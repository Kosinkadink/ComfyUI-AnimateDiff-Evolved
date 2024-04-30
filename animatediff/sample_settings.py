from collections.abc import Iterable
from typing import Union
import torch
from torch import Tensor

import comfy.sample
import comfy.samplers
from comfy.model_patcher import ModelPatcher
from comfy.model_base import BaseModel

from . import freeinit
from .conditioning import LoraHookMode
from .context import ContextOptions, ContextOptionsGroup
from .utils_model import SigmaSchedule
from .utils_motion import extend_to_batch_size, get_sorted_list_via_attr, prepare_mask_batch
from .logger import logger


def prepare_mask_ad(noise_mask, shape, device):
    """ensures noise mask is of proper dimensions"""
    noise_mask = torch.nn.functional.interpolate(noise_mask.reshape((-1, 1, noise_mask.shape[-2], noise_mask.shape[-1])), size=(shape[2], shape[3]), mode="bilinear")
    #noise_mask = noise_mask.round()
    noise_mask = torch.cat([noise_mask] * shape[1], dim=1)
    noise_mask = comfy.utils.repeat_to_batch_size(noise_mask, shape[0])
    noise_mask = noise_mask.to(device)
    return noise_mask


class NoiseLayerType:
    DEFAULT = "default"
    CONSTANT = "constant"
    EMPTY = "empty"
    REPEATED_CONTEXT = "repeated_context"
    FREENOISE = "FreeNoise"

    LIST = [DEFAULT, CONSTANT, EMPTY, REPEATED_CONTEXT, FREENOISE]


class NoiseApplication:
    ADD = "add"
    ADD_WEIGHTED = "add_weighted"
    REPLACE = "replace"
    
    LIST = [ADD, ADD_WEIGHTED, REPLACE]


class NoiseNormalize:
    DISABLE = "disable"
    NORMAL = "normal"

    LIST = [DISABLE, NORMAL]


class SampleSettings:
    def __init__(self, batch_offset: int=0, noise_type: str=None, seed_gen: str=None, seed_offset: int=0, noise_layers: 'NoiseLayerGroup'=None,
                 iteration_opts=None, seed_override:int=None, negative_cond_flipflop=False, adapt_denoise_steps: bool=False,
                 custom_cfg: 'CustomCFGKeyframeGroup'=None, sigma_schedule: SigmaSchedule=None):
        self.batch_offset = batch_offset
        self.noise_type = noise_type if noise_type is not None else NoiseLayerType.DEFAULT
        self.seed_gen = seed_gen if seed_gen is not None else SeedNoiseGeneration.COMFY
        self.noise_layers = noise_layers if noise_layers else NoiseLayerGroup()
        self.iteration_opts = iteration_opts if iteration_opts else IterationOptions()
        self.seed_offset = seed_offset
        self.seed_override = seed_override
        self.negative_cond_flipflop = negative_cond_flipflop
        self.adapt_denoise_steps = adapt_denoise_steps
        self.custom_cfg = custom_cfg.clone() if custom_cfg else custom_cfg
        self.sigma_schedule = sigma_schedule
    
    def prepare_noise(self, seed: int, latents: Tensor, noise: Tensor, extra_seed_offset=0, extra_args:dict={}, force_create_noise=True):
        if self.seed_override is not None:
            seed = self.seed_override
        # if seed is iterable, attempt to do per-latent noises
        if isinstance(seed, Iterable):
            noise = SeedNoiseGeneration.create_noise_individual_seeds(seeds=seed, latents=latents, seed_offset=self.seed_offset+extra_seed_offset, extra_args=extra_args)
            seed = seed[0]+self.seed_offset
        else:
            seed += self.seed_offset
            # replace initial noise if not batch_offset 0 or Comfy seed_gen or not NoiseType default
            if self.batch_offset != 0 or self.seed_offset != 0 or self.noise_type != NoiseLayerType.DEFAULT or self.seed_gen != SeedNoiseGeneration.COMFY or force_create_noise:
                noise = SeedNoiseGeneration.create_noise(seed=seed+extra_seed_offset, latents=latents, existing_seed_gen=self.seed_gen, seed_gen=self.seed_gen,
                                                        noise_type=self.noise_type, batch_offset=self.batch_offset, extra_args=extra_args)
        # apply noise layers
        for noise_layer in self.noise_layers.layers:
            # first, generate new noise matching seed gen override
            layer_noise = noise_layer.create_layer_noise(existing_seed_gen=self.seed_gen, seed=seed, latents=latents,
                                                         extra_seed_offset=extra_seed_offset, extra_args=extra_args)
            # next, get noise after applying layer
            noise = noise_layer.apply_layer_noise(new_noise=layer_noise, old_noise=noise)
        # noise prepared now
        return noise
    
    def pre_run(self, model: ModelPatcher):
        if self.custom_cfg is not None:
            self.custom_cfg.reset()
    
    def cleanup(self):
        if self.custom_cfg is not None:
            self.custom_cfg.reset()

    def clone(self):
        return SampleSettings(batch_offset=self.batch_offset, noise_type=self.noise_type, seed_gen=self.seed_gen, seed_offset=self.seed_offset,
                           noise_layers=self.noise_layers.clone(), iteration_opts=self.iteration_opts, seed_override=self.seed_override,
                           negative_cond_flipflop=self.negative_cond_flipflop, adapt_denoise_steps=self.adapt_denoise_steps, custom_cfg=self.custom_cfg, sigma_schedule=self.sigma_schedule)


class NoiseLayer:
    def __init__(self, noise_type: str, batch_offset: int, seed_gen_override: str, seed_offset: int, seed_override: int=None, mask: Tensor=None):
        self.application: str = NoiseApplication.REPLACE
        self.noise_type = noise_type
        self.batch_offset = batch_offset
        self.seed_gen_override = seed_gen_override
        self.seed_offset = seed_offset
        self.seed_override = seed_override
        self.mask = mask
    
    def create_layer_noise(self, existing_seed_gen: str, seed: int, latents: Tensor, extra_seed_offset=0, extra_args:dict={}) -> Tensor:
        if self.seed_override is not None:
            seed = self.seed_override
         # if seed is iterable, attempt to do per-latent noises
        if isinstance(seed, Iterable):
            return SeedNoiseGeneration.create_noise_individual_seeds(seeds=seed, latents=latents, seed_offset=self.seed_offset+extra_seed_offset, extra_args=extra_args)
        seed += self.seed_offset + extra_seed_offset
        return SeedNoiseGeneration.create_noise(seed=seed, latents=latents, existing_seed_gen=existing_seed_gen, seed_gen=self.seed_gen_override,
                                                noise_type=self.noise_type, batch_offset=self.batch_offset, extra_args=extra_args)

    def apply_layer_noise(self, new_noise: Tensor, old_noise: Tensor) -> Tensor:
        return old_noise
    
    def get_noise_mask(self, noise: Tensor) -> Tensor:
        if self.mask is None:
            return 1
        noise_mask = self.mask.reshape((-1, 1, self.mask.shape[-2], self.mask.shape[-1]))
        return prepare_mask_ad(noise_mask, noise.shape, noise.device)


class NoiseLayerReplace(NoiseLayer):
    def __init__(self, noise_type: str, batch_offset: int, seed_gen_override: str, seed_offset: int, seed_override: int=None, mask: Tensor=None):
        super().__init__(noise_type, batch_offset, seed_gen_override, seed_offset, seed_override, mask)
        self.application = NoiseApplication.REPLACE

    def apply_layer_noise(self, new_noise: Tensor, old_noise: Tensor) -> Tensor:
        noise_mask = self.get_noise_mask(old_noise)
        return (1-noise_mask)*old_noise + noise_mask*new_noise


class NoiseLayerAdd(NoiseLayer):
    def __init__(self, noise_type: str, batch_offset: int, seed_gen_override: str, seed_offset: int, seed_override: int=None, mask: Tensor=None,
                 noise_weight=1.0):
        super().__init__(noise_type, batch_offset, seed_gen_override, seed_offset, seed_override, mask)
        self.noise_weight = noise_weight
        self.application = NoiseApplication.ADD

    def apply_layer_noise(self, new_noise: Tensor, old_noise: Tensor) -> Tensor:
        noise_mask = self.get_noise_mask(old_noise)
        return (1-noise_mask)*old_noise + noise_mask*(old_noise + new_noise * self.noise_weight)


class NoiseLayerAddWeighted(NoiseLayerAdd):
    def __init__(self, noise_type: str, batch_offset: int, seed_gen_override: str, seed_offset: int, seed_override: int=None, mask: Tensor=None,
                 noise_weight=1.0, balance_multiplier=1.0):
        super().__init__(noise_type, batch_offset, seed_gen_override, seed_offset, seed_override, mask, noise_weight)
        self.balance_multiplier = balance_multiplier
        self.application = NoiseApplication.ADD_WEIGHTED

    def apply_layer_noise(self, new_noise: Tensor, old_noise: Tensor) -> Tensor:
        noise_mask = self.get_noise_mask(old_noise)
        return (1-noise_mask)*old_noise + noise_mask*(old_noise * (1.0-(self.noise_weight*self.balance_multiplier)) + new_noise * self.noise_weight)


class NoiseLayerGroup:
    def __init__(self):
        self.layers: list[NoiseLayer] = []
    
    def add(self, layer: NoiseLayer) -> None:
        # add to the end of list
        self.layers.append(layer)

    def add_to_start(self, layer: NoiseLayer) -> None:
        # add to the beginning of list
        self.layers.insert(0, layer)

    def __getitem__(self, index) -> NoiseLayer:
        return self.layers[index]
    
    def is_empty(self) -> bool:
        return len(self.layers) == 0
    
    def clone(self) -> 'NoiseLayerGroup':
        cloned = NoiseLayerGroup()
        for layer in self.layers:
            cloned.add(layer)
        return cloned

class SeedNoiseGeneration:
    COMFY = "comfy"
    AUTO1111 = "auto1111"
    AUTO1111GPU = "auto1111 [gpu]" # TODO: implement this
    USE_EXISTING = "use existing"

    LIST = [COMFY, AUTO1111]
    LIST_WITH_OVERRIDE = [USE_EXISTING, COMFY, AUTO1111]

    @classmethod
    def create_noise(cls, seed: int, latents: Tensor, existing_seed_gen: str=COMFY, seed_gen: str=USE_EXISTING, noise_type: str=NoiseLayerType.DEFAULT, batch_offset: int=0, extra_args: dict={}):
        # determine if should use existing type
        if seed_gen == cls.USE_EXISTING:
            seed_gen = existing_seed_gen
        if seed_gen == cls.COMFY:
            return cls.create_noise_comfy(seed, latents, noise_type, batch_offset, extra_args)
        elif seed_gen in [cls.AUTO1111, cls.AUTO1111GPU]:
            return cls.create_noise_auto1111(seed, latents, noise_type, batch_offset, extra_args)
        raise ValueError(f"Noise seed_gen {seed_gen} is not recognized.")

    @staticmethod
    def create_noise_comfy(seed: int, latents: Tensor, noise_type: str=NoiseLayerType.DEFAULT, batch_offset: int=0, extra_args: dict={}):
        common_noise = SeedNoiseGeneration._create_common_noise(seed, latents, noise_type, batch_offset, extra_args)
        if common_noise is not None:
            return common_noise
        if noise_type == NoiseLayerType.CONSTANT:
            generator = torch.manual_seed(seed)
            length = latents.shape[0]
            single_shape = (1 + batch_offset, latents.shape[1], latents.shape[2], latents.shape[3])
            single_noise = torch.randn(single_shape, dtype=latents.dtype, layout=latents.layout, generator=generator, device="cpu")
            return torch.cat([single_noise[batch_offset:]] * length, dim=0)
        # comfy creates noise with a single seed for the entire shape of the latents batched tensor
        generator = torch.manual_seed(seed)
        offset_shape = (latents.shape[0] + batch_offset, latents.shape[1], latents.shape[2], latents.shape[3])
        final_noise = torch.randn(offset_shape, dtype=latents.dtype, layout=latents.layout, generator=generator, device="cpu")
        final_noise = final_noise[batch_offset:]
        # convert to derivative noise type, if needed
        derivative_noise = SeedNoiseGeneration._create_derivative_noise(final_noise, noise_type=noise_type, seed=seed, extra_args=extra_args)
        if derivative_noise is not None:
            return derivative_noise
        return final_noise
    
    @staticmethod
    def create_noise_auto1111(seed: int, latents: Tensor, noise_type: str=NoiseLayerType.DEFAULT, batch_offset: int=0, extra_args: dict={}):
        common_noise = SeedNoiseGeneration._create_common_noise(seed, latents, noise_type, batch_offset, extra_args)
        if common_noise is not None:
            return common_noise
        if noise_type == NoiseLayerType.CONSTANT:
            generator = torch.manual_seed(seed+batch_offset)
            length = latents.shape[0]
            single_shape = (1, latents.shape[1], latents.shape[2], latents.shape[3])
            single_noise = torch.randn(single_shape, dtype=latents.dtype, layout=latents.layout, generator=generator, device="cpu")
            return torch.cat([single_noise] * length, dim=0)
        # auto1111 applies growing seeds for a batch
        length = latents.shape[0]
        single_shape = (1, latents.shape[1], latents.shape[2], latents.shape[3])
        all_noises = []
        # i starts at 0
        for i in range(length):
            generator = torch.manual_seed(seed+i+batch_offset)
            all_noises.append(torch.randn(single_shape, dtype=latents.dtype, layout=latents.layout, generator=generator, device="cpu"))
        final_noise = torch.cat(all_noises, dim=0)
        # convert to derivative noise type, if needed
        derivative_noise = SeedNoiseGeneration._create_derivative_noise(final_noise, noise_type=noise_type, seed=seed, extra_args=extra_args)
        if derivative_noise is not None:
            return derivative_noise
        return final_noise
    
    @staticmethod
    def create_noise_individual_seeds(seeds: list[int], latents: Tensor, seed_offset: int=0, extra_args: dict={}):
        length = latents.shape[0]
        if len(seeds) < length:
            raise ValueError(f"{len(seeds)} seeds in seed_override were provided, but at least {length} are required to work with the current latents.")
        seeds = seeds[:length]
        single_shape = (1, latents.shape[1], latents.shape[2], latents.shape[3])
        all_noises = []
        for seed in seeds:
            generator = torch.manual_seed(seed+seed_offset)
            all_noises.append(torch.randn(single_shape, dtype=latents.dtype, layout=latents.layout, generator=generator, device="cpu"))
        return torch.cat(all_noises, dim=0)

    @staticmethod
    def _create_common_noise(seed: int, latents: Tensor, noise_type: str=NoiseLayerType.DEFAULT, batch_offset: int=0, extra_args: dict={}):
        if noise_type == NoiseLayerType.EMPTY:
            return torch.zeros_like(latents)
        return None
    
    @staticmethod
    def _create_derivative_noise(noise: Tensor, noise_type: str, seed: int, extra_args: dict):
        derivative_func = DERIVATIVE_NOISE_FUNC_MAP.get(noise_type, None)
        if derivative_func is None:
            return None
        return derivative_func(noise=noise, seed=seed, extra_args=extra_args)

    @staticmethod
    def _convert_to_repeated_context(noise: Tensor, extra_args: dict, **kwargs):
        # if no context_length, return unmodified noise
        opts: ContextOptionsGroup = extra_args["context_options"]
        context_length: int = opts.context_length if not opts.view_options else opts.view_options.context_length
        if context_length is None:
            return noise
        length = noise.shape[0]
        noise = noise[:context_length]
        cat_count = (length // context_length) + 1
        return torch.cat([noise] * cat_count, dim=0)[:length]

    @staticmethod
    def _convert_to_freenoise(noise: Tensor, seed: int, extra_args: dict, **kwargs):
        # if no context_length, return unmodified noise
        opts: ContextOptionsGroup = extra_args["context_options"]
        context_length: int = opts.context_length if not opts.view_options else opts.view_options.context_length
        context_overlap: int = opts.context_overlap if not opts.view_options else opts.view_options.context_overlap
        video_length: int = noise.shape[0]
        if context_length is None:
            return noise
        delta = context_length - context_overlap
        generator = torch.manual_seed(seed)

        for start_idx in range(0, video_length-context_length, delta):
            # start_idx corresponds to the beginning of a context window
            # goal: place shuffled in the delta region right after the end of the context window
            #       if space after context window is not enough to place the noise, adjust and finish
            place_idx = start_idx + context_length
            # if place_idx is outside the valid indexes, we are already finished
            if place_idx >= video_length:
                break
            end_idx = place_idx - 1
            # if there is not enough room to copy delta amount of indexes, copy limited amount and finish
            if end_idx + delta >= video_length:
                final_delta = video_length - place_idx
                # generate list of indexes in final delta region
                list_idx = torch.Tensor(list(range(start_idx,start_idx+final_delta))).to(torch.long)
                # shuffle list
                list_idx = list_idx[torch.randperm(final_delta, generator=generator)]
                # apply shuffled indexes
                noise[place_idx:place_idx+final_delta] = noise[list_idx]
                break
            # otherwise, do normal behavior
            # generate list of indexes in delta region
            list_idx = torch.Tensor(list(range(start_idx,start_idx+delta))).to(torch.long)
            # shuffle list
            list_idx = list_idx[torch.randperm(delta, generator=generator)]
            # apply shuffled indexes
            noise[place_idx:place_idx+delta] = noise[list_idx]
        return noise


DERIVATIVE_NOISE_FUNC_MAP = {
    NoiseLayerType.REPEATED_CONTEXT: SeedNoiseGeneration._convert_to_repeated_context,
    NoiseLayerType.FREENOISE: SeedNoiseGeneration._convert_to_freenoise,
    }


class IterationOptions:
    SAMPLER = "sampler"

    def __init__(self, iterations: int=1, cache_init_noise=False, cache_init_latents=False,
                 iter_batch_offset: int=0, iter_seed_offset: int=0):
        self.iterations = iterations
        self.cache_init_noise = cache_init_noise
        self.cache_init_latents = cache_init_latents
        self.iter_batch_offset = iter_batch_offset
        self.iter_seed_offset = iter_seed_offset
        self.need_sampler = False

    def get_sigma(self, model: ModelPatcher, step: int):
        model_sampling = model.model.model_sampling
        if "model_sampling" in model.object_patches:
            model_sampling = model.object_patches["model_sampling"]
        return model_sampling.sigmas[step]

    def initialize(self, latents: Tensor):
        pass

    def preprocess_latents(self, curr_i: int, model: ModelPatcher, latents: Tensor, noise: Tensor,
                           seed: int, sample_settings: SampleSettings, noise_extra_args: dict, **kwargs):
        if curr_i == 0 or (self.iter_batch_offset == 0 and self.iter_seed_offset == 0):
            return latents, noise
        temp_sample_settings = sample_settings.clone()
        temp_sample_settings.batch_offset += self.iter_batch_offset * curr_i
        temp_sample_settings.seed_offset += self.iter_seed_offset * curr_i
        return latents, temp_sample_settings.prepare_noise(seed=seed, latents=latents, noise=None,
                                                    extra_args=noise_extra_args, force_create_noise=True)


class FreeInitOptions(IterationOptions):
    FREEINIT_SAMPLER = "FreeInit [sampler sigma]"
    FREEINIT_MODEL = "FreeInit [model sigma]"
    DINKINIT_V1 = "DinkInit_v1"

    LIST = [FREEINIT_SAMPLER, FREEINIT_MODEL, DINKINIT_V1]

    def __init__(self, iterations: int, step: int=999, apply_to_1st_iter: bool=False,
                 filter=freeinit.FreeInitFilter.GAUSSIAN, d_s=0.25, d_t=0.25, n=4, init_type=FREEINIT_SAMPLER,
                 iter_batch_offset: int=0, iter_seed_offset: int=1):
        super().__init__(iterations=iterations, cache_init_noise=True, cache_init_latents=True,
                         iter_batch_offset=iter_batch_offset, iter_seed_offset=iter_seed_offset)
        self.apply_to_1st_iter = apply_to_1st_iter
        self.step = step
        self.filter = filter
        self.d_s = d_s
        self.d_t = d_t
        self.n = n
        self.freq_filter = None
        self.freq_filter2 = None
        self.need_sampler = True if init_type in [self.FREEINIT_SAMPLER] else False
        self.init_type = init_type

    def initialize(self, latents: Tensor):
        self.freq_filter = freeinit.get_freq_filter(latents.shape, device=latents.device, filter_type=self.filter,
                                           n=self.n, d_s=self.d_s, d_t=self.d_t)
    
    def preprocess_latents(self, curr_i: int, model: ModelPatcher, latents: Tensor, noise: Tensor, cached_latents: Tensor, cached_noise: Tensor,
                           seed:int, sample_settings: SampleSettings, noise_extra_args: dict, sampler: comfy.samplers.KSampler=None, **kwargs):
        # if first iter and should not apply, do nothing
        if curr_i == 0 and not self.apply_to_1st_iter:
            return latents, noise
        # otherwise, do FreeInit stuff
        if self.init_type in [self.FREEINIT_SAMPLER, self.FREEINIT_MODEL]:
            # NOTE: This should be very close (if not exactly) to how FreeInit is intended to initialize noise the latents.
            #       The trick is that FreeInit is dependent on the behavior of diffuser's DDIMScheduler.add_noise function.
            #       The typical noising method of latents + noise * sigma will NOT work.
            # 1. apply initial noise with appropriate step sigma, normalized against scale_factor
            if sampler is not None:
                sigma = sampler.sigmas[999-self.step].to(latents.device) / (model.model.latent_format.scale_factor)
            else:
                sigma = self.get_sigma(model, self.step-1000).to(latents.device) / (model.model.latent_format.scale_factor)
            alpha_cumprod = 1 / ((sigma * sigma) + 1)
            sqrt_alpha_prod = alpha_cumprod ** 0.5
            sqrt_one_minus_alpha_prod = (1 - alpha_cumprod) ** 0.5
            noised_latents = latents * sqrt_alpha_prod + noise * sqrt_one_minus_alpha_prod
            # 2. create random noise z_rand for high frequency
            temp_sample_settings = sample_settings.clone()
            temp_sample_settings.batch_offset += self.iter_batch_offset * curr_i
            temp_sample_settings.seed_offset += self.iter_seed_offset * curr_i
            z_rand = temp_sample_settings.prepare_noise(seed=seed, latents=latents, noise=None,
                                                    extra_args=noise_extra_args, force_create_noise=True)
            # 3. noise reinitialization - combines low freq. noise from noised_latents and high freq. noise from z_rand
            noised_latents = freeinit.freq_mix_3d(x=noised_latents, noise=z_rand.to(dtype=latents.dtype, device=latents.device), LPF=self.freq_filter)
            return cached_latents, noised_latents
        elif self.init_type == self.DINKINIT_V1:
            # NOTE: This was my first attempt at implementing FreeInit; it sorta works due to my alpha_cumprod shenanigans,
            #       but completely by accident.
            # 1. apply initial noise with appropriate step sigma
            sigma = self.get_sigma(model, self.step-1000).to(latents.device)
            alpha_cumprod = 1 / ((sigma * sigma) + 1) #1 / ((sigma * sigma)) # 1 / ((sigma * sigma) + 1)
            noised_latents = (latents + (cached_noise * sigma)) * alpha_cumprod
            # 2. create random noise z_rand for high frequency
            temp_sample_settings = sample_settings.clone()
            temp_sample_settings.batch_offset += self.iter_batch_offset * curr_i
            temp_sample_settings.seed_offset += self.iter_seed_offset * curr_i
            z_rand = temp_sample_settings.prepare_noise(seed=seed, latents=latents, noise=None,
                                                    extra_args=noise_extra_args, force_create_noise=True)
            ####z_rand = torch.randn_like(latents, dtype=latents.dtype, device=latents.device)
            # 3. noise reinitialization - combines low freq. noise from noised_latents and high freq. noise from z_rand
            noised_latents = freeinit.freq_mix_3d(x=noised_latents, noise=z_rand.to(dtype=latents.dtype, device=latents.device), LPF=self.freq_filter)
            return cached_latents, noised_latents
        else:
            raise ValueError(f"FreeInit init_type '{self.init_type}' is not recognized.")


class CustomCFGKeyframe:
    def __init__(self, cfg_multival: Union[float, Tensor], start_percent=0.0, guarantee_steps=1):
        self.cfg_multival = cfg_multival
        # scheduling
        self.start_percent = float(start_percent)
        self.start_t = 999999999.9
        self.guarantee_steps = guarantee_steps
    
    def clone(self):
        c = CustomCFGKeyframe(cfg_multival=self.cfg_multival,
                              start_percent=self.start_percent, guarantee_steps=self.guarantee_steps)
        c.start_t = self.start_t
        return c


class CustomCFGKeyframeGroup:
    def __init__(self):
        self.keyframes: list[CustomCFGKeyframe] = []
        self._current_keyframe: CustomCFGKeyframe = None
        self._current_used_steps: int = 0
        self._current_index: int = 0
    
    def reset(self):
        self._current_keyframe = None
        self._current_used_steps = 0
        self._current_index = 0
        self._set_first_as_current()

    def add(self, keyframe: CustomCFGKeyframe):
        # add to end of list, then sort
        self.keyframes.append(keyframe)
        self.keyframes = get_sorted_list_via_attr(self.keyframes, "start_percent")
        self._set_first_as_current()

    def _set_first_as_current(self):
        if len(self.keyframes) > 0:
            self._current_keyframe = self.keyframes[0]
        else:
            self._current_keyframe = None

    def has_index(self, index: int) -> int:
        return index >=0 and index < len(self.keyframes)

    def is_empty(self) -> bool:
        return len(self.keyframes) == 0
    
    def clone(self):
        cloned = CustomCFGKeyframeGroup()
        for keyframe in self.keyframes:
            cloned.keyframes.append(keyframe)
        cloned._set_first_as_current()
        return cloned
    
    def initialize_timesteps(self, model: BaseModel):
        for keyframe in self.keyframes:
            keyframe.start_t = model.model_sampling.percent_to_sigma(keyframe.start_percent)
    
    def prepare_current_keyframe(self, t: Tensor):
        curr_t: float = t[0]
        prev_index = self._current_index
        # if met guaranteed steps, look for next keyframe in case need to switch
        if self._current_used_steps >= self._current_keyframe.guarantee_steps:
            # if has next index, loop through and see if need t oswitch
            if self.has_index(self._current_index+1):
                for i in range(self._current_index+1, len(self.keyframes)):
                    eval_c = self.keyframes[i]
                    # check if start_t is greater or equal to curr_t
                    # NOTE: t is in terms of sigmas, not percent, so bigger number = earlier step in sampling
                    if eval_c.start_t >= curr_t:
                        self._current_index = i
                        self._current_keyframe = eval_c
                        self._current_used_steps = 0
                        # if guarantee_steps greater than zero, stop searching for other keyframes
                        if self._current_keyframe.guarantee_steps > 0:
                            break
                    # if eval_c is outside the percent range, stop looking further
                    else: break
        # update steps current context is used
        self._current_used_steps += 1

    def patch_model(self, model: ModelPatcher) -> ModelPatcher:
        def evolved_custom_cfg(args):
            cond: Tensor = args["cond"]
            uncond: Tensor = args["uncond"]
            # cond scale is based purely off of CustomCFG - cond_scale input in sampler is ignored!
            cond_scale = self.cfg_multival
            if isinstance(cond_scale, Tensor):
                cond_scale = prepare_mask_batch(cond_scale.to(cond.dtype).to(cond.device), cond.shape)
                cond_scale = extend_to_batch_size(cond_scale, cond.shape[0])
            return uncond + (cond - uncond) * cond_scale

        model = model.clone()
        model.set_model_sampler_cfg_function(evolved_custom_cfg)
        return model

    # properties shadow those of CustomCFGKeyframe
    @property
    def cfg_multival(self):
        if self._current_keyframe != None:
            return self._current_keyframe.cfg_multival
        return None
