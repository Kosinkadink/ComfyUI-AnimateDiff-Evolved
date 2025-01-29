from __future__ import annotations
from collections.abc import Iterable
from typing import Union, Callable
import torch
from torch import Tensor
import torch.fft as fft
from einops import rearrange

import comfy.k_diffusion.sampling
import comfy.sample
import comfy.samplers
import comfy.model_management
from comfy.patcher_extension import WrappersMP, add_wrapper_with_key
from comfy.model_patcher import ModelPatcher
from comfy.model_base import BaseModel
from comfy.sd import VAE

from . import freeinit
from .context import ContextOptions, ContextOptionsGroup
from .utils_model import SigmaSchedule, BIGMAX_TENSOR
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


class NoiseDeterminism:
    DEFAULT = "default"
    DETERMINISTIC = "deterministic"

    _LIST = [DEFAULT, DETERMINISTIC]


class NoiseLayerType:
    DEFAULT = "default"
    CONSTANT = "constant"
    EMPTY = "empty"
    REPEATED_CONTEXT = "repeated_context"
    FREENOISE = "FreeNoise"

    LIST = [DEFAULT, CONSTANT, EMPTY, REPEATED_CONTEXT, FREENOISE]
    LIST_ANCESTRAL = [DEFAULT, CONSTANT]


class NoiseApplication:
    ADD = "add"
    ADD_WEIGHTED = "add_weighted"
    NORMALIZED_SUM = "normalized_sum"
    REPLACE = "replace"
    
    LIST = [ADD, ADD_WEIGHTED, NORMALIZED_SUM, REPLACE]


class NoiseNormalize:
    DISABLE = "disable"
    NORMAL = "normal"

    LIST = [DISABLE, NORMAL]


class SampleSettings:
    def __init__(self, batch_offset: int=0, noise_type: str=None, seed_gen: str=None, seed_offset: int=0, noise_layers: NoiseLayerGroup=None,
                 iteration_opts=None, seed_override:int=None, negative_cond_flipflop=False, adapt_denoise_steps: bool=False,
                 custom_cfg: CustomCFGKeyframeGroup=None, sigma_schedule: SigmaSchedule=None, image_injection: NoisedImageToInjectGroup=None,
                 noise_calibration: NoiseCalibration=None, ancestral_opts: AncestralOptions=None):
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
        self.image_injection = image_injection.clone() if image_injection else NoisedImageToInjectGroup()
        self.noise_calibration = noise_calibration
        self.ancestral_opts = ancestral_opts
    
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
        if self.image_injection is not None:
            self.image_injection.reset()
    
    def cleanup(self):
        if self.custom_cfg is not None:
            self.custom_cfg.reset()
        if self.image_injection is not None:
            self.image_injection.reset()

    def clone(self):
        return SampleSettings(batch_offset=self.batch_offset, noise_type=self.noise_type, seed_gen=self.seed_gen, seed_offset=self.seed_offset,
                           noise_layers=self.noise_layers.clone(), iteration_opts=self.iteration_opts, seed_override=self.seed_override,
                           negative_cond_flipflop=self.negative_cond_flipflop, adapt_denoise_steps=self.adapt_denoise_steps, custom_cfg=self.custom_cfg,
                           sigma_schedule=self.sigma_schedule, image_injection=self.image_injection, noise_calibration=self.noise_calibration,
                           ancestral_opts=self.ancestral_opts)


class AncestralOptions:
    def __init__(self, noise_type: str, determinism: str, seed_offset: int, seed_override: int=None):
        self.noise_type = noise_type
        self.determinism = determinism
        self.seed_offset = seed_offset
        self.seed_override = seed_override
    
    def init_custom_noise_sampler(self, seed: int):
        if self.seed_override is not None:
            seed = self.seed_override
        if isinstance(seed, Iterable):
            raise Exception("Passing in a list of seeds for Ancestral Options is not supported at this time.")
        seed += self.seed_offset
        return _custom_noise_sampler_factory(real_seed=seed, noise_type=self.noise_type, determinism=self.determinism)

    def add_wrapper_sampler_sample(self, model_options, seed):
        add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, "ADE",
                             _sampler_sample_ancestral_options_factory(self.init_custom_noise_sampler(seed)),
                             model_options, is_model_options=True)


def _sampler_sample_ancestral_options_factory(custom_noise_sampler: Callable):
    def sampler_sample_ancestral_options_wrapper(executor, *args, **kwargs):
        try:
            # TODO: implement this as a model_options thing instead in core ComfyUI
            orig_default_noise_sampler = comfy.k_diffusion.sampling.default_noise_sampler
            comfy.k_diffusion.sampling.default_noise_sampler = custom_noise_sampler
            return executor(*args, **kwargs)
        finally:
            comfy.k_diffusion.sampling.default_noise_sampler = orig_default_noise_sampler
    return sampler_sample_ancestral_options_wrapper


def _custom_noise_sampler_factory(real_seed: int, noise_type: str, determinism: str):
    def custom_noise_sampler(x: Tensor, seed: int=None):
        single_generator = None
        multiple_generators = []
        if determinism == NoiseDeterminism.DEFAULT:
            # prepare generators
            single_generator = torch.Generator(device=x.device)
            single_generator.manual_seed(real_seed)
            # create function to handle determinism type
            def sample_default(sigma, sigma_next):
                if noise_type == NoiseLayerType.CONSTANT:
                    goal_shape = list(x.shape)
                    goal_shape[0] = 1
                    one_noise = torch.randn(goal_shape, dtype=x.dtype, layout=x.layout, device=x.device, generator=single_generator)
                    return torch.cat([one_noise]*x.shape[0], dim=0)
                return torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=single_generator)
            # return function
            return sample_default
        elif determinism == NoiseDeterminism.DETERMINISTIC:
            # prepare generators
            for i in range(x.size(0)):
                generator = torch.Generator(device=x.device)
                multiple_generators.append(generator.manual_seed(real_seed+i))
            # create function to handle determinism type
            def sample_deterministic(sigma, sigma_next):
                goal_shape = list(x.shape)
                goal_shape[0] = 1
                if noise_type == NoiseLayerType.CONSTANT:
                    one_noise = torch.randn(goal_shape, dtype=x.dtype, layout=x.layout, device=x.device, generator=multiple_generators[0])
                    return torch.cat([one_noise]*x.shape[0], dim=0)
                noises = []
                for generator in multiple_generators:
                    one_noise = torch.randn(goal_shape, dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)
                    noises.append(one_noise)
                return torch.cat(noises, dim=0)
            # return function
            return sample_deterministic
        else:
            raise Exception(f"Determinism type '{determinism}' is not recognized.")
    # return function
    return custom_noise_sampler


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
        super().__init__(noise_type, batch_offset, seed_gen_override, seed_offset, seed_override, mask)
        self.noise_weight = noise_weight
        self.balance_multiplier = balance_multiplier
        self.application = NoiseApplication.ADD_WEIGHTED

    def apply_layer_noise(self, new_noise: Tensor, old_noise: Tensor) -> Tensor:
        noise_mask = self.get_noise_mask(old_noise)
        return (1-noise_mask)*old_noise + noise_mask*(old_noise * (1.0-(self.noise_weight*self.balance_multiplier)) + new_noise * self.noise_weight)


class NoiseLayerNormalizedSum(NoiseLayer):
    def __init__(self, noise_type: str, batch_offset: int, seed_gen_override: str, seed_offset: int, seed_override: int=None, mask: Tensor=None,
                 noise_weight=1.0):
        super().__init__(noise_type, batch_offset, seed_gen_override, seed_offset, seed_override, mask)
        self.noise_weight = noise_weight
        self.application = NoiseApplication.NORMALIZED_SUM

    def apply_layer_noise(self, new_noise: Tensor, old_noise: Tensor) -> Tensor:
        noise_mask = self.get_noise_mask(old_noise)
        weight_old = 1.0 - self.noise_weight
        weight_new = self.noise_weight
        
        norm_factor = (weight_old**2 + weight_new**2)**0.5
        weight_old /= norm_factor
        weight_new /= norm_factor

        return (1 - noise_mask) * old_noise + noise_mask * (weight_old * old_noise + weight_new * new_noise)


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


class RandDevice:
    CPU = "cpu"
    GPU = "gpu"
    NV = "nv"


def get_generator(device=RandDevice.CPU, seed: int=None):
    generator = None
    raw_device = None
    if device == RandDevice.CPU:
        raw_device = "cpu"
        generator = torch.Generator(raw_device)
    elif device == RandDevice.GPU:
        raw_device = comfy.model_management.get_torch_device()
        generator = torch.Generator(raw_device)
    # TODO: should I add the NV code from Auto1111?
    # It is AGPL licenced, which should be fine since I will not be modifying it.
    # elif device == RandDevice.NV:
    #     pass
    else:
        raise Exception(f"Unknown noise generator device: '{device}'")
    if seed is not None:
        generator = generator.manual_seed(seed)
    return generator, raw_device


class SeedNoiseGeneration:
    COMFY = "comfy"
    COMFYGPU = "comfy [gpu]"
    #COMFYNV = "comfy [nv]"
    AUTO1111 = "auto1111"
    AUTO1111GPU = "auto1111 [gpu]"
    #AUTO1111NV = "auto1111 [nv]"
    USE_EXISTING = "use existing"

    LIST = [COMFY, COMFYGPU, AUTO1111, AUTO1111GPU]
    LIST_WITH_OVERRIDE = [USE_EXISTING, COMFY, COMFYGPU, AUTO1111, AUTO1111GPU]

    _COMFY_GENS = [COMFY, COMFYGPU]
    _AUTO1111_GENS = [AUTO1111, AUTO1111GPU]

    _SOURCE_DICT = {
        COMFY: RandDevice.CPU, COMFYGPU: RandDevice.GPU,
        AUTO1111: RandDevice.CPU, AUTO1111GPU: RandDevice.GPU,
    }

    @classmethod
    def get_device(cls, seed_gen: str):
        return cls._SOURCE_DICT[seed_gen]

    @classmethod
    def create_noise(cls, seed: int, latents: Tensor, existing_seed_gen: str=COMFY, seed_gen: str=USE_EXISTING, noise_type: str=NoiseLayerType.DEFAULT, batch_offset: int=0, extra_args: dict={}):
        # determine if should use existing type
        if seed_gen == cls.USE_EXISTING:
            seed_gen = existing_seed_gen
        if seed_gen in cls._COMFY_GENS:
            return cls.create_noise_comfy(seed, latents, noise_type, batch_offset, extra_args, cls.get_device(seed_gen))
        elif seed_gen in cls._AUTO1111_GENS:
            return cls.create_noise_auto1111(seed, latents, noise_type, batch_offset, extra_args, cls.get_device(seed_gen))
        raise ValueError(f"Noise seed_gen {seed_gen} is not recognized.")

    @staticmethod
    def create_noise_comfy(seed: int, latents: Tensor, noise_type: str=NoiseLayerType.DEFAULT, batch_offset: int=0, extra_args: dict={}, device=RandDevice.CPU):
        common_noise = SeedNoiseGeneration._create_common_noise(seed, latents, noise_type, batch_offset, extra_args, device)
        if common_noise is not None:
            return common_noise
        if noise_type == NoiseLayerType.CONSTANT:
            generator, raw_device = get_generator(device, seed)
            length = latents.shape[0]
            single_shape = (1 + batch_offset, latents.shape[1], latents.shape[2], latents.shape[3])
            single_noise = torch.randn(single_shape, dtype=latents.dtype, layout=latents.layout, generator=generator, device=raw_device).to(device="cpu")
            return torch.cat([single_noise[batch_offset:]] * length, dim=0)
        # comfy creates noise with a single seed for the entire shape of the latents batched tensor
        generator, raw_device = get_generator(device, seed)
        offset_shape = (latents.shape[0] + batch_offset, latents.shape[1], latents.shape[2], latents.shape[3])
        final_noise = torch.randn(offset_shape, dtype=latents.dtype, layout=latents.layout, generator=generator, device=raw_device).to(device="cpu")
        final_noise = final_noise[batch_offset:]
        # convert to derivative noise type, if needed
        derivative_noise = SeedNoiseGeneration._create_derivative_noise(final_noise, noise_type=noise_type, seed=seed, extra_args=extra_args, device=device)
        if derivative_noise is not None:
            return derivative_noise
        return final_noise
    
    @staticmethod
    def create_noise_auto1111(seed: int, latents: Tensor, noise_type: str=NoiseLayerType.DEFAULT, batch_offset: int=0, extra_args: dict={}, device=RandDevice.CPU):
        common_noise = SeedNoiseGeneration._create_common_noise(seed, latents, noise_type, batch_offset, extra_args, device)
        if common_noise is not None:
            return common_noise
        if noise_type == NoiseLayerType.CONSTANT:
            generator, raw_device = get_generator(device, seed+batch_offset)
            length = latents.shape[0]
            single_shape = (1, latents.shape[1], latents.shape[2], latents.shape[3])
            single_noise = torch.randn(single_shape, dtype=latents.dtype, layout=latents.layout, generator=generator, device=raw_device).to(device="cpu")
            return torch.cat([single_noise] * length, dim=0)
        # auto1111 applies growing seeds for a batch
        length = latents.shape[0]
        single_shape = (1, latents.shape[1], latents.shape[2], latents.shape[3])
        all_noises = []
        # i starts at 0
        for i in range(length):
            generator, raw_device = get_generator(device, seed+i+batch_offset)
            all_noises.append(torch.randn(single_shape, dtype=latents.dtype, layout=latents.layout, generator=generator, device=raw_device).to(device="cpu"))
        final_noise = torch.cat(all_noises, dim=0)
        # convert to derivative noise type, if needed
        derivative_noise = SeedNoiseGeneration._create_derivative_noise(final_noise, noise_type=noise_type, seed=seed, extra_args=extra_args, device=device)
        if derivative_noise is not None:
            return derivative_noise
        return final_noise
    
    @staticmethod
    def create_noise_individual_seeds(seeds: list[int], latents: Tensor, seed_offset: int=0, extra_args: dict={}, device=RandDevice.CPU):
        length = latents.shape[0]
        if len(seeds) < length:
            raise ValueError(f"{len(seeds)} seeds in seed_override were provided, but at least {length} are required to work with the current latents.")
        seeds = seeds[:length]
        single_shape = (1, latents.shape[1], latents.shape[2], latents.shape[3])
        all_noises = []
        for seed in seeds:
            generator, raw_device = get_generator(device, seed+seed_offset)
            all_noises.append(torch.randn(single_shape, dtype=latents.dtype, layout=latents.layout, generator=generator, device=raw_device).to(device="cpu"))
        return torch.cat(all_noises, dim=0)

    @staticmethod
    def _create_common_noise(seed: int, latents: Tensor, noise_type: str=NoiseLayerType.DEFAULT, batch_offset: int=0, extra_args: dict={}, device=RandDevice.CPU):
        if noise_type == NoiseLayerType.EMPTY:
            return torch.zeros_like(latents)
        return None
    
    @staticmethod
    def _create_derivative_noise(noise: Tensor, noise_type: str, seed: int, extra_args: dict, device=RandDevice.CPU):
        derivative_func = DERIVATIVE_NOISE_FUNC_MAP.get(noise_type, None)
        if derivative_func is None:
            return None
        return derivative_func(noise=noise, seed=seed, extra_args=extra_args, device=device)

    @staticmethod
    def _convert_to_repeated_context(noise: Tensor, extra_args: dict, device=RandDevice.CPU, **kwargs):
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
    def _convert_to_freenoise(noise: Tensor, seed: int, extra_args: dict, device=RandDevice.CPU, **kwargs):
        # if no context_length, return unmodified noise
        opts: ContextOptionsGroup = extra_args["context_options"]
        context_length: int = opts.context_length if not opts.view_options else opts.view_options.context_length
        context_overlap: int = opts.context_overlap if not opts.view_options else opts.view_options.context_overlap
        video_length: int = noise.shape[0]
        if context_length is None:
            return noise
        delta = context_length - context_overlap
        generator, _ = get_generator(RandDevice.CPU, seed) # no point in ever using non-CPU to just shuffle indexes

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
            noised_latents = latents * sqrt_alpha_prod + noise.to(dtype=latents.dtype, device=latents.device) * sqrt_one_minus_alpha_prod
            # 2. create random noise z_rand for high frequency
            temp_sample_settings = sample_settings.clone()
            temp_sample_settings.batch_offset += self.iter_batch_offset * curr_i
            temp_sample_settings.seed_offset += self.iter_seed_offset * curr_i
            z_rand = temp_sample_settings.prepare_noise(seed=seed, latents=latents, noise=None,
                                                    extra_args=noise_extra_args, force_create_noise=True)
            # 3. noise reinitialization - combines low freq. noise from noised_latents and high freq. noise from z_rand
            noised_latents = freeinit.freq_mix_3d(x=noised_latents, noise=z_rand, LPF=self.freq_filter)
            return cached_latents, noised_latents
        elif self.init_type == self.DINKINIT_V1:
            # NOTE: This was my first attempt at implementing FreeInit; it sorta works due to my alpha_cumprod shenanigans,
            #       but completely by accident.
            # 1. apply initial noise with appropriate step sigma
            sigma = self.get_sigma(model, self.step-1000).to(latents.device)
            alpha_cumprod = 1 / ((sigma * sigma) + 1) #1 / ((sigma * sigma)) # 1 / ((sigma * sigma) + 1)
            noised_latents = (latents + (cached_noise.to(dtype=latents.dtype, device=latents.device) * sigma)) * alpha_cumprod
            # 2. create random noise z_rand for high frequency
            temp_sample_settings = sample_settings.clone()
            temp_sample_settings.batch_offset += self.iter_batch_offset * curr_i
            temp_sample_settings.seed_offset += self.iter_seed_offset * curr_i
            z_rand = temp_sample_settings.prepare_noise(seed=seed, latents=latents, noise=None,
                                                    extra_args=noise_extra_args, force_create_noise=True)
            ####z_rand = torch.randn_like(latents, dtype=latents.dtype, device=latents.device)
            # 3. noise reinitialization - combines low freq. noise from noised_latents and high freq. noise from z_rand
            noised_latents = freeinit.freq_mix_3d(x=noised_latents, noise=z_rand, LPF=self.freq_filter)
            return cached_latents, noised_latents
        else:
            raise ValueError(f"FreeInit init_type '{self.init_type}' is not recognized.")


class NoiseCalibration:
    def __init__(self, scale: float=0.5, calib_iterations: int=1):
        self.scale = scale
        self.calib_iterations = calib_iterations
    
    def perform_calibration(self, sample_func: Callable, model: ModelPatcher, latents: Tensor, noise: Tensor, is_custom: bool, args: list, kwargs: dict):
        if is_custom:
            return self._perform_calibration_custom(sample_func=sample_func, model=model, latents=latents, noise=noise, _args=args, _kwargs=kwargs)
        return self._perform_calibration_not_custom(sample_func=sample_func, model=model, latents=latents, noise=noise, args=args, kwargs=kwargs)
    
    def _perform_calibration_custom(self, sample_func: Callable, model: ModelPatcher, latents: Tensor, noise: Tensor, _args: list, _kwargs: dict):
        args = _args.copy()
        kwargs = _kwargs.copy()
        # need to get sigmas to be used in sampling and for noise calc
        sigmas = args[2]
        # use first 2 sigmas as real sigmas (2 sigmas = 1 step)
        sigmas = sigmas[:2]
        args[2] = sigmas
        # divide by scale factor
        sigma = sigmas[0] / (model.model.latent_format.scale_factor)
        alpha_cumprod = 1 / ((sigma * sigma) + 1)
        sqrt_alpha_prod = alpha_cumprod ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod) ** 0.5
        zero_noise = torch.zeros_like(noise)
        new_latents = latents# / (model.model.latent_format.scale_factor)
        #new_latents = latents * (model.model.latent_format.scale_factor)
        for _ in range(self.calib_iterations):
            # TODO: do i need to use DDIM noising, or will ComfyUI's work?
            x = new_latents * sqrt_alpha_prod + noise * sqrt_one_minus_alpha_prod
            #x = latents
            #x = latents + noise * sigma #torch.sqrt(1.0 + sigma ** 2.0)
            # replace latents in args with x
            args[-1] = x
            e_t_theta = sample_func(model, zero_noise, *args, **kwargs) * (model.model.latent_format.scale_factor)
            x_0_t = (x - sqrt_one_minus_alpha_prod * e_t_theta) / sqrt_alpha_prod
            freq_delta = (self.get_low_or_high_fft(x_0_t, self.scale, is_low=False) - self.get_low_or_high_fft(new_latents, self.scale, is_low=False))
            noise = e_t_theta + sqrt_alpha_prod / sqrt_one_minus_alpha_prod * freq_delta
        #return latents, noise
        #x = latents * sqrt_alpha_prod + noise * sqrt_one_minus_alpha_prod
        #return zero_noise, x #noise * (model.model.latent_format.scale_factor)
        return latents, noise# * (model.model.latent_format.scale_factor)

    def _perform_calibration_not_custom(self, sample_func: Callable, model: ModelPatcher, latents: Tensor, noise: Tensor, args: list, kwargs: dict):
        return latents, noise
    
    @staticmethod
    # From NoiseCalibration code at https://github.com/yangqy1110/NC-SDEdit/
    def get_low_or_high_fft(x: Tensor, scale: float, is_low=True):
        # reshape to match intended dims; starts in b c h w, turn into c b h w
        x = rearrange(x, "b c h w -> c b h w")
        # FFT
        x_freq = fft.fftn(x, dim=(-2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))
        C, T, H, W = x_freq.shape
        
        # extract
        if is_low:
            mask = torch.zeros((C, T, H, W), device=x.device)
            crow, ccol = H // 2, W // 2
            mask[..., crow - int(crow * scale):crow + int(crow * scale), ccol - int(ccol * scale):ccol + int(ccol * scale)] = 1
        else:
            mask = torch.ones((C, T, H, W), device=x.device)
            crow, ccol = H // 2, W //2
            mask[..., crow - int(crow * scale):crow + int(crow * scale), ccol - int(ccol * scale):ccol + int(ccol * scale)] = 0
        x_freq = x_freq * mask
        
        # IFFT
        x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
        x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
        # rearrange back to ComfyUI expected dims
        x_filtered = rearrange(x_filtered, "c b h w -> b c h w")
        return x_filtered


class CFGExtras:
    def __init__(self, call_fn: Callable):
        self.call_fn = call_fn


class CFGExtrasGroup:
    def __init__(self):
        self.extras: list[CFGExtras] = []
    
    def add(self, extra: CFGExtras):
        self.extras.append(extra)
    
    def is_empty(self) -> bool:
        return len(self.extras) == 0
    
    def clone(self):
        cloned = CFGExtrasGroup()
        cloned.extras = self.extras.copy()
        return cloned


class CustomCFGKeyframe:
    def __init__(self, cfg_multival: Union[float, Tensor], start_percent=0.0, guarantee_steps=1, cfg_extras: CFGExtrasGroup=None):
        self.cfg_multival = cfg_multival
        self.cfg_extras = cfg_extras
        # scheduling
        self.start_percent = float(start_percent)
        self.start_t = 999999999.9
        self.guarantee_steps = guarantee_steps
    
    def get_effective_guarantee_steps(self, max_sigma: torch.Tensor):
        '''If keyframe starts before current sampling range (max_sigma), treat as 0.'''
        if torch.allclose(self.start_t, max_sigma) or self.start_t < max_sigma:
            return self.guarantee_steps
        return 0

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
        self._previous_t = -1
    
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
            to_assign = torch.tensor(model.model_sampling.percent_to_sigma(keyframe.start_percent), device=model.model_sampling.sigma_max.device)
            if keyframe.start_percent == 0.0 and to_assign > model.model_sampling.sigma_max:
                keyframe.start_t = model.model_sampling.sigma_max
            else:
                keyframe.start_t = to_assign
    
    def prepare_current_keyframe(self, t: Tensor, transformer_options: dict[str, Tensor]):
        curr_t: float = t[0]
        # if curr_t same as before, do nothing as step already accounted for
        if curr_t == self._previous_t:
            return
        prev_index = self._current_index
        max_sigma = torch.max(transformer_options.get("sample_sigmas", BIGMAX_TENSOR))
        # if met guaranteed steps, look for next keyframe in case need to switch
        if self._current_used_steps >= self._current_keyframe.get_effective_guarantee_steps(max_sigma):
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
                        if self._current_keyframe.get_effective_guarantee_steps(max_sigma) > 0:
                            break
                    # if eval_c is outside the percent range, stop looking further
                    else: break
        # update steps current context is used
        self._current_used_steps += 1
        # update previous_t
        self._previous_t = curr_t

    def get_cfg_scale(self, cond: Tensor):
        cond_scale = self.cfg_multival
        if isinstance(cond_scale, Tensor):
            cond_scale = prepare_mask_batch(cond_scale.to(cond.dtype).to(cond.device), cond.shape)
            cond_scale = extend_to_batch_size(cond_scale, cond.shape[0])
        return cond_scale
    
    def get_model_options(self, model_options: dict[str]):
        cfg_extras = self.cfg_extras
        if cfg_extras is not None:
            for extra in cfg_extras.extras:
                model_options = extra.call_fn(model_options)
        return model_options

    def patch_model(self, model: ModelPatcher) -> ModelPatcher:
        # NOTE: no longer used at the moment, as most sampler_cfg_function patches should work with tensor cfg_scales,
        # meaning get_cfg_scale is a direct replacement
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
    
    @property
    def cfg_extras(self):
        if self._current_keyframe != None:
            return self._current_keyframe.cfg_extras
        return None


class NoisedImageInjectOptions:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def clone(self):
        return NoisedImageInjectOptions(x=self.x, y=self.y)


class NoisedImageToInject:
    def __init__(self, image: Tensor, mask: Tensor, vae: VAE, start_percent: float, guarantee_steps: int=1,
                 invert_mask=False, resize_image=True, strength_multival=None,
                 img_inject_opts: NoisedImageInjectOptions=None):
        self.image = image
        self.mask = mask
        self.vae = vae
        self.invert_mask = invert_mask
        self.resize_image = resize_image
        self.strength_multival = 1.0 if strength_multival is None else strength_multival
        if img_inject_opts is None:
            img_inject_opts = NoisedImageInjectOptions()
        self.img_inject_opts = img_inject_opts
        # scheduling
        self.start_percent = float(start_percent)
        self.start_t = 999999999.9
        self.start_timestep = 999
        self.guarantee_steps = guarantee_steps

    def clone(self):
        cloned = NoisedImageToInject(image=self.image, vae=self.vae, start_percent=self.start_percent,
                                     guarantee_steps=self.guarantee_steps, invert_mask=self.invert_mask, resize_image=self.resize_image,
                                     img_inject_opts=self.img_inject_opts)
        cloned.start_t = self.start_t
        cloned.start_timestep = self.start_timestep
        return cloned


class NoisedImageToInjectGroup:
    def __init__(self):
        self.injections: list[NoisedImageToInject] = []
        self._current_index: int = -1
        self._current_used_steps: int = 0
    
    @property
    def current_injection(self):
        return self.injections[self._current_index]

    def reset(self):
        self._current_index = -1
        self._current_used_steps: int = 0

    def add(self, to_inject: NoisedImageToInject):
        # add to end of list, then sort
        self.injections.append(to_inject)
        self.injections = get_sorted_list_via_attr(self.injections, "start_percent")

    def is_empty(self) -> bool:
        return len(self.injections) == 0

    def has_index(self, index: int) -> int:
        return index >=0 and index < len(self.injections)

    def clone(self):
        cloned = NoisedImageToInjectGroup()
        for to_inject in self.injections:
            cloned.injections.append(to_inject)
        return cloned
    
    def initialize_timesteps(self, model: BaseModel):
        for to_inject in self.injections:
            to_inject.start_t = model.model_sampling.percent_to_sigma(to_inject.start_percent)
            to_inject.start_timestep = model.model_sampling.timestep(torch.tensor(to_inject.start_t))

    def ksampler_get_injections(self, model: ModelPatcher, scheduler: str, sampler_name: str, denoise: float, force_full_denoise: bool, start_step: int, last_step: int, total_steps: int) -> tuple[list[list[int]], list[NoisedImageToInject]]:
        actual_last_step = min(last_step, total_steps)
        steps = list(range(start_step, actual_last_step+1))
        # create sampler that will be used to get sigmas
        sampler = comfy.samplers.KSampler(model, steps=total_steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)
        # replicate KSampler.sample function to get the exact sigmas
        sigmas = sampler.sigmas
        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0
        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
            else:
                return [[start_step,actual_last_step], []]
        assert len(steps) == len(sigmas)
        model_sampling = model.get_model_object("model_sampling")
        timesteps = [model_sampling.timestep(x) for x in sigmas]
        # get actual ranges + injections
        ranges, injections = self._prepare_injections(timesteps=timesteps)
        # ranges are given with end-exclusive index, so subtract by 1 to get real step value
        steps_list = [[steps[x[0]],steps[x[1]-1]] for x in ranges]
        return steps_list, injections

    def custom_ksampler_get_injections(self, model: ModelPatcher, sigmas: Tensor) -> tuple[list[list[Tensor]], list[NoisedImageToInject]]:
        model_sampling = model.get_model_object("model_sampling")
        timesteps = []
        for i in range(sigmas.shape[0]):
            timesteps.append(model_sampling.timestep(sigmas[i]))
        # get actual ranges + injections
        ranges, injections = self._prepare_injections(timesteps=timesteps)
        sigmas_list = [sigmas[x[0]:x[1]] for x in ranges]
        return sigmas_list, injections

    def _prepare_injections(self, timesteps: list[Tensor]) -> tuple[list[list[Tensor]], list[NoisedImageToInject]]:
        range_start = timesteps[0]
        range_end = timesteps[-1]
        # if nothing to inject, return all indexes of timesteps and no injections
        if self.is_empty():
            return ([(0, len(timesteps))], [])
        # otherwise, need to populate lists
        timesteps_list: list[list[Tensor]] = []
        injection_list: list[NoisedImageToInject] = []
        remaining_timesteps = timesteps.copy()
        remaining_offset = 0
        # NOTE: timesteps start at 999 and end at 0; the smaller the timestep, the 'later' the step
        for eval_c in self.injections:
            if len(remaining_timesteps) <= 2:
                break
            current_used_steps = 0
            # if start_timestep is greater than range_start, ignore it
            if eval_c.start_timestep > range_start:
                continue
            # if start_timestep is less than range_end, ignore it
            if eval_c.start_timestep < range_end:
                continue
            while current_used_steps < eval_c.guarantee_steps:
                if len(remaining_timesteps) <= 2:
                    break
                # otherwise, make a split in timesteps
                broken_nicely = False
                for i in range(1, len(remaining_timesteps)-1):
                    # if smaller than timestep, look at next timestep
                    if eval_c.start_timestep < remaining_timesteps[i]:
                        continue
                    # if only one timestep would be leftover, then end
                    if len(remaining_timesteps[i:]) < 2:
                        broken_nicely = True
                        break
                    new_timestep_range = (remaining_offset, remaining_offset+i+1)
                    timesteps_list.append(new_timestep_range)
                    injection_list.append(eval_c)
                    current_used_steps += 1
                    remaining_timesteps = remaining_timesteps[i:]
                    remaining_offset += i
                    # expected break
                    broken_nicely = True
                    break
                # did not find a match for the timestep, so should break out of while loop
                if not broken_nicely:
                    break

        # add remaining timestep range
        timesteps_list.append((remaining_offset, remaining_offset+len(remaining_timesteps)))
        # return lists - timesteps list len should be one greater than injection list len (fenceposts problem)
        assert len(timesteps_list) == len(injection_list) + 1
        return timesteps_list, injection_list
