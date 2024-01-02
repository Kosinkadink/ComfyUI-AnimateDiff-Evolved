from collections.abc import Iterable
import torch
from torch import Tensor

import comfy.sample
from comfy.model_patcher import ModelPatcher

from .freeinit import FreeInitFilter, freq_mix_3d, get_freq_filter
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

    LIST = [DEFAULT, CONSTANT, EMPTY, REPEATED_CONTEXT]


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
    def __init__(self, batch_offset: int=0, noise_type: str=None, seed_gen: str=None, seed_offset: int=0, noise_layers: 'NoiseLayerGroup'=None, iteration_opts=None, seed_override:int=None, negative_cond_flipflop=False):
        self.batch_offset = batch_offset
        self.noise_type = noise_type if noise_type is not None else NoiseLayerType.DEFAULT
        self.seed_gen = seed_gen if seed_gen is not None else SeedNoiseGeneration.COMFY
        self.noise_layers = noise_layers if noise_layers else NoiseLayerGroup()
        self.iteration_opts = iteration_opts if iteration_opts else IterationOptions()
        self.seed_offset = seed_offset
        self.seed_override = seed_override
        self.negative_cond_flipflop = negative_cond_flipflop
    
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
        elif seed_gen == cls.AUTO1111:
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
        derivative_noise = SeedNoiseGeneration._create_derivative_noise(final_noise, noise_type=noise_type, extra_args=extra_args)
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
        derivative_noise = SeedNoiseGeneration._create_derivative_noise(final_noise, noise_type=noise_type, extra_args=extra_args)
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
    def _create_derivative_noise(noise: Tensor, noise_type: str, extra_args: dict):
        derivative_func = DERIVATIVE_NOISE_FUNC_MAP.get(noise_type, None)
        if derivative_func is None:
            return None
        return derivative_func(noise=noise, extra_args=extra_args)

    @staticmethod
    def _convert_to_repeated_context(noise: Tensor, extra_args: dict):
        # if no context_length, return unmodified noise
        context_length: int = extra_args["context_length"]
        if context_length is None:
            return noise
        length = noise.shape[0]
        noise = noise[:context_length]
        cat_count = (length // context_length) + 1
        return torch.cat([noise] * cat_count, dim=0)[:length]


DERIVATIVE_NOISE_FUNC_MAP = {NoiseLayerType.REPEATED_CONTEXT: SeedNoiseGeneration._convert_to_repeated_context,}


class IterationOptions:
    SAMPLER = "sampler"

    def __init__(self, iterations: int=1, cache_init_noise=False, cache_init_latents=False):
        self.iterations = iterations
        self.cache_init_noise = cache_init_noise
        self.cache_init_latents = cache_init_latents
        self.need_sampler = False

    def get_sigma(self, model: ModelPatcher, step: int):
        model_sampling = model.model.model_sampling
        if "model_sampling" in model.object_patches:
            model_sampling = model.object_patches["model_sampling"]
        return model_sampling.sigmas[step]

    def initialize(self, latents: Tensor):
        pass

    def preprocess_latents(self, curr_i: int, model: ModelPatcher, latents: Tensor, noise: Tensor, **kwargs):
        return latents, noise


class FreeInitOptions(IterationOptions):
    DINKINIT_V1 = "DinkInit_v1"

    LIST = [DINKINIT_V1]

    def __init__(self, iterations: int, step: int=999, apply_to_1st_iter: bool=False,
                 filter=FreeInitFilter.GAUSSIAN, d_s=0.25, d_t=0.25, n=4):
        super().__init__(iterations=iterations, cache_init_noise=True, cache_init_latents=True)
        self.apply_to_1st_iter = apply_to_1st_iter
        self.step = step
        self.filter = filter
        self.d_s = d_s
        self.d_t = d_t
        self.n = n
        self.freq_filter = None
        self.need_sampler = True

    def initialize(self, latents: Tensor):
        self.freq_filter = get_freq_filter(latents.shape, device=latents.device, filter_type=self.filter,
                                           n=self.n, d_s=self.d_s, d_t=self.d_t)
    
    def preprocess_latents(self, curr_i: int, model: ModelPatcher, latents: Tensor, noise: Tensor, cached_latents: Tensor, cached_noise: Tensor,
                           seed:int, sample_settings: SampleSettings, noise_extra_args: dict, **kwargs):
        # if first iter and should not apply, do nothing
        if curr_i == 0 and not self.apply_to_1st_iter:
            return latents, noise
        # otherwise, do FreeInit stuff - DinkInit_v1
        # 1. apply initial noise with appropriate step sigma
        sigma = self.get_sigma(model, self.step).to(latents.device)
        alpha_cumprod = 1 / ((sigma * sigma) + 1) #1 / ((sigma * sigma)) # 1 / ((sigma * sigma) + 1)
        noised_latents = (latents + (cached_noise * sigma)) * alpha_cumprod
        # 2. create random noise z_rand for high frequency
        z_rand = sample_settings.prepare_noise(seed=seed, latents=latents, noise=None, extra_seed_offset=curr_i,
                                                extra_args=noise_extra_args, force_create_noise=True)
        ####z_rand = torch.randn_like(latents, dtype=latents.dtype, device=latents.device)
        # 3. noise reinitialization - combines low freq. noise from noised_latents and high freq. noise from z_rand
        noise = freq_mix_3d(x=noised_latents, noise=z_rand.to(dtype=latents.dtype, device=latents.device), LPF=self.freq_filter)
        return cached_latents, noise
    
        #noised_latents, noise


        # otherwise, do FreeInit stuff
        # 1. apply initial noise with appropriate step sigma
        sigma = self.get_sigma(model, self.step).to(latents.device)
        #####latents = model.model.process_latent_in(latents)
        latents += noise * sigma
        # 2. create random noise z_rand for high frequency
        # Note: original implementation does not use a generator for this... could this cause repeatability issues?
        z_rand = torch.randn_like(latents, dtype=latents.dtype, device=latents.device)
        # 3. noise reinitialization
        #latents = freq_mix_3d(x=latents, noise=z_rand, LPF=self.freq_filter)
        noise = freq_mix_3d(x=noise, noise=z_rand, LPF=self.freq_filter)
        # treat latents as empty, and freq-mixed latents as noise
        #####latents = model.model.process_latent_out(latents)
        return latents, noise
        ######return torch.zeros_like(latents), latents

        # otherwise, do FreeInit stuff
        # 1. apply initial noise with appropriate step sigma (default=999)
        sigma = self.get_sigma(model, self.step).to(latents.device)
        latents += noise * sigma
        # 2. create random noise z_rand for high frequency
        # Note: original implementation does not use a generator for this... could this cause repeatability issues?
        z_rand = torch.randn_like(latents, dtype=latents.dtype, device=latents.device)
        # 3. noise reinitialization
        latents = freq_mix_3d(x=latents, noise=z_rand, LPF=self.freq_filter)
        return latents, noise
