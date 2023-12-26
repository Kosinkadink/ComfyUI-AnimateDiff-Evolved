import torch
from torch import Tensor

import comfy.sample

from .logger import logger


class NoiseLayerType:
    DEFAULT = "default"
    CONSTANT = "constant"

    LIST = [DEFAULT, CONSTANT]


class NoiseApplication:
    ADD = "add"
    REPLACE = "replace"
    
    LIST = [ADD, REPLACE]


class NoiseNormalize:
    DISABLE = "disable"
    NORMAL = "normal"

    LIST = [DISABLE, NORMAL]


class SampleSettings:
    def __init__(self, seed_gen: str=None, noise_layers: 'NoiseLayerGroup'=None, negative_cond_flipflop=False):
        self.seed_gen = seed_gen if seed_gen is not None else SeedNoiseGeneration.COMFY
        self.noise_layers = noise_layers if noise_layers else NoiseLayerGroup()
        self.negative_cond_flipflop = negative_cond_flipflop
    
    def prepare_noise(self, seed: int, latents: Tensor, noise: Tensor):
        if self.seed_gen == SeedNoiseGeneration.AUTO1111:
            noise = SeedNoiseGeneration.create_noise_auto1111(seed, latents)
        # TODO: apply noise layers
        for noise_layer in self.noise_layers.layers:
            # first, generate new noise matching seed gen override
            layer_noise = noise_layer.create_layer_noise(existing_seed_gen=self.seed_gen, seed=seed, latents=latents)
            # next, get noise after applying layer
            noise = noise_layer.apply_layer_noise(new_noise=layer_noise, old_noise=noise)
        # noise prepared now
        return noise


class NoiseLayer:
    def __init__(self, noise_type: str, seed_gen_override: str, seed_offset: int, seed_override: int=None, mask: Tensor=None):
        self.application: str = NoiseApplication.REPLACE
        self.noise_type = noise_type
        self.seed_gen_override = seed_gen_override
        self.seed_offset = seed_offset
        self.seed_override = seed_override
        self.mask = mask
    
    def create_layer_noise(self, existing_seed_gen: str, seed: int, latents: Tensor) -> Tensor:
        if self.seed_override is not None:
            seed = self.seed_override
        seed += self.seed_offset
        return SeedNoiseGeneration.create_noise(existing_seed_gen=existing_seed_gen, seed_gen=self.seed_gen_override, noise_type=self.noise_type, seed=seed, latents=latents)

    def apply_layer_noise(self, new_noise: Tensor, old_noise: Tensor) -> Tensor:
        return old_noise


class NoiseLayerReplace(NoiseLayer):
    def __init__(self, noise_type: str, seed_gen_override: str, seed_offset: int, seed_override: int=None, mask: Tensor=None):
        super().__init__(noise_type, seed_gen_override, seed_offset, seed_override, mask)
        self.application = NoiseApplication.REPLACE

    def apply_layer_noise(self, new_noise: Tensor, old_noise: Tensor) -> Tensor:
        # if no mask, use new_noise
        if self.mask is None:
            return new_noise
        # replicate noise_mask operations
        return new_noise


class NoiseLayerAdd(NoiseLayer):
    def __init__(self, noise_type: str, seed_gen_override: str, seed_offset: int, seed_override: int=None, mask: Tensor=None,
                 noise_weight=1.0, balance_multiplier=1.0, weighted_average=True, normalize:str = NoiseNormalize.DISABLE):
        super().__init__(noise_type, seed_gen_override, seed_offset, seed_override, mask)
        self.application = NoiseApplication.ADD
        self.noise_weight = noise_weight
        self.balance_multiplier = balance_multiplier
        self.weighted_average = weighted_average
        self.normalize = normalize

    def apply_layer_noise(self, new_noise: Tensor, old_noise: Tensor) -> Tensor:
        # if mask is present, apply mask to new_noise
        if self.mask is not None:
            pass
        if self.weighted_average:
            final_noise = old_noise * (1.0-(self.noise_weight*self.balance_multiplier)) + new_noise * self.noise_weight
        else:
            final_noise = old_noise + new_noise * self.noise_weight
        # TODO: perform normalization as requested
        return final_noise


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
    def create_noise(cls, existing_seed_gen: str, seed_gen: str, noise_type: str, seed: int, latents: Tensor):
        # determine if should use existing type
        if seed_gen == cls.USE_EXISTING:
            seed_gen = existing_seed_gen
        if seed_gen == cls.COMFY:
            return cls.create_noise_comfy(seed, latents, noise_type)
        elif seed_gen == cls.AUTO1111:
            return cls.create_noise_auto1111(seed, latents, noise_type)
        raise ValueError(f"Noise seed_gen {seed_gen} is not recognized.")

    @staticmethod
    def create_noise_comfy(seed: int, latents: Tensor, noise_type: str):
        if noise_type == NoiseLayerType.CONSTANT:
            return SeedNoiseGeneration.create_noise_constant(seed, latents)
        noise = comfy.sample.prepare_noise(latents, seed)
        return noise
    
    @staticmethod
    def create_noise_auto1111(seed: int, latents: Tensor, noise_type: str):
        if noise_type == NoiseLayerType.CONSTANT:
            return SeedNoiseGeneration.create_noise_constant(seed, latents)
        # auto1111 applies growing seeds for a batch
        length = latents.shape[0]
        single_shape = (1, latents.shape[1], latents.shape[2], latents.shape[3])
        all_noises = []
        # i starts at 0
        for i in range(length):
            generator = torch.manual_seed(seed+i)
            all_noises.append(torch.randn(single_shape, dtype=latents.dtype, layout=latents.layout, generator=generator, device="cpu"))
        return torch.cat(all_noises, dim=0)

    @staticmethod
    def create_noise_constant(seed: int, latents: Tensor):
        # constant noise is just the noise for the first latent copied latent-length amount of times
        length = latents.shape[0]
        single_shape = (1, latents.shape[1], latents.shape[2], latents.shape[3])
        generator = torch.manual_seed(seed)
        noise = torch.randn(single_shape, dtype=latents.dtype, layout=latents.layout, generator=generator, device="cpu")
        return torch.cat([noise] * length, dim=0)


class __OLDNoiseType:
    DEFAULT = "default"
    REPEATED = "repeated"
    CONSTANT = "constant"
    CONSTANT_ADDED = "constant_added"
    AUTO1111 = "auto1111"

    LIST = [DEFAULT, REPEATED, CONSTANT, CONSTANT_ADDED, AUTO1111]

    @classmethod
    def prepare_noise(cls, noise_type: str, latents: Tensor, noise: Tensor, context_length: int, seed: int):
        if noise_type == cls.DEFAULT:
            return noise
        elif noise_type == cls.REPEATED:
            return cls.prepare_noise_repeated(latents, noise, context_length, seed)
        elif noise_type == cls.CONSTANT:
            return cls.prepare_noise_constant(latents, noise, context_length, seed)
        elif noise_type == cls.CONSTANT_ADDED:
            new_noise = cls.prepare_noise_constant(latents, noise, context_length, seed)
            noise_weight = 0.2
            return noise * (1.0-(noise_weight/3)) + new_noise * noise_weight
        elif noise_type == cls.AUTO1111:
            return cls.prepare_noise_auto1111(latents, noise, context_length, seed)
        logger.warning(f"Noise type {noise_type} not recognized, proceeding with default noise.")
        return noise

    @classmethod
    def prepare_noise_repeated(cls, latents: Tensor, noise: Tensor, context_length: int, seed: int):
        if not context_length:
            return noise
        length = latents.shape[0]
        generator = torch.manual_seed(seed)
        noise = torch.randn(latents.size(), dtype=latents.dtype, layout=latents.layout, generator=generator, device="cpu")
        noise_set = noise[:context_length]
        cat_count = (length // context_length) + 1
        noise_set = torch.cat([noise_set] * cat_count, dim=0)
        noise_set = noise_set[:length]
        return noise_set

    @classmethod
    def prepare_noise_constant(cls, latents: Tensor, noise: Tensor, context_length: int, seed: int):
        length = latents.shape[0]
        single_shape = (1, latents.shape[1], latents.shape[2], latents.shape[3])
        generator = torch.manual_seed(seed)
        noise = torch.randn(single_shape, dtype=latents.dtype, layout=latents.layout, generator=generator, device="cpu")
        return torch.cat([noise] * length, dim=0)

    @classmethod
    def prepare_noise_auto1111(cls, latents: Tensor, noise: Tensor, context_length: int, seed: int):
        # auto1111 applies growing seeds for a batch
        length = latents.shape[0]
        single_shape = (1, latents.shape[1], latents.shape[2], latents.shape[3])
        all_noises = []
        # i starts at 0
        for i in range(length):
            generator = torch.manual_seed(seed+i)
            all_noises.append(torch.randn(single_shape, dtype=latents.dtype, layout=latents.layout, generator=generator, device="cpu"))
        return torch.cat(all_noises, dim=0)
