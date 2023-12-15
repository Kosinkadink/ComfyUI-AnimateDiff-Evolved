import torch
from torch import Tensor

from .logger import logger


class SamplingSettings:
    def __init__(self):
        self.seed_noise_type: str = SeedNoiseType.DEFAULT
        self.noise_layers = NoiseLayerGroup()


class NoiseLayerType:
    DEFAULT = "default"
    CONSTANT = "constant"


class NoiseLayer:
    def __init__(self):
        self.mask: Tensor = None
        self.layer_type: str = NoiseLayerType.DEFAULT
        self.options: dict = {}


class NoiseLayerGroup:
    def __init__(self):
        self.layers = list[NoiseLayer] = []
    
    def add(self, layer: NoiseLayer) -> None:
        # add to the end of list
        self.layers.append(layer)

    def __getitem__(self, index) -> NoiseLayer:
        return self.layers[index]
    
    def is_empty(self) -> bool:
        return len(self.layers) == 0
    
    def clone(self) -> 'NoiseLayerGroup':
        cloned = NoiseLayerGroup()
        for layer in self.layers:
            cloned.add(layer)
        return cloned


class SeedNoiseType:
    DEFAULT = "default"
    AUTO1111 = "auto1111"

    LIST = [DEFAULT, AUTO1111]

    @classmethod
    def prepare_noise(cls, noise_type: str, latents: Tensor, noise: Tensor, context_length: int, seed: int):
        if noise_type == cls.DEFAULT:
            return noise
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
