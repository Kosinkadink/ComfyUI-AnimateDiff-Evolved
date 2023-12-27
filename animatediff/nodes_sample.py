from torch import Tensor

from .sample_settings import NoiseLayerAdd, NoiseLayerAddWeighted, NoiseLayerGroup, NoiseLayerReplace, NoiseLayerType, NoiseNormalize, SeedNoiseGeneration, SampleSettings


class SampleSettingsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_offset": ("INT", {"default": 0, "min": 0}),
                "noise_type": (NoiseLayerType.LIST,),
                "seed_gen": (SeedNoiseGeneration.LIST,),
            },
            "optional": {
                "noise_layers": ("NOISE_LAYERS",),
            }
        }
    
    RETURN_TYPES = ("SAMPLE_SETTINGS",)
    RETURN_NAMES = ("settings",)
    CATEGORY = "Animate Diff 🎭🅐🅓"
    FUNCTION = "create_settings"

    def create_settings(self, batch_offset: int, noise_type: str, seed_gen: str, noise_layers: NoiseLayerGroup=None):
        sampling_settings = SampleSettings(batch_offset=batch_offset, noise_type=noise_type, seed_gen=seed_gen, noise_layers=noise_layers)
        return (sampling_settings,)


class NoiseLayerReplaceNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_offset": ("INT", {"default": 0, "min": 0}),
                "noise_type": (NoiseLayerType.LIST,),
                "seed_gen_override": (SeedNoiseGeneration.LIST_WITH_OVERRIDE,),
                "seed_offset": ("INT", {"default": 0, "min": -999999999999999}),
            },
            "optional": {
                "prev_noise_layers": ("NOISE_LAYERS",),
                "mask_optional": ("MASK",),
                "seed_override": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("NOISE_LAYERS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/noise layers"
    FUNCTION = "create_layers"

    def create_layers(self, batch_offset: int, noise_type: str, seed_gen_override: str, seed_offset: int,
                      prev_noise_layers: NoiseLayerGroup=None, mask_optional: Tensor=None, seed_override: int=None,):
        # prepare prev_noise_layers
        if prev_noise_layers is None:
            prev_noise_layers = NoiseLayerGroup()
        prev_noise_layers = prev_noise_layers.clone()
        # create layer
        layer = NoiseLayerReplace(noise_type=noise_type, batch_offset=batch_offset, seed_gen_override=seed_gen_override, seed_offset=seed_offset,
                                  seed_override=seed_override, mask=mask_optional)
        prev_noise_layers.add_to_start(layer)
        return (prev_noise_layers,)


class NoiseLayerAddNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_offset": ("INT", {"default": 0, "min": 0}),
                "noise_type": (NoiseLayerType.LIST,),
                "seed_gen_override": (SeedNoiseGeneration.LIST_WITH_OVERRIDE,),
                "seed_offset": ("INT", {"default": 0, "min": -999999999999999}),
                "noise_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.001}),
            },
            "optional": {
                "prev_noise_layers": ("NOISE_LAYERS",),
                "mask_optional": ("MASK",),
                "seed_override": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("NOISE_LAYERS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/noise layers"
    FUNCTION = "create_layers"

    def create_layers(self, batch_offset: int, noise_type: str, seed_gen_override: str, seed_offset: int,
                      noise_weight: float,
                      prev_noise_layers: NoiseLayerGroup=None, mask_optional: Tensor=None, seed_override: int=None,):
        # prepare prev_noise_layers
        if prev_noise_layers is None:
            prev_noise_layers = NoiseLayerGroup()
        prev_noise_layers = prev_noise_layers.clone()
        # create layer
        layer = NoiseLayerAdd(noise_type=noise_type, batch_offset=batch_offset, seed_gen_override=seed_gen_override, seed_offset=seed_offset,
                              seed_override=seed_override, mask=mask_optional,
                              noise_weight=noise_weight)
        prev_noise_layers.add_to_start(layer)
        return (prev_noise_layers,)


class NoiseLayerAddWeightedNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_offset": ("INT", {"default": 0, "min": 0}),
                "noise_type": (NoiseLayerType.LIST,),
                "seed_gen_override": (SeedNoiseGeneration.LIST_WITH_OVERRIDE,),
                "seed_offset": ("INT", {"default": 0, "min": -999999999999999}),
                "noise_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.001}),
                "balance_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
            },
            "optional": {
                "prev_noise_layers": ("NOISE_LAYERS",),
                "mask_optional": ("MASK",),
                "seed_override": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("NOISE_LAYERS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/noise layers"
    FUNCTION = "create_layers"

    def create_layers(self, batch_offset: int, noise_type: str, seed_gen_override: str, seed_offset: int,
                      noise_weight: float, balance_multiplier: float,
                      prev_noise_layers: NoiseLayerGroup=None, mask_optional: Tensor=None, seed_override: int=None,):
        # prepare prev_noise_layers
        if prev_noise_layers is None:
            prev_noise_layers = NoiseLayerGroup()
        prev_noise_layers = prev_noise_layers.clone()
        # create layer
        layer = NoiseLayerAddWeighted(noise_type=noise_type, batch_offset=batch_offset, seed_gen_override=seed_gen_override, seed_offset=seed_offset,
                              seed_override=seed_override, mask=mask_optional,
                              noise_weight=noise_weight, balance_multiplier=balance_multiplier)
        prev_noise_layers.add_to_start(layer)
        return (prev_noise_layers,)
