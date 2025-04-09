from typing import Union
from torch import Tensor
from collections.abc import Iterable

from comfy.sd import VAE

from .freeinit import FreeInitFilter
from .sample_settings import (FreeInitOptions, IterationOptions, AncestralOptions,
                              NoiseLayerAdd, NoiseLayerAddWeighted, NoiseLayerNormalizedSum, NoiseLayerGroup, NoiseLayerReplace, NoiseLayerType,
                              SeedNoiseGeneration, SampleSettings, NoiseCalibration, NoiseDeterminism,
                              CustomCFGKeyframeGroup, CustomCFGKeyframe, CFGExtrasGroup, CFGExtras,
                              NoisedImageToInjectGroup, NoisedImageToInject, NoisedImageInjectOptions)
from .utils_model import BIGMIN, BIGMAX, MAX_RESOLUTION, SigmaSchedule, InterpolationMethod
from .cfg_extras import perturbed_attention_guidance_patch, rescale_cfg_patch, set_model_options_sampler_cfg_function, set_model_options_post_cfg_function
from .logger import logger


class SampleSettingsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_offset": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "noise_type": (NoiseLayerType.LIST,),
                "seed_gen": (SeedNoiseGeneration.LIST,),
                "seed_offset": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX}),
            },
            "optional": {
                "noise_layers": ("NOISE_LAYERS",),
                "iteration_opts": ("ITERATION_OPTS",),
                "seed_override": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "adapt_denoise_steps": ("BOOLEAN", {"default": False},),
                "custom_cfg": ("CUSTOM_CFG",),
                "sigma_schedule": ("SIGMA_SCHEDULE",),
                "image_inject": ("IMAGE_INJECT",),
                "ancestral_opts": ("ANCESTRAL_OPTS",),
                #"noise_calib": ("NOISE_CALIBRATION",), # TODO: bring back once NoiseCalibration is working
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("SAMPLE_SETTINGS",)
    RETURN_NAMES = ("settings",)
    CATEGORY = "Animate Diff 🎭🅐🅓"
    FUNCTION = "create_settings"

    def create_settings(self, batch_offset: int, noise_type: str, seed_gen: str, seed_offset: int, noise_layers: NoiseLayerGroup=None,
                        iteration_opts: IterationOptions=None, seed_override: int=None, adapt_denoise_steps=False,
                        custom_cfg: CustomCFGKeyframeGroup=None, sigma_schedule: SigmaSchedule=None, image_inject: NoisedImageToInjectGroup=None,
                        noise_calib: NoiseCalibration=None, ancestral_opts=None):
        sampling_settings = SampleSettings(batch_offset=batch_offset, noise_type=noise_type, seed_gen=seed_gen, seed_offset=seed_offset, noise_layers=noise_layers,
                                           iteration_opts=iteration_opts, seed_override=seed_override, adapt_denoise_steps=adapt_denoise_steps,
                                           custom_cfg=custom_cfg, sigma_schedule=sigma_schedule, image_injection=image_inject, noise_calibration=noise_calib,
                                           ancestral_opts=ancestral_opts)
        return (sampling_settings,)


class AncestralOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                #"batch_offset": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "noise_type": (NoiseLayerType.LIST_ANCESTRAL,),
                #"determinism": (NoiseDeterminism._LIST,),
                "seed_offset": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX}),
                #"seed_gen_override": (SeedNoiseGeneration.LIST_WITH_OVERRIDE,),
            },
            "optional": {
                "seed_override": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("ANCESTRAL_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings"
    FUNCTION = "create_ancestral_opts"

    def create_ancestral_opts(self, noise_type: str, seed_offset: int, determinism: str=NoiseDeterminism.DEFAULT, seed_override: int=None):
        if isinstance(seed_override, Iterable):
            raise Exception("Passing in a list of seeds for Ancestral Options is not supported at this time.")
        return (AncestralOptions(noise_type=noise_type, determinism=determinism, seed_offset=seed_offset, seed_override=seed_override),)


class NoiseLayerReplaceNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_offset": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "noise_type": (NoiseLayerType.LIST,),
                "seed_gen_override": (SeedNoiseGeneration.LIST_WITH_OVERRIDE,),
                "seed_offset": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX}),
            },
            "optional": {
                "prev_noise_layers": ("NOISE_LAYERS",),
                "mask_optional": ("MASK",),
                "seed_override": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
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
                "batch_offset": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "noise_type": (NoiseLayerType.LIST,),
                "seed_gen_override": (SeedNoiseGeneration.LIST_WITH_OVERRIDE,),
                "seed_offset": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX}),
                "noise_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.001}),
            },
            "optional": {
                "prev_noise_layers": ("NOISE_LAYERS",),
                "mask_optional": ("MASK",),
                "seed_override": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
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
                "batch_offset": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "noise_type": (NoiseLayerType.LIST,),
                "seed_gen_override": (SeedNoiseGeneration.LIST_WITH_OVERRIDE,),
                "seed_offset": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX}),
                "noise_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.001}),
                "balance_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
            },
            "optional": {
                "prev_noise_layers": ("NOISE_LAYERS",),
                "mask_optional": ("MASK",),
                "seed_override": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
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


class NoiseLayerNormalizedSumNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_offset": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "noise_type": (NoiseLayerType.LIST,),
                "seed_gen_override": (SeedNoiseGeneration.LIST_WITH_OVERRIDE,),
                "seed_offset": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX}),
                "noise_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "prev_noise_layers": ("NOISE_LAYERS",),
                "mask_optional": ("MASK",),
                "seed_override": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
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
        layer = NoiseLayerNormalizedSum(noise_type=noise_type, batch_offset=batch_offset, seed_gen_override=seed_gen_override, seed_offset=seed_offset,
                              seed_override=seed_override, mask=mask_optional,
                              noise_weight=noise_weight)
        prev_noise_layers.add_to_start(layer)
        return (prev_noise_layers,)


class IterationOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "iterations": ("INT", {"default": 1, "min": 1}),
            },
            "optional": {
                "iter_batch_offset": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "iter_seed_offset": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX}),
            }
        }

    RETURN_TYPES = ("ITERATION_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/iteration opts"
    FUNCTION = "create_iter_opts"

    def create_iter_opts(self, iterations: int, iter_batch_offset: int=0, iter_seed_offset: int=0):
        iter_opts = IterationOptions(iterations=iterations, iter_batch_offset=iter_batch_offset, iter_seed_offset=iter_seed_offset)
        return (iter_opts,)


class FreeInitOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "iterations": ("INT", {"default": 2, "min": 1}),
                "filter": (FreeInitFilter.LIST,),
                "d_s": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.001}),
                "d_t": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.001}),
                "n_butterworth": ("INT", {"default": 4, "min": 1, "max": 100},),
                "sigma_step": ("INT", {"default": 999, "min": 1, "max": 999}),
                "apply_to_1st_iter": ("BOOLEAN", {"default": False}),
                "init_type": (FreeInitOptions.LIST,)
            },
            "optional": {
                "iter_batch_offset": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "iter_seed_offset": ("INT", {"default": 1, "min": BIGMIN, "max": BIGMAX}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("ITERATION_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/iteration opts"
    FUNCTION = "create_iter_opts"

    def create_iter_opts(self, iterations: int, filter: str, d_s: float, d_t: float, n_butterworth: int,
                         sigma_step: int, apply_to_1st_iter: bool, init_type: str,
                         iter_batch_offset: int=0, iter_seed_offset: int=1):
        # init_type does nothing for now, not until I add more methods of applying low+high freq noise
        iter_opts = FreeInitOptions(iterations=iterations, step=sigma_step, apply_to_1st_iter=apply_to_1st_iter,
                                    filter=filter, d_s=d_s, d_t=d_t, n=n_butterworth, init_type=init_type,
                                    iter_batch_offset=iter_batch_offset, iter_seed_offset=iter_seed_offset)
        return (iter_opts,)


class NoiseCalibrationNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "calib_iterations": ("INT", {"default": 1, "min": 1, "step": 1}),
                "thresh_freq": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("NOISE_CALIBRATION",)
    RETURN_NAMES = ("NOISE_CALIB",)
    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings"
    FUNCTION = "create_noisecalibration"

    def create_noisecalibration(self, calib_iterations: int, thresh_freq: float):
        noise_calib = NoiseCalibration(scale=thresh_freq, calib_iterations=calib_iterations)
        return (noise_calib,)


class CustomCFGNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cfg_multival": ("MULTIVAL",),
            },
            "optional": {
                "cfg_extras": ("CFG_EXTRAS",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("CUSTOM_CFG",)
    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/custom cfg"
    FUNCTION = "create_custom_cfg"

    def create_custom_cfg(self, cfg_multival: Union[float, Tensor], cfg_extras: CFGExtrasGroup=None):
        keyframe = CustomCFGKeyframe(cfg_multival=cfg_multival, cfg_extras=cfg_extras)
        cfg_custom = CustomCFGKeyframeGroup()
        cfg_custom.add(keyframe)
        return (cfg_custom,)


class CustomCFGSimpleNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "cfg_extras": ("CFG_EXTRAS",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CUSTOM_CFG",)
    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/custom cfg"
    FUNCTION = "create_custom_cfg"

    def create_custom_cfg(self, cfg: float, cfg_extras: CFGExtrasGroup=None):
        return CustomCFGNode.create_custom_cfg(self, cfg_multival=cfg, cfg_extras=cfg_extras)


class CustomCFGKeyframeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cfg_multival": ("MULTIVAL",),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
            },
            "optional": {
                "prev_custom_cfg": ("CUSTOM_CFG",),
                "cfg_extras": ("CFG_EXTRAS",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("CUSTOM_CFG",)
    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/custom cfg"
    FUNCTION = "create_custom_cfg"

    def create_custom_cfg(self, cfg_multival: Union[float, Tensor], start_percent: float=0.0, guarantee_steps: int=1,
                          prev_custom_cfg: CustomCFGKeyframeGroup=None, cfg_extras: CFGExtrasGroup=None):
        if not prev_custom_cfg:
            prev_custom_cfg = CustomCFGKeyframeGroup()
        prev_custom_cfg = prev_custom_cfg.clone()
        keyframe = CustomCFGKeyframe(cfg_multival=cfg_multival, start_percent=start_percent, guarantee_steps=guarantee_steps, cfg_extras=cfg_extras)
        prev_custom_cfg.add(keyframe)
        return (prev_custom_cfg,)


class CustomCFGKeyframeSimpleNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
            },
            "optional": {
                "prev_custom_cfg": ("CUSTOM_CFG",),
                "cfg_extras": ("CFG_EXTRAS",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 10}),
            }
        }
    
    RETURN_TYPES = ("CUSTOM_CFG",)
    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/custom cfg"
    FUNCTION = "create_custom_cfg"

    def create_custom_cfg(self, cfg: float, start_percent: float=0.0, guarantee_steps: int=1,
                          prev_custom_cfg: CustomCFGKeyframeGroup=None, cfg_extras: CFGExtrasGroup=None):
        return CustomCFGKeyframeNode.create_custom_cfg(self, cfg_multival=cfg, start_percent=start_percent,
                                                       guarantee_steps=guarantee_steps, prev_custom_cfg=prev_custom_cfg, cfg_extras=cfg_extras)


class CustomCFGKeyframeInterpolationNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "cfg_start": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_end": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "interpolation": (InterpolationMethod._LIST, ),
                "intervals": ("INT", {"default": 50, "min": 2, "max": 100, "step": 1}),
                "print_keyframes": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_custom_cfg": ("CUSTOM_CFG",),
                "cfg_extras": ("CFG_EXTRAS",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CUSTOM_CFG",)
    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/custom cfg"
    FUNCTION = "create_custom_cfg"

    def create_custom_cfg(self,
                          start_percent: float, end_percent: float,
                          cfg_start: float, cfg_end: float, interpolation: str, intervals: int,
                          prev_custom_cfg: CustomCFGKeyframeGroup=None, cfg_extras: CFGExtrasGroup=None,
                          print_keyframes=False):
        if not prev_custom_cfg:
            prev_custom_cfg = CustomCFGKeyframeGroup()
        prev_custom_cfg = prev_custom_cfg.clone()
        percents = InterpolationMethod.get_weights(num_from=start_percent, num_to=end_percent, length=intervals, method=InterpolationMethod.LINEAR)
        cfgs = InterpolationMethod.get_weights(num_from=cfg_start, num_to=cfg_end, length=intervals, method=interpolation)

        is_first = True
        for percent, cfg in zip(percents, cfgs):
            guarantee_steps = 0
            if is_first:
                guarantee_steps = 1
                is_first = False
            prev_custom_cfg.add(CustomCFGKeyframe(cfg_multival=float(cfg), start_percent=percent, guarantee_steps=guarantee_steps, cfg_extras=cfg_extras))
            if print_keyframes:
                logger.info(f"CustomCFGKeyframe - start_percent:{percent} = {cfg}")
        return (prev_custom_cfg,)
        

class CustomCFGKeyframeFromListNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cfgs_float": ("FLOAT", {"default": -1, "min": -1, "step": 0.001, "forceInput": True}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "print_keyframes": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_custom_cfg": ("CUSTOM_CFG",),
                "cfg_extras": ("CFG_EXTRAS",),
            }
        }

    RETURN_TYPES = ("CUSTOM_CFG",)
    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/custom cfg"
    FUNCTION = "create_custom_cfg"

    def create_custom_cfg(self, cfgs_float: Union[float, list[float]],
                              start_percent: float, end_percent: float,
                              prev_custom_cfg: CustomCFGKeyframeGroup=None, cfg_extras: CFGExtrasGroup=None,
                              print_keyframes=False):
        if not prev_custom_cfg:
            prev_custom_cfg = CustomCFGKeyframeGroup()
        prev_custom_cfg = prev_custom_cfg.clone()
        if type(cfgs_float) in (float, int):
            cfgs_float = [float(cfgs_float)]
        elif isinstance(cfgs_float, Iterable):
            pass
        else:
            raise Exception(f"strengths_float must be either an interable input or a float, but was {type(cfgs_float).__repr__}.")
        percents = InterpolationMethod.get_weights(num_from=start_percent, num_to=end_percent, length=len(cfgs_float), method=InterpolationMethod.LINEAR)
        
        is_first = True
        for percent, cfg in zip(percents, cfgs_float):
            guarantee_steps = 0
            if is_first:
                guarantee_steps = 1
                is_first = False
            prev_custom_cfg.add(CustomCFGKeyframe(cfg_multival=float(cfg), start_percent=percent, guarantee_steps=guarantee_steps, cfg_extras=cfg_extras))
            if print_keyframes:
                logger.info(f"CustomCFGKeyframe - start_percent:{percent} = {cfg}")
        return (prev_custom_cfg,)


class CFGExtrasPAGNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scale_multival": ("MULTIVAL",),
            },
            "optional": {
                "prev_extras": ("CFG_EXTRAS",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("CFG_EXTRAS",)
    FUNCTION = "add_cfg_extras"

    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/cfg extras"

    def add_cfg_extras(self, scale_multival: Union[float, Tensor], prev_extras: CFGExtrasGroup=None):
        if prev_extras is None:
            prev_extras = CFGExtrasGroup()
        prev_extras = prev_extras.clone()

        patch = perturbed_attention_guidance_patch(scale_multival)
        def call_extras(model_options: dict[str]):
            return set_model_options_post_cfg_function(model_options.copy(), patch)
        
        extra = CFGExtras(call_extras)
        prev_extras.add(extra)
        return (prev_extras,)


class CFGExtrasPAGSimpleNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            },
            "optional": {
                "prev_extras": ("CFG_EXTRAS",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("CFG_EXTRAS",)
    FUNCTION = "add_cfg_extras"

    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/cfg extras"

    def add_cfg_extras(self, scale: float, prev_extras: CFGExtrasGroup=None):
        return CFGExtrasPAGNode.add_cfg_extras(self, scale_multival=scale, prev_extras=prev_extras)


class CFGExtrasRescaleCFGNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mult_multival": ("MULTIVAL",),
            },
            "optional": {
                "prev_extras": ("CFG_EXTRAS",),
            }
        }

    RETURN_TYPES = ("CFG_EXTRAS",)
    FUNCTION = "add_cfg_extras"

    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/cfg extras"

    def add_cfg_extras(self, mult_multival: Union[float, Tensor], prev_extras: CFGExtrasGroup=None):
        if prev_extras is None:
            prev_extras = CFGExtrasGroup()
        prev_extras = prev_extras.clone()

        patch = rescale_cfg_patch(mult_multival)
        def call_extras(model_options: dict[str]):
            return set_model_options_sampler_cfg_function(model_options.copy(), patch)
        
        extra = CFGExtras(call_extras)
        prev_extras.add(extra)
        return (prev_extras,)


class CFGExtrasRescaleCFGSimpleNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "multiplier": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "prev_extras": ("CFG_EXTRAS",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 10}),
            }
        }

    RETURN_TYPES = ("CFG_EXTRAS",)
    FUNCTION = "add_cfg_extras"

    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/cfg extras"

    def add_cfg_extras(self, multiplier: float, prev_extras: CFGExtrasGroup=None):
        return CFGExtrasRescaleCFGNode.add_cfg_extras(self, mult_multival=multiplier, prev_extras=prev_extras)


class NoisedImageInjectionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "vae": ("VAE", ),
            },
            "optional": {
                "mask_opt": ("MASK", ),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "resize_image": ("BOOLEAN", {"default": True}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 1, "max": BIGMAX}),
                "img_inject_opts": ("IMAGE_INJECT_OPTIONS", ),
                "strength_multival": ("MULTIVAL", ),
                "prev_image_inject": ("IMAGE_INJECT", ),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE_INJECT",)
    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/image inject"
    FUNCTION = "create_image_inject"

    def create_image_inject(self, image: Tensor, vae: VAE, invert_mask: bool, resize_image: bool, start_percent: float,
                            mask_opt: Tensor=None, strength_multival: Union[float, Tensor]=None, prev_image_inject: NoisedImageToInjectGroup=None, guarantee_steps=1,
                            img_inject_opts=None):
        if not prev_image_inject:
            prev_image_inject = NoisedImageToInjectGroup()
        prev_image_inject = prev_image_inject.clone()
        to_inject = NoisedImageToInject(image=image, mask=mask_opt, vae=vae, invert_mask=invert_mask, resize_image=resize_image, strength_multival=strength_multival,
                                        start_percent=start_percent, guarantee_steps=guarantee_steps,
                                        img_inject_opts=img_inject_opts)
        prev_image_inject.add(to_inject)
        return (prev_image_inject,)


class NoisedImageInjectOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "composite_x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "composite_y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE_INJECT_OPTIONS",)
    RETURN_NAMES = ("IMG_INJECT_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/image inject"
    FUNCTION = "create_image_inject_opts"

    def create_image_inject_opts(self, x=0, y=0):
        return (NoisedImageInjectOptions(x=x, y=y),)
