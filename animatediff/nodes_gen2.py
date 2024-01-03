from pathlib import Path
import torch

import comfy.sample as comfy_sample
from comfy.model_patcher import ModelPatcher

from .context import ContextOptions, ContextSchedules, UniformContextOptions
from .logger import logger
from .model_utils import BetaSchedules, get_available_motion_loras, get_available_motion_models, get_motion_lora_path
from .motion_lora import MotionLoraInfo, MotionLoraList
from .model_injection import InjectionParams, ModelPatcherAndInjector, MotionModelGroup, MotionModelPatcher, MotionModelSettings, load_motion_module, load_motion_module_gen2, load_motion_lora_as_patches
from .sample_settings import SampleSettings, SeedNoiseGeneration
from .sampling import motion_sample_factory


class UseEvolvedSamplingNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "beta_schedule": (BetaSchedules.ALIAS_LIST, {"default": BetaSchedules.SQRT_LINEAR}),
            },
            "optional": {
                "motion_models": ("M_MODELS",),
                "beta_schedule_override": ("BETA_SCHEDULE",),
                "context_options": ("CONTEXT_OPTIONS",),
                "sample_settings": ("SAMPLE_SETTINGS",),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Animate Diff 🎭🅐🅓"
    FUNCTION = "use_evolved_sampling"

    def use_evolved_sampling(self, model: ModelPatcher, motion_models=None, beta_schedule_override=None,
                               context_options: ContextOptions=None, sample_settings: SampleSettings=None):
        # need to use a ModelPatcher that supports injection of motion modules into unet
        model = ModelPatcherAndInjector(model)
        model.sample_settings = sample_settings if sample_settings is not None else SampleSettings()


class ApplyAnimateDiffModelNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_m": ("MOTION_M",),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "motion_lora": ("MOTION_LORA",),
                "mm_timestep_kf": ("MM_TIMESTEP_KF",),
                "strength_multival": ("MULTIVAL",),
                "scale_multival": ("MULTIVAL",),
                "prev_m_models": ("M_MODELS",),
                "apply_v2_properly": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("M_MODELS",)
    CATEGORY = "Animate Diff 🎭🅐🅓"
    FUNCTION = "apply_motion_model"

    def apply_motion_model(self, motion_m: MotionModelPatcher, start_percent: float=0.0, end_percent: float=1.0,
                           motion_lora: MotionLoraList=None, mm_timestep_kf=None,
                           strength_multival=None, scale_multival=None,
                           prev_m_models: MotionModelGroup=None, apply_v2_properly=True,):
        # set up motion models list
        if prev_m_models is None:
            prev_m_models = MotionModelGroup()
        prev_m_models.clone()
        # apply motion model to loaded_mm
        if motion_lora is not None:
            for lora in motion_lora.loras:
                load_motion_lora_as_patches(motion_m, lora)


class ApplyAnimateDiffModelBasicNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "loaded_mm": ("LOADED_MM",),
            },
            "optional": {
                "motion_lora": ("MOTION_LORA",),
                "mm_timestep_kf": ("MM_TIMESTEP_KF",),
                "scale_multival": ("MULTIVAL",),
                "prev_m_models": ("MOTION_M",),
                "apply_v2_properly": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MOTION_MODELS",)
    CATEGORY = "Animate Diff 🎭🅐🅓"
    FUNCTION = "apply_motion_model"

    def apply_motion_model(self, motion_m: MotionModelPatcher,
                           motion_lora: MotionLoraList=None, mm_timestep_kf=None,
                           scale_multival=None,
                           prev_m_models=None, apply_v2_properly=True,):
        # just a subset of normal ApplyAnimateDiffModelNode inputs
        return ApplyAnimateDiffModelNode.apply_motion_model(self, motion_m, motion_lora=motion_lora,
                                                            mm_timestep_kf=mm_timestep_kf, scale_multival=scale_multival,
                                                            apply_v2_properly=apply_v2_properly)


class LoadAnimateDiffModelNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_available_motion_models(),),
            },
            "optional": {
                "mm_settings": ("MOTION_MODEL_SETTINGS",),
            }
        }

    RETURN_TYPES = ("MOTION_M",)
    CATEGORY = "Animate Diff 🎭🅐🅓"
    FUNCTION = "load_motion_model"

    def load_motion_model(self, model_name: str, mm_settings: MotionModelSettings=None):
        # load motion module and motion settings, if included
        motion_model = load_motion_module_gen2(model_name=model_name, motion_model_settings=mm_settings)
        return (motion_model,)
