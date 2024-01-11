from pathlib import Path
import torch

import comfy.sample as comfy_sample
from comfy.model_patcher import ModelPatcher

from .context import ContextOptions, ContextSchedules
from .logger import logger
from .model_utils import BetaSchedules, get_available_motion_loras, get_available_motion_models, get_motion_lora_path
from .motion_utils import ADKeyframeGroup, ADKeyframe
from .motion_lora import MotionLoraInfo, MotionLoraList
from .model_injection import (InjectionParams, ModelPatcherAndInjector, MotionModelGroup, MotionModelPatcher, MotionModelSettings,
                              load_motion_module, load_motion_module_gen2, load_motion_lora_as_patches, validate_model_compatibility_gen2)
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
                "m_models": ("M_MODELS",),
                "context_options": ("CONTEXT_OPTIONS",),
                "sample_settings": ("SAMPLE_SETTINGS",),
                #"beta_schedule_override": ("BETA_SCHEDULE",),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Animate Diff 🎭🅐🅓"
    FUNCTION = "use_evolved_sampling"

    def use_evolved_sampling(self, model: ModelPatcher, beta_schedule: str, m_models: MotionModelGroup=None, context_options: ContextOptions=None,
                             sample_settings: SampleSettings=None, beta_schedule_override=None):
        if m_models is not None:
            m_models = m_models.clone()
            # for each motion model, confirm that it is compatible with SD model
            for motion_model in m_models.models:
                validate_model_compatibility_gen2(model=model, motion_model=motion_model)
            # create injection params
            model_name_list = [motion_model.model.mm_info.mm_name for motion_model in m_models.models]
            model_names = ",".join(model_name_list)
            # TODO: check if any apply_v2_properly is set to False
            params = InjectionParams(unlimited_area_hack=False, model_name=model_names)
        else:
            params = InjectionParams()
        # apply context options
        if context_options:
            params.set_context(context_options)
        # need to use a ModelPatcher that supports injection of motion modules into unet
        model = ModelPatcherAndInjector(model)
        model.motion_models = m_models
        model.sample_settings = sample_settings if sample_settings is not None else SampleSettings()
        model.motion_injection_params = params

        # save model_sampling from BetaSchedule as object patch
        # TODO: handle autoselect
        new_model_sampling = BetaSchedules.to_model_sampling(beta_schedule, model)
        if new_model_sampling is not None:
            model.add_object_patch("model_sampling", new_model_sampling)
        
        del m_models
        return (model,)


class ApplyAnimateDiffModelNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_model": ("MOTION_MODEL_ADE",),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "motion_lora": ("MOTION_LORA",),
                "scale_multival": ("MULTIVAL",),
                "effect_multival": ("MULTIVAL",),
                "ad_keyframes": ("AD_KEYFRAMES",),
                "prev_m_models": ("M_MODELS",),
            }
        }
    
    RETURN_TYPES = ("M_MODELS",)
    CATEGORY = "Animate Diff 🎭🅐🅓"
    FUNCTION = "apply_motion_model"

    def apply_motion_model(self, motion_model: MotionModelPatcher, start_percent: float=0.0, end_percent: float=1.0,
                           motion_lora: MotionLoraList=None, ad_keyframes: ADKeyframeGroup=None,
                           scale_multival=None, effect_multival=None,
                           prev_m_models: MotionModelGroup=None,):
        # set up motion models list
        if prev_m_models is None:
            prev_m_models = MotionModelGroup()
        prev_m_models = prev_m_models.clone()
        motion_model = motion_model.clone()
        # apply motion model to loaded_mm
        if motion_lora is not None:
            for lora in motion_lora.loras:
                load_motion_lora_as_patches(motion_model, lora)
        motion_model.scale_multival = scale_multival
        motion_model.effect_multival = effect_multival
        motion_model.keyframes = ad_keyframes.clone() if ad_keyframes else ADKeyframeGroup()
        motion_model.timestep_percent_range = (start_percent, end_percent)
        # add to beginning, so that after injection, it will be the earliest of prev_m_models to be run
        prev_m_models.add_to_start(mm=motion_model)
        return (prev_m_models,)
        

class ApplyAnimateDiffModelBasicNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_model": ("MOTION_MODEL_ADE",),
            },
            "optional": {
                "motion_lora": ("MOTION_LORA",),
                "scale_multival": ("MULTIVAL",),
                "effect_multival": ("MULTIVAL",),
                "ad_keyframes": ("AD_KEYFRAMES",),
            }
        }
    
    RETURN_TYPES = ("M_MODELS",)
    CATEGORY = "Animate Diff 🎭🅐🅓"
    FUNCTION = "apply_motion_model"

    def apply_motion_model(self,
                           motion_model: MotionModelPatcher, motion_lora: MotionLoraList=None,
                           scale_multival=None, effect_multival=None, ad_keyframes=None):
        # just a subset of normal ApplyAnimateDiffModelNode inputs
        return ApplyAnimateDiffModelNode.apply_motion_model(self, motion_model, motion_lora=motion_lora,
                                                            scale_multival=scale_multival, effect_multival=effect_multival,
                                                            ad_keyframes=ad_keyframes)


class LoadAnimateDiffModelNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_available_motion_models(),),
            },
            "optional": {
                "ad_settings": ("MOTION_MODEL_SETTINGS",),
            }
        }

    RETURN_TYPES = ("MOTION_MODEL_ADE",)
    RETURN_NAMES = ("MOTION_MODEL",)
    CATEGORY = "Animate Diff 🎭🅐🅓"
    FUNCTION = "load_motion_model"

    def load_motion_model(self, model_name: str, ad_settings: MotionModelSettings=None):
        # load motion module and motion settings, if included
        motion_model = load_motion_module_gen2(model_name=model_name, motion_model_settings=ad_settings)
        return (motion_model,)


class ADKeyframeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}, ),
            },
            "optional": {
                "prev_ad_keyframes": ("AD_KEYFRAMES", ),
                "scale_multival": ("MULTIVAL",),
                "effect_multival": ("MULTIVAL",),
                "inherit_missing": ("BOOLEAN", {"default": True}, ),
                "guarantee_usage": ("BOOLEAN", {"default": True}, ),
            }
        }
    
    RETURN_TYPES = ("AD_KEYFRAMES", )
    FUNCTION = "load_keyframe"

    CATEGORY = "Animate Diff 🎭🅐🅓"

    def load_keyframe(self,
                      start_percent: float, prev_ad_keyframes=None,
                      scale_multival: [float, torch.Tensor]=None, effect_multival: [float, torch.Tensor]=None,
                      inherit_missing: bool=True, guarantee_usage: bool=True):
        if not prev_ad_keyframes:
            prev_ad_keyframes = ADKeyframeGroup()
        prev_ad_keyframes.clone()
        keyframe = ADKeyframe(start_percent=start_percent, scale_multival=scale_multival, effect_multival=effect_multival,
                              inherit_missing=inherit_missing, guarantee_usage=guarantee_usage)
        prev_ad_keyframes.add(keyframe)
        return (prev_ad_keyframes,)
