from pathlib import Path
import torch

import comfy.sample as comfy_sample
from comfy.model_patcher import ModelPatcher

from .ad_settings import AdjustGroup, AnimateDiffSettings, AdjustPE, AdjustWeight
from .context import ContextOptions, ContextOptionsGroup, ContextSchedules
from .logger import logger
from .utils_model import BetaSchedules, get_available_motion_loras, get_available_motion_models, get_motion_lora_path
from .utils_motion import ADKeyframeGroup, get_combined_multival
from .motion_lora import MotionLoraInfo, MotionLoraList
from .model_injection import InjectionParams, ModelPatcherAndInjector, MotionModelGroup, load_motion_lora_as_patches, load_motion_module_gen1, load_motion_module_gen2, validate_model_compatibility_gen2
from .sample_settings import SampleSettings, SeedNoiseGeneration
from .sampling import motion_sample_factory


class AnimateDiffLoaderGen1:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "model_name": (get_available_motion_models(),),
                "beta_schedule": (BetaSchedules.ALIAS_LIST, {"default": BetaSchedules.AUTOSELECT}),
                #"apply_mm_groupnorm_hack": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "context_options": ("CONTEXT_OPTIONS",),
                "motion_lora": ("MOTION_LORA",),
                "ad_settings": ("AD_SETTINGS",),
                "ad_keyframes": ("AD_KEYFRAMES",),
                "sample_settings": ("SAMPLE_SETTINGS",),
                "scale_multival": ("MULTIVAL",),
                "effect_multival": ("MULTIVAL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/① Gen1 nodes ①"
    FUNCTION = "load_mm_and_inject_params"

    def load_mm_and_inject_params(self,
        model: ModelPatcher,
        model_name: str, beta_schedule: str,# apply_mm_groupnorm_hack: bool,
        context_options: ContextOptionsGroup=None, motion_lora: MotionLoraList=None, ad_settings: AnimateDiffSettings=None,
        sample_settings: SampleSettings=None, scale_multival=None, effect_multival=None, ad_keyframes: ADKeyframeGroup=None,
    ):
        # load motion module and motion settings, if included
        motion_model = load_motion_module_gen2(model_name=model_name, motion_model_settings=ad_settings)
        # confirm that it is compatible with SD model
        validate_model_compatibility_gen2(model=model, motion_model=motion_model)
        # apply motion model to loaded_mm
        if motion_lora is not None:
            for lora in motion_lora.loras:
                load_motion_lora_as_patches(motion_model, lora)
        motion_model.scale_multival = scale_multival
        motion_model.effect_multival = effect_multival
        motion_model.keyframes = ad_keyframes.clone() if ad_keyframes else ADKeyframeGroup()
        
        # create injection params
        params = InjectionParams(unlimited_area_hack=False, model_name=motion_model.model.mm_info.mm_name)
        # apply context options
        if context_options:
            params.set_context(context_options)

        # set motion_scale and motion_model_settings
        if not ad_settings:
            ad_settings = AnimateDiffSettings()
        ad_settings.attn_scale = 1.0
        params.set_motion_model_settings(ad_settings)

        # backwards compatibility to support old way of masking scale
        if params.motion_model_settings.mask_attn_scale is not None:
            motion_model.scale_multival = get_combined_multival(scale_multival, (params.motion_model_settings.mask_attn_scale * params.motion_model_settings.attn_scale))
        
        # need to use a ModelPatcher that supports injection of motion modules into unet
        # need to use a ModelPatcher that supports injection of motion modules into unet
        model = ModelPatcherAndInjector.create_from(model, hooks_only=True)
        model.motion_models = MotionModelGroup(motion_model)
        model.sample_settings = sample_settings if sample_settings is not None else SampleSettings()
        model.motion_injection_params = params
        
        if model.sample_settings.custom_cfg is not None:
            logger.info("[Sample Settings] custom_cfg is set; will override any KSampler cfg values or patches.")

        if model.sample_settings.sigma_schedule is not None:
            logger.info("[Sample Settings] sigma_schedule is set; will override beta_schedule.")
            model.add_object_patch("model_sampling", model.sample_settings.sigma_schedule.clone().model_sampling)
        else:
            # save model sampling from BetaSchedule as object patch
            # if autoselect, get suggested beta_schedule from motion model
            if beta_schedule == BetaSchedules.AUTOSELECT and not model.motion_models.is_empty():
                beta_schedule = model.motion_models[0].model.get_best_beta_schedule(log=True)
            new_model_sampling = BetaSchedules.to_model_sampling(beta_schedule, model)
            if new_model_sampling is not None:
                model.add_object_patch("model_sampling", new_model_sampling)

        del motion_model
        return (model,)


class LegacyAnimateDiffLoaderWithContext:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "model_name": (get_available_motion_models(),),
                "beta_schedule": (BetaSchedules.ALIAS_LIST, {"default": BetaSchedules.AUTOSELECT}),
                #"apply_mm_groupnorm_hack": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "context_options": ("CONTEXT_OPTIONS",),
                "motion_lora": ("MOTION_LORA",),
                "ad_settings": ("AD_SETTINGS",),
                "sample_settings": ("SAMPLE_SETTINGS",),
                "motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "apply_v2_models_properly": ("BOOLEAN", {"default": True}),
                "ad_keyframes": ("AD_KEYFRAMES",),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/① Gen1 nodes ①"
    FUNCTION = "load_mm_and_inject_params"


    def load_mm_and_inject_params(self,
        model: ModelPatcher,
        model_name: str, beta_schedule: str,# apply_mm_groupnorm_hack: bool,
        context_options: ContextOptionsGroup=None, motion_lora: MotionLoraList=None, ad_settings: AnimateDiffSettings=None, motion_model_settings: AnimateDiffSettings=None,
        sample_settings: SampleSettings=None, motion_scale: float=1.0, apply_v2_models_properly: bool=False, ad_keyframes: ADKeyframeGroup=None,
    ):
        if ad_settings is not None:
            motion_model_settings = ad_settings
        # load motion module
        motion_model = load_motion_module_gen1(model_name, model, motion_lora=motion_lora, motion_model_settings=motion_model_settings)
        # set injection params
        params = InjectionParams(
                unlimited_area_hack=False,
                model_name=model_name,
                apply_v2_properly=apply_v2_models_properly,
        )
        if context_options:
            params.set_context(context_options)
        # set motion_scale and motion_model_settings
        if not motion_model_settings:
            motion_model_settings = AnimateDiffSettings()
        motion_model_settings.attn_scale = motion_scale
        params.set_motion_model_settings(motion_model_settings)

        if params.motion_model_settings.mask_attn_scale is not None:
            motion_model.scale_multival = params.motion_model_settings.mask_attn_scale * params.motion_model_settings.attn_scale
        else:
            motion_model.scale_multival = params.motion_model_settings.attn_scale

        motion_model.keyframes = ad_keyframes.clone() if ad_keyframes else ADKeyframeGroup()

        model = ModelPatcherAndInjector.create_from(model, hooks_only=True)
        model.motion_models = MotionModelGroup(motion_model)
        model.sample_settings = sample_settings if sample_settings is not None else SampleSettings()
        model.motion_injection_params = params

        # save model sampling from BetaSchedule as object patch
        # if autoselect, get suggested beta_schedule from motion model
        if beta_schedule == BetaSchedules.AUTOSELECT and not model.motion_models.is_empty():
            beta_schedule = model.motion_models[0].model.get_best_beta_schedule(log=True)
        new_model_sampling = BetaSchedules.to_model_sampling(beta_schedule, model)
        if new_model_sampling is not None:
            model.add_object_patch("model_sampling", new_model_sampling)

        del motion_model
        return (model,)
