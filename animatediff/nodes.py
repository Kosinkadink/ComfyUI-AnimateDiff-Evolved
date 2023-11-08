import torch

import comfy.sample as comfy_sample
from comfy.model_patcher import ModelPatcher

from .context import ContextOptions, ContextSchedules, UniformContextOptions
from .logger import logger
from .model_utils import get_available_motion_loras, get_available_motion_models, BetaSchedules
from .motion_lora import MotionLoRAInfo, MotionLoRAList
from .motion_module import InjectionParams, MotionModelSettings
from .motion_module import inject_params_into_model, load_motion_lora, load_motion_module
from .sampling import animatediff_sample_factory

from .nodes_extras import AnimateDiffUnload, EmptyLatentImageLarge, CheckpointLoaderSimpleWithNoiseSelect
from .nodes_experimental import AnimateDiffModelSettingsSimple, AnimateDiffModelSettingsAdvanced, AnimateDiffModelSettingsAdvancedAttnStrengths
from .nodes_deprecated import AnimateDiffLoader_Deprecated, AnimateDiffLoaderAdvanced_Deprecated, AnimateDiffCombine_Deprecated

# override comfy_sample.sample with animatediff-support version
comfy_sample.sample = animatediff_sample_factory(comfy_sample.sample)


class AnimateDiffModelSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min_motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "max_motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
            },
            "optional": {
                "mask_motion_scale": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("MOTION_MODEL_SETTINGS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/motion settings"
    FUNCTION = "get_motion_model_settings"

    def get_motion_model_settings(self, mask_motion_scale: torch.Tensor=None, min_motion_scale: float=1.0, max_motion_scale: float=1.0):
        motion_model_settings = MotionModelSettings(
            mask_attn_scale=mask_motion_scale,
            mask_attn_scale_min=min_motion_scale,
            mask_attn_scale_max=max_motion_scale,
            )

        return (motion_model_settings,)



class AnimateDiffLoRALoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (get_available_motion_loras(),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            },
            "optional": {
                "prev_motion_lora": ("MOTION_LORA",),
            }
        }
    
    RETURN_TYPES = ("MOTION_LORA",)
    CATEGORY = "Animate Diff 🎭🅐🅓"
    FUNCTION = "load_motion_lora"

    def load_motion_lora(self, lora_name: str, strength: float, prev_motion_lora: MotionLoRAList=None):
        if prev_motion_lora is None:
            prev_motion_lora = MotionLoRAList()
        else:
            prev_motion_lora = prev_motion_lora.clone()
        # load lora
        lora = load_motion_lora(lora_name)
        lora_info = MotionLoRAInfo(name=lora_name, strength=strength, hash=lora.hash)
        prev_motion_lora.add_lora(lora_info)

        return (prev_motion_lora,)


class AnimateDiffLoaderWithContext:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "model_name": (get_available_motion_models(),),
                "beta_schedule": (BetaSchedules.get_alias_list_with_first_element(BetaSchedules.SQRT_LINEAR),),
                #"apply_mm_groupnorm_hack": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "context_options": ("CONTEXT_OPTIONS",),
                "motion_lora": ("MOTION_LORA",),
                "motion_model_settings": ("MOTION_MODEL_SETTINGS",),
                "motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "apply_v2_models_properly": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Animate Diff 🎭🅐🅓"
    FUNCTION = "load_mm_and_inject_params"


    def load_mm_and_inject_params(self,
        model: ModelPatcher,
        model_name: str, beta_schedule: str,# apply_mm_groupnorm_hack: bool,
        context_options: ContextOptions=None, motion_lora: MotionLoRAList=None, motion_model_settings: MotionModelSettings=None,
        motion_scale: float=1.0, apply_v2_models_properly: bool=False,
    ):
        # load motion module
        mm = load_motion_module(model_name, motion_lora, model=model, motion_model_settings=motion_model_settings)
        # set injection params
        injection_params = InjectionParams(
                video_length=None,
                unlimited_area_hack=False,
                apply_mm_groupnorm_hack=True,
                beta_schedule=beta_schedule,
                injector=mm.injector_version,
                model_name=model_name,
                apply_v2_models_properly=apply_v2_models_properly,
        )
        if context_options:
            # set context settings TODO: make this dynamic for future purposes
            if type(context_options) == UniformContextOptions:
                injection_params.set_context(
                        context_length=context_options.context_length,
                        context_stride=context_options.context_stride,
                        context_overlap=context_options.context_overlap,
                        context_schedule=context_options.context_schedule,
                        closed_loop=context_options.closed_loop,
                        sync_context_to_pe=context_options.sync_context_to_pe
                )
        if motion_lora:
            injection_params.set_loras(motion_lora)
        # set motion_scale and motion_model_settings
        if not motion_model_settings:
            motion_model_settings = MotionModelSettings()
        motion_model_settings.attn_scale = motion_scale
        injection_params.set_motion_model_settings(motion_model_settings)
        # inject for use in sampling code
        model = inject_params_into_model(model, injection_params)

        return (model,)


class AnimateDiffUniformContextOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_length": ("INT", {"default": 16, "min": 1, "max": 32}), # keep an eye on these max values
                "context_stride": ("INT", {"default": 1, "min": 1, "max": 32}),  # would need to be updated
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": 32}), # if new motion modules come out
                "context_schedule": (ContextSchedules.CONTEXT_SCHEDULE_LIST,),
                "closed_loop": ("BOOLEAN", {"default": False},),
                #"sync_context_to_pe": ("BOOLEAN", {"default": False},),
            },
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    CATEGORY = "Animate Diff 🎭🅐🅓"
    FUNCTION = "create_options"

    def create_options(self, context_length: int, context_stride: int, context_overlap: int, context_schedule: int, closed_loop: bool):
        context_options = UniformContextOptions(
            context_length=context_length,
            context_stride=context_stride,
            context_overlap=context_overlap,
            context_schedule=context_schedule,
            closed_loop=closed_loop
            )
        #context_options.set_sync_context_to_pe(sync_context_to_pe)
        return (context_options,)


NODE_CLASS_MAPPINGS = {
    "ADE_AnimateDiffUniformContextOptions": AnimateDiffUniformContextOptions,
    "ADE_AnimateDiffLoaderWithContext": AnimateDiffLoaderWithContext,
    "ADE_AnimateDiffLoRALoader": AnimateDiffLoRALoader,
    "ADE_AnimateDiffModelSettings_Release": AnimateDiffModelSettings,
    # Experimental Nodes
    "ADE_AnimateDiffModelSettingsSimple": AnimateDiffModelSettingsSimple,
    "ADE_AnimateDiffModelSettings": AnimateDiffModelSettingsAdvanced,
    "ADE_AnimateDiffModelSettingsAdvancedAttnStrengths": AnimateDiffModelSettingsAdvancedAttnStrengths,
    # Extras Nodes
    "ADE_AnimateDiffUnload": AnimateDiffUnload,
    "ADE_EmptyLatentImageLarge": EmptyLatentImageLarge,
    "CheckpointLoaderSimpleWithNoiseSelect": CheckpointLoaderSimpleWithNoiseSelect,
    # Deprecated Nodes
    "AnimateDiffLoaderV1": AnimateDiffLoader_Deprecated,
    "ADE_AnimateDiffLoaderV1Advanced": AnimateDiffLoaderAdvanced_Deprecated,
    "ADE_AnimateDiffCombine": AnimateDiffCombine_Deprecated,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ADE_AnimateDiffUniformContextOptions": "Uniform Context Options 🎭🅐🅓",
    "ADE_AnimateDiffLoaderWithContext": "AnimateDiff Loader 🎭🅐🅓",
    "ADE_AnimateDiffLoRALoader": "AnimateDiff LoRA Loader 🎭🅐🅓",
    "ADE_AnimateDiffModelSettings_Release": "Motion Model Settings 🎭🅐🅓",
    # Experimental Nodes
    "ADE_AnimateDiffModelSettingsSimple": "EXP Motion Model Settings (Simple) 🎭🅐🅓",
    "ADE_AnimateDiffModelSettings": "EXP Motion Model Settings (Advanced) 🎭🅐🅓",
    "ADE_AnimateDiffModelSettingsAdvancedAttnStrengths": "EXP Motion Model Settings (Adv. Attn) 🎭🅐🅓",
    # Extras Nodes
    "ADE_AnimateDiffUnload": "AnimateDiff Unload 🎭🅐🅓",
    "ADE_EmptyLatentImageLarge": "Empty Latent Image (Big Batch) 🎭🅐🅓",
    "CheckpointLoaderSimpleWithNoiseSelect": "Load Checkpoint w/ Noise Select 🎭🅐🅓",
    # Deprecated Nodes
    "AnimateDiffLoaderV1": "AnimateDiff Loader [DEPRECATED] 🎭🅐🅓",
    "ADE_AnimateDiffLoaderV1Advanced": "AnimateDiff Loader (Advanced) [DEPRECATED] 🎭🅐🅓",
    "ADE_AnimateDiffCombine": "DO NOT USE, USE VideoCombine from ComfyUI-VideoHelperSuite instead! AnimateDiff Combine [DEPRECATED, DO NOT USE] 🎭🅐🅓",
}
