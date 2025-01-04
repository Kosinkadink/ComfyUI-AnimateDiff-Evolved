from typing import Union
import torch

from comfy.model_patcher import ModelPatcher

from .ad_settings import AnimateDiffSettings
from .context import ContextOptionsGroup
from .logger import logger
from .utils_model import BIGMAX, BetaSchedules, get_available_motion_models
from .utils_motion import ADKeyframeGroup, ADKeyframe, InputPIA, AllPerBlocks
from .motion_lora import MotionLoraList
from .model_injection import (ModelPatcherHelper,
                              InjectionParams, MotionModelGroup, MotionModelPatcher, get_mm_attachment, create_fresh_motion_module,
                              load_motion_module_gen2, load_motion_lora_as_patches, validate_model_compatibility_gen2,
                              validate_per_block_compatibility, validate_per_block_compatibility_keyframes)
from .sample_settings import SampleSettings
from .sampling import outer_sample_wrapper, sliding_calc_cond_batch


class UseEvolvedSamplingNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "beta_schedule": (BetaSchedules.ALIAS_LIST, {"default": BetaSchedules.AUTOSELECT}),
            },
            "optional": {
                "m_models": ("M_MODELS",),
                "context_options": ("CONTEXT_OPTIONS",),
                "sample_settings": ("SAMPLE_SETTINGS",),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②"
    FUNCTION = "use_evolved_sampling"

    def use_evolved_sampling(self, model: ModelPatcher, beta_schedule: str, m_models: MotionModelGroup=None, context_options: ContextOptionsGroup=None,
                             sample_settings: SampleSettings=None):
        model = model.clone()
        helper = ModelPatcherHelper(model)
        params = InjectionParams()
        if m_models is not None:
            m_models = m_models.clone()
            # for each motion model, confirm that it is compatible with SD model
            for motion_model in m_models.models:
                validate_model_compatibility_gen2(model=model, motion_model=motion_model)
            # TODO: check if any apply_v2_properly is set to False
        # apply context options
        if context_options:
            params.set_context(context_options)
        
        # attach all properties to model to enable AnimateDiff functionality
        helper.set_all_properties(
            outer_sampler_wrapper=outer_sample_wrapper,
            calc_cond_batch_wrapper=sliding_calc_cond_batch,
            params=params,
            sample_settings=sample_settings,
            motion_models=m_models,
        )

        sample_settings = helper.get_sample_settings()
        if sample_settings.custom_cfg is not None:
            logger.info("[Sample Settings] custom_cfg is set; will override any KSampler cfg values or patches.")

        if sample_settings.sigma_schedule is not None:
            logger.info("[Sample Settings] sigma_schedule is set; will override beta_schedule.")
            model.add_object_patch("model_sampling", sample_settings.sigma_schedule.clone().model_sampling)
        else:
            # save model_sampling from BetaSchedule as object patch
            # if autoselect, get suggested beta_schedule from motion model
            if beta_schedule == BetaSchedules.AUTOSELECT:
                if helper.get_motion_models():
                    beta_schedule = helper.get_motion_models()[0].model.get_best_beta_schedule(log=True)
                else:
                    beta_schedule = BetaSchedules.USE_EXISTING
                    
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
                "per_block": ("PER_BLOCK",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("M_MODELS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②"
    FUNCTION = "apply_motion_model"

    def apply_motion_model(self, motion_model: MotionModelPatcher, start_percent: float=0.0, end_percent: float=1.0,
                           motion_lora: MotionLoraList=None, ad_keyframes: ADKeyframeGroup=None,
                           scale_multival=None, effect_multival=None, per_block: AllPerBlocks=None,
                           prev_m_models: MotionModelGroup=None,):
        # set up motion models list
        if prev_m_models is None:
            prev_m_models = MotionModelGroup()
        prev_m_models = prev_m_models.clone()
        motion_model = motion_model.clone()
        # check if internal motion model already present in previous model - create new if so
        for prev_model in prev_m_models.models:
            if motion_model.model is prev_model.model:
                # need to create new internal model based on same state_dict
                motion_model = create_fresh_motion_module(motion_model)
        # apply motion model to loaded_mm
        if motion_lora is not None:
            for lora in motion_lora.loras:
                load_motion_lora_as_patches(motion_model, lora)
        attachment = get_mm_attachment(motion_model)
        attachment.scale_multival = scale_multival
        attachment.effect_multival = effect_multival
        if per_block is not None:
            validate_per_block_compatibility(motion_model=motion_model, all_per_blocks=per_block)
            attachment.per_block_list = per_block.per_block_list
        attachment.keyframes = ad_keyframes.clone() if ad_keyframes else ADKeyframeGroup()
        validate_per_block_compatibility_keyframes(motion_model=motion_model, keyframes=attachment.keyframes)
        attachment.timestep_percent_range = (start_percent, end_percent)
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
                "per_block": ("PER_BLOCK",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("M_MODELS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②"
    FUNCTION = "apply_motion_model"

    def apply_motion_model(self,
                           motion_model: MotionModelPatcher, motion_lora: MotionLoraList=None,
                           scale_multival=None, effect_multival=None, ad_keyframes=None,
                           per_block: AllPerBlocks=None):
        # just a subset of normal ApplyAnimateDiffModelNode inputs
        return ApplyAnimateDiffModelNode.apply_motion_model(self, motion_model, motion_lora=motion_lora,
                                                            scale_multival=scale_multival, effect_multival=effect_multival,
                                                            ad_keyframes=ad_keyframes, per_block=per_block)


class LoadAnimateDiffModelNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_available_motion_models(),),
            },
            "optional": {
                "ad_settings": ("AD_SETTINGS",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 50}),
            }
        }

    RETURN_TYPES = ("MOTION_MODEL_ADE",)
    RETURN_NAMES = ("MOTION_MODEL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②"
    FUNCTION = "load_motion_model"

    def load_motion_model(self, model_name: str, ad_settings: AnimateDiffSettings=None):
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
                "per_block_replace": ("PER_BLOCK",),
                "inherit_missing": ("BOOLEAN", {"default": True}, ),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("AD_KEYFRAMES", )
    FUNCTION = "load_keyframe"

    CATEGORY = "Animate Diff 🎭🅐🅓"

    def load_keyframe(self,
                      start_percent: float, prev_ad_keyframes=None,
                      scale_multival: Union[float, torch.Tensor]=None, effect_multival: Union[float, torch.Tensor]=None,
                      per_block_replace: AllPerBlocks=None,
                      cameractrl_multival: Union[float, torch.Tensor]=None, pia_input: InputPIA=None,
                      inherit_missing: bool=True, guarantee_steps: int=1):
        if not prev_ad_keyframes:
            prev_ad_keyframes = ADKeyframeGroup()
        prev_ad_keyframes = prev_ad_keyframes.clone()
        keyframe = ADKeyframe(start_percent=start_percent,
                              scale_multival=scale_multival, effect_multival=effect_multival,
                              per_block_replace=per_block_replace,
                              cameractrl_multival=cameractrl_multival, pia_input=pia_input,
                              inherit_missing=inherit_missing, guarantee_steps=guarantee_steps)
        prev_ad_keyframes.add(keyframe)
        return (prev_ad_keyframes,)
