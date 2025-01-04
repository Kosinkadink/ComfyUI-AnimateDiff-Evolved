from comfy.model_patcher import ModelPatcher

from .ad_settings import AnimateDiffSettings
from .context import ContextOptionsGroup
from .logger import logger
from .utils_model import BetaSchedules, get_available_motion_models
from .utils_motion import ADKeyframeGroup, AllPerBlocks, get_combined_multival
from .motion_lora import MotionLoraList
from .model_injection import (ModelPatcherHelper, InjectionParams, MotionModelGroup, get_mm_attachment,
                              load_motion_lora_as_patches, load_motion_module_gen2, validate_model_compatibility_gen2,
                              validate_per_block_compatibility, validate_per_block_compatibility_keyframes)
from .sample_settings import SampleSettings
from .sampling import outer_sample_wrapper, sliding_calc_cond_batch


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
                "per_block": ("PER_BLOCK",),
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
        per_block: AllPerBlocks=None,
    ):
        # load motion module and motion settings, if included
        motion_model = load_motion_module_gen2(model_name=model_name, motion_model_settings=ad_settings)
        # confirm that it is compatible with SD model
        validate_model_compatibility_gen2(model=model, motion_model=motion_model)
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

        # create injection params
        params = InjectionParams(unlimited_area_hack=False)
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
            attachment.scale_multival = get_combined_multival(scale_multival, (params.motion_model_settings.mask_attn_scale * params.motion_model_settings.attn_scale))
        
        # need to use a ModelPatcher that supports injection of motion modules into unet
        model = model.clone()
        helper = ModelPatcherHelper(model)
        helper.set_all_properties(
            outer_sampler_wrapper=outer_sample_wrapper,
            calc_cond_batch_wrapper=sliding_calc_cond_batch,
            params=params,
            sample_settings=sample_settings,
            motion_models=MotionModelGroup(motion_model),
        )
        
        sample_settings = helper.get_sample_settings()
        if sample_settings.custom_cfg is not None:
            logger.info("[Sample Settings] custom_cfg is set; will override any KSampler cfg values or patches.")

        if sample_settings.sigma_schedule is not None:
            logger.info("[Sample Settings] sigma_schedule is set; will override beta_schedule.")
            model.add_object_patch("model_sampling", sample_settings.sigma_schedule.clone().model_sampling)
        else:
            # save model sampling from BetaSchedule as object patch
            # if autoselect, get suggested beta_schedule from motion model
            if beta_schedule == BetaSchedules.AUTOSELECT and helper.get_motion_models():
                beta_schedule = helper.get_motion_models()[0].model.get_best_beta_schedule(log=True)
            new_model_sampling = BetaSchedules.to_model_sampling(beta_schedule, model)
            if new_model_sampling is not None:
                model.add_object_patch("model_sampling", new_model_sampling)

        del motion_model
        return (model,)
