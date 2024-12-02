from typing import Union
import torch

from nodes import VAEEncode
import comfy.utils
from comfy.sd import VAE

from .ad_settings import AnimateDiffSettings
from .logger import logger
from .utils_model import ScaleMethods, CropMethods, get_available_motion_models, vae_encode_raw_batched
from .utils_motion import ADKeyframeGroup
from .motion_lora import MotionLoraList
from .model_injection import (MotionModelGroup, MotionModelPatcher, get_mm_attachment, create_fresh_encoder_only_model,
                              load_motion_module_gen2, inject_img_encoder_into_model)
from .motion_module_ad import AnimateDiffFormat
from .nodes_gen2 import ApplyAnimateDiffModelNode


class ApplyAnimateLCMI2VModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_model": ("MOTION_MODEL_ADE",),
                "ref_latent": ("LATENT",),
                "ref_drift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "apply_ref_when_disabled": ("BOOLEAN", {"default": False}),
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
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/AnimateLCM-I2V"
    FUNCTION = "apply_motion_model"

    def apply_motion_model(self, motion_model: MotionModelPatcher, ref_latent: dict, ref_drift: float=0.0, apply_ref_when_disabled=False, start_percent: float=0.0, end_percent: float=1.0,
                           motion_lora: MotionLoraList=None, ad_keyframes: ADKeyframeGroup=None,
                           scale_multival=None, effect_multival=None, per_block=None,
                           prev_m_models: MotionModelGroup=None,):
        new_m_models = ApplyAnimateDiffModelNode.apply_motion_model(self, motion_model, start_percent=start_percent, end_percent=end_percent,
                                                                    motion_lora=motion_lora, ad_keyframes=ad_keyframes,
                                                                    scale_multival=scale_multival, effect_multival=effect_multival, per_block=per_block,
                                                                    prev_m_models=prev_m_models)
        # most recent added model will always be first in list;
        curr_model = new_m_models[0].models[0]
        # confirm that model contains img_encoder
        if curr_model.model.img_encoder is None:
            raise Exception(f"Motion model '{curr_model.model.mm_info.mm_name}' does not contain an img_encoder; cannot be used with Apply AnimateLCM-I2V Model node.")
        attachment = get_mm_attachment(curr_model)
        attachment.orig_img_latents = ref_latent["samples"]
        attachment.orig_ref_drift = ref_drift
        attachment.orig_apply_ref_when_disabled = apply_ref_when_disabled
        return new_m_models


class LoadAnimateLCMI2VModelNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_available_motion_models(),),
            },
            "optional": {
                "ad_settings": ("AD_SETTINGS",),
            }
        }
    
    RETURN_TYPES = ("MOTION_MODEL_ADE", "MOTION_MODEL_ADE")
    RETURN_NAMES = ("MOTION_MODEL", "encoder_only")
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/AnimateLCM-I2V"
    FUNCTION = "load_motion_model"

    def load_motion_model(self, model_name: str, ad_settings: AnimateDiffSettings=None):
        # load motion module and motion settings, if included
        motion_model = load_motion_module_gen2(model_name=model_name, motion_model_settings=ad_settings)
        # make sure model is an AnimateLCM-I2V model
        if motion_model.model.mm_info.mm_format != AnimateDiffFormat.ANIMATELCM:
            raise Exception(f"Motion model '{motion_model.model.mm_info.mm_name}' is not an AnimateLCM-I2V model; selected model is not AnimateLCM, and does not contain an img_encoder.")
        if motion_model.model.img_encoder is None:
            raise Exception(f"Motion model '{motion_model.model.mm_info.mm_name}' is not an AnimateLCM-I2V model; selected model IS AnimateLCM, but does NOT contain an img_encoder.")
        # create encoder-only motion model
        encoder_only_motion_model = create_fresh_encoder_only_model(motion_model=motion_model)
        return (motion_model, encoder_only_motion_model)


class LoadAnimateDiffAndInjectI2VNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_available_motion_models(),),
                "motion_model": ("MOTION_MODEL_ADE",),
            },
            "optional": {
                "ad_settings": ("AD_SETTINGS",),
                "deprecation_warning": ("ADEWARN", {"text": "Experimental. Don't expect to work.", "warn_type": "experimental", "color": "#CFC"}),
            }
        }
    
    RETURN_TYPES = ("MOTION_MODEL_ADE",)
    RETURN_NAMES = ("MOTION_MODEL",)

    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/AnimateLCM-I2V/🧪experimental"
    FUNCTION = "load_motion_model"

    def load_motion_model(self, model_name: str, motion_model: MotionModelPatcher, ad_settings: AnimateDiffSettings=None):
        # make sure model w/ encoder actually has encoder
        if motion_model.model.img_encoder is None:
            raise Exception("Passed-in motion model was expected to have an img_encoder, but did not.")
        # load motion module and motion settings, if included
        loaded_motion_model = load_motion_module_gen2(model_name=model_name, motion_model_settings=ad_settings)
        inject_img_encoder_into_model(motion_model=loaded_motion_model, w_encoder=motion_model)
        return (loaded_motion_model,)


class UpscaleAndVaeEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "latent_size": ("LATENT",),
                "scale_method": (ScaleMethods._LIST_IMAGE,),
                "crop": (CropMethods._LIST, {"default": CropMethods.CENTER},),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "preprocess_images"

    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/AnimateLCM-I2V"

    def preprocess_images(self, image: torch.Tensor, vae: VAE, latent_size: torch.Tensor, scale_method: str, crop: str):
        b, c, h, w = latent_size["samples"].size()
        image = image.movedim(-1,1)
        image = comfy.utils.common_upscale(samples=image, width=w*8, height=h*8, upscale_method=scale_method, crop=crop)
        image = image.movedim(1,-1)
        # now that images are the expected size, VAEEncode them
        return ({"samples": vae_encode_raw_batched(vae, image)},)
