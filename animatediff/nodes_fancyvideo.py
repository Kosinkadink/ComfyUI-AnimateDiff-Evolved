from typing import Union
import torch
from torch import Tensor
import math

from comfy.sd import VAE

from .ad_settings import AnimateDiffSettings
from .logger import logger
from .utils_model import BIGMIN, BIGMAX, get_available_motion_models
from .utils_motion import ADKeyframeGroup, InputPIA, InputPIA_Multival, extend_list_to_batch_size, extend_to_batch_size, prepare_mask_batch
from .motion_lora import MotionLoraList
from .model_injection import MotionModelGroup, MotionModelPatcher, get_mm_attachment, load_motion_module_gen2, inject_pia_conv_in_into_model
from .motion_module_ad import AnimateDiffFormat
from .nodes_gen2 import ApplyAnimateDiffModelNode, ADKeyframeNode


class ApplyAnimateDiffFancyVideo:
    NodeID = 'ADE_ApplyAnimateDiffFancyVideo'
    NodeName = 'Apply AD-FancyVideo Model 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_model": ("MOTION_MODEL_ADE",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
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
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/FancyVideo"
    FUNCTION = "apply_motion_model"

    def apply_motion_model(self, motion_model: MotionModelPatcher, image: Tensor, vae: VAE,
                           start_percent: float=0.0, end_percent: float=1.0,
                           motion_lora: MotionLoraList=None, ad_keyframes: ADKeyframeGroup=None,
                           scale_multival=None, effect_multival=None, ref_multival=None, per_block=None,
                           prev_m_models: MotionModelGroup=None,):
        new_m_models = ApplyAnimateDiffModelNode.apply_motion_model(self, motion_model, start_percent=start_percent, end_percent=end_percent,
                                                                    motion_lora=motion_lora, ad_keyframes=ad_keyframes,
                                                                    scale_multival=scale_multival, effect_multival=effect_multival, per_block=per_block,
                                                                    prev_m_models=prev_m_models)
        # most recent added model will always be first in list;
        curr_model = new_m_models[0].models[0]
        # confirm that model is FancyVideo
        if curr_model.model.mm_info.mm_format != AnimateDiffFormat.FANCYVIDEO:
            raise Exception(f"Motion model '{curr_model.model.mm_info.mm_name}' is not a FancyVideo model; cannot be used with Apply AD-FancyModel Model node.")
        attachment = get_mm_attachment(curr_model)
        attachment.orig_fancy_images = image
        attachment.fancy_vae = vae
        return new_m_models
