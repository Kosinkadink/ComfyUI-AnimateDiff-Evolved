from typing import Union
import torch
from torch import Tensor

from comfy.sd import VAE

from .logger import logger
from .utils_model import BIGMAX
from .utils_motion import ADKeyframeGroup, InputPIA, InputPIA_Multival
from .motion_lora import MotionLoraList
from .model_injection import MotionModelGroup, MotionModelPatcher
from .motion_module_ad import AnimateDiffFormat
from .nodes_gen2 import ApplyAnimateDiffModelNode, ADKeyframeNode


class ApplyAnimateDiffPIAModel:
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
                "pia_input": ("PIA_INPUT",),
                "motion_lora": ("MOTION_LORA",),
                "scale_multival": ("MULTIVAL",),
                "effect_multival": ("MULTIVAL",),
                "ad_keyframes": ("AD_KEYFRAMES",),
                "prev_m_models": ("M_MODELS",),
            }
        }

    RETURN_TYPES = ("M_MODELS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA"
    FUNCTION = "apply_motion_model"

    def apply_motion_model(self, motion_model: MotionModelPatcher, image: Tensor, vae: VAE,
                           start_percent: float=0.0, end_percent: float=1.0, pia_input: InputPIA=None,
                           motion_lora: MotionLoraList=None, ad_keyframes: ADKeyframeGroup=None,
                           scale_multival=None, effect_multival=None, ref_multival=None,
                           prev_m_models: MotionModelGroup=None,):
        new_m_models = ApplyAnimateDiffModelNode.apply_motion_model(self, motion_model, start_percent=start_percent, end_percent=end_percent,
                                                                    motion_lora=motion_lora, ad_keyframes=ad_keyframes,
                                                                    scale_multival=scale_multival, effect_multival=effect_multival, prev_m_models=prev_m_models)
        # most recent added model will always be first in list;
        curr_model = new_m_models[0].models[0]
        # confirm that model is PIA
        if curr_model.model.mm_info.mm_format != AnimateDiffFormat.PIA:
            raise Exception(f"Motion model '{curr_model.model.mm_info.mm_name}' is not a PIA model; cannot be used with Apply AnimateDiff-PIA Model node.")
        curr_model.orig_pia_images = image
        curr_model.pia_vae = vae
        if pia_input is None:
            pia_input = InputPIA_Multival(1.0)
        curr_model.pia_input = pia_input
        #curr_model.pia_multival = ref_multival
        return new_m_models


class PIA_ADKeyframeNode:
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
                "pia_input": ("PIA_INPUT",),
                "inherit_missing": ("BOOLEAN", {"default": True}, ),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
            }
        }
    
    RETURN_TYPES = ("AD_KEYFRAMES", )
    FUNCTION = "load_keyframe"

    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA"

    def load_keyframe(self,
                      start_percent: float, prev_ad_keyframes=None,
                      scale_multival: Union[float, torch.Tensor]=None, effect_multival: Union[float, torch.Tensor]=None,
                      pia_input: InputPIA=None,
                      inherit_missing: bool=True, guarantee_steps: int=1):
        return ADKeyframeNode.load_keyframe(self,
                    start_percent=start_percent, prev_ad_keyframes=prev_ad_keyframes,
                    scale_multival=scale_multival, effect_multival=effect_multival, pia_input=pia_input,
                    inherit_missing=inherit_missing, guarantee_steps=guarantee_steps
                )


class InputPIA_MultivalNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "multival": ("MULTIVAL",),
            },
        }
    
    RETURN_TYPES = ("PIA_INPUT",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA"
    FUNCTION = "create_pia_input"

    def create_pia_input(self, multival: Union[float, Tensor]):
        return (InputPIA_Multival(multival),)
