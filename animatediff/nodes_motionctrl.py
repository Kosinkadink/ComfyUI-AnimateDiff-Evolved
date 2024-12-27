import torch
from torch import Tensor

from .ad_settings import AnimateDiffSettings
from .adapter_motionctrl import injection_motionctrl_cmcm, load_motionctrl_omcm

from .motion_module_ad import AllPerBlocks
from .model_injection import MotionModelPatcher, MotionModelGroup, load_motion_module_gen2
from .motion_lora import MotionLoraList

from .nodes_gen2 import ApplyAnimateDiffModelNode
from .utils_model import get_available_motion_models
from .utils_motion import ADKeyframeGroup


class LoadMotionCtrlCMCM:
    NodeID = "ADE_LoadMotionCtrl_CMCMMOdel"
    NodeName = "Load AnimateDiff+MotionCtrl Camera Model 🎭🅐🅓②"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_available_motion_models(),),
                "motionctrl_cmcm": (get_available_motion_models(),),
            },
            "optional": {
                "ad_settings": ("AD_SETTINGS",),
            }
        }
    
    RETURN_TYPES = ("MOTION_MODEL_ADE",)
    RETURN_NAMES = ("MOTION_MODEL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/MotionCtrl"
    FUNCTION = "load_motionctrl_cmcm"

    def load_motionctrl_cmcm(self, model_name: str, motionctrl_cmcm: str, ad_settings: AnimateDiffSettings=None):
        motion_model = load_motion_module_gen2(model_name=model_name, motion_model_settings=ad_settings)
        motion_model = injection_motionctrl_cmcm(motion_model, cmcm_name=motionctrl_cmcm)
        return (motion_model,)


class LoadMotionCtrlOMCM:
    NodeID = "ADE_LoadMotionCtrl_OMCMMOdel"
    NodeName = "Load MotionCtrl Object Model 🎭🅐🅓②"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motionctrl_omcm": (get_available_motion_models(),),
            }
        }
    
    RETURN_TYPES = ("OMCM_MOTIONCTRL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/MotionCtrl"
    FUNCTION = "load_motionctrl_omcm"

    def load_motionctrl_omcm(self, motionctrl_omcm: str):
        omcm_modelpatcher = load_motionctrl_omcm(motionctrl_omcm)
        return (omcm_modelpatcher,)


class ApplyAnimateDiffMotionCtrlModel:
    NodeID = "ADE_ApplyAnimateDiffModelWithMotionCtrl"
    NodeName = "Apply AnimateDiff+MotionCtrl Model 🎭🅐🅓②"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_model": ("MOTION_MODEL_ADE",),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "omcm_motionctrl": ("OMCM_MOTIONCTRL",),
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
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/MotionCtrl"
    FUNCTION = "apply_motion_model"

    def apply_motion_model(self, motion_model: MotionModelPatcher, start_percent: float=0.0, end_percent: float=1.0,
                           motion_lora: MotionLoraList=None, ad_keyframes: ADKeyframeGroup=None,
                           scale_multival=None, effect_multival=None, per_block: AllPerBlocks=None,
                           prev_m_models: MotionModelGroup=None,):
        (new_m_models,) = ApplyAnimateDiffModelNode.apply_motion_model(self, motion_model, start_percent=start_percent, end_percent=end_percent,
                                                                    motion_lora=motion_lora, ad_keyframes=ad_keyframes, per_block=per_block,
                                                                    scale_multival=scale_multival, effect_multival=effect_multival, prev_m_models=prev_m_models)
        # most recent added model will always be first in list
        curr_model = new_m_models.models[0]
        # check if model has CMCM; if so, make sure something is provided for it
        # check if OMCM is provided; if so, make sure something is provided for it
        return (new_m_models,)
