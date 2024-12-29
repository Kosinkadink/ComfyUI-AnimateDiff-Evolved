import torch
from torch import Tensor
import numpy as np
import os
import json

import folder_paths

from .ad_settings import AnimateDiffSettings
from .adapter_motionctrl import (ObjectControlModelPatcher, inject_motionctrl_cmcm, load_motionctrl_omcm,
                                 convert_cameractrl_poses_to_RT)

from .model_injection import MotionModelPatcher, MotionModelGroup, load_motion_module_gen2, get_mm_attachment
from .motion_lora import MotionLoraList

from .nodes_gen2 import ApplyAnimateDiffModelNode
from .utils_model import get_available_motion_models
from .utils_motion import ADKeyframeGroup, AllPerBlocks


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
                "override_ad_weights": ("BOOLEAN", {"default": True}),
                "ad_settings": ("AD_SETTINGS",),
            }
        }
    
    RETURN_TYPES = ("MOTION_MODEL_ADE",)
    RETURN_NAMES = ("MOTION_MODEL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/MotionCtrl"
    FUNCTION = "load_motionctrl_cmcm"

    def load_motionctrl_cmcm(self, model_name: str, motionctrl_cmcm: str,
                             override_ad_weights=True, ad_settings: AnimateDiffSettings=None,):
        motion_model = load_motion_module_gen2(model_name=model_name, motion_model_settings=ad_settings)
        inject_motionctrl_cmcm(motion_model.model, cmcm_name=motionctrl_cmcm, apply_non_ccs=override_ad_weights)
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


class LoadMotionCtrlCameraPosesFromFile:
    NodeID = "ADE_LoadMotionCtrlCameraPosesFromFile"
    NodeName = "Load MotionCtrl Camera Poses 🎭🅐🅓"
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = [f for f in files if f.endswith(".json")]
        return {
            "required": {
                "pose_filename": (sorted(files),),
            }
        }
    
    RETURN_TYPES = ("CAMERA_MOTIONCTRL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/MotionCtrl"
    FUNCTION = "load_camera_poses"

    def load_camera_poses(self, pose_filename):
        file_path = folder_paths.get_annotated_filepath(pose_filename)
        with open(file_path, 'r') as f:
            RT = json.load(f)
        RT = np.array(RT)
        RT = torch.tensor(RT).float() # [t, 12]
        return (RT,)


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
                "cameractrl_poses": ("CAMERACTRL_POSES",),
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
                           omcm_motionctrl: ObjectControlModelPatcher=None,
                           cameractrl_poses: list[list[float]]=None,
                           motion_lora: MotionLoraList=None, ad_keyframes: ADKeyframeGroup=None,
                           scale_multival=None, effect_multival=None, per_block: AllPerBlocks=None,
                           prev_m_models: MotionModelGroup=None,):
        (new_m_models,) = ApplyAnimateDiffModelNode.apply_motion_model(self, motion_model, start_percent=start_percent, end_percent=end_percent,
                                                                    motion_lora=motion_lora, ad_keyframes=ad_keyframes, per_block=per_block,
                                                                    scale_multival=scale_multival, effect_multival=effect_multival, prev_m_models=prev_m_models)
        # most recent added model will always be first in list
        curr_model = new_m_models.models[0]
        # check if model has CMCM; if so, make sure something is provided for it
        if curr_model.model.is_motionctrl_cc_enabled():
            attachment = get_mm_attachment(curr_model)
            if cameractrl_poses is not None:
                RT = convert_cameractrl_poses_to_RT(cameractrl_poses)
                attachment.orig_RT = RT
            else:
                attachment.orig_RT = torch.zeros((1, 12))
            #     attachment.orig_RT = cameractrl_poses
            # else:
            #     attachment.orig_RT = torch.zeros([])

        # check if OMCM is provided; if so, make sure something is provided for it
        return (new_m_models,)
