from typing import Union
import os

import folder_paths

from .ad_settings import AnimateDiffSettings
from .adapter_cameractrl import CameraEntry
from .logger import logger
from .utils_model import get_available_motion_models
from .utils_motion import ADKeyframeGroup
from .motion_lora import MotionLoraList
from .model_injection import (MotionModelGroup, MotionModelPatcher, load_motion_module_gen2, inject_camera_encoder_into_model)
from .nodes_gen2 import ApplyAnimateDiffModelNode


class ApplyAnimateDiffWithCameraCtrl:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_model": ("MOTION_MODEL_ADE",),
                "cameractrl_poses": ("CAMERACTRL_POSES",),
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
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl"
    FUNCTION = "apply_motion_model"

    def apply_motion_model(self, motion_model: MotionModelPatcher, cameractrl_poses: list[list[float]], start_percent: float=0.0, end_percent: float=1.0,
                           motion_lora: MotionLoraList=None, ad_keyframes: ADKeyframeGroup=None,
                           scale_multival=None, effect_multival=None,
                           prev_m_models: MotionModelGroup=None,):
        new_m_models = ApplyAnimateDiffModelNode.apply_motion_model(self, motion_model, start_percent=start_percent, end_percent=end_percent,
                                                                    motion_lora=motion_lora, ad_keyframes=ad_keyframes,
                                                                    scale_multival=scale_multival, effect_multival=effect_multival, prev_m_models=prev_m_models)
        # most recent added model will always be first in list;
        curr_model = new_m_models[0].models[0]
        # confirm that model contains camera_encoder
        if curr_model.model.camera_encoder is None:
            raise Exception(f"Motion model '{curr_model.model.mm_info.mm_name}' does not contain a camera_encoder; cannot be used with Apply AnimateDiff-CameraCtrl Model node.")
        camera_entries = [CameraEntry(entry) for entry in cameractrl_poses]
        curr_model.orig_camera_entries = camera_entries
        return new_m_models


class LoadAnimateDiffModelWithCameraCtrl:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_available_motion_models(),),
                "camera_ctrl": (get_available_motion_models(),),
            },
            "optional": {
                "ad_settings": ("AD_SETTINGS",),
            }
        }

    RETURN_TYPES = ("MOTION_MODEL_ADE",)
    RETURN_NAMES = ("MOTION_MODEL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl"
    FUNCTION = "load_camera_ctrl"

    def load_camera_ctrl(self, model_name: str, camera_ctrl: str, ad_settings: AnimateDiffSettings=None):
        loaded_motion_model = load_motion_module_gen2(model_name=model_name, motion_model_settings=ad_settings)
        inject_camera_encoder_into_model(motion_model=loaded_motion_model, camera_ctrl_name=camera_ctrl)
        return (loaded_motion_model,)


class LoadCameraPoses:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = [f for f in files if f.endswith(".txt")]
        return {
            "required": {
                "pose_filename": (sorted(files),),
            }
        }

    RETURN_TYPES = ("CAMERACTRL_POSES",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl"
    FUNCTION = "load_camera_poses"

    def load_camera_poses(self, pose_filename):
        file_path = folder_paths.get_annotated_filepath(pose_filename)
        with open(file_path, 'r') as f:
            poses = f.readlines()
        # first line of file is the link to source, so can be skipped,
        # and the rest is a header-less CSV file separated by single spaces
        poses = [pose.strip().split(' ') for pose in poses[1:]]
        poses = [[float(x) for x in pose] for pose in poses]
        return (poses,)
