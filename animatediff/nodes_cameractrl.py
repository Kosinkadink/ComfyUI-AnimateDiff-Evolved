from typing import Union
import os
import torch

import folder_paths
import copy
import json
import numpy as np

from .ad_settings import AnimateDiffSettings
from .adapter_cameractrl import CameraEntry
from .logger import logger
from .utils_model import get_available_motion_models, BIGMAX
from .utils_motion import ADKeyframeGroup
from .motion_lora import MotionLoraList
from .model_injection import (MotionModelGroup, MotionModelPatcher, load_motion_module_gen2, inject_camera_encoder_into_model)
from .nodes_gen2 import ApplyAnimateDiffModelNode, ADKeyframeNode

CAMERA = {
    # T
    "base_T_norm": 1.5,
    "base_angle": np.pi/3,

    "Static": {     "angle":[0., 0., 0.],   "T":[0., 0., 0.]},
    "Pan Up": {     "angle":[0., 0., 0.],   "T":[0., 1., 0.]},
    "Pan Down": {   "angle":[0., 0., 0.],   "T":[0.,-1.,0.]},
    "Pan Left": {   "angle":[0., 0., 0.],   "T":[1.,0.,0.]},
    "Pan Right": {  "angle":[0., 0., 0.],   "T": [-1.,0.,0.]},
    "Zoom In": {    "angle":[0., 0., 0.],   "T": [0.,0.,-2.]},
    "Zoom Out": {   "angle":[0., 0., 0.],   "T": [0.,0.,2.]},
    "ACW": {        "angle": [0., 0., 1.],  "T":[0., 0., 0.]},
    "CW": {         "angle": [0., 0., -1.], "T":[0., 0., 0.]},
}

def compute_R_form_rad_angle(angles):
    theta_x, theta_y, theta_z = angles
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def get_camera_motion(angle, T, speed, n=16):
    RT = []
    for i in range(n):
        _angle = (i/n)*speed*(CAMERA["base_angle"])*angle
        R = compute_R_form_rad_angle(_angle) 
        # _T = (i/n)*speed*(T.reshape(3,1))
        _T=(i/n)*speed*(CAMERA["base_T_norm"])*(T.reshape(3,1))
        _RT = np.concatenate([R,_T], axis=1)
        RT.append(_RT)
    RT = np.stack(RT)
    return RT
    
def combine_camera_motion(RT_0, RT_1):
    RT = copy.deepcopy(RT_0[-1])
    R = RT[:,:3]
    R_inv = RT[:,:3].T
    T =  RT[:,-1]

    temp = []
    for _RT in RT_1:
        _RT[:,:3] = np.dot(_RT[:,:3], R)
        _RT[:,-1] =  _RT[:,-1] + np.dot(np.dot(_RT[:,:3], R_inv), T) 
        temp.append(_RT)

    RT_1 = np.stack(temp)

    return np.concatenate([RT_0, RT_1], axis=0)


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
                "cameractrl_multival": ("MULTIVAL",),
                "ad_keyframes": ("AD_KEYFRAMES",),
                "prev_m_models": ("M_MODELS",),
            }
        }
    
    RETURN_TYPES = ("M_MODELS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl"
    FUNCTION = "apply_motion_model"

    def apply_motion_model(self, motion_model: MotionModelPatcher, cameractrl_poses: list[list[float]], start_percent: float=0.0, end_percent: float=1.0,
                           motion_lora: MotionLoraList=None, ad_keyframes: ADKeyframeGroup=None,
                           scale_multival=None, effect_multival=None, cameractrl_multival=None,
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
        curr_model.cameractrl_multival = cameractrl_multival
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


class CameraCtrlADKeyframeNode:
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
                "cameractrl_multival": ("MULTIVAL",),
                "inherit_missing": ("BOOLEAN", {"default": True}, ),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
            }
        }
    
    RETURN_TYPES = ("AD_KEYFRAMES", )
    FUNCTION = "load_keyframe"

    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl"

    def load_keyframe(self,
                      start_percent: float, prev_ad_keyframes=None,
                      scale_multival: Union[float, torch.Tensor]=None, effect_multival: Union[float, torch.Tensor]=None,
                      cameractrl_multival: Union[float, torch.Tensor]=None,
                      inherit_missing: bool=True, guarantee_steps: int=1):
        return ADKeyframeNode.load_keyframe(self,
                    start_percent=start_percent, prev_ad_keyframes=prev_ad_keyframes,
                    scale_multival=scale_multival, effect_multival=effect_multival, cameractrl_multival=cameractrl_multival,
                    inherit_missing=inherit_missing, guarantee_steps=guarantee_steps
                )


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
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses"
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


class CameraPoseBasic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pose":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "speed":("FLOAT",{"default":1.0}),
                "video_length":("INT",{"default":16}),
            },
        }

    RETURN_TYPES = ("CameraPose",)
    FUNCTION = "camera_pose_basic"
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses"

    def camera_pose_basic(self,camera_pose,speed,video_length):
        motion_list = [camera_pose]
        mode = "Basic Camera Poses" 
        angle = np.array(CAMERA[motion_list[0]]["angle"])
        T = np.array(CAMERA[motion_list[0]]["T"])
        RT = get_camera_motion(angle, T, speed, video_length)
        return (RT,)


class CameraPoseJoin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pose1":("CameraPose",),
                "camera_pose2":("CameraPose",),
            },
        }

    RETURN_TYPES = ("CameraPose",)
    FUNCTION = "camera_pose_join"
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses"

    def camera_pose_join(self,camera_pose1,camera_pose2):
        RT = combine_camera_motion(camera_pose1, camera_pose2)
        return (RT,)


class CameraPoseCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pose1":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "camera_pose2":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "camera_pose3":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "camera_pose4":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "speed":("FLOAT",{"default":1.0}),
                "video_length":("INT",{"default":16}),
            },
        }

    RETURN_TYPES = ("CameraPose",)
    FUNCTION = "camera_pose_combine"
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses"

    def camera_pose_combine(self,camera_pose1,camera_pose2,camera_pose3,camera_pose4,speed,video_length):
        angle = np.array(CAMERA[camera_pose1]["angle"]) + np.array(CAMERA[camera_pose2]["angle"]) + np.array(CAMERA[camera_pose3]["angle"]) + np.array(CAMERA[camera_pose4]["angle"])
        T = np.array(CAMERA[camera_pose1]["T"]) + np.array(CAMERA[camera_pose2]["T"]) + np.array(CAMERA[camera_pose3]["T"]) + np.array(CAMERA[camera_pose4]["T"])
        RT = get_camera_motion(angle, T, speed, video_length)
        return (RT,)


class CameraCtrlPose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pose":("CameraPose",),
                "fx":("FLOAT",{"default":0.474812461, "min": 0, "max": 1, "step": 0.000000001}),
                "fy":("FLOAT",{"default":0.844111024, "min": 0, "max": 1, "step": 0.000000001}),
                "cx":("FLOAT",{"default":0.5, "min": 0, "max": 1, "step": 0.01}),
                "cy":("FLOAT",{"default":0.5, "min": 0, "max": 1, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("CAMERACTRL_POSES","INT",)
    RETURN_NAMES = ("cameractrl_poses","video_length",)
    FUNCTION = "camera_ctrl_pose"
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses"

    def camera_ctrl_pose(self,camera_pose,fx,fy,cx,cy):
        camera_pose_list=camera_pose.tolist()
        trajs=[]
        for cp in camera_pose_list:
            traj=[0,fx,fy,cx,cy,0,0]
            traj.extend(cp[0])
            traj.extend(cp[1])
            traj.extend(cp[2])
            trajs.append(traj)
        
        return (trajs,len(trajs),)
