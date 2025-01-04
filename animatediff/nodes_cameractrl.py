from typing import Union
import os
import torch

import math
import folder_paths
import copy
import json
import numpy as np
from pathlib import Path
from collections import OrderedDict

from .ad_settings import AnimateDiffSettings
from .adapter_cameractrl import CameraEntry
from .logger import logger
from .utils_model import get_available_motion_models, calculate_file_hash, strip_path, BIGMAX
from .utils_motion import ADKeyframeGroup
from .motion_lora import MotionLoraList
from .model_injection import (MotionModelGroup, MotionModelPatcher, get_mm_attachment, load_motion_module_gen2, inject_camera_encoder_into_model)
from .nodes_gen2 import ApplyAnimateDiffModelNode, ADKeyframeNode


class CameraMotion:
    def __init__(self, rotate: tuple[float], translate: tuple[float]):
        assert len(rotate) == 3
        assert len(translate) == 3
        self.rotate = np.array(rotate)
        self.translate = np.array(translate)

    def multiply(self, mult: float):
        if math.isclose(mult, 1.0):
            return self.clone()
        new_rotate = self.rotate.copy()
        new_translate = self.translate.copy()
        new_rotate *= mult
        new_translate *= mult
        return CameraMotion(rotate=new_rotate, translate=new_translate)

    def clone(self):
        return CameraMotion(rotate=self.rotate.copy(), translate=self.translate.copy())

    @staticmethod
    def combine(deltas: list['CameraMotion']) -> 'CameraMotion':
        new_rotate = np.array([0., 0., 0.])
        new_translate = np.array([0., 0., 0.])
        for delta in deltas:
            new_rotate += delta.rotate
            new_translate += delta.translate
        return CameraMotion(rotate=new_rotate, translate=new_translate)


class CAM:
    BASE_T_NORM = 1.5
    BASE_ANGLE = np.pi/3

    DEFAULT_FX = 0.474812461
    DEFAULT_FY = 0.844111024
    DEFAULT_CX = 0.5
    DEFAULT_CY = 0.5

    DEFAULT_POSE_WIDTH = 1280
    DEFAULT_POSE_HEIGHT = 720

    STATIC = "Static"
    PAN_UP = "Pan Up"
    PAN_DOWN = "Pan Down"
    PAN_LEFT = "Pan Left"
    PAN_RIGHT = "Pan Right"
    ZOOM_IN = "Zoom In"
    ZOOM_OUT = "Zoom Out"
    ROLL_CLOCKWISE = "Roll Clockwise"
    ROLL_ANTICLOCKWISE = "Roll Anticlockwise"
    TILT_UP = "Tilt Up"
    TILT_DOWN = "Tilt Down"
    TILT_LEFT = "Tilt Left"
    TILT_RIGHT = "Tilt Right"
    
    _PAIRS = [
        (STATIC,        CameraMotion(rotate=(0., 0., 0.), translate=(0., 0., 0.))),
        (PAN_UP,        CameraMotion(rotate=(0., 0., 0.), translate=(0., 1., 0.))),
        (PAN_DOWN,      CameraMotion(rotate=(0., 0., 0.), translate=(0., -1., 0.))),
        (PAN_LEFT,      CameraMotion(rotate=(0., 0., 0.), translate=(1., 0., 0.))),
        (PAN_RIGHT,     CameraMotion(rotate=(0., 0., 0.), translate=(-1., 0., 0.))),
        (ZOOM_IN,       CameraMotion(rotate=(0., 0., 0.), translate=(0., 0., -2.))),
        (ZOOM_OUT,      CameraMotion(rotate=(0., 0., 0.), translate=(0., 0., 2.))),
        (ROLL_CLOCKWISE,     CameraMotion(rotate=(0., 0., -1.), translate=(0., 0., 0.))),
        (ROLL_ANTICLOCKWISE, CameraMotion(rotate=(0., 0., 1.), translate=(0., 0., 0.))),
        (TILT_DOWN,     CameraMotion(rotate=(1., 0., 0.), translate=(0., 0., 0.))),
        (TILT_UP,    CameraMotion(rotate=(-1., 0., 0.), translate=(0., 0., 0.))),
        (TILT_LEFT,       CameraMotion(rotate=(0., 1., 0.), translate=(0., 0., 0.))),
        (TILT_RIGHT,     CameraMotion(rotate=(0., -1., 0.), translate=(0., 0., 0.))),
    ]
    _DICT: dict[str, CameraMotion] = OrderedDict(_PAIRS)
    _LIST = list(_DICT.keys())

    @staticmethod
    def get(motion: str):
        return CAM._DICT[motion]


def compute_R_from_rad_angle(angles: np.ndarray):
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

def get_camera_motion(angle: np.ndarray, T: np.ndarray, speed: float, n=16, base=16):
    RT = []
    for i in range(n):
        _angle = (i/base)*speed*(CAM.BASE_ANGLE)*angle
        R = compute_R_from_rad_angle(_angle) 
        # _T = (i/n)*speed*(T.reshape(3,1))
        _T=(i/base)*speed*(CAM.BASE_T_NORM)*(T.reshape(3,1))
        _RT = np.concatenate([R,_T], axis=1)
        RT.append(_RT)
    RT = np.stack(RT)
    return RT
    
def combine_RTs(RT_0: np.ndarray, RT_1: np.ndarray):
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

def stack_RTs(RT_0: np.ndarray, RT_1: np.ndarray):
    RT_target = copy.deepcopy(RT_1)
    static_motion = CAM.get(CAM.STATIC)
    RT_static = get_camera_motion(static_motion.rotate, static_motion.translate, 1.0, 1)
    RT_offset = RT_0[-1] - RT_static[-1]

    temp = []
    for sub_RT in RT_target:
        temp.append(sub_RT + RT_offset)

    RT_1 = np.stack(temp)
    RT_0 = RT_0[:-1]

    return np.concatenate([RT_0, RT_1], axis=0)


def set_original_pose_dims(poses: list[list[float]], pose_width, pose_height):
    # indexes 5 and 6 are not used for anything in the poses, so can use 5 and 6 to set original pose width/height
    new_poses = copy.deepcopy(poses)
    for pose in new_poses:
        pose[5] = pose_width
        pose[6] = pose_height
    return new_poses

def combine_poses(poses0: list[list[float]], poses1: list[list[float]]):
    new_poses = copy.deepcopy(poses0) + copy.deepcopy(poses1)
    new_RT = combine_RTs(poses_to_ndarray(poses0), poses_to_ndarray(poses1))
    inter_poses = ndarray_to_poses(new_RT)
    # maintain fx, fy, cx, and cy values by pasting only the movement portion of poses
    for i in range(len(new_poses)):
        new_poses[i][7:] = inter_poses[i][7:]
    return new_poses


def combine_poses_redux(poses0: list[list[float]], poses1: list[list[float]]):
    new_poses = copy.deepcopy(poses0[:-1]) + copy.deepcopy(poses1)
    new_RT = stack_RTs(poses_to_ndarray(poses0), poses_to_ndarray(poses1))
    inter_poses = ndarray_to_poses(new_RT)
    # maintain fx, fy, cx, and cy values by pasting only the movement portion of poses
    for i in range(len(new_poses)):
        new_poses[i][7:] = inter_poses[i][7:]
    return new_poses


def combine_poses_with_ndarray(poses: list[list[float]], RT: np.ndarray):
    return combine_poses(poses0=poses, poses1=ndarray_to_poses(RT))


def ndarray_to_poses(RT: np.ndarray, fx=CAM.DEFAULT_FX, fy=CAM.DEFAULT_FY, cx=CAM.DEFAULT_CX, cy=CAM.DEFAULT_CY) -> list[list[float]]:
    '''
    Converts ndarray (motion) to cameractrl_poses.
    '''
    motion_list=RT.tolist()
    poses = []
    for motion in motion_list:
        traj = [0, fx, fy, cx, cy, CAM.DEFAULT_POSE_WIDTH, CAM.DEFAULT_POSE_HEIGHT]
        traj.extend(motion[0])
        traj.extend(motion[1])
        traj.extend(motion[2])
        poses.append(traj)
    return poses

def poses_to_ndarray(poses: list[list[float]]) -> np.ndarray:
    '''
    Converts cameractrl_poses (list) to ndarray (motion) to be used for math stuff.
    '''
    motion_list = []
    for pose in poses:
        # pose will have 19 components;
        # idx 7-10 have first column, idx 11-14 have second column, idx 15-18 have third column
        motion_list.append(np.array(pose[7:]).reshape(3, 4))
    RT = np.array(motion_list)
    return RT


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
                "per_block": ("PER_BLOCK",),
            }
        }
    
    RETURN_TYPES = ("M_MODELS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl"
    FUNCTION = "apply_motion_model"

    def apply_motion_model(self, motion_model: MotionModelPatcher, cameractrl_poses: list[list[float]], start_percent: float=0.0, end_percent: float=1.0,
                           motion_lora: MotionLoraList=None, ad_keyframes: ADKeyframeGroup=None,
                           scale_multival=None, effect_multival=None, cameractrl_multival=None, per_block=None,
                           prev_m_models: MotionModelGroup=None,):
        new_m_models = ApplyAnimateDiffModelNode.apply_motion_model(self, motion_model, start_percent=start_percent, end_percent=end_percent,
                                                                    motion_lora=motion_lora, ad_keyframes=ad_keyframes, per_block=per_block,
                                                                    scale_multival=scale_multival, effect_multival=effect_multival, prev_m_models=prev_m_models)
        # most recent added model will always be first in list;
        curr_model = new_m_models[0].models[0]
        # confirm that model contains camera_encoder
        if curr_model.model.camera_encoder is None:
            raise Exception(f"Motion model '{curr_model.model.mm_info.mm_name}' does not contain a camera_encoder; cannot be used with Apply AnimateDiff-CameraCtrl Model node.")
        camera_entries = [CameraEntry(entry) for entry in cameractrl_poses]
        attachment = get_mm_attachment(curr_model)
        attachment.orig_camera_entries = camera_entries
        attachment.cameractrl_multival = cameractrl_multival
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
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
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


class LoadCameraPosesFromFile:
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

    def load_camera_poses(self, pose_filename: str):
        file_path = folder_paths.get_annotated_filepath(pose_filename)
        with open(file_path, 'r') as f:
            poses = f.readlines()
        # first line of file is the link to source, so can be skipped,
        # and the rest is a header-less CSV file separated by single spaces
        poses = [pose.strip().split(' ') for pose in poses[1:]]
        poses = [[float(x) for x in pose] for pose in poses]
        poses = set_original_pose_dims(poses, pose_width=CAM.DEFAULT_POSE_WIDTH, pose_height=CAM.DEFAULT_POSE_HEIGHT)
        return (poses,)
    

class LoadCameraPosesFromPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "file_path": ("STRING", {"default": "X://path/to/pose_file.txt"}),
            }
        }
    
    @classmethod
    def IS_CHANGED(s, file_path, **kwargs):
        if Path(file_path).is_file():
            return calculate_file_hash(strip_path(file_path))
        return False
    
    @classmethod
    def VALIDATE_INPUTS(s, file_path, **kwargs):
        # This function never gets ran for some reason, I don't care enough to figure out why right now.
        if not Path(strip_path(file_path)).is_file():
            return f"Pose file not found: {file_path}"
        return True

    RETURN_TYPES = ("CAMERACTRL_POSES",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses"
    FUNCTION = "load_camera_poses"

    def load_camera_poses(self, file_path: str):
        file_path = strip_path(file_path)
        if not Path(file_path).is_file():
            raise Exception(f"Pose file not found: {file_path}")
        with open(file_path, 'r') as f:
            poses = f.readlines()
        # first line of file is the link to source, so can be skipped,
        # and the rest is a header-less CSV file separated by single spaces
        poses = [pose.strip().split(' ') for pose in poses[1:]]
        poses = [[float(x) for x in pose] for pose in poses]
        poses = set_original_pose_dims(poses, pose_width=CAM.DEFAULT_POSE_WIDTH, pose_height=CAM.DEFAULT_POSE_HEIGHT)
        return (poses,)


class CameraCtrlPoseBasic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_type": (CAM._LIST,),
                "speed": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "frame_length": ("INT", {"default": 16}),
            },
            "optional": {
                "prev_poses": ("CAMERACTRL_POSES",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("CAMERACTRL_POSES",)
    FUNCTION = "camera_pose_basic"
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses"

    def camera_pose_basic(self, motion_type: str, speed: float, frame_length: int, prev_poses: list[list[float]]=None):
        motion = CAM.get(motion_type)
        RT = get_camera_motion(motion.rotate, motion.translate, speed, frame_length)
        new_motion = ndarray_to_poses(RT=RT)
        if prev_poses is not None:
            new_motion = combine_poses(prev_poses, new_motion)
        return (new_motion,)


class CameraCtrlPoseCombo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_type1": (CAM._LIST,),
                "motion_type2": (CAM._LIST,),
                "motion_type3": (CAM._LIST,),
                "motion_type4": (CAM._LIST,),
                "motion_type5": (CAM._LIST,),
                "motion_type6": (CAM._LIST,),
                "speed": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "frame_length": ("INT", {"default": 16}),
            },
            "optional": {
                "prev_poses": ("CAMERACTRL_POSES",),
            }
        }

    RETURN_TYPES = ("CAMERACTRL_POSES",)
    FUNCTION = "camera_pose_combo"
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses"

    def camera_pose_combo(self,
                          motion_type1: str, motion_type2: str, motion_type3: str,
                          motion_type4: str, motion_type5: str, motion_type6: str,
                          speed: float, frame_length: int,
                          prev_poses: list[list[float]]=None,
                          strength1=1.0, strength2=1.0, strength3=1.0, strength4=1.0, strength5=1.0, strength6=1.0):
        combined_motion = CameraMotion.combine([
            CAM.get(motion_type1).multiply(strength1), CAM.get(motion_type2).multiply(strength2), CAM.get(motion_type3).multiply(strength3),
            CAM.get(motion_type4).multiply(strength4), CAM.get(motion_type5).multiply(strength5), CAM.get(motion_type6).multiply(strength6)
            ])
        RT = get_camera_motion(combined_motion.rotate, combined_motion.translate, speed, frame_length)
        new_motion = ndarray_to_poses(RT=RT)
        if prev_poses is not None:
            new_motion = combine_poses(prev_poses, new_motion)
        return (new_motion,)


class CameraCtrlPoseAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_type1": (CAM._LIST,),
                "strength1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "motion_type2": (CAM._LIST,),
                "strength2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "motion_type3": (CAM._LIST,),
                "strength3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "motion_type4": (CAM._LIST,),
                "strength4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "motion_type5": (CAM._LIST,),
                "strength5": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "motion_type6": (CAM._LIST,),
                "strength6": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "speed": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "frame_length": ("INT", {"default": 16}),
            },
            "optional": {
                "prev_poses": ("CAMERACTRL_POSES",),
            }
        }

    RETURN_TYPES = ("CAMERACTRL_POSES",)
    FUNCTION = "camera_pose_combo"
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses"

    def camera_pose_combo(self,
                          motion_type1: str, motion_type2: str, motion_type3: str,
                          motion_type4: str, motion_type5: str, motion_type6: str,
                          speed: float, frame_length: int,
                          prev_poses: list[list[float]]=None,
                          strength1=1.0, strength2=1.0, strength3=1.0, strength4=1.0, strength5=1.0, strength6=1.0):
        return CameraCtrlPoseCombo.camera_pose_combo(self,
                                                     motion_type1=motion_type1, motion_type2=motion_type2, motion_type3=motion_type3,
                                                     motion_type4=motion_type4, motion_type5=motion_type5, motion_type6=motion_type6,
                                                     speed=speed, frame_length=frame_length, prev_poses=prev_poses,
                                                     strength1=strength1, strength2=strength2, strength3=strength3,
                                                     strength4=strength4, strength5=strength5, strength6=strength6)


class CameraCtrlManualAppendPose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "poses_first": ("CAMERACTRL_POSES",),
                "poses_last": ("CAMERACTRL_POSES",),
            }
        }
    
    RETURN_TYPES = ("CAMERACTRL_POSES",)
    FUNCTION = "camera_manual_append"
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses"

    def camera_manual_append(self, poses_first: list[list[float]], poses_last: list[list[float]]):
        return (combine_poses(poses0=poses_first, poses1=poses_last),)


class CameraCtrlReplaceCameraParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "poses":("CAMERACTRL_POSES",),
                "fx": ("FLOAT", {"default": CAM.DEFAULT_FX, "min": 0, "max": 1, "step": 0.000000001}),
                "fy": ("FLOAT", {"default": CAM.DEFAULT_FY, "min": 0, "max": 1, "step": 0.000000001}),
                "cx": ("FLOAT", {"default": CAM.DEFAULT_CX, "min": 0, "max": 1, "step": 0.01}),
                "cy": ("FLOAT", {"default": CAM.DEFAULT_CY, "min": 0, "max": 1, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = ("CAMERACTRL_POSES",)
    FUNCTION = "set_camera_parameters"
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses"

    def set_camera_parameters(self, poses: list[list[float]], fx: float, fy: float, cx: float, cy: float):
        new_poses = copy.deepcopy(poses)
        for pose in new_poses:
            # fx,fy,cx,fy are in indexes 1-4 of the 19-long pose list
            pose[1] = fx
            pose[2] = fy
            pose[3] = cx
            pose[4] = cy
        return (new_poses,)


class CameraCtrlSetOriginalAspectRatio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "poses":("CAMERACTRL_POSES",),
                "orig_pose_width": ("INT", {"default": 1280, "min": 1, "max": BIGMAX}),
                "orig_pose_height": ("INT", {"default": 720, "min": 1, "max": BIGMAX}),
            }
        }
    
    RETURN_TYPES = ("CAMERACTRL_POSES",)
    FUNCTION = "set_aspect_ratio"
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses"

    def set_aspect_ratio(self, poses: list[list[float]], orig_pose_width: int, orig_pose_height: int):
        return (set_original_pose_dims(poses, pose_width=orig_pose_width, pose_height=orig_pose_height),)
