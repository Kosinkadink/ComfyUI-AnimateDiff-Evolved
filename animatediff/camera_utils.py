import copy
# import plotly.express as px
# import plotly.graph_objects as go
import json

import numpy as np

CAMERA_MOTION_MODE = ["Basic Camera Poses", "Provided Complex Camera Poses", "Custom Camera Poses"]

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

COMPLEX_CAMERA = {
    "Pose_1": "examples/camera_poses/test_camera_1424acd0007d40b5.json",
    "Pose_2": "examples/camera_poses/test_camera_d971457c81bca597.json",
    "Pose_3": "examples/camera_poses/test_camera_Round-ZoomIn.json",
    "Pose_4": "examples/camera_poses/test_camera_Round-RI_90.json",
    "Pose_5": "examples/camera_poses/test_camera_Round-RI-120.json",
    "Pose_6": "examples/camera_poses/test_camera_018f7907401f2fef.json",
    "Pose_7": "examples/camera_poses/test_camera_088b93f15ca8745d.json",
    "Pose_8": "examples/camera_poses/test_camera_b133a504fc90a2d1.json",
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
    
    # 计算相机外参的旋转矩阵
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
    
def create_relative(RT_list, K_1=4.7, dataset="syn"):
    RT = copy.deepcopy(RT_list[0])
    R_inv = RT[:,:3].T
    T =  RT[:,-1]

    temp = []
    for _RT in RT_list:
        _RT[:,:3] = np.dot(_RT[:,:3], R_inv)
        _RT[:,-1] =  _RT[:,-1] - np.dot(_RT[:,:3], T)
        temp.append(_RT)
    RT_list = temp

    return RT_list
    
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

def process_camera(camera_dict):
    # "First A then B", "Both A and B", "Custom"
    if camera_dict['complex'] is not None:
        with open(COMPLEX_CAMERA[camera_dict['complex']]) as f:
            RT = json.load(f) # [16, 12]
        RT = np.array(RT).reshape(-1, 3, 4)
        print(RT.shape)
        return RT


    motion_list = camera_dict['motion']
    mode = camera_dict['mode']
    speed = camera_dict['speed']
    print(len(motion_list))
    if len(motion_list) == 0:
        angle = np.array([0,0,0])
        T = np.array([0,0,0])
        RT = get_camera_motion(angle, T, speed, 16)


    elif len(motion_list) == 1:
        angle = np.array(CAMERA[motion_list[0]]["angle"])
        T = np.array(CAMERA[motion_list[0]]["T"])
        print(angle, T)
        RT = get_camera_motion(angle, T, speed, 16)
        
        
    
    elif len(motion_list) == 2:
        if mode == "Customized Mode 1: First A then B":
            angle = np.array(CAMERA[motion_list[0]]["angle"]) 
            T = np.array(CAMERA[motion_list[0]]["T"]) 
            RT_0 = get_camera_motion(angle, T, speed, 8)

            angle = np.array(CAMERA[motion_list[1]]["angle"]) 
            T = np.array(CAMERA[motion_list[1]]["T"]) 
            RT_1 = get_camera_motion(angle, T, speed, 8)

            RT = combine_camera_motion(RT_0, RT_1)

        elif mode == "Customized Mode 2: Both A and B":
            angle = np.array(CAMERA[motion_list[0]]["angle"]) + np.array(CAMERA[motion_list[1]]["angle"])
            T = np.array(CAMERA[motion_list[0]]["T"]) + np.array(CAMERA[motion_list[1]]["T"])
            RT = get_camera_motion(angle, T, speed, 16)


    # return RT.reshape(-1, 12)
    return RT

