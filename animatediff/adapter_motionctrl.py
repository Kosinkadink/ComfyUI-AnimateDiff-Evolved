# main code adapted from https://github.com/TencentARC/MotionCtrl/tree/animatediff
from __future__ import annotations
from torch import nn, Tensor

from comfy.model_patcher import ModelPatcher
import comfy.model_management
import comfy.ops
import comfy.utils

from .adapter_cameractrl import ResnetBlockCameraCtrl
from .motion_module_ad import AnimateDiffModel
from .utils_model import get_motion_model_path

# cmcm (Camera Control)
def injection_motionctrl_cmcm(motion_model: AnimateDiffModel, cmcm_name: str):
    pass


# omcm (Object Control)
def load_motionctrl_omcm(omcm_name: str):
    omcm_path = get_motion_model_path(omcm_name)
    state_dict = comfy.utils.load_torch_file(omcm_path, safe_load=True)
    for key in list(state_dict.keys()):
        # remove 'module.' prefix
        if key.startswith('module.'):
            new_key = key.replace('module.', '')
            state_dict[new_key] = state_dict[key]
            state_dict.pop(key)
    
    if comfy.model_management.unet_manual_cast(comfy.model_management.unet_dtype(), comfy.model_management.get_torch_device()) is None:
        ops = comfy.ops.disable_weight_init
    else:
        ops = comfy.ops.manual_cast
    adapter = MotionCtrlAdapter(ops=ops)
    adapter.load_state_dict(state_dict=state_dict, strict=True)
    adapter.to(
        device = comfy.model_management.unet_offload_device(),
        dtype = comfy.model_management.unet_dtype()
    )
    omcm_modelpatcher = _create_OMCMModelPatcher(model=adapter,
                                                load_device=comfy.model_management.get_torch_device(),
                                                offload_device=comfy.model_management.unet_offload_device())
    return omcm_modelpatcher


def _create_OMCMModelPatcher(model, load_device, offload_device) -> ObjectControlModelPatcher:
    patcher = ModelPatcher(model, load_device=load_device, offload_device=offload_device)
    return patcher


class ObjectControlModelPatcher(ModelPatcher):
    '''Class only used for type hints.'''
    def __init__(self):
        self.model: MotionCtrlAdapter


class MotionCtrlAdapter(nn.Module):
    def __init__(self,
                 downscale_factor=8,
                 channels=[320, 640, 1280, 1280],
                 nums_rb=2, cin=128, # 2*8*8
                 ksize=3, sk=True,
                 use_conv=False,
                 ops=comfy.ops.disable_weight_init):
        super(MotionCtrlAdapter, self).__init__()
        self.downscale_factor = downscale_factor
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i != 0) and (j == 0):
                    self.body.append(
                        ResnetBlockCameraCtrl(channels[i - 1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv, ops=ops))
                else:
                    self.body.append(
                        ResnetBlockCameraCtrl(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv, ops=ops))
        self.body = nn.ModuleList(self.body)
        self.conv_in = ops.Conv2d(cin, channels[0], 3, 1, 1)
    
    def forward(self, x: Tensor):
        x = self.unshuffle(x)
        # extract features
        features = []
        x = self.conv_in(x)
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)
            features.append(x)
        return features
