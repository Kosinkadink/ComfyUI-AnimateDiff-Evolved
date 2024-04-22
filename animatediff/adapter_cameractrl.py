# Modified from https://github.com/hehao13/CameraCtrl/blob/main/cameractrl/models/pose_adaptor.py
# (whose parts were also taken from https://github.com/TencentARC/T2I-Adapter)
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from einops import rearrange

import comfy.ops

from .context import ContextOptions, ContextFuseMethod, ContextSchedules
from .motion_module_ad import TemporalTransformerBlock, get_position_encoding_max_len
from .logger import logger


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class CameraEntry:
    def __init__(self, entry: list[float]):
        self.entry = entry.copy()
        self.orig_pose_width = entry[5]
        self.orig_pose_height = entry[6]
        # focal length/intrinsic camera parameters
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)
    
    def clone(self):
        return CameraEntry(entry=self.entry)


def get_parameter_dtype(parameter: torch.nn.Module):
    params = tuple(parameter.parameters())
    if len(params) > 0:
        return params[0].dtype

    buffers = tuple(parameter.buffers())
    if len(buffers) > 0:
        return buffers[0].dtype


def get_parameter_device(parameter: torch.nn.Module):
    params = tuple(parameter.parameters())
    if len(params) > 0:
        return params[0].device

    buffers = tuple(parameter.buffers())
    if len(buffers) > 0:
        return buffers[0].device


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    return torch.meshgrid(*args, indexing='ij')


def get_relative_pose(cam_params: list[CameraEntry]):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def ray_condition(K: Tensor, c2w: Tensor, H, W, device):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ directions
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


def prepare_pose_embedding(cam_params: list[CameraEntry], image_width, image_height):
    # clone each CameraEntry in list so that CameraEntries don't get spoiled after a single run
    cam_params = [entry.clone() for entry in cam_params]
    sample_wh_ratio = image_width / image_height

    for cam_param in cam_params:
        pose_wh_ratio = cam_param.orig_pose_width / cam_param.orig_pose_height

        if pose_wh_ratio > sample_wh_ratio:
            resized_ori_w = image_height * pose_wh_ratio
            cam_param.fx = resized_ori_w * cam_param.fx / image_width
        else:
            resized_ori_h = image_width / pose_wh_ratio
            cam_param.fy = resized_ori_h * cam_param.fy / image_height
    intrinsic = np.asarray([[cam_param.fx * image_width,
                             cam_param.fy * image_height,
                             cam_param.cx * image_width,
                             cam_param.cy * image_height]
                            for cam_param in cam_params], dtype=np.float32)
    
    K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
    c2ws = get_relative_pose(cam_params)
    c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
    plucker_embedding = ray_condition(K, c2ws, image_height, image_width, device='cpu')[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
    plucker_embedding = rearrange(plucker_embedding, "f c h w -> c f h w")
    return plucker_embedding


class CameraPoseEncoder(nn.Module):
    def __init__(self,
                 downscale_factor=8,
                 channels=[320, 640, 1280, 1280],
                 nums_rb=2,
                 cin=384,
                 ksize=1,
                 sk=True,
                 use_conv=False,
                 compression_factor=1,
                 temporal_attention_nhead=8,
                 attention_block_types=("Temporal_Self", ),
                 temporal_position_encoding=True,
                 temporal_position_encoding_max_len=16,
                 rescale_output_factor=1.0,
                 ops=comfy.ops.disable_weight_init):
        super(CameraPoseEncoder, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.channels = channels
        self.nums_rb = nums_rb
        self.encoder_conv_in = ops.Conv2d(cin, channels[0], 3, 1, 1)
        self.encoder_down_conv_blocks = nn.ModuleList()
        self.encoder_down_attention_blocks = nn.ModuleList()
        for i in range(len(channels)):
            conv_layers = nn.ModuleList()
            temporal_attention_layers = nn.ModuleList() 
            for j in range(nums_rb):
                if j == 0 and i != 0:
                    in_dim = channels[i - 1]
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlockCameraCtrl(in_dim, out_dim, down=True, ksize=ksize, sk=sk, use_conv=use_conv, ops=ops)
                elif j == 0:
                    in_dim = channels[0]
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlockCameraCtrl(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv, ops=ops)
                elif j == nums_rb - 1:
                    in_dim = channels[i] / compression_factor
                    out_dim = channels[i]
                    conv_layer = ResnetBlockCameraCtrl(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv, ops=ops)
                else:
                    in_dim = int(channels[i] / compression_factor)
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlockCameraCtrl(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv, ops=ops)
                temporal_attention_layer = TemporalTransformerBlock(dim=out_dim,
                                                                    num_attention_heads=temporal_attention_nhead,
                                                                    attention_head_dim=int(out_dim / temporal_attention_nhead),
                                                                    attention_block_types=attention_block_types,
                                                                    dropout=0.0,
                                                                    cross_attention_dim=None,
                                                                    temporal_pe=temporal_position_encoding,
                                                                    temporal_pe_max_len=temporal_position_encoding_max_len,
                                                                    ops=ops)
                conv_layers.append(conv_layer)
                temporal_attention_layers.append(temporal_attention_layer)
            self.encoder_down_conv_blocks.append(conv_layers)
            self.encoder_down_attention_blocks.append(temporal_attention_layers)
            self.temporal_pe_max_len = 16

    def forward(self, x: Tensor, video_length: int, batched_number: int=1):
        # rearrange to match expected format
        x = rearrange(x, "c f h w -> f c h w")
        # logger.info(f"x: {x.shape}, {float(x[0][0][0][-1])}")
        # unshuffle
        x = self.unshuffle(x)
        # extract features
        features = []
        # prepare view_options, if needed
        view_options = ContextOptions(
            context_length=self.temporal_pe_max_len,
            context_overlap=self.temporal_pe_max_len//2, # at 16 max_len, context_overlap will be 8
            context_schedule=ContextSchedules.STATIC_STANDARD,
            fuse_method=ContextFuseMethod.PYRAMID,
        )
        # logger.warn(f"x dtype: {x.dtype}, device: {x.device}")
        # logger.warn(f"dtype: {get_parameter_dtype(self)}, device: {get_parameter_device(self)}")
        x = self.encoder_conv_in(x.to(dtype=get_parameter_dtype(self), device=get_parameter_device(self)))
        for res_block, attention_block in zip(self.encoder_down_conv_blocks, self.encoder_down_attention_blocks):
            for res_layer, attention_layer in zip(res_block, attention_block):
                x = res_layer(x)
                h, w = x.shape[-2:]
                x = rearrange(x, 'b c h w -> b (h w) c')  # h w are in middle instead of beginning like in diffusers
                x = attention_layer(x, video_length=video_length, view_options=view_options)
                x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # h w are in middle instead of beginning like in diffusers
            features.append(x)
        # for idx, feature in enumerate(features):
        #     logger.info(f"{idx}: {feature.shape}, {float(feature[0][0][0][0])}")
        for idx, x1 in enumerate(features):
            x1 = x1.to(x.dtype).to(x.device)
            x1 = rearrange(x1, 'b c h w -> (h w) b c')
            x1 = torch.cat([x1] * batched_number, dim=0)
            features[idx] = x1
        return features


class ResnetBlockCameraCtrl(nn.Module):
    def __init__(self, in_c, out_c, down: bool, ksize=3, sk=False, use_conv=True,
                 ops=comfy.ops.disable_weight_init):
        super().__init__()
        ps = ksize // 2 # padding size
        if in_c != out_c or sk == False:
            self.in_conv = ops.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = ops.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = ops.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = ops.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = DownsampleCameraCtrl(in_c, use_conv=use_conv)

    def forward(self, x: Tensor):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class DownsampleCameraCtrl(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv: bool, dims=2, out_channels=None, padding=1,
                 ops=comfy.ops.disable_weight_init):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.operation = ops.conv_nd(dims, in_channels=self.channels, out_channels=self.out_channels,
                                         kernel_size=3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.operation = avg_pool_nd(dims, kernel_size=stride, stride=stride)  # both are stride value on purpose
    
    def forward(self, x: Tensor):
        assert x.shape[1] == self.channels
        return self.operation(x)
