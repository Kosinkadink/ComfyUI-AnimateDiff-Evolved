
# Modified from https://github.com/hehao13/CameraCtrl/blob/main/cameractrl/models/pose_adaptor.py
# (whose parts were also taken from https://github.com/TencentARC/T2I-Adapter)
import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict
from einops import rearrange

import comfy.ops

from .motion_module_ad import TemporalTransformerBlock


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


# class PoseAdapter(nn.Module):
#     def __init__(self, unet, pose_encoder):
#         super().__init__()
#         self.unet = unet
#         self.pose_encoder = pose_encoder
    
#     def forward(self, noisy_latents: Tensor, timesteps, encoder_hidden_states: Tensor, pose_embedding):
#         # original code needed to convert from 4 dims (bf c h w) to 5 dims (b c f h w),
#         # but ComfyUI already deals with everything in 4 dims
#         pose_embedding_features = self.pose_encoder(pose_embedding)
#         noise_pred = self.unet(noisy_latents,
#                                timesteps,
#                                encoder_hidden_states,
#                                pose_embedding_features).sample
#         return noise_pred


class CameraPoseEncoder(nn.Module):
    def __init__(self,
                 downscale_factor=8,
                 channels=[320, 640, 1280, 1280],
                 nums_rb=3,
                 cin=64,
                 ksize=3,
                 sk=False,
                 use_conv=True,
                 compression_factor=1,
                 temporal_attention_nhead=8,
                 attention_block_types=("Temporal_Self", ),
                 temporal_position_encoding=False,
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
            for j in range(len(nums_rb)):
                if j == 0 and i != 0:
                    in_dim = channels[i - 1]
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=True, ksize=ksize, sk=sk, use_conv=use_conv, ops=ops)
                elif j == 0:
                    in_dim = channels[0]
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv, ops=ops)
                elif j == nums_rb - 1:
                    in_dim = channels[i] / compression_factor
                    out_dim = channels[i]
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv, ops=ops)
                else:
                    in_dim = int(channels[i] / compression_factor)
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv, ops=ops)
                temporal_attention_layer = TemporalTransformerBlock(dim=out_dim,
                                                                    num_attention_heads=temporal_attention_nhead,
                                                                    attention_head_dim=int(out_dim / temporal_attention_nhead),
                                                                    attention_block_types=attention_block_types,
                                                                    dropout=0.0,
                                                                    cross_attention_dim=None,
                                                                    temporal_position_encoding=temporal_position_encoding,
                                                                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                                                                    rescale_output_factor=rescale_output_factor,
                                                                    ops=ops)
                conv_layers.append(conv_layer)
                temporal_attention_layers.append(temporal_attention_layer)
            self.encoder_down_conv_blocks.append(conv_layers)
            self.encoder_down_attention_blocks.append(temporal_attention_layers)

    def forward(self, x: Tensor):
        # unshuffle
        x = self.unshuffle
        # extract features
        features = []
        x = self.encoder_conv_in(x)
        for res_block, attention_block in zip(self.encoder_down_conv_blocks, self.encoder_down_attention_blocks):
            for res_layer, attention_layer in zip(res_block, attention_block):
                x = res_layer(x)
                h, w = x.shape[-2:]
                x = rearrange(x, 'b c h w -> (h w) b c')
                x = attention_layer(x)
                x = rearrange(x, '(h w) b c -> b c h w', h=h, w=w)
            features.append(x)
        return features


class ResnetBlock(nn.Module):
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
            self.down_opt = Downsample(in_c, use_conv=use_conv)

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


class Downsample(nn.Module):
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
