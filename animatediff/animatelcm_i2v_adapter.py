# Modified from code provided by Fu-Yun Wang (G-U-N on github)
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import comfy.ops
import comfy.model_management


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


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


# based on PositionalEncoding of AnimateDiff
def fixed_positional_embedding(t, d_model):
    position = torch.arange(0, t, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float()
                         * (-np.log(10000.0) / d_model))
    pos_embedding = torch.zeros(t, d_model)
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)
    return pos_embedding


class AdapterEmbed(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280],
                 nums_rb=3, cin=64, ksize=3, sk=False, use_conv=True,
                 ops=comfy.ops.disable_weight_init):
        super(AdapterEmbed, self).__init__()
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i != 0) and (j == 0):
                    self.body.append(ResnetBlockEmbed(
                        channels[i-1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv, ops=ops
                    ))
                else:
                    self.body.append(ResnetBlockEmbed(
                        channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv, ops=ops
                    ))
        self.body = nn.ModuleList(self.body)
        self.conv_in = zero_module(ops.Conv2d(in_channels=cin, out_channels=channels[0],
                                              kernel_size=3, stride=1, padding=1))
        self.d_model = channels[0]
        # settable
        self.ref_drift = 0.5
        self.insertion_weights = [1.0, 1.0, 1.0, 1.0]

    def set_ref_drift(self, ref_drift: float):
        if ref_drift is None:
            ref_drift = 0.5
        self.ref_drift = ref_drift
    
    def set_insertion_weights(self, insertion_weights: list[float]):
        if insertion_weights is None:
            insertion_weights = [1.0, 1.0, 1.0, 1.0]
        assert len(insertion_weights) == 4
        self.insertion_weights = insertion_weights
    
    def cleanup(self):
        self.set_ref_drift(None)
        self.set_insertion_weights(None)

    def forward(self, x: Tensor, video_length: int, batched_number: int):
        b, c, h, w = x.shape

        features = []

        use_dtype = comfy.model_management.unet_dtype()
        # allow fp8 to work
        if comfy.model_management.dtype_size(use_dtype) == 1:
            use_dtype = x.dtype

        x = self.conv_in(x.to(use_dtype))

        pos_embedding = fixed_positional_embedding(
            video_length, self.d_model).to(use_dtype).to(x.device)
        pos_embedding = pos_embedding.unsqueeze(-1).unsqueeze(-1)
        pos_embedding = pos_embedding.expand(-1, -1, h, w)
        # add x_pos with influence amount
        x = x + (pos_embedding * self.ref_drift)

        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                # get real index in self.body that corresponds to current channel/resnetblock
                idx = i*self.nums_rb + j
                x = self.body[idx](x)
            # match real_x to batched_number
            real_x = x.repeat(batched_number, 1, 1, 1)
            features.append(real_x)
        features = [weight * feature for weight, feature in zip(features, self.insertion_weights)]
        return features


class ResnetBlockEmbed(nn.Module):
    def __init__(self, in_c, out_c, down: bool, ksize=3, sk=False, use_conv=True,
                 ops=comfy.ops.disable_weight_init):
        super().__init__()
        ps = ksize // 2 # padding size
        if in_c != out_c or sk == False:
            self.in_conv = zero_module(ops.Conv2d(in_c, out_c, ksize, 1, ps))
        else:
            self.in_conv = None
        self.block1 = ops.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = zero_module(ops.Conv2d(out_c, out_c, ksize, 1, ps))
        if sk == False:
            self.skep = ops.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None
        
        self.down = down
        if self.down == True:
            self.down_opt = DownsampleEmbed(in_c, use_conv=use_conv, ops=ops)
    
    def forward(self, x: Tensor):
        if self.down == True:
            x = self.down_opt(x)
        
        if self.in_conv is not None:
            x = self.in_conv(x)
        
        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)

        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class DownsampleEmbed(nn.Module):
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
        self.out_channels = out_channels or channels  # use channels if out_channels is None
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

        kernel_size = (2, 2)

        input_height, input_width = x.size(-2), x.size(-1)

        padding_height = (
            math.ceil(input_height / kernel_size[0]) * kernel_size[0]) - input_height
        padding_width = (
            math.ceil(input_width / kernel_size[1]) * kernel_size[1]) - input_width

        x = F.pad(x, (0, padding_width, 0, padding_height), mode='replicate')

        return self.operation(x)
