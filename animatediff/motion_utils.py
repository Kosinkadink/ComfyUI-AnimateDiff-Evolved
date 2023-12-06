from abc import ABC, abstractmethod
from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import comfy.model_management as model_management
import comfy.ops
from comfy.cli_args import args
from comfy.ldm.modules.attention import attention_basic, attention_pytorch, attention_split, attention_sub_quad, default
from comfy.controlnet import broadcast_image_to
from comfy.utils import repeat_to_batch_size

from .motion_lora import MotionLoraInfo
from .logger import logger


# until xformers bug is fixed, do not use xformers for VersatileAttention! TODO: change this when fix is out
# logic for choosing optimized_attention method taken from comfy/ldm/modules/attention.py
optimized_attention_mm = attention_basic
if model_management.xformers_enabled():
    pass
    #optimized_attention_mm = attention_xformers
if model_management.pytorch_attention_enabled():
    optimized_attention_mm = attention_pytorch
else:
    if args.use_split_cross_attention:
        optimized_attention_mm = attention_split
    else:
        optimized_attention_mm = attention_sub_quad


class CrossAttentionMM(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., dtype=None, device=None, operations=comfy.ops):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = None
        self.default_scale = dim_head ** -0.5

        self.to_q = operations.Linear(query_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = operations.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = operations.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)

        self.to_out = nn.Sequential(operations.Linear(inner_dim, query_dim, dtype=dtype, device=device), nn.Dropout(dropout))

    def forward(self, x, context=None, value=None, mask=None, scale_mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k: Tensor = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        # apply custom scale by multiplying k by scale factor
        if self.scale is not None:
            k *= self.scale
        
        # apply scale mask, if present
        if scale_mask is not None:
            k *= scale_mask

        out = optimized_attention_mm(q, k, v, self.heads, mask)
        return self.to_out(out)


# super class to TemporalTransformer-like classes
class TemporalTransformerGeneric:
    def temporal_transformer_init(self, default_length: int):
        self.video_length = default_length
        self.full_length = default_length
        self.scale_min = 1.0
        self.scale_max = 1.0
        self.raw_scale_mask: Union[Tensor, None] = None
        self.temp_scale_mask: Union[Tensor, None] = None
        self.sub_idxs: Union[list[int], None] = None
        self.prev_hidden_states_batch = 0

    def reset_temp_vars(self):
        del self.temp_scale_mask
        self.temp_scale_mask = None
        self.prev_hidden_states_batch = 0

    def get_scale_mask(self, hidden_states: Tensor) -> Union[Tensor, None]:
        # if no raw mask, return None
        if self.raw_scale_mask is None:
            return None
        shape = hidden_states.shape
        batch, channel, height, width = shape
        # if temp mask already calculated, return it
        if self.temp_scale_mask != None:
            # check if hidden_states batch matches
            if batch == self.prev_hidden_states_batch:
                if self.sub_idxs is not None:
                    return self.temp_scale_mask[:, self.sub_idxs, :]
                return self.temp_scale_mask
            # if does not match, reset cached temp_scale_mask and recalculate it
            del self.temp_scale_mask
            self.temp_scale_mask = None
        # otherwise, calculate temp mask
        self.prev_hidden_states_batch = batch
        mask = prepare_mask_batch(self.raw_scale_mask, shape=(self.full_length, 1, height, width))
        mask = repeat_to_batch_size(mask, self.full_length)
        # if mask not the same amount length as full length, make it match
        if self.full_length != mask.shape[0]:
            mask = broadcast_image_to(mask, self.full_length, 1)
        # reshape mask to attention K shape (h*w, latent_count, 1)
        batch, channel, height, width = mask.shape
        # first, perform same operations as on hidden_states,
        # turning (b, c, h, w) -> (b, h*w, c)
        mask = mask.permute(0, 2, 3, 1).reshape(batch, height*width, channel)
        # then, make it the same shape as attention's k, (h*w, b, c)
        mask = mask.permute(1, 0, 2)
        # make masks match the expected length of h*w
        batched_number = shape[0] // self.video_length
        if batched_number > 1:
            mask = torch.cat([mask] * batched_number, dim=0)
        # cache mask and set to proper device
        self.temp_scale_mask = mask
        # move temp_scale_mask to proper dtype + device
        self.temp_scale_mask = self.temp_scale_mask.to(dtype=hidden_states.dtype, device=hidden_states.device)
        # return subset of masks, if needed
        if self.sub_idxs is not None:
            return self.temp_scale_mask[:, self.sub_idxs, :]
        return self.temp_scale_mask


class BlockType:
    UP = "up"
    DOWN = "down"
    MID = "mid"


class GenericMotionWrapper(nn.Module, ABC):
    def __init__(self, mm_hash: str, mm_name: str, loras: list[MotionLoraInfo]):
        super().__init__()
        self.down_blocks: nn.ModuleList = None
        self.up_blocks: nn.ModuleList = None
        self.mid_block = None
        self.mm_hash = mm_hash
        self.mm_name = mm_name
        self.version = "FILLTHISIN"
        self.injector_version = "VERYIMPORTANT_FILLTHISIN"
        self.AD_video_length: int = 0
        self.loras = loras

    def has_loras(self) -> bool:
        # TODO: fix this to return False if has an empty list as well
        # but only after implementing a fix for lowvram loading
        return self.loras is not None

    @abstractmethod
    def set_video_length(self, video_length: int, full_length: int):
        pass

    @abstractmethod
    def set_scale_multiplier(self, multiplier: Union[float, None]):
        pass

    @abstractmethod
    def set_masks(self, masks: Tensor, min_val: float, max_val: float):
        pass

    @abstractmethod
    def set_sub_idxs(self, sub_idxs: list[int]):
        pass
    
    @abstractmethod
    def reset_temp_vars(self):
        pass

    def reset_scale_multiplier(self):
        self.set_scale_multiplier(None)

    def reset_sub_idxs(self):
        self.set_sub_idxs(None)

    def reset(self):
        self.reset_sub_idxs()
        self.reset_scale_multiplier()
        self.reset_temp_vars()


class GroupNormAD(torch.nn.GroupNorm):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine, device=device, dtype=dtype)
    
    def forward(self, input: Tensor) -> Tensor:
        return F.group_norm(
             input, self.num_groups, self.weight, self.bias, self.eps)


# applies min-max normalization, from:
# https://stackoverflow.com/questions/68791508/min-max-normalization-of-a-tensor-in-pytorch
def normalize_min_max(x: Tensor, new_min = 0.0, new_max = 1.0):
    x_min, x_max = x.min(), x.max()
    return (((x - x_min)/(x_max - x_min)) * (new_max - new_min)) + new_min


# adapted from comfy/sample.py
def prepare_mask_batch(mask: Tensor, shape: Tensor, multiplier: int=1, match_dim1=False):
    mask = mask.clone()
    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[2]*multiplier, shape[3]*multiplier), mode="bilinear")
    if match_dim1:
        mask = torch.cat([mask] * shape[1], dim=1)
    return mask


class NoiseType:
    DEFAULT = "default"
    REPEATED = "repeated"
    CONSTANT = "constant"
    AUTO1111 = "auto1111"

    LIST = [DEFAULT, REPEATED, CONSTANT, AUTO1111]

    @classmethod
    def prepare_noise(cls, noise_type: str, latents: Tensor, noise: Tensor, context_length: int, seed: int):
        if noise_type == cls.DEFAULT:
            return noise
        elif noise_type == cls.REPEATED:
            return cls.prepare_noise_repeated(latents, noise, context_length, seed)
        elif noise_type == cls.CONSTANT:
            return cls.prepare_noise_constant(latents, noise, context_length, seed)
        elif noise_type == cls.AUTO1111:
            return cls.prepare_noise_auto1111(latents, noise, context_length, seed)
        logger.warning(f"Noise type {noise_type} not recognized, proceeding with default noise.")
        return noise

    @classmethod
    def prepare_noise_repeated(cls, latents: Tensor, noise: Tensor, context_length: int, seed: int):
        if not context_length:
            return noise
        length = latents.shape[0]
        generator = torch.manual_seed(seed)
        noise = torch.randn(latents.size(), dtype=latents.dtype, layout=latents.layout, generator=generator, device="cpu")
        noise_set = noise[:context_length]
        cat_count = (length // context_length) + 1
        noise_set = torch.cat([noise_set] * cat_count, dim=0)
        noise_set = noise_set[:length]
        return noise_set

    @classmethod
    def prepare_noise_constant(cls, latents: Tensor, noise: Tensor, context_length: int, seed: int):
        length = latents.shape[0]
        single_shape = (1, latents.shape[1], latents.shape[2], latents.shape[3])
        generator = torch.manual_seed(seed)
        noise = torch.randn(single_shape, dtype=latents.dtype, layout=latents.layout, generator=generator, device="cpu")
        return torch.cat([noise] * length, dim=0)

    @classmethod
    def prepare_noise_auto1111(cls, latents: Tensor, noise: Tensor, context_length: int, seed: int):
        # auto1111 applies growing seeds for a batch
        length = latents.shape[0]
        single_shape = (1, latents.shape[1], latents.shape[2], latents.shape[3])
        all_noises = []
        # i starts at 0
        for i in range(length):
            generator = torch.manual_seed(seed+i)
            all_noises.append(torch.randn(single_shape, dtype=latents.dtype, layout=latents.layout, generator=generator, device="cpu"))
        return torch.cat(all_noises, dim=0)


class MotionCompatibilityError(ValueError):
    pass
