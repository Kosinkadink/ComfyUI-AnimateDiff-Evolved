from typing import Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import comfy.model_management as model_management
import comfy.ops
import comfy.utils
from comfy.cli_args import args
from comfy.ldm.modules.attention import attention_basic, attention_pytorch, attention_split, attention_sub_quad, default

from .logger import logger


# until xformers bug is fixed, do not use xformers for VersatileAttention! TODO: change this when fix is out
# logic for choosing optimized_attention method taken from comfy/ldm/modules/attention.py
# a fallback_attention_mm is selected to avoid CUDA configuration limitation with pytorch's scaled_dot_product
optimized_attention_mm = attention_basic
fallback_attention_mm = attention_basic
if model_management.xformers_enabled():
    pass
    #optimized_attention_mm = attention_xformers
if model_management.pytorch_attention_enabled():
    optimized_attention_mm = attention_pytorch
    if args.use_split_cross_attention:
        fallback_attention_mm = attention_split
    else:
        fallback_attention_mm = attention_sub_quad
else:
    if args.use_split_cross_attention:
        optimized_attention_mm = attention_split
    else:
        optimized_attention_mm = attention_sub_quad


class CrossAttentionMM(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., dtype=None, device=None,
                 operations=comfy.ops.disable_weight_init):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.actual_attention = optimized_attention_mm
        self.heads = heads
        self.dim_head = dim_head
        self.scale = None
        self.default_scale = dim_head ** -0.5

        self.to_q = operations.Linear(query_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = operations.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = operations.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)

        self.to_out = nn.Sequential(operations.Linear(inner_dim, query_dim, dtype=dtype, device=device), nn.Dropout(dropout))

    def reset_attention_type(self):
        self.actual_attention = optimized_attention_mm

    def forward(self, x, context=None, value=None, mask=None, scale_mask=None, mm_kwargs=None, transformer_options=None):
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

        try:
            out = self.actual_attention(q, k, v, self.heads, mask)
        except RuntimeError as e:
            if str(e).startswith("CUDA error: invalid configuration argument"):
                self.actual_attention = fallback_attention_mm
                out = self.actual_attention(q, k, v, self.heads, mask)
            else:
                raise
        return self.to_out(out)

# TODO: set up comfy.ops style classes for groupnorm and other functions
class GroupNormAD(torch.nn.GroupNorm):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine, device=device, dtype=dtype)
    
    def forward(self, input: Tensor) -> Tensor:
        return F.group_norm(
             input, self.num_groups, self.weight, self.bias, self.eps)


# applies min-max normalization, from:
# https://stackoverflow.com/questions/68791508/min-max-normalization-of-a-tensor-in-pytorch
def normalize_min_max(x: Tensor, new_min=0.0, new_max=1.0):
    return linear_conversion(x, x_min=x.min(), x_max=x.max(), new_min=new_min, new_max=new_max)


def linear_conversion(x, x_min=0.0, x_max=1.0, new_min=0.0, new_max=1.0):
    return (((x - x_min)/(x_max - x_min)) * (new_max - new_min)) + new_min


# adapted from comfy/sample.py
def prepare_mask_batch(mask: Tensor, shape: Tensor, multiplier: int=1, match_dim1=False):
    mask = mask.clone()
    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[2]*multiplier, shape[3]*multiplier), mode="bilinear")
    if match_dim1:
        mask = torch.cat([mask] * shape[1], dim=1)
    return mask


def extend_to_batch_size(tensor: Tensor, batch_size: int):
    if tensor.shape[0] > batch_size:
        return tensor[:batch_size]
    elif tensor.shape[0] < batch_size:
        remainder = batch_size-tensor.shape[0]
        return torch.cat([tensor] + [tensor[-1:]]*remainder, dim=0)
    return tensor


def extend_list_to_batch_size(_list: list, batch_size: int):
    if len(_list) > batch_size:
        return _list[:batch_size]
    elif len(_list) < batch_size:
        return _list + _list[-1:]*(batch_size-len(_list))
    return _list.copy()


# from comfy/controlnet.py
def ade_broadcast_image_to(tensor, target_batch_size, batched_number):
    current_batch_size = tensor.shape[0]
    #print(current_batch_size, target_batch_size)
    if current_batch_size == 1:
        return tensor

    per_batch = target_batch_size // batched_number
    tensor = tensor[:per_batch]

    if per_batch > tensor.shape[0]:
        tensor = torch.cat([tensor] * (per_batch // tensor.shape[0]) + [tensor[:(per_batch % tensor.shape[0])]], dim=0)

    current_batch_size = tensor.shape[0]
    if current_batch_size == target_batch_size:
        return tensor
    else:
        return torch.cat([tensor] * batched_number, dim=0)


# originally from comfy_extras/nodes_mask.py::composite function
def composite_extend(destination: Tensor, source: Tensor, x: int, y: int, mask: Tensor = None, multiplier = 8, resize_source = False):
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

    source = extend_to_batch_size(source, destination.shape[0])

    x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
    y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (left + source.shape[3], top + source.shape[2],)

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
        mask = extend_to_batch_size(mask, source.shape[0])

    # calculate the bounds of the source that will be overlapping the destination
    # this prevents the source trying to overwrite latent pixels that are out of bounds
    # of the destination
    visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

    mask = mask[:, :, :visible_height, :visible_width]
    inverse_mask = torch.ones_like(mask) - mask

    source_portion = mask * source[:, :, :visible_height, :visible_width]
    destination_portion = inverse_mask  * destination[:, :, top:bottom, left:right]

    destination[:, :, top:bottom, left:right] = source_portion + destination_portion
    return destination


def get_sorted_list_via_attr(objects: list, attr: str) -> list:
    if not objects:
        return objects
    elif len(objects) <= 1:
        return [x for x in objects]
    # now that we know we have to sort, do it following these rules:
    # a) if objects have same value of attribute, maintain their relative order
    # b) perform sorting of the groups of objects with same attributes
    unique_attrs = {}
    for o in objects:
        val_attr = getattr(o, attr)
        attr_list: list = unique_attrs.get(val_attr, list())
        attr_list.append(o)
        if val_attr not in unique_attrs:
            unique_attrs[val_attr] = attr_list
    # now that we have the unique attr values grouped together in relative order, sort them by key
    sorted_attrs = dict(sorted(unique_attrs.items()))
    # now flatten out the dict into a list to return
    sorted_list = []
    for object_list in sorted_attrs.values():
        sorted_list.extend(object_list)
    return sorted_list


class MotionCompatibilityError(ValueError):
    pass


class InputPIA(ABC):
    def __init__(self, effect_multival: Union[float, Tensor]=None):
        self.effect_multival = effect_multival if effect_multival is not None else 1.0

    @abstractmethod
    def get_mask(self, x: Tensor):
        pass


class InputPIA_Multival(InputPIA):
    def __init__(self, multival: Union[float, Tensor], effect_multival: Union[float, Tensor]=None):
        super().__init__(effect_multival=effect_multival)
        self.multival = multival

    def get_mask(self, x: Tensor):
        if type(self.multival) is Tensor:
            return self.multival
        # if not Tensor, then is float, and simply return a mask with the right dimensions + value
        b, c, h, w = x.shape
        mask = torch.ones(size=(b, h, w))
        return mask * self.multival


def create_multival_combo(float_val: Union[float, list[float]], mask_optional: Tensor=None):
    # first, normalize inputs
    # if float_val is iterable, treat as a list and assume inputs are floats
    float_is_iterable = False
    if isinstance(float_val, Iterable):
        float_is_iterable = True
        float_val = list(float_val)
        # if mask present, make sure float_val list can be applied to list - match lengths
        if mask_optional is not None:
            if len(float_val) < mask_optional.shape[0]:
                # copies last entry enough times to match mask shape
                float_val = extend_list_to_batch_size(float_val, mask_optional.shape[0])
            if mask_optional.shape[0] < len(float_val):
                mask_optional = extend_to_batch_size(mask_optional, len(float_val))
            float_val = float_val[:mask_optional.shape[0]]
        float_val: Tensor = torch.tensor(float_val).unsqueeze(-1).unsqueeze(-1)
    # now that inputs are normalized, figure out what value to actually return
    if mask_optional is not None:
        mask_optional = mask_optional.clone()
        if float_is_iterable:
            mask_optional = mask_optional[:] * float_val.to(mask_optional.dtype).to(mask_optional.device)
        else:
            mask_optional = mask_optional * float_val
        return mask_optional
    else:
        if not float_is_iterable:
            return float_val
        # create a dummy mask of b,h,w=float_len,1,1 (sigle pixel)
        # purpose is for float input to work with mask code, without special cases
        float_len = float_val.shape[0] if float_is_iterable else 1
        shape = (float_len,1,1)
        mask_optional = torch.ones(shape)
        mask_optional = mask_optional[:] * float_val.to(mask_optional.dtype).to(mask_optional.device)
        return mask_optional


def get_combined_multival(multivalA: Union[float, Tensor], multivalB: Union[float, Tensor], force_leader_A=False) -> Union[float, Tensor]:
    if multivalA is None and multivalB is None:
        return 1.0
    # if one is None, use the other
    if multivalA is None:
        return multivalB
    elif multivalB is None:
        return multivalA 
    # both have a value - combine them based on type
    # if both are Tensors, make dims match before multiplying
    if type(multivalA) == Tensor and type(multivalB) == Tensor:
        if force_leader_A:
            leader,follower = (multivalA,multivalB)
            batch_size = multivalA.shape[0]
        else:
            areaA = multivalA.shape[1]*multivalA.shape[2]
            areaB = multivalB.shape[1]*multivalB.shape[2]
            # match height/width to mask with larger area
            leader,follower = (multivalA,multivalB) if areaA >= areaB else (multivalB,multivalA)
            batch_size = multivalA.shape[0] if multivalA.shape[0] >= multivalB.shape[0] else multivalB.shape[0]
        # make follower same dimensions as leader
        follower = torch.unsqueeze(follower, 1)
        follower = comfy.utils.common_upscale(follower, leader.shape[-1], leader.shape[-2], "bilinear", "center")
        follower = torch.squeeze(follower, 1)
        # make sure batch size will match
        leader = extend_to_batch_size(leader, batch_size)
        follower = extend_to_batch_size(follower, batch_size)
        return leader * follower
    # otherwise, just multiply them together - one of them is a float
    return multivalA * multivalB


def resize_multival(multival: Union[float, Tensor], batch_size: int, height: int, width: int):
    if multival is None:
        return 1.0
    if type(multival) != Tensor:
        return multival
    multival = torch.unsqueeze(multival, 1)
    multival = comfy.utils.common_upscale(multival, height, width, "bilinear", "center")
    multival = torch.squeeze(multival, 1)
    multival = extend_to_batch_size(multival, batch_size)
    return multival


def get_combined_input(inputA: Union[InputPIA, None], inputB: Union[InputPIA, None], x: Tensor):
    if inputA is None:
        inputA = InputPIA_Multival(1.0)
    if inputB is None:
        inputB = InputPIA_Multival(1.0)
    return get_combined_multival(inputA.get_mask(x), inputB.get_mask(x))


def get_combined_input_effect_multival(inputA: Union[InputPIA, None], inputB: Union[InputPIA, None]):
    if inputA is None:
        inputA = InputPIA_Multival(1.0)
    if inputB is None:
        inputB = InputPIA_Multival(1.0)
    return get_combined_multival(inputA.effect_multival, inputB.effect_multival)


#######################
# Facilitate Per-Block Effect and Scale Control
class PerAttn:
    def __init__(self, attn_idx: Union[int, None], scale: Union[float, Tensor, None]):
        self.attn_idx = attn_idx
        self.scale = scale
    
    def matches(self, id: int):
        if self.attn_idx is None:
            return True
        return self.attn_idx == id


class PerBlockId:
    def __init__(self, block_type: str, block_idx: Union[int, None]=None, module_idx: Union[int, None]=None):
        self.block_type = block_type
        self.block_idx = block_idx
        self.module_idx = module_idx
    
    def matches(self, other: 'PerBlockId') -> bool:
        # block_type
        if other.block_type != self.block_type:
            return False
        # block_idx
        if other.block_idx is None:
            return True
        elif other.block_idx != self.block_idx:
            return False
        # module_idx
        if other.module_idx is None:
            return True
        return other.module_idx == self.module_idx
    
    def __str__(self):
        return f"PerBlockId({self.block_type},{self.block_idx},{self.module_idx})"


class PerBlock:
    def __init__(self, id: PerBlockId, effect: Union[float, Tensor, None]=None,
                 scales: Union[list[Union[float, Tensor, None]], None]=None):
        self.id = id
        self.effect = effect
        self.scales = scales

    def matches(self, id: PerBlockId):
        return self.id.matches(id)
    

@dataclass
class AllPerBlocks:
    per_block_list: list[PerBlock]
    sd_type: Union[str, None] = None


def get_combined_per_block_list(listDefault: Union[list[PerBlock], None], listNew: Union[list[PerBlock], None]):
    if listDefault is None:
        return listNew
    elif listNew is None:
        return listDefault
    else:
        return listNew
#----------------------
#######################


class ADKeyframe:
    def __init__(self,
                 start_percent: float = 0.0,
                 scale_multival: Union[float, Tensor]=None,
                 effect_multival: Union[float, Tensor]=None,
                 per_block_replace: AllPerBlocks=None,
                 cameractrl_multival: Union[float, Tensor]=None,
                 pia_input: InputPIA=None,
                 inherit_missing: bool=True,
                 guarantee_steps: int=1,
                 default: bool=False,
                 ):
        self.start_percent = start_percent
        self.start_t = 999999999.9
        self.scale_multival = scale_multival
        self.effect_multival = effect_multival
        self._per_block_replace = per_block_replace
        self.cameractrl_multival = cameractrl_multival
        self.pia_input = pia_input
        self.inherit_missing = inherit_missing
        self.guarantee_steps = guarantee_steps
        self.default = default
    
    @property
    def per_block_list(self):
        if self._per_block_replace is None:
            return None
        return self._per_block_replace.per_block_list

    def has_scale(self):
        return self.scale_multival is not None
    
    def has_effect(self):
        return self.effect_multival is not None

    def has_per_block_replace(self):
        return self._per_block_replace is not None

    def has_cameractrl_effect(self):
        return self.cameractrl_multival is not None
    
    def has_pia_input(self):
        return self.pia_input is not None

    def get_effective_guarantee_steps(self, max_sigma: torch.Tensor):
        '''If keyframe starts before current sampling range (max_sigma), treat as 0.'''
        if self.start_t > max_sigma:
            return 0
        return self.guarantee_steps


class ADKeyframeGroup:
    def __init__(self):
        self.keyframes: list[ADKeyframe] = []
        self.keyframes.append(ADKeyframe(guarantee_steps=1, default=True))
    
    def add(self, keyframe: ADKeyframe):
        # remove any default keyframes that match start_percent of new keyframe
        default_to_delete = []
        for i in range(len(self.keyframes)):
            if self.keyframes[i].default and self.keyframes[i].start_percent == keyframe.start_percent:
                default_to_delete.append(i)
        for i in reversed(default_to_delete):
            self.keyframes.pop(i)
        # add to end of list, then sort
        self.keyframes.append(keyframe)
        self.keyframes = get_sorted_list_via_attr(self.keyframes, "start_percent")
    
    def get_index(self, index: int) -> Union[ADKeyframe, None]:
        try:
            return self.keyframes[index]
        except IndexError:
            return None
    
    def has_index(self, index: int) -> int:
        return index >=0 and index < len(self.keyframes)

    def __getitem__(self, index) -> ADKeyframe:
        return self.keyframes[index]

    def __len__(self) -> int:
        return len(self.keyframes)

    def is_empty(self) -> bool:
        return len(self.keyframes) == 0

    def clone(self) -> 'ADKeyframeGroup':
        cloned = ADKeyframeGroup()
        for tk in self.keyframes:
            if not tk.default:
                cloned.add(tk)
        return cloned


class DummyNNModule(nn.Module):
    class DoNothingWhenCalled:
        def __call__(self, *args, **kwargs):
            return

    '''
    Class that does not throw exceptions for almost anything you throw at it. As name implies, does nothing.
    '''
    def __init__(self):
        super().__init__()

    def __getattr__(self, *args, **kwargs):
        return self.DoNothingWhenCalled()
    
    def __setattr__(self, name, value):
        pass
    
    def __iter__(self, *args, **kwargs):
        pass
    
    def __next__(self, *args, **kwargs):
        pass

    def __len__(self, *args, **kwargs):
        pass
    
    def __getitem__(self, *args, **kwargs):
        pass
    
    def __setitem__(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        pass
