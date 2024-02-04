from collections.abc import Iterable
from typing import Union

import torch
from torch import Tensor

from .utils_motion import linear_conversion, normalize_min_max, extend_to_batch_size


class ScaleType:
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    LIST = [ABSOLUTE, RELATIVE]


class MultivalDynamicNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float_val": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001},),
            },
            "optional": {
                "mask_optional": ("MASK",)
            }
        }
    
    RETURN_TYPES = ("MULTIVAL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/multival"
    FUNCTION = "create_multival"

    def create_multival(self, float_val: Union[float, list[float]]=1.0, mask_optional: Tensor=None):
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
                    float_val = float_val + float_val[-1]*(mask_optional.shape[0]-len(float_val))
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
            return (mask_optional,)
        else:
            if not float_is_iterable:
                return (float_val,)
            # create a dummy mask of b,h,w=float_len,1,1 (sigle pixel)
            # purpose is for float input to work with mask code, without special cases
            float_len = float_val.shape[0] if float_is_iterable else 1
            shape = (float_len,1,1)
            mask_optional = torch.ones(shape)
            mask_optional = mask_optional[:] * float_val.to(mask_optional.dtype).to(mask_optional.device)
            return (mask_optional,)


class MultivalScaledMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min_float_val": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.001}),
                "max_float_val": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "mask": ("MASK",),
            },
            "optional": {
                "scaling": (ScaleType.LIST,),
            }
        }

    RETURN_TYPES = ("MULTIVAL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/multival"
    FUNCTION = "create_multival"

    def create_multival(self, min_float_val: float, max_float_val: float, mask: Tensor, scaling: str=ScaleType.ABSOLUTE):
        # TODO: allow min_float_val and max_float_val to be list[float]
        if isinstance(min_float_val, Iterable):
            raise ValueError(f"min_float_val must be type float (no lists allowed here), not {type(min_float_val).__name__}.")
        if isinstance(max_float_val, Iterable):
            raise ValueError(f"max_float_val must be type float (no lists allowed here), not {type(max_float_val).__name__}.")
        
        if scaling == ScaleType.ABSOLUTE:
            mask = linear_conversion(mask.clone(), new_min=min_float_val, new_max=max_float_val)
        elif scaling == ScaleType.RELATIVE:
            mask = normalize_min_max(mask.clone(), new_min=min_float_val, new_max=max_float_val)
        else:
            raise ValueError(f"scaling '{scaling}' not recognized.")
        return MultivalDynamicNode.create_multival(self, mask_optional=mask)


class MultivalDynamicFloatInputNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float_val": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "forceInput": True},),
            },
            "optional": {
                "mask_optional": ("MASK",)
            }
        }
    
    RETURN_TYPES = ("MULTIVAL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/multival"
    FUNCTION = "create_multival"

    def create_multival(self, float_val: Union[float, list[float]]=None, mask_optional: Tensor=None):
        return MultivalDynamicNode.create_multival(self, float_val=float_val, mask_optional=mask_optional)


class MultivalFloatNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float_val": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001},),
            },
        }
    
    RETURN_TYPES = ("MULTIVAL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/multival"
    FUNCTION = "create_multival"

    def create_multival(self, float_val: Union[float, list[float]]=None):
        return MultivalDynamicNode.create_multival(self, float_val=float_val)
