from collections.abc import Iterable
from typing import Union

import torch
from torch import Tensor

from .utils_motion import create_multival_combo, linear_conversion, normalize_min_max, extend_to_batch_size, extend_list_to_batch_size


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
                "mask_optional": ("MASK",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("MULTIVAL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/multival"
    FUNCTION = "create_multival"

    def create_multival(self, float_val: Union[float, list[float]]=1.0, mask_optional: Tensor=None):
        return (create_multival_combo(float_val=float_val, mask_optional=mask_optional),)


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
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("MULTIVAL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/multival"
    FUNCTION = "create_multival"

    def create_multival(self, min_float_val: float, max_float_val: float, mask: Tensor, scaling: str=ScaleType.ABSOLUTE):
        lengths = [mask.shape[0]]
        iterable_inputs = [False, False]
        val_inputs = [min_float_val, max_float_val]
        if isinstance(min_float_val, Iterable):
            iterable_inputs[0] = True
            val_inputs[0] = list(min_float_val)
            lengths.append(len(min_float_val))
        if isinstance(max_float_val, Iterable):
            iterable_inputs[1] = True
            val_inputs[1] = list(max_float_val)
            lengths.append(len(max_float_val))
        # make sure mask and any iterable float_vals match max length
        max_length = max(lengths)
        mask = extend_to_batch_size(mask, max_length)
        for i in range(len(iterable_inputs)):
            if iterable_inputs[i] == True:
                # make sure tensors will match dimensions of mask
                val_inputs[i] = torch.tensor(extend_list_to_batch_size(val_inputs[i], max_length)).unsqueeze(-1).unsqueeze(-1)
        min_float_val, max_float_val = val_inputs
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
                "mask_optional": ("MASK",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("MULTIVAL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/multival"
    FUNCTION = "create_multival"

    def create_multival(self, float_val: Union[float, list[float]]=None, mask_optional: Tensor=None):
        return MultivalDynamicNode.create_multival(self, float_val=float_val, mask_optional=mask_optional)


class MultivalDynamicFloatsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "floats": ("FLOATS", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001},),
            },
            "optional": {
                "mask_optional": ("MASK",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("MULTIVAL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/multival"
    FUNCTION = "create_multival"

    def create_multival(self, floats: Union[float, list[float]]=None, mask_optional: Tensor=None):
        return MultivalDynamicNode.create_multival(self, float_val=floats, mask_optional=mask_optional)


class MultivalFloatNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float_val": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001},),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("MULTIVAL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/multival"
    FUNCTION = "create_multival"

    def create_multival(self, float_val: Union[float, list[float]]=None):
        return MultivalDynamicNode.create_multival(self, float_val=float_val)


class MultivalConvertToMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "multival": ("MULTIVAL",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    CATEGORY = "Animate Diff 🎭🅐🅓/multival"
    FUNCTION = "convert_multival_to_mask"

    def convert_multival_to_mask(self, multival: Union[float, Tensor]):
        # if already tensor, assume is a valid mask
        if type(multival) == Tensor:
            return (multival,)
        # otherwise, make a single 1x1 mask with the proper value
        shape = (1,1,1)
        converted_multival = torch.ones(shape) * multival
        return (converted_multival,)
