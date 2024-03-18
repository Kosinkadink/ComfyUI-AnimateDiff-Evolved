from abc import ABC, abstractmethod
from typing import Union
from torch import Tensor
import math

from .utils_motion import normalize_min_max
from .logger import logger


class AnimateDiffSettings:
    def __init__(self,
                 adjust_pe: 'AdjustGroup'=None,
                 adjust_weight: 'AdjustGroup'=None,
                 attn_scale: float=1.0,
                 mask_attn_scale: Tensor=None,
                 mask_attn_scale_min: float=1.0,
                 mask_attn_scale_max: float=1.0,
                 ):
        # PE-interpolation settings
        self.adjust_pe = adjust_pe if adjust_pe is not None else AdjustGroup()
        # Weight settings
        self.adjust_weight = adjust_weight if adjust_weight is not None else AdjustGroup()
        # attention scale settings - DEPRECATED (part of scale_multival now)
        self.attn_scale = attn_scale
        # attention scale mask settings - DEPRECATED (part of scale_multival now)
        self.mask_attn_scale = mask_attn_scale.clone() if mask_attn_scale is not None else mask_attn_scale
        self.mask_attn_scale_min = mask_attn_scale_min
        self.mask_attn_scale_max = mask_attn_scale_max
        self._prepare_mask_attn_scale()
    
    def _prepare_mask_attn_scale(self):
        if self.mask_attn_scale is not None:
            self.mask_attn_scale = normalize_min_max(self.mask_attn_scale, self.mask_attn_scale_min, self.mask_attn_scale_max)

    def has_mask_attn_scale(self) -> bool:
        return self.mask_attn_scale is not None

    def has_anything_to_apply(self) -> bool:
        return self.adjust_pe.has_anything_to_apply() \
            or self.adjust_weight.has_anything_to_apply()


class AdjustAbstract(ABC):
    def __init__(self, print_adjustment=False):
        self.print_adjustment = print_adjustment
    
    @abstractmethod
    def has_anything_to_apply(self):
        return False


class AdjustPE(AdjustAbstract):
    def __init__(self,
                 cap_initial_pe_length: int=0, interpolate_pe_to_length: int=0,
                 initial_pe_idx_offset: int=0, final_pe_idx_offset: int=0,
                 motion_pe_stretch: int=0, print_adjustment=False):
        super().__init__(print_adjustment=print_adjustment)
        # PE-interpolation settings
        self.cap_initial_pe_length = cap_initial_pe_length
        self.interpolate_pe_to_length = interpolate_pe_to_length
        self.initial_pe_idx_offset = initial_pe_idx_offset
        self.final_pe_idx_offset = final_pe_idx_offset
        self.motion_pe_stretch = motion_pe_stretch

    def has_cap_initial_pe_length(self) -> bool:
        return self.cap_initial_pe_length > 0
    
    def has_interpolate_pe_to_length(self) -> bool:
        return self.interpolate_pe_to_length > 0
    
    def has_initial_pe_idx_offset(self) -> bool:
        return self.initial_pe_idx_offset > 0
    
    def has_final_pe_idx_offset(self) -> bool:
        return self.final_pe_idx_offset > 0

    def has_motion_pe_stretch(self) -> bool:
        return self.motion_pe_stretch > 0
    
    def has_anything_to_apply(self) -> bool:
        return self.has_cap_initial_pe_length() \
            or self.has_interpolate_pe_to_length() \
            or self.has_initial_pe_idx_offset() \
            or self.has_final_pe_idx_offset() \
            or self.has_motion_pe_stretch()


class AdjustWeight(AdjustAbstract):
    # possible operations
    OP_ANY = "_____ANY"
    OP_ADD = "_ADD"
    OP_MULT = "_MULT"
    OPS = [OP_ADD, OP_MULT]
    # possible attributes
    ATTR_ALL = "all"
    ATTR_PE = "pe"
    ATTR_ATTN = "attn"
    ATTR_ATTN_Q = "attn_q"
    ATTR_ATTN_K = "attn_k"
    ATTR_ATTN_V = "attn_v"
    ATTR_ATTN_OUT_WEIGHT = "attn_out_weight"
    ATTR_ATTN_OUT_BIAS = "attn_out_bias"
    ATTR_OTHER = "other"
    ATTRS = [ATTR_ALL, ATTR_PE, ATTR_ATTN, ATTR_ATTN_Q, ATTR_ATTN_K, ATTR_ATTN_V, ATTR_ATTN_OUT_WEIGHT, ATTR_ATTN_OUT_BIAS, ATTR_OTHER]
    
    def __init__(self,
                 all_ADD=0.0, all_MULT=1.0,
                 pe_ADD=0.0, pe_MULT=1.0,
                 attn_ADD=0.0, attn_MULT=1.0,
                 attn_q_ADD=0.0, attn_q_MULT=1.0,
                 attn_k_ADD=0.0, attn_k_MULT=1.0,
                 attn_v_ADD=0.0, attn_v_MULT=1.0,
                 attn_out_weight_ADD=0.0, attn_out_weight_MULT=1.0,
                 attn_out_bias_ADD=0.0, attn_out_bias_MULT=1.0,
                 other_ADD=0.0, other_MULT=1.0,
                 print_adjustment=False):
        # all
        self.all_ADD = all_ADD
        self.all_MULT = all_MULT
        # pe
        self.pe_ADD = pe_ADD
        self.pe_MULT = pe_MULT
        # attn
        self.attn_ADD = attn_ADD
        self.attn_MULT = attn_MULT
        # attn_q
        self.attn_q_ADD = attn_q_ADD
        self.attn_q_MULT = attn_q_MULT
        # attn_k
        self.attn_k_ADD = attn_k_ADD
        self.attn_k_MULT = attn_k_MULT
        # attn_v
        self.attn_v_ADD = attn_v_ADD
        self.attn_v_MULT = attn_v_MULT
        # attn_out_weight
        self.attn_out_weight_ADD = attn_out_weight_ADD
        self.attn_out_weight_MULT = attn_out_weight_MULT
        # attn_out_bias
        self.attn_out_bias_ADD = attn_out_bias_ADD
        self.attn_out_bias_MULT = attn_out_bias_MULT
        # other
        self.other_ADD = other_ADD
        self.other_MULT = other_MULT
        # additional vars
        self.print_adjustment = print_adjustment
        # temp var
        self.already_printed: dict[str, bool] = {}
        self.mark_attrs_as_unprinted()

    def mark_attrs_as_unprinted(self):
        for attr in self.ATTRS:
            for op in self.OPS:
                self.already_printed[attr+op] = False
    
    def mask_as_printed(self, attr: str, op: str):
        self.already_printed[attr+op] = True
    
    def is_already_printed(self, attr: str, op: str):
        return self.already_printed.get(attr+op, False)

    def _get_val(self, op: str, attr: str) -> float:
        try:
            return getattr(self, attr+op)
        except AttributeError:
            raise Exception(f"Parameter '{attr+op}' could not be found in AdjustWeight class.")

    def _has_OP(self, op: str, attr: str):
        value = self._get_val(op=op, attr=attr)
        if op == self.OP_ADD:
            return not math.isclose(value, 0.0)
        elif op == self.OP_MULT:
            return not math.isclose(value, 1.0)
        else:
            raise Exception(f"Operation '{op}' not recognized in AdjustWeight.")

    def _has_apply(self, op: str, attr: str):
        # determine if attr with specific operation is to be applied
        if op == self.OP_ANY:
            any = False
            for one_op in self.OPS:
                any = any or self._has_OP(op=one_op, attr=attr)
            return any
        return self._has_OP(op=op, attr=attr)

    def has_all(self, op: str) -> bool:
        return self._has_apply(op, self.ATTR_ALL)
    
    def has_pe(self, op: str) -> bool:
        return self._has_apply(op, self.ATTR_PE)
    
    def has_attn(self, op: str) -> bool:
        return self._has_apply(op, self.ATTR_ATTN)
    
    def has_attn_q(self, op: str) -> bool:
        return self._has_apply(op, self.ATTR_ATTN_Q)
    
    def has_attn_k(self, op: str) -> bool:
        return self._has_apply(op, self.ATTR_ATTN_K)
    
    def has_attn_v(self, op: str) -> bool:
        return self._has_apply(op, self.ATTR_ATTN_V)
    
    def has_attn_out_weight(self, op: str) -> bool:
        return self._has_apply(op, self.ATTR_ATTN_OUT_WEIGHT)
    
    def has_attn_out_bias(self, op: str) -> bool:
        return self._has_apply(op, self.ATTR_ATTN_OUT_BIAS)
    
    def has_other(self, op: str) -> bool:
        return self._has_apply(op, self.ATTR_OTHER)

    def has_anything_to_apply(self):
        return self.has_all(self.OP_ANY) \
            or self.has_pe(self.OP_ANY) \
            or self.has_attn(self.OP_ANY) \
            or self.has_attn_q(self.OP_ANY) \
            or self.has_attn_k(self.OP_ANY) \
            or self.has_attn_v(self.OP_ANY) \
            or self.has_attn_out_weight(self.OP_ANY) \
            or self.has_attn_out_bias(self.OP_ANY) \
            or self.has_other(self.OP_ANY)
    
    def _perform_op(self, model_dict: dict[str, Tensor], key: str, op: str, attr: str):
        val = self._get_val(op=op, attr=attr)
        specific_str = f"'{attr}' weights" if attr == self.ATTR_ALL else f"every '{attr}' weight"
        if op == self.OP_ADD:
            model_dict[key] += val
            if self.print_adjustment and not self.is_already_printed(attr=attr, op=op):
                logger.info(f"[Adjust Weight]: Adding to {specific_str} value {val}")
                self.mask_as_printed(attr=attr, op=op)
        elif op == self.OP_MULT:
            model_dict[key] *= val
            if self.print_adjustment and not self.is_already_printed(attr=attr, op=op):
                logger.info(f"[Adjust Weight]: Multiplying {specific_str} by {val}")
                self.mask_as_printed(attr=attr, op=op)
        else:
            raise Exception(f"Operation '{op}' not recognized in AdjustWeight.")

    def perform_applicable_ops(self, attr: str, model_dict: dict[str, Tensor], key: str):
        for op in self.OPS:
            if self._has_apply(op=op, attr=attr):
                self._perform_op(model_dict=model_dict, key=key, op=op, attr=attr)


ADJUST_TYPES = Union[AdjustPE, AdjustWeight]
class AdjustGroup:
    def __init__(self, initial: ADJUST_TYPES=None):
        self.adjusts: list[ADJUST_TYPES] = []
        if initial is not None:
            self.add(initial)
        
    def add(self, adjust: ADJUST_TYPES):
        self.adjusts.append(adjust)
    
    def has_anything_to_apply(self):
        for adjust in self.adjusts:
            if adjust.has_anything_to_apply():
                return True
        return False

    def clone(self):
        new_group = AdjustGroup()
        for adjust in self.adjusts:
            new_group.add(adjust=adjust)
        return new_group
