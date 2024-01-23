from torch import Tensor

from .utils_motion import normalize_min_max


class AnimateDiffSettings:
    def __init__(self,
                 adjust_pe: 'AdjustPEGroup'=None,
                 pe_strength: float=1.0,
                 attn_strength: float=1.0,
                 attn_q_strength: float=1.0,
                 attn_k_strength: float=1.0,
                 attn_v_strength: float=1.0,
                 attn_out_weight_strength: float=1.0,
                 attn_out_bias_strength: float=1.0,
                 other_strength: float=1.0,
                 attn_scale: float=1.0,
                 mask_attn_scale: Tensor=None,
                 mask_attn_scale_min: float=1.0,
                 mask_attn_scale_max: float=1.0,
                 ):
        # PE-interpolation settings
        self.adjust_pe = adjust_pe if adjust_pe is not None else AdjustPEGroup()
        # general strengths
        self.pe_strength = pe_strength
        self.attn_strength = attn_strength
        self.other_strength = other_strength
        # specific attn strengths
        self.attn_q_strength = attn_q_strength
        self.attn_k_strength = attn_k_strength
        self.attn_v_strength = attn_v_strength
        self.attn_out_weight_strength = attn_out_weight_strength
        self.attn_out_bias_strength = attn_out_bias_strength
        # attention scale settings - DEPRECATED
        self.attn_scale = attn_scale
        # attention scale mask settings - DEPRECATED
        self.mask_attn_scale = mask_attn_scale.clone() if mask_attn_scale is not None else mask_attn_scale
        self.mask_attn_scale_min = mask_attn_scale_min
        self.mask_attn_scale_max = mask_attn_scale_max
        self._prepare_mask_attn_scale()
    
    def _prepare_mask_attn_scale(self):
        if self.mask_attn_scale is not None:
            self.mask_attn_scale = normalize_min_max(self.mask_attn_scale, self.mask_attn_scale_min, self.mask_attn_scale_max)

    def has_mask_attn_scale(self) -> bool:
        return self.mask_attn_scale is not None

    def has_pe_strength(self) -> bool:
        return self.pe_strength != 1.0
    
    def has_attn_strength(self) -> bool:
        return self.attn_strength != 1.0
    
    def has_other_strength(self) -> bool:
        return self.other_strength != 1.0

    def has_anything_to_apply(self) -> bool:
        return self.adjust_pe.has_anything_to_apply() \
            or self.has_pe_strength() \
            or self.has_attn_strength() \
            or self.has_other_strength() \
            or self.has_any_attn_sub_strength()

    def has_any_attn_sub_strength(self) -> bool:
        return self.has_attn_q_strength() \
            or self.has_attn_k_strength() \
            or self.has_attn_v_strength() \
            or self.has_attn_out_weight_strength() \
            or self.has_attn_out_bias_strength()

    def has_attn_q_strength(self) -> bool:
        return self.attn_q_strength != 1.0

    def has_attn_k_strength(self) -> bool:
        return self.attn_k_strength != 1.0

    def has_attn_v_strength(self) -> bool:
        return self.attn_v_strength != 1.0

    def has_attn_out_weight_strength(self) -> bool:
        return self.attn_out_weight_strength != 1.0

    def has_attn_out_bias_strength(self) -> bool:
        return self.attn_out_bias_strength != 1.0


class AdjustPE:
    def __init__(self,
                 cap_initial_pe_length: int=0, interpolate_pe_to_length: int=0,
                 initial_pe_idx_offset: int=0, final_pe_idx_offset: int=0,
                 motion_pe_stretch: int=0, print_adjustment=False):
        # PE-interpolation settings
        self.cap_initial_pe_length = cap_initial_pe_length
        self.interpolate_pe_to_length = interpolate_pe_to_length
        self.initial_pe_idx_offset = initial_pe_idx_offset
        self.final_pe_idx_offset = final_pe_idx_offset
        self.motion_pe_stretch = motion_pe_stretch
        self.print_adjustment = print_adjustment

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


class AdjustPEGroup:
    def __init__(self, initial: AdjustPE=None):
        self.adjusts: list[AdjustPE] = []
        if initial is not None:
            self.add(initial)

    def add(self, adjust_pe: AdjustPE):
        self.adjusts.append(adjust_pe)
    
    def has_anything_to_apply(self):
        for adjust in self.adjusts:
            if adjust.has_anything_to_apply():
                return True
        return False

    def clone(self):
        new_group = AdjustPEGroup()
        for adjust in self.adjusts:
            new_group.add(adjust)
        return new_group
