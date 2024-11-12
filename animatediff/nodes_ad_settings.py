from .ad_settings import AdjustPE, AdjustWeight, AdjustGroup, AnimateDiffSettings
from .utils_model import BIGMAX


class AnimateDiffSettingsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "pe_adjust": ("PE_ADJUST",),
                "weight_adjust": ("WEIGHT_ADJUST",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("AD_SETTINGS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/ad settings"
    FUNCTION = "get_ad_settings"

    def get_ad_settings(self, pe_adjust: AdjustGroup=None, weight_adjust: AdjustGroup=None):
        return (AnimateDiffSettings(adjust_pe=pe_adjust, adjust_weight=weight_adjust),)


class ManualAdjustPENode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cap_initial_pe_length": ("INT", {"default": 0, "min": 0, "step": 1}),
                "interpolate_pe_to_length": ("INT", {"default": 0, "min": 0, "step": 1}),
                "initial_pe_idx_offset": ("INT", {"default": 0, "min": 0, "step": 1}),
                "final_pe_idx_offset": ("INT", {"default": 0, "min": 0, "step": 1}),
                "print_adjustment": ("BOOLEAN", {"default": False}),
                
            },
            "optional": {
                "prev_pe_adjust": ("PE_ADJUST",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("PE_ADJUST",)
    CATEGORY = "Animate Diff 🎭🅐🅓/ad settings/pe adjust"
    FUNCTION = "get_pe_adjust"

    def get_pe_adjust(self, cap_initial_pe_length: int, interpolate_pe_to_length: int, 
                      initial_pe_idx_offset: int, final_pe_idx_offset: int, print_adjustment: bool,
                      prev_pe_adjust: AdjustGroup=None):
        if prev_pe_adjust is None:
            prev_pe_adjust = AdjustGroup()
        prev_pe_adjust = prev_pe_adjust.clone()
        adjust = AdjustPE(cap_initial_pe_length=cap_initial_pe_length, interpolate_pe_to_length=interpolate_pe_to_length,
                          initial_pe_idx_offset=initial_pe_idx_offset, final_pe_idx_offset=final_pe_idx_offset,
                          print_adjustment=print_adjustment)
        prev_pe_adjust.add(adjust)
        return (prev_pe_adjust,)


class SweetspotStretchPENode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sweetspot": ("INT", {"default": 16, "min": 0, "max": BIGMAX},),
                "new_sweetspot": ("INT", {"default": 16, "min": 0, "max": BIGMAX},),
                "print_adjustment": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_pe_adjust": ("PE_ADJUST",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("PE_ADJUST",)
    CATEGORY = "Animate Diff 🎭🅐🅓/ad settings/pe adjust"
    FUNCTION = "get_pe_adjust"

    def get_pe_adjust(self, sweetspot: int, new_sweetspot: int, print_adjustment: bool, prev_pe_adjust: AdjustGroup=None):
        if prev_pe_adjust is None:
            prev_pe_adjust = AdjustGroup()
        prev_pe_adjust = prev_pe_adjust.clone()
        adjust = AdjustPE(cap_initial_pe_length=sweetspot, interpolate_pe_to_length=new_sweetspot,
                          print_adjustment=print_adjustment)
        prev_pe_adjust.add(adjust)
        return (prev_pe_adjust,)


class FullStretchPENode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pe_stretch": ("INT", {"default": 0, "min": 0, "max": BIGMAX},),
                "print_adjustment": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_pe_adjust": ("PE_ADJUST",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("PE_ADJUST",)
    CATEGORY = "Animate Diff 🎭🅐🅓/ad settings/pe adjust"
    FUNCTION = "get_pe_adjust"

    def get_pe_adjust(self, pe_stretch: int, print_adjustment: bool, prev_pe_adjust: AdjustGroup=None):
        if prev_pe_adjust is None:
            prev_pe_adjust = AdjustGroup()
        prev_pe_adjust = prev_pe_adjust.clone()
        adjust = AdjustPE(motion_pe_stretch=pe_stretch,
                          print_adjustment=print_adjustment)
        prev_pe_adjust.add(adjust)
        return (prev_pe_adjust,)


class WeightAdjustAllAddNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "all_ADD": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.000001}),
                "print_adjustment": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_weight_adjust": ("WEIGHT_ADJUST",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("WEIGHT_ADJUST",)
    CATEGORY = "Animate Diff 🎭🅐🅓/ad settings/weight adjust"
    FUNCTION = "get_weight_adjust"

    def get_weight_adjust(self, all_ADD: float, print_adjustment: bool, prev_weight_adjust: AdjustGroup=None):
        if prev_weight_adjust is None:
            prev_weight_adjust = AdjustGroup()
        prev_weight_adjust = prev_weight_adjust.clone()
        adjust = AdjustWeight(
            all_ADD=all_ADD,
            print_adjustment=print_adjustment
        )
        prev_weight_adjust.add(adjust)
        return (prev_weight_adjust,)


class WeightAdjustAllMultNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "all_MULT": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.000001}),
                "print_adjustment": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_weight_adjust": ("WEIGHT_ADJUST",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("WEIGHT_ADJUST",)
    CATEGORY = "Animate Diff 🎭🅐🅓/ad settings/weight adjust"
    FUNCTION = "get_weight_adjust"

    def get_weight_adjust(self, all_MULT: float, print_adjustment: bool, prev_weight_adjust: AdjustGroup=None):
        if prev_weight_adjust is None:
            prev_weight_adjust = AdjustGroup()
        prev_weight_adjust = prev_weight_adjust.clone()
        adjust = AdjustWeight(
            all_MULT=all_MULT,
            print_adjustment=print_adjustment
        )
        prev_weight_adjust.add(adjust)
        return (prev_weight_adjust,)


class WeightAdjustIndivAddNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pe_ADD": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.000001}),
                "attn_ADD": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.000001}),
                "other_ADD": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.000001}),
                "print_adjustment": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_weight_adjust": ("WEIGHT_ADJUST",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("WEIGHT_ADJUST",)
    CATEGORY = "Animate Diff 🎭🅐🅓/ad settings/weight adjust"
    FUNCTION = "get_weight_adjust"

    def get_weight_adjust(self, pe_ADD: float, attn_ADD: float, other_ADD: float, print_adjustment: bool, prev_weight_adjust: AdjustGroup=None):
        if prev_weight_adjust is None:
            prev_weight_adjust = AdjustGroup()
        prev_weight_adjust = prev_weight_adjust.clone()
        adjust = AdjustWeight(
            pe_ADD=pe_ADD,
            attn_ADD=attn_ADD,
            other_ADD=other_ADD,
            print_adjustment=print_adjustment
        )
        prev_weight_adjust.add(adjust)
        return (prev_weight_adjust,)


class WeightAdjustIndivMultNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pe_MULT": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.000001}),
                "attn_MULT": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.000001}),
                "other_MULT": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.000001}),
                "print_adjustment": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_weight_adjust": ("WEIGHT_ADJUST",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("WEIGHT_ADJUST",)
    CATEGORY = "Animate Diff 🎭🅐🅓/ad settings/weight adjust"
    FUNCTION = "get_weight_adjust"

    def get_weight_adjust(self, pe_MULT: float, attn_MULT: float, other_MULT: float, print_adjustment: bool, prev_weight_adjust: AdjustGroup=None):
        if prev_weight_adjust is None:
            prev_weight_adjust = AdjustGroup()
        prev_weight_adjust = prev_weight_adjust.clone()
        adjust = AdjustWeight(
            pe_MULT=pe_MULT,
            attn_MULT=attn_MULT,
            other_MULT=other_MULT,
            print_adjustment=print_adjustment
        )
        prev_weight_adjust.add(adjust)
        return (prev_weight_adjust,)


class WeightAdjustIndivAttnAddNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pe_ADD": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.000001}),
                "attn_ADD": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.000001}),
                "attn_q_ADD": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.000001}),
                "attn_k_ADD": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.000001}),
                "attn_v_ADD": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.000001}),
                "attn_out_weight_ADD": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.000001}),
                "attn_out_bias_ADD": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.000001}),
                "other_ADD": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.000001}),
                "print_adjustment": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_weight_adjust": ("WEIGHT_ADJUST",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("WEIGHT_ADJUST",)
    CATEGORY = "Animate Diff 🎭🅐🅓/ad settings/weight adjust"
    FUNCTION = "get_weight_adjust"

    def get_weight_adjust(self, pe_ADD: float, attn_ADD: float,
                          attn_q_ADD: float, attn_k_ADD: float, attn_v_ADD: float,
                          attn_out_weight_ADD: float, attn_out_bias_ADD: float,
                          other_ADD: float, print_adjustment: bool, prev_weight_adjust: AdjustGroup=None):
        if prev_weight_adjust is None:
            prev_weight_adjust = AdjustGroup()
        prev_weight_adjust = prev_weight_adjust.clone()
        adjust = AdjustWeight(
            pe_ADD=pe_ADD,
            attn_ADD=attn_ADD,
            attn_q_ADD=attn_q_ADD,
            attn_k_ADD=attn_k_ADD,
            attn_v_ADD=attn_v_ADD,
            attn_out_weight_ADD=attn_out_weight_ADD,
            attn_out_bias_ADD=attn_out_bias_ADD,
            other_ADD=other_ADD,
            print_adjustment=print_adjustment
        )
        prev_weight_adjust.add(adjust)
        return (prev_weight_adjust,)


class WeightAdjustIndivAttnMultNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pe_MULT": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.000001}),
                "attn_MULT": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.000001}),
                "attn_q_MULT": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.000001}),
                "attn_k_MULT": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.000001}),
                "attn_v_MULT": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.000001}),
                "attn_out_weight_MULT": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.000001}),
                "attn_out_bias_MULT": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.000001}),
                "other_MULT": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.000001}),
                "print_adjustment": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_weight_adjust": ("WEIGHT_ADJUST",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("WEIGHT_ADJUST",)
    CATEGORY = "Animate Diff 🎭🅐🅓/ad settings/weight adjust"
    FUNCTION = "get_weight_adjust"

    def get_weight_adjust(self, pe_MULT: float, attn_MULT: float,
                          attn_q_MULT: float, attn_k_MULT: float, attn_v_MULT: float,
                          attn_out_weight_MULT: float, attn_out_bias_MULT: float,
                          other_MULT: float, print_adjustment: bool, prev_weight_adjust: AdjustGroup=None):
        if prev_weight_adjust is None:
            prev_weight_adjust = AdjustGroup()
        prev_weight_adjust = prev_weight_adjust.clone()
        adjust = AdjustWeight(
            pe_MULT=pe_MULT,
            attn_MULT=attn_MULT,
            attn_q_MULT=attn_q_MULT,
            attn_k_MULT=attn_k_MULT,
            attn_v_MULT=attn_v_MULT,
            attn_out_weight_MULT=attn_out_weight_MULT,
            attn_out_bias_MULT=attn_out_bias_MULT,
            other_MULT=other_MULT,
            print_adjustment=print_adjustment
        )
        prev_weight_adjust.add(adjust)
        return (prev_weight_adjust,)
