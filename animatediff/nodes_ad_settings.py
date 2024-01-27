from .ad_settings import AdjustPE, AdjustPEGroup, AnimateDiffSettings
from .utils_model import BIGMAX


class AnimateDiffSettingsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "pe_adjust": ("PE_ADJUST",),
            }
        }
    
    RETURN_TYPES = ("AD_SETTINGS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/ad settings"
    FUNCTION = "get_ad_settings"

    def get_ad_settings(self, pe_adjust: AdjustPEGroup=None):
        return (AnimateDiffSettings(adjust_pe=pe_adjust),)


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
            }
        }
    
    RETURN_TYPES = ("PE_ADJUST",)
    CATEGORY = "Animate Diff 🎭🅐🅓/ad settings/pe adjust"
    FUNCTION = "get_pe_adjust"

    def get_pe_adjust(self, cap_initial_pe_length: int, interpolate_pe_to_length: int, 
                      initial_pe_idx_offset: int, final_pe_idx_offset: int, print_adjustment: bool,
                      prev_pe_adjust: AdjustPEGroup=None):
        if prev_pe_adjust is None:
            prev_pe_adjust = AdjustPEGroup()
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
            }
        }
    
    RETURN_TYPES = ("PE_ADJUST",)
    CATEGORY = "Animate Diff 🎭🅐🅓/ad settings/pe adjust"
    FUNCTION = "get_pe_adjust"

    def get_pe_adjust(self, sweetspot: int, new_sweetspot: int, print_adjustment: bool, prev_pe_adjust: AdjustPEGroup=None):
        if prev_pe_adjust is None:
            prev_pe_adjust = AdjustPEGroup()
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
            }
        }
    
    RETURN_TYPES = ("PE_ADJUST",)
    CATEGORY = "Animate Diff 🎭🅐🅓/ad settings/pe adjust"
    FUNCTION = "get_pe_adjust"

    def get_pe_adjust(self, pe_stretch: int, print_adjustment: bool, prev_pe_adjust: AdjustPEGroup=None):
        if prev_pe_adjust is None:
            prev_pe_adjust = AdjustPEGroup()
        prev_pe_adjust = prev_pe_adjust.clone()
        adjust = AdjustPE(motion_pe_stretch=pe_stretch,
                          print_adjustment=print_adjustment)
        prev_pe_adjust.add(adjust)
        return (prev_pe_adjust,)
