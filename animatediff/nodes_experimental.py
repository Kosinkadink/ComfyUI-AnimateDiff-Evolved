
import torch

from .motion_module import MotionModelSettings


class AnimateDiffModelSettingsSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_pe_stretch": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
                "min_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "max_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
            }
        }
    
    RETURN_TYPES = ("MOTION_MODEL_SETTINGS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/motion settings/experimental"
    FUNCTION = "get_motion_model_settings"

    def get_motion_model_settings(self, motion_pe_stretch: int, mask: torch.Tensor=None, min_scale: float=1.0, max_scale: float=1.0):
        motion_model_settings = MotionModelSettings(
            motion_pe_stretch=motion_pe_stretch
            )

        return (motion_model_settings,)


class AnimateDiffModelSettingsAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pe_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "other_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "motion_pe_stretch": ("INT", {"default": 0, "min": 0, "step": 1}),
                "cap_initial_pe_length": ("INT", {"default": 0, "min": 0, "step": 1}),
                "interpolate_pe_to_length": ("INT", {"default": 0, "min": 0, "step": 1}),
                "initial_pe_idx_offset": ("INT", {"default": 0, "min": 0, "step": 1}),
                "final_pe_idx_offset": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("MOTION_MODEL_SETTINGS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/motion settings/experimental"
    FUNCTION = "get_motion_model_settings"

    def get_motion_model_settings(self, pe_strength: float, attn_strength: float, other_strength: float,
                                  motion_pe_stretch: int,
                                  cap_initial_pe_length: int, interpolate_pe_to_length: int,
                                  initial_pe_idx_offset: int, final_pe_idx_offset: int):
        motion_model_settings = MotionModelSettings(
            pe_strength=pe_strength,
            attn_strength=attn_strength,
            other_strength=other_strength,
            cap_initial_pe_length=cap_initial_pe_length,
            interpolate_pe_to_length=interpolate_pe_to_length,
            initial_pe_idx_offset=initial_pe_idx_offset,
            final_pe_idx_offset=final_pe_idx_offset,
            motion_pe_stretch=motion_pe_stretch
            )

        return (motion_model_settings,)


class AnimateDiffModelSettingsAdvancedAttnStrengths:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pe_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_q_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_k_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_v_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_out_weight_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_out_bias_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "other_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "motion_pe_stretch": ("INT", {"default": 0, "min": 0, "step": 1}),
                "cap_initial_pe_length": ("INT", {"default": 0, "min": 0, "step": 1}),
                "interpolate_pe_to_length": ("INT", {"default": 0, "min": 0, "step": 1}),
                "initial_pe_idx_offset": ("INT", {"default": 0, "min": 0, "step": 1}),
                "final_pe_idx_offset": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("MOTION_MODEL_SETTINGS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/motion settings/experimental"
    FUNCTION = "get_motion_model_settings"

    def get_motion_model_settings(self, pe_strength: float, attn_strength: float,
                                  attn_q_strength: float,
                                  attn_k_strength: float,
                                  attn_v_strength: float,
                                  attn_out_weight_strength: float,
                                  attn_out_bias_strength: float,
                                  other_strength: float,
                                  motion_pe_stretch: int,
                                  cap_initial_pe_length: int, interpolate_pe_to_length: int,
                                  initial_pe_idx_offset: int, final_pe_idx_offset: int):
        motion_model_settings = MotionModelSettings(
            pe_strength=pe_strength,
            attn_strength=attn_strength,
            attn_q_strength=attn_q_strength,
            attn_k_strength=attn_k_strength,
            attn_v_strength=attn_v_strength,
            attn_out_weight_strength=attn_out_weight_strength,
            attn_out_bias_strength=attn_out_bias_strength,
            other_strength=other_strength,
            cap_initial_pe_length=cap_initial_pe_length,
            interpolate_pe_to_length=interpolate_pe_to_length,
            initial_pe_idx_offset=initial_pe_idx_offset,
            final_pe_idx_offset=final_pe_idx_offset,
            motion_pe_stretch=motion_pe_stretch
            )

        return (motion_model_settings,)
