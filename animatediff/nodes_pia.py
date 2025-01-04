from typing import Union
import torch
from torch import Tensor
import math

from comfy.sd import VAE

from .ad_settings import AnimateDiffSettings
from .logger import logger
from .utils_model import BIGMIN, BIGMAX, get_available_motion_models
from .utils_motion import ADKeyframeGroup, InputPIA, InputPIA_Multival, extend_list_to_batch_size, extend_to_batch_size, prepare_mask_batch
from .motion_lora import MotionLoraList
from .model_injection import MotionModelGroup, MotionModelPatcher, get_mm_attachment, load_motion_module_gen2, inject_pia_conv_in_into_model
from .motion_module_ad import AnimateDiffFormat
from .nodes_gen2 import ApplyAnimateDiffModelNode, ADKeyframeNode


# Preset values ported over from PIA repository:
# https://github.com/open-mmlab/PIA/blob/main/animatediff/utils/util.py
class PIA_RANGES:
    ANIMATION_SMALL = "Animation (Small Motion)"
    ANIMATION_MEDIUM = "Animation (Medium Motion)"
    ANIMATION_LARGE = "Animation (Large Motion)"
    LOOP_SMALL = "Loop (Small Motion)"
    LOOP_MEDIUM = "Loop (Medium Motion)"
    LOOP_LARGE = "Loop (Large Motion)"
    STYLE_TRANSFER_SMALL = "Style Transfer (Small Motion)"
    STYLE_TRANSFER_MEDIUM = "Style Transfer (Medium Motion)"
    STYLE_TRANSFER_LARGE = "Style Transfer (Large Motion)"

    _LOOPED = [LOOP_SMALL, LOOP_MEDIUM, LOOP_LARGE]
    _LIST_ALL = [ANIMATION_SMALL, ANIMATION_MEDIUM, ANIMATION_LARGE,
                 LOOP_SMALL, LOOP_MEDIUM, LOOP_LARGE,
                 STYLE_TRANSFER_SMALL, STYLE_TRANSFER_MEDIUM, STYLE_TRANSFER_LARGE]

    _MAPPING = {
        ANIMATION_SMALL: [1.0, 0.9, 0.85, 0.85, 0.85, 0.8],
        ANIMATION_MEDIUM: [1.0, 0.8, 0.8, 0.8, 0.79, 0.78, 0.75],
        ANIMATION_LARGE: [1.0, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5, 0.5],
        LOOP_SMALL: [1.0, 0.9, 0.85, 0.85, 0.85, 0.8],
        LOOP_MEDIUM: [1.0, 0.8, 0.8, 0.8, 0.79, 0.78, 0.75],
        LOOP_LARGE: [1.0, 0.8, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5],
        STYLE_TRANSFER_SMALL: [0.5, 0.4, 0.4, 0.4, 0.35, 0.3],
        STYLE_TRANSFER_MEDIUM: [0.5, 0.4, 0.4, 0.4, 0.35, 0.35, 0.3, 0.25, 0.2],
        STYLE_TRANSFER_LARGE: [0.5, 0.2],
    }

    @classmethod
    def get_preset(cls, preset: str) -> list[float]:
        if preset in cls._MAPPING:
            return cls._MAPPING[preset]
        raise Exception(f"PIA Preset '{preset}' is not recognized.")
    
    @classmethod
    def is_looped(cls, preset: str) -> bool:
        return preset in cls._LOOPED


class InputPIA_PaperPresets(InputPIA):
    def __init__(self, preset: str, index: int, mult_multival: Union[float, Tensor]=None, effect_multival: Union[float, Tensor]=None):
        super().__init__(effect_multival=effect_multival)
        self.preset = preset
        self.index = index
        self.mult_multival = mult_multival if mult_multival is not None else 1.0
    
    def get_mask(self, x: Tensor):
        b, c, h, w = x.shape
        values = PIA_RANGES.get_preset(self.preset)
        # if preset is looped, make values loop
        if PIA_RANGES.is_looped(self.preset):
            # even length
            if b % 2 == 0:
                # extend to half length to get half of the loop
                values = extend_list_to_batch_size(values, b // 2)
                # apply second half of loop (just reverse it)
                values += list(reversed(values))
            # odd length
            else:
                inter_values = extend_list_to_batch_size(values, b // 2)
                middle_vals = [values[min(len(inter_values), len(values)-1)]]
                # make middle vals long enough to fill in gaps (or none if not needed)
                middle_vals = middle_vals * (max(0, b-2*len(inter_values)))
                values = inter_values + middle_vals + list(reversed(inter_values))
        # otherwise, just extend values to desired length
        else:
            values = extend_list_to_batch_size(values, b)
        assert len(values) == b

        index = self.index
        # handle negative index
        if index < 0:
            index = b + index
        # constrain index between 0 and b-1
        index = max(0, min(b-1, index))
        # center values around targer index
        order = [abs(i - index) for i in range(b)]
        real_values = [values[order[i]] for i in range(b)]
        # using real values, generate masks
        tensor_values = torch.tensor(real_values).unsqueeze(-1).unsqueeze(-1)
        mask = torch.ones(size=(b, h, w)) * tensor_values
        # apply multi_multival to mask
        if type(self.mult_multival) == Tensor or not math.isclose(self.mult_multival, 1.0):
            real_mult = self.mult_multival
            if type(real_mult) == Tensor:
                real_mult = extend_to_batch_size(prepare_mask_batch(real_mult, x.shape), b).squeeze(1)
            mask = mask * real_mult
        return mask


class ApplyAnimateDiffPIAModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_model": ("MOTION_MODEL_ADE",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "pia_input": ("PIA_INPUT",),
                "motion_lora": ("MOTION_LORA",),
                "scale_multival": ("MULTIVAL",),
                "effect_multival": ("MULTIVAL",),
                "ad_keyframes": ("AD_KEYFRAMES",),
                "prev_m_models": ("M_MODELS",),
                "per_block": ("PER_BLOCK",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("M_MODELS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA"
    FUNCTION = "apply_motion_model"

    def apply_motion_model(self, motion_model: MotionModelPatcher, image: Tensor, vae: VAE,
                           start_percent: float=0.0, end_percent: float=1.0, pia_input: InputPIA=None,
                           motion_lora: MotionLoraList=None, ad_keyframes: ADKeyframeGroup=None,
                           scale_multival=None, effect_multival=None, ref_multival=None, per_block=None,
                           prev_m_models: MotionModelGroup=None,):
        new_m_models = ApplyAnimateDiffModelNode.apply_motion_model(self, motion_model, start_percent=start_percent, end_percent=end_percent,
                                                                    motion_lora=motion_lora, ad_keyframes=ad_keyframes,
                                                                    scale_multival=scale_multival, effect_multival=effect_multival, per_block=per_block,
                                                                    prev_m_models=prev_m_models)
        # most recent added model will always be first in list;
        curr_model = new_m_models[0].models[0]
        # confirm that model is PIA
        if curr_model.model.mm_info.mm_format != AnimateDiffFormat.PIA:
            raise Exception(f"Motion model '{curr_model.model.mm_info.mm_name}' is not a PIA model; cannot be used with Apply AnimateDiff-PIA Model node.")
        attachment = get_mm_attachment(curr_model)
        attachment.orig_pia_images = image
        attachment.pia_vae = vae
        if pia_input is None:
            pia_input = InputPIA_Multival(1.0)
        attachment.pia_input = pia_input
        #curr_model.pia_multival = ref_multival
        return new_m_models


class LoadAnimateDiffAndInjectPIANode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_available_motion_models(),),
                "motion_model": ("MOTION_MODEL_ADE",),
            },
            "optional": {
                "ad_settings": ("AD_SETTINGS",),
                "deprecation_warning": ("ADEWARN", {"text": "Experimental. Don't expect to work.", "warn_type": "experimental", "color": "#CFC"}),
            }
        }
    
    RETURN_TYPES = ("MOTION_MODEL_ADE",)
    RETURN_NAMES = ("MOTION_MODEL",)

    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA/🧪experimental"
    FUNCTION = "load_motion_model"
    
    def load_motion_model(self, model_name: str, motion_model: MotionModelPatcher, ad_settings: AnimateDiffSettings=None):
        # make sure model actually has PIA conv_in
        if motion_model.model.conv_in is None:
            raise Exception("Passed-in motion model was expected to be PIA (contain conv_in), but did not.")
        # load motion module and motion settings, if included
        loaded_motion_model = load_motion_module_gen2(model_name=model_name, motion_model_settings=ad_settings)
        inject_pia_conv_in_into_model(motion_model=loaded_motion_model, w_pia=motion_model)
        return (loaded_motion_model,)


class PIA_ADKeyframeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}, ),
            },
            "optional": {
                "prev_ad_keyframes": ("AD_KEYFRAMES", ),
                "scale_multival": ("MULTIVAL",),
                "effect_multival": ("MULTIVAL",),
                "pia_input": ("PIA_INPUT",),
                "inherit_missing": ("BOOLEAN", {"default": True}, ),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("AD_KEYFRAMES", )
    FUNCTION = "load_keyframe"

    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA"

    def load_keyframe(self,
                      start_percent: float, prev_ad_keyframes=None,
                      scale_multival: Union[float, torch.Tensor]=None, effect_multival: Union[float, torch.Tensor]=None,
                      pia_input: InputPIA=None,
                      inherit_missing: bool=True, guarantee_steps: int=1):
        return ADKeyframeNode.load_keyframe(self,
                    start_percent=start_percent, prev_ad_keyframes=prev_ad_keyframes,
                    scale_multival=scale_multival, effect_multival=effect_multival, pia_input=pia_input,
                    inherit_missing=inherit_missing, guarantee_steps=guarantee_steps
                )


class InputPIA_MultivalNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "multival": ("MULTIVAL",),
            },
            # "optional": {
            #     "effect_multival": ("MULTIVAL",),
            # }
        }
    
    RETURN_TYPES = ("PIA_INPUT",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA"
    FUNCTION = "create_pia_input"

    def create_pia_input(self, multival: Union[float, Tensor], effect_multival: Union[float, Tensor]=None):
        return (InputPIA_Multival(multival, effect_multival),)


class InputPIA_PaperPresetsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (PIA_RANGES._LIST_ALL,),
                "batch_index": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX, "step": 1}),
            },
            "optional": {
                "mult_multival": ("MULTIVAL",),
                "print_values": ("BOOLEAN", {"default": False},),
                #"effect_multival": ("MULTIVAL",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("PIA_INPUT",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA"
    FUNCTION = "create_pia_input"

    def create_pia_input(self, preset: str, batch_index: int, mult_multival: Union[float, Tensor]=None, print_values: bool=False, effect_multival: Union[float, Tensor]=None):
        # verify preset exists - function will throw error if does not
        values = PIA_RANGES.get_preset(preset)
        if print_values:
            logger.info(f"PIA Preset '{preset}': {values}")
        return (InputPIA_PaperPresets(preset=preset, index=batch_index, mult_multival=mult_multival, effect_multival=effect_multival),)
