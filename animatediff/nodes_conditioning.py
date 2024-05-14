import uuid
import folder_paths
from typing import Union
from torch import Tensor
from collections.abc import Iterable

from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP
import comfy.sd
import comfy.utils

from .conditioning import (COND_CONST, TimestepsCond, set_mask_conds, set_mask_and_combine_conds, set_unmasked_and_combine_conds,
                           LoraHook, LoraHookGroup, LoraHookKeyframe, LoraHookKeyframeGroup)
from .model_injection import ModelPatcherAndInjector, CLIPWithHooks, load_hooked_lora_for_models, load_model_as_hooked_lora_for_models
from .utils_model import BIGMAX, InterpolationMethod
from .logger import logger


###############################################
### Mask, Combine, and Hook Conditioning
###############################################
class PairedConditioningSetMaskHooked:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_ADD": ("CONDITIONING", ),
                "negative_ADD": ("CONDITIONING", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (COND_CONST._LIST_COND_AREA,),
            },
            "optional": {
                "opt_mask": ("MASK", ),
                "opt_lora_hook": ("LORA_HOOK",),
                "opt_timesteps": ("TIMESTEPS_COND",)
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning"
    FUNCTION = "append_and_hook"

    def append_and_hook(self, positive_ADD, negative_ADD,
                        strength: float, set_cond_area: str,
                        opt_mask: Tensor=None, opt_lora_hook: LoraHookGroup=None, opt_timesteps: TimestepsCond=None):
        final_positive, final_negative = set_mask_conds(conds=[positive_ADD, negative_ADD],
                                                        strength=strength, set_cond_area=set_cond_area,
                                                        opt_mask=opt_mask, opt_lora_hook=opt_lora_hook, opt_timesteps=opt_timesteps)
        return (final_positive, final_negative)


class ConditioningSetMaskHooked:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond_ADD": ("CONDITIONING",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (COND_CONST._LIST_COND_AREA,),
            },
            "optional": {
                "opt_mask": ("MASK", ),
                "opt_lora_hook": ("LORA_HOOK",),
                "opt_timesteps": ("TIMESTEPS_COND",)
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/single cond ops"
    FUNCTION = "append_and_hook"

    def append_and_hook(self, cond_ADD,
                        strength: float, set_cond_area: str,
                        opt_mask: Tensor=None, opt_lora_hook: LoraHookGroup=None, opt_timesteps: TimestepsCond=None):
        (final_conditioning,) = set_mask_conds(conds=[cond_ADD],
                                               strength=strength, set_cond_area=set_cond_area,
                                               opt_mask=opt_mask, opt_lora_hook=opt_lora_hook, opt_timesteps=opt_timesteps)
        return (final_conditioning,) 


class PairedConditioningSetMaskAndCombineHooked:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "positive_ADD": ("CONDITIONING",),
                "negative_ADD": ("CONDITIONING",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (COND_CONST._LIST_COND_AREA,),
            },
            "optional": {
                "opt_mask": ("MASK", ),
                "opt_lora_hook": ("LORA_HOOK",),
                "opt_timesteps": ("TIMESTEPS_COND",)
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning"
    FUNCTION = "append_and_combine"

    def append_and_combine(self, positive, negative, positive_ADD, negative_ADD,
                           strength: float, set_cond_area: str,
                           opt_mask: Tensor=None, opt_lora_hook: LoraHookGroup=None, opt_timesteps: TimestepsCond=None):
        final_positive, final_negative = set_mask_and_combine_conds(conds=[positive, negative], new_conds=[positive_ADD, negative_ADD],
                                                                    strength=strength, set_cond_area=set_cond_area,
                                                                    opt_mask=opt_mask, opt_lora_hook=opt_lora_hook, opt_timesteps=opt_timesteps)
        return (final_positive, final_negative,)


class ConditioningSetMaskAndCombineHooked:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond": ("CONDITIONING",),
                "cond_ADD": ("CONDITIONING",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (COND_CONST._LIST_COND_AREA,),
            },
            "optional": {
                "opt_mask": ("MASK", ),
                "opt_lora_hook": ("LORA_HOOK",),
                "opt_timesteps": ("TIMESTEPS_COND",)
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/single cond ops"
    FUNCTION = "append_and_combine"

    def append_and_combine(self, cond, cond_ADD,
                           strength: float, set_cond_area: str,
                           opt_mask: Tensor=None, opt_lora_hook: LoraHookGroup=None, opt_timesteps: TimestepsCond=None):
        (final_conditioning,) = set_mask_and_combine_conds(conds=[cond], new_conds=[cond_ADD],
                                                                    strength=strength, set_cond_area=set_cond_area,
                                                                    opt_mask=opt_mask, opt_lora_hook=opt_lora_hook, opt_timesteps=opt_timesteps)
        return (final_conditioning,)


class PairedConditioningSetUnmaskedAndCombineHooked:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "positive_DEFAULT": ("CONDITIONING",),
                "negative_DEFAULT": ("CONDITIONING",),
            },
            "optional": {
                "opt_lora_hook": ("LORA_HOOK",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning"
    FUNCTION = "append_and_combine"

    def append_and_combine(self, positive, negative, positive_DEFAULT, negative_DEFAULT,
                           opt_lora_hook: LoraHookGroup=None):
        final_positive, final_negative = set_unmasked_and_combine_conds(conds=[positive, negative], new_conds=[positive_DEFAULT, negative_DEFAULT],
                                                                        opt_lora_hook=opt_lora_hook)
        return (final_positive, final_negative,)
    

class ConditioningSetUnmaskedAndCombineHooked:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond": ("CONDITIONING",),
                "cond_DEFAULT": ("CONDITIONING",),
            },
            "optional": {
                "opt_lora_hook": ("LORA_HOOK",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/single cond ops"
    FUNCTION = "append_and_combine"

    def append_and_combine(self, cond, cond_DEFAULT,
                           opt_lora_hook: LoraHookGroup=None):
        (final_conditioning,) = set_unmasked_and_combine_conds(conds=[cond], new_conds=[cond_DEFAULT],
                                                                        opt_lora_hook=opt_lora_hook)
        return (final_conditioning,)
###############################################
###############################################
###############################################



###############################################
### Scheduling
###############################################
class ConditioningTimestepsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            }
        }
    
    RETURN_TYPES = ("TIMESTEPS_COND",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning"
    FUNCTION = "create_schedule"

    def create_schedule(self, start_percent: float, end_percent: float):
        return (TimestepsCond(start_percent=start_percent, end_percent=end_percent),)


class SetLoraHookKeyframes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_hook": ("LORA_HOOK",), 
                "hook_kf": ("LORA_HOOK_KEYFRAMES",),
            }
        }
    
    RETURN_TYPES = ("LORA_HOOK",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning"
    FUNCTION = "set_hook_keyframes"

    def set_hook_keyframes(self, lora_hook: LoraHookGroup, hook_kf: LoraHookKeyframeGroup):
        new_lora_hook = lora_hook.clone()
        new_lora_hook.set_keyframes_on_hooks(hook_kf=hook_kf)
        return (new_lora_hook,)


class CreateLoraHookKeyframe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
            },
            "optional": {
                "prev_hook_kf": ("LORA_HOOK_KEYFRAMES",),
            }
        }
    
    RETURN_TYPES = ("LORA_HOOK_KEYFRAMES",)
    RETURN_NAMES = ("HOOK_KF",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/schedule lora hooks"
    FUNCTION = "create_hook_keyframe"

    def create_hook_keyframe(self, strength_model: float, start_percent: float, guarantee_steps: float,
                             prev_hook_kf: LoraHookKeyframeGroup=None):
        if prev_hook_kf:
            prev_hook_kf = prev_hook_kf.clone()
        else:
            prev_hook_kf = LoraHookKeyframeGroup()
        keyframe = LoraHookKeyframe(strength=strength_model, start_percent=start_percent, guarantee_steps=guarantee_steps)
        prev_hook_kf.add(keyframe)
        return (prev_hook_kf,)


class CreateLoraHookKeyframeInterpolation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "strength_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "interpolation": (InterpolationMethod._LIST, ),
                "intervals": ("INT", {"default": 5, "min": 2, "max": 100, "step": 1}),
                "print_keyframes": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_hook_kf": ("LORA_HOOK_KEYFRAMES",),
            }
        }
    
    RETURN_TYPES = ("LORA_HOOK_KEYFRAMES",)
    RETURN_NAMES = ("HOOK_KF",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/schedule lora hooks"
    FUNCTION = "create_hook_keyframes"

    def create_hook_keyframes(self,
                              start_percent: float, end_percent: float,
                              strength_start: float, strength_end: float, interpolation: str, intervals: int,
                              prev_hook_kf: LoraHookKeyframeGroup=None, print_keyframes=False):
        if prev_hook_kf:
            prev_hook_kf = prev_hook_kf.clone()
        else:
            prev_hook_kf = LoraHookKeyframeGroup()
        percents = InterpolationMethod.get_weights(num_from=start_percent, num_to=end_percent, length=intervals, method=InterpolationMethod.LINEAR)
        strengths = InterpolationMethod.get_weights(num_from=strength_start, num_to=strength_end, length=intervals, method=interpolation)
        
        is_first = True
        for percent, strength in zip(percents, strengths):
            guarantee_steps = 0
            if is_first:
                guarantee_steps = 1
                is_first = False
            prev_hook_kf.add(LoraHookKeyframe(strength=strength, start_percent=percent, guarantee_steps=guarantee_steps))
            if print_keyframes:
                logger.info(f"LoraHookKeyframe - start_percent:{percent} = {strength}")
        return (prev_hook_kf,)
    

class CreateLoraHookKeyframeFromStrengthList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "strengths_float": ("FLOAT", {"default": -1, "min": -1, "step": 0.001, "forceInput": True}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "print_keyframes": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_hook_kf": ("LORA_HOOK_KEYFRAMES",),
            }
        }
    
    RETURN_TYPES = ("LORA_HOOK_KEYFRAMES",)
    RETURN_NAMES = ("HOOK_KF",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/schedule lora hooks"
    FUNCTION = "create_hook_keyframes"

    def create_hook_keyframes(self, strengths_float: Union[float, list[float]],
                              start_percent: float, end_percent: float,
                              prev_hook_kf: LoraHookKeyframeGroup=None, print_keyframes=False):
        if prev_hook_kf:
            prev_hook_kf = prev_hook_kf.clone()
        else:
            prev_hook_kf = LoraHookKeyframeGroup()
        if type(strengths_float) in (float, int):
            strengths_float = [float(strengths_float)]
        elif isinstance(strengths_float, Iterable):
            pass
        else:
            raise Exception(f"strengths_floast must be either an interable input or a float, but was {type(strengths_float).__repr__}.")
        percents = InterpolationMethod.get_weights(num_from=start_percent, num_to=end_percent, length=len(strengths_float), method=InterpolationMethod.LINEAR)

        is_first = True
        for percent, strength in zip(percents, strengths_float):
            guarantee_steps = 0
            if is_first:
                guarantee_steps = 1
                is_first = False
            prev_hook_kf.add(LoraHookKeyframe(strength=strength, start_percent=percent, guarantee_steps=guarantee_steps))
            if print_keyframes:
                logger.info(f"LoraHookKeyframe - start_percent:{percent} = {strength}")
        return (prev_hook_kf,)
###############################################
###############################################
###############################################


###############################################
### Register LoRA Hooks
###############################################
# based on ComfyUI's nodes.py LoraLoader
class MaskableLoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "LORA_HOOK")
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/register lora hooks"
    FUNCTION = "load_lora"

    def load_lora(self, model: Union[ModelPatcher, ModelPatcherAndInjector], clip: CLIP, lora_name: str, strength_model: float, strength_clip: float):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)
        
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp
        
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        lora_hook = LoraHook(lora_name=lora_name)
        lora_hook_group = LoraHookGroup()
        lora_hook_group.add(lora_hook)
        model_lora, clip_lora = load_hooked_lora_for_models(model=model, clip=clip, lora=lora, lora_hook=lora_hook,
                                                            strength_model=strength_model, strength_clip=strength_clip)
        return (model_lora, clip_lora, lora_hook_group)


class MaskableLoraLoaderModelOnly(MaskableLoraLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "LORA_HOOK")
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/register lora hooks"
    FUNCTION = "load_lora_model_only"

    def load_lora_model_only(self, model: ModelPatcher, lora_name: str, strength_model: float):
        model_lora, clip_lora, lora_hook = self.load_lora(model=model, clip=None, lora_name=lora_name,
                                                          strength_model=strength_model, strength_clip=0)
        return (model_lora, lora_hook)


class MaskableSDModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "LORA_HOOK")
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/register lora hooks"
    FUNCTION = "load_model_as_lora"

    def load_model_as_lora(self, model: ModelPatcher, clip: CLIP, ckpt_name: str, strength_model: float, strength_clip: float):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        model_loaded = out[0]
        clip_loaded = out[1]

        lora_hook = LoraHook(lora_name=ckpt_name)
        lora_hook_group = LoraHookGroup()
        lora_hook_group.add(lora_hook)
        model_lora, clip_lora = load_model_as_hooked_lora_for_models(model=model, clip=clip,
                                                                     model_loaded=model_loaded, clip_loaded=clip_loaded,
                                                                     lora_hook=lora_hook,
                                                                     strength_model=strength_model, strength_clip=strength_clip)
        return (model_lora, clip_lora, lora_hook_group)


class MaskableSDModelLoaderModelOnly(MaskableSDModelLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "LORA_HOOK")
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/register lora hooks"
    FUNCTION = "load_model_as_lora_model_only"

    def load_model_as_lora_model_only(self, model: ModelPatcher, ckpt_name: str, strength_model: float):
        model_lora, clip_lora, lora_hook = self.load_model_as_lora(model=model, clip=None, ckpt_name=ckpt_name,
                                                                   strength_model=strength_model, strength_clip=0)
        return (model_lora, lora_hook)
###############################################
###############################################
###############################################



###############################################
### Set LoRA Hooks
###############################################
class SetModelLoraHook:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "lora_hook": ("LORA_HOOK",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/single cond ops"
    FUNCTION = "attach_lora_hook"

    def attach_lora_hook(self, conditioning, lora_hook: LoraHookGroup):
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]["lora_hook"] = lora_hook
            c.append(n)
        return (c, )
    

class SetClipLoraHook:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "lora_hook": ("LORA_HOOK",),
            }
        }
    
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("hook_CLIP",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning"
    FUNCTION = "apply_lora_hook"

    def apply_lora_hook(self, clip: CLIP, lora_hook: LoraHookGroup):
        new_clip = CLIPWithHooks(clip)
        new_clip.set_desired_hooks(lora_hooks=lora_hook)
        return (new_clip, )


class CombineLoraHooks:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "lora_hook_A": ("LORA_HOOK",),
                "lora_hook_B": ("LORA_HOOK",),
            }
        }
    
    RETURN_TYPES = ("LORA_HOOK",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/combine lora hooks"
    FUNCTION = "combine_lora_hooks"

    def combine_lora_hooks(self, lora_hook_A: LoraHookGroup=None, lora_hook_B: LoraHookGroup=None):
        candidates = [lora_hook_A, lora_hook_B]
        return (LoraHookGroup.combine_all_lora_hooks(candidates),)


class CombineLoraHookFourOptional:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "lora_hook_A": ("LORA_HOOK",),
                "lora_hook_B": ("LORA_HOOK",),
                "lora_hook_C": ("LORA_HOOK",),
                "lora_hook_D": ("LORA_HOOK",),
            }
        }

    RETURN_TYPES = ("LORA_HOOK",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/combine lora hooks"
    FUNCTION = "combine_lora_hooks"

    def combine_lora_hooks(self,
                           lora_hook_A: LoraHookGroup=None, lora_hook_B: LoraHookGroup=None,
                           lora_hook_C: LoraHookGroup=None, lora_hook_D: LoraHookGroup=None,):
        candidates = [lora_hook_A, lora_hook_B, lora_hook_C, lora_hook_D]
        return (LoraHookGroup.combine_all_lora_hooks(candidates),)


class CombineLoraHookEightOptional:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "lora_hook_A": ("LORA_HOOK",),
                "lora_hook_B": ("LORA_HOOK",),
                "lora_hook_C": ("LORA_HOOK",),
                "lora_hook_D": ("LORA_HOOK",),
                "lora_hook_E": ("LORA_HOOK",),
                "lora_hook_F": ("LORA_HOOK",),
                "lora_hook_G": ("LORA_HOOK",),
                "lora_hook_H": ("LORA_HOOK",),
            }
        }

    RETURN_TYPES = ("LORA_HOOK",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/combine lora hooks"
    FUNCTION = "combine_lora_hooks"

    def combine_lora_hooks(self,
                           lora_hook_A: LoraHookGroup=None, lora_hook_B: LoraHookGroup=None,
                           lora_hook_C: LoraHookGroup=None, lora_hook_D: LoraHookGroup=None,
                           lora_hook_E: LoraHookGroup=None, lora_hook_F: LoraHookGroup=None,
                           lora_hook_G: LoraHookGroup=None, lora_hook_H: LoraHookGroup=None):
        candidates = [lora_hook_A, lora_hook_B, lora_hook_C, lora_hook_D,
                      lora_hook_E, lora_hook_F, lora_hook_G, lora_hook_H]
        return (LoraHookGroup.combine_all_lora_hooks(candidates),)

# NOTE: if at some point I add more Javascript stuff to this repo, there should be a combine node
#   that dynamically increases the hooks available to plug in on the node
###############################################
###############################################
###############################################
