import uuid
import folder_paths
from typing import Union
from torch import Tensor
from collections.abc import Iterable

from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP
import comfy.sd
from comfy.hooks import HookGroup, HookKeyframeGroup, HookKeyframe
import comfy_extras.nodes_hooks
import comfy.hooks
import comfy.utils

from .utils_model import BIGMAX, InterpolationMethod
from .logger import logger


###################################################################
# EVERYTHING BELOW HERE IS DEPRECATED;
# Can be replaced with vanilla ComfyUI nodes
#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------
class COND_CONST:
    COND_AREA_DEFAULT = "default"
    COND_AREA_MASK_BOUNDS = "mask bounds"
    _LIST_COND_AREA = [COND_AREA_DEFAULT, COND_AREA_MASK_BOUNDS]


class CreateLoraHookKeyframeInterpolationDEPR:
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
                "prev_hook_kf": ("HOOK_KEYFRAMES",),
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    DEPRECATED = True
    RETURN_TYPES = ("HOOK_KEYFRAMES",)
    RETURN_NAMES = ("HOOK_KF",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/schedule lora hooks"
    FUNCTION = "create_hook_keyframes"

    def create_hook_keyframes(self,
                              start_percent: float, end_percent: float,
                              strength_start: float, strength_end: float, interpolation: str, intervals: int,
                              prev_hook_kf: HookKeyframeGroup=None, print_keyframes=False):
        if prev_hook_kf:
            prev_hook_kf = prev_hook_kf.clone()
        else:
            prev_hook_kf = HookKeyframeGroup()
        percents = InterpolationMethod.get_weights(num_from=start_percent, num_to=end_percent, length=intervals, method=InterpolationMethod.LINEAR)
        strengths = InterpolationMethod.get_weights(num_from=strength_start, num_to=strength_end, length=intervals, method=interpolation)
        
        is_first = True
        for percent, strength in zip(percents, strengths):
            guarantee_steps = 0
            if is_first:
                guarantee_steps = 1
                is_first = False
            prev_hook_kf.add(HookKeyframe(strength=strength, start_percent=percent, guarantee_steps=guarantee_steps))
            if print_keyframes:
                logger.info(f"HookKeyframe - start_percent:{percent} = {strength}")
        return (prev_hook_kf,)


###############################################
### Mask, Combine, and Hook Conditioning
###############################################
class PairedConditioningSetMaskHookedDEPR:
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
                "opt_lora_hook": ("HOOKS",),
                "opt_timesteps": ("TIMESTEPS_RANGE",),
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning"
    FUNCTION = "append_and_hook"
    DEPRECATED = True

    def append_and_hook(self, positive_ADD, negative_ADD,
                        strength: float, set_cond_area: str,
                        opt_mask: Tensor=None, opt_lora_hook: HookGroup=None, opt_timesteps: tuple=None):
        final_positive, final_negative = comfy.hooks.set_conds_props(conds=[positive_ADD, negative_ADD],
                                                        strength=strength, set_cond_area=set_cond_area,
                                                        mask=opt_mask, hooks=opt_lora_hook, timesteps_range=opt_timesteps)
        return (final_positive, final_negative)


class ConditioningSetMaskHookedDEPR:
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
                "opt_lora_hook": ("HOOKS",),
                "opt_timesteps": ("TIMESTEPS_RANGE",),
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/single cond ops"
    FUNCTION = "append_and_hook"
    DEPRECATED = True

    def append_and_hook(self, cond_ADD,
                        strength: float, set_cond_area: str,
                        opt_mask: Tensor=None, opt_lora_hook: HookGroup=None, opt_timesteps: tuple=None):
        (final_conditioning,) = comfy.hooks.set_conds_props(conds=[cond_ADD],
                                               strength=strength, set_cond_area=set_cond_area,
                                               mask=opt_mask, hooks=opt_lora_hook, timesteps_range=opt_timesteps)
        return (final_conditioning,) 


class PairedConditioningSetMaskAndCombineHookedDEPR:
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
                "opt_lora_hook": ("HOOKS",),
                "opt_timesteps": ("TIMESTEPS_RANGE",),
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning"
    FUNCTION = "append_and_combine"
    DEPRECATED = True

    def append_and_combine(self, positive, negative, positive_ADD, negative_ADD,
                           strength: float, set_cond_area: str,
                           opt_mask: Tensor=None, opt_lora_hook: HookGroup=None, opt_timesteps: tuple=None):
        final_positive, final_negative = comfy.hooks.set_conds_props_and_combine(conds=[positive, negative], new_conds=[positive_ADD, negative_ADD],
                                                                    strength=strength, set_cond_area=set_cond_area,
                                                                    mask=opt_mask, hooks=opt_lora_hook, timesteps_range=opt_timesteps)
        return (final_positive, final_negative,)


class ConditioningSetMaskAndCombineHookedDEPR:
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
                "opt_lora_hook": ("HOOKS",),
                "opt_timesteps": ("TIMESTEPS_RANGE",),
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/single cond ops"
    FUNCTION = "append_and_combine"
    DEPRECATED = True

    def append_and_combine(self, cond, cond_ADD,
                           strength: float, set_cond_area: str,
                           opt_mask: Tensor=None, opt_lora_hook: HookGroup=None, opt_timesteps: tuple=None):
        (final_conditioning,) = comfy.hooks.set_conds_props_and_combine(conds=[cond], new_conds=[cond_ADD],
                                                                    strength=strength, set_cond_area=set_cond_area,
                                                                    mask=opt_mask, hooks=opt_lora_hook, timesteps_range=opt_timesteps)
        return (final_conditioning,)


class PairedConditioningSetUnmaskedAndCombineHookedDEPR:
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
                "opt_lora_hook": ("HOOKS",),
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning"
    FUNCTION = "append_and_combine"
    DEPRECATED = True

    def append_and_combine(self, positive, negative, positive_DEFAULT, negative_DEFAULT,
                           opt_lora_hook: HookGroup=None):
        final_positive, final_negative = comfy.hooks.set_default_conds_and_combine(conds=[positive, negative], new_conds=[positive_DEFAULT, negative_DEFAULT],
                                                                        hooks=opt_lora_hook)
        return (final_positive, final_negative,)
    

class ConditioningSetUnmaskedAndCombineHookedDEPR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond": ("CONDITIONING",),
                "cond_DEFAULT": ("CONDITIONING",),
            },
            "optional": {
                "opt_lora_hook": ("HOOKS",),
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/single cond ops"
    FUNCTION = "append_and_combine"
    DEPRECATED = True

    def append_and_combine(self, cond, cond_DEFAULT,
                           opt_lora_hook: HookGroup=None):
        (final_conditioning,) = comfy.hooks.set_default_conds_and_combine(conds=[cond], new_conds=[cond_DEFAULT],
                                                                        hooks=opt_lora_hook)
        return (final_conditioning,)
    

class PairedConditioningCombineDEPR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_A": ("CONDITIONING",),
                "negative_A": ("CONDITIONING",),
                "positive_B": ("CONDITIONING",),
                "negative_B": ("CONDITIONING",),
            },
            "optional": {
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning"
    FUNCTION = "combine"
    DEPRECATED = True

    def combine(self, positive_A, negative_A, positive_B, negative_B):
        final_positive, final_negative = comfy.hooks.set_conds_props_and_combine(conds=[positive_A, negative_A], new_conds=[positive_B, negative_B],)
        return (final_positive, final_negative,)


class ConditioningCombineDEPR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond_A": ("CONDITIONING",),
                "cond_B": ("CONDITIONING",),
            },
            "optional": {
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/single cond ops"
    FUNCTION = "combine"
    DEPRECATED = True

    def combine(self, cond_A, cond_B):
        (final_conditioning,) = comfy.hooks.set_conds_props_and_combine(conds=[cond_A], new_conds=[cond_B],)
        return (final_conditioning,)
###############################################
###############################################
###############################################



###############################################
### Scheduling
###############################################
class ConditioningTimestepsNodeDEPR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
            "optional": {
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("TIMESTEPS_RANGE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning"
    FUNCTION = "create_schedule"
    DEPRECATED = True

    def create_schedule(self, start_percent: float, end_percent: float):
        return ((start_percent, end_percent),)


class SetLoraHookKeyframesDEPR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_hook": ("HOOKS",), 
                "hook_kf": ("HOOK_KEYFRAMES",),
            },
            "optional": {
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("HOOKS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning"
    FUNCTION = "set_hook_keyframes"
    DEPRECATED = True

    def set_hook_keyframes(self, lora_hook: HookGroup, hook_kf: HookKeyframeGroup):
        new_lora_hook = lora_hook.clone()
        new_lora_hook.set_keyframes_on_hooks(hook_kf=hook_kf)
        return (new_lora_hook,)


class CreateLoraHookKeyframeDEPR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
            },
            "optional": {
                "prev_hook_kf": ("HOOK_KEYFRAMES",),
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("HOOK_KEYFRAMES",)
    RETURN_NAMES = ("HOOK_KF",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/schedule lora hooks"
    FUNCTION = "create_hook_keyframe"
    DEPRECATED = True

    def create_hook_keyframe(self, strength_model: float, start_percent: float, guarantee_steps: float,
                             prev_hook_kf: HookKeyframeGroup=None):
        if prev_hook_kf:
            prev_hook_kf = prev_hook_kf.clone()
        else:
            prev_hook_kf = HookKeyframeGroup()
        keyframe = HookKeyframe(strength=strength_model, start_percent=start_percent, guarantee_steps=guarantee_steps)
        prev_hook_kf.add(keyframe)
        return (prev_hook_kf,)
    

class CreateLoraHookKeyframeFromStrengthListDEPR:
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
                "prev_hook_kf": ("HOOK_KEYFRAMES",),
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            }
        }
    
    RETURN_TYPES = ("HOOK_KEYFRAMES",)
    RETURN_NAMES = ("HOOK_KF",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/schedule lora hooks"
    FUNCTION = "create_hook_keyframes"
    DEPRECATED = True

    def create_hook_keyframes(self, strengths_float: Union[float, list[float]],
                              start_percent: float, end_percent: float,
                              prev_hook_kf: HookKeyframeGroup=None, print_keyframes=False):
        if prev_hook_kf:
            prev_hook_kf = prev_hook_kf.clone()
        else:
            prev_hook_kf = HookKeyframeGroup()
        if type(strengths_float) in (float, int):
            strengths_float = [float(strengths_float)]
        elif isinstance(strengths_float, Iterable):
            pass
        else:
            raise Exception(f"strengths_float must be either an interable input or a float, but was {type(strengths_float).__repr__}.")
        percents = InterpolationMethod.get_weights(num_from=start_percent, num_to=end_percent, length=len(strengths_float), method=InterpolationMethod.LINEAR)

        is_first = True
        for percent, strength in zip(percents, strengths_float):
            guarantee_steps = 0
            if is_first:
                guarantee_steps = 1
                is_first = False
            prev_hook_kf.add(HookKeyframe(strength=strength, start_percent=percent, guarantee_steps=guarantee_steps))
            if print_keyframes:
                logger.info(f"HookKeyframe - start_percent:{percent} = {strength}")
        return (prev_hook_kf,)
###############################################
###############################################
###############################################


###############################################
### Register LoRA Hooks
###############################################
# based on ComfyUI's nodes.py LoraLoader
class MaskableLoraLoaderDEPR:
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
            },
            "optional": {
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "HOOKS")
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/register lora hooks"
    FUNCTION = "load_lora"
    DEPRECATED = True

    def load_lora(self, model: Union[ModelPatcher], clip: CLIP, lora_name: str, strength_model: float, strength_clip: float):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, None)
        
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
        
        model_lora, clip_lora, hooks = comfy.hooks.load_hook_lora_for_models(model=model, clip=clip, lora=lora,
                                                                             strength_model=strength_model, strength_clip=strength_clip)
        return (model_lora, clip_lora, hooks)


class MaskableLoraLoaderModelOnlyDEPR(MaskableLoraLoaderDEPR):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
            "optional": {
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            }
        }

    RETURN_TYPES = ("MODEL", "HOOKS")
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/register lora hooks"
    FUNCTION = "load_lora_model_only"
    DEPRECATED = True

    def load_lora_model_only(self, model: ModelPatcher, lora_name: str, strength_model: float):
        model_lora, _, hooks = self.load_lora(model=model, clip=None, lora_name=lora_name,
                                              strength_model=strength_model, strength_clip=0)
        return (model_lora, hooks)


class MaskableSDModelLoaderDEPR(comfy_extras.nodes_hooks.CreateHookModelAsLora):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
            "optional": {
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "HOOKS")
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/register lora hooks"
    FUNCTION = "load_model_as_lora"
    DEPRECATED = True

    def load_model_as_lora(self, model: ModelPatcher, clip: CLIP, ckpt_name: str, strength_model: float, strength_clip: float):
        returned = self.create_hook(ckpt_name=ckpt_name, strength_model=strength_model, strength_clip=strength_clip)
        return (model.clone(), clip.clone(), returned[0])


class MaskableSDModelLoaderModelOnlyDEPR(MaskableSDModelLoaderDEPR):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
            "optional": {
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "HOOKS")
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/register lora hooks"
    FUNCTION = "load_model_as_lora_model_only"
    DEPRECATED = True

    def load_model_as_lora_model_only(self, model: ModelPatcher, ckpt_name: str, strength_model: float):
        model_lora, _, hooks = self.load_model_as_lora(model=model, clip=None, ckpt_name=ckpt_name,
                                                       strength_model=strength_model, strength_clip=0)
        return (model_lora, hooks)
###############################################
###############################################
###############################################



###############################################
### Set LoRA Hooks
###############################################
class SetModelLoraHookDEPR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "lora_hook": ("HOOKS",),
            },
            "optional": {
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/single cond ops"
    FUNCTION = "attach_lora_hook"
    DEPRECATED = True

    def attach_lora_hook(self, conditioning, lora_hook: HookGroup):
        return (comfy.hooks.set_hooks_for_conditioning(conditioning, lora_hook),)
    

class SetClipLoraHookDEPR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "lora_hook": ("HOOKS",),
            },
            "optional": {
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("hook_CLIP",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning"
    FUNCTION = "apply_lora_hook"
    DEPRECATED = True

    def apply_lora_hook(self, clip: CLIP, lora_hook: HookGroup):
        return comfy_extras.nodes_hooks.SetClipHooks.apply_hooks(self, clip, False, lora_hook)


class CombineLoraHooksDEPR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "lora_hook_A": ("HOOKS",),
                "lora_hook_B": ("HOOKS",),
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("HOOKS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/combine lora hooks"
    FUNCTION = "combine_lora_hooks"
    DEPRECATED = True

    def combine_lora_hooks(self, lora_hook_A: HookGroup=None, lora_hook_B: HookGroup=None):
        candidates = [lora_hook_A, lora_hook_B]
        return (HookGroup.combine_all_hooks(candidates),)


class CombineLoraHookFourOptionalDEPR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "lora_hook_A": ("HOOKS",),
                "lora_hook_B": ("HOOKS",),
                "lora_hook_C": ("HOOKS",),
                "lora_hook_D": ("HOOKS",),
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("HOOKS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/combine lora hooks"
    FUNCTION = "combine_lora_hooks"
    DEPRECATED = True

    def combine_lora_hooks(self,
                           lora_hook_A: HookGroup=None, lora_hook_B: HookGroup=None,
                           lora_hook_C: HookGroup=None, lora_hook_D: HookGroup=None,):
        candidates = [lora_hook_A, lora_hook_B, lora_hook_C, lora_hook_D]
        return (HookGroup.combine_all_hooks(candidates),)


class CombineLoraHookEightOptionalDEPR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "lora_hook_A": ("HOOKS",),
                "lora_hook_B": ("HOOKS",),
                "lora_hook_C": ("HOOKS",),
                "lora_hook_D": ("HOOKS",),
                "lora_hook_E": ("HOOKS",),
                "lora_hook_F": ("HOOKS",),
                "lora_hook_G": ("HOOKS",),
                "lora_hook_H": ("HOOKS",),
                "deprecation_warning": ("ADEWARN", {"text": "Deprecated - use native ComfyUI nodes instead."}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("HOOKS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/combine lora hooks"
    FUNCTION = "combine_lora_hooks"
    DEPRECATED = True

    def combine_lora_hooks(self,
                           lora_hook_A: HookGroup=None, lora_hook_B: HookGroup=None,
                           lora_hook_C: HookGroup=None, lora_hook_D: HookGroup=None,
                           lora_hook_E: HookGroup=None, lora_hook_F: HookGroup=None,
                           lora_hook_G: HookGroup=None, lora_hook_H: HookGroup=None):
        candidates = [lora_hook_A, lora_hook_B, lora_hook_C, lora_hook_D,
                      lora_hook_E, lora_hook_F, lora_hook_G, lora_hook_H]
        return (HookGroup.combine_all_hooks(candidates),)

# NOTE: if at some point I add more Javascript stuff to this repo, there should be a combine node
#   that dynamically increases the hooks available to plug in on the node
###############################################
###############################################
###############################################
