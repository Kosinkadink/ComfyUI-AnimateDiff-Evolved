from torch import Tensor
from typing import Union

from .context import (ContextOptionsGroup)
from .context_extras import (ContextExtrasGroup,
                             ContextRef, ContextRefTune, ContextRefMode, ContextRefKeyframeGroup, ContextRefKeyframe,
                             NaiveReuse, NaiveReuseKeyframe, NaiveReuseKeyframeGroup)
from .utils_model import BIGMAX
from .utils_scheduling import convert_str_to_indexes


class SetContextExtrasOnContextOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_opts": ("CONTEXT_OPTIONS",),
                "context_extras": ("CONTEXT_EXTRAS",),
            },
            "optional": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    RETURN_NAMES = ("CONTEXT_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/context extras"
    FUNCTION = "set_context_extras"

    def set_context_extras(self, context_opts: ContextOptionsGroup, context_extras: ContextExtrasGroup):
        context_opts = context_opts.clone()
        context_opts.extras = context_extras.clone()
        return (context_opts,)


#########################################
# NaiveReuse
class ContextExtras_NaiveReuse:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "prev_extras": ("CONTEXT_EXTRAS",),
                "strength_multival": ("MULTIVAL",),
                "naivereuse_kf": ("NAIVEREUSE_KEYFRAME",),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.001}),
                "weighted_mean": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.001}),
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_EXTRAS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/context extras"
    FUNCTION = "create_context_extra"

    def create_context_extra(self, start_percent=0.0, end_percent=0.1, weighted_mean=0.95, strength_multival: Union[float, Tensor]=None,
                             naivereuse_kf: NaiveReuseKeyframeGroup=None, prev_extras: ContextExtrasGroup=None):
        if prev_extras is None:
            prev_extras = prev_extras = ContextExtrasGroup()
        prev_extras = prev_extras.clone()
        # create extra
        naive_reuse = NaiveReuse(start_percent=start_percent, end_percent=end_percent, weighted_mean=weighted_mean, multival_opt=strength_multival,
                                 naivereuse_kf=naivereuse_kf)
        prev_extras.add(naive_reuse)
        return (prev_extras,)


class NaiveReuse_KeyframeMultivalNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "prev_kf": ("NAIVEREUSE_KEYFRAME",),
                "mult_multival": ("MULTIVAL",),
                "mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("NAIVEREUSE_KEYFRAME",)
    RETURN_NAMES = ("NAIVEREUSE_KF",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/context extras/naivereuse"
    FUNCTION = "create_keyframe"

    def create_keyframe(self, prev_kf=None, mult=1.0, mult_multival=1.0,
                        start_percent=0.0, guarantee_steps=1):
        if prev_kf is None:
            prev_kf = NaiveReuseKeyframeGroup()
        prev_kf = prev_kf.clone()
        kf = NaiveReuseKeyframe(mult=mult, mult_multival=mult_multival, start_percent=start_percent, guarantee_steps=guarantee_steps)
        prev_kf.add(kf)
        return (prev_kf,)
#----------------------------------------
#########################################


#########################################
# ContextRef
class ContextExtras_ContextRef:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "prev_extras": ("CONTEXT_EXTRAS",),
                "strength_multival": ("MULTIVAL",),
                "contextref_mode": ("CONTEXTREF_MODE",),
                "contextref_tune": ("CONTEXTREF_TUNE",),
                "contextref_kf": ("CONTEXTREF_KEYFRAME",),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.001}),
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_EXTRAS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/context extras"
    FUNCTION = "create_context_extra"

    def create_context_extra(self, start_percent=0.0, end_percent=0.1, strength_multival: Union[float, Tensor]=None,
                             contextref_mode: ContextRefMode=None, contextref_tune: ContextRefTune=None,
                             contextref_kf: ContextRefKeyframeGroup=None, prev_extras: ContextExtrasGroup=None):
        if prev_extras is None:
            prev_extras = prev_extras = ContextExtrasGroup()
        prev_extras = prev_extras.clone()
        # create extra
        # TODO: make customizable, and allow mask input
        if contextref_tune is None:
            contextref_tune = ContextRefTune(attn_style_fidelity=1.0, attn_ref_weight=1.0, attn_strength=1.0)
        if contextref_mode is None:
            contextref_mode = ContextRefMode.init_first()
        context_ref = ContextRef(start_percent=start_percent, end_percent=end_percent,
                                 strength_multival=strength_multival, tune=contextref_tune, mode=contextref_mode,
                                 keyframe=contextref_kf)
        prev_extras.add(context_ref)
        return (prev_extras,)


class ContextRef_KeyframeMultivalNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "prev_kf": ("CONTEXTREF_KEYFRAME",),
                "mult_multival": ("MULTIVAL",),
                "mode_replace": ("CONTEXTREF_MODE",),
                "tune_replace": ("CONTEXTREF_TUNE",),
                "mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
                "inherit_missing": ("BOOLEAN", {"default": True}, ),
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CONTEXTREF_KEYFRAME",)
    RETURN_NAMES = ("CONTEXTREF_KF",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/context extras/contextref"
    FUNCTION = "create_keyframe"

    def create_keyframe(self, prev_kf=None, mult=1.0, mult_multival=1.0, mode_replace=None, tune_replace=None,
                        start_percent=1.0, guarantee_steps=1, inherit_missing=True):
        if prev_kf is None:
            prev_kf = ContextRefKeyframeGroup()
        prev_kf = prev_kf.clone()
        kf = ContextRefKeyframe(mult=mult, mult_multival=mult_multival, tune_replace=tune_replace, mode_replace=mode_replace,
                                start_percent=start_percent, guarantee_steps=guarantee_steps, inherit_missing=inherit_missing)
        prev_kf.add(kf)
        return (prev_kf,)


class ContextRef_ModeFirst:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "autosize": ("ADEAUTOSIZE", {"padding": 25}),
            },
        }
    
    RETURN_TYPES = ("CONTEXTREF_MODE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/context extras/contextref"
    FUNCTION = "create_contextref_mode"

    def create_contextref_mode(self):
        mode = ContextRefMode.init_first()
        return (mode,)


class ContextRef_ModeSliding:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "sliding_width": ("INT", {"default": 2, "min": 2, "max": BIGMAX, "step": 1}),
                "autosize": ("ADEAUTOSIZE", {"padding": 42}),
            }
        }
    
    RETURN_TYPES = ("CONTEXTREF_MODE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/context extras/contextref"
    FUNCTION = "create_contextref_mode"

    def create_contextref_mode(self, sliding_width):
        mode = ContextRefMode.init_sliding(sliding_width=sliding_width)
        return (mode,)


class ContextRef_ModeIndexes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "switch_on_idxs": ("STRING", {"default": ""}),
                "always_include_0": ("BOOLEAN", {"default": True},),
                "autosize": ("ADEAUTOSIZE", {"padding": 50}),
            },
        }
    
    RETURN_TYPES = ("CONTEXTREF_MODE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/context extras/contextref"
    FUNCTION = "create_contextref_mode"

    def create_contextref_mode(self, switch_on_idxs: str, always_include_0: bool):
        idxs = set(convert_str_to_indexes(indexes_str=switch_on_idxs, length=0, allow_range=False))
        if always_include_0 and 0 not in idxs:
            idxs.add(0)
        mode = ContextRefMode.init_indexes(indexes=idxs)
        return (mode,)


class ContextRef_TuneAttnAdain:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "attn_style_fidelity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "attn_ref_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "attn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "adain_style_fidelity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "adain_ref_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "adain_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "autosize": ("ADEAUTOSIZE", {"padding": 65}),
            }
        }
    
    RETURN_TYPES = ("CONTEXTREF_TUNE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/context extras/contextref"
    FUNCTION = "create_contextref_tune"

    def create_contextref_tune(self, attn_style_fidelity=1.0, attn_ref_weight=1.0, attn_strength=1.0,
                        adain_style_fidelity=1.0, adain_ref_weight=1.0, adain_strength=1.0):
        params = ContextRefTune(attn_style_fidelity=attn_style_fidelity, adain_style_fidelity=adain_style_fidelity,
                                  attn_ref_weight=attn_ref_weight, adain_ref_weight=adain_ref_weight,
                                  attn_strength=attn_strength, adain_strength=adain_strength)
        return (params,)


class ContextRef_TuneAttn:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "attn_style_fidelity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "attn_ref_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "attn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "autosize": ("ADEAUTOSIZE", {"padding": 15}),
            }
        }
    
    RETURN_TYPES = ("CONTEXTREF_TUNE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/context extras/contextref"
    FUNCTION = "create_contextref_tune"

    def create_contextref_tune(self, attn_style_fidelity=1.0, attn_ref_weight=1.0, attn_strength=1.0):
        return ContextRef_TuneAttnAdain.create_contextref_tune(self,
                                                               attn_style_fidelity=attn_style_fidelity, attn_ref_weight=attn_ref_weight, attn_strength=attn_strength,
                                                               adain_ref_weight=0.0, adain_style_fidelity=0.0, adain_strength=0.0)
#----------------------------------------
#########################################