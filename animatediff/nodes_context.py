from torch import Tensor
from typing import Union

import comfy.samplers
from comfy.model_patcher import ModelPatcher

from .context import (ContextFuseMethod, ContextOptions, ContextOptionsGroup, ContextSchedules,
                      generate_context_visualization)
from .utils_model import BIGMAX, MAX_RESOLUTION


LENGTH_MAX = 128   # keep an eye on these max values;
STRIDE_MAX = 32    # would need to be updated
OVERLAP_MAX = 128  # if new motion modules come out


class LoopedUniformContextOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_length": ("INT", {"default": 16, "min": 1, "max": LENGTH_MAX}),
                "context_stride": ("INT", {"default": 1, "min": 1, "max": STRIDE_MAX}),
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": OVERLAP_MAX}),
                "closed_loop": ("BOOLEAN", {"default": False},),
                #"sync_context_to_pe": ("BOOLEAN", {"default": False},),
            },
            "optional": {
                "fuse_method": (ContextFuseMethod.LIST,),
                "use_on_equal_length": ("BOOLEAN", {"default": False},),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
                "prev_context": ("CONTEXT_OPTIONS",),
                "view_opts": ("VIEW_OPTS",),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    RETURN_NAMES = ("CONTEXT_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts"
    FUNCTION = "create_options"

    def create_options(self, context_length: int, context_stride: int, context_overlap: int, closed_loop: bool,
                       fuse_method: str=ContextFuseMethod.FLAT, use_on_equal_length=False, start_percent: float=0.0, guarantee_steps: int=1,
                       view_opts: ContextOptions=None, prev_context: ContextOptionsGroup=None):
        if prev_context is None:
            prev_context = ContextOptionsGroup()
        prev_context = prev_context.clone()

        context_options = ContextOptions(
            context_length=context_length,
            context_stride=context_stride,
            context_overlap=context_overlap,
            context_schedule=ContextSchedules.UNIFORM_LOOPED,
            closed_loop=closed_loop,
            fuse_method=fuse_method,
            use_on_equal_length=use_on_equal_length,
            start_percent=start_percent,
            guarantee_steps=guarantee_steps,
            view_options=view_opts,
            )
        #context_options.set_sync_context_to_pe(sync_context_to_pe)
        prev_context.add(context_options)
        return (prev_context,)


# This Legacy version exists to maintain compatiblity with old workflows
class LegacyLoopedUniformContextOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_length": ("INT", {"default": 16, "min": 1, "max": LENGTH_MAX}),
                "context_stride": ("INT", {"default": 1, "min": 1, "max": STRIDE_MAX}),
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": OVERLAP_MAX}),
                "context_schedule": (ContextSchedules.LEGACY_UNIFORM_SCHEDULE_LIST,),
                "closed_loop": ("BOOLEAN", {"default": False},),
                #"sync_context_to_pe": ("BOOLEAN", {"default": False},),
            },
            "optional": {
                "fuse_method": (ContextFuseMethod.LIST, {"default": ContextFuseMethod.FLAT}),
                "use_on_equal_length": ("BOOLEAN", {"default": False},),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
                "prev_context": ("CONTEXT_OPTIONS",),
                "view_opts": ("VIEW_OPTS",),
                "deprecation_warning": ("ADEWARN", {"text": ""}),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    RETURN_NAMES = ("CONTEXT_OPTS",)
    CATEGORY = ""  # No Category, so will not appear in menu
    FUNCTION = "create_options"

    def create_options(self, fuse_method: str=ContextFuseMethod.FLAT, context_schedule: str=None, **kwargs):
        return LoopedUniformContextOptionsNode.create_options(self, fuse_method=fuse_method, **kwargs)


class StandardUniformContextOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_length": ("INT", {"default": 16, "min": 1, "max": LENGTH_MAX}),
                "context_stride": ("INT", {"default": 1, "min": 1, "max": STRIDE_MAX}),
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": OVERLAP_MAX}),
            },
            "optional": {
                "fuse_method": (ContextFuseMethod.LIST,),
                "use_on_equal_length": ("BOOLEAN", {"default": False},),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
                "prev_context": ("CONTEXT_OPTIONS",),
                "view_opts": ("VIEW_OPTS",),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    RETURN_NAMES = ("CONTEXT_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts"
    FUNCTION = "create_options"

    def create_options(self, context_length: int, context_stride: int, context_overlap: int,
                       fuse_method: str=ContextFuseMethod.PYRAMID, use_on_equal_length=False, start_percent: float=0.0, guarantee_steps: int=1,
                       view_opts: ContextOptions=None, prev_context: ContextOptionsGroup=None):
        if prev_context is None:
            prev_context = ContextOptionsGroup()
        prev_context = prev_context.clone()

        context_options = ContextOptions(
            context_length=context_length,
            context_stride=context_stride,
            context_overlap=context_overlap,
            context_schedule=ContextSchedules.UNIFORM_STANDARD,
            closed_loop=False,
            fuse_method=fuse_method,
            use_on_equal_length=use_on_equal_length,
            start_percent=start_percent,
            guarantee_steps=guarantee_steps,
            view_options=view_opts,
            )
        prev_context.add(context_options)
        return (prev_context,)


class StandardStaticContextOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_length": ("INT", {"default": 16, "min": 1, "max": LENGTH_MAX}),
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": OVERLAP_MAX}),
            },
            "optional": {
                "fuse_method": (ContextFuseMethod.LIST_STATIC,),
                "use_on_equal_length": ("BOOLEAN", {"default": False},),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
                "prev_context": ("CONTEXT_OPTIONS",),
                "view_opts": ("VIEW_OPTS",),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    RETURN_NAMES = ("CONTEXT_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts"
    FUNCTION = "create_options"

    def create_options(self, context_length: int, context_overlap: int,
                       fuse_method: str=ContextFuseMethod.PYRAMID, use_on_equal_length=False, start_percent: float=0.0, guarantee_steps: int=1,
                       view_opts: ContextOptions=None, prev_context: ContextOptionsGroup=None):
        if prev_context is None:
            prev_context = ContextOptionsGroup()
        prev_context = prev_context.clone()
        
        context_options = ContextOptions(
            context_length=context_length,
            context_stride=None,
            context_overlap=context_overlap,
            context_schedule=ContextSchedules.STATIC_STANDARD,
            fuse_method=fuse_method,
            use_on_equal_length=use_on_equal_length,
            start_percent=start_percent,
            guarantee_steps=guarantee_steps,
            view_options=view_opts,
            )
        prev_context.add(context_options)
        return (prev_context,)


class BatchedContextOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_length": ("INT", {"default": 16, "min": 1, "max": LENGTH_MAX}),
            },
            "optional": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
                "prev_context": ("CONTEXT_OPTIONS",),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    RETURN_NAMES = ("CONTEXT_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts"
    FUNCTION = "create_options"

    def create_options(self, context_length: int, start_percent: float=0.0, guarantee_steps: int=1,
                       prev_context: ContextOptionsGroup=None):
        if prev_context is None:
            prev_context = ContextOptionsGroup()
        prev_context = prev_context.clone()
        
        context_options = ContextOptions(
            context_length=context_length,
            context_overlap=0,
            context_schedule=ContextSchedules.BATCHED,
            start_percent=start_percent,
            guarantee_steps=guarantee_steps,
            )
        prev_context.add(context_options)
        return (prev_context,)


class ViewAsContextOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "view_opts_req": ("VIEW_OPTS",),
            },
            "optional": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
                "prev_context": ("CONTEXT_OPTIONS",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    RETURN_NAMES = ("CONTEXT_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts"
    FUNCTION = "create_options"

    def create_options(self, view_opts_req: ContextOptions, start_percent: float=0.0, guarantee_steps: int=1,
                       prev_context: ContextOptionsGroup=None):
        if prev_context is None:
            prev_context = ContextOptionsGroup()
        prev_context = prev_context.clone()
        context_options = ContextOptions(
            context_schedule=ContextSchedules.VIEW_AS_CONTEXT,
            start_percent=start_percent,
            guarantee_steps=guarantee_steps,
            view_options=view_opts_req,
            use_on_equal_length=True
        )
        prev_context.add(context_options)
        return (prev_context,)


#########################
# View Options
class StandardStaticViewOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "view_length": ("INT", {"default": 16, "min": 1, "max": LENGTH_MAX}),
                "view_overlap": ("INT", {"default": 4, "min": 0, "max": OVERLAP_MAX}),
            },
            "optional": {
                "fuse_method": (ContextFuseMethod.LIST,),
            }
        }
    
    RETURN_TYPES = ("VIEW_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/view opts"
    FUNCTION = "create_options"

    def create_options(self, view_length: int, view_overlap: int,
                       fuse_method: str=ContextFuseMethod.FLAT,):
        view_options = ContextOptions(
            context_length=view_length,
            context_stride=None,
            context_overlap=view_overlap,
            context_schedule=ContextSchedules.STATIC_STANDARD,
            fuse_method=fuse_method,
            )
        return (view_options,)


class StandardUniformViewOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "view_length": ("INT", {"default": 16, "min": 1, "max": LENGTH_MAX}),
                "view_stride": ("INT", {"default": 1, "min": 1, "max": STRIDE_MAX}),
                "view_overlap": ("INT", {"default": 4, "min": 0, "max": OVERLAP_MAX}),
            },
            "optional": {
                "fuse_method": (ContextFuseMethod.LIST,),
            }
        }
    
    RETURN_TYPES = ("VIEW_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/view opts"
    FUNCTION = "create_options"

    def create_options(self, view_length: int, view_overlap: int, view_stride: int,
                       fuse_method: str=ContextFuseMethod.PYRAMID,):
        view_options = ContextOptions(
            context_length=view_length,
            context_stride=view_stride,
            context_overlap=view_overlap,
            context_schedule=ContextSchedules.UNIFORM_STANDARD,
            fuse_method=fuse_method,
            )
        return (view_options,)


class LoopedUniformViewOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "view_length": ("INT", {"default": 16, "min": 1, "max": LENGTH_MAX}),
                "view_stride": ("INT", {"default": 1, "min": 1, "max": STRIDE_MAX}),
                "view_overlap": ("INT", {"default": 4, "min": 0, "max": OVERLAP_MAX}),
                "closed_loop": ("BOOLEAN", {"default": False},),
            },
            "optional": {
                "fuse_method": (ContextFuseMethod.LIST,),
                "use_on_equal_length": ("BOOLEAN", {"default": False},),
            }
        }
    
    RETURN_TYPES = ("VIEW_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/view opts"
    FUNCTION = "create_options"

    def create_options(self, view_length: int, view_overlap: int, view_stride: int, closed_loop: bool,
                       fuse_method: str=ContextFuseMethod.PYRAMID, use_on_equal_length=False):
        view_options = ContextOptions(
            context_length=view_length,
            context_stride=view_stride,
            context_overlap=view_overlap,
            context_schedule=ContextSchedules.UNIFORM_LOOPED,
            closed_loop=closed_loop,
            fuse_method=fuse_method,
            use_on_equal_length=use_on_equal_length,
            )
        return (view_options,)


class VisualizeContextOptionsKAdv:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
            },
            "optional": {
                "context_opts": ("CONTEXT_OPTIONS",),
                "visual_width": ("INT", {"min": 32, "max": MAX_RESOLUTION, "default": 1440}),
                "latents_length": ("INT", {"min": 1, "max": BIGMAX, "default": 32}),
                "steps": ("INT", {"min": 0, "max": BIGMAX, "default": 20}),
                "start_step": ("INT", {"min": 0, "max": BIGMAX, "default": 0}),
                "end_step": ("INT", {"min": 1, "max": BIGMAX, "default": 20}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/visualize"
    FUNCTION = "visualize"

    def visualize(self, model: ModelPatcher, sampler_name: str, scheduler: str, context_opts: ContextOptionsGroup=None,
                  visual_width=1440, latents_length=32, steps=20, start_step=0, end_step=20):
        images = generate_context_visualization(model=model, context_opts=context_opts, width=visual_width, video_length=latents_length,
                                                sampler_name=sampler_name, scheduler=scheduler,
                                                steps=steps, start_step=start_step, end_step=end_step)
        return (images,)


class VisualizeContextOptionsK:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
            },
            "optional": {
                "context_opts": ("CONTEXT_OPTIONS",),
                "visual_width": ("INT", {"min": 32, "max": MAX_RESOLUTION, "default": 1440}),
                "latents_length": ("INT", {"min": 1, "max": BIGMAX, "default": 32}),
                "steps": ("INT", {"min": 0, "max": BIGMAX, "default": 20}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/visualize"
    FUNCTION = "visualize"

    def visualize(self, model: ModelPatcher, sampler_name: str, scheduler: str, context_opts: ContextOptionsGroup=None,
                  visual_width=1440, latents_length=32, steps=20, denoise=1.0):
        images = generate_context_visualization(model=model, context_opts=context_opts, width=visual_width, video_length=latents_length,
                                                sampler_name=sampler_name, scheduler=scheduler,
                                                steps=steps, denoise=denoise)
        return (images,)


class VisualizeContextOptionsSCustom:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sigmas": ("SIGMAS", ),
            },
            "optional": {
                "context_opts": ("CONTEXT_OPTIONS",),
                "visual_width": ("INT", {"min": 32, "max": MAX_RESOLUTION, "default": 1440}),
                "latents_length": ("INT", {"min": 1, "max": BIGMAX, "default": 32}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/visualize"
    FUNCTION = "visualize"

    def visualize(self, model: ModelPatcher, sigmas,  context_opts: ContextOptionsGroup=None,
                  visual_width=1440, latents_length=32):
        images = generate_context_visualization(model=model, context_opts=context_opts, width=visual_width, video_length=latents_length,
                                                sigmas=sigmas)
        return (images,)
