from .context import ContextFuseMethod, ContextOptions, ContextSchedules


class LoopedUniformContextOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_length": ("INT", {"default": 16, "min": 1, "max": 128}), # keep an eye on these max values
                "context_stride": ("INT", {"default": 1, "min": 1, "max": 32}),  # would need to be updated
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": 128}), # if new motion modules come out
                "context_schedule": (ContextSchedules.UNIFORM_SCHEDULE_LIST,),
                "closed_loop": ("BOOLEAN", {"default": False},),
                #"sync_context_to_pe": ("BOOLEAN", {"default": False},),
            },
            "optional": {
                "fuse_method": (ContextFuseMethod.LIST,),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts"
    FUNCTION = "create_options"

    def create_options(self, context_length: int, context_stride: int, context_overlap: int, context_schedule: int, closed_loop: bool,
                       fuse_method: str=ContextFuseMethod.FLAT):
        context_options = ContextOptions(
            context_length=context_length,
            context_stride=context_stride,
            context_overlap=context_overlap,
            context_schedule=context_schedule,
            closed_loop=closed_loop,
            fuse_method=fuse_method,
            )
        #context_options.set_sync_context_to_pe(sync_context_to_pe)
        return (context_options,)


class StandardUniformContextOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_length": ("INT", {"default": 16, "min": 1, "max": 128}), # keep an eye on these max values
                "context_stride": ("INT", {"default": 1, "min": 1, "max": 32}),  # would need to be updated
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": 128}), # if new motion modules come out
            },
            "optional": {
                "fuse_method": (ContextFuseMethod.LIST,),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts"
    FUNCTION = "create_options"

    def create_options(self, context_length: int, context_stride: int, context_overlap: int,
                       fuse_method: str=ContextFuseMethod.FLAT):
        context_options = ContextOptions(
            context_length=context_length,
            context_stride=context_stride,
            context_overlap=context_overlap,
            context_schedule=ContextSchedules.UNIFORM_STANDARD,
            closed_loop=False,
            fuse_method=fuse_method,
            )
        return (context_options,)


class StandardStaticContextOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_length": ("INT", {"default": 16, "min": 1, "max": 128}), # keep an eye on these max values
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": 128}), # if new motion modules come out
            },
            "optional": {
                "fuse_method": (ContextFuseMethod.LIST, {"default": ContextFuseMethod.PYRAMID}),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts"
    FUNCTION = "create_options"

    def create_options(self, context_length: int, context_overlap: int,
                       fuse_method: str=ContextFuseMethod.FLAT):
        context_options = ContextOptions(
            context_length=context_length,
            context_stride=None,
            context_overlap=context_overlap,
            context_schedule=ContextSchedules.STATIC_STANDARD,
            fuse_method=fuse_method,
            )
        return (context_options,)


class BatchedContextOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_length": ("INT", {"default": 16, "min": 1, "max": 128}), # keep an eye on these max values
            },
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts"
    FUNCTION = "create_options"

    def create_options(self, context_length: int):
        context_options = ContextOptions(
            context_length=context_length,
            context_overlap=0,
            context_schedule=ContextSchedules.BATCHED,
            )
        return (context_options,)
