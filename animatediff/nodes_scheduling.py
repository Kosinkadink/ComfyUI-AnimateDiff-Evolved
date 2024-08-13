from .scheduling import evaluate_prompt_schedule, evaluate_value_schedule, TensorInterp
from .utils_model import BIGMAX
from .logger import logger


class PromptSchedulingLatentsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompts": ("STRING", {"multiline": True, "default": ""}),
                "clip": ("CLIP",),
                "latent": ("LATENT",),
            },
            "optional": {
                "print_schedule": ("BOOLEAN", {"default": False}),
                "tensor_interp": (TensorInterp._LIST,)
            },
        }
    
    RETURN_TYPES = ("CONDITIONING", "LATENT",)

    CATEGORY = "Animate Diff 🎭🅐🅓/scheduling"
    FUNCTION = "create_schedule"

    def create_schedule(self, prompts: str, clip, latent: dict, print_schedule=False, tensor_interp=TensorInterp.LERP):
        conditioning = evaluate_prompt_schedule(prompts, latent["samples"].size(0), clip, tensor_interp)
        return (conditioning, latent)


class ValueSchedulingLatentsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "values": ("STRING", {"multiline": True, "default": ""}),
                "latent": ("LATENT",),
            },
            "optional": {
                "print_schedule": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("FLOAT", "FLOATS", "INT", "INTS")
    CATEGORY = "Animate Diff 🎭🅐🅓/scheduling"
    FUNCTION = "create_schedule"

    def create_schedule(self, values: str, latent: dict, print_schedule=False):
        float_vals = evaluate_value_schedule(values, latent["samples"].size(0))
        int_vals = [round(x) for x in float_vals]
        if print_schedule:
            for i, val in enumerate(float_vals):
                logger.info(f"ValueScheduling: {i} = {val}")
        return (float_vals, float_vals, int_vals, int_vals)


class ValueSchedulingNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "values": ("STRING", {"multiline": True, "default": ""}),
                "max_length": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
            },
            "optional": {
                "print_schedule": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOATS", "INT", "INTS")
    CATEGORY = "Animate Diff 🎭🅐🅓/scheduling"
    FUNCTION = "create_schedule"

    def create_schedule(self, values: str, max_length: int, print_schedule=False):
        float_vals = evaluate_value_schedule(values, max_length)
        int_vals = [round(x) for x in float_vals]
        if print_schedule:
            for i, val in enumerate(float_vals):
                logger.info(f"ValueScheduling: {i} = {val}")
        return (float_vals, float_vals, int_vals, int_vals)
