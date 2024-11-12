from pathlib import Path

import folder_paths
import comfy.utils
import comfy.sd

from .logger import logger
from .utils_model import get_available_motion_loras, get_motion_lora_path
from .motion_lora import MotionLoraInfo, MotionLoraList


class AnimateDiffLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": (get_available_motion_loras(),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            },
            "optional": {
                "prev_motion_lora": ("MOTION_LORA",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 30}),
            }
        }
    
    RETURN_TYPES = ("MOTION_LORA",)
    CATEGORY = "Animate Diff 🎭🅐🅓"
    FUNCTION = "load_motion_lora"

    def load_motion_lora(self, name: str, strength: float, prev_motion_lora: MotionLoraList=None, lora_name: str=None):
        if prev_motion_lora is None:
            prev_motion_lora = MotionLoraList()
        else:
            prev_motion_lora = prev_motion_lora.clone()
        if lora_name is not None: # backwards compatibility
            name = lora_name
        # check if motion lora with name exists
        lora_path = get_motion_lora_path(name)
        if not Path(lora_path).is_file():
            raise FileNotFoundError(f"Motion lora with name '{name}' not found.")
        # create motion lora info to be loaded in AnimateDiff Loader
        lora_info = MotionLoraInfo(name=name, strength=strength)
        prev_motion_lora.add_lora(lora_info)

        return (prev_motion_lora,)
