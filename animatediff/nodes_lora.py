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
                "lora_name": (get_available_motion_loras(),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            },
            "optional": {
                "prev_motion_lora": ("MOTION_LORA",),
            }
        }
    
    RETURN_TYPES = ("MOTION_LORA",)
    CATEGORY = "Animate Diff 🎭🅐🅓"
    FUNCTION = "load_motion_lora"

    def load_motion_lora(self, lora_name: str, strength: float, prev_motion_lora: MotionLoraList=None):
        if prev_motion_lora is None:
            prev_motion_lora = MotionLoraList()
        else:
            prev_motion_lora = prev_motion_lora.clone()
        # check if motion lora with name exists
        lora_path = get_motion_lora_path(lora_name)
        if not Path(lora_path).is_file():
            raise FileNotFoundError(f"Motion lora with name '{lora_name}' not found.")
        # create motion lora info to be loaded in AnimateDiff Loader
        lora_info = MotionLoraInfo(name=lora_name, strength=strength)
        prev_motion_lora.add_lora(lora_info)

        return (prev_motion_lora,)


class MaskedLoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name": (folder_paths.get_filename_list("loras"), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              }}
    #RETURN_TYPES = ()
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
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

        from pathlib import Path
        with open(Path(__file__).parent.parent.parent / "sd_lora_keys.txt", "w") as lfile:
            for key in lora:
                lfile.write(f"{key}:\t{lora[key].size()}\n")

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        #return (model_lora, clip_lora)
        return (model, clip)
        
            