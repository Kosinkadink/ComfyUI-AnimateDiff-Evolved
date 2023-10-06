import hashlib
import os
from pathlib import Path
import time
import json
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

import folder_paths
from comfy.model_base import SDXL, BaseModel
from comfy.model_patcher import ModelPatcher
from comfy.model_management import xformers_enabled


class IsChangedHelper:
    def __init__(self):
        self.val = 0
    
    def no_change(self):
        return self.val
    
    def change(self):
        self.val = (self.val + 1) % 100


class BetaSchedules:
    SQRT_LINEAR = "sqrt_linear (AnimateDiff)"
    LINEAR = "linear (HotshotXL/default)"
    SQRT = "sqrt"
    COSINE = "cosine"
    SQUAREDCOS_CAP_V2 = "squaredcos_cap_v2"

    ALIAS_LIST = [SQRT_LINEAR, LINEAR, SQRT, COSINE, SQUAREDCOS_CAP_V2]

    ALIAS_MAP = {
        SQRT_LINEAR: "sqrt_linear",
        LINEAR: "linear",
        SQRT: "sqrt",
        COSINE: "cosine",
        SQUAREDCOS_CAP_V2: "squaredcos_cap_v2",
    }

    @classmethod
    def to_name(cls, alias: str):
        return cls.ALIAS_MAP[alias]

    @staticmethod
    def get_alias_list_with_first_element(first_element: str):
        new_list = BetaSchedules.ALIAS_LIST.copy()
        element_index = new_list.index(first_element)
        new_list[0], new_list[element_index] = new_list[element_index], new_list[0]
        return new_list


class BetaScheduleCache:
    def __init__(self, model: ModelPatcher): 
        self.betas = model.model.betas.cpu().clone().detach()
        self.linear_start = model.model.linear_start
        self.linear_end = model.model.linear_end

    def use_cached_beta_schedule_and_clean(self, model: ModelPatcher):
        model.model.register_schedule(given_betas=self.betas.clone().detach(), linear_start=self.linear_start, linear_end=self.linear_end)
        self.clean()

    def clean(self):
        self.betas = None
        self.linear_start = None
        self.linear_end = None


class Folders:
    ANIMATEDIFF_MODELS = "AnimateDiffEvolved_Models"
    MOTION_LORA = "AnimateDiffMotion_LoRA"


# register motion models folder(s)
folder_paths.folder_names_and_paths[Folders.ANIMATEDIFF_MODELS] = (
    [
        str(Path(__file__).parent.parent / "models")
    ],
    folder_paths.supported_pt_extensions
)

# register motion LoRA folder(s)
folder_paths.folder_names_and_paths[Folders.MOTION_LORA] = (
    [
        str(Path(__file__).parent.parent / "motion_lora")
    ],
    folder_paths.supported_pt_extensions
)


#Register video_formats folder
folder_paths.folder_names_and_paths["video_formats"] = (
    [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video_formats"),
    ],
    [".json"]
)


def get_available_motion_models():
    return folder_paths.get_filename_list(Folders.ANIMATEDIFF_MODELS)


def get_motion_model_path(model_name: str):
    return folder_paths.get_full_path(Folders.ANIMATEDIFF_MODELS, model_name)


def get_available_motion_loras():
    return folder_paths.get_filename_list(Folders.MOTION_LORA)


def get_motion_lora_path(lora_name: str):
    return folder_paths.get_full_path(Folders.MOTION_LORA, lora_name)


# modified from https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
def calculate_file_hash(filename: str, hash_every_n: int = 50):
    h = hashlib.sha256()
    b = bytearray(1024*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        i = 0
        # don't hash entire file, only portions of it
        while n := f.readinto(mv):
            if i%hash_every_n == 0:
                h.update(mv[:n])
            i += 1
    return h.hexdigest()


def calculate_model_hash(model: ModelPatcher):
    unet = model.model.diff
    t = unet.input_blocks[1]
    m = hashlib.sha256()
    for buf in t.buffers():
        m.update(buf.cpu().numpy().view(np.uint8))
    return m.hexdigest()


class ModelTypesSD:
    SD1_5 = "sd1_5"
    SDXL = "sdxl"


def get_sd_model_type(model: ModelPatcher) -> str:
    if model is None:
        return None
    if is_checkpoint_sd1_5(model):
        return ModelTypesSD.SD1_5
    elif is_checkpoint_sdxl(model):
        return ModelTypesSD.SDXL
    return False


def is_checkpoint_sd1_5(model: ModelPatcher):
    if model is None:
        return False
    model_type = type(model.model)
    return model_type == BaseModel

def is_checkpoint_sdxl(model: ModelPatcher):
    if model is None:
        return False
    model_type = type(model.model)
    return model_type == SDXL


def raise_if_not_checkpoint_sd1_5(model: ModelPatcher):
    if not is_checkpoint_sd1_5(model):
        raise ValueError(f"For AnimateDiff, SD Checkpoint (model) is expected to be SD1.5-based (BaseModel), but was: {type(model.model).__name__}")


# TODO: remove this filth when xformers bug gets fixed in future xformers version
def wrap_function_to_inject_xformers_bug_info(function_to_wrap: Callable) -> Callable:
    if not xformers_enabled:
        return function_to_wrap
    else:
        def wrapped_function(*args, **kwargs):
            try:
                return function_to_wrap(*args, **kwargs)
            except RuntimeError as e:
                if str(e).startswith("CUDA error: invalid configuration argument"):
                    raise RuntimeError(f"An xformers bug was encountered in AnimateDiff - this is unexpected, \
                                       report this to Kosinkadink/ComfyUI-AnimateDiff-Evolved repo as an issue, \
                                       and a workaround for now is to run ComfyUI with the --disable-xformers argument.")
                raise
        return wrapped_function


# TODO: possibly add configuration file in future when needed?
# # Load config settings
# ADE_DIR = Path(__file__).parent.parent
# ADE_CONFIG_FILE = ADE_DIR / "ade_config.json"

# class ADE_Settings:
#     USE_XFORMERS_IN_VERSATILE_ATTENTION = "use_xformers_in_VersatileAttention"

# # Create ADE config if not present
# ABS_CONFIG = {
#     ADE_Settings.USE_XFORMERS_IN_VERSATILE_ATTENTION: True
# }
# if not ADE_CONFIG_FILE.exists():
#     with ADE_CONFIG_FILE.open("w") as f:
#         json.dumps(ABS_CONFIG, indent=4)
# # otherwise, load it and use values
# else:
#     loaded_values: dict = None
#     with ADE_CONFIG_FILE.open("r") as f:
#         loaded_values = json.load(f)
#     if loaded_values is not None:
#         for key, value in loaded_values.items():
#             if key in ABS_CONFIG:
#                 ABS_CONFIG[key] = value
