import os
from pathlib import Path
import time
import json
from typing import Callable

import folder_paths
from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher
from comfy.model_management import xformers_enabled


class BetaSchedules:
    SQRT_LINEAR = "sqrt_linear (AnimateDiff)"
    LINEAR = "linear (default)"
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


class Folders:
    MODELS = "models"


# create and handle directories for models
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../models"))

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

folder_names_and_paths = {}
folder_names_and_paths[Folders.MODELS] = ([MODEL_DIR], folder_paths.supported_pt_extensions)

filename_list_cache = {}

# Load config settings
ADE_DIR = Path(__file__).parent.parent
ADE_CONFIG_FILE = ADE_DIR / "ade_config.json"

class ADE_Settings:
    USE_XFORMERS_IN_VERSATILE_ATTENTION = "use_xformers_in_VersatileAttention"

# Create ADE config if not present
ABS_CONFIG = {
    ADE_Settings.USE_XFORMERS_IN_VERSATILE_ATTENTION: True
}
if not ADE_CONFIG_FILE.exists():
    with ADE_CONFIG_FILE.open("w") as f:
        json.dumps(ABS_CONFIG, indent=4)
# otherwise, load it and use values
else:
    loaded_values: dict = None
    with ADE_CONFIG_FILE.open("r") as f:
        loaded_values = json.load(f)
    if loaded_values is not None:
        for key, value in loaded_values.items():
            if key in ABS_CONFIG:
                ABS_CONFIG[key] = value
        


#Register video_formats folder
folder_paths.folder_names_and_paths["video_formats"] = (
    [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video_formats"),
    ],
    [".json"]
)

def get_filename_list_(folder_name):
    global folder_names_and_paths
    output_list = set()
    folders = folder_names_and_paths[folder_name]
    output_folders = {}
    for x in folders[0]:
        files, folders_all = folder_paths.recursive_search(x)
        output_list.update(folder_paths.filter_files_extensions(files, folders[1]))  # folders[1] is extensions
        output_folders = {**output_folders, **folders_all}

    return (sorted(list(output_list)), output_folders, time.perf_counter())


def cached_filename_list_(folder_name):
    global filename_list_cache
    global folder_names_and_paths
    if folder_name not in filename_list_cache:
        return None
    out = filename_list_cache[folder_name]
    if time.perf_counter() < (out[2] + 0.5):
        return out
    for x in out[1]:
        time_modified = out[1][x]
        folder = x
        if os.path.getmtime(folder) != time_modified:
            return None
        
    folders = folder_names_and_paths[folder_name]
    for x in folders[0]:
        if os.path.isdir(x):
            if not x in out[1]:
                return None
            
    return out


def get_filename_list(folder_name):
    out = cached_filename_list_(folder_name)
    if out is None:
        out = get_filename_list_(folder_name)
        global filename_list_cache
        filename_list_cache[folder_name] = out
    return list(out[0])


def get_folder_path(folder_name):
    return folder_names_and_paths.get(folder_name, ([""],set()))[0][0]


def get_full_path(folder_name, filename):
    global folder_names_and_paths
    if folder_name not in folder_names_and_paths:
        return None
    folders = folder_names_and_paths[folder_name]
    filename = os.path.relpath(os.path.join("/", filename), "/")
    for path in folders[0]:
        full_path = os.path.join(path, filename)
        if os.path.isfile(full_path):
            return full_path
        
    return None


def get_available_models():
    return get_filename_list(Folders.MODELS)


def raise_if_not_checkpoint_sd1_5(model: ModelPatcher):
    model_type = type(model.model)
    if model_type != BaseModel:
        raise ValueError(f"For AnimateDiff, SD Checkpoint (model) is expected to be SD1.5-based (BaseModel), but was: {model_type.__name__}")


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
                    raise RuntimeError(f"An xformers bug was encountered in AnimateDiff - to run your workflow, \
                                       disable xformers for AD only by going to '{ADE_CONFIG_FILE}', and set \
                                        '{ADE_Settings.USE_XFORMERS_IN_VERSATILE_ATTENTION}' to false. Reboot ComfyUI, \
                                            and then AD will use the next-best available attention method.")
                raise
        return wrapped_function
