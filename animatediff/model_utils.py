import os
import time
from huggingface_hub import hf_hub_download

import folder_paths

HF_REPO = "guoyww/animatediff"
MODEL_FILES = ["mm_sd_v14.ckpt", "mm_sd_v15.ckpt"]


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


class Folders:
    MODELS = "models"


# create and handle directories for models
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../models"))

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

folder_names_and_paths = {}
folder_names_and_paths[Folders.MODELS] = ([MODEL_DIR], folder_paths.supported_ckpt_extensions)

filename_list_cache = {}


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


def download(model_file=MODEL_FILES[-1]):
    if not os.path.exists(os.path.join(MODEL_DIR, model_file)):
        hf_hub_download(
            HF_REPO,
            model_file,
            cache_dir=MODEL_DIR,
            force_download=True,
            force_filename=model_file,
        )
