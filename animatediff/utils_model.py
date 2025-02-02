import hashlib
from pathlib import Path
from typing import Callable, Union
from collections.abc import Iterable
from time import time
import copy

from torch import Tensor
import torch
import numpy as np

import folder_paths
from comfy.model_base import SD21UNCLIP, SDXL, BaseModel, SDXLRefiner, SVD_img2vid, model_sampling, ModelType
from comfy.model_management import xformers_enabled
from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE
from comfy.utils import ProgressBar

import comfy.model_sampling
import comfy_extras.nodes_model_advanced

from .logger import logger

BIGMIN = -(2**53-1)
BIGMAX = (2**53-1)
BIGMAX_TENSOR = torch.tensor(9999999999.9)

MAX_RESOLUTION = 16384  # mirrors ComfyUI's nodes.py MAX_RESOLUTION

class MachineState:
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"
    OFF = "off"


def vae_encode_raw_dynamic_batched(vae: VAE, pixels: Tensor, max_batch=16, min_batch=1, max_size=512*512, show_pbar=False):
    b, h, w, c = pixels.shape
    actual_size = h*w
    actual_batch_size = int(max(min_batch, min(max_batch, max_batch // max((actual_size / max_size), 1.0))))
    return vae_encode_raw_batched(vae=vae, pixels=pixels, per_batch=actual_batch_size, show_pbar=show_pbar)


def vae_decode_raw_dynamic_batched(vae: VAE, latents: Tensor, max_batch=16, min_batch=1, max_size=512*512, show_pbar=False):
    b, c, h, w = latents.shape
    actual_size = (h*vae.downscale_ratio)*(w*vae.downscale_ratio)
    actual_batch_size = int(max(min_batch, min(max_batch, max_batch // max((actual_size / max_size), 1.0))))
    return vae_decode_raw_batched(vae=vae, latents=latents, per_batch=actual_batch_size, show_pbar=show_pbar)


def vae_encode_raw_batched(vae: VAE, pixels: Tensor, per_batch=16, show_pbar=False):
    encoded = []
    pbar = None
    if show_pbar:
        pbar = ProgressBar(pixels.shape[0])
    for start_idx in range(0, pixels.shape[0], per_batch):
        sub_encoded = vae.encode(pixels[start_idx:start_idx+per_batch][:,:,:,:3])
        encoded.append(sub_encoded)
        if pbar is not None:
            pbar.update(sub_encoded.shape[0])
    return torch.cat(encoded, dim=0)


def vae_decode_raw_batched(vae: VAE, latents: Tensor, per_batch=16, show_pbar=False):
    decoded = []
    pbar = None
    if show_pbar:
        pbar = ProgressBar(latents.shape[0])
    for start_idx in range(0, latents.shape[0], per_batch):
        sub_decoded = vae.decode(latents[start_idx:start_idx+per_batch])
        decoded.append(sub_decoded)
        if pbar is not None:
            pbar.update(sub_decoded.shape[0])
    return torch.cat(decoded, dim=0)


class ModelSamplingConfig:
    def __init__(self, beta_schedule: str, linear_start: float=None, linear_end: float=None, given_betas: Tensor=None, timesteps: int=None):
        self.sampling_settings = {"beta_schedule": beta_schedule}
        if linear_start is not None:
            self.sampling_settings["linear_start"] = linear_start
        if linear_end is not None:
            self.sampling_settings["linear_end"] = linear_end
        if given_betas is not None:
            self.sampling_settings["given_betas"] = given_betas
        if timesteps is not None:
            self.sampling_settings["timesteps"] = timesteps


class ModelSamplingType:
    EPS = "eps"
    V_PREDICTION = "v_prediction"
    LCM = "lcm"

    _NON_LCM_LIST = [EPS, V_PREDICTION]
    _FULL_LIST = [EPS, V_PREDICTION, LCM]

    MAP = {
        EPS: ModelType.EPS,
        V_PREDICTION: ModelType.V_PREDICTION,
        LCM: comfy_extras.nodes_model_advanced.LCM,
    }

    @classmethod
    def from_alias(cls, alias: str):
        return cls.MAP[alias]


def factory_model_sampling_discrete_distilled(original_timesteps=50):
    class ModelSamplingDiscreteDistilledEvolved(comfy_extras.nodes_model_advanced.ModelSamplingDiscreteDistilled):
        def __init__(self, *args, **kwargs):
            self.original_timesteps = original_timesteps  # normal LCM has 50
            super().__init__(*args, **kwargs)
    return ModelSamplingDiscreteDistilledEvolved


# based on code in comfy_extras/nodes_model_advanced.py
def evolved_model_sampling(model_config: ModelSamplingConfig, model_type: ModelType, alias: str, original_timesteps: Union[int, None]=None):
    # if LCM, need to handle manually
    if BetaSchedules.is_lcm(alias) or original_timesteps is not None:
        sampling_type = comfy_extras.nodes_model_advanced.LCM
        if original_timesteps is not None:
            sampling_base = factory_model_sampling_discrete_distilled(original_timesteps=original_timesteps)
        elif alias == BetaSchedules.LCM_100:
            sampling_base = factory_model_sampling_discrete_distilled(original_timesteps=100)
        elif alias == BetaSchedules.LCM_25:
            sampling_base = factory_model_sampling_discrete_distilled(original_timesteps=25)
        else:
            sampling_base = comfy_extras.nodes_model_advanced.ModelSamplingDiscreteDistilled
        class ModelSamplingAdvancedEvolved(sampling_base, sampling_type):
            pass
        # NOTE: if I want to support zsnr, this is where I would add that code
        return ModelSamplingAdvancedEvolved(model_config)
    # otherwise, use vanilla model_sampling function
    ms = model_sampling(model_config, model_type)
    if "given_betas" in model_config.sampling_settings:
        beta_schedule = model_config.sampling_settings.get("beta_schedule", "linear")
        linear_start = model_config.sampling_settings.get("linear_start", 0.00085)
        linear_end = model_config.sampling_settings.get("linear_end", 0.012)
        timesteps = model_config.sampling_settings.get("timesteps", 1000)
        given_betas = model_config.sampling_settings.get("given_betas", None)
        ms._register_schedule(given_betas=given_betas, beta_schedule=beta_schedule,
                              timesteps=timesteps, linear_start=linear_start, linear_end=linear_end)
    return ms


class BetaSchedules:
    AUTOSELECT = "autoselect"
    SQRT_LINEAR = "sqrt_linear (AnimateDiff)"
    LINEAR_ADXL = "linear (AnimateDiff-SDXL)"
    LINEAR = "linear (HotshotXL/default)"
    AVG_LINEAR_SQRT_LINEAR = "avg(sqrt_linear,linear)"
    LCM_AVG_LINEAR_SQRT_LINEAR = "lcm avg(sqrt_linear,linear)"
    LCM = "lcm"
    LCM_100 = "lcm[100_ots]"
    LCM_25 = "lcm[25_ots]"
    LCM_SQRT_LINEAR = "lcm >> sqrt_linear"
    USE_EXISTING = "use existing"
    SQRT = "sqrt"
    COSINE = "cosine"
    SQUAREDCOS_CAP_V2 = "squaredcos_cap_v2"
    RAW_LINEAR = "linear"
    RAW_SQRT_LINEAR = "sqrt_linear"

    RAW_BETA_SCHEDULE_LIST = [RAW_LINEAR, RAW_SQRT_LINEAR, SQRT, COSINE, SQUAREDCOS_CAP_V2]

    ALIAS_LCM_LIST = [LCM, LCM_100, LCM_25, LCM_SQRT_LINEAR]

    ALIAS_ACTIVE_LIST = [SQRT_LINEAR, LINEAR_ADXL, LINEAR, AVG_LINEAR_SQRT_LINEAR, LCM_AVG_LINEAR_SQRT_LINEAR, LCM, LCM_100, LCM_SQRT_LINEAR, # LCM_25 is purposely omitted
                  SQRT, COSINE, SQUAREDCOS_CAP_V2]

    ALIAS_LIST = [AUTOSELECT, USE_EXISTING] + ALIAS_ACTIVE_LIST

    

    ALIAS_MAP = {
        SQRT_LINEAR: "sqrt_linear",
        LINEAR_ADXL: "linear", # also linear, but has different linear_end (0.020)
        LINEAR: "linear",
        LCM_100: "linear",  # distilled, 100 original timesteps
        LCM_25: "linear",  # distilled, 25 original timesteps
        LCM: "linear",  # distilled
        LCM_SQRT_LINEAR: "sqrt_linear", # distilled, sqrt_linear
        SQRT: "sqrt",
        COSINE: "cosine",
        SQUAREDCOS_CAP_V2: "squaredcos_cap_v2",
        RAW_LINEAR: "linear",
        RAW_SQRT_LINEAR: "sqrt_linear"
    }

    @classmethod
    def is_lcm(cls, alias: str):
        return alias in cls.ALIAS_LCM_LIST

    @classmethod
    def to_name(cls, alias: str):
        return cls.ALIAS_MAP[alias]
    
    @classmethod
    def to_config(cls, alias: str) -> ModelSamplingConfig:
        linear_start = None
        linear_end = None
        if alias == cls.LINEAR_ADXL:
            # uses linear_end=0.020
            linear_end = 0.020
        return ModelSamplingConfig(cls.to_name(alias), linear_start=linear_start, linear_end=linear_end)
    
    @classmethod
    def _to_model_sampling(cls, alias: str, model_type: ModelType, config_override: Union[ModelSamplingConfig,None]=None, original_timesteps: Union[int,None]=None):
        if alias == cls.USE_EXISTING:
            return None
        elif config_override != None:
            return evolved_model_sampling(config_override, model_type=model_type, alias=alias, original_timesteps=original_timesteps)
        elif alias == cls.AVG_LINEAR_SQRT_LINEAR:
            ms_linear = evolved_model_sampling(cls.to_config(cls.LINEAR), model_type=model_type, alias=cls.LINEAR)
            ms_sqrt_linear = evolved_model_sampling(cls.to_config(cls.SQRT_LINEAR), model_type=model_type, alias=cls.SQRT_LINEAR)
            avg_sigmas = (ms_linear.sigmas + ms_sqrt_linear.sigmas) / 2
            ms_linear.set_sigmas(avg_sigmas)
            return ms_linear
        elif alias == cls.LCM_AVG_LINEAR_SQRT_LINEAR:
            ms_linear = evolved_model_sampling(cls.to_config(cls.LCM), model_type=model_type, alias=cls.LCM)
            ms_sqrt_linear = evolved_model_sampling(cls.to_config(cls.LCM_SQRT_LINEAR), model_type=model_type, alias=cls.LCM_SQRT_LINEAR)
            avg_sigmas = (ms_linear.sigmas + ms_sqrt_linear.sigmas) / 2
            ms_linear.set_sigmas(avg_sigmas)
            return ms_linear
            # average out the sigmas
        ms_obj = evolved_model_sampling(cls.to_config(alias), model_type=model_type, alias=alias, original_timesteps=original_timesteps)
        return ms_obj

    @classmethod
    def to_model_sampling(cls, alias: str, model: ModelPatcher):
        return cls._to_model_sampling(alias=alias, model_type=model.model.model_type)

    @staticmethod
    def get_alias_list_with_first_element(first_element: str):
        new_list = BetaSchedules.ALIAS_LIST.copy()
        element_index = new_list.index(first_element)
        new_list[0], new_list[element_index] = new_list[element_index], new_list[0]
        return new_list


class SigmaSchedule:
    def __init__(self, model_sampling: comfy.model_sampling.ModelSamplingDiscrete, model_type: ModelType):
        self.model_sampling = model_sampling
        #self.config = config
        self.model_type = model_type
        self.original_timesteps = getattr(self.model_sampling, "original_timesteps", None)
    
    def is_lcm(self):
        return self.original_timesteps is not None

    def total_sigmas(self):
        return len(self.model_sampling.sigmas)
    
    def clone(self) -> 'SigmaSchedule':
        new_model_sampling = copy.deepcopy(self.model_sampling)
        #new_config = copy.deepcopy(self.config)
        return SigmaSchedule(model_sampling=new_model_sampling, model_type=self.model_type)

    # def clone(self):
    #     pass

    @staticmethod
    def apply_zsnr(new_model_sampling: comfy.model_sampling.ModelSamplingDiscrete):
        new_model_sampling.set_sigmas(comfy_extras.nodes_model_advanced.rescale_zero_terminal_snr_sigmas(new_model_sampling.sigmas))

    # def get_lcmified(self, original_timesteps=50, zsnr=False) -> 'SigmaSchedule':
    #     new_model_sampling = evolved_model_sampling(model_config=self.config, model_type=self.model_type, alias=None, original_timesteps=original_timesteps)
    #     if zsnr:
    #         new_model_sampling.set_sigmas(comfy_extras.nodes_model_advanced.rescale_zero_terminal_snr_sigmas(new_model_sampling.sigmas))
    #     return SigmaSchedule(model_sampling=new_model_sampling, config=self.config, model_type=self.model_type, is_lcm=True)
        

class InterpolationMethod:
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"

    _LIST = [LINEAR, EASE_IN, EASE_OUT, EASE_IN_OUT]

    @classmethod
    def get_weights(cls, num_from: float, num_to: float, length: int, method: str, reverse=False):
        diff = num_to - num_from
        if method == cls.LINEAR:
            weights = torch.linspace(num_from, num_to, length)
        elif method == cls.EASE_IN:
            index = torch.linspace(0, 1, length)
            weights = diff * np.power(index, 2) + num_from
        elif method == cls.EASE_OUT:
            index = torch.linspace(0, 1, length)
            weights = diff * (1 - np.power(1 - index, 2)) + num_from
        elif method == cls.EASE_IN_OUT:
            index = torch.linspace(0, 1, length)
            weights = diff * ((1 - np.cos(index * np.pi)) / 2) + num_from
        else:
            raise ValueError(f"Unrecognized interpolation method '{method}'.")
        if reverse:
            weights = weights.flip(dims=(0,))
        return weights


class ScaleMethods:
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    AREA = "area"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"

    _LIST_IMAGE = [NEAREST_EXACT, BILINEAR, AREA, BICUBIC, LANCZOS]


class CropMethods:
    DISABLED = "disabled"
    CENTER = "center"

    _LIST = [DISABLED, CENTER]


class Folders:
    ANIMATEDIFF_MODELS = "animatediff_models"
    MOTION_LORA = "animatediff_motion_lora"
    VIDEO_FORMATS = "animatediff_video_formats"


def add_extension_to_folder_path(folder_name: str, extensions: Union[str, list[str]]):
    if folder_name in folder_paths.folder_names_and_paths:
        if isinstance(extensions, str):
            folder_paths.folder_names_and_paths[folder_name][1].add(extensions)
        elif isinstance(extensions, Iterable):
            for ext in extensions:
                folder_paths.folder_names_and_paths[folder_name][1].add(ext) 


def try_mkdir(full_path: str):
    try:
        Path(full_path).mkdir()
    except Exception:
        pass


# register motion models folder(s)
folder_paths.add_model_folder_path(Folders.ANIMATEDIFF_MODELS, str(Path(__file__).parent.parent / "models"))
folder_paths.add_model_folder_path(Folders.ANIMATEDIFF_MODELS, str(Path(folder_paths.models_dir) / Folders.ANIMATEDIFF_MODELS))
add_extension_to_folder_path(Folders.ANIMATEDIFF_MODELS, folder_paths.supported_pt_extensions)
try_mkdir(str(Path(folder_paths.models_dir) / Folders.ANIMATEDIFF_MODELS))

# register motion LoRA folder(s)
folder_paths.add_model_folder_path(Folders.MOTION_LORA, str(Path(__file__).parent.parent / "motion_lora"))
folder_paths.add_model_folder_path(Folders.MOTION_LORA, str(Path(folder_paths.models_dir) / Folders.MOTION_LORA))
add_extension_to_folder_path(Folders.MOTION_LORA, folder_paths.supported_pt_extensions)
try_mkdir(str(Path(folder_paths.models_dir) / Folders.MOTION_LORA))

# register video_formats folder
folder_paths.add_model_folder_path(Folders.VIDEO_FORMATS, str(Path(__file__).parent.parent / "video_formats"))
add_extension_to_folder_path(Folders.VIDEO_FORMATS, ".json")


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


def strip_path(path):
    # removes whitespace and single quotes from either end of string, if present
    path = path.strip()
    if path.startswith("\""):
        path = path[1:]
    if path.endswith("\""):
        path = path[:-1]
    return path


class ModelTypeSD:
    SD1_5 = "SD1.5"
    SD2_1 = "SD2.1"
    SDXL = "SDXL"
    SDXL_REFINER = "SDXL_Refiner"
    SVD = "SVD"

    _LIST = [SD1_5, SD2_1, SDXL, SDXL_REFINER, SVD]


def get_sd_model_type(model: ModelPatcher) -> str:
    if model is None:
        return None
    type_str = str(type(model.model).__name__)
    # instructpix2pix models should be allowed to work with AD
    if type(model.model) == BaseModel or type_str == "SD15_instructpix2pix":
        return ModelTypeSD.SD1_5
    elif type(model.model) == SDXL or type_str == "SDXL_instructpix2pix":
        return ModelTypeSD.SDXL
    elif type(model.model) == SD21UNCLIP:
        return ModelTypeSD.SD2_1
    elif type(model.model) == SDXLRefiner:
        return ModelTypeSD.SDXL_REFINER
    elif type(model.model) == SVD_img2vid:
        return ModelTypeSD.SVD
    else:
        return type_str

def is_checkpoint_sd1_5(model: ModelPatcher):
    return False if model is None else type(model.model) == BaseModel

def is_checkpoint_sdxl(model: ModelPatcher):
    return False if model is None else type(model.model) == SDXL


def raise_if_not_checkpoint_sd1_5(model: ModelPatcher):
    if not is_checkpoint_sd1_5(model):
        raise ValueError(f"For AnimateDiff, SD Checkpoint (model) is expected to be SD1.5-based (BaseModel), but was: {type(model.model).__name__}")


# TODO: remove this filth when xformers bug gets fixed in future xformers version
# NOTE: avoid using this for now to avoid false positives with pytorch or non-AD stuff like SVD
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


class Timer(object):
    __slots__ = ("start_time", "end_time")

    def __init__(self) -> None:
        self.start_time = 0.0
        self.end_time = 0.0

    def start(self) -> None:
        self.start_time = time()

    def update(self) -> None:
        self.start()

    def stop(self) -> float:
        self.end_time = time()
        return self.get_time_diff()

    def get_time_diff(self) -> float:
        return self.end_time - self.start_time

    def get_time_current(self) -> float:
        return time() - self.start_time


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
