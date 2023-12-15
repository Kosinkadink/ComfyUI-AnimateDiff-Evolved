import copy

from einops import rearrange
from torch import Tensor
import torch.nn.functional as F
import torch

import comfy.model_management
import comfy.utils
from comfy.model_patcher import ModelPatcher

from .motion_module_ad import AnimateDiffModel, has_mid_block, normalize_ad_state_dict
from .logger import logger
from .motion_utils import MotionCompatibilityError, NoiseType, normalize_min_max
from .motion_lora import MotionLoraInfo, MotionLoraList
from .model_utils import get_motion_lora_path, get_motion_model_path, get_sd_model_type


# some motion_model casts here might fail if model becomes metatensor or is not castable;
# should not really matter if it fails, so ignore raised Exceptions
class ModelPatcherAndInjector(ModelPatcher):
    def __init__(self, m: ModelPatcher):
        # replicate ModelPatcher.clone() to initialize ModelPatcherAndInjector
        super().__init__(m.model, m.load_device, m.offload_device, m.size, m.current_device, weight_inplace_update=m.weight_inplace_update)
        self.patches = {}
        for k in m.patches:
            self.patches[k] = m.patches[k][:]

        self.object_patches = m.object_patches.copy()
        self.model_options = copy.deepcopy(m.model_options)
        self.model_keys = m.model_keys

        # injection stuff
        self.motion_injection_params: InjectionParams = None
        self.motion_model: MotionModelPatcher = None
        self.motion_model_sampling = None
    
    def model_patches_to(self, device):
        super().model_patches_to(device)
        if self.motion_model is not None:
            try:
                self.motion_model.model.to(device)
            except Exception:
                pass

    def patch_model(self, device_to=None):
        # first, perform model patching
        patched_model = super().patch_model(device_to)
        # finally, perform motion model injection
        self.inject_model(device_to=device_to)
        return patched_model

    def unpatch_model(self, device_to=None):
        # first, eject motion model from unet
        self.eject_model(device_to=device_to)
        # finally, do normal model unpatching
        return super().unpatch_model(device_to)

    def inject_model(self, device_to=None):
        if self.motion_model is not None:
            self.motion_model.model.eject(self)
            self.motion_model.model.inject(self)
            try:
                self.motion_model.model.to(device_to)
            except Exception:
                pass

    def eject_model(self, device_to=None):
        if self.motion_model is not None:
            self.motion_model.model.eject(self)
            try:
                self.motion_model.model.to(device_to)
            except Exception:
                pass

    def clone(self):
        cloned = ModelPatcherAndInjector(self)
        cloned.motion_model = self.motion_model
        cloned.motion_injection_params = self.motion_injection_params
        cloned.motion_model_sampling = self.motion_model_sampling
        return cloned


class MotionModelPatcher(ModelPatcher):
    # Mostly here so that type hints work in IDEs
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: AnimateDiffModel = self.model

    def patch_model(self, *args, **kwargs):
        # patch as normal, but prepare_weights so that lowvram meta device works properly
        patched_model = super().patch_model(*args, **kwargs)
        self.prepare_weights()
        return patched_model

    def prepare_weights(self):
        # in case lowvram is active and meta device is used, need to convert weights
        # otherwise, will get exceptions thrown related to meta device
        state_dict = self.model.state_dict()
        for key in state_dict:
            weight = comfy.model_management.resolve_lowvram_weight(state_dict[key], self.model, key)
            try:
                comfy.utils.set_attr(self.model, key, weight)
            except Exception:
                pass
    
    def pre_run(self):
        # just in case, prepare_weights before every run
        self.prepare_weights()

    def cleanup(self):
        if self.model is not None:
            self.model.cleanup()


def get_vanilla_model_patcher(m: ModelPatcher) -> ModelPatcher:
    model = ModelPatcher(m.model, m.load_device, m.offload_device, m.size, m.current_device, weight_inplace_update=m.weight_inplace_update)
    model.patches = {}
    for k in m.patches:
        model.patches[k] = m.patches[k][:]

    model.object_patches = m.object_patches.copy()
    model.model_options = copy.deepcopy(m.model_options)
    model.model_keys = m.model_keys
    return model

# adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/utils/convert_lora_safetensor_to_diffusers.py
# Example LoRA keys:
#   down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0.processor.to_q_lora.down.weight
#   down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0.processor.to_q_lora.up.weight
#
# Example model keys: 
#   down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0.to_q.weight
#
def load_motion_lora_as_patches(motion_model: MotionModelPatcher, lora: MotionLoraInfo) -> None:
    def get_version(has_midblock: bool):
        return "v2" if has_midblock else "v1"

    lora_path = get_motion_lora_path(lora.name)
    logger.info(f"Loading motion LoRA {lora.name}")
    state_dict = comfy.utils.load_torch_file(lora_path)

    # remove all non-temporal keys (in case model has extra stuff in it)
    for key in list(state_dict.keys()):
        if "temporal" not in key:
            del state_dict[key]
    if len(state_dict) == 0:
        raise ValueError(f"'{lora.name}' contains no temporal keys; it is not a valid motion LoRA!")

    model_has_midblock = motion_model.model.mid_block != None
    lora_has_midblock = has_mid_block(state_dict)
    logger.info(f"Applying a {get_version(lora_has_midblock)} LoRA ({lora.name}) to a { motion_model.model.mm_info.mm_version} motion model.")

    patches = {}
    # convert lora state dict to one that matches motion_module keys and tensors
    for key in state_dict:
        # if motion_module doesn't have a midblock, skip mid_block entries
        if not model_has_midblock:
            if "mid_block" in key: continue
        # only process lora down key (we will process up at the same time as down)
        if "up." in key: continue

        # get up key version of down key
        up_key = key.replace(".down.", ".up.")

        # adapt key to match motion_module key format - remove 'processor.', '_lora', 'down.', and 'up.'
        model_key = key.replace("processor.", "").replace("_lora", "").replace("down.", "").replace("up.", "")
        # motion_module keys have a '0.' after all 'to_out.' weight keys
        model_key = model_key.replace("to_out.", "to_out.0.")
        
        weight_down = state_dict[key]
        weight_up = state_dict[up_key]
        # actual weights obtained by matrix multiplication of up and down weights
        # save as a tuple, so that (Motion)ModelPatcher's calculate_weight function detects len==1, applying it correctly
        patches[model_key] = (torch.mm(weight_up, weight_down),)
    del state_dict
    # add patches to motion ModelPatcher
    motion_model.add_patches(patches=patches, strength_patch=lora.strength)


def load_motion_module(model_name: str, model: ModelPatcher, motion_lora: MotionLoraList = None, motion_model_settings: 'MotionModelSettings' = None) -> MotionModelPatcher:
    model_path = get_motion_model_path(model_name)
    logger.info(f"Loading motion module {model_name}")
    mm_state_dict = comfy.utils.load_torch_file(model_path, safe_load=True)
    # TODO: check for empty state dict?
    # get normalized state_dict and motion model info
    mm_state_dict, mm_info = normalize_ad_state_dict(mm_state_dict=mm_state_dict, mm_name=model_name)
    # check that motion model is compatible with sd model
    model_sd_type = get_sd_model_type(model)
    if model_sd_type != mm_info.sd_type:
        raise MotionCompatibilityError(f"Motion module '{mm_info.mm_name}' is intended for {mm_info.sd_type} models, " \
                                       + f"but the provided model is type {model_sd_type}.")
    # apply motion model settings
    mm_state_dict = apply_mm_settings(model_dict=mm_state_dict, mm_settings=motion_model_settings)
    # initialize AnimateDiffModelWrapper
    ad_wrapper = AnimateDiffModel(mm_state_dict=mm_state_dict, mm_info=mm_info)
    ad_wrapper.to(model.model_dtype())
    ad_wrapper.to(model.offload_device)
    load_result = ad_wrapper.load_state_dict(mm_state_dict)
    # TODO: report load_result of motion_module loading?
    # wrap motion_module into a ModelPatcher, to allow motion lora patches
    motion_model = MotionModelPatcher(model=ad_wrapper, load_device=model.load_device, offload_device=model.offload_device)
    # load motion_lora, if present
    if motion_lora is not None:
        for lora in motion_lora.loras:
            load_motion_lora_as_patches(motion_model, lora)
    return motion_model


def interpolate_pe_to_length(model_dict: dict[str, Tensor], key: str, new_length: int):
    pe_shape = model_dict[key].shape
    temp_pe = rearrange(model_dict[key], "(t b) f d -> t b f d", t=1)
    temp_pe = F.interpolate(temp_pe, size=(new_length, pe_shape[-1]), mode="bilinear")
    temp_pe = rearrange(temp_pe, "t b f d -> (t b) f d", t=1)
    model_dict[key] = temp_pe
    del temp_pe


def interpolate_pe_to_length_diffs(model_dict: dict[str, Tensor], key: str, new_length: int):
    # TODO: fill out and try out
    pe_shape = model_dict[key].shape
    temp_pe = rearrange(model_dict[key], "(t b) f d -> t b f d", t=1)
    temp_pe = F.interpolate(temp_pe, size=(new_length, pe_shape[-1]), mode="bilinear")
    temp_pe = rearrange(temp_pe, "t b f d -> (t b) f d", t=1)
    model_dict[key] = temp_pe
    del temp_pe


def interpolate_pe_to_length_pingpong(model_dict: dict[str, Tensor], key: str, new_length: int):
    if model_dict[key].shape[1] < new_length:
        temp_pe = model_dict[key]
        flipped_temp_pe = torch.flip(temp_pe[:, 1:-1, :], [1])
        use_flipped = True
        preview_pe = None
        while model_dict[key].shape[1] < new_length:
            preview_pe = model_dict[key]
            model_dict[key] = torch.cat([model_dict[key], flipped_temp_pe if use_flipped else temp_pe], dim=1)
            use_flipped = not use_flipped
        del temp_pe
        del flipped_temp_pe
        del preview_pe
    model_dict[key] = model_dict[key][:, :new_length]


def freeze_mask_of_pe(model_dict: dict[str, Tensor], key: str):
    pe_portion = model_dict[key].shape[2] // 64
    first_pe = model_dict[key][:,:1,:]
    model_dict[key][:,:,pe_portion:] = first_pe[:,:,pe_portion:]
    del first_pe


def freeze_mask_of_attn(model_dict: dict[str, Tensor], key: str):
    attn_portion = model_dict[key].shape[0] // 2
    model_dict[key][:attn_portion,:attn_portion] *= 1.5


def apply_mm_settings(model_dict: dict[str, Tensor], mm_settings: 'MotionModelSettings') -> dict[str, Tensor]:
    if mm_settings is None:
        return model_dict
    if not mm_settings.has_anything_to_apply():
        return model_dict
    for key in model_dict:
        if "attention_blocks" in key:
            if "pos_encoder" in key:
                # apply simple motion pe stretch, if needed
                if mm_settings.has_motion_pe_stretch():
                    new_pe_length = model_dict[key].shape[1] + mm_settings.motion_pe_stretch
                    interpolate_pe_to_length(model_dict, key, new_length=new_pe_length)
                # apply pe_strength, if needed
                if mm_settings.has_pe_strength():
                    model_dict[key] *= mm_settings.pe_strength
                # apply pe_idx_offset, if needed
                if mm_settings.has_initial_pe_idx_offset():
                    model_dict[key] = model_dict[key][:, mm_settings.initial_pe_idx_offset:]
                # apply has_cap_initial_pe_length, if needed
                if mm_settings.has_cap_initial_pe_length():
                    model_dict[key] = model_dict[key][:, :mm_settings.cap_initial_pe_length]
                # apply interpolate_pe_to_length, if needed
                if mm_settings.has_interpolate_pe_to_length():
                    interpolate_pe_to_length(model_dict, key, new_length=mm_settings.interpolate_pe_to_length)
                # apply final_pe_idx_offset, if needed
                if mm_settings.has_final_pe_idx_offset():
                    model_dict[key] = model_dict[key][:, mm_settings.final_pe_idx_offset:]
            else:
                # apply attn_strenth, if needed
                if mm_settings.has_attn_strength():
                    model_dict[key] *= mm_settings.attn_strength
                # apply specific attn_strengths, if needed
                if mm_settings.has_any_attn_sub_strength():
                    if "to_q" in key and mm_settings.has_attn_q_strength():
                        model_dict[key] *= mm_settings.attn_q_strength
                    elif "to_k" in key and mm_settings.has_attn_k_strength():
                        model_dict[key] *= mm_settings.attn_k_strength
                    elif "to_v" in key and mm_settings.has_attn_v_strength():
                        model_dict[key] *= mm_settings.attn_v_strength
                    elif "to_out" in key:
                        if key.strip().endswith("weight") and mm_settings.has_attn_out_weight_strength():
                            model_dict[key] *= mm_settings.attn_out_weight_strength
                        elif key.strip().endswith("bias") and mm_settings.has_attn_out_bias_strength():
                            model_dict[key] *= mm_settings.attn_out_bias_strength
        # apply other strength, if needed
        elif mm_settings.has_other_strength():
            model_dict[key] *= mm_settings.other_strength
    return model_dict


class InjectionParams:
    def __init__(self, video_length: int, unlimited_area_hack: bool, apply_mm_groupnorm_hack: bool, beta_schedule: str, model_name: str,
                 apply_v2_models_properly: bool=False) -> None:
        self.video_length = video_length
        self.full_length = None
        self.unlimited_area_hack = unlimited_area_hack
        self.apply_mm_groupnorm_hack = apply_mm_groupnorm_hack
        self.beta_schedule = beta_schedule
        self.model_name = model_name
        self.apply_v2_models_properly = apply_v2_models_properly
        self.context_length: int = None
        self.context_stride: int = None
        self.context_overlap: int = None
        self.context_schedule: str = None
        self.closed_loop: bool = False
        self.sync_context_to_pe = False
        self.loras: MotionLoraList = None
        self.motion_model_settings = MotionModelSettings()
        self.noise_type: str = NoiseType.DEFAULT
        self.sub_idxs = None  # value should NOT be included in clone, so it will auto reset
    

    def set_context(self, context_length: int, context_stride: int, context_overlap: int, context_schedule: str, closed_loop: bool, sync_context_to_pe: bool=False):
        self.context_length = context_length
        self.context_stride = context_stride
        self.context_overlap = context_overlap
        self.context_schedule = context_schedule
        self.closed_loop = closed_loop
        self.sync_context_to_pe = sync_context_to_pe
    
    def set_loras(self, loras: MotionLoraList):
        self.loras = loras.clone()
    
    def set_motion_model_settings(self, motion_model_settings: 'MotionModelSettings'):
        if motion_model_settings is None:
            self.motion_model_settings = MotionModelSettings()
        else:
            self.motion_model_settings = motion_model_settings

    def reset_context(self):
        self.context_length = None
        self.context_stride = None
        self.context_overlap = None
        self.context_schedule = None
        self.closed_loop = False
    
    def clone(self) -> 'InjectionParams':
        new_params = InjectionParams(
            self.video_length, self.unlimited_area_hack, self.apply_mm_groupnorm_hack,
            self.beta_schedule, self.model_name, apply_v2_models_properly=self.apply_v2_models_properly,
            )
        new_params.full_length = self.full_length
        new_params.noise_type = self.noise_type
        new_params.set_context(
            context_length=self.context_length, context_stride=self.context_stride,
            context_overlap=self.context_overlap, context_schedule=self.context_schedule,
            closed_loop=self.closed_loop, sync_context_to_pe=self.sync_context_to_pe,
            )
        if self.loras is not None:
            new_params.loras = self.loras.clone()
        new_params.set_motion_model_settings(self.motion_model_settings)
        return new_params


class MotionModelSettings:
    def __init__(self,
                 pe_strength: float=1.0,
                 attn_strength: float=1.0,
                 attn_q_strength: float=1.0,
                 attn_k_strength: float=1.0,
                 attn_v_strength: float=1.0,
                 attn_out_weight_strength: float=1.0,
                 attn_out_bias_strength: float=1.0,
                 other_strength: float=1.0,
                 cap_initial_pe_length: int=0, interpolate_pe_to_length: int=0,
                 initial_pe_idx_offset: int=0, final_pe_idx_offset: int=0,
                 motion_pe_stretch: int=0,
                 attn_scale: float=1.0,
                 mask_attn_scale: Tensor=None,
                 mask_attn_scale_min: float=1.0,
                 mask_attn_scale_max: float=1.0,
                 ):
        # general strengths
        self.pe_strength = pe_strength
        self.attn_strength = attn_strength
        self.other_strength = other_strength
        # specific attn strengths
        self.attn_q_strength = attn_q_strength
        self.attn_k_strength = attn_k_strength
        self.attn_v_strength = attn_v_strength
        self.attn_out_weight_strength = attn_out_weight_strength
        self.attn_out_bias_strength = attn_out_bias_strength
        # PE-interpolation settings
        self.cap_initial_pe_length = cap_initial_pe_length
        self.interpolate_pe_to_length = interpolate_pe_to_length
        self.initial_pe_idx_offset = initial_pe_idx_offset
        self.final_pe_idx_offset = final_pe_idx_offset
        self.motion_pe_stretch = motion_pe_stretch
        # attention scale settings
        self.attn_scale = attn_scale
        # attention scale mask settings
        self.mask_attn_scale = mask_attn_scale.clone() if mask_attn_scale is not None else mask_attn_scale
        self.mask_attn_scale_min = mask_attn_scale_min
        self.mask_attn_scale_max = mask_attn_scale_max
        self._prepare_mask_attn_scale()
    
    def _prepare_mask_attn_scale(self):
        if self.mask_attn_scale is not None:
            self.mask_attn_scale = normalize_min_max(self.mask_attn_scale, self.mask_attn_scale_min, self.mask_attn_scale_max)

    def has_mask_attn_scale(self) -> bool:
        return self.mask_attn_scale is not None

    def has_pe_strength(self) -> bool:
        return self.pe_strength != 1.0
    
    def has_attn_strength(self) -> bool:
        return self.attn_strength != 1.0
    
    def has_other_strength(self) -> bool:
        return self.other_strength != 1.0

    def has_cap_initial_pe_length(self) -> bool:
        return self.cap_initial_pe_length > 0
    
    def has_interpolate_pe_to_length(self) -> bool:
        return self.interpolate_pe_to_length > 0
    
    def has_initial_pe_idx_offset(self) -> bool:
        return self.initial_pe_idx_offset > 0
    
    def has_final_pe_idx_offset(self) -> bool:
        return self.final_pe_idx_offset > 0

    def has_motion_pe_stretch(self) -> bool:
        return self.motion_pe_stretch > 0

    def has_anything_to_apply(self) -> bool:
        return self.has_pe_strength() \
            or self.has_attn_strength() \
            or self.has_other_strength() \
            or self.has_cap_initial_pe_length() \
            or self.has_interpolate_pe_to_length() \
            or self.has_initial_pe_idx_offset() \
            or self.has_final_pe_idx_offset() \
            or self.has_motion_pe_stretch() \
            or self.has_any_attn_sub_strength()

    def has_any_attn_sub_strength(self) -> bool:
        return self.has_attn_q_strength() \
            or self.has_attn_k_strength() \
            or self.has_attn_v_strength() \
            or self.has_attn_out_weight_strength() \
            or self.has_attn_out_bias_strength()

    def has_attn_q_strength(self) -> bool:
        return self.attn_q_strength != 1.0

    def has_attn_k_strength(self) -> bool:
        return self.attn_k_strength != 1.0

    def has_attn_v_strength(self) -> bool:
        return self.attn_v_strength != 1.0

    def has_attn_out_weight_strength(self) -> bool:
        return self.attn_out_weight_strength != 1.0

    def has_attn_out_bias_strength(self) -> bool:
        return self.attn_out_bias_strength != 1.0
