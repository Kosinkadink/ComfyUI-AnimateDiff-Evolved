import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

import comfy.model_management as model_management
import comfy.model_patcher as comfy_model_patcher
from comfy.ldm.modules.diffusionmodules import openaimodel
from comfy.ldm.modules.diffusionmodules.openaimodel import ResBlock, SpatialTransformer
from comfy.model_patcher import ModelPatcher
from comfy.utils import calculate_parameters, load_torch_file
from .logger import logger
from .model_utils import ModelTypesSD, calculate_file_hash, get_motion_lora_path, get_motion_model_path, \
    get_sd_model_type
from .motion_lora import MotionLoRAList, MotionLoRAWrapper
from .motion_module_ad import AnimDiffMotionWrapper, VanillaTemporalModule, has_mid_block
from .motion_module_adxl import AnimDiffSDXLMotionWrapper
from .motion_module_hsxl import HotShotXLMotionWrapper, TransformerTemporal
from .motion_utils import GenericMotionWrapper, InjectorVersion, MotionCompatibilityError, NoiseType, normalize_min_max

# inject into ModelPatcher.clone to carry over injected params over to cloned ModelPatcher
orig_modelpatcher_clone = comfy_model_patcher.ModelPatcher.clone
def clone_injection(self, *args, **kwargs):
    model = orig_modelpatcher_clone(self, *args, **kwargs)
    if is_injected_mm_params(self):
         set_injected_mm_params(model, get_injected_mm_params(self))
    return model
comfy_model_patcher.ModelPatcher.clone = clone_injection


# cached motion modules
motion_modules: dict[str, GenericMotionWrapper] = {}
# cached motion loras
motion_loras: dict[str, MotionLoRAWrapper] = {}


# adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/utils/convert_lora_safetensor_to_diffusers.py
# Example LoRA keys:
# down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0.processor.to_q_lora.down.weight
# down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0.processor.to_q_lora.up.weight
#
# Example model keys: 
# down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0.to_q.weight
#
def apply_lora_to_mm_state_dict(model_dict: dict[str, Tensor], lora: MotionLoRAWrapper):
    # TODO: generalize for both AD and HSXL
    model_has_midblock = has_mid_block(model_dict)
    lora_has_midblock = has_mid_block(lora.state_dict)

    def get_version(has_midblock: bool):
        return "v2" if has_midblock else "v1"

    logger.info(f"Applying a {get_version(lora_has_midblock)} LoRA ({lora.info.name}) to a {get_version(model_has_midblock)} motion model.")

    for key in lora.state_dict:
        # if motion model doesn't have a mid_block, skip mid_block entries
        if not model_has_midblock:
            if "mid_block" in key: continue
        # only process lora down key (we will process up at the same time as down)
        if "up." in key: continue
        
        # key to get up value
        up_key = key.replace(".down.", ".up.")
        # adapt key to match model_dict format - remove 'processor.', '_lora', 'down.', and 'up.'
        model_key = key.replace("processor.", "").replace("_lora", "").replace("down.", "").replace("up.", "")
        # model keys have a '0.' after all 'to_out.' weight keys
        model_key = model_key.replace("to_out.", "to_out.0.")

        weight_down = lora.state_dict[key]
        weight_up = lora.state_dict[up_key]
        # apply weights to model_dict - multiply strength by matrix multiplication of up and down weights
        model_dict[model_key] += lora.info.strength * torch.mm(weight_up, weight_down).to(model_dict[model_key].device)


def load_motion_lora(lora_name: str) -> MotionLoRAWrapper:
    # if already loaded, return it
    lora_path = get_motion_lora_path(lora_name)
    lora_hash = calculate_file_hash(lora_path, hash_every_n=3)

    if lora_hash in motion_loras:
        return motion_loras[lora_hash]
    
    logger.info(f"Loading motion LoRA {lora_name}")
    l_state_dict = load_torch_file(lora_path)
    lora = MotionLoRAWrapper(l_state_dict, lora_hash)
    # add motion LoRA to cache
    motion_loras[lora_hash] = lora
    return lora


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

def load_motion_module(model_name: str, motion_lora: MotionLoRAList = None, model: ModelPatcher = None, motion_model_settings = None) -> GenericMotionWrapper:
    # if already loaded, return it
    model_path = get_motion_model_path(model_name)
    model_hash = calculate_file_hash(model_path, hash_every_n=50)

    # load lora, if present
    loras = []
    if motion_lora is not None:
        for lora_info in motion_lora.loras:
            lora = load_motion_lora(lora_info.name)
            lora.set_info(lora_info)
            loras.append(lora)
        loras.sort(key=lambda x: x.hash)
        # use lora hashes with model hash
        for lora in loras:
            model_hash += lora.hash
        model_hash = str(hash(model_hash))

    # models are determined by combo self + applied loras
    if model_hash in motion_modules:
        return motion_modules[model_hash]

    logger.info(f"Loading motion module {model_name}")
    mm_state_dict = load_torch_file(model_path)

    if motion_model_settings != None:
        mm_state_dict = apply_mm_settings(mm_state_dict, motion_model_settings)

    # load lora state dicts if exist
    if len(loras) > 0:
        for lora in loras:
            # apply LoRA to mm_state_dict
            apply_lora_to_mm_state_dict(mm_state_dict, lora)


    # determine if motion module is SD_1.5 compatible or SDXL compatible
    sd_model_type = ModelTypesSD.SD1_5
    if model is not None:
        sd_model_type = get_sd_model_type(model)
    
    motion_module: GenericMotionWrapper = None
    if sd_model_type == ModelTypesSD.SD1_5:
        try:
            motion_module = AnimDiffMotionWrapper(mm_state_dict=mm_state_dict, mm_hash=model_hash, mm_name=model_name, loras=loras)
        except MotionCompatibilityError as e:
            raise ValueError(f"Motion model {model_name} is not compatible with SD1.5-based model.", e)
    elif sd_model_type == ModelTypesSD.SDXL:
        # determine if motion module is a AnimateDiffXL model or a HotshotXL model
        try:
            # try to load as HotShotXL model first
            motion_module = HotShotXLMotionWrapper(mm_state_dict=mm_state_dict, mm_hash=model_hash, mm_name=model_name, loras=loras)
        except MotionCompatibilityError as e:
            # if not compatible, try to load as AnimateDiff-SDXL
            try:
                motion_module = AnimDiffSDXLMotionWrapper(mm_state_dict=mm_state_dict, mm_hash=model_hash, mm_name=model_name, loras=loras)
            except MotionCompatibilityError as e:
                raise ValueError(f"Motion model {model_name} is not compatible with neither AnimateDiff-SDXL nor HotShotXL.", e)
    else:
        raise ValueError(f"SD model must be either SD1.5-based for AnimateDiff or SDXL-based for HotShotXL.")


    # continue loading model
    parameters = calculate_parameters(mm_state_dict, "")
    usefp16 = model_management.should_use_fp16(model_params=parameters)
    if usefp16:
        logger.info("Using fp16, converting motion module to fp16")
        motion_module.half()
    offload_device = model_management.unet_offload_device()
    motion_module = motion_module.to(offload_device)
    motion_module.load_state_dict(mm_state_dict)

    # add to motion_module cache
    motion_modules[model_hash] = motion_module
    return motion_module


def unload_motion_module(motion_module: GenericMotionWrapper):
    logger.info(f"Removing motion module {motion_module.mm_name} from cache")
    motion_modules.pop(motion_module.mm_hash, None)


##################################################################################
##################################################################################
# Injection-related classes and functions
def inject_params_into_model(model: ModelPatcher, params: 'InjectionParams') -> ModelPatcher:
    model = model.clone()
    # clean unet, if necessary
    clean_contained_unet(model)
    set_injected_mm_params(model, params)
    return model


def eject_params_from_model(model: ModelPatcher) -> ModelPatcher:
    model = model.clone()
    # clean unet, if necessary
    clean_contained_unet(model)
    del_injected_mm_params(model)
    return model
    

def inject_motion_module(model: ModelPatcher, motion_module: GenericMotionWrapper, params: 'InjectionParams'):
    if params.context_length and params.video_length > params.context_length:
        logger.info(f"Sliding context window activated - latents passed in ({params.video_length}) greater than context_length {params.context_length}.")
    else:
        logger.info(f"Regular AnimateDiff activated - latents passed in ({params.video_length}) less or equal to context_length {params.context_length}.")
        params.reset_context()
    # if no context_length, treat video length as intended AD frame window
    if not params.context_length:
        if params.video_length > motion_module.encoding_max_len:
            raise ValueError(f"Without a context window, AnimateDiff model {motion_module.mm_name} has upper limit of {motion_module.encoding_max_len} frames, but received {params.video_length} latents.")
        motion_module.set_video_length(params.video_length, params.full_length)
    # otherwise, treat context_length as intended AD frame window
    else:
        if params.context_length > motion_module.encoding_max_len:
            raise ValueError(f"AnimateDiff model {motion_module.mm_name} has upper limit of {motion_module.encoding_max_len} frames for a context window, but received context length of {params.context_length}.")
        motion_module.set_video_length(params.context_length, params.full_length)
    # inject model
    params.set_version(motion_module)
    logger.info(f"Injecting motion module {motion_module.mm_name} version {motion_module.version}.")
    injectors[params.injector](model, motion_module)


def eject_motion_module(model: ModelPatcher):
    try:
        # handle injected params
        if is_injected_mm_params(model):
            params = get_injected_mm_params(model)
            logger.info(f"Ejecting motion module {params.model_name} version {params.version}.")
        else:
            logger.info(f"Motion module not injected, skip unloading.")
        # clean unet, just in case
    finally:
        clean_contained_unet(model)


def clean_contained_unet(model: ModelPatcher):
    if is_injected_unet_version(model):
        logger.info("Cleaning motion module from unet.")
        injector = get_injected_unet_version(model)
        ejectors[injector](model)

############################################################################################################
## AnimateDiff
def _inject_motion_module_to_unet(model: ModelPatcher, motion_module: 'AnimDiffMotionWrapper'):
    unet: openaimodel.UNetModel = model.model.diffusion_model
    for mm_idx, unet_idx in enumerate([1, 2, 4, 5, 7, 8, 10, 11]):
        mm_idx0, mm_idx1 = mm_idx // 2, mm_idx % 2
        unet.input_blocks[unet_idx].append(
            motion_module.down_blocks[mm_idx0].motion_modules[mm_idx1]
        )

    for unet_idx in range(12):
        mm_idx0, mm_idx1 = unet_idx // 3, unet_idx % 3
        if unet_idx % 3 == 2 and unet_idx != 11:
            unet.output_blocks[unet_idx].insert(
                -1, motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1]
            )
        else:
            unet.output_blocks[unet_idx].append(
                motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1]
            )

    if motion_module.mid_block is not None:
        unet.middle_block.insert(-1, motion_module.mid_block.motion_modules[0]) # only 1 VanillaTemporalModule
    # keep track of if unet blocks actually affected
    set_injected_unet_version(model, InjectorVersion.V1_V2)


def _eject_motion_module_from_unet(model: ModelPatcher):
    unet: openaimodel.UNetModel = model.model.diffusion_model
    for unet_idx in [1, 2, 4, 5, 7, 8, 10, 11]:
        unet.input_blocks[unet_idx].pop(-1)

    for unet_idx in range(12):
        if unet_idx % 3 == 2 and unet_idx != 11:
            unet.output_blocks[unet_idx].pop(-2)
        else:
            unet.output_blocks[unet_idx].pop(-1)
    
    if len(unet.middle_block) > 3: # SD1.5 UNet has 3 expected middle_blocks - more means injected
        unet.middle_block.pop(-2)
    # remove attr; ejected
    del_injected_unet_version(model)
############################################################################################################


############################################################################################################
## AnimateDiff-SDXL
def _inject_adxl_motion_module_to_unet(model: ModelPatcher, motion_module: 'AnimDiffSDXLMotionWrapper'):
    unet: openaimodel.UNetModel = model.model.diffusion_model
    # inject input (down) blocks
    # AnimateDiffSDXL mm contains 3 downblocks, each with 2 TransformerTemporals - 6 in total
    # per_block is the amount of Temporal Blocks per down block
    _perform_adxl_motion_module_injection(unet.input_blocks, motion_module.down_blocks, injection_goal=6, per_block=2)

    # inject output (up) blocks
    # AnimateDiffSDXL mm contains 3 upblocks, each with 3 TransformerTemporals - 9 in total
    _perform_adxl_motion_module_injection(unet.output_blocks, motion_module.up_blocks, injection_goal=9, per_block=3)

    # inject mid block, if needed (encapsulate in list to make structure compatible)
    if motion_module.mid_block is not None:
        _perform_adxl_motion_module_injection(unet.middle_block, [motion_module.mid_block], injection_goal=1, per_block=1)

    # keep track of if unet blocks actually affected
    set_injected_unet_version(model, InjectorVersion.ADXL_V1_V2)

def _perform_adxl_motion_module_injection(unet_blocks: nn.ModuleList, mm_blocks: nn.ModuleList, injection_goal: int, per_block: int):
    # Rules for injection:
    # For each component list in a unet block:
    #     if SpatialTransformer exists in list, place next block after last occurrence
    #     elif ResBlock exists in list, place next block after first occurrence
    #     else don't place block
    injection_count = 0
    unet_idx = 0
    # only stop injecting when modules exhausted
    while injection_count < injection_goal:
        # figure out which VanillaTemporalModule from mm to inject
        mm_blk_idx, mm_vtm_idx = injection_count // per_block, injection_count % per_block
        # figure out layout of unet block components
        st_idx = -1 # SpatialTransformer index
        res_idx = -1 # first ResBlock index
        # first, figure out indeces of relevant blocks
        for idx, component in enumerate(unet_blocks[unet_idx]):
            if type(component) == SpatialTransformer:
                st_idx = idx
            elif type(component) == ResBlock and res_idx < 0:
                res_idx = idx
        # if SpatialTransformer exists, inject right after
        if st_idx >= 0:
            #logger.info(f"ADXL: injecting after ST({st_idx})")
            unet_blocks[unet_idx].insert(st_idx+1, mm_blocks[mm_blk_idx].motion_modules[mm_vtm_idx])
            injection_count += 1
        # otherwise, if only ResBlock exists, inject right after
        elif res_idx >= 0:
            #logger.info(f"ADXL: injecting after Res({res_idx})")
            unet_blocks[unet_idx].insert(res_idx+1, mm_blocks[mm_blk_idx].motion_modules[mm_vtm_idx])
            injection_count += 1
        # increment unet_idx
        unet_idx += 1

def _eject_adxl_motion_module_from_unet(model: ModelPatcher):
    unet: openaimodel.UNetModel = model.model.diffusion_model
    # remove from input blocks
    _perform_adxl_motion_module_ejection(unet.input_blocks)
    # remove from output blocks
    _perform_adxl_motion_module_ejection(unet.output_blocks)
    # remove from middle block (encapsulate in list to make structure compatible)
    _perform_adxl_motion_module_ejection([unet.middle_block])
    # remove attr; ejected
    del_injected_unet_version(model)

def _perform_adxl_motion_module_ejection(unet_blocks: nn.ModuleList):
    # eject all TemporalTransformer3DModel objects from all blocks
    for block in unet_blocks:
        idx_to_pop = []
        for idx, component in enumerate(block):
            if type(component) == VanillaTemporalModule:
                idx_to_pop.append(idx)
        # pop in backwards order, as to not disturb what the indeces refer to
        for idx in sorted(idx_to_pop, reverse=True):
            block.pop(idx)
        #logger.info(f"ADXL: ejecting {idx_to_pop}")
############################################################################################################


############################################################################################################
## HotShot XL
def _inject_hsxl_motion_module_to_unet(model: ModelPatcher, motion_module: 'HotShotXLMotionWrapper'):
    unet: openaimodel.UNetModel = model.model.diffusion_model
    # inject input (down) blocks
    # HotShotXL mm contains 3 downblocks, each with 2 TransformerTemporals - 6 in total
    # per_block is the amount of Temporal Blocks per down block
    _perform_hsxl_motion_module_injection(unet.input_blocks, motion_module.down_blocks, injection_goal=6, per_block=2)

    # inject output (up) blocks
    # HotShotXL mm contains 3 upblocks, each with 3 TransformerTemporals - 9 in total
    _perform_hsxl_motion_module_injection(unet.output_blocks, motion_module.up_blocks, injection_goal=9, per_block=3)

    # inject mid block, if needed (encapsulate in list to make structure compatible)
    if motion_module.mid_block is not None:
        _perform_hsxl_motion_module_injection(unet.middle_block, [motion_module.mid_block], injection_goal=1, per_block=1)

    # keep track of if unet blocks actually affected
    set_injected_unet_version(model, InjectorVersion.HOTSHOTXL_V1)

def _perform_hsxl_motion_module_injection(unet_blocks: nn.ModuleList, mm_blocks: nn.ModuleList, injection_goal: int, per_block: int):
    # Rules for injection:
    # For each component list in a unet block:
    #     if SpatialTransformer exists in list, place next block after last occurrence
    #     elif ResBlock exists in list, place next block after first occurrence
    #     else don't place block
    injection_count = 0
    unet_idx = 0
    # only stop injecting when modules exhausted
    while injection_count < injection_goal:
        # figure out which TransformerTemporal from mm to inject
        mm_blk_idx, mm_tt_idx = injection_count // per_block, injection_count % per_block
        # figure out layout of unet block components
        st_idx = -1 # SpatialTransformer index
        res_idx = -1 # first ResBlock index
        # first, figure out indeces of relevant blocks
        for idx, component in enumerate(unet_blocks[unet_idx]):
            if type(component) == SpatialTransformer:
                st_idx = idx
            elif type(component) == ResBlock and res_idx < 0:
                res_idx = idx
        # if SpatialTransformer exists, inject right after
        if st_idx >= 0:
            #logger.info(f"HSXL: injecting after ST({st_idx})")
            unet_blocks[unet_idx].insert(st_idx+1, mm_blocks[mm_blk_idx].temporal_attentions[mm_tt_idx])
            injection_count += 1
        # otherwise, if only ResBlock exists, inject right after
        elif res_idx >= 0:
            #logger.info(f"HSXL: injecting after Res({res_idx})")
            unet_blocks[unet_idx].insert(res_idx+1, mm_blocks[mm_blk_idx].temporal_attentions[mm_tt_idx])
            injection_count += 1
        # increment unet_idx
        unet_idx += 1

def _eject_hsxl_motion_module_from_unet(model: ModelPatcher):
    unet: openaimodel.UNetModel = model.model.diffusion_model
    # remove from input blocks
    _perform_hsxl_motion_module_ejection(unet.input_blocks)
    # remove from output blocks
    _perform_hsxl_motion_module_ejection(unet.output_blocks)
    # remove from middle block (encapsulate in list to make structure compatible)
    _perform_hsxl_motion_module_ejection([unet.middle_block])
    # remove attr; ejected
    del_injected_unet_version(model)

def _perform_hsxl_motion_module_ejection(unet_blocks: nn.ModuleList):
    # eject all TransformerTemporal objects from all blocks
    for block in unet_blocks:
        idx_to_pop = []
        for idx, component in enumerate(block):
            if type(component) == TransformerTemporal:
                idx_to_pop.append(idx)
        # pop in backwards order, as to not disturb what the indeces refer to
        for idx in sorted(idx_to_pop, reverse=True):
            block.pop(idx)
        #logger.info(f"HSXL: ejecting {idx_to_pop}")
############################################################################################################



injectors = {
    InjectorVersion.V1_V2: _inject_motion_module_to_unet,
    InjectorVersion.ADXL_V1_V2: _inject_adxl_motion_module_to_unet,
    InjectorVersion.HOTSHOTXL_V1: _inject_hsxl_motion_module_to_unet,
}

ejectors = {
    InjectorVersion.V1_V2: _eject_motion_module_from_unet,
    InjectorVersion.ADXL_V1_V2: _eject_adxl_motion_module_from_unet,
    InjectorVersion.HOTSHOTXL_V1: _eject_hsxl_motion_module_from_unet,
}


MM_INJECTED_ATTR = "_mm_injected_params"
MM_UNET_INJECTION_ATTR = "_mm_is_unet_injected"

class InjectionParams:
    def __init__(self, video_length: int, unlimited_area_hack: bool, apply_mm_groupnorm_hack: bool, beta_schedule: str, injector: str, model_name: str,
                 apply_v2_models_properly: bool=False) -> None:
        self.video_length = video_length
        self.full_length = None
        self.unlimited_area_hack = unlimited_area_hack
        self.apply_mm_groupnorm_hack = apply_mm_groupnorm_hack
        self.beta_schedule = beta_schedule
        self.injector = injector
        self.model_name = model_name
        self.apply_v2_models_properly = apply_v2_models_properly
        self.context_length: int = None
        self.context_stride: int = None
        self.context_overlap: int = None
        self.context_schedule: str = None
        self.closed_loop: bool = False
        self.sync_context_to_pe = False
        self.version: str = None
        self.loras: MotionLoRAList = None
        self.motion_model_settings = MotionModelSettings()
        self.noise_type: str = NoiseType.DEFAULT
        self.sub_idxs = None  # value should NOT be included in clone, so it will auto reset
    
    def set_version(self, motion_module: GenericMotionWrapper):
        self.version = motion_module.version

    def set_context(self, context_length: int, context_stride: int, context_overlap: int, context_schedule: str, closed_loop: bool, sync_context_to_pe: bool=False):
        self.context_length = context_length
        self.context_stride = context_stride
        self.context_overlap = context_overlap
        self.context_schedule = context_schedule
        self.closed_loop = closed_loop
        self.sync_context_to_pe = sync_context_to_pe
    
    def set_loras(self, loras: MotionLoRAList):
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
            self.beta_schedule, self.injector, self.model_name, apply_v2_models_properly=self.apply_v2_models_properly,
            )
        new_params.full_length = self.full_length
        new_params.noise_type = self.noise_type
        new_params.version = self.version
        new_params.set_context(
            context_length=self.context_length, context_stride=self.context_stride,
            context_overlap=self.context_overlap, context_schedule=self.context_schedule,
            closed_loop=self.closed_loop, sync_context_to_pe=self.sync_context_to_pe,
            )
        if self.loras is not None:
            new_params.loras = self.loras.clone()
        new_params.set_motion_model_settings(self.motion_model_settings)
        return new_params
        

# Injected Param Functions
def is_injected_mm_params(model: ModelPatcher) -> bool:
    return hasattr(model, MM_INJECTED_ATTR)

def get_injected_mm_params(model: ModelPatcher) -> InjectionParams:
    if is_injected_mm_params(model):
        return getattr(model, MM_INJECTED_ATTR)
    return None

def set_injected_mm_params(model: ModelPatcher, injection_params: InjectionParams):
    setattr(model, MM_INJECTED_ATTR, injection_params)

def del_injected_mm_params(model: ModelPatcher):
    if is_injected_mm_params(model):
        delattr(model, MM_INJECTED_ATTR)
    
# Injected Unet Functions
def is_injected_unet_version(model: ModelPatcher) -> bool:
    return hasattr(model.model.diffusion_model, MM_UNET_INJECTION_ATTR)

def get_injected_unet_version(model: ModelPatcher) -> str:
    if is_injected_unet_version(model):
        return getattr(model.model.diffusion_model, MM_UNET_INJECTION_ATTR) 

def set_injected_unet_version(model: ModelPatcher, value: str):
    setattr(model.model.diffusion_model, MM_UNET_INJECTION_ATTR, value)

def del_injected_unet_version(model: ModelPatcher):
    if is_injected_unet_version(model):
        delattr(model.model.diffusion_model, MM_UNET_INJECTION_ATTR)


##################################################################################
##################################################################################


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
