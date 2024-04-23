import math
from typing import Iterable, Tuple, Union, TYPE_CHECKING
import re

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from comfy.ldm.modules.attention import FeedForward, SpatialTransformer
from comfy.model_patcher import ModelPatcher
from comfy.ldm.modules.diffusionmodules import openaimodel
from comfy.ldm.modules.diffusionmodules.openaimodel import SpatialTransformer
from comfy.controlnet import broadcast_image_to
from comfy.utils import repeat_to_batch_size
import comfy.ops
import comfy.model_management

from .context import ContextFuseMethod, ContextOptions, get_context_weights, get_context_windows
from .adapter_animatelcm_i2v import AdapterEmbed
if TYPE_CHECKING:  # avoids circular import
    from .adapter_cameractrl import CameraPoseEncoder
from .utils_motion import CrossAttentionMM, MotionCompatibilityError, DummyNNModule, extend_to_batch_size, prepare_mask_batch
from .utils_model import BetaSchedules, ModelTypeSD
from .logger import logger


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


class AnimateDiffFormat:
    ANIMATEDIFF = "AnimateDiff"
    HOTSHOTXL = "HotshotXL"
    ANIMATELCM = "AnimateLCM"


class AnimateDiffVersion:
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


class AnimateDiffInfo:
    def __init__(self, sd_type: str, mm_format: str, mm_version: str, mm_name: str):
        self.sd_type = sd_type
        self.mm_format = mm_format
        self.mm_version = mm_version
        self.mm_name = mm_name
    
    def get_string(self):
        return f"{self.mm_name}:{self.mm_version}:{self.mm_format}:{self.sd_type}"


def is_hotshotxl(mm_state_dict: dict[str, Tensor]) -> bool:
    # use pos_encoder naming to determine if hotshotxl model
    for key in mm_state_dict.keys():
        if key.endswith("pos_encoder.positional_encoding"):
            return True
    return False


def is_animatelcm(mm_state_dict: dict[str, Tensor]) -> bool:
    # use lack of ANY pos_encoder keys to determine if animatelcm model
    for key in mm_state_dict.keys():
        if "pos_encoder" in key:
            return False
    return True


def get_down_block_max(mm_state_dict: dict[str, Tensor]) -> int:
    # keep track of biggest down_block count in module
    biggest_block = 0
    for key in mm_state_dict.keys():
        if "down_blocks" in key:
            try:
                block_int = key.split(".")[1]
                block_num = int(block_int)
                if block_num > biggest_block:
                    biggest_block = block_num
            except ValueError:
                pass
    return biggest_block


def has_mid_block(mm_state_dict: dict[str, Tensor]):
    # check if keys contain mid_block
    for key in mm_state_dict.keys():
        if key.startswith("mid_block."):
            return True
    return False


def get_position_encoding_max_len(mm_state_dict: dict[str, Tensor], mm_name: str, mm_format: str) -> Union[int, None]:
    # use pos_encoder.pe entries to determine max length - [1, {max_length}, {320|640|1280}]
    for key in mm_state_dict.keys():
        if key.endswith("pos_encoder.pe"):
            return mm_state_dict[key].size(1) # get middle dim
    # AnimateLCM models should have no pos_encoder entries, and assumed to be 64
    if mm_format == AnimateDiffFormat.ANIMATELCM:
        return 64
    raise MotionCompatibilityError(f"No pos_encoder.pe found in mm_state_dict - {mm_name} is not a valid AnimateDiff motion module!")


_regex_hotshotxl_module_num = re.compile(r'temporal_attentions\.(\d+)\.')
def find_hotshot_module_num(key: str) -> Union[int, None]:
    found = _regex_hotshotxl_module_num.search(key)
    if found:
        return int(found.group(1))
    return None


def has_img_encoder(mm_state_dict: dict[str, Tensor]):
    for key in mm_state_dict.keys():
        if key.startswith("img_encoder."):
            return True
    return False


def normalize_ad_state_dict(mm_state_dict: dict[str, Tensor], mm_name: str) -> Tuple[dict[str, Tensor], AnimateDiffInfo]:
    # from pathlib import Path
    # with open(Path(__file__).parent.parent.parent / f"keys_{mm_name}.txt", "w") as afile:
    #     for key, value in mm_state_dict.items():
    #         afile.write(f"{key}:\t{value.shape}\n")
    
    # determine what SD model the motion module is intended for
    sd_type: str = None
    down_block_max = get_down_block_max(mm_state_dict)
    if down_block_max == 3:
        sd_type = ModelTypeSD.SD1_5
    elif down_block_max == 2:
        sd_type = ModelTypeSD.SDXL
    else:
        raise ValueError(f"'{mm_name}' is not a valid SD1.5 nor SDXL motion module - contained {down_block_max} downblocks.")
    # determine the model's format
    mm_format = AnimateDiffFormat.ANIMATEDIFF
    if is_hotshotxl(mm_state_dict):
        mm_format = AnimateDiffFormat.HOTSHOTXL
    if is_animatelcm(mm_state_dict):
        mm_format = AnimateDiffFormat.ANIMATELCM
    # for AnimateLCM-I2V purposes, check for img_encoder keys
    contains_img_encoder = has_img_encoder(mm_state_dict)
    # remove all non-temporal keys (in case model has extra stuff in it)
    for key in list(mm_state_dict.keys()):
        if "temporal" not in key:
            if mm_format == AnimateDiffFormat.ANIMATELCM and contains_img_encoder and key.startswith("img_encoder."):
                continue
            del mm_state_dict[key]
    # determine the model's version
    mm_version = AnimateDiffVersion.V1
    if has_mid_block(mm_state_dict):
        mm_version = AnimateDiffVersion.V2
    elif sd_type==ModelTypeSD.SD1_5 and get_position_encoding_max_len(mm_state_dict, mm_name, mm_format)==32:
        mm_version = AnimateDiffVersion.V3
    info = AnimateDiffInfo(sd_type=sd_type, mm_format=mm_format, mm_version=mm_version, mm_name=mm_name)
    # convert to AnimateDiff format, if needed
    if mm_format == AnimateDiffFormat.HOTSHOTXL:
        # HotshotXL is AD-based architecture applied to SDXL instead of SD1.5
        # By renaming the keys, no code needs to be adapted at all
        #
        # reformat temporal_attentions:
        # HSXL: temporal_attentions.#.
        #   AD: motion_modules.#.temporal_transformer.
        # HSXL: pos_encoder.positional_encoding
        #   AD: pos_encoder.pe
        for key in list(mm_state_dict.keys()):
            module_num = find_hotshot_module_num(key)
            if module_num is not None:
                new_key = key.replace(f"temporal_attentions.{module_num}",
                                      f"motion_modules.{module_num}.temporal_transformer", 1)
                new_key = new_key.replace("pos_encoder.positional_encoding", "pos_encoder.pe")
                mm_state_dict[new_key] = mm_state_dict[key]
                del mm_state_dict[key]
    # return adjusted mm_state_dict and info
    return mm_state_dict, info


class BlockType:
    UP = "up"
    DOWN = "down"
    MID = "mid"


class AnimateDiffModel(nn.Module):
    def __init__(self, mm_state_dict: dict[str, Tensor], mm_info: AnimateDiffInfo):
        super().__init__()
        self.mm_info = mm_info
        self.down_blocks: Iterable[MotionModule] = nn.ModuleList([])
        self.up_blocks: Iterable[MotionModule] = nn.ModuleList([])
        self.mid_block: Union[MotionModule, None] = None
        self.encoding_max_len = get_position_encoding_max_len(mm_state_dict, mm_info.mm_name, mm_info.mm_format)
        self.has_position_encoding = self.encoding_max_len is not None
        # determine ops to use (to support fp8 properly)
        if comfy.model_management.unet_manual_cast(comfy.model_management.unet_dtype(), comfy.model_management.get_torch_device()) is None:
            ops = comfy.ops.disable_weight_init
        else:
            ops = comfy.ops.manual_cast
        self.ops = ops
        # SDXL has 3 up/down blocks, SD1.5 has 4 up/down blocks
        if mm_info.sd_type == ModelTypeSD.SDXL:
            layer_channels = (320, 640, 1280)
        else:
            layer_channels = (320, 640, 1280, 1280)
        self.layer_channels = layer_channels
        # fill out down/up blocks and middle block, if present
        for idx, c in enumerate(layer_channels):
            self.down_blocks.append(MotionModule(c, temporal_pe=self.has_position_encoding,
                                                 temporal_pe_max_len=self.encoding_max_len, block_type=BlockType.DOWN, block_idx=idx, ops=ops))
        for idx, c in enumerate(list(reversed(layer_channels))):
            self.up_blocks.append(MotionModule(c, temporal_pe=self.has_position_encoding,
                                               temporal_pe_max_len=self.encoding_max_len, block_type=BlockType.UP, block_idx=idx, ops=ops))
        if has_mid_block(mm_state_dict):
            self.mid_block = MotionModule(1280, temporal_pe=self.has_position_encoding,
                                          temporal_pe_max_len=self.encoding_max_len, block_type=BlockType.MID, ops=ops)
        self.AD_video_length: int = 24
        # create AdapterEmbed if keys present for it
        self.img_encoder: AdapterEmbed = None
        if has_img_encoder(mm_state_dict):
            self.init_img_encoder()
        self.camera_encoder: 'CameraPoseEncoder' = None

    def init_img_encoder(self):
        del self.img_encoder
        self.img_encoder = AdapterEmbed(cin=4, channels=self.layer_channels, nums_rb=2, ksize=1, sk=True, use_conv=False, ops=self.ops)

    def set_camera_encoder(self, camera_encoder: 'CameraPoseEncoder'):
        del self.camera_encoder
        self.camera_encoder = camera_encoder

    def get_device_debug(self):
        return self.down_blocks[0].motion_modules[0].temporal_transformer.proj_in.weight.device

    def is_length_valid_for_encoding_max_len(self, length: int):
        if self.encoding_max_len is None:
            return True
        return length <= self.encoding_max_len

    def get_best_beta_schedule(self, log=False) -> str:
        to_return = None
        if self.mm_info.sd_type == ModelTypeSD.SD1_5:
            if self.mm_info.mm_format == AnimateDiffFormat.ANIMATELCM:
                to_return = BetaSchedules.LCM  # while LCM_100 is the intended schedule, I find LCM to have much less flicker
            else:
                to_return = BetaSchedules.SQRT_LINEAR
        elif self.mm_info.sd_type == ModelTypeSD.SDXL:
            if self.mm_info.mm_format == AnimateDiffFormat.HOTSHOTXL:
                to_return = BetaSchedules.LINEAR
            else:
                to_return = BetaSchedules.LINEAR_ADXL
        if to_return is not None:
            if log: logger.info(f"[Autoselect]: '{to_return}' beta_schedule for {self.mm_info.get_string()}")
        else:
            to_return = BetaSchedules.USE_EXISTING
            if log: logger.info(f"[Autoselect]: could not find beta_schedule for {self.mm_info.get_string()}, defaulting to '{to_return}'")
        return to_return

    def cleanup(self):
        self._reset_sub_idxs()
        self._reset_scale_multiplier()
        self._reset_temp_vars()
        if self.img_encoder is not None:
            self.img_encoder.cleanup()

    def inject(self, model: ModelPatcher):
        unet: openaimodel.UNetModel = model.model.diffusion_model
        # inject input (down) blocks
        # SD15 mm contains 4 downblocks, each with 2 TemporalTransformers - 8 in total
        # SDXL mm contains 3 downblocks, each with 2 TemporalTransformers - 6 in total
        if self.down_blocks is not None:
            self._inject(unet.input_blocks, self.down_blocks)
        # inject output (up) blocks
        # SD15 mm contains 4 upblocks, each with 3 TemporalTransformers - 12 in total
        # SDXL mm contains 3 upblocks, each with 3 TemporalTransformers - 9 in total
        if self.up_blocks is not None:
            self._inject(unet.output_blocks, self.up_blocks)
        # inject mid block, if needed (encapsulate in list to make structure compatible)
        if self.mid_block is not None:
            self._inject([unet.middle_block], [self.mid_block])
        del unet

    def _inject(self, unet_blocks: nn.ModuleList, mm_blocks: nn.ModuleList):
        # Rules for injection:
        # For each component list in a unet block:
        #     if SpatialTransformer exists in list, place next block after last occurrence
        #     elif ResBlock exists in list, place next block after first occurrence
        #     else don't place block
        injection_count = 0
        unet_idx = 0
        # details about blocks passed in
        per_block = len(mm_blocks[0].motion_modules)
        injection_goal = len(mm_blocks) * per_block
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
                elif type(component).__name__ == "ResBlock" and res_idx < 0:
                    res_idx = idx
            # if SpatialTransformer exists, inject right after
            if st_idx >= 0:
                #logger.info(f"AD: injecting after ST({st_idx})")
                unet_blocks[unet_idx].insert(st_idx+1, mm_blocks[mm_blk_idx].motion_modules[mm_vtm_idx])
                injection_count += 1
            # otherwise, if only ResBlock exists, inject right after
            elif res_idx >= 0:
                #logger.info(f"AD: injecting after Res({res_idx})")
                unet_blocks[unet_idx].insert(res_idx+1, mm_blocks[mm_blk_idx].motion_modules[mm_vtm_idx])
                injection_count += 1
            # increment unet_idx
            unet_idx += 1

    def eject(self, model: ModelPatcher):
        unet: openaimodel.UNetModel = model.model.diffusion_model
        # remove from input blocks (downblocks)
        self._eject(unet.input_blocks)
        # remove from output blocks (upblocks)
        self._eject(unet.output_blocks)
        # remove from middle block (encapsulate in list to make compatible)
        self._eject([unet.middle_block])
        del unet

    def _eject(self, unet_blocks: nn.ModuleList):
        # eject all VanillaTemporalModule objects from all blocks
        for block in unet_blocks:
            idx_to_pop = []
            for idx, component in enumerate(block):
                if type(component) == VanillaTemporalModule:
                    idx_to_pop.append(idx)
            # pop in backwards order, as to not disturb what the indeces refer to
            for idx in sorted(idx_to_pop, reverse=True):
                block.pop(idx)

    def set_video_length(self, video_length: int, full_length: int):
        self.AD_video_length = video_length
        if self.down_blocks is not None:
            for block in self.down_blocks:
                block.set_video_length(video_length, full_length)
        if self.up_blocks is not None:
            for block in self.up_blocks:
                block.set_video_length(video_length, full_length)
        if self.mid_block is not None:
            self.mid_block.set_video_length(video_length, full_length)
    
    def set_scale(self, multival: Union[float, Tensor]):
        if multival is None:
            multival = 1.0
        if type(multival) == Tensor:
            self._set_scale_multiplier(1.0)
            self._set_scale_mask(multival)
        else:
            self._set_scale_multiplier(multival)
            self._set_scale_mask(None)
    
    def set_effect(self, multival: Union[float, Tensor]):
        if self.down_blocks is not None:
            for block in self.down_blocks:
                block.set_effect(multival)
        if self.up_blocks is not None:
            for block in self.up_blocks:
                block.set_effect(multival)
        if self.mid_block is not None:
            self.mid_block.set_effect(multival)

    def set_cameractrl_effect(self, multival: Union[float, Tensor]):
        # cameractrl should only impact down and up blocks
        if self.down_blocks is not None:
            for block in self.down_blocks:
                block.set_cameractrl_effect(multival)
        if self.up_blocks is not None:
            for block in self.up_blocks:
                block.set_cameractrl_effect(multival)

    def set_sub_idxs(self, sub_idxs: list[int]):
        if self.down_blocks is not None:
            for block in self.down_blocks:
                block.set_sub_idxs(sub_idxs)
        if self.up_blocks is not None:
            for block in self.up_blocks:
                block.set_sub_idxs(sub_idxs)
        if self.mid_block is not None:
            self.mid_block.set_sub_idxs(sub_idxs)

    def set_view_options(self, view_options: ContextOptions):
        if self.down_blocks is not None:
            for block in self.down_blocks:
                block.set_view_options(view_options)
        if self.up_blocks is not None:
            for block in self.up_blocks:
                block.set_view_options(view_options)
        if self.mid_block is not None:
            self.mid_block.set_view_options(view_options)

    def set_img_features(self, img_features: list[Tensor], apply_ref_when_disabled=False):
        # img_features should only impact downblocks
        if self.down_blocks is not None:
            for block in self.down_blocks:
                block.set_img_features(img_features=img_features, apply_ref_when_disabled=apply_ref_when_disabled)

    def set_camera_features(self, camera_features: list[Tensor]):
        # camera features should only impact down and up blocks
        if self.down_blocks is not None:
            for block in self.down_blocks:
                block.set_camera_features(camera_features=camera_features)
        if self.up_blocks is not None:
            for block in self.up_blocks:
                block.set_camera_features(camera_features=list(reversed(camera_features)))

    def _set_scale_multiplier(self, multiplier: Union[float, None]):
        if self.down_blocks is not None:
            for block in self.down_blocks:
                block.set_scale_multiplier(multiplier)
        if self.up_blocks is not None:
            for block in self.up_blocks:
                block.set_scale_multiplier(multiplier)
        if self.mid_block is not None:
            self.mid_block.set_scale_multiplier(multiplier)

    def _set_scale_mask(self, mask: Tensor):
        if self.down_blocks is not None:
            for block in self.down_blocks:
                block.set_scale_mask(mask)
        if self.up_blocks is not None:
            for block in self.up_blocks:
                block.set_scale_mask(mask)
        if self.mid_block is not None:
            self.mid_block.set_scale_mask(mask)
    
    def _reset_temp_vars(self):
        if self.down_blocks is not None:
            for block in self.down_blocks:
                block.reset_temp_vars()
        if self.up_blocks is not None:
            for block in self.up_blocks:
                block.reset_temp_vars()
        if self.mid_block is not None:
            self.mid_block.reset_temp_vars()

    def _reset_scale_multiplier(self):
        self._set_scale_multiplier(None)

    def _reset_sub_idxs(self):
        self.set_sub_idxs(None)


class MotionModule(nn.Module):
    def __init__(self,
            in_channels,
            temporal_pe=True,
            temporal_pe_max_len=24,
            block_type: str=BlockType.DOWN,
            block_idx: int=0,
            ops=comfy.ops.disable_weight_init
        ):
        super().__init__()
        if block_type == BlockType.MID:
            # mid blocks contain only a single VanillaTemporalModule
            self.motion_modules: Iterable[VanillaTemporalModule] = nn.ModuleList([get_motion_module(in_channels, block_type, block_idx, module_idx=0, temporal_pe=temporal_pe, temporal_pe_max_len=temporal_pe_max_len, ops=ops)])
        else:
            # down blocks contain two VanillaTemporalModules
            self.motion_modules: Iterable[VanillaTemporalModule] = nn.ModuleList(
                [
                    get_motion_module(in_channels, block_type, block_idx, module_idx=0, temporal_pe=temporal_pe, temporal_pe_max_len=temporal_pe_max_len, ops=ops),
                    get_motion_module(in_channels, block_type, block_idx, module_idx=1, temporal_pe=temporal_pe, temporal_pe_max_len=temporal_pe_max_len, ops=ops)
                ]
            )
            # up blocks contain one additional VanillaTemporalModule
            if block_type == BlockType.UP: 
                self.motion_modules.append(get_motion_module(in_channels, block_type, block_idx, module_idx=2, temporal_pe=temporal_pe, temporal_pe_max_len=temporal_pe_max_len, ops=ops))
    
    def set_video_length(self, video_length: int, full_length: int):
        for motion_module in self.motion_modules:
            motion_module.set_video_length(video_length, full_length)
    
    def set_scale_multiplier(self, multiplier: Union[float, None]):
        for motion_module in self.motion_modules:
            motion_module.set_scale_multiplier(multiplier)
    
    def set_scale_mask(self, mask: Tensor):
        for motion_module in self.motion_modules:
            motion_module.set_scale_mask(mask)
    
    def set_effect(self, multival: Union[float, Tensor]):
        for motion_module in self.motion_modules:
            motion_module.set_effect(multival)
    
    def set_cameractrl_effect(self, multival: Union[float, Tensor]):
        for motion_module in self.motion_modules:
            motion_module.set_cameractrl_effect(multival)
    
    def set_sub_idxs(self, sub_idxs: list[int]):
        for motion_module in self.motion_modules:
            motion_module.set_sub_idxs(sub_idxs)

    def set_view_options(self, view_options: ContextOptions):
        for motion_module in self.motion_modules:
            motion_module.set_view_options(view_options=view_options)

    def set_img_features(self, img_features: list[Tensor], apply_ref_when_disabled=False):
        for motion_module in self.motion_modules:
            motion_module.set_img_features(img_features=img_features, apply_ref_when_disabled=apply_ref_when_disabled)

    def set_camera_features(self, camera_features: list[Tensor]):
        for idx, motion_module in enumerate(self.motion_modules):
            #if idx == 0:
            motion_module.set_camera_features(camera_features=camera_features)

    def reset_temp_vars(self):
        for motion_module in self.motion_modules:
            motion_module.reset_temp_vars()


def get_motion_module(in_channels, block_type: str, block_idx: int, module_idx: int,
                      temporal_pe, temporal_pe_max_len, ops=comfy.ops.disable_weight_init):
    return VanillaTemporalModule(in_channels=in_channels, block_type=block_type, block_idx=block_idx, module_idx=module_idx,
                                 temporal_pe=temporal_pe, temporal_pe_max_len=temporal_pe_max_len, ops=ops)


class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        block_type: str,
        block_idx: int,
        module_idx: int,
        num_attention_heads=8,
        num_transformer_block=1,
        attention_block_types=("Temporal_Self", "Temporal_Self"),
        cross_frame_attention_mode=None,
        temporal_pe=True,
        temporal_pe_max_len=24,
        temporal_attention_dim_div=1,
        zero_initialize=True,
        ops=comfy.ops.disable_weight_init,
    ):
        super().__init__()

        self.video_length = 16
        self.full_length = 16
        self.sub_idxs = None
        self.view_options = None
        # keep track of module's position in unet
        self.block_type = block_type
        self.block_idx = block_idx
        self.module_idx = module_idx
        # effect vars
        self.effect = None
        self.temp_effect_mask: Tensor = None
        self.prev_input_tensor_batch = 0
        # AnimateLCM-I2V vars
        self.img_features: list[Tensor] = None
        self.apply_ref_when_disabled = False
        # CameraCtrl vars
        self.camera_features: list[Tensor] = None

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels
            // num_attention_heads
            // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_pe=temporal_pe,
            temporal_pe_max_len=temporal_pe_max_len,
            ops=ops
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(
                self.temporal_transformer.proj_out
            )

    def set_video_length(self, video_length: int, full_length: int):
        self.video_length = video_length
        self.full_length = full_length
        self.temporal_transformer.set_video_length(video_length, full_length)
    
    def set_scale_multiplier(self, multiplier: Union[float, None]):
        self.temporal_transformer.set_scale_multiplier(multiplier)

    def set_scale_mask(self, mask: Tensor):
        self.temporal_transformer.set_scale_mask(mask)

    def set_effect(self, multival: Union[float, Tensor]):
        if type(multival) == Tensor:
            self.effect = multival
        elif multival is not None and math.isclose(multival, 1.0):
            self.effect = None
        else:
            self.effect = multival
        self.temp_effect_mask = None
    
    def set_cameractrl_effect(self, multival: Union[float, Tensor]):
        if type(multival) == Tensor:
            pass
        elif multival is None:
            multival = 1.0
        elif multival is not None and math.isclose(multival, 1.0):
            multival = 1.0
        self.temporal_transformer.set_cameractrl_effect(multival)
        

    def set_sub_idxs(self, sub_idxs: list[int]):
        self.sub_idxs = sub_idxs
        self.temporal_transformer.set_sub_idxs(sub_idxs)

    def set_view_options(self, view_options: ContextOptions):
        self.view_options = view_options

    def set_img_features(self, img_features: list[Tensor], apply_ref_when_disabled=False):
        del self.img_features
        self.img_features = img_features
        self.apply_ref_when_disabled = apply_ref_when_disabled

    def set_camera_features(self, camera_features: list[Tensor]):
        del self.camera_features
        self.camera_features = camera_features

    def reset_temp_vars(self):
        self.set_effect(None)
        self.set_view_options(None)
        self.set_img_features(None)
        self.set_camera_features(None)
        self.temporal_transformer.reset_temp_vars()

    def get_effect_mask(self, input_tensor: Tensor):
        batch, channel, height, width = input_tensor.shape
        batched_number = batch // self.video_length
        full_batched_idxs = list(range(self.video_length))*batched_number
        # if there is a cached temp_effect_mask and it is valid for current input, return it
        if batch == self.prev_input_tensor_batch and self.temp_effect_mask is not None:
            if self.sub_idxs is not None:
                return self.temp_effect_mask[self.sub_idxs*batched_number]
            return self.temp_effect_mask[full_batched_idxs]
        # clear any existing mask
        del self.temp_effect_mask
        self.temp_effect_mask = None
        # recalculate temp mask
        self.prev_input_tensor_batch = batch
        # make sure mask matches expected dimensions
        mask = prepare_mask_batch(self.effect, shape=(self.full_length, 1, height, width))
        # make sure mask is as long as full_length - clone last element of list if too short
        self.temp_effect_mask = extend_to_batch_size(mask, self.full_length).to(
            dtype=input_tensor.dtype, device=input_tensor.device)
        # return finalized mask
        if self.sub_idxs is not None:
            return self.temp_effect_mask[self.sub_idxs*batched_number]
        return self.temp_effect_mask[full_batched_idxs]

    def should_handle_img_features(self):
        return self.img_features is not None and self.block_type == BlockType.DOWN and self.module_idx == 1

    def should_handle_camera_features(self):
        return self.camera_features is not None and self.block_type != BlockType.MID# and self.module_idx == 0

    def forward(self, input_tensor: Tensor, encoder_hidden_states=None, attention_mask=None):
        mm_kwargs = None
        if self.should_handle_camera_features():
            mm_kwargs = {"camera_feature": self.camera_features[self.block_idx]}
        if self.effect is None:
            # do AnimateLCM-I2V stuff if needed
            if self.should_handle_img_features():
                input_tensor += self.img_features[self.block_idx]
            return self.temporal_transformer(input_tensor, encoder_hidden_states, attention_mask, self.view_options, mm_kwargs)
        # return weighted average of input_tensor and AD output
        if type(self.effect) != Tensor:
            effect = self.effect
            # do nothing if effect is 0
            if math.isclose(effect, 0.0):
                # do AnimateLCM-I2V stuff if needed
                if self.apply_ref_when_disabled and self.should_handle_img_features():
                    input_tensor += self.img_features[self.block_idx]
                return input_tensor
        else:
            effect = self.get_effect_mask(input_tensor)
        # do AnimateLCM-I2V stuff if needed
        if self.should_handle_img_features():
            return input_tensor*(1.0-effect) + self.temporal_transformer(input_tensor+self.img_features[self.block_idx], encoder_hidden_states, attention_mask, self.view_options, mm_kwargs)*effect
        return input_tensor*(1.0-effect) + self.temporal_transformer(input_tensor, encoder_hidden_states, attention_mask, self.view_options, mm_kwargs)*effect


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_pe=False,
        temporal_pe_max_len=24,
        ops=comfy.ops.disable_weight_init,
    ):
        super().__init__()
        self.video_length = 16
        self.full_length = 16
        self.raw_scale_mask: Union[Tensor, None] = None
        self.temp_scale_mask: Union[Tensor, None] = None
        self.sub_idxs: Union[list[int], None] = None
        self.prev_hidden_states_batch = 0

        # cameractrl stuff
        self.raw_cameractrl_effect: Union[float, Tensor] = None
        self.temp_cameractrl_effect: Union[float, Tensor] = None
        self.prev_cameractrl_hidden_states_batch = 0

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = ops.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = ops.Linear(in_channels, inner_dim)

        self.transformer_blocks: Iterable[TemporalTransformerBlock] = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_pe=temporal_pe,
                    temporal_pe_max_len=temporal_pe_max_len,
                    ops=ops,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = ops.Linear(inner_dim, in_channels)

    def set_video_length(self, video_length: int, full_length: int):
        self.video_length = video_length
        self.full_length = full_length
    
    def set_scale_multiplier(self, multiplier: Union[float, None]):
        for block in self.transformer_blocks:
            block.set_scale_multiplier(multiplier)

    def set_scale_mask(self, mask: Tensor):
        self.raw_scale_mask = mask
        self.temp_scale_mask = None

    def set_cameractrl_effect(self, multival: Union[float, Tensor]):
        self.raw_cameractrl_effect = multival
        self.temp_cameractrl_effect = None

    def set_sub_idxs(self, sub_idxs: list[int]):
        self.sub_idxs = sub_idxs
        for block in self.transformer_blocks:
            block.set_sub_idxs(sub_idxs)

    def reset_temp_vars(self):
        del self.temp_scale_mask
        self.temp_scale_mask = None
        self.prev_hidden_states_batch = 0
        del self.temp_cameractrl_effect
        self.temp_cameractrl_effect = None
        self.prev_cameractrl_hidden_states_batch = 0
        for block in self.transformer_blocks:
            block.reset_temp_vars()

    def get_scale_mask(self, hidden_states: Tensor) -> Union[Tensor, None]:
        # if no raw mask, return None
        if self.raw_scale_mask is None:
            return None
        shape = hidden_states.shape
        batch, channel, height, width = shape
        # if temp mask already calculated, return it
        if self.temp_scale_mask != None:
            # check if hidden_states batch matches
            if batch == self.prev_hidden_states_batch:
                if self.sub_idxs is not None:
                    return self.temp_scale_mask[:, self.sub_idxs, :]
                return self.temp_scale_mask
            # if does not match, reset cached temp_scale_mask and recalculate it
            del self.temp_scale_mask
            self.temp_scale_mask = None
        # otherwise, calculate temp mask
        self.prev_hidden_states_batch = batch
        mask = prepare_mask_batch(self.raw_scale_mask, shape=(self.full_length, 1, height, width))
        mask = repeat_to_batch_size(mask, self.full_length)
        # if mask not the same amount length as full length, make it match
        if self.full_length != mask.shape[0]:
            mask = broadcast_image_to(mask, self.full_length, 1)
        # reshape mask to attention K shape (h*w, latent_count, 1)
        batch, channel, height, width = mask.shape
        # first, perform same operations as on hidden_states,
        # turning (b, c, h, w) -> (b, h*w, c)
        mask = mask.permute(0, 2, 3, 1).reshape(batch, height*width, channel)
        # then, make it the same shape as attention's k, (h*w, b, c)
        mask = mask.permute(1, 0, 2)
        # make masks match the expected length of h*w
        batched_number = shape[0] // self.video_length
        if batched_number > 1:
            mask = torch.cat([mask] * batched_number, dim=0)
        # cache mask and set to proper device
        self.temp_scale_mask = mask
        # move temp_scale_mask to proper dtype + device
        self.temp_scale_mask = self.temp_scale_mask.to(dtype=hidden_states.dtype, device=hidden_states.device)
        # return subset of masks, if needed
        if self.sub_idxs is not None:
            return self.temp_scale_mask[:, self.sub_idxs, :]
        return self.temp_scale_mask

    def get_cameractrl_effect(self, hidden_states: Tensor) -> Union[float, Tensor, None]:
        # if no raw camera_Ctrl, return None
        if self.raw_cameractrl_effect is None:
            return 1.0
        # if raw_cameractrl is not a Tensor, return it (should be a float)
        if type(self.raw_cameractrl_effect) != Tensor:
            return self.raw_cameractrl_effect
        shape = hidden_states.shape
        batch, channel, height, width = shape
        # if temp_cameractrl already calculated, return it
        if self.temp_cameractrl_effect != None:
            # check if hidden_states batch matches
            if batch == self.prev_cameractrl_hidden_states_batch:
                if self.sub_idxs is not None:
                    return self.temp_cameractrl_effect[:, self.sub_idxs, :]
                return self.temp_cameractrl_effect
            # if does not match, reset cached temp_cameractrl and recalculate it
            del self.temp_cameractrl_effect
            self.temp_cameractrl_effect = None
        # otherwise, calculate temp_cameractrl
        self.prev_cameractrl_hidden_states_batch = batch
        mask = prepare_mask_batch(self.raw_scale_mask, shape=(self.full_length, 1, height, width))
        mask = repeat_to_batch_size(mask, self.full_length)
        # if mask not the same amount length as full length, make it match
        if self.full_length != mask.shape[0]:
            mask = broadcast_image_to(mask, self.full_length, 1)
        # reshape mask to attention K shape (h*w, latent_count, 1)
        batch, channel, height, width = mask.shape
        # first, perform same operations as on hidden_states,
        # turning (b, c, h, w) -> (b, h*w, c)
        mask = mask.permute(0, 2, 3, 1).reshape(batch, height*width, channel)
        # then, make it the same shape as attention's k, (h*w, b, c)
        mask = mask.permute(1, 0, 2)
        # make masks match the expected length of h*w
        batched_number = shape[0] // self.video_length
        if batched_number > 1:
            mask = torch.cat([mask] * batched_number, dim=0)
        # cache mask and set to proper device
        self.temp_cameractrl_effect = mask
        # move temp_cameractrl to proper dtype + device
        self.temp_cameractrl_effect = self.temp_cameractrl_effect.to(dtype=hidden_states.dtype, device=hidden_states.device)
        # return subset of masks, if needed
        if self.sub_idxs is not None:
            return self.temp_cameractrl_effect[:, self.sub_idxs, :]
        return self.temp_cameractrl_effect

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, view_options: ContextOptions=None, mm_kwargs: dict[str]=None):
        batch, channel, height, width = hidden_states.shape
        residual = hidden_states
        scale_mask = self.get_scale_mask(hidden_states)
        cameractrl_effect = self.get_cameractrl_effect(hidden_states)
        # add some casts for fp8 purposes - does not affect speed otherwise
        hidden_states = self.norm(hidden_states).to(hidden_states.dtype)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * width, inner_dim
        )
        hidden_states = self.proj_in(hidden_states).to(hidden_states.dtype)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                video_length=self.video_length,
                scale_mask=scale_mask,
                cameractrl_effect=cameractrl_effect,
                view_options=view_options,
                mm_kwargs=mm_kwargs
            )

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, width, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        output = hidden_states + residual

        return output


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_pe=False,
        temporal_pe_max_len=24,
        ops=comfy.ops.disable_weight_init,
    ):
        super().__init__()

        attention_blocks: Iterable[VersatileAttention] = []
        norms = []

        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    context_dim=cross_attention_dim # called context_dim for ComfyUI impl
                    if block_name.endswith("_Cross")
                    else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    #bias=attention_bias, # remove for Comfy CrossAttention
                    #upcast_attention=upcast_attention, # remove for Comfy CrossAttention
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_pe=temporal_pe,
                    temporal_pe_max_len=temporal_pe_max_len,
                    ops=ops,
                )
            )
            norms.append(ops.LayerNorm(dim))

        attention_blocks[0].camera_feature_enabled = True
        self.attention_blocks: Iterable[VersatileAttention] = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, glu=(activation_fn == "geglu"), operations=ops)
        self.ff_norm = ops.LayerNorm(dim)

    def set_scale_multiplier(self, multiplier: Union[float, None]):
        for block in self.attention_blocks:
            block.set_scale_multiplier(multiplier)

    def set_sub_idxs(self, sub_idxs: list[int]):
        for block in self.attention_blocks:
            block.set_sub_idxs(sub_idxs)

    def reset_temp_vars(self):
        for block in self.attention_blocks:
            block.reset_temp_vars()

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor=None,
        attention_mask: Tensor=None,
        video_length: int=None,
        scale_mask: Tensor=None,
        cameractrl_effect: Union[float, Tensor] = None,
        view_options: ContextOptions=None,
        mm_kwargs: dict[str]=None,
    ):
        # make view_options None if context_length > video_length, or if equal and equal not allowed
        if view_options:
            if view_options.context_length > video_length:
                view_options = None
            elif view_options.context_length == video_length and not view_options.use_on_equal_length:
                view_options = None
        if not view_options:
            for attention_block, norm in zip(self.attention_blocks, self.norms):
                norm_hidden_states = norm(hidden_states).to(hidden_states.dtype)
                hidden_states = (
                    attention_block(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states
                        if attention_block.is_cross_attention
                        else None,
                        attention_mask=attention_mask,
                        video_length=video_length,
                        scale_mask=scale_mask,
                        cameractrl_effect=cameractrl_effect,
                        mm_kwargs=mm_kwargs
                    ) + hidden_states
                )
        else:
            # views idea gotten from diffusers AnimateDiff FreeNoise implementation:
            # https://github.com/arthur-qiu/FreeNoise-AnimateDiff/blob/main/animatediff/models/motion_module.py
            # apply sliding context windows (views)
            views = get_context_windows(num_frames=video_length, opts=view_options)
            hidden_states = rearrange(hidden_states, "(b f) d c -> b f d c", f=video_length)
            value_final = torch.zeros_like(hidden_states)
            count_final = torch.zeros_like(hidden_states)
            # bias_final = [0.0] * video_length
            batched_conds = hidden_states.size(1) // video_length
            # store original camera_feature, if present
            has_camera_feature = False
            if mm_kwargs is not None:
                has_camera_feature = True
                orig_camera_feature = mm_kwargs["camera_feature"]
            # perform view options
            for sub_idxs in views:
                sub_hidden_states = rearrange(hidden_states[:, sub_idxs], "b f d c -> (b f) d c")
                if has_camera_feature:
                    mm_kwargs["camera_feature"] = orig_camera_feature[:, sub_idxs, :]
                for attention_block, norm in zip(self.attention_blocks, self.norms):
                    norm_hidden_states = norm(sub_hidden_states).to(sub_hidden_states.dtype)
                    sub_hidden_states = (
                        attention_block(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states # do these need to be changed for sub_idxs too?
                            if attention_block.is_cross_attention
                            else None,
                            attention_mask=attention_mask,
                            video_length=len(sub_idxs),
                            scale_mask=scale_mask[:, sub_idxs, :] if scale_mask is not None else scale_mask,
                            cameractrl_effect=cameractrl_effect[:, sub_idxs, :] if type(cameractrl_effect) == Tensor else cameractrl_effect,
                            mm_kwargs=mm_kwargs
                        ) + sub_hidden_states
                    )
                sub_hidden_states = rearrange(sub_hidden_states, "(b f) d c -> b f d c", f=len(sub_idxs))

                # if view_options.fuse_method == ContextFuseMethod.RELATIVE:
                #     for pos, idx in enumerate(sub_idxs):
                #         # bias is the influence of a specific index in relation to the whole context window
                #         bias = 1 - abs(idx - (sub_idxs[0] + sub_idxs[-1]) / 2) / ((sub_idxs[-1] - sub_idxs[0] + 1e-2) / 2)
                #         bias = max(1e-2, bias)
                #         # take weighted averate relative to total bias of current idx
                #         bias_total = bias_final[idx]
                #         prev_weight = torch.tensor([bias_total / (bias_total + bias)],
                #                                    dtype=value_final.dtype, device=value_final.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                #         #prev_weight = torch.cat([prev_weight]*value_final.shape[1], dim=1)
                #         new_weight = torch.tensor([bias / (bias_total + bias)],
                #                                    dtype=value_final.dtype, device=value_final.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                #         #new_weight = torch.cat([new_weight]*value_final.shape[1], dim=1)
                #         test = value_final[:, idx:idx+1, :, :]
                #         value_final[:, idx:idx+1, :, :] = value_final[:, idx:idx+1, :, :] * prev_weight + sub_hidden_states[:, pos:pos+1, : ,:] * new_weight
                #         bias_final[idx] = bias_total + bias
                # else:
                weights = get_context_weights(len(sub_idxs), view_options.fuse_method) * batched_conds
                weights_tensor = torch.Tensor(weights).to(device=hidden_states.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                value_final[:, sub_idxs] += sub_hidden_states * weights_tensor
                count_final[:, sub_idxs] += weights_tensor
            # restore original camera_feature
            if has_camera_feature:
                mm_kwargs["camera_feature"] = orig_camera_feature
                del orig_camera_feature
            # get weighted average of sub_hidden_states, if fuse method requires it
            # if view_options.fuse_method != ContextFuseMethod.RELATIVE:
            hidden_states = value_final / count_final
            hidden_states = rearrange(hidden_states, "b f d c -> (b f) d c")
            del value_final
            del count_final
            # del bias_final

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=24):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.sub_idxs = None
        self.pe: Tensor

    def set_sub_idxs(self, sub_idxs: list[int]):
        self.sub_idxs = sub_idxs

    def forward(self, x: Tensor):
        #if self.sub_idxs is not None:
        #    x = x + self.pe[:, self.sub_idxs]
        #else:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class VersatileAttention(CrossAttentionMM):
    def __init__(
        self,
        attention_mode=None,
        cross_frame_attention_mode=None,
        temporal_pe=False,
        temporal_pe_max_len=24,
        ops=comfy.ops.disable_weight_init,
        *args,
        **kwargs,
    ):
        super().__init__(operations=ops, *args, **kwargs)
        assert attention_mode == "Temporal"

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs["context_dim"] is not None

        self.query_dim: int = kwargs["query_dim"]
        self.qkv_merge: comfy.ops.disable_weight_init.Linear = None
        self.camera_feature_enabled = False

        self.pos_encoder = (
            PositionalEncoding(
                kwargs["query_dim"],
                dropout=0.0,
                max_len=temporal_pe_max_len,
            )
            if (temporal_pe and attention_mode == "Temporal")
            else None
        )

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def set_scale_multiplier(self, multiplier: Union[float, None]):
        if multiplier is None or math.isclose(multiplier, 1.0):
            self.scale = 1.0
        else:
            self.scale = multiplier

    def set_sub_idxs(self, sub_idxs: list[int]):
        if self.pos_encoder != None:
            self.pos_encoder.set_sub_idxs(sub_idxs)

    def init_qkv_merge(self, ops=comfy.ops.disable_weight_init):
        self.qkv_merge = zero_module(ops.Linear(in_features=self.query_dim, out_features=self.query_dim))

    def reset_temp_vars(self):
        self.reset_attention_type()

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        scale_mask=None,
        cameractrl_effect: Union[float, Tensor] = 1.0,
        mm_kwargs: dict[str]={},
    ):
        if self.attention_mode != "Temporal":
            raise NotImplementedError

        d = hidden_states.shape[1]
        hidden_states = rearrange(
            hidden_states, "(b f) d c -> (b d) f c", f=video_length
        )

        if self.pos_encoder is not None:
           hidden_states = self.pos_encoder(hidden_states).to(hidden_states.dtype)

        encoder_hidden_states = (
            repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
            if encoder_hidden_states is not None
            else encoder_hidden_states
        )

        if self.camera_feature_enabled and self.qkv_merge is not None and mm_kwargs is not None and "camera_feature" in mm_kwargs:
            camera_feature: Tensor = mm_kwargs["camera_feature"]
            hidden_states = (self.qkv_merge(hidden_states + camera_feature) + hidden_states) * cameractrl_effect + hidden_states * (1. - cameractrl_effect)

        hidden_states = super().forward(
            hidden_states,
            encoder_hidden_states,
            value=None,
            mask=attention_mask,
            scale_mask=scale_mask,
        )

        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states

############################################################################
### EncoderOnly Version
############################################################################
class EncoderOnlyAnimateDiffModel(AnimateDiffModel):
    def __init__(self, mm_state_dict: dict[str, Tensor], mm_info: AnimateDiffInfo):
        super().__init__(mm_state_dict=mm_state_dict, mm_info=mm_info)
        self.down_blocks: Iterable[EncoderOnlyMotionModule] = nn.ModuleList([])
        self.up_blocks = None
        self.mid_block = None
        # fill out down/up blocks and middle block, if present
        for idx, c in enumerate(self.layer_channels):
            self.down_blocks.append(EncoderOnlyMotionModule(c, block_type=BlockType.DOWN, block_idx=idx, ops=self.ops))
    
    def _eject(self, unet_blocks: nn.ModuleList):
        # eject all EncoderOnlyTemporalModule objects from all blocks
        for block in unet_blocks:
            idx_to_pop = []
            for idx, component in enumerate(block):
                if type(component) == EncoderOnlyTemporalModule:
                    idx_to_pop.append(idx)
            # pop in backwards order, as to not disturb what the indeces refer to
            for idx in sorted(idx_to_pop, reverse=True):
                block.pop(idx)


class EncoderOnlyMotionModule(MotionModule):
    '''
    MotionModule that will store EncoderOnlyTemporalModule objects instead of VanillaTemporalModules
    '''
    def __init__(
            self,
            in_channels,
            block_type: str=BlockType.DOWN,
            block_idx: int=0,
            ops=comfy.ops.disable_weight_init
        ):
        super().__init__(in_channels=in_channels, block_type=block_type, block_idx=block_idx, ops=ops)
        if block_type == BlockType.MID:
            # mid blocks contain only a single VanillaTemporalModule
            self.motion_modules: Iterable[EncoderOnlyTemporalModule] = nn.ModuleList([EncoderOnlyTemporalModule.create(in_channels, block_type, block_idx, module_idx=0, ops=ops)])
        else:
            # down blocks contain two VanillaTemporalModules
            self.motion_modules: Iterable[EncoderOnlyTemporalModule] = nn.ModuleList(
                [
                    EncoderOnlyTemporalModule.create(in_channels, block_type, block_idx, module_idx=0, ops=ops),
                    EncoderOnlyTemporalModule.create(in_channels, block_type, block_idx, module_idx=1, ops=ops)
                ]
            )
            # up blocks contain one additional VanillaTemporalModule
            if block_type == BlockType.UP: 
                self.motion_modules.append(EncoderOnlyTemporalModule.create(in_channels, block_type, block_idx, module_idx=2, ops=ops))


class EncoderOnlyTemporalModule(VanillaTemporalModule):
    '''
    VanillaTemporalModule that will only add img_features to input_tensor while respecting effect_multival
    '''
    def __init__(
            self,
            in_channels,
            block_type: str,
            block_idx: int,
            module_idx: int,
            ops=comfy.ops.disable_weight_init,
        ):
        super().__init__(in_channels=in_channels, block_type=block_type, block_idx=block_idx, module_idx=module_idx, zero_initialize=False, ops=ops)
        # make temporal_transformer a dummy class that does nothing, but will allow inherited VanillaTemporalModule code to work
        self.temporal_transformer = DummyNNModule()

    @classmethod
    def create(cls, in_channels, block_type: str, block_idx: int, module_idx: int, ops=comfy.ops.disable_weight_init):
        return cls(in_channels=in_channels, block_type=block_type, block_idx=block_idx, module_idx=module_idx, ops=ops)

    def forward(self, input_tensor: Tensor, encoder_hidden_states=None, attention_mask=None):
        if self.effect is None:
            # do AnimateLCM-I2V stuff if needed
            if self.should_handle_img_features():
                input_tensor += self.img_features[self.block_idx]
            return input_tensor
        # handle effect
        if type(self.effect) != Tensor:
            effect = self.effect
            # do nothing if effect is 0
            if math.isclose(effect, 0.0):
                # do AnimateLCM-I2V stuff if needed
                if self.apply_ref_when_disabled and self.should_handle_img_features():
                    input_tensor += self.img_features[self.block_idx]
                return input_tensor
        else:
            effect = self.get_effect_mask(input_tensor)
        if self.should_handle_img_features():
            return input_tensor*(1.0-effect) + (input_tensor+self.img_features[self.block_idx])*effect
        return input_tensor  # since no img_features to apply, no need for weighted average
############################################################################
