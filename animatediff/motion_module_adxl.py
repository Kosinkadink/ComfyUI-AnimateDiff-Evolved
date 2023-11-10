import math
from typing import Callable, Iterable, Optional, Union

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from comfy.ldm.modules.attention import FeedForward
from .motion_lora import MotionLoRAInfo
from .motion_utils import GenericMotionWrapper, GroupNormAD, InjectorVersion, BlockType, CrossAttentionMM, MotionCompatibilityError, TemporalTransformerGeneric, prepare_mask_batch
from .motion_module_ad import MotionModule


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


def get_ad_sdxl_temporal_position_encoding_max_len(mm_state_dict: dict[str, Tensor], mm_type: str) -> int:
    # use pos_encoder.pe entries to determine max length - [1, {max_length}, {320|640|1280}]
    for key in mm_state_dict.keys():
        if key.endswith("pos_encoder.pe"):
            return mm_state_dict[key].size(1) # get middle dim
    raise MotionCompatibilityError(f"No pos_encoder.pe found in mm_state_dict - {mm_type} is not a valid AnimateDiff-SDXL motion module!")


def validate_ad_sdxl_block_count(mm_state_dict: dict[str, Tensor], mm_type: str) -> None:
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
    if biggest_block != 2:
        raise MotionCompatibilityError(f"Expected biggest down_block to be 2, but was {biggest_block} - {mm_type} is not a valid AnimateDiff-SDXL motion module!")


def has_mid_block(mm_state_dict: dict[str, Tensor]):
    # check if keys contain mid_block
    for key in mm_state_dict.keys():
        if key.startswith("mid_block."):
            return True
    return False


class AnimDiffSDXLMotionWrapper(GenericMotionWrapper):
    def __init__(self, mm_state_dict: dict[str, Tensor], mm_hash: str, mm_name: str="mm_sd_v15.ckpt" , loras: list[MotionLoRAInfo]=None):
        super().__init__(mm_hash, mm_name, loras)
        self.down_blocks: Iterable[MotionModule] = nn.ModuleList([])
        self.up_blocks: Iterable[MotionModule] = nn.ModuleList([])
        self.mid_block: Union[MotionModule, None] = None
        self.encoding_max_len = get_ad_sdxl_temporal_position_encoding_max_len(mm_state_dict, mm_name)
        validate_ad_sdxl_block_count(mm_state_dict, mm_name)
        for c in (320, 640, 1280):
            self.down_blocks.append(MotionModule(c, temporal_position_encoding_max_len=self.encoding_max_len, block_type=BlockType.DOWN))
        for c in (1280, 640, 320):
            self.up_blocks.append(MotionModule(c, temporal_position_encoding_max_len=self.encoding_max_len, block_type=BlockType.UP))
        if has_mid_block(mm_state_dict):
            self.mid_block = MotionModule(1280, temporal_position_encoding_max_len=self.encoding_max_len, block_type=BlockType.MID)
        self.mm_hash = mm_hash
        self.mm_name = mm_name
        self.version = "v1" if self.mid_block is None else "v2"
        self.injector_version = InjectorVersion.ADXL_V1_V2
        self.AD_video_length: int = 24
        self.loras = loras
    
    def has_loras(self):
        # TODO: fix this to return False if has an empty list as well
        # but only after implementing a fix for lowvram loading
        return self.loras is not None
    
    def set_video_length(self, video_length: int, full_length: int):
        self.AD_video_length = video_length
        for block in self.down_blocks:
            block.set_video_length(video_length, full_length)
        for block in self.up_blocks:
            block.set_video_length(video_length, full_length)
        if self.mid_block is not None:
            self.mid_block.set_video_length(video_length, full_length)
    
    def set_scale_multiplier(self, multiplier: Union[float, None]):
        for block in self.down_blocks:
            block.set_scale_multiplier(multiplier)
        for block in self.up_blocks:
            block.set_scale_multiplier(multiplier)
        if self.mid_block is not None:
            self.mid_block.set_scale_multiplier(multiplier)

    def set_masks(self, masks: Tensor, min_val: float, max_val: float):
        for block in self.down_blocks:
            block.set_masks(masks, min_val, max_val)
        for block in self.up_blocks:
            block.set_masks(masks, min_val, max_val)
        if self.mid_block is not None:
            self.mid_block.set_masks(masks, min_val, max_val)

    def set_sub_idxs(self, sub_idxs: list[int]):
        for block in self.down_blocks:
            block.set_sub_idxs(sub_idxs)
        for block in self.up_blocks:
            block.set_sub_idxs(sub_idxs)
        if self.mid_block is not None:
            self.mid_block.set_sub_idxs(sub_idxs)
    
    def reset_temp_vars(self):
        for block in self.down_blocks:
            block.reset_temp_vars()
        for block in self.up_blocks:
            block.reset_temp_vars()
        if self.mid_block is not None:
            self.mid_block.reset_temp_vars()
