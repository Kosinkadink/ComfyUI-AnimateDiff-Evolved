import torch
from torch import Tensor, nn
from abc import ABC, abstractmethod

from comfy.ldm.modules.attention import CrossAttentionBirchSan, CrossAttentionDoggettx, CrossAttentionPytorch, FeedForward, CrossAttention, MemoryEfficientCrossAttention
import comfy.model_management as model_management
from comfy.cli_args import args

from .motion_lora import MotionLoRAInfo


CrossAttentionMM = CrossAttention
# until xformers bug is fixed, do not use xformers for VersatileAttention! TODO: change this when fix is out
# logic for choosing CrossAttention method taken from comfy/ldm/modules/attention.py
if model_management.xformers_enabled():
    pass
    # CrossAttentionMM = MemoryEfficientCrossAttention
if model_management.pytorch_attention_enabled():
    CrossAttentionMM = CrossAttentionPytorch
else:
    if args.use_split_cross_attention:
        CrossAttentionMM = CrossAttentionDoggettx
    else:
        CrossAttentionMM = CrossAttentionBirchSan


class BlockType:
    UP = "up"
    DOWN = "down"
    MID = "mid"


class InjectorVersion:
    V1_V2 = "v1/v2"
    HOTSHOTXL_V1 = "HSXL v1"


class GenericMotionWrapper(nn.Module, ABC):
    def __init__(self, mm_hash: str, mm_name: str, loras: list[MotionLoRAInfo]):
        super().__init__()
        self.down_blocks: nn.ModuleList = None
        self.up_blocks: nn.ModuleList = None
        self.mid_block = None
        self.mm_hash = mm_hash
        self.mm_name = mm_name
        self.version = "FILLTHISIN"
        self.injector_version = "VERYIMPORTANT_FILLTHISIN"
        self.AD_video_length: int = 0
        self.loras = loras

    def has_loras(self) -> bool:
        # TODO: fix this to return False if has an empty list as well
        # but only after implementing a fix for lowvram loading
        return self.loras is not None

    @abstractmethod
    def set_video_length(self, video_length: int):
        pass
