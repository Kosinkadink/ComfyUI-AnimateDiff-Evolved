# original HotShotXL components adapted from https://github.com/hotshotco/Hotshot-XL/blob/main/hotshot_xl/models/transformer_temporal.py
from typing import Optional
import torch
from torch import Tensor, nn

import math
from einops import rearrange, repeat

from comfy.ldm.modules.attention import FeedForward

from .motion_utils import GenericMotionWrapper, GroupNormAD, InjectorVersion, BlockType, CrossAttentionMM
from .motion_lora import MotionLoRAInfo


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


def get_hsxl_temporal_position_encoding_max_len(mm_state_dict: dict[str, Tensor], mm_type: str) -> int:
    # use pos_encoder.positional_encoding entries to determine max length - [1, {max_length}, {320|640|1280}]
    for key in mm_state_dict.keys():
        if key.endswith("pos_encoder.positional_encoding"):
            return mm_state_dict[key].size(1) # get middle dim
    raise ValueError(f"No pos_encoder.positional_encoding found in mm_state_dict - {mm_type} is not a valid HotShotXL motion module!")


def has_mid_block(mm_state_dict: dict[str, Tensor]):
    # check if keys contain mid_block (temporal)
    for key in mm_state_dict.keys():
        if key.startswith("mid_block.") and "temporal" in key:
            return True
    return False


#########################################################################################
# Explanation for future me and other developers:
# Goal of the Wrapper and HotShotXLMotionModule is to create a structure compatible with the motion module to be loaded.
# Names of nn.ModuleList objects match that of entries of the motion module
#########################################################################################


class HotShotXLMotionWrapper(GenericMotionWrapper):
    def __init__(self, mm_state_dict: dict[str, Tensor], mm_hash: str, mm_name: str="mm_sd_v15.ckpt", loras: list[MotionLoRAInfo]=None):
        super().__init__(mm_hash, mm_name, loras)
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.mid_block = None
        self.encoding_max_len = get_hsxl_temporal_position_encoding_max_len(mm_state_dict, mm_name)
        for c in (320, 640, 1280):
            self.down_blocks.append(HotShotXLMotionModule(c, block_type=BlockType.DOWN))
        for c in (1280, 640, 320):
            self.up_blocks.append(HotShotXLMotionModule(c, block_type=BlockType.UP))
        if has_mid_block(mm_state_dict):
            self.mid_block = HotShotXLMotionModule(1280, BlockType=BlockType.MID)
        self.mm_hash = mm_hash
        self.mm_name = mm_name
        self.version = "HSXL v1" if self.mid_block is None else "HSXL v2"
        self.injector_version = InjectorVersion.HOTSHOTXL_V1
        self.AD_video_length: int = 8
        self.loras = loras
    
    def has_loras(self):
        # TODO: fix this to return False if has an empty list as well
        # but only after implementing a fix for lowvram loading
        return self.loras is not None
    
    def set_video_length(self, video_length: int):
        self.AD_video_length = video_length
        for block in self.down_blocks:
            block.set_video_length(video_length)
        for block in self.up_blocks:
            block.set_video_length(video_length)
        if self.mid_block is not None:
            self.mid_block.set_video_length(video_length)
        

class HotShotXLMotionModule(nn.Module):
    def __init__(self, in_channels, block_type: str=BlockType.DOWN):
        super().__init__()
        if block_type == BlockType.MID:
            # mid blocks contain only a single TransformerTemporal
            self.temporal_attentions = nn.ModuleList([get_transformer_temporal(in_channels)])
        else:
            # down blocks contain two TransformerTemporals
            self.temporal_attentions = nn.ModuleList(
                [
                    get_transformer_temporal(in_channels),
                    get_transformer_temporal(in_channels)
                ]
            )
            # up blocks contain one additional TransformerTemporal
            if block_type == BlockType.UP:
                self.temporal_attentions.append(get_transformer_temporal(in_channels))

    def set_video_length(self, video_length: int):
        for tt in self.temporal_attentions:
            tt.set_video_length(video_length)


def get_transformer_temporal(in_channels) -> 'TransformerTemporal':
    num_attention_heads = 8
    return TransformerTemporal(
        num_attention_heads=num_attention_heads,
        attention_head_dim=in_channels // num_attention_heads,
        in_channels=in_channels,
    )


class TransformerTemporal(nn.Module):
    def __init__(
            self,
            num_attention_heads: int,
            attention_head_dim: int,
            in_channels: int,
            num_layers: int = 1,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = None,
            attention_bias: bool = False,
            activation_fn: str = "geglu",
            upcast_attention: bool = False,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = GroupNormAD(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_attention_dim=cross_attention_dim
                )
                for _ in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)
        self.video_length = 8

    def set_video_length(self, video_length: int):
        self.video_length = video_length

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * weight, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                number_of_frames=self.video_length)

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, weight, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        output = hidden_states + residual

        return output


class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_attention_heads,
            attention_head_dim,
            dropout=0.0,
            activation_fn="geglu",
            attention_bias=False,
            upcast_attention=False,
            depth=2,
            cross_attention_dim: Optional[int] = None
    ):
        super().__init__()

        self.is_cross = cross_attention_dim is not None

        attention_blocks = []
        norms = []

        for _ in range(depth):
            attention_blocks.append(
                TemporalAttention(
                    query_dim=dim,
                    context_dim=cross_attention_dim, # called context_dim for ComfyUI impl
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    #bias=attention_bias, # remove for Comfy CrossAttention
                    #upcast_attention=upcast_attention, # remove for Comfy CrossAttention
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, glu=(activation_fn == "geglu"))
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, number_of_frames=None):

        if not self.is_cross:
            encoder_hidden_states = None

        for block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = block(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                number_of_frames=number_of_frames
            ) + hidden_states

        norm_hidden_states = self.ff_norm(hidden_states)
        hidden_states = self.ff(norm_hidden_states) + hidden_states

        output = hidden_states
        return output


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as described in "Attention Is All You Need".
    Adds sinusoidal based positional encodings to the input tensor.
    """

    _SCALE_FACTOR = 10000.0  # Scale factor used in the positional encoding computation.

    def __init__(self, dim: int, dropout: float = 0.0, max_length: int = 24):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # The size is (1, max_length, dim) to allow easy addition to input tensors.
        positional_encoding = torch.zeros(1, max_length, dim)

        # Position and dim are used in the sinusoidal computation.
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(self._SCALE_FACTOR) / dim))

        positional_encoding[0, :, 0::2] = torch.sin(position * div_term)
        positional_encoding[0, :, 1::2] = torch.cos(position * div_term)

        # Register the positional encoding matrix as a buffer,
        # so it's part of the model's state but not the parameters.
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, hidden_states: torch.Tensor, length: int) -> torch.Tensor:
        hidden_states = hidden_states + self.positional_encoding[:, :length]
        return self.dropout(hidden_states)


class TemporalAttention(CrossAttentionMM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_encoder = PositionalEncoding(kwargs["query_dim"], dropout=0)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, number_of_frames=8):
        sequence_length = hidden_states.shape[1]
        hidden_states = rearrange(hidden_states, "(b f) s c -> (b s) f c", f=number_of_frames)
        hidden_states = self.pos_encoder(hidden_states, length=number_of_frames)

        if encoder_hidden_states:
            encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b s) n c", s=sequence_length)

        hidden_states = super().forward(hidden_states, encoder_hidden_states, mask=attention_mask)

        return rearrange(hidden_states, "(b s) f c -> (b f) s c", s=sequence_length)
