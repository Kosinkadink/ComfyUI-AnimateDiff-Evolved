import math
from typing import Iterable, Tuple, Union
import re

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from comfy.ldm.modules.attention import FeedForward, SpatialTransformer
from comfy.model_patcher import ModelPatcher
from comfy.ldm.modules.diffusionmodules import openaimodel
from comfy.ldm.modules.diffusionmodules.openaimodel import SpatialTransformer

from .motion_utils import GroupNormAD, BlockType, CrossAttentionMM, MotionCompatibilityError, TemporalTransformerGeneric
from .model_utils import ModelTypeSD


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


class AnimateDiffFormat:
    ANIMATEDIFF = "AnimateDiff"
    HOTSHOTXL = "HotshotXL"


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


def is_hotshotxl(mm_state_dict: dict[str, Tensor]) -> bool:
    # use pos_encoder naming to determine if hotshotxl model
    for key in mm_state_dict.keys():
        if key.endswith("pos_encoder.positional_encoding"):
            return True
    return False


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


def get_position_encoding_max_len(mm_state_dict: dict[str, Tensor], mm_name: str) -> int:
    # use pos_encoder.pe entries to determine max length - [1, {max_length}, {320|640|1280}]
    for key in mm_state_dict.keys():
        if key.endswith("pos_encoder.pe"):
            return mm_state_dict[key].size(1) # get middle dim
    raise MotionCompatibilityError(f"No pos_encoder.pe found in mm_state_dict - {mm_name} is not a valid AnimateDiff motion module!")


_regex_hotshotxl_module_num = re.compile(r'temporal_attentions\.(\d+)\.')
def find_hotshot_module_num(key: str) -> Union[int, None]:
    found = _regex_hotshotxl_module_num.search(key)
    if found:
        return int(found.group(1))
    return None


def normalize_ad_state_dict(mm_state_dict: dict[str, Tensor], mm_name: str) -> Tuple[dict[str, Tensor], AnimateDiffInfo]:
    # remove all non-temporal keys (in case model has extra stuff in it)
    for key in list(mm_state_dict.keys()):
        if "temporal" not in key:
            del mm_state_dict[key]
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
    # determine the model's version
    mm_version = AnimateDiffVersion.V1
    if has_mid_block(mm_state_dict):
        mm_version = AnimateDiffVersion.V2
    elif sd_type==ModelTypeSD.SD1_5 and get_position_encoding_max_len(mm_state_dict, mm_name)==32:
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


class AnimateDiffModel(nn.Module):
    def __init__(self, mm_state_dict: dict[str, Tensor], mm_info: AnimateDiffInfo):
        super().__init__()
        self.mm_info = mm_info
        self.down_blocks: Iterable[MotionModule] = nn.ModuleList([])
        self.up_blocks: Iterable[MotionModule] = nn.ModuleList([])
        self.mid_block: Union[MotionModule, None] = None
        self.encoding_max_len = get_position_encoding_max_len(mm_state_dict, mm_info.mm_name)
        # SDXL has 3 up/down blocks, SD1.5 has 4 up/down blocks
        if mm_info.sd_type == ModelTypeSD.SDXL:
            layer_channels = (320, 640, 1280)
        else:
            layer_channels = (320, 640, 1280, 1280)
        # fill out down/up blocks and middle block, if present
        for c in layer_channels:
            self.down_blocks.append(MotionModule(c, temporal_position_encoding_max_len=self.encoding_max_len, block_type=BlockType.DOWN))
        for c in reversed(layer_channels):
            self.up_blocks.append(MotionModule(c, temporal_position_encoding_max_len=self.encoding_max_len, block_type=BlockType.UP))
        if has_mid_block(mm_state_dict):
            self.mid_block = MotionModule(1280, temporal_position_encoding_max_len=self.encoding_max_len, block_type=BlockType.MID)
        self.AD_video_length: int = 24

    def get_device_debug(self):
        return self.down_blocks[0].motion_modules[0].temporal_transformer.proj_in.weight.device

    def cleanup(self):
        pass

    def inject(self, model: ModelPatcher):
        unet: openaimodel.UNetModel = model.model.diffusion_model
        # inject input (down) blocks
        # SD15 mm contains 4 downblocks, each with 2 TemporalTransformers - 8 in total
        # SDXL mm contains 3 downblocks, each with 2 TemporalTransformers - 6 in total
        self._inject(unet.input_blocks, self.down_blocks)
        # inject output (up) blocks
        # SD15 mm contains 4 upblocks, each with 3 TemporalTransformers - 12 in total
        # SDXL mm contains 3 upblocks, each with 3 TemporalTransformers - 9 in total
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

    def reset_scale_multiplier(self):
        self.set_scale_multiplier(None)

    def reset_sub_idxs(self):
        self.set_sub_idxs(None)

    def reset(self):
        self.reset_sub_idxs()
        self.reset_scale_multiplier()
        self.reset_temp_vars()


class MotionModule(nn.Module):
    def __init__(self, in_channels, temporal_position_encoding_max_len=24, block_type: str=BlockType.DOWN):
        super().__init__()
        if block_type == BlockType.MID:
            # mid blocks contain only a single VanillaTemporalModule
            self.motion_modules: Iterable[VanillaTemporalModule] = nn.ModuleList([get_motion_module(in_channels, temporal_position_encoding_max_len)])
        else:
            # down blocks contain two VanillaTemporalModules
            self.motion_modules: Iterable[VanillaTemporalModule] = nn.ModuleList(
                [
                    get_motion_module(in_channels, temporal_position_encoding_max_len),
                    get_motion_module(in_channels, temporal_position_encoding_max_len)
                ]
            )
            # up blocks contain one additional VanillaTemporalModule
            if block_type == BlockType.UP: 
                self.motion_modules.append(get_motion_module(in_channels, temporal_position_encoding_max_len))
    
    def set_video_length(self, video_length: int, full_length: int):
        for motion_module in self.motion_modules:
            motion_module.set_video_length(video_length, full_length)
    
    def set_scale_multiplier(self, multiplier: Union[float, None]):
        for motion_module in self.motion_modules:
            motion_module.set_scale_multiplier(multiplier)
    
    def set_masks(self, masks: Tensor, min_val: float, max_val: float):
        for motion_module in self.motion_modules:
            motion_module.set_masks(masks, min_val, max_val)
    
    def set_sub_idxs(self, sub_idxs: list[int]):
        for motion_module in self.motion_modules:
            motion_module.set_sub_idxs(sub_idxs)

    def reset_temp_vars(self):
        for motion_module in self.motion_modules:
            motion_module.reset_temp_vars()


def get_motion_module(in_channels, temporal_position_encoding_max_len):
    return VanillaTemporalModule(in_channels=in_channels, temporal_position_encoding_max_len=temporal_position_encoding_max_len)


class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        num_transformer_block=1,
        attention_block_types=("Temporal_Self", "Temporal_Self"),
        cross_frame_attention_mode=None,
        temporal_position_encoding=True,
        temporal_position_encoding_max_len=24,
        temporal_attention_dim_div=1,
        zero_initialize=True,
    ):
        super().__init__()

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels
            // num_attention_heads
            // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(
                self.temporal_transformer.proj_out
            )

    def set_video_length(self, video_length: int, full_length: int):
        self.temporal_transformer.set_video_length(video_length, full_length)
    
    def set_scale_multiplier(self, multiplier: Union[float, None]):
        self.temporal_transformer.set_scale_multiplier(multiplier)

    def set_masks(self, masks: Tensor, min_val: float, max_val: float):
        self.temporal_transformer.set_masks(masks, min_val, max_val)
    
    def set_sub_idxs(self, sub_idxs: list[int]):
        self.temporal_transformer.set_sub_idxs(sub_idxs)

    def reset_temp_vars(self):
        self.temporal_transformer.reset_temp_vars()

    def forward(self, input_tensor, encoder_hidden_states=None, attention_mask=None):
        return self.temporal_transformer(input_tensor, encoder_hidden_states, attention_mask)
        #portion = output_tensor.shape[2] // 4 + output_tensor.shape[2] // 2
        portion = output_tensor.shape[2] // 2
        ad_effect = 0.7
        #output_tensor[:,:,portion:] = input_tensor[:,:,portion:] * (1-ad_effect) + output_tensor[:,:,portion:] * ad_effect
        #output_tensor[:,:,portion:] = input_tensor[:,:,portion:] #* 0.5
        return output_tensor


class TemporalTransformer3DModel(nn.Module, TemporalTransformerGeneric):
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
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()
        super().temporal_transformer_init(default_length=16)

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = GroupNormAD(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Linear(in_channels, inner_dim)

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
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def set_video_length(self, video_length: int, full_length: int):
        self.video_length = video_length
        self.full_length = full_length
    
    def set_scale_multiplier(self, multiplier: Union[float, None]):
        for block in self.transformer_blocks:
            block.set_scale_multiplier(multiplier)

    def set_masks(self, masks: Tensor, min_val: float, max_val: float):
        self.scale_min = min_val
        self.scale_max = max_val
        self.raw_scale_mask = masks

    def set_sub_idxs(self, sub_idxs: list[int]):
        self.sub_idxs = sub_idxs
        for block in self.transformer_blocks:
            block.set_sub_idxs(sub_idxs)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch, channel, height, width = hidden_states.shape
        residual = hidden_states
        scale_mask = self.get_scale_mask(hidden_states)
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
                scale_mask=scale_mask
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
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        attention_blocks = []
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
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks: Iterable[VersatileAttention] = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, glu=(activation_fn == "geglu"))
        self.ff_norm = nn.LayerNorm(dim)

    def set_scale_multiplier(self, multiplier: Union[float, None]):
        for block in self.attention_blocks:
            block.set_scale_multiplier(multiplier)

    def set_sub_idxs(self, sub_idxs: list[int]):
        for block in self.attention_blocks:
            block.set_sub_idxs(sub_idxs)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        scale_mask=None
    ):
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
                    scale_mask=scale_mask
                )
                + hidden_states
            )

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

    def set_sub_idxs(self, sub_idxs: list[int]):
        self.sub_idxs = sub_idxs

    def forward(self, x):
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
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal"

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs["context_dim"] is not None

        self.pos_encoder = (
            PositionalEncoding(
                kwargs["query_dim"],
                dropout=0.0,
                max_len=temporal_position_encoding_max_len,
            )
            if (temporal_position_encoding and attention_mode == "Temporal")
            else None
        )

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def set_scale_multiplier(self, multiplier: Union[float, None]):
        if multiplier is None or math.isclose(multiplier, 1.0):
            self.scale = None
        else:
            self.scale = multiplier

    def set_sub_idxs(self, sub_idxs: list[int]):
        if self.pos_encoder != None:
            self.pos_encoder.set_sub_idxs(sub_idxs)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        scale_mask=None,
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

        hidden_states = super().forward(
            hidden_states,
            encoder_hidden_states,
            value=None,
            mask=attention_mask,
            scale_mask=scale_mask,
        )

        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
