import torch
from torch import Tensor, nn

import math
from einops import rearrange, repeat

from comfy.ldm.modules.attention import CrossAttentionBirchSan, CrossAttentionDoggettx, CrossAttentionPytorch, FeedForward, CrossAttention, MemoryEfficientCrossAttention
from ldm.modules.diffusionmodules import openaimodel
import comfy.model_patcher as comfy_model_patcher
from comfy.model_patcher import ModelPatcher
import comfy.model_management as model_management
from comfy.cli_args import args
from comfy.utils import calculate_parameters, load_torch_file

from .model_utils import Folders, calculate_file_hash, get_full_path, is_checkpoint_sd1_5
from .logger import logger


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


# inject into ModelPatcher.clone to carry over injected params over to cloned ModelPatcher
orig_modelpatcher_clone = comfy_model_patcher.ModelPatcher.clone
def clone_injection(self, *args, **kwargs):
    model = orig_modelpatcher_clone(self, *args, **kwargs)
    if is_checkpoint_sd1_5(model) and is_injected_mm_params(self):
         set_injected_mm_params(model, get_injected_mm_params(self))
    return model
comfy_model_patcher.ModelPatcher.clone = clone_injection


# cache loaded motion modules
motion_modules: dict[str, 'MotionWrapper'] = {}


def load_motion_module(model_name: str):
    # if already loaded, return it
    model_path = get_full_path(Folders.MODELS, model_name)
    model_hash = calculate_file_hash(model_path)

    if model_hash in motion_modules:
        return motion_modules[model_hash]

    logger.info(f"Loading motion module {model_name}")
    mm_state_dict = load_torch_file(model_path)
    motion_module = MotionWrapper(mm_state_dict=mm_state_dict, mm_name=model_name)

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
    

def inject_motion_module(model: ModelPatcher, motion_module: 'MotionWrapper', params: 'InjectionParams'):
    if params.context_length and params.video_length > params.context_length:
        logger.info(f"Sliding context window activated - latents passed in ({params.video_length}) greater than context_length {params.context_length}.")
    else:
        logger.info(f"Regular AnimateDiff activated - latents passed in ({params.video_length}) less or equal to context_length {params.context_length}.")
        params.reset_context()
    # if no context_length, treat video length as intended AD frame window
    if not params.context_length:
        if params.video_length > motion_module.encoding_max_len:
            raise ValueError(f"Without a context window, AnimateDiff model {motion_module.mm_name} has upper limit of {motion_module.encoding_max_len} frames, but received {params.video_length} latents.")
        motion_module.set_video_length(params.video_length)
    # otherwise, treat context_length as intended AD frame window
    else:
        if params.context_length > motion_module.encoding_max_len:
            raise ValueError(f"AnimateDiff model {motion_module.mm_name} has upper limit of {motion_module.encoding_max_len} frames for a context window, but received context length of {params.context_length}.")
        motion_module.set_video_length(params.context_length)
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


def _inject_motion_module_to_unet(model: ModelPatcher, motion_module: 'MotionWrapper'):
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
    logger.info(f"Ejecting motion module from UNet input blocks.")
    for unet_idx in [1, 2, 4, 5, 7, 8, 10, 11]:
        unet.input_blocks[unet_idx].pop(-1)

    logger.info(f"Ejecting motion module from UNet output blocks.")
    for unet_idx in range(12):
        if unet_idx % 3 == 2 and unet_idx != 11:
            unet.output_blocks[unet_idx].pop(-2)
        else:
            unet.output_blocks[unet_idx].pop(-1)
    
    if len(unet.middle_block) > 3: # SD1.5 UNet has 3 expected middle_blocks - more means injected
        logger.info(f"Ejecting motion module from UNet middle blocks.")
        unet.middle_block.pop(-2)
    # remove attr; ejected
    del_injected_unet_version(model)


class InjectorVersion:
    V1_V2 = "v1/v2"


injectors = {
    InjectorVersion.V1_V2: _inject_motion_module_to_unet,
}

ejectors = {
    InjectorVersion.V1_V2: _eject_motion_module_from_unet,
}


MM_INJECTED_ATTR = "_mm_injected_params"
MM_UNET_INJECTION_ATTR = "_mm_is_unet_injected"

class InjectionParams:
    def __init__(self, video_length: int, unlimited_area_hack: bool, beta_schedule: str, injector: str, model_name: str) -> None:
        self.video_length = video_length
        self.unlimited_area_hack = unlimited_area_hack
        self.beta_schedule = beta_schedule
        self.injector = injector
        self.model_name = model_name
        self.context_length: int = None
        self.context_stride: int = None
        self.context_overlap: int = None
        self.context_schedule: str = None
        self.closed_loop: bool = False
        self.version: str = None
    
    def set_version(self, motion_module: 'MotionWrapper'):
        self.version = motion_module.version

    def set_context(self, context_length: int, context_stride: int, context_overlap: int, context_schedule: str, closed_loop: bool):
        self.context_length = context_length
        self.context_stride = context_stride
        self.context_overlap = context_overlap
        self.context_schedule = context_schedule
        self.closed_loop = closed_loop
    
    def reset_context(self):
        self.context_length = None
        self.context_stride = None
        self.context_overlap = None
        self.context_schedule = None
        self.closed_loop = False

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
    if is_checkpoint_sd1_5(model):
        return hasattr(model.model.diffusion_model, MM_UNET_INJECTION_ATTR)

def get_injected_unet_version(model: ModelPatcher) -> str:
    if is_checkpoint_sd1_5(model):
        if is_injected_unet_version(model):
            return getattr(model.model.diffusion_model, MM_UNET_INJECTION_ATTR) 

def set_injected_unet_version(model: ModelPatcher, value: str):
    if is_checkpoint_sd1_5(model):
        setattr(model, MM_UNET_INJECTION_ATTR, value)

def del_injected_unet_version(model: ModelPatcher):
    if is_checkpoint_sd1_5(model):
        if is_injected_unet_version(model):
            delattr(model, MM_UNET_INJECTION_ATTR)


##################################################################################
##################################################################################


class BlockType:
    UP = "up"
    DOWN = "down"
    MID = "mid"


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


def get_temporal_position_encoding_max_len(mm_state_dict: dict[str, Tensor], mm_type: str) -> int:
    # use pos_encoder.pe entries to determine max length - [1, {max_length}, {320|640|1280}]
    for key in mm_state_dict.keys():
        if key.endswith("pos_encoder.pe"):
            return mm_state_dict[key].size(1) # get middle dim
    raise ValueError(f"No pos_encoder.pe found in mm_state_dict - {mm_type} is not a valid motion module!")


def has_mid_block(mm_state_dict: dict[str, Tensor]):
    # check if keys contain mid_block
    for key in mm_state_dict.keys():
        if key.startswith("mid_block."):
            return True
    return False


class MotionWrapper(nn.Module):
    def __init__(self, mm_state_dict: dict[str, Tensor], mm_name: str="mm_sd_v15.ckpt"):
        super().__init__()
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.mid_block = None
        self.encoding_max_len = get_temporal_position_encoding_max_len(mm_state_dict, mm_name)
        for c in (320, 640, 1280, 1280):
            self.down_blocks.append(MotionModule(c, temporal_position_encoding_max_len=self.encoding_max_len, block_type=BlockType.DOWN))
        for c in (1280, 1280, 640, 320):
            self.up_blocks.append(MotionModule(c, temporal_position_encoding_max_len=self.encoding_max_len, block_type=BlockType.UP))
        if has_mid_block(mm_state_dict):
            self.mid_block = MotionModule(1280, temporal_position_encoding_max_len=self.encoding_max_len, block_type=BlockType.MID)
        self.mm_name = mm_name
        self.version = "v1" if self.mid_block is None else "v2"
        self.AD_video_length: int = 24
    
    def set_video_length(self, video_length: int):
        self.AD_video_length = video_length
        for block in self.down_blocks:
            block.set_video_length(video_length)
        for block in self.up_blocks:
            block.set_video_length(video_length)
        if self.mid_block is not None:
            self.mid_block.set_video_length(video_length)


class MotionModule(nn.Module):
    def __init__(self, in_channels, temporal_position_encoding_max_len=24, block_type: str=BlockType.DOWN):
        super().__init__()
        if block_type == BlockType.MID:
            # mid blocks contain only a single VanillaTemporalModule
            self.motion_modules = nn.ModuleList([get_motion_module(in_channels, temporal_position_encoding_max_len)])
        else:
            # down blocks contain two VanillaTemporalModules
            self.motion_modules = nn.ModuleList(
                [
                    get_motion_module(in_channels, temporal_position_encoding_max_len),
                    get_motion_module(in_channels, temporal_position_encoding_max_len)
                ]
            )
            # up blocks contain one additional VanillaTemporalModule
            if block_type == BlockType.UP: 
                self.motion_modules.append(get_motion_module(in_channels, temporal_position_encoding_max_len))
    
    def set_video_length(self, video_length: int):
        for motion_module in self.motion_modules:
            motion_module.set_video_length(video_length)


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

    def set_video_length(self, video_length: int):
        self.temporal_transformer.set_video_length(video_length)

    def forward(self, input_tensor, encoder_hidden_states, attention_mask=None):
        return self.temporal_transformer(input_tensor, encoder_hidden_states, attention_mask)


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
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
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
        self.video_length = 16

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
                video_length=self.video_length,
            )

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, weight, inner_dim)
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

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, glu=(activation_fn == "geglu"))
        self.ff_norm = nn.LayerNorm(dim)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
    ):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = (
                attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states
                    if attention_block.is_cross_attention
                    else None,
                    video_length=video_length,
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

    def forward(self, x):
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

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
    ):
        if self.attention_mode != "Temporal":
            raise NotImplementedError

        d = hidden_states.shape[1]
        hidden_states = rearrange(
            hidden_states, "(b f) d c -> (b d) f c", f=video_length
        )

        if self.pos_encoder is not None:
           hidden_states = self.pos_encoder(hidden_states)

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
        )

        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
