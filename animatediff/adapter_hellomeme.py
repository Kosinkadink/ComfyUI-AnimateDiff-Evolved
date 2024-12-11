# main code adapted from HelloMeme: https://github.com/HelloVision/HelloMeme
from typing import Optional, Union, Callable

import copy
import math
import torch
from torch import Tensor, nn

from einops import rearrange

import comfy.ops
from comfy.ldm.modules.diffusionmodules import openaimodel
from comfy.ldm.modules.attention import CrossAttention, FeedForward
from comfy.model_patcher import ModelPatcher, PatcherInjection


def zero_module(module: nn.Module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def create_HM_forward_timestep_embed_patch():
    return (SKReferenceAttention, _hm_forward_timestep_embed_patch_ade)

def _hm_forward_timestep_embed_patch_ade(layer, x, emb, context, transformer_options, *args, **kwargs):
    return layer(x, transformer_options=transformer_options)



class HMReferenceAdapter(nn.Module):
    def __init__(self,
                 block_out_channels: tuple[int] = (320, 640, 1280, 1280),
                 num_attention_heads: Optional[Union[int, tuple[int]]] = 8,
                 ops=comfy.ops.disable_weight_init
                 ):
        super().__init__()

        self.block_out_channels = block_out_channels
        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(block_out_channels)
        self.num_attention_heads = num_attention_heads

        self.reference_modules_down = nn.ModuleList([])
        self.reference_modules_mid = None
        self.reference_modules_up = nn.ModuleList([])

        for i in range(len(block_out_channels)):
            output_channel = block_out_channels[i]
            
            self.reference_modules_down.append(
                SKReferenceAttention(
                    in_channels=output_channel,
                    num_attention_heads=num_attention_heads[i],
                    ops=ops
                )
            )

        self.reference_modules_mid = SKReferenceAttention(
            in_channels=block_out_channels[-1],
            num_attention_heads=num_attention_heads[-1],
            ops=ops
        )

        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))

        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            if i > 0:
                self.reference_modules_up.append(
                    SKReferenceAttention(
                        in_channels=prev_output_channel,
                        num_attention_heads=reversed_num_attention_heads[i],
                        ops=ops
                    )
                )
    
    def inject(self, model: ModelPatcher):
        unet: openaimodel.UNetModel = model.model.diffusion_model
        # inject input (down) blocks
        if self.reference_modules_down is not None:
            self._inject_down(unet.input_blocks)
        # inject mid block
        if self.reference_modules_mid is not None:
            self._inject_mid([unet.middle_block])
        # inject output (up) blocks
        if self.reference_modules_up is not None:
            self._inject_up(unet.output_blocks)
        del unet

    def _inject_down(self, unet_blocks: nn.ModuleList):
        b = 20

    def _inject_up(self, unet_blocks: nn.ModuleList):
        b = 20

    def _inject_mid(self, unet_blocks: nn.ModuleList):
        # add middle block at the end
        injection_count = 0
        unet_idx = 0
        injection_goal = 1

    def eject(self, model: ModelPatcher):
        unet: openaimodel.UNetModel = model.model.diffusion_model
        # eject input (down) blocks
        if hasattr(unet, "input_blocks"):
            self._eject(unet.input_blocks)
        # eject mid block (encapsulate in list to make compatible)
        if hasattr(unet, "middle_block"):
            self._eject([unet.middle_block])
        # eject output (up) blocks
        if hasattr(unet, "output_blocks"):
            self._eject(unet.output_blocks)
        del unet
    
    def _eject(self, unet_blocks: nn.ModuleList):
        # eject all SKReferenceAttention objects from all blocks
        for block in unet_blocks:
            idx_to_pop = []
            for idx, component in enumerate(block):
                if type(component) == SKReferenceAttention:
                    idx_to_pop.append(idx)
            # pop in reverse order, as to not disturb what the indeces refer to
            for idx in sorted(idx_to_pop, reverse=True):
                block.pop(idx)

    def create_injector(self):
        return PatcherInjection(inject=self.inject, eject=self.eject)


class SKReferenceAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_attention_heads: int=1,
                 norm_elementwise_affine: bool = True,
                 norm_eps: float = 1e-5,
                 num_positional_embeddings: int = 64*2,
                 ops = comfy.ops.disable_weight_init,
                 ):
        super().__init__()
        self.pos_embed = SinusoidalPositionalEmbedding(in_channels, max_seq_length=num_positional_embeddings)
        self.attn1 = CrossAttention(
            query_dim=in_channels,
            heads=num_attention_heads,
            dim_head=in_channels // num_attention_heads,
            dropout=0.0,
        )
        self.attn2 = CrossAttention(
            query_dim=in_channels,
            heads=num_attention_heads,
            dim_head=in_channels // num_attention_heads,
            dropout=0.0,
        )
        self.norm = ops.LayerNorm(in_channels, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.proj = zero_module(ops.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))

    # def forward(self, hidden_states: Tensor, ref_states: Tensor, num_frames: int):
    def forward(self, hidden_states: Tensor, transformer_options: dict[str]):
        h, w = hidden_states.shape[-2:]

        ref_states: Tensor = transformer_options["ade_ref_states"]
        ad_params: dict[str] = transformer_options["ad_params"]
        num_frames = ad_params.get("context_length", ad_params["full_length"])

        if ref_states.shape[0] != hidden_states.shape[0]:
            ref_states = ref_states.repeat_interleave(num_frames, dim=0)
        cat_states = torch.cat([hidden_states, ref_states], dim=-1)

        cat_states = rearrange(cat_states.contiguous(), "b c h w -> (b h) w c")
        res1 = self.attn1(self.norm(self.pos_embed(cat_states)))
        res1 = rearrange(res1[:, :w, :], "(b h) w c -> b c h w", h=h)

        cat_states2 = torch.cat([res1, ref_states], dim=-2)
        cat_states2 = rearrange(cat_states2.contiguous(), "b c h w -> (b w) h c")
        res2 = self.attn2(self.norm(self.pos_embed(cat_states2)))

        res2 = rearrange(res2[:, :h, :], "(b w) h c -> b c h w", w=w)

        return hidden_states + self.proj(res2)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, ops=comfy.ops.disable_weight_init):
    """3x3 convolution with padding"""
    return ops.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        ops=comfy.ops.disable_weight_init,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, ops=ops)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, ops=ops)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SKCrossAttention(nn.Module):
    def __init__(self,
                 channel_in,
                 channel_out,
                 heads: int=8,
                 cross_attention_dim: int=320,
                 norm_elementwise_affine: bool = True,
                 norm_eps: float = 1e-5,
                 num_positional_embeddings: int = 64,
                 num_positional_embeddings_hidden: int = 64,
                 ops=comfy.ops.disable_weight_init
                 ):
        super().__init__()
        self.conv = BasicBlock(
                        inplanes=channel_in,
                        planes=channel_out,
                        stride=2,
                        downsample=nn.Sequential(
                            ops.Conv2d(channel_in, channel_out, kernel_size=1, stride=2, bias=False),
                            nn.InstanceNorm2d(channel_out),
                            nn.SiLU(),
                        ),
                        norm_layer=nn.InstanceNorm2d
                        )
        
        self.pos_embed = SinusoidalPositionalEmbedding(channel_out, max_seq_length=num_positional_embeddings)
        self.pos_embed_hidden = SinusoidalPositionalEmbedding(cross_attention_dim, max_seq_length=num_positional_embeddings_hidden)

        self.norm1 = ops.LayerNorm(channel_out, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn1 = CrossAttention(
            query_dim=channel_out,
            heads=heads,
            dim_head=channel_out // heads,
            dropout=0.0,
            context_dim=cross_attention_dim,
        )

        self.norm2 = nn.LayerNorm(channel_out, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn2 = CrossAttention(
            query_dim=channel_out,
            heads=heads,
            dim_head=channel_out // heads,
            dropout=0.0,
            context_dim=cross_attention_dim,
        )

        self.ff = FeedForward(
            channel_out,
            mult=2,
            dropout=0.0,
            glu=True,
            operations=ops,
        )

        self.proj = zero_module(ops.Conv2d(channel_out, channel_out, kernel_size=3, padding=1))

    def forward(self, input: Tensor, hidden_states: Tensor):
        x: Tensor = self.conv(input)
        h, w = x.shape[-2:]
        x = rearrange(x, "b c h w -> (b h) w c")
        x = self.attn1(self.norm1(self.pos_embed(x)), self.pos_embed_hidden(hidden_states.repeat_interleave(h, dim=0).contiguous()))
        x = rearrange(x, "(b h) w c -> (b w) h c", h=h)
        x = self.ff(self.attn2(self.norm2(self.pos_embed(x)), self.pos_embed_hidden(hidden_states.repeat_interleave(w, dim=0).contiguous())))
        x = rearrange(x, "(b w) h c -> b c h w", w=w)
        x = self.proj(x)
        return x


# from diffusers
class SinusoidalPositionalEmbedding(nn.Module):
    """Apply positional information to a sequence of embeddings.

    Takes in a sequence of embeddings with shape (batch_size, seq_length, embed_dim) and adds positional embeddings to
    them

    Args:
        embed_dim: (int): Dimension of the positional embedding.
        max_seq_length: Maximum sequence length to apply positional embeddings

    """

    def __init__(self, embed_dim: int, max_seq_length: int = 32):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_seq_length, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        _, seq_length, _ = x.shape
        x = x + self.pe[:, :seq_length]
        return x


class InsertReferenceAdapter(object):
    def __init__(self):
        self.reference_modules_down = None
        self.reference_modules_mid = None
        self.reference_modules_up = None

    def insert_reference_adapter(self, adapter: HMReferenceAdapter):
        self.reference_modules_down = copy.deepcopy(adapter.reference_modules_down)
        self.reference_modules_mid = copy.deepcopy(adapter.reference_modules_mid)
        self.reference_modules_up = copy.deepcopy(adapter.reference_modules_up)
