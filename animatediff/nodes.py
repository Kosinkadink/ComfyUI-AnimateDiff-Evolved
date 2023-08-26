import os
import json
import hashlib
import torch
import numpy as np
from typing import Dict, List, Tuple
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from einops import rearrange

import folder_paths
import comfy.ldm.modules.diffusionmodules.openaimodel as openaimodel
import comfy.model_management as model_management
from comfy.ldm.modules.attention import SpatialTransformer
from comfy.ldm.modules.diffusionmodules.util import GroupNorm32
from comfy.utils import load_torch_file, calculate_parameters
from comfy.sd import ModelPatcher

from .logger import logger
from .motion_module import MotionWrapper, VanillaTemporalModule
from .model_utils import MODEL_DIR, get_available_models


orig_forward_timestep_embed = openaimodel.forward_timestep_embed
groupnorm32_original_forward = GroupNorm32.forward


def forward_timestep_embed(
    ts, x, emb, context=None, transformer_options={}, output_shape=None
):
    for layer in ts:
        if isinstance(layer, openaimodel.TimestepBlock):
            x = layer(x, emb)
        elif isinstance(layer, VanillaTemporalModule):
            x = layer(x, context)
        elif isinstance(layer, SpatialTransformer):
            x = layer(x, context, transformer_options)
            transformer_options["current_index"] += 1
        elif isinstance(layer, openaimodel.Upsample):
            x = layer(x, output_shape=output_shape)
        else:
            x = layer(x)
    return x


def groupnorm32_mm_forward(self, x):
    x = rearrange(x, "(b f) c h w -> b c f h w", b=2)
    x = groupnorm32_original_forward(self, x)
    x = rearrange(x, "b c f h w -> (b f) c h w", b=2)
    return x


openaimodel.forward_timestep_embed = forward_timestep_embed

motion_modules: Dict[str, MotionWrapper] = {}
original_model_hashs = set()
injected_model_hashs: Dict[str, Tuple[str, str]] = {}


def calculate_model_hash(unet):
    t = unet.input_blocks[1]
    m = hashlib.sha256()
    for buf in t.buffers():
        m.update(buf.cpu().numpy().view(np.uint8))
    return m.hexdigest()


def load_motion_module(model_name: str):
    model_path = os.path.join(MODEL_DIR, model_name)

    logger.info(f"Loading motion module {model_name}")
    mm_state_dict = load_torch_file(model_path)
    motion_module = MotionWrapper(model_name)

    parameters = calculate_parameters(mm_state_dict, "")
    usefp16 = model_management.should_use_fp16(model_params=parameters)
    if usefp16:
        logger.info("Using fp16, converting motion module to fp16")
        motion_module.half()
    offload_device = model_management.unet_offload_device()
    motion_module = motion_module.to(offload_device)
    motion_module.load_state_dict(mm_state_dict)

    return motion_module


def inject_motion_module_to_unet_legacy(unet, motion_module: MotionWrapper):
    logger.info(f"Injecting motion module into UNet input blocks.")
    for mm_idx, unet_idx in enumerate([1, 2, 4, 5, 7, 8, 10, 11]):
        mm_idx0, mm_idx1 = mm_idx // 2, mm_idx % 2
        unet.input_blocks[unet_idx].append(
            motion_module.down_blocks[mm_idx0].motion_modules[mm_idx1]
        )

    logger.info(f"Injecting motion module into UNet output blocks.")
    for unet_idx in range(12):
        mm_idx0, mm_idx1 = unet_idx // 3, unet_idx % 3
        if unet_idx % 2 == 2:
            unet.output_blocks[unet_idx].insert(
                -1, motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1]
            )
        else:
            unet.output_blocks[unet_idx].append(
                motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1]
            )


def eject_motion_module_from_unet_legacy(unet):
    logger.info(f"Ejecting motion module from UNet input blocks.")
    for unet_idx in [1, 2, 4, 5, 7, 8, 10, 11]:
        unet.input_blocks[unet_idx].pop(-1)

    logger.info(f"Ejecting motion module from UNet output blocks.")
    for unet_idx in range(12):
        if unet_idx % 2 == 2:
            unet.output_blocks[unet_idx].pop(-2)
        else:
            unet.output_blocks[unet_idx].pop(-1)


def inject_motion_module_to_unet(unet, motion_module: MotionWrapper):
    logger.info(f"Injecting motion module into UNet input blocks.")
    for mm_idx, unet_idx in enumerate([1, 2, 4, 5, 7, 8, 10, 11]):
        mm_idx0, mm_idx1 = mm_idx // 2, mm_idx % 2
        unet.input_blocks[unet_idx].append(
            motion_module.down_blocks[mm_idx0].motion_modules[mm_idx1]
        )

    logger.info(f"Injecting motion module into UNet output blocks.")
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


def eject_motion_module_from_unet(unet):
    logger.info(f"Ejecting motion module from UNet input blocks.")
    for unet_idx in [1, 2, 4, 5, 7, 8, 10, 11]:
        unet.input_blocks[unet_idx].pop(-1)

    logger.info(f"Ejecting motion module from UNet output blocks.")
    for unet_idx in range(12):
        if unet_idx % 3 == 2 and unet_idx != 11:
            unet.output_blocks[unet_idx].pop(-2)
        else:
            unet.output_blocks[unet_idx].pop(-1)


injectors = {
    "legacy": inject_motion_module_to_unet_legacy,
    "v1": inject_motion_module_to_unet,
}

ejectors = {
    "legacy": eject_motion_module_from_unet_legacy,
    "v1": eject_motion_module_from_unet,
}


class AnimateDiffLoaderLegacy:
    def __init__(self) -> None:
        self.version = "legacy"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "model_name": (get_available_models(),),
                "width": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 8}),
                "frame_number": (
                    "INT",
                    {"default": 16, "min": 2, "max": 24, "step": 1},
                ),
            },
            "optional": {
                "init_latent": ("LATENT",),
            },
        }

    @classmethod
    def IS_CHANGED(s, model: ModelPatcher):
        unet = model.model.diffusion_model
        return calculate_model_hash(unet) not in injected_model_hashs

    RETURN_TYPES = ("MODEL", "LATENT")
    CATEGORY = "Animate Diff"
    FUNCTION = "inject_motion_modules"

    def inject_motion_modules(
        self,
        model: ModelPatcher,
        model_name: str,
        width: int,
        height: int,
        frame_number=16,
        init_latent: Dict[str, torch.Tensor] = None,
    ):
        model = model.clone()

        if model_name not in motion_modules:
            motion_modules[model_name] = load_motion_module(model_name)

        motion_module = motion_modules[model_name]
        unet = model.model.diffusion_model
        unet_hash = calculate_model_hash(unet)

        need_inject = unet_hash not in injected_model_hashs

        if unet_hash in injected_model_hashs:
            (mm_type, version) = injected_model_hashs[unet_hash]
            if version != self.version or mm_type != motion_module.mm_type:
                # injected by another motion module, unload first
                logger.info(f"Ejecting motion module {mm_type} version {version}.")
                ejectors[version](unet)
                GroupNorm32.forward = groupnorm32_original_forward
                need_inject = True
            else:
                logger.info(f"Motion module already injected, skipping injection.")

        if need_inject:
            logger.info(f"Injecting motion module {model_name} version {self.version}.")
            injectors[self.version](unet, motion_module)
            unet_hash = calculate_model_hash(unet)
            injected_model_hashs[unet_hash] = (motion_module.mm_type, self.version)

            logger.info(f"Hacking GroupNorm32 forward function.")
            GroupNorm32.forward = groupnorm32_mm_forward

        if init_latent is None:
            latent = torch.zeros([frame_number, 4, height // 8, width // 8]).cpu()
        else:
            # clone value of first frame
            latent = init_latent["samples"][:1, :, :, :].clone().cpu()
            # repeat for all frames
            latent = latent.repeat(frame_number, 1, 1, 1)

        return (model, {"samples": latent})


class MotionModuleLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_available_models(),),
            },
        }

    RETURN_TYPES = ("MOTION_MODULE",)
    CATEGORY = "Animate Diff"
    FUNCTION = "load_motion_module"

    def load_motion_module(
        self,
        model_name: str,
    ):
        if not model_name in motion_modules is None:
            motion_modules[model_name] = load_motion_module(model_name)

        return (motion_modules[model_name],)


class AnimateDiffLoader:
    def __init__(self) -> None:
        self.version = "v1"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "init_latent": ("LATENT",),
                "model_name": (get_available_models(),),
                "frame_number": (
                    "INT",
                    {"default": 16, "min": 2, "max": 24, "step": 1},
                ),
            },
        }

    @classmethod
    def IS_CHANGED(s, model: ModelPatcher, _):
        unet = model.model.diffusion_model
        return calculate_model_hash(unet) not in injected_model_hashs

    RETURN_TYPES = ("MODEL", "LATENT")
    CATEGORY = "Animate Diff"
    FUNCTION = "inject_motion_modules"

    def inject_motion_modules(
        self,
        model: ModelPatcher,
        init_latent: Dict[str, torch.Tensor],
        model_name: str,
        frame_number=16,
    ):
        if model_name not in motion_modules:
            motion_modules[model_name] = load_motion_module(model_name)

        motion_module = motion_modules[model_name]

        model = model.clone()
        unet = model.model.diffusion_model
        unet_hash = calculate_model_hash(unet)
        need_inject = unet_hash not in injected_model_hashs

        if unet_hash in injected_model_hashs:
            (mm_type, version) = injected_model_hashs[unet_hash]
            if version != self.version or mm_type != motion_module.mm_type:
                # injected by another motion module, unload first
                logger.info(f"Ejecting motion module {mm_type} version {version}.")
                ejectors[version](unet)
                GroupNorm32.forward = groupnorm32_original_forward
                need_inject = True
            else:
                logger.info(f"Motion module already injected, skipping injection.")

        if need_inject:
            logger.info(f"Injecting motion module {model_name} version {self.version}.")
            injectors[self.version](unet, motion_module)
            unet_hash = calculate_model_hash(unet)
            injected_model_hashs[unet_hash] = (motion_module.mm_type, self.version)

            logger.info(f"Hacking GroupNorm32 forward function.")
            GroupNorm32.forward = groupnorm32_mm_forward
        
        init_frames = len(init_latent["samples"])
        samples = init_latent["samples"][:init_frames, :, :, :].clone().cpu()
        
        if init_frames < frame_number:
            last_frame = samples[-1].unsqueeze(0)
            repeated_last_frames = last_frame.repeat(frame_number - init_frames, 1, 1, 1)
            samples = torch.cat((samples, repeated_last_frames), dim=0)

        return (model, {"samples": samples})


class AnimateDiffUnload:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",)}}

    @classmethod
    def IS_CHANGED(s, model: ModelPatcher):
        unet = model.model.diffusion_model
        return calculate_model_hash(unet) in injected_model_hashs

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Animate Diff"
    FUNCTION = "unload_motion_modules"

    def unload_motion_modules(self, model: ModelPatcher):
        model = model.clone()
        unet = model.model.diffusion_model
        model_hash = calculate_model_hash(unet)
        if model_hash in injected_model_hashs:
            (model_name, version) = injected_model_hashs[model_hash]
            logger.info(f"Ejecting motion module {model_name} version {version}.")
            ejectors[version](unet)
            GroupNorm32.forward = groupnorm32_original_forward
        else:
            logger.info(f"Motion module not injected, skip unloading.")

        return (model,)


class AnimateDiffCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "INT",
                    {"default": 8, "min": 1, "max": 24, "step": 1},
                ),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "save_image": (["Enabled", "Disabled"],),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "Animate Diff"
    FUNCTION = "generate_gif"

    def generate_gif(
        self,
        images,
        frame_rate: int,
        loop_count: int,
        save_image="Enabled",
        filename_prefix="AnimateDiff",
        prompt=None,
        extra_pnginfo=None,
    ):
        # convert images to numpy
        pil_images: List[Image.Image] = []
        for image in images:
            img = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            pil_images.append(img)

        # save image
        output_dir = (
            folder_paths.get_output_directory()
            if save_image == "Enabled"
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)

        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        # save first frame as png to keep metadata
        file = f"{filename}_{counter:05}_.png"
        file_path = os.path.join(full_output_folder, file)
        pil_images[0].save(
            file_path,
            pnginfo=metadata,
            compress_level=4,
        )

        # save gif
        file = f"{filename}_{counter:05}_.gif"
        file_path = os.path.join(full_output_folder, file)
        pil_images[0].save(
            file_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=round(1000 / frame_rate),
            loop=loop_count,
            compress_level=4,
        )

        print("Saved gif to", file_path, os.path.exists(file_path))

        previews = [
            {
                "filename": file,
                "subfolder": subfolder,
                "type": "output" if save_image == "Enabled" else "temp",
            }
        ]
        return {"ui": {"images": previews}}


NODE_CLASS_MAPPINGS = {
    "AnimateDiffLoader": AnimateDiffLoaderLegacy,
    "AnimateDiffLoader_v2": AnimateDiffLoader,
    "AnimateDiffUnload": AnimateDiffUnload,
    "AnimateDiffCombine": AnimateDiffCombine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimateDiffLoader": "[DEPRECATED] Animate Diff Loader Legacy",
    "AnimateDiffLoader_v2": "Animate Diff Loader",
    "AnimateDiffUnload": "Animate Diff Unload",
    "AnimateDiffCombine": "Animate Diff Combine",
}
