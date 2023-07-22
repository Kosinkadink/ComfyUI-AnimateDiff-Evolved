import os
import json
import hashlib
import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from typing import Dict, List

import folder_paths
import comfy.ldm.modules.diffusionmodules.openaimodel as openaimodel
import comfy.model_management as model_management
from comfy.ldm.modules.attention import SpatialTransformer
from comfy.utils import load_torch_file
from comfy.sd import ModelPatcher, calculate_parameters

from .logger import logger
from .motion_module import MotionWrapper, VanillaTemporalModule
from .model_utils import MODEL_DIR, get_available_models


orig_forward_timestep_embed = openaimodel.forward_timestep_embed


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


openaimodel.forward_timestep_embed = forward_timestep_embed

motion_module: MotionWrapper = None


class AnimateDiffLoader:
    def __init__(self) -> None:
        self.last_injected_model_hash = set()

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
        model_path = os.path.join(MODEL_DIR, model_name)

        global motion_module
        if motion_module is None:
            logger.info(f"Loading motion module {model_name} from {model_path}")
            mm_state_dict = load_torch_file(model_path)
            motion_module = MotionWrapper()

            parameters = calculate_parameters(mm_state_dict, "")
            usefp16 = model_management.should_use_fp16(model_params=parameters)
            if usefp16:
                print("Using fp16, converting motion module to fp16")
                motion_module.half()
            offload_device = model_management.unet_offload_device()
            motion_module = motion_module.to(offload_device)
            motion_module.load_state_dict(mm_state_dict)

        unet = model.model.diffusion_model
        if self.calculate_model_hash(unet) in self.last_injected_model_hash:
            logger.info(f"Motion module already injected, skipping injection.")
        else:
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
                        -1, motion_module.up_blocks[mm_idx0].motion_modules[mm_idx]
                    )
                else:
                    unet.output_blocks[unet_idx].append(
                        motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1]
                    )

            self.last_injected_model_hash.add(self.calculate_model_hash(unet))

        if init_latent is None:
            latent = torch.zeros([frame_number, 4, width // 8, height // 8]).cpu()
        else:
            # clone value of first frame
            latent = init_latent["samples"].clone().cpu()
            # repeat for all frames
            latent = latent.repeat(frame_number, 1, 1, 1)

        return (model, {"samples": latent})

    def calculate_model_hash(self, unet):
        t = unet.input_blocks[1]
        m = hashlib.sha256()
        for buf in t.buffers():
            m.update(buf.numpy().view(np.uint8))
        return m.hexdigest()


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

        previews = [
            {
                "filename": file,
                "subfolder": subfolder,
                "type": "output" if save_image == "Enabled" else "temp",
            }
        ]
        return {"ui": {"images": previews}}


NODE_CLASS_MAPPINGS = {
    "AnimateDiffLoader": AnimateDiffLoader,
    "AnimateDiffCombine": AnimateDiffCombine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimateDiffLoader": "Animate Diff Loader",
    "AnimateDiffCombine": "Animate Diff Combine",
}
