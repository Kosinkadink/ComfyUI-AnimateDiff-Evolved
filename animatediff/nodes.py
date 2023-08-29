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
from comfy.sd import VAE, load_checkpoint_guess_config
from comfy.model_patcher import ModelPatcher

from .logger import logger
from .motion_module import MotionWrapper, VanillaTemporalModule
from .model_utils import Folders, get_available_models, get_full_path, BetaSchedules


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
    model_path = get_full_path(Folders.MODELS, model_name)

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
        # set motion_module's video_length to match latent length
        motion_module.set_video_length(frame_number)

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


class AnimateDiffLoader:
    def __init__(self) -> None:
        self.version = "v1"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "model_name": (get_available_models(),),
                "latents": ("LATENT",),
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
        model_name: str,
        latents: Dict[str, torch.Tensor],
    ):
        if model_name not in motion_modules:
            motion_modules[model_name] = load_motion_module(model_name)

        motion_module = motion_modules[model_name]
        # check that latents don't exceed max frame size
        init_frames = len(latents["samples"])
        if init_frames > 24:
            raise ValueError(f"AnimateDiff has upper limit of 24 frames, but received {init_frames} latents.")
        # set motion_module's video_length to match latent length
        motion_module.set_video_length(init_frames)

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

        return (model, latents)


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


class CheckpointLoaderSimpleWithNoiseSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "beta_schedule": (BetaSchedules.ALIAS_LIST, )
            },
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "Animate Diff"

    def load_checkpoint(self, ckpt_name, beta_schedule, output_vae=True, output_clip=True):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        # register chosen beta schedule on model - convert to beta_schedule name recognized by ComfyUI
        beta_schedule_name = BetaSchedules.to_name(beta_schedule)
        out[0].model.register_schedule(given_betas=None, beta_schedule=beta_schedule_name, timesteps=1000, linear_start=0.00085, linear_end=0.012, cosine_s=8e-3)
        return out


def decode_injected(self, samples_in):
    self.first_stage_model = self.first_stage_model.to(self.device)
    try:
        print("$$$$ in decode_injected!")
        memory_used = (2562 * samples_in.shape[2] * samples_in.shape[3] * 64) * 1.7
        model_management.free_memory(memory_used, self.device)
        free_memory = model_management.get_free_memory(self.device)
        batch_number = int(free_memory / memory_used)
        batch_number = max(1, batch_number)

        latent_length = samples_in.shape[0]

        #samples_in = 1 / 0.18215 * samples_in
        #samples_in = rearrange(samples_in, "(b f) c h w -> b c f h w")
        #x = groupnorm32_original_forward(self, x)
        

        pixel_samples = torch.empty((samples_in.shape[0], 3, round(samples_in.shape[2] * 8), round(samples_in.shape[3] * 8)), device="cpu")
        for x in range(0, samples_in.shape[0], batch_number):
            samples = samples_in[x:x+batch_number].to(self.vae_dtype).to(self.device)
            pixel_samples[x:x+batch_number] = torch.clamp((self.first_stage_model.decode(samples) + 1.0) / 2.0, min=0.0, max=1.0).cpu().float()

    except model_management.OOM_EXCEPTION as e:
        print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
        raise ValueError("Ran out of memory! ")
        pixel_samples = self.decode_tiled_(samples_in)

    #pixel_samples = (pixel_samples / 2 + 0.5).clamp(0, 1).cpu().float()

    # batch_images = []
    # for i, x_sample in enumerate(pixel_samples):
    #     x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
    #     x_sample = x_sample.astype(np.uint8)
    #     image = Image.fromarray(x_sample)
    #     #image = images.resize_image(0, image, target_width, target_height, upscaler_name=self.hr_upscaler)
    #     image = np.array(image).astype(np.float32) / 255.0
    #     image = np.moveaxis(image, 2, 0)
    #     batch_images.append(image)
    
    # pixel_samples = torch.from_numpy(np.array(batch_images))
    # pixel_samples = pixel_samples.to(self.device)
    # pixel_samples = 2. * pixel_samples - 1.

    self.first_stage_model = self.first_stage_model.to(self.offload_device)
    pixel_samples = pixel_samples.cpu().movedim(1,-1)
    #samples_in = rearrange(samples_in, "b c f h w -> (b f) c h w", f=latent_length)
    return pixel_samples


class VAEDecodeNormalized:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "Animate Diff"

    def inject_decode(self, vae: VAE):
        vae.decode = decode_injected.__get__(vae, type(vae))

    def eject_decode(self, vae: VAE, original_decode):
        vae.decode = original_decode.__get__(vae, type(vae))

    def decode(self, vae: VAE, samples):
        original_decode = vae.decode
        try:
            self.inject_decode(vae)
            return (vae.decode(samples["samples"]), )
        finally:
            self.eject_decode(vae, original_decode)


NODE_CLASS_MAPPINGS = {
    "CheckpointLoaderSimpleWithNoiseSelect": CheckpointLoaderSimpleWithNoiseSelect,
    # "VAEDecodeNormalized": VAEDecodeNormalized,
    # AnimateDiff-specific
    "AnimateDiffLoaderLegacy": AnimateDiffLoaderLegacy,
    "AnimateDiffLoaderV1": AnimateDiffLoader,
    "AnimateDiffUnload": AnimateDiffUnload,
    "AnimateDiffCombine": AnimateDiffCombine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointLoaderSimpleWithNoiseSelect": "Load Checkpoint w/ Noise Select",
    # "VAEDecodeNormalized": "VAE Decode (Normalized)",
    # AnimateDiff-specific
    "AnimateDiffLoaderLegacy": "[DEPRECATED] AnimateDiff Loader Legacy",
    "AnimateDiffLoaderV1": "AnimateDiff Loader",
    "AnimateDiffUnload": "AnimateDiff Unload",
    "AnimateDiffCombine": "AnimateDiff Combine",
}
