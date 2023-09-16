import os
import sys
import json
import subprocess
import shutil
import hashlib
import torch
from torch import Tensor
from torch.nn.functional import group_norm
import numpy as np
from typing import Dict, List, Tuple
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from einops import rearrange

import folder_paths
import comfy.ldm.modules.diffusionmodules.openaimodel as openaimodel
import comfy.model_management as model_management
import comfy.sample as comfy_sample
from comfy.ldm.modules.attention import SpatialTransformer
from comfy.utils import load_torch_file, calculate_parameters
from comfy.sd import load_checkpoint_guess_config
from comfy.model_patcher import ModelPatcher
from .logger import logger
from .motion_module import MotionWrapper, VanillaTemporalModule
from .model_utils import Folders, get_available_models, get_full_path, BetaSchedules

#############################################
#### Code Injection #########################
MM_INJECTED_ATTR = "_mm_injected"

class InjectionParams:
    def __init__(self, video_length: int, unlimited_area_hack: bool) -> None:
        self.video_length = video_length
        self.unlimited_area_hack = unlimited_area_hack

def is_mm_injected_into_model(model: ModelPatcher):
    return hasattr(model.model.diffusion_model, MM_INJECTED_ATTR)

def get_mm_injected_params(model: ModelPatcher) -> InjectionParams:
    return getattr(model.model.diffusion_model, MM_INJECTED_ATTR)

def set_mm_injected_params(model: ModelPatcher, injection_params: InjectionParams):
    setattr(model.model.diffusion_model, MM_INJECTED_ATTR, injection_params)



orig_comfy_sample = comfy_sample.sample # wrapper will go around this to inject/eject GroupNorm hack
orig_maximum_batch_area = model_management.maximum_batch_area # allows for "unlimited area hack" to prevent halving of conds/unconds
orig_forward_timestep_embed = openaimodel.forward_timestep_embed # needed to account for VanillaTemporalModule
orig_groupnorm_forward = torch.nn.GroupNorm.forward # used to normalize latents to remove "flickering" of colors/brightness between frames


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

# inject forward_timestep embed
openaimodel.forward_timestep_embed = forward_timestep_embed


def unlimited_batch_area():
    return int(sys.maxsize)


def groupnorm_mm_factory(params: InjectionParams):
    def groupnorm_mm_forward(self, input: Tensor) -> Tensor:
        # axes_factor normalizes batch based on total conds and unconds passed in batch;
        # the conds and unconds per batch can change based on VRAM optimizations that may kick in
        axes_factor = input.size(0)//params.video_length

        input = rearrange(input, "(b f) c h w -> b c f h w", b=axes_factor)
        input = group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
        input = rearrange(input, "b c f h w -> (b f) c h w", b=axes_factor)
        return input
    return groupnorm_mm_forward


def sample_wrapper(model, *args, **kwargs):
    #previous_groupnorm_forward = torch.nn.GroupNorm.forward
    # check if model is currently injected
    if is_mm_injected_into_model(model):
        params = get_mm_injected_params(model)
        if params.unlimited_area_hack:
            logger.info(f"Hacking model_management.maximum_batch_area function.")
            model_management.maximum_batch_area = unlimited_batch_area
        logger.info(f"Hacking torch.nn.GroupNorm forward function.")
        torch.nn.GroupNorm.forward = groupnorm_mm_factory(params)
    try:
        return orig_comfy_sample(model, *args, **kwargs)
    except:
        raise
    finally:
        # maintain functions present prior to sampling
        model_management.maximum_batch_area = orig_maximum_batch_area
        torch.nn.GroupNorm.forward = orig_groupnorm_forward

# inject sample_wrapper to wrap original sample function
comfy_sample.sample = sample_wrapper


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
    motion_module = MotionWrapper(mm_state_dict=mm_state_dict, mm_type=model_name)

    parameters = calculate_parameters(mm_state_dict, "")
    usefp16 = model_management.should_use_fp16(model_params=parameters)
    if usefp16:
        logger.info("Using fp16, converting motion module to fp16")
        motion_module.half()
    offload_device = model_management.unet_offload_device()
    motion_module = motion_module.to(offload_device)
    motion_module.load_state_dict(mm_state_dict)

    return motion_module


def inject_motion_module_to_unet_legacy(unet: openaimodel.UNetModel, motion_module: MotionWrapper, injection_params: InjectionParams):
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
    
    if motion_module.mid_block is not None:
        logger.info(f"Injecting motion module into UNet middle blocks.")
        unet.middle_block.insert(-1, motion_module.mid_block.motion_modules[0]) # only 1 VanillaTemporalModule
    setattr(unet, MM_INJECTED_ATTR, injection_params)


def eject_motion_module_from_unet_legacy(unet: openaimodel.UNetModel):
    logger.info(f"Ejecting motion module from UNet input blocks.")
    for unet_idx in [1, 2, 4, 5, 7, 8, 10, 11]:
        unet.input_blocks[unet_idx].pop(-1)

    logger.info(f"Ejecting motion module from UNet output blocks.")
    for unet_idx in range(12):
        if unet_idx % 2 == 2:
            unet.output_blocks[unet_idx].pop(-2)
        else:
            unet.output_blocks[unet_idx].pop(-1)
        
    if len(unet.middle_block) > 3: # SD1.5 UNet has 3 expected middle_blocks - more means injected
        logger.info(f"Ejecting motion module from UNet middle blocks.")
        unet.middle_block.pop(-2)
    delattr(unet, MM_INJECTED_ATTR)


def inject_motion_module_to_unet(unet: openaimodel.UNetModel, motion_module: MotionWrapper, injection_params: InjectionParams):
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

    if motion_module.mid_block is not None:
        logger.info(f"Injecting motion module into UNet middle blocks.")
        unet.middle_block.insert(-1, motion_module.mid_block.motion_modules[0]) # only 1 VanillaTemporalModule
    setattr(unet, MM_INJECTED_ATTR, injection_params)


def eject_motion_module_from_unet(unet: openaimodel.UNetModel):
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
    delattr(unet, MM_INJECTED_ATTR)


class InjectorVersion:
    LEGACY = "legacy"
    V1_V2 = "v1/v2"


injectors = {
    InjectorVersion.LEGACY: inject_motion_module_to_unet_legacy,
    InjectorVersion.V1_V2: inject_motion_module_to_unet,
}

ejectors = {
    InjectorVersion.LEGACY: eject_motion_module_from_unet_legacy,
    InjectorVersion.V1_V2: eject_motion_module_from_unet,
}
#############################################
#############################################


class AnimateDiffLoaderLegacy:
    def __init__(self) -> None:
        self.version =  InjectorVersion.LEGACY

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latents": ("LATENT",),
                "model_name": (get_available_models(),),
                "unlimited_area_hack": ([False, True],),
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
            latents: Dict[str, torch.Tensor],
            model_name: str, unlimited_area_hack: bool
        ):
        if model_name not in motion_modules:
            motion_modules[model_name] = load_motion_module(model_name)

        motion_module = motion_modules[model_name]
        # check that latents don't exceed max frame size
        init_frames_len = len(latents["samples"])
        if init_frames_len > motion_module.encoding_max_len:
            # TODO: warning and cutoff frames instead of error
            raise ValueError(f"AnimateDiff model {model_name} has upper limit of {motion_module.encoding_max_len} frames, but received {init_frames_len} latents.")
        # set motion_module's video_length to match latent length
        motion_module.set_video_length(init_frames_len)

        model = model.clone()
        unet = model.model.diffusion_model
        unet_hash = calculate_model_hash(unet)
        need_inject = unet_hash not in injected_model_hashs

        injection_params = InjectionParams(
            video_length=init_frames_len,
            unlimited_area_hack=unlimited_area_hack,
        )

        if unet_hash in injected_model_hashs:
            (mm_type, version) = injected_model_hashs[unet_hash]
            if version != self.version or mm_type != motion_module.mm_type:
                # injected by another motion module, unload first
                logger.info(f"Ejecting motion module {mm_type} version {version} - {motion_module.version}.")
                ejectors[version](unet)
                need_inject = True
            else:
                logger.info(f"Motion module already injected, only injecting params.")
                set_mm_injected_params(model, injection_params)

        if need_inject:
            logger.info(f"Injecting motion module {model_name} version {self.version} - {motion_module.version}.")
            
            injectors[self.version](unet, motion_module, injection_params)
            unet_hash = calculate_model_hash(unet)
            injected_model_hashs[unet_hash] = (motion_module.mm_type, self.version)

        return (model, latents)


class AnimateDiffLoader:
    def __init__(self) -> None:
        self.version = InjectorVersion.V1_V2

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latents": ("LATENT",),
                "model_name": (get_available_models(),),
                "unlimited_area_hack": ([False, True],),
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
        latents: Dict[str, torch.Tensor],
        model_name: str, unlimited_area_hack: bool
    ):
        if model_name not in motion_modules:
            motion_modules[model_name] = load_motion_module(model_name)

        motion_module = motion_modules[model_name]
        # check that latents don't exceed max frame size
        init_frames_len = len(latents["samples"])
        if init_frames_len > motion_module.encoding_max_len:
            # TODO: warning and cutoff frames instead of error
            raise ValueError(f"AnimateDiff model {model_name} has upper limit of {motion_module.encoding_max_len} frames, but received {init_frames_len} latents.")
        # set motion_module's video_length to match latent length
        motion_module.set_video_length(init_frames_len)

        model = model.clone()
        unet = model.model.diffusion_model
        unet_hash = calculate_model_hash(unet)
        need_inject = unet_hash not in injected_model_hashs

        injection_params = InjectionParams(
            video_length=init_frames_len,
            unlimited_area_hack=unlimited_area_hack,
        )

        if unet_hash in injected_model_hashs:
            (mm_type, version) = injected_model_hashs[unet_hash]
            if version != self.version or mm_type != motion_module.mm_type:
                # injected by another motion module, unload first
                logger.info(f"Ejecting motion module {mm_type} version {version} - {motion_module.version}.")
                ejectors[version](unet)
                need_inject = True
            else:
                logger.info(f"Motion module already injected, only injecting params.")
                set_mm_injected_params(model, injection_params)

        if need_inject:
            logger.info(f"Injecting motion module {model_name} version {motion_module.version}.")
            
            injectors[self.version](unet, motion_module, injection_params)
            unet_hash = calculate_model_hash(unet)
            injected_model_hashs[unet_hash] = (motion_module.mm_type, self.version)

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
            # just in case (is done automatically on mm eject anyway)
            torch.nn.GroupNorm.forward = orig_groupnorm_forward
            model_management.maximum_batch_area = orig_maximum_batch_area
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
                "save_image": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "pingpong": ("BOOLEAN", {"default": False}),
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
        save_image=True,
        filename_prefix="AnimateDiff",
        pingpong=False,
        prompt=None,
        extra_pnginfo=None,
    ):
        # convert images to numpy
        pil_images: List[Image.Image] = []
        for image in images:
            img = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            pil_images.append(img)
        if pingpong:
            pil_images += pil_images[::-1]
            
        # save image
        output_dir = (
            folder_paths.get_output_directory()
            if save_image
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
                "subfolder": "",
                "type": "output" if save_image else "temp",
            }
        ]
        print(previews)
        return {"ui": {"gif": previews}}

class AnimateDiffCombineVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "INT",
                    {"default": 8, "min": 1, "max": 24, "step": 1},
                ),
                "save_image": (["Enabled", "Disabled"],),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "pingpong": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "Animate Diff"
    FUNCTION = "generate_video"

    def generate_video(
        self,
        images,
        frame_rate: int,
        save_image="Enabled",
        filename_prefix="AnimateDiff",
        pingpong=False,
        prompt=None,
        extra_pnginfo=None,
    ):
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
        first_image = Image.fromarray(np.clip(255.0*images[0].cpu().numpy(),0,255).astype(np.uint8))
        first_image.save(
            file_path,
            pnginfo=metadata,
            compress_level=4,
        )
        
        # save webm
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise ProcessLookupError("Could not find ffmpeg")
        file_webm = f"{filename}_{counter:05}_.webm"
        file_path = os.path.join(full_output_folder, file_webm)
        dimensions = f"{first_image.width}x{first_image.height}"
        args = [ffmpeg_path, "-v", "panic", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                dimensions, "-r", str(frame_rate), "-i", "-", "-pix_fmt", "yuv420p", file_path]
        # convert images to numpy
        frames: List[Image.Image] = []
        for image in images:
            img = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            frames.append(img)
        if pingpong:
            frames += frames[::-1]

        with subprocess.Popen(args, stdin=subprocess.PIPE) as proc:
            for frame in frames:
                proc.stdin.write(frame.tobytes())
        print("Saved webm to", file_path, os.path.exists(file_path))


        previews = [
            {
                "filename": file_webm,
                "subfolder": subfolder,
                "type": "output" if save_image == "Enabled" else "temp",
            }
        ]
        return {"ui": {"video": previews}}


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


NODE_CLASS_MAPPINGS = {
    "CheckpointLoaderSimpleWithNoiseSelect": CheckpointLoaderSimpleWithNoiseSelect,
    "AnimateDiffLoaderV1": AnimateDiffLoader,
    "ADE_AnimateDiffUnload": AnimateDiffUnload,
    "ADE_AnimateDiffCombine": AnimateDiffCombine,
    "ADE_AnimateDiffCombineVideo": AnimateDiffCombineVideo,
    # AnimateDiff-specific
    "ADE_AnimateDiffLoaderLegacy": AnimateDiffLoaderLegacy,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointLoaderSimpleWithNoiseSelect": "Load Checkpoint w/ Noise Select",
    "AnimateDiffLoaderV1": "AnimateDiff Loader",
    "ADE_AnimateDiffUnload": "AnimateDiff Unload",
    "ADE_AnimateDiffCombine": "AnimateDiff Combine",
    "ADE_AnimateDiffCombineVideo": "AnimateDiff Combine Video",
    # AnimateDiff-specific
    "ADE_AnimateDiffLoaderLegacy": "[DEPRECATED] AnimateDiff Loader Legacy",
    
}
