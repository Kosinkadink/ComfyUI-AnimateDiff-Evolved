from typing import Union

import torch
from torch import Tensor

import folder_paths
import nodes as comfy_nodes
from comfy.model_patcher import ModelPatcher
import comfy.model_patcher
import comfy.samplers
from comfy.sd import load_checkpoint_guess_config

from .logger import logger
from .utils_model import BetaSchedules
from .utils_motion import extend_to_batch_size, prepare_mask_batch
from .model_injection import get_vanilla_model_patcher
from .cfg_extras import perturbed_attention_guidance_patch, rescale_cfg_patch


class AnimateDiffUnload:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/extras"
    FUNCTION = "unload_motion_modules"

    def unload_motion_modules(self, model: ModelPatcher):
        # return model clone with ejected params
        #model = eject_params_from_model(model)
        model = get_vanilla_model_patcher(model)
        return (model.clone(),)


class CheckpointLoaderSimpleWithNoiseSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "beta_schedule": (BetaSchedules.ALIAS_LIST, {"default": BetaSchedules.USE_EXISTING}, )
            },
            "optional": {
                "use_custom_scale_factor": ("BOOLEAN", {"default": False}),
                "scale_factor": ("FLOAT", {"default": 0.18215, "min": 0.0, "max": 1.0, "step": 0.00001})
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "Animate Diff 🎭🅐🅓/extras"

    def load_checkpoint(self, ckpt_name, beta_schedule, output_vae=True, output_clip=True, use_custom_scale_factor=False, scale_factor=0.18215):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        # register chosen beta schedule on model - convert to beta_schedule name recognized by ComfyUI
        new_model_sampling = BetaSchedules.to_model_sampling(beta_schedule, out[0])
        if new_model_sampling is not None:
            out[0].model.model_sampling = new_model_sampling
        if use_custom_scale_factor:
            out[0].model.latent_format.scale_factor = scale_factor
        return out


class EmptyLatentImageLarge:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 64, "max": comfy_nodes.MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 512, "min": 64, "max": comfy_nodes.MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 262144})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "Animate Diff 🎭🅐🅓/extras"

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return ({"samples":latent}, )


class PerturbedAttentionGuidanceMultival:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale_multival": ("MULTIVAL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "Animate Diff 🎭🅐🅓/extras"

    def patch(self, model: ModelPatcher, scale_multival: Union[float, Tensor]):
        m = model.clone()
        m.set_model_sampler_post_cfg_function(perturbed_attention_guidance_patch(scale_multival))

        return (m,)


class RescaleCFGMultival:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "mult_multival": ("MULTIVAL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "Animate Diff 🎭🅐🅓/extras"

    def patch(self, model: ModelPatcher, mult_multival: Union[float, Tensor]):
        m = model.clone()
        m.set_model_sampler_cfg_function(rescale_cfg_patch(mult_multival))
        return (m, )
