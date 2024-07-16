from typing import Union

import inspect
import torch
from torch import Tensor

import comfy.model_patcher
import comfy.samplers

from .utils_motion import extend_to_batch_size, prepare_mask_batch


################################################################################
# helpers for modifying model_options to apply cfg function patches;
# taken from comfy/model_patcher.py
def set_model_options_sampler_cfg_function(model_options: dict[str], sampler_cfg_function, disable_cfg1_optimization=False):
    if len(inspect.signature(sampler_cfg_function).parameters) == 3:
        model_options["sampler_cfg_function"] = lambda args: sampler_cfg_function(args["cond"], args["uncond"], args["cond_scale"]) #Old way
    else:
        model_options["sampler_cfg_function"] = sampler_cfg_function
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options

def set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=False):
    model_options["sampler_post_cfg_function"] = model_options.get("sampler_post_cfg_function", []) + [post_cfg_function]
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options
#-------------------------------------------------------------------------------


# this is a modified version of PerturbedAttentionGuidance from comfy_extras/nodes_pag.py
def perturbed_attention_guidance_patch(scale_multival: Union[float, Tensor]):
    unet_block = "middle"
    unet_block_id = 0

    def perturbed_attention(q, k, v, extra_options, mask=None):
        return v

    def post_cfg_function(args):
        model = args["model"]
        cond_pred: Tensor = args["cond_denoised"]
        cond = args["cond"]
        cfg_result = args["denoised"]
        sigma = args["sigma"]
        model_options = args["model_options"].copy()
        x = args["input"]

        if type(scale_multival) != Tensor and scale_multival == 0:
            return cfg_result
        
        scale = scale_multival
        if isinstance(scale, Tensor):
            scale = prepare_mask_batch(scale.to(cond_pred.dtype).to(cond_pred.device), cond_pred.shape)
            scale = extend_to_batch_size(scale, cond_pred.shape[0])

        # Replace Self-attention with PAG
        model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, perturbed_attention, "attn1", unet_block, unet_block_id)
        (pag,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)

        return cfg_result + (cond_pred - pag) * scale
    
    return post_cfg_function


# this is a modified version of RescaleCFG from comfy_extras/nodes_model_advanced.py
def rescale_cfg_patch(multiplier_multival: Union[float, Tensor]):
    def cfg_function(args):
        cond: Tensor = args["cond"]
        uncond = args["uncond"]
        cond_scale = args["cond_scale"]
        sigma = args["sigma"]
        sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
        x_orig = args["input"]

        #rescale cfg has to be done on v-pred model output
        x = x_orig / (sigma * sigma + 1.0)
        cond = ((x - (x_orig - cond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)
        uncond = ((x - (x_orig - uncond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)

        #rescalecfg
        x_cfg = uncond + cond_scale * (cond - uncond)
        ro_pos = torch.std(cond, dim=(1,2,3), keepdim=True)
        ro_cfg = torch.std(x_cfg, dim=(1,2,3), keepdim=True)

        multiplier = multiplier_multival
        if isinstance(multiplier, Tensor):
            multiplier = prepare_mask_batch(multiplier.to(cond.dtype).to(cond.device), cond.shape)
            multiplier = extend_to_batch_size(multiplier, cond.shape[0])

        x_rescaled = x_cfg * (ro_pos / ro_cfg)
        x_final = multiplier * x_rescaled + (1.0 - multiplier) * x_cfg

        return x_orig - (x - x_final * sigma / (sigma * sigma + 1.0) ** 0.5)
    
    return cfg_function
