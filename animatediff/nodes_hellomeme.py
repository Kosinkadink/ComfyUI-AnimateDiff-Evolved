from typing import Union
import torch
from torch import Tensor

from comfy.sd import VAE
from comfy.model_patcher import ModelPatcher
import comfy.model_management


from .adapter_hellomeme import (HMRefConst, HMModelPatcher, HMRefAttachment, load_hmreferenceadapter,
                                create_hmref_attachment,
                                create_HM_forward_timestep_embed_patch)
from .model_injection import ModelPatcherHelper
from .sampling import outer_sample_wrapper
from .utils_model import get_available_motion_models


class TestHMRefNetInjection:
    NodeID = "ADE_TestHMRefNetInjection"
    NodeName = "Test HMRefNetInjection"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "hmref": (get_available_motion_models(),),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/HelloMeme"
    FUNCTION = "inject_hmref"

    def inject_hmref(self, model: ModelPatcher, image: Tensor, vae: VAE, 
                     hmref: str):
        model = model.clone()

        mp_hmref: HMModelPatcher = load_hmreferenceadapter(hmref)
        model.set_additional_models(HMRefConst.HMREF, [mp_hmref])
        model.set_model_forward_timestep_embed_patch(create_HM_forward_timestep_embed_patch())
        model.set_injections(HMRefConst.HMREF, [mp_hmref.model.create_injector()])
        create_hmref_attachment(model, HMRefAttachment(image=image, vae=vae))
        
        helper = ModelPatcherHelper(model)
        helper.set_outer_sample_wrapper(outer_sample_wrapper)
        del helper

        return (model,)
