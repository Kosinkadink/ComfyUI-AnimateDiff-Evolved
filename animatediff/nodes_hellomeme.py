from typing import Union
import torch

from comfy.model_patcher import ModelPatcher
import comfy.model_management


from .adapter_hellomeme import HMReferenceAdapter, create_HM_forward_timestep_embed_patch


class TestHMRefNetInjection:
    NodeID = "ADE_TestHMRefNetInjection"
    NodeName = "Test HMRefNetInjection"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/HelloMeme"
    FUNCTION = "inject_hmref"

    def inject_hmref(self, model: ModelPatcher):
        model = model.clone()

        hmref = HMReferenceAdapter()
        hmref.to(comfy.model_management.unet_dtype())
        hmref.to(comfy.model_management.unet_offload_device())
        mp_hmref = ModelPatcher(model=hmref,
                                load_device=comfy.model_management.get_torch_device(),
                                offload_device=comfy.model_management.unet_offload_device())
        model.set_additional_models("ADE_HMREF", [mp_hmref])
        model.set_model_forward_timestep_embed_patch(create_HM_forward_timestep_embed_patch())
        model.set_injections("ADE_HMREF", [hmref.create_injector()])

        return (model,)
