from pathlib import Path
import torch

import comfy.sample as comfy_sample
from comfy.model_patcher import ModelPatcher

from .logger import logger
from .utils_model import BetaSchedules, get_available_motion_loras, get_available_motion_models, get_motion_lora_path
from .motion_lora import MotionLoraInfo, MotionLoraList
from .model_injection import InjectionParams, ModelPatcherAndInjector, MotionModelSettings, load_motion_module
from .sample_settings import SampleSettings, SeedNoiseGeneration
from .sampling import motion_sample_factory

from .nodes_gen1 import AnimateDiffLoaderWithContext
from .nodes_gen2 import UseEvolvedSamplingNode, ApplyAnimateDiffModelNode, ApplyAnimateDiffModelBasicNode, LoadAnimateDiffModelNode, ADKeyframeNode
from .nodes_multival import MultivalDynamicNode, MultivalFloatNode, MultivalScaledMaskNode
from .nodes_sample import FreeInitOptionsNode, NoiseLayerAddWeightedNode, SampleSettingsNode, NoiseLayerAddNode, NoiseLayerReplaceNode, IterationOptionsNode
from .nodes_context import (LoopedUniformContextOptionsNode, LoopedUniformViewOptionsNode, StandardUniformContextOptionsNode, StandardStaticContextOptionsNode, BatchedContextOptionsNode,
                            StandardStaticViewOptionsNode, StandardUniformViewOptionsNode, ViewAsContextOptionsNode)
from .nodes_extras import AnimateDiffUnload, EmptyLatentImageLarge, CheckpointLoaderSimpleWithNoiseSelect
from .nodes_experimental import AnimateDiffModelSettingsSimple, AnimateDiffModelSettingsAdvanced, AnimateDiffModelSettingsAdvancedAttnStrengths
from .nodes_deprecated import AnimateDiffLoader_Deprecated, AnimateDiffLoaderAdvanced_Deprecated, AnimateDiffCombine_Deprecated

# override comfy_sample.sample with animatediff-support version
comfy_sample.sample = motion_sample_factory(comfy_sample.sample)
comfy_sample.sample_custom = motion_sample_factory(comfy_sample.sample_custom, is_custom=True)


class AnimateDiffModelSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min_motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "max_motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
            },
            "optional": {
                "mask_motion_scale": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("MOTION_MODEL_SETTINGS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/gen1 nodes ①/motion settings"
    FUNCTION = "get_motion_model_settings"

    def get_motion_model_settings(self, mask_motion_scale: torch.Tensor=None, min_motion_scale: float=1.0, max_motion_scale: float=1.0):
        motion_model_settings = MotionModelSettings(
            mask_attn_scale=mask_motion_scale,
            mask_attn_scale_min=min_motion_scale,
            mask_attn_scale_max=max_motion_scale,
            )

        return (motion_model_settings,)


class AnimateDiffLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (get_available_motion_loras(),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            },
            "optional": {
                "prev_motion_lora": ("MOTION_LORA",),
            }
        }
    
    RETURN_TYPES = ("MOTION_LORA",)
    CATEGORY = "Animate Diff 🎭🅐🅓"
    FUNCTION = "load_motion_lora"

    def load_motion_lora(self, lora_name: str, strength: float, prev_motion_lora: MotionLoraList=None):
        if prev_motion_lora is None:
            prev_motion_lora = MotionLoraList()
        else:
            prev_motion_lora = prev_motion_lora.clone()
        # check if motion lora with name exists
        lora_path = get_motion_lora_path(lora_name)
        if not Path(lora_path).is_file():
            raise FileNotFoundError(f"Motion lora with name '{lora_name}' not found.")
        # create motion lora info to be loaded in AnimateDiff Loader
        lora_info = MotionLoraInfo(name=lora_name, strength=strength)
        prev_motion_lora.add_lora(lora_info)

        return (prev_motion_lora,)


NODE_CLASS_MAPPINGS = {
    # Unencapsulated/Gen2
    "ADE_UseEvolvedSampling": UseEvolvedSamplingNode,
    "ADE_ApplyAnimateDiffModel": ApplyAnimateDiffModelNode,
    "ADE_ApplyAnimateDiffModelSimple": ApplyAnimateDiffModelBasicNode,
    "ADE_LoadAnimateDiffModel": LoadAnimateDiffModelNode,
    "ADE_AnimateDiffLoRALoader": AnimateDiffLoraLoader,
    "ADE_AnimateDiffSamplingSettings": SampleSettingsNode,
    "ADE_AnimateDiffKeyframe": ADKeyframeNode,
    # Multival Nodes
    "ADE_MultivalDynamic": MultivalDynamicNode,
    "ADE_MultivalScaledMask": MultivalScaledMaskNode,
    # Context Opts
    "ADE_StandardStaticContextOptions": StandardStaticContextOptionsNode,
    "ADE_StandardUniformContextOptions": StandardUniformContextOptionsNode,
    "ADE_AnimateDiffUniformContextOptions": LoopedUniformContextOptionsNode,
    "ADE_ViewsOnlyContextOptions": ViewAsContextOptionsNode,
    "ADE_BatchedContextOptions": BatchedContextOptionsNode,
    # View Opts
    "ADE_StandardStaticViewOptions": StandardStaticViewOptionsNode,
    "ADE_StandardUniformViewOptions": StandardUniformViewOptionsNode,
    "ADE_LoopedUniformViewOptions": LoopedUniformViewOptionsNode,
    # Iteration Opts
    "ADE_IterationOptsDefault": IterationOptionsNode,
    "ADE_IterationOptsFreeInit": FreeInitOptionsNode,
    # Noise Layer Nodes
    "ADE_NoiseLayerAdd": NoiseLayerAddNode,
    "ADE_NoiseLayerAddWeighted": NoiseLayerAddWeightedNode,
    "ADE_NoiseLayerReplace": NoiseLayerReplaceNode,
    # Extras Nodes
    "ADE_AnimateDiffUnload": AnimateDiffUnload,
    "ADE_EmptyLatentImageLarge": EmptyLatentImageLarge,
    "CheckpointLoaderSimpleWithNoiseSelect": CheckpointLoaderSimpleWithNoiseSelect,
    # Gen1 Nodes
    "ADE_AnimateDiffLoaderWithContext": AnimateDiffLoaderWithContext,
    "ADE_AnimateDiffModelSettings_Release": AnimateDiffModelSettings,
    "ADE_AnimateDiffModelSettingsSimple": AnimateDiffModelSettingsSimple,
    "ADE_AnimateDiffModelSettings": AnimateDiffModelSettingsAdvanced,
    "ADE_AnimateDiffModelSettingsAdvancedAttnStrengths": AnimateDiffModelSettingsAdvancedAttnStrengths,
    # Deprecated Nodes
    "AnimateDiffLoaderV1": AnimateDiffLoader_Deprecated,
    "ADE_AnimateDiffLoaderV1Advanced": AnimateDiffLoaderAdvanced_Deprecated,
    "ADE_AnimateDiffCombine": AnimateDiffCombine_Deprecated,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    # Unencapsulated/Gen2
    "ADE_UseEvolvedSampling": "Use Evolved Sampling 🎭🅐🅓②",
    "ADE_ApplyAnimateDiffModel": "Apply AnimateDiff Model (Adv.) 🎭🅐🅓②",
    "ADE_ApplyAnimateDiffModelSimple": "Apply AnimateDiff Model (Basic) 🎭🅐🅓②",
    "ADE_LoadAnimateDiffModel": "Load AnimateDiff Model 🎭🅐🅓②",
    "ADE_AnimateDiffLoRALoader": "Load AnimateDiff LoRA 🎭🅐🅓",
    "ADE_AnimateDiffSamplingSettings": "Sample Settings 🎭🅐🅓",
    "ADE_AnimateDiffKeyframe": "AnimateDiff Keyframe 🎭🅐🅓",
    # Multival Nodes
    "ADE_MultivalDynamic": "Multival Dynamic 🎭🅐🅓",
    "ADE_MultivalScaledMask": "Multival Scaled Mask 🎭🅐🅓",
    # Context Opts
    "ADE_StandardStaticContextOptions": "Context Options◆Standard Static 🎭🅐🅓",
    "ADE_StandardUniformContextOptions": "Context Options◆Standard Uniform 🎭🅐🅓",
    "ADE_AnimateDiffUniformContextOptions": "Context Options◆Looped Uniform 🎭🅐🅓",
    "ADE_ViewsOnlyContextOptions": "Context Options◆Views Only [VRAM⇈] 🎭🅐🅓",
    "ADE_BatchedContextOptions": "Context Options◆Batched [Non-AD] 🎭🅐🅓",
    # View Opts
    "ADE_StandardStaticViewOptions": "View Options◆Standard Static 🎭🅐🅓",
    "ADE_StandardUniformViewOptions": "View Options◆Standard Uniform 🎭🅐🅓",
    "ADE_LoopedUniformViewOptions": "View Options◆Looped Uniform 🎭🅐🅓",
    # Iteration Opts
    "ADE_IterationOptsDefault": "Default Iteration Options 🎭🅐🅓",
    "ADE_IterationOptsFreeInit": "FreeInit Iteration Options 🎭🅐🅓",
    # Noise Layer Nodes
    "ADE_NoiseLayerAdd": "Noise Layer [Add] 🎭🅐🅓",
    "ADE_NoiseLayerAddWeighted": "Noise Layer [Add Weighted] 🎭🅐🅓",
    "ADE_NoiseLayerReplace": "Noise Layer [Replace] 🎭🅐🅓",
    # Extras Nodes
    "ADE_AnimateDiffUnload": "AnimateDiff Unload 🎭🅐🅓",
    "ADE_EmptyLatentImageLarge": "Empty Latent Image (Big Batch) 🎭🅐🅓",
    "CheckpointLoaderSimpleWithNoiseSelect": "Load Checkpoint w/ Noise Select 🎭🅐🅓",
    # Gen1 Nodes
    "ADE_AnimateDiffLoaderWithContext": "AnimateDiff Loader 🎭🅐🅓①",
    "ADE_AnimateDiffModelSettings_Release": "Motion Model Settings 🎭🅐🅓①",
    "ADE_AnimateDiffModelSettingsSimple": "EXP Motion Model Settings (Simple) 🎭🅐🅓①",
    "ADE_AnimateDiffModelSettings": "EXP Motion Model Settings (Advanced) 🎭🅐🅓①",
    "ADE_AnimateDiffModelSettingsAdvancedAttnStrengths": "EXP Motion Model Settings (Adv. Attn) 🎭🅐🅓①",
    # Deprecated Nodes
    "AnimateDiffLoaderV1": "AnimateDiff Loader [DEPRECATED] 🎭🅐🅓",
    "ADE_AnimateDiffLoaderV1Advanced": "AnimateDiff Loader (Advanced) [DEPRECATED] 🎭🅐🅓",
    "ADE_AnimateDiffCombine": "DO NOT USE, USE VideoCombine from ComfyUI-VideoHelperSuite instead! AnimateDiff Combine [DEPRECATED, DO NOT USE] 🎭🅐🅓",
}
