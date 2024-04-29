import comfy.sample as comfy_sample

from .sampling import motion_sample_factory

from .nodes_gen1 import (AnimateDiffLoaderGen1, LegacyAnimateDiffLoaderWithContext)
from .nodes_gen2 import (UseEvolvedSamplingNode, ApplyAnimateDiffModelNode, ApplyAnimateDiffModelBasicNode, ADKeyframeNode,
                         LoadAnimateDiffModelNode)
from .nodes_animatelcmi2v import (ApplyAnimateLCMI2VModel, LoadAnimateLCMI2VModelNode, LoadAnimateDiffAndInjectI2VNode, UpscaleAndVaeEncode)
from .nodes_cameractrl import (LoadAnimateDiffModelWithCameraCtrl, ApplyAnimateDiffWithCameraCtrl, CameraCtrlADKeyframeNode, LoadCameraPoses,
                               CameraCtrlPoseBasic, CameraCtrlPoseCombo, CameraCtrlPoseAdvanced, CameraCtrlManualAppendPose,
                               CameraCtrlReplaceCameraParameters, CameraCtrlSetOriginalAspectRatio)
from .nodes_multival import MultivalDynamicNode, MultivalScaledMaskNode
from .nodes_conditioning import (MaskableLoraLoader, MaskableLoraLoaderModelOnly, MaskableSDModelLoader, MaskableSDModelLoaderModelOnly,
                                 SetModelLoraHook, SetClipLoraHook,
                                 CombineLoraHooks, CombineLoraHookFourOptional, CombineLoraHookEightOptional,
                                 PairedConditioningSetMaskHooked, ConditioningSetMaskHooked,
                                 PairedConditioningSetMaskAndCombineHooked, ConditioningSetMaskAndCombineHooked,
                                 PairedConditioningSetUnmaskedAndCombineHooked, ConditioningSetUnmaskedAndCombineHooked,
                                 ConditioningTimestepsNode, SetLoraHookKeyframes,
                                 CreateLoraHookKeyframe, CreateLoraHookKeyframeInterpolation, CreateLoraHookKeyframeFromStrengthList)
from .nodes_sample import (FreeInitOptionsNode, NoiseLayerAddWeightedNode, SampleSettingsNode, NoiseLayerAddNode, NoiseLayerReplaceNode, IterationOptionsNode,
                           CustomCFGNode, CustomCFGKeyframeNode)
from .nodes_sigma_schedule import (SigmaScheduleNode, RawSigmaScheduleNode, WeightedAverageSigmaScheduleNode, InterpolatedWeightedAverageSigmaScheduleNode, SplitAndCombineSigmaScheduleNode)
from .nodes_context import (LegacyLoopedUniformContextOptionsNode, LoopedUniformContextOptionsNode, LoopedUniformViewOptionsNode, StandardUniformContextOptionsNode, StandardStaticContextOptionsNode, BatchedContextOptionsNode,
                            StandardStaticViewOptionsNode, StandardUniformViewOptionsNode, ViewAsContextOptionsNode)
from .nodes_ad_settings import (AnimateDiffSettingsNode, ManualAdjustPENode, SweetspotStretchPENode, FullStretchPENode,
                                WeightAdjustAllAddNode, WeightAdjustAllMultNode, WeightAdjustIndivAddNode, WeightAdjustIndivMultNode,
                                WeightAdjustIndivAttnAddNode, WeightAdjustIndivAttnMultNode)
from .nodes_extras import AnimateDiffUnload, EmptyLatentImageLarge, CheckpointLoaderSimpleWithNoiseSelect
from .nodes_deprecated import (AnimateDiffLoader_Deprecated, AnimateDiffLoaderAdvanced_Deprecated, AnimateDiffCombine_Deprecated,
                               AnimateDiffModelSettings, AnimateDiffModelSettingsSimple, AnimateDiffModelSettingsAdvanced, AnimateDiffModelSettingsAdvancedAttnStrengths)
from .nodes_lora import AnimateDiffLoraLoader

from .logger import logger

# override comfy_sample.sample with animatediff-support version
comfy_sample.sample = motion_sample_factory(comfy_sample.sample)
comfy_sample.sample_custom = motion_sample_factory(comfy_sample.sample_custom, is_custom=True)


NODE_CLASS_MAPPINGS = {
    # Unencapsulated
    "ADE_AnimateDiffLoRALoader": AnimateDiffLoraLoader,
    "ADE_AnimateDiffSamplingSettings": SampleSettingsNode,
    "ADE_AnimateDiffKeyframe": ADKeyframeNode,
    # Multival Nodes
    "ADE_MultivalDynamic": MultivalDynamicNode,
    "ADE_MultivalScaledMask": MultivalScaledMaskNode,
    # Context Opts
    "ADE_StandardStaticContextOptions": StandardStaticContextOptionsNode,
    "ADE_StandardUniformContextOptions": StandardUniformContextOptionsNode,
    "ADE_LoopedUniformContextOptions": LoopedUniformContextOptionsNode,
    "ADE_ViewsOnlyContextOptions": ViewAsContextOptionsNode,
    "ADE_BatchedContextOptions": BatchedContextOptionsNode,
    "ADE_AnimateDiffUniformContextOptions": LegacyLoopedUniformContextOptionsNode, # Legacy
    # View Opts
    "ADE_StandardStaticViewOptions": StandardStaticViewOptionsNode,
    "ADE_StandardUniformViewOptions": StandardUniformViewOptionsNode,
    "ADE_LoopedUniformViewOptions": LoopedUniformViewOptionsNode,
    # Iteration Opts
    "ADE_IterationOptsDefault": IterationOptionsNode,
    "ADE_IterationOptsFreeInit": FreeInitOptionsNode,
    # Conditioning
    "ADE_RegisterLoraHook": MaskableLoraLoader,
    "ADE_RegisterLoraHookModelOnly": MaskableLoraLoaderModelOnly,
    "ADE_RegisterModelAsLoraHook": MaskableSDModelLoader,
    "ADE_RegisterModelAsLoraHookModelOnly": MaskableSDModelLoaderModelOnly,
    "ADE_CombineLoraHooks": CombineLoraHooks,
    "ADE_CombineLoraHooksFour": CombineLoraHookFourOptional,
    "ADE_CombineLoraHooksEight": CombineLoraHookEightOptional,
    "ADE_SetLoraHookKeyframe": SetLoraHookKeyframes,
    "ADE_AttachLoraHookToCLIP": SetClipLoraHook,
    "ADE_LoraHookKeyframe": CreateLoraHookKeyframe,
    "ADE_LoraHookKeyframeInterpolation": CreateLoraHookKeyframeInterpolation,
    "ADE_LoraHookKeyframeFromStrengthList": CreateLoraHookKeyframeFromStrengthList,
    "ADE_AttachLoraHookToConditioning": SetModelLoraHook,
    "ADE_PairedConditioningSetMask": PairedConditioningSetMaskHooked,
    "ADE_ConditioningSetMask": ConditioningSetMaskHooked,
    "ADE_PairedConditioningSetMaskAndCombine": PairedConditioningSetMaskAndCombineHooked,
    "ADE_ConditioningSetMaskAndCombine": ConditioningSetMaskAndCombineHooked,
    "ADE_PairedConditioningSetUnmaskedAndCombine": PairedConditioningSetUnmaskedAndCombineHooked,
    "ADE_ConditioningSetUnmaskedAndCombine": ConditioningSetUnmaskedAndCombineHooked,
    "ADE_TimestepsConditioning": ConditioningTimestepsNode,
    # Noise Layer Nodes
    "ADE_NoiseLayerAdd": NoiseLayerAddNode,
    "ADE_NoiseLayerAddWeighted": NoiseLayerAddWeightedNode,
    "ADE_NoiseLayerReplace": NoiseLayerReplaceNode,
    # AnimateDiff Settings
    "ADE_AnimateDiffSettings": AnimateDiffSettingsNode,
    "ADE_AdjustPESweetspotStretch": SweetspotStretchPENode,
    "ADE_AdjustPEFullStretch": FullStretchPENode,
    "ADE_AdjustPEManual": ManualAdjustPENode,
    "ADE_AdjustWeightAllAdd": WeightAdjustAllAddNode,
    "ADE_AdjustWeightAllMult": WeightAdjustAllMultNode,
    "ADE_AdjustWeightIndivAdd": WeightAdjustIndivAddNode,
    "ADE_AdjustWeightIndivMult": WeightAdjustIndivMultNode,
    "ADE_AdjustWeightIndivAttnAdd": WeightAdjustIndivAttnAddNode,
    "ADE_AdjustWeightIndivAttnMult": WeightAdjustIndivAttnMultNode,
    # Sample Settings
    "ADE_CustomCFG": CustomCFGNode,
    "ADE_CustomCFGKeyframe": CustomCFGKeyframeNode,
    "ADE_SigmaSchedule": SigmaScheduleNode,
    "ADE_RawSigmaSchedule": RawSigmaScheduleNode,
    "ADE_SigmaScheduleWeightedAverage": WeightedAverageSigmaScheduleNode,
    "ADE_SigmaScheduleWeightedAverageInterp": InterpolatedWeightedAverageSigmaScheduleNode,
    "ADE_SigmaScheduleSplitAndCombine": SplitAndCombineSigmaScheduleNode,
    # Extras Nodes
    "ADE_AnimateDiffUnload": AnimateDiffUnload,
    "ADE_EmptyLatentImageLarge": EmptyLatentImageLarge,
    "CheckpointLoaderSimpleWithNoiseSelect": CheckpointLoaderSimpleWithNoiseSelect,
    # Gen1 Nodes
    "ADE_AnimateDiffLoaderGen1": AnimateDiffLoaderGen1,
    "ADE_AnimateDiffLoaderWithContext": LegacyAnimateDiffLoaderWithContext,
    # Gen2 Nodes
    "ADE_UseEvolvedSampling": UseEvolvedSamplingNode,
    "ADE_ApplyAnimateDiffModelSimple": ApplyAnimateDiffModelBasicNode,
    "ADE_ApplyAnimateDiffModel": ApplyAnimateDiffModelNode,
    "ADE_LoadAnimateDiffModel": LoadAnimateDiffModelNode,
    # AnimateLCM-I2V Nodes
    "ADE_ApplyAnimateLCMI2VModel": ApplyAnimateLCMI2VModel,
    "ADE_LoadAnimateLCMI2VModel": LoadAnimateLCMI2VModelNode,
    "ADE_UpscaleAndVAEEncode": UpscaleAndVaeEncode,
    "ADE_InjectI2VIntoAnimateDiffModel": LoadAnimateDiffAndInjectI2VNode,
    # CameraCtrl Nodes
    "ADE_ApplyAnimateDiffModelWithCameraCtrl": ApplyAnimateDiffWithCameraCtrl,
    "ADE_LoadAnimateDiffModelWithCameraCtrl": LoadAnimateDiffModelWithCameraCtrl,
    "ADE_CameraCtrlAnimateDiffKeyframe": CameraCtrlADKeyframeNode,
    "ADE_LoadCameraPoses": LoadCameraPoses,
    "ADE_CameraPoseBasic": CameraCtrlPoseBasic,
    "ADE_CameraPoseCombo": CameraCtrlPoseCombo,
    "ADE_CameraPoseAdvanced": CameraCtrlPoseAdvanced,
    "ADE_CameraManualPoseAppend": CameraCtrlManualAppendPose,
    "ADE_ReplaceCameraParameters": CameraCtrlReplaceCameraParameters,
    "ADE_ReplaceOriginalPoseAspectRatio": CameraCtrlSetOriginalAspectRatio,
    # MaskedLoraLoader
    #"ADE_MaskedLoadLora": MaskedLoraLoader,
    # Deprecated Nodes
    "AnimateDiffLoaderV1": AnimateDiffLoader_Deprecated,
    "ADE_AnimateDiffLoaderV1Advanced": AnimateDiffLoaderAdvanced_Deprecated,
    "ADE_AnimateDiffCombine": AnimateDiffCombine_Deprecated,
    "ADE_AnimateDiffModelSettings_Release": AnimateDiffModelSettings,
    "ADE_AnimateDiffModelSettingsSimple": AnimateDiffModelSettingsSimple,
    "ADE_AnimateDiffModelSettings": AnimateDiffModelSettingsAdvanced,
    "ADE_AnimateDiffModelSettingsAdvancedAttnStrengths": AnimateDiffModelSettingsAdvancedAttnStrengths,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    # Unencapsulated
    "ADE_AnimateDiffLoRALoader": "Load AnimateDiff LoRA 🎭🅐🅓",
    "ADE_AnimateDiffSamplingSettings": "Sample Settings 🎭🅐🅓",
    "ADE_AnimateDiffKeyframe": "AnimateDiff Keyframe 🎭🅐🅓",
    # Multival Nodes
    "ADE_MultivalDynamic": "Multival Dynamic 🎭🅐🅓",
    "ADE_MultivalScaledMask": "Multival Scaled Mask 🎭🅐🅓",
    # Context Opts
    "ADE_StandardStaticContextOptions": "Context Options◆Standard Static 🎭🅐🅓",
    "ADE_StandardUniformContextOptions": "Context Options◆Standard Uniform 🎭🅐🅓",
    "ADE_LoopedUniformContextOptions": "Context Options◆Looped Uniform 🎭🅐🅓",
    "ADE_ViewsOnlyContextOptions": "Context Options◆Views Only [VRAM⇈] 🎭🅐🅓",
    "ADE_BatchedContextOptions": "Context Options◆Batched [Non-AD] 🎭🅐🅓",
    "ADE_AnimateDiffUniformContextOptions": "Context Options◆Looped Uniform 🎭🅐🅓", # Legacy
    # View Opts
    "ADE_StandardStaticViewOptions": "View Options◆Standard Static 🎭🅐🅓",
    "ADE_StandardUniformViewOptions": "View Options◆Standard Uniform 🎭🅐🅓",
    "ADE_LoopedUniformViewOptions": "View Options◆Looped Uniform 🎭🅐🅓",
    # Iteration Opts
    "ADE_IterationOptsDefault": "Default Iteration Options 🎭🅐🅓",
    "ADE_IterationOptsFreeInit": "FreeInit Iteration Options 🎭🅐🅓",
    # Conditioning
    "ADE_RegisterLoraHook": "Register LoRA Hook 🎭🅐🅓",
    "ADE_RegisterLoraHookModelOnly": "Register LoRA Hook (Model Only) 🎭🅐🅓",
    "ADE_RegisterModelAsLoraHook": "Register Model as LoRA Hook 🎭🅐🅓",
    "ADE_RegisterModelAsLoraHookModelOnly": "Register Model as LoRA Hook (MO) 🎭🅐🅓",
    "ADE_CombineLoraHooks": "Combine LoRA Hooks [2] 🎭🅐🅓",
    "ADE_CombineLoraHooksFour": "Combine LoRA Hooks [4] 🎭🅐🅓",
    "ADE_CombineLoraHooksEight": "Combine LoRA Hooks [8] 🎭🅐🅓",
    "ADE_SetLoraHookKeyframe": "Set LoRA Hook Keyframes 🎭🅐🅓",
    "ADE_AttachLoraHookToCLIP": "Set CLIP LoRA Hook 🎭🅐🅓",
    "ADE_LoraHookKeyframe": "LoRA Hook Keyframe 🎭🅐🅓",
    "ADE_LoraHookKeyframeInterpolation": "LoRA Hook Keyframes Interpolation 🎭🅐🅓",
    "ADE_LoraHookKeyframeFromStrengthList": "LoRA Hook Keyframes From List 🎭🅐🅓",
    "ADE_AttachLoraHookToConditioning": "Set Model LoRA Hook 🎭🅐🅓",
    "ADE_PairedConditioningSetMask": "Set Props on Conds 🎭🅐🅓",
    "ADE_ConditioningSetMask": "Set Props on Cond 🎭🅐🅓",
    "ADE_PairedConditioningSetMaskAndCombine": "Set Props and Combine Conds 🎭🅐🅓",
    "ADE_ConditioningSetMaskAndCombine": "Set Props and Combine Cond 🎭🅐🅓",
    "ADE_PairedConditioningSetUnmaskedAndCombine": "Set Unmasked Conds 🎭🅐🅓",
    "ADE_ConditioningSetUnmaskedAndCombine": "Set Unmasked Cond 🎭🅐🅓",
    "ADE_TimestepsConditioning": "Timesteps Conditioning 🎭🅐🅓",
    # Noise Layer Nodes
    "ADE_NoiseLayerAdd": "Noise Layer [Add] 🎭🅐🅓",
    "ADE_NoiseLayerAddWeighted": "Noise Layer [Add Weighted] 🎭🅐🅓",
    "ADE_NoiseLayerReplace": "Noise Layer [Replace] 🎭🅐🅓",
    # AnimateDiff Settings
    "ADE_AnimateDiffSettings": "AnimateDiff Settings 🎭🅐🅓",
    "ADE_AdjustPESweetspotStretch": "Adjust PE [Sweetspot Stretch] 🎭🅐🅓",
    "ADE_AdjustPEFullStretch": "Adjust PE [Full Stretch] 🎭🅐🅓",
    "ADE_AdjustPEManual": "Adjust PE [Manual] 🎭🅐🅓",
    "ADE_AdjustWeightAllAdd": "Adjust Weight [All◆Add] 🎭🅐🅓",
    "ADE_AdjustWeightAllMult": "Adjust Weight [All◆Mult] 🎭🅐🅓",
    "ADE_AdjustWeightIndivAdd": "Adjust Weight [Indiv◆Add] 🎭🅐🅓",
    "ADE_AdjustWeightIndivMult": "Adjust Weight [Indiv◆Mult] 🎭🅐🅓",
    "ADE_AdjustWeightIndivAttnAdd": "Adjust Weight [Indiv-Attn◆Add] 🎭🅐🅓",
    "ADE_AdjustWeightIndivAttnMult": "Adjust Weight [Indiv-Attn◆Mult] 🎭🅐🅓",
    # Sample Settings
    "ADE_CustomCFG": "Custom CFG 🎭🅐🅓",
    "ADE_CustomCFGKeyframe": "Custom CFG Keyframe 🎭🅐🅓",
    "ADE_SigmaSchedule": "Create Sigma Schedule 🎭🅐🅓",
    "ADE_RawSigmaSchedule": "Create Raw Sigma Schedule 🎭🅐🅓",
    "ADE_SigmaScheduleWeightedAverage": "Sigma Schedule Weighted Mean 🎭🅐🅓",
    "ADE_SigmaScheduleWeightedAverageInterp": "Sigma Schedule Interpolated Mean 🎭🅐🅓",
    "ADE_SigmaScheduleSplitAndCombine": "Sigma Schedule Split Combine 🎭🅐🅓",
    # Extras Nodes
    "ADE_AnimateDiffUnload": "AnimateDiff Unload 🎭🅐🅓",
    "ADE_EmptyLatentImageLarge": "Empty Latent Image (Big Batch) 🎭🅐🅓",
    "CheckpointLoaderSimpleWithNoiseSelect": "Load Checkpoint w/ Noise Select 🎭🅐🅓",
    # Gen1 Nodes
    "ADE_AnimateDiffLoaderGen1": "AnimateDiff Loader 🎭🅐🅓①",
    "ADE_AnimateDiffLoaderWithContext": "AnimateDiff Loader [Legacy] 🎭🅐🅓①",
    # Gen2 Nodes
    "ADE_UseEvolvedSampling": "Use Evolved Sampling 🎭🅐🅓②",
    "ADE_ApplyAnimateDiffModelSimple": "Apply AnimateDiff Model 🎭🅐🅓②",
    "ADE_ApplyAnimateDiffModel": "Apply AnimateDiff Model (Adv.) 🎭🅐🅓②",
    "ADE_LoadAnimateDiffModel": "Load AnimateDiff Model 🎭🅐🅓②",
    # AnimateLCM-I2V Nodes
    "ADE_ApplyAnimateLCMI2VModel": "Apply AnimateLCM-I2V Model 🎭🅐🅓②",
    "ADE_LoadAnimateLCMI2VModel": "Load AnimateLCM-I2V Model 🎭🅐🅓②",
    "ADE_UpscaleAndVAEEncode": "Scale Ref Image and VAE Encode 🎭🅐🅓②",
    "ADE_InjectI2VIntoAnimateDiffModel": "🧪Inject I2V into AnimateDiff Model 🎭🅐🅓②",
    # CameraCtrl Nodes
    "ADE_ApplyAnimateDiffModelWithCameraCtrl": "Apply AnimateDiff+CameraCtrl Model 🎭🅐🅓②",
    "ADE_LoadAnimateDiffModelWithCameraCtrl": "Load AnimateDiff+CameraCtrl Model 🎭🅐🅓②",
    "ADE_CameraCtrlAnimateDiffKeyframe": "AnimateDiff+CameraCtrl Keyframe 🎭🅐🅓",
    "ADE_LoadCameraPoses": "Load CameraCtrl Poses (File) 🎭🅐🅓②",
    "ADE_CameraPoseBasic": "Create CameraCtrl Poses 🎭🅐🅓②",
    "ADE_CameraPoseCombo": "Create CameraCtrl Poses (Combo) 🎭🅐🅓②",
    "ADE_CameraPoseAdvanced": "Create CameraCtrl Poses (Adv.) 🎭🅐🅓②",
    "ADE_CameraManualPoseAppend": "Manual Append CameraCtrl Poses 🎭🅐🅓②",
    "ADE_ReplaceCameraParameters": "Replace Camera Parameters 🎭🅐🅓②",
    "ADE_ReplaceOriginalPoseAspectRatio": "Replace Orig. Pose Aspect Ratio 🎭🅐🅓②",
    # MaskedLoraLoader
    #"ADE_MaskedLoadLora": "Load LoRA (Masked) 🎭🅐🅓",
    # Deprecated Nodes
    "AnimateDiffLoaderV1": "🚫AnimateDiff Loader [DEPRECATED] 🎭🅐🅓",
    "ADE_AnimateDiffLoaderV1Advanced": "🚫AnimateDiff Loader (Advanced) [DEPRECATED] 🎭🅐🅓",
    "ADE_AnimateDiffCombine": "🚫AnimateDiff Combine [DEPRECATED, Use Video Combine (VHS) Instead!] 🎭🅐🅓",
    "ADE_AnimateDiffModelSettings_Release": "🚫[DEPR] Motion Model Settings 🎭🅐🅓①",
    "ADE_AnimateDiffModelSettingsSimple": "🚫[DEPR] Motion Model Settings (Simple) 🎭🅐🅓①",
    "ADE_AnimateDiffModelSettings": "🚫[DEPR] Motion Model Settings (Advanced) 🎭🅐🅓①",
    "ADE_AnimateDiffModelSettingsAdvancedAttnStrengths": "🚫[DEPR] Motion Model Settings (Adv. Attn) 🎭🅐🅓①",
}
