import comfy.sample as comfy_sample

from .sampling import motion_sample_factory

from .nodes_gen1 import (AnimateDiffLoaderGen1, LegacyAnimateDiffLoaderWithContext)
from .nodes_gen2 import (UseEvolvedSamplingNode, ApplyAnimateDiffModelNode, ApplyAnimateDiffModelBasicNode, ADKeyframeNode,
                         LoadAnimateDiffModelNode)
from .nodes_animatelcmi2v import (ApplyAnimateLCMI2VModel, LoadAnimateLCMI2VModelNode, LoadAnimateDiffAndInjectI2VNode, UpscaleAndVaeEncode)
from .nodes_cameractrl import (LoadAnimateDiffModelWithCameraCtrl, ApplyAnimateDiffWithCameraCtrl, CameraCtrlADKeyframeNode,
                               LoadCameraPosesFromFile, LoadCameraPosesFromPath,
                               CameraCtrlPoseBasic, CameraCtrlPoseCombo, CameraCtrlPoseAdvanced, CameraCtrlManualAppendPose,
                               CameraCtrlReplaceCameraParameters, CameraCtrlSetOriginalAspectRatio)
from .nodes_pia import (ApplyAnimateDiffPIAModel, LoadAnimateDiffAndInjectPIANode, InputPIA_MultivalNode, InputPIA_PaperPresetsNode, PIA_ADKeyframeNode)
from .nodes_multival import MultivalDynamicNode, MultivalScaledMaskNode, MultivalDynamicFloatInputNode, MultivalDynamicFloatsNode, MultivalConvertToMaskNode
from .nodes_conditioning import (MaskableLoraLoader, MaskableLoraLoaderModelOnly, MaskableSDModelLoader, MaskableSDModelLoaderModelOnly,
                                 SetModelLoraHook, SetClipLoraHook,
                                 CombineLoraHooks, CombineLoraHookFourOptional, CombineLoraHookEightOptional,
                                 PairedConditioningSetMaskHooked, ConditioningSetMaskHooked,
                                 PairedConditioningSetMaskAndCombineHooked, ConditioningSetMaskAndCombineHooked,
                                 PairedConditioningSetUnmaskedAndCombineHooked, ConditioningSetUnmaskedAndCombineHooked,
                                 PairedConditioningCombine, ConditioningCombine,
                                 ConditioningTimestepsNode, SetLoraHookKeyframes,
                                 CreateLoraHookKeyframe, CreateLoraHookKeyframeInterpolation, CreateLoraHookKeyframeFromStrengthList)
from .nodes_sample import (FreeInitOptionsNode, NoiseLayerAddWeightedNode, SampleSettingsNode, NoiseLayerAddNode, NoiseLayerReplaceNode, IterationOptionsNode,
                           CustomCFGNode, CustomCFGSimpleNode, CustomCFGKeyframeNode, CustomCFGKeyframeSimpleNode, CustomCFGKeyframeInterpolationNode, CustomCFGKeyframeFromListNode,
                           CFGExtrasPAGNode, CFGExtrasPAGSimpleNode, CFGExtrasRescaleCFGNode, CFGExtrasRescaleCFGSimpleNode,
                           NoisedImageInjectionNode, NoisedImageInjectOptionsNode)
from .nodes_sigma_schedule import (SigmaScheduleNode, RawSigmaScheduleNode, WeightedAverageSigmaScheduleNode, InterpolatedWeightedAverageSigmaScheduleNode, SplitAndCombineSigmaScheduleNode, SigmaScheduleToSigmasNode)
from .nodes_context import (LegacyLoopedUniformContextOptionsNode, LoopedUniformContextOptionsNode, LoopedUniformViewOptionsNode, StandardUniformContextOptionsNode, StandardStaticContextOptionsNode, BatchedContextOptionsNode,
                            StandardStaticViewOptionsNode, StandardUniformViewOptionsNode, ViewAsContextOptionsNode,
                            VisualizeContextOptionsK, VisualizeContextOptionsKAdv, VisualizeContextOptionsSCustom)
from .nodes_context_extras import (SetContextExtrasOnContextOptions, ContextExtras_NaiveReuse, ContextExtras_ContextRef,
                            ContextRef_ModeFirst, ContextRef_ModeSliding, ContextRef_ModeIndexes,
                            ContextRef_TuneAttn, ContextRef_TuneAttnAdain,
                            ContextRef_KeyframeMultivalNode, ContextRef_KeyframeInterpolationNode, ContextRef_KeyframeFromListNode,
                            NaiveReuse_KeyframeMultivalNode, NaiveReuse_KeyframeInterpolationNode, NaiveReuse_KeyframeFromListNode)
from .nodes_ad_settings import (AnimateDiffSettingsNode, ManualAdjustPENode, SweetspotStretchPENode, FullStretchPENode,
                                WeightAdjustAllAddNode, WeightAdjustAllMultNode, WeightAdjustIndivAddNode, WeightAdjustIndivMultNode,
                                WeightAdjustIndivAttnAddNode, WeightAdjustIndivAttnMultNode)
from .nodes_scheduling import (PromptSchedulingNode, PromptSchedulingLatentsNode, ValueSchedulingNode, ValueSchedulingLatentsNode,
                               AddValuesReplaceNode, FloatToFloatsNode)
from .nodes_extras import AnimateDiffUnload, EmptyLatentImageLarge, CheckpointLoaderSimpleWithNoiseSelect, PerturbedAttentionGuidanceMultival, RescaleCFGMultival
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
    "ADE_MultivalDynamicFloatInput": MultivalDynamicFloatInputNode,
    "ADE_MultivalDynamicFloats": MultivalDynamicFloatsNode,
    "ADE_MultivalScaledMask": MultivalScaledMaskNode,
    "ADE_MultivalConvertToMask": MultivalConvertToMaskNode,
    ###############################################################################
    #------------------------------------------------------------------------------
    # Context Opts
    "ADE_StandardStaticContextOptions": StandardStaticContextOptionsNode,
    "ADE_StandardUniformContextOptions": StandardUniformContextOptionsNode,
    "ADE_LoopedUniformContextOptions": LoopedUniformContextOptionsNode,
    "ADE_ViewsOnlyContextOptions": ViewAsContextOptionsNode,
    "ADE_BatchedContextOptions": BatchedContextOptionsNode,
    "ADE_AnimateDiffUniformContextOptions": LegacyLoopedUniformContextOptionsNode, # Legacy/Deprecated
    "ADE_VisualizeContextOptionsK": VisualizeContextOptionsK,
    "ADE_VisualizeContextOptionsKAdv": VisualizeContextOptionsKAdv,
    "ADE_VisualizeContextOptionsSCustom": VisualizeContextOptionsSCustom,
    # View Opts
    "ADE_StandardStaticViewOptions": StandardStaticViewOptionsNode,
    "ADE_StandardUniformViewOptions": StandardUniformViewOptionsNode,
    "ADE_LoopedUniformViewOptions": LoopedUniformViewOptionsNode,
    # Context Extras
    "ADE_ContextExtras_Set": SetContextExtrasOnContextOptions,
    "ADE_ContextExtras_ContextRef": ContextExtras_ContextRef,
    "ADE_ContextExtras_ContextRef_ModeFirst": ContextRef_ModeFirst,
    "ADE_ContextExtras_ContextRef_ModeSliding": ContextRef_ModeSliding,
    "ADE_ContextExtras_ContextRef_ModeIndexes": ContextRef_ModeIndexes,
    "ADE_ContextExtras_ContextRef_TuneAttn": ContextRef_TuneAttn,
    "ADE_ContextExtras_ContextRef_TuneAttnAdain": ContextRef_TuneAttnAdain,
    "ADE_ContextExtras_ContextRef_Keyframe": ContextRef_KeyframeMultivalNode,
    "ADE_ContextExtras_ContextRef_KeyframeInterpolation": ContextRef_KeyframeInterpolationNode,
    "ADE_ContextExtras_ContextRef_KeyframeFromList": ContextRef_KeyframeFromListNode,
    "ADE_ContextExtras_NaiveReuse": ContextExtras_NaiveReuse,
    "ADE_ContextExtras_NaiveReuse_Keyframe": NaiveReuse_KeyframeMultivalNode,
    "ADE_ContextExtras_NaiveReuse_KeyframeInterpolation": NaiveReuse_KeyframeInterpolationNode,
    "ADE_ContextExtras_NaiveReuse_KeyframeFromList": NaiveReuse_KeyframeFromListNode,
    #------------------------------------------------------------------------------
    ###############################################################################
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
    "ADE_PairedConditioningCombine": PairedConditioningCombine,
    "ADE_ConditioningCombine": ConditioningCombine,
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
    "ADE_CustomCFGSimple": CustomCFGSimpleNode,
    "ADE_CustomCFG": CustomCFGNode,
    "ADE_CustomCFGKeyframeSimple": CustomCFGKeyframeSimpleNode,
    "ADE_CustomCFGKeyframe": CustomCFGKeyframeNode,
    "ADE_CustomCFGKeyframeInterpolation": CustomCFGKeyframeInterpolationNode,
    "ADE_CustomCFGKeyframeFromList": CustomCFGKeyframeFromListNode,
    "ADE_CFGExtrasPAGSimple": CFGExtrasPAGSimpleNode,
    "ADE_CFGExtrasPAG": CFGExtrasPAGNode,
    "ADE_CFGExtrasRescaleCFGSimple": CFGExtrasRescaleCFGSimpleNode,
    "ADE_CFGExtrasRescaleCFG": CFGExtrasRescaleCFGNode,
    "ADE_SigmaSchedule": SigmaScheduleNode,
    "ADE_RawSigmaSchedule": RawSigmaScheduleNode,
    "ADE_SigmaScheduleWeightedAverage": WeightedAverageSigmaScheduleNode,
    "ADE_SigmaScheduleWeightedAverageInterp": InterpolatedWeightedAverageSigmaScheduleNode,
    "ADE_SigmaScheduleSplitAndCombine": SplitAndCombineSigmaScheduleNode,
    "ADE_SigmaScheduleToSigmas": SigmaScheduleToSigmasNode,
    "ADE_NoisedImageInjection": NoisedImageInjectionNode,
    "ADE_NoisedImageInjectOptions": NoisedImageInjectOptionsNode,
    # Scheduling
    "ADE_PromptScheduling": PromptSchedulingNode,
    "ADE_PromptSchedulingLatents": PromptSchedulingLatentsNode,
    "ADE_ValueScheduling": ValueSchedulingNode,
    "ADE_ValueSchedulingLatents": ValueSchedulingLatentsNode,
    "ADE_ValuesReplace": AddValuesReplaceNode,
    "ADE_FloatToFloats": FloatToFloatsNode,
    # Extras Nodes
    "ADE_AnimateDiffUnload": AnimateDiffUnload,
    "ADE_EmptyLatentImageLarge": EmptyLatentImageLarge,
    "CheckpointLoaderSimpleWithNoiseSelect": CheckpointLoaderSimpleWithNoiseSelect,
    "ADE_PerturbedAttentionGuidanceMultival": PerturbedAttentionGuidanceMultival,
    "ADE_RescaleCFGMultival": RescaleCFGMultival,
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
    "ADE_LoadCameraPoses": LoadCameraPosesFromFile,
    "ADE_LoadCameraPosesFromPath": LoadCameraPosesFromPath,
    "ADE_CameraPoseBasic": CameraCtrlPoseBasic,
    "ADE_CameraPoseCombo": CameraCtrlPoseCombo,
    "ADE_CameraPoseAdvanced": CameraCtrlPoseAdvanced,
    "ADE_CameraManualPoseAppend": CameraCtrlManualAppendPose,
    "ADE_ReplaceCameraParameters": CameraCtrlReplaceCameraParameters,
    "ADE_ReplaceOriginalPoseAspectRatio": CameraCtrlSetOriginalAspectRatio,
    # PIA Nodes
    "ADE_ApplyAnimateDiffModelWithPIA": ApplyAnimateDiffPIAModel,
    "ADE_InputPIA_Multival": InputPIA_MultivalNode,
    "ADE_InputPIA_PaperPresets": InputPIA_PaperPresetsNode,
    "ADE_PIA_AnimateDiffKeyframe": PIA_ADKeyframeNode,
    "ADE_InjectPIAIntoAnimateDiffModel": LoadAnimateDiffAndInjectPIANode,
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
    "ADE_MultivalDynamic": "Multival 🎭🅐🅓",
    "ADE_MultivalDynamicFloatInput": "Multival [Float List] 🎭🅐🅓",
    "ADE_MultivalDynamicFloats": "Multival [Floats] 🎭🅐🅓",
    "ADE_MultivalScaledMask": "Multival Scaled Mask 🎭🅐🅓",
    "ADE_MultivalConvertToMask": "Multival to Mask 🎭🅐🅓",
    ###############################################################################
    #------------------------------------------------------------------------------
    # Context Opts
    "ADE_StandardStaticContextOptions": "Context Options◆Standard Static 🎭🅐🅓",
    "ADE_StandardUniformContextOptions": "Context Options◆Standard Uniform 🎭🅐🅓",
    "ADE_LoopedUniformContextOptions": "Context Options◆Looped Uniform 🎭🅐🅓",
    "ADE_ViewsOnlyContextOptions": "Context Options◆Views Only [VRAM⇈] 🎭🅐🅓",
    "ADE_BatchedContextOptions": "Context Options◆Batched [Non-AD] 🎭🅐🅓",
    "ADE_AnimateDiffUniformContextOptions": "Context Options◆Looped Uniform 🎭🅐🅓", # Legacy/Deprecated
    "ADE_VisualizeContextOptionsK": "Visualize Context Options (K.) 🎭🅐🅓",
    "ADE_VisualizeContextOptionsKAdv": "Visualize Context Options (K.Adv.) 🎭🅐🅓",
    "ADE_VisualizeContextOptionsSCustom": "Visualize Context Options (S.Cus.) 🎭🅐🅓",
    # View Opts
    "ADE_StandardStaticViewOptions": "View Options◆Standard Static 🎭🅐🅓",
    "ADE_StandardUniformViewOptions": "View Options◆Standard Uniform 🎭🅐🅓",
    "ADE_LoopedUniformViewOptions": "View Options◆Looped Uniform 🎭🅐🅓",
    # Context Extras
    "ADE_ContextExtras_Set": "Set Context Extras 🎭🅐🅓",
    "ADE_ContextExtras_ContextRef": "Context Extras◆ContextRef 🎭🅐🅓",
    "ADE_ContextExtras_ContextRef_ModeFirst": "ContextRef Mode◆First 🎭🅐🅓",
    "ADE_ContextExtras_ContextRef_ModeSliding": "ContextRef Mode◆Sliding 🎭🅐🅓",
    "ADE_ContextExtras_ContextRef_ModeIndexes": "ContextRef Mode◆Indexes 🎭🅐🅓",
    "ADE_ContextExtras_ContextRef_TuneAttn": "ContextRef Tune◆Attn 🎭🅐🅓",
    "ADE_ContextExtras_ContextRef_TuneAttnAdain": "ContextRef Tune◆Attn+Adain 🎭🅐🅓",
    "ADE_ContextExtras_ContextRef_Keyframe": "ContextRef Keyframe 🎭🅐🅓",
    "ADE_ContextExtras_ContextRef_KeyframeInterpolation": "ContextRef Keyframes Interp. 🎭🅐🅓",
    "ADE_ContextExtras_ContextRef_KeyframeFromList": "ContextRef Keyframes From List 🎭🅐🅓",
    "ADE_ContextExtras_NaiveReuse": "Context Extras◆NaiveReuse 🎭🅐🅓",
    "ADE_ContextExtras_NaiveReuse_Keyframe": "NaiveReuse Keyframe 🎭🅐🅓",
    "ADE_ContextExtras_NaiveReuse_KeyframeInterpolation": "NaiveReuse Keyframes Interp. 🎭🅐🅓",
    "ADE_ContextExtras_NaiveReuse_KeyframeFromList": "NaiveReuse Keyframes From List 🎭🅐🅓",
    #------------------------------------------------------------------------------
    ###############################################################################
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
    "ADE_LoraHookKeyframeInterpolation": "LoRA Hook Keyframes Interp. 🎭🅐🅓",
    "ADE_LoraHookKeyframeFromStrengthList": "LoRA Hook Keyframes From List 🎭🅐🅓",
    "ADE_AttachLoraHookToConditioning": "Set Model LoRA Hook 🎭🅐🅓",
    "ADE_PairedConditioningSetMask": "Set Props on Conds 🎭🅐🅓",
    "ADE_ConditioningSetMask": "Set Props on Cond 🎭🅐🅓",
    "ADE_PairedConditioningSetMaskAndCombine": "Set Props and Combine Conds 🎭🅐🅓",
    "ADE_ConditioningSetMaskAndCombine": "Set Props and Combine Cond 🎭🅐🅓",
    "ADE_PairedConditioningSetUnmaskedAndCombine": "Set Unmasked Conds 🎭🅐🅓",
    "ADE_ConditioningSetUnmaskedAndCombine": "Set Unmasked Cond 🎭🅐🅓",
    "ADE_PairedConditioningCombine": "Manual Combine Conds 🎭🅐🅓",
    "ADE_ConditioningCombine": "Manual Combine Cond 🎭🅐🅓",
    "ADE_TimestepsConditioning": "Timesteps Conditioning 🎭🅐🅓",
    # Noise Layer Nodes
    "ADE_NoiseLayerAdd": "Noise Layer [Add] 🎭🅐🅓",
    "ADE_NoiseLayerAddWeighted": "Noise Layer [Add Weighted] 🎭🅐🅓",
    "ADE_NoiseLayerReplace": "Noise Layer [Replace] 🎭🅐🅓",
    # AnimateDiff Settings
    "ADE_AnimateDiffSettings": "AnimateDiff Settings 🎭🅐🅓",
    "ADE_AdjustPESweetspotStretch": "Adjust PE [Sweetspot] 🎭🅐🅓",
    "ADE_AdjustPEFullStretch": "Adjust PE [Full Stretch] 🎭🅐🅓",
    "ADE_AdjustPEManual": "Adjust PE [Manual] 🎭🅐🅓",
    "ADE_AdjustWeightAllAdd": "Adjust Weight [All◆Add] 🎭🅐🅓",
    "ADE_AdjustWeightAllMult": "Adjust Weight [All◆Mult] 🎭🅐🅓",
    "ADE_AdjustWeightIndivAdd": "Adjust Weight [Indiv◆Add] 🎭🅐🅓",
    "ADE_AdjustWeightIndivMult": "Adjust Weight [Indiv◆Mult] 🎭🅐🅓",
    "ADE_AdjustWeightIndivAttnAdd": "Adjust Weight [Indiv-Attn◆Add] 🎭🅐🅓",
    "ADE_AdjustWeightIndivAttnMult": "Adjust Weight [Indiv-Attn◆Mult] 🎭🅐🅓",
    # Sample Settings
    "ADE_CustomCFGSimple": "Custom CFG 🎭🅐🅓",
    "ADE_CustomCFG": "Custom CFG [Multival] 🎭🅐🅓",
    "ADE_CustomCFGKeyframeSimple": "Custom CFG Keyframe 🎭🅐🅓",
    "ADE_CustomCFGKeyframe": "Custom CFG Keyframe [Multival] 🎭🅐🅓",
    "ADE_CustomCFGKeyframeInterpolation": "Custom CFG Keyframes Interp. 🎭🅐🅓",
    "ADE_CustomCFGKeyframeFromList": "Custom CFG Keyframes From List 🎭🅐🅓",
    "ADE_CFGExtrasPAGSimple": "CFG Extras◆PAG 🎭🅐🅓",
    "ADE_CFGExtrasPAG": "CFG Extras◆PAG [Multival] 🎭🅐🅓",
    "ADE_CFGExtrasRescaleCFGSimple": "CFG Extras◆RescaleCFG 🎭🅐🅓",
    "ADE_CFGExtrasRescaleCFG": "CFG Extras◆RescaleCFG [Multival] 🎭🅐🅓",
    "ADE_SigmaSchedule": "Create Sigma Schedule 🎭🅐🅓",
    "ADE_RawSigmaSchedule": "Create Raw Sigma Schedule 🎭🅐🅓",
    "ADE_SigmaScheduleWeightedAverage": "Sigma Schedule Weighted Mean 🎭🅐🅓",
    "ADE_SigmaScheduleWeightedAverageInterp": "Sigma Schedule Interp. Mean 🎭🅐🅓",
    "ADE_SigmaScheduleSplitAndCombine": "Sigma Schedule Split Combine 🎭🅐🅓",
    "ADE_SigmaScheduleToSigmas": "Sigma Schedule To Sigmas 🎭🅐🅓",
    "ADE_NoisedImageInjection": "Image Injection 🎭🅐🅓",
    "ADE_NoisedImageInjectOptions": "Image Injection Options 🎭🅐🅓",
    # Scheduling
    "ADE_PromptScheduling": "Prompt Scheduling 🎭🅐🅓",
    "ADE_PromptSchedulingLatents": "Prompt Scheduling [Latents] 🎭🅐🅓",
    "ADE_ValueScheduling": "Value Scheduling 🎭🅐🅓",
    "ADE_ValueSchedulingLatents": "Value Scheduling [Latents] 🎭🅐🅓",
    "ADE_ValuesReplace": "Add Values Replace 🎭🅐🅓",
    "ADE_FloatToFloats": "Float to Floats 🎭🅐🅓",
    # Extras Nodes
    "ADE_AnimateDiffUnload": "AnimateDiff Unload 🎭🅐🅓",
    "ADE_EmptyLatentImageLarge": "Empty Latent Image (Big Batch) 🎭🅐🅓",
    "CheckpointLoaderSimpleWithNoiseSelect": "Load Checkpoint w/ Noise Select 🎭🅐🅓",
    "ADE_PerturbedAttentionGuidanceMultival": "PerturbedAttnGuide [Multival] 🎭🅐🅓",
    "ADE_RescaleCFGMultival": "RescaleCFG [Multival] 🎭🅐🅓",
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
    "ADE_LoadCameraPosesFromPath": "Load CameraCtrl Poses (Path) 🎭🅐🅓②",
    "ADE_CameraPoseBasic": "Create CameraCtrl Poses 🎭🅐🅓②",
    "ADE_CameraPoseCombo": "Create CameraCtrl Poses (Combo) 🎭🅐🅓②",
    "ADE_CameraPoseAdvanced": "Create CameraCtrl Poses (Adv.) 🎭🅐🅓②",
    "ADE_CameraManualPoseAppend": "Manual Append CameraCtrl Poses 🎭🅐🅓②",
    "ADE_ReplaceCameraParameters": "Replace Camera Parameters 🎭🅐🅓②",
    "ADE_ReplaceOriginalPoseAspectRatio": "Replace Orig. Pose Aspect Ratio 🎭🅐🅓②",
    # PIA Nodes
    "ADE_ApplyAnimateDiffModelWithPIA": "Apply AnimateDiff-PIA Model 🎭🅐🅓②",
    "ADE_InputPIA_Multival": "PIA Input [Multival] 🎭🅐🅓②",
    "ADE_InputPIA_PaperPresets": "PIA Input [Paper Presets] 🎭🅐🅓②",
    "ADE_PIA_AnimateDiffKeyframe": "AnimateDiff-PIA Keyframe 🎭🅐🅓",
    "ADE_InjectPIAIntoAnimateDiffModel": "🧪Inject PIA into AnimateDiff Model 🎭🅐🅓②",
    # Deprecated Nodes
    "AnimateDiffLoaderV1": "🚫AnimateDiff Loader [DEPRECATED] 🎭🅐🅓",
    "ADE_AnimateDiffLoaderV1Advanced": "🚫AnimateDiff Loader (Advanced) [DEPRECATED] 🎭🅐🅓",
    "ADE_AnimateDiffCombine": "🚫AnimateDiff Combine [DEPRECATED, Use Video Combine (VHS) Instead!] 🎭🅐🅓",
    "ADE_AnimateDiffModelSettings_Release": "🚫[DEPR] Motion Model Settings 🎭🅐🅓①",
    "ADE_AnimateDiffModelSettingsSimple": "🚫[DEPR] Motion Model Settings (Simple) 🎭🅐🅓①",
    "ADE_AnimateDiffModelSettings": "🚫[DEPR] Motion Model Settings (Advanced) 🎭🅐🅓①",
    "ADE_AnimateDiffModelSettingsAdvancedAttnStrengths": "🚫[DEPR] Motion Model Settings (Adv. Attn) 🎭🅐🅓①",
}
