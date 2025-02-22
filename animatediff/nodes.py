import comfy.sample as comfy_sample

from .nodes_gen1 import (AnimateDiffLoaderGen1,)
from .nodes_gen2 import (UseEvolvedSamplingNode, ApplyAnimateDiffModelNode, ApplyAnimateDiffModelBasicNode, ADKeyframeNode,
                         LoadAnimateDiffModelNode)
from .nodes_animatelcmi2v import (ApplyAnimateLCMI2VModel, LoadAnimateLCMI2VModelNode, LoadAnimateDiffAndInjectI2VNode, UpscaleAndVaeEncode)
from .nodes_cameractrl import (LoadAnimateDiffModelWithCameraCtrl, ApplyAnimateDiffWithCameraCtrl, CameraCtrlADKeyframeNode,
                               LoadCameraPosesFromFile, LoadCameraPosesFromPath,
                               CameraCtrlPoseBasic, CameraCtrlPoseCombo, CameraCtrlPoseAdvanced, CameraCtrlManualAppendPose,
                               CameraCtrlReplaceCameraParameters, CameraCtrlSetOriginalAspectRatio)
from .nodes_motionctrl import (LoadMotionCtrlCMCM, LoadMotionCtrlOMCM, ApplyAnimateDiffMotionCtrlModel, LoadMotionCtrlCameraPosesFromFile)
from .nodes_pia import (ApplyAnimateDiffPIAModel, LoadAnimateDiffAndInjectPIANode, InputPIA_MultivalNode, InputPIA_PaperPresetsNode, PIA_ADKeyframeNode)
from .nodes_fancyvideo import (ApplyAnimateDiffFancyVideo,)
from .nodes_hellomeme import (TestHMRefNetInjection,)
from .nodes_multival import MultivalDynamicNode, MultivalScaledMaskNode, MultivalDynamicFloatInputNode, MultivalDynamicFloatsNode, MultivalConvertToMaskNode
from .nodes_conditioning import (CreateLoraHookKeyframeInterpolationDEPR,
                                 MaskableLoraLoaderDEPR, MaskableLoraLoaderModelOnlyDEPR, MaskableSDModelLoaderDEPR, MaskableSDModelLoaderModelOnlyDEPR, 
                                 SetModelLoraHookDEPR, SetClipLoraHookDEPR,
                                 CombineLoraHooksDEPR, CombineLoraHookFourOptionalDEPR, CombineLoraHookEightOptionalDEPR,
                                 PairedConditioningSetMaskHookedDEPR, ConditioningSetMaskHookedDEPR,
                                 PairedConditioningSetMaskAndCombineHookedDEPR, ConditioningSetMaskAndCombineHookedDEPR,
                                 PairedConditioningSetUnmaskedAndCombineHookedDEPR, ConditioningSetUnmaskedAndCombineHookedDEPR,
                                 PairedConditioningCombineDEPR, ConditioningCombineDEPR,
                                 ConditioningTimestepsNodeDEPR, SetLoraHookKeyframesDEPR,
                                 CreateLoraHookKeyframeDEPR, CreateLoraHookKeyframeFromStrengthListDEPR)
from .nodes_sample import (FreeInitOptionsNode, NoiseLayerAddWeightedNode, NoiseLayerNormalizedSumNode, SampleSettingsNode, NoiseLayerAddNode, NoiseLayerReplaceNode, IterationOptionsNode,
                           CustomCFGNode, CustomCFGSimpleNode, CustomCFGKeyframeNode, CustomCFGKeyframeSimpleNode, CustomCFGKeyframeInterpolationNode, CustomCFGKeyframeFromListNode,
                           CFGExtrasPAGNode, CFGExtrasPAGSimpleNode, CFGExtrasRescaleCFGNode, CFGExtrasRescaleCFGSimpleNode,
                           NoisedImageInjectionNode, NoisedImageInjectOptionsNode, NoiseCalibrationNode, AncestralOptionsNode)
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
from .nodes_scheduling import (ConditionExtractionNode, PromptSchedulingNode, PromptSchedulingLatentsNode, ValueSchedulingNode, ValueSchedulingLatentsNode,
                               AddValuesReplaceNode, FloatToFloatsNode)
from .nodes_per_block import (ADBlockComboNode, ADBlockIndivNode, PerBlockHighLevelNode,
                              PerBlock_SD15_LowLevelNode, PerBlock_SD15_MidLevelNode, PerBlock_SD15_FromFloatsNode,
                              PerBlock_SDXL_LowLevelNode, PerBlock_SDXL_MidLevelNode, PerBlock_SDXL_FromFloatsNode)
from .nodes_extras import AnimateDiffUnload, EmptyLatentImageLarge, CheckpointLoaderSimpleWithNoiseSelect, PerturbedAttentionGuidanceMultival, RescaleCFGMultival
from .nodes_deprecated import (AnimateDiffLoaderDEPR, AnimateDiffLoaderAdvancedDEPR, LegacyAnimateDiffLoaderWithContextDEPR, AnimateDiffCombineDEPR,
                               AnimateDiffModelSettingsDEPR, AnimateDiffModelSettingsSimpleDEPR, AnimateDiffModelSettingsAdvancedDEPR, AnimateDiffModelSettingsAdvancedAttnStrengthsDEPR)
from .nodes_lora import AnimateDiffLoraLoader

from .logger import logger


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
    # Conditioning (DEPRECATED)
    "ADE_RegisterLoraHook": MaskableLoraLoaderDEPR,
    "ADE_RegisterLoraHookModelOnly": MaskableLoraLoaderModelOnlyDEPR,
    "ADE_RegisterModelAsLoraHook": MaskableSDModelLoaderDEPR,
    "ADE_RegisterModelAsLoraHookModelOnly": MaskableSDModelLoaderModelOnlyDEPR,
    "ADE_CombineLoraHooks": CombineLoraHooksDEPR,
    "ADE_CombineLoraHooksFour": CombineLoraHookFourOptionalDEPR,
    "ADE_CombineLoraHooksEight": CombineLoraHookEightOptionalDEPR,
    "ADE_SetLoraHookKeyframe": SetLoraHookKeyframesDEPR,
    "ADE_AttachLoraHookToCLIP": SetClipLoraHookDEPR,
    "ADE_LoraHookKeyframe": CreateLoraHookKeyframeDEPR,
    "ADE_LoraHookKeyframeInterpolation": CreateLoraHookKeyframeInterpolationDEPR,
    "ADE_LoraHookKeyframeFromStrengthList": CreateLoraHookKeyframeFromStrengthListDEPR,
    "ADE_AttachLoraHookToConditioning": SetModelLoraHookDEPR,
    "ADE_PairedConditioningSetMask": PairedConditioningSetMaskHookedDEPR,
    "ADE_ConditioningSetMask": ConditioningSetMaskHookedDEPR,
    "ADE_PairedConditioningSetMaskAndCombine": PairedConditioningSetMaskAndCombineHookedDEPR,
    "ADE_ConditioningSetMaskAndCombine": ConditioningSetMaskAndCombineHookedDEPR,
    "ADE_PairedConditioningSetUnmaskedAndCombine": PairedConditioningSetUnmaskedAndCombineHookedDEPR,
    "ADE_ConditioningSetUnmaskedAndCombine": ConditioningSetUnmaskedAndCombineHookedDEPR,
    "ADE_PairedConditioningCombine": PairedConditioningCombineDEPR,
    "ADE_ConditioningCombine": ConditioningCombineDEPR,
    "ADE_TimestepsConditioning": ConditioningTimestepsNodeDEPR,
    # Noise Layer Nodes
    "ADE_NoiseLayerAdd": NoiseLayerAddNode,
    "ADE_NoiseLayerAddWeighted": NoiseLayerAddWeightedNode,
    "ADE_NoiseLayerNormalizedSum": NoiseLayerNormalizedSumNode,
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
    "ADE_AncestralOptions": AncestralOptionsNode,
    #"ADE_NoiseCalibration": NoiseCalibrationNode,
    # Scheduling
    PromptSchedulingNode.NodeID: PromptSchedulingNode,
    PromptSchedulingLatentsNode.NodeID: PromptSchedulingLatentsNode,
    ValueSchedulingNode.NodeID: ValueSchedulingNode,
    ValueSchedulingLatentsNode.NodeID: ValueSchedulingLatentsNode,
    ConditionExtractionNode.NodeID: ConditionExtractionNode,
    AddValuesReplaceNode.NodeID: AddValuesReplaceNode,
    FloatToFloatsNode.NodeID: FloatToFloatsNode,
    # Per-Block
    ADBlockComboNode.NodeID: ADBlockComboNode,
    ADBlockIndivNode.NodeID: ADBlockIndivNode,
    PerBlockHighLevelNode.NodeID: PerBlockHighLevelNode,
    PerBlock_SD15_MidLevelNode.NodeID: PerBlock_SD15_MidLevelNode,
    PerBlock_SD15_LowLevelNode.NodeID: PerBlock_SD15_LowLevelNode,
    PerBlock_SD15_FromFloatsNode.NodeID: PerBlock_SD15_FromFloatsNode,
    PerBlock_SDXL_MidLevelNode.NodeID: PerBlock_SDXL_MidLevelNode,
    PerBlock_SDXL_LowLevelNode.NodeID: PerBlock_SDXL_LowLevelNode,
    PerBlock_SDXL_FromFloatsNode.NodeID: PerBlock_SDXL_FromFloatsNode,
    # Extras Nodes
    "ADE_AnimateDiffUnload": AnimateDiffUnload,
    "ADE_EmptyLatentImageLarge": EmptyLatentImageLarge,
    "CheckpointLoaderSimpleWithNoiseSelect": CheckpointLoaderSimpleWithNoiseSelect,
    "ADE_PerturbedAttentionGuidanceMultival": PerturbedAttentionGuidanceMultival,
    "ADE_RescaleCFGMultival": RescaleCFGMultival,
    # Gen1 Nodes
    "ADE_AnimateDiffLoaderGen1": AnimateDiffLoaderGen1,
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
    # MotionCtrl Nodes
    #LoadMotionCtrlCMCM.NodeID: LoadMotionCtrlCMCM,
    #LoadMotionCtrlOMCM.NodeID: LoadMotionCtrlOMCM,
    #ApplyAnimateDiffMotionCtrlModel.NodeID: ApplyAnimateDiffMotionCtrlModel,
    #LoadMotionCtrlCameraPosesFromFile.NodeID: LoadMotionCtrlCameraPosesFromFile,
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
    # FancyVideo
    #ApplyAnimateDiffFancyVideo.NodeID: ApplyAnimateDiffFancyVideo,
    # HelloMeme
    #TestHMRefNetInjection.NodeID: TestHMRefNetInjection,
    # Deprecated Nodes
    "ADE_AnimateDiffLoaderWithContext": LegacyAnimateDiffLoaderWithContextDEPR,
    "AnimateDiffLoaderV1": AnimateDiffLoaderDEPR,
    "ADE_AnimateDiffLoaderV1Advanced": AnimateDiffLoaderAdvancedDEPR,
    "ADE_AnimateDiffCombine": AnimateDiffCombineDEPR,
    "ADE_AnimateDiffModelSettings_Release": AnimateDiffModelSettingsDEPR,
    "ADE_AnimateDiffModelSettingsSimple": AnimateDiffModelSettingsSimpleDEPR,
    "ADE_AnimateDiffModelSettings": AnimateDiffModelSettingsAdvancedDEPR,
    "ADE_AnimateDiffModelSettingsAdvancedAttnStrengths": AnimateDiffModelSettingsAdvancedAttnStrengthsDEPR,
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
    # Conditioning (DEPRECATED)
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
    "ADE_NoiseLayerNormalizedSum": "Noise Layer [Normalized Sum] 🎭🅐🅓",
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
    "ADE_NoiseCalibration": "Noise Calibration 🎭🅐🅓",
    "ADE_AncestralOptions": "Ancestral Options 🎭🅐🅓",
    # Scheduling
    PromptSchedulingNode.NodeID: PromptSchedulingNode.NodeName,
    PromptSchedulingLatentsNode.NodeID: PromptSchedulingLatentsNode.NodeName,
    ValueSchedulingNode.NodeID: ValueSchedulingNode.NodeName,
    ValueSchedulingLatentsNode.NodeID: ValueSchedulingLatentsNode.NodeName,
    ConditionExtractionNode.NodeID: ConditionExtractionNode.NodeName,
    AddValuesReplaceNode.NodeID: AddValuesReplaceNode.NodeName,
    FloatToFloatsNode.NodeID:FloatToFloatsNode.NodeName,
    # Per-Block
    ADBlockComboNode.NodeID: ADBlockComboNode.NodeName,
    ADBlockIndivNode.NodeID: ADBlockIndivNode.NodeName,
    PerBlockHighLevelNode.NodeID: PerBlockHighLevelNode.NodeName,
    PerBlock_SD15_MidLevelNode.NodeID: PerBlock_SD15_MidLevelNode.NodeName,
    PerBlock_SD15_LowLevelNode.NodeID: PerBlock_SD15_LowLevelNode.NodeName,
    PerBlock_SD15_FromFloatsNode.NodeID: PerBlock_SD15_FromFloatsNode.NodeName,
    PerBlock_SDXL_MidLevelNode.NodeID: PerBlock_SDXL_MidLevelNode.NodeName,
    PerBlock_SDXL_LowLevelNode.NodeID: PerBlock_SDXL_LowLevelNode.NodeName,
    PerBlock_SDXL_FromFloatsNode.NodeID: PerBlock_SDXL_FromFloatsNode.NodeName,
    # Extras Nodes
    "ADE_AnimateDiffUnload": "AnimateDiff Unload 🎭🅐🅓",
    "ADE_EmptyLatentImageLarge": "Empty Latent Image (Big Batch) 🎭🅐🅓",
    "CheckpointLoaderSimpleWithNoiseSelect": "Load Checkpoint w/ Noise Select 🎭🅐🅓",
    "ADE_PerturbedAttentionGuidanceMultival": "PerturbedAttnGuide [Multival] 🎭🅐🅓",
    "ADE_RescaleCFGMultival": "RescaleCFG [Multival] 🎭🅐🅓",
    # Gen1 Nodes
    "ADE_AnimateDiffLoaderGen1": "AnimateDiff Loader 🎭🅐🅓①",
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
    # MotionCtrl Nodes
    LoadMotionCtrlCMCM.NodeID: LoadMotionCtrlCMCM.NodeName,
    LoadMotionCtrlOMCM.NodeID: LoadMotionCtrlOMCM.NodeName,
    ApplyAnimateDiffMotionCtrlModel.NodeID: ApplyAnimateDiffMotionCtrlModel.NodeName,
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
    # FancyVideo
    ApplyAnimateDiffFancyVideo.NodeID: ApplyAnimateDiffFancyVideo.NodeName,
    # HelloMeme
    TestHMRefNetInjection.NodeID: TestHMRefNetInjection.NodeName,
    # Deprecated Nodes
    "ADE_AnimateDiffLoaderWithContext": "AnimateDiff Loader [Legacy] 🎭🅐🅓①",
    "AnimateDiffLoaderV1": "🚫AnimateDiff Loader [DEPRECATED] 🎭🅐🅓",
    "ADE_AnimateDiffLoaderV1Advanced": "🚫AnimateDiff Loader (Advanced) [DEPRECATED] 🎭🅐🅓",
    "ADE_AnimateDiffCombine": "🚫AnimateDiff Combine [DEPRECATED, Use Video Combine (VHS) Instead!] 🎭🅐🅓",
    "ADE_AnimateDiffModelSettings_Release": "🚫[DEPR] Motion Model Settings 🎭🅐🅓①",
    "ADE_AnimateDiffModelSettingsSimple": "🚫[DEPR] Motion Model Settings (Simple) 🎭🅐🅓①",
    "ADE_AnimateDiffModelSettings": "🚫[DEPR] Motion Model Settings (Advanced) 🎭🅐🅓①",
    "ADE_AnimateDiffModelSettingsAdvancedAttnStrengths": "🚫[DEPR] Motion Model Settings (Adv. Attn) 🎭🅐🅓①",
}
