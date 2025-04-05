from typing import Callable

import math
import torch
from torch import Tensor
from torch.nn.functional import group_norm
from einops import rearrange
import collections
import threading

import comfy.model_management
import comfy.model_patcher
import comfy.patcher_extension
import comfy.samplers
import comfy.sampler_helpers
import comfy.utils
from comfy.controlnet import ControlBase
from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrapperExecutor, WrappersMP
import comfy.conds
import comfy.ops
if hasattr(comfy, 'multigpu'):
    import comfy.multigpu

from .context import ContextFuseMethod, ContextSchedules, get_context_weights, get_context_windows
from .context_extras import ContextRefHandler, NaiveReuseHandler
from .sample_settings import SampleSettings, NoisedImageToInject
from .utils_model import vae_encode_raw_batched, vae_decode_raw_batched
from .utils_motion import composite_extend, prepare_mask_batch, extend_to_batch_size
from .model_injection import InjectionParams, ModelPatcherHelper, MotionModelGroup, get_mm_attachment
from .motion_module_ad import AnimateDiffFormat, AnimateDiffInfo, AnimateDiffVersion
from .adapter_hellomeme import HMRefConst, HMRefStates, get_hmref_attachment, create_hmref_apply_model_wrapper
from .logger import logger


##################################################################################
######################################################################
# Global variable to use to more conveniently hack variable access into samplers
class AnimateDiffGlobalState:
    def __init__(self):
        self.model_patcher: ModelPatcher = None
        self.motion_models: MotionModelGroup = None

        self.model_patcher_devices: dict[torch.device, ModelPatcher] = {}
        self.motion_models_devices: dict[torch.device, MotionModelGroup] = {}

        self.params: InjectionParams = None
        self.sample_settings: SampleSettings = None
        self.callback_output_dict: dict[str] = {}
        self.function_injections: FunctionInjectionHolder = None
        self.reset()

    def initialize(self, model: BaseModel, model_options: dict[str]):
        # this function is to be run in sampling func
        if not self.initialized:
            self.initialized = True
            # initialize multigpu stuff
            if "multigpu_clones" in model_options:
                self.model_patcher_devices = model_options["multigpu_clones"].copy()
                for device, patcher in self.model_patcher_devices.items():
                    mm_list = ModelPatcherHelper(patcher).get_motion_models()
                    if len(mm_list) > 0:
                        self.motion_models_devices[device] = MotionModelGroup(mm_list)
            else:
                self.model_patcher_devices[self.model_patcher.load_device] = self.model_patcher
                if self.motion_models is not None:
                    self.motion_models_devices[self.model_patcher.load_device] = self.motion_models
            # initialize timesteps
            for device, motion_models in self.motion_models_devices.items():
                base_model = self.model_patcher_devices[device].model
                motion_models.initialize_timesteps(base_model)
            if self.params.context_options is not None:
                self.params.context_options.initialize_timesteps(model)
            if self.sample_settings.custom_cfg is not None:
                self.sample_settings.custom_cfg.initialize_timesteps(model)

    def prepare_current_keyframes(self, x: Tensor, timestep: Tensor, transformer_options: dict[str, Tensor]):
        for motion_models in self.motion_models_devices.values():
            motion_models.prepare_current_keyframe(x=x, t=timestep, transformer_options=transformer_options)
        if self.params.context_options is not None:
            self.params.context_options.prepare_current(t=timestep, transformer_options=transformer_options)
        if self.sample_settings.custom_cfg is not None:
            self.sample_settings.custom_cfg.prepare_current_keyframe(t=timestep, transformer_options=transformer_options)

    def perform_special_model_features(self, conds: list, x_in: Tensor, model_options: dict[str]):
        # make sure multigpu clones of same model are not causing duplicate wraps
        clone_uuids = set()
        for device, motion_models in self.motion_models_devices.items():
            model: BaseModel = self.model_patcher_devices[device].model
            special_models = motion_models.get_special_models()
            if len(special_models) > 0:
                for special_model in special_models:
                    has_clone_base_uuid = hasattr(special_model, "clone_base_uuid") # backwards compatibility
                    if special_model.model.is_in_effect():
                        attachment = get_mm_attachment(special_model)
                        if attachment.is_pia(special_model):
                            special_model.model.inject_unet_conv_in_pia_fancyvideo(model)
                            if not has_clone_base_uuid or special_model.clone_base_uuid not in clone_uuids:
                                if has_clone_base_uuid:
                                    clone_uuids.add(special_model.clone_base_uuid)
                                conds = get_conds_with_c_concat(conds,
                                                                attachment.get_pia_c_concat(model, x_in))
                        elif attachment.is_fancyvideo(special_model):
                            # TODO: handle other weights
                            special_model.model.inject_unet_conv_in_pia_fancyvideo(model)
                            if not has_clone_base_uuid or special_model.clone_base_uuid not in clone_uuids:
                                if has_clone_base_uuid:
                                    clone_uuids.add(special_model.clone_base_uuid)
                                conds = get_conds_with_c_concat(conds,
                                                                attachment.get_fancy_c_concat(model, x_in))
                                # add fps_embedding/motion_embedding patches
                                emb_patches = special_model.model.get_fancyvideo_emb_patches(dtype=x_in.dtype, device=x_in.device)
                                transformer_patches = model_options["transformer_options"].get("patches", {})
                                transformer_patches["emb_patch"] = emb_patches
                                model_options["transformer_options"]["patches"] = transformer_patches
        return conds

    def restore_special_model_features(self):
        for device, motion_models in self.motion_models_devices.items():
            model: BaseModel = self.model_patcher_devices[device].model
            special_models = motion_models.get_special_models()
            if len(special_models) > 0:
                for special_model in reversed(special_models):
                    attachment = get_mm_attachment(special_model)
                    if attachment.is_pia(special_model):
                        special_model.model.restore_unet_conv_in_pia_fancyvideo(model)
                    elif attachment.is_fancyvideo(special_model):
                        # TODO: fill out
                        special_model.model.restore_unet_conv_in_pia_fancyvideo(model)

    def interrupt_processing(self):
        self.processing_interrupted = True

    def is_processing_interrupted(self):
        return self.processing_interrupted

    def reset(self):
        self.processing_interrupted = False
        self.initialized = False
        self.hooks_initialized = False
        self.start_step: int = 0
        self.last_step: int = 0
        self.current_step: int = 0
        self.total_steps: int = 0
        self.callback_output_dict.clear()
        self.callback_output_dict = {}
        # model patchers
        self.model_patcher_devices.clear()
        if self.model_patcher is not None:
            del self.model_patcher
            self.model_patcher = None
        # motion models
        self.motion_models_devices.clear()
        if self.motion_models is not None:
            del self.motion_models
            self.motion_models = None
        # other
        if self.params is not None:
            self.params.context_options.reset()
            del self.params
            self.params = None
        if self.sample_settings is not None:
            del self.sample_settings
            self.sample_settings = None
        if self.function_injections is not None:
            del self.function_injections
            self.function_injections = None

    def update_with_inject_params(self, params: InjectionParams):
        self.params = params

    def is_using_sliding_context(self):
        return self.params is not None and self.params.is_using_sliding_context()

    def create_exposed_params(self):
        # This dict will be exposed to be used by other extensions
        # DO NOT change any of the key names
        # or I will find you 👁.👁
        return {
            "full_length": self.params.full_length,
            "context_length": self.params.context_options.context_length,
            "sub_idxs": self.params.sub_idxs,
        }
######################################################################
##################################################################################


##################################################################################
#### Code Injection ##################################################
def unlimited_memory_required(*args, **kwargs):
    return 0


def groupnorm_mm_factory(params: InjectionParams, manual_cast=False):
    def groupnorm_mm_forward(self, input: Tensor) -> Tensor:
        # axes_factor normalizes batch based on total conds and unconds passed in batch;
        # the conds and unconds per batch can change based on VRAM optimizations that may kick in
        if not params.is_using_sliding_context():
            batched_conds = input.size(0)//params.full_length
        else:
            batched_conds = input.size(0)//params.context_options.context_length

        input = rearrange(input, "(b f) c h w -> b c f h w", b=batched_conds)
        if manual_cast:
            weight, bias = comfy.ops.cast_bias_weight(self, input)
        else:
            weight, bias = self.weight, self.bias
        input = group_norm(input, self.num_groups, weight, bias, self.eps)
        input = rearrange(input, "b c f h w -> (b f) c h w", b=batched_conds)
        return input
    return groupnorm_mm_forward


def create_special_model_apply_model_wrapper(model_options: dict):
    comfy.patcher_extension.add_wrapper_with_key(WrappersMP.APPLY_MODEL,
                                                 "ADE_special_model_apply_model",
                                                 _apply_model_wrapper,
                                                 model_options, is_model_options=True)

def _apply_model_wrapper(executor, *args, **kwargs):
    # args (from BaseModel._apply_model):
    # 0: x
    # 1: t
    # 2: c_concat
    # 3: c_crossattn
    # 4: control
    # 5: transformer_options
    x: Tensor = args[0]
    transformer_options = args[5]
    cond_or_uncond = transformer_options["cond_or_uncond"]
    ad_params = transformer_options["ad_params"]
    ADGS: AnimateDiffGlobalState = transformer_options["ADGS"]
    motion_models = ADGS.motion_models_devices.get(x.device, None)
    if motion_models is not None:
        for motion_model in motion_models.models:
            attachment = get_mm_attachment(motion_model)
            attachment.prepare_alcmi2v_features(motion_model, x=x, cond_or_uncond=cond_or_uncond, ad_params=ad_params, latent_format=executor.class_obj.latent_format)
            attachment.prepare_camera_features(motion_model, x=x, cond_or_uncond=cond_or_uncond, ad_params=ad_params)
            attachment.prepare_motionctrl_camera(motion_model, x=x, transformer_options=transformer_options)
    del x
    return executor(*args, **kwargs)


def create_diffusion_model_groupnormed_wrapper(model_options: dict, inject_helper: 'GroupnormInjectHelper'):
    comfy.patcher_extension.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL,
                                                 "ADE_groupnormed_diffusion_model",
                                                 _diffusion_model_groupnormed_wrapper_factory(inject_helper),
                                                 model_options, is_model_options=True)

def _diffusion_model_groupnormed_wrapper_factory(inject_helper: 'GroupnormInjectHelper'):
    def _diffusion_model_groupnormed_wrapper(executor, *args, **kwargs):
        with inject_helper:
            return executor(*args, **kwargs)
    return _diffusion_model_groupnormed_wrapper
######################################################################
##################################################################################


def create_prepare_sampling_wrapper(model_options: dict, params: InjectionParams):
    # keep backwards compatibility
    if hasattr(WrappersMP, "PREPARE_SAMPLING"):
        comfy.patcher_extension.add_wrapper_with_key(WrappersMP.PREPARE_SAMPLING,
                                                     "ADE_prepare_sampling",
                                                     _prepare_sampling_wrapper_factory(params),
                                                     model_options, is_model_options=True)


def _prepare_sampling_wrapper_factory(params: InjectionParams):
    def _prepare_sampling_wrapper(executor, model: ModelPatcher, noise_shape: Tensor, *args, **kwargs):
        # TODO: handle various dims instead of defaulting to 0th
        # limit noise_shape length to context_length for more accurate vram use estimation
        noise_shape = [min(noise_shape[0], params.context_options.context_length)] + list(noise_shape[1:])
        return executor(model, noise_shape, *args, **kwargs)
    return _prepare_sampling_wrapper


def apply_params_to_motion_models(helper: ModelPatcherHelper, params: InjectionParams, model_options: dict[str]):
    params = params.clone()
    for context in params.context_options.contexts:
        if context.context_schedule == ContextSchedules.VIEW_AS_CONTEXT:
            context.context_length = params.full_length
    # TODO: check (and message) should be different based on use_on_equal_length setting
    if params.context_options.context_length:
        pass

    allow_equal = params.context_options.use_on_equal_length
    if params.context_options.context_length:
        enough_latents = params.full_length >= params.context_options.context_length if allow_equal else params.full_length > params.context_options.context_length
    else:
        enough_latents = False
    if params.context_options.context_length and enough_latents:
        logger.info(f"Sliding context window sampling activated - latents passed in ({params.full_length}) greater than context_length {params.context_options.context_length}.")
        create_prepare_sampling_wrapper(model_options, params)
    else:
        logger.info(f"Regular sampling activated - latents passed in ({params.full_length}) less or equal to context_length {params.context_options.context_length}.")
        params.reset_context()
    if helper.get_motion_models():
        # if no context_length, treat video length as intended AD frame window
        if not params.context_options.context_length:
            for motion_model in helper.get_motion_models():
                if not motion_model.model.is_length_valid_for_encoding_max_len(params.full_length):
                    raise ValueError(f"Without a context window, AnimateDiff model {motion_model.model.mm_info.mm_name} has upper limit of {motion_model.model.encoding_max_len} frames, but received {params.full_length} latents.")
            helper.set_video_length(params.full_length, params.full_length)
        # otherwise, treat context_length as intended AD frame window
        else:
            for motion_model in helper.get_motion_models():
                view_options = params.context_options.view_options
                context_length = view_options.context_length if view_options else params.context_options.context_length
                if not motion_model.model.is_length_valid_for_encoding_max_len(context_length):
                    raise ValueError(f"AnimateDiff model {motion_model.model.mm_info.mm_name} has upper limit of {motion_model.model.encoding_max_len} frames for a context window, but received context length of {params.context_options.context_length}.")
            helper.set_video_length(params.context_options.context_length, params.full_length)
        # inject model
        module_str = "modules" if len(helper.get_motion_models()) > 1 else "module"
        logger.info(f"Using motion {module_str} {helper.get_name_string(show_version=True)}.")
    return params


class FunctionInjectionHolder:
    def __init__(self):
        self.temp_uninjector: GroupnormUninjectHelper = GroupnormUninjectHelper()
        self.groupnorm_injector: GroupnormInjectHelper = GroupnormInjectHelper()
    
    def inject_functions(self, helper: ModelPatcherHelper, params: InjectionParams, model_options: dict):
        # Save Original Functions - order must match between here and restore_functions
        self.orig_memory_required = None
        self.orig_groupnorm_forward = torch.nn.GroupNorm.forward # used to normalize latents to remove "flickering" of colors/brightness between frames
        self.orig_groupnorm_forward_comfy_cast_weights = comfy.ops.disable_weight_init.GroupNorm.forward_comfy_cast_weights
        self.orig_sampling_function = comfy.samplers.sampling_function # used to support sliding context windows in samplers
        # Inject Functions
        if params.unlimited_area_hack:
            # allows for "unlimited area hack" to prevent halving of conds/unconds
            self.orig_memory_required = helper.model.model.memory_required
            helper.model.model.memory_required = unlimited_memory_required
        if helper.get_motion_models():
            # only apply groupnorm hack if PIA, v2 and not properly applied, or v1
            info: AnimateDiffInfo = helper.get_motion_models()[0].model.mm_info
            if ((info.mm_format == AnimateDiffFormat.PIA) or
                (info.mm_version == AnimateDiffVersion.V2 and not params.apply_v2_properly) or
                (info.mm_version == AnimateDiffVersion.V1)):
                self.inject_groupnorm_forward = groupnorm_mm_factory(params)
                self.inject_groupnorm_forward_comfy_cast_weights = groupnorm_mm_factory(params, manual_cast=True)
                self.groupnorm_injector = GroupnormInjectHelper(self)
                create_diffusion_model_groupnormed_wrapper(model_options, self.groupnorm_injector)
                # if mps device (Apple Silicon), disable batched conds to avoid black images with groupnorm hack
                try:
                    if helper.model.load_device.type == "mps":
                        self.orig_memory_required = helper.model.model.memory_required
                        helper.model.model.memory_required = unlimited_memory_required
                except Exception:
                    pass
            # if AnimateLCM-I2V, CameraCtrl, or MotionCtrl present, inject apply_model to handle correctly
            for motion_model in helper.get_motion_models():
                if motion_model.model.needs_apply_model_wrapper():
                    create_special_model_apply_model_wrapper(model_options)
                    break
            del info
        comfy.samplers.sampling_function = evolved_sampling_function
        # create temp_uninjector to help facilitate uninjecting functions
        self.temp_uninjector = GroupnormUninjectHelper(self)

    def restore_functions(self, helper: ModelPatcherHelper):
        # Restoration
        try:
            if self.orig_memory_required is not None:
                helper.model.model.memory_required = self.orig_memory_required
            torch.nn.GroupNorm.forward = self.orig_groupnorm_forward
            comfy.ops.disable_weight_init.GroupNorm.forward_comfy_cast_weights = self.orig_groupnorm_forward_comfy_cast_weights
            comfy.samplers.sampling_function = self.orig_sampling_function
        except AttributeError:
            logger.error("Encountered AttributeError while attempting to restore functions - likely, an error occured while trying " + \
                         "to save original functions before injection, and a more specific error was thrown by ComfyUI.")


class GroupnormUninjectHelper:
    def __init__(self, holder: FunctionInjectionHolder=None):
        self.holder = holder
        self.previous_gn_forward = None
        self.previous_dwi_gn_cast_weights = None
    
    def __enter__(self):
        if self.holder is None:
            return self
        # backup current groupnorm funcs
        self.previous_gn_forward = torch.nn.GroupNorm.forward
        self.previous_dwi_gn_cast_weights = comfy.ops.disable_weight_init.GroupNorm.forward_comfy_cast_weights
        # restore groupnorm to default state
        torch.nn.GroupNorm.forward = self.holder.orig_groupnorm_forward
        comfy.ops.disable_weight_init.GroupNorm.forward_comfy_cast_weights = self.holder.orig_groupnorm_forward_comfy_cast_weights
        return self

    def __exit__(self, *args, **kwargs):
        if self.holder is None:
            return
        # bring groupnorm back to previous state
        torch.nn.GroupNorm.forward = self.previous_gn_forward
        comfy.ops.disable_weight_init.GroupNorm.forward_comfy_cast_weights = self.previous_dwi_gn_cast_weights
        self.previous_gn_forward = None
        self.previous_dwi_gn_cast_weights = None


class GroupnormInjectHelper:
    def __init__(self, holder: FunctionInjectionHolder=None):
        self.holder = holder
        self.previous_gn_forward = None
        self.previous_dwi_gn_cast_weights = None
    
    def __enter__(self):
        if self.holder is None:
            return self
        # store previous gn_forward
        self.previous_gn_forward = torch.nn.GroupNorm.forward
        self.previous_dwi_gn_cast_weights = comfy.ops.disable_weight_init.GroupNorm.forward_comfy_cast_weights
        # inject groupnorm functions
        torch.nn.GroupNorm.forward = self.holder.inject_groupnorm_forward
        comfy.ops.disable_weight_init.GroupNorm.forward_comfy_cast_weights = self.holder.inject_groupnorm_forward_comfy_cast_weights
        return self

    def __exit__(self, *args, **kwargs):
        if self.holder is None:
            return
        # bring groupnorm back to previous state
        torch.nn.GroupNorm.forward = self.previous_gn_forward
        comfy.ops.disable_weight_init.GroupNorm.forward_comfy_cast_weights = self.previous_dwi_gn_cast_weights
        self.previous_gn_forward = None
        self.previous_dwi_gn_cast_weights = None     


def outer_sample_wrapper(executor: WrapperExecutor, *args, **kwargs):
    # NOTE: OUTER_SAMPLE wrapper patch in ModelPatcher
    latents = None
    cached_latents = None
    cached_noise = None
    function_injections = FunctionInjectionHolder()
    hmref_attachment = None

    try:
        guider: comfy.samplers.CFGGuider = executor.class_obj
        helper = ModelPatcherHelper(guider.model_patcher)

        orig_model_options = guider.model_options
        guider.model_options = comfy.model_patcher.create_model_options_clone(guider.model_options)
        # create ADGS in transformer_options
        ADGS = AnimateDiffGlobalState()
        guider.model_options["transformer_options"]["ADGS"] = ADGS

        args = list(args)
        # clone params from model
        params = helper.get_params().clone()
        # get amount of latents passed in, and store in params
        noise: Tensor = args[0]
        latents: Tensor = args[1]
        params.full_length = latents.size(0)
        # reset global state
        ADGS.reset()

        # apply custom noise, if needed
        disable_noise = math.isclose(noise.max(), 0.0)
        seed = args[-1]

        # apply params to motion model
        params = apply_params_to_motion_models(helper, params, model_options=guider.model_options)

        # store and inject funtions
        function_injections.inject_functions(helper, params, guider.model_options)

        # prepare noise_extra_args for noise generation purposes
        noise_extra_args = {"disable_noise": disable_noise}
        params.set_noise_extra_args(noise_extra_args)
        # if noise is not disabled, do noise stuff
        if not disable_noise:
            noise = helper.get_sample_settings().prepare_noise(seed, latents, noise, extra_args=noise_extra_args, force_create_noise=False)
        # handle AncestralOptions, if present
        if helper.get_sample_settings().ancestral_opts is not None:
            helper.get_sample_settings().ancestral_opts.add_wrapper_sampler_sample(guider.model_options, seed)

        # callback setup
        original_callback = args[-3]
        def ad_callback(step, x0, x, total_steps):
            if original_callback is not None:
                original_callback(step, x0, x, total_steps)
            # store denoised latents if image_injection will be used
            if not helper.get_sample_settings().image_injection.is_empty():
                ADGS.callback_output_dict["x0"] = x0
            # update GLOBALSTATE for next iteration
            ADGS.current_step = ADGS.start_step + step + 1
        args[-3] = ad_callback
        ADGS.model_patcher = helper.model
        ADGS.motion_models = MotionModelGroup(helper.get_motion_models())
        ADGS.sample_settings = helper.get_sample_settings()
        ADGS.function_injections = function_injections

        # apply adapt_denoise_steps - does not work here! would need to mess with this elsewhere...
        # TODO: implement proper wrapper to handle this feature...

        iter_opts = helper.get_sample_settings().iteration_opts
        iter_opts.initialize(latents)
        # cache initial noise and latents, if needed
        if iter_opts.cache_init_latents:
            cached_latents = latents.clone()
        if iter_opts.cache_init_noise:
            cached_noise = noise.clone()
        # prepare iter opts preprocess kwargs, if needed
        iter_kwargs = {}
        # NOTE: original KSampler stuff is not doable here, so skipping...

    
        # NOTE: this will never be used as I have hidden HelloMeme RefNet nodes from being loaded
        # if have HMRef, then do what's needed
        hmref_attachment = get_hmref_attachment(helper.model)
        if hmref_attachment is not None:
            ref_latents = None
            #sigmas: Tensor = args[3]
            try:
                hmref_attachment.prepare_ref_latent(helper.model.model, x=latents)
                # NOTE: trying out using the hmrefnet more like other types of refnets
                create_hmref_apply_model_wrapper(guider.model_options)
                # ref_latents = hmref_attachment.prepare_ref_latent(helper.model.model, x=latents)
                # ref_sigmas = torch.tensor([helper.model.model.model_sampling.sigma_min, torch.tensor(0.0)]).to(device=sigmas.device, dtype=sigmas.dtype)
                # new_args = args.copy()
                # new_args[3] = ref_sigmas
                # make sure transformer_options has necessary HMREF stuff
                # guider.model_options["transformer_options"][HMRefConst.REF_STATES] = HMRefStates()
                # guider.model_options["transformer_options"][HMRefConst.REF_MODE] = HMRefConst.WRITE
                # ADGS.update_with_inject_params(params)
                # ADGS.start_step = 0
                # ADGS.current_step = ADGS.start_step
                # ADGS.last_step = 0
                # executor(*tuple(new_args), **kwargs)
                # guider.model_options["transformer_options"][HMRefConst.REF_MODE] = HMRefConst.READ
            finally:
                del ref_latents
                #del sigmas

        for curr_i in range(iter_opts.iterations):
            # handle GLOBALSTATE vars and step tally
            # NOTE: only KSampler/KSampler (Advanced) would have steps;
            # explore modifying ComfyUI to provide this when possible?
            ADGS.update_with_inject_params(params)
            ADGS.start_step = kwargs.get("start_step") or 0
            ADGS.current_step = ADGS.start_step
            ADGS.last_step = kwargs.get("last_step") or 0
            if iter_opts.iterations > 1:
                logger.info(f"Iteration {curr_i+1}/{iter_opts.iterations}")
            # perform any iter_opts preprocessing on latents
            latents, noise = iter_opts.preprocess_latents(curr_i=curr_i, model=helper.model, latents=latents, noise=noise,
                                                          cached_latents=cached_latents, cached_noise=cached_noise,
                                                          seed=seed,
                                                          sample_settings=helper.get_sample_settings(), noise_extra_args=noise_extra_args,
                                                          **iter_kwargs)
            if helper.get_sample_settings().noise_calibration is not None:
                    latents, noise = helper.get_sample_settings().noise_calibration.perform_calibration(sample_func=executor, model=helper.model, latents=latents, noise=noise,
                                                                                                 is_custom=True, args=args, kwargs=kwargs)
            # finalize latent_image in args
            args[0] = noise
            args[1] = latents

            helper.pre_run()

            if ADGS.sample_settings.image_injection.is_empty():
                latents = executor(*tuple(args), **kwargs)
            else:
                ADGS.sample_settings.image_injection.initialize_timesteps(helper.model.model)
                sigmas = args[3]
                sigmas_list, injection_list = ADGS.sample_settings.image_injection.custom_ksampler_get_injections(helper.model, sigmas)
                # useful logging
                if len(injection_list) > 0:
                    inj_str = "s" if len(injection_list) > 1 else ""
                    logger.info(f"Found {len(injection_list)} applicable image injection{inj_str}; sampling will be split into {len(sigmas_list)}.")
                else:
                    logger.info(f"Found 0 applicable image injections within the step bounds of this sampler; sampling unaffected.")
                is_first = True
                new_noise = noise
                for i in range(len(sigmas_list)):
                    args[0] = new_noise
                    args[1] = latents
                    args[3] = sigmas_list[i]
                    latents = executor(*tuple(args), **kwargs)
                    if is_first:
                        new_noise = torch.zeros_like(latents)
                    # if injection expected, perform injection
                    if i < len(injection_list):
                        to_inject = injection_list[i]
                        latents = perform_image_injection(ADGS, helper.model.model, latents, to_inject)
        return latents
    finally:
        guider.model_options = orig_model_options
        del noise
        del latents
        del cached_latents
        del cached_noise
        del orig_model_options
        if hmref_attachment is not None:
            hmref_attachment.cleanup()
        del hmref_attachment
        # reset global state
        ADGS.reset()
        # clean motion_models
        helper.cleanup_motion_models()
        # restore injected functions
        function_injections.restore_functions(helper)
        del function_injections
        del helper


def evolved_sampling_function(model, x: Tensor, timestep: Tensor, uncond, cond, cond_scale, model_options: dict={}, seed=None):
    ADGS: AnimateDiffGlobalState = model_options["transformer_options"]["ADGS"]
    ADGS.initialize(model, model_options)
    ADGS.prepare_current_keyframes(x=x, timestep=timestep, transformer_options=model_options["transformer_options"])
    try:
        # add AD/evolved-sampling params to model_options (transformer_options)
        model_options = model_options.copy()
        if "transformer_options" not in model_options:
            model_options["transformer_options"] = {}
        else:
            model_options["transformer_options"] = model_options["transformer_options"].copy()
        model_options["transformer_options"]["ad_params"] = ADGS.create_exposed_params()

        cond, uncond = ADGS.perform_special_model_features([cond, uncond], x, model_options)

        # only use cfg1_optimization if not using custom_cfg or explicitly set to 1.0
        uncond_ = uncond
        if ADGS.sample_settings.custom_cfg is None and math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
            uncond_ = None
        elif ADGS.sample_settings.custom_cfg is not None:
            cfg_multival = ADGS.sample_settings.custom_cfg.cfg_multival
            if type(cfg_multival) != Tensor and math.isclose(cfg_multival, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
                uncond_ = None
            del cfg_multival

        cond_pred, uncond_pred = comfy.samplers.calc_cond_batch(model, [cond, uncond_], x, timestep, model_options)

        if ADGS.sample_settings.custom_cfg is not None:
            cond_scale = ADGS.sample_settings.custom_cfg.get_cfg_scale(cond_pred)
            model_options = ADGS.sample_settings.custom_cfg.get_model_options(model_options)
        
        return comfy.samplers.cfg_function(model, cond_pred, uncond_pred, cond_scale, x, timestep, model_options, cond, uncond)
    finally:
        ADGS.restore_special_model_features()


def perform_image_injection(ADGS: AnimateDiffGlobalState, model: BaseModel, latents: Tensor, to_inject: NoisedImageToInject) -> Tensor:
    # NOTE: the latents here have already been process_latent_out'ed
    # get currently used models so they can be properly reloaded after perfoming VAE Encoding
    cached_loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
    try:
        orig_device = latents.device
        orig_dtype = latents.dtype
        # follow same steps as in KSampler Custom to get same denoised_x0 value
        x0 = ADGS.callback_output_dict.get("x0", None)
        if x0 is None:
            return latents
        # x0 should be process_latent_out'ed to match expected state of latents between nodes
        x0 = model.process_latent_out(x0)
    
        # first, decode x0 into images, and then re-encode
        decoded_images = vae_decode_raw_batched(to_inject.vae, x0)
        encoded_x0 = vae_encode_raw_batched(to_inject.vae, decoded_images)

        # get difference between sampled latents and encoded_x0
        latents = latents.to(device=encoded_x0.device)
        encoded_x0 = latents - encoded_x0

        # get mask, or default to full mask
        mask = to_inject.mask
        b, c, h, w = encoded_x0.shape
        # need to resize images and masks to match expected dims
        if mask is None:
            mask = torch.ones(1, h, w)
        if to_inject.invert_mask:
            mask = 1.0 - mask
        opts = to_inject.img_inject_opts
        # composite decoded_x0 with image to inject;
        # make sure to move dims to match expectation of (b,c,h,w)
        composited = composite_extend(destination=decoded_images.movedim(-1, 1), source=to_inject.image.movedim(-1, 1), x=opts.x, y=opts.y, mask=mask,
                                      multiplier=to_inject.vae.downscale_ratio, resize_source=to_inject.resize_image).movedim(1, -1)
        # encode composited to get latent representation
        composited = vae_encode_raw_batched(to_inject.vae, composited)
        # add encoded_x0 diff to composited 
        composited += encoded_x0
        if type(to_inject.strength_multival) == float and math.isclose(1.0, to_inject.strength_multival):
            return composited.to(dtype=orig_dtype, device=orig_device)
        strength = to_inject.strength_multival
        if type(strength) == Tensor:
            strength = extend_to_batch_size(prepare_mask_batch(strength, composited.shape), b)
        return (composited * strength + latents * (1.0 - strength)).to(dtype=orig_dtype, device=orig_device)
    finally:
        comfy.model_management.load_models_gpu(cached_loaded_models)


def prepare_control_objects(control: ControlBase, full_idxs: list[int], ADGS: AnimateDiffGlobalState, device=None):
    # get controlnet matching device (multigpu only)
    if device is not None:
        control = control.get_instance_for_device(device)
    if control.previous_controlnet is not None:
        prepare_control_objects(control.previous_controlnet, full_idxs, ADGS, device)
    if not hasattr(control, "sub_idxs"):
        raise ValueError(f"Control type {type(control).__name__} may not support required features for sliding context window; \
                            use ControlNet nodes from Kosinkadink/ComfyUI-Advanced-ControlNet, or make sure ComfyUI-Advanced-ControlNet is updated.")
    if not hasattr(control, "ACN_VERSION"):
        control.sub_idxs = full_idxs
        control.full_latent_length = ADGS.params.full_length
        control.context_length = ADGS.params.context_options.context_length
    return control

# initial sliding_calc_conds_batch inspired by ashen's initial hack for 16-frame sliding context:
# https://github.com/comfyanonymous/ComfyUI/compare/master...ashen-sensored:ComfyUI:master
def get_resized_cond(cond_in, x_in: Tensor, full_idxs: list[int], context_length: int, ADGS: AnimateDiffGlobalState, device=None) -> list:
    if cond_in is None:
        return None
    # reuse or resize cond items to match context requirements
    resized_cond = []
    # cond object is a list containing a dict - outer list is irrelevant, so just loop through it
    for actual_cond in cond_in:
        resized_actual_cond = actual_cond.copy()
        # now we are in the inner dict - "pooled_output" is a tensor, "control" is a ControlBase object, "model_conds" is dictionary
        for key in actual_cond:
            try:
                cond_item = actual_cond[key]
                if isinstance(cond_item, Tensor):
                    # check that tensor is the expected length - x.size(0)
                    if cond_item.size(0) == x_in.size(0):
                        # if so, it's subsetting time - tell controls the expected indeces so they can handle them
                        actual_cond_item = cond_item[full_idxs]
                        resized_actual_cond[key] = actual_cond_item.to(device)
                    else:
                        resized_actual_cond[key] = cond_item.to(device)
                # look for control
                elif key == "control":
                    resized_actual_cond[key] = prepare_control_objects(cond_item, full_idxs, ADGS, device)
                elif isinstance(cond_item, dict):
                    new_cond_item = cond_item.copy()
                    # when in dictionary, look for tensors and CONDCrossAttn [comfy/conds.py] (has cond attr that is a tensor)
                    for cond_key, cond_value in new_cond_item.items():
                        if isinstance(cond_value, Tensor):
                            if cond_value.size(0) == x_in.size(0):
                                new_cond_item[cond_key] = cond_value[full_idxs].to(device)
                        # if has cond that is a Tensor, check if needs to be subset
                        elif hasattr(cond_value, "cond") and isinstance(cond_value.cond, Tensor):
                            if cond_value.cond.size(0) == x_in.size(0):
                                new_cond_item[cond_key] = cond_value._copy_with(cond_value.cond[full_idxs].to(device))
                        elif cond_key == "num_video_frames": # for SVD
                            new_cond_item[cond_key] = cond_value._copy_with(cond_value.cond)
                            new_cond_item[cond_key].cond = context_length
                    resized_actual_cond[key] = new_cond_item
                else:
                    resized_actual_cond[key] = cond_item
            finally:
                del cond_item  # just in case to prevent VRAM issues
        resized_cond.append(resized_actual_cond)
    return resized_cond


def sliding_calc_cond_batch(executor: Callable, model, conds: list[list[dict]], x_in: Tensor, timestep, model_options):
    ADGS: AnimateDiffGlobalState = model_options["transformer_options"]["ADGS"]
    if not ADGS.is_using_sliding_context():
        return executor(model, conds, x_in, timestep, model_options)

    # get context windows
    ADGS.params.context_options.step = ADGS.current_step
    context_windows = get_context_windows(ADGS.params.full_length, ADGS.params.context_options)
    enumerated_context_windows = list(enumerate(context_windows))

    if ADGS.motion_models is not None:
        ADGS.motion_models.set_view_options(ADGS.params.context_options.view_options)
    
    # prepare final conds, out_counts, and biases
    conds_final = [torch.zeros_like(x_in) for _ in conds]
    if ADGS.params.context_options.fuse_method == ContextFuseMethod.RELATIVE:
        # counts_final not used for RELATIVE fuse_method
        counts_final = [torch.ones((x_in.shape[0], 1, 1, 1), device=x_in.device) for _ in conds]
    else:
        # default counts_final initialization
        counts_final = [torch.zeros((x_in.shape[0], 1, 1, 1), device=x_in.device) for _ in conds]
    biases_final = [([0.0] * x_in.shape[0]) for _ in conds]

    CREF = ContextRefHandler()
    NAIVE = NaiveReuseHandler()
    allow_multigpu_contexts = 'multigpu_clones' in model_options
    ctxs_relative_work = None
    # need to make sure that contextref stuff gets cleaned up, no matter what
    try:
        if ADGS.params.context_options.extras.should_run_context_ref():
            CREF.initialize_step(timestep, model_options, ADGS)
            allow_multigpu_contexts = False  # ContextRef requires windows get run in a specific order
        if ADGS.params.context_options.extras.should_run_naive_reuse():
            NAIVE.initialize_step(x_in, conds)
        
        if allow_multigpu_contexts:
            cond_count = len([x for x in conds if x is not None])
            context_count = len(context_windows)
            # figure out if conds or ctxs would be better for splitting work
            cond_relative_work, idle_cond = comfy.multigpu.load_balance_devices(model_options, cond_count, return_idle_time=True)
            ctxs_relative_work, idle_ctxs = comfy.multigpu.load_balance_devices(model_options, context_count, return_idle_time=True,
                                                                                work_normalized=cond_count)
            # if splitting conds is better to splitting contexts, don't split contexts
            if idle_cond < idle_ctxs:
                allow_multigpu_contexts = False


        if allow_multigpu_contexts:
            multigpu_windows = {}
            start_idx = 0
            for device, work in ctxs_relative_work.items():
                if work == 0:
                    continue
                end_idx = start_idx + work
                multigpu_windows[device] = enumerated_context_windows[start_idx:end_idx]
                start_idx = end_idx
            first_device = list(multigpu_windows.keys())[0]
            def _handle_context_batch(device: torch.device, batch_windows, model_options_batch, results: list[list[ContextResults]]):
                model_options_batch = comfy.model_patcher.create_model_options_clone(model_options_batch)
                comfy.samplers.cast_transformer_options(model_options_batch["transformer_options"], device=device)
                with torch.no_grad():
                    results.append(evaluate_context_windows(executor, model, x_in, conds, timestep, batch_windows, model_options_batch, CREF, ADGS,
                                                       device=device, first_device=first_device))
            combined_results: list[list[ContextResults]] = []
            threads: list[threading.Thread] = []
            try:
                multigpu_entry = model_options.pop('multigpu_clones')
                # for device, thread_windows in multigpu_windows.items():
                #     _handle_context_batch(device, thread_windows, combined_results)
                for device, thread_windows in multigpu_windows.items():
                    new_thread = threading.Thread(target=_handle_context_batch, args=(device, thread_windows, model_options, combined_results))
                    threads.append(new_thread)
                    new_thread.start()
                for thread in threads:
                    thread.join()
                if ADGS.is_processing_interrupted():
                    raise comfy.model_management.InterruptProcessingException()
            finally:
                model_options['multigpu_clones'] = multigpu_entry

            for results in combined_results:
                for result in results:
                    combine_context_window_results(x_in, result.sub_conds_out, result.sub_conds, result.ctx_idxs, result.window_idx, len(enumerated_context_windows), timestep,
                                                ADGS, NAIVE, CREF, conds_final, counts_final, biases_final)
            
        else:
            for enum_window in enumerated_context_windows:
                results = evaluate_context_windows(executor, model, x_in, conds, timestep, [enum_window], model_options, CREF, ADGS)
                for result in results:
                    combine_context_window_results(x_in, result.sub_conds_out, result.sub_conds, result.ctx_idxs, result.window_idx, len(enumerated_context_windows), timestep,
                                                ADGS, NAIVE, CREF, conds_final, counts_final, biases_final)
    finally:
        CREF.cleanup(model_options)

    # handle NaiveReuse
    NAIVE.apply_cached(x_in, conds_final, counts_final, ADGS)

    # finalize conds
    if ADGS.params.context_options.fuse_method == ContextFuseMethod.RELATIVE:
        # relative is already normalized, so return as is
        del counts_final
        return conds_final
    else:
        # normalize conds via division by context usage counts
        for i in range(len(conds_final)):
            conds_final[i] /= counts_final[i]
        del counts_final
        return conds_final

ContextResults = collections.namedtuple("ContextResults", ['window_idx', 'sub_conds_out', 'sub_conds', 'ctx_idxs'])
def evaluate_context_windows(executor, model: BaseModel, x_in: Tensor, conds, timestep: Tensor, enumerated_context_windows: list[tuple[int, list[int]]],
                             model_options, CREF: ContextRefHandler, ADGS: AnimateDiffGlobalState, device=None, first_device=None):
    results: list[ContextResults] = []
    for window_idx, ctx_idxs in enumerated_context_windows:
        # allow processing to end between context window executions for faster Cancel
        if device is None:
            # non-MultiGPU execution
            comfy.model_management.throw_exception_if_processing_interrupted()
        elif first_device == device:
            # MultiGPU execution
            try:
                comfy.model_management.throw_exception_if_processing_interrupted()
                if ADGS.is_processing_interrupted():
                    break
            except comfy.model_management.InterruptProcessingException:
                ADGS.interrupt_processing()
                break
        else:
            # MultiGPU execution
            if ADGS.is_processing_interrupted():
                break
        ADGS.params.sub_idxs = ctx_idxs

        if device is None:
            motion_models_devices = ADGS.motion_models_devices.values()
        else:
            motion_models_devices = ADGS.motion_models_devices.get(device, None)
            if motion_models_devices is None:
                motion_models_devices = []
            else:
                motion_models_devices = [motion_models_devices]
            model = ADGS.model_patcher_devices[device].model
        for motion_models in motion_models_devices:
            motion_models.set_sub_idxs(ctx_idxs)
            motion_models.set_video_length(len(ctx_idxs), ADGS.params.full_length)
        # update exposed params
        model_options["transformer_options"]["ad_params"]["sub_idxs"] = ctx_idxs
        model_options["transformer_options"]["ad_params"]["context_length"] = len(ctx_idxs)
        # get subsections of x, timestep, conds
        sub_x = x_in[ctx_idxs].to(device)
        sub_timestep = timestep[ctx_idxs].to(device)
        sub_conds = [get_resized_cond(cond, x_in, ctx_idxs, len(ctx_idxs), ADGS, device) for cond in conds]

        CREF.prepare_referencecn(ctx_idxs, window_idx, model_options)
        
        sub_conds_out = executor(model, sub_conds, sub_x, sub_timestep, model_options)
        if device is not None:
            for i in range(len(sub_conds_out)):
                sub_conds_out[i] = sub_conds_out[i].to(x_in.device)
        results.append(ContextResults(window_idx, sub_conds_out, sub_conds, ctx_idxs))
    return results


def combine_context_window_results(x_in: Tensor, sub_conds_out, sub_conds, ctx_idxs: list[int], window_idx: int, total_windows: int, timestep,
                                   ADGS: AnimateDiffGlobalState, NAIVE: NaiveReuseHandler, CREF: ContextRefHandler,
                                   conds_final: list[Tensor], counts_final: list[Tensor], biases_final: list[Tensor]):
    if ADGS.params.context_options.fuse_method == ContextFuseMethod.RELATIVE:
        for pos, idx in enumerate(ctx_idxs):
            # bias is the influence of a specific index in relation to the whole context window
            bias = 1 - abs(idx - (ctx_idxs[0] + ctx_idxs[-1]) / 2) / ((ctx_idxs[-1] - ctx_idxs[0] + 1e-2) / 2)
            bias = max(1e-2, bias)
            # take weighted average relative to total bias of current idx
            for i in range(len(sub_conds_out)):
                bias_total = biases_final[i][idx]
                prev_weight = (bias_total / (bias_total + bias))
                new_weight = (bias / (bias_total + bias))
                conds_final[i][idx] = conds_final[i][idx] * prev_weight + sub_conds_out[i][pos] * new_weight
                biases_final[i][idx] = bias_total + bias
    else:
        # add conds and counts based on weights of fuse method
        weights = get_context_weights(len(ctx_idxs), x_in.shape[0], ctx_idxs, ADGS.params.context_options, sigma=timestep)
        weights_tensor = torch.Tensor(weights).to(device=x_in.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        for i in range(len(sub_conds_out)):
            conds_final[i][ctx_idxs] += sub_conds_out[i] * weights_tensor
            counts_final[i][ctx_idxs] += weights_tensor
    # handle NaiveReuse
    NAIVE.cache_first_context_results(window_idx, ctx_idxs, sub_conds, conds_final, counts_final)
    # handle ContextRef
    CREF.finalize_step()


def get_conds_with_c_concat(conds: list[dict], c_concat: comfy.conds.CONDNoiseShape):
    new_conds = []
    for cond in conds:
        resized_cond = None
        if cond is not None:
            # reuse or resize cond items to match context requirements
            resized_cond = []
            # cond object is a list containing a dict - outer list is irrelevant, so just loop through it
            for actual_cond in cond:
                resized_actual_cond = actual_cond.copy()
                # now we are in the inner dict - "pooled_output" is a tensor, "control" is a ControlBase object, "model_conds" is dictionary
                for key in actual_cond:
                    if key == "model_conds":
                        new_model_conds = actual_cond[key].copy()
                        if "c_concat" in new_model_conds:
                            new_model_conds["c_concat"] = comfy.conds.CONDNoiseShape(torch.cat(new_model_conds["c_concat"].cond, c_concat.cond, dim=1))
                        else:
                            new_model_conds["c_concat"] = c_concat
                        resized_actual_cond[key] = new_model_conds
                resized_cond.append(resized_actual_cond)
        new_conds.append(resized_cond)
    return new_conds
