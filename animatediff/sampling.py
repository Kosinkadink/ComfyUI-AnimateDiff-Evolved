from typing import Callable

import collections
import math
import torch
from torch import Tensor
from torch.nn.functional import group_norm
from einops import rearrange
from types import MethodType

import comfy.ldm.modules.attention as attention
from comfy.ldm.modules.diffusionmodules import openaimodel
import comfy.model_management
import comfy.samplers
import comfy.sample
SAMPLE_FALLBACK = False
try:
    import comfy.sampler_helpers
except ImportError:
    SAMPLE_FALLBACK = True
import comfy.utils
from comfy.controlnet import ControlBase
from comfy.model_base import BaseModel
import comfy.ops

from .conditioning import COND_CONST, LoraHookGroup
from .context import ContextFuseMethod, ContextSchedules, get_context_weights, get_context_windows
from .sample_settings import IterationOptions, SampleSettings, SeedNoiseGeneration
from .utils_model import ModelTypeSD
from .model_injection import InjectionParams, ModelPatcherAndInjector, MotionModelGroup, MotionModelPatcher
from .motion_module_ad import AnimateDiffFormat, AnimateDiffInfo, AnimateDiffVersion, VanillaTemporalModule
from .logger import logger


##################################################################################
######################################################################
# Global variable to use to more conveniently hack variable access into samplers
class AnimateDiffHelper_GlobalState:
    def __init__(self):
        self.model_patcher: ModelPatcherAndInjector = None
        self.motion_models: MotionModelGroup = None
        self.params: InjectionParams = None
        self.sample_settings: SampleSettings = None
        self.reset()

    def initialize(self, model: BaseModel):
        # this function is to be run in sampling func
        if not self.initialized:
            self.initialized = True
            if self.motion_models is not None:
                self.motion_models.initialize_timesteps(model)
            if self.params.context_options is not None:
                self.params.context_options.initialize_timesteps(model)
            if self.sample_settings.custom_cfg is not None:
                self.sample_settings.custom_cfg.initialize_timesteps(model)

    def hooks_initialize(self, model: BaseModel, hook_groups: list[LoraHookGroup]):
        # this function is to be run the first time all gathered
        if not self.hooks_initialized:
            self.hooks_initialized = True
            for hook_group in hook_groups:
                for hook in hook_group.hooks:
                    hook.reset()
                    hook.initialize_timesteps(model)

    def prepare_current_keyframes(self, timestep: Tensor):
        if self.motion_models is not None:
            self.motion_models.prepare_current_keyframe(t=timestep)
        if self.params.context_options is not None:
            self.params.context_options.prepare_current_context(t=timestep)
        if self.sample_settings.custom_cfg is not None:
            self.sample_settings.custom_cfg.prepare_current_keyframe(t=timestep)

    def prepare_hooks_current_keyframes(self, timestep: Tensor, hook_groups: list[LoraHookGroup]):
        if self.model_patcher is not None:
            self.model_patcher.prepare_hooked_patches_current_keyframe(t=timestep, hook_groups=hook_groups)

    def reset(self):
        self.initialized = False
        self.hooks_initialized = False
        self.start_step: int = 0
        self.last_step: int = 0
        self.current_step: int = 0
        self.total_steps: int = 0
        if self.model_patcher is not None:
            self.model_patcher.clean_hooks()
            del self.model_patcher
            self.model_patcher = None
        if self.motion_models is not None:
            del self.motion_models
            self.motion_models = None
        if self.params is not None:
            del self.params
            self.params = None
        if self.sample_settings is not None:
            del self.sample_settings
            self.sample_settings = None

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

ADGS = AnimateDiffHelper_GlobalState()
######################################################################
##################################################################################


##################################################################################
#### Code Injection ##################################################

# refer to forward_timestep_embed in comfy/ldm/modules/diffusionmodules/openaimodel.py
def forward_timestep_embed_factory() -> Callable:
    def forward_timestep_embed(ts, x, emb, context=None, transformer_options={}, output_shape=None, time_context=None, num_video_frames=None, image_only_indicator=None):
        for layer in ts:
            if isinstance(layer, openaimodel.VideoResBlock):
                x = layer(x, emb, num_video_frames, image_only_indicator)
            elif isinstance(layer, openaimodel.TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, VanillaTemporalModule):
                x = layer(x, context)
            elif isinstance(layer, attention.SpatialVideoTransformer):
                x = layer(x, context, time_context, num_video_frames, image_only_indicator, transformer_options)
                if "transformer_index" in transformer_options:
                    transformer_options["transformer_index"] += 1
                if "current_index" in transformer_options: # keep this for backward compat, for now
                    transformer_options["current_index"] += 1
            elif isinstance(layer, attention.SpatialTransformer):
                x = layer(x, context, transformer_options)
                if "transformer_index" in transformer_options:
                    transformer_options["transformer_index"] += 1
                if "current_index" in transformer_options:  # keep this for backward compat, for now
                    transformer_options["current_index"] += 1
            elif isinstance(layer, openaimodel.Upsample):
                x = layer(x, output_shape=output_shape)
            else:
                x = layer(x)
        return x
    return forward_timestep_embed


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


def get_additional_models_factory(orig_get_additional_models: Callable, motion_models: MotionModelGroup):
    def get_additional_models_with_motion(*args, **kwargs):
        models, inference_memory = orig_get_additional_models(*args, **kwargs)
        if motion_models is not None:
            for motion_model in motion_models.models:
                models.append(motion_model)
        # TODO: account for inference memory as well?
        return models, inference_memory
    return get_additional_models_with_motion

def apply_model_factory(orig_apply_model: Callable):
    def apply_model_ade_wrapper(self, *args, **kwargs):
        x: Tensor = args[0]
        cond_or_uncond = kwargs["transformer_options"]["cond_or_uncond"]
        ad_params = kwargs["transformer_options"]["ad_params"]
        if ADGS.motion_models is not None:
            for motion_model in ADGS.motion_models.models:
                motion_model.prepare_img_features(x=x, cond_or_uncond=cond_or_uncond, ad_params=ad_params, latent_format=self.latent_format)
                motion_model.prepare_camera_features(x=x, cond_or_uncond=cond_or_uncond, ad_params=ad_params)
        del x
        return orig_apply_model(*args, **kwargs)
    return apply_model_ade_wrapper
######################################################################
##################################################################################


def apply_params_to_motion_models(motion_models: MotionModelGroup, params: InjectionParams):
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
        logger.info(f"Sliding context window activated - latents passed in ({params.full_length}) greater than context_length {params.context_options.context_length}.")
    else:
        logger.info(f"Regular AnimateDiff activated - latents passed in ({params.full_length}) less or equal to context_length {params.context_options.context_length}.")
        params.reset_context()
    if motion_models is not None:
        # if no context_length, treat video length as intended AD frame window
        if not params.context_options.context_length:
            for motion_model in motion_models.models:
                if not motion_model.model.is_length_valid_for_encoding_max_len(params.full_length):
                    raise ValueError(f"Without a context window, AnimateDiff model {motion_model.model.mm_info.mm_name} has upper limit of {motion_model.model.encoding_max_len} frames, but received {params.full_length} latents.")
            motion_models.set_video_length(params.full_length, params.full_length)
        # otherwise, treat context_length as intended AD frame window
        else:
            for motion_model in motion_models.models:
                view_options = params.context_options.view_options
                context_length = view_options.context_length if view_options else params.context_options.context_length
                if not motion_model.model.is_length_valid_for_encoding_max_len(context_length):
                    raise ValueError(f"AnimateDiff model {motion_model.model.mm_info.mm_name} has upper limit of {motion_model.model.encoding_max_len} frames for a context window, but received context length of {params.context_options.context_length}.")
            motion_models.set_video_length(params.context_options.context_length, params.full_length)
        # inject model
        module_str = "modules" if len(motion_models.models) > 1 else "module"
        logger.info(f"Using motion {module_str} {motion_models.get_name_string(show_version=True)}.")
    return params


class FunctionInjectionHolder:
    def __init__(self):
        pass
    
    def inject_functions(self, model: ModelPatcherAndInjector, params: InjectionParams):
        # Save Original Functions - order must match between here and restore_functions
        self.orig_forward_timestep_embed = openaimodel.forward_timestep_embed # needed to account for VanillaTemporalModule
        self.orig_memory_required = model.model.memory_required # allows for "unlimited area hack" to prevent halving of conds/unconds
        self.orig_groupnorm_forward = torch.nn.GroupNorm.forward # used to normalize latents to remove "flickering" of colors/brightness between frames
        self.orig_groupnorm_forward_comfy_cast_weights = comfy.ops.disable_weight_init.GroupNorm.forward_comfy_cast_weights
        self.orig_sampling_function = comfy.samplers.sampling_function # used to support sliding context windows in samplers
        self.orig_get_area_and_mult = comfy.samplers.get_area_and_mult
        if SAMPLE_FALLBACK:  # for backwards compatibility, for now
            self.orig_get_additional_models = comfy.sample.get_additional_models
        else:
            self.orig_get_additional_models = comfy.sampler_helpers.get_additional_models
        self.orig_apply_model = model.model.apply_model
        # Inject Functions
        openaimodel.forward_timestep_embed = forward_timestep_embed_factory()
        if params.unlimited_area_hack:
            model.model.memory_required = unlimited_memory_required
        if model.motion_models is not None:
            # only apply groupnorm hack if not [v3 or ([not Hotshot] and SD1.5 and v2 and apply_v2_properly)]
            info: AnimateDiffInfo = model.motion_models[0].model.mm_info
            if not (info.mm_version == AnimateDiffVersion.V3 or
                    (info.mm_format not in [AnimateDiffFormat.HOTSHOTXL] and info.sd_type == ModelTypeSD.SD1_5 and info.mm_version == AnimateDiffVersion.V2 and params.apply_v2_properly)):
                torch.nn.GroupNorm.forward = groupnorm_mm_factory(params)
                comfy.ops.disable_weight_init.GroupNorm.forward_comfy_cast_weights = groupnorm_mm_factory(params, manual_cast=True)
                # if mps device (Apple Silicon), disable batched conds to avoid black images with groupnorm hack
                try:
                    if model.load_device.type == "mps":
                        model.model.memory_required = unlimited_memory_required
                except Exception:
                    pass
            # if img_encoder or camera_encoder present, inject apply_model to handle correctly
            for motion_model in model.motion_models:
                if (motion_model.model.img_encoder is not None) or (motion_model.model.camera_encoder is not None):
                    model.model.apply_model = apply_model_factory(self.orig_apply_model).__get__(model.model, type(model.model))
                    break
            del info
        comfy.samplers.sampling_function = evolved_sampling_function
        comfy.samplers.get_area_and_mult = get_area_and_mult_ADE
        if SAMPLE_FALLBACK:  # for backwards compatibility, for now
            comfy.sample.get_additional_models = get_additional_models_factory(self.orig_get_additional_models, model.motion_models)
        else:
            comfy.sampler_helpers.get_additional_models = get_additional_models_factory(self.orig_get_additional_models, model.motion_models)

    def restore_functions(self, model: ModelPatcherAndInjector):
        # Restoration
        try:
            model.model.memory_required = self.orig_memory_required
            openaimodel.forward_timestep_embed = self.orig_forward_timestep_embed
            torch.nn.GroupNorm.forward = self.orig_groupnorm_forward
            comfy.ops.disable_weight_init.GroupNorm.forward_comfy_cast_weights = self.orig_groupnorm_forward_comfy_cast_weights
            comfy.samplers.sampling_function = self.orig_sampling_function
            comfy.samplers.get_area_and_mult = self.orig_get_area_and_mult
            if SAMPLE_FALLBACK:  # for backwards compatibility, for now
                comfy.sample.get_additional_models = self.orig_get_additional_models
            else:
                comfy.sampler_helpers.get_additional_models = self.orig_get_additional_models
            model.model.apply_model = self.orig_apply_model
        except AttributeError:
            logger.error("Encountered AttributeError while attempting to restore functions - likely, an error occured while trying " + \
                         "to save original functions before injection, and a more specific error was thrown by ComfyUI.")


def motion_sample_factory(orig_comfy_sample: Callable, is_custom: bool=False) -> Callable:
    def motion_sample(model: ModelPatcherAndInjector, noise: Tensor, *args, **kwargs):
        # check if model is intended for injecting
        if type(model) != ModelPatcherAndInjector:
            return orig_comfy_sample(model, noise, *args, **kwargs)
        # otherwise, injection time
        latents = None
        cached_latents = None
        cached_noise = None
        function_injections = FunctionInjectionHolder()
        try:
            if model.sample_settings.custom_cfg is not None:
                model = model.sample_settings.custom_cfg.patch_model(model)
            # clone params from model
            params = model.motion_injection_params.clone()
            # get amount of latents passed in, and store in params
            latents: Tensor = args[-1]
            params.full_length = latents.size(0)
            # reset global state
            ADGS.reset()

            # apply custom noise, if needed
            disable_noise = kwargs.get("disable_noise") or False
            seed = kwargs["seed"]

            # apply params to motion model
            params = apply_params_to_motion_models(model.motion_models, params)

            # store and inject functions
            function_injections.inject_functions(model, params)

            # prepare noise_extra_args for noise generation purposes
            noise_extra_args = {"disable_noise": disable_noise}
            params.set_noise_extra_args(noise_extra_args)
            # if noise is not disabled, do noise stuff
            if not disable_noise:
                noise = model.sample_settings.prepare_noise(seed, latents, noise, extra_args=noise_extra_args, force_create_noise=False)

            # callback setup
            original_callback = kwargs.get("callback", None)
            def ad_callback(step, x0, x, total_steps):
                if original_callback is not None:
                    original_callback(step, x0, x, total_steps)
                # update GLOBALSTATE for next iteration
                ADGS.current_step = ADGS.start_step + step + 1
            kwargs["callback"] = ad_callback
            ADGS.model_patcher = model
            ADGS.motion_models = model.motion_models
            ADGS.sample_settings = model.sample_settings

            # apply adapt_denoise_steps
            args = list(args)
            if model.sample_settings.adapt_denoise_steps and not is_custom:
                # only applicable when denoise and steps are provided (from simple KSampler nodes)
                denoise = kwargs.get("denoise", None)
                steps = args[0]
                if denoise is not None and type(steps) == int:
                    args[0] = max(int(denoise * steps), 1)


            iter_opts = IterationOptions()
            if model.sample_settings is not None:
                iter_opts = model.sample_settings.iteration_opts
            iter_opts.initialize(latents)
            # cache initial noise and latents, if needed
            if iter_opts.cache_init_latents:
                cached_latents = latents.clone()
            if iter_opts.cache_init_noise:
                cached_noise = noise.clone()
            # prepare iter opts preprocess kwargs, if needed
            iter_kwargs = {}
            if iter_opts.need_sampler:
                # -5 for sampler_name (not custom) and sampler (custom)
                if is_custom:
                    iter_kwargs[IterationOptions.SAMPLER] = None #args[-5]
                else:
                    if SAMPLE_FALLBACK:  # backwards compatibility, for now
                        # in older comfy, model needs to be loaded to get proper model_sampling to be used for sigmas
                        comfy.model_management.load_model_gpu(model)
                        iter_model = model.model
                    else:
                        iter_model = model
                    iter_kwargs[IterationOptions.SAMPLER] = comfy.samplers.KSampler(
                        iter_model, steps=999, #steps=args[-7],
                        device=model.current_device, sampler=args[-5],
                        scheduler=args[-4], denoise=kwargs.get("denoise", None),
                        model_options=model.model_options)
                    del iter_model

            for curr_i in range(iter_opts.iterations):
                # handle GLOBALSTATE vars and step tally
                ADGS.update_with_inject_params(params)
                ADGS.start_step = kwargs.get("start_step") or 0
                ADGS.current_step = ADGS.start_step
                ADGS.last_step = kwargs.get("last_step") or 0
                ADGS.hooks_initialized = False
                if iter_opts.iterations > 1:
                    logger.info(f"Iteration {curr_i+1}/{iter_opts.iterations}")
                # perform any iter_opts preprocessing on latents
                latents, noise = iter_opts.preprocess_latents(curr_i=curr_i, model=model, latents=latents, noise=noise,
                                                              cached_latents=cached_latents, cached_noise=cached_noise,
                                                              seed=seed,
                                                              sample_settings=model.sample_settings, noise_extra_args=noise_extra_args,
                                                              **iter_kwargs)
                args[-1] = latents

                if model.motion_models is not None:
                    model.motion_models.pre_run(model)
                if model.sample_settings is not None:
                    model.sample_settings.pre_run(model)
                latents = orig_comfy_sample(model, noise, *args, **kwargs)
            return latents
        finally:
            del latents
            del noise
            del cached_latents
            del cached_noise
            # reset global state
            ADGS.reset()
            # clean motion_models
            if model.motion_models is not None:
                model.motion_models.cleanup()
            # restore injected functions
            function_injections.restore_functions(model)
            del function_injections
    return motion_sample


def evolved_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options: dict={}, seed=None):
    ADGS.initialize(model)
    ADGS.prepare_current_keyframes(timestep=timestep)

    # never use cfg1 optimization if using custom_cfg (since can have timesteps and such)
    if ADGS.sample_settings.custom_cfg is None and math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    # add AD/evolved-sampling params to model_options (transformer_options)
    model_options = model_options.copy()
    if "tranformer_options" not in model_options:
        model_options["tranformer_options"] = {}
    model_options["transformer_options"]["ad_params"] = ADGS.create_exposed_params()

    if not ADGS.is_using_sliding_context():
        cond_pred, uncond_pred = calc_cond_uncond_batch_wrapper(model, [cond, uncond_], x, timestep, model_options)
    else:
        cond_pred, uncond_pred = sliding_calc_conds_batch(model, [cond, uncond_], x, timestep, model_options)

    if hasattr(comfy.samplers, "cfg_function"):
        try:
            cached_calc_cond_batch = comfy.samplers.calc_cond_batch
            # support hooks and sliding context for PAG/other sampler_post_cfg_function tech that may use calc_cond_batch
            comfy.samplers.calc_cond_batch = wrapped_cfg_sliding_calc_cond_batch_factory(cached_calc_cond_batch)
            return comfy.samplers.cfg_function(model, cond_pred, uncond_pred, cond_scale, x, timestep, model_options, cond, uncond)
        finally:
            comfy.samplers.calc_cond_batch = cached_calc_cond_batch
    else: # for backwards compatibility, for now
        if "sampler_cfg_function" in model_options:
            args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                    "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
            cfg_result = x - model_options["sampler_cfg_function"](args)
        else:
            cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                    "sigma": timestep, "model_options": model_options, "input": x}
            cfg_result = fn(args)

        return cfg_result


def wrapped_cfg_sliding_calc_cond_batch_factory(orig_calc_cond_batch):
    def wrapped_cfg_sliding_calc_cond_batch(model, conds, x_in, timestep, model_options):
        # current call to calc_cond_batch should refer to sliding version
        try:
            current_calc_cond_batch = comfy.samplers.calc_cond_batch
            # when inside sliding_calc_conds_batch, should return to original calc_cond_batch
            comfy.samplers.calc_cond_batch = orig_calc_cond_batch
            if not ADGS.is_using_sliding_context():
                return calc_cond_uncond_batch_wrapper(model, conds, x_in, timestep, model_options)
            else:
                return sliding_calc_conds_batch(model, conds, x_in, timestep, model_options)
        finally:
            # make sure calc_cond_batch will become wrapped again
            comfy.samplers.calc_cond_batch = current_calc_cond_batch
    return wrapped_cfg_sliding_calc_cond_batch


# sliding_calc_conds_batch inspired by ashen's initial hack for 16-frame sliding context:
# https://github.com/comfyanonymous/ComfyUI/compare/master...ashen-sensored:ComfyUI:master
def sliding_calc_conds_batch(model, conds, x_in: Tensor, timestep, model_options):
    def prepare_control_objects(control: ControlBase, full_idxs: list[int]):
        if control.previous_controlnet is not None:
            prepare_control_objects(control.previous_controlnet, full_idxs)
        if not hasattr(control, "sub_idxs"):
            raise ValueError(f"Control type {type(control).__name__} may not support required features for sliding context window; \
                                use ControlNet nodes from Kosinkadink/ComfyUI-Advanced-ControlNet, or make sure ComfyUI-Advanced-ControlNet is updated.")
        control.sub_idxs = full_idxs
        control.full_latent_length = ADGS.params.full_length
        control.context_length = ADGS.params.context_options.context_length
    
    def get_resized_cond(cond_in, full_idxs: list[int], context_length: int) -> list:
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
                            resized_actual_cond[key] = actual_cond_item
                        else:
                            resized_actual_cond[key] = cond_item
                    # look for control
                    elif key == "control":
                        control_item = cond_item
                        prepare_control_objects(control_item, full_idxs)
                        resized_actual_cond[key] = control_item
                        del control_item
                    elif isinstance(cond_item, dict):
                        new_cond_item = cond_item.copy()
                        # when in dictionary, look for tensors and CONDCrossAttn [comfy/conds.py] (has cond attr that is a tensor)
                        for cond_key, cond_value in new_cond_item.items():
                            if isinstance(cond_value, Tensor):
                                if cond_value.size(0) == x_in.size(0):
                                    new_cond_item[cond_key] = cond_value[full_idxs]
                            # if has cond that is a Tensor, check if needs to be subset
                            elif hasattr(cond_value, "cond") and isinstance(cond_value.cond, Tensor):
                                if cond_value.cond.size(0) == x_in.size(0):
                                    new_cond_item[cond_key] = cond_value._copy_with(cond_value.cond[full_idxs])
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

    # get context windows
    ADGS.params.context_options.step = ADGS.current_step
    context_windows = get_context_windows(ADGS.params.full_length, ADGS.params.context_options)
    # figure out how input is split
    batched_conds = x_in.size(0)//ADGS.params.full_length

    if ADGS.motion_models is not None:
        ADGS.motion_models.set_view_options(ADGS.params.context_options.view_options)
    
    # prepare final conds, out_counts, and biases
    conds_final = [torch.zeros_like(x_in) for _ in conds]
    counts_final = [torch.zeros((x_in.shape[0], 1, 1, 1), device=x_in.device) for _ in conds]
    biases_final = [([0.0] * x_in.shape[0]) for _ in conds]

    # perform calc_conds_batch per context window
    for ctx_idxs in context_windows:
        ADGS.params.sub_idxs = ctx_idxs
        if ADGS.motion_models is not None:
            ADGS.motion_models.set_sub_idxs(ctx_idxs)
            ADGS.motion_models.set_video_length(len(ctx_idxs), ADGS.params.full_length)
        # update exposed params
        model_options["transformer_options"]["ad_params"]["sub_idxs"] = ctx_idxs
        model_options["transformer_options"]["ad_params"]["context_length"] = len(ctx_idxs)
        # account for all portions of input frames
        full_idxs = []
        for n in range(batched_conds):
            for ind in ctx_idxs:
                full_idxs.append((ADGS.params.full_length*n)+ind)
        # get subsections of x, timestep, conds
        sub_x = x_in[full_idxs]
        sub_timestep = timestep[full_idxs]
        sub_conds = [get_resized_cond(cond, full_idxs, len(ctx_idxs)) for cond in conds]

        sub_conds_out = calc_cond_uncond_batch_wrapper(model, sub_conds, sub_x, sub_timestep, model_options)

        if ADGS.params.context_options.fuse_method == ContextFuseMethod.RELATIVE:
            full_length = ADGS.params.full_length
            for pos, idx in enumerate(ctx_idxs):
                # bias is the influence of a specific index in relation to the whole context window
                bias = 1 - abs(idx - (ctx_idxs[0] + ctx_idxs[-1]) / 2) / ((ctx_idxs[-1] - ctx_idxs[0] + 1e-2) / 2)
                bias = max(1e-2, bias)
                # take weighted average relative to total bias of current idx
                # and account for batched_conds
                for i in range(len(sub_conds_out)):
                    for n in range(batched_conds):
                        bias_total = biases_final[i][(full_length*n)+idx]
                        prev_weight = (bias_total / (bias_total + bias))
                        new_weight = (bias / (bias_total + bias))
                        conds_final[i][(full_length*n)+idx] = conds_final[i][(full_length*n)+idx] * prev_weight + sub_conds_out[i][(full_length*n)+pos] * new_weight
                        biases_final[i][(full_length*n)+idx] = bias_total + bias
        else:
            # add conds and counts based on weights of fuse method
            weights = get_context_weights(len(ctx_idxs), ADGS.params.context_options.fuse_method, sigma=timestep) * batched_conds
            weights_tensor = torch.Tensor(weights).to(device=x_in.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            for i in range(len(sub_conds_out)):
                conds_final[i][full_idxs] += sub_conds_out[i] * weights_tensor
                counts_final[i][full_idxs] += weights_tensor
        
    if ADGS.params.context_options.fuse_method == ContextFuseMethod.RELATIVE:
        # already normalized, so return as is
        del counts_final
        return conds_final
    else:
        # normalize conds via division by context usage counts
        for i in range(len(conds_final)):
            conds_final[i] /= counts_final[i]
        del counts_final
        return conds_final


def calc_cond_uncond_batch_wrapper(model, conds: list[dict], x_in: Tensor, timestep, model_options):
    # check if conds or unconds contain lora_hook or default_cond
    contains_lora_hooks = False
    has_default_cond = False
    hook_groups = []
    for cond_uncond in conds:
        if cond_uncond is None:
            continue
        for t in cond_uncond:
            if COND_CONST.KEY_LORA_HOOK in t:
                contains_lora_hooks = True
                hook_groups.append(t[COND_CONST.KEY_LORA_HOOK])
            if COND_CONST.KEY_DEFAULT_COND in t:
                has_default_cond = True
        # if contains_lora_hooks:
        #     break
    if contains_lora_hooks or has_default_cond:
       ADGS.hooks_initialize(model, hook_groups=hook_groups)
       ADGS.prepare_hooks_current_keyframes(timestep, hook_groups=hook_groups)
       return calc_conds_batch_lora_hook(model, conds, x_in, timestep, model_options, has_default_cond)
    # keep for backwards compatibility, for now
    if not hasattr(comfy.samplers, "calc_cond_batch"):
        return comfy.samplers.calc_cond_uncond_batch(model, conds[0], conds[1], x_in, timestep, model_options)
    return comfy.samplers.calc_cond_batch(model, conds, x_in, timestep, model_options) 


# modified from comfy.samplers.get_area_and_mult
COND_OBJ = collections.namedtuple('cond_obj', ['input_x', 'mult', 'conditioning', 'area', 'control', 'patches'])
def get_area_and_mult_ADE(conds, x_in, timestep_in):
    area = (x_in.shape[2], x_in.shape[3], 0, 0)
    strength = 1.0

    if 'timestep_start' in conds:
        timestep_start = conds['timestep_start']
        if timestep_in[0] > timestep_start:
            return None
    if 'timestep_end' in conds:
        timestep_end = conds['timestep_end']
        if timestep_in[0] < timestep_end:
            return None
    if 'area' in conds:
        area = conds['area']
    if 'strength' in conds:
        strength = conds['strength']

    input_x = x_in[:,:,area[2]:area[0] + area[2],area[3]:area[1] + area[3]]
    if 'mask' in conds:
        # Scale the mask to the size of the input
        # The mask should have been resized as we began the sampling process
        mask_strength = 1.0
        if "mask_strength" in conds:
            mask_strength = conds["mask_strength"]
        mask = conds['mask']
        assert(mask.shape[1] == x_in.shape[2])
        assert(mask.shape[2] == x_in.shape[3])
        # make sure mask is capped at input_shape batch length to prevent 0 as dimension
        mask = mask[:input_x.shape[0], area[2]:area[0] + area[2], area[3]:area[1] + area[3]] * mask_strength
        mask = mask.unsqueeze(1).repeat(input_x.shape[0] // mask.shape[0], input_x.shape[1], 1, 1)
    else:
        mask = torch.ones_like(input_x)
    mult = mask * strength

    if 'mask' not in conds:
        rr = 8
        if area[2] != 0:
            for t in range(rr):
                mult[:,:,t:1+t,:] *= ((1.0/rr) * (t + 1))
        if (area[0] + area[2]) < x_in.shape[2]:
            for t in range(rr):
                mult[:,:,area[0] - 1 - t:area[0] - t,:] *= ((1.0/rr) * (t + 1))
        if area[3] != 0:
            for t in range(rr):
                mult[:,:,:,t:1+t] *= ((1.0/rr) * (t + 1))
        if (area[1] + area[3]) < x_in.shape[3]:
            for t in range(rr):
                mult[:,:,:,area[1] - 1 - t:area[1] - t] *= ((1.0/rr) * (t + 1))

    conditioning = {}
    model_conds = conds["model_conds"]
    for c in model_conds:
        conditioning[c] = model_conds[c].process_cond(batch_size=x_in.shape[0], device=x_in.device, area=area)

    control = conds.get('control', None)

    patches = None
    if 'gligen' in conds:
        gligen = conds['gligen']
        patches = {}
        gligen_type = gligen[0]
        gligen_model = gligen[1]
        if gligen_type == "position":
            gligen_patch = gligen_model.model.set_position(input_x.shape, gligen[2], input_x.device)
        elif gligen_type == "position_batched":
            try:
                gligen_model.model.set_position_batched_ADE = MethodType(gligen_batch_set_position_ADE, gligen_model.model)
                gligen_patch = gligen_model.model.set_position_batched_ADE(input_x.shape, gligen[2], input_x.device)
            finally:
                delattr(gligen_model.model, "set_position_batched_ADE")
        else:
            gligen_patch = gligen_model.model.set_empty(input_x.shape, input_x.device)

        patches['middle_patch'] = [gligen_patch]

    return COND_OBJ(input_x, mult, conditioning, area, control, patches)


def separate_default_conds(conds: list[dict]):
    normal_conds = []
    default_conds = []
    for i in range(len(conds)):
        c = []
        default_c = []
        # if cond is None, make normal/default_conds reflect that too
        if conds[i] is None:
            c = None
            default_c = []
        else:
            for t in conds[i]:
                # check if cond is a default cond
                if COND_CONST.KEY_DEFAULT_COND in t:
                    default_c.append(t)
                else:
                    c.append(t)
        normal_conds.append(c)
        default_conds.append(default_c)
    return normal_conds, default_conds


def finalize_default_conds(hooked_to_run: dict[LoraHookGroup,list[tuple[COND_OBJ,int]]], default_conds: list[list[dict]], x_in: Tensor, timestep):
    # need to figure out remaining unmasked area for conds
    default_mults = []
    for d in default_conds:
        default_mults.append(torch.ones_like(x_in))
    
    # look through each finalized cond in hooked_to_run for 'mult' and subtract it from each cond
    for lora_hooks, to_run in hooked_to_run.items():
        for cond_obj, i in to_run:
            # if no default_cond for cond_type, do nothing
            if len(default_conds[i]) == 0:
                continue
            area: list[int] = cond_obj.area
            default_mults[i][:,:,area[2]:area[0] + area[2],area[3]:area[1] + area[3]] -= cond_obj.mult
    
    # for each default_mult, ReLU to make negatives=0, and then check for any nonzeros
    for i, mult in enumerate(default_mults):
        # if no default_cond for cond type, do nothing
        if len(default_conds[i]) == 0:
            continue
        torch.nn.functional.relu(mult, inplace=True)
        # if mult is all zeros, then don't add default_cond
        if torch.max(mult) == 0.0:
            continue

        cond = default_conds[i]
        for x in cond:
            # do get_area_and_mult to get all the expected values
            p = comfy.samplers.get_area_and_mult(x, x_in, timestep)
            if p is None:
                continue
            # replace p's mult with calculated mult
            p = p._replace(mult=mult)
            hook: LoraHookGroup = x.get(COND_CONST.KEY_LORA_HOOK, None)
            hooked_to_run.setdefault(hook, list())
            hooked_to_run[hook] += [(p, i)]


# based on comfy.samplers.calc_conds_batch
def calc_conds_batch_lora_hook(model: BaseModel, conds: list[list[dict]], x_in: Tensor, timestep, model_options: dict, has_default_cond=False):
    out_conds = []
    out_counts = []
    # separate conds by matching lora_hooks
    hooked_to_run: dict[LoraHookGroup,list[tuple[collections.namedtuple,int]]] = {}

    # separate out default_conds, if needed
    if has_default_cond:
        conds, default_conds = separate_default_conds(conds)

    # cond is i=0, uncond is i=1
    for i in range(len(conds)):
        out_conds.append(torch.zeros_like(x_in))
        out_counts.append(torch.ones_like(x_in) * 1e-37)

        cond = conds[i]
        if cond is not None:
            for x in cond:
                p = comfy.samplers.get_area_and_mult(x, x_in, timestep)
                if p is None:
                    continue
                hook: LoraHookGroup = x.get(COND_CONST.KEY_LORA_HOOK, None)
                hooked_to_run.setdefault(hook, list())
                hooked_to_run[hook] += [(p, i)]
    
    # finalize default_conds, if needed
    if has_default_cond:
        finalize_default_conds(hooked_to_run, default_conds, x_in, timestep)

    # run every hooked_to_run separately
    for lora_hooks, to_run in hooked_to_run.items():
        while len(to_run) > 0:
            first = to_run[0]
            first_shape = first[0][0].shape
            to_batch_temp = []
            for x in range(len(to_run)):
                if comfy.samplers.can_concat_cond(to_run[x][0], first[0]):
                    to_batch_temp += [x]

            to_batch_temp.reverse()
            to_batch = to_batch_temp[:1]

            free_memory = comfy.model_management.get_free_memory(x_in.device)
            for i in range(1, len(to_batch_temp) + 1):
                batch_amount = to_batch_temp[:len(to_batch_temp)//i]
                input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
                if model.memory_required(input_shape) < free_memory:
                    to_batch = batch_amount
                    break
            ADGS.model_patcher.apply_lora_hooks(lora_hooks=lora_hooks)

            input_x = []
            mult = []
            c = []
            cond_or_uncond = []
            area = []
            control = None
            patches = None
            for x in to_batch:
                o = to_run.pop(x)
                p = o[0]
                input_x.append(p.input_x)
                mult.append(p.mult)
                c.append(p.conditioning)
                area.append(p.area)
                cond_or_uncond.append(o[1])
                control = p.control
                patches = p.patches

            batch_chunks = len(cond_or_uncond)
            input_x = torch.cat(input_x)
            c = comfy.samplers.cond_cat(c)
            timestep_ = torch.cat([timestep] * batch_chunks)

            if control is not None:
                c['control'] = control.get_control(input_x, timestep_, c, len(cond_or_uncond))

            transformer_options = {}
            if 'transformer_options' in model_options:
                transformer_options = model_options['transformer_options'].copy()

            if patches is not None:
                if "patches" in transformer_options:
                    cur_patches = transformer_options["patches"].copy()
                    for p in patches:
                        if p in cur_patches:
                            cur_patches[p] = cur_patches[p] + patches[p]
                        else:
                            cur_patches[p] = patches[p]
                    transformer_options["patches"] = cur_patches
                else:
                    transformer_options["patches"] = patches

            transformer_options["cond_or_uncond"] = cond_or_uncond[:]
            transformer_options["sigmas"] = timestep

            c['transformer_options'] = transformer_options

            if 'model_function_wrapper' in model_options:
                output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
            else:
                output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)

            for o in range(batch_chunks):
                cond_index = cond_or_uncond[o]
                out_conds[cond_index][:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_counts[cond_index][:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]

    for i in range(len(out_conds)):
        out_conds[i] /= out_counts[i]

    return out_conds

def gligen_batch_set_position_ADE(self, latent_image_shape: torch.Size, position_params_batch: list[list[tuple[Tensor, int, int, int, int]]], device):
    batch, c, h, w = latent_image_shape

    all_boxes = []
    all_masks = []
    all_conds = []

    # make sure there are enough position_params to match expected amount
    if len(position_params_batch) < ADGS.params.full_length:
        position_params_batch = position_params_batch.copy()
        for _ in range(ADGS.params.full_length-len(position_params_batch)):
            position_params_batch.append(position_params_batch[-1])

    for batch_idx in range(batch):
        if ADGS.params.sub_idxs is not None:
            position_params = position_params_batch[ADGS.params.sub_idxs[batch_idx]]
        else:
            position_params = position_params_batch[batch_idx]
        masks = torch.zeros([self.max_objs], device="cpu")
        boxes = []
        positive_embeddings = []

        for p in position_params:
            x1 = (p[4]) / w
            y1 = (p[3]) / h
            x2 = (p[4] + p[2]) / w
            y2 = (p[3] + p[1]) / h
            masks[len(boxes)] = 1.0
            boxes.append(torch.tensor((x1, y1, x2, y2)).unsqueeze(0))
            positive_embeddings.append(p[0])

        if len(boxes) < self.max_objs:
            append_boxes = torch.zeros([self.max_objs - len(boxes), 4], device="cpu")
            append_conds = torch.zeros([self.max_objs - len(boxes), self.key_dim], device="cpu")
            boxes = torch.cat(boxes + [append_boxes])
            conds = torch.cat(positive_embeddings + [append_conds])
        else:
            boxes = torch.cat(boxes)
            conds = torch.cat(positive_embeddings)
        all_boxes.append(boxes)
        all_masks.append(masks)
        all_conds.append(conds)

    box_out = torch.stack(all_boxes).to(device)
    masks_out = torch.stack(all_masks).to(device)
    conds_out = torch.stack(all_conds).to(device)
    
    return self._set_position(box_out, masks_out, conds_out)
