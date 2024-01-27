from typing import Callable

import math
import torch
from torch import Tensor
from torch.nn.functional import group_norm
from einops import rearrange

import comfy.ldm.modules.attention as attention
from comfy.ldm.modules.diffusionmodules import openaimodel
import comfy.model_management as model_management
import comfy.samplers
import comfy.sample
import comfy.utils
from comfy.controlnet import ControlBase
import comfy.ops

from .context import ContextFuseMethod, ContextSchedules, get_context_weights, get_context_windows
from .sample_settings import IterationOptions, SeedNoiseGeneration, prepare_mask_ad
from .utils_model import ModelTypeSD, wrap_function_to_inject_xformers_bug_info
from .model_injection import InjectionParams, ModelPatcherAndInjector, MotionModelGroup, MotionModelPatcher
from .motion_module_ad import AnimateDiffFormat, AnimateDiffInfo, AnimateDiffVersion, VanillaTemporalModule
from .logger import logger


##################################################################################
######################################################################
# Global variable to use to more conveniently hack variable access into samplers
class AnimateDiffHelper_GlobalState:
    def __init__(self):
        self.motion_models: MotionModelGroup = None
        self.params: InjectionParams = None
        self.reset()
    
    def initialize(self, model):
        # this function is to be run in sampling func
        if not self.initialized:
            self.initialized = True
            if self.motion_models is not None:
                self.motion_models.initialize_timesteps(model)
            if self.params.context_options is not None:
                self.params.context_options.initialize_timesteps(model)

    def reset(self):
        self.initialized = False
        self.start_step: int = 0
        self.last_step: int = 0
        self.current_step: int = 0
        self.total_steps: int = 0
        if self.motion_models is not None:
            del self.motion_models
            self.motion_models = None
        if self.params is not None:
            del self.params
            self.params = None
    
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
######################################################################
##################################################################################


def apply_params_to_motion_models(motion_models: MotionModelGroup, params: InjectionParams):
    params = params.clone()
    if params.context_options.context_schedule == ContextSchedules.VIEW_AS_CONTEXT:
        params.context_options._current_context.context_length = params.full_length
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
                if params.full_length > motion_model.model.encoding_max_len:
                    raise ValueError(f"Without a context window, AnimateDiff model {motion_model.model.mm_info.mm_name} has upper limit of {motion_model.model.encoding_max_len} frames, but received {params.full_length} latents.")
            motion_models.set_video_length(params.full_length, params.full_length)
        # otherwise, treat context_length as intended AD frame window
        else:
            for motion_model in motion_models.models:
                view_options = params.context_options.view_options
                context_length = view_options.context_length if view_options else params.context_options.context_length
                if context_length > motion_model.model.encoding_max_len:
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
        # Save Original Functions
        self.orig_forward_timestep_embed = openaimodel.forward_timestep_embed # needed to account for VanillaTemporalModule
        self.orig_memory_required = model.model.memory_required # allows for "unlimited area hack" to prevent halving of conds/unconds
        self.orig_groupnorm_forward = torch.nn.GroupNorm.forward # used to normalize latents to remove "flickering" of colors/brightness between frames
        self.orig_groupnorm_manual_cast_forward = comfy.ops.manual_cast.GroupNorm.forward_comfy_cast_weights
        self.orig_sampling_function = comfy.samplers.sampling_function # used to support sliding context windows in samplers
        self.orig_prepare_mask = comfy.sample.prepare_mask
        self.orig_get_additional_models = comfy.sample.get_additional_models
        # Inject Functions
        openaimodel.forward_timestep_embed = forward_timestep_embed_factory()
        if params.unlimited_area_hack:
            model.model.memory_required = unlimited_memory_required
        if model.motion_models is not None:
            # only apply groupnorm hack if not [v3 or (AnimateDiff SD1.5 and v2 and should apply v2 properly)]
            info: AnimateDiffInfo = model.motion_models[0].model.mm_info
            if not (info.mm_version == AnimateDiffVersion.V3 or (info.mm_format == AnimateDiffFormat.ANIMATEDIFF and info.sd_type == ModelTypeSD.SD1_5 and
                    info.mm_version == AnimateDiffVersion.V2 and params.apply_v2_properly)):
                torch.nn.GroupNorm.forward = groupnorm_mm_factory(params)
                comfy.ops.manual_cast.GroupNorm.forward_comfy_cast_weights = groupnorm_mm_factory(params, manual_cast=True)
                # if mps device (Apple Silicon), disable batched conds to avoid black images with groupnorm hack
                try:
                    if model.load_device.type == "mps":
                        model.model.memory_required = unlimited_memory_required
                except Exception:
                    pass
            del info
        comfy.samplers.sampling_function = evolved_sampling_function
        comfy.sample.prepare_mask = prepare_mask_ad
        comfy.sample.get_additional_models = get_additional_models_factory(self.orig_get_additional_models, model.motion_models)

    def restore_functions(self, model: ModelPatcherAndInjector):
        # Restoration
        try:
            model.model.memory_required = self.orig_memory_required
            openaimodel.forward_timestep_embed = self.orig_forward_timestep_embed
            torch.nn.GroupNorm.forward = self.orig_groupnorm_forward
            comfy.ops.manual_cast.GroupNorm.forward_comfy_cast_weights = self.orig_groupnorm_manual_cast_forward
            comfy.samplers.sampling_function = self.orig_sampling_function
            comfy.sample.prepare_mask = self.orig_prepare_mask
            comfy.sample.get_additional_models = self.orig_get_additional_models
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
            ADGS.motion_models = model.motion_models

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
                model_management.load_model_gpu(model)
                if is_custom:
                    iter_kwargs[IterationOptions.SAMPLER] = None #args[-5]
                else:
                    iter_kwargs[IterationOptions.SAMPLER] = comfy.samplers.KSampler(
                        model.model, steps=999, #steps=args[-7],
                        device=model.current_device, sampler=args[-5],
                        scheduler=args[-4], denoise=kwargs.get("denoise", None),
                        model_options=model.model_options)

            for curr_i in range(iter_opts.iterations):
                # handle GLOBALSTATE vars and step tally
                ADGS.update_with_inject_params(params)
                ADGS.start_step = kwargs.get("start_step") or 0
                ADGS.current_step = ADGS.start_step
                ADGS.last_step = kwargs.get("last_step") or 0
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
                latents = wrap_function_to_inject_xformers_bug_info(orig_comfy_sample)(model, noise, *args, **kwargs)
            return latents
        finally:
            del latents
            del noise
            del cached_latents
            del cached_noise
            # reset global state
            ADGS.reset()
            # restore injected functions
            function_injections.restore_functions(model)
            del function_injections
    return motion_sample


def evolved_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options: dict={}, seed=None):
    ADGS.initialize(model)
    if ADGS.motion_models is not None:
        ADGS.motion_models.prepare_current_keyframe(t=timestep)
    if ADGS.params.context_options is not None:
        ADGS.params.context_options.prepare_current_context(t=timestep)

    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    # add AD/evolved-sampling params to model_options (transformer_options)
    model_options = model_options.copy()
    if "tranformer_options" not in model_options:
        model_options["tranformer_options"] = {}
    model_options["transformer_options"]["ad_params"] = ADGS.create_exposed_params()

    if not ADGS.is_using_sliding_context():
        cond_pred, uncond_pred = comfy.samplers.calc_cond_uncond_batch(model, cond, uncond_, x, timestep, model_options)
    else:
        cond_pred, uncond_pred = sliding_calc_cond_uncond_batch(model, cond, uncond_, x, timestep, model_options)

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


# sliding_calc_cond_uncond_batch inspired by ashen's initial hack for 16-frame sliding context:
# https://github.com/comfyanonymous/ComfyUI/compare/master...ashen-sensored:ComfyUI:master
def sliding_calc_cond_uncond_batch(model, cond, uncond, x_in: Tensor, timestep, model_options):
    def prepare_control_objects(control: ControlBase, full_idxs: list[int]):
        if control.previous_controlnet is not None:
            prepare_control_objects(control.previous_controlnet, full_idxs)
        control.sub_idxs = full_idxs
        control.full_latent_length = ADGS.params.full_length
        control.context_length = ADGS.params.context_options.context_length
    
    def get_resized_cond(cond_in, full_idxs) -> list:
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
                        if hasattr(control_item, "sub_idxs"):
                            prepare_control_objects(control_item, full_idxs)
                        else:
                            raise ValueError(f"Control type {type(control_item).__name__} may not support required features for sliding context window; \
                                                use Control objects from Kosinkadink/ComfyUI-Advanced-ControlNet nodes, or make sure Advanced-ControlNet is updated.")
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

    # prepare final cond, uncond, and out_count
    cond_final = torch.zeros_like(x_in)
    uncond_final = torch.zeros_like(x_in)
    out_count_final = torch.zeros((x_in.shape[0], 1, 1, 1), device=x_in.device)

    # perform calc_cond_uncond_batch per context window
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
        # get subsections of x, timestep, cond, uncond, cond_concat
        sub_x = x_in[full_idxs]
        sub_timestep = timestep[full_idxs]
        sub_cond = get_resized_cond(cond, full_idxs) if cond is not None else None
        sub_uncond = get_resized_cond(uncond, full_idxs) if uncond is not None else None

        sub_cond_out, sub_uncond_out = comfy.samplers.calc_cond_uncond_batch(model, sub_cond, sub_uncond, sub_x, sub_timestep, model_options)

        # add conds and counts based on weights of fuse method
        weights = get_context_weights(len(ctx_idxs), ADGS.params.context_options.fuse_method) * batched_conds
        weights_tensor = torch.Tensor(weights).to(device=x_in.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        cond_final[full_idxs] += sub_cond_out * weights_tensor
        uncond_final[full_idxs] += sub_uncond_out * weights_tensor
        out_count_final[full_idxs] += weights_tensor


    # normalize cond and uncond via division by context usage counts
    cond_final /= out_count_final
    uncond_final /= out_count_final
    del out_count_final
    return cond_final, uncond_final
