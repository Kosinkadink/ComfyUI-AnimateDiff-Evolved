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

from .context import get_context_scheduler
from .sample_settings import IterationOptions, SeedNoiseGeneration, prepare_mask_ad
from .motion_utils import GroupNormAD
from .model_utils import ModelTypeSD, wrap_function_to_inject_xformers_bug_info
from .model_injection import InjectionParams, ModelPatcherAndInjector, MotionModelPatcher
from .motion_module_ad import AnimateDiffFormat, AnimateDiffInfo, AnimateDiffVersion, VanillaTemporalModule
from .logger import logger


##################################################################################
######################################################################
# Global variable to use to more conveniently hack variable access into samplers
class AnimateDiffHelper_GlobalState:
    def __init__(self):
        self.motion_model: MotionModelPatcher = None
        self.params: InjectionParams = None
        self.reset()
    
    def reset(self):
        self.start_step: int = 0
        self.last_step: int = 0
        self.current_step: int = 0
        self.total_steps: int = 0
        if self.motion_model is not None:
            del self.motion_model
            self.motion_model = None
        if self.params is not None:
            del self.params
            self.params = None
    
    def update_with_inject_params(self, params: InjectionParams):
        self.params = params

    def is_using_sliding_context(self):
        return self.params is not None and self.params.context_length is not None
    
    def create_exposed_params(self):
        # This dict will be exposed to be used by other extensions
        # DO NOT change any of the key names
        # or I will find you 👁.👁
        return {
            "full_length": self.params.video_length,
            "context_length": self.params.context_length,
            "sub_idxs": self.params.sub_idxs,
        }

ADGS = AnimateDiffHelper_GlobalState()
######################################################################
##################################################################################


##################################################################################
#### Code Injection ##################################################

# refer to forward_timestep_embed in comfy/ldm/modules/diffusionmodules/openaimodel.py
def forward_timestep_embed_factory() -> Callable:
    if hasattr(attention, "SpatialVideoTransformer"):
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
    # keep old version for backwards compatibility (TODO: remove at end of 2023)
    else:
        def forward_timestep_embed(ts, x, emb, context=None, transformer_options={}, output_shape=None):
            for layer in ts:
                if isinstance(layer, openaimodel.TimestepBlock):
                    x = layer(x, emb)
                elif isinstance(layer, VanillaTemporalModule):
                    x = layer(x, context)
                elif isinstance(layer, attention.SpatialTransformer):
                    x = layer(x, context, transformer_options)
                    if "current_index" in transformer_options:
                        transformer_options["current_index"] += 1
                elif isinstance(layer, openaimodel.Upsample):
                    x = layer(x, output_shape=output_shape)
                else:
                    x = layer(x)
            return x
    return forward_timestep_embed


def unlimited_memory_required(*args, **kwargs):
    return 0


def groupnorm_mm_factory(params: InjectionParams):
    def groupnorm_mm_forward(self, input: Tensor) -> Tensor:
        # axes_factor normalizes batch based on total conds and unconds passed in batch;
        # the conds and unconds per batch can change based on VRAM optimizations that may kick in
        if not ADGS.is_using_sliding_context():
            axes_factor = input.size(0)//params.video_length
        else:
            axes_factor = input.size(0)//params.context_length

        input = rearrange(input, "(b f) c h w -> b c f h w", b=axes_factor)
        input = group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
        input = rearrange(input, "b c f h w -> (b f) c h w", b=axes_factor)
        return input
    return groupnorm_mm_forward


def get_additional_models_factory(orig_get_additional_models: Callable, motion_model: MotionModelPatcher):
    def get_additional_models_with_motion(*args, **kwargs):
        models, inference_memory = orig_get_additional_models(*args, **kwargs)
        models.append(motion_model)
        # TODO: account for inference memory as well?
        return models, inference_memory
    return get_additional_models_with_motion
######################################################################
##################################################################################


def apply_params_to_motion_model(motion_model: MotionModelPatcher, params: InjectionParams):
    if params.context_length and params.video_length > params.context_length:
        logger.info(f"Sliding context window activated - latents passed in ({params.video_length}) greater than context_length {params.context_length}.")
    else:
        logger.info(f"Regular AnimateDiff activated - latents passed in ({params.video_length}) less or equal to context_length {params.context_length}.")
        params.reset_context()
    # if no context_length, treat video length as intended AD frame window
    if not params.context_length:
        if params.video_length > motion_model.model.encoding_max_len:
            raise ValueError(f"Without a context window, AnimateDiff model {motion_model.model.mm_info.mm_name} has upper limit of {motion_model.model.encoding_max_len} frames, but received {params.video_length} latents.")
        motion_model.model.set_video_length(params.video_length, params.full_length)
    # otherwise, treat context_length as intended AD frame window
    else:
        if params.context_length > motion_model.model.encoding_max_len:
            raise ValueError(f"AnimateDiff model {motion_model.model.mm_info.mm_name} has upper limit of {motion_model.model.encoding_max_len} frames for a context window, but received context length of {params.context_length}.")
        motion_model.model.set_video_length(params.context_length, params.full_length)
    # inject model
    logger.info(f"Using motion module {motion_model.model.mm_info.mm_name} version {motion_model.model.mm_info.mm_version}.")


class FunctionInjectionHolder:
    def __init__(self):
        pass
    
    def inject_functions(self, model: ModelPatcherAndInjector, params: InjectionParams):
        # Save Original Functions
        self.orig_forward_timestep_embed = openaimodel.forward_timestep_embed # needed to account for VanillaTemporalModule
        self.orig_memory_required = model.model.memory_required # allows for "unlimited area hack" to prevent halving of conds/unconds
        self.orig_groupnorm_forward = torch.nn.GroupNorm.forward # used to normalize latents to remove "flickering" of colors/brightness between frames
        self.orig_groupnormad_forward = GroupNormAD.forward
        self.orig_sampling_function = comfy.samplers.sampling_function # used to support sliding context windows in samplers
        self.orig_prepare_mask = comfy.sample.prepare_mask
        self.orig_get_additional_models = comfy.sample.get_additional_models
        # Inject Functions
        openaimodel.forward_timestep_embed = forward_timestep_embed_factory()
        if params.unlimited_area_hack:
            model.model.memory_required = unlimited_memory_required
        # only apply groupnorm hack if not [v3 or (AnimateDiff SD1.5 and v2 and should apply v2 properly)]
        info: AnimateDiffInfo = model.motion_model.model.mm_info
        if not (info.mm_version == AnimateDiffVersion.V3 or (info.mm_format == AnimateDiffFormat.ANIMATEDIFF and info.sd_type == ModelTypeSD.SD1_5 and
                info.mm_version == AnimateDiffVersion.V2 and params.apply_v2_models_properly)):
            torch.nn.GroupNorm.forward = groupnorm_mm_factory(params)
            if params.apply_mm_groupnorm_hack:
                GroupNormAD.forward = groupnorm_mm_factory(params)
        comfy.samplers.sampling_function = sliding_sampling_function
        comfy.sample.prepare_mask = prepare_mask_ad
        comfy.sample.get_additional_models = get_additional_models_factory(self.orig_get_additional_models, model.motion_model)
        del info

    def restore_functions(self, model: ModelPatcherAndInjector):
        # Restoration
        try:
            model.model.memory_required = self.orig_memory_required
            openaimodel.forward_timestep_embed = self.orig_forward_timestep_embed
            torch.nn.GroupNorm.forward = self.orig_groupnorm_forward
            GroupNormAD.forward = self.orig_groupnormad_forward
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
            params.video_length = latents.size(0)
            params.full_length = latents.size(0)
            # reset global state
            ADGS.reset()
            # store and inject functions
            function_injections.inject_functions(model, params)

            # apply custom noise, if needed
            disable_noise = kwargs.get("disable_noise") or False
            seed = kwargs["seed"]

            # apply params to motion model
            apply_params_to_motion_model(model.motion_model, params)

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
            ADGS.motion_model = model.motion_model

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
                    iter_kwargs[IterationOptions.SAMPLER] = args[-5]
                else:
                    iter_kwargs[IterationOptions.SAMPLER] = comfy.samplers.KSampler(
                        model.model, steps=args[-7],
                        device=model.current_device, sampler=args[-5],
                        scheduler=args[-4], denoise=kwargs["denoise"],
                        model_options=model.model_options)

            args = list(args)
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
                                                              sample_settings=model.sample_settings, noise_extra_args=noise_extra_args)
                args[-1] = latents

                model.motion_model.pre_run()
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



def sliding_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
        def get_area_and_mult(conds, x_in, timestep_in):
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
                mask = mask[:,area[2]:area[0] + area[2],area[3]:area[1] + area[3]] * mask_strength
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

            conditionning = {}
            model_conds = conds["model_conds"]
            for c in model_conds:
                conditionning[c] = model_conds[c].process_cond(batch_size=x_in.shape[0], device=x_in.device, area=area)

            control = None
            if 'control' in conds:
                control = conds['control']

            patches = None
            if 'gligen' in conds:
                gligen = conds['gligen']
                patches = {}
                gligen_type = gligen[0]
                gligen_model = gligen[1]
                if gligen_type == "position":
                    gligen_patch = gligen_model.model.set_position(input_x.shape, gligen[2], input_x.device)
                else:
                    gligen_patch = gligen_model.model.set_empty(input_x.shape, input_x.device)

                patches['middle_patch'] = [gligen_patch]

            return (input_x, mult, conditionning, area, control, patches)

        def cond_equal_size(c1, c2):
            if c1 is c2:
                return True
            if c1.keys() != c2.keys():
                return False
            for k in c1:
                if not c1[k].can_concat(c2[k]):
                    return False
            return True

        def can_concat_cond(c1, c2):
            if c1[0].shape != c2[0].shape:
                return False

            #control
            if (c1[4] is None) != (c2[4] is None):
                return False
            if c1[4] is not None:
                if c1[4] is not c2[4]:
                    return False

            #patches
            if (c1[5] is None) != (c2[5] is None):
                return False
            if (c1[5] is not None):
                if c1[5] is not c2[5]:
                    return False

            return cond_equal_size(c1[2], c2[2])

        def cond_cat(c_list):
            c_crossattn = []
            c_concat = []
            c_adm = []
            crossattn_max_len = 0

            temp = {}
            for x in c_list:
                for k in x:
                    cur = temp.get(k, [])
                    cur.append(x[k])
                    temp[k] = cur

            out = {}
            for k in temp:
                conds = temp[k]
                out[k] = conds[0].concat(conds[1:])

            return out

        def calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options):
            out_cond = torch.zeros_like(x_in)
            out_count = torch.ones_like(x_in) * 1e-37

            out_uncond = torch.zeros_like(x_in)
            out_uncond_count = torch.ones_like(x_in) * 1e-37

            COND = 0
            UNCOND = 1

            to_run = []
            for x in cond:
                p = get_area_and_mult(x, x_in, timestep)
                if p is None:
                    continue

                to_run += [(p, COND)]
            if uncond is not None:
                for x in uncond:
                    p = get_area_and_mult(x, x_in, timestep)
                    if p is None:
                        continue

                    to_run += [(p, UNCOND)]

            while len(to_run) > 0:
                first = to_run[0]
                first_shape = first[0][0].shape
                to_batch_temp = []
                for x in range(len(to_run)):
                    if can_concat_cond(to_run[x][0], first[0]):
                        to_batch_temp += [x]

                to_batch_temp.reverse()
                to_batch = to_batch_temp[:1]

                free_memory = model_management.get_free_memory(x_in.device)
                for i in range(1, len(to_batch_temp) + 1):
                    batch_amount = to_batch_temp[:len(to_batch_temp)//i]
                    input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
                    if model.memory_required(input_shape) < free_memory:
                        to_batch = batch_amount
                        break

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
                    input_x += [p[0]]
                    mult += [p[1]]
                    c += [p[2]]
                    area += [p[3]]
                    cond_or_uncond += [o[1]]
                    control = p[4]
                    patches = p[5]

                batch_chunks = len(cond_or_uncond)
                input_x = torch.cat(input_x)
                c = cond_cat(c)
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
                    else:
                        transformer_options["patches"] = patches

                transformer_options["cond_or_uncond"] = cond_or_uncond
                transformer_options["sigmas"] = timestep
                transformer_options["ad_params"] = ADGS.create_exposed_params()

                c['transformer_options'] = transformer_options

                if 'model_function_wrapper' in model_options:
                    output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
                else:
                    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
                del input_x

                for o in range(batch_chunks):
                    if cond_or_uncond[o] == COND:
                        out_cond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                        out_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
                    else:
                        out_uncond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                        out_uncond_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
                del mult

            out_cond /= out_count
            del out_count
            out_uncond /= out_uncond_count
            del out_uncond_count
            return out_cond, out_uncond

        # sliding_calc_cond_uncond_batch inspired by ashen's initial hack for 16-frame sliding context:
        # https://github.com/comfyanonymous/ComfyUI/compare/master...ashen-sensored:ComfyUI:master
        def sliding_calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options):
            # get context scheduler
            context_scheduler = get_context_scheduler(ADGS.params.context_schedule)
            # figure out how input is split
            axes_factor = x_in.size(0)//ADGS.params.video_length

            # prepare final cond, uncond, and out_count
            cond_final = torch.zeros_like(x_in)
            uncond_final = torch.zeros_like(x_in)
            out_count_final = torch.zeros((x_in.shape[0], 1, 1, 1), device=x_in.device)

            def prepare_control_objects(control: ControlBase, full_idxs: list[int]):
                if control.previous_controlnet is not None:
                    prepare_control_objects(control.previous_controlnet, full_idxs)
                control.sub_idxs = full_idxs
                control.full_latent_length = ADGS.params.video_length
                control.context_length = ADGS.params.context_length
            
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

            # perform calc_cond_uncond_batch per context window
            for ctx_idxs in context_scheduler(ADGS.current_step, ADGS.total_steps, ADGS.params.video_length, ADGS.params.context_length, ADGS.params.context_stride, ADGS.params.context_overlap, ADGS.params.closed_loop):
                ADGS.params.sub_idxs = ctx_idxs
                ADGS.motion_model.model.set_sub_idxs(ctx_idxs)
                # account for all portions of input frames
                full_idxs = []
                for n in range(axes_factor):
                    for ind in ctx_idxs:
                        full_idxs.append((ADGS.params.video_length*n)+ind)
                # get subsections of x, timestep, cond, uncond, cond_concat
                sub_x = x_in[full_idxs]
                sub_timestep = timestep[full_idxs]
                sub_cond = get_resized_cond(cond, full_idxs) if cond is not None else None
                sub_uncond = get_resized_cond(uncond, full_idxs) if uncond is not None else None

                sub_cond_out, sub_uncond_out = calc_cond_uncond_batch(model, sub_cond, sub_uncond, sub_x, sub_timestep, model_options)

                cond_final[full_idxs] += sub_cond_out
                uncond_final[full_idxs] += sub_uncond_out
                out_count_final[full_idxs] += 1 # increment which indeces were used

            # normalize cond and uncond via division by context usage counts
            cond_final /= out_count_final
            uncond_final /= out_count_final
            del out_count_final
            return cond_final, uncond_final


        if math.isclose(cond_scale, 1.0):
            uncond = None

        if not ADGS.is_using_sliding_context():
            cond, uncond = calc_cond_uncond_batch(model, cond, uncond, x, timestep, model_options)
        else:
            cond, uncond = sliding_calc_cond_uncond_batch(model, cond, uncond, x, timestep, model_options)
        if "sampler_cfg_function" in model_options:
            args = {"cond": x - cond, "uncond": x - uncond, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep}
            return x - model_options["sampler_cfg_function"](args)
        else:
            return uncond + (cond - uncond) * cond_scale
