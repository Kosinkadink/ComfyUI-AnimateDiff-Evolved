from __future__ import annotations
import copy
from typing import Union, Callable
from collections import namedtuple

from einops import rearrange
from torch import Tensor
import torch.nn.functional as F
import torch
import uuid
import math

import comfy.conds
import comfy.lora
import comfy.model_management
import comfy.utils
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import CallbacksMP, WrappersMP, PatcherInjection
from comfy.model_base import BaseModel
from comfy.sd import CLIP, VAE

from .ad_settings import AnimateDiffSettings, AdjustPE, AdjustWeight
from .adapter_cameractrl import CameraPoseEncoder, CameraEntry, prepare_pose_embedding
from .context import ContextOptions, ContextOptions, ContextOptionsGroup
from .motion_module_ad import (AnimateDiffModel, AnimateDiffFormat, AnimateDiffInfo, EncoderOnlyAnimateDiffModel, VersatileAttention,
                               VanillaTemporalModule, has_mid_block, normalize_ad_state_dict, get_position_encoding_max_len)
from .logger import logger
from .utils_motion import (ADKeyframe, ADKeyframeGroup, MotionCompatibilityError, InputPIA,
                           PerBlock, AllPerBlocks, get_combined_per_block_list,
                           get_combined_multival, get_combined_input, get_combined_input_effect_multival,
                           ade_broadcast_image_to, extend_to_batch_size, prepare_mask_batch)
from .motion_lora import MotionLoraInfo, MotionLoraList
from .utils_model import get_motion_lora_path, get_motion_model_path, get_sd_model_type, vae_encode_raw_batched, BIGMAX_TENSOR
from .sample_settings import SampleSettings, SeedNoiseGeneration
from .dinklink import DinkLinkConst, get_dinklink, get_acn_outer_sample_wrapper


def prepare_dinklink_register_definitions():
    # expose create_MotionModelPatcher
    d = get_dinklink()
    link_ade = d.setdefault(DinkLinkConst.ADE, {})
    link_ade[DinkLinkConst.ADE_CREATE_MOTIONMODELPATCHER] = create_MotionModelPatcher


class MotionModelPatcher(ModelPatcher):
    '''Class used only for type hints.'''
    def __init__(self):
        self.model: AnimateDiffModel


class ModelPatcherHelper:
    SAMPLE_SETTINGS = "ADE_sample_settings"
    PARAMS = "ADE_params"
    ADE = "ADE"

    def __init__(self, model: ModelPatcher):
        self.model = model

    def set_all_properties(self, outer_sampler_wrapper: Callable, calc_cond_batch_wrapper: Callable,
                           params: InjectionParams, sample_settings: SampleSettings=None, motion_models: MotionModelGroup=None):
        self.set_outer_sample_wrapper(outer_sampler_wrapper)
        self.set_calc_cond_batch_wrapper(calc_cond_batch_wrapper)
        self.set_sample_settings(sample_settings = sample_settings if sample_settings is not None else SampleSettings())
        self.set_params(params)
        if motion_models is not None:
            self.set_motion_models(motion_models.models.copy())
            self.set_forward_timestep_embed_patch()
        else:
            self.remove_motion_models()
            self.remove_forward_timestep_embed_patch()

    def get_motion_models(self, all_devices=False) -> list[MotionModelPatcher]:
        if all_devices:
            patchers = [self.model]
            patchers.extend(self.model.get_additional_models_with_key("multigpu"))
            all_motion_models = []
            for patcher in patchers:
                all_motion_models.extend(patcher.get_additional_models_with_key(self.ADE))
            return all_motion_models
        return self.model.additional_models.get(self.ADE, [])

    def set_motion_models(self, motion_models: list[MotionModelPatcher]):
        self.model.set_additional_models(self.ADE, motion_models)
        self.model.set_injections(self.ADE,
                                  [PatcherInjection(inject=inject_motion_models, eject=eject_motion_models)])

    def remove_motion_models(self):
        self.model.remove_additional_models(self.ADE)
        self.model.remove_injections(self.ADE)

    def cleanup_motion_models(self):
        for motion_model in self.get_motion_models(all_devices=True):
            motion_model.cleanup()


    def set_forward_timestep_embed_patch(self):
        self.remove_forward_timestep_embed_patch()
        self.model.set_model_forward_timestep_embed_patch(create_forward_timestep_embed_patch())

    def remove_forward_timestep_embed_patch(self):
        if "transformer_options" in self.model.model_options:
            transformer_options = self.model.model_options["transformer_options"]
            if "patches" in transformer_options:
                patches = transformer_options["patches"]
                if "forward_timestep_embed_patch" in patches:
                    forward_timestep_patches: list = patches["forward_timestep_embed_patch"]
                    to_remove = []
                    for idx, patch in enumerate(forward_timestep_patches):
                        if patch[1] == forward_timestep_embed_patch_ade:
                            to_remove.append(idx)
                    for idx in to_remove:
                        forward_timestep_patches.pop(idx)


    ##########################
    # motion models helpers
    def set_video_length(self, video_length: int, full_length: int):
        for motion_model in self.get_motion_models(all_devices=True):
            motion_model.model.set_video_length(video_length=video_length, full_length=full_length)
    
    def get_name_string(self, show_version=False):
        identifiers = []
        for motion_model in self.get_motion_models():
            id = motion_model.model.mm_info.mm_name
            if show_version:
                id += f":{motion_model.model.mm_info.mm_version}"
            identifiers.append(id)
        return ", ".join(identifiers)
    ##########################


    def get_sample_settings(self) -> SampleSettings:
        sample_settings = self.model.get_attachment(self.SAMPLE_SETTINGS)
        return sample_settings if sample_settings is not None else SampleSettings()
    
    def set_sample_settings(self, sample_settings: SampleSettings):
        self.model.set_attachments(self.SAMPLE_SETTINGS, sample_settings)
    

    def get_params(self) -> InjectionParams:
        params = self.model.get_attachment(self.PARAMS)
        return params if params is not None else InjectionParams()
    
    def set_params(self, params: InjectionParams):
        self.model.set_attachments(self.PARAMS, params)
        if params.context_options.context_length is not None:
            self.set_ACN_outer_sample_wrapper(throw_exception=False)
        elif params.context_options.extras.context_ref is not None:
            self.set_ACN_outer_sample_wrapper(throw_exception=True)

    def set_ACN_outer_sample_wrapper(self, throw_exception=True):
        # get wrapper to register from Advanced-ControlNet via DinkLink shared dict
        wrapper_info = get_acn_outer_sample_wrapper(throw_exception)
        if wrapper_info is None:
            return
        wrapper_type, key, wrapper = wrapper_info
        if len(self.model.get_wrappers(wrapper_type, key)) == 0:
            self.model.add_wrapper_with_key(wrapper_type, key, wrapper)

    def set_outer_sample_wrapper(self, wrapper: Callable):
        self.model.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE, self.ADE)
        self.model.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, self.ADE, wrapper)
    
    def set_calc_cond_batch_wrapper(self, wrapper: Callable):
        self.model.remove_wrappers_with_key(WrappersMP.CALC_COND_BATCH, self.ADE)
        self.model.add_wrapper_with_key(WrappersMP.CALC_COND_BATCH, self.ADE, wrapper)

    def remove_wrappers(self):
        self.model.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE, self.ADE)
        self.model.remove_wrappers_with_key(WrappersMP.CALC_COND_BATCH, self.ADE)

    def pre_run(self):
        # TODO: could implement this as a ModelPatcher ON_PRE_RUN callback
        for motion_model in self.get_motion_models(all_devices=True):
            motion_model.pre_run()
        self.get_sample_settings().pre_run(self.model)


def inject_motion_models(patcher: ModelPatcher):
    helper = ModelPatcherHelper(patcher)
    motion_models = helper.get_motion_models()
    for mm in motion_models:
        mm.model.inject(patcher)


def eject_motion_models(patcher: ModelPatcher):
    helper = ModelPatcherHelper(patcher)
    motion_models = helper.get_motion_models()
    for mm in motion_models:
        mm.model.eject(patcher)


def create_forward_timestep_embed_patch():
    return (VanillaTemporalModule, forward_timestep_embed_patch_ade)

def forward_timestep_embed_patch_ade(layer, x, emb, context, transformer_options, output_shape, time_context, num_video_frames, image_only_indicator, *args, **kwargs):
    return layer(x, context, transformer_options=transformer_options)


def create_MotionModelPatcher(model, load_device, offload_device) -> MotionModelPatcher:
    patcher = ModelPatcher(model, load_device=load_device, offload_device=offload_device)
    ade = ModelPatcherHelper.ADE
    patcher.add_callback_with_key(CallbacksMP.ON_LOAD, ade, _mm_patch_lowvram_extras_callback)
    patcher.add_callback_with_key(CallbacksMP.ON_LOAD, ade, _mm_handle_float8_pe_tensors_callback)
    patcher.add_callback_with_key(CallbacksMP.ON_PRE_RUN, ade, _mm_pre_run_callback)
    patcher.add_callback_with_key(CallbacksMP.ON_CLEANUP, ade, _mm_clean_callback)
    patcher.set_attachments(ade, MotionModelAttachment())
    return patcher


def _mm_patch_lowvram_extras_callback(self: MotionModelPatcher, device_to, lowvram_model_memory, *args, **kwargs):
    if lowvram_model_memory > 0:
        # figure out the tensors (likely pe's) that should be cast to device besides just the named_modules
        remaining_tensors = list(self.model.state_dict().keys())
        named_modules = []
        for n, _ in self.model.named_modules():
            named_modules.append(n)
            named_modules.append(f"{n}.weight")
            named_modules.append(f"{n}.bias")
        for name in named_modules:
            if name in remaining_tensors:
                remaining_tensors.remove(name)

        for key in remaining_tensors:
            self.patch_weight_to_device(key, device_to)
            if device_to is not None:
                comfy.utils.set_attr(self.model, key, comfy.utils.get_attr(self.model, key).to(device_to))

def _mm_handle_float8_pe_tensors_callback(self: MotionModelPatcher, *args, **kwargs):
    remaining_tensors = list(self.model.state_dict().keys())
    pe_tensors = [x for x in remaining_tensors if '.pe' in x]
    is_first = True
    for key in pe_tensors:
        if is_first:
            is_first = False
            if comfy.utils.get_attr(self.model, key).dtype not in [torch.float8_e5m2, torch.float8_e4m3fn]:
                break
        comfy.utils.set_attr(self.model, key, comfy.utils.get_attr(self.model, key).half())

def _mm_pre_run_callback(self: MotionModelPatcher, *args, **kwargs):
    attachment = get_mm_attachment(self)
    attachment.pre_run(self)

def _mm_clean_callback(self: MotionModelPatcher, *args, **kwargs):
    attachment = get_mm_attachment(self)
    attachment.cleanup(self)


def get_mm_attachment(patcher: MotionModelPatcher) -> MotionModelAttachment:
    return patcher.get_attachment(ModelPatcherHelper.ADE)


class MotionModelAttachment:
    def __init__(self):
        self.timestep_percent_range = (0.0, 1.0)
        self.timestep_range: tuple[float, float] = None
        self.keyframes: ADKeyframeGroup = ADKeyframeGroup()

        self.scale_multival: Union[float, Tensor, None] = None
        self.effect_multival: Union[float, Tensor, None] = None
        self.per_block_list: Union[list[PerBlock], None] = None

        # AnimateLCM-I2V
        self.orig_ref_drift: float = None
        self.orig_insertion_weights: list[float] = None
        self.orig_apply_ref_when_disabled = False
        self.orig_img_latents: Tensor = None
        self.img_features: list[int, Tensor] = None  # temporary
        self.img_latents_shape: tuple = None

        # CameraCtrl
        self.orig_camera_entries: list[CameraEntry] = None
        self.camera_features: list[Tensor] = None  # temporary
        self.camera_features_shape: tuple = None
        self.cameractrl_multival: Union[float, Tensor] = None
        ## temp
        self.current_cameractrl_effect: Union[float, Tensor] = None
        self.combined_cameractrl_effect: Union[float, Tensor] = None

        # PIA
        self.orig_pia_images: Tensor = None
        self.pia_vae: VAE = None
        self.pia_input: InputPIA = None
        self.cached_pia_c_concat: comfy.conds.CONDNoiseShape = None  # cached
        self.prev_pia_latents_shape: tuple = None
        self.prev_current_pia_input: InputPIA = None
        self.pia_multival: Union[float, Tensor] = None
        ## temp
        self.current_pia_input: InputPIA = None
        self.combined_pia_mask: Union[float, Tensor] = None
        self.combined_pia_effect: Union[float, Tensor] = None

        # FancyVideo
        self.orig_fancy_images: Tensor = None
        self.fancy_vae: VAE = None
        self.cached_fancy_c_concat: comfy.conds.CONDNoiseShape = None  # cached
        self.prev_fancy_latents_shape: tuple = None
        self.fancy_multival: Union[float, Tensor] = None

        # MotionCtrl
        self.orig_RT: Tensor = None
        self.RT: Tensor = None
        self.prev_RT_shape: tuple = None
        self.prev_RT_uuids: list = None

        # temporary variables
        self.current_used_steps = 0
        self.current_keyframe: ADKeyframe = None
        self.current_index = -1
        self.previous_t = -1
        self.current_scale: Union[float, Tensor] = None
        self.current_effect: Union[float, Tensor] = None
        self.current_per_block_list: Union[list[PerBlock], None] = None
        self.combined_scale: Union[float, Tensor] = None
        self.combined_effect: Union[float, Tensor] = None
        self.combined_per_block_list: Union[list[PerBlock], None] = None
        self.was_within_range = False
        self.prev_sub_idxs = None
        self.prev_batched_number = None

    def pre_run(self, patcher: MotionModelPatcher):
        self.cleanup(patcher)
        patcher.model.set_scale(self.scale_multival, self.per_block_list)
        patcher.model.set_effect(self.effect_multival, self.per_block_list)
        patcher.model.set_cameractrl_effect(self.cameractrl_multival)
        if patcher.model.img_encoder is not None:
            patcher.model.img_encoder.set_ref_drift(self.orig_ref_drift)
            patcher.model.img_encoder.set_insertion_weights(self.orig_insertion_weights)

    def initialize_timesteps(self, model: BaseModel):
        self.timestep_range = (model.model_sampling.percent_to_sigma(self.timestep_percent_range[0]),
                               model.model_sampling.percent_to_sigma(self.timestep_percent_range[1]))
        if self.keyframes is not None:
            for keyframe in self.keyframes.keyframes:
                keyframe.start_t = model.model_sampling.percent_to_sigma(keyframe.start_percent)

    def prepare_current_keyframe(self, patcher: MotionModelPatcher, x: Tensor, t: Tensor, transformer_options: dict[str, Tensor]):
        curr_t: float = t[0]
        # if curr_t was previous_t, then do nothing (already accounted for this step)
        if curr_t == self.previous_t:
            return
        prev_index = self.current_index
        max_sigma = torch.max(transformer_options.get("sample_sigmas", BIGMAX_TENSOR))
        # if met guaranteed steps, look for next keyframe in case need to switch
        if self.current_keyframe is None or self.current_used_steps >= self.current_keyframe.get_effective_guarantee_steps(max_sigma):
            # if has next index, loop through and see if need to switch
            if self.keyframes.has_index(self.current_index+1):
                for i in range(self.current_index+1, len(self.keyframes)):
                    eval_kf = self.keyframes[i]
                    # check if start_t is greater or equal to curr_t
                    # NOTE: t is in terms of sigmas, not percent, so bigger number = earlier step in sampling
                    if eval_kf.start_t >= curr_t:
                        self.current_index = i
                        self.current_keyframe = eval_kf
                        self.current_used_steps = 0
                        # NOTE: handle possible inputs from keyframe, taking into account inherit_missing
                        # scale
                        if self.current_keyframe.has_scale():
                            self.current_scale = self.current_keyframe.scale_multival
                        elif not self.current_keyframe.inherit_missing:
                            self.current_scale = None
                        # effect
                        if self.current_keyframe.has_effect():
                            self.current_effect = self.current_keyframe.effect_multival
                        elif not self.current_keyframe.inherit_missing:
                            self.current_effect = None
                        # per_block_list
                        if self.current_keyframe.has_per_block_replace():
                            self.current_per_block_list = self.current_keyframe.per_block_list
                        elif not self.current_keyframe.inherit_missing:
                            self.current_per_block_list = None
                        # cameractrl_effect
                        if self.current_keyframe.has_cameractrl_effect():
                            self.current_cameractrl_effect = self.current_keyframe.cameractrl_multival
                        elif not self.current_keyframe.inherit_missing:
                            self.current_cameractrl_effect = None
                        # pia_input
                        if self.current_keyframe.has_pia_input():
                            self.current_pia_input = self.current_keyframe.pia_input
                        elif not self.current_keyframe.inherit_missing:
                            self.current_pia_input = None
                        # if guarantee_steps greater than zero, stop searching for other keyframes
                        if self.current_keyframe.get_effective_guarantee_steps(max_sigma) > 0:
                            break
                    # if eval_kf is outside the percent range, stop looking further
                    else:
                        break
        # if index changed, apply new combined values
        if prev_index != self.current_index:
            # combine model's scale and effect with keyframe's scale and effect
            self.combined_scale = get_combined_multival(self.scale_multival, self.current_scale)
            self.combined_effect = get_combined_multival(self.effect_multival, self.current_effect)
            self.combined_per_block_list = get_combined_per_block_list(self.per_block_list, self.current_per_block_list)
            self.combined_cameractrl_effect = get_combined_multival(self.cameractrl_multival, self.current_cameractrl_effect)
            self.combined_pia_mask = get_combined_input(self.pia_input, self.current_pia_input, x)
            self.combined_pia_effect = get_combined_input_effect_multival(self.pia_input, self.current_pia_input)
            # apply scale and effect
            patcher.model.set_scale(self.combined_scale, self.combined_per_block_list)
            patcher.model.set_effect(self.combined_effect, self.combined_per_block_list)
            patcher.model.set_cameractrl_effect(self.combined_cameractrl_effect)
        # apply effect - if not within range, set effect to 0, effectively turning model off
        if curr_t > self.timestep_range[0] or curr_t < self.timestep_range[1]:
            patcher.model.set_effect(0.0)
            self.was_within_range = False
        else:
            # if was not in range last step, apply effect to toggle AD status
            if not self.was_within_range:
                patcher.model.set_effect(self.combined_effect, self.combined_per_block_list)
                self.was_within_range = True
        # update steps current keyframe is used
        self.current_used_steps += 1
        # update previous_t
        self.previous_t = curr_t

    def prepare_alcmi2v_features(self, patcher: MotionModelPatcher, x: Tensor, cond_or_uncond: list[int], ad_params: dict[str], latent_format):
        '''Used for AnimateLCM-I2V'''
        # if no img_encoder, done
        if patcher.model.img_encoder is None:
            return
        batched_number = len(cond_or_uncond)
        full_length = ad_params["full_length"]
        sub_idxs = ad_params["sub_idxs"]
        goal_length = x.size(0) // batched_number
        # calculate img_features if needed
        if (self.img_latents_shape is None or sub_idxs != self.prev_sub_idxs or batched_number != self.prev_batched_number
                or x.shape[2] != self.img_latents_shape[2] or x.shape[3] != self.img_latents_shape[3]):
            if sub_idxs is not None and self.orig_img_latents.size(0) >= full_length:
                img_latents = comfy.utils.common_upscale(self.orig_img_latents[sub_idxs], x.shape[3], x.shape[2], 'nearest-exact', 'center').to(x.dtype).to(x.device)
            else:
                img_latents = comfy.utils.common_upscale(self.orig_img_latents, x.shape[3], x.shape[2], 'nearest-exact', 'center').to(x.dtype).to(x.device)
            img_latents: Tensor = latent_format.process_in(img_latents)
            # make sure img_latents matches goal_length
            if goal_length != img_latents.shape[0]:
                img_latents = ade_broadcast_image_to(img_latents, goal_length, batched_number)
            img_features = patcher.model.img_encoder(img_latents, goal_length, batched_number)
            patcher.model.set_img_features(img_features=img_features, apply_ref_when_disabled=self.orig_apply_ref_when_disabled)
            # cache values for next step
            self.img_latents_shape = img_latents.shape
        self.prev_sub_idxs = sub_idxs
        self.prev_batched_number = batched_number

    def prepare_camera_features(self, patcher: MotionModelPatcher, x: Tensor, cond_or_uncond: list[int], ad_params: dict[str]):
        '''Used for CameraCtrl'''
        # if no camera_encoder, done
        if patcher.model.camera_encoder is None:
            return
        batched_number = len(cond_or_uncond)
        full_length = ad_params["full_length"]
        sub_idxs = ad_params["sub_idxs"]
        goal_length = x.size(0) // batched_number
        # calculate camera_features if needed
        if self.camera_features_shape is None or sub_idxs != self.prev_sub_idxs or batched_number != self.prev_batched_number:
            # make sure there are enough camera_poses to match full_length
            camera_poses = self.orig_camera_entries.copy()
            if len(camera_poses) < full_length:
                for i in range(full_length-len(camera_poses)):
                    camera_poses.append(camera_poses[-1])
            if sub_idxs is not None:
                camera_poses = [camera_poses[idx] for idx in sub_idxs]
            # make sure camera_poses matches goal_length
            if len(camera_poses) > goal_length:
                camera_poses = camera_poses[:goal_length]
            elif len(camera_poses) < goal_length:
                # pad the camera_poses with the last element to match goal_length
                for i in range(goal_length-len(camera_poses)):
                    camera_poses.append(camera_poses[-1])
            # create encoded embeddings
            b, c, h, w = x.shape
            plucker_embedding = prepare_pose_embedding(camera_poses, image_width=w*8, image_height=h*8).to(dtype=x.dtype, device=x.device)
            camera_embedding = patcher.model.camera_encoder(plucker_embedding, video_length=goal_length, batched_number=batched_number)
            patcher.model.set_camera_features(camera_features=camera_embedding)
            self.camera_features_shape = len(camera_embedding)
        self.prev_sub_idxs = sub_idxs
        self.prev_batched_number = batched_number

    def prepare_motionctrl_camera(self, patcher: MotionModelPatcher, x: Tensor, transformer_options: dict[str]):
        '''Used for MotionCtrl'''
        # if no cc enabled, done
        if not patcher.model.is_motionctrl_cc_enabled():
            if "ADE_RT" in transformer_options:
                transformer_options.pop("ADE_RT")
            return
        cond_or_uncond: list[int] = transformer_options["cond_or_uncond"]
        uuids: list = transformer_options["uuids"]
        batched_number = len(cond_or_uncond)
        ad_params = transformer_options["ad_params"]
        full_length = ad_params["full_length"]
        sub_idxs = ad_params["sub_idxs"]
        goal_length = x.size(0) // batched_number
        if self.prev_RT_shape != x.shape or sub_idxs != self.prev_sub_idxs or uuids != self.prev_RT_uuids:
            real_RT = self.orig_RT.clone().to(dtype=x.dtype, device=x.device) # [t, 12]
            # make sure RT is of the valid length
            real_RT = extend_to_batch_size(real_RT, full_length)
            if sub_idxs is not None:
                real_RT = real_RT[sub_idxs]
            real_RT = real_RT.unsqueeze(0) # [1, t, 12]
            # match batch length - conds get real_RT, unconds get empty
            if batched_number > 1:
                batched_RTs = []
                for condtype in cond_or_uncond:
                    if condtype == 0: # cond
                        batched_RTs.append(real_RT)
                    else: # uncond
                        batched_RTs.append(torch.zeros_like(real_RT))
                real_RT = torch.cat(batched_RTs, dim=0)
            self.RT = real_RT.to(dtype=x.dtype, device=x.device)
            self.prev_RT_shape = x.shape
        transformer_options["ADE_RT"] = self.RT
        self.prev_sub_idxs = sub_idxs
        self.prev_batched_number = batched_number


    def get_pia_c_concat(self, model: BaseModel, x: Tensor) -> Tensor:
        '''Used for PIA'''
        # if have cached shape, check if matches - if so, return cached pia_latents
        if self.prev_pia_latents_shape is not None:
            if self.prev_pia_latents_shape[0] == x.shape[0] and self.prev_pia_latents_shape[2] == x.shape[2] and self.prev_pia_latents_shape[3] == x.shape[3]:
                # if mask is also the same for this timestep, then return cached
                if self.prev_current_pia_input == self.current_pia_input:
                    return self.cached_pia_c_concat
                # otherwise, adjust new mask, and create new cached_pia_c_concat
                b, c, h ,w = x.shape
                mask = prepare_mask_batch(self.combined_pia_mask, x.shape)
                mask = extend_to_batch_size(mask, b)
                # make sure to update prev_current_pia_input to know when is changed
                self.prev_current_pia_input = self.current_pia_input
                # TODO: handle self.combined_pia_effect eventually (feature hidden for now)
                # the first index in dim=1 is the mask that needs to be updated - update in place
                self.cached_pia_c_concat.cond[:, :1, :, :] = mask
                return self.cached_pia_c_concat
        self.prev_pia_latents_shape = None
        # otherwise, x shape should be the cached pia_latents_shape
        # get currently used models so they can be properly reloaded after perfoming VAE Encoding
        cached_loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
        try:
            b, c, h ,w = x.shape
            usable_ref = self.orig_pia_images[:b]
            # in diffusers, the image is scaled from [-1, 1] instead of default [0, 1],
            # but form my testing, that blows out the images here, so I skip it
            # usable_images = usable_images * 2 - 1
            # resize images to latent's dims
            usable_ref = usable_ref.movedim(-1,1)
            usable_ref = comfy.utils.common_upscale(samples=usable_ref, width=w*self.pia_vae.downscale_ratio, height=h*self.pia_vae.downscale_ratio,
                                                    upscale_method="bilinear", crop="center")
            usable_ref = usable_ref.movedim(1,-1)
            # VAE encode images
            logger.info("VAE Encoding PIA input images...")
            usable_ref = model.process_latent_in(vae_encode_raw_batched(vae=self.pia_vae, pixels=usable_ref, show_pbar=False))
            logger.info("VAE Encoding PIA input images complete.")
            # make pia_latents match expected length
            usable_ref = extend_to_batch_size(usable_ref, b)
            self.prev_pia_latents_shape = x.shape
            # now, take care of the mask
            mask = prepare_mask_batch(self.combined_pia_mask, x.shape)
            mask = extend_to_batch_size(mask, b)
            #mask = mask.unsqueeze(1)
            self.prev_current_pia_input = self.current_pia_input
            if type(self.combined_pia_effect) == Tensor or not math.isclose(self.combined_pia_effect, 1.0):
                real_pia_effect = self.combined_pia_effect
                if type(self.combined_pia_effect) == Tensor:
                    real_pia_effect = extend_to_batch_size(prepare_mask_batch(self.combined_pia_effect, x.shape), b)
                zero_mask = torch.zeros_like(mask)
                mask = mask * real_pia_effect + zero_mask * (1.0 - real_pia_effect)
                del zero_mask
                zero_usable_ref = torch.zeros_like(usable_ref)
                usable_ref = usable_ref * real_pia_effect + zero_usable_ref * (1.0 - real_pia_effect)
                del zero_usable_ref
            # cache pia c_concat
            self.cached_pia_c_concat = comfy.conds.CONDNoiseShape(torch.cat([mask, usable_ref], dim=1))
            return self.cached_pia_c_concat
        finally:
            comfy.model_management.load_models_gpu(cached_loaded_models)

    def get_fancy_c_concat(self, model: BaseModel, x: Tensor) -> Tensor:
        '''Used for FancyVideo'''
        # if have cached shape, check if matches - if so, return cached fancy_latents
        if self.prev_fancy_latents_shape is not None:
            if self.prev_fancy_latents_shape[0] == x.shape[0] and self.prev_fancy_latents_shape[-2] == x.shape[-2] and self.prev_fancy_latents_shape[-1] == x.shape[-1]:
                # TODO: if mask is also the same for this timestep, then retucn cached
                return self.cached_fancy_c_concat
        self.prev_fancy_latents_shape = None
        # otherwise, x shape should be the cached fancy_latents_shape
        # get currently used models so they can be properly reloaded after performing VAE Encoding
        cached_loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
        try:
            b, c, h, w = x.shape
            usable_ref = self.orig_fancy_images[:b]
            # resize images to latent's dims
            usable_ref = usable_ref.movedim(-1,1)
            usable_ref = comfy.utils.common_upscale(samples=usable_ref, width=w*self.fancy_vae.downscale_ratio, height=h*self.fancy_vae.downscale_ratio,
                                                    upscale_method="bilinear", crop="center")
            usable_ref = usable_ref.movedim(1,-1)
            # VAE encode images
            logger.info("VAE Encoding FancyVideo input images...")
            usable_ref: Tensor = model.process_latent_in(vae_encode_raw_batched(vae=self.fancy_vae, pixels=usable_ref, show_pbar=False))
            logger.info("VAE Encoding FancyVideo input images complete.")
            self.prev_fancy_latents_shape = x.shape
            # TODO: experiment with indexes that aren't the first
            # pad usable_ref with zeros
            ref_length = usable_ref.shape[0]
            pad_length = b - ref_length
            zero_ref = torch.zeros([pad_length, c, h, w], dtype=usable_ref.dtype, device=usable_ref.device)
            usable_ref = torch.cat([usable_ref, zero_ref], dim=0)
            del zero_ref
            # create mask
            mask_ones = torch.ones([ref_length, 1, h, w], dtype=usable_ref.dtype, device=usable_ref.device)
            mask_zeros = torch.zeros([pad_length, 1, h, w], dtype=usable_ref.dtype, device=usable_ref.device)
            mask = torch.cat([mask_ones, mask_zeros], dim=0)
            # TODO: experiment with mask strength
            # cache fancy c_concat - ref first, then mask
            self.cached_fancy_c_concat = comfy.conds.CONDNoiseShape(torch.cat([usable_ref, mask], dim=1))
            return self.cached_fancy_c_concat
        finally:
            comfy.model_management.load_models_gpu(cached_loaded_models)

    def is_pia(self, patcher: MotionModelPatcher):
        return patcher.model.mm_info.mm_format == AnimateDiffFormat.PIA and self.orig_pia_images is not None

    def is_fancyvideo(self, patcher: MotionModelPatcher):
        return patcher.model.mm_info.mm_format == AnimateDiffFormat.FANCYVIDEO

    def cleanup(self, patcher: MotionModelPatcher):
        if patcher.model is not None:
            patcher.model.cleanup()
        # AnimateLCM-I2V
        del self.img_features
        self.img_features = None
        self.img_latents_shape = None
        # CameraCtrl
        del self.camera_features
        self.camera_features = None
        self.camera_features_shape = None
        # PIA
        self.combined_pia_mask = None
        self.combined_pia_effect = None
        # MotionCtrl
        self.RT = None
        self.prev_RT_shape = None
        self.prev_RT_uuids = None
        # Default
        self.current_used_steps = 0
        self.current_keyframe = None
        self.current_index = -1
        self.previous_t = -1
        self.current_scale = None
        self.current_effect = None
        self.current_per_block_list = None
        self.combined_scale = None
        self.combined_effect = None
        self.combined_per_block_list = None
        self.was_within_range = False
        self.prev_sub_idxs = None
        self.prev_batched_number = None

    def on_model_patcher_clone(self):
        n = MotionModelAttachment()
        # extra cloned params
        n.timestep_percent_range = self.timestep_percent_range
        n.timestep_range = self.timestep_range
        n.keyframes = self.keyframes.clone()
        n.scale_multival = self.scale_multival
        n.effect_multival = self.effect_multival
        # AnimateLCM-I2V
        n.orig_img_latents = self.orig_img_latents
        n.orig_ref_drift = self.orig_ref_drift
        n.orig_insertion_weights = self.orig_insertion_weights.copy() if self.orig_insertion_weights is not None else self.orig_insertion_weights
        n.orig_apply_ref_when_disabled = self.orig_apply_ref_when_disabled
        # CameraCtrl
        n.orig_camera_entries = self.orig_camera_entries
        n.cameractrl_multival = self.cameractrl_multival
        # PIA
        n.orig_pia_images = self.orig_pia_images
        n.pia_vae = self.pia_vae
        n.pia_input = self.pia_input
        n.pia_multival = self.pia_multival
        return n


class MotionModelGroup:
    def __init__(self, init_motion_model: Union[MotionModelPatcher, list[MotionModelPatcher]]=None):
        self.models: list[MotionModelPatcher] = []
        if init_motion_model is not None:
            if isinstance(init_motion_model, list):
                for m in init_motion_model:
                    self.add(m)
            else:
                self.add(init_motion_model)

    def add(self, mm: MotionModelPatcher):
        # add to end of list
        self.models.append(mm)

    def add_to_start(self, mm: MotionModelPatcher):
        self.models.insert(0, mm)

    def __getitem__(self, index) -> MotionModelPatcher:
        return self.models[index]
    
    def is_empty(self) -> bool:
        return len(self.models) == 0
    
    def clone(self) -> MotionModelGroup:
        cloned = MotionModelGroup()
        for mm in self.models:
            cloned.add(mm)
        return cloned
    
    def set_sub_idxs(self, sub_idxs: list[int]):
        for motion_model in self.models:
            motion_model.model.set_sub_idxs(sub_idxs=sub_idxs)
    
    def set_view_options(self, view_options: ContextOptions):
        for motion_model in self.models:
            motion_model.model.set_view_options(view_options)

    def set_video_length(self, video_length: int, full_length: int):
        for motion_model in self.models:
            motion_model.model.set_video_length(video_length=video_length, full_length=full_length)
    
    def initialize_timesteps(self, model: BaseModel):
        for motion_model in self.models:
            attachment = get_mm_attachment(motion_model)
            attachment.initialize_timesteps(model)

    def pre_run(self, model: ModelPatcher):
        for motion_model in self.models:
            motion_model.pre_run()
    
    def cleanup(self):
        for motion_model in self.models:
            motion_model.cleanup()
    
    def prepare_current_keyframe(self, x: Tensor, t: Tensor, transformer_options: dict[str, Tensor]):
        for motion_model in self.models:
            attachment = get_mm_attachment(motion_model)
            attachment.prepare_current_keyframe(motion_model, x=x, t=t, transformer_options=transformer_options)

    def get_special_models(self):
        pia_motion_models: list[MotionModelPatcher] = []
        for motion_model in self.models:
            attachment = get_mm_attachment(motion_model)
            if attachment.is_pia(motion_model) or attachment.is_fancyvideo(motion_model):
                pia_motion_models.append(motion_model)
        return pia_motion_models

    def get_name_string(self, show_version=False):
        identifiers = []
        for motion_model in self.models:
            id = motion_model.model.mm_info.mm_name
            if show_version:
                id += f":{motion_model.model.mm_info.mm_version}"
            identifiers.append(id)
        return ", ".join(identifiers)


def get_vanilla_model_patcher(m: ModelPatcher) -> ModelPatcher:
    model = ModelPatcher(m.model, m.load_device, m.offload_device, m.size, weight_inplace_update=m.weight_inplace_update)
    model.patches = {}
    for k in m.patches:
        model.patches[k] = m.patches[k][:]

    model.object_patches = m.object_patches.copy()
    model.model_options = copy.deepcopy(m.model_options)
    if hasattr(model, "model_keys"):
        model.model_keys = m.model_keys
    return model


# adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/utils/convert_lora_safetensor_to_diffusers.py
# Example LoRA keys:
#   down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0.processor.to_q_lora.down.weight
#   down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0.processor.to_q_lora.up.weight
#
# Example model keys: 
#   down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0.to_q.weight
#
def load_motion_lora_as_patches(motion_model: MotionModelPatcher, lora: MotionLoraInfo) -> None:
    def get_version(has_midblock: bool):
        return "v2" if has_midblock else "v1"

    lora_path = get_motion_lora_path(lora.name)
    logger.info(f"Loading motion LoRA {lora.name}")
    state_dict = comfy.utils.load_torch_file(lora_path)

    # remove all non-temporal keys (in case model has extra stuff in it)
    for key in list(state_dict.keys()):
        if "temporal" not in key:
            del state_dict[key]
    if len(state_dict) == 0:
        raise ValueError(f"'{lora.name}' contains no temporal keys; it is not a valid motion LoRA!")

    model_has_midblock = motion_model.model.mid_block != None
    lora_has_midblock = has_mid_block(state_dict)
    logger.info(f"Applying a {get_version(lora_has_midblock)} LoRA ({lora.name}) to a { motion_model.model.mm_info.mm_version} motion model.")

    patches = {}
    # convert lora state dict to one that matches motion_module keys and tensors
    for key in state_dict:
        # if motion_module doesn't have a midblock, skip mid_block entries
        if not model_has_midblock:
            if "mid_block" in key: continue
        # only process lora down key (we will process up at the same time as down)
        if "up." in key: continue

        # get up key version of down key
        up_key = key.replace(".down.", ".up.")

        # adapt key to match motion_module key format - remove 'processor.', '_lora', 'down.', and 'up.'
        model_key = key.replace("processor.", "").replace("_lora", "").replace("down.", "").replace("up.", "")

        # motion_module keys have a '0.' after all 'to_out.' weight keys
        if "to_out.0." not in model_key:
            model_key = model_key.replace("to_out.", "to_out.0.")
        
        weight_down = state_dict[key]
        weight_up = state_dict[up_key]
        # actual weights obtained by matrix multiplication of up and down weights
        # save as a tuple, so that (Motion)ModelPatcher's calculate_weight function detects len==1, applying it correctly
        patches[model_key] = (torch.mm(
            comfy.model_management.cast_to_device(weight_up, weight_up.device, torch.float32),
            comfy.model_management.cast_to_device(weight_down, weight_down.device, torch.float32)
            ),)
    del state_dict
    # add patches to motion ModelPatcher
    motion_model.add_patches(patches=patches, strength_patch=lora.strength)


def load_motion_module_gen1(model_name: str, model: ModelPatcher, motion_lora: MotionLoraList = None, motion_model_settings: AnimateDiffSettings = None) -> MotionModelPatcher:
    model_path = get_motion_model_path(model_name)
    logger.info(f"Loading motion module {model_name}")
    mm_state_dict = comfy.utils.load_torch_file(model_path, safe_load=True)
    # TODO: check for empty state dict?
    # get normalized state_dict and motion model info
    mm_state_dict, mm_info = normalize_ad_state_dict(mm_state_dict=mm_state_dict, mm_name=model_name)
    # check that motion model is compatible with sd model
    model_sd_type = get_sd_model_type(model)
    if model_sd_type != mm_info.sd_type:
        raise MotionCompatibilityError(f"Motion module '{mm_info.mm_name}' is intended for {mm_info.sd_type} models, " \
                                       + f"but the provided model is type {model_sd_type}.")
    # apply motion model settings
    mm_state_dict = apply_mm_settings(model_dict=mm_state_dict, mm_settings=motion_model_settings)
    # initialize AnimateDiffModelWrapper
    ad_wrapper = AnimateDiffModel(mm_state_dict=mm_state_dict, mm_info=mm_info)
    ad_wrapper.to(model.model_dtype())
    ad_wrapper.to(model.offload_device)
    load_result = ad_wrapper.load_state_dict(mm_state_dict, strict=False)
    verify_load_result(load_result=load_result, mm_info=mm_info)
    # wrap motion_module into a ModelPatcher, to allow motion lora patches
    motion_model = create_MotionModelPatcher(model=ad_wrapper, load_device=model.load_device, offload_device=model.offload_device)
    # load motion_lora, if present
    if motion_lora is not None:
        for lora in motion_lora.loras:
            load_motion_lora_as_patches(motion_model, lora)
    return motion_model


def load_motion_module_gen2(model_name: str, motion_model_settings: AnimateDiffSettings = None) -> MotionModelPatcher:
    model_path = get_motion_model_path(model_name)
    logger.info(f"Loading motion module {model_name} via Gen2")
    mm_state_dict = comfy.utils.load_torch_file(model_path, safe_load=True)
    # TODO: check for empty state dict?
    # get normalized state_dict and motion model info (converts alternate AD models like HotshotXL into AD keys)
    mm_state_dict, mm_info = normalize_ad_state_dict(mm_state_dict=mm_state_dict, mm_name=model_name)
    # apply motion model settings
    mm_state_dict = apply_mm_settings(model_dict=mm_state_dict, mm_settings=motion_model_settings)
    # initialize AnimateDiffModelWrapper
    ad_wrapper = AnimateDiffModel(mm_state_dict=mm_state_dict, mm_info=mm_info)
    ad_wrapper.to(comfy.model_management.unet_dtype())
    ad_wrapper.to(comfy.model_management.unet_offload_device())
    load_result = ad_wrapper.load_state_dict(mm_state_dict, strict=False)
    verify_load_result(load_result=load_result, mm_info=mm_info)
    # wrap motion_module into a ModelPatcher, to allow motion lora patches
    motion_model = create_MotionModelPatcher(model=ad_wrapper, load_device=comfy.model_management.get_torch_device(),
                                              offload_device=comfy.model_management.unet_offload_device())
    return motion_model


IncompatibleKeys = namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])
def verify_load_result(load_result: IncompatibleKeys, mm_info: AnimateDiffInfo):
    error_msgs: list[str] = []
    is_animatelcm = mm_info.mm_format==AnimateDiffFormat.ANIMATELCM

    remove_missing_idxs = []
    remove_unexpected_idxs = []
    for idx, key in enumerate(load_result.missing_keys):
        # NOTE: AnimateLCM has no pe keys in the model file, so any errors associated with missing pe keys can be ignored
        if is_animatelcm and "pos_encoder.pe" in key:
            remove_missing_idxs.append(idx)
    # remove any keys to ignore in reverse order (to preserve idx correlation)
    for idx in reversed(remove_unexpected_idxs):
        load_result.unexpected_keys.pop(idx)
    for idx in reversed(remove_missing_idxs):
        load_result.missing_keys.pop(idx)
    # copied over from torch.nn.Module.module class Module's load_state_dict func
    if len(load_result.unexpected_keys) > 0:
        error_msgs.insert(
            0, 'Unexpected key(s) in state_dict: {}. '.format(
                ', '.join(f'"{k}"' for k in load_result.unexpected_keys)))
    if len(load_result.missing_keys) > 0:
        error_msgs.insert(
            0, 'Missing key(s) in state_dict: {}. '.format(
                ', '.join(f'"{k}"' for k in load_result.missing_keys)))
    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                            mm_info.mm_name, "\n\t".join(error_msgs)))
    

def create_fresh_motion_module(motion_model: MotionModelPatcher) -> MotionModelPatcher:
    ad_wrapper = AnimateDiffModel(mm_state_dict=motion_model.model.state_dict(), mm_info=motion_model.model.mm_info)
    ad_wrapper.to(comfy.model_management.unet_dtype())
    ad_wrapper.to(comfy.model_management.unet_offload_device())
    ad_wrapper.load_state_dict(motion_model.model.state_dict())
    return create_MotionModelPatcher(model=ad_wrapper, load_device=comfy.model_management.get_torch_device(),
                                      offload_device=comfy.model_management.unet_offload_device())


def create_fresh_encoder_only_model(motion_model: MotionModelPatcher) -> MotionModelPatcher:
    ad_wrapper = EncoderOnlyAnimateDiffModel(mm_state_dict=motion_model.model.state_dict(), mm_info=motion_model.model.mm_info)
    ad_wrapper.to(comfy.model_management.unet_dtype())
    ad_wrapper.to(comfy.model_management.unet_offload_device())
    ad_wrapper.load_state_dict(motion_model.model.state_dict(), strict=False)
    return create_MotionModelPatcher(model=ad_wrapper, load_device=comfy.model_management.get_torch_device(),
                                      offload_device=comfy.model_management.unet_offload_device())


def inject_img_encoder_into_model(motion_model: MotionModelPatcher, w_encoder: MotionModelPatcher):
    motion_model.model.init_img_encoder()
    motion_model.model.img_encoder.to(comfy.model_management.unet_dtype())
    motion_model.model.img_encoder.to(comfy.model_management.unet_offload_device())
    motion_model.model.img_encoder.load_state_dict(w_encoder.model.img_encoder.state_dict())


def inject_pia_conv_in_into_model(motion_model: MotionModelPatcher, w_pia: MotionModelPatcher):
    motion_model.model.init_conv_in(w_pia.model.state_dict())
    motion_model.model.conv_in.to(comfy.model_management.unet_dtype())
    motion_model.model.conv_in.to(comfy.model_management.unet_offload_device())
    motion_model.model.conv_in.load_state_dict(w_pia.model.conv_in.state_dict())
    motion_model.model.mm_info.mm_format = AnimateDiffFormat.PIA


def inject_camera_encoder_into_model(motion_model: MotionModelPatcher, camera_ctrl_name: str):
    camera_ctrl_path = get_motion_model_path(camera_ctrl_name)
    full_state_dict = comfy.utils.load_torch_file(camera_ctrl_path, safe_load=True)
    camera_state_dict: dict[str, Tensor] = dict()
    attention_state_dict: dict[str, Tensor] = dict()
    for key in full_state_dict:
        if key.startswith("encoder"):
            camera_state_dict[key] = full_state_dict[key]
        elif "qkv_merge" in key:
            attention_state_dict[key] = full_state_dict[key]
    # verify has necessary keys
    if len(camera_state_dict) == 0:
        raise Exception("Provided CameraCtrl model had no Camera Encoder-related keys; not a valid CameraCtrl model!")
    if len(attention_state_dict) == 0:
        raise Exception("Provided CameraCtrl model had no qkv_merge keys; not a valid CameraCtrl model!")
    # initialize CameraPoseEncoder on motion model, and load keys
    camera_encoder = CameraPoseEncoder(channels=motion_model.model.layer_channels, nums_rb=2, ops=motion_model.model.ops).to(
        device=comfy.model_management.unet_offload_device(),
        dtype=comfy.model_management.unet_dtype()
    )
    camera_encoder.load_state_dict(camera_state_dict)
    camera_encoder.temporal_pe_max_len = get_position_encoding_max_len(camera_state_dict, mm_name=camera_ctrl_name, mm_format=AnimateDiffFormat.ANIMATEDIFF)
    motion_model.model.set_camera_encoder(camera_encoder=camera_encoder)
    # initialize qkv_merge on specific attention blocks, and load keys
    for key in attention_state_dict:
        key = key.strip()
        # to avoid handling the same qkv_merge twice, only pay attention to the bias keys (bias+weight handled together)
        if key.endswith("weight"):
            continue
        attr_path = key.split(".processor.qkv_merge")[0]
        base_key = key.split(".bias")[0]
        # first, initialize qkv_merge on model
        attention_obj: VersatileAttention  = comfy.utils.get_attr(motion_model.model, attr_path)
        attention_obj.init_qkv_merge(ops=motion_model.model.ops)
        # then, apply weights to qkv_merge
        qkv_merge_state_dict = {}
        qkv_merge_state_dict["weight"] = attention_state_dict[f"{base_key}.weight"]
        qkv_merge_state_dict["bias"] = attention_state_dict[f"{base_key}.bias"]
        attention_obj.qkv_merge.load_state_dict(qkv_merge_state_dict)
        attention_obj.qkv_merge = attention_obj.qkv_merge.to(
            device=comfy.model_management.unet_offload_device(),
            dtype=comfy.model_management.unet_dtype()
        )
    

def validate_model_compatibility_gen2(model: ModelPatcher, motion_model: MotionModelPatcher):
    # check that motion model is compatible with sd model
    model_sd_type = get_sd_model_type(model)
    mm_info = motion_model.model.mm_info
    if model_sd_type != mm_info.sd_type:
        raise MotionCompatibilityError(f"Motion module '{mm_info.mm_name}' is intended for {mm_info.sd_type} models, " \
                                       + f"but the provided model is type {model_sd_type}.")


def validate_per_block_compatibility(motion_model: MotionModelPatcher, all_per_blocks: AllPerBlocks):
    if all_per_blocks is None or all_per_blocks.sd_type is None:
        return
    mm_info = motion_model.model.mm_info
    if all_per_blocks.sd_type != mm_info.sd_type:
        raise Exception(f"Per-Block provided is meant for {all_per_blocks.sd_type}, but provided motion module is for {mm_info.sd_type}.")


def validate_per_block_compatibility_keyframes(motion_model: MotionModelPatcher, keyframes: ADKeyframeGroup):
    if keyframes is None:
        return
    for keyframe in keyframes.keyframes:
        validate_per_block_compatibility(motion_model, keyframe._per_block_replace)


def interpolate_pe_to_length(model_dict: dict[str, Tensor], key: str, new_length: int):
    pe_shape = model_dict[key].shape
    temp_pe = rearrange(model_dict[key], "(t b) f d -> t b f d", t=1)
    temp_pe = F.interpolate(temp_pe, size=(new_length, pe_shape[-1]), mode="bilinear")
    temp_pe = rearrange(temp_pe, "t b f d -> (t b) f d", t=1)
    model_dict[key] = temp_pe
    del temp_pe


def interpolate_pe_to_length_diffs(model_dict: dict[str, Tensor], key: str, new_length: int):
    # TODO: fill out and try out
    pe_shape = model_dict[key].shape
    temp_pe = rearrange(model_dict[key], "(t b) f d -> t b f d", t=1)
    temp_pe = F.interpolate(temp_pe, size=(new_length, pe_shape[-1]), mode="bilinear")
    temp_pe = rearrange(temp_pe, "t b f d -> (t b) f d", t=1)
    model_dict[key] = temp_pe
    del temp_pe


def interpolate_pe_to_length_pingpong(model_dict: dict[str, Tensor], key: str, new_length: int):
    if model_dict[key].shape[1] < new_length:
        temp_pe = model_dict[key]
        flipped_temp_pe = torch.flip(temp_pe[:, 1:-1, :], [1])
        use_flipped = True
        preview_pe = None
        while model_dict[key].shape[1] < new_length:
            preview_pe = model_dict[key]
            model_dict[key] = torch.cat([model_dict[key], flipped_temp_pe if use_flipped else temp_pe], dim=1)
            use_flipped = not use_flipped
        del temp_pe
        del flipped_temp_pe
        del preview_pe
    model_dict[key] = model_dict[key][:, :new_length]


def freeze_mask_of_pe(model_dict: dict[str, Tensor], key: str):
    pe_portion = model_dict[key].shape[2] // 64
    first_pe = model_dict[key][:,:1,:]
    model_dict[key][:,:,pe_portion:] = first_pe[:,:,pe_portion:]
    del first_pe


def freeze_mask_of_attn(model_dict: dict[str, Tensor], key: str):
    attn_portion = model_dict[key].shape[0] // 2
    model_dict[key][:attn_portion,:attn_portion] *= 1.5


def apply_mm_settings(model_dict: dict[str, Tensor], mm_settings: AnimateDiffSettings) -> dict[str, Tensor]:
    if mm_settings is None:
        return model_dict
    if not mm_settings.has_anything_to_apply():
        return model_dict
    # first, handle PE Adjustments
    for adjust_pe in mm_settings.adjust_pe.adjusts:
        adjust_pe: AdjustPE
        if adjust_pe.has_anything_to_apply():
            already_printed = False
            for key in model_dict:
                if "attention_blocks" in key and "pos_encoder" in key:
                    # apply simple motion pe stretch, if needed
                    if adjust_pe.has_motion_pe_stretch():
                        original_length = model_dict[key].shape[1]
                        new_pe_length = original_length + adjust_pe.motion_pe_stretch
                        interpolate_pe_to_length(model_dict, key, new_length=new_pe_length)
                        if adjust_pe.print_adjustment and not already_printed:
                            logger.info(f"[Adjust PE]: PE Stretch from {original_length} to {new_pe_length}.")
                    # apply pe_idx_offset, if needed
                    if adjust_pe.has_initial_pe_idx_offset():
                        original_length = model_dict[key].shape[1]
                        model_dict[key] = model_dict[key][:, adjust_pe.initial_pe_idx_offset:]
                        if adjust_pe.print_adjustment and not already_printed:
                            logger.info(f"[Adjust PE]: Offsetting PEs by {adjust_pe.initial_pe_idx_offset}; PE length to shortens from {original_length} to {model_dict[key].shape[1]}.")
                    # apply has_cap_initial_pe_length, if needed
                    if adjust_pe.has_cap_initial_pe_length():
                        original_length = model_dict[key].shape[1]
                        model_dict[key] = model_dict[key][:, :adjust_pe.cap_initial_pe_length]
                        if adjust_pe.print_adjustment and not already_printed:
                            logger.info(f"[Adjust PE]: Capping PEs (initial) from {original_length} to {model_dict[key].shape[1]}.")
                    # apply interpolate_pe_to_length, if needed
                    if adjust_pe.has_interpolate_pe_to_length():
                        original_length = model_dict[key].shape[1]
                        interpolate_pe_to_length(model_dict, key, new_length=adjust_pe.interpolate_pe_to_length)
                        if adjust_pe.print_adjustment and not already_printed:
                            logger.info(f"[Adjust PE]: Interpolating PE length from {original_length} to {model_dict[key].shape[1]}.")
                    # apply final_pe_idx_offset, if needed
                    if adjust_pe.has_final_pe_idx_offset():
                        original_length = model_dict[key].shape[1]
                        model_dict[key] = model_dict[key][:, adjust_pe.final_pe_idx_offset:]
                        if adjust_pe.print_adjustment and not already_printed:
                            logger.info(f"[Adjust PE]: Capping PEs (final) from {original_length} to {model_dict[key].shape[1]}.")
                    already_printed = True
    # finally, handle Weight Adjustments
    for adjust_w in mm_settings.adjust_weight.adjusts:
        adjust_w: AdjustWeight
        if adjust_w.has_anything_to_apply():
            adjust_w.mark_attrs_as_unprinted()
            for key in model_dict:
                # apply global weight adjustments, if needed
                adjust_w.perform_applicable_ops(attr=AdjustWeight.ATTR_ALL, model_dict=model_dict, key=key)
                if "attention_blocks" in key:
                    # apply pe change, if needed
                    if "pos_encoder" in key:
                        adjust_w.perform_applicable_ops(attr=AdjustWeight.ATTR_PE, model_dict=model_dict, key=key)
                    else:
                        # apply attn change, if needed
                        adjust_w.perform_applicable_ops(attr=AdjustWeight.ATTR_ATTN, model_dict=model_dict, key=key)
                        # apply specific attn changes, if needed
                        # apply attn_q change, if needed
                        if "to_q" in key:
                            adjust_w.perform_applicable_ops(attr=AdjustWeight.ATTR_ATTN_Q, model_dict=model_dict, key=key)
                        # apply attn_q change, if needed
                        elif "to_k" in key:
                            adjust_w.perform_applicable_ops(attr=AdjustWeight.ATTR_ATTN_K, model_dict=model_dict, key=key)
                        # apply attn_q change, if needed
                        elif "to_v" in key:
                            adjust_w.perform_applicable_ops(attr=AdjustWeight.ATTR_ATTN_V, model_dict=model_dict, key=key)
                        # apply to_out changes, if needed
                        elif "to_out" in key:
                            if key.strip().endswith("weight"):
                                adjust_w.perform_applicable_ops(attr=AdjustWeight.ATTR_ATTN_OUT_WEIGHT, model_dict=model_dict, key=key)
                            elif key.strip().endswith("bias"):
                                adjust_w.perform_applicable_ops(attr=AdjustWeight.ATTR_ATTN_OUT_BIAS, model_dict=model_dict, key=key)
                else:
                    adjust_w.perform_applicable_ops(attr=AdjustWeight.ATTR_OTHER, model_dict=model_dict, key=key)
    return model_dict


class InjectionParams:
    def __init__(self, unlimited_area_hack: bool=False, apply_mm_groupnorm_hack: bool=True,
                 apply_v2_properly: bool=True) -> None:
        self.full_length = None
        self.unlimited_area_hack = unlimited_area_hack
        self.apply_mm_groupnorm_hack = apply_mm_groupnorm_hack
        self.apply_v2_properly = apply_v2_properly
        self.context_options: ContextOptionsGroup = ContextOptionsGroup.default()
        self.motion_model_settings = AnimateDiffSettings() # Gen1
        self.sub_idxs = None  # value should NOT be included in clone, so it will auto reset
    
    def set_noise_extra_args(self, noise_extra_args: dict):
        noise_extra_args["context_options"] = self.context_options.clone()

    def set_context(self, context_options: ContextOptionsGroup):
        self.context_options = context_options.clone() if context_options else ContextOptionsGroup.default()
    
    def is_using_sliding_context(self) -> bool:
        return self.context_options.context_length is not None

    def set_motion_model_settings(self, motion_model_settings: AnimateDiffSettings): # Gen1
        if motion_model_settings is None:
            self.motion_model_settings = AnimateDiffSettings()
        else:
            self.motion_model_settings = motion_model_settings

    def reset_context(self):
        self.context_options = ContextOptionsGroup.default()
    
    def clone(self) -> InjectionParams:
        new_params = InjectionParams(
            self.unlimited_area_hack, self.apply_mm_groupnorm_hack, apply_v2_properly=self.apply_v2_properly,
            )
        new_params.full_length = self.full_length
        new_params.set_context(self.context_options)
        new_params.set_motion_model_settings(self.motion_model_settings) # Gen1
        return new_params
    
    def on_model_patcher_clone(self):
        return self.clone()
