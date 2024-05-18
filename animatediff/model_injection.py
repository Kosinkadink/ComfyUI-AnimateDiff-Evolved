import copy
from typing import Union, Callable

from einops import rearrange
from torch import Tensor
import torch.nn.functional as F
import torch
import uuid
import math

import comfy.lora
import comfy.model_management
import comfy.utils
from comfy.model_patcher import ModelPatcher
from comfy.model_base import BaseModel
from comfy.sd import CLIP

from .ad_settings import AnimateDiffSettings, AdjustPE, AdjustWeight
from .adapter_cameractrl import CameraPoseEncoder, CameraEntry, prepare_pose_embedding
from .context import ContextOptions, ContextOptions, ContextOptionsGroup
from .motion_module_ad import (AnimateDiffModel, AnimateDiffFormat, EncoderOnlyAnimateDiffModel, VersatileAttention,
                               has_mid_block, normalize_ad_state_dict, get_position_encoding_max_len)
from .logger import logger
from .utils_motion import ADKeyframe, ADKeyframeGroup, MotionCompatibilityError, get_combined_multival, ade_broadcast_image_to, normalize_min_max
from .conditioning import HookRef, LoraHook, LoraHookGroup, LoraHookMode
from .motion_lora import MotionLoraInfo, MotionLoraList
from .utils_model import get_motion_lora_path, get_motion_model_path, get_sd_model_type
from .sample_settings import SampleSettings, SeedNoiseGeneration


# some motion_model casts here might fail if model becomes metatensor or is not castable;
# should not really matter if it fails, so ignore raised Exceptions
class ModelPatcherAndInjector(ModelPatcher):
    def __init__(self, m: ModelPatcher):
        # replicate ModelPatcher.clone() to initialize ModelPatcherAndInjector
        super().__init__(m.model, m.load_device, m.offload_device, m.size, m.current_device, weight_inplace_update=m.weight_inplace_update)
        self.patches = {}
        for k in m.patches:
            self.patches[k] = m.patches[k][:]
        if hasattr(m, "patches_uuid"):
            self.patches_uuid = m.patches_uuid

        self.object_patches = m.object_patches.copy()
        self.model_options = copy.deepcopy(m.model_options)
        self.model_keys = m.model_keys
        if hasattr(m, "backup"):
            self.backup = m.backup
        if hasattr(m, "object_patches_backup"):
            self.object_patches_backup = m.object_patches_backup

        # lora hook stuff
        self.hooked_patches: dict[HookRef] = {} # binds LoraHook to specific keys
        self.hooked_backup: dict[str, tuple[Tensor, torch.device]] = {}
        self.cached_hooked_patches: dict[LoraHookGroup, dict[str, Tensor]] = {} # binds LoraHookGroup to pre-calculated weights (speed optimization)
        self.current_lora_hooks = None
        self.lora_hook_mode = LoraHookMode.MAX_SPEED
        self.model_params_lowvram = False 
        self.model_params_lowvram_keys = {} # keeps track of keys with applied 'weight_function' or 'bias_function'
        # injection stuff
        self.currently_injected = False
        self.motion_injection_params: InjectionParams = InjectionParams()
        self.sample_settings: SampleSettings = SampleSettings()
        self.motion_models: MotionModelGroup = None
    
    def clone(self, hooks_only=False):
        cloned = ModelPatcherAndInjector(self)
        # copy lora hooks
        for hook_ref in self.hooked_patches:
            cloned.hooked_patches[hook_ref] = {}
            for k in self.hooked_patches[hook_ref]:
                cloned.hooked_patches[hook_ref][k] = self.hooked_patches[hook_ref][k][:]
        # copy pre-calc weights bound to LoraHookGroups
        for group in self.cached_hooked_patches:
            cloned.cached_hooked_patches[group] = {}
            for k in self.cached_hooked_patches[group]:
                cloned.cached_hooked_patches[group][k] = self.cached_hooked_patches[group][k]
        cloned.hooked_backup = self.hooked_backup
        cloned.current_lora_hooks = self.current_lora_hooks
        cloned.currently_injected = self.currently_injected
        cloned.lora_hook_mode = self.lora_hook_mode
        if not hooks_only:
            cloned.motion_models = self.motion_models.clone() if self.motion_models else self.motion_models
            cloned.sample_settings = self.sample_settings
            cloned.motion_injection_params = self.motion_injection_params.clone() if self.motion_injection_params else self.motion_injection_params
        return cloned
    
    @classmethod
    def create_from(cls, model: Union[ModelPatcher, 'ModelPatcherAndInjector'], hooks_only=False) -> 'ModelPatcherAndInjector':
        if isinstance(model, ModelPatcherAndInjector):
            return model.clone(hooks_only=hooks_only)
        else:
            return ModelPatcherAndInjector(model)

    def clone_has_same_weights(self, clone: 'ModelPatcherCLIPHooks'):
        returned = super().clone_has_same_weights(clone)
        if not returned:
            return returned
        # currently, hook patches require that model gets loaded when sampled, so always say is not a clone if hooks present
        if len(self.hooked_patches) > 0:
            return False
        if type(self) != type(clone):
            return False
        if self.current_lora_hooks != clone.current_lora_hooks:
            return False
        if self.hooked_patches.keys() != clone.hooked_patches.keys():
            return False
        return returned

    def set_lora_hook_mode(self, lora_hook_mode: str):
        self.lora_hook_mode = lora_hook_mode
    
    def prepare_hooked_patches_current_keyframe(self, t: Tensor, hook_groups: list[LoraHookGroup]):
        curr_t = t[0]
        for hook_group in hook_groups:
            for hook in hook_group.hooks:
                changed = hook.lora_keyframe.prepare_current_keyframe(curr_t=curr_t)
                # if keyframe changed, remove any cached LoraHookGroups that contain hook with the same hook_ref;
                # this will cause the weights to be recalculated when sampling
                if changed:
                    # reset current_lora_hooks if contains lora hook that changed
                    if self.current_lora_hooks is not None:
                        for current_hook in self.current_lora_hooks.hooks:
                            if current_hook == hook:
                                self.current_lora_hooks = None
                                break
                    for cached_group in list(self.cached_hooked_patches.keys()):
                        if cached_group.contains(hook):
                            self.cached_hooked_patches.pop(cached_group)

    def clean_hooks(self):
        self.unpatch_hooked()
        self.clear_cached_hooked_weights()
        # for lora_hook in self.hooked_patches:
        #     lora_hook.reset()

    def add_hooked_patches(self, lora_hook: LoraHook, patches, strength_patch=1.0, strength_model=1.0):
        '''
        Based on add_patches, but for hooked weights.
        '''
        # TODO: make this work with timestep scheduling
        current_hooked_patches: dict[str,list] = self.hooked_patches.get(lora_hook.hook_ref, {})
        p = set()
        for key in patches:
            if key in self.model_keys:
                p.add(key)
                current_patches: list[tuple] = current_hooked_patches.get(key, [])
                current_patches.append((strength_patch, patches[key], strength_model))
                current_hooked_patches[key] = current_patches
        self.hooked_patches[lora_hook.hook_ref] = current_hooked_patches
        # since should care about these patches too to determine if same model, reroll patches_uuid
        self.patches_uuid = uuid.uuid4()
        return list(p)
    
    def add_hooked_patches_as_diffs(self, lora_hook: LoraHook, patches: dict, strength_patch=1.0, strength_model=1.0):
        '''
        Based on add_hooked_patches, but intended for using a model's weights as lora hook.
        '''
        # TODO: make this work with timestep scheduling
        current_hooked_patches: dict[str,list] = self.hooked_patches.get(lora_hook.hook_ref, {})
        p = set()
        for key in patches:
            if key in self.model_keys:
                p.add(key)
                current_patches: list[tuple] = current_hooked_patches.get(key, [])
                # take difference between desired weight and existing weight to get diff
                current_patches.append((strength_patch, (patches[key]-comfy.utils.get_attr(self.model, key),), strength_model))
                current_hooked_patches[key] = current_patches
        self.hooked_patches[lora_hook.hook_ref] = current_hooked_patches
        # since should care about these patches too to determine if same model, reroll patches_uuid
        self.patches_uuid = uuid.uuid4()
        return list(p)

    def get_combined_hooked_patches(self, lora_hooks: LoraHookGroup):
        '''
        Returns patches for selected lora_hooks.
        '''
        # combined_patches will contain weights of all relevant lora_hooks, per key
        combined_patches = {}
        if lora_hooks is not None:
            for hook in lora_hooks.hooks:
                hook_patches: dict = self.hooked_patches.get(hook.hook_ref, {})
                for key in hook_patches.keys():
                    current_patches: list[tuple] = combined_patches.get(key, [])
                    if math.isclose(hook.strength, 1.0):
                        # if hook strength is 1.0, can just add it directly
                        current_patches.extend(hook_patches[key])
                    else:
                        # otherwise, need to multiply original patch strength by hook strength
                        # patches are stored as tuples: (strength_patch, (tuple_with_weights,), strength_model)
                        for patch in hook_patches[key]:
                            new_patch = list(patch)
                            new_patch[0] *= hook.strength
                            current_patches.append(tuple(new_patch))
                    combined_patches[key] = current_patches
        return combined_patches

    def model_patches_to(self, device):
        super().model_patches_to(device)

    def patch_model(self, device_to=None, patch_weights=True):
        # first, perform model patching
        if patch_weights: # TODO: keep only 'else' portion when don't need to worry about past comfy versions
            patched_model = super().patch_model(device_to)
        else:
            patched_model = super().patch_model(device_to, patch_weights)
        # finally, perform motion model injection
        self.inject_model()
        return patched_model

    def patch_model_lowvram(self, *args, **kwargs):
        try:
            return super().patch_model_lowvram(*args, **kwargs)
        finally:
            # check if any modules have weight_function or bias_function that is not None
            # NOTE: this serves no purpose currently, but I have it here for future reasons
            for n, m in self.model.named_modules():
                if not hasattr(m, "comfy_cast_weights"):
                    continue
                if getattr(m, "weight_function", None) is not None:
                    self.model_params_lowvram = True
                    self.model_params_lowvram_keys[f"{n}.weight"] = n
                if getattr(m, "bias_function", None) is not None:
                    self.model_params_lowvram = True
                    self.model_params_lowvram_keys[f"{n}.weight"] = n

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        # first, eject motion model from unet
        self.eject_model()
        # finally, do normal model unpatching
        if unpatch_weights: # TODO: keep only 'else' portion when don't need to worry about past comfy versions
            # handle hooked_patches first
            self.clean_hooks()
            try:
                return super().unpatch_model(device_to)
            finally:
                self.model_params_lowvram = False
                self.model_params_lowvram_keys.clear()
        else:
            try:
                return super().unpatch_model(device_to, unpatch_weights)
            finally:
                self.model_params_lowvram = False
                self.model_params_lowvram_keys.clear()

    def inject_model(self):
        if self.motion_models is not None:
            for motion_model in self.motion_models.models:
                self.currently_injected = True
                motion_model.model.inject(self)

    def eject_model(self):
        if self.motion_models is not None:
            for motion_model in self.motion_models.models:
                motion_model.model.eject(self)
            self.currently_injected = False

    def apply_lora_hooks(self, lora_hooks: LoraHookGroup):
        # first, determine if need to reapply patches
        if self.current_lora_hooks == lora_hooks:
            return
        # patch hooks
        self.patch_hooked(lora_hooks=lora_hooks)

    def patch_hooked(self, lora_hooks: LoraHookGroup) -> None:
        # first, unpatch any previous patches
        self.unpatch_hooked()
        # eject model, if needed
        was_injected = self.currently_injected
        if was_injected:
            self.eject_model()

        model_sd = self.model_state_dict()
        # if have cached weights for lora_hooks, use it
        cached_weights = self.cached_hooked_patches.get(lora_hooks, None)
        if cached_weights is not None:
            for key in cached_weights:
                if key not in model_sd:
                    logger.warning(f"Cached LoraHook could not patch. key doesn't exist in model: {key}")
                self.patch_cached_hooked_weight(cached_weights=cached_weights, key=key)
        else:
            # get combined patches of relevant lora_hooks
            relevant_patches = self.get_combined_hooked_patches(lora_hooks=lora_hooks)
            for key in relevant_patches:
                if key not in model_sd:
                    logger.warning(f"LoraHook could not patch. key doesn't exist in model: {key}")
                    continue
                self.patch_hooked_weight_to_device(lora_hooks=lora_hooks, combined_patches=relevant_patches, key=key)
        self.current_lora_hooks = lora_hooks
        # reinject model, if needed
        if was_injected:
            self.inject_model()

    def patch_cached_hooked_weight(self, cached_weights: dict, key: str):
        # TODO: handle model_params_lowvram stuff if necessary
        inplace_update = self.weight_inplace_update
        if key not in self.hooked_backup:
            weight: Tensor = comfy.utils.get_attr(self.model, key)
            target_device = self.offload_device
            if self.lora_hook_mode == LoraHookMode.MAX_SPEED:
                target_device = weight.device
            self.hooked_backup[key] = (weight.to(device=target_device, copy=inplace_update), weight.device)
        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, cached_weights[key])
        else:
            comfy.utils.set_attr_param(self.model, key, cached_weights[key])

    def clear_cached_hooked_weights(self):
        self.cached_hooked_patches.clear()
        self.current_lora_hooks = None

    def patch_hooked_weight_to_device(self, lora_hooks: LoraHookGroup, combined_patches: dict, key: str):
        if key not in combined_patches:
            return

        inplace_update = self.weight_inplace_update
        weight: Tensor = comfy.utils.get_attr(self.model, key)
        if key not in self.hooked_backup:
            target_device = self.offload_device
            if self.lora_hook_mode == LoraHookMode.MAX_SPEED:
                target_device = weight.device
            self.hooked_backup[key] = (weight.to(device=target_device, copy=inplace_update), weight.device)

        # TODO: handle model_params_lowvram stuff if necessary
        temp_weight = comfy.model_management.cast_to_device(weight, weight.device, torch.float32, copy=True)
        out_weight = self.calculate_weight(combined_patches[key], temp_weight, key).to(weight.dtype)
        if self.lora_hook_mode == LoraHookMode.MAX_SPEED:
            self.cached_hooked_patches.setdefault(lora_hooks, {})
            self.cached_hooked_patches[lora_hooks][key] = out_weight
        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def patch_hooked_replace_weight_to_device(self, lora_hooks: LoraHookGroup, model_sd: dict, replace_patches: dict):
        # first handle replace_patches
        for key in replace_patches:
            if key not in model_sd:
                logger.warning(f"LoraHook could not replace patch. key doesn't exist in model: {key}")
                continue

            inplace_update = self.weight_inplace_update
            weight: Tensor = comfy.utils.get_attr(self.model, key)
            if key not in self.hooked_backup:
                # TODO: handle model_params_lowvram stuff if necessary
                target_device = self.offload_device
                if self.lora_hook_mode == LoraHookMode.MAX_SPEED:
                    target_device = weight.device
                self.hooked_backup[key] = (weight.to(device=target_device, copy=inplace_update), weight.device)

            out_weight = replace_patches[key].to(weight.device)
            if self.lora_hook_mode == LoraHookMode.MAX_SPEED:
                self.cached_hooked_patches.setdefault(lora_hooks, {})
                self.cached_hooked_patches[lora_hooks][key] = out_weight
            if inplace_update:
                comfy.utils.copy_to_param(self.model, key, out_weight)
            else:
                comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_hooked(self) -> None:
        # if no backups from before hook, then nothing to unpatch
        if len(self.hooked_backup) == 0:
            return
        was_injected = self.currently_injected
        if was_injected:
            self.eject_model()
        # TODO: handle model_params_lowvram stuff if necessary
        keys = list(self.hooked_backup.keys())
        if self.weight_inplace_update:
            for k in keys:
                if self.lora_hook_mode == LoraHookMode.MAX_SPEED: # does not need to be casted - cache device matches needed device
                    comfy.utils.copy_to_param(self.model, k, self.hooked_backup[k][0])
                else: # should be casted as may not match needed device
                    comfy.utils.copy_to_param(self.model, k, self.hooked_backup[k][0].to(device=self.hooked_backup[k][1]))
        else:
            for k in keys:
                if self.lora_hook_mode == LoraHookMode.MAX_SPEED:
                    comfy.utils.set_attr_param(self.model, k, self.hooked_backup[k][0])
                else: # should be casted as may not match needed device
                    comfy.utils.set_attr_param(self.model, k, self.hooked_backup[k][0].to(device=self.hooked_backup[k][1]))
        # clear hooked_backup
        self.hooked_backup.clear()
        self.current_lora_hooks = None
        # reinject model, if necessary
        if was_injected:
            self.inject_model()


class CLIPWithHooks(CLIP):
    def __init__(self, clip: Union[CLIP, 'CLIPWithHooks']):
        super().__init__(no_init=True)
        self.patcher = ModelPatcherCLIPHooks.create_from(clip.patcher)
        self.cond_stage_model = clip.cond_stage_model
        self.tokenizer = clip.tokenizer
        self.layer_idx = clip.layer_idx
        self.desired_hooks: LoraHookGroup = None
        if hasattr(clip, "desired_hooks"):
            self.set_desired_hooks(clip.desired_hooks)
    
    def clone(self):
        cloned = CLIPWithHooks(clip=self)
        return cloned

    def set_desired_hooks(self, lora_hooks: LoraHookGroup):
        self.desired_hooks = lora_hooks
        self.patcher.set_desired_hooks(lora_hooks=lora_hooks)

    def add_hooked_patches(self, lora_hook: LoraHook, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_hooked_patches(lora_hook=lora_hook, patches=patches, strength_patch=strength_patch, strength_model=strength_model)
    
    def add_hooked_patches_as_diffs(self, lora_hook: LoraHook, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_hooked_patches_as_diffs(lora_hook=lora_hook, patches=patches, strength_patch=strength_patch, strength_model=strength_model)


class ModelPatcherCLIPHooks(ModelPatcher):
    def __init__(self, m: ModelPatcher):
        # replicate ModelPatcher.clone() to initialize
        super().__init__(m.model, m.load_device, m.offload_device, m.size, m.current_device, weight_inplace_update=m.weight_inplace_update)
        self.patches = {}
        for k in m.patches:
            self.patches[k] = m.patches[k][:]
        if hasattr(m, "patches_uuid"):
            self.patches_uuid = m.patches_uuid

        self.object_patches = m.object_patches.copy()
        self.model_options = copy.deepcopy(m.model_options)
        self.model_keys = m.model_keys
        if hasattr(m, "backup"):
            self.backup = m.backup
        if hasattr(m, "object_patches_backup"):
            self.object_patches_backup = m.object_patches_backup
        # lora hook stuff
        self.hooked_patches = {} # binds LoraHook to specific keys
        self.patches_backup = {}
        self.hooked_backup: dict[str, tuple[Tensor, torch.device]] = {}

        self.current_lora_hooks = None
        self.desired_lora_hooks = None
        self.lora_hook_mode = LoraHookMode.MAX_SPEED

        self.model_params_lowvram = False
        self.model_params_lowvram_keys = {} # keeps track of keys with applied 'weight_function' or 'bias_function'

    def clone(self):
        cloned = ModelPatcherCLIPHooks(self)
        # copy lora hooks
        for hook in self.hooked_patches:
            cloned.hooked_patches[hook] = {}
            for k in self.hooked_patches[hook]:
                cloned.hooked_patches[hook][k] = self.hooked_patches[hook][k][:]
        cloned.patches_backup = self.patches_backup
        cloned.hooked_backup = self.hooked_backup
        cloned.current_lora_hooks = self.current_lora_hooks
        cloned.desired_lora_hooks = self.desired_lora_hooks
        cloned.lora_hook_mode = self.lora_hook_mode
        return cloned

    @classmethod
    def create_from(cls, model: Union[ModelPatcher, 'ModelPatcherCLIPHooks']):
        if isinstance(model, ModelPatcherCLIPHooks):
            return model.clone()
        return ModelPatcherCLIPHooks(model)
    
    def clone_has_same_weights(self, clone: 'ModelPatcherCLIPHooks'):
        returned = super().clone_has_same_weights(clone)
        if not returned:
            return returned
        if type(self) != type(clone):
            return False
        if self.desired_lora_hooks != clone.desired_lora_hooks:
            return False
        if self.current_lora_hooks != clone.current_lora_hooks:
            return False
        if self.hooked_patches.keys() != clone.hooked_patches.keys():
            return False
        return returned

    def set_desired_hooks(self, lora_hooks: LoraHookGroup):
        self.desired_lora_hooks = lora_hooks

    def add_hooked_patches(self, lora_hook: LoraHook, patches, strength_patch=1.0, strength_model=1.0):
        '''
        Based on add_patches, but for hooked weights.
        '''
        current_hooked_patches: dict[str,list] = self.hooked_patches.get(lora_hook, {})
        p = set()
        for key in patches:
            if key in self.model_keys:
                p.add(key)
                current_patches: list[tuple] = current_hooked_patches.get(key, [])
                current_patches.append((strength_patch, patches[key], strength_model))
                current_hooked_patches[key] = current_patches
        self.hooked_patches[lora_hook] = current_hooked_patches
        # since should care about these patches too to determine if same model, reroll patches_uuid
        self.patches_uuid = uuid.uuid4()
        return list(p)
    
    def add_hooked_patches_as_diffs(self, lora_hook: LoraHook, patches, strength_patch=1.0, strength_model=1.0):
        '''
        Based on add_hooked_patches, but intended for using a model's weights as lora hook.
        '''
        current_hooked_patches: dict[str,list] = self.hooked_patches.get(lora_hook, {})
        p = set()
        for key in patches:
            if key in self.model_keys:
                p.add(key)
                current_patches: list[tuple] = current_hooked_patches.get(key, [])
                # take difference between desired weight and existing weight to get diff
                current_patches.append((strength_patch, (patches[key]-comfy.utils.get_attr(self.model, key),), strength_model))
                current_hooked_patches[key] = current_patches
        self.hooked_patches[lora_hook] = current_hooked_patches
        # since should care about these patches too to determine if same model, reroll patches_uuid
        self.patches_uuid = uuid.uuid4()
        return list(p)
    
    def get_combined_hooked_patches(self, lora_hooks: LoraHookGroup):
        '''
        Returns patches for selected lora_hooks.
        '''
        # combined_patches will contain weights of all relevant lora_hooks, per key
        combined_patches = {}
        if lora_hooks is not None:
            for hook in lora_hooks.hooks:
                hook_patches: dict = self.hooked_patches.get(hook, {})
                for key in hook_patches.keys():
                    current_patches: list[tuple] = combined_patches.get(key, [])
                    current_patches.extend(hook_patches[key])
                    combined_patches[key] = current_patches
        return combined_patches
    
    def patch_hooked_replace_weight_to_device(self, model_sd: dict, replace_patches: dict):
        # first handle replace_patches
        for key in replace_patches:
            if key not in model_sd:
                logger.warning(f"CLIP LoraHook could not replace patch. key doesn't exist in model: {key}")
                continue
            weight: Tensor = comfy.utils.get_attr(self.model, key)
            inplace_update = self.weight_inplace_update
            target_device = weight.device
            
            if key not in self.hooked_backup:
                self.hooked_backup[key] = (weight.to(device=target_device, copy=inplace_update), weight.device)
            out_weight = replace_patches[key].to(target_device)
            if inplace_update:
                comfy.utils.copy_to_param(self.model, key, out_weight)
            else:
                comfy.utils.set_attr_param(self.model, key, out_weight)

    def patch_model(self, device_to=None, patch_weights=True, *args, **kwargs):
        if self.desired_lora_hooks is not None:
            self.patches_backup = self.patches.copy()
            relevant_patches = self.get_combined_hooked_patches(lora_hooks=self.desired_lora_hooks)
            for key in relevant_patches:
                self.patches.setdefault(key, [])
                self.patches[key].extend(relevant_patches[key])
            self.current_lora_hooks = self.desired_lora_hooks
        return super().patch_model(device_to, patch_weights, *args, **kwargs)

    def patch_model_lowvram(self, *args, **kwargs):
        try:
            return super().patch_model_lowvram(*args, **kwargs)
        finally:
            # check if any modules have weight_function or bias_function that is not None
            # NOTE: this serves no purpose currently, but I have it here for future reasons
            for n, m in self.model.named_modules():
                if not hasattr(m, "comfy_cast_weights"):
                    continue
                if getattr(m, "weight_function", None) is not None:
                    self.model_params_lowvram = True
                    self.model_params_lowvram_keys[f"{n}.weight"] = n
                if getattr(m, "bias_function", None) is not None:
                    self.model_params_lowvram = True
                    self.model_params_lowvram_keys[f"{n}.weight"] = n

    def unpatch_model(self, device_to=None, unpatch_weights=True, *args, **kwargs):
        try:
            return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)
        finally:
            self.patches = self.patches_backup.copy()
            self.patches_backup.clear()
            # handle replace patches
            keys = list(self.hooked_backup.keys())
            if self.weight_inplace_update:
                for k in keys:
                    comfy.utils.copy_to_param(self.model, k, self.hooked_backup[k][0].to(device=self.hooked_backup[k][1]))
            else:
                for k in keys:
                    comfy.utils.set_attr_param(self.model, k, self.hooked_backup[k][0].to(device=self.hooked_backup[k][1]))
            self.model_params_lowvram = False
            self.model_params_lowvram_keys.clear()
            # clear hooked_backup
            self.hooked_backup.clear()
            self.current_lora_hooks = None


def load_hooked_lora_for_models(model: Union[ModelPatcher, ModelPatcherAndInjector], clip: CLIP, lora: dict[str, Tensor], lora_hook: LoraHook,
                                strength_model: float, strength_clip: float):
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    loaded: dict[str] = comfy.lora.load_lora(lora, key_map)
    if model is not None:
        new_modelpatcher = ModelPatcherAndInjector.create_from(model)
        k = new_modelpatcher.add_hooked_patches(lora_hook=lora_hook, patches=loaded, strength_patch=strength_model)
    else:
        k = ()
        new_modelpatcher = None
    
    if clip is not None:
        new_clip = CLIPWithHooks(clip)
        k1 = new_clip.add_hooked_patches(lora_hook=lora_hook, patches=loaded, strength_patch=strength_clip)
    else:
        k1 = ()
        new_clip = None
    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            logger.warning(f"NOT LOADED {x}")
    return (new_modelpatcher, new_clip)


def load_model_as_hooked_lora_for_models(model: Union[ModelPatcher, ModelPatcherAndInjector], clip: CLIP, model_loaded: ModelPatcher, clip_loaded: CLIP, lora_hook: LoraHook,
                                         strength_model: float, strength_clip: float):
    if model is not None and model_loaded is not None:
        new_modelpatcher = ModelPatcherAndInjector.create_from(model)
        comfy.model_management.unload_model_clones(new_modelpatcher)
        expected_model_keys = model_loaded.model_keys.copy()
        patches_model: dict[str, Tensor] = model_loaded.model.state_dict()
        # do not include ANY model_sampling components of the model that should act as a patch
        for key in list(patches_model.keys()):
            if key.startswith("model_sampling"):
                expected_model_keys.discard(key)
                patches_model.pop(key, None)
        k = new_modelpatcher.add_hooked_patches_as_diffs(lora_hook=lora_hook, patches=patches_model, strength_patch=strength_model)
    else:
        k = ()
        new_modelpatcher = None
        
    if clip is not None and clip_loaded is not None:
        new_clip = CLIPWithHooks(clip)
        comfy.model_management.unload_model_clones(new_clip.patcher)
        expected_clip_keys = clip_loaded.patcher.model_keys.copy()
        patches_clip: dict[str, Tensor] = clip_loaded.cond_stage_model.state_dict()
        k1 = new_clip.add_hooked_patches_as_diffs(lora_hook=lora_hook, patches=patches_clip, strength_patch=strength_clip)
    else:
        k1 = ()
        new_clip = None
    
    k = set(k)
    k1 = set(k1)
    if model is not None and model_loaded is not None:
        for key in expected_model_keys:
            if key not in k:
                logger.warning(f"MODEL-AS-LORA NOT LOADED {key}")
    if clip is not None and clip_loaded is not None:
        for key in expected_clip_keys:
            if key not in k1:
                logger.warning(f"CLIP-AS-LORA NOT LOADED {key}")
    
    return (new_modelpatcher, new_clip)


class MotionModelPatcher(ModelPatcher):
    # Mostly here so that type hints work in IDEs
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: AnimateDiffModel = self.model
        self.timestep_percent_range = (0.0, 1.0)
        self.timestep_range: tuple[float, float] = None
        self.keyframes: ADKeyframeGroup = ADKeyframeGroup()

        self.scale_multival = None
        self.effect_multival = None

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
        self.cameractrl_multival = None

        # temporary variables
        self.current_used_steps = 0
        self.current_keyframe: ADKeyframe = None
        self.current_index = -1
        self.current_scale: Union[float, Tensor] = None
        self.current_effect: Union[float, Tensor] = None
        self.current_cameractrl_effect: Union[float, Tensor] = None
        self.combined_scale: Union[float, Tensor] = None
        self.combined_effect: Union[float, Tensor] = None
        self.combined_cameractrl_effect: Union[float, Tensor] = None
        self.was_within_range = False
        self.prev_sub_idxs = None
        self.prev_batched_number = None
    
    def patch_model_lowvram(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, *args, **kwargs):
        patched_model = super().patch_model_lowvram(device_to, lowvram_model_memory, force_patch_weights, *args, **kwargs)

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

        return patched_model

    def pre_run(self, model: ModelPatcherAndInjector):
        self.cleanup()
        self.model.set_scale(self.scale_multival)
        self.model.set_effect(self.effect_multival)
        self.model.set_cameractrl_effect(self.cameractrl_multival)
        if self.model.img_encoder is not None:
            self.model.img_encoder.set_ref_drift(self.orig_ref_drift)
            self.model.img_encoder.set_insertion_weights(self.orig_insertion_weights)

    def initialize_timesteps(self, model: BaseModel):
        self.timestep_range = (model.model_sampling.percent_to_sigma(self.timestep_percent_range[0]),
                               model.model_sampling.percent_to_sigma(self.timestep_percent_range[1]))
        if self.keyframes is not None:
            for keyframe in self.keyframes.keyframes:
                keyframe.start_t = model.model_sampling.percent_to_sigma(keyframe.start_percent)

    def prepare_current_keyframe(self, t: Tensor):
        curr_t: float = t[0]
        prev_index = self.current_index
        # if met guaranteed steps, look for next keyframe in case need to switch
        if self.current_keyframe is None or self.current_used_steps >= self.current_keyframe.guarantee_steps:
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
                        # keep track of scale and effect multivals, accounting for inherit_missing
                        if self.current_keyframe.has_scale():
                            self.current_scale = self.current_keyframe.scale_multival
                        elif not self.current_keyframe.inherit_missing:
                            self.current_scale = None
                        if self.current_keyframe.has_effect():
                            self.current_effect = self.current_keyframe.effect_multival
                        elif not self.current_keyframe.inherit_missing:
                            self.current_effect = None
                        if self.current_keyframe.has_cameractrl_effect():
                            self.current_cameractrl_effect = self.current_keyframe.cameractrl_multival
                        elif not self.current_keyframe.inherit_missing:
                            self.current_cameractrl_effect = None
                        # if guarantee_steps greater than zero, stop searching for other keyframes
                        if self.current_keyframe.guarantee_steps > 0:
                            break
                    # if eval_kf is outside the percent range, stop looking further
                    else:
                        break
        # if index changed, apply new combined values
        if prev_index != self.current_index:
            # combine model's scale and effect with keyframe's scale and effect
            self.combined_scale = get_combined_multival(self.scale_multival, self.current_scale)
            self.combined_effect = get_combined_multival(self.effect_multival, self.current_effect)
            self.combined_cameractrl_effect = get_combined_multival(self.cameractrl_multival, self.current_cameractrl_effect)
            # apply scale and effect
            self.model.set_scale(self.combined_scale)
            self.model.set_effect(self.combined_effect)
            self.model.set_cameractrl_effect(self.combined_cameractrl_effect)
        # apply effect - if not within range, set effect to 0, effectively turning model off
        if curr_t > self.timestep_range[0] or curr_t < self.timestep_range[1]:
            self.model.set_effect(0.0)
            self.was_within_range = False
        else:
            # if was not in range last step, apply effect to toggle AD status
            if not self.was_within_range:
                self.model.set_effect(self.combined_effect)
                self.was_within_range = True
        # update steps current keyframe is used
        self.current_used_steps += 1

    def prepare_img_features(self, x: Tensor, cond_or_uncond: list[int], ad_params: dict[str], latent_format):
        # if no img_encoder, done
        if self.model.img_encoder is None:
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
            img_latents = latent_format.process_in(img_latents)
            # make sure img_latents matches goal_length
            if goal_length != img_latents.shape[0]:
                img_latents = ade_broadcast_image_to(img_latents, goal_length, batched_number)
            img_features = self.model.img_encoder(img_latents, goal_length, batched_number)
            self.model.set_img_features(img_features=img_features, apply_ref_when_disabled=self.orig_apply_ref_when_disabled)
            # cache values for next step
            self.img_latents_shape = img_latents.shape
        self.prev_sub_idxs = sub_idxs
        self.prev_batched_number = batched_number

    def prepare_camera_features(self, x: Tensor, cond_or_uncond: list[int], ad_params: dict[str]):
        # if no camera_encoder, done
        if self.model.camera_encoder is None:
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
            camera_embedding = self.model.camera_encoder(plucker_embedding, video_length=goal_length, batched_number=batched_number)
            self.model.set_camera_features(camera_features=camera_embedding)
            self.camera_features_shape = len(camera_embedding)
        self.prev_sub_idxs = sub_idxs
        self.prev_batched_number = batched_number

    def cleanup(self):
        if self.model is not None:
            self.model.cleanup()
        # AnimateLCM-I2V
        del self.img_features
        self.img_features = None
        self.img_latents_shape = None
        # CameraCtrl
        del self.camera_features
        self.camera_features = None
        self.camera_features_shape = None
        # Default
        self.current_used_steps = 0
        self.current_keyframe = None
        self.current_index = -1
        self.current_scale = None
        self.current_effect = None
        self.combined_scale = None
        self.combined_effect = None
        self.was_within_range = False
        self.prev_sub_idxs = None
        self.prev_batched_number = None

    def clone(self):
        # normal ModelPatcher clone actions
        n = MotionModelPatcher(self.model, self.load_device, self.offload_device, self.size, self.current_device, weight_inplace_update=self.weight_inplace_update)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        if hasattr(n, "patches_uuid"):
            self.patches_uuid = n.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.model_keys = self.model_keys
        if hasattr(n, "backup"):
            self.backup = n.backup
        if hasattr(n, "object_patches_backup"):
            self.object_patches_backup = n.object_patches_backup
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
        return n


class MotionModelGroup:
    def __init__(self, init_motion_model: MotionModelPatcher=None):
        self.models: list[MotionModelPatcher] = []
        if init_motion_model is not None:
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
    
    def clone(self) -> 'MotionModelGroup':
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
            motion_model.initialize_timesteps(model)

    def pre_run(self, model: ModelPatcherAndInjector):
        for motion_model in self.models:
            motion_model.pre_run(model)
    
    def cleanup(self):
        for motion_model in self.models:
            motion_model.cleanup()
    
    def prepare_current_keyframe(self, t: Tensor):
        for motion_model in self.models:
            motion_model.prepare_current_keyframe(t=t)

    def get_name_string(self, show_version=False):
        identifiers = []
        for motion_model in self.models:
            id = motion_model.model.mm_info.mm_name
            if show_version:
                id += f":{motion_model.model.mm_info.mm_version}"
            identifiers.append(id)
        return ", ".join(identifiers)


def get_vanilla_model_patcher(m: ModelPatcher) -> ModelPatcher:
    model = ModelPatcher(m.model, m.load_device, m.offload_device, m.size, m.current_device, weight_inplace_update=m.weight_inplace_update)
    model.patches = {}
    for k in m.patches:
        model.patches[k] = m.patches[k][:]

    model.object_patches = m.object_patches.copy()
    model.model_options = copy.deepcopy(m.model_options)
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
    is_animatelcm = mm_info.mm_format==AnimateDiffFormat.ANIMATELCM
    load_result = ad_wrapper.load_state_dict(mm_state_dict, strict=not is_animatelcm)
    # TODO: report load_result of motion_module loading?
    # wrap motion_module into a ModelPatcher, to allow motion lora patches
    motion_model = MotionModelPatcher(model=ad_wrapper, load_device=model.load_device, offload_device=model.offload_device)
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
    is_animatelcm = mm_info.mm_format==AnimateDiffFormat.ANIMATELCM
    load_result = ad_wrapper.load_state_dict(mm_state_dict, strict=not is_animatelcm)
    # TODO: manually check load_results for AnimateLCM models
    if is_animatelcm:
        pass
    # TODO: report load_result of motion_module loading?
    # wrap motion_module into a ModelPatcher, to allow motion lora patches
    motion_model = MotionModelPatcher(model=ad_wrapper, load_device=comfy.model_management.get_torch_device(),
                                      offload_device=comfy.model_management.unet_offload_device())
    return motion_model


def create_fresh_motion_module(motion_model: MotionModelPatcher) -> MotionModelPatcher:
    ad_wrapper = AnimateDiffModel(mm_state_dict=motion_model.model.state_dict(), mm_info=motion_model.model.mm_info)
    ad_wrapper.to(comfy.model_management.unet_dtype())
    ad_wrapper.to(comfy.model_management.unet_offload_device())
    ad_wrapper.load_state_dict(motion_model.model.state_dict())
    return MotionModelPatcher(model=ad_wrapper, load_device=comfy.model_management.get_torch_device(),
                                      offload_device=comfy.model_management.unet_offload_device())


def create_fresh_encoder_only_model(motion_model: MotionModelPatcher) -> MotionModelPatcher:
    ad_wrapper = EncoderOnlyAnimateDiffModel(mm_state_dict=motion_model.model.state_dict(), mm_info=motion_model.model.mm_info)
    ad_wrapper.to(comfy.model_management.unet_dtype())
    ad_wrapper.to(comfy.model_management.unet_offload_device())
    ad_wrapper.load_state_dict(motion_model.model.state_dict(), strict=False)
    return MotionModelPatcher(model=ad_wrapper, load_device=comfy.model_management.get_torch_device(),
                                      offload_device=comfy.model_management.unet_offload_device()) 


def inject_img_encoder_into_model(motion_model: MotionModelPatcher, w_encoder: MotionModelPatcher):
    motion_model.model.init_img_encoder()
    motion_model.model.img_encoder.to(comfy.model_management.unet_dtype())
    motion_model.model.img_encoder.to(comfy.model_management.unet_offload_device())
    motion_model.model.img_encoder.load_state_dict(w_encoder.model.img_encoder.state_dict())


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
    def __init__(self, unlimited_area_hack: bool=False, apply_mm_groupnorm_hack: bool=True, model_name: str="",
                 apply_v2_properly: bool=True) -> None:
        self.full_length = None
        self.unlimited_area_hack = unlimited_area_hack
        self.apply_mm_groupnorm_hack = apply_mm_groupnorm_hack
        self.model_name = model_name
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
    
    def clone(self) -> 'InjectionParams':
        new_params = InjectionParams(
            self.unlimited_area_hack, self.apply_mm_groupnorm_hack,
            self.model_name, apply_v2_properly=self.apply_v2_properly,
            )
        new_params.full_length = self.full_length
        new_params.set_context(self.context_options)
        new_params.set_motion_model_settings(self.motion_model_settings) # Gen1
        return new_params
