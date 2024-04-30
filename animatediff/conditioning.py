from torch import Tensor

from comfy.model_base import BaseModel

from .utils_motion import get_sorted_list_via_attr


class LoraHookMode:
    MIN_VRAM = "min_vram"
    MAX_SPEED = "max_speed"
    #MIN_VRAM_LOWVRAM = "min_vram_lowvram"
    #MAX_SPEED_LOWVRAM = "max_speed_lowvram"


# Acts simply as a way to track unique LoraHooks
class HookRef:
    pass


class LoraHook:
    def __init__(self, lora_name: str):
        self.lora_name = lora_name
        self.lora_keyframe = LoraHookKeyframeGroup()
        self.hook_ref = HookRef()
    
    def initialize_timesteps(self, model: BaseModel):
        self.lora_keyframe.initialize_timesteps(model)

    def reset(self):
        self.lora_keyframe.reset()


    def get_copy(self):
        '''
        Copies LoraHook, but maintains same HookRef
        '''
        c = LoraHook(lora_name=self.lora_name)
        c.lora_keyframe = self.lora_keyframe
        c.hook_ref = self.hook_ref # same instance that acts as ref
        return c

    @property
    def strength(self):
        return self.lora_keyframe.strength

    def __eq__(self, other: 'LoraHook'):
        return self.__class__ == other.__class__ and self.hook_ref == other.hook_ref

    def __hash__(self):
        return hash(self.hook_ref)


class LoraHookGroup:
    '''
    Stores LoRA hooks to apply for conditioning
    '''
    def __init__(self):
        self.hooks: list[LoraHook] = []
    
    def names(self):
        names = []
        for hook in self.hooks:
            names.append(hook.lora_name)
        return ",".join(names)

    def add(self, hook: LoraHook):
        if hook not in self.hooks:
            self.hooks.append(hook)
    
    def is_empty(self):
        return len(self.hooks) == 0

    def contains(self, lora_hook: LoraHook):
        return lora_hook in self.hooks

    def clone(self):
        cloned = LoraHookGroup()
        for hook in self.hooks:
            cloned.add(hook.get_copy())
        return cloned

    def clone_and_combine(self, other: 'LoraHookGroup'):
        cloned = self.clone()
        for hook in other.hooks:
            cloned.add(hook.get_copy())
        return cloned
    
    def set_keyframes_on_hooks(self, hook_kf: 'LoraHookKeyframeGroup'):
        hook_kf = hook_kf.clone()
        for hook in self.hooks:
            hook.lora_keyframe = hook_kf

    @staticmethod
    def combine_all_lora_hooks(lora_hooks_list: list['LoraHookGroup'], require_count=1) -> 'LoraHookGroup':
        actual: list[LoraHookGroup] = []
        for group in lora_hooks_list:
            if group is not None:
                actual.append(group)
        if len(actual) < require_count:
            raise Exception(f"Need at least {require_count} LoRA Hooks to combine, but only had {len(actual)}.")
        # if only 1 hook, just return itself without any cloning
        if len(actual) == 1:
            return actual[0]
        final_hook: LoraHookGroup = None
        for hook in actual:
            if final_hook is None:
                final_hook = hook.clone()
            else:
                final_hook = final_hook.clone_and_combine(hook)
        return final_hook


class LoraHookKeyframe:
    def __init__(self, strength: float, start_percent=0.0, guarantee_steps=1):
        self.strength = strength
        # scheduling
        self.start_percent = float(start_percent)
        self.start_t = 999999999.9
        self.guarantee_steps = guarantee_steps
    
    def clone(self):
        c = LoraHookKeyframe(strength=self.strength,
                             start_percent=self.start_percent, guarantee_steps=self.guarantee_steps)
        c.start_t = self.start_t
        return c

class LoraHookKeyframeGroup:
    def __init__(self):
        self.keyframes: list[LoraHookKeyframe] = []
        self._current_keyframe: LoraHookKeyframe = None
        self._current_used_steps: int = 0
        self._current_index: int = 0
        self._curr_t: float = -1
    
    def reset(self):
        self._current_keyframe = None
        self._current_used_steps = 0
        self._current_index = 0
        self._curr_t = -1
        self._set_first_as_current()

    def add(self, keyframe: LoraHookKeyframe):
        # add to end of list, then sort
        self.keyframes.append(keyframe)
        self.keyframes = get_sorted_list_via_attr(self.keyframes, "start_percent")
        self._set_first_as_current()

    def _set_first_as_current(self):
        if len(self.keyframes) > 0:
            self._current_keyframe = self.keyframes[0]
        else:
            self._current_keyframe = None
    
    def has_index(self, index: int) -> int:
        return index >= 0 and index < len(self.keyframes)
    
    def is_empty(self) -> bool:
        return len(self.keyframes) == 0
    
    def clone(self):
        cloned = LoraHookKeyframeGroup()
        for keyframe in self.keyframes:
            cloned.keyframes.append(keyframe)
        cloned._set_first_as_current()
        return cloned
    
    def initialize_timesteps(self, model: BaseModel):
        for keyframe in self.keyframes:
            keyframe.start_t = model.model_sampling.percent_to_sigma(keyframe.start_percent)

    def prepare_current_keyframe(self, curr_t: float) -> bool:
        if self.is_empty():
            return False
        if curr_t == self._curr_t:
            return False
        prev_index = self._current_index
        # if met guaranteed steps, look for next keyframe in case need to switch
        if self._current_used_steps >= self._current_keyframe.guarantee_steps:
            # if has next index, loop through and see if need t oswitch
            if self.has_index(self._current_index+1):
                for i in range(self._current_index+1, len(self.keyframes)):
                    eval_c = self.keyframes[i]
                    # check if start_t is greater or equal to curr_t
                    # NOTE: t is in terms of sigmas, not percent, so bigger number = earlier step in sampling
                    if eval_c.start_t >= curr_t:
                        self._current_index = i
                        self._current_keyframe = eval_c
                        self._current_used_steps = 0
                        # if guarantee_steps greater than zero, stop searching for other keyframes
                        if self._current_keyframe.guarantee_steps > 0:
                            break
                    # if eval_c is outside the percent range, stop looking further
                    else: break
        # update steps current context is used
        self._current_used_steps += 1
        # update current timestep this was performed on
        self._curr_t = curr_t
        # return True if keyframe changed, False if no change
        return prev_index != self._current_index

    # properties shadow those of LoraHookKeyframe
    @property
    def strength(self):
        if self._current_keyframe is not None:
            return self._current_keyframe.strength
        return 1.0


class COND_CONST:
    KEY_LORA_HOOK = "lora_hook"
    KEY_DEFAULT_COND = "default_cond"

    COND_AREA_DEFAULT = "default"
    COND_AREA_MASK_BOUNDS = "mask bounds"
    _LIST_COND_AREA = [COND_AREA_DEFAULT, COND_AREA_MASK_BOUNDS]


class TimestepsCond:
    def __init__(self, start_percent: float, end_percent: float):
        self.start_percent = start_percent
        self.end_percent = end_percent


def conditioning_set_values(conditioning, values={}):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)
    return c

def set_lora_hook_for_conditioning(conditioning, lora_hook: LoraHookGroup):
    if lora_hook is None:
        return conditioning
    return conditioning_set_values(conditioning, {COND_CONST.KEY_LORA_HOOK: lora_hook})

def set_timesteps_for_conditioning(conditioning, timesteps_cond: TimestepsCond):
    if timesteps_cond is None:
        return conditioning
    return conditioning_set_values(conditioning, {"start_percent": timesteps_cond.start_percent,
                                                  "end_percent": timesteps_cond.end_percent})

def set_mask_for_conditioning(conditioning, mask: Tensor, set_cond_area: str, strength: float):
    if mask is None:
        return conditioning
    set_area_to_bounds = False
    if set_cond_area != COND_CONST.COND_AREA_DEFAULT:
        set_area_to_bounds = True
    if len(mask.shape) < 3:
        mask = mask.unsqueeze(0)

    return conditioning_set_values(conditioning, {"mask": mask,
                                               "set_area_to_bounds": set_area_to_bounds,
                                               "mask_strength": strength})

def combine_conditioning(conds: list):
    combined_conds = []
    for cond in conds:
        combined_conds.extend(cond)
    return combined_conds

def set_mask_conds(conds: list, strength: float, set_cond_area: str,
                   opt_mask: Tensor=None, opt_lora_hook: LoraHookGroup=None, opt_timesteps: TimestepsCond=None):
    masked_conds = []
    for c in conds:
        # first, apply lora_hook to conditioning, if provided
        c = set_lora_hook_for_conditioning(c, opt_lora_hook)
        # next, apply mask to conditioning
        c = set_mask_for_conditioning(conditioning=c, mask=opt_mask, strength=strength, set_cond_area=set_cond_area)
        # apply timesteps, if present
        c = set_timesteps_for_conditioning(conditioning=c, timesteps_cond=opt_timesteps)
        # finally, apply mask to conditioning and store
        masked_conds.append(c)
    return masked_conds

def set_mask_and_combine_conds(conds: list, new_conds: list, strength: float, set_cond_area: str,
                               opt_mask: Tensor=None, opt_lora_hook: LoraHookGroup=None, opt_timesteps: TimestepsCond=None):
    combined_conds = []
    for c, masked_c in zip(conds, new_conds):
        # first, apply lora_hook to new conditioning, if provided
        masked_c = set_lora_hook_for_conditioning(masked_c, opt_lora_hook)
        # next, apply mask to new conditioning, if provided
        masked_c = set_mask_for_conditioning(conditioning=masked_c, mask=opt_mask, set_cond_area=set_cond_area, strength=strength)
        # apply timesteps, if present
        masked_c = set_timesteps_for_conditioning(conditioning=masked_c, timesteps_cond=opt_timesteps)
        # finally, combine with existing conditioning and store
        combined_conds.append(combine_conditioning([c, masked_c]))
    return combined_conds

def set_unmasked_and_combine_conds(conds: list, new_conds: list,
                                   opt_lora_hook: LoraHookGroup, opt_timesteps: TimestepsCond=None):
    combined_conds = []
    for c, new_c in zip(conds, new_conds):
        # first, apply lora_hook to new conditioning, if provided
        new_c = set_lora_hook_for_conditioning(new_c, opt_lora_hook)
        # next, add default_cond key to cond so that during sampling, it can be identified
        new_c = conditioning_set_values(new_c, {COND_CONST.KEY_DEFAULT_COND: True})
        # apply timesteps, if present
        new_c = set_timesteps_for_conditioning(conditioning=new_c, timesteps_cond=opt_timesteps)
        # finally, combine with existing conditioning and store
        combined_conds.append(combine_conditioning([c, new_c]))
    return combined_conds
