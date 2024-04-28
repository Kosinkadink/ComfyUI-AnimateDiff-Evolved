import uuid
from torch import Tensor


class LoraHookMode:
    MIN_VRAM = "min_vram"
    MAX_SPEED = "max_speed"
    #MIN_VRAM_LOWVRAM = "min_vram_lowvram"
    #MAX_SPEED_LOWVRAM = "max_speed_lowvram"


class LoraHook:
    def __init__(self, lora_name: str):
        self.lora_name = lora_name
        self.id = f"{lora_name}|{uuid.uuid4()}"
    
    # def __eq__(self, other: 'LoraHook'):
    #     return self.id == other.id


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

    def add(self, hook: str):
        if hook not in self.hooks:
            self.hooks.append(hook)
    
    def is_empty(self):
        return len(self.hooks) == 0

    def clone(self):
        cloned = LoraHookGroup()
        for hook in self.hooks:
            cloned.add(hook)
        return cloned

    def clone_and_combine(self, other: 'LoraHookGroup'):
        cloned = self.clone()
        for hook in other.hooks:
            cloned.add(hook)
        return cloned

    @staticmethod
    def combine_all_lora_hooks(lora_hooks_list: list['LoraHookGroup'], require_count=2) -> 'LoraHookGroup':
        actual: list[LoraHookGroup] = []
        for group in lora_hooks_list:
            if group is not None:
                actual.append(group)
        if len(actual) < require_count:
            raise Exception(f"Need at least {require_count} LoRA Hooks to combine, but only had {len(actual)}.")
        final_hook: LoraHookGroup = None
        for hook in actual:
            if final_hook is None:
                final_hook = hook.clone()
            else:
                final_hook = final_hook.clone_and_combine(hook)
        return final_hook


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
