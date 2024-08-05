from typing import Union
import math
import torch
from torch import Tensor

from comfy.model_base import BaseModel

from .utils_motion import (prepare_mask_batch, extend_to_batch_size, get_combined_multival, resize_multival,
                           get_sorted_list_via_attr)


class ContextExtra:
    def __init__(self, start_percent: float, end_percent: float):
        # scheduling
        self.start_percent = float(start_percent)
        self.start_t = 999999999.9
        self.end_percent = float(end_percent)
        self.end_t = 0.0
        self.curr_t = 999999999.9

    def initialize_timesteps(self, model: BaseModel):
        self.start_t = model.model_sampling.percent_to_sigma(self.start_percent)
        self.end_t = model.model_sampling.percent_to_sigma(self.end_percent)

    def prepare_current(self, t: Tensor):
        self.curr_t = t[0]

    def should_run(self):
        if self.curr_t > self.start_t or self.curr_t < self.end_t:
            return False
        return True

    def cleanup(self):
        pass


################################
# ContextRef
class ContextRefParams:
    def __init__(self,
                 attn_style_fidelity=0.0, attn_ref_weight=0.0, attn_strength=0.0,
                 adain_style_fidelity=0.0, adain_ref_weight=0.0, adain_strength=0.0):
        # attn1
        self.attn_style_fidelity = float(attn_style_fidelity)
        self.attn_ref_weight = float(attn_ref_weight)
        self.attn_strength = float(attn_strength)
        # adain
        self.adain_style_fidelity = float(adain_style_fidelity)
        self.adain_ref_weight = float(adain_ref_weight)
        self.adain_strength = float(adain_strength)
    
    def create_dict(self):
        return {
            "attn_style_fidelity": self.attn_style_fidelity,
            "attn_ref_weight": self.attn_ref_weight,
            "attn_strength": self.attn_strength,
            "adain_style_fidelity": self.adain_style_fidelity,
            "adain_ref_weight": self.adain_ref_weight,
            "adain_strength": self.adain_strength,
        }


class ContextRefMode:
    FIRST = "first"
    SLIDING = "sliding"
    INDEXES = "indexes"
    _LIST = [FIRST, SLIDING, INDEXES]

    def __init__(self, mode: str, sliding_width=2, indexes: set[int]=set([0])):
        self.mode = mode
        self.sliding_width = sliding_width
        self.indexes = indexes
        self.single_trigger = True

    @classmethod
    def init_first(cls):
        return ContextRefMode(cls.FIRST)
    
    @classmethod
    def init_sliding(cls, sliding_width: int):
        return ContextRefMode(cls.SLIDING, sliding_width=sliding_width)
    
    @classmethod
    def init_indexes(cls, indexes: set[int]):
        return ContextRefMode(cls.INDEXES, indexes=indexes)


class ContextRef(ContextExtra):
    def __init__(self, start_percent: float, end_percent: float, params: ContextRefParams, mode: ContextRefMode):
        super().__init__(start_percent=start_percent, end_percent=end_percent)
        self.params = params
        self.mode = mode

    def should_run(self):
        return super().should_run()
#--------------------------------


################################
# NaiveReuse 
class NaiveReuseKeyframe:
    def __init__(self, mult_multival: Union[float, Tensor], start_percent=0.0, guarantee_steps=1):
        self.mult_multival = mult_multival
        # scheduling
        self.start_percent = float(start_percent)
        self.start_t = 999999999.9
        self.guarantee_steps = guarantee_steps
    
    def clone(self):
        c = NaiveReuseKeyframe(mult_multival=self.mult_multival,
                               start_percent=self.start_percent, guarantee_steps=self.guarantee_steps)
        c.start_t = self.start_t
        return c

class NaiveReuseKeyframeGroup:
    def __init__(self):
        self.keyframes: list[NaiveReuseKeyframe] = []
        self._current_keyframe: NaiveReuseKeyframe = None
        self._current_used_steps: int = 0
        self._current_index: int = 0
        self._previous_t = -1

    def reset(self):
        self._current_keyframe = None
        self._current_used_steps = 0
        self._current_index = 0
        self._set_first_as_current()

    def add(self, keyframe: NaiveReuseKeyframe):
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
        return index >=0 and index < len(self.keyframes)

    def is_empty(self) -> bool:
        return len(self.keyframes) == 0
    
    def clone(self):
        cloned = NaiveReuseKeyframeGroup()
        for keyframe in self.keyframes:
            cloned.keyframes.append(keyframe)
        cloned._set_first_as_current()
        return cloned
    
    def initialize_timesteps(self, model: BaseModel):
        for keyframe in self.keyframes:
            keyframe.start_t = model.model_sampling.percent_to_sigma(keyframe.start_percent)
    
    def prepare_current_keyframe(self, t: Tensor):
        curr_t: float = t[0]
        # if curr_t same as before, do nothing as step already accounted for
        if curr_t == self._previous_t:
            return
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
        # update previous_t
        self._previous_t = curr_t
    
    # properties shadow those of NaiveReuseKeyframe
    @property
    def mult_multival(self):
        if self._current_keyframe != None:
            return self._current_keyframe.mult_multival
        return None

class NaiveReuse(ContextExtra):
    def __init__(self, start_percent: float, end_percent: float, weighted_mean: float, multival_opt: Union[float, Tensor]=None, naivereuse_kf: NaiveReuseKeyframeGroup=None):
        super().__init__(start_percent=start_percent, end_percent=end_percent)
        self.weighted_mean = weighted_mean
        self.orig_multival = multival_opt
        self.mask: Tensor = None
        self.keyframe = naivereuse_kf if naivereuse_kf else NaiveReuseKeyframeGroup()
        self._prev_keyframe = None
    
    def cleanup(self):
        super().cleanup()
        del self.mask
        self.mask = None
        self._prev_keyframe = None
        self.keyframe.reset()

    def initialize_timesteps(self, model: BaseModel):
        super().initialize_timesteps(model)
        self.keyframe.initialize_timesteps(model)

    def prepare_current(self, t: Tensor):
        super().prepare_current(t)
        self.keyframe.prepare_current_keyframe(t)

    def get_effective_weighted_mean(self, x: Tensor, idxs: list[int]):
        if self.orig_multival is None and self.keyframe.mult_multival is None:
            return self.weighted_mean
        # check if keyframe changed
        keyframe_changed = False
        if self.keyframe._current_keyframe != self._prev_keyframe:
            keyframe_changed = True
        self._prev_keyframe = self.keyframe._current_keyframe

        if type(self.orig_multival) != Tensor and type(self.keyframe.mult_multival) != Tensor:
            return self.weighted_mean * get_combined_multival(self.orig_multival, self.keyframe.mult_multival)

        if self.mask is None or keyframe_changed or self.mask.shape[0] != x.shape[0] or self.mask.shape[-1] != x.shape[-1] or self.mask.shape[-2] != x.shape[-2]:
            del self.mask
            real_mult_multival = resize_multival(self.keyframe.mult_multival, batch_size=x.shape[0], height=x.shape[-1], width=x.shape[-2])
            self.mask = resize_multival(self.orig_multival, batch_size=x.shape[0], height=x.shape[-1], width=x.shape[-2])
            self.mask = get_combined_multival(self.mask, real_mult_multival)
        return self.weighted_mean * self.mask[idxs].to(dtype=x.dtype, device=x.device)

    def should_run(self):
        to_return = super().should_run()
        # if keyframe has 0.0 val, should not run
        if self.keyframe.mult_multival is not None and type(self.keyframe.mult_multival) != Tensor and math.isclose(self.keyframe.mult_multival, 0.0):
            return False
        # if weighted_mean is 0.0, then reuse will take no effect anyway
        return to_return and self.weighted_mean > 0.0
#--------------------------------


class ContextExtrasGroup:
    def __init__(self):
        self.context_ref: ContextRef = None
        self.naive_reuse: NaiveReuse = None
    
    def get_extras_list(self) -> list[ContextExtra]:
        extras_list = []
        if self.context_ref is not None:
            extras_list.append(self.context_ref)
        if self.naive_reuse is not None:
            extras_list.append(self.naive_reuse)
        return extras_list

    def initialize_timesteps(self, model: BaseModel):
        for extra in self.get_extras_list():
            extra.initialize_timesteps(model)

    def prepare_current(self, t: Tensor):
        for extra in self.get_extras_list():
            extra.prepare_current(t)

    def should_run_context_ref(self):
        if not self.context_ref:
            return False
        return self.context_ref.should_run()
    
    def should_run_naive_reuse(self):
        if not self.naive_reuse:
            return False
        return self.naive_reuse.should_run()

    def add(self, extra: ContextExtra):
        if type(extra) == ContextRef:
            self.context_ref = extra
        elif type(extra) == NaiveReuse:
            self.naive_reuse = extra
        else:
            raise Exception(f"Unrecognized ContextExtras type: {type(extra)}")
    
    def cleanup(self):
        for extra in self.get_extras_list():
            extra.cleanup()

    def clone(self):
        cloned = ContextExtrasGroup()
        cloned.context_ref = self.context_ref
        cloned.naive_reuse = self.naive_reuse
        return cloned
