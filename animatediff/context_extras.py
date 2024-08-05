from torch import Tensor

from comfy.model_base import BaseModel

from .utils_motion import prepare_mask_batch, extend_to_batch_size, get_combined_multival


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
class NaiveReuse(ContextExtra):
    def __init__(self, start_percent: float, end_percent: float, weighted_mean: float, multival_opt: Tensor=None):
        super().__init__(start_percent=start_percent, end_percent=end_percent)
        self.weighted_mean = weighted_mean
        self.orig_multival = multival_opt
        self.mask: Tensor = None
    
    def cleanup(self):
        super().cleanup()
        del self.mask
        self.mask = None

    def get_effective_weighted_mean(self, x: Tensor, idxs: list[int]):
        if self.orig_multival is None:
            return self.weighted_mean
        # otherwise, is Tensor and should be extended to match dims and size of x;
        # see if needs to be recalculated
        if type(self.orig_multival) != Tensor:
            return self.weighted_mean * self.orig_multival
        elif self.mask is None or self.mask.shape[0] != x.shape[0] or self.mask.shape[-1] != x.shape[-1] or self.mask.shape[-2] != x.shape[-2]:
            del self.mask
            self.mask = prepare_mask_batch(self.orig_multival, x.shape)
            self.mask = extend_to_batch_size(self.mask, x.shape[0])
        return self.weighted_mean * self.mask[idxs].to(dtype=x.dtype, device=x.device)

    def should_run(self):
        to_return = super().should_run()
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
