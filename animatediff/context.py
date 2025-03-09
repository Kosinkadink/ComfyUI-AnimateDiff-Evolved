from typing import Union

import torch
import torchvision
from PIL import Image, ImageFont, ImageDraw

import numpy as np
from torch import Tensor

import comfy.samplers
from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher

from .context_extras import ContextExtrasGroup
from .utils_model import BIGMAX_TENSOR
from .utils_motion import get_sorted_list_via_attr


class ContextFuseMethod:
    FLAT = "flat"
    PYRAMID = "pyramid"
    RELATIVE = "relative"
    OVERLAP_LINEAR = "overlap-linear"

    RANDOM = "🔬random"
    RANDOM_DEPR = "random"
    GAUSS_SIGMA = "🔬gauss-sigma"
    GAUSS_SIGMA_DEPR = "gauss-sigma"
    GAUSS_SIGMA_INV = "🔬gauss-sigma inverse"
    GAUSS_SIGMA_INV_DEPR = "gauss-sigma inverse"
    DELAYED_REVERSE_SAWTOOTH = "🔬delayed reverse sawtooth"
    DELAYED_REVERSE_SAWTOOTH_DEPR = "delayed reverse sawtooth"
    PYRAMID_SIGMA = "🔬pyramid-sigma"
    PYRAMID_SIGMA_DEPR = "pyramid-sigma"
    PYRAMID_SIGMA_INV = "🔬pyramid-sigma inverse"
    PYRAMID_SIGMA_INV_DEPR = "pyramid-sigma inverse"

    LIST = [PYRAMID, FLAT, OVERLAP_LINEAR, DELAYED_REVERSE_SAWTOOTH, PYRAMID_SIGMA, PYRAMID_SIGMA_INV, GAUSS_SIGMA, GAUSS_SIGMA_INV, RANDOM]
    LIST_STATIC = [PYRAMID, RELATIVE, FLAT, OVERLAP_LINEAR, DELAYED_REVERSE_SAWTOOTH, PYRAMID_SIGMA, PYRAMID_SIGMA_INV, GAUSS_SIGMA, GAUSS_SIGMA_INV, RANDOM]


class ContextType:
    UNIFORM_WINDOW = "uniform window"


class ContextOptions:
    def __init__(self, context_length: int=None, context_stride: int=None, context_overlap: int=None,
                 context_schedule: str=None, closed_loop: bool=False, fuse_method: str=ContextFuseMethod.FLAT,
                 use_on_equal_length: bool=False, view_options: 'ContextOptions'=None,
                 start_percent=0.0, guarantee_steps=1):
        # permanent settings
        self.context_length = context_length
        self.context_stride = context_stride
        self.context_overlap = context_overlap
        self.context_schedule = context_schedule
        self.closed_loop = closed_loop
        self.fuse_method = fuse_method
        self.sync_context_to_pe = False  # this feature is likely bad and stay unused, so I might remove this
        self.use_on_equal_length = use_on_equal_length
        self.view_options = view_options.clone() if view_options else view_options
        # scheduling
        self.start_percent = float(start_percent)
        self.start_t = 999999999.9
        self.guarantee_steps = guarantee_steps
        # temporary vars
        self._step: int = 0
    
    @property
    def step(self):
        return self._step
    @step.setter
    def step(self, value: int):
        self._step = value
        if self.view_options:
            self.view_options.step = value

    def get_effective_guarantee_steps(self, max_sigma: torch.Tensor):
        '''If keyframe starts before current sampling range (max_sigma), treat as 0.'''
        if self.start_t > max_sigma:
            return 0
        return self.guarantee_steps

    def clone(self):
        n = ContextOptions(context_length=self.context_length, context_stride=self.context_stride,
                                  context_overlap=self.context_overlap, context_schedule=self.context_schedule,
                                  closed_loop=self.closed_loop, fuse_method=self.fuse_method,
                                  use_on_equal_length=self.use_on_equal_length, view_options=self.view_options,
                                  start_percent=self.start_percent, guarantee_steps=self.guarantee_steps)
        n.start_t = self.start_t
        return n


class ContextOptionsGroup:
    def __init__(self):
        self.contexts: list[ContextOptions] = []
        self.extras = ContextExtrasGroup()
        self._current_context: ContextOptions = None
        self._current_used_steps: int = 0
        self._current_index: int = 0
        self._previous_t = -1
        self._step = 0

    def reset(self):
        self._current_context = None
        self._current_used_steps = 0
        self._current_index = 0
        self._previous_t = -1
        self.step = 0
        self._set_first_as_current()
        self.extras.cleanup()

    @property
    def step(self):
        return self._step
    @step.setter
    def step(self, value: int):
        self._step = value
        if self._current_context is not None:
            self._current_context.step = value

    @classmethod
    def default(cls):
        def_context = ContextOptions()
        new_group = ContextOptionsGroup()
        new_group.add(def_context)
        return new_group

    def add(self, context: ContextOptions):
        # add to end of list, then sort
        self.contexts.append(context)
        self.contexts = get_sorted_list_via_attr(self.contexts, "start_percent")
        self._set_first_as_current()

    def add_to_start(self, context: ContextOptions):
        # add to start of list, then sort
        self.contexts.insert(0, context)
        self.contexts = get_sorted_list_via_attr(self.contexts, "start_percent")
        self._set_first_as_current()

    def has_index(self, index: int) -> int:
        return index >=0 and index < len(self.contexts)

    def is_empty(self) -> bool:
        return len(self.contexts) == 0

    def clone(self):
        cloned = ContextOptionsGroup()
        cloned.extras = self.extras.clone()
        for context in self.contexts:
            cloned.contexts.append(context)
        cloned._set_first_as_current()
        return cloned

    def initialize_timesteps(self, model: BaseModel):
        for context in self.contexts:
            context.start_t = model.model_sampling.percent_to_sigma(context.start_percent)
        self.extras.initialize_timesteps(model)

    def prepare_current(self, t: Tensor, transformer_options):
        self.prepare_current_context(t, transformer_options)
        self.extras.prepare_current(t, transformer_options)

    def prepare_current_context(self, t: Tensor, transformer_options: dict[str, Tensor]):
        curr_t: float = t[0]
        # if same as previous, do nothing as step already accounted for
        if curr_t == self._previous_t:
            return
        prev_index = self._current_index
        max_sigma = torch.max(transformer_options.get("sample_sigmas", BIGMAX_TENSOR))
        # if met guaranteed steps, look for next context in case need to switch
        if self._current_used_steps >= self._current_context.get_effective_guarantee_steps(max_sigma):
            # if has next index, loop through and see if need to switch
            if self.has_index(self._current_index+1):
                for i in range(self._current_index+1, len(self.contexts)):
                    eval_c  = self.contexts[i]
                    # check if start_t is greater or equal to curr_t
                    # NOTE: t is in terms of sigmas, not percent, so bigger number = earlier step in sampling
                    if eval_c.start_t >= curr_t:
                        self._current_index = i
                        self._current_context = eval_c
                        self._current_used_steps = 0
                        # if guarantee_steps greater than zero, stop searching for other keyframes
                        if self._current_context.get_effective_guarantee_steps(max_sigma) > 0:
                            break
                    # if eval_c is outside the percent range, stop looking further
                    else:
                        break
        # update steps current context is used
        self._current_used_steps += 1
        # update previous_t
        self._previous_t = curr_t

    def _set_first_as_current(self):
        if len(self.contexts) > 0:
            self._current_context = self.contexts[0]

    # properties shadow those of ContextOptions
    @property
    def context_length(self):
        return self._current_context.context_length
    
    @property
    def context_overlap(self):
        return self._current_context.context_overlap
    
    @property
    def context_stride(self):
        return self._current_context.context_stride
    
    @property
    def context_schedule(self):
        return self._current_context.context_schedule
    
    @property
    def closed_loop(self):
        return self._current_context.closed_loop
    
    @property
    def fuse_method(self):
        return self._current_context.fuse_method
    
    @property
    def use_on_equal_length(self):
        return self._current_context.use_on_equal_length
    
    @property
    def view_options(self):
        return self._current_context.view_options


class ContextSchedules:
    UNIFORM_LOOPED = "looped_uniform"
    UNIFORM_STANDARD = "standard_uniform"
    STATIC_STANDARD = "standard_static"
    BATCHED = "batched"
    VIEW_AS_CONTEXT = "view_as_context"
    SVD_EXTENSION = "svd_extension"

    LEGACY_UNIFORM_LOOPED = "uniform"
    LEGACY_UNIFORM_SCHEDULE_LIST = [LEGACY_UNIFORM_LOOPED]


# from https://github.com/neggles/animatediff-cli/blob/main/src/animatediff/pipelines/context.py
def create_windows_uniform_looped(num_frames: int, opts: Union[ContextOptionsGroup, ContextOptions]):
    windows = []
    if num_frames < opts.context_length:
        windows.append(list(range(num_frames)))
        return windows
    
    context_stride = min(opts.context_stride, int(np.ceil(np.log2(num_frames / opts.context_length))) + 1)
    # obtain uniform windows as normal, looping and all
    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(opts.step)))
        for j in range(
            int(ordered_halving(opts.step) * context_step) + pad,
            num_frames + pad + (0 if opts.closed_loop else -opts.context_overlap),
            (opts.context_length * context_step - opts.context_overlap),
        ):
            windows.append([e % num_frames for e in range(j, j + opts.context_length * context_step, context_step)])

    return windows


def create_windows_uniform_standard(num_frames: int, opts: Union[ContextOptionsGroup, ContextOptions]):
    # unlike looped, uniform_straight does NOT allow windows that loop back to the beginning;
    # instead, they get shifted to the corresponding end of the frames.
    # in the case that a window (shifted or not) is identical to the previous one, it gets skipped.
    windows = []
    if num_frames <= opts.context_length:
        windows.append(list(range(num_frames)))
        return windows
    
    context_stride = min(opts.context_stride, int(np.ceil(np.log2(num_frames / opts.context_length))) + 1)
    # first, obtain uniform windows as normal, looping and all
    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(opts.step)))
        for j in range(
            int(ordered_halving(opts.step) * context_step) + pad,
            num_frames + pad + (-opts.context_overlap),
            (opts.context_length * context_step - opts.context_overlap),
        ):
            windows.append([e % num_frames for e in range(j, j + opts.context_length * context_step, context_step)])
    
    # now that windows are created, shift any windows that loop, and delete duplicate windows
    delete_idxs = []
    win_i = 0
    while win_i < len(windows):
        # if window is rolls over itself, need to shift it
        is_roll, roll_idx = does_window_roll_over(windows[win_i], num_frames)
        if is_roll:
            roll_val = windows[win_i][roll_idx]  # roll_val might not be 0 for windows of higher strides
            shift_window_to_end(windows[win_i], num_frames=num_frames)
            # check if next window (cyclical) is missing roll_val
            if roll_val not in windows[(win_i+1) % len(windows)]:
                # need to insert new window here - just insert window starting at roll_val
                windows.insert(win_i+1, list(range(roll_val, roll_val + opts.context_length)))
        # delete window if it's not unique
        for pre_i in range(0, win_i):
            if windows[win_i] == windows[pre_i]:
                delete_idxs.append(win_i)
                break
        win_i += 1

    # reverse delete_idxs so that they will be deleted in an order that doesn't break idx correlation
    delete_idxs.reverse()
    for i in delete_idxs:
        windows.pop(i)

    return windows


def create_windows_static_standard(num_frames: int, opts: Union[ContextOptionsGroup, ContextOptions]):
    windows = []
    if num_frames <= opts.context_length:
        windows.append(list(range(num_frames)))
        return windows
    # always return the same set of windows
    delta = opts.context_length - opts.context_overlap
    for start_idx in range(0, num_frames, delta):
        # if past the end of frames, move start_idx back to allow same context_length
        ending = start_idx + opts.context_length
        if ending >= num_frames:
            final_delta = ending - num_frames
            final_start_idx = start_idx - final_delta
            windows.append(list(range(final_start_idx, final_start_idx + opts.context_length)))
            break
        windows.append(list(range(start_idx, start_idx + opts.context_length)))
    return windows


def create_windows_batched(num_frames: int, opts: Union[ContextOptionsGroup, ContextOptions]):
    windows = []
    if num_frames <= opts.context_length:
        windows.append(list(range(num_frames)))
        return windows
    # always return the same set of windows;
    # no overlap, just cut up based on context_length;
    # last window size will be different if num_frames % opts.context_length != 0
    for start_idx in range(0, num_frames, opts.context_length):
        windows.append(list(range(start_idx, min(start_idx + opts.context_length, num_frames))))
    return windows


def create_windows_default(num_frames: int, opts: Union[ContextOptionsGroup, ContextOptions]):
    return [list(range(num_frames))]


def get_context_windows(num_frames: int, opts: Union[ContextOptionsGroup, ContextOptions]) -> list[list[int]]:
    context_func = CONTEXT_MAPPING.get(opts.context_schedule, None)
    if not context_func:
        raise ValueError(f"Unknown context_schedule '{opts.context_schedule}'.")
    return context_func(num_frames, opts)


CONTEXT_MAPPING = {
    ContextSchedules.UNIFORM_LOOPED: create_windows_uniform_looped,
    ContextSchedules.UNIFORM_STANDARD: create_windows_uniform_standard,
    ContextSchedules.STATIC_STANDARD: create_windows_static_standard,
    ContextSchedules.BATCHED: create_windows_batched,
    ContextSchedules.SVD_EXTENSION: create_windows_batched,
    ContextSchedules.VIEW_AS_CONTEXT: create_windows_default,  # just return all to allow Views to do all the work
}


def get_context_weights(length: int, full_length: int, idxs: list[int], ctx_opts: ContextOptions, sigma: Tensor=None):
    weights_func = FUSE_MAPPING.get(ctx_opts.fuse_method, None)
    if not weights_func:
        raise ValueError(f"Unknown fuse_method '{ctx_opts.fuse_method}'.")
    return weights_func(length, sigma=sigma, ctx_opts=ctx_opts, full_length=full_length, idxs=idxs)


def create_weights_flat(length: int, **kwargs) -> list[float]:
    # weight is the same for all
    return [1.0] * length

def create_weights_pyramid(length: int, **kwargs) -> list[float]:
    # weight is based on the distance away from the edge of the context window;
    # based on weighted average concept in FreeNoise paper
    if length % 2 == 0:
        max_weight = length // 2
        weight_sequence = list(range(1, max_weight + 1, 1)) + list(range(max_weight, 0, -1))
    else:
        max_weight = (length + 1) // 2
        weight_sequence = list(range(1, max_weight, 1)) + [max_weight] + list(range(max_weight - 1, 0, -1))
    return weight_sequence

def create_weights_overlap_linear(length: int, full_length: int, idxs: list[int], ctx_opts: ContextOptions, **kwargs):
    # based on code in Kijai's WanVideoWrapper: https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/dbb2523b37e4ccdf45127e5ae33e31362f755c8e/nodes.py#L1302
    # only expected overlap is given different weights
    weights_torch = torch.ones((length))
    # blend left-side on all except first window
    if min(idxs) > 0:
        ramp_up = torch.linspace(1e-37, 1, ctx_opts.context_overlap)
        weights_torch[:ctx_opts.context_overlap] = ramp_up
    # blend right-side on all except last window
    if max(idxs) < full_length-1:
        ramp_down = torch.linspace(1, 1e-37, ctx_opts.context_overlap)
        weights_torch[-ctx_opts.context_overlap:] = ramp_down
    return weights_torch

def create_weights_random(length: int, **kwargs) -> list[float]:
    if length % 2 == 0:
        max_weight = length // 2
    else:
        max_weight = (length + 1) // 2
    return list(np.random.random(length)*max_weight+0.001)
    
def create_weights_gauss_sigma(length: int, **kwargs) -> list[float]:
    sigma = 1.0 + 8.0*(min(4.0, kwargs["sigma"].mean().cpu()) / 4.0)
    ax = np.linspace(-(length - 1) / 2., (length - 1) / 2., length)
    w = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    if length % 2 == 0:
        max_weight = length // 2
    else:
        max_weight = (length + 1) // 2
    w *= max_weight / np.linalg.norm(w)
    #print("create_weights_gauss_sigma sigma",sigma,w)
    return list(w)
    
def create_weights_gauss_sigma_inv(length: int, **kwargs) -> list[float]:
    sigma = 1.0 + 8.0*(1.0-min(4.0, kwargs["sigma"].mean().cpu()) / 4.0)
    ax = np.linspace(-(length - 1) / 2., (length - 1) / 2., length)
    w = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    if length % 2 == 0:
        max_weight = length // 2
    else:
        max_weight = (length + 1) // 2
    w *= max_weight / np.linalg.norm(w)
    #print("create_weights_gauss_sigma_inv sigma",sigma,w)
    return list(w)

def create_weights_pyramid_sigma_inv(length: int, **kwargs) -> list[float]:
    sigma = min(4.0, kwargs["sigma"].mean().cpu()) / 4.0
    
    if length % 2 == 0:
        max_weight = length // 2
        weight_sequence = np.array(list(range(1, max_weight + 1, 1)) + list(range(max_weight, 0, -1)))
        weight_sequence2 = np.array([-max_weight]*(max_weight-1) +[max_weight,max_weight] + [-max_weight]*(max_weight-1))
    else:
        max_weight = (length + 1) // 2
        weight_sequence = list(range(1, max_weight, 1)) + [max_weight] + list(range(max_weight - 1, 0, -1))
        weight_sequence2 = np.array([-max_weight]*(max_weight) +[max_weight] + [-max_weight]*(max_weight-1))
    weight_sequence = (sigma * weight_sequence2 + (1.0-sigma) * weight_sequence).clip(0.001,max_weight)
    #print("create_weights_pyramid_sigma_inv",kwargs["sigma"].mean(),sigma, len(weight_sequence),weight_sequence)
    return list(weight_sequence)

def create_weights_pyramid_sigma(length: int, **kwargs) -> list[float]:
    sigma = min(4.0, kwargs["sigma"].mean().cpu()) / 4.0
    
    if length % 2 == 0:
        max_weight = length // 2
        weight_sequence = np.array(list(range(1, max_weight + 1, 1)) + list(range(max_weight, 0, -1)))
        weight_sequence2 = np.array([-max_weight]*(max_weight-1) +[max_weight,max_weight] + [-max_weight]*(max_weight-1))
    else:
        max_weight = (length + 1) // 2
        weight_sequence = list(range(1, max_weight, 1)) + [max_weight] + list(range(max_weight - 1, 0, -1))
        weight_sequence2 = np.array([-max_weight]*(max_weight) +[max_weight] + [-max_weight]*(max_weight-1))
    weight_sequence = (sigma * weight_sequence + (1.0-sigma) * weight_sequence2).clip(0.001,max_weight)
    #print("create_weights_pyramid_sigma",kwargs["sigma"].mean(),sigma, len(weight_sequence),weight_sequence)
    return list(weight_sequence)

def create_weights_delayed_reverse_sawtooth(length: int, **kwargs) -> list[float]:
    # assigns 0.01 to first half (or half-1 if even) of weights, then the rest of the weights are basically
    # based on distance from context edge
    if length % 2 == 0:
        max_weight = length // 2
        weight_sequence = [0.01]*(max_weight-1) + [max_weight] + list(range(max_weight, 0, -1))
    else:
        max_weight = (length + 1) // 2
        weight_sequence = [0.01]*max_weight + [max_weight] + list(range(max_weight - 1, 0, -1))
    #print("create_weights_delayed_falling_edge",len(weight_sequence),weight_sequence)
    return weight_sequence


FUSE_MAPPING = {
    ContextFuseMethod.FLAT: create_weights_flat,
    ContextFuseMethod.PYRAMID: create_weights_pyramid,
    ContextFuseMethod.RELATIVE: create_weights_pyramid,
    ContextFuseMethod.OVERLAP_LINEAR: create_weights_overlap_linear,
    # experimental
    ContextFuseMethod.GAUSS_SIGMA: create_weights_gauss_sigma,
    ContextFuseMethod.GAUSS_SIGMA_DEPR: create_weights_gauss_sigma,
    ContextFuseMethod.GAUSS_SIGMA_INV: create_weights_gauss_sigma_inv,
    ContextFuseMethod.GAUSS_SIGMA_INV_DEPR: create_weights_gauss_sigma_inv,
    ContextFuseMethod.RANDOM: create_weights_random,
    ContextFuseMethod.RANDOM_DEPR: create_weights_random,
    ContextFuseMethod.DELAYED_REVERSE_SAWTOOTH: create_weights_delayed_reverse_sawtooth,
    ContextFuseMethod.DELAYED_REVERSE_SAWTOOTH_DEPR: create_weights_delayed_reverse_sawtooth,
    ContextFuseMethod.PYRAMID_SIGMA: create_weights_pyramid_sigma,
    ContextFuseMethod.PYRAMID_SIGMA_DEPR: create_weights_pyramid_sigma,
    ContextFuseMethod.PYRAMID_SIGMA_INV: create_weights_pyramid_sigma_inv,
    ContextFuseMethod.PYRAMID_SIGMA_INV_DEPR: create_weights_pyramid_sigma_inv,
}


# Returns fraction that has denominator that is a power of 2
def ordered_halving(val):
    # get binary value, padded with 0s for 64 bits
    bin_str = f"{val:064b}"
    # flip binary value, padding included
    bin_flip = bin_str[::-1]
    # convert binary to int
    as_int = int(bin_flip, 2)
    # divide by 1 << 64, equivalent to 2**64, or 18446744073709551616,
    # or b10000000000000000000000000000000000000000000000000000000000000000 (1 with 64 zero's)
    return as_int / (1 << 64)


def get_missing_indexes(windows: list[list[int]], num_frames: int) -> list[int]:
    all_indexes = list(range(num_frames))
    for w in windows:
        for val in w:
            try:
                all_indexes.remove(val)
            except ValueError:
                pass
    return all_indexes


def does_window_roll_over(window: list[int], num_frames: int) -> tuple[bool, int]:
    prev_val = -1
    for i, val in enumerate(window):
        val = val % num_frames
        if val < prev_val:
            return True, i
        prev_val = val
    return False, -1


def shift_window_to_start(window: list[int], num_frames: int):
    start_val = window[0]
    for i in range(len(window)):
        # 1) subtract each element by start_val to move vals relative to the start of all frames
        # 2) add num_frames and take modulus to get adjusted vals
        window[i] = ((window[i] - start_val) + num_frames) % num_frames


def shift_window_to_end(window: list[int], num_frames: int):
    # 1) shift window to start
    shift_window_to_start(window, num_frames)
    end_val = window[-1]
    end_delta = num_frames - end_val - 1
    for i in range(len(window)):
        # 2) add end_delta to each val to slide windows to end
        window[i] = window[i] + end_delta


##########################
# Context Visualization
##########################
class Colors:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    CYAN = (0, 255, 255)


class BorderWidth:
    INDEXES = 2
    CONTEXT = 4


class VisualizeSettings:
    def __init__(self, img_width: int, video_length: int):
        self.video_length = video_length
        self.img_width = img_width
        self.grid = img_width // video_length
        self.img_height = self.grid * 5
        self.pil_to_tensor = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])
        self.font_size = int(self.grid * 0.5)
        self.font = ImageFont.load_default(size=self.font_size)
        #self.title_font = ImageFont.load_default(size=int(self.font_size * 1.5))
        self.title_font = ImageFont.load_default(size=int(self.font_size * 1.2))

        self.background_color = Colors.BLACK
        self.grid_outline_color = Colors.WHITE
        self.start_idx_fill_color = Colors.MAGENTA
        self.subidx_end_color = Colors.YELLOW

        self.context_color = Colors.GREEN
        self.view_color = Colors.RED


class GridDisplay:
    def __init__(self, draw: ImageDraw.ImageDraw, vs: VisualizeSettings, home_x: int=0, home_y: int=0):
        self.home_x = home_x
        self.home_y = home_y
        self.draw = draw
        self.vs = vs


def get_text_xy(input: str, font: ImageFont, x: int, y: int, centered=True):
    return (x, y,)


def draw_text(text: str, font: ImageFont, gd: GridDisplay, x: int, y: int, color=Colors.WHITE, centered=True):
    x, y = get_text_xy(text, font, x, y, centered=centered)
    gd.draw.text(xy=(gd.home_x+x, gd.home_y+y), text=text, fill=color, font=font)


def draw_first_grid_row(total_length: int, gd: GridDisplay, start_idx=-1):
    vs = gd.vs
    # the first row is white squares, with the indexes drawed in
    for i in range(total_length):
        x1 = gd.home_x+(vs.grid*i)
        y1 = gd.home_y
        x2 = x1 + vs.grid
        y2 = y1 + vs.grid
        
        fill = None
        if i==start_idx:
            fill=vs.start_idx_fill_color
        gd.draw.rectangle(xy=(x1, y1, x2, y2), fill=fill, outline=vs.grid_outline_color, width=BorderWidth.INDEXES)
        draw_text(text=str(i), font=vs.font, gd=gd, x=vs.grid*i, y=0)


def draw_subidxs(window: list[int], gd: GridDisplay, y_grid_offset: int, color: tuple):
    vs = gd.vs
    # with no indexes drawed in- just solid squares, mostly
    y_offset = vs.grid * y_grid_offset
    for i, val in enumerate(window):
        x1 = gd.home_x+(vs.grid*val)
        y1 = gd.home_y+y_offset
        x2 = x1 + vs.grid
        y2 = y1 + vs.grid
        fill_color = color
        # if at an end of indexes, make inside be different color
        if i == 0 or i == len(window)-1:
            fill_color = vs.subidx_end_color
        gd.draw.rectangle(xy=(x1, y1, x2, y2), fill=fill_color, outline=color, width=BorderWidth.CONTEXT)


def draw_context(window: list[int], gd: GridDisplay):
    draw_subidxs(window=window, gd=gd, y_grid_offset=1, color=gd.vs.context_color)


def draw_view(window: list[int], gd: GridDisplay):
    draw_subidxs(window=window, gd=gd, y_grid_offset=2, color=gd.vs.view_color)


def generate_context_visualization(model: ModelPatcher, context_opts: ContextOptionsGroup=None, sampler_name: str=None, scheduler: str=None,
                                   width=1440, height=200, video_length=32,
                                   steps=None, start_step=None, end_step=None, sigmas=None, force_full_denoise=False, denoise=None):
    if context_opts is None:
        context_opts = ContextOptionsGroup.default()
        params = model.get_attachment("ADE_params")
        if params is not None:
            context_opts = params.context_options
    context_opts = context_opts.clone()
    vs = VisualizeSettings(width, video_length)
    all_imgs = []

    if sigmas is None:
        sampler = comfy.samplers.KSampler(
            model=model, steps=steps, device="cpu", sampler=sampler_name, scheduler=scheduler,
            denoise=denoise, model_options=model.model_options,
        )
        sigmas = sampler.sigmas
        if end_step is not None and end_step < (len(sigmas) - 1):
            sigmas = sigmas[:end_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0
        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
    # remove last sigma, as sampling uses pairs of sigmas at a time (fence post problem)
    sigmas = sigmas[:-1]

    context_opts.reset()
    context_opts.initialize_timesteps(model.model)

    if start_step is None:
        start_step = 0  # use this in case start_step is provided, to display accurate step
    if steps is None:
        steps = len(sigmas)

    for i, t in enumerate(sigmas):
        # make context_opts reflect current step/sigma
        context_opts.prepare_current([t], {})
        context_opts.step = start_step+i

        # check if context should even be active in this case
        context_active = True
        if context_opts.context_length is None:
            context_active = False
        elif video_length < context_opts.context_length:
            context_active = False
        elif video_length == context_opts.context_length and not context_opts.use_on_equal_length:
            context_active = False

        if context_active:
            context_windows = get_context_windows(num_frames=video_length, opts=context_opts)
        else:
            context_windows = [list(range(video_length))]
        start_idx = -1
        for j,window in enumerate(context_windows):
            repeat_count = 0
            view_windows = []
            total_repeats = 1
            view_options = context_opts.view_options
            if view_options is not None:
                view_active = True
                if len(window) < view_options.context_length:
                    view_active = False
                elif video_length == view_options.context_length and not view_options.use_on_equal_length:
                    view_active = False
                if view_active:
                    view_windows = get_context_windows(num_frames=len(window), opts=view_options)
                    total_repeats = len(view_windows)
            while total_repeats > repeat_count:
                # create new frame
                frame: Image = Image.new(mode="RGB", size=(vs.img_width, vs.img_height), color=vs.background_color)
                draw = ImageDraw.Draw(frame)
                gd = GridDisplay(draw=draw, vs=vs, home_x=0, home_y=vs.grid)
                # if views present, do view stuff
                if len(view_windows) > 0:
                    converted_view = [window[x] for x in view_windows[repeat_count]]
                    draw_view(window=converted_view, gd=gd)
                # draw context_type + current step
                title_str = f"{context_opts.context_schedule} - Step {context_opts.step+1}/{steps} (Context {j+1}/{len(context_windows)})"
                if len(view_windows) > 0:
                    title_str = f"{title_str} (View {repeat_count+1}/{len(view_windows)})"
                draw_text(text=title_str, font=vs.title_font, gd=gd, x=0-gd.home_x, y=0-gd.home_y, centered=False)
                # draw first row (total length, white)
                if j == 0:
                    start_idx = window[0]
                draw_first_grid_row(total_length=video_length, gd=gd, start_idx=start_idx)
                # draw context row
                draw_context(window=window, gd=gd)
                # save image + iterate repeat_count
                img: Tensor = vs.pil_to_tensor(frame)
                all_imgs.append(img)
                repeat_count += 1

    images = torch.stack(all_imgs)
    images = images.movedim(1, -1).to(torch.float32)
    return images
