# from https://github.com/neggles/animatediff-cli/blob/main/src/animatediff/pipelines/context.py
from typing import Callable, Optional

import numpy as np


class ContextType:
    UNIFORM_WINDOW = "uniform window"


class ContextOptions:
    CONTEXT_TYPE = None

    def __init__(self):
        pass


class UniformContextOptions(ContextOptions):
    CONTEXT_TYPE = ContextType.UNIFORM_WINDOW

    def __init__(self, context_length: int, context_stride: int, context_overlap: int, context_schedule: int, closed_loop: bool):
        self.context_length = context_length
        self.context_stride = context_stride
        self.context_overlap = context_overlap
        self.context_schedule = context_schedule
        self.closed_loop = closed_loop


class ContextSchedules:
    UNIFORM = "uniform"
    UNIFORM_CONSTANT = "uniform_constant"
    UNIFORM_V2 = "uniform v2"

    CONTEXT_SCHEDULE_LIST = [UNIFORM] # only include somewhat functional contexts here


# Returns fraction that has denominator that is a power of 2
def ordered_halving(val, print_final=False):
    # get binary value, padded with 0s for 64 bits
    bin_str = f"{val:064b}"
    # flip binary value, padding included
    bin_flip = bin_str[::-1]
    # convert binary to int
    as_int = int(bin_flip, 2)
    # divide by 1 << 64, equivalent to 2**64, or 18446744073709551616,
    # or b10000000000000000000000000000000000000000000000000000000000000000 (1 with 64 zero's)
    final = as_int / (1 << 64)
    if print_final:
        print(f"$$$$ final: {final}")
    return final


# Generator that returns lists of latent indeces to diffuse on
def uniform(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
    print_final: bool = False,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step, print_final)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            yield [e % num_frames for e in range(j, j + context_size * context_step, context_step)]

def uniform_v2(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
    print_final: bool = False,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)

    pad = int(round(num_frames * ordered_halving(step, print_final)))
    for context_step in 1 << np.arange(context_stride):
        j_initial = int(ordered_halving(step) * context_step) + pad
        for j in range(
            j_initial,
            num_frames + pad - context_overlap,
            (context_size * context_step - context_overlap),
        ):
            if context_size * context_step > num_frames:
                # On the final context_step,
                # ensure no frame appears in the window twice
                yield [e % num_frames for e in range(j, j + num_frames, context_step)]
                continue
            j = j % num_frames
            if j > (j + context_size * context_step) % num_frames and not closed_loop:
                yield  [e for e in range(j, num_frames, context_step)]
                j_stop = (j + context_size * context_step) % num_frames
                # When  ((num_frames % (context_size - context_overlap)+context_overlap) % context_size != 0,
                # This can cause 'superflous' runs where all frames in
                # a context window have already been processed during
                # the first context window of this stride and step.
                # While the following commented if should prevent this,
                # I believe leaving it in is more correct as it maintains
                # the total conditional passes per frame over a large total steps
                # if j_stop > context_overlap:
                yield [e for e in range(0, j_stop, context_step)]
                continue
            yield [e % num_frames for e in range(j, j + context_size * context_step, context_step)]


def uniform_constant(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
    print_final: bool = False,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)

    # want to avoid loops that connect end to beginning

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step, print_final)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            skip_this_window = False
            prev_val = -1
            to_yield = []
            for e in range(j, j + context_size * context_step, context_step):
                e = e % num_frames
                # if not a closed loop and loops back on itself, should be skipped
                if not closed_loop and e < prev_val:
                    skip_this_window = True
                    break
                to_yield.append(e)
                prev_val = e
            if skip_this_window:
                continue
            # yield if not skipped
            yield to_yield


# This needs to stay here below the context functions
UNIFORM_CONTEXT_MAPPING = {
    ContextSchedules.UNIFORM: uniform,
    ContextSchedules.UNIFORM_CONSTANT: uniform_constant,
    ContextSchedules.UNIFORM_V2: uniform_v2,
}


# TODO: expand to support other context window types (future feature)
def get_context_scheduler(name: str) -> Callable:
    context_func = UNIFORM_CONTEXT_MAPPING.get(name, None)
    if not context_func:
        raise ValueError(f"Unknown context_overlap policy {name}")
    return context_func


def get_total_steps(
    scheduler,
    timesteps: list[int],
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    return sum(
        len(
            list(
                scheduler(
                    i,
                    num_steps,
                    num_frames,
                    context_size,
                    context_stride,
                    context_overlap,
                )
            )
        )
        for i in range(len(timesteps))
    )


def get_total_steps_fixed(
    scheduler,
    timesteps: list[int],
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    total_loops = 0
    for i, t in enumerate(timesteps):
        for context in scheduler(i, num_steps, num_frames, context_size, context_stride, context_overlap, closed_loop=closed_loop):
            total_loops += 1
    return total_loops
