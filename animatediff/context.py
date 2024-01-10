from typing import Callable, Optional

import numpy as np

class ContextFuseMethod:
    FLAT = "flat"
    PYRAMID = "pyramid"

    LIST = [FLAT, PYRAMID]


class ContextType:
    UNIFORM_WINDOW = "uniform window"


class ContextOptions:
    def __init__(self, context_length: int=None, context_stride: int=None, context_overlap: int=None,
                 context_schedule: str=None, closed_loop: bool=False, fuse_method: str=ContextFuseMethod.FLAT):
        self.context_length = context_length
        self.context_stride = context_stride
        self.context_overlap = context_overlap
        self.context_schedule = context_schedule
        self.closed_loop = closed_loop
        self.fuse_method = fuse_method
        self.sync_context_to_pe = False
    
    def clone(self):
        n = ContextOptions(context_length=self.context_length, context_stride=self.context_stride,
                                  context_overlap=self.context_overlap, context_schedule=self.context_schedule,
                                  closed_loop=self.closed_loop, fuse_method=self.fuse_method)
        return n


class ContextSchedules:
    UNIFORM_LOOPED = "uniform"
    UNIFORM_STANDARD = "uniform_standard"

    STATIC_STANDARD = "static_standard"

    BATCHED = "batched"

    UNIFORM_SCHEDULE_LIST = [UNIFORM_LOOPED] # only include somewhat functional contexts here
    STATIC_SCHEDULE_LIST = [STATIC_STANDARD]


# from https://github.com/neggles/animatediff-cli/blob/main/src/animatediff/pipelines/context.py
def create_windows_uniform_looped(step: int, num_frames: int, opts: ContextOptions):
    windows = []
    if num_frames <= opts.context_length:
        windows.append(list(range(num_frames)))
        return windows
    
    context_stride = min(opts.context_stride, int(np.ceil(np.log2(num_frames / opts.context_length))) + 1)
    # obtain uniform windows as normal, looping and all
    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if opts.closed_loop else -opts.context_overlap),
            (opts.context_length * context_step - opts.context_overlap),
        ):
            windows.append([e % num_frames for e in range(j, j + opts.context_length * context_step, context_step)])

    return windows


def create_windows_uniform_standard(step: int, num_frames: int, opts: ContextOptions):
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
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
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

    # reverse delete_idxs so that they will be deleted in an order that does break idx correlation
    delete_idxs.reverse()
    for i in delete_idxs:
        windows.pop(i)

    return windows


def create_windows_static_standard(step: int, num_frames: int, opts: ContextOptions):
    windows = []
    if num_frames <= opts.context_length:
        windows.append(list(range(num_frames)))
        return windows
    # always return the same set of windows
    delta = opts.context_length - opts.context_overlap
    for start_idx in range(0, num_frames, delta):
        # if past the end of frames, move start_idx back to allow same context_length
        ending = start_idx + opts.context_length
        if ending > num_frames:
            final_delta = ending - num_frames
            final_start_idx = start_idx - final_delta
            windows.append(list(range(final_start_idx, final_start_idx + opts.context_length)))
            break
        windows.append(list(range(start_idx, start_idx + opts.context_length)))
    return windows


def create_windows_batched(step: int, num_frames: int, opts: ContextOptions):
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


def get_context_windows(step: int, num_frames: int, opts: ContextOptions):
    context_func = CONTEXT_MAPPING.get(opts.context_schedule, None)
    if not context_func:
        raise ValueError(f"Unknown context_schedule '{opts.context_schedule}'")
    return context_func(step, num_frames, opts)


CONTEXT_MAPPING = {
    ContextSchedules.UNIFORM_LOOPED: create_windows_uniform_looped,
    ContextSchedules.UNIFORM_STANDARD: create_windows_uniform_standard,
    ContextSchedules.STATIC_STANDARD: create_windows_static_standard,
    ContextSchedules.BATCHED: create_windows_batched,
}


def generate_distance_weight(n):
    if n % 2 == 0:
        max_weight = n // 2
        weight_sequence = list(range(1, max_weight + 1, 1)) + list(range(max_weight, 0, -1))
    else:
        max_weight = (n + 1) // 2
        weight_sequence = list(range(1, max_weight, 1)) + [max_weight] + list(range(max_weight - 1, 0, -1))
    return weight_sequence


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






################################################################################################

# Generator that returns lists of latent indeces to diffuse on
def uniform(
    step: int,
    num_frames: int,
    opts: ContextOptions,
    print_final: bool = False,
):
    if num_frames <= opts.context_length:
        yield list(range(num_frames))
        return

    context_stride = min(opts.context_stride, int(np.ceil(np.log2(num_frames / opts.context_length))) + 1)

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step, print_final)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if opts.closed_loop else -opts.context_overlap),
            (opts.context_length * context_step - opts.context_overlap),
        ):
            yield [e % num_frames for e in range(j, j + opts.context_length * context_step, context_step)]


#################################
#  helper funcs for testing
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


def uniform_v2(
    step: int = ...,
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
