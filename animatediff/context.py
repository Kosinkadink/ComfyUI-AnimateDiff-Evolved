# from https://github.com/neggles/animatediff-cli/blob/main/src/animatediff/pipelines/context.py
from typing import Callable, Optional

import numpy as np


CONTEXT_SCHEDULE_LIST = ["uniform"]

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


def get_context_scheduler(name: str) -> Callable:
    match name:
        case "uniform":
            return uniform
        case _:
            raise ValueError(f"Unknown context_overlap policy {name}")


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


################################################
# Manual testing code
def context_test():
    video_length = 32
    context_frames = 16
    context_stride = 4
    context_overlap = 4
    context_schedule = "uniform"
    closed_loop = False

    context_scheduler = get_context_scheduler(context_schedule)

    num_inference_steps = 20
    timesteps = [num_inference_steps]*num_inference_steps
    total_loops = 0
    for i, t in enumerate(timesteps):
        print(f"@@@@ i,t: {i},{t}")
        for context in context_scheduler(i, num_inference_steps, video_length, context_frames, context_stride, context_overlap, closed_loop=closed_loop, print_final=True):
            print(context)
            total_loops += 1
    print(f"actual total loops: {total_loops}")
    print(f"total calculated steps: {get_total_steps(context_scheduler, timesteps, num_inference_steps, video_length, context_frames, context_stride, context_overlap, closed_loop=closed_loop)}")
    print(f"total calculated steps fixed: {get_total_steps_fixed(context_scheduler, timesteps, num_inference_steps, video_length, context_frames, context_stride, context_overlap, closed_loop=closed_loop)}")
    


if __name__ == "__main__":
    context_test()
################################################
