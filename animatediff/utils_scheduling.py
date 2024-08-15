from typing import Union

import torch
from torch import Tensor
import math


class TensorInterp:
    LERP = "lerp"
    SLERP = "slerp"
    _LIST = [LERP, SLERP]


class SelectError(Exception):
    pass


def lerp_tensors(tensor_from: Tensor, tensor_to: Tensor, strength_to: Tensor):
    # basic weighted average to combine conds
    # TODO: see how far we can generalize this, and if some params need to change
    return torch.mul(tensor_from, (1.0-strength_to)) + torch.mul(tensor_to, strength_to)


# https://matilabs.ai/2024/03/05/slerp-model-merging-primer/#slerp-code
# https://medium.com/@akp83540/slerp-algorithm-a4ce1bacee4a
def slerp_tensors(tensor_from: Tensor, tensor_to: Tensor, strength_to: Tensor, dot_threshold=0.9995):
    # normalize tensors
    normal_from = tensor_from / tensor_from.norm()
    normal_to = tensor_to / tensor_to.norm()
    # get dot product to find the cosine of the angle between the tensors (vectors)
    dot = (normal_from * normal_to).sum()
    # if tensors (vectors) nearly parallel (dot product ~ 1.0), simplify to lerp
    if dot.abs() > dot_threshold:
        return lerp_tensors(tensor_from=tensor_from, tensor_to=tensor_to, strength_to=strength_to)
    # omega (Ω)
    omega = dot.acos()
    # apply formula:
    # q(t) = (q₀ * sin((1 — t) * Ω)) / sin(Ω) + (q₁ * sin(t * Ω)) / sin(Ω)
    # simplified to (extract sin(Ω)):
    # q(t) = ((q₀ * sin((1 — t) * Ω)) + (q₁ * sin(t * Ω))) / sin(Ω)
    sin_from = ((1.0 - strength_to) * omega).sin()
    sin_to = (strength_to * omega).sin()
    return (tensor_from * sin_from + tensor_to * sin_to) / omega.sin()


def validate_index(raw_index: Union[str, int, float], length: int=0, is_range: bool=False, allow_negative=False, allow_missing=False, allow_decimal=False) -> int:
    is_decimal = False
    if isinstance(raw_index, str):
        if '.' in raw_index:
            is_decimal = True
    if is_decimal:
        if not allow_decimal:
            raise SelectError(f"Index '{raw_index}' contains a decimal, but decimal inputs are not allowed.")
        if length == 0:
            raise SelectError(f"Decimal indexes are not allowed when no explicit length ({length}) is provided.")
        try:
            index_float = float(raw_index)
        except ValueError as e:
            raise SelectError(f"Decimal index '{raw_index}' isn't a valid float. ", e)
        if index_float < 0.0 or index_float > 1.0:
            raise SelectError(f"Decimal index must be between 0.0 and 1.0, but was '{index_float}'.")
        if math.isclose(index_float, 1.0):
            index = length-1
        else:
            index = int(index_float * length)
    else:
        try:
            index = int(raw_index)
        except ValueError as e:
            raise SelectError(f"Index '{raw_index}' must be an integer.", e)
    # if part of range, do nothing
    if is_range:
        if index < 0:
            conv_index = length+index
            if conv_index < 0:
                conv_index = 0
            index = conv_index
        return index
    # otherwise, validate index
    # validate not out of range - only when latent_count is passed in
    if length > 0 and index > length-1 and not allow_missing:
        raise SelectError(f"Index '{index}' out of range for {length} item(s).")
    # if negative, validate not out of range
    if index < 0:
        if not allow_negative:
            raise SelectError(f"Negative indeces not allowed, but was '{index}'.")
        conv_index = length+index
        if conv_index < 0 and not allow_missing:
            raise SelectError(f"Index '{index}', converted to '{conv_index}' out of range for {length} item(s).")
        index = conv_index
    return index


def convert_to_index_int(raw_index: str, length: int=0, is_range: bool=False, allow_negative=False, allow_missing=False, allow_decimal=False) -> int:
    return validate_index(raw_index, length=length, is_range=is_range, allow_negative=allow_negative, allow_missing=allow_missing, allow_decimal=allow_decimal)


def convert_str_to_indexes(indexes_str: str, length: int=0, allow_range=True, allow_missing=False, fix_reverse=False, same_is_one=False, allow_decimal=False) -> list[int]:
    if not indexes_str:
        return []
    int_indexes = list(range(0, length))
    allow_negative = length > 0
    chosen_indexes = []
    # parse string - allow positive ints, negative ints, and ranges separated by ':'
    groups = indexes_str.split(",")
    groups = [g.strip() for g in groups]
    for g in groups:
        # parse range of indeces (e.g. 2:16)
        if ':' in g:
            if not allow_range:
                raise SelectError("Ranges (:) not allowed for this input.")
            index_range = g.split(":", 2)
            index_range = [r.strip() for r in index_range]

            start_index = index_range[0]
            if len(start_index) > 0:
                start_index = convert_to_index_int(start_index, length=length, is_range=True, allow_negative=allow_negative, allow_missing=allow_missing, allow_decimal=allow_decimal)
            else:
                start_index = 0
            end_index = index_range[1]
            if len(end_index) > 0:
                end_index = convert_to_index_int(end_index, length=length, is_range=True, allow_negative=allow_negative, allow_missing=allow_missing, allow_decimal=allow_decimal)
            else:
                end_index = length
            # support step as well, to allow things like reversing, every-other, etc.
            step = 1
            if len(index_range) > 2:
                step = index_range[2]
                if len(step) > 0:
                    step = convert_to_index_int(step, length=length, is_range=True, allow_negative=True, allow_missing=True)
                else:
                    step = 1
            # if supposed to treat same start and end as one entry, do so
            if same_is_one and start_index == end_index:
                chosen_indexes.append(convert_to_index_int(start_index, length=length, allow_negative=allow_negative, allow_missing=allow_missing, allow_decimal=allow_decimal))
            else:
                # if should fix_reverse and reverse detected, then swap start and end indexes
                do_reverse = False
                if fix_reverse and end_index < start_index:
                    start_index, end_index = end_index, start_index
                    #do_reverse = True
                # if latents were passed in, base indeces on known latent count
                if len(int_indexes) > 0 and not allow_missing:
                    new_indexes = int_indexes[start_index:end_index][::step]
                    if do_reverse:
                        new_indexes.reverse()
                    chosen_indexes.extend(new_indexes)
                # otherwise, assume indeces are valid
                else:
                    new_indexes = list(range(start_index, end_index, step))
                    if do_reverse:
                        new_indexes.reverse()
                    chosen_indexes.extend(new_indexes)
        # parse individual indeces
        else:
            chosen_indexes.append(convert_to_index_int(g, length=length, allow_negative=allow_negative, allow_missing=allow_missing, allow_decimal=allow_decimal))
    return chosen_indexes


def select_indexes(input_obj: Union[Tensor, list], idxs: list):
    if type(input_obj) == Tensor:
        return input_obj[idxs]
    else:
        return [input_obj[i] for i in idxs]


def select_indexes_from_str(input_obj: Union[Tensor, list], indexes: str, allow_range=True, err_if_missing=True, err_if_empty=True):
    real_idxs = convert_str_to_indexes(indexes, len(input_obj), allow_range=allow_range, allow_missing=not err_if_missing)
    if err_if_empty and len(real_idxs) == 0:
        raise Exception(f"Nothing was selected based on indexes found in '{indexes}'.")
    return select_indexes(input_obj, real_idxs)
