from typing import Union

from torch import Tensor


class SelectError(Exception):
    pass


def validate_index(index: int, length: int=0, is_range: bool=False, allow_negative=False, allow_missing=False) -> int:
    # if part of range, do nothing
    if is_range:
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


def convert_to_index_int(raw_index: str, length: int=0, is_range: bool=False, allow_negative=False, allow_missing=False) -> int:
    try:
        return validate_index(int(raw_index), length=length, is_range=is_range, allow_negative=allow_negative, allow_missing=allow_missing)
    except SelectError as e:
        raise SelectError(f"Index '{raw_index}' must be an integer.", e)


def convert_str_to_indexes(indexes_str: str, length: int=0, allow_range=True, allow_missing=False) -> list[int]:
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
                start_index = convert_to_index_int(start_index, length=length, is_range=True, allow_negative=allow_negative, allow_missing=allow_missing)
            else:
                start_index = 0
            end_index = index_range[1]
            if len(end_index) > 0:
                end_index = convert_to_index_int(end_index, length=length, is_range=True, allow_negative=allow_negative, allow_missing=allow_missing)
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
            # if latents were passed in, base indeces on known latent count
            if len(int_indexes) > 0:
                chosen_indexes.extend(int_indexes[start_index:end_index][::step])
            # otherwise, assume indeces are valid
            else:
                chosen_indexes.extend(list(range(start_index, end_index, step)))
        # parse individual indeces
        else:
            chosen_indexes.append(convert_to_index_int(g, length=length, allow_negative=allow_negative, allow_missing=allow_missing))
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
