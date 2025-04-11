import re
import math
from typing import Union
import torch
from torch import Tensor
import torch.nn.functional as F
from dataclasses import dataclass, replace

from comfy.sd import CLIP
from comfy.utils import ProgressBar
import comfy.model_management

from .utils_model import InterpolationMethod
from .utils_motion import extend_list_to_batch_size
from .utils_scheduling import SelectError, TensorInterp, convert_str_to_indexes, lerp_tensors, slerp_tensors
from .logger import logger

###############################################
#----------------------------------------------
# JSON prompt format is as follows:
# "idxs": "prompt", ...
_regex_prompt_json = re.compile(r'"([\d:\-\.]+)\s*"\s*:\s*"([^"]*)"(?:\s*,\s*|$)')
# NOTE: I used ChatGPT to generate this regex and summary, as I couldn't be bothered.
# ([\d:\-\.]+): Matches idxs, which can be any combination of digits, colons, and periods.
# \s*: Matches optional whitespace.
# ":\s*": Matches the ":" separator with optional spaces.
# "([^"]*)": Captures the prompt, which can be any character except for double quotation marks.
# (?:\s*,\s*|$): This non-capturing group (?: ... ) matches either a comma (with optional spaces before or after) or the end of the string ($).

# pythonic prompt format is as follows:
# idxs = "prompt", ...
_regex_prompt_pyth = re.compile(r'([\d:\-\.]+)\s*=\s*"([^"]*)"(?:\s*,\s*|$)')
# NOTE: I used ChatGPT to generate this regex and summary, as I couldn't be bothered.
# ([\d:\-\.]+): Matches idx, which can be any combination of digits, colons, and periods.
# \s*=\s*: Matches the equal sign (=) with optional spaces on both sides.
# "([^"]*)": Captures the prompt, which can be any character except for double quotation marks.
# (?:\s*,\s*|$): Matches either a comma (with optional spaces before or after) or the end of the string ($).


# JSON value format is as follows:
# "idxs": value, ...
_regex_value_json = re.compile(r'"([\d:\-\.]+)\s*"\s*:\s*([^,]+)(?:\s*,\s*|$)')
# NOTE: I used ChatGPT to generate this regex and summary, as I couldn't be bothered.
# ([\d:\-\.]+): Matches idxs, which can be any combination of digits, colons, and periods.
# \s*: Matches optional whitespace.
# ":\s*: Matches the ":" separator with optional spaces.
# ([^,]+): Captures the value, which can be any character except for commas (this ensures that values are correctly separated).
# (?:\s*,\s*|$): Matches either a comma (with optional spaces before or after) or the end of the string ($).

# pythonic value format is as follows:
# idxs = value, ...
_regex_value_pyth = re.compile(r'([\d:\-\.]+)\s*=\s*([^,]+)(?:\s*,\s*|$)')
# NOTE: I used ChatGPT to generate this regex and summary, as I couldn't be bothered.
# ([\d:\-\.]+): Matches idx, which can be any combination of digits, colons, and periods.
# \s*=\s*: Matches the equal sign (=) with optional spaces on both sides.
# ([^,]+): Captures the value, which can be any character except for commas (this ensures that values are correctly separated).
# (?:\s*,\s*|$): Matches either a comma (with optional spaces before or after) or the end of the string ($).
#----------------------------------------------
###############################################


# verify that string only contains a-z, A-Z, 0-9, or _
_regex_key_value = re.compile(r'^[a-zA-Z0-9_]+$')
def verify_key_value(key: str, raise_error=True):
    match = re.match(_regex_key_value, key)
    if not match and raise_error:
        raise Exception(f"Value key may only contain 'a-z', 'A-Z', '0-9', or '_', but was: '{key}'.")
    return match is not None


class SFormat:
    JSON = "json"
    PYTH = "pythonic"

@dataclass
class RegexErrorReport:
    start: int
    end: int
    text: str
    reason: str = None

@dataclass
class InputPair:
    idx: int
    val: Union[int, str, Tensor]
    hold: bool = False
    end: bool = False

@dataclass
class CondHolder:
    idx: int
    prompt: str
    raw_prompt: str
    cond: Tensor
    pooled: Tensor
    hold: bool = False
    interp_weight: float = None
    interp_prompt: str = None

@dataclass
class ParseErrorReport:
    idx_str: str
    val_str: str
    reason: str

@dataclass
class PromptOptions:
    interp: str = TensorInterp.LERP
    prepend_text: str = ''
    append_text: str = ''
    values_replace: dict[str, list[float]] = None
    print_schedule: bool = False
    add_dict: dict[str] = None

IndividualConditioning = tuple[torch.Tensor, dict[str, torch.Tensor]]
Conditioning = list[IndividualConditioning]

def evaluate_prompt_schedule(text: str, length: int, clip: CLIP, options: PromptOptions):
    text = strip_input(text)
    if len(text) == 0:
        raise Exception("No text provided to Prompt Scheduling.")
    # prioritize formats based on best guess to minimize redo's
    if text.startswith('"'):
        formats = [SFormat.JSON, SFormat.PYTH]
    else:
        formats = [SFormat.PYTH, SFormat.JSON]
    for format in formats:
        if format is SFormat.JSON:
            # check JSON format
            # if no errors found, assume this is the right format and pass on to parsing individual values
            json_matches, json_errors = get_matches_and_errors(text, _regex_prompt_json)
            if len(json_errors) == 0:
                return parse_prompt_groups(json_matches, length, clip, options)
        elif format is SFormat.PYTH:
            # check pythonic format
            # if no errors found, assume this is the right format and pass on to parsing individual values
            pyth_matches, pyth_errors = get_matches_and_errors(text, _regex_prompt_pyth)
            if len(pyth_errors) == 0:
                return parse_prompt_groups(pyth_matches, length, clip, options)
    # since both formats have errors, check which format is more 'correct' for the input
    # priority:
    # 1 - most matches
    # 2 - least errors
    if len(json_matches) > len(pyth_matches):
        real_errors = json_errors
        assumed = SFormat.JSON
    elif len(json_matches) < len(pyth_matches):
        real_errors = pyth_errors
        assumed = SFormat.PYTH
    elif len(json_errors) < len(pyth_errors):
        real_errors = json_errors
        assumed = SFormat.JSON
    else:
        logger.warn("same amount of matches+errors for prompt!")
        real_errors = pyth_errors
        assumed = SFormat.PYTH
    # TODO: make separate case for when format is unknown, so that both are displayed to the user
    error_msg_list = []
    if len(real_errors) == 1:
        error_msg_list.append(f"Found 1 issue in prompt schedule (assumed {assumed} format):")
    else:
        error_msg_list.append(f"Found {len(real_errors)} issues in prompt schedule (assumed {assumed} format):")
    for error in real_errors:
        error_msg_list.append(f"Position {error.start} to {error.end}: '{error.text}'")
    error_msg = "\n".join(error_msg_list)
    raise Exception(error_msg)


def parse_prompt_groups(groups: list[tuple], length: int, clip: CLIP, options: PromptOptions):
    pairs: list[InputPair]
    errors: list[ParseErrorReport]
    # turn group tuples into InputPairs
    pairs = [InputPair(x[0], x[1]) for x in groups]
    # perform first parse, to get idea of indexes to handle
    pairs, errors = handle_group_idxs(pairs, length)
    if len(errors) > 0:
        error_msg_list = []
        issues_formatted = f"{len(errors)} issue{'s' if len(errors)> 1 else ''}"
        error_msg_list.append(f"Found {issues_formatted} with idxs:")
        for error in errors:
            error_msg_list.append(f"{error.idx_str}: {error.reason}")
        error_msg = "\n".join(error_msg_list)
        raise Exception(error_msg)
    prepare_prompts(pairs, options)
    final_vals = handle_prompt_interpolation(pairs, length, clip, options)
    return final_vals


def prepare_prompts(pairs: list[InputPair], options: PromptOptions):
    for pair in pairs:
        prepend_text = options.prepend_text.strip()
        append_text = options.append_text.strip()
        prompt = pair.val.strip()
        # when adding prepend and append text, handle commas properly
        # prepend text
        if len(prepend_text) > 0:
            while prepend_text.endswith(','):
                prepend_text = prepend_text[:-1].strip()
            if prompt.startswith(','):
                prepend_text = f"{prepend_text}"
            else:
                prepend_text = f"{prepend_text}, "
            prompt = prepend_text + prompt
        # append text
        if len(append_text) > 0:
            while append_text.startswith(','):
                append_text = append_text[1:].strip()
            if prompt.endswith(','):
                append_text = f" {append_text}"
            else:
                append_text = f", {append_text}"
            prompt = prompt + append_text
        # update value w/ prompt
        pair.val = prompt


def apply_values_replace_to_prompt(prompt: str, idx: int, values_replace: Union[None, dict[str, list[float]]]):
    # if no values to replace, do nothing
    if values_replace is None:
        return prompt
    for key, value in values_replace.items():
        # use FizzNodes `` notation
        match_str = '`' + key + '`'
        value_str = f"{value[idx]}"
        prompt = prompt.replace(match_str, value_str)
    return prompt


def handle_prompt_interpolation(pairs: list[InputPair], length: int, clip: CLIP, options: PromptOptions):
    if length == 0:
        length = max(pairs, key=lambda x: x.idx).idx+1
    # prepare values_replace (should match length)
    values_replace = options.values_replace
    if values_replace is not None:
        values_replace.copy()
        for key, value in values_replace.items():
            if len(value) < length:
                values_replace[key] = extend_list_to_batch_size(value, length)

    scheduled_keyframes = []
    if clip.use_clip_schedule:
        clip = clip.clone()
        scheduled_keyframes = clip.patcher.forced_hooks.get_hooks_for_clip_schedule()

    pairs_lengths = len(pairs) * max(1, len(scheduled_keyframes))
    pbar_total = length + pairs_lengths
    pbar = ProgressBar(pbar_total)
    # for now, use FizzNodes approach of calculating max size of tokens beforehand;
    # this can up to double total encoding time, as this will be done again.
    # TODO: do this dynamically to save encoding time
    max_size = 0
    for pair in pairs:
        prepared_prompt = apply_values_replace_to_prompt(pair.val, 0, values_replace=values_replace)
        cond: Tensor = clip.encode_from_tokens(clip.tokenize(prepared_prompt))
        max_size = max(max_size, cond.shape[1])
        pbar.update(1)

    # if do not need to schedule clip with hooks, do nothing special
    if not clip.use_clip_schedule:
        return _handle_prompt_interpolation(pairs, length, clip, options, values_replace, max_size, pbar)
    # otherwise, need to account for keyframes on forced_hooks
    full_output = []
    for i, scheduled_opts in enumerate(scheduled_keyframes):
        clip.patcher.forced_hooks.reset()
        clip.patcher.unpatch_hooks()

        t_range = scheduled_opts[0]
        hooks_keyframes = scheduled_opts[1]
        for hook, keyframe in hooks_keyframes:
            hook.hook_keyframe._current_keyframe = keyframe
        try:
            # don't print_schedule on non-first iteration
            orig_print_schedule = options.print_schedule
            if orig_print_schedule and i != 0:
                options.print_schedule = False
            schedule_output = _handle_prompt_interpolation(pairs, length, clip, options, values_replace, max_size, pbar)
        finally:
            options.print_schedule = orig_print_schedule
        for cond, pooled_dict in schedule_output:
            pooled_dict: dict[str]
            # add clip_start_percent and clip_end_percent in pooled
            pooled_dict["clip_start_percent"] = t_range[0]
            pooled_dict["clip_end_percent"] = t_range[1]
        full_output.extend(schedule_output)
    return full_output


def _handle_prompt_interpolation(pairs: list[InputPair], length: int, clip: CLIP, options: PromptOptions,
                                 values_replace: dict[str, list[float]], max_size: int, pbar: ProgressBar):
    real_holders: list[CondHolder] = [None] * length
    real_cond = [None] * length
    real_pooled = [None] * length
    prev_holder: Union[CondHolder, None] = None
    for idx, pair in enumerate(pairs):
        holder = None
        is_over_length = False
        # if no last pair is set, then use first provided val up to the idx
        if prev_holder is None:
            for i in range(idx, pair.idx+1):
                if i >= length:
                    is_over_length = True
                    continue
                real_prompt = apply_values_replace_to_prompt(pair.val, i, values_replace=values_replace)
                if holder is None or holder.prompt != real_prompt:
                    cond, pooled = clip.encode_from_tokens(clip.tokenize(real_prompt), return_pooled=True)
                    cond = pad_cond(cond, target_length=max_size)
                    holder = CondHolder(idx=i, prompt=real_prompt, raw_prompt=pair.val, cond=cond, pooled=pooled, hold=pair.hold)
                else:
                    holder = replace(holder)
                    holder.idx = i
                real_cond[i] = cond
                real_pooled[i] = pooled
                real_holders[i] = holder
                pbar.update(1)
                comfy.model_management.throw_exception_if_processing_interrupted()
        # if idx is exactly one greater than the one before, nothing special
        elif prev_holder.idx == pair.idx-1:
            comfy.model_management.throw_exception_if_processing_interrupted()
            holder = prev_holder
            if pair.idx < length:
                real_prompt = apply_values_replace_to_prompt(pair.val, pair.idx, values_replace=values_replace)
                cond, pooled = clip.encode_from_tokens(clip.tokenize(real_prompt), return_pooled=True)
                cond = pad_cond(cond, target_length=max_size)
                holder = CondHolder(idx=pair.idx, prompt=real_prompt, raw_prompt=pair.val, cond=cond, pooled=pooled, hold=pair.hold)
                real_cond[pair.idx] = cond
                real_pooled[pair.idx] = pooled
                real_holders[pair.idx] = holder
                pbar.update(1)
        else:
            # if holding value, no interpolation
            if prev_holder.hold:
                # keep same value as last_holder, then calculate current index cond;
                # however, need to check if real_prompt remains the same
                for i in range(prev_holder.idx+1, pair.idx):
                    if i >= length:
                        is_over_length = True
                        continue
                    if holder is None:
                        holder = prev_holder
                    real_prompt = apply_values_replace_to_prompt(holder.raw_prompt, i, values_replace=values_replace)
                    if holder.prompt != real_prompt:
                        cond, pooled = clip.encode_from_tokens(clip.tokenize(real_prompt), return_pooled=True)
                        cond = pad_cond(cond, target_length=max_size)
                        holder = replace(holder, idx=i, prompt=real_prompt, cond=cond, pooled=pooled)
                    else:
                        holder = replace(holder)
                        holder.idx = i
                    real_cond[i] = holder.cond
                    real_pooled[i] = holder.pooled
                    real_holders[i] = holder
                    pbar.update(1)
                    comfy.model_management.throw_exception_if_processing_interrupted()
                if pair.idx < length:
                    real_prompt = apply_values_replace_to_prompt(pair.val, pair.idx, values_replace=values_replace)
                    cond, pooled = clip.encode_from_tokens(clip.tokenize(real_prompt), return_pooled=True)
                    cond = pad_cond(cond, target_length=max_size)
                    holder = CondHolder(idx=pair.idx, prompt=real_prompt, raw_prompt=pair.val, cond=cond, pooled=pooled, hold=pair.hold)
                    real_cond[pair.idx] = cond
                    real_pooled[pair.idx] = pooled
                    real_holders[pair.idx] = holder
                    pbar.update(1)
                    comfy.model_management.throw_exception_if_processing_interrupted()
            # otherwise, interpolate
            else:
                diff_len = abs(pair.idx-prev_holder.idx)+1
                interp_idxs = InterpolationMethod.get_weights(num_from=prev_holder.idx, num_to=pair.idx, length=diff_len,
                                                              method=InterpolationMethod.LINEAR)
                interp_weights = InterpolationMethod.get_weights(num_from=0.0, num_to=1.0, length=diff_len,
                                                              method=InterpolationMethod.LINEAR)
                cond_to = None
                pooled_to = None
                cond_from = None
                holder = None
                interm_holder = prev_holder
                for raw_idx, weight in zip(interp_idxs, interp_weights):
                    if raw_idx >= length:
                        is_over_length = True
                        continue
                    idx_int = round(float(raw_idx))
                    # calculate cond_to stuff if not done yet
                    real_prompt = apply_values_replace_to_prompt(pair.val, idx_int, values_replace=values_replace)
                    if holder is None or holder.prompt != real_prompt:
                        cond_to, pooled_to = clip.encode_from_tokens(clip.tokenize(real_prompt), return_pooled=True)
                        cond_to = pad_cond(cond_to, target_length=max_size)
                        holder = CondHolder(idx=pair.idx, prompt=real_prompt, raw_prompt=pair.val, cond=cond_to, pooled=pooled_to, hold=pair.hold)
                    # calculate interm_holder stuff if needed
                    real_prompt = apply_values_replace_to_prompt(interm_holder.raw_prompt, idx_int, values_replace=values_replace)
                    if interm_holder.prompt != real_prompt:
                        cond_from, pooled_from = clip.encode_from_tokens(clip.tokenize(real_prompt), return_pooled=True)
                        cond_from = pad_cond(cond_from, target_length=max_size)
                        interm_holder = CondHolder(idx=idx_int, prompt=real_prompt, raw_prompt=interm_holder.raw_prompt, cond=cond_from, pooled=pooled_from, hold=holder.hold)
                    else:
                        interm_holder = CondHolder(idx=interm_holder.idx, prompt=interm_holder.prompt, raw_prompt=interm_holder.raw_prompt, cond=interm_holder.cond, pooled=interm_holder.pooled, hold=interm_holder.hold)
                    # interpolate conds
                    if options.interp == TensorInterp.LERP:
                        cond_interp = lerp_tensors(tensor_from=interm_holder.cond, tensor_to=cond_to, strength_to=weight)
                    elif options.interp == TensorInterp.SLERP:
                        cond_interp = slerp_tensors(tensor_from=interm_holder.cond, tensor_to=cond_to, strength_to=weight)
                    pooled_interp = pooled_to
                    if math.isclose(weight, 0.0):
                        pooled_interp = interm_holder.pooled
                    interm_holder = CondHolder(idx=idx_int, prompt=interm_holder.prompt, raw_prompt=interm_holder.raw_prompt, cond=cond_interp, pooled=pooled_interp, hold=holder.hold,
                                               interp_weight=weight, interp_prompt=holder.prompt)
                    real_cond[idx_int] = cond_interp
                    real_pooled[idx_int] = pooled_interp
                    real_holders[idx_int] = interm_holder
                    pbar.update(1)
                    comfy.model_management.throw_exception_if_processing_interrupted()
        if is_over_length:
            break
        assert holder is not None
        prev_holder = holder

    # fill in None gaps with last used values
    # TODO: review if this works as intended, or if needs to be a bit more thorough
    prev_holder = None
    for i in range(len(real_holders)):
        if real_holders[i] is None:
            # check if any value replacement needs to be accounted for
            real_prompt = apply_values_replace_to_prompt(prev_holder.raw_prompt, i, values_replace=values_replace)
            if prev_holder.prompt != real_prompt:
                cond, pooled = clip.encode_from_tokens(clip.tokenize(real_prompt), return_pooled=True)
                cond = pad_cond(cond, target_length=max_size)
                prev_holder = CondHolder(idx=i, prompt=real_prompt, raw_prompt=prev_holder.raw_prompt, cond=cond, pooled=pooled, hold=prev_holder.hold)
            real_cond[i] = prev_holder.cond
            real_pooled[i] = prev_holder.pooled
            real_holders[i] = prev_holder
            pbar.update(1)
        else:
            prev_holder = real_holders[i]
    
    final_cond = torch.cat(real_cond, dim=0)
    final_pooled = torch.cat(real_pooled, dim=0)

    if options.print_schedule:
        logger.info(f"PromptScheduling ({len(real_holders)} prompts)")
        for i, holder in enumerate(real_holders):
            if holder.interp_prompt is None:
                logger.info(f'{i} = "{holder.prompt}"')
            else:
                logger.info(f'{i} = ({1.-holder.interp_weight:.2f})"{holder.prompt}" -> ({holder.interp_weight:.2f})"{holder.interp_prompt}"')
    # cond is a list[list[Tensor, dict[str: Any]]] format
    final_pooled_dict = {"pooled_output": final_pooled}
    if options.add_dict is not None:
        final_pooled_dict.update(options.add_dict)
    # add hooks, if needed
    clip.add_hooks_to_dict(final_pooled_dict)
    return [[final_cond, final_pooled_dict]]

def extract_cond_from_schedule(conditioning: Conditioning, index: int) -> Conditioning:
    return [_extract_single_cond(t, index) for t in conditioning]

def _extract_single_cond(single_cond: IndividualConditioning, index:int) -> IndividualConditioning:
    if index < 0:
        return single_cond

    cond, kwargs = single_cond[0], single_cond[1].copy()
    original_pooled = kwargs["pooled_output"]

    cond_schedules =  cond.shape[0]
    pooled_schedules = original_pooled.shape[0]

    if cond_schedules <= index or pooled_schedules <= index:
        logger.warning(f"Trying to get index {index}, only have {cond_schedules} items")
        return single_cond

    cond_chunks = cond.chunk(cond_schedules)
    chosen_cond = cond_chunks[index]

    pool_chunks = original_pooled.chunk(pooled_schedules)
    kwargs["pooled_output"] = pool_chunks[index]
    return [chosen_cond, kwargs]

def pad_cond(cond: Tensor, target_length: int):
    # FizzNodes-style cond padding
    # TODO: test out other methods of padding
    curr_length = cond.shape[1]
    if curr_length < target_length:
        pad_length = target_length - curr_length
        # FizzNodes pads the tensor on both ends
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad
        # perform padding
        cond = F.pad(cond, (0, 0, left_pad, right_pad))
    return cond


def evaluate_value_schedule(text: str, length: int):
    text = strip_input(text)
    if len(text) == 0:
        raise Exception("No text provided to Value Scheduling.")
    # prioritize formats based on best guess to minimize redo's
    if text.startswith('"'):
        formats = [SFormat.JSON, SFormat.PYTH]
    else:
        formats = [SFormat.PYTH, SFormat.JSON]
    for format in formats:
        if format is SFormat.JSON:
            # check JSON format
            # if no errors found, assume this is the right format and pass on to parsing individual values
            json_matches, json_errors = get_matches_and_errors(text, _regex_value_json)
            if len(json_errors) == 0:
                return parse_value_groups(json_matches, length)
        elif format is SFormat.PYTH:
            # check pythonic format
            # if no errors found, assume this is the right format and pass on to parsing individual values
            pyth_matches, pyth_errors = get_matches_and_errors(text, _regex_value_pyth)
            if len(pyth_errors) == 0:
                return parse_value_groups(pyth_matches, length)
    # since both formats have errors, check which format is more 'correct' for the input
    # priority:
    # 1 - most matches
    # 2 - least errors
    if len(json_matches) > len(pyth_matches):
        real_errors = json_errors
        assumed = SFormat.JSON
    elif len(json_matches) < len(pyth_matches):
        real_errors = pyth_errors
        assumed = SFormat.PYTH
    elif len(json_errors) < len(pyth_errors):
        real_errors = json_errors
        assumed = SFormat.JSON
    else:
        #logger.info("same amount of matches+errors for value!")
        if text.startswith('"'):
            real_errors = json_errors
            assumed = SFormat.JSON
        else:
            real_errors = pyth_errors
            assumed = SFormat.PYTH
    # TODO: make separate case for when format is unknown, so that both are displayed to the user
    error_msg_list = []
    if len(real_errors) == 1:
        error_msg_list.append(f"Found 1 issue in value schedule (assumed {assumed} format):")
    else:
        error_msg_list.append(f"Found {len(real_errors)} issues in value schedule (assumed {assumed} format):")
    for error in real_errors:
        error_msg_list.append(f"Position {error.start} to {error.end}: '{error.text}'")
    error_msg = "\n".join(error_msg_list)
    raise Exception(error_msg)


def parse_value_groups(groups: list[tuple], length: int):
    #logger.info(groups)
    pairs: list[InputPair]
    errors: list[ParseErrorReport]
    # perform first parse, where we convert vals to floats
    pairs, errors = handle_float_vals(groups)
    if len(errors) == 0:
        # perform second parse, to get idea of indexes to handle
        pairs, errors = handle_group_idxs(pairs, length)
        if len(pairs) == 0:
            errors.append(ParseErrorReport(idx_str='No valid idxs provided', val_str='', reason='Provided ranges might not be selecting anything.'))
    if len(errors) > 0:
        error_msg_list = []
        issues_formatted = f"{len(errors)} issue{'s' if len(errors)> 1 else ''}"
        error_msg_list.append(f"Found {issues_formatted} with idxs/vals:")
        for error in errors:
            error_msg_list.append(f"{error.idx_str}: {error.reason}")
        error_msg = "\n".join(error_msg_list)
        raise Exception(error_msg)
    # perform third parse, where hold and interpolation is used to fill in any in-between values
    final_vals = handle_val_interpolation(pairs, length)
    return final_vals


def handle_float_vals(groups: list[tuple]):
    actual_pairs: list[InputPair] = []
    errors: list[ParseErrorReport] = []
    for idx_str, val_str in groups:
        val_str = strip_value(val_str)
        try:
            val = float(val_str)
        except ValueError:
            errors.append(ParseErrorReport(idx_str, val_str, f"Value '{val_str}' is not a valid number"))
            continue
        actual_pairs.append(InputPair(idx_str, val))
    return actual_pairs, errors


def handle_val_interpolation(pairs: list[InputPair], length: int):
    if length == 0:
        length = max(pairs, key=lambda x: x.idx).idx+1
    real_vals = [None] * length

    prev_pair = None
    for pair in pairs:
        # if no last pair is set, then use first provided val up to the idx
        if prev_pair is None:
            for i in range(0, pair.idx+1):
                if i >= length:
                    break
                real_vals[i] = pair.val
        # if idx is exactly one greater than the one before, nothing special
        elif prev_pair.idx == pair.idx-1:
            if pair.idx < length:
                real_vals[pair.idx] = pair.val
        else:
            # if holding value, no interpolation
            if prev_pair.hold:
                # keep same value as last_pair, then assign current index value
                for i in range(prev_pair.idx+1, pair.idx):
                    if i >= length:
                        continue
                    real_vals[i] = prev_pair.val
                if pair.idx < length:
                    real_vals[pair.idx] = pair.val
            # otherwise, interpolate
            else:
                diff_len = abs(pair.idx-prev_pair.idx)+1
                interp_idxs = InterpolationMethod.get_weights(num_from=prev_pair.idx, num_to=pair.idx, length=diff_len,
                                                              method=InterpolationMethod.LINEAR)
                interp_vals = InterpolationMethod.get_weights(num_from=prev_pair.val, num_to=pair.val, length=diff_len,
                                                              method=InterpolationMethod.LINEAR)
                for idx, val in zip(interp_idxs, interp_vals):
                    if idx >= length:
                        continue
                    real_vals[round(float(idx))] = float(val)
        prev_pair = pair
    # fill in None gaps with last used value
    # TODO: review if this works as intended, or if needs to be a bit more thorough
    prev_val = None
    for i in range(len(real_vals)):
        if real_vals[i] is None:
            real_vals[i] = prev_val
        else:
            prev_val = real_vals[i]
    return real_vals


def handle_group_idxs(pairs: list[InputPair], length: int):
    actual_pairs: list[InputPair] = []
    errors: list[ParseErrorReport] = []
    for pair in pairs:
        idx_str, val_str = pair.idx, pair.val
        idx_str: str = idx_str.strip()
        hold = False
        # if starts with :, wrong
        if idx_str.startswith(':'):
            errors.append(ParseErrorReport(idx_str, val_str, "Idx can't begin with ':'"))
            continue
        # if has more than one :, wrong
        if idx_str.count(':') > 1:
            errors.append(ParseErrorReport(idx_str, val_str, "Idx can't have more than one ':'"))
        if idx_str.endswith(':'):
            hold = True
            idx_str = idx_str[:-1]
        try:
            idxs = convert_str_to_indexes(idx_str, length, allow_range=True, allow_missing=True, fix_reverse=True, same_is_one=True, allow_decimal=True)
        except SelectError as e:
            errors.append(ParseErrorReport(idx_str, val_str, f"Couldn't convert idxs; {str(e)}"))
            continue
        for idx in idxs:
            actual_pairs.append(InputPair(idx, val_str, hold))
    return actual_pairs, errors


def get_matches_and_errors(text: str, pattern: re.Pattern) -> tuple[list, list[RegexErrorReport]]:
    last_match_end = 0
    matches = []
    errors: list[RegexErrorReport] = []

    for match in re.finditer(pattern, text):
        start, end = match.span()
        # if there is any text between last match and current, consider as error
        if start != last_match_end:
            errors.append(RegexErrorReport(last_match_end, start, text[last_match_end:start].replace('\n','\t')))
        # update match
        last_match_end = end
        # store match
        matches.append(match.groups())
    
    # check for any trailing unmatched text
    if last_match_end != len(text):
        errors.append(RegexErrorReport(last_match_end, len(text), text[last_match_end:].replace('\n','\t')))
    
    return matches, errors


def is_surrounded(text: str, pair):
    return text.startswith(pair[0]) and text.endswith(pair[1])


def is_surrounded_pairs(text: str, pairs):
    for pair in pairs:
        if is_surrounded(text, pair):
            return True
    return False


def strip_value(text: str, limit=-1):
    text = text.strip()
    # strip common paired symbols
    symbol_pairs = [
        ("(", ")"),
        ("[", "]"),
        ("{", "}"),
        ('"', '"'),
        ("'", "'"),
    ]
    while limit != 0 and is_surrounded_pairs(text, symbol_pairs):
        text = text[1:-1].strip()
        limit -= 1
    return text


def strip_input(text: str):
    text = text.strip()
    # strip JSON brackets, if needed
    if text.startswith('{') and text.endswith('}'):
        return text[1:-1].strip()
    return text
