import re
import math
from typing import Union
from collections import namedtuple
from dataclasses import dataclass

from .utils_model import InterpolationMethod
from .utils_scheduling import SelectError, convert_str_to_indexes
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


class SFormat:
    JSON = "json"
    PYTH = "pythonic"

@dataclass
class RegexErrorReport:
    start: int
    end: int
    text: str
    reason: str = None


def evaluate_prompt_schedule(text: str, length: int):
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
                return parse_prompt_groups(json_matches, length)
        elif format is SFormat.PYTH:
            # check pythonic format
            # if no errors found, assume this is the right format and pass on to parsing individual values
            pyth_matches, pyth_errors = get_matches_and_errors(text, _regex_prompt_pyth)
            if len(pyth_errors) == 0:
                return parse_prompt_groups(pyth_matches, length)
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


def parse_prompt_groups(groups: tuple, length: int):
    pairs: list[InputPair]
    errors: list[ParseErrorReport]
    # perform first parse, to get idea of indexes to handle
    pairs, errors = handle_group_idxs(groups, length)
    if len(errors) == 0:
        # do next step
        raise Exception("Looks good.")
    if len(errors) > 0:
        error_msg_list = []
        issues_formatted = f"{len(errors)} issue{'s' if len(errors)> 1 else ''}"
        error_msg_list.append(f"Found {issues_formatted} with idxs:")
        for error in errors:
            error_msg_list.append(f"{error.idx_str}: {error.reason}")
        error_msg = "\n".join(error_msg_list)
        raise Exception(error_msg)
    final_vals = []
    return final_vals


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


def parse_value_groups(groups: tuple, length: int):
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
        try:
            val = float(val_str)
        except ValueError:
            errors.append(ParseErrorReport(idx_str, val_str, f"Value '{val_str}' is not a valid number"))
            continue
        actual_pairs.append(InputPair(idx_str, val))
    return actual_pairs, errors


@dataclass
class InputPair:
    idx: int
    val: Union[int, str]
    hold: bool = False
    end: bool = False

@dataclass
class ParseErrorReport:
    idx_str: str
    val_str: str
    reason: str


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


def handle_val_interpolation(pairs: list[InputPair], length: int):
    if length == 0:
        length = max(pairs, key=lambda x: x.idx).idx+1
    real_vals = [None] * length

    last_pair = None
    for pair in pairs:
        # if no last pair is set, then use first provided val up to the idx
        if last_pair is None:
            for i in range(0, pair.idx+1):
                if i >= length:
                    break
                real_vals[i] = pair.val
        # if idx is exactly one greater than the one before, nothing special
        elif last_pair.idx == pair.idx-1:
            if pair.idx < length:
                real_vals[pair.idx] = pair.val
        else:
            # if holding value, no interpolation
            if last_pair.hold:
                # keep same value as last_pair, then assign current index value
                for i in range(last_pair.idx+1, pair.idx):
                    if i >= length:
                        continue
                    real_vals[i] = last_pair.val
                if pair.idx < length:
                    real_vals[pair.idx] = pair.val
            # otherwise, interpolate
            else:
                diff_len = abs(pair.idx-last_pair.idx)+1
                interp_idxs = InterpolationMethod.get_weights(num_from=last_pair.idx, num_to=pair.idx, length=diff_len,
                                                              method=InterpolationMethod.LINEAR)
                interp_vals = InterpolationMethod.get_weights(num_from=last_pair.val, num_to=pair.val, length=diff_len,
                                                              method=InterpolationMethod.LINEAR)
                for idx, val in zip(interp_idxs, interp_vals):
                    if idx >= length:
                        continue
                    real_vals[round(float(idx))] = float(val)
        last_pair = pair
    # fill in None gaps with last used value
    # TODO: review if this works as intended, or if needs to be a bit more thorough
    last_val = None
    for i in range(len(real_vals)):
        if real_vals[i] is None:
            real_vals[i] = last_val
        else:
            last_val = real_vals[i]
    return real_vals


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


def strip_input(text: str):
    text = text.strip()
    # strip JSON brackets, if needed
    if text.startswith('{') and text.endswith('}'):
        return text[1:-1].strip()
    return text
