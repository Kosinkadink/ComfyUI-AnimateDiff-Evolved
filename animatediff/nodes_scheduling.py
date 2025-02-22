from typing import Union

from .documentation import register_description, short_desc, coll, DocHelper
from .scheduling import (evaluate_prompt_schedule, evaluate_value_schedule, extract_cond_from_schedule, TensorInterp, PromptOptions,
                         verify_key_value)
from .utils_model import BIGMAX
from .logger import logger


desc_values = {coll('values'): 'Write your values here.'}
desc_prompts = {coll('prompts'): 'Write your prompts here.'}
desc_clip = {'clip': 'CLIP to use for encoding prompts.'}
desc_latent = {'latent': 'Used to get the amount of frames (max_length) to use for scheduling.'}

desc_prepend_text = {'prepend_text': 'OPTIONAL, adds text before all prompts.'}
desc_append_text = {'append_text': 'OPTIONAL, adds text after all prompts.'}
desc_values_replace = {'values_replace': 'OPTIONAL, replaces keys from value_replace keys with provided value schedules. Keys in the prompt are written as `some_key`, surrounded by the ` characters.'}
desc_tensor_interp = {'tensor_interp': 'Selects method of interpolating prompt conds - defaults to lerp.'}
desc_print_schedule = {'print_schedule': 'When True, prints output values for each frame.'}

desc_max_length = {'max_length': 'Used to select the intended length of schedule. If set to 0, will use the largest index in the schedule as max_length, but will disable relative indexes (negative and decimal).'}
desc_floats = {'floats': 'List of floats, likely outputted by a Value Scheduling node.'}
desc_FLOAT = {'FLOAT': 'Float (or list of floats) to convert to FLOATS type.'}
desc_value_key = {'value_key': 'Key to use for value schedule in Prompt Scheduling node. Can only contain a-z, A-Z, 0-9, and _ characters. In Prompt Scheduling, keys can be referred to as `some_key`, where the key is surrounded by ` characters.'}
desc_prev_replace = {'prev_replace': 'OPTIONAL, other values_replace can be chained.'}

desc_input_conditioning = {'conditioning': 'Encoded prompts. The output of a Prompt Scheduling node.'}
desc_index = {'index': 'The index to extract. Must be within the range [0,N] where N is the length of scheduled prompts.'}
desc_output_conditioning_single = {'CONDITIONING': 'The single step conditioning from the schedule.'}

desc_output_conditioning = {'CONDITIONING': 'Encoded prompts.'}
desc_output_latent = {'LATENT': 'Unmodified input latents; can be used as pipe, or can be ignored.'}

desc_format_allowed_idxs = {'allowed idxs':
        {'single': 'A positive integer (e.g. 0, 2) schedules value for frame. A negative integer (e.g. -1, -5) schedules value for frame from the end (-1 would be the last frame). ' + 
            'A decimal (e.g. 0.5, 1.0) selects frame based relative location in whole schedule (0.5 would be halfway, 1.0 would be last frame).',
        'range': 'Using rules above, single:single chooses uninterpolated prompts from start idx (included) to end idx (excluded). Examples -> 0:12, 0:-5, 2:0.5',
        'hold': 'Putting a colon after a single idx stops interpolation until the next provided index. Examples -> 0:, 0.5:, 16: '}
    }

desc_format_prompt = [
    'Scheduling supports two formats: JSON and pythonic.',
    {'JSON': ['"idx": "your prompt here", ...'],
     'pythonic': ['idx = "your prompt here", ...']},
    'The idx is the index of the frame - first frame is 0, last frame is max_frames-1. An idx may be the following:',
    desc_format_allowed_idxs,
    'The prompts themselves should be surrounded by double quotes ("your prompt here"). Portions of prompts can use value schedules provided values_replace.',
    {'JSON': ['"0": "blue rock on mountain",', '"16": "green rock in lake"'],
     'pythonic': ['0 = "blue rock on mountain",', '16 = "green rock in lake"']}
]

desc_format_values = [
    'Scheduling supports two formats: JSON and pythonic.',
    {'JSON': ['"idx": float/int_value, ...'],
     'pythonic': ['idx = float/int_value, ...']},
    'The idx is the index of the frame - first frame is 0, last frame is max_frames-1. An idx may be the following:',
    desc_format_allowed_idxs,
    'The values can be written without any special formatting.',
    {'JSON': ['"0": 1.0,', '"16": 1.3'],
     'pythonic': ['0 = 1.0,', '16 = 1.3']}
]


class PromptSchedulingLatentsNode:
    NodeID = 'ADE_PromptSchedulingLatents'
    NodeName = 'Prompt Scheduling [Latents] 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompts": ("STRING", {"multiline": True, "default": ''}),
                "clip": ("CLIP",),
                "latent": ("LATENT",),
            },
            "optional": {
                "prepend_text": ("STRING", {"multiline": True, "default": '', "forceInput": True}),
                "append_text": ("STRING", {"multiline": True, "default": '', "forceInput": True}),
                "values_replace": ("VALUES_REPLACE",),
                "print_schedule": ("BOOLEAN", {"default": False}),
                "tensor_interp": (TensorInterp._LIST,)
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT",)
    CATEGORY = "Animate Diff 🎭🅐🅓/scheduling"
    FUNCTION = "create_schedule"

    Desc = [
        short_desc('Encode a schedule of prompts with automatic interpolation, its length matching passed-in latent count.'),
        {'Format': desc_format_prompt},
        {coll('Inputs'): DocHelper.combine(desc_prompts, desc_clip, desc_latent, desc_values_replace, desc_prepend_text, desc_append_text, desc_tensor_interp, desc_print_schedule)},
        {coll('Outputs'): DocHelper.combine(desc_output_conditioning, desc_output_latent)}
    ]
    register_description(NodeID, Desc)

    def create_schedule(self, prompts: str, clip, latent: dict, print_schedule=False, tensor_interp=TensorInterp.LERP,
                        prepend_text='', append_text='', values_replace=None):
        options = PromptOptions(interp=tensor_interp, prepend_text=prepend_text, append_text=append_text,
                                values_replace=values_replace, print_schedule=print_schedule)
        conditioning = evaluate_prompt_schedule(prompts, latent["samples"].size(0), clip, options)
        return (conditioning, latent)


class PromptSchedulingNode:
    NodeID = 'ADE_PromptScheduling'
    NodeName = 'Prompt Scheduling 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompts": ("STRING", {"multiline": True, "default": ''}),
                "clip": ("CLIP",),
            },
            "optional": {
                "prepend_text": ("STRING", {"multiline": True, "default": '', "forceInput": True}),
                "append_text": ("STRING", {"multiline": True, "default": '', "forceInput": True}),
                "values_replace": ("VALUES_REPLACE",),
                "print_schedule": ("BOOLEAN", {"default": False}),
                "max_length": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "tensor_interp": (TensorInterp._LIST,)
            },
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "Animate Diff 🎭🅐🅓/scheduling"
    FUNCTION = "create_schedule"

    Desc = [
        short_desc('Encode a schedule of prompts with automatic interpolation.'),
        {'Format': desc_format_prompt},
        {coll('Inputs'): DocHelper.combine(desc_prompts, desc_clip, desc_values_replace, desc_prepend_text, desc_append_text, desc_max_length, desc_tensor_interp, desc_print_schedule)},
        {coll('Outputs'): DocHelper.combine(desc_output_conditioning)}
    ]
    register_description(NodeID, Desc)

    def create_schedule(self, prompts: str, clip, print_schedule=False, max_length: int=0, tensor_interp=TensorInterp.LERP,
                        prepend_text='', append_text='', values_replace=None):
        options = PromptOptions(interp=tensor_interp, prepend_text=prepend_text, append_text=append_text,
                                values_replace=values_replace, print_schedule=print_schedule)
        conditioning = evaluate_prompt_schedule(prompts, max_length, clip, options)
        return (conditioning,)


class ValueSchedulingLatentsNode:
    NodeID = 'ADE_ValueSchedulingLatents'
    NodeName = 'Value Scheduling [Latents] 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "values": ("STRING", {"multiline": True, "default": ""}),
                "latent": ("LATENT",),
            },
            "optional": {
                "print_schedule": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "FLOATS", "INT", "INTS")
    CATEGORY = "Animate Diff 🎭🅐🅓/scheduling"
    FUNCTION = "create_schedule"

    Desc = [
        short_desc('Create a list of values with automatic interpolation, its length matching passed-in latent count.'),
        {'Format': desc_format_values},
        {coll('Inputs'): DocHelper.combine(desc_values, desc_latent, desc_print_schedule)},
    ]
    register_description(NodeID, Desc)

    def create_schedule(self, values: str, latent: dict, print_schedule=False):
        float_vals = evaluate_value_schedule(values, latent["samples"].size(0))
        int_vals = [round(x) for x in float_vals]
        if print_schedule:
            logger.info(f"ValueScheduling ({len(float_vals)} values):")
            for i, val in enumerate(float_vals):
                logger.info(f"{i} = {val}")
        return (float_vals, float_vals, int_vals, int_vals)


class ValueSchedulingNode:
    NodeID = 'ADE_ValueScheduling'
    NodeName = 'Value Scheduling 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "values": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "print_schedule": ("BOOLEAN", {"default": False}),
                "max_length": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOATS", "INT", "INTS")
    CATEGORY = "Animate Diff 🎭🅐🅓/scheduling"
    FUNCTION = "create_schedule"

    Desc = [
        short_desc('Create a list of values with automatic interpolation.'),
        {'Format': desc_format_values},
        {coll('Inputs'): DocHelper.combine(desc_values, desc_max_length, desc_print_schedule)},
    ]
    register_description(NodeID, Desc)

    def create_schedule(self, values: str, max_length: int, print_schedule=False):
        float_vals = evaluate_value_schedule(values, max_length)
        int_vals = [round(x) for x in float_vals]
        if print_schedule:
            logger.info(f"ValueScheduling ({len(float_vals)} values):")
            for i, val in enumerate(float_vals):
                logger.info(f"{i} = {val}")
        return (float_vals, float_vals, int_vals, int_vals)


class AddValuesReplaceNode:
    NodeID = 'ADE_ValuesReplace'
    NodeName = 'Add Values Replace 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value_key": ("STRING", {"default": ""}),
                "floats": ("FLOATS",)
            },
            "optional": {
                "prev_replace": ("VALUES_REPLACE",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("VALUES_REPLACE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/scheduling"
    FUNCTION = "add_values_replace"

    Desc = [
        short_desc('Add a values schedule bound to a key to be used in Prompt Scheduling node.'),
        {'Inputs': DocHelper.combine(desc_value_key, desc_floats, desc_prev_replace)},
    ]
    register_description(NodeID, Desc)

    def add_values_replace(self, value_key: str, floats: Union[list[float]], prev_replace: dict=None):
        # key can only have a-z, A-Z, 0-9, and _ characters
        verify_key_value(key=value_key)
        # add/replace value floats
        if prev_replace is None:
            prev_replace = {}
        prev_replace = prev_replace.copy()
        if value_key in prev_replace:
            logger.warn(f"Value key '{value_key}' is already present - corresponding floats value will be overriden.")
        prev_replace[value_key] = floats
        return (prev_replace,)


class FloatToFloatsNode:
    NodeID = 'ADE_FloatToFloats'
    NodeName = 'Float to Floats 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "FLOAT": ("FLOAT", {"default": 39, "forceInput": True}),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("FLOATS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/scheduling"
    FUNCTION = "convert_to_floats"

    def convert_to_floats(self, FLOAT: Union[float, list[float]]):
        floats = None
        if isinstance(FLOAT, float):
            floats = [float(FLOAT)]
        else:
            floats = list(FLOAT)
        return (floats,)

class ConditionExtractionNode:
    NodeID = 'ADE_ConditionExtraction'
    NodeName = 'Condition Step Extraction 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "index": ("INT", {"default": 0, "min": 0, "step": 1})
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "Animate Diff 🎭🅐🅓/scheduling"
    FUNCTION = "extract_conditioning"

    Desc = [
        short_desc('Extract a single conditioning step from a schedule of prompts.'),
        {coll('Inputs'): DocHelper.combine(desc_input_conditioning, desc_index)},
        {coll('Outputs'): DocHelper.combine(desc_output_conditioning)}
    ]
    register_description(NodeID, Desc)

    def extract_conditioning(self, conditioning, index: int=0):
        conditioning_step = extract_cond_from_schedule(conditioning, index)
        return (conditioning_step,)
