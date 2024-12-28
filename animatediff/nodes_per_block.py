from typing import Union
from torch import Tensor

from .documentation import short_desc, register_description, coll, DocHelper
from .motion_module_ad import BlockType
from .utils_model import ModelTypeSD
from .utils_motion import AllPerBlocks, PerBlock, PerBlockId, extend_list_to_batch_size


class ADBlockHolder:
    def __init__(self, effect: Union[float, Tensor, None]=None,
                 scales: Union[list[float, Tensor], None]=list()):
        self.effect = effect
        self.scales = scales

    def has_effect(self):
        return self.effect is not None

    def has_scale(self):
        for scale in self.scales:
            if scale is not None:
                return True
        return False

    def is_empty(self):
        has_anything = self.has_effect() or self.has_scale()
        return not has_anything


class ADBlockComboNode:
    NodeID = 'ADE_ADBlockCombo'
    NodeName = 'AD Block 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "effect": ("MULTIVAL",),
                "scale": ("MULTIVAL",),
            },
            "hidden": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("AD_BLOCK",)
    CATEGORY = "Animate Diff 🎭🅐🅓/per block"
    FUNCTION = "block_control"

    def block_control(self, effect: Union[float, Tensor, None]=None, scale: Union[float, Tensor, None]=None):
        scales = [scale, scale]
        block = ADBlockHolder(effect=effect, scales=scales)
        if block.is_empty():
            block = None
        return (block,)


class ADBlockIndivNode:
    NodeID = 'ADE_ADBlockIndiv'
    NodeName = 'AD Block+ 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "effect": ("MULTIVAL",),
                "scale_0": ("MULTIVAL",),
                "scale_1": ("MULTIVAL",),
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("AD_BLOCK",)
    CATEGORY = "Animate Diff 🎭🅐🅓/per block"
    FUNCTION = "block_control"

    def block_control(self, effect: Union[float, Tensor, None]=None,
                      scale_0: Union[float, Tensor, None]=None, scale_1: Union[float, Tensor, None]=None):
        scales = [scale_0, scale_1]
        block = ADBlockHolder(effect=effect, scales=scales)
        if block.is_empty():
            block = None
        return (block,)


class PerBlockHighLevelNode:
    NodeID = 'ADE_PerBlockHighLevel'
    NodeName = 'AD Per Block 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "down": ("AD_BLOCK",),
                "mid": ("AD_BLOCK",),
                "up": ("AD_BLOCK",),
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("PER_BLOCK",)
    CATEGORY = "Animate Diff 🎭🅐🅓/per block"
    FUNCTION = "create_per_block"

    def create_per_block(self,
                         down: Union[ADBlockHolder, None]=None,
                         mid: Union[ADBlockHolder, None]=None,
                         up: Union[ADBlockHolder, None]=None):
        blocks = []
        d = {
            PerBlockId(block_type=BlockType.DOWN): down,
            PerBlockId(block_type=BlockType.MID): mid,
            PerBlockId(block_type=BlockType.UP): up,
        }
        for id, block in d.items():
            if block is not None:
                blocks.append(PerBlock(id=id, effect=block.effect, scales=block.scales))
        if len(blocks) == 0:
            blocks = None
        return (AllPerBlocks(blocks),)


class PerBlock_SD15_MidLevelNode:
    NodeID = 'ADE_PerBlock_SD15_MidLevel'
    NodeName = 'AD Per Block+ (SD1.5) 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "down_0": ("AD_BLOCK",),
                "down_1": ("AD_BLOCK",),
                "down_2": ("AD_BLOCK",),
                "down_3": ("AD_BLOCK",),
                "mid": ("AD_BLOCK",),
                "up_0": ("AD_BLOCK",),
                "up_1": ("AD_BLOCK",),
                "up_2": ("AD_BLOCK",),
                "up_3": ("AD_BLOCK",),
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("PER_BLOCK",)
    CATEGORY = "Animate Diff 🎭🅐🅓/per block"
    FUNCTION = "create_per_block"

    def create_per_block(self,
                         down_0: Union[ADBlockHolder, None]=None,
                         down_1: Union[ADBlockHolder, None]=None,
                         down_2: Union[ADBlockHolder, None]=None,
                         down_3: Union[ADBlockHolder, None]=None,
                         mid: Union[ADBlockHolder, None]=None,
                         up_0: Union[ADBlockHolder, None]=None,
                         up_1: Union[ADBlockHolder, None]=None,
                         up_2: Union[ADBlockHolder, None]=None,
                         up_3: Union[ADBlockHolder, None]=None):
        blocks = []
        d = {
            PerBlockId(block_type=BlockType.DOWN, block_idx=0): down_0,
            PerBlockId(block_type=BlockType.DOWN, block_idx=1): down_1,
            PerBlockId(block_type=BlockType.DOWN, block_idx=2): down_2,
            PerBlockId(block_type=BlockType.DOWN, block_idx=3): down_3,
            PerBlockId(block_type=BlockType.MID): mid,
            PerBlockId(block_type=BlockType.UP, block_idx=0): up_0,
            PerBlockId(block_type=BlockType.UP, block_idx=1): up_1,
            PerBlockId(block_type=BlockType.UP, block_idx=2): up_2,
            PerBlockId(block_type=BlockType.UP, block_idx=3): up_3,
        }
        for id, block in d.items():
            if block is not None:
                blocks.append(PerBlock(id=id, effect=block.effect, scales=block.scales))
        if len(blocks) == 0:
            blocks = None
        return (AllPerBlocks(blocks, ModelTypeSD.SD1_5),)


class PerBlock_SD15_LowLevelNode:
    NodeID = 'ADE_PerBlock_SD15_LowLevel'
    NodeName = 'AD Per Block++ (SD1.5) 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "down_0__0": ("AD_BLOCK",),
                "down_0__1": ("AD_BLOCK",),
                "down_1__0": ("AD_BLOCK",),
                "down_1__1": ("AD_BLOCK",),
                "down_2__0": ("AD_BLOCK",),
                "down_2__1": ("AD_BLOCK",),
                "down_3__0": ("AD_BLOCK",),
                "down_3__1": ("AD_BLOCK",),
                "mid": ("AD_BLOCK",),
                "up_0__0": ("AD_BLOCK",),
                "up_0__1": ("AD_BLOCK",),
                "up_0__2": ("AD_BLOCK",),
                "up_1__0": ("AD_BLOCK",),
                "up_1__1": ("AD_BLOCK",),
                "up_1__2": ("AD_BLOCK",),
                "up_2__0": ("AD_BLOCK",),
                "up_2__1": ("AD_BLOCK",),
                "up_2__2": ("AD_BLOCK",),
                "up_3__0": ("AD_BLOCK",),
                "up_3__1": ("AD_BLOCK",),
                "up_3__2": ("AD_BLOCK",),
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("PER_BLOCK",)
    CATEGORY = "Animate Diff 🎭🅐🅓/per block"
    FUNCTION = "create_per_block"

    def create_per_block(self,
                         down_0__0: Union[ADBlockHolder, None]=None,
                         down_0__1: Union[ADBlockHolder, None]=None,
                         down_1__0: Union[ADBlockHolder, None]=None,
                         down_1__1: Union[ADBlockHolder, None]=None,
                         down_2__0: Union[ADBlockHolder, None]=None,
                         down_2__1: Union[ADBlockHolder, None]=None,
                         down_3__0: Union[ADBlockHolder, None]=None,
                         down_3__1: Union[ADBlockHolder, None]=None,
                         mid: Union[ADBlockHolder, None]=None,
                         up_0__0: Union[ADBlockHolder, None]=None,
                         up_0__1: Union[ADBlockHolder, None]=None,
                         up_0__2: Union[ADBlockHolder, None]=None,
                         up_1__0: Union[ADBlockHolder, None]=None,
                         up_1__1: Union[ADBlockHolder, None]=None,
                         up_1__2: Union[ADBlockHolder, None]=None,
                         up_2__0: Union[ADBlockHolder, None]=None,
                         up_2__1: Union[ADBlockHolder, None]=None,
                         up_2__2: Union[ADBlockHolder, None]=None,
                         up_3__0: Union[ADBlockHolder, None]=None,
                         up_3__1: Union[ADBlockHolder, None]=None,
                         up_3__2: Union[ADBlockHolder, None]=None):
        blocks = []
        d = {
            PerBlockId(block_type=BlockType.DOWN, block_idx=0, module_idx=0): down_0__0,
            PerBlockId(block_type=BlockType.DOWN, block_idx=0, module_idx=1): down_0__1,
            PerBlockId(block_type=BlockType.DOWN, block_idx=1, module_idx=0): down_1__0,
            PerBlockId(block_type=BlockType.DOWN, block_idx=1, module_idx=1): down_1__1,
            PerBlockId(block_type=BlockType.DOWN, block_idx=2, module_idx=0): down_2__0,
            PerBlockId(block_type=BlockType.DOWN, block_idx=2, module_idx=1): down_2__1,
            PerBlockId(block_type=BlockType.DOWN, block_idx=3, module_idx=0): down_3__0,
            PerBlockId(block_type=BlockType.DOWN, block_idx=3, module_idx=1): down_3__1,
            PerBlockId(block_type=BlockType.MID): mid,
            PerBlockId(block_type=BlockType.UP, block_idx=0, module_idx=0): up_0__0,
            PerBlockId(block_type=BlockType.UP, block_idx=0, module_idx=1): up_0__1,
            PerBlockId(block_type=BlockType.UP, block_idx=0, module_idx=2): up_0__2,
            PerBlockId(block_type=BlockType.UP, block_idx=1, module_idx=0): up_1__0,
            PerBlockId(block_type=BlockType.UP, block_idx=1, module_idx=1): up_1__1,
            PerBlockId(block_type=BlockType.UP, block_idx=1, module_idx=2): up_1__2,
            PerBlockId(block_type=BlockType.UP, block_idx=2, module_idx=0): up_2__0,
            PerBlockId(block_type=BlockType.UP, block_idx=2, module_idx=1): up_2__1,
            PerBlockId(block_type=BlockType.UP, block_idx=2, module_idx=2): up_2__2,
            PerBlockId(block_type=BlockType.UP, block_idx=3, module_idx=0): up_3__0,
            PerBlockId(block_type=BlockType.UP, block_idx=3, module_idx=1): up_3__1,
            PerBlockId(block_type=BlockType.UP, block_idx=3, module_idx=2): up_3__2,
        }
        for id, block in d.items():
            if block is not None:
                blocks.append(PerBlock(id=id, effect=block.effect, scales=block.scales))
        if len(blocks) == 0:
            blocks = None
        return (AllPerBlocks(blocks, ModelTypeSD.SD1_5),)


class PerBlock_SD15_FromFloatsNode:
    NodeID = 'ADE_PerBlock_SD15_FromFloats'
    NodeName = 'AD Per Block Floats (SD1.5) 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "effect_21_floats": ("FLOATS",),
                "scale_21_floats": ("FLOATS",),
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("PER_BLOCK",)
    CATEGORY = "Animate Diff 🎭🅐🅓/per block"
    FUNCTION = "create_per_block"

    Desc = [
        short_desc('Use Floats from Value Schedules to select SD1.5 effect/scale values for blocks.'),
        'SD1.5 Motion Modules contain 21 blocks:',
        'idx 0 - start of down blocks (down_0__0)',
        'idx 7 - end of down blocks   (down_3__1)',
        'idx 8 - mid block            (mid)',
        'idx 9 - start of up blocks   (up_0__0)',
        'idx 20 - end of up blocks    (up_3__2)',
    ]
    register_description(NodeID, Desc)

    def create_per_block(self,
                         effect_21_floats: Union[list[float], None]=None,
                         scale_21_floats: Union[list[float], None]=None):
        if effect_21_floats is None and scale_21_floats is None:
            return (AllPerBlocks(None, ModelTypeSD.SD1_5),)
        # SD1.5 has 21 blocks
        block_total = 21
        holders = [ADBlockHolder() for _ in range(block_total)]
        if effect_21_floats is not None:
            effect_21_floats = extend_list_to_batch_size(effect_21_floats, block_total)
            for effect, holder in zip(effect_21_floats, holders):
                holder.effect = effect
        if scale_21_floats is not None:
            scale_21_floats = extend_list_to_batch_size(scale_21_floats, block_total)
            for scale, holder in zip(scale_21_floats, holders):
                holder.scales = [scale, scale]
        return PerBlock_SD15_LowLevelNode.create_per_block(self, *holders)


class PerBlock_SDXL_MidLevelNode:
    NodeID = 'ADE_PerBlock_SDXL_MidLevel'
    NodeName = 'AD Per Block+ (SDXL) 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "down_0": ("AD_BLOCK",),
                "down_1": ("AD_BLOCK",),
                "down_2": ("AD_BLOCK",),
                "mid": ("AD_BLOCK",),
                "up_0": ("AD_BLOCK",),
                "up_1": ("AD_BLOCK",),
                "up_2": ("AD_BLOCK",),
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("PER_BLOCK",)
    CATEGORY = "Animate Diff 🎭🅐🅓/per block"
    FUNCTION = "create_per_block"

    def create_per_block(self,
                         down_0: Union[ADBlockHolder, None]=None,
                         down_1: Union[ADBlockHolder, None]=None,
                         down_2: Union[ADBlockHolder, None]=None,
                         mid: Union[ADBlockHolder, None]=None,
                         up_0: Union[ADBlockHolder, None]=None,
                         up_1: Union[ADBlockHolder, None]=None,
                         up_2: Union[ADBlockHolder, None]=None):
        blocks = []
        d = {
            PerBlockId(block_type=BlockType.DOWN, block_idx=0): down_0,
            PerBlockId(block_type=BlockType.DOWN, block_idx=1): down_1,
            PerBlockId(block_type=BlockType.DOWN, block_idx=2): down_2,
            PerBlockId(block_type=BlockType.MID): mid,
            PerBlockId(block_type=BlockType.UP, block_idx=0): up_0,
            PerBlockId(block_type=BlockType.UP, block_idx=1): up_1,
            PerBlockId(block_type=BlockType.UP, block_idx=2): up_2,
        }
        for id, block in d.items():
            if block is not None:
                blocks.append(PerBlock(id=id, effect=block.effect, scales=block.scales))
        if len(blocks) == 0:
            blocks = None
        return (AllPerBlocks(blocks, ModelTypeSD.SDXL),)


class PerBlock_SDXL_LowLevelNode:
    NodeID = 'ADE_PerBlock_SDXL_LowLevel'
    NodeName = 'AD Per Block++ (SDXL) 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "down_0__0": ("AD_BLOCK",),
                "down_0__1": ("AD_BLOCK",),
                "down_1__0": ("AD_BLOCK",),
                "down_1__1": ("AD_BLOCK",),
                "down_2__0": ("AD_BLOCK",),
                "down_2__1": ("AD_BLOCK",),
                "mid": ("AD_BLOCK",),
                "up_0__0": ("AD_BLOCK",),
                "up_0__1": ("AD_BLOCK",),
                "up_0__2": ("AD_BLOCK",),
                "up_1__0": ("AD_BLOCK",),
                "up_1__1": ("AD_BLOCK",),
                "up_1__2": ("AD_BLOCK",),
                "up_2__0": ("AD_BLOCK",),
                "up_2__1": ("AD_BLOCK",),
                "up_2__2": ("AD_BLOCK",),
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("PER_BLOCK",)
    CATEGORY = "Animate Diff 🎭🅐🅓/per block"
    FUNCTION = "create_per_block"

    def create_per_block(self,
                         down_0__0: Union[ADBlockHolder, None]=None,
                         down_0__1: Union[ADBlockHolder, None]=None,
                         down_1__0: Union[ADBlockHolder, None]=None,
                         down_1__1: Union[ADBlockHolder, None]=None,
                         down_2__0: Union[ADBlockHolder, None]=None,
                         down_2__1: Union[ADBlockHolder, None]=None,
                         mid: Union[ADBlockHolder, None]=None,
                         up_0__0: Union[ADBlockHolder, None]=None,
                         up_0__1: Union[ADBlockHolder, None]=None,
                         up_0__2: Union[ADBlockHolder, None]=None,
                         up_1__0: Union[ADBlockHolder, None]=None,
                         up_1__1: Union[ADBlockHolder, None]=None,
                         up_1__2: Union[ADBlockHolder, None]=None,
                         up_2__0: Union[ADBlockHolder, None]=None,
                         up_2__1: Union[ADBlockHolder, None]=None,
                         up_2__2: Union[ADBlockHolder, None]=None,):
        blocks = []
        d = {
            PerBlockId(block_type=BlockType.DOWN, block_idx=0, module_idx=0): down_0__0,
            PerBlockId(block_type=BlockType.DOWN, block_idx=0, module_idx=1): down_0__1,
            PerBlockId(block_type=BlockType.DOWN, block_idx=1, module_idx=0): down_1__0,
            PerBlockId(block_type=BlockType.DOWN, block_idx=1, module_idx=1): down_1__1,
            PerBlockId(block_type=BlockType.DOWN, block_idx=2, module_idx=0): down_2__0,
            PerBlockId(block_type=BlockType.DOWN, block_idx=2, module_idx=1): down_2__1,
            PerBlockId(block_type=BlockType.MID): mid,
            PerBlockId(block_type=BlockType.UP, block_idx=0, module_idx=0): up_0__0,
            PerBlockId(block_type=BlockType.UP, block_idx=0, module_idx=1): up_0__1,
            PerBlockId(block_type=BlockType.UP, block_idx=0, module_idx=2): up_0__2,
            PerBlockId(block_type=BlockType.UP, block_idx=1, module_idx=0): up_1__0,
            PerBlockId(block_type=BlockType.UP, block_idx=1, module_idx=1): up_1__1,
            PerBlockId(block_type=BlockType.UP, block_idx=1, module_idx=2): up_1__2,
            PerBlockId(block_type=BlockType.UP, block_idx=2, module_idx=0): up_2__0,
            PerBlockId(block_type=BlockType.UP, block_idx=2, module_idx=1): up_2__1,
            PerBlockId(block_type=BlockType.UP, block_idx=2, module_idx=2): up_2__2,
        }
        for id, block in d.items():
            if block is not None:
                blocks.append(PerBlock(id=id, effect=block.effect, scales=block.scales))
        if len(blocks) == 0:
            blocks = None
        return (AllPerBlocks(blocks, ModelTypeSD.SDXL),)


class PerBlock_SDXL_FromFloatsNode:
    NodeID = 'ADE_PerBlock_SDXL_FromFloats'
    NodeName = 'AD Per Block Floats (SDXL) 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "effect_16_floats": ("FLOATS",),
                "scale_16_floats": ("FLOATS",),
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("PER_BLOCK",)
    CATEGORY = "Animate Diff 🎭🅐🅓/per block"
    FUNCTION = "create_per_block"

    Desc = [
        short_desc('Use Floats from Value Schedules to select SDXL effect/scale values for blocks.'),
        'SDXL Motion Modules contain 16 blocks:',
        'idx 0 - start of down blocks (down_0__0)',
        'idx 5 - end of down blocks   (down_2__1)',
        'idx 6 - mid block            (mid)',
        'idx 7 - start of up blocks   (up_0__0)',
        'idx 15 - end of up blocks    (up_2__2)',
    ]
    register_description(NodeID, Desc)

    def create_per_block(self,
                         effect_16_floats: Union[list[float], None]=None,
                         scale_16_floats: Union[list[float], None]=None):
        if effect_16_floats is None and scale_16_floats is None:
            return (AllPerBlocks(None, ModelTypeSD.SDXL),)
        # SDXL has 16 blocks
        block_total = 16
        holders = [ADBlockHolder() for _ in range(block_total)]
        if effect_16_floats is not None:
            effect_16_floats = extend_list_to_batch_size(effect_16_floats, block_total)
            for effect, holder in zip(effect_16_floats, holders):
                holder.effect = effect
        if scale_16_floats is not None:
            scale_16_floats = extend_list_to_batch_size(scale_16_floats, block_total)
            for scale, holder in zip(scale_16_floats, holders):
                holder.scales = [scale, scale]
        return PerBlock_SDXL_LowLevelNode.create_per_block(self, *holders)
