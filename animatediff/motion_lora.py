from torch import Tensor

from pathlib import Path

class MotionLoRAInfo:
    def __init__(self, name: str, strength: float = 1.0, hash: str=""):
        self.name = name
        self.strength = strength
        self.hash = ""
    
    def set_hash(self, hash: str):
        self.hash = hash
    
    def clone(self):
        return MotionLoRAInfo(self.name, self.strength, self.hash)


class MotionLoRAWrapper:
    def __init__(self, state_dict: dict[str, Tensor], hash: str):
        self.state_dict = state_dict
        self.hash = hash
        self.info: MotionLoRAInfo = None
    
    def set_info(self, info: MotionLoRAInfo):
        self.info = info


class MotionLoRAList:
    def __init__(self):
        self.loras: list[MotionLoRAInfo] = []
    
    def add_lora(self, lora: MotionLoRAInfo):
        self.loras.append(lora)
    
    def clone(self):
        new_list = MotionLoRAList()
        for lora in self.loras:
            new_list.add_lora(lora.clone())
        return new_list
