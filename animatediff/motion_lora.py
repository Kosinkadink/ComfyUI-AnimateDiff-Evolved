class MotionLoraInfo:
    def __init__(self, name: str, strength: float = 1.0, hash: str=""):
        self.name = name
        self.strength = strength
        self.hash = ""
    
    def set_hash(self, hash: str):
        self.hash = hash
    
    def clone(self):
        return MotionLoraInfo(self.name, self.strength, self.hash)


class MotionLoraList:
    def __init__(self):
        self.loras: list[MotionLoraInfo] = []
    
    def add_lora(self, lora: MotionLoraInfo):
        self.loras.append(lora)
    
    def clone(self):
        new_list = MotionLoraList()
        for lora in self.loras:
            new_list.add_lora(lora.clone())
        return new_list
