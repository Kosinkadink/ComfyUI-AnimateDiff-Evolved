

class LoraHookGroup:
    '''
    Stores LoRA hooks to apply for conditioning
    '''
    def __init__(self):
        self.hooks = []
    
    def add(self, hook: str):
        if hook not in self.hooks:
            self.hooks.append(hook)
    
    def is_empty(self):
        return len(self.hooks) == 0

    def clone(self):
        cloned = LoraHookGroup()
        for hook in self.hooks:
            cloned.add(hook)
        return cloned

    def clone_and_combine(self, other: 'LoraHookGroup'):
        cloned = self.clone()
        for hook in other.hooks:
            cloned.add(hook)
        return cloned
