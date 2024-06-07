from abc import ABC, abstractmethod
import torch
from torch import Tensor
from typing import Union


class InputPIA(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_mask(self, x: Tensor):
        pass


class InputPIA_Multival(InputPIA):
    def __init__(self, multival: Union[float, Tensor]):
        self.multival = multival

    def get_mask(self, x: Tensor):
        if type(self.multival) is Tensor:
            return self.multival
        # if not Tensor, then is float, and simply return a mask with the right dimensions + value
        b, c, h, w = x.shape
        mask = torch.ones(size=(b, h, w))
        return mask * self.multival
