from torch import nn

import comfy.ops


FancyVideoKeys = [
    'fps_embedding.linear.bias',
    'fps_embedding.linear.weight',
    'motion_embedding.linear.bias',
    'motion_embedding.linear.weight',
    'conv_in.bias',
    'conv_in.weight',
]


def initialize_weights_to_zero(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class FancyVideoCondEmbedding(nn.Module):
    def __init__(self, in_channels: int, cond_embed_dim: int, act_fn: str = "silu", ops=comfy.ops.disable_weight_init):
        super().__init__()

        self.linear = ops.Linear(in_channels, cond_embed_dim)
        self.act = None
        if act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "mish":
            self.act = nn.Mish()

    def forward(self, sample):
        sample = self.linear(sample)

        if self.act is not None:
            sample = self.act(sample)

        return sample
