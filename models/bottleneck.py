import torch
from torch import nn


class RoundBottleneck(nn.Module):
    """
    Rounding bottleneck with optional training-time dither.
    Eval: always hard-round.
    """
    def __init__(self, latent_dim: int = 32, dither: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.dither = dither

    def forward(self, x):
        assert x.shape[1] == self.latent_dim, f"Expected channels={self.latent_dim}, got {x.shape[1]}"
        if self.training and self.dither:
            x = x + (torch.rand_like(x) - 0.5)
            q = x
        else:
            q = x.round()
        return q


