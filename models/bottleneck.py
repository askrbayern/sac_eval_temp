import torch
from torch import nn

# bottleneck, always round (since this repo is inference only)

class RoundBottleneck(nn.Module):
    def __init__(self, latent_dim: int = 32, dither: bool = False):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, x):
        assert x.shape[1] == self.latent_dim, f"Expected channels={self.latent_dim}, got {x.shape[1]}"
        q = x.round()
        return q


