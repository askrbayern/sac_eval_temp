import math
from typing import List, Optional

import torch
from torch import nn
from torch.nn.utils import weight_norm
from einops import rearrange
from .bottleneck import RoundBottleneck
from .transformer import SnakeBeta

# Structure

class AudioAutoencoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int, downsampling_ratio: int, sample_rate: int, io_channels: int = 1, bottleneck: Optional[nn.Module] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        self.io_channels = io_channels
        self.bottleneck = bottleneck # if it is None, the latent will be continuous values

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        z = self.encoder(audio)
        if self.bottleneck is not None:
            z = self.bottleneck(z)
        return z

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)


# ================= Oobleck Encoder/Decoder (weight_norm + SnakeBeta) =================

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def get_activation(activation: str, channels: int):
    if activation == "snake":
        return SnakeBeta(channels)
    if activation == "elu":
        return nn.ELU()
    return nn.Identity()


class ResidualUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, use_snake: bool = False):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", out_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, dilation=dilation, padding=padding),
            get_activation("snake" if use_snake else "elu", out_channels),
            WNConv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, use_snake: bool = False):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=9, use_snake=use_snake),
            get_activation("snake" if use_snake else "elu", in_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, use_snake: bool = False, use_nearest_upsample: bool = False):
        super().__init__()
        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2 * stride, stride=1, bias=False, padding='same')
            )
        else:
            upsample_layer = WNConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))
        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", in_channels),
            upsample_layer,
            ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=9, use_snake=use_snake),
        )

    def forward(self, x):
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(self, in_channels: int = 2, channels: int = 64, latent_dim: int = 2048, c_mults = [1,2,4,8], strides = [2,4,4,4], use_snake: bool = True):
        super().__init__()
        c_mults = [1] + c_mults
        self.depth = len(c_mults)
        layers = [WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)]
        for i in range(self.depth - 1):
            layers += [EncoderBlock(in_channels=c_mults[i] * channels, out_channels=c_mults[i + 1] * channels, stride=strides[i], use_snake=use_snake)]
        layers += [
            get_activation("snake" if use_snake else "elu", c_mults[-1] * channels),
            WNConv1d(in_channels=c_mults[-1] * channels, out_channels=latent_dim, kernel_size=3, padding=1)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(self, out_channels: int = 2, channels: int = 64, latent_dim: int = 2048, c_mults = [1,2,4,8], strides = [2,4,4,4], use_snake: bool = True, final_tanh: bool = False, use_nearest_upsample: bool = False):
        super().__init__()
        c_mults = [1] + c_mults
        self.depth = len(c_mults)
        layers = [WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1] * channels, kernel_size=7, padding=3)]
        for i in range(self.depth - 1, 0, -1):
            layers += [DecoderBlock(in_channels=c_mults[i] * channels, out_channels=c_mults[i - 1] * channels, stride=strides[i - 1], use_snake=use_snake, use_nearest_upsample=use_nearest_upsample)]
        layers += [
            get_activation("snake" if use_snake else "elu", c_mults[0] * channels),
            WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False),
            nn.Tanh() if final_tanh else nn.Identity()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


