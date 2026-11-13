"""Fully convolutional 2D U-Net on equiangular global grids (design doc §3).

No attention and no absolute position input, so the same weights apply at every rung's
grid size and the receptive field in pixels is the same at every scale.

Sphere-aware padding: longitude wraps around; rows beyond a pole continue on the other
side of the pole, i.e. they are the first rows mirrored in latitude and shifted by half
the longitudes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sphere_pad(x: torch.Tensor, p: int) -> torch.Tensor:
    """Pad (B, C, H, W) by p pixels on every side, consistently with the sphere."""
    if p == 0:
        return x
    W = x.shape[-1]
    top = torch.roll(x[..., :p, :].flip(-2), W // 2, dims=-1)
    bottom = torch.roll(x[..., -p:, :].flip(-2), W // 2, dims=-1)
    x = torch.cat([top, x, bottom], dim=-2)
    return torch.cat([x[..., -p:], x, x[..., :p]], dim=-1)


class SConv(nn.Module):
    """Conv2d with sphere padding."""

    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__()
        self.p = k // 2
        self.conv = nn.Conv2d(cin, cout, k, padding=0)

    def forward(self, x):
        return self.conv(sphere_pad(x, self.p))


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb: int, dropout: float = 0.0):
        super().__init__()
        self.n1 = nn.GroupNorm(min(32, cin // 4) or 1, cin)
        self.c1 = SConv(cin, cout)
        self.emb = nn.Linear(emb, 2 * cout)
        self.n2 = nn.GroupNorm(min(32, cout // 4) or 1, cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = SConv(cout, cout)
        nn.init.zeros_(self.c2.conv.weight)
        nn.init.zeros_(self.c2.conv.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.emb(e)[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return self.skip(x) + h


class FourierEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float = 16.0):
        super().__init__()
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, x):
        x = x[:, None] * self.freqs[None] * 2 * math.pi
        return torch.cat([x.cos(), x.sin()], dim=1)


class UNet(nn.Module):
    """in_ch input channels -> out_ch output channels, conditioned on a scalar (noise level)."""

    def __init__(self, in_ch: int = 2, out_ch: int = 1, base: int = 64,
                 mults: tuple[int, ...] = (1, 2, 2, 4), blocks: int = 2, emb: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        self.fourier = FourierEmbedding(emb)
        self.emb = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inc = SConv(in_ch, base)
        self.down = nn.ModuleList()
        chans = [base]
        c = base
        for i, m in enumerate(mults):
            for _ in range(blocks):
                self.down.append(ResBlock(c, base * m, emb, dropout))
                c = base * m
                chans.append(c)
            if i < len(mults) - 1:
                self.down.append(nn.AvgPool2d(2))
                chans.append(c)
        self.mid = nn.ModuleList([ResBlock(c, c, emb, dropout), ResBlock(c, c, emb, dropout)])
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(blocks + 1):
                self.up.append(ResBlock(c + chans.pop(), base * m, emb, dropout))
                c = base * m
            if i > 0:
                self.up.append(nn.Upsample(scale_factor=2, mode="nearest"))
        self.out = nn.Sequential(nn.GroupNorm(min(32, c // 4), c), nn.SiLU(), SConv(c, out_ch))
        nn.init.zeros_(self.out[-1].conv.weight)
        nn.init.zeros_(self.out[-1].conv.bias)
        self.levels = len(mults) - 1

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        if H % (2 ** self.levels) or W % (2 ** self.levels):
            raise ValueError(f"grid {H}x{W} not divisible by {2 ** self.levels}")
        e = self.emb(self.fourier(noise))
        h = self.inc(x)
        hs = [h]
        for layer in self.down:
            h = layer(h, e) if isinstance(layer, ResBlock) else layer(h)
            hs.append(h)
        for layer in self.mid:
            h = layer(h, e)
        for layer in self.up:
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, hs.pop()], dim=1), e)
            else:
                h = layer(h)
        return self.out(h)
