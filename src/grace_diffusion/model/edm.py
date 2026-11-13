"""EDM preconditioning, loss and sampler (Karras et al. 2022), as in imonet but 2D.

The diffusion model generates the added detail d = target - input (normalised units),
conditioned on the input field as a second channel. A deterministic regression model
(`Regressor`) with the same backbone predicts d directly; it is the `regress` baseline.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .unet import UNet


class EDM(nn.Module):
    def __init__(self, net: UNet, sigma_data: float = 1.0):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data

    def denoise(self, x_noisy: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        sd = self.sigma_data
        s = sigma.view(-1, 1, 1, 1)
        c_skip = sd ** 2 / (s ** 2 + sd ** 2)
        c_out = s * sd / torch.sqrt(s ** 2 + sd ** 2)
        c_in = 1 / torch.sqrt(s ** 2 + sd ** 2)
        c_noise = torch.log(sigma) / 4
        F_x = self.net(torch.cat([c_in * x_noisy, cond], dim=1), c_noise)
        return c_skip * x_noisy + c_out * F_x

    def loss(self, detail: torch.Tensor, cond: torch.Tensor, weight: torch.Tensor,
             P_mean: float = -1.2, P_std: float = 1.2) -> torch.Tensor:
        """EDM loss, weighted per pixel by area x loss mask, normalised by the weight sum."""
        sigma = torch.exp(torch.randn(detail.shape[0], device=detail.device) * P_std + P_mean)
        s = sigma.view(-1, 1, 1, 1)
        lam = (s ** 2 + self.sigma_data ** 2) / (s * self.sigma_data) ** 2
        noisy = detail + torch.randn_like(detail) * s
        D = self.denoise(noisy, sigma, cond)
        return (lam * weight * (D - detail) ** 2).sum() / weight.sum()

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps: int = 32, sigma_min: float = 0.002,
               sigma_max: float = 80.0, rho: float = 7.0,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """Deterministic Heun sampler (EDM Algorithm 1) from Gaussian noise."""
        dev = cond.device
        i = torch.arange(steps, device=dev, dtype=torch.float64)
        t = (sigma_max ** (1 / rho) + i / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t = torch.cat([t, torch.zeros(1, device=dev, dtype=torch.float64)])
        x = torch.randn(cond.shape, generator=generator, device=dev, dtype=torch.float64) * t[0]
        for k in range(steps):
            tc, tn = t[k], t[k + 1]
            sig = torch.full((cond.shape[0],), float(tc), device=dev)
            d = (x - self.denoise(x.float(), sig, cond).double()) / tc
            xn = x + (tn - tc) * d
            if tn > 0:
                sig_n = torch.full((cond.shape[0],), float(tn), device=dev)
                d2 = (xn - self.denoise(xn.float(), sig_n, cond).double()) / tn
                xn = x + (tn - tc) * 0.5 * (d + d2)
            x = xn
        return x.float()


class Regressor(nn.Module):
    """Deterministic baseline: predicts the detail from the input with the same U-Net."""

    def __init__(self, net: UNet):
        super().__init__()
        self.net = net

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros(cond.shape[0], device=cond.device)
        return self.net(torch.cat([torch.zeros_like(cond), cond], dim=1), zero)

    def loss(self, detail, cond, weight):
        return (weight * (self(cond) - detail) ** 2).sum() / weight.sum()


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def seeded_generator(device, *seed) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(np.random.SeedSequence(list(seed)).generate_state(1)[0]))
    return g
