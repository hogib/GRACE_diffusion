"""Baseline generators (design doc §5). Numpy only; `regress` (a U-Net) lives with the model code.

All generators work on one month's coefficients `c` (already rotated, EWH mm) for a rung
with target degree L, and return an ensemble of target-rung grids (M, 4L, 8L) in mm.
Every member satisfies the consistency constraint exactly: degrees the input carries
(input taper > 0) equal input / taper, i.e. the true target's coefficients there.
"""
from __future__ import annotations

import numpy as np

from . import sh


def _fixed_and_free(L: int, flat: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Input taper, target taper (to degree L), and the boolean 'free' degrees."""
    w_in = sh.taper(L // 2, L, flat)
    w_out = sh.taper(L, L, flat)
    free = w_in == 0
    return w_in, w_out, free


def known_part(c: np.ndarray, L: int, flat: float = 0.8) -> np.ndarray:
    """Target-rung coefficients that the input determines; zero in the free degrees."""
    w_in, w_out, free = _fixed_and_free(L, flat)
    cc = np.zeros((2, L + 1, L + 1))
    k = min(L, c.shape[-1] - 1)
    cc[:, : k + 1, : k + 1] = c[:, : k + 1, : k + 1]
    out = sh.degree_weights(cc, np.where(free, 0.0, 1.0))      # = input / w_in where w_in > 0
    return out


def input_only(c: np.ndarray, L: int, members: int = 1, rng=None, flat: float = 0.8) -> np.ndarray:
    """No added detail: the target-rung field with the free degrees set to zero."""
    g = sh.synth(known_part(c, L, flat), L)
    return np.repeat(g[None], members, axis=0)


def _random_coeffs(dv: np.ndarray, L: int, rng: np.random.Generator) -> np.ndarray:
    """Isotropic Gaussian coefficients with degree variance dv[l] (sum over orders)."""
    out = np.zeros((2, L + 1, L + 1))
    for l in range(2, L + 1):
        if dv[l] <= 0:
            continue
        sd = np.sqrt(dv[l] / (2 * l + 1))
        out[0, l, : l + 1] = rng.normal(0, sd, l + 1)
        out[1, l, 1: l + 1] = rng.normal(0, sd, l)
    return out


def extrapolated_spectrum(c: np.ndarray, L: int, flat: float = 0.8) -> np.ndarray:
    """Degree variance for degrees 0..L: a power law fitted to the degrees the input carries.

    The input carries degrees with input taper > 0 (for L=8 that is only l = 2, 3).
    The fit uses the untapered coefficients there, i.e. input / taper.
    """
    w_in, _w_out, free = _fixed_and_free(L, flat)
    known = known_part(c, L, flat)
    dv = sh.degree_variance(known)
    l = np.arange(L + 1)
    use = (~free) & (l >= 2) & (dv > 0)
    if use.sum() < 2:
        return np.zeros(L + 1)
    b, a = np.polyfit(np.log(l[use]), np.log(dv[use]), 1)
    out = np.zeros(L + 1)
    out[2:] = np.exp(a + b * np.log(l[2:]))
    return out


def spectral_gauss(c: np.ndarray, L: int, members: int, rng: np.random.Generator,
                   spectrum: np.ndarray | None = None, flat: float = 0.8) -> np.ndarray:
    """Input plus isotropic Gaussian detail in the free degrees.

    spectrum: degree variance of the untapered field per degree (length >= L+1). Default:
    extrapolated from the input month (fair for the transfer test). The target taper is
    applied to the random coefficients, as it is to the true target.
    """
    _w_in, w_out, free = _fixed_and_free(L, flat)
    dv = extrapolated_spectrum(c, L, flat) if spectrum is None else np.asarray(spectrum)[: L + 1]
    base = known_part(c, L, flat)
    out = np.empty((members, *sh.grid_shape(L)))
    for i in range(members):
        r = _random_coeffs(np.where(free, dv, 0.0), L, rng)
        out[i] = sh.synth(base + sh.degree_weights(r, w_out), L)
    return out


def climatological_spectrum(coeffs: np.ndarray, train_idx: np.ndarray, L: int) -> np.ndarray:
    """Mean degree variance over training months (rotation invariant, so frame-free)."""
    return np.mean([sh.degree_variance(coeffs[i][:, : L + 1, : L + 1].astype(np.float64))
                    for i in train_idx], axis=0)
