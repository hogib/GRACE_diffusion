"""Scores for ensembles of fields (design doc §6). Numpy only.

Conventions: `ens` is (M, H, W), `truth` is (H, W), `w` is (H, W) weights (pixel area x
region mask; zero outside the scored region). Scores are computed on the *added detail*
(field minus input) by the caller passing detail arrays, or on full fields: for the
pixelwise scores below the difference only matters for correlation.
"""
from __future__ import annotations

import numpy as np

from . import sh


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float((x * w).sum() / w.sum())


def crps_fair(ens: np.ndarray, truth: np.ndarray, w: np.ndarray) -> float:
    """Area-weighted mean of the fair (unbiased) ensemble CRPS per pixel.

    CRPS = mean|X - y| - sum_{i,j}|X_i - X_j| / (2 M (M - 1)); for M = 1 it is |X - y|.
    """
    M = ens.shape[0]
    t1 = np.abs(ens - truth[None]).mean(axis=0)
    if M == 1:
        return _wmean(t1, w)
    xs = np.sort(ens, axis=0)
    k = (2 * np.arange(1, M + 1) - M - 1).reshape(-1, 1, 1)
    pair = 2 * (k * xs).sum(axis=0)                 # sum_{i,j} |x_i - x_j|
    return _wmean(t1 - pair / (2 * M * (M - 1)), w)


def energy_score(ens: np.ndarray, truth: np.ndarray, w: np.ndarray) -> float:
    """Fair energy score with the area-weighted L2 norm over the region."""
    M = ens.shape[0]
    norm = lambda d: np.sqrt((d ** 2 * w).sum(axis=(-2, -1)) / w.sum())
    t1 = norm(ens - truth[None]).mean()
    if M == 1:
        return float(t1)
    d = ens[:, None] - ens[None]
    t2 = norm(d).sum() / (2 * M * (M - 1))
    return float(t1 - t2)


def ensemble_stats(ens: np.ndarray, truth: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """RMSE of the mean, spread/error ratio, 50/90% coverage."""
    M = ens.shape[0]
    mean = ens.mean(axis=0)
    rmse = np.sqrt(_wmean((mean - truth) ** 2, w))
    out = {"rmse_mean": float(rmse)}
    if M > 1:
        spread = np.sqrt(_wmean(ens.var(axis=0, ddof=1), w))
        out["spread_error"] = float(np.sqrt((M + 1) / M) * spread / rmse) if rmse > 0 else np.nan
        p = pit(ens, truth)
        for c in (50, 90):
            out[f"cov{c}"] = _wmean((np.abs(p - 0.5) <= c / 200).astype(float), w)
    return out


def pit(ens: np.ndarray, truth: np.ndarray, seed: int = 0) -> np.ndarray:
    """Randomised probability integral transform per pixel: (rank + U) / (M + 1).

    Uniform on [0, 1] for a calibrated ensemble of any size M. Coverage of the central
    c% interval is the fraction with |PIT - 0.5| <= c/200. (Percentile intervals of the
    members are biased low for finite M, e.g. ~0.84 instead of 0.90 for M = 30.)
    """
    M = ens.shape[0]
    below = (ens < truth[None]).sum(axis=0)
    ties = (ens == truth[None]).sum(axis=0)
    u = np.random.default_rng(seed).random(truth.shape)
    return (below + u * (ties + 1)) / (M + 1)


def correlation(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    am, bm = _wmean(a, w), _wmean(b, w)
    cov = _wmean((a - am) * (b - bm), w)
    return float(cov / np.sqrt(_wmean((a - am) ** 2, w) * _wmean((b - bm) ** 2, w)))


def detail_power_ratio(ens: np.ndarray, inp: np.ndarray, truth: np.ndarray, w: np.ndarray) -> float:
    """Mean over members of the weighted power of the added detail, relative to the truth's.

    Computed in the scored region only (unlike band_power_ratio, which is global and so
    dominated by the ice sheets that are masked out of training and scoring).
    """
    pt = _wmean((truth - inp) ** 2, w)
    pg = np.mean([_wmean((e - inp) ** 2, w) for e in ens])
    return float(pg / pt) if pt > 0 else np.nan


def band_power_ratio(ens: np.ndarray, truth: np.ndarray, l0: int, l1: int) -> float:
    """Mean over members of (generated power / true power) in degrees l0..l1 (global)."""
    pt = sh.degree_variance(sh.analyse(truth))[l0: l1 + 1].sum()
    pg = np.mean([sh.degree_variance(sh.analyse(e))[l0: l1 + 1].sum() for e in ens])
    return float(pg / pt) if pt > 0 else np.nan


def consistency_rms(ens: np.ndarray, inp: np.ndarray, L_in: int, flat: float = 0.8) -> float:
    """RMS of lowpass(member, L_in) - input, relative to the input RMS (global)."""
    L = inp.shape[0] // 4
    r = []
    for e in ens:
        back = sh.synth(sh.lowpass(sh.analyse(e), L_in, flat), L)
        r.append(np.sqrt(np.mean((back - inp) ** 2)))
    return float(np.mean(r) / np.sqrt(np.mean(inp ** 2)))
