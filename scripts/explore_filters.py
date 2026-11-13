"""Exploratory look at ITSG monthly solutions: where do stripes start, and what do filters do?

Usage: uv run python scripts/explore_filters.py data/raw runs/explore

Units: approximate equivalent water height (EWH) in mm, computed WITHOUT load Love
numbers (k_l = 0). This overstates degree-2 amplitudes by ~40% and degrees >= 15 by
a few percent; it is good enough to locate stripes, not for physical numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyshtools as pysh

A = 6378136.3          # m
RHO_E, RHO_W = 5517.0, 1000.0
LMAX = 120
BOX = (30.0, 45.0, 20.0, 50.0)   # lat0, lat1, lon0, lon1 (design doc test box)


def read_month(path: Path) -> tuple[np.ndarray, np.ndarray]:
    c, _gm, _r0, err = pysh.shio.read_icgem_gfc(str(path), lmax=LMAX, errors="formal")
    return c, err


def ewh_factor(lmax: int = LMAX) -> np.ndarray:
    l = np.arange(lmax + 1)
    return A * RHO_E / (3 * RHO_W) * (2 * l + 1) * 1e3   # mm, k_l = 0


def gaussian_weights(radius_km: float, lmax: int = LMAX) -> np.ndarray:
    """Normalised Gaussian smoothing weights (W_0 = 1), exponential approximation."""
    b = np.log(2) / (1 - np.cos(radius_km * 1e3 / A))
    l = np.arange(lmax + 1)
    return np.exp(-l * (l + 1) / (2 * b))


def degree_amplitude(c: np.ndarray) -> np.ndarray:
    return np.sqrt((c ** 2).sum(axis=(0, 2)))


def main(raw: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(raw.glob("*/n120/*.gfc"))
    months = [f.stem for f in files]
    coeffs, errs = zip(*(read_month(f) for f in files))
    coeffs, errs = np.array(coeffs), np.array(errs)
    ref = np.array([m[:4] >= "2004" and m[:4] <= "2009" for m in months])
    anom = coeffs - coeffs[ref].mean(axis=0)
    anom[:, :, :2, :] = 0.0                      # drop degrees 0 and 1 (not observed)
    f = ewh_factor()[None, None, :, None]
    l = np.arange(LMAX + 1)

    # 1. degree amplitudes: signal, formal error, month-to-month difference / sqrt(2)
    sig = np.array([degree_amplitude(a * f[0]) for a in anom])
    err = np.array([degree_amplitude(e * f[0]) for e in errs])
    diff = np.array([degree_amplitude((anom[i + 1] - anom[i]) * f[0]) / np.sqrt(2)
                     for i in range(len(anom) - 1)])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arr, lab in [(sig, "monthly anomaly"), (diff, "month-to-month diff / √2"), (err, "formal error")]:
        med = np.median(arr, axis=0)
        ax.semilogy(l[2:], med[2:], label=lab)
    for L in (15, 30, 60, 120):
        ax.axvline(L, color="0.8", lw=0.8)
    ax.set_xlabel("degree l")
    ax.set_ylabel("degree amplitude, EWH mm (k_l = 0)")
    ax.set_title(f"ITSG monthly solutions, median over {len(months)} months")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "degree_amplitudes.png", dpi=130)
    np.savetxt(out / "degree_amplitudes.csv",
               np.c_[l, np.median(sig, 0), np.median(diff, 0), np.median(err, 0)],
               delimiter=",", header="l,signal,diff_over_sqrt2,formal", comments="")

    # 2. maps for one month: truncations and Gaussian smoothing
    month = "2010-04" if "2010-04" in months else months[len(months) // 2]
    a = anom[months.index(month)] * f[0]
    variants = [
        ("l ≤ 30, unfiltered", 30, None),
        ("l ≤ 60, unfiltered", 60, None),
        ("l ≤ 96, unfiltered", 96, None),
        ("l ≤ 60, Gaussian 300 km", 60, 300),
        ("l ≤ 96, Gaussian 300 km", 96, 300),
        ("l ≤ 120, Gaussian 500 km", 120, 500),
    ]
    lat = np.linspace(BOX[1], BOX[0], 61)
    lon = np.linspace(BOX[2], BOX[3], 121)
    fig_g, axes_g = plt.subplots(3, 2, figsize=(12, 10))
    fig_r, axes_r = plt.subplots(3, 2, figsize=(12, 9))
    for (title, L, r), axg, axr in zip(variants, axes_g.flat, axes_r.flat):
        c = a.copy()
        c[:, L + 1:, :] = 0.0
        if r:
            c = c * gaussian_weights(r)[None, :, None]
        sh = pysh.SHCoeffs.from_array(c, normalization="4pi", csphase=1)
        grid = sh.expand(grid="DH2", extend=True).to_array()
        vmax = np.percentile(np.abs(grid), 99)
        im = axg.imshow(grid, extent=(0, 360, -90, 90), cmap="RdBu", vmin=-vmax, vmax=vmax)
        axg.set_title(f"{month}: {title}")
        fig_g.colorbar(im, ax=axg, shrink=0.8, label="mm")
        lo, la = np.meshgrid(lon, lat)
        reg = sh.expand(lat=la.ravel(), lon=lo.ravel()).reshape(la.shape)
        vr = np.percentile(np.abs(reg), 99)
        im = axr.imshow(reg, extent=(BOX[2], BOX[3], BOX[0], BOX[1]), cmap="RdBu", vmin=-vr, vmax=vr)
        axr.set_title(f"{month}: {title}")
        fig_r.colorbar(im, ax=axr, shrink=0.8, label="mm")
    fig_g.tight_layout()
    fig_g.savefig(out / "maps_global.png", dpi=110)
    fig_r.tight_layout()
    fig_r.savefig(out / "maps_turkiye_box.png", dpi=110)

    # 3. fraction of signal power each filter keeps per rung band, and estimated SNR per band
    bands = {"R0 add (8-15)": (8, 15), "R1 add (16-30)": (16, 30),
             "R2 add (31-60)": (31, 60), "R3 add (61-120)": (61, 120)}
    S2 = np.median(sig, 0) ** 2
    N2 = np.median(diff, 0) ** 2
    lines = ["band, signal/noise power (median anomaly vs month-diff proxy), "
             + ", ".join(f"kept by Gaussian {r} km" for r in (200, 300, 500))]
    for name, (l0, l1) in bands.items():
        s = S2[l0:l1 + 1].sum()
        n = N2[l0:l1 + 1].sum()
        kept = [(S2[l0:l1 + 1] * gaussian_weights(r)[l0:l1 + 1] ** 2).sum() / s for r in (200, 300, 500)]
        lines.append(f"{name}, {s / n:.2f}, " + ", ".join(f"{k:.2f}" for k in kept))
    (out / "bands.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("months:", len(months), "map month:", month)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
