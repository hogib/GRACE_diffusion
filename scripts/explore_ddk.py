"""Compare DDK and Gaussian filters per rung band on ITSG monthly solutions.

Usage: uv run python scripts/explore_ddk.py data/raw runs/explore

EWH in mm with PREM load Love numbers (gravity_toolkit, Han & Wahr 1995 set).
Per band and filter we estimate, from median degree variances over months:
  S_tot  = power of the monthly anomaly (signal + noise)
  N      = power of the month-to-month difference / 2 (noise proxy; includes real
           month-to-month change, so it overstates noise and S below is a lower bound)
  S      = max(S_tot - N, 0), the signal estimate
and after filtering: S_f = max(S_tot_f - N_f, 0), N_f.
Reported: kept = S_f / S, residual SNR = S_f / N_f.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyshtools as pysh
import scipy.sparse as sp
from gravity_toolkit import load_love_numbers
from shxarray.io.binv_legacy import readBINV

A = 6378136.3
RHO_E, RHO_W = 5517.0, 1000.0
LMAX = 120
BOX = (30.0, 45.0, 20.0, 50.0)
DDK = {"DDK1": "1d14p_4", "DDK2": "1d13p_4", "DDK3": "1d12p_4", "DDK4": "5d11p_4", "DDK5": "1d11p_4"}
BANDS = {"R0 7-15": (8, 15), "R1 16-30": (16, 30), "R2 31-60": (31, 60), "R3 61-120": (61, 120)}


def ewh_factor() -> np.ndarray:
    _h, k, _l = load_love_numbers(LMAX, LOVE_NUMBERS=0, REFERENCE="CF")
    l = np.arange(LMAX + 1)
    return A * RHO_E / (3 * RHO_W) * (2 * l + 1) / (1 + k) * 1e3


def gauss_w(r_km: float) -> np.ndarray:
    b = np.log(2) / (1 - np.cos(r_km * 1e3 / A))
    l = np.arange(LMAX + 1)
    return np.exp(-l * (l + 1) / (2 * b))


class DDKFilter:
    """Apply a block-diagonal DDK matrix to pyshtools-ordered coefficient arrays (2, L+1, L+1)."""

    def __init__(self, path: Path):
        ds = readBINV(str(path))
        n, m = ds["n"].values, ds["m"].values
        self.mat = sp.csr_matrix(ds["mat"].data.tocsr()) if hasattr(ds["mat"].data, "tocsr") else sp.csr_matrix(ds["mat"].data.to_scipy_sparse())
        # vector index -> (cs, l, |m|); negative m = sine
        self.cs = (m < 0).astype(int)
        self.l = n
        self.m = np.abs(m)
        # output side ordering (nm_) assumed identical to input (nm); checked below
        assert np.array_equal(ds["n_"].values, n) and np.array_equal(ds["m_"].values, m)

    def __call__(self, c: np.ndarray) -> np.ndarray:
        v = c[self.cs, self.l, self.m]
        out = c.copy()
        out[self.cs, self.l, self.m] = self.mat @ v
        return out


def band_power(c: np.ndarray, l0: int, l1: int) -> float:
    return float((c[:, l0:l1 + 1, :] ** 2).sum())


def main(raw: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(raw.glob("itsg*/n120/*.gfc"))
    months = [f.stem for f in files]
    C = np.array([pysh.shio.read_icgem_gfc(str(f), lmax=LMAX)[0] for f in files])
    ref = np.array(["2004" <= m[:4] <= "2009" for m in months])
    anom = C - C[ref].mean(axis=0)
    anom[:, :, :2, :] = 0.0
    fac = ewh_factor()[None, :, None]
    anom = anom * fac[None]                      # EWH mm coefficients
    diff = (anom[1:] - anom[:-1]) / np.sqrt(2)   # noise proxy

    filters = {"none": lambda c: c}
    for r in (200, 300, 500):
        w = gauss_w(r)[None, :, None]
        filters[f"Gauss{r}"] = (lambda w: (lambda c: c * w))(w)
    for name, suf in DDK.items():
        filters[name] = DDKFilter(raw / "ddk" / f"Wbd_2-120.a_{suf}")

    rows = []
    for fname, flt in filters.items():
        fa = np.array([flt(a) for a in anom])
        fd = np.array([flt(d) for d in diff])
        for bname, (l0, l1) in BANDS.items():
            S_tot = np.median([band_power(a, l0, l1) for a in anom])
            N = np.median([band_power(d, l0, l1) for d in diff])
            S = max(S_tot - N, 0.0)
            S_tot_f = np.median([band_power(a, l0, l1) for a in fa])
            N_f = np.median([band_power(d, l0, l1) for d in fd])
            S_f = max(S_tot_f - N_f, 0.0)
            rows.append((fname, bname, S_f / S if S > 0 else np.nan, S_f / N_f if N_f > 0 else np.nan,
                         np.sqrt(S_f), np.sqrt(N_f)))
    lines = ["filter,band,signal_kept,residual_snr,signal_rms_mm,noise_rms_mm"]
    lines += [f"{f},{b},{k:.3f},{s:.2f},{sr:.1f},{nr:.1f}" for f, b, k, s, sr, nr in rows]
    (out / "filters_by_band.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # maps: one month over the test box, l <= 60 and l <= 96, for DDK2/3/5 and Gauss300
    month = "2010-04"
    a = anom[months.index(month)]
    lat = np.linspace(BOX[1], BOX[0], 61)
    lon = np.linspace(BOX[2], BOX[3], 121)
    lo, la = np.meshgrid(lon, lat)
    names = ["none", "DDK2", "DDK3", "DDK5", "Gauss300"]
    fig, axes = plt.subplots(len(names), 2, figsize=(11, 3.0 * len(names)))
    for i, fname in enumerate(names):
        for j, L in enumerate((60, 96)):
            c = filters[fname](a).copy()
            c[:, L + 1:, :] = 0.0
            sh = pysh.SHCoeffs.from_array(c, normalization="4pi", csphase=1)
            reg = sh.expand(lat=la.ravel(), lon=lo.ravel()).reshape(la.shape)
            v = max(np.percentile(np.abs(reg), 99), 1)
            im = axes[i, j].imshow(reg, extent=(BOX[2], BOX[3], BOX[0], BOX[1]), cmap="RdBu", vmin=-v, vmax=v)
            axes[i, j].set_title(f"{month} {fname}, l ≤ {L}")
            fig.colorbar(im, ax=axes[i, j], shrink=0.8, label="EWH mm")
    fig.tight_layout()
    fig.savefig(out / "maps_ddk_turkiye_box.png", dpi=100)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
