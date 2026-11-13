"""Build cached coefficient series (EWH mm anomalies) from the raw downloads.

Outputs in data/proc/:
  grace_raw.npz    ITSG unfiltered, degrees <= 64
  grace_ddk5.npz   ITSG DDK5-filtered, degrees <= 64
  twin.npz         WaterGAP v2.2e TWS, degrees <= 128
Each holds `months` ('YYYY-MM' strings) and `coeffs` (n, 2, L+1, L+1, float32),
anomalies relative to the 2004–2009 mean, degrees 0 and 1 set to zero.

GRACE corrections applied before anomalies (design doc §4): C20, and C30 where
given, replaced by GSFC TN-14 SLR values; GIA removed with the Caron et al. (2018)
expectation model (JPL VESL). Degree 1 (geocentre) is not observed and stays zero.
The twin needs neither correction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyshtools as pysh

from . import sh

GRACE_LMAX = 64
TWIN_LMAX = 128
TWIN_FILE = "watergap/watergap22e_gswp3-w5e5_tws_histsoc_monthly_1901_2019.nc"


TN14_FILE = "aux/TN-14_C30_C20_GSFC_SLR.txt"
GIA_FILE = "aux/exp_Stokes_Caron_et_al_2018"


def build_grace(raw: Path, out: Path) -> None:
    s = sh.load_series(raw)
    missing = sh.replace_low_degrees(s, sh.read_tn14(raw / TN14_FILE))
    if missing:
        print(f"no TN-14 C20 for {len(missing)} months (GRACE value kept): {', '.join(missing)}")
    sh.remove_gia(s, sh.read_gia_rates(raw / GIA_FILE, s.lmax))
    anom = sh.to_ewh(sh.anomalies(s))
    ddk5 = sh.DDKFilter.named("DDK5", raw / "ddk")
    filt = np.array([ddk5(a) for a in anom])
    k = GRACE_LMAX + 1
    for name, arr in (("grace_raw", anom), ("grace_ddk5", filt)):
        np.savez(out / f"{name}.npz", months=np.array(s.months),
                 coeffs=arr[:, :, :k, :k].astype(np.float32), source=np.array(s.source))


def watergap_to_dh(field: np.ndarray) -> np.ndarray:
    """WaterGAP 0.5° cell-centred grid (lat 89.75…-89.75, lon -179.75…179.75) to a DH
    grid of 360 × 720 (lat 90…-89.5, lon 0…359.5).

    Ocean (NaN) becomes 0. Points of the DH grid sit on cell corners, so each is the
    mean of the four surrounding cells (a 0.5° box-car, negligible below degree 128).
    """
    f = np.nan_to_num(field.astype(np.float64), nan=0.0)
    f = np.roll(f, 360, axis=1)                    # lon now 0.25 … 359.75
    left = np.roll(f, 1, axis=1)                   # cell at lon - 0.5
    f = 0.5 * (f + left)                           # value at lon 0, 0.5, … (corners)
    pad = np.vstack([f[:1], f])                    # row above the first (north pole row)
    dh = 0.5 * (pad[:-1] + pad[1:])                # lat 90, 89.5, … -89.5
    return dh


def runaway_cells(tws, threshold_mm: float = 1000.0) -> np.ndarray:
    """Cells whose storage drifts by more than threshold_mm between 1901–1910 and 2010–2019.

    In WaterGAP v2.2e these are mostly glacier cells where snow accumulates without
    bound (~2,000 cells, 90% at high latitude) and city cells with point-like
    groundwater abstraction (New York −100 m, Los Angeles −30 m, ...). Left in, they
    would dominate the twin's fine-scale structure. Design doc §4.
    """
    early = tws.sel(time=slice("1901", "1910")).mean("time").values
    late = tws.sel(time=slice("2010", "2019")).mean("time").values
    return np.abs(late - early) > threshold_mm


def build_twin(raw: Path, out: Path, threshold_mm: float = 1000.0) -> None:
    import xarray as xr

    ds = xr.open_dataset(raw / TWIN_FILE)
    tws = ds["tws"]
    bad = runaway_cells(tws, threshold_mm)
    months = [f"{t.year:04d}-{t.month:02d}" for t in tws["time"].values]
    coeffs = np.empty((len(months), 2, TWIN_LMAX + 1, TWIN_LMAX + 1), dtype=np.float64)
    for i in range(len(months)):
        f = tws.isel(time=i).values.copy()
        f[bad] = np.nan                                # excluded cells count as zero anomaly
        dh = watergap_to_dh(f)
        coeffs[i] = pysh.SHGrid.from_array(dh, grid="DH").expand(
            **sh.NORM, lmax_calc=TWIN_LMAX).to_array()
    ref = np.array(["2004" <= m[:4] <= "2009" for m in months])
    coeffs -= coeffs[ref].mean(axis=0)
    coeffs[:, :, :2, :] = 0.0
    np.savez(out / "twin.npz", months=np.array(months), coeffs=coeffs.astype(np.float32))


def build_landmask(raw: Path, out: Path) -> None:
    """Land mask on WaterGAP's 0.5° cell-centred grid (cells with any valid value)."""
    import xarray as xr

    ds = xr.open_dataset(raw / TWIN_FILE)
    valid = np.isfinite(ds["tws"].isel(time=0).values)
    np.savez(out / "landmask.npz", land=valid, lat=ds["lat"].values, lon=ds["lon"].values)


def load(proc: Path, name: str) -> tuple[list[str], np.ndarray]:
    d = np.load(Path(proc) / f"{name}.npz")
    return [str(m) for m in d["months"]], d["coeffs"]


if __name__ == "__main__":
    import sys

    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    which = sys.argv[3:] or ["grace", "twin", "land"]
    if "grace" in which:
        build_grace(raw, out)
    if "land" in which:
        build_landmask(raw, out)
    if "twin" in which:
        build_twin(raw, out)
