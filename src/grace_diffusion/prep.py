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


def build_grace_sigma(raw: Path, out: Path, lmax: int = 120) -> None:
    """Formal errors of the monthly solutions as EWH coefficients (mm), unfiltered, to lmax.

    Written as grace_sigma.npz (months, sigma). Used to perturb the GRACE input of ensemble
    members (GraceNoise); degrees 0 and 1 are zero, as in the fields.
    """
    import pyshtools as pysh

    months, sig = [], []
    for name in ("itsg2018", "itsg_operational"):
        for f in sorted(Path(raw, name, f"n{lmax}").glob("*.gfc")):
            _c, _gm, _r0, err = pysh.shio.read_icgem_gfc(str(f), lmax=lmax, errors="formal")
            months.append(f.stem)
            sig.append(sh.to_ewh(err))
    order = np.argsort(months)
    sig = np.array(sig)[order]
    sig[:, :, :2, :] = 0.0
    np.savez(out / "grace_sigma.npz", months=np.array(months)[order], sigma=sig.astype(np.float32))


class GraceNoise:
    """Draws of GRACE's coefficient noise, processed like a GRACE dataset.

    A draw is independent Gaussian noise with the month's formal errors, passed through the
    DDK5 filter (for 'ddk5' datasets) and the land-only masking (for '_land' datasets), both
    of which are linear, and truncated to the dataset's degree.
    """

    def __init__(self, proc: Path, raw: Path, dataset: str, lmax: int = 64, grid_L: int = 64):
        from . import grids

        d = np.load(Path(proc) / "grace_sigma.npz")
        self.sigma = {str(m): s for m, s in zip(d["months"], d["sigma"])}
        self.ddk = sh.DDKFilter.named("DDK5", Path(raw) / "ddk") if "ddk5" in dataset else None
        self.lmax, self.grid_L = lmax, grid_L
        self.mask = None
        if dataset.endswith("_land"):
            self.mask = grids.land_mask(grid_L, None, str(proc)) * grids.loss_mask(
                grid_L, None, hold_out_test_box=False, mask_ice=True, margin_km=0.0)

    def draw(self, month: str, rng: np.random.Generator) -> np.ndarray | None:
        if month not in self.sigma:
            return None
        n = rng.standard_normal(self.sigma[month].shape) * self.sigma[month]
        if self.ddk is not None:
            n = self.ddk(n)
        n = n[:, : self.lmax + 1, : self.lmax + 1]
        if self.mask is not None:
            n = sh.analyse(sh.synth(n, self.grid_L) * self.mask, lmax=self.lmax)
        n[:, :2, :] = 0.0
        return n


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


def build_model(path: Path, out_file: Path, threshold_mm: float = 1000.0) -> dict:
    """One hydrological model's monthly TWS (0.5 deg cell-centred, variable 'tws', mm) to
    spherical-harmonic anomalies to degree TWIN_LMAX, stored like twin.npz.

    Runaway cells (see runaway_cells) count as zero anomaly; anomalies are relative to
    2004-2009, as for GRACE; degrees 0 and 1 are zeroed.
    """
    import xarray as xr

    ds = xr.open_dataset(path, use_cftime=True)
    tws = ds["tws"]
    assert tws.shape[1:] == (360, 720), tws.shape
    assert float(ds["lat"][0]) > 0 and float(ds["lon"][0]) < 0, "unexpected grid orientation"
    bad = runaway_cells(tws, threshold_mm)
    months = [f"{t.year:04d}-{t.month:02d}" for t in tws["time"].values]
    coeffs = np.empty((len(months), 2, TWIN_LMAX + 1, TWIN_LMAX + 1), dtype=np.float64)
    for i in range(len(months)):
        f = tws.isel(time=i).values.astype(np.float64)
        f[bad] = np.nan                                # excluded cells count as zero anomaly
        dh = watergap_to_dh(f)
        coeffs[i] = pysh.SHGrid.from_array(dh, grid="DH").expand(
            **sh.NORM, lmax_calc=TWIN_LMAX).to_array()
    ref = np.array(["2004" <= m[:4] <= "2009" for m in months])
    coeffs -= coeffs[ref].mean(axis=0)
    coeffs[:, :, :2, :] = 0.0
    np.savez(out_file, months=np.array(months), coeffs=coeffs.astype(np.float32))
    return {"months": len(months), "runaway_cells": int(bad.sum()),
            "land_cells": int(np.isfinite(tws.isel(time=0).values).sum())}


def build_twin(raw: Path, out: Path, threshold_mm: float = 1000.0) -> None:
    build_model(raw / TWIN_FILE, out / "twin.npz", threshold_mm)


def build_isimip(raw: Path, out: Path, threshold_mm: float = 1000.0) -> None:
    """Every ISIMIP3a TWS file in raw/isimip/ to out/isimip_<model>.npz."""
    for path in sorted((raw / "isimip").glob("*_tws_global_monthly_*.nc")):
        model = path.name.split("_")[0]
        target = out / f"isimip_{model}.npz"
        if target.exists():
            continue
        info = build_model(path, target, threshold_mm)
        print(model, info, flush=True)


def build_landmask(raw: Path, out: Path) -> None:
    """Land mask on WaterGAP's 0.5° cell-centred grid (cells with any valid value)."""
    import xarray as xr

    ds = xr.open_dataset(raw / TWIN_FILE)
    valid = np.isfinite(ds["tws"].isel(time=0).values)
    np.savez(out / "landmask.npz", land=valid, lat=ds["lat"].values, lon=ds["lon"].values)


def land_only(proc: Path, src: str, dst: str, mask_ice: bool = True) -> None:
    """A dataset with ocean (and, by default, Greenland and Antarctica) set to zero, like
    the hydrological models: synthesise each month on a fine grid, mask, re-expand to the
    source degree, zero degrees 0 and 1. Input and truth are then both built from the
    masked field, so the experiment stays consistent (design doc §4).
    """
    from . import grids

    months, coeffs = load(proc, src)
    lmax = coeffs.shape[-1] - 1
    L = 128                                           # 512 x 1024 grid, exact to degree 255
    mask = grids.land_mask(L, None, str(proc))
    if mask_ice:
        mask = mask * grids.loss_mask(L, None, hold_out_test_box=False, mask_ice=True,
                                      margin_km=0.0)
    out = np.empty_like(coeffs)
    for i in range(len(months)):
        g = sh.synth(coeffs[i].astype(np.float64), L) * mask
        out[i] = sh.analyse(g, lmax=lmax)
    out[:, :, :2, :] = 0.0
    np.savez(Path(proc) / f"{dst}.npz", months=np.array(months), coeffs=out.astype(np.float32))


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
    if "isimip" in which:
        build_isimip(raw, out)
    if "grace_sigma" in which:
        build_grace_sigma(raw, out)
    if "grace_land" in which:
        for name in ("grace_ddk5", "grace_raw"):
            land_only(out, name, f"{name}_land")
