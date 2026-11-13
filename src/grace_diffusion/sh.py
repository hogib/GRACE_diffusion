"""Spherical-harmonic core: reading, anomalies, EWH, filters, tapers, rotations, rung grids.

Coefficient arrays follow pyshtools: shape (2, L+1, L+1), [0] cosine, [1] sine,
4π (geodesy) normalisation, no Condon–Shortley phase (csphase=1), i.e. ICGEM's
convention. Degree l carries detail of half-wavelength ≈ 20,000 km / l.

Rung grids (design doc §2): a field band-limited to degree L is synthesised on a
Driscoll–Healy equiangular grid of 4L × 8L (lat × lon) points, i.e. 4 pixels per
half-wavelength of degree L. That grid is exactly invertible up to degree 2L − 1,
so analysis of a degree-L field on it is exact.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyshtools as pysh
from scipy.spatial.transform import Rotation

A_EARTH = 6378136.3      # m, ITSG reference radius
RHO_EARTH = 5517.0       # kg m^-3, mean density
RHO_WATER = 1000.0
NORM = dict(normalization="4pi", csphase=1)


# ----------------------------------------------------------------------------- reading

def read_gfc(path: str | Path, lmax: int) -> np.ndarray:
    """Read an ICGEM .gfc file, truncated to lmax."""
    c, _gm, _r0 = pysh.shio.read_icgem_gfc(str(path), lmax=lmax)
    return c


@dataclass
class Series:
    months: list[str]          # 'YYYY-MM'
    coeffs: np.ndarray         # (n_months, 2, L+1, L+1)
    source: list[str]          # series name per month

    @property
    def lmax(self) -> int:
        return self.coeffs.shape[-1] - 1


def load_series(raw: str | Path, names=("itsg2018", "itsg_operational"), lmax: int = 120) -> Series:
    """Load monthly solutions of several series, one per month (later series win on overlap)."""
    by_month: dict[str, tuple[str, Path]] = {}
    for name in names:
        for f in sorted(Path(raw, name, f"n{lmax}").glob("*.gfc")):
            by_month[f.stem] = (name, f)
    months = sorted(by_month)
    coeffs = np.array([read_gfc(by_month[m][1], lmax) for m in months])
    return Series(months, coeffs, [by_month[m][0] for m in months])


# ----------------------------------------------------------------------------- anomalies, EWH

def anomalies(s: Series, ref: tuple[str, str] = ("2004", "2009")) -> np.ndarray:
    """Coefficients minus the mean over the reference years; degrees 0 and 1 set to zero."""
    sel = np.array([ref[0] <= m[:4] <= ref[1] for m in s.months])
    if not sel.any():
        raise ValueError("no months in the reference period")
    out = s.coeffs - s.coeffs[sel].mean(axis=0)
    out[:, :, :2, :] = 0.0
    return out


@lru_cache(maxsize=4)
def love_k(lmax: int) -> np.ndarray:
    """Load Love numbers k_l (PREM, Han & Wahr 1995, centre-of-figure frame)."""
    from gravity_toolkit import load_love_numbers

    _h, k, _l = load_love_numbers(lmax, LOVE_NUMBERS=0, REFERENCE="CF")
    return np.asarray(k, dtype=float)


def ewh_factor(lmax: int) -> np.ndarray:
    """Per-degree factor from dimensionless Stokes coefficients to equivalent water height in mm."""
    l = np.arange(lmax + 1)
    return A_EARTH * RHO_EARTH / (3 * RHO_WATER) * (2 * l + 1) / (1 + love_k(lmax)) * 1e3


def to_ewh(c: np.ndarray) -> np.ndarray:
    """Stokes coefficients (..., 2, L+1, L+1) to EWH coefficients in mm."""
    lmax = c.shape[-1] - 1
    return c * ewh_factor(lmax)[:, None]


# ----------------------------------------------------------------------------- filters, tapers

def degree_weights(c: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Multiply coefficients by per-degree weights w (length >= L+1)."""
    lmax = c.shape[-1] - 1
    return c * np.asarray(w)[: lmax + 1, None]


def taper(L: int, lmax: int, flat: float = 0.8) -> np.ndarray:
    """Low-pass weights for 'degree L': 1 up to flat*L, cosine roll-off to 0 at L, 0 beyond.

    The input taper of a rung, taper(L/2), lies inside the flat part of taper(L), so
    lowpass(lowpass(x, L), L/2) == lowpass(x, L/2).
    """
    l = np.arange(lmax + 1, dtype=float)
    w = np.zeros(lmax + 1)
    l0 = flat * L
    w[l <= l0] = 1.0
    roll = (l > l0) & (l < L)
    w[roll] = 0.5 * (1 + np.cos(np.pi * (l[roll] - l0) / (L - l0)))
    return w


def lowpass(c: np.ndarray, L: int, flat: float = 0.8) -> np.ndarray:
    return degree_weights(c, taper(L, c.shape[-1] - 1, flat))


def gaussian_weights(radius_km: float, lmax: int) -> np.ndarray:
    """Normalised Gaussian smoothing weights (W_0 = 1), exponential approximation."""
    b = np.log(2) / (1 - np.cos(radius_km * 1e3 / A_EARTH))
    l = np.arange(lmax + 1)
    return np.exp(-l * (l + 1) / (2 * b))


class DDKFilter:
    """Kusche DDK decorrelation filter (block-diagonal, degrees 2–120), applied to coefficient arrays.

    The matrix files are the Wbd_2-120.a_* files of github.com/strawpants/GRACE-filter.
    """

    FILES = {"DDK1": "1d14p_4", "DDK2": "1d13p_4", "DDK3": "1d12p_4", "DDK4": "5d11p_4",
             "DDK5": "1d11p_4"}

    def __init__(self, path: str | Path):
        import scipy.sparse as sp
        from shxarray.io.binv_legacy import readBINV

        ds = readBINV(str(path))
        n, m = ds["n"].values, ds["m"].values
        if not (np.array_equal(ds["n_"].values, n) and np.array_equal(ds["m_"].values, m)):
            raise ValueError("DDK matrix with different row and column ordering")
        mat = ds["mat"].data
        self.mat = sp.csr_matrix(mat.tocsr() if hasattr(mat, "tocsr") else mat.to_scipy_sparse())
        self.cs, self.l, self.m = (m < 0).astype(int), n, np.abs(m)
        self.lmax = int(n.max())

    @classmethod
    def named(cls, name: str, folder: str | Path) -> "DDKFilter":
        return cls(Path(folder) / f"Wbd_2-120.a_{cls.FILES[name]}")

    def __call__(self, c: np.ndarray) -> np.ndarray:
        if c.shape[-1] - 1 > self.lmax:
            raise ValueError(f"DDK matrix covers degrees up to {self.lmax}")
        ok = self.l <= c.shape[-1] - 1
        out = c.copy()
        v = np.zeros(len(self.l))
        v[ok] = c[self.cs[ok], self.l[ok], self.m[ok]]
        fv = self.mat @ v
        out[self.cs[ok], self.l[ok], self.m[ok]] = fv[ok]
        return out


# ----------------------------------------------------------------------------- rotations

def _angles(rot: Rotation) -> tuple[float, float, float]:
    """pyshtools rotate() angles (deg, convention 'y', space frame) for a scipy rotation.

    rotate(c, *angles) moves the value at point p to rot.apply(p). Determined
    empirically (tests/test_sh.py::test_rotation_moves_points).
    """
    a, b, g = rot.as_euler("zyz", degrees=True)
    return -a, -b, -g


@lru_cache(maxsize=8)
def _dj(lmax: int) -> np.ndarray:
    return pysh.rotate.djpi2(lmax)


def rotate(c: np.ndarray, rot: Rotation) -> np.ndarray:
    """Rotate a field: the value at point p ends up at rot.apply(p)."""
    lmax = c.shape[-1] - 1
    sh = pysh.SHCoeffs.from_array(c, **NORM)
    a, b, g = _angles(rot)
    return sh.rotate(a, b, g, degrees=True, convention="y", body=False, dj_matrix=_dj(lmax)).to_array()


def random_rotation(rng: np.random.Generator) -> Rotation:
    """Uniformly distributed rotation of the sphere."""
    return Rotation.random(random_state=rng)


def rotation_to_equator(lat: float, lon: float) -> Rotation:
    """Rotation moving (lat, lon) to (0, 0) with north kept up along that meridian."""
    return Rotation.from_euler("zy", [-lon, lat], degrees=True)


def xyz(lat, lon) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def latlon(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v)
    return np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1))), np.degrees(np.arctan2(v[..., 1], v[..., 0]))


# ----------------------------------------------------------------------------- rung grids

def grid_shape(L: int) -> tuple[int, int]:
    """(n_lat, n_lon) of the rung grid for target degree L: 4 px per half-wavelength."""
    return 4 * L, 8 * L


def grid_coords(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Latitudes (deg, north to south, from +90) and longitudes (deg, from 0) of the rung grid."""
    nlat, nlon = grid_shape(L)
    return 90.0 - 180.0 * np.arange(nlat) / nlat, 360.0 * np.arange(nlon) / nlon


def synth(c: np.ndarray, L: int) -> np.ndarray:
    """Synthesise coefficients (degrees above 2L-1 are ignored) on the rung grid of degree L."""
    lgrid = 2 * L - 1
    cc = np.zeros((2, lgrid + 1, lgrid + 1))
    k = min(lgrid, c.shape[-1] - 1)
    cc[:, : k + 1, : k + 1] = c[:, : k + 1, : k + 1]
    sh = pysh.SHCoeffs.from_array(cc, **NORM)
    return sh.expand(grid="DH2", extend=False).to_array()


def analyse(grid: np.ndarray, lmax: int | None = None) -> np.ndarray:
    """Spherical-harmonic analysis of a rung grid (exact up to degree n_lat/2 - 1)."""
    g = pysh.SHGrid.from_array(np.asarray(grid, dtype=float), grid="DH")
    nlat = grid.shape[0]
    lcalc = nlat // 2 - 1 if lmax is None else lmax
    return g.expand(**NORM, lmax_calc=lcalc).to_array()


def project_consistent(inp_grid: np.ndarray, out_grid: np.ndarray, L_in: int,
                       flat: float = 0.8) -> np.ndarray:
    """Hard consistency: make lowpass(result, L_in) equal the input exactly.

    The input is w x with w = taper(L_in). The taper is not a projection (w^2 != w in
    the roll-off), so the constraint is set per degree: wherever w_l > 0 the output's
    degree-l coefficients are fixed to input_l / w_l (which equals the true target's,
    because the target taper is 1 there); all degrees with w_l = 0 are left free.
    The operation is a projection: idempotent, and the true target is a fixed point.
    """
    L = out_grid.shape[0] // 4
    c_out = analyse(out_grid)
    c_in = analyse(inp_grid)
    w = taper(L_in, c_out.shape[-1] - 1, flat)
    fixed = np.nonzero(w > 0)[0]
    c_out[:, fixed] = c_in[:, fixed] / w[fixed][:, None]
    return synth(c_out, L)


def degree_variance(c: np.ndarray) -> np.ndarray:
    """Sum over orders of squared coefficients, per degree."""
    return (np.asarray(c) ** 2).sum(axis=(-3, -1))
