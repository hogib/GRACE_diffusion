"""Loss masks and pixel weights on rung grids, in any rotated frame.

Regions are defined in geographic coordinates. A training field is rotated by
`rot` (the value at geographic point p ends up at rot.apply(p)), so a grid point
q of the rotated field shows geographic point rot.inv().apply(q). Masks are
evaluated there, which makes them rotate with the field.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from . import sh

KM_PER_DEG = 111.195


@dataclass(frozen=True)
class Box:
    lat0: float
    lat1: float
    lon0: float
    lon1: float

    def contains(self, lat, lon, margin_km: float = 0.0):
        """True where (lat, lon) lies in the box expanded by margin_km."""
        dlat = margin_km / KM_PER_DEG
        coslat = np.cos(np.radians(np.clip(np.abs(lat), 0, 89.0)))
        dlon = margin_km / (KM_PER_DEG * coslat)
        lon = (np.asarray(lon) - self.lon0 + 180.0) % 360.0 - 180.0   # relative to lon0, in [-180, 180)
        width = (self.lon1 - self.lon0) % 360.0
        return ((lat >= self.lat0 - dlat) & (lat <= self.lat1 + dlat)
                & (lon >= -dlon) & (lon <= width + dlon))

    @property
    def centre(self) -> tuple[float, float]:
        return 0.5 * (self.lat0 + self.lat1), self.lon0 + 0.5 * ((self.lon1 - self.lon0) % 360.0)


# Design doc: test box 30–45° N, 20–50° E.
TEST_BOX = Box(30.0, 45.0, 20.0, 50.0)
# Ice sheets, kept out of the loss (design doc §4). Simple geographic outlines.
GREENLAND = Box(59.0, 84.0, -75.0, -10.0)
ANTARCTICA_LAT = -60.0


def half_wavelength_km(L: int) -> float:
    return 20000.0 / L


def rotated_coords(L: int, rot: Rotation | None) -> tuple[np.ndarray, np.ndarray]:
    """Geographic (lat, lon) of every pixel of the rung-L grid of a field rotated by rot."""
    lat, lon = sh.grid_coords(L)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    if rot is None:
        return LAT, LON
    p = rot.inv().apply(sh.xyz(LAT.ravel(), LON.ravel()))
    glat, glon = sh.latlon(p)
    return glat.reshape(LAT.shape), glon.reshape(LAT.shape)


def area_weights(L: int) -> np.ndarray:
    """Pixel area weights of the rung grid (cos latitude in the grid's own frame), mean 1."""
    lat, _ = sh.grid_coords(L)
    w = np.cos(np.radians(lat))[:, None] * np.ones(sh.grid_shape(L))
    return w / w.mean()


def loss_mask(L: int, rot: Rotation | None, hold_out_test_box: bool = True,
              mask_ice: bool = True, margin_km: float | None = None) -> np.ndarray:
    """1 where a pixel counts in the training loss, 0 where it is held out.

    The margin defaults to one half-wavelength of the target degree L, so that the
    held-out region is not leaked through detail generated right next to it.
    """
    glat, glon = rotated_coords(L, rot)
    m = np.ones(glat.shape, dtype=bool)
    margin = half_wavelength_km(L) if margin_km is None else margin_km
    if hold_out_test_box:
        m &= ~TEST_BOX.contains(glat, glon, margin)
    if mask_ice:
        m &= ~GREENLAND.contains(glat, glon, margin)
        m &= ~(glat <= ANTARCTICA_LAT + margin / KM_PER_DEG)
    return m.astype(np.float32)


def region_mask(L: int, rot: Rotation | None, box: Box = TEST_BOX) -> np.ndarray:
    """1 inside the box (no margin), for scoring."""
    glat, glon = rotated_coords(L, rot)
    return box.contains(glat, glon).astype(np.float32)
