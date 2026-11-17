"""Training and evaluation samples over (month, rung, rotation). Numpy only.

A sample for rung target degree L (input degree L/2) is built from one month's EWH
coefficients:
  1. rotate the coefficients (random for training, test box to the equator for scoring)
  2. input  = synth(lowpass(c, L/2), L)
     target = synth(lowpass(c, L), L)
  3. scale  = area-weighted RMS of the input; input and target are divided by it
  4. weight = area weights x loss mask (test box + ice sheets held out, rotating with the field)
Arrays have shape (1, 4L, 8L), float32. Batches are always a single rung, because the
grid size depends on the rung.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from . import grids, sh

# Block splits by month, inclusive (design doc §4). Never random months.
SPLITS = {
    "grace": {"train": ("2002-04", "2019-12"), "val": ("2020-01", "2022-12"),
              "test": ("2023-01", "9999-12")},
    "twin": {"train": ("1901-01", "1989-12"), "val": ("1990-01", "2001-12"),
             "test": ("2002-01", "2019-12")},
}


def split_months(months: list[str], kind: str, split: str) -> np.ndarray:
    """Indices of months in a split. kind is 'grace' or 'twin'."""
    lo, hi = SPLITS[kind][split]
    return np.array([i for i, m in enumerate(months) if lo <= m <= hi], dtype=int)


@dataclass
class Sample:
    inp: np.ndarray          # (1, H, W) normalised input
    target: np.ndarray       # (1, H, W) normalised target
    weight: np.ndarray       # (1, H, W) area weight x loss mask
    scale: float             # multiply by this to get EWH mm
    month: str
    L: int
    rotation: np.ndarray     # quaternion (x, y, z, w)


def area_rms(grid: np.ndarray, L: int) -> float:
    w = grids.area_weights(L)
    return float(np.sqrt((w * grid ** 2).mean()))


def make_sample(c: np.ndarray, month: str, L: int, rot: Rotation | None, *,
                hold_out_test_box: bool = True, mask_ice: bool = True,
                flat: float = 0.8) -> Sample:
    if c.shape[-1] - 1 < L:
        raise ValueError(f"coefficients only to degree {c.shape[-1] - 1}, rung needs {L}")
    cc = c[:, : L + 1, : L + 1].astype(np.float64)
    if rot is not None:
        cc = sh.rotate(cc, rot)
    inp = sh.synth(sh.lowpass(cc, L // 2, flat), L)
    tgt = sh.synth(sh.lowpass(cc, L, flat), L)
    s = area_rms(inp, L)
    if s == 0:
        s = 1.0
    wt = grids.area_weights(L) * grids.loss_mask(L, rot, hold_out_test_box, mask_ice)
    q = (rot or Rotation.identity()).as_quat()
    f32 = lambda a: a[None].astype(np.float32)
    return Sample(f32(inp / s), f32(tgt / s), f32(wt), s, month, L, q)


def stack(samples: list[Sample]) -> dict[str, np.ndarray]:
    return {
        "inp": np.stack([s.inp for s in samples]),
        "target": np.stack([s.target for s in samples]),
        "weight": np.stack([s.weight for s in samples]),
        "scale": np.array([s.scale for s in samples], dtype=np.float32),
        "L": samples[0].L,
        "months": [s.month for s in samples],
    }


@dataclass
class Sampler:
    """Draws batches for one split.

    rungs: target degrees to draw from, e.g. (8, 16) for training on R1+R2.
    rung_probs: probability of each rung per batch (default uniform).
    rotate: 'random' (uniform on the sphere), 'polar' (about the polar axis only),
            'none', or 'test_box' (test box to the equator; for scoring).
    """
    months: list[str]
    coeffs: np.ndarray
    kind: str                       # 'grace' or 'twin'
    split: str                      # 'train', 'val' or 'test'
    rungs: tuple[int, ...]
    rotate: str = "random"
    rung_probs: tuple[float, ...] | None = None
    hold_out_test_box: bool = True
    mask_ice: bool = True
    idx: np.ndarray = field(init=False)

    def __post_init__(self):
        self.idx = split_months(self.months, self.kind, self.split)
        if len(self.idx) == 0:
            raise ValueError(f"no months in {self.kind}/{self.split}")
        if self.rotate not in ("random", "polar", "none", "test_box"):
            raise ValueError(self.rotate)

    def rotation(self, rng: np.random.Generator) -> Rotation | None:
        if self.rotate == "random":
            return sh.random_rotation(rng)
        if self.rotate == "polar":
            return Rotation.from_euler("z", rng.uniform(0, 360), degrees=True)
        if self.rotate == "test_box":
            return sh.rotation_to_equator(*grids.TEST_BOX.centre)
        return None

    def batch(self, rng: np.random.Generator, batch_size: int, L: int | None = None) -> dict:
        if L is None:
            L = int(rng.choice(self.rungs, p=self.rung_probs))
        months = rng.choice(self.idx, size=batch_size, replace=True)
        samples = [make_sample(self.coeffs[i], self.months[i], L, self.rotation(rng),
                               hold_out_test_box=self.hold_out_test_box, mask_ice=self.mask_ice)
                   for i in months]
        return stack(samples)

    def batches(self, seed: int, batch_size: int, n_batches: int):
        """Reproducible stream of batches: batch k uses rng seeded by (seed, k)."""
        for k in range(n_batches):
            yield self.batch(np.random.default_rng([seed, k]), batch_size)

    def fixed_set(self, L: int, rotations_per_month: int = 1, seed: int = 0) -> list[Sample]:
        """Deterministic evaluation set: every month of the split at rung L."""
        rng = np.random.default_rng(seed)
        out = []
        for i in self.idx:
            for _ in range(rotations_per_month):
                out.append(make_sample(self.coeffs[i], self.months[i], L, self.rotation(rng),
                                       hold_out_test_box=self.hold_out_test_box,
                                       mask_ice=self.mask_ice))
        return out


class MultiSampler:
    """Several datasets (e.g. hydrological models) behind the Sampler interface.

    Each sample of a batch comes from a uniformly chosen dataset; a batch has one rung.
    fixed_set concatenates every dataset's evaluation set.
    """

    def __init__(self, samplers: list[Sampler]):
        if not samplers:
            raise ValueError("no samplers")
        rungs = {s.rungs for s in samplers}
        if len(rungs) != 1:
            raise ValueError(f"samplers disagree on rungs: {rungs}")
        self.samplers = samplers
        self.rungs = samplers[0].rungs
        self.rung_probs = samplers[0].rung_probs

    def batch(self, rng: np.random.Generator, batch_size: int, L: int | None = None) -> dict:
        if L is None:
            L = int(rng.choice(self.rungs, p=self.rung_probs))
        which = rng.integers(len(self.samplers), size=batch_size)
        samples = []
        for k in which:
            s = self.samplers[k]
            i = rng.choice(s.idx)
            samples.append(make_sample(s.coeffs[i], s.months[i], L, s.rotation(rng),
                                       hold_out_test_box=s.hold_out_test_box,
                                       mask_ice=s.mask_ice))
        return stack(samples)

    def batches(self, seed: int, batch_size: int, n_batches: int):
        for k in range(n_batches):
            yield self.batch(np.random.default_rng([seed, k]), batch_size)

    def fixed_set(self, L: int, rotations_per_month: int = 1, seed: int = 0) -> list[Sample]:
        return [x for s in self.samplers for x in s.fixed_set(L, rotations_per_month, seed)]

