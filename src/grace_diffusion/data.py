"""Training and evaluation samples over (month, rung, rotation). Numpy only.

A step is (L, Lin): target degree L, input degree Lin (default L/2; Lin = L/4 spans two
octaves). A sample is built from one month's EWH coefficients:
  1. rotate the coefficients (random for training, test box to the equator for scoring)
  2. input  = synth(lowpass(c, Lin), L)
     target = synth(lowpass(c, L), L)
  3. scale  = area-weighted RMS of the input; input and target are divided by it
  4. weight = area weights x loss mask (test box + ice sheets held out, rotating with the field)
Arrays have shape (1, 4L, 8L), float32. Batches are always a single step, because the
grid size depends on L.

Paired samples (design doc §7.1) take the input from one dataset and the detail from another:
the month's coefficients are spliced, degrees below Lin from the input dataset (GRACE) and
degrees from Lin upward from the target dataset (WaterGAP), and then treated as above. The
input is then exactly the blurred GRACE field, and the target is consistent with it.
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
    # months with both GRACE and WaterGAP; the GRACE / GRACE-FO gap separates val and test
    "paired": {"train": ("2002-04", "2014-12"), "val": ("2015-01", "2017-12"),
               "test": ("2018-01", "2019-12")},
}

Step = tuple[int, int]          # (target degree L, input degree Lin)


def as_step(x) -> Step:
    """16 -> (16, 8); (32, 8) stays."""
    if isinstance(x, (int, np.integer)):
        return int(x), int(x) // 2
    L, Lin = x
    return int(L), int(Lin)


def parse_steps(spec: str) -> tuple[Step, ...]:
    """'8,16' -> ((8, 4), (16, 8)); '8-32' -> ((32, 8),): input degree 8, target degree 32."""
    out = []
    for tok in spec.split(","):
        if "-" in tok:
            lin, L = tok.split("-")
            out.append((int(L), int(lin)))
        else:
            out.append(as_step(int(tok)))
    return tuple(out)


def step_name(step) -> str:
    L, Lin = as_step(step)
    return str(L) if Lin * 2 == L else f"{Lin}-{L}"


def split_months(months: list[str], kind: str, split: str) -> np.ndarray:
    """Indices of months in a split. kind is 'grace', 'twin' or 'paired'."""
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
    Lin: int = 0             # input degree


def area_rms(grid: np.ndarray, L: int) -> float:
    w = grids.area_weights(L)
    return float(np.sqrt((w * grid ** 2).mean()))


def splice_coeffs(c_in: np.ndarray, c_tgt: np.ndarray, Lin: int, L: int) -> np.ndarray:
    """Coefficients to degree L: degrees < Lin from c_in, degrees >= Lin from c_tgt.

    lowpass(result, Lin) is the blurred c_in exactly (the taper is zero from Lin upward), and
    the detail above it comes from c_tgt alone.
    """
    if c_tgt.shape[-1] - 1 < L:
        raise ValueError(f"target coefficients only to degree {c_tgt.shape[-1] - 1}, need {L}")
    if c_in.shape[-1] < Lin:
        raise ValueError(f"input coefficients only to degree {c_in.shape[-1] - 1}, need {Lin - 1}")
    out = c_tgt[:, : L + 1, : L + 1].astype(np.float64).copy()
    out[:, :Lin, :] = 0.0
    out[:, :Lin, :Lin] = c_in[:, :Lin, :Lin]
    return out


def make_sample(c: np.ndarray, month: str, L: int, rot: Rotation | None, *,
                hold_out_test_box: bool = True, mask_ice: bool = True,
                flat: float = 0.8, Lin: int | None = None) -> Sample:
    Lin = L // 2 if Lin is None else Lin
    if c.shape[-1] - 1 < L:
        raise ValueError(f"coefficients only to degree {c.shape[-1] - 1}, rung needs {L}")
    cc = c[:, : L + 1, : L + 1].astype(np.float64)
    if rot is not None:
        cc = sh.rotate(cc, rot)
    inp = sh.synth(sh.lowpass(cc, Lin, flat), L)
    tgt = sh.synth(sh.lowpass(cc, L, flat), L)
    s = area_rms(inp, L)
    if s == 0:
        s = 1.0
    wt = grids.area_weights(L) * grids.loss_mask(L, rot, hold_out_test_box, mask_ice)
    q = (rot or Rotation.identity()).as_quat()
    f32 = lambda a: a[None].astype(np.float32)
    return Sample(f32(inp / s), f32(tgt / s), f32(wt), s, month, L, q, Lin)


def stack(samples: list[Sample]) -> dict[str, np.ndarray]:
    return {
        "inp": np.stack([s.inp for s in samples]),
        "target": np.stack([s.target for s in samples]),
        "weight": np.stack([s.weight for s in samples]),
        "scale": np.array([s.scale for s in samples], dtype=np.float32),
        "L": samples[0].L,
        "Lin": samples[0].Lin,
        "months": [s.month for s in samples],
    }


@dataclass
class Sampler:
    """Draws batches for one split.

    rungs: steps to draw from: target degrees, e.g. (8, 16) for R1+R2, or (L, Lin) pairs.
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
        self.rungs = tuple(as_step(r) for r in self.rungs)
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

    def draw_step(self, rng: np.random.Generator) -> Step:
        return self.rungs[int(rng.choice(len(self.rungs), p=self.rung_probs))]

    def sample(self, i: int, step, rot: Rotation | None) -> Sample:
        """One sample: month index i of this dataset at a step, in a rotated frame."""
        L, Lin = as_step(step)
        return make_sample(self.coeffs[i], self.months[i], L, rot, Lin=Lin,
                           hold_out_test_box=self.hold_out_test_box, mask_ice=self.mask_ice)

    def batch(self, rng: np.random.Generator, batch_size: int, L=None) -> dict:
        step = self.draw_step(rng) if L is None else as_step(L)
        months = rng.choice(self.idx, size=batch_size, replace=True)
        return stack([self.sample(i, step, self.rotation(rng)) for i in months])

    def batches(self, seed: int, batch_size: int, n_batches: int):
        """Reproducible stream of batches: batch k uses rng seeded by (seed, k)."""
        for k in range(n_batches):
            yield self.batch(np.random.default_rng([seed, k]), batch_size)

    def fixed_set(self, L, rotations_per_month: int = 1, seed: int = 0) -> list[Sample]:
        """Deterministic evaluation set: every month of the split at a step (L or (L, Lin))."""
        rng = np.random.default_rng(seed)
        out = []
        for i in self.idx:
            for _ in range(rotations_per_month):
                out.append(self.sample(i, L, self.rotation(rng)))
        return out


def paired_months(months_in: list[str], months_tgt: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Months present in both datasets, with their indices in each."""
    pos = {m: j for j, m in enumerate(months_tgt)}
    both = [(m, i, pos[m]) for i, m in enumerate(months_in) if m in pos]
    return ([b[0] for b in both], np.array([b[1] for b in both], dtype=int),
            np.array([b[2] for b in both], dtype=int))


@dataclass
class PairedSampler(Sampler):
    """Input from one dataset, detail from another, for the months both have.

    months/coeffs are the input dataset (e.g. land-only GRACE); tgt_months/tgt_coeffs the
    target dataset (e.g. WaterGAP). Each sample is built from spliced coefficients
    (see splice_coeffs), so its input is the blurred input-dataset field.
    """
    tgt_months: list[str] = field(default_factory=list)
    tgt_coeffs: np.ndarray | None = None

    def __post_init__(self):
        common, i_in, i_tgt = paired_months(self.months, self.tgt_months)
        self.months = common
        self.coeffs = self.coeffs[i_in]
        self.tgt_coeffs = self.tgt_coeffs[i_tgt]
        super().__post_init__()

    def sample(self, i: int, step, rot: Rotation | None) -> Sample:
        L, Lin = as_step(step)
        c = splice_coeffs(self.coeffs[i], self.tgt_coeffs[i], Lin, L)
        return make_sample(c, self.months[i], L, rot, Lin=Lin,
                           hold_out_test_box=self.hold_out_test_box, mask_ice=self.mask_ice)


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

    def batch(self, rng: np.random.Generator, batch_size: int, L=None) -> dict:
        step = self.samplers[0].draw_step(rng) if L is None else as_step(L)
        which = rng.integers(len(self.samplers), size=batch_size)
        samples = []
        for k in which:
            s = self.samplers[k]
            samples.append(s.sample(rng.choice(s.idx), step, s.rotation(rng)))
        return stack(samples)

    def batches(self, seed: int, batch_size: int, n_batches: int):
        for k in range(n_batches):
            yield self.batch(np.random.default_rng([seed, k]), batch_size)

    def fixed_set(self, L, rotations_per_month: int = 1, seed: int = 0) -> list[Sample]:
        return [x for s in self.samplers for x in s.fixed_set(L, rotations_per_month, seed)]

