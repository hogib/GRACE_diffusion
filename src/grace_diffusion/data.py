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

Paired samples (the paired experiment, design doc Appendix B.5) take the input from one dataset and the detail from another:
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


# ------------------------------------------------------------------ extra input channels

@dataclass(frozen=True)
class Extras:
    """Extra input channels next to the blurred field (all in the sample's rotated frame).

    hist    lags in months: the blurred field of month t - lag, in the units of the current
            input (so changes in amplitude are visible). Zero where that month is missing.
    models  the hydrological models' detail for the same month: mean over the models (the other
            models, when the sample itself is a model) of each model's detail divided by the
            RMS of its own blurred field. One channel, dimensionless like the network's output.
    clim    the same from each model's mean seasonal cycle (calendar-month means over the
            training years minus the annual mean). Needs no model run for the month itself.
    land    the land mask.
    lat     sine of the geographic latitude (a plain indication of climate zone).
    drop    in training, each group of channels is zeroed with this probability, so the network
            also learns to work without it (missing months, months after the model runs end).
    """
    hist: tuple[int, ...] = ()
    models: bool = False
    clim: bool = False
    land: bool = False
    lat: bool = False
    drop: float = 0.0

    @property
    def n(self) -> int:
        return len(self.hist) + int(self.models) + int(self.clim) + int(self.land) + int(self.lat)

    @property
    def needs_bank(self) -> bool:
        return self.models or self.clim


def parse_extras(spec: str | None, drop: float = 0.0) -> Extras:
    """'hist:1,2,3,6,12+models+land' -> Extras. Empty or None: no extra channels."""
    kw: dict = {"drop": drop}
    for tok in (spec or "").split("+"):
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("hist:"):
            kw["hist"] = tuple(int(x) for x in tok[5:].split(","))
        elif tok in ("models", "clim", "land", "lat"):
            kw[tok] = True
        else:
            raise ValueError(f"unknown extra input '{tok}'")
    return Extras(**kw)


def month_shift(month: str, lag: int) -> str:
    k = int(month[:4]) * 12 + int(month[5:7]) - 1 - lag
    return f"{k // 12:04d}-{k % 12 + 1:02d}"


def sh_rms(c: np.ndarray) -> float:
    """Global RMS of a field from its (4-pi normalised) coefficients; rotation invariant."""
    return float(np.sqrt((np.asarray(c, dtype=np.float64) ** 2).sum()))


class ModelBank:
    """The hydrological models' coefficients by month, for the 'models' and 'clim' channels."""

    def __init__(self, sets: dict[str, tuple[list[str], np.ndarray]], lmax: int,
                 clim_years: tuple[str, str] = ("1901", "1989")):
        self.names = sorted(sets)
        self.lmax = lmax
        self.coeffs = {n: np.asarray(sets[n][1][:, :, : lmax + 1, : lmax + 1], dtype=np.float32)
                       for n in self.names}
        self.pos = {n: {m: j for j, m in enumerate(sets[n][0])} for n in self.names}
        self.clim = {}
        for n in self.names:
            months = sets[n][0]
            cm = np.zeros((12, 2, lmax + 1, lmax + 1))
            for k in range(12):
                sel = [j for j, m in enumerate(months)
                       if int(m[5:7]) == k + 1 and clim_years[0] <= m[:4] <= clim_years[1]]
                cm[k] = self.coeffs[n][sel].mean(0)
            self.clim[n] = cm - cm.mean(0, keepdims=True)

    @classmethod
    def load(cls, proc, lmax: int, pattern: str = "isimip_*.npz") -> "ModelBank":
        from pathlib import Path

        from . import prep
        names = sorted(f.stem for f in Path(proc).glob(pattern))
        if not names:
            raise FileNotFoundError(f"no {pattern} in {proc}")
        return cls({n: prep.load(Path(proc), n) for n in names}, lmax)

    def _mean_detail(self, fields: list[np.ndarray], Lin: int, L: int) -> np.ndarray | None:
        acc, k = np.zeros((2, L + 1, L + 1)), 0
        for c in fields:
            c = c[:, : L + 1, : L + 1].astype(np.float64)
            s = sh_rms(sh.lowpass(c, Lin))
            if s > 0:
                acc += (sh.lowpass(c, L) - sh.lowpass(c, Lin)) / s
                k += 1
        return acc / k if k else None

    def month_detail(self, month: str, Lin: int, L: int, exclude: str = "") -> np.ndarray | None:
        """Coefficients of the mean scaled detail of the models (but `exclude`) for a month."""
        return self._mean_detail([self.coeffs[n][self.pos[n][month]] for n in self.names
                                  if n != exclude and month in self.pos[n]], Lin, L)

    def clim_detail(self, month: str, Lin: int, L: int, exclude: str = "") -> np.ndarray | None:
        k = int(month[5:7]) - 1
        return self._mean_detail([self.clim[n][k] for n in self.names if n != exclude], Lin, L)


class NoiseBank:
    """Draws of GRACE-like input noise for training on model fields (ideas list B4).

    coeffs: (K, 2, l+1, l+1) noise draws in mm, processed like the GRACE dataset
    (prep.GraceNoise). ref[Lin][k]: RMS of the GRACE field of that draw's month limited to Lin.
    A draw is scaled so that noise relative to the input is what it is for GRACE: models are
    weaker than GRACE, so unscaled noise would be too strong for them.
    """

    def __init__(self, coeffs: np.ndarray, ref: dict[int, np.ndarray]):
        self.coeffs, self.ref = coeffs, ref

    def draw(self, rng: np.random.Generator, c: np.ndarray, Lin: int, fmax: float) -> np.ndarray:
        """Noise coefficients (2, Lin+1, Lin+1) for a sample with coefficients c: a random
        draw times a factor uniform in [0, fmax], scaled to the sample's input RMS."""
        k = int(rng.integers(len(self.coeffs)))
        s_model = sh_rms(sh.lowpass(c[:, : Lin + 1, : Lin + 1].astype(np.float64), Lin))
        f = rng.uniform(0.0, fmax) * s_model / self.ref[Lin][k]
        return f * self.coeffs[k][:, : Lin + 1, : Lin + 1].astype(np.float64)


def extra_grids(ex: Extras, L: int, Lin: int, rot: Rotation | None, scale: float, *,
                month: str, name: str = "", prev=None, bank: ModelBank | None = None,
                rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """The extra channels of one sample, in the order hist, models, clim, land, lat.

    prev(lag) returns the coefficients of month t - lag or None. scale is the RMS of the
    current input. With rng (training), groups are dropped with probability ex.drop.
    """
    shape = sh.grid_shape(L)
    keep = lambda: rng is None or ex.drop <= 0 or rng.random() >= ex.drop
    turn = lambda c: c if rot is None else sh.rotate(c, rot)
    out = []
    if ex.hist:
        on = keep()
        for lag in ex.hist:
            c = prev(lag) if (on and prev is not None) else None
            if c is None:
                out.append(np.zeros(shape))
            else:
                c = c[:, : L + 1, : L + 1].astype(np.float64)
                out.append(sh.synth(sh.lowpass(turn(c), Lin), L) / scale)
    for flag, fn in ((ex.models, "month_detail"), (ex.clim, "clim_detail")):
        if flag:
            c = getattr(bank, fn)(month, Lin, L, exclude=name) if keep() else None
            out.append(np.zeros(shape) if c is None else sh.synth(turn(c), L))
    if ex.land:
        out.append(grids.land_mask(L, rot) if keep() else np.zeros(shape))
    if ex.lat:
        out.append(np.sin(np.radians(grids.rotated_coords(L, rot)[0])) if keep() else np.zeros(shape))
    return out


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
    inp: np.ndarray          # (1 + extra channels, H, W): normalised input, then the extras
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
                flat: float = 0.8, Lin: int | None = None, extra=None,
                noise: np.ndarray | None = None) -> Sample:
    """extra: callable(scale) -> list of grids appended to the input as further channels.
    noise: coefficients (unrotated, to degree >= Lin) added to the input only: the input
    becomes lowpass(c + noise, Lin) while the detail to generate stays that of c."""
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
    if noise is not None:
        nn = noise[:, : Lin + 1, : Lin + 1].astype(np.float64)
        n_grid = sh.synth(sh.lowpass(nn if rot is None else sh.rotate(nn, rot), Lin, flat), L)
        inp, tgt = inp + n_grid, tgt + n_grid
    wt = grids.area_weights(L) * grids.loss_mask(L, rot, hold_out_test_box, mask_ice)
    q = (rot or Rotation.identity()).as_quat()
    f32 = lambda a: a[None].astype(np.float32)
    inp_ch = np.stack([inp / s, *(extra(s) if extra else [])]).astype(np.float32)
    return Sample(inp_ch, f32(tgt / s), f32(wt), s, month, L, q, Lin)


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
    extras: Extras | None = None    # extra input channels
    bank: ModelBank | None = None   # needed for the 'models' and 'clim' channels
    name: str = ""                  # dataset name (a model is left out of its own 'models' channel)
    noise_bank: NoiseBank | None = None   # GRACE-like input noise, training samples only
    noise_max: float = 0.0                # largest noise factor (times the formal errors)
    idx: np.ndarray = field(init=False)

    def __post_init__(self):
        self.rungs = tuple(as_step(r) for r in self.rungs)
        self._pos = {m: j for j, m in enumerate(self.months)}
        if self.extras is not None and self.extras.needs_bank and self.bank is None:
            raise ValueError("extra inputs 'models'/'clim' need a ModelBank")
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

    def prev(self, i: int):
        """prev(lag) -> coefficients of the month `lag` months before month i, or None."""
        def get(lag):
            j = self._pos.get(month_shift(self.months[i], lag))
            return None if j is None else self.coeffs[j]
        return get

    def extra(self, i: int, L: int, Lin: int, rot, rng=None):
        if self.extras is None or self.extras.n == 0:
            return None
        return lambda scale: extra_grids(self.extras, L, Lin, rot, scale, month=self.months[i],
                                         name=self.name, prev=self.prev(i), bank=self.bank, rng=rng)

    def sample(self, i: int, step, rot: Rotation | None, rng=None) -> Sample:
        """One sample: month index i of this dataset at a step, in a rotated frame.

        rng: given in training, so that extra channels are dropped at random (Extras.drop).
        """
        L, Lin = as_step(step)
        noise = None
        if rng is not None and self.noise_bank is not None and self.noise_max > 0:
            noise = self.noise_bank.draw(rng, self.coeffs[i], Lin, self.noise_max)
        return make_sample(self.coeffs[i], self.months[i], L, rot, Lin=Lin,
                           hold_out_test_box=self.hold_out_test_box, mask_ice=self.mask_ice,
                           extra=self.extra(i, L, Lin, rot, rng), noise=noise)

    def batch(self, rng: np.random.Generator, batch_size: int, L=None) -> dict:
        step = self.draw_step(rng) if L is None else as_step(L)
        months = rng.choice(self.idx, size=batch_size, replace=True)
        drop = rng if self.split == "train" else None
        return stack([self.sample(i, step, self.rotation(rng), drop) for i in months])

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

    def sample(self, i: int, step, rot: Rotation | None, rng=None) -> Sample:
        if self.extras is not None and self.extras.n:
            raise NotImplementedError("extra inputs with paired samples")
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
            samples.append(s.sample(rng.choice(s.idx), step, s.rotation(rng),
                                    rng if s.split == "train" else None))
        return stack(samples)

    def batches(self, seed: int, batch_size: int, n_batches: int):
        for k in range(n_batches):
            yield self.batch(np.random.default_rng([seed, k]), batch_size)

    def fixed_set(self, L, rotations_per_month: int = 1, seed: int = 0) -> list[Sample]:
        return [x for s in self.samplers for x in s.fixed_set(L, rotations_per_month, seed)]

