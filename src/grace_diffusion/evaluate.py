"""Score generators on a split (design doc §6, §7).

A method is a callable (c, L, members, rng) -> ensemble of target-rung grids (M, 4L, 8L)
in mm, where c is one month's (rotated) EWH coefficients. Scores are per month, on the
added detail band, weighted by pixel area x region mask; results are written per month so
that paired comparisons (block bootstrap over months) can be made later.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import baselines, data, grids, metrics, prep, sh

Method = Callable[[np.ndarray, int, int, np.random.Generator], np.ndarray]


def region_weight(L: int, rot, region: str) -> np.ndarray:
    """'val_land': land outside the test box and ice sheets; 'test_box': the box only."""
    area = grids.area_weights(L)
    if region == "val_land":
        return area * grids.loss_mask(L, rot) * grids.land_mask(L, rot)
    if region == "test_box":
        return area * grids.region_mask(L, rot)
    if region == "global":
        return area
    raise ValueError(region)


def score_month(method: Method, c: np.ndarray, L: int, members: int, rng: np.random.Generator,
                region: str, rot=None) -> dict[str, float]:
    cc = c[:, : L + 1, : L + 1].astype(np.float64)
    if rot is not None:
        cc = sh.rotate(cc, rot)
    truth = sh.synth(sh.lowpass(cc, L), L)
    inp = sh.synth(sh.lowpass(cc, L // 2), L)
    ens = method(cc, L, members, rng)
    w = region_weight(L, rot, region)
    detail_t = truth - inp
    out = {"crps": metrics.crps_fair(ens, truth, w),
           "energy": metrics.energy_score(ens, truth, w),
           "rms_truth_detail": float(np.sqrt((detail_t ** 2 * w).sum() / w.sum()))}
    out.update(metrics.ensemble_stats(ens, truth, w))
    out["corr_mean_detail"] = metrics.correlation(ens.mean(0) - inp, detail_t, w) \
        if np.abs(ens.mean(0) - inp).max() > 0 else np.nan
    out["power_ratio"] = metrics.detail_power_ratio(ens, inp, truth, w)
    out["consistency"] = metrics.consistency_rms(ens[:2], inp, L // 2)
    return out


def run(methods: dict[str, Method], dataset: str, kind: str, split: str, rungs: tuple[int, ...],
        members: int, region: str, out_csv: Path, proc: Path = Path("data/proc"),
        seed: int = 0, rotate: str = "none") -> list[dict]:
    months, coeffs = prep.load(proc, dataset)
    idx = data.split_months(months, kind, split)
    rot = sh.rotation_to_equator(*grids.TEST_BOX.centre) if rotate == "test_box" else None
    rows = []
    for L in rungs:
        for name, method in methods.items():
            t0 = time.time()
            for i in idx:
                rng = np.random.default_rng([seed, L, int(i)])
                r = score_month(method, coeffs[i], L, members, rng, region, rot)
                rows.append({"dataset": dataset, "split": split, "L": L, "method": name,
                             "month": months[i], **r})
            print(f"{dataset} L={L} {name}: {len(idx)} months in {time.time() - t0:.0f}s")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    return rows


def summarise(rows: list[dict]) -> str:
    """Mean over months per (dataset, L, method)."""
    import collections

    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["L"], r["method"])].append(r)
    cols = ["crps", "energy", "rmse_mean", "spread_error", "cov50", "cov90", "corr_mean_detail",
            "power_ratio", "consistency", "rms_truth_detail"]
    lines = ["dataset,L,method,n," + ",".join(cols)]
    for (ds, L, m), g in groups.items():
        vals = [np.nanmean([x.get(c, np.nan) for x in g]) for c in cols]
        lines.append(f"{ds},{L},{m},{len(g)}," + ",".join(f"{v:.4g}" for v in vals))
    return "\n".join(lines)


def baseline_methods(coeffs: np.ndarray, train_idx: np.ndarray, max_L: int) -> dict[str, Method]:
    clim = baselines.climatological_spectrum(coeffs, train_idx, max_L)
    return {
        "input": lambda c, L, M, rng: baselines.input_only(c, L, 1),
        "spectral-gauss": lambda c, L, M, rng: baselines.spectral_gauss(c, L, M, rng),
        "spectral-gauss-clim": lambda c, L, M, rng: baselines.spectral_gauss(c, L, M, rng, spectrum=clim),
    }


if __name__ == "__main__":
    import sys

    # usage: python -m grace_diffusion.evaluate <dataset> <kind> <split> <rungs comma> <members> <out.csv>
    ds, kind, split, rungs, members, out = sys.argv[1:7]
    rungs = tuple(int(x) for x in rungs.split(","))
    months, coeffs = prep.load(Path("data/proc"), ds)
    methods = baseline_methods(coeffs, data.split_months(months, kind, "train"), max(rungs))
    rows = run(methods, ds, kind, split, rungs, int(members), "val_land", Path(out))
    print(summarise(rows))
