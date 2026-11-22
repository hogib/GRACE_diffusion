"""Grade detail generated for degrees 33-64 against what DDK5-filtered GRACE retains there
(ideas list B1a; signal check in design doc B.12).

GRACE's field above degree 32 is not the truth: the DDK5 filter has removed most of it and
noise remains. A generated field is therefore first treated like GRACE (DDK5 filter, land
mask), and only then compared with GRACE in the band of the step 32 -> 64:

  raw        the band of the generated field as it is          (for reference)
  filtered   the band after DDK5 and the land mask             (the comparison that counts)

Methods: the baselines of evaluate.py, checkpoints (--ckpt name=path), and with --models the
ten hydrological models' own fields for the month as a ten-member ensemble ("models-month";
only months the simulations cover).

  uv run python scripts/score_like_grace.py --first 2004-01 --last 2010-12 --stride 2 \
      --baselines --models --ckpt net=runs/std-r4/best.pt --out runs/eval/like_grace_2004.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from grace_diffusion import data, evaluate, grids, metrics, prep, sh

L, LIN, LM = 64, 32, 128


def band(c):
    c = c[:, : L + 1, : L + 1]
    return sh.synth(sh.lowpass(c, L) - sh.lowpass(c, LIN), L)


def wrms(x, w):
    return float(np.sqrt((w * x * x).sum() / w.sum()))


def score(bands: np.ndarray, truth: np.ndarray, w: np.ndarray) -> dict[str, float]:
    mean = bands.mean(0)
    out = {"corr": metrics.correlation(mean, truth, w) if np.abs(mean).max() > 0 else np.nan,
           "crps": metrics.crps_fair(bands, truth, w),
           "rms_member": float(np.mean([wrms(b, w) for b in bands])),
           "rms_truth": wrms(truth, w)}
    out.update(metrics.ensemble_stats(bands, truth, w))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="grace_ddk5_land")
    p.add_argument("--first", default="2004-01")
    p.add_argument("--last", default="2010-12")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--members", type=int, default=16)
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--models", action="store_true")
    p.add_argument("--ckpt", action="append", default=[])
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--churn", type=float, default=20.0)
    p.add_argument("--region", default="val_land")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    proc, raw = Path("data/proc"), Path("data/raw")
    months, coeffs = prep.load(proc, a.dataset)
    pos = {m: j for j, m in enumerate(months)}
    idx = [i for i, m in enumerate(months) if a.first <= m <= a.last][:: a.stride]

    ddk = sh.DDKFilter.named("DDK5", raw / "ddk")
    mask = grids.land_mask(LM, None, str(proc)) * grids.loss_mask(
        LM, None, hold_out_test_box=False, mask_ice=True, margin_km=0.0)
    w = evaluate.region_weight(L, None, a.region)

    def like_grace(c):                                # coefficients -> DDK5 -> land mask
        f = ddk(c)[:, : L + 1, : L + 1]
        f = sh.analyse(sh.synth(f, LM) * mask, lmax=L)
        f[:, :2, :] = 0.0
        return f

    methods: dict[str, evaluate.Method] = {}
    if a.baselines:
        train = data.split_months(months, "grace", "train")
        methods.update(evaluate.baseline_methods(coeffs, train, L))
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        methods[name] = evaluate.model_method(path, steps=a.steps, chunk=a.chunk, churn=a.churn)
    models = {}
    if a.models:
        for f in sorted(proc.glob("isimip_*.npz")):
            mm, mc = prep.load(proc, f.stem)
            models[f.stem] = ({m: j for j, m in enumerate(mm)}, mc)

    def prev_of(i):
        def get(lag):
            j = pos.get(data.month_shift(months[i], lag))
            return None if j is None else coeffs[j]
        return get

    rows = []

    def add(name, month, raw_bands, filt_bands, truth):
        for kind, b in (("raw", raw_bands), ("filtered", filt_bands)):
            rows.append({"method": name, "kind": kind, "month": month, **score(b, truth, w)})

    t0 = time.time()
    for i in idx:
        month = months[i]
        g = coeffs[i].astype(np.float64)
        truth = band(g)
        cc = g[:, : L + 1, : L + 1]
        for name, method in methods.items():
            rng = np.random.default_rng([a.seed, L, int(i)])
            ctx = {"month": month, "name": a.dataset, "prev": prev_of(i),
                   "noise_seed": (a.seed, L, LIN)}
            ens = method(cc, L, a.members, rng, LIN, ctx)
            xc = [sh.analyse(e, lmax=L) for e in ens]
            add(name, month, np.array([band(x) for x in xc]),
                np.array([band(like_grace(x)) for x in xc]), truth)
        have = [n for n, (mp, _) in models.items() if month in mp]
        if have:
            mcs = [models[n][1][models[n][0][month]].astype(np.float64) for n in have]
            add("models-month", month, np.array([band(m) for m in mcs]),
                np.array([band(like_grace(m[:, :121, :121])) for m in mcs]), truth)
        print(f"{month} {time.time() - t0:.0f}s", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    cols = ["corr", "crps", "rmse_mean", "spread_error", "rms_member", "rms_truth"]
    print(f"\n{a.dataset} {LIN} -> {L}, {len(idx)} months {months[idx[0]]}..{months[idx[-1]]}")
    print("method,kind,n," + ",".join(cols))
    for name in dict.fromkeys(r["method"] for r in rows):
        for kind in ("raw", "filtered"):
            sel = [r for r in rows if r["method"] == name and r["kind"] == kind]
            vals = [np.nanmean([r.get(c, np.nan) for r in sel]) for c in cols]
            print(f"{name},{kind},{len(sel)}," + ",".join(f"{v:.4g}" for v in vals))


if __name__ == "__main__":
    main()
