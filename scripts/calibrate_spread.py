"""Fit the spread of a diffusion checkpoint: stochastic sampler (churn) and inflation k.

For each churn value, sample one ensemble per month (no inflation), then score it for a grid
of inflation factors k (members stretched away from the ensemble mean by k). Prints mean
CRPS, spread/error and 90% coverage per (rung, churn, k), and the k that minimises CRPS.

Fit on months the checkpoint did not train on and that are not the validation years used
for the gate, e.g. GRACE training years for a network trained only on models, or the
models' own held-out years:

  uv run python scripts/calibrate_spread.py runs/isimip10-r23/best.pt \
      --dataset grace_ddk5_land --kind grace --split train --rungs 16,32 \
      --months 36 --stride 4 --churn 0,10,40 --out runs/eval/cal_grace_train.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from grace_diffusion import data, evaluate, metrics, prep, sh

KS = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--dataset", required=True, help="comma-separated; months pooled")
    p.add_argument("--kind", default="grace")
    p.add_argument("--split", default="train")
    p.add_argument("--rungs", default="16,32")
    p.add_argument("--months", type=int, default=36, help="per dataset")
    p.add_argument("--stride", type=int, default=4, help="take every n-th month of the split")
    p.add_argument("--churn", default="0")
    p.add_argument("--members", type=int, default=16)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--region", default="val_land")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    rows = []
    for churn in (float(x) for x in a.churn.split(",")):
        method = evaluate.model_method(a.ckpt, steps=a.steps, churn=churn)
        for ds in a.dataset.split(","):
            months, coeffs = prep.load(Path("data/proc"), ds)
            idx = data.split_months(months, a.kind, a.split)[:: a.stride][: a.months]
            for L in (int(x) for x in a.rungs.split(",")):
                w = evaluate.region_weight(L, None, a.region)
                t0 = time.time()
                for i in idx:
                    cc = coeffs[i][:, : L + 1, : L + 1].astype(np.float64)
                    truth = sh.synth(sh.lowpass(cc, L), L)
                    rng = np.random.default_rng([0, L, int(i)])
                    ens = method(cc, L, a.members, rng)
                    mean = ens.mean(0)
                    for k in KS:
                        e = mean[None] + k * (ens - mean[None])
                        st = metrics.ensemble_stats(e, truth, w)
                        rows.append({"dataset": ds, "L": L, "churn": churn, "k": k,
                                     "month": months[i],
                                     "crps": metrics.crps_fair(e, truth, w),
                                     "spread_error": st["spread_error"],
                                     "cov50": st["cov50"], "cov90": st["cov90"]})
                print(f"{ds} L={L} churn={churn:g}: {len(idx)} months in {time.time() - t0:.0f}s",
                      flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    print("\nL  churn  k     crps   spread/err  cov50  cov90")
    keys = sorted({(r["L"], r["churn"], r["k"]) for r in rows})
    best = {}
    for key in keys:
        sel = [r for r in rows if (r["L"], r["churn"], r["k"]) == key]
        m = {c: float(np.mean([r[c] for r in sel])) for c in ("crps", "spread_error", "cov50", "cov90")}
        print(f"{key[0]:<3d}{key[1]:5g}  {key[2]:.1f}  {m['crps']:7.2f}  {m['spread_error']:6.2f}"
              f"     {m['cov50']:.2f}   {m['cov90']:.2f}")
        b = best.get(key[:2])
        if b is None or m["crps"] < b[1]:
            best[key[:2]] = (key[2], m["crps"])
    print("\nbest k by CRPS:", {f"L={L} churn={c:g}": f"k={k} ({v:.2f})" for (L, c), (k, v) in best.items()})


if __name__ == "__main__":
    main()
