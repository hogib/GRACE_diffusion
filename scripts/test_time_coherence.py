"""Are generated ensemble members coherent from month to month?

Each month is generated independently. This compares, for consecutive months, how much the
generated detail of one member changes with how much GRACE's real detail changes:
  independent   fresh noise every month (the current behaviour)
  shared        member j uses the same noise in every month
for the stochastic and the deterministic sampler. Reports the RMS month-to-month change of the
detail (mm) and the correlation between consecutive months' detail, for the truth and for
single members, and the CRPS (which shared noise should leave unchanged).

  uv run python scripts/test_time_coherence.py runs/isimip10-r23/best.pt --out runs/eval/coherence.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from grace_diffusion import baselines, data, evaluate, metrics, prep, sh


def wcorr(a, b, w):
    ma, mb = (w * a).sum() / w.sum(), (w * b).sum() / w.sum()
    return float((w * (a - ma) * (b - mb)).sum()
                 / np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--dataset", default="grace_ddk5_land")
    p.add_argument("--kind", default="grace")
    p.add_argument("--split", default="val")
    p.add_argument("--rungs", default="16,32")
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    months, coeffs = prep.load(Path("data/proc"), a.dataset)
    idx = data.split_months(months, a.kind, a.split)
    ym = lambda s: int(s[:4]) * 12 + int(s[5:7])
    rows = []
    for L, Lin in data.parse_steps(a.rungs):
        w = evaluate.region_weight(L, None, "val_land")
        rms = lambda x: float(np.sqrt((w * x ** 2).sum() / w.sum()))
        truth, base = [], []
        for i in idx:
            g = coeffs[i][:, : L + 1, : L + 1].astype(np.float64)
            base.append(baselines.input_only(g, L, 1, Lin=Lin)[0])
            truth.append(sh.synth(sh.lowpass(g, L), L) - base[-1])
        pairs = [k for k in range(len(idx) - 1) if ym(months[idx[k + 1]]) - ym(months[idx[k]]) == 1]
        t_change = np.mean([rms(truth[k + 1] - truth[k]) for k in pairs])
        t_corr = np.mean([wcorr(truth[k + 1], truth[k], w) for k in pairs])
        rows.append({"L": L, "variant": "truth (GRACE)", "change_rms": t_change, "lag1_corr": t_corr,
                     "detail_rms": np.mean([rms(t) for t in truth]), "crps": np.nan})
        for churn, sname in ((20.0, "stochastic"), (0.0, "deterministic")):
            net = evaluate.model_method(a.ckpt, churn=churn)
            for noise in ("independent", "shared"):
                det = []
                for k, i in enumerate(idx):
                    g = coeffs[i][:, : L + 1, : L + 1].astype(np.float64)
                    rng = np.random.default_rng([7, L] if noise == "shared" else [7, L, int(i)])
                    det.append(net(g, L, a.members, rng, Lin) - base[k][None])
                det = np.stack(det)                                  # months, members, H, W
                ch = np.mean([rms(det[k + 1, j] - det[k, j]) for k in pairs for j in range(a.members)])
                co = np.mean([wcorr(det[k + 1, j], det[k, j], w) for k in pairs for j in range(a.members)])
                cr = np.mean([metrics.crps_fair(det[k], truth[k], w) for k in range(len(idx))])
                rows.append({"L": L, "variant": f"{sname}, {noise} noise", "change_rms": ch,
                             "lag1_corr": co, "detail_rms": np.mean([rms(det[k, j]) for k in range(len(idx))
                                                                     for j in range(a.members)]),
                             "crps": cr})
                print(rows[-1], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n{'L':>3} {'variant':34s} {'change':>7} {'lag-1 corr':>10} {'detail rms':>10} {'crps':>6}")
    for r in rows:
        print(f"{r['L']:>3} {r['variant']:34s} {r['change_rms']:7.1f} {r['lag1_corr']:10.2f} "
              f"{r['detail_rms']:10.1f} {r['crps']:6.1f}")


if __name__ == "__main__":
    main()
