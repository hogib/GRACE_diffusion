"""Does the models' detail for the month add anything to the network's detail?

For each month and step, against GRACE's own detail d (the part the input does not determine):
  n  the network's ensemble-mean detail (GRACE-only input)
  m  the ten-model mean detail for that month, scaled to the GRACE input (see
     test_model_month_detail.py)
Fits d ~ a*n + b*m with pooled weights, cross-fitted over two halves of the months (fit on one
half, apply to the other), and reports correlation and RMSE of n, m and the mix, and the CRPS
of the network ensemble with its mean replaced by the mix.

  uv run python scripts/test_mixing.py runs/isimip10-r23/best.pt --out runs/eval/mixing_val.csv
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
    p.add_argument("--kind", default="paired")
    p.add_argument("--split", default="val")
    p.add_argument("--rungs", default="16,32")
    p.add_argument("--members", type=int, default=32)
    p.add_argument("--churn", type=float, default=20.0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    proc = Path("data/proc")
    gm, gc = prep.load(proc, a.dataset)
    names = sorted(f.stem for f in proc.glob("isimip_*.npz"))
    models = {n: prep.load(proc, n) for n in names}
    pos = {n: {mo: j for j, mo in enumerate(models[n][0])} for n in names}
    common = set(gm).intersection(*[set(models[n][0]) for n in names])
    idx = [i for i in data.split_months(gm, a.kind, a.split) if gm[i] in common]
    net = evaluate.model_method(a.ckpt, churn=a.churn)
    rows = []
    for L, Lin in data.parse_steps(a.rungs):
        w = evaluate.region_weight(L, None, "val_land")
        D, N, M, E, B = [], [], [], [], []
        for i in idx:
            g = gc[i][:, : L + 1, : L + 1].astype(np.float64)
            inp = sh.synth(sh.lowpass(g, Lin), L)
            base = baselines.input_only(g, L, 1, Lin=Lin)[0]       # everything the input determines
            truth = sh.synth(sh.lowpass(g, L), L)
            ens = net(g, L, a.members, np.random.default_rng([0, L, int(i)]), Lin)
            s_g = data.area_rms(inp, L)
            md = []
            for n in names:
                mc = models[n][1][pos[n][gm[i]]].astype(np.float64)
                full = sh.synth(sh.lowpass(data.splice_coeffs(g, mc, Lin, L), L), L)
                s_m = data.area_rms(sh.synth(sh.lowpass(mc[:, : L + 1, : L + 1], Lin), L), L)
                md.append((full - base) * (s_g / s_m if s_m > 0 else 1.0))
            D.append(truth - base); N.append(ens.mean(0) - base); M.append(np.mean(md, 0))
            E.append(ens); B.append(base)
        print(f"L={L}: {len(idx)} months sampled", flush=True)
        half = np.arange(len(idx)) < len(idx) // 2
        for fit, app in ((half, ~half), (~half, half)):
            f = np.nonzero(fit)[0]
            X = np.stack([np.concatenate([(np.sqrt(w) * N[k]).ravel() for k in f]),
                          np.concatenate([(np.sqrt(w) * M[k]).ravel() for k in f])], 1)
            y = np.concatenate([(np.sqrt(w) * D[k]).ravel() for k in f])
            (ca, cb), *_ = np.linalg.lstsq(X, y, rcond=None)
            ca1 = float((X[:, 0] @ y) / (X[:, 0] @ X[:, 0]))       # network alone, rescaled
            for k in np.nonzero(app)[0]:
                mix = ca * N[k] + cb * M[k]
                rm = lambda x: float(np.sqrt((w * (x - D[k]) ** 2).sum() / w.sum()))
                truth = B[k] + D[k]
                shift = lambda new: E[k] - E[k].mean(0)[None] + (B[k] + new)[None]
                rows.append({
                    "L": L, "month": gm[idx[k]], "a": ca, "b": cb, "a_alone": ca1,
                    "corr_net": wcorr(N[k], D[k], w), "corr_models": wcorr(M[k], D[k], w),
                    "corr_mix": wcorr(mix, D[k], w), "corr_net_models": wcorr(N[k], M[k], w),
                    "rmse_net": rm(N[k]), "rmse_net_rescaled": rm(ca1 * N[k]),
                    "rmse_models": rm(M[k]), "rmse_mix": rm(mix),
                    "crps_net": metrics.crps_fair(E[k], truth, w),
                    "crps_net_rescaled": metrics.crps_fair(shift(ca1 * N[k]), truth, w),
                    "crps_mix": metrics.crps_fair(shift(mix), truth, w)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for L in sorted({r["L"] for r in rows}):
        sel = [r for r in rows if r["L"] == L]
        print(f"\nL={L}  " + "  ".join(f"{k}={np.mean([r[k] for r in sel]):.3f}"
                                       for k in rows[0] if k not in ("L", "month")))


if __name__ == "__main__":
    main()
