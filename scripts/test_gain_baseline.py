"""Baseline "model pattern, GRACE amplitude" (ideas list C1). No training, CPU only.

For each model and each pixel, a gain is fitted so that the model's blurred field matches
GRACE's blurred field over the fitting months:  gain(x) = sum_t G(x,t) M(x,t) / sum_t M(x,t)^2
(both limited to the input degree). On other months, the model's detail for the month is
multiplied by that gain and added to the GRACE input. The ten models give a ten-member ensemble,
scored against GRACE's own detail like the networks.

Variants:
  gain          per-pixel gain fitted over the fitting months
  gain-global   one gain per model and month: RMS of the GRACE input over RMS of the model input
                (the 'models-month-scaled' of test_model_month_detail.py, for reference)
  gain-both     per-pixel gain, then rescaled per month so that the gained model input has the
                RMS of the GRACE input
  *-mean        the ten-member mean as a single prediction

  uv run python scripts/test_gain_baseline.py --out runs/eval/gain_val.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from grace_diffusion import data, evaluate, prep, sh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="grace_ddk5_land")
    p.add_argument("--kind", default="paired")
    p.add_argument("--fit", default="train")
    p.add_argument("--split", default="val")
    p.add_argument("--rungs", default="16,32")
    p.add_argument("--clip", type=float, default=4.0, help="gains are limited to [0, clip]")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    proc = Path("data/proc")
    gm, gc = prep.load(proc, a.dataset)
    names = sorted(f.stem for f in proc.glob("isimip_*.npz"))
    models = {n: prep.load(proc, n) for n in names}
    pos = {n: {m: j for j, m in enumerate(models[n][0])} for n in names}
    common = set(gm).intersection(*[set(models[n][0]) for n in names])
    fit = [i for i in data.split_months(gm, a.kind, a.fit) if gm[i] in common]
    idx = [i for i in data.split_months(gm, a.kind, a.split) if gm[i] in common]
    print(f"{len(names)} models; fit on {len(fit)} months, score {len(idx)} months "
          f"{gm[idx[0]]}..{gm[idx[-1]]}", flush=True)
    rows = []
    for L, Lin in data.parse_steps(a.rungs):
        blur = lambda c: sh.synth(sh.lowpass(c[:, : L + 1, : L + 1].astype(np.float64), Lin), L)
        num = {n: 0.0 for n in names}
        den = {n: 0.0 for n in names}
        for i in fit:
            G = blur(gc[i])
            for n in names:
                M = blur(models[n][1][pos[n][gm[i]]])
                num[n] = num[n] + G * M
                den[n] = den[n] + M * M
        gain = {n: np.clip(num[n] / np.maximum(den[n], 1e-12), 0.0, a.clip) for n in names}
        w = evaluate.region_weight(L, None, "val_land")
        print(f"L={L}: median gain over land per model: "
              + ", ".join(f"{np.median(gain[n][w > 0]):.2f}" for n in names), flush=True)

        def finish(inp, detail):
            """Detail multiplied on the grid: limit it to degree L again, keep the input exact."""
            c = sh.lowpass(sh.analyse(detail)[:, : L + 1, : L + 1], L)
            return sh.project_consistent(inp, inp + sh.synth(c, L), Lin)

        for i in idx:
            g = gc[i].astype(np.float64)
            inp = blur(g)
            s_g = data.area_rms(inp, L)
            ens = {"gain": [], "gain-global": [], "gain-both": []}
            for n in names:
                mc = models[n][1][pos[n][gm[i]]].astype(np.float64)
                M = blur(mc)
                d = sh.synth(sh.lowpass(data.splice_coeffs(g, mc, Lin, L), L), L) - inp
                s_m = data.area_rms(M, L)
                s_gm = data.area_rms(gain[n] * M, L)
                ens["gain"].append(finish(inp, gain[n] * d))
                ens["gain-global"].append(finish(inp, d * (s_g / s_m if s_m > 0 else 1.0)))
                ens["gain-both"].append(finish(inp, gain[n] * d * (s_g / s_gm if s_gm > 0 else 1.0)))
            for name, e in ens.items():
                e = np.stack(e)
                for label, arr in ((name, e), (name + "-mean", e.mean(0)[None])):
                    r = evaluate.score_month(lambda *_, arr=arr: arr, g, L, len(arr),
                                             np.random.default_rng(0), "val_land", None, Lin)
                    rows.append({"dataset": a.dataset, "split": a.split, "L": L, "Lin": Lin,
                                 "method": label, "month": gm[i], **r})
        print(f"L={L} done", flush=True)
    keys = list(rows[0])
    for r in rows:
        keys += [k for k in r if k not in keys]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(evaluate.summarise(rows))


if __name__ == "__main__":
    main()
