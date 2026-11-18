"""Do the hydrological models' own fields for a month say anything about GRACE's detail?

For each month, the GRACE field limited to the input degree is completed with each model's
detail for that same month (spliced coefficients: GRACE below the input degree, the model from
there up). The ten completions are scored as a ten-member ensemble against GRACE's own detail,
exactly like the networks. No training.

Variants:
  models-month         the ten models' detail as it is
  models-month-scaled  each model's detail scaled by rms(GRACE input) / rms(model input), the
                       same normalisation the networks use
  mean / mean-scaled   the ten-model mean as a single prediction
  watergap             WaterGAP alone as a single prediction

  uv run python scripts/test_model_month_detail.py --dataset grace_ddk5_land --split val \
      --out runs/eval/models_month_val.csv
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
    p.add_argument("--split", default="val")
    p.add_argument("--rungs", default="16,32")
    p.add_argument("--region", default="val_land")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    proc = Path("data/proc")
    gm, gc = prep.load(proc, a.dataset)
    names = sorted(f.stem for f in proc.glob("isimip_*.npz"))
    models = {n: prep.load(proc, n) for n in names}
    wg = next(n for n in names if "watergap" in n)
    common = set(gm)
    for m, _ in models.values():
        common &= set(m)
    idx = [i for i in data.split_months(gm, a.kind, a.split) if gm[i] in common]
    pos = {n: {m: j for j, m in enumerate(models[n][0])} for n in names}
    print(f"{len(names)} models, {len(idx)} months {gm[idx[0]]}..{gm[idx[-1]]}", flush=True)

    rows = []
    for L, Lin in data.parse_steps(a.rungs):
        for i in idx:
            g = gc[i].astype(np.float64)
            inp = sh.synth(sh.lowpass(g[:, : L + 1, : L + 1], Lin), L)
            s_g = data.area_rms(inp, L)
            raw, scaled = {}, {}
            for n in names:
                mc = models[n][1][pos[n][gm[i]]].astype(np.float64)
                full = sh.synth(sh.lowpass(data.splice_coeffs(g, mc, Lin, L), L), L)
                d = full - inp
                s_m = data.area_rms(sh.synth(sh.lowpass(mc[:, : L + 1, : L + 1], Lin), L), L)
                raw[n], scaled[n] = inp + d, inp + d * (s_g / s_m if s_m > 0 else 1.0)
            ens_raw, ens_sc = np.stack(list(raw.values())), np.stack(list(scaled.values()))
            methods = {
                "models-month": lambda *_: ens_raw,
                "models-month-scaled": lambda *_: np.stack(
                    [sh.project_consistent(inp, e, Lin) for e in ens_sc]),
                "mean": lambda *_: ens_raw.mean(0)[None],
                "mean-scaled": lambda *_: sh.project_consistent(inp, ens_sc.mean(0), Lin)[None],
                "watergap": lambda *_: raw[wg][None],
            }
            for name, meth in methods.items():
                r = evaluate.score_month(meth, g, L, 10, np.random.default_rng(0), a.region, None, Lin)
                rows.append({"dataset": a.dataset, "split": a.split, "L": L, "Lin": Lin,
                             "method": name, "month": gm[i], **r})
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
