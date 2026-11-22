"""Is there a signal in DDK5-filtered GRACE at degrees 33-64 that generated detail could be
graded against (ideas list B1a)? No training.

For each month and step (16 -> 32 as the reference, 32 -> 64 as the question):
  grace      the detail of land-only DDK5 GRACE in the band
  noise      the same band of draws of GRACE's formal errors, filtered and masked like the data
  model      each hydrological model's detail in the band, as it is
  model-f    the same after the model field is passed through DDK5 and the land mask, i.e.
             what GRACE's processing would leave of it

Printed: RMS of each over the scoring region; the share of GRACE's band variance that the
formal errors explain, and the correlation ceiling that follows from it; the correlation of
GRACE's detail with the filtered and unfiltered model detail (single models and their mean).

  uv run python scripts/test_band_signal.py --out runs/eval/band_signal.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from grace_diffusion import data, evaluate, grids, metrics, prep, sh


def wrms(x, w):
    return float(np.sqrt((w * x * x).sum() / w.sum()))


SHARP: dict[int, tuple[int, int]] = {}                # L -> (lo, hi): strict degree band


def band(c, L, Lin):
    c = c[:, : L + 1, : L + 1]
    if L in SHARP:                                    # degrees lo..hi only, no taper zone
        lo, hi = SHARP[L]
        out = np.zeros_like(c)
        out[:, lo: hi + 1, :] = c[:, lo: hi + 1, :]
        return sh.synth(out, L)
    return sh.synth(sh.lowpass(c, L) - sh.lowpass(c, Lin), L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="grace_ddk5_land")
    p.add_argument("--rungs", default="32,64")
    p.add_argument("--first", default="2002-04")
    p.add_argument("--last", default="2017-06")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--draws", type=int, default=4)
    p.add_argument("--sharp", default="", help="strict degree bands, e.g. '32:17-25,64:33-60'; "
                   "the default band of a step includes its taper zone (0.8 Lin .. Lin)")
    p.add_argument("--region", default="val_land")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    for part in filter(None, a.sharp.split(",")):
        k, v = part.split(":")
        SHARP[int(k)] = tuple(int(x) for x in v.split("-"))
    proc, raw = Path("data/proc"), Path("data/raw")
    gm, gc = prep.load(proc, a.dataset)
    names = sorted(f.stem for f in proc.glob("isimip_*.npz"))
    models = {n: prep.load(proc, n) for n in names}
    pos = {n: {m: j for j, m in enumerate(models[n][0])} for n in names}
    common = set(gm)
    for n in names:
        common &= set(models[n][0])
    idx = [i for i, m in enumerate(gm) if a.first <= m <= a.last and m in common][:: a.stride]
    print(f"{len(names)} models, {len(idx)} months {gm[idx[0]]}..{gm[idx[-1]]}", flush=True)

    ddk = sh.DDKFilter.named("DDK5", raw / "ddk")
    LM = 128                                          # masking grid, as in prep.land_only
    mask = grids.land_mask(LM, None, str(proc)) * grids.loss_mask(
        LM, None, hold_out_test_box=False, mask_ice=True, margin_km=0.0)
    noise = prep.GraceNoise(proc, raw, a.dataset)
    steps = data.parse_steps(a.rungs)
    wts = {L: evaluate.region_weight(L, None, a.region) for L, _ in steps}

    def like_grace(mc):                               # model field -> DDK5 -> land mask
        f = ddk(mc[:, :121, :121])[:, :65, :65]
        f = sh.analyse(sh.synth(f, LM) * mask, lmax=64)
        f[:, :2, :] = 0.0
        return f

    rows = []
    for i in idx:
        month = gm[i]
        g = gc[i].astype(np.float64)
        rng = np.random.default_rng([7, i])
        draws = [noise.draw(month, rng) for _ in range(a.draws)]
        mcs = {n: models[n][1][pos[n][month]].astype(np.float64) for n in names}
        mfs = {n: like_grace(mcs[n]) for n in names}
        for L, Lin in steps:
            w = wts[L]
            dg = band(g, L, Lin)
            r = {"month": month, "L": L, "rms_grace": wrms(dg, w),
                 "rms_noise": float(np.sqrt(np.mean([wrms(band(d, L, Lin), w) ** 2 for d in draws])))}
            dm = {n: band(mcs[n], L, Lin) for n in names}
            df = {n: band(mfs[n], L, Lin) for n in names}
            r["rms_model"] = float(np.mean([wrms(dm[n], w) for n in names]))
            r["rms_model_f"] = float(np.mean([wrms(df[n], w) for n in names]))
            r["corr_model"] = float(np.mean([metrics.correlation(dg, dm[n], w) for n in names]))
            r["corr_model_f"] = float(np.mean([metrics.correlation(dg, df[n], w) for n in names]))
            r["corr_mean"] = metrics.correlation(dg, np.mean(list(dm.values()), 0), w)
            r["corr_mean_f"] = metrics.correlation(dg, np.mean(list(df.values()), 0), w)
            r["corr_between_models_f"] = float(np.mean(
                [metrics.correlation(df[x], df[y], w) for k, x in enumerate(names) for y in names[k + 1:]]))
            rows.append(r)
        print(month, flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    cols = [c for c in rows[0] if c not in ("month", "L")]
    for label, lo, hi in (("all", "0", "9"), ("2004-2010", "2004-01", "2010-12"),
                          ("2015-2017", "2015-01", "2017-12")):
        for L, Lin in steps:
            sel = [r for r in rows if r["L"] == L and lo <= r["month"] <= hi]
            if not sel:
                continue
            m = {c: float(np.mean([r[c] for r in sel])) for c in cols}
            share = float(np.mean([r["rms_noise"] ** 2 for r in sel]) / np.mean([r["rms_grace"] ** 2 for r in sel]))
            print(f"\n{label}, {Lin} -> {L}, {len(sel)} months")
            print(f"  rms: grace {m['rms_grace']:.1f}  formal-error noise {m['rms_noise']:.1f}  "
                  f"model {m['rms_model']:.1f}  model filtered {m['rms_model_f']:.1f}")
            print(f"  noise share of GRACE band variance {share:.2f}; correlation ceiling "
                  f"{np.sqrt(max(0.0, 1 - share)):.2f}")
            print(f"  corr with GRACE: single model {m['corr_model']:.2f}, filtered {m['corr_model_f']:.2f}; "
                  f"model mean {m['corr_mean']:.2f}, filtered {m['corr_mean_f']:.2f}; "
                  f"between filtered models {m['corr_between_models_f']:.2f}")


if __name__ == "__main__":
    main()
