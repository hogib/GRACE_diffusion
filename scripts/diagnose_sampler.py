"""Why is the diffusion model weaker than `regress` at its own training rungs?

For each validation month (unrotated, loss-mask x area weights, normalised units) compare
against the true detail d = target - input:
  regress           the deterministic U-Net
  D(sigma)          the diffusion denoiser applied to pure noise of level sigma: at large
                    sigma this is the model's own estimate of E[d | input]
  mean@S            the mean of M sampled members with an S-step Heun sampler
  member@S          a single member
Reports weighted correlation and RMSE relative to the RMS of d (1 = no better than zero).

  uv run python scripts/diagnose_sampler.py runs/diffusion-ddk5-r12/best.pt \
      runs/regress-ddk5-r12/best.pt --dataset grace_ddk5 --kind grace --rungs 8,16,32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from grace_diffusion import data, prep
from grace_diffusion.model.edm import seeded_generator
from grace_diffusion.train import load_checkpoint


def scores(pred, d, w):
    """Weighted correlation and relative RMSE, pooled over the batch."""
    mp, md = (w * pred).sum() / w.sum(), (w * d).sum() / w.sum()
    cov = (w * (pred - mp) * (d - md)).sum()
    corr = cov / np.sqrt((w * (pred - mp) ** 2).sum() * (w * (d - md) ** 2).sum())
    rel = np.sqrt((w * (pred - d) ** 2).sum() / (w * d ** 2).sum())
    return corr, rel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("diffusion")
    p.add_argument("regress", nargs="?")
    p.add_argument("--dataset", default="grace_ddk5")
    p.add_argument("--kind", default="grace")
    p.add_argument("--rungs", default="8,16,32")
    p.add_argument("--members", type=int, default=32)
    p.add_argument("--steps", default="32,128")
    p.add_argument("--max-months", type=int, default=36)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diff, _ = load_checkpoint(a.diffusion, dev)
    reg = load_checkpoint(a.regress, dev)[0] if a.regress else None
    months, coeffs = prep.load(Path("data/proc"), a.dataset)
    for L in (int(x) for x in a.rungs.split(",")):
        val = data.Sampler(months, coeffs, a.kind, "val", rungs=(L,), rotate="none")
        samples = val.fixed_set(L)[: a.max_months]
        b = data.stack(samples)
        cond = torch.from_numpy(b["inp"]).to(dev)
        d = b["target"] - b["inp"]
        w = b["weight"]
        rows = []
        with torch.no_grad():
            if reg is not None:
                rows.append(("regress", reg(cond).cpu().numpy()))
            g = seeded_generator(dev, 0, L)
            for s in (80.0, 10.0, 2.0, 0.5):
                x = torch.randn(cond.shape, generator=g, device=dev) * s
                sig = torch.full((cond.shape[0],), s, device=dev)
                rows.append((f"D(sigma={s:g})", diff.denoise(x, sig, cond).cpu().numpy()))
        for S in (int(x) for x in a.steps.split(",")):
            ens = []
            for k in range(a.members):
                ens.append(diff.sample(cond, steps=S, generator=g).cpu().numpy())
            ens = np.stack(ens)
            rows.append((f"member@{S}", ens[0]))
            rows.append((f"mean@{S}", ens.mean(0)))
            spread = np.sqrt((w * ens.var(0, ddof=1)).sum() / w.sum())
            err = np.sqrt((w * (ens.mean(0) - d) ** 2).sum() / w.sum())
            rows.append((f"spread/err@{S}", None, spread / err * np.sqrt((a.members + 1) / a.members)))
        print(f"\n{a.dataset} L={L} ({len(samples)} months)  rms(d)={np.sqrt((w*d**2).sum()/w.sum()):.3f}")
        for r in rows:
            if r[1] is None:
                print(f"  {r[0]:16s} {r[2]:.3f}")
            else:
                c, e = scores(r[1], d, w)
                print(f"  {r[0]:16s} corr {c:+.3f}  rel.rmse {e:.3f}")


if __name__ == "__main__":
    main()
