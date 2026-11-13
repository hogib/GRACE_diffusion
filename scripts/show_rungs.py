"""Show the octave ladder on real GRACE: input and target maps of rungs R1-R3 for one month.

Usage: uv run python scripts/show_rungs.py data/raw runs/explore [YYYY-MM] [--ddk5] [--rotate]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grace_diffusion import sh

RUNGS = {"R1": 8, "R2": 16, "R3": 32}


def main(raw: Path, out: Path, month: str, ddk5: bool, rotate: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s = sh.load_series(raw)
    anom = sh.anomalies(s)
    c = sh.to_ewh(anom[s.months.index(month)])
    if ddk5:
        c = sh.DDKFilter.named("DDK5", Path(raw) / "ddk")(c)
    c = c[:, :65, :65]
    if rotate:
        c = sh.rotate(c, sh.rotation_to_equator(37.5, 35.0))
    print(f"loaded {len(s.months)} months in {time.time() - t0:.1f}s")
    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    for row, (name, L) in zip(axes, RUNGS.items()):
        inp = sh.synth(sh.lowpass(c, L // 2), L)
        tgt = sh.synth(sh.lowpass(c, L), L)
        v = np.percentile(np.abs(tgt), 99.5)
        for ax, img, title in zip(row, (inp, tgt, tgt - inp),
                                  (f"{name} input (≤{L // 2})", f"{name} target (≤{L})", f"{name} added detail")):
            im = ax.imshow(img, cmap="RdBu", vmin=-v, vmax=v, extent=(0, 360, -90, 90))
            ax.set_title(f"{title}  {img.shape[1]}×{img.shape[0]} px")
            fig.colorbar(im, ax=ax, shrink=0.7, label="EWH mm")
    tag = ("ddk5" if ddk5 else "raw") + ("_rot" if rotate else "")
    fig.suptitle(f"{month}, ITSG {'DDK5' if ddk5 else 'unfiltered'}"
                 + (", test box rotated to (0, 0)" if rotate else ""))
    fig.tight_layout()
    fig.savefig(out / f"rungs_{month}_{tag}.png", dpi=90)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(Path(args[0]), Path(args[1]), args[2] if len(args) > 2 else "2010-04",
         "--ddk5" in sys.argv, "--rotate" in sys.argv)
