"""Figure for the proposal: which degrees come from GRACE and which are generated.

  uv run python reports/proposal/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from grace_diffusion import sh  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
C1, C2 = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
})

LIN, L = 32, 128
l = np.arange(0, L + 1)
w_in, w_out = sh.taper(LIN, L), sh.taper(L, L)
fig, ax = plt.subplots(figsize=(5.2, 2.2))
ax.axvspan(2, LIN, color=C1, alpha=0.10, lw=0)
ax.axvspan(LIN, L, color=C2, alpha=0.10, lw=0)
ax.plot(l[2:], w_in[2:], color=C1, label=r"input: GRACE limited to degree 32")
ax.plot(l[2:], w_out[2:], color=C2, label=r"output: limited to degree 128")
ax.set_xscale("log", base=2)
deg = [2, 4, 8, 16, 32, 64, 128]
ax.set_xticks(deg, [f"{d}\n{20000 / d:.0f} km" for d in deg])
ax.set_xlim(2, 140)
ax.set_ylim(0, 1.45)
ax.set_yticks([0, 0.5, 1])
ax.set_xlabel("spherical-harmonic degree and half-wavelength")
ax.set_ylabel("weight")
ax.text(8, 1.2, "from GRACE\n(kept exactly)", ha="center", va="center", color=C1, fontsize=8)
ax.text(64, 1.2, "generated\n(learned from hydrological models)", ha="center", va="center", color=C2, fontsize=8)
ax.legend(loc="lower left", fontsize=7.5, bbox_to_anchor=(0.0, 0.02))
fig.tight_layout()
fig.savefig(OUT / "bands.pdf")
print("wrote", OUT / "bands.pdf")
