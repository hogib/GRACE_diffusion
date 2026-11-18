"""Figures for the Phase 1 lab report. Reads runs/eval/*.csv (copied from the server).

  uv run --with pandas python reports/phase1/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "runs" / "eval"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# reference categorical palette (light), fixed order
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
})


def crps(files, L):
    d = pd.concat([pd.read_csv(EV / f) for f in files])
    return d[d.L == L].groupby("method").crps.mean()


# --- Figure 1: CRPS where trained (R2) vs one octave up (R3), GRACE DDK5 -------------------
r2 = crps(["A_grace_ddk5.csv", "A1b_grace_ddk5.csv"], 16)
r3 = crps(["B_grace_ddk5.csv", "B1b_grace_ddk5.csv"], 32)
rows = [("input", "input only"), ("spectral-gauss", "spectral baseline"),
        ("regress", "regression U-Net"), ("diffusion", "diffusion, default"),
        ("logu", "diffusion, log-uniform"), ("wide", "diffusion, widened"),
        ("oracle", "R3-trained reference")]
fig, ax = plt.subplots(figsize=(3.45, 2.5))
for y, (m, lab) in enumerate(rows):
    a, b = r2.get(m), r3.get(m)
    if a is not None and b is not None:
        ax.plot([a, b], [y, y], color=GRID, lw=2.5, zorder=1)
    if a is not None:
        ax.scatter(a, y, s=34, color=C1, zorder=2, edgecolor="white", linewidth=1)
    if b is not None:
        ax.scatter(b, y, s=34, color=C2, zorder=2, edgecolor="white", linewidth=1)
ax.set_yticks(range(len(rows)), [r[1] for r in rows])
ax.invert_yaxis()
ax.set_xlabel("CRPS (mm EWH), lower is better")
ax.axvline(r3["spectral-gauss"], color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=0)
ax.scatter([], [], s=34, color=C1, label="R2 (training scale)")
ax.scatter([], [], s=34, color=C2, label="R3 (one octave finer)")
fig.legend(loc="upper center", ncol=2, handletextpad=0.1, columnspacing=1.2, fontsize=7.5)
ax.grid(axis="y", visible=False)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT / "crps_r2_r3.pdf")

# --- Figure 2: twin, CRPS by rung (Phase 1 twin model trained on R1+R2) -------------------
tw = pd.concat([pd.read_csv(EV / "C_twin.csv"), pd.read_csv(EV / "C_twin_r5.csv")])
g = tw.groupby(["method", "L"]).crps.mean().unstack(0)
fig, ax = plt.subplots(figsize=(3.45, 2.2))
x = range(len(g.index))
for m, col, lab in [("input", INK2, "input"), ("spectral-gauss", C2, "spectral baseline"),
                    ("diffusion", C1, "diffusion, default")]:
    ax.plot(x, g[m].values, color=col, marker="o", ms=4.5, mec="white", mew=1, label=lab)
wide = pd.read_csv(EV / "C1c_twin.csv").groupby("L").crps.mean()
ax.plot([list(g.index).index(L) for L in wide.index], wide.values, color=C3, marker="o", ms=4.5,
        mec="white", mew=1, label="diffusion, widened")
ax.set_xticks(list(x), [f"R{i + 2}\n{L // 2}→{L}" for i, L in enumerate(g.index)])
ax.axvspan(-0.3, 0.3, color=GRID, alpha=0.6, lw=0, zorder=0)
ax.set_ylabel("CRPS (mm EWH)")
ax.set_ylim(0, 24)
ax.set_xlim(-0.4, len(g.index) - 0.6)
ax.text(0, 22.6, "training", ha="center", va="center", color=INK2, fontsize=7)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.02), fontsize=7, labelspacing=0.25)
fig.tight_layout()
fig.savefig(OUT / "twin_rungs.pdf")

# --- Figure 3: denoiser vs noise level (diagnose_sampler.py output, 2026-11-16/17) ---------
# Correlation of D(pure noise of level sigma | input) with the true detail, validation months.
diag = {  # rung: {model: (corr at sigma 10, corr at sigma 80), ensemble-mean corr}
    8: {"Phase 1": ((0.258, -0.003), 0.457), "wide": ((0.844, 0.881), 0.806),
        "regress": 0.890},
    16: {"Phase 1": ((0.404, -0.001), 0.563), "wide": ((0.906, 0.918), 0.911),
         "regress": 0.920},
    32: {"Phase 1": ((0.200, -0.000), 0.232), "wide": ((0.196, 0.216), 0.155),
         "oracle": ((0.794, -0.001), 0.839), "regress": 0.169},
}
fig, ax = plt.subplots(figsize=(3.45, 2.2))
labels = ["R1 (L=8)", "R2 (L=16)", "R3 (L=32)"]
series = [("Phase 1", C1), ("wide", C2), ("oracle", C3), ("regress", INK2)]
names = {"Phase 1": "default", "wide": "widened", "oracle": "R3-trained", "regress": "regression"}
w = 0.19
for i, (m, col) in enumerate(series):
    xs, ys = [], []
    for j, L in enumerate((8, 16, 32)):
        v = diag[L].get(m)
        if v is None:
            continue
        xs.append(j + (i - 1.5) * w)
        ys.append(v if isinstance(v, float) else v[1])
    ax.bar(xs, ys, width=w - 0.03, color=col, label=names[m])
ax.set_xticks(range(3), labels)
ax.set_ylabel("corr(ensemble mean, true detail)")
ax.set_ylim(0, 1)
ax.grid(axis="x", visible=False)
ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(-0.02, 1.0), fontsize=7, columnspacing=0.7,
          handlelength=1.0, handletextpad=0.3)
fig.tight_layout()
fig.savefig(OUT / "diagnosis_corr.pdf")
print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
