"""Score generators on a split (design doc §6, §7).

A method is a callable (c, L, members, rng, Lin, ctx) -> ensemble of target grids (M, 4L, 8L)
in mm, where c is one month's (rotated) EWH coefficients, Lin the input degree, and ctx what a
network with extra input channels needs (month, dataset name, earlier months, rotation). With a
target dataset (paired scoring, design doc Appendix B.5), c is spliced: degrees below Lin from the
input dataset, the rest from the target dataset. Scores are per month, on the
added detail band, weighted by pixel area x region mask; results are written per month so
that paired comparisons (block bootstrap over months) can be made later.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import baselines, data, grids, metrics, prep, sh

Method = Callable[[np.ndarray, int, int, np.random.Generator, int], np.ndarray]


def region_weight(L: int, rot, region: str) -> np.ndarray:
    """'val_land': land outside the test box and ice sheets; 'test_box': the box only."""
    area = grids.area_weights(L)
    if region == "val_land":
        return area * grids.loss_mask(L, rot) * grids.land_mask(L, rot)
    if region == "test_box":
        return area * grids.region_mask(L, rot)
    if region == "global":
        return area
    raise ValueError(region)


def score_month(method: Method, c: np.ndarray, L: int, members: int, rng: np.random.Generator,
                region: str, rot=None, Lin: int | None = None, ctx: dict | None = None) -> dict[str, float]:
    Lin = L // 2 if Lin is None else Lin
    cc = c[:, : L + 1, : L + 1].astype(np.float64)
    if rot is not None:
        cc = sh.rotate(cc, rot)
    truth = sh.synth(sh.lowpass(cc, L), L)
    inp = sh.synth(sh.lowpass(cc, Lin), L)
    ens = method(cc, L, members, rng, Lin) if ctx is None else \
        method(cc, L, members, rng, Lin, {**ctx, "rot": rot})
    w = region_weight(L, rot, region)
    detail_t = truth - inp
    out = {"crps": metrics.crps_fair(ens, truth, w),
           "energy": metrics.energy_score(ens, truth, w),
           "rms_truth_detail": float(np.sqrt((detail_t ** 2 * w).sum() / w.sum()))}
    out.update(metrics.ensemble_stats(ens, truth, w))
    out["corr_mean_detail"] = metrics.correlation(ens.mean(0) - inp, detail_t, w) \
        if np.abs(ens.mean(0) - inp).max() > 0 else np.nan
    out["power_ratio"] = metrics.detail_power_ratio(ens, inp, truth, w)
    out["consistency"] = metrics.consistency_rms(ens[:2], inp, Lin)
    return out


def run(methods: dict[str, Method], dataset: str, kind: str, split: str, rungs: tuple,
        members: int, region: str, out_csv: Path, proc: Path = Path("data/proc"),
        seed: int = 0, rotate: str = "none", max_months: int = 0,
        target: str | None = None) -> list[dict]:
    """rungs: steps (L or (L, Lin)). target: dataset supplying the detail (paired scoring)."""
    months, coeffs = prep.load(proc, dataset)
    tgt = None
    if target:
        months, i_in, i_tgt = data.paired_months(months, prep.load(proc, target)[0])
        coeffs, tgt = coeffs[i_in], prep.load(proc, target)[1][i_tgt]
    idx = data.split_months(months, kind, split)
    pos = {m: j for j, m in enumerate(months)}

    def prev_of(i):
        def get(lag):
            j = pos.get(data.month_shift(months[i], lag))
            return None if j is None else coeffs[j]
        return get
    if max_months:
        idx = idx[:max_months]
    rot = sh.rotation_to_equator(*grids.TEST_BOX.centre) if rotate == "test_box" else None
    rows = []
    for L, Lin in (data.as_step(r) for r in rungs):
        for name, method in methods.items():
            t0 = time.time()
            for i in idx:
                rng = np.random.default_rng([seed, L, int(i)])
                c = coeffs[i] if tgt is None else data.splice_coeffs(coeffs[i], tgt[i], Lin, L)
                ctx = {"month": months[i], "name": dataset, "prev": prev_of(i),
                       "noise_seed": (seed, L, Lin)}
                r = score_month(method, c, L, members, rng, region, rot, Lin, ctx)
                rows.append({"dataset": dataset if tgt is None else f"{dataset}>{target}",
                             "split": split, "L": L, "Lin": Lin, "method": name,
                             "month": months[i], **r})
            print(f"{dataset} L={data.step_name((L, Lin))} {name}: {len(idx)} months "
                  f"in {time.time() - t0:.0f}s", flush=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    return rows


def summarise(rows: list[dict]) -> str:
    """Mean over months per (dataset, step, method)."""
    import collections

    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["dataset"], data.step_name((r["L"], r.get("Lin", r["L"] // 2))), r["method"])].append(r)
    cols = ["crps", "energy", "rmse_mean", "spread_error", "cov50", "cov90", "corr_mean_detail",
            "power_ratio", "consistency", "rms_truth_detail"]
    lines = ["dataset,L,method,n," + ",".join(cols)]
    for (ds, L, m), g in groups.items():
        vals = [np.nanmean([x.get(c, np.nan) for x in g]) for c in cols]
        lines.append(f"{ds},{L},{m},{len(g)}," + ",".join(f"{v:.4g}" for v in vals))
    return "\n".join(lines)


def baseline_methods(coeffs: np.ndarray, train_idx: np.ndarray, max_L: int) -> dict[str, Method]:
    clim = baselines.climatological_spectrum(coeffs, train_idx, max_L)
    return {
        "input": lambda c, L, M, rng, Lin=None, ctx=None: baselines.input_only(c, L, 1, Lin=Lin),
        "spectral-gauss": lambda c, L, M, rng, Lin=None, ctx=None: baselines.spectral_gauss(
            c, L, M, rng, Lin=Lin),
        "spectral-gauss-clim": lambda c, L, M, rng, Lin=None, ctx=None: baselines.spectral_gauss(
            c, L, M, rng, spectrum=clim, Lin=Lin),
    }


def model_method(ckpt: str | Path, device: str | None = None, steps: int = 32,
                 project: bool = True, chunk: int = 32, churn: float = 0.0,
                 inflate: float | dict[int, float] = 1.0, chain: bool = False) -> Method:
    """A trained checkpoint as a Method: diffusion samples M members, regress returns 1.

    churn: stochastic sampler (see EDM.sample). inflate: stretch each member away from the
    ensemble mean by this factor (one value, or one per target degree L). The projection is
    linear and all members share the input's coarse degrees, so inflation keeps them exact.
    chain: a step spanning several octaves (e.g. 8-32) is generated one octave at a time, each
    member's output feeding the next step; otherwise it is generated in one pass.
    Members are generated in the normalised units of training (input / area RMS of input),
    rescaled to mm, and projected onto exact consistency with the input.
    """
    import torch

    from .model.edm import seeded_generator
    from .train import load_checkpoint

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = load_checkpoint(ckpt, dev)
    extras = data.parse_extras(cfg.get("extras", ""))
    banks: dict = {}

    def extra_channels(L, Lin, scale, ctx):
        """Extra input channels (n, 4L, 8L) for one month, as in training but never dropped."""
        if extras.n == 0:
            return None
        if ctx is None:
            raise ValueError("this checkpoint has extra input channels; the caller must pass ctx")
        bank = None
        if extras.needs_bank:
            if L not in banks:
                banks[L] = data.ModelBank.load("data/proc", L)
            bank = banks[L]
        return np.stack(data.extra_grids(extras, L, Lin, ctx.get("rot"), scale, month=ctx["month"],
                                         name=ctx.get("name", ""), prev=ctx.get("prev"), bank=bank))

    def generate(inps, L, Lin, g, extra=None):
        """One ensemble member per input grid (m, 4L, 8L), in mm, consistent with its input."""
        out = []
        for k in range(0, len(inps), chunk):
            part = inps[k: k + chunk]
            s = np.array([data.area_rms(x, L) or 1.0 for x in part])
            cond = torch.from_numpy((part / s[:, None, None])[:, None].astype(np.float32)).to(dev)
            if extra is not None:
                e = torch.from_numpy(extra.astype(np.float32)).to(dev)
                cond = torch.cat([cond, e[None].expand(len(part), -1, -1, -1)], dim=1)
            with torch.no_grad():
                if cfg["model"] == "regress":
                    det = model(cond)
                else:
                    det = model.sample(cond, steps=steps, generator=g, churn=churn)
            det = det.double().cpu().numpy()[:, 0]
            for x, sc, d in zip(part, s, det):
                e = x + sc * d
                out.append(sh.project_consistent(x, e, Lin) if project else e)
        return np.stack(out)

    def method(c, L, M, rng, Lin=None, ctx=None):
        Lin = L // 2 if Lin is None else Lin
        inp = sh.synth(sh.lowpass(c, Lin), L)
        extra = extra_channels(L, Lin, data.area_rms(inp, L) or 1.0, ctx)
        if extra is not None and chain and L > 2 * Lin:
            raise NotImplementedError("chained generation with extra input channels")
        # shared noise: member j starts from the same noise in every month, which keeps each
        # member coherent in time (scripts/test_time_coherence.py); scores are unaffected
        if ctx is not None and "noise_seed" in ctx:
            g = seeded_generator(dev, *ctx["noise_seed"])
        else:
            g = seeded_generator(dev, int(rng.integers(2**31)))
        m = 1 if cfg["model"] == "regress" else M
        if chain and L > 2 * Lin:
            # one octave at a time: each member's output is the next step's input
            Lk = 2 * Lin
            ens = generate(np.repeat(sh.synth(sh.lowpass(c, Lin), Lk)[None], m, axis=0), Lk, Lin, g)
            while Lk < L:
                nxt = np.stack([sh.synth(sh.analyse(e), 2 * Lk) for e in ens])
                ens = generate(nxt, 2 * Lk, Lk, g)
                Lk *= 2
        else:
            ens = generate(np.repeat(inp[None], m, axis=0), L, Lin, g, extra)
        k = inflate.get(L, 1.0) if isinstance(inflate, dict) else inflate
        if k != 1.0 and len(ens) > 1:
            mean = ens.mean(0)
            ens = mean[None] + k * (ens - mean[None])
        return ens

    return method


def parse_inflate(spec: str) -> float | dict[int, float]:
    if ":" not in spec:
        return float(spec)
    return {int(k): float(v) for k, v in (x.split(":") for x in spec.split(","))}


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("kind")
    p.add_argument("split")
    p.add_argument("rungs", help="steps: '16,32' (one octave each) or '8-32' (input 8, target 32)")
    p.add_argument("--target", default=None,
                   help="dataset supplying the true detail (paired scoring); default: the dataset itself")
    p.add_argument("--members", type=int, default=16)
    p.add_argument("--out", required=True)
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--ckpt", action="append", default=[],
                   help="name=path/to/best.pt; a name starting with 'chain' generates steps "
                        "spanning several octaves one octave at a time")
    p.add_argument("--region", default="val_land")
    p.add_argument("--rotate", default="none", choices=["none", "test_box"])
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--chunk", type=int, default=32, help="members sampled per GPU batch")
    p.add_argument("--churn", type=float, default=0.0, help="stochastic sampler (EDM S_churn)")
    p.add_argument("--inflate", default="1", help="spread factor: '1.4' or per rung '16:1.4,32:1.5'")
    p.add_argument("--max-months", type=int, default=0, help="score only the first N months")
    a = p.parse_args(argv)
    inflate = parse_inflate(a.inflate)
    rungs = data.parse_steps(a.rungs)
    # climatological spectrum: from the dataset that supplies the detail
    months, coeffs = prep.load(Path("data/proc"), a.target or a.dataset)
    methods: dict[str, Method] = {}
    if a.baselines:
        methods.update(baseline_methods(coeffs, data.split_months(months, a.kind, "train"),
                                        max(L for L, _ in rungs)))
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        methods[name] = model_method(path, steps=a.steps, chunk=a.chunk, churn=a.churn,
                                     inflate=inflate, chain=name.startswith("chain"))
    rows = run(methods, a.dataset, a.kind, a.split, rungs, a.members, a.region, Path(a.out),
               rotate=a.rotate, max_months=a.max_months, target=a.target)
    print(summarise(rows))


if __name__ == "__main__":
    main()
