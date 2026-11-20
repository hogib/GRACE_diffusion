"""Train the diffusion model or the `regress` baseline (design doc §3, §7).

Examples:
  # pre-train on WaterGAP alone
  uv run python -m grace_diffusion.train --dataset twin --kind twin --rungs 16,32 \
      --p-mean -0.4 --p-std 1.6 --sigma-max 20 --out runs/pre-wg-r23
  # fine-tune on paired months: land-only GRACE in, WaterGAP detail out
  uv run python -m grace_diffusion.train --dataset grace_ddk5_land --target twin --kind paired \
      --rungs 16,32 --init runs/pre-wg-r23/best.pt --lr 5e-5 --steps 10000 --out runs/ft-wg-r23
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data, prep, sh
from .model.edm import EDM, EMA, Regressor, count_params
from .model.unet import UNet


class BatchStream(torch.utils.data.IterableDataset):
    """Endless stream of single-rung batches; reproducible per (seed, worker, k)."""

    def __init__(self, sampler: data.Sampler | data.MultiSampler, batch_size: int, seed: int):
        self.sampler, self.batch_size, self.seed = sampler, batch_size, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        k = 0
        while True:
            b = self.sampler.batch(np.random.default_rng([self.seed, wid, k]), self.batch_size)
            k += 1
            yield {key: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v)
                   for key, v in b.items() if key in ("inp", "target", "weight", "L")}


def to_dev(b, dev):
    cond = b["inp"].to(dev, non_blocking=True)          # blurred field, then any extra channels
    detail = b["target"].to(dev, non_blocking=True) - cond[:, :1]
    return detail, cond, b["weight"].to(dev, non_blocking=True)


def build_noise_bank(proc, raw, dataset: str, rungs, n: int = 512, seed: int = 0,
                     last_month: str = "2019-12") -> data.NoiseBank:
    """n draws of GRACE noise (formal errors, processed like `dataset`) from random months up to
    last_month, with the RMS of that month's GRACE field at each input degree."""
    steps = [data.as_step(r) for r in rungs]
    lmax = max(Lin for _, Lin in steps)
    months, coeffs = prep.load(Path(proc), dataset)
    gen = prep.GraceNoise(Path(proc), Path(raw), dataset)
    ok = [i for i, m in enumerate(months) if m <= last_month and m in gen.sigma]
    rng = np.random.default_rng([seed, 99])
    pick = rng.choice(ok, size=n)
    draws = np.array([gen.draw(months[i], rng)[:, : lmax + 1, : lmax + 1] for i in pick], dtype=np.float32)
    ref = {Lin: np.array([data.sh_rms(sh.lowpass(coeffs[i][:, : Lin + 1, : Lin + 1].astype(np.float64), Lin))
                          for i in pick]) for Lin in {Lin for _, Lin in steps}}
    return data.NoiseBank(draws, ref)


def make_samplers(proc, datasets: str, kind: str, rungs, rotate: str, target: str | None = None,
                  extras: data.Extras | None = None, noise_bank: data.NoiseBank | None = None,
                  noise_max: float = 0.0, bank_exclude=()):
    """Train and validation samplers; `datasets` is one name or a comma-separated list.

    target: dataset supplying the detail (paired samples); the datasets then supply the input.
    extras: extra input channels (data.Extras).
    """
    tr, va = [], []
    bank = None
    if extras is not None and extras.needs_bank:
        bank = data.ModelBank.load(proc, max(L for L, _ in (data.as_step(r) for r in rungs)),
                                   exclude=bank_exclude)
    ex = dict(extras=extras, bank=bank)
    tm, tc = prep.load(Path(proc), target) if target else (None, None)
    for name in datasets.split(","):
        months, coeffs = prep.load(Path(proc), name)
        for split, rot, dst in (("train", rotate, tr), ("val", "none", va)):
            if target:
                dst.append(data.PairedSampler(months, coeffs, kind, split, rungs=rungs, rotate=rot,
                                              tgt_months=tm, tgt_coeffs=tc))
            else:
                dst.append(data.Sampler(months, coeffs, kind, split, rungs=rungs, rotate=rot,
                                        name=name, noise_bank=noise_bank, noise_max=noise_max, **ex))
    if len(tr) == 1:
        return tr[0], va[0]
    return data.MultiSampler(tr), data.MultiSampler(va)


def estimate_sigma_data(sampler: data.Sampler | data.MultiSampler, n: int = 40, batch: int = 16) -> float:
    """Weighted RMS of the normalised detail over random training batches."""
    num = den = 0.0
    for k in range(n):
        b = sampler.batch(np.random.default_rng([12345, k]), batch)
        d = b["target"] - b["inp"][:, :1]
        num += float((b["weight"] * d ** 2).sum())
        den += float(b["weight"].sum())
    return float(np.sqrt(num / den))


def build(cfg: dict) -> torch.nn.Module:
    net = UNet(in_ch=2 + cfg.get("extra_ch", 0), out_ch=1, base=cfg["base"], mults=tuple(cfg["mults"]),
               blocks=cfg["blocks"], dropout=cfg["dropout"])
    if cfg["model"] == "diffusion":
        return EDM(net, sigma_data=cfg["sigma_data"], p_mean=cfg.get("p_mean", -1.2),
                   p_std=cfg.get("p_std", 1.2), noise=cfg.get("noise", "lognormal"),
                   sigma_max=cfg.get("sigma_max", 80.0))
    return Regressor(net)


def load_checkpoint(path: str | Path, device="cpu") -> tuple[torch.nn.Module, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build(ck["cfg"]).to(device)
    m.load_state_dict(ck["ema"])
    m.eval()
    return m, ck["cfg"]


@torch.no_grad()
def val_loss(model, batches, dev) -> float:
    model.eval()
    torch.manual_seed(0)
    tot = n = 0.0
    for b in batches:
        detail, cond, w = to_dev(b, dev)
        tot += float(model.loss(detail, cond, w)) * detail.shape[0]
        n += detail.shape[0]
    model.train()
    return tot / n


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="grace_ddk5")
    p.add_argument("--kind", default="grace", choices=sorted(data.SPLITS))
    p.add_argument("--target", default=None,
                   help="dataset supplying the detail (paired samples: input from --dataset)")
    p.add_argument("--init", default=None,
                   help="checkpoint to start from (fine-tuning); its network settings are kept")
    p.add_argument("--rungs", default="8,16", help="steps: '8,16', or '8-32' (input 8, target 32)")
    p.add_argument("--extras", default="",
                   help="extra input channels, e.g. 'hist:1,2,3,6,12+models+clim+land' (data.Extras)")
    p.add_argument("--input-noise", type=float, default=0.0,
                   help="train with GRACE-like noise on the input: formal errors times a factor "
                        "drawn uniformly from [0, this value]; 0 = clean inputs")
    p.add_argument("--noise-dataset", default="grace_ddk5_land",
                   help="GRACE dataset whose processing and amplitude the input noise follows")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--bank-exclude", default="",
                   help="comma-separated datasets kept out of the 'models'/'clim' inputs "
                        "(a model withheld from training must not leak in through them)")
    p.add_argument("--extras-drop", type=float, default=0.15,
                   help="probability of zeroing each group of extra channels in training")
    p.add_argument("--model", choices=["diffusion", "regress"], default="diffusion")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--mults", default="1,2,2,4")
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--rotate", default="random", choices=["random", "polar", "none"])
    p.add_argument("--noise", default="lognormal", choices=["lognormal", "loguniform"],
                   help="training noise-level distribution (diffusion only)")
    p.add_argument("--p-mean", type=float, default=-1.2)
    p.add_argument("--p-std", type=float, default=1.2)
    p.add_argument("--sigma-max", type=float, default=80.0, help="sampler start; loguniform upper end")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--proc", default="data/proc")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rungs = data.parse_steps(a.rungs)
    extras = data.parse_extras(a.extras, a.extras_drop)
    bank = build_noise_bank(a.proc, a.raw, a.noise_dataset, rungs) if a.input_noise > 0 else None
    train_s, val_s = make_samplers(a.proc, a.dataset, a.kind, rungs, a.rotate, a.target, extras,
                                   bank, a.input_noise,
                                   tuple(x for x in a.bank_exclude.split(",") if x))
    cfg = dict(model=a.model, base=a.base, mults=[int(x) for x in a.mults.split(",")],
               blocks=a.blocks, dropout=a.dropout, rungs=a.rungs, dataset=a.dataset,
               target=a.target, kind=a.kind, rotate=a.rotate, noise=a.noise, p_mean=a.p_mean,
               p_std=a.p_std, sigma_max=a.sigma_max, seed=a.seed, steps=a.steps, batch=a.batch,
               lr=a.lr, init=a.init, input_noise=a.input_noise, noise_dataset=a.noise_dataset,
               bank_exclude=a.bank_exclude, extras=a.extras, extras_drop=a.extras_drop, extra_ch=extras.n)
    init = None
    if a.init:
        # fine-tuning: keep the network and its preconditioning scale; the model type must match
        init = torch.load(a.init, map_location="cpu", weights_only=False)
        if init["cfg"]["model"] != a.model:
            raise SystemExit(f"--init is a {init['cfg']['model']} checkpoint, --model is {a.model}")
        for k in ("base", "mults", "blocks", "dropout", "sigma_data"):
            cfg[k] = init["cfg"][k]
        if init["cfg"].get("extra_ch", 0) != extras.n:
            raise SystemExit("--init has a different number of extra input channels")
        cfg["sigma_data_here"] = estimate_sigma_data(train_s)
    else:
        cfg["sigma_data"] = estimate_sigma_data(train_s)
    model = build(cfg).to(dev)
    if init is not None:
        model.load_state_dict(init["ema"])
    cfg["params"] = count_params(model)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg), flush=True)

    # fixed validation batches: every val month at every training rung, unrotated
    val_batches = []
    for step_ in rungs:
        samples = val_s.fixed_set(step_)
        for i in range(0, len(samples), a.batch):
            b = data.stack(samples[i: i + a.batch])
            val_batches.append({k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v)
                                for k, v in b.items() if k in ("inp", "target", "weight", "L")})

    loader = torch.utils.data.DataLoader(BatchStream(train_s, a.batch, a.seed), batch_size=None,
                                         num_workers=a.workers, pin_memory=True,
                                         persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000))
    ema = EMA(model, a.ema)
    ema_model = build(cfg).to(dev)
    log = open(out / "log.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["step", "train_loss", "val_loss_ema", "lr", "sec"])
    t0, run_loss, best = time.time(), [], np.inf
    if init is not None:          # where fine-tuning starts from, on this run's validation set
        ema_model.load_state_dict(ema.shadow)
        vl0 = val_loss(ema_model, val_batches, dev)
        wr.writerow([0, "", f"{vl0:.5f}", "", "0"])
        print(f"step 0 val(ema) {vl0:.4f} (from {a.init})", flush=True)
    for step, b in enumerate(loader, start=1):
        detail, cond, w = to_dev(b, dev)
        loss = model.loss(detail, cond, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(model)
        run_loss.append(float(loss.detach()))
        if step % a.val_every == 0 or step == a.steps:
            ema_model.load_state_dict(ema.shadow)
            vl = val_loss(ema_model, val_batches, dev)
            tl = float(np.mean(run_loss))
            run_loss = []
            wr.writerow([step, f"{tl:.5f}", f"{vl:.5f}", f"{sched.get_last_lr()[0]:.2e}",
                         f"{time.time() - t0:.0f}"])
            log.flush()
            print(f"step {step} train {tl:.4f} val(ema) {vl:.4f} {time.time() - t0:.0f}s", flush=True)
            ck = {"cfg": cfg, "ema": ema.shadow, "model": model.state_dict(), "step": step, "val": vl}
            torch.save(ck, out / "last.pt")
            if vl < best:
                best = vl
                torch.save(ck, out / "best.pt")
        if step >= a.steps:
            break
    log.close()


if __name__ == "__main__":
    main()
