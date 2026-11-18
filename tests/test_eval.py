

def test_parse_inflate():
    from grace_diffusion.evaluate import parse_inflate
    assert parse_inflate("1.4") == 1.4
    assert parse_inflate("16:1.3,32:1.5") == {16: 1.3, 32: 1.5}


def test_baselines_and_scoring_with_two_octave_step():
    import numpy as np

    from grace_diffusion import baselines, evaluate, sh
    from tests.test_sh import random_coeffs
    c = random_coeffs(32, seed=0)
    inp = sh.synth(sh.lowpass(c, 8), 32)
    ens = baselines.spectral_gauss(c, 32, 3, np.random.default_rng(0), Lin=8)
    for e in ens:                                    # members keep the input exactly
        np.testing.assert_allclose(sh.project_consistent(inp, e, 8), e, atol=1e-6)
    assert np.abs(ens[0] - ens[1]).max() > 0
    m = evaluate.baseline_methods(np.stack([c, c]), np.array([0, 1]), 32)
    r = evaluate.score_month(m["spectral-gauss"], c, 32, 4, np.random.default_rng(1), "global", Lin=8)
    assert r["consistency"] < 1e-6 and r["crps"] > 0
    r0 = evaluate.score_month(m["input"], c, 32, 1, np.random.default_rng(1), "global", Lin=8)
    assert r0["crps"] > r["crps"] * 0.5


def test_finetune_starts_from_checkpoint(tmp_path, monkeypatch):
    import numpy as np
    import torch

    from grace_diffusion import prep, train
    from tests.test_sh import random_coeffs
    gm = [f"{y}-{m:02d}" for y in range(2002, 2020) for m in range(1, 13)]
    wm = [f"{y}-{m:02d}" for y in range(1980, 2020) for m in range(1, 13)]
    sets = {"g": (gm, np.array([random_coeffs(8, seed=i) for i in range(len(gm))])),
            "w": (wm, np.array([random_coeffs(8, seed=1000 + i) for i in range(len(wm))]))}
    monkeypatch.setattr(prep, "load", lambda proc, name: sets[name])
    common = ["--rungs", "8", "--base", "8", "--mults", "1,2", "--blocks", "1", "--batch", "2",
              "--workers", "0", "--val-every", "2", "--steps", "2", "--sigma-max", "20"]
    train.main(["--dataset", "w", "--kind", "twin", "--out", str(tmp_path / "pre"), *common])
    train.main(["--dataset", "g", "--target", "w", "--kind", "paired", "--lr", "1e-5",
                "--init", str(tmp_path / "pre" / "best.pt"), "--out", str(tmp_path / "ft"), *common])
    pre = torch.load(tmp_path / "pre" / "best.pt", weights_only=False)
    ft = torch.load(tmp_path / "ft" / "best.pt", weights_only=False)
    assert ft["cfg"]["sigma_data"] == pre["cfg"]["sigma_data"] and ft["cfg"]["target"] == "w"
    k = next(iter(pre["ema"]))
    assert torch.allclose(pre["ema"][k], ft["ema"][k], atol=1e-2)      # two tiny steps away
    log = (tmp_path / "ft" / "log.csv").read_text().splitlines()
    assert log[1].startswith("0,")                                     # starting point logged


def test_chained_generation_keeps_the_input(tmp_path):
    import numpy as np
    import torch

    from grace_diffusion import evaluate, sh
    from grace_diffusion.train import build
    from tests.test_sh import random_coeffs
    cfg = dict(model="diffusion", base=8, mults=[1, 2], blocks=1, dropout=0.0, sigma_data=1.0,
               p_mean=-0.4, p_std=1.6, sigma_max=20.0)
    torch.manual_seed(0)
    torch.save({"cfg": cfg, "ema": build(cfg).state_dict()}, tmp_path / "m.pt")
    c = random_coeffs(16, seed=0)
    inp = sh.synth(sh.lowpass(c, 4), 16)
    for chain in (False, True):
        m = evaluate.model_method(tmp_path / "m.pt", device="cpu", steps=3, chain=chain)
        ens = m(c, 16, 2, np.random.default_rng(0), 4)
        assert ens.shape == (2, 64, 128) and np.abs(ens[0] - ens[1]).max() > 0
        for e in ens:                                # degrees below 4 are the input's
            np.testing.assert_allclose(sh.project_consistent(inp, e, 4), e, atol=1e-6)
