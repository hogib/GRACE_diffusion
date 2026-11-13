import numpy as np
import pytest

from grace_diffusion import baselines, metrics, sh
from tests.test_sh import random_coeffs


def brute_crps(ens, y):
    M = len(ens)
    t1 = np.mean(np.abs(ens - y))
    t2 = sum(abs(a - b) for a in ens for b in ens) / (2 * M * (M - 1))
    return t1 - t2


def test_crps_fair_matches_bruteforce():
    rng = np.random.default_rng(0)
    ens = rng.normal(size=(7, 3, 4))
    y = rng.normal(size=(3, 4))
    w = np.ones((3, 4))
    expect = np.mean([brute_crps(ens[:, i, j], y[i, j]) for i in range(3) for j in range(4)])
    assert np.isclose(metrics.crps_fair(ens, y, w), expect)


def test_crps_single_member_is_mae():
    rng = np.random.default_rng(1)
    x, y, w = rng.normal(size=(1, 5, 5)), rng.normal(size=(5, 5)), np.ones((5, 5))
    assert np.isclose(metrics.crps_fair(x, y, w), np.abs(x[0] - y).mean())


def test_calibrated_gaussian_ensemble_scores():
    """Members and truth from the same distribution: spread/error ~ 1, coverage ~ nominal."""
    rng = np.random.default_rng(2)
    ens = rng.normal(size=(30, 60, 60))
    y = rng.normal(size=(60, 60))
    s = metrics.ensemble_stats(ens, y, np.ones((60, 60)))
    assert 0.9 < s["spread_error"] < 1.1
    assert abs(s["cov90"] - 0.9) < 0.03 and abs(s["cov50"] - 0.5) < 0.04


def test_energy_score_zero_spread_equals_rmse_like_norm():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(8, 8))
    x = y + 0.5
    es = metrics.energy_score(x[None], y, np.ones((8, 8)))
    assert np.isclose(es, 0.5)


@pytest.mark.parametrize("L", [8, 16, 32])
def test_baselines_are_consistent(L):
    c = random_coeffs(2 * L, seed=L)
    rng = np.random.default_rng(0)
    inp = sh.synth(sh.lowpass(c, L // 2), L)
    for ens in (baselines.input_only(c, L, 2),
                baselines.spectral_gauss(c, L, 3, rng)):
        assert ens.shape == (len(ens), 4 * L, 8 * L)
        assert metrics.consistency_rms(ens, inp, L // 2) < 1e-8


def test_spectral_gauss_power_follows_spectrum():
    L = 32
    c = random_coeffs(L, seed=9, slope=1.0)
    dv = np.full(L + 1, 4.0)
    ens = baselines.spectral_gauss(c, L, 40, np.random.default_rng(1), spectrum=dv)
    w_out = sh.taper(L, L)
    free = sh.taper(L // 2, L) == 0
    got = np.mean([sh.degree_variance(sh.analyse(e))[: L + 1] for e in ens], axis=0)
    expect = 4.0 * w_out ** 2
    ok = free & (expect > 0.5)
    assert np.allclose(got[ok], expect[ok], rtol=0.35)
