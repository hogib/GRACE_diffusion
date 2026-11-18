import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from grace_diffusion import data, grids, prep, sh
from tests.test_sh import random_coeffs


def fake_series(n=24, start=2002, L=40):
    months = [f"{start + i // 12:04d}-{i % 12 + 1:02d}" for i in range(n)]
    coeffs = np.array([random_coeffs(L, seed=i) for i in range(n)])
    return months, coeffs


# ---------------------------------------------------------------- splits

def test_splits_are_disjoint_blocks():
    months = [f"{y:04d}-{m:02d}" for y in range(2002, 2027) for m in range(1, 13)]
    sets = {s: set(data.split_months(months, "grace", s)) for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]) and not (sets["val"] & sets["test"])
    assert max(months[i] for i in sets["train"]) < min(months[i] for i in sets["val"])
    assert max(months[i] for i in sets["val"]) < min(months[i] for i in sets["test"])
    assert months[min(sets["train"])] == "2002-04"


# ---------------------------------------------------------------- masks

def test_box_contains_and_margin():
    b = grids.TEST_BOX
    assert b.contains(37.5, 35.0) and not b.contains(37.5, 55.0)
    assert not b.contains(46.0, 35.0) and b.contains(46.0, 35.0, margin_km=200)
    assert b.contains(37.5, 35.0 - 360)          # longitude wrap


def test_mask_rotates_with_field():
    """After rotating the test box to the equator, the box is masked at (0, 0)."""
    L = 32
    rot = sh.rotation_to_equator(*grids.TEST_BOX.centre)
    m = grids.loss_mask(L, rot, mask_ice=False)
    lat, lon = sh.grid_coords(L)
    i0, j0 = np.argmin(np.abs(lat)), 0
    assert m[i0, j0] == 0
    far = np.argmin(np.abs(lat - 0)), np.argmin(np.abs(lon - 180))
    assert m[far] == 1
    # unrotated: masked at the box centre itself
    m0 = grids.loss_mask(L, None, mask_ice=False)
    i, j = np.argmin(np.abs(lat - 37.5)), np.argmin(np.abs(lon - 35.0))
    assert m0[i, j] == 0


def test_ice_mask():
    L = 16
    m = grids.loss_mask(L, None, hold_out_test_box=False)
    lat, lon = sh.grid_coords(L)
    assert m[np.argmin(np.abs(lat + 80)), 10] == 0                       # Antarctica
    assert m[np.argmin(np.abs(lat - 72)), np.argmin(np.abs(lon - 318))] == 0   # Greenland
    assert m[np.argmin(np.abs(lat - 0)), np.argmin(np.abs(lon - 20))] == 1


def test_area_weights_mean_one():
    w = grids.area_weights(16)
    assert np.isclose(w.mean(), 1.0) and w[0].max() < w[w.shape[0] // 2].max()


# ---------------------------------------------------------------- samples

@pytest.mark.parametrize("L", [8, 16, 32])
def test_sample_input_is_blurred_target(L):
    months, coeffs = fake_series()
    rot = Rotation.from_euler("zyz", [10, 50, 80], degrees=True)
    s = data.make_sample(coeffs[0], months[0], L, rot)
    assert s.inp.shape == s.target.shape == s.weight.shape == (1, 4 * L, 8 * L)
    # blurring the target to L/2 reproduces the input (consistency holds for the truth)
    ct = sh.analyse(s.target[0].astype(np.float64))
    back = sh.synth(sh.lowpass(ct, L // 2), L)
    np.testing.assert_allclose(back, s.inp[0], atol=1e-4 * np.abs(s.inp).max())
    # normalisation: area-weighted RMS of the input is 1
    assert np.isclose(data.area_rms(s.inp[0].astype(np.float64), L), 1.0, rtol=1e-4)


def test_sampler_batches_reproducible_and_single_rung():
    months, coeffs = fake_series(n=36)
    smp = data.Sampler(months, coeffs, "grace", "train", rungs=(8, 16))
    b1 = list(smp.batches(seed=1, batch_size=3, n_batches=4))
    b2 = list(smp.batches(seed=1, batch_size=3, n_batches=4))
    for x, y in zip(b1, b2):
        assert x["L"] == y["L"] and x["months"] == y["months"]
        np.testing.assert_array_equal(x["inp"], y["inp"])
        assert x["inp"].shape == (3, 1, 4 * x["L"], 8 * x["L"])
    assert all(m <= "2019-12" for b in b1 for m in b["months"])


def test_rotation_mode_test_box_puts_box_on_equator():
    months, coeffs = fake_series(n=24, start=2020)
    smp = data.Sampler(months, coeffs, "grace", "val", rungs=(16,), rotate="test_box",
                       hold_out_test_box=False)
    s = smp.fixed_set(16)[0]
    lat, lon = sh.grid_coords(16)
    reg = grids.region_mask(16, sh.rotation_to_equator(*grids.TEST_BOX.centre))
    assert reg[np.argmin(np.abs(lat)), 0] == 1


# ---------------------------------------------------------------- twin regridding

def test_watergap_to_dh_preserves_smooth_field():
    lat_c = 89.75 - 0.5 * np.arange(360)
    lon_c = -179.75 + 0.5 * np.arange(720)
    LA, LO = np.meshgrid(lat_c, lon_c, indexing="ij")
    f = np.sin(np.radians(LA)) + np.cos(np.radians(LO))
    dh = prep.watergap_to_dh(f)
    lat_dh = 90 - 0.5 * np.arange(360)
    lon_dh = 0.5 * np.arange(720)
    LA2, LO2 = np.meshgrid(lat_dh, lon_dh, indexing="ij")
    expect = np.sin(np.radians(LA2)) + np.cos(np.radians(LO2))
    assert np.abs(dh - expect)[1:].max() < 1e-3        # row 0 is the pole (extrapolated)


def test_multisampler_mixes_datasets():
    from grace_diffusion.data import MultiSampler, Sampler
    months = [f"{y}-{m:02d}" for y in range(1980, 1995) for m in range(1, 13)]
    rng = np.random.default_rng(0)
    a = rng.normal(size=(len(months), 2, 17, 17)) * 0
    b = rng.normal(size=(len(months), 2, 17, 17))
    a[:, :, :2] = b[:, :, :2] = 0
    ms = MultiSampler([Sampler(months, a, "twin", "train", rungs=(16,), rotate="none"),
                       Sampler(months, b, "twin", "train", rungs=(16,), rotate="none")])
    bt = ms.batch(np.random.default_rng(1), 16)
    zero = np.array([np.allclose(t, 0) for t in bt["target"]])
    assert 0 < zero.sum() < 16                       # both datasets appear
    val = MultiSampler([Sampler(months, a, "twin", "val", rungs=(16,), rotate="none"),
                        Sampler(months, b, "twin", "val", rungs=(16,), rotate="none")])
    assert len(val.fixed_set(16)) == 2 * 60          # 1990-1994 val months, both datasets


def test_land_only_zeroes_ocean(tmp_path):
    land = np.zeros((360, 720), dtype=bool)
    land[100:200, 100:300] = True                    # one block of "land", away from ice
    np.savez(tmp_path / "landmask.npz", land=land, lat=np.zeros(360), lon=np.zeros(720))
    c = random_coeffs(16, np.random.default_rng(0))[None]
    np.savez(tmp_path / "x.npz", months=np.array(["2005-01"]), coeffs=c.astype(np.float32))
    prep.land_only(tmp_path, "x", "x_land")
    months, cl = prep.load(tmp_path, "x_land")
    assert months == ["2005-01"] and cl.shape == c.shape
    g = sh.synth(cl[0].astype(float), 64)
    glat, glon = sh.grid_coords(64)
    ocean = (glat[:, None] > 60) | (glat[:, None] < -60)   # far from the land block
    ocean = np.broadcast_to(ocean, g.shape)
    assert np.abs(g[ocean]).mean() < 0.15 * np.abs(sh.synth(c[0], 64)[ocean]).mean()


# ---------------------------------------------------------------- steps and paired samples

def test_parse_steps():
    assert data.parse_steps("8,16") == ((8, 4), (16, 8))
    assert data.parse_steps("8-32,32") == ((32, 8), (32, 16))
    assert data.step_name((32, 8)) == "8-32" and data.step_name(16) == "16"


def test_two_octave_step_is_consistent():
    c = random_coeffs(32, seed=3)
    s = data.make_sample(c, "2005-01", 32, None, Lin=8)
    assert s.Lin == 8 and s.inp.shape == (1, 128, 256)
    tgt = s.target[0].astype(np.float64) * s.scale
    inp = s.inp[0].astype(np.float64) * s.scale
    np.testing.assert_allclose(sh.project_consistent(inp, tgt, 8), tgt, atol=1e-3)
    assert np.abs(sh.analyse(inp)[:, 8:]).max() < 1e-3                             # nothing from 8 up


def test_splice_takes_input_below_and_target_above():
    g, w = random_coeffs(20, seed=1), random_coeffs(40, seed=2)
    c = data.splice_coeffs(g, w, 8, 16)
    assert c.shape == (2, 17, 17)
    np.testing.assert_array_equal(c[:, :8, :8], g[:, :8, :8])
    np.testing.assert_array_equal(c[:, 8:, :], w[:, 8:17, :17])
    np.testing.assert_allclose(sh.lowpass(c, 8), sh.lowpass(g[:, :17, :17], 8))   # input is blurred g
    # the detail does not depend on the input dataset above the input degree
    c2 = data.splice_coeffs(random_coeffs(20, seed=9), w, 8, 16)
    d1, d2 = sh.lowpass(c, 16) - sh.lowpass(c, 8), sh.lowpass(c2, 16) - sh.lowpass(c2, 8)
    np.testing.assert_allclose(d1[:, 8:], d2[:, 8:])


def test_paired_sampler_matches_months_and_splits():
    gm = [f"{y}-{m:02d}" for y in range(2002, 2021) for m in range(1, 13) if not (y == 2017 and m > 6)]
    wm = [f"{y}-{m:02d}" for y in range(1990, 2020) for m in range(1, 13)]
    rng = np.random.default_rng(0)
    g = rng.normal(size=(len(gm), 2, 17, 17))
    w = rng.normal(size=(len(wm), 2, 17, 17))
    g[:, :, :2] = w[:, :, :2] = 0
    tri = np.tril(np.ones((17, 17)))                       # orders m <= l only
    g, w = g * tri, w * tri
    g[:, 1, :, 0] = w[:, 1, :, 0] = 0                      # no sine term at order 0
    kw = dict(rungs=(16,), tgt_months=wm, tgt_coeffs=w)
    tr = data.PairedSampler(gm, g, "paired", "train", rotate="random", **kw)
    va = data.PairedSampler(gm, g, "paired", "val", rotate="none", **kw)
    te = data.PairedSampler(gm, g, "paired", "test", rotate="none", **kw)
    assert max(tr.months[i] for i in tr.idx) == "2014-12"
    assert [va.months[i] for i in va.idx][-1] == "2017-06"
    assert [te.months[i] for i in te.idx][-1] == "2019-12"      # 2020 has no target
    b = tr.batch(np.random.default_rng(1), 4)
    assert b["inp"].shape == (4, 1, 64, 128) and b["Lin"] == 8
    s = va.fixed_set(16)[0]                                     # 2015-01, unrotated
    i, j = gm.index("2015-01"), wm.index("2015-01")
    want_in = sh.synth(sh.lowpass(g[i].astype(np.float64), 8), 16)
    np.testing.assert_allclose(s.inp[0] * s.scale, want_in, atol=1e-4)
    want_d = sh.synth(sh.lowpass(w[j].astype(np.float64), 16), 16)
    got_d = sh.analyse((s.target[0] - s.inp[0]).astype(np.float64) * s.scale)
    np.testing.assert_allclose(got_d[:, 8:17, :17], sh.lowpass(w[j].astype(np.float64), 16)[:, 8:], atol=1e-3)
    assert want_d.shape == s.target[0].shape


# ---------------------------------------------------------------- extra input channels

def test_sh_rms_matches_grid_rms():
    c = sh.lowpass(random_coeffs(16, seed=4), 16)
    assert abs(data.sh_rms(c) - data.area_rms(sh.synth(c, 16), 16)) / data.sh_rms(c) < 0.02


def test_parse_extras_and_month_shift():
    ex = data.parse_extras("hist:1,12+models+land", 0.2)
    assert ex.hist == (1, 12) and ex.models and ex.land and not ex.clim and ex.n == 4
    assert data.parse_extras("").n == 0
    assert data.month_shift("2003-01", 1) == "2002-12" and data.month_shift("2003-01", 12) == "2002-01"


def _bank_and_series():
    months = [f"{y}-{m:02d}" for y in range(1980, 1992) for m in range(1, 13)]
    sets = {f"m{k}": (months, np.array([random_coeffs(16, seed=100 * k + i) for i in range(len(months))]))
            for k in range(3)}
    return months, sets, data.ModelBank(sets, 16, clim_years=("1980", "1989"))


def test_extra_channels_history_models_clim_land():
    months, sets, bank = _bank_and_series()
    ex = data.parse_extras("hist:1,2+models+clim")
    smp = data.Sampler(months, sets["m0"][1], "twin", "train", rungs=(16,), rotate="none",
                       extras=ex, bank=bank, name="m0")
    s = smp.sample(5, 16, None)
    assert s.inp.shape == (5, 64, 128) and s.target.shape == (1, 64, 128)
    # history channel 1 is last month's blurred field in the current input's units
    want = sh.synth(sh.lowpass(sets["m0"][1][4].astype(np.float64), 8), 16) / s.scale
    np.testing.assert_allclose(s.inp[1], want, atol=1e-4)
    # the 'models' channel leaves the sample's own model out
    own = data.Sampler(months, sets["m0"][1], "twin", "train", rungs=(16,), rotate="none",
                       extras=ex, bank=bank, name="").sample(5, 16, None)
    assert np.abs(own.inp[3] - s.inp[3]).max() > 1e-3
    assert np.abs(s.inp[4]).max() > 0                           # climatology channel
    # the first month has no history: zeros
    first = smp.sample(0, 16, None)
    assert np.all(first.inp[1] == 0) and np.all(first.inp[2] == 0)
    # dropout zeroes whole groups in training batches
    drop = data.Sampler(months, sets["m0"][1], "twin", "train", rungs=(16,), rotate="none",
                        extras=data.parse_extras("hist:1+models", 1.0), bank=bank, name="m0")
    b = drop.batch(np.random.default_rng(0), 2)
    assert b["inp"].shape == (2, 3, 64, 128) and np.all(b["inp"][:, 1:] == 0)


def test_extra_channels_rotate_with_the_field():
    months, sets, bank = _bank_and_series()
    ex = data.parse_extras("hist:1")
    smp = data.Sampler(months, sets["m0"][1], "twin", "train", rungs=(16,), rotate="none", extras=ex)
    rot = Rotation.from_euler("z", 90, degrees=True)
    a, b = smp.sample(5, 16, None), smp.sample(5, 16, rot)
    np.testing.assert_allclose(np.roll(a.inp[1], 32, axis=1), b.inp[1], atol=1e-4)   # 90 deg = 32 columns
