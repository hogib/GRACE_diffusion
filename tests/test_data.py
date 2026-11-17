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
