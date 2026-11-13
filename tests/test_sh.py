import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from grace_diffusion import sh


def random_coeffs(L, seed=0, slope=2.0):
    """Random band-limited field with a red spectrum, degrees >= 2."""
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((2, L + 1, L + 1))
    c[1, :, 0] = 0.0
    for l in range(L + 1):
        c[:, l, l + 1:] = 0.0
        c[:, l] *= (max(l, 1)) ** -slope
    c[:, :2] = 0.0
    return c


def value_at(c, lat, lon):
    import pyshtools as pysh
    return pysh.SHCoeffs.from_array(c, **sh.NORM).expand(lat=np.atleast_1d(lat), lon=np.atleast_1d(lon))


# ---------------------------------------------------------------- tapers

@pytest.mark.parametrize("L", [8, 16, 32, 64])
def test_taper_nesting(L):
    """Blurring a rung-L target to L/2 gives exactly the rung-L/2 input."""
    lmax = 2 * L
    c = random_coeffs(lmax, seed=L)
    target = sh.lowpass(c, L)
    np.testing.assert_allclose(sh.lowpass(target, L // 2), sh.lowpass(c, L // 2), atol=0)


def test_taper_shape():
    w = sh.taper(32, 40)
    assert w[0] == 1 and w[25] == 1 and w[32] == 0 and w[40] == 0
    assert np.all(np.diff(w) <= 0)


# ---------------------------------------------------------------- grids

@pytest.mark.parametrize("L", [8, 16, 32])
def test_grid_shape_and_coords(L):
    g = sh.synth(random_coeffs(L), L)
    assert g.shape == sh.grid_shape(L) == (4 * L, 8 * L)
    lat, lon = sh.grid_coords(L)
    assert lat[0] == 90 and lat[-1] > -90 and lon[0] == 0 and lon[-1] < 360


@pytest.mark.parametrize("L", [8, 16, 32])
def test_synth_matches_pointwise(L):
    c = random_coeffs(L, seed=1)
    g = sh.synth(c, L)
    lat, lon = sh.grid_coords(L)
    i, j = L, 3 * L
    assert np.isclose(g[i, j], value_at(c, lat[i], lon[j])[0], atol=1e-10)


@pytest.mark.parametrize("L", [8, 16, 32])
def test_analysis_inverts_synthesis(L):
    c = random_coeffs(L, seed=2)
    back = sh.analyse(sh.synth(c, L))
    np.testing.assert_allclose(back[:, : L + 1, : L + 1], c, atol=1e-10)
    assert np.abs(back[:, L + 1:]).max() < 1e-10


# ---------------------------------------------------------------- consistency projection

@pytest.mark.parametrize("L", [8, 16, 32])
def test_projection_is_consistent_and_idempotent(L):
    x = random_coeffs(2 * L, seed=3)
    inp = sh.synth(sh.lowpass(x, L // 2), L)
    out = sh.synth(random_coeffs(L, seed=4), L)          # an arbitrary "generated" field
    p = sh.project_consistent(inp, out, L // 2)
    low = sh.synth(sh.lowpass(sh.analyse(p), L // 2), L)
    np.testing.assert_allclose(low, inp, atol=1e-9 * np.abs(inp).max())
    p2 = sh.project_consistent(inp, p, L // 2)
    np.testing.assert_allclose(p2, p, atol=1e-9 * np.abs(p).max())


def test_projection_keeps_true_target():
    """The true target already satisfies the constraint and is left unchanged."""
    L = 16
    x = random_coeffs(2 * L, seed=5)
    target = sh.synth(sh.lowpass(x, L), L)
    inp = sh.synth(sh.lowpass(x, L // 2), L)
    np.testing.assert_allclose(sh.project_consistent(inp, target, L // 2), target,
                               atol=1e-9 * np.abs(target).max())


# ---------------------------------------------------------------- rotations

def test_rotation_moves_points():
    """rotate(c, rot) puts the value of point p at rot.apply(p)."""
    c = random_coeffs(30, seed=6)
    rot = Rotation.from_euler("zyz", [40, 25, -70], degrees=True)
    r = sh.rotate(c, rot)
    for lat, lon in [(10, 20), (-30, 100), (50, -60), (0, 0)]:
        q_lat, q_lon = sh.latlon(rot.apply(sh.xyz(lat, lon)))
        assert np.isclose(value_at(r, q_lat, q_lon)[0], value_at(c, lat, lon)[0], atol=1e-9)


def test_rotation_preserves_degree_variance():
    c = random_coeffs(40, seed=7)
    r = sh.rotate(c, sh.random_rotation(np.random.default_rng(0)))
    np.testing.assert_allclose(sh.degree_variance(r), sh.degree_variance(c), rtol=1e-8)


def test_rotation_to_equator():
    lat0, lon0 = 37.5, 35.0
    rot = sh.rotation_to_equator(lat0, lon0)
    q = sh.latlon(rot.apply(sh.xyz(lat0, lon0)))
    assert np.allclose(q, (0, 0), atol=1e-9)
    # north stays up: a point due north of the centre maps due north of (0, 0)
    qn = sh.latlon(rot.apply(sh.xyz(lat0 + 5, lon0)))
    assert np.isclose(qn[1], 0, atol=1e-9) and qn[0] > 0


# ---------------------------------------------------------------- EWH

def test_love_numbers_and_ewh_factor():
    k = sh.love_k(120)
    assert np.isclose(k[2], -0.3025, atol=1e-3)
    f = sh.ewh_factor(120)
    assert f[2] > 0 and np.all(np.diff(f[2:]) > 0)


# ---------------------------------------------------------------- C20 / GIA

def test_read_tn14_assigns_by_largest_overlap(tmp_path):
    p = tmp_path / "tn14.txt"
    p.write_text("header\nProduct:\n"
                 "55197.0 2010.0000  -4.84E-04  0.1 0.1  NaN NaN NaN 55227.0 2010.08\n"       # 1-31 Jan 2010
                 "55228.0 2010.0849  -4.85E-04  0.1 0.1  9.57E-07 0.1 0.1 55255.0 2010.16\n"  # Feb 2010
                 # 16 Dec 2011 - 15 Jan 2012 (JPL's Dec 2011), then 1-31 Jan 2012
                 "55911.0 2011.9589  -4.86E-04  0.1 0.1  NaN NaN NaN 55941.0 2012.04\n"
                 "55927.0 2012.0000  -4.87E-04  0.1 0.1  NaN NaN NaN 55958.0 2012.08\n")
    t = sh.read_tn14(p)
    assert np.isclose(t["2010-01"][0], -4.84e-4) and np.isnan(t["2010-01"][1])
    assert np.isclose(t["2010-02"][1], 9.57e-7)
    assert np.isclose(t["2011-12"][0], -4.86e-4)      # straddling span fills December
    assert np.isclose(t["2012-01"][0], -4.87e-4)      # January keeps its own row
    assert "2011-06" not in t                          # months with no overlap stay absent


def test_replace_low_degrees_and_gia(tmp_path):
    months = ["2010-01", "2010-02", "2011-01"]
    s = sh.Series(months, np.zeros((3, 2, 6, 6)), ["x"] * 3)
    missing = sh.replace_low_degrees(s, {"2010-01": (-1.0, np.nan), "2010-02": (-2.0, 3.0)})
    assert missing == ["2011-01"]
    assert s.coeffs[0, 0, 2, 0] == -1.0 and s.coeffs[0, 0, 3, 0] == 0.0
    assert s.coeffs[1, 0, 3, 0] == 3.0
    g = tmp_path / "gia.txt"
    g.write_text("% comment\n  2 -1 +2.0e-11\n  2 0 +1.0e-11\n  9 9 +5.0e-12\n")
    rates = sh.read_gia_rates(g, 5)
    assert rates[1, 2, 1] == 2e-11 and rates[0, 2, 0] == 1e-11 and rates.shape == (2, 6, 6)
    before = s.coeffs.copy()
    sh.remove_gia(s, rates, t0=2010.0)
    dt = np.array([sh.decimal_year(m) - 2010.0 for m in months])
    np.testing.assert_allclose(s.coeffs[:, 1, 2, 1], before[:, 1, 2, 1] - 2e-11 * dt)
