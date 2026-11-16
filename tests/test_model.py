import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grace_diffusion import sh  # noqa: E402
from grace_diffusion.model.edm import EDM, Regressor  # noqa: E402
from grace_diffusion.model.unet import UNet, sphere_pad  # noqa: E402
from tests.test_sh import random_coeffs  # noqa: E402


def test_sphere_pad_continues_field_across_dateline_and_pole():
    """Padded values equal the field evaluated just past the grid edge."""
    L = 16
    c = random_coeffs(L, seed=1)
    g = sh.synth(c, L)
    x = torch.from_numpy(g)[None, None]
    p = sphere_pad(x, 1)[0, 0].numpy()
    lat, lon = sh.grid_coords(L)
    dlat, dlon = lat[0] - lat[1], lon[1] - lon[0]
    import pyshtools as pysh
    f = lambda la, lo: pysh.SHCoeffs.from_array(c, **sh.NORM).expand(lat=[la], lon=[lo])[0]
    # dateline: column left of lon[0] is lon = -dlon
    assert np.isclose(p[5, 0], f(lat[4], -dlon), atol=1e-8)
    # north pole: row above lat[0] = 90 is at 90 + dlat, i.e. latitude 90 - dlat on the far side
    j = 3
    assert np.isclose(p[0, j + 1], f(90 - dlat, lon[j] + 180), atol=1e-8)


@pytest.mark.parametrize("L", [8, 16, 32])
def test_unet_runs_at_every_rung(L):
    net = UNet(base=16, mults=(1, 2, 2, 4), blocks=1)
    x = torch.randn(2, 2, 4 * L, 8 * L)
    y = net(x, torch.zeros(2))
    assert y.shape == (2, 1, 4 * L, 8 * L)


def test_edm_loss_backward_and_sample_shapes():
    torch.manual_seed(0)
    m = EDM(UNet(base=16, mults=(1, 2), blocks=1), sigma_data=1.0)
    cond = torch.randn(3, 1, 32, 64)
    detail = torch.randn(3, 1, 32, 64)
    w = torch.rand(3, 1, 32, 64)
    loss = m.loss(detail, cond, w)
    loss.backward()
    assert torch.isfinite(loss)
    s = m.sample(cond, steps=4)
    assert s.shape == cond.shape and torch.isfinite(s).all()


def test_regressor_shapes():
    r = Regressor(UNet(base=16, mults=(1, 2), blocks=1))
    cond = torch.randn(2, 1, 64, 128)
    assert r(cond).shape == cond.shape
    r.loss(torch.randn_like(cond), cond, torch.ones_like(cond)).backward()


def test_noise_schedules():
    from grace_diffusion.model.edm import EDM
    net = UNet(base=8, mults=(1, 2), blocks=1)
    s = EDM(net, noise="loguniform", sigma_max=20.0).draw_sigma(20000, "cpu")
    assert s.min() >= 0.002 and s.max() <= 20.0
    assert abs(float((s > 1.0).float().mean()) - np.log(20) / np.log(1e4)) < 0.02
    s = EDM(net, p_mean=-0.4, p_std=1.6).draw_sigma(20000, "cpu")
    assert abs(float(s.log().mean()) + 0.4) < 0.05
