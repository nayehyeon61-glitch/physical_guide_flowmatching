import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import baselines as bl  # noqa: E402


def _step(**kw):
    torch.manual_seed(0)
    return bl.UCastStep(9, 16, torch.full((66,), 0.5), model_channels=16, channel_mult=(1, 2), num_blocks=1, **kw)


def test_ucast_step_starts_at_persistence_and_restores_conv2d():
    """U-Cast zero-initializes the output conv, so an untrained step is persistence; the upstream
    circular padding patch is undone after every forward."""
    orig = F.conv2d
    step = _step().eval()
    x, st, d = torch.randn(2, 66, 9, 16), torch.randn(2, 1, 9, 16), torch.tensor([3, 200])
    torch.testing.assert_close(step(x, st, d), x)
    assert F.conv2d is orig
    loss = step.loss(x, x + 0.1, st, d, "wmae")
    assert torch.isfinite(loss) and F.conv2d is orig


def test_ucast_longitude_is_periodic():
    """Rolling the input in longitude rolls the output (circular padding on U-Cast's height axis)."""
    step = _step().eval()
    with torch.no_grad():
        for p in step.net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x, st, d = torch.randn(1, 66, 9, 16), torch.randn(1, 1, 9, 16), torch.tensor([10])
    out = step(x, st, d)
    rolled = step(x.roll(4, -1), st.roll(4, -1), d)
    torch.testing.assert_close(rolled, out.roll(4, -1), atol=1e-4, rtol=1e-4)


def test_ucast_crps_stage_uses_dropout_members():
    step = _step(loss_function="wcrps", members=2)
    with torch.no_grad():
        for p in step.net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    step.train()
    x, st, d = torch.randn(2, 66, 9, 16), torch.randn(2, 1, 9, 16), torch.tensor([3, 4])
    loss = step.loss(x, x, st, d, "wcrps")
    loss.backward()
    assert torch.isfinite(loss)
    step.eval()
    step.set_dropout(True, 0.2)
    a, b = step(x, st, d), step(x, st, d)
    assert not torch.allclose(a, b)
    step.set_dropout(False)
    torch.testing.assert_close(step(x, st, d), step(x, st, d))


def test_scher_perturbations():
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(2, 66, 9, 16)
    e = bl.scher_rand(x0, 6, 0.3, g)
    torch.testing.assert_close(e.mean(1), x0, atol=1e-5, rtol=0)
    torch.testing.assert_close(e.std(1, unbiased=False), torch.full_like(x0, 0.3), atol=1e-5, rtol=0)
    sv = torch.linalg.qr(torch.randn(x0[0].numel(), 3))[0].T[None].repeat(2, 1, 1)
    p = bl.scher_svd(x0, sv, 6, 1.0, g)
    torch.testing.assert_close(p[:, 0::2] + p[:, 1::2], 2 * x0[:, None].expand(2, 3, *x0.shape[1:]))


def test_jacobian_svecs_are_orthonormal_and_leading():
    step = _step().eval()
    with torch.no_grad():
        for p in step.net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x, st, d = torch.randn(66, 9, 16), torch.randn(1, 9, 16), torch.tensor([5])
    V = bl.jacobian_svecs(step, x, st, d, k=3, iters=8)
    torch.testing.assert_close(V @ V.T, torch.eye(3), atol=1e-4, rtol=0)
    f = lambda v: torch.autograd.functional.jvp(  # noqa: E731
        lambda z: step(z, st[None], d), x[None], v.view(1, *x.shape))[1].norm()
    lead = f(V[0])
    rand = torch.stack([f(r / r.norm()) for r in torch.randn(5, x.numel())])
    assert (lead > rand).all()


def test_crps_normal_matches_scoringrules():
    sr = pytest.importorskip("scoringrules")
    import numpy as np
    g = np.random.default_rng(0)
    mu, s, y = g.normal(size=20), g.uniform(0.2, 2, size=20), g.normal(size=20)
    ours = bl.crps_normal(torch.tensor(mu), torch.tensor(s), torch.tensor(y)).numpy()
    np.testing.assert_allclose(ours, sr.crps_normal(y, mu, s), rtol=1e-6, atol=1e-8)


def test_drn_quantile_members():
    torch.manual_seed(0)
    fc = torch.randn(8, 2, 3, 4, 5)
    obs = fc + 0.5 * torch.randn_like(fc)
    models = bl.fit_drn(fc, obs, n_valid=2, epochs=3, batch=64)
    ens = bl.drn_members(models, fc, 4)
    assert ens.shape == (8, 4, 2, 3, 4, 5)
    assert (ens[:, 1:] >= ens[:, :-1]).all()  # quantile members are ordered
