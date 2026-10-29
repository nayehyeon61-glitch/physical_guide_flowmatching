import torch

from pgcfm.core import Core, CoreConfig
from pgcfm.fno_core import FNONet, double_sphere


def _norm(u):  # domain-normalized L2 over (C, H, W) per sample
    return (u ** 2).mean((-2, -1)).sum(-1).sqrt()


def test_double_sphere_is_an_isometry_and_continues_over_the_pole():
    x = torch.randn(2, 3, 8, 16)
    y = double_sphere(x)
    assert y.shape == (2, 3, 16, 16)
    assert torch.allclose(_norm(y), _norm(x))
    # the row after the last latitude is the last latitude at the opposite longitude
    assert torch.allclose(y[..., 8, :], torch.roll(x[..., 7, :], 8, dims=-1))


def test_projection_bounds_every_part_and_the_lipschitz_bound_holds():
    torch.manual_seed(0)
    net = FNONet(5, 4, width=8, modes=(3, 3), n_layers=2, bound=1.5)
    with torch.no_grad():
        net.head.weight.normal_()  # the zero head would make the test trivial
        for blk in net.blocks:
            blk.spec.w_pos.mul_(50)
    net.project_()
    assert net.clip_log[-1] > 0  # the scaled weights were above the bound
    for blk in net.blocks:
        assert float(blk.l2_operator_norm()) <= 1.5 + 1e-4
    L = net.lipschitz_bound()
    assert L <= 2 ** 0.5 * (1.129 * 1.5) ** 2 + 1e-4
    x, y = torch.randn(16, 5, 8, 16), torch.randn(16, 5, 8, 16)
    ratio = _norm(net(x) - net(y)) / _norm(x - y)
    assert float(ratio.max()) <= L


def test_core_with_fno_starts_as_persistence():
    cfg = CoreConfig(mode="jump6", net="fno", fno_width=8, fno_modes=3, fno_layers=2, bound=1.0)
    tend = {6: torch.ones(5), 24: torch.ones(5)}
    core = Core(cfg, torch.zeros(8, 8, 16), tend)
    frames = torch.randn(2, 3, 5, 8, 16)
    out = core.rollout(frames, torch.zeros(2), 3)
    assert torch.allclose(out, frames[:, 2:3].expand(-1, 3, -1, -1, -1))
