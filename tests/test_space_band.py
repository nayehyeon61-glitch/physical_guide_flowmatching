import torch

from pgcfm.adaptive_band import BandPool, OrthoDCT2
from pgcfm.ens_guide import band_index
from pgcfm.space_band import (BandProjector, CondSpaceBandVariance, SpaceBandOperator, band_cell_grids,
                              local_error_energy, space_band_nll, upsample_sphere)


def _setup(H=16, W=32, n_min=4):
    bidx = band_index(H, W)
    nb = int(bidx.max()) + 1
    pool = BandPool(bidx, nb)
    return BandProjector(OrthoDCT2(H, W), pool), band_cell_grids(pool.n_b, H, W, n_min), nb


def test_upsample_is_convex_periodic_in_longitude_and_keeps_constants():
    m = torch.rand(2, 3, 5)
    up = upsample_sphere(m, 16, 32)
    assert up.shape == (2, 16, 32)
    assert float(up.max()) <= float(m.max()) + 1e-6 and float(up.min()) >= float(m.min()) - 1e-6
    torch.testing.assert_close(upsample_sphere(torch.full((1, 3, 7), 2.0), 32, 64), torch.full((1, 32, 64), 2.0))
    shifted = upsample_sphere(torch.roll(m, 1, -1), 16, 30)  # one cell = 6 columns at w = 5, W = 30
    torch.testing.assert_close(shifted, torch.roll(upsample_sphere(m, 16, 30), 6, -1), atol=1e-5, rtol=1e-5)


def test_band_projections_sum_to_identity_and_cell_energies_add_up():
    torch.manual_seed(0)
    proj, grids, nb = _setup()
    r = torch.randn(3, 2, 16, 32)
    torch.testing.assert_close(sum(proj.project_all(r)), r, atol=1e-4, rtol=1e-4)
    divisible = [(1, 1)] * nb  # exact partition: the energy of the band-passed fields adds up to |r|^2
    tot = sum(E.mean((-2, -1)) for E in local_error_energy(proj, r, divisible))
    torch.testing.assert_close(tot, (r ** 2).mean((-2, -1)), atol=1e-4, rtol=1e-4)


def test_space_band_operator_is_bounded_by_the_largest_multiplier_and_starts_at_climatology():
    torch.manual_seed(0)
    proj, grids, nb = _setup()
    clim = [torch.randn(2, 2, *g) for g in grids]
    net = CondSpaceBandVariance(3, 2, 2, grids, clim, width=8)
    feats, scal, lead = torch.randn(4, 3, 16, 32), torch.randn(4, 2), torch.tensor([0, 1, 1, 0])
    out = net(feats, scal, lead)
    for b in range(nb):
        torch.testing.assert_close(out[b], clim[b][lead])
    assert torch.isfinite(space_band_nll(out, [o.exp() for o in out], torch.ones(nb)))
    x, mu = torch.randn(4, 6, 2, 16, 32), torch.zeros(4, 2, 16, 32)
    y, info = SpaceBandOperator(proj, grids, gamma_max=1.0)(x, mu, out)
    lhs = (y - mu[:, None]).pow(2).sum((2, 3, 4)).sqrt()
    rhs = (x - x.mean(1, keepdim=True)).pow(2).sum((2, 3, 4)).sqrt()
    assert (lhs <= info["op_norm_bound"][:, None] * rhs * 1.0001 + 1e-5).all()
    assert float(info["op_norm_bound"].max()) <= 1.0 + 1e-6
