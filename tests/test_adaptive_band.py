import torch

from pgcfm.adaptive_band import (AdaptiveBandOperator, BandPool, CondBandVariance, OrthoDCT2, chi2_band_nll,
                                 estimate_n_eff)
from pgcfm.ens_guide import band_index
from pgcfm.seqdct import dct_2d


def _setup(H=16, W=32):
    bidx = band_index(H, W)
    nb = int(bidx.max()) + 1
    return OrthoDCT2(H, W), BandPool(bidx, nb), nb


def test_ortho_dct_matches_the_regm_dct_and_pool_keeps_parseval():
    dct, pool, _ = _setup()
    u = torch.randn(3, 2, 16, 32)
    torch.testing.assert_close(dct(u), dct_2d(u), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(dct.inverse(dct(u)), u, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(pool.mean_energy(dct(u)) @ pool.n_b, (u ** 2).sum((-2, -1)), atol=1e-3, rtol=1e-4)


def test_operator_norm_is_the_largest_gamma_and_realized_spread_matches_s2():
    torch.manual_seed(0)
    dct, pool, nb = _setup()
    x, mu = torch.randn(2, 8, 3, 16, 32), torch.zeros(2, 3, 16, 32)
    log_s2 = torch.full((2, 3, nb), -1.0)
    T = AdaptiveBandOperator(dct, pool, gamma_max=10.0, normalize="realized")
    y, g, info = T(x, mu, log_s2)
    torch.testing.assert_close(T.ensemble_band_spread(dct(y - y.mean(1, keepdim=True))), log_s2.exp(), rtol=1e-3,
                               atol=1e-5)
    torch.testing.assert_close(info["op_norm"], g.amax((-1, -2)))
    dev, out = x - x.mean(1, keepdim=True), y - mu[:, None]
    ratio = (out ** 2).sum((-1, -2, -3)).sqrt() / (dev ** 2).sum((-1, -2, -3)).sqrt()
    assert float(ratio.max()) <= float(g.max()) + 1e-4
    y1, g1, info1 = AdaptiveBandOperator(dct, pool, gamma_max=1.0)(x, mu, torch.full((2, 3, nb), 5.0))
    assert float(g1.max()) <= 1.0 and float(info1["clip_frac"]) == 1.0


def test_band_variance_starts_at_climatology_and_the_nll_is_minimal_at_the_energy():
    torch.manual_seed(0)
    clim = torch.randn(3, 2, 5)
    net = CondBandVariance(4, 2, 2, 5, clim, width=8)
    lead = torch.tensor([0, 2, 1])
    torch.testing.assert_close(net(torch.randn(3, 4, 8, 16), torch.randn(3, 2), lead), clim[lead])
    Eb, n_eff = torch.rand(3, 2, 5) + 0.1, torch.ones(5)
    best = chi2_band_nll(Eb.log(), Eb, n_eff)
    assert best < chi2_band_nll(Eb.log() + 0.1, Eb, n_eff) and best < chi2_band_nll(Eb.log() - 0.1, Eb, n_eff)
    n = estimate_n_eff(torch.distributions.Chi2(20.0).sample((4000, 2, 5)) / 20, torch.ones(4000, 2, 5),
                       torch.full((5,), 100.0))
    assert ((n - 20).abs() < 3).all()
