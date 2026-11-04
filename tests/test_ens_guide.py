import torch

from pgcfm.ens_guide import (BandGuide, apply_gamma, band_counts, band_index, band_spread2, calibrated_energy,
                             doob_stats, fair_crps, monotone_gamma, spread_skill_gamma)
from pgcfm.seqdct import dct_2d

N, M, C, H, W = 6, 8, 2, 16, 32


def test_gamma_one_is_identity_and_energy_follows_parseval():
    torch.manual_seed(0)
    ens = torch.randn(N, M, C, H, W)
    bidx = band_index(H, W)
    nb = int(bidx.max()) + 1
    assert nb == 5 and (bidx[0, 0] == 0) and (bidx[-1, -1] == nb - 1)
    out = apply_gamma(ens.mean(1), ens, torch.ones(N, C, nb), bidx)
    assert torch.allclose(out, ens, atol=1e-5)
    g = torch.rand(N, C, nb) + 0.5
    out = apply_gamma(ens.mean(1), ens, g, bidx)
    var = ((out - out.mean(1, keepdim=True)) ** 2).sum(1).sum((-1, -2)) / (M - 1)  # grid energy (N, C)
    assert torch.allclose(var, calibrated_energy(g, band_spread2(ens, bidx, nb), band_counts(bidx, nb)), rtol=1e-4)


def test_spread_skill_gamma_recovers_an_overdispersion_factor():
    torch.manual_seed(1)
    truth_sd, k = 1.0, 2.5
    mu = torch.zeros(4000, 1, 4, 4)
    obs = truth_sd * torch.randn(4000, 1, 4, 4)
    ens = k * truth_sd * torch.randn(4000, M, 1, 4, 4)  # k times too wide
    one = torch.zeros(4, 4, dtype=torch.long)
    err2 = (dct_2d(obs - mu) ** 2).mean((-1, -2))[..., None]
    g = spread_skill_gamma(err2, band_spread2(ens, one, 1), M)
    assert abs(float(g) - (M / (M + 1)) ** 0.5 / k) < 0.03


def test_monotone_gamma_and_doob_stats():
    gamma = torch.ones(3, 1, 2)
    spread2 = torch.tensor([[[1.0, 1.0]], [[0.5, 0.5]], [[2.0, 2.0]]])  # energy falls at lead 2
    counts = torch.tensor([1.0, 3.0])
    g = monotone_gamma(gamma, spread2, counts)
    S = calibrated_energy(g, spread2, counts)[:, 0]
    assert torch.all(S[1:] >= S[:-1] - 1e-6) and torch.allclose(g[2], gamma[2])
    e = torch.tensor([1.0, 3.0, 2.0]).view(1, 1, 3, 1).expand(5, 2, 3, 1)
    d = doob_stats(e)
    assert abs(d["doob_ratio"][0] - 1.5) < 1e-6 and abs(d["falling_steps"][0] - 0.5) < 1e-6


def test_fair_crps_and_band_guide_start():
    torch.manual_seed(2)
    obs = torch.randn(200, 1, 4, 4)
    good, wide = torch.randn(200, M, 1, 4, 4), 3 * torch.randn(200, M, 1, 4, 4)
    w = torch.ones(4, 1)
    assert fair_crps(good, obs, w) < fair_crps(wide, obs, w)
    gd = BandGuide(C, 5, 3)
    gd.base.copy_(torch.full((3, C, 5), 0.3).log())
    sp = torch.rand(N, C, 5) + 0.1
    gd.fit_inputs(sp)
    out = gd(sp, torch.zeros(N, dtype=torch.long), torch.zeros(N))
    assert torch.allclose(out, torch.full_like(out, 0.3), atol=1e-6)
