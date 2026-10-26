import torch

from pgcfm.flow_sigma import FlowSigma, fit_flow_sigma, flow_features, predict_sigma

N, H, W = 64, 16, 32


def inputs(seed):
    g = torch.Generator().manual_seed(seed)
    x0, mu = torch.randn(N, 5, H, W, generator=g), torch.randn(N, 5, H, W, generator=g)
    feats = flow_features(x0, mu, torch.full((N,), 24.0), 48.0 * torch.arange(N), torch.randn(8, H, W, generator=g))
    true_sd = 0.3 + 1.2 * (x0 > 0).float()  # the error size follows the "flow" x0
    return feats, mu, mu + true_sd * torch.randn(N, 5, H, W, generator=g), true_sd


def test_starts_at_the_climatological_sigma():
    feats, *_ = inputs(0)
    clim = torch.rand(6, 5, H, W) + 0.5
    lead_idx = torch.randint(0, 6, (N,))
    assert torch.allclose(predict_sigma(FlowSigma(feats.shape[1], clim), feats, lead_idx), clim[lead_idx])


def test_learns_a_flow_dependent_sigma():
    torch.manual_seed(0)
    feats, mu, tgt, true_sd = inputs(1)
    clim = ((tgt - mu) ** 2).mean(0, keepdim=True).sqrt()
    model = FlowSigma(feats.shape[1], clim, layers=(1, 1), hidden=(16, 16))
    lead_idx = torch.zeros(N, dtype=torch.long)
    info = fit_flow_sigma(model, feats, mu, tgt, lead_idx, torch.arange(48), torch.arange(48, N), epochs=8,
                          lr=3e-3, log=lambda q: None)
    assert info["best"] < info["val_nll_start"] - 0.1
    sig = predict_sigma(model, feats, lead_idx)[48:]
    a, b = sig.flatten() - sig.mean(), true_sd[48:].flatten() - true_sd[48:].mean()
    assert float((a * b).sum() / (a.norm() * b.norm())) > 0.5
