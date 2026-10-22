import pytest
import torch

from pgcfm import contract, metrics
from pgcfm.cfm import TISR_DYN, CFMConfig, CFMTransition, heun, rollout

H, W = 25, 48
SMALL = dict(model_channels=32, channel_mult=(1, 2), num_res_blocks=1)


def batch(B=2, h=H, w=W, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, 66, h, w, generator=g), torch.randn(B, 66, h, w, generator=g),
            torch.randn(B, 1, h, w, generator=g), torch.tensor([10, 200])[:B])


@pytest.mark.parametrize("mode", ["target", "forcing"])
def test_loss_backward_and_sample_shapes(mode):
    torch.manual_seed(0)
    m = CFMTransition(CFMConfig(tisr_mode=mode, **SMALL), H, W)
    x, y, s, d = batch()
    loss = m.loss(x, y, s, d)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    out = m.sample(x, s, d, tisr_next=y[:, TISR_DYN:TISR_DYN + 1], steps=2)
    assert out.shape == (2, 66, H, W)
    assert not out.requires_grad
    if mode == "forcing":
        assert torch.equal(out[:, TISR_DYN], y[:, TISR_DYN])


def test_sample_is_differentiable_for_physics_loss():
    """PG-CFM needs dL/dtheta through the generated endpoint (no detach)."""
    torch.manual_seed(0)
    m = CFMTransition(CFMConfig(**SMALL), H, W)
    x, _, s, d = batch()
    out = m.sample(x, s, d, steps=2, grad=True)
    out.pow(2).mean().backward()
    g = [p.grad for p in m.parameters()]
    assert all(t is not None for t in g) and sum(t.abs().sum() for t in g) > 0


def test_cfm_target_matches_paper_eq3():
    """torchcfm with source=eps, target=y gives z = (1-tau) eps + tau y, u = y - eps."""
    m = CFMTransition(CFMConfig(**SMALL), H, W)
    eps, y = torch.randn(3, 66, 4, 4), torch.randn(3, 66, 4, 4)
    tau = torch.tensor([0.0, 0.3, 1.0])
    _, z, u = m.fm.sample_location_and_conditional_flow(eps, y, t=tau)
    t = tau[:, None, None, None]
    assert torch.allclose(z, (1 - t) * eps + t * y)
    assert torch.allclose(u, y - eps)


def test_heun_exact_for_constant_velocity_and_linear_field():
    z0 = torch.randn(2, 3)
    target = torch.randn(2, 3)
    # Optimal CFM field for a point mass at `target`: v = (target - z) / (1 - tau)
    # is singular at tau=1; the straight-line field from a fixed eps is constant.
    v = target - z0
    assert torch.allclose(heun(lambda t, z: v, z0, 4), target, atol=1e-6)
    # dz/dtau = tau -> z(1) = z0 + 1/2, Heun integrates linear-in-tau exactly.
    out = heun(lambda t, z: t[:, None].expand_as(z), z0, 3)
    assert torch.allclose(out, z0 + 0.5, atol=1e-6)


def test_rollout_members_are_markov_and_distinct():
    torch.manual_seed(0)
    m = CFMTransition(CFMConfig(tisr_mode="forcing", **SMALL), H, W).eval()
    x, _, s, _ = batch(B=1)
    doy = torch.tensor([[5, 6, 7]])
    tisr = torch.randn(1, 2, 1, H, W)
    ens = rollout(m, x, s, doy, n_members=3, tisr=tisr, steps=2, seed=1)
    assert ens.shape == (1, 3, 2, 66, H, W)
    # Fresh noise per member: members differ.
    assert (ens[0, 0, 0] - ens[0, 1, 0]).abs().mean() > 1e-3
    # Known forcing is copied, never generated.
    assert torch.equal(ens[0, :, 1, TISR_DYN], tisr[0, 1, 0].expand(3, H, W))
    # Same seed -> same ensemble; lead 2 of member 0 is one transition from its own lead 1.
    again = rollout(m, x, s, doy, n_members=3, tisr=tisr, steps=2, seed=1)
    assert torch.equal(ens, again)


def test_full_kai_grid_forward():
    """121x240 needs lat padding for the U-Net; output must be cropped back."""
    m = CFMTransition(CFMConfig(**SMALL), contract.N_LAT, contract.N_LON)
    x, y, s, d = batch(B=1, h=contract.N_LAT, w=contract.N_LON)
    with torch.no_grad():
        out = m.sample(x, s, d, steps=1)
    assert out.shape == (1, 66, contract.N_LAT, contract.N_LON)


def test_scores_on_known_ensembles():
    torch.manual_seed(0)
    obs = torch.zeros(1, 30, 40)
    # Calibrated: obs and members exchangeable N(0,1) -> (M+1)/M * spread^2 ~ MSE.
    truth, ens = torch.randn(2000, 5, 4), torch.randn(16, 2000, 5, 4)
    ratio = (17 / 16 * metrics.ens_spread(ens).pow(2).mean() / metrics.ens_mean_rmse(ens, truth).pow(2).mean()).sqrt()
    assert 0.95 < ratio < 1.05
    # Point ensemble at the truth has zero CRPS; CRPS of N(0,1) vs 0 is ~0.234.
    assert metrics.crps(obs.expand(5, 1, 30, 40).clone(), obs).abs().max() < 1e-7
    c = metrics.crps(torch.randn(200, 1, 30, 40), obs)
    assert abs(c.item() - 0.2337) < 0.02


def test_velocity_is_equivariant_to_longitude_roll():
    """No dateline seam: rolling all inputs in longitude rolls the output."""
    torch.manual_seed(0)
    m = CFMTransition(CFMConfig(**SMALL), 24, 48).eval()
    with torch.no_grad():  # torchcfm zero-initializes output convs; make the map non-trivial
        for p in m.parameters():
            p.normal_(0.0, 0.1)
    x, y, s, d = batch(h=24, w=48)
    cond = m.condition(x, s, d)
    tau = torch.tensor([0.3, 0.7])
    with torch.no_grad():
        v = m.net(tau, y, cond)
        v_roll = m.net(tau, y.roll(16, -1), cond.roll(16, -1))
    # A shift by a multiple of the total downsampling (2) keeps the pooling grid aligned.
    assert v.abs().mean() > 1e-2
    assert torch.allclose(v.roll(16, -1), v_roll, atol=1e-4 * v.abs().max().item())
