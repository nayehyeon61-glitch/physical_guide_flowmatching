import torch

from pgcfm import contract
from pgcfm.cfm import CFMConfig, CFMTransition, rollout
from pgcfm.coupling import PGCFM, PhysicsConfig, PhysicsGuide, deterministic_rollout, estimate_scales
from pgcfm.projection import (GradientFlowProjection, ODEProjection, ProjectionConfig, freeze,
                              sequence_terms, train_pinn, train_projection)

H, W = 17, 32


def _batch(B=2, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x0": 0.3 * torch.randn(B, 66, H, W, generator=g),
            "targets": 0.3 * torch.randn(B, K, 66, H, W, generator=g),
            "static": torch.zeros(B, 1, H, W), "doy": torch.arange(10, 11 + K).repeat(B, 1)}


def _model():
    torch.manual_seed(0)
    cfm = CFMTransition(CFMConfig(channel_mult=(1, 2)), H, W)
    guide = PhysicsGuide(H, W, contract.REFERENCE_MEAN, contract.REFERENCE_STD, PhysicsConfig())
    return PGCFM(cfm, guide)


def _proj(m, **kw):
    pc = ProjectionConfig(model_channels=32, channel_mult=(1, 2), ode_steps=1, cfm_steps=1, **kw)
    return ODEProjection(m.cfm, pc, H, W)


def test_projection_starts_as_identity_and_freezes_cfm():
    """The torchcfm U-Net's last layer is zero-initialized, so g_phi = 0 and the
    untrained projection returns the CFM proposal unchanged."""
    m, b = _model(), _batch()
    proj = _proj(m)
    x, st, d = b["x0"], b["static"], b["doy"][:, 1]
    eps = torch.randn(2, 66, H, W)
    torch.testing.assert_close(proj.sample(x, st, d, steps=1, eps=eps), m.cfm.sample(x, st, d, steps=1, eps=eps))
    assert not any(p.requires_grad for p in proj.cfm.parameters())
    assert all(p.requires_grad for p in proj.field.parameters())


def test_sequence_loss_backprops_through_rollout():
    """Lead-2 CRPS reaches g_phi through the lead-1 projection and the frozen CFM (BPTT)."""
    m, b = _model(), _batch(K=2)
    proj = _proj(m)
    for p in proj.field.unet.out.parameters():  # leave the identity so x_1 depends on phi
        torch.nn.init.normal_(p, std=1e-2)
    terms, logs = sequence_terms(proj, m.guide, b, horizon=2)
    assert set(logs) >= {"crps", "crps_d1", "crps_d2"}
    terms["crps"].backward()
    assert proj.field.unet.out[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in proj.cfm.parameters())


def test_frozen_pinn_physics_term_and_grad_cap():
    m, b = _model(), _batch(K=2)
    m.guide.scales = estimate_scales(m.guide, [b])
    freeze(m.guide)
    proj = _proj(m, lambda_p=0.1, grad_ratio=0.5, curriculum=(1, 2))
    hist = train_projection(proj, m.guide, [b], iters=2, log=lambda *a: None)
    assert hist[0]["horizon"] == 1 and hist[1]["horizon"] == 2
    assert 0 < hist[0]["phys_scale"] <= 1.0 and "grad_phys" in hist[0]
    assert all(p.grad is None for p in m.guide.parameters())


def test_train_pinn_stops_and_freezes():
    m, b = _model(), _batch(K=1)
    m.guide.scales = estimate_scales(m.guide, [b])
    hist = train_pinn(m, [b], [b], max_iters=6, window=2, tol=1e9, patience=1, log=lambda *a: None)
    assert len(hist) == 2  # one window, then "no improvement > tol" stops it
    assert not any(p.requires_grad for p in m.guide.parameters())


def test_rollouts_run_on_projections():
    m, b = _model(), _batch(B=1, K=2)
    m.guide.scales = estimate_scales(m.guide, [b])
    for proj in (_proj(m), GradientFlowProjection(m.cfm, freeze(m.guide), steps=2)):
        ens = rollout(proj, b["x0"], b["static"], b["doy"], n_members=2, steps=1)
        det = deterministic_rollout(proj, b["x0"], b["static"], b["doy"], steps=1)
        assert ens.shape == (1, 2, 2, 66, H, W) and det.shape == (1, 2, 66, H, W)
        assert torch.isfinite(ens).all()
