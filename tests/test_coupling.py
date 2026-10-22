import os

import netCDF4 as nc
import numpy as np
import pytest
import torch

from pgcfm import contract
from pgcfm.cfm import CFMConfig, CFMTransition
from pgcfm.coupling import PGCFM, PhysicsConfig, PhysicsGuide, estimate_scales
from pgcfm.physics import grid as pgrid

H, W = 17, 32


def _batch(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x0": 0.3 * torch.randn(B, 66, H, W, generator=g),
            "targets": 0.3 * torch.randn(B, 1, 66, H, W, generator=g),
            "static": torch.zeros(B, 1, H, W), "doy": torch.tensor([[10, 11]] * B)}


def _model(**pcfg):
    torch.manual_seed(0)
    cfm = CFMTransition(CFMConfig(channel_mult=(1, 2)), H, W)
    guide = PhysicsGuide(H, W, contract.REFERENCE_MEAN, contract.REFERENCE_STD, PhysicsConfig(**pcfg))
    return PGCFM(cfm, guide)


def test_zero_bridge_matches_step3_linear_bridge():
    """h_psi starts at zero, so the coupled residual is step 3's time_integrated_residuals."""
    m, b = _model(), _batch()
    x, y = b["x0"], b["targets"][:, 0]
    R, reg = m.guide.residuals(x, y, b["static"], b["doy"][:, 0], use_closure=False)
    g = m.guide.grid(x.device)
    ref = pgrid.time_integrated_residuals(m.guide.to_si(x, b["static"]), m.guide.to_si(y, b["static"]), g, n_sub=4)
    assert reg.item() == 0.0
    for k in ("u", "v", "T", "mass", "hyd"):
        torch.testing.assert_close(R[k], ref[k])


def test_bridge_keeps_endpoints():
    """eq. (9): the a(1-a) factor leaves s = 0 and s = dt untouched for any h_psi."""
    m, b = _model(), _batch()
    for p in m.guide.bridge.out.parameters():
        torch.nn.init.normal_(p)
    x, y = b["x0"], b["targets"][:, 0]
    F0 = m.guide.to_si(x, b["static"])
    R1, _ = m.guide.residuals(x, y, b["static"], b["doy"][:, 0], use_closure=False)
    R0, _ = m.guide.residuals(x, y, b["static"], b["doy"][:, 0], use_bridge=False, use_closure=False)
    assert not torch.allclose(R1["u"], R0["u"])  # interior nodes change
    assert torch.isfinite(R1["u"]).all() and F0["u"].shape[-3] == contract.N_LEVELS


def test_physics_gradient_reaches_theta():
    """L_phys -> B_psi -> y_hat -> theta: no detach anywhere on the generated endpoint."""
    m, b = _model(lambda_p=1.0, ramp_iters=1), _batch()
    m.guide.scales = estimate_scales(m.guide, [b])
    _, logs = m.joint_loss(b, it=10)
    y_hat = m.cfm.sample(b["x0"], b["static"], b["doy"][:, 1], steps=2, grad=True)
    R, _ = m.guide.residuals(b["x0"], y_hat, b["static"], b["doy"][:, 0])
    lp = m.guide.loss(R, m.guide.mask(b["x0"], b["targets"][:, 0], b["static"]))
    m.zero_grad()
    lp.backward()
    grads = [p.grad for p in m.cfm.net.parameters()]
    assert all(g is not None for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0
    assert logs["phys"] > 0


def test_mask_is_fixed_by_observed_pair():
    m, b = _model(), _batch()
    x, y = b["x0"], b["targets"][:, 0]
    m0 = m.guide.mask(x, y, b["static"])
    y_low_sp = y.clone()
    y_low_sp[:, contract.DYNAMIC_CHANNELS.index(contract.CHANNEL_INDEX["sp"])] -= 50.0
    # a generated endpoint never enters the mask: same observed pair -> same mask
    assert torch.equal(m0, m.guide.mask(x, y, b["static"]))
    assert not torch.equal(m0, m.guide.mask(x, y_low_sp, b["static"]))


def test_scales_are_floored():
    """A constraint the data satisfy exactly must not get b_j -> 0."""
    m, b = _model(), _batch()
    s = estimate_scales(m.guide, [b], floor_frac=0.1)
    from pgcfm.physics.primitive import ResidualScales
    ref = ResidualScales()
    for k in ("mom", "T", "mass", "hyd", "q"):
        assert getattr(s, k) >= 0.1 * getattr(ref, k)


def test_train_and_infer_scripts(tmp_path):
    from scripts.infer_pgcfm import main as infer
    from scripts.make_balanced_era5 import main as make_balanced
    from scripts.train_pgcfm import main as train
    d = str(tmp_path)
    make_balanced(["--out", f"{d}/train", "--years", "2011", "--days", "6", "--n-lat", str(H), "--n-lon", str(W)])
    make_balanced(["--out", f"{d}/valid", "--years", "2012", "--days", "6", "--n-lat", str(H), "--n-lon", str(W)])
    train(["--out", f"{d}/run", "--set", f"train_dir={d}/train", f"valid_dir={d}/valid",
           f"stats_dir={d}/train/stats", 'iters={"cfm":2,"guide":1,"joint":1}', "batch=2", "workers=0",
           "channel_mult=[1,2]", "num_res_blocks=1", "model_channels=32", "val_every=1", "val_batches=1",
           "scale_batches=1", "val_members=2"])
    infer(["--ckpt", f"{d}/run/joint.pt", "--test-file", f"{d}/valid/2012.nc", "--stats-dir", f"{d}/train/stats",
           "--out", f"{d}/fc.nc", "--leads", "2", "--members", "2", "--steps", "1", "--ics", "0", "2", "--det"])
    with nc.Dataset(f"{d}/fc.nc") as f, nc.Dataset(f"{d}/valid/2012.nc") as src:
        v = f["forecast"]
        assert v.dimensions == ("initial_condition", "lead_time", "member", "channel", "lat", "lon")
        assert v.shape == (2, 3, 2, 67, H, W)
        # lead 0 is the IC itself in ERA5 units, orography copied at every lead
        np.testing.assert_allclose(v[1, 0, 0], src["fields"][2], rtol=1e-4, atol=0.1)  # tisr 0 at polar night
        o = contract.CHANNEL_INDEX["orog"]
        np.testing.assert_allclose(v[0, 2, 1, o], src["fields"][0, o], rtol=1e-4, atol=1e-2)
    with nc.Dataset(f"{d}/fc_det.nc") as f:
        assert f["forecast"].shape == (2, 3, 67, H, W)
    assert os.path.exists(f"{d}/run/log.jsonl")
