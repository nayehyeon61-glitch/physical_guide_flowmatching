import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import metrics  # noqa: E402
from pgcfm.structure import (SpreadCalibrator, StructureOperator, crps_from_parts, crps_parts,  # noqa: E402
                             match_spread, rescale_ensemble_spread, structure_scores)


def test_rescale_ensemble_spread_classical1():
    """The owner's classical1 exercise, checks as written there."""
    members = torch.tensor([[[[[0.0]]], [[[2.0]]]]], requires_grad=True)  # [B=1, K=2, C=1, H=1, W=1]
    channel_scales = torch.tensor([2.0])
    out = rescale_ensemble_spread(members, channel_scales)
    torch.testing.assert_close(out["mean"], torch.tensor([[[[1.0]]]]))
    torch.testing.assert_close(out["anomalies"], torch.tensor([[[[[-1.0]]], [[[1.0]]]]]))
    torch.testing.assert_close(out["calibrated_members"], torch.tensor([[[[[-1.0]]], [[[3.0]]]]]))
    torch.testing.assert_close(out["spread_before"], torch.tensor([[[[1.0]]]]))
    torch.testing.assert_close(out["spread_after"], torch.tensor([[[[2.0]]]]))
    torch.testing.assert_close(out["calibrated_members"].mean(dim=1), out["mean"])
    torch.testing.assert_close(out["spread_after"], out["spread_before"] * channel_scales.reshape(1, 1, 1, 1))
    target = torch.tensor([[[[3.0]]]])
    mean_rmse = (out["mean"] - target).square().mean().sqrt()
    torch.testing.assert_close(mean_rmse, torch.tensor(2.0))
    torch.testing.assert_close(out["spread_before"].square().mean().sqrt() / mean_rmse, torch.tensor(0.5))
    torch.testing.assert_close(out["spread_after"].square().mean().sqrt() / mean_rmse, torch.tensor(1.0))
    identity = rescale_ensemble_spread(members.detach(), torch.tensor([1.0]))
    torch.testing.assert_close(identity["calibrated_members"], members.detach())
    collapsed = rescale_ensemble_spread(members.detach(), torch.tensor([0.0]))
    torch.testing.assert_close(collapsed["calibrated_members"], torch.ones_like(members))
    torch.testing.assert_close(collapsed["spread_after"], torch.zeros_like(out["spread_after"]))
    permuted = rescale_ensemble_spread(members.detach().flip(1), channel_scales)
    torch.testing.assert_close(permuted["mean"], out["mean"].detach())
    torch.testing.assert_close(permuted["calibrated_members"], out["calibrated_members"].detach().flip(1))
    loss = (out["calibrated_members"] - target[:, None]).square().mean()
    loss.backward()
    assert members.grad is not None and torch.isfinite(members.grad).all() and members.grad.abs().sum() > 0


def test_crps_parts_match_metrics():
    g = torch.Generator().manual_seed(0)
    e, o = torch.randn(6, 3, 4, 5, generator=g), torch.randn(3, 4, 5, generator=g)
    skill, pair = crps_parts(e, o)
    brute = (e[:, None] - e[None]).abs().sum((0, 1))
    torch.testing.assert_close(pair, brute)
    w = metrics.area_weights(4)
    fair = (crps_from_parts(skill, pair, 6, "fair") * w).mean((-2, -1))
    torch.testing.assert_close(fair, metrics.crps(e, o))
    emp = crps_from_parts(skill, pair, 6, "empirical")
    torch.testing.assert_close(emp, skill - 0.5 * brute / 36)


def test_spread_calibrator_recovers_scale():
    """Members drawn with half the truth's spread in channel 0 and the right spread in channel 1.
    For mu + s (x - mu) the expected empirical CRPS is minimized at the true scale (s = 1 for a
    perfect ensemble, any M; ~1.9 for half the spread), so classical4's empirical criterion picks
    (2, 1). The fair CRPS of the rescaled (no longer independent) members keeps decreasing past
    s = 1 and picks larger scales. The mean is untouched."""
    g = torch.Generator().manual_seed(1)
    B, M, K, C, H, W = 64, 8, 2, 2, 6, 8
    sd = torch.tensor([0.5, 1.0]).view(1, 1, 1, C, 1, 1)
    obs = torch.randn(B, K, C, H, W, generator=g)
    ens = torch.randn(B, M, K, C, H, W, generator=g) * sd
    grid = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    emp = SpreadCalibrator(C)
    info = emp.fit_validation(ens, obs, candidates=grid)
    assert emp.scales.tolist() == [[2.0, 1.0]] and info["edge"] == 0
    fair = SpreadCalibrator(C)
    fair.fit_validation(ens, obs, candidates=grid, criterion="fair")
    assert (fair.scales >= emp.scales).all() and (fair.scales > emp.scales).any()
    out = emp(ens)
    torch.testing.assert_close(out.mean(1), ens.mean(1))
    per = SpreadCalibrator(C, per_lead=True, leads=K)
    per.fit_validation(ens, obs)
    assert per.scales.shape == (K, C)


def test_structure_operator_identity_equivariance_and_spread_match():
    torch.manual_seed(0)
    B, M, C, H, W = 2, 4, 3, 8, 16
    op = StructureOperator(C=C, K=2, n_modes=(4, 4), hidden=8)
    U, cond = torch.randn(B, M, C, H, W), torch.randn(B, M, 2, H, W)
    torch.testing.assert_close(op(U, cond), U)  # gate 0
    with torch.no_grad():
        op.gate.fill_(0.5)
    out = op(U, cond)
    perm = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(op(U[:, perm], cond[:, perm]), out[:, perm], atol=1e-5, rtol=1e-5)
    V = match_spread(out, U)
    torch.testing.assert_close(V.mean(1), out.mean(1))
    var = lambda T: metrics._wmean((T - T.mean(1, keepdim=True)).pow(2).mean(1))  # noqa: E731
    torch.testing.assert_close(var(V), var(U), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_structure_scores_invariant_to_channel_scale(scale):
    """The correlation diagnostics cannot be changed by a positive channel spread scale."""
    g = torch.Generator().manual_seed(2)
    M, B, K, C, H, W = 6, 3, 2, 65, 6, 8
    e, o = torch.randn(M, B, K, C, H, W, generator=g), torch.randn(B, K, C, H, W, generator=g)
    mu = e.mean(0, keepdim=True)
    s = structure_scores(mu + scale * (e - mu), o)
    s1 = structure_scores(e, o)
    for k in ("lead_xcorr_err", "lead_xcorr_err_phys", "lead_lagcorr_err"):
        assert s[k] == pytest.approx(s1[k], rel=1e-4)


def test_scores_match_scoringrules():
    """Cross-check with scoringrules (search-first: the reference implementation of these scores)."""
    sr = pytest.importorskip("scoringrules")
    import numpy as np
    from pgcfm.coupling import energy_score
    g = np.random.default_rng(0)
    e, o = g.normal(size=(8, 50)), g.normal(size=50)
    skill, pair = crps_parts(torch.tensor(e), torch.tensor(o))
    for crit, est in (("empirical", "nrg"), ("fair", "fair")):
        ref = sr.crps_ensemble(o, e.T, estimator=est)
        np.testing.assert_allclose(crps_from_parts(skill, pair, 8, crit).numpy(), ref, rtol=1e-10)
    E, O = g.normal(size=(8, 3, 4, 5)), g.normal(size=(3, 4, 5))
    ours = float(energy_score(torch.tensor(E)[:, None], torch.tensor(O)[None], torch.ones(4, 1, dtype=torch.float64)))
    ref = sr.es_ensemble(O.reshape(-1), E.reshape(8, -1), estimator="fair") / np.sqrt(O.size)  # ours: RMS norm
    assert ours == pytest.approx(float(ref), rel=1e-10)


def test_sfno_structure_operator():
    pytest.importorskip("torch_harmonics")
    torch.manual_seed(0)
    op = StructureOperator(C=3, K=2, n_modes=(4, 8), hidden=8, net="sfno")
    U, cond = torch.randn(2, 3, 3, 9, 16), torch.randn(2, 3, 2, 9, 16)
    torch.testing.assert_close(op(U, cond), U)
    with torch.no_grad():
        op.gate.fill_(0.5)
    out = op(U, cond)
    perm = torch.tensor([1, 2, 0])
    torch.testing.assert_close(op(U[:, perm], cond[:, perm]), out[:, perm], atol=1e-5, rtol=1e-5)


def test_quick_experiment(tmp_path):
    """Step 7 quick serial chain -> step 9 quick: every calibration on base, a spread-preserving
    operator and an operator on the in-loop calibrated chain."""
    from scripts.run_step7_latent import main as step7
    from scripts.run_step9_structure import CALS, main as step9
    s7 = str(tmp_path / "s7")
    step7(["--out", s7, "--quick", "--arms", "lat_serial"])
    r = step9(["--out", str(tmp_path / "s9"), "--step7", s7, "--test16", str(tmp_path / "t16"), "--quick",
               "--arms", "base", "op_sp", "loop_op_sp"])
    for arm in ("base", "op_sp", "loop_op_sp"):
        for c in CALS:
            s = r[f"{arm}/{c}"]["test16"]
            assert len(s["lead_crps_emp"]) == 2 and len(s["lead_xcorr_err_phys"]) == 2, (arm, c)
    assert r["base/cal_c"]["test16"]["lead_rmse"] == r["base/raw"]["test16"]["lead_rmse"]  # mean kept
