import torch

from pgcfm import contract
from pgcfm.spread import CovarianceOperator, DCTBands, HydrostaticTangent


def test_bands_partition_and_identity():
    b = DCTBands(9, 16, 3, 4, n_bands=3)
    assert set(b.idx.unique().tolist()) == {0, 1, 2, 3}
    d = torch.randn(2, 5, 9, 16)
    assert torch.allclose(b.scale(d, [1.0, 1.0, 1.0]), d, atol=1e-5)
    assert torch.allclose(b.scale(d, [2.0, 2.0, 2.0]) - d, b.scale(d, [2.0, 2.0, 2.0]) - d)  # finite
    assert b.power(d).shape == (4,)


def test_projection_removes_constraint_directions():
    torch.manual_seed(0)
    J = torch.randn(2, 3, 4, 5, contract.N_DYNAMIC)  # (N, H, W, L, C)
    v = torch.randn(2, contract.N_DYNAMIC, 3, 4)
    pv = HydrostaticTangent.project(J, v)
    jpv = J.double() @ pv.permute(0, 2, 3, 1).double()[..., None]
    assert jpv.abs().max() < 1e-4
    assert torch.allclose(HydrostaticTangent.project(J, pv), pv, atol=1e-4)  # a projection
    vm = torch.randn(2, 3, contract.N_DYNAMIC, 3, 4)  # with a member axis
    assert torch.allclose(HydrostaticTangent.project(J, vm)[:, 1], HydrostaticTangent.project(J, vm[:, 1]), atol=1e-5)


def test_covariance_operator_keeps_mean_and_starts_at_identity():
    op = CovarianceOperator(n_modes=(4, 4), hidden=4)
    U = torch.randn(2, 3, contract.N_DYNAMIC, 9, 16)
    cond = torch.randn(2, 3, contract.N_DYNAMIC + 3, 9, 16)
    assert torch.allclose(op(U, cond), U, atol=1e-5)
    with torch.no_grad():
        op.gate.fill_(1.0)
        op.gate_mix.fill_(0.5)
    out = op(U, cond)
    assert torch.allclose(out.mean(1), U.mean(1), atol=1e-4)
    assert not torch.allclose(out, U)


def test_quick_experiment(tmp_path):
    from scripts.run_step7_latent import main as step7
    from scripts.run_step8_spread import main as step8
    s7 = str(tmp_path / "s7")
    step7(["--out", s7, "--quick", "--arms", "lat_serial", "lat_serial_nophys"])
    r = step8(["--out", str(tmp_path / "s8"), "--step7", s7, "--quick"])
    for n in ("serial", "inflate", "posthoc", "band", "noQ", "noQ_hyd", "qmean", "tangent", "tangent_inflate",
              "covop", "covop_nophys", "serial_nophys"):
        assert len(r[n]["test16"]["lead_crps"]) == 2, n
    assert r["contraction"]["kept_total"] > 0 and "nophys_contraction" in r


def test_fit_spread_and_infer_scripts(tmp_path):
    import json

    import netCDF4 as nc
    from scripts.fit_spread import main as fit
    from scripts.infer_pgcfm import main as infer
    from scripts.make_balanced_era5 import main as make_balanced
    from scripts.train_latent import main as train
    H, W, d = 9, 16, str(tmp_path)
    make_balanced(["--out", f"{d}/train", "--years", "2011", "--days", "6", "--n-lat", str(H), "--n-lon", str(W)])
    make_balanced(["--out", f"{d}/valid", "--years", "2012", "--days", "8", "--n-lat", str(H), "--n-lon", str(W)])
    train(["--out", f"{d}/run", "--stages", "stage1", "stage2", "--set", f"train_dir={d}/train", f"valid_dir={d}/valid",
           f"stats_dir={d}/train/stats", "batch=2", "workers=0", "k_lat=3", "k_lon=4", "latent=8", "enc_hidden=[16]",
           "pinn_hidden=[16]", "cfm_width=16", "stage1_iters=2", "serial_iters=2", "dct_fit_batches=2",
           "scale_batches=1", "val_batches=1", "val_members=2"])
    calib = fit(["--ckpt", f"{d}/run/latent.pt", "--valid-dir", f"{d}/valid", "--stats-dir", f"{d}/train/stats",
                 "--out", f"{d}/calib.json", "--n-ics", "4", "--ic-stride", "1", "--leads", "2", "--members", "2",
                 "--steps", "1", "--chunk", "2", "--mode", "band", "--device", "cpu"])
    assert calib["mode"] == "band" and len(calib["band_alpha"]) == 3
    assert set(calib["scores"]) == {"serial", "inflate", "band", "posthoc", "tangent_inflate", "classical",
                                    "classical_lead", "classical_tan", "mbm", "mbm_tan"}
    assert len(calib["candidates"]["classical"]["scales"][0]) == 66
    for mode in ("band", "posthoc", "classical_tan", "mbm_tan"):
        c = dict(json.load(open(f"{d}/calib.json")), mode=mode)
        json.dump(c, open(f"{d}/c_{mode}.json", "w"))
        infer(["--ckpt", f"{d}/run/latent.pt", "--test-file", f"{d}/valid/2012.nc", "--stats-dir", f"{d}/train/stats",
               "--out", f"{d}/fc_{mode}.nc", "--leads", "2", "--members", "2", "--steps", "1", "--ics", "0",
               "--spread", f"{d}/c_{mode}.json", "--device", "cpu"])
        with nc.Dataset(f"{d}/fc_{mode}.nc") as f:
            assert f["forecast"].shape == (1, 3, 2, 67, H, W)
