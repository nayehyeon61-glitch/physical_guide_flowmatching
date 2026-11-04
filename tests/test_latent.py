import torch

from pgcfm import contract
from pgcfm.latent import DCTTruncation, LatentConfig, LatentPGCFM, dct, idct

H, W = 9, 16


def _batch(B=3):
    return {"x0": torch.randn(B, contract.N_DYNAMIC, H, W), "targets": torch.randn(B, 1, contract.N_DYNAMIC, H, W),
            "static": torch.randn(B, 1, H, W), "doy": torch.tensor([[10, 11]] * B)}


def test_dct_roundtrip_and_truncation():
    x = torch.randn(2, 3, H, W)
    assert torch.allclose(idct(dct(x)), x, atol=1e-5)
    full = DCTTruncation(3, H, W, H, W)
    assert torch.allclose(full.filter(x), x, atol=1e-5)
    t = DCTTruncation(3, H, W, 3, 4)
    t.fit(x)
    once = t.filter(x)
    assert torch.allclose(t.filter(once), once, atol=1e-5)  # a projection
    smooth = torch.cos(torch.pi * (torch.arange(W) + 0.5) / W).expand(2, 3, H, W)  # DCT-II mode 1
    assert torch.allclose(t.filter(smooth), smooth, atol=1e-4)  # lowest modes pass


def test_latent_model_shapes_and_eps0():
    m = LatentPGCFM(LatentConfig(k_lat=3, k_lon=4, latent=8, enc_hidden=(32, 16), dec_hidden=(16, 32),
                                 pinn_hidden=(16,), cfm_width=16), H, W)
    b = _batch()
    m.fit_dct(b["x0"], b["static"])
    terms, logs = m.stage1_terms(b)
    assert set(terms) == {"data"} and torch.isfinite(terms["data"])
    loss, _ = m.cfm_loss(b, train_ae=True)
    loss.backward()
    assert m.encoder.layers[0].weight.grad is not None
    x = m.sample(b["x0"], b["static"], b["doy"][:, 1], steps=2)
    assert x.shape == b["x0"].shape
    zeros = torch.zeros(3, len(m.gen_idx), H, W)  # deterministic_rollout passes grid-shaped eps = 0
    a1 = m.sample(b["x0"], b["static"], b["doy"][:, 1], steps=2, eps=zeros)
    a2 = m.sample(b["x0"], b["static"], b["doy"][:, 1], steps=2, eps=zeros)
    assert torch.equal(a1, a2)
    # a decoded member lives in the kept DCT modes
    assert torch.allclose(m.dct_x.filter(x), x, atol=1e-4)


def test_quick_experiment(tmp_path):
    from scripts.run_step7_latent import main
    r = main(["--out", str(tmp_path), "--quick"])
    for n in ("cfm", "cfm_dct", "pinn_det", "lat_frozen", "lat_retrain", "lat_finetune", "lat_nophys", "lat_serial",
              "lat_serial_nophys", "lat_serial_op", "lat_serial_op_nophys", "lat_serial_op_band", "lat_serial_op2",
              "lat_serial_op2_nophys"):
        assert n in r and len(r[n]["lead_crps"]) == 2
    assert all(v == v for v in r["floors"].values())


def test_train_latent_and_infer_scripts(tmp_path, capsys):
    import netCDF4 as nc
    from scripts.infer_pgcfm import main as infer
    from scripts.make_balanced_era5 import main as make_balanced
    from scripts.train_latent import main as train
    d = str(tmp_path)
    make_balanced(["--out", f"{d}/train", "--years", "2011", "--days", "6", "--n-lat", str(H), "--n-lon", str(W)])
    make_balanced(["--out", f"{d}/valid", "--years", "2012", "--days", "6", "--n-lat", str(H), "--n-lon", str(W)])
    train(["--out", f"{d}/run", "--stages", "stage1", "stage2", "decft", "--set", f"train_dir={d}/train",
           f"valid_dir={d}/valid", f"stats_dir={d}/train/stats", "batch=2", "workers=0", "k_lat=3", "k_lon=4",
           "latent=8", "enc_hidden=[16]", "pinn_hidden=[16]", "cfm_width=16", "stage1_iters=2", "stage2_iters=2",
           "decft_iters=1", "dct_fit_batches=2", "scale_batches=1", "val_batches=1", "val_members=2"])
    capsys.readouterr()
    before = torch.load(f"{d}/run/latent.pt")["latent"]
    train(["--out", f"{d}/run", "--resume", "--stages", "stage2", "decft", "--set", f"train_dir={d}/train",
           f"valid_dir={d}/valid", f"stats_dir={d}/train/stats", "batch=2", "workers=0", "k_lat=3", "k_lon=4",
           "latent=8", "enc_hidden=[16]", "pinn_hidden=[16]", "cfm_width=16", "val_batches=1"])
    assert "resumed stage1, stage2, decft" in capsys.readouterr().out  # finished stages are not retrained
    after = torch.load(f"{d}/run/latent.pt")["latent"]
    assert all(torch.equal(before[k], after[k]) for k in before)
    infer(["--ckpt", f"{d}/run/latent.pt", "--test-file", f"{d}/valid/2012.nc", "--stats-dir", f"{d}/train/stats",
           "--out", f"{d}/fc.nc", "--leads", "2", "--members", "2", "--steps", "1", "--ics", "0", "--det"])
    with nc.Dataset(f"{d}/fc.nc") as f:
        assert f["forecast"].shape == (1, 3, 2, 67, H, W)
    with nc.Dataset(f"{d}/fc_det.nc") as f:
        assert f["forecast"].shape == (1, 3, 67, H, W)


def test_carry_highpass_keeps_unresolved_scales():
    m = LatentPGCFM(LatentConfig(k_lat=3, k_lon=4, latent=8, enc_hidden=(32,), dec_hidden=(32,), pinn_hidden=(16,),
                                 cfm_width=16, carry_highpass=True), H, W)
    b = _batch()
    m.fit_dct(b["x0"], b["static"])
    x = b["x0"]
    y = m.sample(x, b["static"], b["doy"][:, 1], steps=2)
    hp = lambda t: t - m.dct_x.filter(t)  # noqa: E731
    assert torch.allclose(hp(y), hp(x), atol=1e-4)  # high modes persist, low modes generated


def test_carry_tendency_cancels_the_autoencoder_error():
    from pgcfm.latent import floors, serial_terms
    m = LatentPGCFM(LatentConfig(k_lat=3, k_lon=4, latent=8, enc_hidden=(32,), dec_hidden=(32,), pinn_hidden=(16,),
                                 cfm_width=16, carry_tendency=True, coupling="serial"), H, W)
    b = _batch()
    m.fit_dct(b["x0"], b["static"])
    x, st, d = b["x0"], b["static"], b["doy"]
    z0 = m.encode(x, st, d[:, 0])
    assert torch.allclose(m.decode(z0, x, z0), x, atol=1e-5)  # a latent step that does nothing = persistence
    assert m.sample(x, st, d[:, 1], steps=2).shape == x.shape
    terms, _ = m.stage1_terms(b)
    assert torch.isfinite(terms["data"])
    loss, logs = m.cfm_loss(b, train_ae=True)  # stage 2 retrain still trains the decoder
    loss.backward()
    assert logs["recon"] > 0 and m.decoder.layers[0].weight.grad.abs().sum() > 0
    m.zero_grad()
    terms, _ = serial_terms(m, b, members=2, steps=2)
    assert torch.isfinite(terms["crps"])
    f = floors(m, [b])
    assert f["rmse_ae"] < 1e-6 and f["rmse_ae_tendency"] > 0


def test_checkpoint_resume_roundtrip(tmp_path):
    import numpy as np
    from pgcfm.ens_operator import EnsembleOperator
    from pgcfm.latent import _checkpoint, _resume

    def make():
        op = EnsembleOperator(n_modes=(4, 4), hidden=8)
        opt = torch.optim.Adam(op.parameters())
        return op, opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10)
    op, opt, sched = make()
    with torch.no_grad():
        op.gate.fill_(0.3)
    ck = str(tmp_path / "op.pt")
    _checkpoint(ck, {"op": op}, opt, sched, [{"es": np.float64(1.0)}], 4)  # FNO "_metadata", numpy logs
    op2, opt2, sched2 = make()
    hist, it = _resume(ck, {"op": op2}, opt2, sched2, "cpu")
    assert it == 4 and len(hist) == 1 and op2.gate.item() == op.gate.item()
