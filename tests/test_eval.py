import os

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pgcfm import contract
from pgcfm.eval import physics, scores, synthetic
from pgcfm.eval.wb2 import LEAD, REALIZATION, select_headline, to_dataset
from scripts.evaluate import main as evaluate
from scripts.make_synthetic_era5 import main as make_synthetic

INIT = np.array(["2020-01-10", "2020-02-10", "2020-03-10", "2020-04-10"], dtype="datetime64[D]")


def _toy(rng, shape):
    """(N, L, [M], 67, 13, 24) random fields around the channel means."""
    return (contract.REFERENCE_MEAN[:, None, None] + contract.REFERENCE_STD[:, None, None] * rng.standard_normal(shape)).astype(np.float32)


def test_deterministic_crps_is_mae_and_calibrated_ssr_is_one():
    rng = np.random.default_rng(0)
    mu = _toy(rng, (4, 2, 67, 13, 24))
    sd = contract.REFERENCE_STD[:, None, None]
    truth = mu + sd * rng.standard_normal(mu.shape)
    ens = mu[:, :, None] + sd * rng.standard_normal((4, 2, 40, 67, 13, 24))
    t = select_headline(to_dataset(truth, INIT, [1, 2], False))
    det = scores.score_forecast(select_headline(to_dataset(mu, INIT, [1, 2], False)), t, region_names=["global"])
    mae = np.abs(mu - truth)
    w = np.cos(np.deg2rad(np.linspace(-90, 90, 13)))
    assert float(det["z500"].sel(metric="crps", region="global").isel({LEAD: 0})) == pytest.approx(
        float((mae[:, 0, contract.CHANNEL_INDEX["z500"]] * w[:, None]).sum((-2, -1)).mean() / (w.sum() * 24)), rel=0.05)
    e = scores.score_forecast(select_headline(to_dataset(ens.astype(np.float32), INIT, [1, 2], True)), t, region_names=["global"])
    ssr = e.sel(metric="ssr", region="global").to_array()
    assert np.allclose(ssr, 1.0, atol=0.1)
    # The ensemble beats its own mean on CRPS but not on member RMSE.
    assert (e.sel(metric="crps").to_array() < det.sel(metric="crps").to_array()).all()


@pytest.fixture(scope="module")
def world():
    return synthetic.make_world(n_ic=2, n_lead=2, seed=3)


def _full(x, member):
    return to_dataset(x, INIT[:2], [1, 2], member)


def test_tendency_ratio_truth_one_persistence_zero(world):
    truth = _full(world.truth[:, 1:], False)
    ic = to_dataset(world.truth[:, :1], INIT[:2], [0], False).isel({LEAD: 0}, drop=True).drop_vars("valid_time")
    v = ["temperature", "u_component_of_wind"]
    r1 = physics.tendency_amplitude_ratio(truth[v], truth[v], ic[v]).to_array()
    assert np.allclose(r1, 1.0)
    pers = np.repeat(world.truth[:, :1], 2, axis=1)
    r0 = physics.tendency_amplitude_ratio(_full(pers, False)[v], truth[v], ic[v]).to_array()
    assert np.allclose(r0.isel({LEAD: 0}), 0.0)


def test_balanced_state_passes_hydrostatic_and_geostrophic(world):
    truth = _full(world.truth[:, 1:], False)
    h = physics.hydrostatic_residual(truth)
    assert float(h.max()) < 1.0  # m2 s-2; exact up to float32
    g = physics.geostrophic_balance(truth)
    assert float(g.ratio.sel(level=500).max()) < 0.2
    broken = world.truth[:, 1:].copy()
    broken[:, :, contract.level_slice("z")] += 100.0 * np.random.default_rng(0).standard_normal(broken[:, :, contract.level_slice("z")].shape)
    assert float(physics.hydrostatic_residual(_full(broken, False)).min()) > 50.0


def test_spectral_ratio_detects_blur(world):
    truth = select_headline(_full(world.truth[:, 1:], False))["z500"]
    blur = synthetic.blurry(world, M=2)
    b = select_headline(_full(blur, True))["z500"]
    assert float(physics.spectral_ratio(truth, truth).mean()) == pytest.approx(1.0)
    assert float(physics.spectral_ratio(b, truth).mean()) < 0.5


def _write_forecast(path, x, ic_index, member, channel=None):
    with nc.Dataset(path, "w") as d:
        d.createDimension("initial_condition", x.shape[0])
        d.createDimension("lead_time", x.shape[1])
        dims = ["initial_condition", "lead_time"]
        if member:
            d.createDimension("member", x.shape[2])
            dims.append("member")
        for n, s in zip(("channel", "lat", "lon"), x.shape[-3:]):
            d.createDimension(n, s)
        d.createVariable("initial_condition", "i4", ("initial_condition",))[:] = ic_index
        d.createVariable("lead_time", "i4", ("lead_time",))[:] = np.arange(x.shape[1])
        d.createVariable("channel", "i4", ("channel",))[:] = np.arange(x.shape[-3]) if channel is None else channel
        d.createVariable("forecast", "f4", tuple(dims) + ("channel", "lat", "lon"))[:] = x


def test_cli_on_kai_format_files(tmp_path):
    data = str(tmp_path / "era5")
    make_synthetic(["--out", data, "--years", "2018", "2019", "2020", "--days", "8"])
    clim = str(tmp_path / "clim.nc")
    evaluate(["climatology", "--data-dir", data, "--years", "2018-2019", "--out", clim, "--window", "3"])
    with xr.open_dataset(os.path.join(data, "2020.nc")) as ds:
        f = ds["fields"].values
    ic = np.array([0, 2])
    pers = np.stack([np.repeat(f[i: i + 1], 4, axis=0) for i in ic])  # (2, 4, 67, H, W), lead 0 = IC
    rng = np.random.default_rng(0)
    ens = pers[:, :, None] + 0.1 * contract.REFERENCE_STD[:, None, None] * rng.standard_normal((2, 4, 3) + f.shape[1:])
    _write_forecast(tmp_path / "det.nc", pers, ic, False)
    _write_forecast(tmp_path / "ens.nc", ens.astype(np.float32), ic, True)
    perm = np.random.default_rng(1).permutation(contract.N_CHANNELS)  # KAI out_channels in another order
    _write_forecast(tmp_path / "shuffled.nc", pers[:, :, perm], ic, False, channel=perm)
    out = str(tmp_path / "eval")
    evaluate(["score", "--data-dir", data, "--year", "2020", "--forecast", f"det={tmp_path / 'det.nc'}",
              "--forecast", f"ens={tmp_path / 'ens.nc'}", "--forecast", f"shuffled={tmp_path / 'shuffled.nc'}", "--climatology", clim, "--out", out])
    df = pd.read_csv(os.path.join(out, "scores.csv"))
    # A forecast that equals persistence must score exactly like the persistence baseline.
    a = df[(df.model == "det") & (df.metric == "rmse")].value.values
    b = df[(df.model == "persistence") & (df.metric == "rmse")].value.values
    assert np.allclose(a, b)
    c = df[(df.model == "shuffled") & (df.metric == "rmse")].value.values
    assert np.allclose(a, c)
    assert {"acc", "crps", "ssr", "rmsb"} <= set(df.metric)
    assert len(os.listdir(os.path.join(out, "figures"))) >= 6


def test_streamed_climatology_matches_wb2_on_all_years(tmp_path):
    from pgcfm.eval import baselines

    data = str(tmp_path / "era5")
    make_synthetic(["--out", data, "--years", "2018", "2019", "--days", "5"])
    out = str(tmp_path / "clim.nc")
    baselines.climatology_from_kai_dir(data, [2018, 2019], out, window_size=3)
    paths = [os.path.join(data, f"{y}.nc") for y in (2018, 2019)]
    ref = baselines.daily_climatology(baselines.kai_years_to_dataset(paths), window_size=3)
    with xr.open_dataset(out) as got:
        xr.testing.assert_allclose(got.load(), ref, rtol=1e-5)


def test_hydrostatic_mask_drops_below_ground_points(world):
    truth = _full(world.truth[:, 1:], False)
    bad = world.truth[:, 1:].copy()
    bad[:, :, contract.CHANNEL_INDEX["z1000"]] += 1e4  # only the 925-1000 layer is wrong
    ds = _full(bad, False)
    everywhere_low_sp = truth["surface_pressure"] * 0 + 95000.0  # 1000 hPa below ground everywhere
    r = physics.hydrostatic_residual(ds, everywhere_low_sp)
    assert bool(np.isnan(r.sel(layer=925)).all())  # no valid point: not counted as balanced
    assert float(r.sel(layer=850).max()) < 1.0
