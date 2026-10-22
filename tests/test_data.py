import os

import numpy as np
import pytest
import torch

from pgcfm import contract
from pgcfm.data import PGCFMSequenceDataset, assemble_full_state, kai_data_config, kai_utils
from scripts.make_synthetic_era5 import main as make_synthetic

DAYS = 6


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("era5"))
    # Reduced grid for speed; the full-grid case is covered separately.
    make_synthetic(["--out", out, "--years", "2011", "2012", "--days", str(DAYS),
                    "--n-lat", "25", "--n-lon", "48"])
    stats = os.path.join(out, "stats")
    cfg = kai_data_config(os.path.join(stats, "global_mean.npy"), os.path.join(stats, "global_std.npy"))
    return out, cfg


def test_horizon1_matches_kai_loader(synth):
    out, cfg = synth
    ours = PGCFMSequenceDataset(cfg, out, horizon=1)
    kai = kai_utils.ClimateNetCDFDataset(cfg, out)
    # KAI drops the last day of each yearly file; so do we.
    assert len(ours) == len(kai) == 2 * (DAYS - 1)
    for i in [0, DAYS - 2, DAYS - 1, len(kai) - 1]:
        inp, tar = kai[i]
        s = ours[i]
        assert torch.equal(s["x0"], inp[contract.DYNAMIC_CHANNELS])
        assert torch.equal(s["targets"][0], tar[contract.DYNAMIC_CHANNELS])
        assert torch.equal(s["static"], inp[contract.STATIC_CHANNELS])


def test_sequences_never_cross_year_boundary(synth):
    out, cfg = synth
    K = 4
    ds = PGCFMSequenceDataset(cfg, out, horizon=K)
    assert len(ds) == 2 * (DAYS - K)
    for i in range(len(ds)):
        doy = ds[i]["doy"].numpy()
        assert np.array_equal(np.diff(doy), np.ones(K))
    # First sample of the second year starts on its 1 January.
    assert ds[DAYS - K]["doy"][0] == 0


def test_shapes_and_static_copy(synth):
    out, cfg = synth
    ds = PGCFMSequenceDataset(cfg, out, horizon=3)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2)))
    assert batch["x0"].shape == (2, 66, 25, 48)
    assert batch["targets"].shape == (2, 3, 66, 25, 48)
    assert batch["static"].shape == (2, 1, 25, 48)
    full = assemble_full_state(batch["targets"], batch["static"][:, None])
    assert full.shape == (2, 3, 67, 25, 48)
    assert torch.equal(full[:, :, contract.DYNAMIC_CHANNELS], batch["targets"])
    for k in range(3):
        assert torch.equal(full[:, k, 64], batch["static"][:, 0])


def test_full_kai_grid(tmp_path):
    make_synthetic(["--out", str(tmp_path), "--years", "2020", "--days", "2"])
    stats = tmp_path / "stats"
    cfg = kai_data_config(str(stats / "global_mean.npy"), str(stats / "global_std.npy"))
    ds = PGCFMSequenceDataset(cfg, str(tmp_path), horizon=1)
    s = ds[0]
    assert s["x0"].shape == (66, contract.N_LAT, contract.N_LON)
    assert s["x0"].dtype == torch.float32
    assert torch.isfinite(s["x0"]).all()


def test_doy_decoded_from_time_units(tmp_path):
    """Absolute or sub-daily time origins must still give 0-based day-of-year."""
    import netCDF4 as nc

    make_synthetic(["--out", str(tmp_path), "--years", "2012", "--days", "3",
                    "--n-lat", "25", "--n-lon", "48"])
    path = tmp_path / "2012.nc"
    with nc.Dataset(path, "a") as ds:
        base = nc.date2num(nc.num2date(0, "days since 2012-03-01"), "hours since 1900-01-01")
        ds["time"].units = "hours since 1900-01-01"
        ds["time"][:] = base + 24 * np.arange(3)
    stats = tmp_path / "stats"
    cfg = kai_data_config(str(stats / "global_mean.npy"), str(stats / "global_std.npy"))
    ds = PGCFMSequenceDataset(cfg, str(tmp_path), horizon=2)
    # 2012 is a leap year: 1 March is day 60 (0-based).
    assert ds[0]["doy"].tolist() == [60, 61, 62]
