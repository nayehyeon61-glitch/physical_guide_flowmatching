"""Evaluate KAI / PG-CFM forecast files against ERA5 with the NeuralGCM (WB2) suite.

Forecast files follow KAI ``src/inference.py``: variable ``forecast`` with dims
(initial_condition, lead_time, channel, lat, lon), de-normalized, lead 0 = IC,
``initial_condition`` = index into the test-year file. A PG-CFM ensemble file
adds a ``member`` dim after ``lead_time``.

    # once, on the cluster: day-of-year climatology from the training years
    python scripts/evaluate.py climatology --data-dir /lustre/.../era5 --years 1980-2011 --out clim_1980_2011.nc

    python scripts/evaluate.py score --data-dir /lustre/.../era5 --year 2020 \
        --forecast KAI=runs/kai/forecast.nc --forecast PG-CFM=runs/pgcfm/ensemble.nc \
        --forecast "PG-CFM eps=0"=runs/pgcfm/zero_noise.nc \
        --climatology clim_1980_2011.nc --out eval/2020
"""

import argparse
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import contract  # noqa: E402
from pgcfm.eval import baselines  # noqa: E402
from pgcfm.eval.suite import run_suite  # noqa: E402
from pgcfm.eval.wb2 import from_kai_forecast_nc  # noqa: E402


def _years(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",")]


def load_truth(data_dir, year, ic_index, n_lead):
    with xr.open_dataset(os.path.join(data_dir, f"{year}.nc")) as ds:
        fields = ds["fields"]
        times = ds["time"].values
        if fields.shape[1] != contract.N_CHANNELS:
            raise ValueError(f"expected {contract.N_CHANNELS} channels, got {fields.shape[1]}")
        last = int(ic_index.max()) + n_lead
        if last >= fields.shape[0]:
            raise ValueError(f"IC {ic_index.max()} + {n_lead} leads runs past the end of {year} (no cross-year pairs)")
        truth = np.stack([fields[i: i + n_lead + 1].values for i in ic_index]).astype(np.float32)
    return truth, times[ic_index].astype("datetime64[D]")


def cmd_score(a):
    forecasts, ic_ref = {}, None
    for spec in a.forecast:
        name, path = spec.split("=", 1)
        f = from_kai_forecast_nc(path)
        if ic_ref is None:
            ic_ref = f["ic_index"]
        elif not np.array_equal(ic_ref, f["ic_index"]):
            raise ValueError(f"{name}: initial conditions differ from the first forecast")
        ch = np.asarray(f["channel"])
        if sorted(ch.tolist()) != list(range(contract.N_CHANNELS)):
            raise ValueError(f"{name}: needs each of the 67 contract channels once, got {ch.tolist()}")
        # KAI allows any out_channels order; put channels back in contract order.
        x = np.take(f["forecast"][: a.max_ic], np.argsort(ch), axis=-3)
        forecasts[name] = x[:, 1:]  # drop lead 0 (= IC)
    ic_index = ic_ref[: a.max_ic]
    n_lead = next(iter(forecasts.values())).shape[1]
    truth, init_times = load_truth(a.data_dir, a.year, ic_index, n_lead)
    clim = xr.open_dataset(a.climatology).load() if a.climatology else None
    res = run_suite(forecasts, truth, init_times, clim, a.out, map_model=a.map_model)
    print(res["physics"].round(4).to_string())
    print(f"wrote {a.out}/scores.csv, physics.csv, figures/")


def cmd_climatology(a):
    out = baselines.climatology_from_kai_dir(a.data_dir, _years(a.years), a.out, a.window)
    print(f"wrote {out}")


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--data-dir", required=True)
    s.add_argument("--year", type=int, default=2020)
    s.add_argument("--forecast", action="append", required=True, help="NAME=path/to/forecast.nc (repeatable)")
    s.add_argument("--climatology", default=None)
    s.add_argument("--out", required=True)
    s.add_argument("--max-ic", type=int, default=None)
    s.add_argument("--map-model", default=None)
    s.set_defaults(fn=cmd_score)
    c = sub.add_parser("climatology")
    c.add_argument("--data-dir", required=True)
    c.add_argument("--years", required=True, help="e.g. 1980-2011")
    c.add_argument("--out", required=True)
    c.add_argument("--window", type=int, default=61)
    c.set_defaults(fn=cmd_climatology)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
