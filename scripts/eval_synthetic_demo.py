"""Small test of the evaluation suite on the balanced toy atmosphere.

Runs the six failure-mode models of ``pgcfm.eval.synthetic`` through the full
NeuralGCM/WB2 suite and writes tables + figures, so we can see which metric
catches which weakness before running on real PG-CFM output.

    python scripts/eval_synthetic_demo.py --out docs/eval/synthetic --n-ic 3 --leads 6 --members 8
"""

import argparse
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm.eval import synthetic  # noqa: E402
from pgcfm.eval.suite import run_suite  # noqa: E402
from pgcfm.eval.wb2 import to_dataset  # noqa: E402


def toy_climatology(world, n_lead):
    doys = sorted({int(d) + k for d in world.init_doy for k in range(n_lead + 1)})
    states = np.stack([synthetic.climatology_state(d) for d in doys])  # (D, 67, H, W)
    ds = to_dataset(states[:, None], np.array(["2000-01-01"] * len(doys), dtype="datetime64[D]"), [0], member_axis=False)
    ds = ds.isel(init_time=slice(None), prediction_timedelta=0, drop=True).drop_vars("valid_time", errors="ignore")
    return ds.rename(init_time="dayofyear").assign_coords(dayofyear=doys)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/eval/synthetic")
    p.add_argument("--n-ic", type=int, default=3)
    p.add_argument("--leads", type=int, default=6)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    world = synthetic.make_world(a.n_ic, a.leads, seed=a.seed)
    clim = toy_climatology(world, a.leads)
    forecasts = {}
    for name, fn in synthetic.MODELS.items():
        kw = {} if name == "zero-noise" else {"M": a.members}
        x = fn(world, **kw)
        forecasts[name] = x[:, :, 0] if name == "zero-noise" else x
        print(f"generated {name}: {x.shape}", flush=True)
    res = run_suite(forecasts, world.truth, world.init_times, clim, a.out, map_model="unbalanced")
    g = res["scores"]
    g = g[(g.region == "global") & (g.variable.isin(["z500", "t850", "u850"])) & (g.lead_days.isin([1, a.leads]))]
    print(g.pivot_table(index=["model"], columns=["metric", "variable", "lead_days"], values="value").round(3).to_string())
    print(res["physics"].round(3).to_string())


if __name__ == "__main__":
    main()
