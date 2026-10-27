"""Per-window metrics of saved step 17 grid arms (for runs whose json predates the ``windows`` field):
regenerates the test windows with the run's seeds and re-evaluates ``<out>/<setting>_<arm>.pt``.

    python scripts/step17_windows.py --out runs/step17 --setting coarse
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pgcfm.swe import Normalizer, TendencyNet, World, rollout_heun, rollout_node  # noqa: E402
from scripts.run_step17_swe import GRID_ARMS, evaluate  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/step17")
    p.add_argument("--setting", default="coarse")
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    path = os.path.join(a.out, f"{a.setting}.json")
    res = json.load(open(path))
    ra = argparse.Namespace(**res["args"])
    rng = np.random.default_rng(ra.seed)
    learn = World(64, 128, 22, 300.0)
    truth = learn if ra.setting == "full" else World(128, 256, 43, 150.0)
    to_learn = lambda us: learn.grid(truth.truncate(us, learn.lmax)).float()  # noqa: E731
    F_of = lambda us: learn.grid(truth.truncate(truth.tendency_spec(us, closure=True), learn.lmax)).float()  # noqa: E731

    def trajectory(us0, hours, spinup=0):
        if spinup:
            us0 = truth.step_hours(us0, spinup)[-1]
        specs = [us0] + truth.step_hours(us0, hours)
        return torch.stack([to_learn(s) for s in specs]), torch.stack([F_of(s) for s in specs])

    for i in range(ra.n_train):  # consume the same parameter draws (the training trajectories are not needed)
        rng.uniform(60, 90), rng.uniform(0, 2 * math.pi), rng.uniform(0.5, 1.5)
    tests = []
    gal_hours = max(ra.test_t0) + ra.test_hours
    gx, gF = trajectory(truth.galewsky(), gal_hours)
    for t in ra.test_t0:
        tests.append(("galewsky", t, gx[t:t + ra.test_hours + 1], gF[t:t + ra.test_hours + 1]))
    for i in range(2):
        torch.manual_seed(ra.seed * 100 + 50 + i)
        rx, rF = trajectory(truth.random_ic(mach=0.1), ra.test_hours, max(24, ra.spinup // 4))
        tests.append(("random", 0, rx, rF))
    d = res["data"]
    nz = Normalizer.__new__(Normalizer)
    nz.mean, nz.std = torch.tensor(d["mean"]).view(1, 3, 1, 1), torch.tensor(d["std"]).view(1, 3, 1, 1)
    nz.t1 = torch.tensor(d["t1"])
    b = torch.tensor(d["b"])

    def rollout(arm, net, z0, hours):
        if arm == "node":
            return rollout_node(net, z0, hours, method=ra.node_method, step=ra.node_step)
        return rollout_heun(net, z0, hours, substeps=ra.substeps)

    for arm in [q for q in GRID_ARMS if q in res["arms"]]:
        net = TendencyNet(learn, nz.t1, ra.hidden, tuple(ra.blocks))
        net.load_state_dict(torch.load(os.path.join(a.out, f"{a.setting}_{arm}.pt"), map_location="cpu"))
        net.eval()
        r = evaluate(arm, net, tests, learn, nz, b, rollout, ra.test_hours)
        same = abs(r["rmse"]["Phi"][-1] - res["arms"][arm]["rmse"]["Phi"][-1]) / res["arms"][arm]["rmse"]["Phi"][-1]
        res["arms"][arm]["windows"] = r["windows"]
        print(f"[{arm}] 24 h Phi RMSE recomputed {r['rmse']['Phi'][-1]:.1f} (saved {res['arms'][arm]['rmse']['Phi'][-1]:.1f}, "
              f"rel diff {same:.2g}); per window: " + " ".join(f"{w['kind']}@{w['t0']} {w['rmse_Phi'][-1]:.0f}/{w['pers_Phi'][-1]:.0f}"
                                                                 for w in r["windows"]), flush=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
