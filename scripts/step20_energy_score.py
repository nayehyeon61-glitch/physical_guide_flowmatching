"""Step 20 (3): multivariate scores of saved ensembles, where a physics projection can show up.

The ClimODE CRPS is per variable (marginal), so a projection that only changes the joint structure of
(z500, t850, t2m, u10, v10) at a grid point (e.g. the hydrostatic z-T relation) cannot move it. Here:
  energy score  ES = E|X - y| - 1/2 E|X - X'| over the 5-variable vector at each grid point (fair pair term),
                variables standardized by the truth's climatological std, area-weighted mean per lead
  z-t energy score  the same on the (z500, t850) pair only (the hydrostatic pair)
  marginal CRPS sum  sum over the 5 variables of the fair CRPS in the same units (= ES if variables were
                independent; ES / sum tells how much the joint structure counts)
Arms are prefixes of ``<prefix>_ens_<year>.npz`` (E_minus_x0, O, X0, ics, leads).

    python scripts/step20_energy_score.py --era5 /home/claude/era5/era5_data --out docs/step20/energy_score.json \
        --arms runs/step13/seqfm_dct_raw runs/step13/seqfm_dct_pinn ...
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from scripts.run_step11_climode import VARS, climode_truth  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402


def energy_score(E, O, w):
    """E (b, M, L, C, H, W), O (b, L, C, H, W) standardized -> (b, L) area-weighted fair energy score
    over the C-vector at each grid point."""
    M = E.shape[1]
    skill = torch.linalg.vector_norm(E - O[:, None], dim=3).mean(1)  # (b, L, H, W)
    pair = 0.0
    for i in range(M):
        for j in range(i + 1, M):
            pair = pair + torch.linalg.vector_norm(E[:, i] - E[:, j], dim=2)
    pair = pair / (M * (M - 1) / 2)
    return ((skill - 0.5 * pair) * w).mean((-1, -2))


def fair_crps(E, O, w):
    """per variable fair CRPS (b, L, C)."""
    M = E.shape[1]
    skill = (E - O[:, None]).abs().mean(1)
    pair = 0.0
    for i in range(M):
        for j in range(i + 1, M):
            pair = pair + (E[:, i] - E[:, j]).abs()
    pair = pair / (M * (M - 1) / 2)
    return ((skill - 0.5 * pair) * w).mean((-1, -2))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--era5", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--test-years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--chunk", type=int, default=12)
    p.add_argument("--max-ics", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    lat, lon, _, _ = load_static(a.era5)
    w = torch.from_numpy((np.cos(np.deg2rad(lat)) / np.cos(np.deg2rad(lat)).mean()).astype(np.float32))[:, None]
    std = np.stack([climode_truth(a.era5, y) for y in (2015, 2016)]).reshape(-1, 5, len(lat), len(lon)).std((0, 2, 3))
    std = torch.from_numpy(std.astype(np.float32))[:, None, None]
    iz, it = VARS.index("z"), VARS.index("t")
    res = {"std": std.view(-1).tolist(), "arms": {}}
    for arm in a.arms:
        name = os.path.basename(arm)
        acc = {"es": 0.0, "es_zt": 0.0, "crps": 0.0, "n": 0}
        leads = None
        for y in a.test_years:
            z = np.load(f"{arm}_ens_{y}.npz")
            E16, O, X0, leads = z["E_minus_x0"], z["O"], z["X0"], z["leads"]
            n = len(E16) if not a.max_ics else min(len(E16), a.max_ics)
            for c in range(0, n, a.chunk):
                sl = slice(c, min(n, c + a.chunk))
                E = torch.from_numpy(E16[sl].astype(np.float32) + X0[sl][:, None, None]) / std
                Ot = torch.from_numpy(O[sl]) / std
                acc["es"] = acc["es"] + energy_score(E, Ot, w).sum(0)
                acc["es_zt"] = acc["es_zt"] + energy_score(E[:, :, :, [iz, it]], Ot[:, :, [iz, it]], w).sum(0)
                acc["crps"] = acc["crps"] + fair_crps(E, Ot, w).sum(0)
                acc["n"] += E.shape[0]
        r = {"leads": np.asarray(leads).tolist(), "n_ic": acc["n"],
             "energy_score": (acc["es"] / acc["n"]).tolist(), "energy_score_zt": (acc["es_zt"] / acc["n"]).tolist(),
             "crps_sum": (acc["crps"] / acc["n"]).sum(-1).tolist(), "crps": (acc["crps"] / acc["n"]).tolist()}
        r["es_over_crps_sum"] = (np.asarray(r["energy_score"]) / np.asarray(r["crps_sum"])).tolist()
        res["arms"][name] = r
        print(f"[{name}] n {acc['n']} | ES by lead " + " ".join(f"{v:.4f}" for v in r["energy_score"])
              + " | z-t ES " + " ".join(f"{v:.4f}" for v in r["energy_score_zt"])
              + " | ES / sum CRPS " + " ".join(f"{v:.3f}" for v in r["es_over_crps_sum"]), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
