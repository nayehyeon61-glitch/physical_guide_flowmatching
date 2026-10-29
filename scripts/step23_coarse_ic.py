"""Step 23 equal-information ablation: NeuralGCM run from our 5.625 deg / 12-level IC vs from the full ERA5 IC.

Both inputs are run_neuralgcm_eval.py outputs (neuralgcm_pred.npz: mean, truth, ics, lead_hours). On the ICs they
share, prints and writes the latitude-weighted z500 / t850 RMSE of each and the paired difference (coarse - full)
with an IC bootstrap 95 % interval. Rule H10 (docs/step23_neuralgcm.md 4.4): coarse-IC z500 24 h RMSE below 100 m2/s2
means the 24 h gap to our model is mostly the model, not the IC information.

    python scripts/step23_coarse_ic.py --era5 /home/claude/era5/era5_data \
        --full runs/step23/neuralgcm_2_8/neuralgcm_pred.npz --coarse runs/step23/neuralgcm_2_8_coarse/neuralgcm_pred.npz \
        --out runs/step23/coarse_ic.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.report_step11 import boot_ci  # noqa: E402
from scripts.run_step11_climode import VARS  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--era5", required=True)
    p.add_argument("--full", required=True)
    p.add_argument("--coarse", required=True)
    p.add_argument("--out", default="runs/step23/coarse_ic.json")
    a = p.parse_args(argv)
    lat = load_static(a.era5)[0]
    w = np.cos(np.deg2rad(lat))
    w = (w / w.mean())[:, None]
    F, C = dict(np.load(a.full)), dict(np.load(a.coarse))
    leads = [int(h) for h in F["lead_hours"]]
    common = [int(i) for i in C["ics"] if i in set(F["ics"].tolist())]
    fi = [int(np.flatnonzero(F["ics"] == i)[0]) for i in common]
    ci = [int(np.flatnonzero(C["ics"] == i)[0]) for i in common]

    def rmse(d, rows):  # (n, L, 5): truth[:, :, 1:] are the verifying fields, [0] is the IC
        m, t = d["mean"][0, rows], d["truth"][0, rows, 1:]
        return np.sqrt((((m - t) ** 2) * w).mean((-1, -2)))

    rf, rc = rmse(F, fi), rmse(C, ci)
    assert np.allclose(F["truth"][0, fi], C["truth"][0, ci], equal_nan=True), "truth differs between runs"
    out = {"ics": common, "leads": leads, "full": {}, "coarse": {}, "coarse-full": {}}
    for v in ("z", "t"):
        k = VARS.index(v)
        out["full"][v] = rf[:, :, k].mean(0).tolist()
        out["coarse"][v] = rc[:, :, k].mean(0).tolist()
        d = rc[:, :, k] - rf[:, :, k]
        out["coarse-full"][v] = {"mean": d.mean(0).tolist(), "ci95": [boot_ci(d[:, j]) for j in range(len(leads))]}
        print(f"{v}: full " + " ".join(f"{x:.2f}" for x in out["full"][v]) + " | coarse "
              + " ".join(f"{x:.2f}" for x in out["coarse"][v]) + " | coarse - full "
              + " ".join(f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]" for m, (lo, hi) in zip(d.mean(0), out["coarse-full"][v]["ci95"])))
    z24 = out["coarse"]["z"][leads.index(24)]
    out["h10_rule"] = {"coarse_z500_24h": z24, "threshold": 100.0, "gap_mostly_model": bool(z24 < 100.0)}
    print(f"[H10] coarse-IC z500 24 h RMSE {z24:.1f}: " + ("< 100 -> the 24 h gap is mostly the model"
                                                          if z24 < 100 else ">= 100 -> IC information matters"))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
