"""Step 13 vs ClimODE on the same ICs (the formal paired comparison of steps 11-12, for step 13 arms).

ClimODE was reproduced on its first 8 ICs of each test year (runs/step11/climode/climode_pred.npz,
scripts/run_climode_eval.py; about 18 min per IC pair on CPU). Every step 13 arm saved in the stage-2
format (<prefix>_ens_<year>.npz) and the step 12 core are scored on exactly those ICs with the
vectorized ClimODE scores (scripts/run_step11_climode.py: rmse, acc, crps_gauss), and the paired
difference arm - ClimODE gets a 95% bootstrap interval over the ICs. Ensembles also get the ensemble
CRPS (properscoring, ClimODE's min-max units), which ClimODE's Gaussian output does not have.

    python scripts/compare_step13_climode.py --era5 /path/era5_data --out runs/step13/climode_paired.json \
        --arms runs/step13/seqfm_dct_raw runs/step13/seqfm_dct_guided ... --core runs/step12/core_jump24
"""

import argparse
import json
import os
import sys

import numpy as np
import properscoring as ps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.report_step11 import boot_ci  # noqa: E402
from scripts.run_step11_climode import VARS, climode_minmax, climode_scores, climode_truth  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402

YEARS = (2017, 2018)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--era5", required=True)
    p.add_argument("--climode", default="runs/step11/climode/climode_pred.npz")
    p.add_argument("--arms", nargs="*", default=[], help="prefixes of saved ensembles (<prefix>_ens_<year>.npz)")
    p.add_argument("--core", nargs="*", default=[], help="step 12 core prefixes (<prefix>_pred.npz)")
    p.add_argument("--out", default="runs/step13/climode_paired.json")
    a = p.parse_args(argv)
    cp = dict(np.load(a.climode))
    ics, leads = [int(i) for i in cp["ics"]], [int(h) for h in cp["lead_hours"]]
    mx, mn = climode_minmax(a.era5)
    lat = load_static(a.era5)[0]
    sc_ = (mx - mn)[:, None, None]
    scores = {}  # name -> lead -> metric -> list over ICs of (5,)

    def add(name, h, met, v):
        scores.setdefault(name, {}).setdefault(str(h), {}).setdefault(met, []).append(np.asarray(v, np.float64))

    for yi, y in enumerate(YEARS):
        truth = climode_truth(a.era5, y)
        clim = truth.mean(0)  # ClimODE: time mean of the test year
        assert np.allclose(cp["truth"][yi, :, 1:], truth[[[i + h // 6 for h in leads] for i in ics]], rtol=1e-4,
                           atol=1e-3), "ClimODE's saved truth does not match the ICs"
        sources = {}
        for prefix in a.arms:
            f = f"{prefix}_ens_{y}.npz"
            if os.path.exists(f):
                z = np.load(f)
                rows = [int(np.flatnonzero(z["ics"] == i)[0]) for i in ics]
                lz = [int(h) for h in z["leads"]]
                ens = z["E_minus_x0"][rows].astype(np.float32) + z["X0"][rows][:, None, None]  # (n, M, L, 5, H, W)
                sources[os.path.basename(prefix)] = ("ens", ens, lz)
            else:
                print(f"  {y}: {f} missing, skipped")
        for prefix in a.core:
            with np.load(prefix + "_pred.npz") as z:
                if str(y) in z.files:
                    sources[os.path.basename(prefix)] = ("det", z[str(y)][[(i - 2) // 8 for i in ics]],
                                                         [int(h) for h in z["leads"]])
        for e, i0 in enumerate(ics):
            for j, h in enumerate(leads):
                obs = truth[i0 + h // 6]
                s_c = climode_scores(cp["mean"][yi, e, j], obs, cp["std"][yi, e, j], clim, lat, mx, mn)
                for met in ("rmse", "acc", "crps_gauss"):
                    add("climode", h, met, s_c[met])
                pers = climode_scores(truth[i0], obs, np.full_like(obs, 1e-6), clim, lat, mx, mn)
                for met in ("rmse", "acc", "crps_gauss"):
                    add("persistence", h, met, pers[met])
                    add("persistence-climode", h, met, pers[met] - s_c[met])
                for name, (kind, x, lz) in sources.items():
                    k = lz.index(h)
                    if kind == "ens":
                        mem = x[e, :, k]
                        s = climode_scores(mem.mean(0), obs, mem.std(0, ddof=1), clim, lat, mx, mn)
                        s["crps_ens"] = ps.crps_ensemble((obs - mn[:, None, None]) / sc_,
                                                         np.moveaxis((mem - mn[:, None, None]) / sc_, 0, -1)).mean((-1, -2))
                    else:
                        s = climode_scores(x[e, k], obs, np.full_like(obs, 1e-6), clim, lat, mx, mn)
                    for met, v in s.items():
                        add(name, h, met, v)
                        if met in s_c:
                            add(f"{name}-climode", h, met, v - s_c[met])
    out = {"n_ic": {}, "leads": leads, "vars": VARS, "scores": {}}
    for name, by_h in scores.items():
        for h, d in by_h.items():
            for met, v in d.items():
                v = np.stack(v)
                out["n_ic"][name] = len(v)
                out["scores"].setdefault(name, {}).setdefault(h, {})[met] = {
                    var: {"mean": float(v[:, i].mean()), "ci95": boot_ci(v[:, i]) if name.endswith("-climode") else None}
                    for i, var in enumerate(VARS)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[paired] ClimODE's first {len(ics)} ICs of each test year; ICs per model {out['n_ic']}")
    for name, by_h in out["scores"].items():
        for met in ("rmse", "crps_gauss", "crps_ens"):
            if met not in by_h[str(leads[0])]:
                continue
            row = []
            for h in (24, 72, 144):
                r = by_h[str(h)][met]
                row.append(" ".join((f"{r[v]['mean']:.0f}" if met == "rmse" and v == "z" else f"{r[v]['mean']:.4g}")
                                    + (f" [{r[v]['ci95'][0]:.3g}, {r[v]['ci95'][1]:.3g}]" if r[v]["ci95"] else "")
                                    for v in ("z", "t")))
            print(f"[{name}] {met} z500 t850 at 24/72/144 h: " + " | ".join(row))
    return out


if __name__ == "__main__":
    main()
