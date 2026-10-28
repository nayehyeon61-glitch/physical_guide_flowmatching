"""Merge several run_climode_eval.py prediction files (different IC subsets of the same test years) into one
climode_pred.npz for scripts/compare_step13_climode.py, sorted by IC.

    python scripts/merge_climode_pred.py --out runs/step22/climode_pred_34.npz \
        runs/step11/climode/climode_pred.npz runs/step22/climode_spread/climode_pred.npz
"""

import argparse

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    zs = [dict(np.load(f)) for f in a.files]
    for z in zs[1:]:
        assert np.array_equal(z["lead_hours"], zs[0]["lead_hours"]) and np.allclose(z["max_lev"], zs[0]["max_lev"])
    ics = np.concatenate([z["ics"] for z in zs])
    assert len(set(ics.tolist())) == len(ics), "duplicate ICs"
    order = np.argsort(ics)
    out = {k: zs[0][k] for k in ("lead_hours", "max_lev", "min_lev")}
    for k in ("mean", "std", "truth"):
        out[k] = np.concatenate([z[k] for z in zs], 1)[:, order]
    out["ics"] = ics[order]
    np.savez_compressed(a.out, **out)
    print(f"[merge] {a.out}: ICs {out['ics'].tolist()}, mean {out['mean'].shape}")


if __name__ == "__main__":
    main()
