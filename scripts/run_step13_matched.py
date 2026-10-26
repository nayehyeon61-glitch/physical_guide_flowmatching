"""Step 13 control for the band x cell result (docs/step13_sequence.md 4.4): is its gain a spatial
redistribution of the spread, or only a larger total spread?

    <ref>_matched  the members of ``--ref`` with their deviations about the ensemble mean rescaled by one factor
                   per (lead, variable), so the area-weighted mean spread equals that of ``--target`` in the same
                   year; no grid-point or case information is added and the mean is unchanged. The factor is
                   computed from the two ensembles only (no truth).

    python scripts/run_step13_matched.py --ref runs/step13/seqfm_dct_pinn_regm_band \
        --target runs/step13/seqfm_dct_pinn_regm_sband_clim
"""

import argparse
import json

import numpy as np


def area_weights(H: int) -> np.ndarray:
    lat = np.linspace(-90 + 90 / H, 90 - 90 / H, H)
    w = np.cos(np.deg2rad(lat))
    return (w / w.mean())[:, None]


def mean_spread2(E: np.ndarray, w: np.ndarray) -> np.ndarray:
    """E (n, M, L, C, H, W) -> area-weighted mean member variance about the ensemble mean, (L, C)."""
    d = E - E.mean(1, keepdims=True)
    return ((d ** 2).sum(1) / (E.shape[1] - 1) * w).mean((0, -1, -2))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="prefix of the arm to rescale (<prefix>_ens_<year>.npz)")
    p.add_argument("--target", required=True, help="prefix of the arm whose mean spread is matched")
    p.add_argument("--years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--suffix", default="_matched")
    a = p.parse_args(argv)
    out = {}
    for y in a.years:
        r, t = np.load(f"{a.ref}_ens_{y}.npz"), np.load(f"{a.target}_ens_{y}.npz")
        assert (r["ics"] == t["ics"]).all() and np.allclose(r["O"], t["O"])
        E = r["E_minus_x0"].astype(np.float32)
        w = area_weights(E.shape[-2])
        f = np.sqrt(mean_spread2(t["E_minus_x0"].astype(np.float32), w) / mean_spread2(E, w))  # (L, C)
        mu = E.mean(1, keepdims=True)
        E = mu + f[None, None, :, :, None, None] * (E - mu)
        np.savez(f"{a.ref}{a.suffix}_ens_{y}.npz", E_minus_x0=E.astype(np.float16), O=r["O"], X0=r["X0"],
                 ics=r["ics"], leads=r["leads"])
        out[y] = f.tolist()
        print(f"[matched] {y}: spread factor by lead (mean over variables) "
              + " ".join(f"{v:.3f}" for v in f.mean(1)) + f" -> {a.ref}{a.suffix}_ens_{y}.npz", flush=True)
    with open(f"{a.ref}{a.suffix}_factors.json", "w") as fh:
        json.dump(dict(ref=a.ref, target=a.target, factor=out), fh, indent=1)


if __name__ == "__main__":
    main()
