"""Step 16: at which zonal scales does the strong residual see the tangent vector?

On ERA5 states x with the centred 12 h difference tendency xdot: R_full = xdot + A_h(x) (= the operator
error e(x) if ERA5 solved the continuous equations) and the tendency term alone D = xdot. Per zonal
wavenumber k: signal / error = |D_k|^2 / |R_full,k|^2 for u, v, T. Above 1 the residual's implied
tangent vector -A_h(x) is closer to the truth than the rest state; below 1 it is not.

    python scripts/step16_tangent_spectrum.py --kai ... --era5 ... --stats runs/step16/stats --out docs/step16
"""

import argparse
import json
import os
import sys

import matplotlib
import netCDF4 as nc
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pgcfm import contract  # noqa: E402
from pgcfm.coupling import PhysicsConfig, PhysicsGuide  # noqa: E402
from pgcfm.dyncore import StrongPhysics, finite_difference_tendency  # noqa: E402
from pgcfm.runs import load_years  # noqa: E402
from scripts.run_step13_pinn import sample  # noqa: E402

DYN = contract.DYNAMIC_CHANNELS


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--kai", required=True)
    p.add_argument("--era5", required=True)
    p.add_argument("--stats", required=True, help="folder with global_mean.npy / global_std.npy")
    p.add_argument("--year", type=int, default=2016)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--out", default="docs/step16")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    mean, std = np.load(os.path.join(a.stats, "global_mean.npy")), np.load(os.path.join(a.stats, "global_std.npy"))
    data = load_years(a.kai, [a.year], mean, std)
    with nc.Dataset(os.path.join(a.kai, f"{a.year}.nc")) as d:
        contract.register_latitudes(np.asarray(d["latitude"][:]))
    guide = PhysicsGuide(32, 64, mean, std, PhysicsConfig(dt_seconds=6 * 3600.0))
    phys = StrongPhysics(guide, torch.ones(contract.N_DYNAMIC))
    rng = np.random.default_rng(7)
    sig, err, n = {}, {}, 0
    with torch.no_grad():
        for _ in range(0, a.n, 8):
            x0, d, orog, _ = sample(data, [a.year], 2, rng, 8)
            x1 = x0 + d[:, 0]
            R1, m = phys.residuals(x1, orog, finite_difference_tendency(x0, x0 + d[:, 1]))
            R0, _ = phys.residuals(x1, orog, torch.zeros_like(x1))
            g = guide.grid(x1.device)
            wt = (g.latitude_weight("area") * m).double()  # (N, L, H-2, W) broadcast
            for q in ("u", "v", "T"):
                D, E = R1[q] - R0[q], R1[q]
                for name, r, acc in (("sig", D, sig), ("err", E, err)):
                    r = torch.where(wt > 0, r, torch.zeros_like(r))
                    pk = (torch.fft.rfft(r, dim=-1).abs() ** 2) * wt.mean(-1, keepdim=True)
                    acc[q] = acc.get(q, 0.0) + pk.sum((0, 1, 2)).numpy() / wt.sum((0, 1, 2)).mean().item()
            n += 1
    k = np.arange(len(next(iter(sig.values()))))
    ratio = {q: (sig[q] / err[q]).tolist() for q in sig}
    cum = {q: (np.cumsum(sig[q]) / np.cumsum(err[q])).tolist() for q in sig}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "tangent_spectrum.json"), "w") as f:
        json.dump({"k": k.tolist(), "ratio": ratio, "cumulative_ratio": cum, "n_states": 8 * n}, f, indent=1)
    for f_ in ("WenQuanYi Zen Hei", "Noto Sans CJK KR", "DejaVu Sans"):
        if any(f_ == x.name for x in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f_; plt.rcParams["axes.unicode_minus"] = False
            break
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for q, c in zip(("u", "v", "T"), ("#4477AA", "#EE6677", "#228833")):
        ax[0].plot(k[1:], ratio[q][1:], color=c, lw=2, marker="o", ms=3, label=q)
        ax[1].plot(k, cum[q], color=c, lw=2, marker="o", ms=3, label=q)
    for q in ax:
        q.axhline(1.0, color="k", ls=":", lw=1), q.set_yscale("log"), q.grid(alpha=0.3), q.legend()
        q.set_xlabel("zonal wavenumber k")
    ax[0].set_title("경향항 / 잔차(연산자 오차) 파워 비, 파수별")
    ax[1].set_title("누적 (k 이하 모드만 남긴 잔차)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_tangent_spectrum.png"), dpi=130)
    print("k   " + " ".join(f"{q:>6}" for q in ratio))
    for i in k:
        print(f"{i:2d}  " + " ".join(f"{ratio[q][i]:6.2f}" for q in ratio))
    print("cumulative k<=3 " + " ".join(f"{q} {cum[q][3]:.2f}" for q in cum) + " | k<=8 "
          + " ".join(f"{q} {cum[q][8]:.2f}" for q in cum))


if __name__ == "__main__":
    main()
