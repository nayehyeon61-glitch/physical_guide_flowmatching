"""Step 13 maps of the best ensembles (regression mean + FM deviations, energy-scaled per DCT band) and their energy.

    fig_maps_<var>.png   one IC (ClimODE's first 2017 IC), rows = 24 / 72 / 144 h: truth | regression mean |
                         member 1 and spread under the closed-form band Gamma (reg_band) and under the learned
                         guide (reg_guide) | mean error | ClimODE mean error | ClimODE spread (its Gaussian std)
    fig_energy.png       the energy the guide controls: radial DCT spectrum (orthonormal 2-D DCT, the guide's
                         basis) of the member deviations, scaled by (1 + 1/M), against the error of the mean
                         at 24 / 72 / 144 h (a calibrated ensemble lies on its error line at every scale), and
                         the realized band Gamma of both arms, sqrt(deviation energy / raw FM deviation energy)

    python scripts/plot_step13_maps.py --era5 /path/era5_data --out docs/step13
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.fft import dctn  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm.ens_guide import BAND_EDGES, band_index  # noqa: E402
from scripts.plot_step13 import GRID, INK, MUTED, SLOTS  # noqa: E402
from scripts.run_step11_climode import VARS  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402

O_ = "runs/step13"
RAW = f"{O_}/seqfm_dct_pinn_raw"
BEST = {"band": f"{O_}/seqfm_dct_pinn_reg_band", "guide": f"{O_}/seqfm_dct_pinn_reg_guide"}
SPECTRA = [  # (prefix, label, line style, slot); reg_* share the regression mean, so no own error line
    (f"{O_}/seqregress_pinn", "Regression + statistical errors", "-", 6),
    (RAW, "FM + PINN members (raw)", "-", 0),
    (BEST["band"], "Regression mean + FM deviations, closed-form band Γ", "-", 5),
    (BEST["guide"], "Regression mean + FM deviations, learned band guide", "--", 5),
]
UNITS = {"z": "m²/s²", "t": "K"}
TITLES = {"z": "Geopotential at 500 hPa", "t": "Temperature at 850 hPa"}


def load_var(prefix, year, v):
    """Members (n, M, L, H, W), truth (n, L, H, W), ics, leads of one variable, float32."""
    iv = VARS.index(v)
    with np.load(f"{prefix}_ens_{year}.npz") as z:
        ens = z["E_minus_x0"][:, :, :, iv].astype(np.float32) + z["X0"][:, None, None, iv]
        return ens, z["O"][:, :, iv].astype(np.float32), z["ics"], [int(h) for h in z["leads"]]


def style(a_):
    a_.set_xticks([])
    a_.set_yticks([])
    for s_ in a_.spines.values():
        s_.set_color(GRID)


def plot_maps(v, year, era5, cp, out, leads_show=(24, 72, 144)):
    lat, lon, _, lsm = load_static(era5)
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    ens = {}
    for k, prefix in BEST.items():
        ens[k], obs, ics, leads = load_var(prefix, year, v)
    i0 = int(cp["ics"][0])
    r = int(np.flatnonzero(ics == i0)[0])
    yi, iv = (2017, 2018).index(year), VARS.index(v)
    w = np.cos(np.deg2rad(lat))[:, None]
    cols = ["Truth (ERA5)", "Ours: regression mean", "Member 1, band Γ", "Member 1, learned guide",
            "Spread, band Γ", "Spread, learned guide", "Ours: mean − truth", "ClimODE: mean − truth",
            "ClimODE: spread (σ)"]
    fig, ax = plt.subplots(len(leads_show), len(cols), figsize=(29, 2.9 * len(leads_show) + 1.3),
                           layout="constrained")
    rmse = lambda e: np.sqrt((e ** 2 * w).mean() / w.mean())  # noqa: E731
    for row, h in enumerate(leads_show):
        j, jc = leads.index(h), [int(q) for q in cp["lead_hours"]].index(h)
        tru = obs[r, j]
        mb, mg = ens["band"][r, :, j], ens["guide"][r, :, j]
        mu = mb.mean(0)  # = mg.mean(0): both arms keep the regression mean exactly
        sb, sg = mb.std(0, ddof=1), mg.std(0, ddof=1)
        c_mu, c_sd = cp["mean"][yi, 0, jc, iv], cp["std"][yi, 0, jc, iv]
        lo, hi = np.percentile(tru, [1, 99])
        e_max = np.percentile(np.abs(np.stack([mu - tru, c_mu - tru])), 99)
        s_max = np.percentile(np.stack([sb, sg, c_sd]), 99)
        panels = [(tru, "viridis", lo, hi, ""), (mu, "viridis", lo, hi, ""), (mb[0], "viridis", lo, hi, ""),
                  (mg[0], "viridis", lo, hi, ""), (sb, "Purples", 0, s_max, f"mean {sb.mean():.3g}"),
                  (sg, "Purples", 0, s_max, f"mean {sg.mean():.3g}"),
                  (mu - tru, "RdBu_r", -e_max, e_max, f"RMSE {rmse(mu - tru):.3g}"),
                  (c_mu - tru, "RdBu_r", -e_max, e_max, f"RMSE {rmse(c_mu - tru):.3g}"),
                  (c_sd, "Purples", 0, s_max, f"mean {c_sd.mean():.3g}")]
        ims = []
        for c, (f, cmap, a, b, note) in enumerate(panels):
            a_ = ax[row, c]
            ims.append(a_.imshow(f, origin="lower", extent=ext, cmap=cmap, vmin=a, vmax=b, aspect="auto"))
            if cmap == "viridis":
                a_.contour(lon, lat, f, levels=8, colors="white", linewidths=0.5, alpha=0.7)
            a_.contour(lon, lat, lsm, levels=[0.5], colors=INK if cmap != "viridis" else "#ffffff",
                       linewidths=0.4, alpha=0.5)
            style(a_)
            if note:
                a_.text(0.02, 0.04, note, transform=a_.transAxes, fontsize=9, color=INK,
                        bbox=dict(facecolor="white", alpha=0.75, lw=0, pad=1.5))
            if row == 0:
                a_.set_title(cols[c], fontsize=11)
        ax[row, 0].set_ylabel(f"+{h // 24} d ({h} h)", fontsize=12)
        u = UNITS[v]
        fig.colorbar(ims[0], ax=ax[row, :4], shrink=0.85, pad=0.01, label=u)
        fig.colorbar(ims[4], ax=ax[row, 4:6], shrink=0.85, pad=0.01, label=f"spread ({u})")
        fig.colorbar(ims[6], ax=ax[row, 6:8], shrink=0.85, pad=0.01, label=f"error ({u})")
        fig.colorbar(ims[8], ax=ax[row, 8], shrink=0.85, pad=0.01, label=f"spread ({u})")
    day = f"{year}-01-0{1 + i0 // 4} {6 * (i0 % 4):02d} UTC"
    fig.suptitle(f"Best step 13 ensembles on ClimODE's first {year} IC ({day}): {TITLES[v]}\n"
                 "Ours: sequence regression mean + FM + PINN member deviations, energy-scaled per DCT band "
                 "(closed-form Γ or learned guide), 8 members. ClimODE: authors' checkpoint, Gaussian mean and σ "
                 "(spread colour scale shared)", fontsize=13)
    p = os.path.join(out, f"fig_maps_{v}.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"[maps] {p}")
    return p


def radial_spectrum(f, nbin=16):
    """Mean squared orthonormal 2-D DCT coefficient per radial bin of the last two axes -> (..., nbin)."""
    H, W = f.shape[-2:]
    rad = np.sqrt((np.arange(H)[:, None] / H) ** 2 + (np.arange(W)[None] / W) ** 2)
    edges = np.linspace(0, rad.max() + 1e-9, nbin + 1)
    b = np.digitize(rad, edges) - 1
    c2 = dctn(f, axes=(-2, -1), norm="ortho") ** 2
    return np.stack([c2[..., b == k].mean(-1) for k in range(nbin)], -1), 0.5 * (edges[1:] + edges[:-1])


def band_energy(dev):
    """Mean squared DCT coefficient of deviations (n, M, L, H, W) per guide band -> (L, nb)."""
    bidx = band_index(*dev.shape[-2:]).numpy()
    c2 = dctn(dev, axes=(-2, -1), norm="ortho") ** 2
    return np.stack([c2[..., bidx == b].mean(-1).mean((0, 1)) for b in range(int(bidx.max()) + 1)], -1)


def plot_energy(year, out, v="z", leads_show=(24, 72, 144)):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10.4), layout="constrained")
    flat = [ax[0, 0], ax[0, 1], ax[1, 0]]
    band_e = {}
    for prefix, lab, ls, slot in SPECTRA:
        ens, obs, _, leads = load_var(prefix, year, v)
        M = ens.shape[1]
        dev_all = ens - ens.mean(1, keepdims=True)
        if prefix in (RAW, *BEST.values()):
            band_e[prefix] = band_energy(dev_all)
        for a_, h in zip(flat, leads_show):
            j = leads.index(h)
            dev, k = radial_spectrum(dev_all[:, :, j])
            a_.plot(k, dev.sum(1).mean(0) / (M - 1) * (1 + 1 / M), ls, color=SLOTS[slot], lw=1.8,
                    label=f"{lab}: spread")
            if prefix not in BEST.values():
                err, _ = radial_spectrum(ens[:, :, j].mean(1) - obs[:, j])
                a_.plot(k, err.mean(0), ":", color=SLOTS[slot], lw=2, label=f"{lab}: error of its mean")
        del ens, obs, dev_all
    for a_, h in zip(flat, leads_show):
        for e in BAND_EDGES:
            a_.axvline(e, color=GRID, lw=1, zorder=0)
        a_.set_yscale("log")
        a_.set_title(f"{TITLES[v]}, +{h // 24} d", fontsize=13)
        a_.set_xlabel("Normalized DCT radius (large scales left; vertical lines = guide bands)")
        a_.set_ylabel(f"Energy per coefficient ({UNITS[v]})²")
        a_.grid(True, color=GRID, lw=0.8, which="major")
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    a_ = ax[1, 1]
    days = np.array(leads) / 24
    nb = band_e[RAW].shape[1]
    ramp = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, nb))
    names = ["largest scales", "", "", "", "smallest scales"]
    for key, ls in (("band", "-"), ("guide", "--")):
        g = np.sqrt(band_e[BEST[key]] / band_e[RAW])
        for b in range(nb):
            a_.plot(days, g[:, b], ls, marker="o", color=ramp[b], lw=1.8, ms=4,
                    label=(f"band {b}" + (f" ({names[b]})" if names[b] else "")) if key == "band" else None)
    a_.plot([], [], "-", color=MUTED, label="closed-form band Γ")
    a_.plot([], [], "--", color=MUTED, label="learned guide")
    a_.axhline(1, color=MUTED, ls=":", lw=1.5)
    a_.set_yscale("log")
    a_.set_title("Realized Γ on the FM deviations, √(energy after / before)", fontsize=13)
    a_.set_xlabel("Forecast lead (days)")
    a_.set_ylabel("Γ (1 = FM spread kept; < 1 = energy removed)")
    a_.set_xlim(1, 6)
    a_.legend(fontsize=9, frameon=False, ncol=2)
    a_.grid(True, color=GRID, lw=0.8)
    for s_ in ("top", "right"):
        a_.spines[s_].set_visible(False)
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=9,
               frameon=False)
    fig.suptitle(f"Step 13 energy control, {TITLES[v]}, {year} (180 ICs, 8 members)\n"
                 "Solid/dashed: member deviation energy × (1 + 1/M); dotted: energy of the mean's error\n"
                 "Calibrated = spread on its error line (purple dotted for both regression-mean arms)", fontsize=12)
    p = os.path.join(out, "fig_energy.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[energy] {p}")
    return p


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--era5", required=True)
    p.add_argument("--year", type=int, default=2017)
    p.add_argument("--climode", default="runs/step11/climode/climode_pred.npz")
    p.add_argument("--out", default="docs/step13")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    cp = dict(np.load(a.climode))
    out = [plot_maps(v, a.year, a.era5, cp, a.out) for v in ("z", "t")]
    out.append(plot_energy(a.year, a.out))
    return out


if __name__ == "__main__":
    main()
