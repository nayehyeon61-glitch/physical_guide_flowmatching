"""Step 13 maps of the best ensembles (regression mean + FM deviations, energy-scaled per DCT band) and their energy.

    fig_maps_<var>.png   one IC (ClimODE's first 2017 IC), rows = 24 / 72 / 144 h: truth | regression mean |
                         member 1 and spread under the closed-form band Gamma (reg_band) and under the learned
                         guide (reg_guide) | mean error | ClimODE mean error | ClimODE spread (its Gaussian std)
    fig_energy.png       the energy the guide controls: radial DCT spectrum (orthonormal 2-D DCT, the guide's
                         basis) of the member deviations, scaled by (1 + 1/M), against the error of the mean
                         at 24 / 72 / 144 h (a calibrated ensemble lies on its error line at every scale), and
                         the realized band Gamma of both arms, sqrt(deviation energy / raw FM deviation energy)
    fig_basis.png        the owner's DCT basis against the grid control (same FM, same frozen PINN): z500 CRPS,
                         lag-24 h member correlation, small-scale power, and band spread / error energy
                         (score_b.json from scripts/score_step13.py, band energies from the saved ensembles)
    fig_flow_fm.png      what the flow-dependent spread field (docs/step13_sequence.md 4.3) does to the FM
                         members: spread maps of the raw FM + PINN members, regm_band and the sigma arms with
                         their factor f, on the hardest and the easiest 2017 case (error of mu* at 72 h), and
                         the case-to-case domain-mean spread against the RMSE (``--only flow``)

    python scripts/plot_step13_maps.py --era5 /path/era5_data --out docs/step13
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.fft import dctn  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm.ens_guide import BAND_EDGES, band_index  # noqa: E402
from scripts.plot_step13 import GRID, INK, MUTED, SLOTS, persistence  # noqa: E402
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


FLOW = [  # (prefix, short name, title, slot); slots as in scripts/plot_step13_flowdep.py
    (f"{O_}/seqfm_dct_pinn", "fm", "FM + PINN members (raw)", 0),
    (f"{O_}/seqfm_dct_pinn_regm_band", "band", "regm_band: FM deviations × band Γ", 7),
    (f"{O_}/seqfm_dct_pinn_regm_flow", "flow", "regm_flow: × f(x), ClimODE net", 0),
    (f"{O_}/seqfm_dct_pinn_regm_flowop", "flowop", "regm_flowop: × f(x), operator", 2),
]


def plot_flow(year, era5, out, v="z", lead=72):
    from datetime import datetime, timedelta
    from matplotlib.colors import LogNorm
    lat, lon, _, lsm = load_static(era5)
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    w = w / w.mean()
    arms = [q for q in FLOW if os.path.exists(f"{q[0]}_ens_{year}.npz")]
    mem, obs = {}, None
    for prefix, k, *_ in arms:
        ens, obs, ics, leads = load_var(prefix, year, v)
        mem[k] = ens[:, :, leads.index(lead)]
    obs = obs[:, leads.index(lead)]
    sd = {k: m.std(1, ddof=1) for k, m in mem.items()}
    err = {k: m.mean(1) - obs for k, m in mem.items()}  # regm_* share mu*
    dom = lambda f: (f * w).mean((-1, -2))  # noqa: E731
    rmse = np.sqrt(dom(err["band"] ** 2))
    cases = [("Hardest case", int(rmse.argmax())), ("Easiest case", int(rmse.argmin()))]
    sig = [k for k in ("flow", "flowop") if k in mem]
    cols = [("|μ* − truth|", None)] + [(t, k) for _, k, t, _ in arms] + [(f"factor f (regm_{k})", k) for k in sig]
    fig = plt.figure(figsize=(4.1 * len(cols) + 2, 15.5), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[2, 1.15])
    ax = top.subplots(2, len(cols))
    u = UNITS[v]
    for row, (name, r) in enumerate(cases):
        mags = [np.abs(err["band"][r])] + [sd[k][r] for _, k, *_ in arms]
        vmax = np.percentile(np.stack(mags), 99)
        for c, f in enumerate(mags):
            a_ = ax[row, c]
            im = a_.imshow(f, origin="lower", extent=ext, cmap="Purples", vmin=0, vmax=vmax, aspect="auto")
            note = f"mean {dom(f):.3g}"
            if c and arms[c - 1][1] != "fm":
                note += f", corr with |err| {np.corrcoef(f.ravel(), mags[0].ravel())[0, 1]:.2f}"
            a_.text(0.02, 0.04, note, transform=a_.transAxes, fontsize=9, color=INK,
                    bbox=dict(facecolor="white", alpha=0.75, lw=0, pad=1.5))
        top.colorbar(im, ax=ax[row, :len(mags)], shrink=0.85, pad=0.01, label=f"{u}")
        for c, k in enumerate(sig):
            a_ = ax[row, len(mags) + c]
            fac = sd[k][r] / sd["band"][r]
            imf = a_.imshow(fac, origin="lower", extent=ext, cmap="RdBu_r", norm=LogNorm(0.5, 2), aspect="auto")
            a_.text(0.02, 0.04, f"domain mean {dom(fac):.2f}", transform=a_.transAxes, fontsize=9, color=INK,
                    bbox=dict(facecolor="white", alpha=0.75, lw=0, pad=1.5))
        if sig:
            top.colorbar(imf, ax=ax[row, len(mags):], shrink=0.85, pad=0.01, label="f = spread / regm_band spread")
        for c in range(len(cols)):
            ax[row, c].contour(lon, lat, lsm, levels=[0.5], colors=INK, linewidths=0.4, alpha=0.5)
            style(ax[row, c])
            if row == 0:
                ax[row, c].set_title(cols[c][0], fontsize=11)
        day = (datetime(year, 1, 1) + timedelta(hours=6 * int(ics[r]))).strftime("%Y-%m-%d %H UTC")
        ax[row, 0].set_ylabel(f"{name}\nIC {day}", fontsize=11)
    bx = bot.subplots(1, len(arms))
    for c, (_, k, title, slot) in enumerate(arms):
        a_ = bx[c]
        x, y = np.sqrt(dom(sd[k] ** 2)), np.sqrt(dom(err[k] ** 2))
        a_.scatter(x, y, s=14, color=SLOTS[slot], alpha=0.7, lw=0)
        for (name, r), mk in zip(cases, ("^", "s")):
            a_.scatter(x[r], y[r], s=80, marker=mk, facecolor="none", edgecolor=INK, lw=1.4, label=name)
        lim = [0.9 * min(x.min(), y.min()), 1.05 * max(x.max(), y.max())]
        a_.plot(lim, lim, ":", color=MUTED, lw=1.5, label="spread = RMSE")
        a_.set_xlim(lim)
        a_.set_ylim(lim)
        a_.set_title(f"{title}\ncase corr {np.corrcoef(x, y)[0, 1]:.2f}, spread CV {x.std() / x.mean():.2f}",
                     fontsize=11)
        a_.set_xlabel(f"domain-mean spread ({u})")
        a_.grid(True, color=GRID, lw=0.8)
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    bx[0].set_ylabel(f"RMSE of the ensemble mean ({u})")
    bx[0].legend(fontsize=9, frameon=False, loc="upper left")
    fig.suptitle(f"Step 13: what the flow-dependent spread field does to the FM members, {TITLES[v]}, +{lead // 24} d, "
                 f"{year} ({len(ics)} ICs)\nThe FM is not retrained: f(x) = σ(x) / ρ rescales its member deviations "
                 "after the flow and band Γ, case by case; the mean μ* is unchanged", fontsize=13)
    p = os.path.join(out, "fig_flow_fm.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"[flow] {p}: " + ", ".join(k for _, k, *_ in arms))
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


BASIS = [  # (score key, label, line style, slot)
    ("seqfm_dct_raw", "FM on 3-D DCT coefficients (owner's design)", "-", 0),
    ("seqfm_dct_pinn", "DCT FM + frozen PINN", "--", 0),
    ("seqfm_grid_raw", "FM on grid values (control)", "-", 1),
    ("seqfm_grid_pinn", "grid FM + frozen PINN", "--", 1),
]


def plot_basis(era5, out, scores=f"{O_}/score_b.json", years=(2017, 2018), v="z"):
    d = json.load(open(scores))
    A, iv = d["arms"], d["vars"].index(v)
    lat = load_static(era5)[0]
    from scripts.run_step11_climode import climode_minmax
    mx, mn = climode_minmax(era5)
    _, p_crps = persistence(era5, list(years), lat, mx, mn)
    zs = (mx - mn)[iv]
    leads = A[BASIS[0][0]]["leads"]
    days = np.array(leads) / 24
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.8), layout="constrained")
    for k, lab, ls, slot in BASIS:
        r = A[k]
        ax[0, 0].plot(days, np.asarray(r["crps"])[:, iv] * zs, ls, color=SLOTS[slot], lw=1.8, label=lab)
        ax[0, 1].plot(days[1:], np.asarray(r["lag24_member"])[:, iv], ls, color=SLOTS[slot], lw=1.8)
        ax[1, 0].plot(days, np.asarray(r["power_ratio_k13-32"])[:, iv], ls, color=SLOTS[slot], lw=1.8)
    ax[0, 0].plot(days, p_crps[:, iv] * zs, ":", color=MUTED, lw=2, label="Persistence")
    ax[0, 1].plot(days[1:], np.asarray(A[BASIS[0][0]]["lag24_truth"])[:, iv], "-", color=INK, lw=1.2, marker="s",
                  ms=5, label="ERA5 truth")
    ax[1, 0].axhline(1, color=INK, lw=1.2, label="ERA5 truth")
    ax[1, 0].set_yscale("log")
    for k, lab, _, slot in BASIS[::2]:  # raw arms; P barely changes the spread (0.99-1.00 through P)
        rat = []
        for y in years:
            ens, obs, _, lz = load_var(f"{O_}/{k}", y, v)
            M, mu = ens.shape[1], ens.mean(1)
            dev = band_energy(ens - mu[:, None]) / (M - 1) * M * (1 + 1 / M)  # band_energy averages over members
            bidx = band_index(*obs.shape[-2:]).numpy()
            c2 = dctn(mu - obs, axes=(-2, -1), norm="ortho") ** 2
            err = np.stack([c2[..., bidx == b].mean(-1).mean(0) for b in range(int(bidx.max()) + 1)], -1)
            rat.append((dev, err))
            del ens, obs
        dev, err = sum(q[0] for q in rat), sum(q[1] for q in rat)
        for h, ls_ in ((24, "--"), (144, "-")):
            j = lz.index(h)
            ax[1, 1].plot(range(dev.shape[1]), dev[j] / err[j], ls_, marker="o", color=SLOTS[slot], lw=1.8, ms=5,
                          label=f"{lab.split(' (')[0]}, +{h // 24} d")
    ax[1, 1].axhline(1, color=INK, lw=1.2)
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xticks(range(5), ["0\nlargest", "1", "2", "3", "4\nsmallest"])
    titles = [(ax[0, 0], "Geopotential at 500 hPa: ensemble CRPS", "CRPS (m²/s²)", "Forecast lead (days)"),
              (ax[0, 1], "Member correlation over 24 h (time coherence)", "corr(x(τ), x(τ + 24 h)) of anomalies",
               "Forecast lead of the later field (days)"),
              (ax[1, 0], "Small-scale power, zonal wavenumbers 13–32", "Member power / ERA5 power",
               "Forecast lead (days)"),
              (ax[1, 1], "Spread / error energy per guide band (1 = calibrated)", "(1 + 1/M) spread² / error²",
               "DCT band of the guide")]
    for a_, t_, yl, xl in titles:
        a_.set_title(t_, fontsize=13)
        a_.set_ylabel(yl)
        a_.set_xlabel(xl)
        a_.grid(True, color=GRID, lw=0.8)
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
        if a_ is not ax[1, 1]:
            a_.set_xlim(1, 6)
    ax[1, 1].legend(fontsize=9, frameon=False)
    h0, l0 = ax[0, 0].get_legend_handles_labels()
    h1, l1 = ax[0, 1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Step 13: DCT basis vs grid control for the sequence FM, z500, test 2017–2018 (360 ICs, 8 members)\n"
                 "Same U-Net, same 3000 iterations, same frozen PINN; only the space the flow runs in differs",
                 fontsize=13)
    p = os.path.join(out, "fig_basis.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[basis] {p}")
    return p


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--era5", required=True)
    p.add_argument("--year", type=int, default=2017)
    p.add_argument("--climode", default="runs/step11/climode/climode_pred.npz")
    p.add_argument("--out", default="docs/step13")
    p.add_argument("--only", choices=["maps", "energy", "basis", "flow"], default=None)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.only == "basis":
        return [plot_basis(a.era5, a.out)]
    if a.only == "flow":
        return [plot_flow(a.year, a.era5, a.out)]
    cp = dict(np.load(a.climode))
    out = [plot_maps(v, a.year, a.era5, cp, a.out) for v in ("z", "t")] if a.only in (None, "maps") else []
    if a.only in (None, "energy"):
        out.append(plot_energy(a.year, a.out))
    if a.only is None:
        out.append(plot_basis(a.era5, a.out))
    return out


if __name__ == "__main__":
    main()
