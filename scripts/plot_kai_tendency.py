"""Figures of "where does the tendency core go in KAI" (scripts/run_kai_tendency.py --score).

    python scripts/plot_kai_tendency.py runs/kaitend docs/kai_tendency [tags to draw ...]
fig_kai_lead.png      lead-time lines (normalized RMSE, z500, t850, t2m, q700 RMSE, z500 ACC), persistence dotted
fig_kai_where.png     relative RMSE change vs KAI per channel x lead, one panel per position (x = not significant)
fig_kai_scale.png     error power relative to KAI by zonal wavenumber band, and prediction / truth power
fig_kai_effect.png    the core against its controls (out vs res / stack / seeds, blend controls), out vs res per channel
fig_kai_bias.png      global-mean bias per lead (the core's drift)
fig_kai_reliance.png  normalized RMSE of each position with tau, with tau = 0 and with the core open loop;
                      z500 small-scale power; the blend's KAI weight
where_summary.json    the numbers behind the figures
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STYLE = {"persistence": dict(color="k", ls=":", label="persistence"),
         "climatology": dict(color="0.6", ls=":", label="climatology"),
         "kai": dict(color="k", lw=2, label="KAI original"),
         "kai_long": dict(color="k", ls="--", label="KAI original, 4x iterations"),
         "core": dict(color="0.5", lw=2, label="tendency core alone"),
         "res": dict(color="0.5", ls="--", label="res: x + net (no core)"),
         "in": dict(color="#1f77b4", label="in: tau at the input"),
         "s1": dict(color="#9ecae1", label="s1: after stage 1"),
         "s2": dict(color="#41ab5d", label="s2: after stage 2"),
         "s3": dict(color="#a1d99b", label="s3: after stage 3"),
         "s4": dict(color="#006d2c", label="s4: after stage 4 (before head)"),
         "inres": dict(color="#1f77b4", ls="--", label="inres: res + tau at the input"),
         "out": dict(color="#d62728", lw=2, label="out: core forecast + net"),
         "inout": dict(color="#ff7f0e", label="inout: out + tau at the input"),
         "stack": dict(color="#8c564b", label="stack: KAI(res) forecast + net (control of out)"),
         "blend": dict(color="#9467bd", lw=2, label="blend of KAI and core (no training)")}
STYLE.update({
    "core_gn": dict(color="#17becf", lw=2, label="core, no BatchNorm (gn)"),
    "core_gn2": dict(color="#17becf", ls="--", label="core gn + 2 Heun steps per day"),
    "core_gnk2": dict(color="#17becf", ls=":", label="core gn, rollout curriculum 1-2 days"),
    "out_gn": dict(color="#e377c2", lw=2, label="out with core gn"),
    "out_gn2": dict(color="#e377c2", ls="--", label="out with core gn2"),
    "out_gnk2": dict(color="#e377c2", ls=":", label="out with core gnk2"),
    "in_gn": dict(color="#1f77b4", ls=":", label="in with core gn"),
    "kai_s1": dict(color="k", ls=":", lw=1, label="KAI original, seed 1"),
    "kai_ro": dict(color="k", ls="-.", lw=1.5, label="KAI original + rollout fine-tuning"),
    "blend_core_gn": dict(color="#9467bd", ls="--", label="blend of KAI and core gn"),
    "blend_kai_s1": dict(color="#c5b0d5", lw=1.5, label="blend of two KAI seeds (control)"),
    "loop_core_gn": dict(color="#bcbd22", lw=2, label="KAI and core gn mixed every step"),
})
BANDS = [(0, 0), (1, 3), (4, 9), (10, 20), (21, 32)]
FALLBACK = ["#7f7f7f", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5"]


def style(k, used={}):
    """A style for any scored tag: the table above, else a fallback colour with the tag as label."""
    if k in STYLE:
        return dict(STYLE[k])
    if k not in used:
        used[k] = FALLBACK[len(used) % len(FALLBACK)]
    return dict(color=used[k], lw=1, label=k)


def bname(lo, hi):
    return f"k{lo}" if lo == hi else f"k{lo}-{hi}"


def main(run, out, only=None):
    os.makedirs(out, exist_ok=True)
    S = json.load(open(os.path.join(run, "score.json")))
    raw = np.load(os.path.join(run, "score_raw.npz"))
    names = S["channels"]
    L = S["leads"]
    days = np.arange(1, L + 1)
    arms = [k for k in STYLE if k in S["arms"]] + [k for k in S["arms"] if k not in STYLE and "~" not in k]
    if only:
        arms = [k for k in arms if k in only or k in ("persistence", "climatology")]
    summary = {"ics": S["ics"]}

    # ---- 1. lead lines
    panels = [("nrmse", None, "normalized RMSE (65 channels)"), ("rmse", "z500", "z500 RMSE (m$^2$/s$^2$)"),
              ("rmse", "t850", "t850 RMSE (K)"), ("rmse", "t2m", "t2m RMSE (K)"),
              ("rmse", "q700", "q700 RMSE (kg/kg)"), ("acc", "z500", "z500 ACC")]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for p, (key, ch, title) in zip(ax.flat, panels):
        for k in arms:
            v = np.array(S["arms"][k][key])
            p.plot(days, v if ch is None else v[:, names.index(ch)], **style(k))
        if key != "acc":  # a diverging rollout leaves the panel instead of flattening every other line
            v = np.array(S["arms"]["persistence"][key])
            p.set_ylim(0, 1.25 * float((v if ch is None else v[:, names.index(ch)]).max()))
        p.set_title(title)
        p.set_xlabel("lead (days)")
        p.grid(alpha=0.3)
    ax.flat[0].legend(fontsize=6.5, ncol=2, loc="lower right")
    fig.suptitle(f"Where to put the tendency core in KAI-alpha: ERA5 5.625 deg daily means, test 2017 "
                 f"({S['ics']} ICs), train 2012-15 (a line leaving the panel = the rollout diverges)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_kai_lead.png"), dpi=130)
    plt.close(fig)

    # ---- 2. where: each position vs KAI, relative RMSE change per channel x lead (paired bootstrap in score.json)
    rows = [i for i, n in enumerate(names) if n not in ("orog", "tisr")]
    P = S["paired_vs_kai"]
    show = [k for k in arms if k in P and k != "persistence"]
    fig, ax = plt.subplots(1, len(show), figsize=(2.3 * len(show) + 1.5, 12), sharey=True)
    ax = np.atleast_1d(ax)
    for p, k in zip(ax, show):
        r, lo, hi = (np.array(P[k][q]) for q in ("rel", "lo", "hi"))
        sig = (lo > 0) | (hi < 0)
        im = p.imshow(100 * r[:, rows].T, cmap="RdBu_r", vmin=-30, vmax=30, aspect="auto")
        yy, xx = np.where(~sig[:, rows].T)
        p.scatter(xx, yy, marker="x", s=5, c="k", lw=0.4)
        p.set_title(k, fontsize=10)
        p.set_xticks(range(0, L, 2))
        p.set_xticklabels(days[::2])
        p.set_xlabel("lead (days)")
        b, wr = ((r < 0) & sig)[:, rows], ((r > 0) & sig)[:, rows]
        summary[f"{k}/kai"] = {"better_cells": int(b.sum()), "worse_cells": int(wr.sum()),
                               "cells": int(len(rows) * L),
                               "nrmse_rel_pct_lead_1_3_5_10": [round(100 * P[k]["nrmse_rel"][q], 2) for q in (0, 2, 4, L - 1)],
                               "nrmse_ci_pct_lead_1_3_5_10": [[round(100 * P[k]["nrmse_lo"][q], 2),
                                                               round(100 * P[k]["nrmse_hi"][q], 2)] for q in (0, 2, 4, L - 1)]}
        # where it helps most / hurts most: mean change per variable group
        grp = {}
        for g in ("z", "q", "t", "u", "v"):
            ii = [i for i, n in enumerate(names) if n[0] == g and n[1:].isdigit()]
            grp[g] = [round(100 * float(r[q, ii].mean()), 1) for q in (0, 2, 4, L - 1)]
        ii = [names.index(n) for n in ("t2m", "sp", "tcwv", "msl", "skt")]
        grp["sfc"] = [round(100 * float(r[q, ii].mean()), 1) for q in (0, 2, 4, L - 1)]
        for lev, sl in (("50-200", slice(0, 3)), ("250-500", slice(3, 7)), ("600-1000", slice(7, 12))):
            ii = [g * 12 + j for g in range(5) for j in range(12)[sl]]
            grp[f"lev{lev}"] = [round(100 * float(r[q, ii].mean()), 1) for q in (0, 2, 4, L - 1)]
        summary[f"{k}/kai"]["rmse_change_pct_by_group_lead_1_3_5_10"] = grp
    ax[0].set_yticks(range(len(rows)))
    ax[0].set_yticklabels([names[i] for i in rows], fontsize=5.5)
    fig.colorbar(im, ax=ax, shrink=0.4, label="RMSE change vs KAI original, % (blue = better)")
    fig.suptitle("Where each position changes the error (x = 95 % paired bootstrap CI over ICs contains 0)")
    fig.savefig(os.path.join(out, "fig_kai_where.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- 3. scales: band error power relative to KAI (rows = variable, columns = band), prediction / truth power
    chans = ["z500", "t850", "u850", "q700"]
    comp = [k for k in arms if f"{k}__sp_err" in raw and k not in ("kai", "persistence", "climatology", "s1", "s3",
                                                                    "s4", "res", "inres", "stack")][:9]
    fig, ax = plt.subplots(len(chans), len(BANDS), figsize=(3.3 * len(BANDS), 2.9 * len(chans)), sharex=True)
    sc = {}
    for i, ch in enumerate(chans):
        c = names.index(ch)
        for j, (lo, hi) in enumerate(BANDS):
            base = raw["kai__sp_err"][:, c, lo:hi + 1].sum(-1)
            for k in comp:
                r = raw[f"{k}__sp_err"][:, c, lo:hi + 1].sum(-1) / base
                ax[i, j].plot(days, r, **style(k))
                sc[f"{k}/{ch}/{bname(lo, hi)}"] = [round(float(r[q]), 3) for q in (0, 2, 4, L - 1)]
            ax[i, j].axhline(1, color="k", lw=0.6)
            ax[i, j].set_yscale("log")
            ax[i, j].set_title(f"{ch}  {bname(lo, hi)}", fontsize=9)
            ax[i, j].grid(alpha=0.3)
    for p in ax[-1]:
        p.set_xlabel("lead (days)")
    ax[0, 0].legend(fontsize=6)
    fig.suptitle("Scales: error power / KAI original's, by zonal wavenumber band (below 1 = less error than KAI)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_kai_scale.png"), dpi=130)
    plt.close(fig)
    summary["band_err_power_ratio_vs_kai_lead_1_3_5_10"] = sc
    pw = {}
    for k in ["kai"] + comp:
        for ch in chans:
            c = names.index(ch)
            for lo, hi in BANDS[1:]:
                r = raw[f"{k}__sp_pred"][:, c, lo:hi + 1].sum(-1) / raw[f"{k}__sp_true"][:, c, lo:hi + 1].sum(-1)
                pw[f"{k}/{ch}/{bname(lo, hi)}"] = [round(float(r[q]), 3) for q in (0, 2, 4, L - 1)]
    summary["pred_over_truth_power_lead_1_3_5_10"] = pw

    # ---- 4. reliance on the core (tau = 0 at inference), prediction / truth power, the blend's weight
    inj = [k for k in S["arms"] if f"{k}~notau" in S["arms"]]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for k in inj:
        st = style(k)
        ax[0].plot(days, S["arms"][k]["nrmse"], **st)
        st.update(ls=":", label=None)
        ax[0].plot(days, S["arms"][f"{k}~notau"]["nrmse"], **st)
        a1, a0 = np.array(S["arms"][k]["nrmse"]), np.array(S["arms"][f"{k}~notau"]["nrmse"])
        summary[f"reliance/{k}"] = {"nrmse_notau_over_tau_lead_1_3_5_10": [round(float(a0[q] / a1[q]), 3) for q in (0, 2, 4, L - 1)]}
        if f"{k}~open" in S["arms"]:
            st.update(ls="--")
            ax[0].plot(days, S["arms"][f"{k}~open"]["nrmse"], **st)
    ax[0].plot(days, S["arms"]["kai"]["nrmse"], **style("kai"))
    ax[0].set_title("solid = trained coupling, dotted = tau zeroed, dashed = core open loop", fontsize=9)
    ax[0].set_ylabel("normalized RMSE")
    ax[0].set_ylim(0, 1.25 * max(S["arms"]["persistence"]["nrmse"]))
    ax[0].legend(fontsize=6.5)
    c = names.index("z500")
    for k in ["kai"] + comp:
        r = raw[f"{k}__sp_pred"][:, c, 10:21].sum(-1) / raw[f"{k}__sp_true"][:, c, 10:21].sum(-1)
        ax[1].plot(days, r, **style(k))
    ax[1].axhline(1, color="k", lw=0.6)
    ax[1].set_title("z500 prediction / truth power, k10-20 (below 1 = smoothed)")
    if "blend_kai_weight" in S:
        wk = np.array(S["blend_kai_weight"])
        for n, col in (("z500", "C0"), ("t850", "C1"), ("u850", "C2"), ("q700", "C3"), ("t2m", "C4"), ("msl", "C5")):
            ax[2].plot(days, wk[:, names.index(n)], color=col, label=n)
        scored = [i for i, n in enumerate(names) if n not in ("orog", "tisr")]
        ax[2].plot(days, wk[:, scored].mean(1), color="k", lw=2, label="mean of 65 channels")
        ax[2].set_ylim(0, 1)
        ax[2].set_title("blend: weight of KAI (1 - weight = core), fitted on 2016")
        ax[2].legend(fontsize=7)
        summary["blend_kai_weight_mean_lead_1_3_5_10"] = [round(float(wk[q, scored].mean()), 3) for q in (0, 2, 4, L - 1)]
    for p in ax:
        p.set_xlabel("lead (days)")
        p.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_kai_reliance.png"), dpi=130)
    plt.close(fig)

    # ---- 5. is the core's effect real? the core against its controls (paired, same ICs)
    PR = S.get("pairs", {})
    if PR:
        fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1, 1, 0.8]})

        def line(p, key, color, label, ls="-"):
            if key not in PR:
                return
            e = PR[key]
            p.plot(days, 100 * np.array(e["nrmse_rel"]), color=color, ls=ls, label=label)
            p.fill_between(days, 100 * np.array(e["nrmse_lo"]), 100 * np.array(e["nrmse_hi"]), color=color, alpha=0.15)
            summary[f"pair/{key}"] = {"nrmse_rel_pct_lead_1_3_5_10": [round(100 * e["nrmse_rel"][q], 2) for q in (0, 2, 4, L - 1)],
                                      "ci": [[round(100 * e["nrmse_lo"][q], 2), round(100 * e["nrmse_hi"][q], 2)] for q in (0, 2, 4, L - 1)],
                                      "ic_win_frac": [round(e["ic_win_frac"][q], 2) for q in (0, 2, 4, L - 1)],
                                      "better_cells": e["better_cells"], "worse_cells": e["worse_cells"]}

        line(ax[0], "out/res", "#d62728", "out vs res (same net, anchor = core instead of x)")
        line(ax[0], "out_s1/res_s1", "#d62728", "same, second training seed", "--")
        line(ax[0], "out/res_s1", "#ff9896", "seed 0 vs seed 1 (cross)", ":")
        line(ax[0], "out_s1/res", "#ff9896", None, ":")
        line(ax[0], "out/stack", "#8c564b", "out vs stack (first stage = a trained KAI, not the core)")
        line(ax[0], "stack/res", "0.4", "stack vs res (what any first stage adds)", "--")
        line(ax[0], "out/kai_long", "k", "out vs KAI original with 4x iterations", "-.")
        ax[0].set_title("The core at the output: normalized RMSE change, %", fontsize=10)
        line(ax[1], "blend/blend_kai_s1", "#9467bd", "blend(KAI, core) vs blend(KAI, second KAI seed)")
        line(ax[1], "blend/blend_res", "#c5b0d5", "blend(KAI, core) vs blend(KAI, res)")
        line(ax[1], "kai_s1/kai", "0.3", "seed noise: KAI seed 1 vs seed 0", ":")
        line(ax[1], "out_s1/out", "0.5", "seed noise: out", ":")
        line(ax[1], "res_s1/res", "0.7", "seed noise: res", ":")
        ax[1].set_title("The core in a blend, and the training-seed noise, %", fontsize=10)
        for p in ax[:2]:
            p.axhline(0, color="k", lw=0.6)
            p.set_xlabel("lead (days)")
            p.grid(alpha=0.3)
            p.legend(fontsize=6.5)
        if "out/res" in PR:
            r, lo, hi = (np.array(PR["out/res"][q]) for q in ("rel", "lo", "hi"))
            im = ax[2].imshow(100 * r[:, rows].T, cmap="RdBu_r", vmin=-20, vmax=20, aspect="auto")
            yy, xx = np.where(~((lo > 0) | (hi < 0))[:, rows].T)
            ax[2].scatter(xx, yy, marker="x", s=4, c="k", lw=0.4)
            ax[2].set_yticks(range(0, len(rows), 1))
            ax[2].set_yticklabels([names[i] for i in rows], fontsize=3.6)
            ax[2].set_xticks(range(0, L, 2))
            ax[2].set_xticklabels(days[::2])
            ax[2].set_xlabel("lead (days)")
            ax[2].set_title("out vs res per channel (RMSE change %)", fontsize=10)
            fig.colorbar(im, ax=ax[2], shrink=0.7)
            grp = {}
            for g in ("z", "q", "t", "u", "v"):
                ii = [i for i, n in enumerate(names) if n[0] == g and n[1:].isdigit()]
                grp[g] = [round(100 * float(r[q, ii].mean()), 1) for q in (0, 2, 4, L - 1)]
            ii = [names.index(n) for n in ("t2m", "sp", "tcwv", "msl", "skt")]
            grp["sfc"] = [round(100 * float(r[q, ii].mean()), 1) for q in (0, 2, 4, L - 1)]
            summary["pair/out/res"]["rmse_change_pct_by_group_lead_1_3_5_10"] = grp
        fig.suptitle("Is the tendency core's effect real? Paired comparisons on the same ICs (band = 95 % bootstrap CI over ICs)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_kai_effect.png"), dpi=140)
        plt.close(fig)

    # ---- 6. the systematic part: IC-mean of the global-mean error per lead (units of the channel's std)
    GB = S.get("global_bias", {})
    if GB:
        chs = ["z500", "z1000", "z50", "t850", "q700", "t2m"]
        fig, ax = plt.subplots(1, len(chs), figsize=(3.4 * len(chs), 3.8))
        for p, ch in zip(ax, chs):
            c = names.index(ch)
            for k in arms:
                if k in GB and k not in ("persistence", "climatology"):
                    p.plot(days, np.array(GB[k])[:, c], **style(k))
            p.axhline(0, color="k", lw=0.6)
            p.set_title(f"{ch}: global-mean bias (sigma)", fontsize=9)
            p.set_xlabel("lead (days)")
            p.grid(alpha=0.3)
        ax[0].legend(fontsize=5.5)
        fig.suptitle("Systematic drift: mean over ICs of the global-mean error")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_kai_bias.png"), dpi=130)
        plt.close(fig)
        summary["global_bias_sigma_lead_1_3_5_10"] = {
            k: {ch: [round(float(np.array(GB[k])[q, names.index(ch)]), 3) for q in (0, 2, 4, L - 1)] for ch in chs}
            for k in GB if "~" not in k}

    summary["nrmse_lead_1_3_5_10"] = {k: [round(S["arms"][k]["nrmse"][q], 4) for q in (0, 2, 4, L - 1)] for k in S["arms"]}
    for n in ("z500", "t850", "t2m", "u850", "q700"):
        c = names.index(n)
        summary[f"rmse_{n}_lead_1_3_5_10"] = {k: [float(f"{S['arms'][k]['rmse'][q][c]:.4g}") for q in (0, 2, 4, L - 1)]
                                              for k in S["arms"]}
    summary["train_seconds"] = {k: S["arms"][k].get("train_seconds") for k in S["arms"] if "train_seconds" in S["arms"][k]}
    summary["params"] = {k: S["arms"][k].get("params") for k in S["arms"] if "params" in S["arms"][k]}
    json.dump(summary, open(os.path.join(out, "where_summary.json"), "w"), indent=1)
    print(json.dumps(summary["nrmse_lead_1_3_5_10"], indent=0))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or None)  # optional: the tags to draw
