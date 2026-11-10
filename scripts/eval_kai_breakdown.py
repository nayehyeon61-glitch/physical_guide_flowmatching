"""Where does a forecast get better or worse? The breakdown of one score (scripts/run_kai_tendency.py --score)
for pairs A:B (A relative to B).

Sign convention (the owner's, 2026-11-10): every change is written as a GAIN of A over B, + = A better,
- = A worse, for every quantity (RMSE gain = the reduction of the RMSE in %, ACC gain = the increase).

    python scripts/eval_kai_breakdown.py runs/kaitend/score_Ld.json runs/kaitend/score_Ld_raw.npz outdir \
        [--stats runs/kaitend/stats.npz] "A:B[=title]" ...

Per pair one figure (outdir/breakdown_<A>__vs__<B>.png) and one table (printed; all numbers in
outdir/breakdown.json):

  by item      RMSE gain (%) of every variable x level x lead, paired bootstrap over the ICs
               (a dot = the 95 % interval contains zero)
  by term      the normalized MSE of every forecast split exactly into three terms, on anomalies from the
               climatology (s_f, s_o = pattern amplitude of forecast / truth, a = s_f / s_o, r = their correlation):
               mean_w(e^2) = (mean_w e)^2 + s_o^2 (1 - r^2) + s_o^2 (a - r)^2
               bias + pattern (information) term: what is left at the best amplitude, falls only with r
               + amplitude cost: what RMSE charges for an amplitude other than a = r (a full-strength field,
               a = 1, pays (1 - r)^2; a field damped to a = r pays nothing)
               The textbook split (s_f - s_o)^2 + 2 s_f s_o (1 - r) is printed too; its second term also falls
               when the field is only damped.
  by scale     error in zonal wavenumber bands k0 (zonal mean) / k1-3 / k4-9 / k10+
  by region    tropics (22.5S-22.5N: the 8 rows with |lat| < 20), north and south of it
  by season    DJF / MAM / JJA / SON of the IC (table)
  ACC by variable group, activity (pattern amplitude s_f / s_o of the anomalies; 1 = the truth's)

The term, region and activity parts need the metrics ``dec`` / ``mse_reg`` / ``act`` of the score; an older score
gives the other parts only.
"""

import datetime
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]  # fixed order, never cycled
# diverging: red (a loss, below 0) <-> blue (a gain, above 0), neutral gray midpoint
DIV = LinearSegmentedColormap.from_list("lossgain", ["#8a1f1e", "#e34948", "#f0efec", "#2a78d6", "#12407a"])
LEVELS = [50, 100, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
LEVVARS = ["z", "q", "t", "u", "v"]
SFC = ["t2m", "msl", "sp", "skt", "tcwv"]
BANDS = [("k0 (zonal mean)", 0, 0), ("k1-3 (planetary)", 1, 3), ("k4-9 (synoptic)", 4, 9), ("k10+ (small)", 10, 32)]
REGIONS = ["tropics (22.5S-22.5N)", "north of 22.5N", "south of 22.5S"]
TERMS = ["bias", "amplitude", "phase"]  # the textbook split
TERMS2 = ["bias", "pattern (information)", "amplitude cost"]
SEASONS = ["DJF", "MAM", "JJA", "SON"]


def boot_gain(ma, mb, n=400, seed=0):
    """RMSE gain of a over b, 1 - RMSE_a / RMSE_b, per (lead, channel), and is the 95 % interval clear of zero?"""
    rng = np.random.default_rng(seed)
    N = ma.shape[0]
    gain = 1 - np.sqrt(ma.mean(0)) / np.sqrt(mb.mean(0) + 1e-30)
    bs = []
    for _ in range(n // 50):
        idx = rng.integers(0, N, (50, N))
        bs.append(1 - np.sqrt(ma[idx].mean(1)) / np.sqrt(mb[idx].mean(1) + 1e-30))
    lo, hi = np.percentile(np.concatenate(bs), [2.5, 97.5], axis=0)
    return gain, (hi < 0) | (lo > 0)


def breakdown(S, raw, std, a, b):
    names = S["channels"]
    L = S["leads"]
    sc = np.array([i for i, n in enumerate(names) if n not in ("orog", "tisr")])
    ch = names.index
    groups = {v: [ch(f"{v}{p}") for p in LEVELS] for v in LEVVARS}
    groups["surface"] = [ch(n) for n in SFC]
    ma, mb = raw[f"{a}__mse"].astype(np.float64), raw[f"{b}__mse"].astype(np.float64)
    N = ma.shape[0]
    gain, sig = boot_gain(ma, mb)
    out = {"a": a, "b": b, "ics": int(N), "leads": L, "gain": gain, "sig": sig, "groups": groups, "names": names}

    def nrmse(m):  # the score's 65-channel number: mean over the channels of RMSE / std, per lead
        return (np.sqrt(m.mean(0))[:, sc] / std[sc]).mean(1)

    out["nrmse"] = {a: nrmse(ma), b: nrmse(mb)}
    out["gain_group"] = {g: gain[:, c].mean(1) for g, c in groups.items()}
    out["cells"] = {g: (int((sig & (gain > 0))[:, c].sum()), int((sig & (gain < 0))[:, c].sum()), len(c) * L)
                    for g, c in groups.items()}
    aa, ab = raw[f"{a}__acc"], raw[f"{b}__acc"]
    out["acc"] = {a: aa.mean(0)[:, sc].mean(1), b: ab.mean(0)[:, sc].mean(1)}
    out["dacc_group"] = {g: (aa.mean(0) - ab.mean(0))[:, c].mean(1) for g, c in groups.items()}
    # scale: the error in zonal wavenumber bands, as the score's nRMSE of the band-filtered error
    # (Parseval weights of the rfft bins: the bands add up to the MSE)
    if f"{a}__sp_err" in raw and f"{b}__sp_err" in raw:
        ea, eb = raw[f"{a}__sp_err"].astype(np.float64), raw[f"{b}__sp_err"].astype(np.float64)
        K = ea.shape[-1]
        wk = np.full(K, 2.0)
        wk[0] = wk[-1] = 1.0
        pw = lambda e, lo, hi: (e[:, sc, lo:hi + 1] * wk[lo:hi + 1]).sum(-1)  # noqa: E731  (L, 65)
        nb = lambda e, lo, hi: (np.sqrt(pw(e, lo, hi)) / std[sc]).mean(1)  # noqa: E731
        out["scale"] = {n: 1 - nb(ea, lo, hi) / nb(eb, lo, hi) for n, lo, hi in BANDS}
        out["scale_share"] = {n: (pw(eb, lo, hi) / pw(eb, 0, K - 1)).mean(1) for n, lo, hi in BANDS}
    # terms. RMSE_A - RMSE_B = (MSE_A - MSE_B) / (RMSE_A + RMSE_B) per channel, and the MSE is a sum of terms, so
    # the gain of the 65-channel nRMSE splits exactly into one contribution per term (in % of B's nRMSE)
    ra, rb = np.sqrt(ma.mean(0)), np.sqrt(mb.mean(0))  # (L, 67)

    def contrib(ta, tb, c):
        num = ((tb - ta)[:, c] / ((ra + rb)[:, c, None] * std[c, None])).mean(1)  # + = the term fell
        return 100 * num / (rb[:, c] / std[c]).mean(1)[:, None]

    def share(t, r, c):  # the part of the nRMSE that belongs to each term
        return (t[:, c] / (r[:, c, None] * std[c, None])).mean(1) / (r[:, c] / std[c]).mean(1)[:, None]

    if all(f"{k}__{m}" in raw for k in (a, b) for m in ("dec", "act")) and "climatology__dec" in raw:
        so2 = raw["climatology__dec"][..., 1].astype(np.float64)  # the truth's pattern variance (s_f = 0 there)

        def split(k):  # (L, 67, 3): bias, s_o^2 (1 - r^2), s_o^2 (a - r)^2, mean over the forecasts
            d, am = raw[f"{k}__dec"].astype(np.float64), raw[f"{k}__act"].astype(np.float64)
            cov = am * so2 - d[..., 2] / 2
            info = so2 - np.where(am > 1e-6, cov ** 2 / (am ** 2 * so2 + 1e-30), 0.0)
            return np.stack([d[..., 0], info, d[..., 1] + d[..., 2] - info], -1).mean(0)

        sa, sb = split(a), split(b)
        out["dterm2_pct"] = contrib(sa, sb, sc)
        out["term2_share"] = {a: share(sa, ra, sc), b: share(sb, rb, sc)}
        out["dterm2_group_pct"] = {g: contrib(sa, sb, np.array(c)) for g, c in groups.items()}
    if f"{a}__dec" in raw and f"{b}__dec" in raw:  # the textbook split
        ta, tb = raw[f"{a}__dec"].astype(np.float64).mean(0), raw[f"{b}__dec"].astype(np.float64).mean(0)
        out["dterm_pct"] = contrib(ta, tb, sc)
        out["term_share"] = {a: share(ta, ra, sc), b: share(tb, rb, sc)}
    if f"{a}__mse_reg" in raw and f"{b}__mse_reg" in raw:
        ra, rb = raw[f"{a}__mse_reg"].astype(np.float64), raw[f"{b}__mse_reg"].astype(np.float64)
        out["region"] = {n: 1 - nrmse(ra[..., i]) / nrmse(rb[..., i]) for i, n in enumerate(REGIONS)}
    if f"{a}__act" in raw and f"{b}__act" in raw:
        out["act"] = {a: raw[f"{a}__act"].mean(0)[:, sc].mean(1), b: raw[f"{b}__act"].mean(0)[:, sc].mean(1)}
        out["act_group"] = {k: {g: raw[f"{k}__act"].mean(0)[:, c].mean(1) for g, c in groups.items()} for k in (a, b)}
    # season of the IC (ICs every (365 - L) // N days from 1 January of the test year)
    stride = max(1, round((365 - L) / N))
    month = np.array([(datetime.date(2017, 1, 1) + datetime.timedelta(days=int(i * stride))).month for i in range(N)])
    seas = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
    out["season"] = {s: 1 - nrmse(ma[np.isin(month, m)]) / nrmse(mb[np.isin(month, m)]) for s, m in seas.items()}
    out["season_n"] = {s: int(np.isin(month, m).sum()) for s, m in seas.items()}
    return out


def figure(B, path, title):
    a, b, L = B["a"], B["b"], B["leads"]
    days = np.arange(1, L + 1)
    rel, sig, groups = 100 * B["gain"], B["sig"], B["groups"]
    allc = sum(groups.values(), [])
    vmax = float(np.clip(5 * np.ceil(np.percentile(np.abs(rel[:, allc]), 98) / 5), 5, 60))
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK})
    lines = [k for k in ("dterm2_pct", "scale", "region", "dacc_group", "act") if k in B]
    fig = plt.figure(figsize=(19, 10.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 60, height_ratios=[1.0, 0.9], hspace=0.42, wspace=0.0, left=0.04, right=0.955,
                          top=0.9, bottom=0.07)
    im = None
    for i, g in enumerate(LEVVARS + ["surface"]):  # row 1: the scorecard
        p = fig.add_subplot(gs[0, i * 9 + (1 if i else 0):i * 9 + 8 + (1 if i else 0)] if i < 5 else gs[0, 47:55])
        c = groups[g]
        M, Sg = rel[:, c].T, sig[:, c].T  # (rows, leads)
        im = p.pcolormesh(np.arange(L + 1) + 0.5, np.arange(len(c) + 1), M, cmap=DIV, vmin=-vmax, vmax=vmax,
                          edgecolors=SURFACE, linewidth=1.2)
        yy, xx = np.nonzero(~Sg)
        p.scatter(xx + 1, yy + 0.5, s=5, color=MUTED, linewidths=0)
        p.set_ylim(len(c), 0)
        p.set_yticks(np.arange(len(c)) + 0.5)
        p.set_yticklabels([str(q) for q in LEVELS] if g != "surface" else SFC)
        if i not in (0, 5):
            p.set_yticklabels([])
        if i == 0:
            p.set_ylabel("level (hPa)")
        p.set_xticks(days)
        p.set_xlabel("lead (days)")
        nb, nw, nt = B["cells"][g]
        p.set_title(f"{g}   {nb} better / {nw} worse of {nt}", fontsize=10, loc="left")
        p.set_facecolor(SURFACE)
        p.tick_params(length=0)
        for s in p.spines.values():
            s.set_visible(False)
    cax = fig.add_subplot(gs[0, 58:59])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("RMSE gain of A over B (%):  + (blue) = A better,  - (red) = A worse", color=MUTED)
    cb.outline.set_visible(False)

    def base(p, ttl, zero=0.0):
        p.set_facecolor(SURFACE)
        p.set_title(ttl, fontsize=10, loc="left")
        p.set_xlabel("lead (days)")
        p.set_xticks(days)
        p.grid(color=GRID, lw=0.8)
        if zero is not None:
            p.axhline(zero, color=INK, lw=0.8)
        for s in ("top", "right"):
            p.spines[s].set_visible(False)

    n = len(lines)
    span = 60 // max(n, 1)
    for j, k in enumerate(lines):  # row 2: the same change split in other ways
        p = fig.add_subplot(gs[1, j * span + 2:(j + 1) * span - 2])
        if k == "dterm2_pct":
            base(p, "by error term: contribution to the nRMSE gain\n(%; the bars add up to the total; + = A better)")
            wdt = 0.26
            for t, name in enumerate(TERMS2):
                p.bar(days + (t - 1) * wdt, B[k][:, t], wdt * 0.9, color=CAT[t], label=name)
            p.plot(days, B[k].sum(1), color=INK, lw=1.6, marker="o", ms=4, label="total")
            p.legend(frameon=False, fontsize=8.5)
        elif k == "scale":
            base(p, "by scale: nRMSE gain per wavenumber band\n(%; + = A better; legend: share of B's MSE)")
            for t, (name, _, _) in enumerate(BANDS):
                sh = B["scale_share"][name]
                p.plot(days, 100 * B[k][name], color=CAT[t], lw=2,
                       label=f"{name}: {100 * sh[0]:.0f} / {100 * sh[-1]:.0f} %")
            p.legend(frameon=False, fontsize=9)
        elif k == "region":
            base(p, "by region: 65-channel nRMSE gain\n(%; + = A better)")
            for t, name in enumerate(REGIONS):
                p.plot(days, 100 * B[k][name], color=CAT[t], lw=2, label=name)
            p.legend(frameon=False, fontsize=9)
        elif k == "dacc_group":
            base(p, "anomaly correlation gain: A - B by variable\n(+ = A better)")
            for t, g in enumerate(LEVVARS + ["surface"]):
                p.plot(days, B[k][g], color=CAT[t], lw=2, label=g)
            p.legend(frameon=False, fontsize=9, ncol=2)
        elif k == "act":
            base(p, "activity: pattern amplitude of the forecast\nanomaly / truth's (1 = full strength)", zero=1.0)
            p.plot(days, B[k][a], color=CAT[0], lw=2.2, label="A")
            p.plot(days, B[k][b], color=MUTED, lw=2.2, label="B")
            p.legend(frameon=False, fontsize=9)
    na, nb_ = B["nrmse"][a], B["nrmse"][b]
    q = [0, 2, 4, L - 1]
    fig.suptitle(f"{title}\nA = {a},  B = {b};  65-channel normalized RMSE at day "
                 + "/".join(str(x + 1) for x in q) + ":  A " + "/".join(f"{na[x]:.3f}" for x in q)
                 + ",  B " + "/".join(f"{nb_[x]:.3f}" for x in q)
                 + f"  (gain {'/'.join(f'{100 * (1 - na[x] / nb_[x]):+.1f}' for x in q)} %; + = A better);  {B['ics']} ICs, test 2017",
                 fontsize=11.5, x=0.04, ha="left", y=0.985)
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def table(B):
    a, b, L = B["a"], B["b"], B["leads"]
    q = [x for x in (0, 2, 4, 6, L - 1) if x < L]
    hd = "/".join(str(x + 1) for x in q)
    f = lambda v, s=100, d=1: "/".join(f"{s * v[x]:+.{d}f}" for x in q)  # noqa: E731
    print(f"\n=== {a} against {b}  (day {hd}); every change is a gain of A over B: + = A better, - = A worse ===")
    print(f"  65-ch nRMSE   A {'/'.join(f'{B['nrmse'][a][x]:.3f}' for x in q)}  B {'/'.join(f'{B['nrmse'][b][x]:.3f}' for x in q)}"
          f"  gain % {f(1 - B['nrmse'][a] / B['nrmse'][b])}")
    print(f"  65-ch ACC     A {'/'.join(f'{B['acc'][a][x]:.3f}' for x in q)}  B {'/'.join(f'{B['acc'][b][x]:.3f}' for x in q)}")
    print("  by variable: RMSE gain %  |  ACC gain  |  cells better/worse (95 %)")
    for g in B["groups"]:
        nb, nw, nt = B["cells"][g]
        print(f"    {g:8s} {f(B['gain_group'][g]):38s} | {f(B['dacc_group'][g], 1, 3):38s} | {nb}/{nw} of {nt}")
    if "dterm2_pct" in B:
        print("  by term: contribution to the gain of the 65-channel nRMSE, % (sum = total)")
        for t, name in enumerate(TERMS2):
            print(f"    {name:22s} {f(B['dterm2_pct'][:, t], 1)}   part of the nRMSE of A: "
                  + "/".join(f"{100 * B['term2_share'][a][x, t]:.0f}" for x in q) + " %, of B: "
                  + "/".join(f"{100 * B['term2_share'][b][x, t]:.0f}" for x in q) + " %")
        print(f"    {'total':22s} {f(B['dterm2_pct'].sum(1), 1)}")
        for g in B["groups"]:
            print(f"      {g:8s} " + "  ".join(f"{name.split(' ')[0]} {f(B['dterm2_group_pct'][g][:, t], 1)}"
                                              for t, name in enumerate(TERMS2)))
    if "dterm_pct" in B:
        print("  textbook split (s_f - s_o)^2 + 2 s_f s_o (1 - r): contribution to the nRMSE gain, %")
        for t, name in enumerate(TERMS):
            print(f"    {name:22s} {f(B['dterm_pct'][:, t], 1)}")
    if "scale" in B:
        print("  by scale: nRMSE gain %  (share of B's error)")
        for name, _, _ in BANDS:
            print(f"    {name:18s} {f(B['scale'][name]):38s} ({'/'.join(f'{100 * B['scale_share'][name][x]:.0f}' for x in q)} %)")
    if "region" in B:
        print("  by region: nRMSE gain %")
        for name in REGIONS:
            print(f"    {name:18s} {f(B['region'][name])}")
    if "act" in B:
        print(f"  activity      A {'/'.join(f'{B['act'][a][x]:.2f}' for x in q)}  B {'/'.join(f'{B['act'][b][x]:.2f}' for x in q)}")
    print("  by season of the IC: nRMSE gain %")
    for s in SEASONS:
        print(f"    {s} ({B['season_n'][s]:2d} ICs)  {f(B['season'][s])}")


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return np.round(x.astype(np.float64), 5).tolist() if x.dtype != bool else x.astype(int).tolist()
    return x


def main(argv):
    stats = "runs/kaitend/stats.npz"
    if "--stats" in argv:
        i = argv.index("--stats")
        stats = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    S = json.load(open(argv[0]))
    raw = np.load(argv[1])
    out = argv[2]
    os.makedirs(out, exist_ok=True)
    std = np.load(stats)["std"].astype(np.float64) + 1e-8
    allb = {}
    for spec in argv[3:]:
        pair, title = (spec.split("=", 1) + [None])[:2]
        a, b = pair.split(":")
        if f"{a}__mse" not in raw or f"{b}__mse" not in raw:
            print("skipped (not in this score):", pair)
            continue
        B = breakdown(S, raw, std, a, b)
        path = os.path.join(out, f"breakdown_{a}__vs__{b}.png".replace(">", "-").replace("@", "-").replace("~", "-"))
        figure(B, path, title or f"Where A gets better or worse than B")
        table(B)
        print("  wrote", path)
        allb[pair] = jsonable({k: v for k, v in B.items() if k not in ("names",)})
    with open(os.path.join(out, "breakdown.json"), "w") as fh:
        json.dump(allb, fh)


if __name__ == "__main__":
    main(sys.argv[1:])
