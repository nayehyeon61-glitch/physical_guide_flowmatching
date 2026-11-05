"""Table + 2x2 lead-time figure: our blend_band (37 y, GPU, 2020 hold-out) vs recent models on the same ICs.

Our numbers and ClimODE / persistence come from docs/results_2026_11 (per-lead means over the 179 ICs; per-IC
values live only on the owner's machine). Every other model is scored here with scripts/compare_recent/common.py,
which reproduces that pipeline exactly (persistence RMSE / ACC / CRPS identical to 3 decimals).
CRPS = ensemble CRPS, unweighted, ClimODE min-max units x (max - min) -> physical units (m2/s2, K).

    python scripts/compare_recent/report.py --runs runs/step27 --out docs/step27
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import common as C

NAMES = {"ours": "ours blend_band (8 m)", "climode": "ClimODE", "persistence": "persistence",
         "keisler": "Keisler GNN 2022", "sphericalcnn": "Spherical CNN 2023 (00 UTC)", "pangu": "Pangu-Weather",
         "graphcast": "GraphCast", "fuxi": "FuXi", "hres": "IFS HRES", "neuralgcm_det": "NeuralGCM 0.7 det",
         "neuralgcm_ens_m8": "NeuralGCM ENS (8 m)", "ifs_ens_m8": "IFS ENS (8 m)", "gencast_m8": "GenCast (8 m)",
         "andrae_iterative-24h_m8": "Andrae 2025 AR-24h (8 m)", "andrae_iterative-24h": "Andrae 2025 AR-24h (50 m)",
         "andrae_continuous-24+6h_m8": "Andrae 2025 ARCI (8 m)"}
GROUP = {"ours": "ours", "climode": "5.625", "persistence": "ref"}


def load(runs, extra=()):
    d = json.load(open(C.PAIRED))["scores"]
    mx, mn = C.minmax()
    out = {}
    src_list = [(d, "ours", "seqfm_dct_pinn_blend_band"), (d, "climode", "climode"), (d, "persistence", "persistence")]
    for e in extra:  # JSON:ARM:LABEL, a compare_step13_climode.py output on the 2020 ICs (e.g. blend_band+)
        path, arm, label = e.split(":", 2)
        j = json.load(open(path))
        assert j["ics"] == C.ics_2020() and j["years"] == [2020], f"{path}: not the 179 2020 ICs"
        NAMES[label] = label
        src_list.append((j["scores"], label, arm))
    for d, key, arm in src_list:
        s = d[arm]
        m = {}
        for met, src in (("rmse", "rmse"), ("acc", "acc"), ("spread", "spread"),
                         ("crps", "crps_ens" if "crps_ens" in s["24"] else "crps_gauss")):
            if src not in s["24"]:
                continue
            m[met] = np.array([[s[str(h)][src][v]["mean"] if s[str(h)][src][v]["mean"] is not None else np.nan
                                for v in C.VARS] for h in C.LEADS], float)
        m["crps"] = m["crps"] * (mx - mn)
        m.setdefault("spread", np.full_like(m["rmse"], np.nan))
        m["n"] = 179
        out[key] = m
    for f in sorted(glob.glob(os.path.join(runs, "*", "*.npz"))):
        k = os.path.basename(f)[:-4]
        if k.endswith("_ens") or k not in NAMES:
            continue
        z = np.load(f)
        with np.errstate(all="ignore"):
            out[k] = {"rmse": np.nanmean(z["rmse"], 0), "acc": np.nanmean(z["acc"], 0),
                      "spread": np.nanmean(z["spread"], 0), "crps": np.nanmean(z["crps_ens"], 0) * (mx - mn),
                      "n": int((~np.isnan(z["rmse"][:, 0, 0])).sum())}
    return out


def table(out):
    rows = ["| model | ICs | z500 RMSE 24/72/144 h | t850 RMSE 24/72/144 h | t2m RMSE 72 h | z500 CRPS 24/72/144 h | z500 spread/RMSE 72 h |",
            "|---|---:|---|---|---:|---|---:|"]
    for k, m in out.items():
        r = m["rmse"]; c = m["crps"]; s = m["spread"]
        f = lambda x, fmt: "–" if np.isnan(x) else format(x, fmt)  # noqa: E731
        ssr = s[2, 0] / r[2, 0] if not np.isnan(s[2, 0]) else np.nan
        rows.append(f"| {NAMES[k]} | {m['n']} | {f(r[0,0],'.0f')} / {f(r[2,0],'.0f')} / {f(r[5,0],'.0f')} | "
                    f"{f(r[0,1],'.2f')} / {f(r[2,1],'.2f')} / {f(r[5,1],'.2f')} | {f(r[2,2],'.2f')} | "
                    f"{f(c[0,0],'.0f')} / {f(c[2,0],'.0f')} / {f(c[5,0],'.0f')} | {f(ssr,'.2f')} |")
    return "\n".join(rows)


def figure(out, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    days = np.array(C.LEADS) / 24
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    panels = [("rmse", 0, "z500 RMSE (m$^2$/s$^2$)"), ("rmse", 1, "t850 RMSE (K)"),
              ("crps", 0, "z500 CRPS (m$^2$/s$^2$, unweighted)"), ("ssr", 0, "z500 spread / RMSE")]
    style = {"ours": dict(color="k", lw=2.6), "persistence": dict(color="0.5", ls=":", lw=1.5),
             "climode": dict(color="tab:red", lw=1.8)}
    cyc = iter(["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive",
                "tab:cyan", "navy", "goldenrod", "teal", "darkviolet", "lime", "slateblue"] * 2)
    for k in out:
        if k not in style:
            style[k] = dict(color=next(cyc), lw=1.2, ls="-" if k.startswith(("andrae", "keisler", "spherical")) else "--")
    for a, (met, v, lab) in zip(ax.flat, panels):
        for k, m in out.items():
            y = m["spread"][:, v] / m["rmse"][:, v] if met == "ssr" else m[met][:, v]
            if np.all(np.isnan(y)) or (met == "ssr" and k == "persistence"):
                continue
            a.plot(days, y, marker="o", ms=3, label=NAMES[k], **style[k])
        a.set_xlabel("lead (days)"); a.set_title(lab); a.grid(alpha=.3)
        if met == "ssr":
            a.axhline(1, color="0.3", lw=.8)
    ax[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("2020 hold-out, 179 ICs (12 UTC every 2 days), WB2 ERA5 64x32 truth; ensembles scored with 8 members")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.join(C.R, "docs", "step27"), help="folder with <sub>/<model>.npz per-IC scores")
    p.add_argument("--out", default=os.path.join(C.R, "docs", "step27"))
    p.add_argument("--extra", nargs="*", default=[], help="JSON:ARM:LABEL, e.g. "
                   "runs/blend_plus/y2020/climode_paired_plus_2020.json:seqfm_dct_pinn_blend_plus:'ours blend_band+'")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = load(a.runs, a.extra)
    t = table(out)
    print(t)
    open(os.path.join(a.out, "table_2020.md"), "w").write(t + "\n")
    json.dump({k: {m: (v.tolist() if isinstance(v, np.ndarray) else v) for m, v in d.items()} for k, d in out.items()},
              open(os.path.join(a.out, "scores_2020.json"), "w"), indent=1)
    figure(out, os.path.join(a.out, "fig_recent_2020.png"))


if __name__ == "__main__":
    main()
