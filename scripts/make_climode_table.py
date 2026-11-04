"""Ours (37-year ngcm profile, FM blend_band) vs ClimODE public checkpoint only, 2020 hold-out, 5.625 deg, paired table."""
import json, sys
p = json.load(open(sys.argv[1])); out = sys.argv[2]
O, C, D = "seqfm_dct_pinn_blend_band", "climode", "seqfm_dct_pinn_blend_band-climode"
S = p["scores"]; L = [str(h) for h in p["leads"]]
V = [("z", "z500", "m²/s²", "{:.0f}", "{:.1f}"), ("t", "t850", "K", "{:.2f}", "{:.3f}"), ("t2m", "t2m", "K", "{:.2f}", "{:.3f}"),
     ("u10", "u10", "m/s", "{:.2f}", "{:.3f}"), ("v10", "v10", "m/s", "{:.2f}", "{:.3f}")]
H = "| 변수 | 모델 | " + " | ".join(f"{int(h)//24} d" for h in L) + " |\n|---|---|" + "---:|" * len(L) + "\n"
def sect(title, met, omet=None, lo=True):
    s = f"\n### {title}\n\n" + H
    for v, n, u, f, fd in V:
        o = [S[O][h][omet or met][v]["mean"] for h in L]; c = [S[C][h][met][v]["mean"] for h in L]
        w = lambda a, b: "**" + f4(a) + "**" if (a < b if lo else a > b) else f4(a)
        f4 = (lambda x: "{:.4f}".format(x)) if "crps" in met else ((lambda x: "{:.3f}".format(x)) if met == "acc" else (lambda x: f.format(x)))
        s += f"| {n} | ours | " + " | ".join(w(a, b) for a, b in zip(o, c)) + " |\n"
        s += f"| {n} | ClimODE | " + " | ".join(w(b, a) for a, b in zip(o, c)) + " |\n"
    return s
md = f"# 2020 hold-out: ours (FM blend_band) vs ClimODE\n\nICs {p['n_ic'][C]}, 12 UTC 2-day stride, 5.625 deg. IC 970 excluded for both (ClimODE NaN).\n"
md += sect("RMSE", "rmse") + sect("ACC", "acc", lo=False)
md += sect("CRPS (same Gaussian estimator, min-max units)", "crps_gauss") + sect("CRPS (ours 8-member ensemble vs ClimODE Gaussian)", "crps_gauss", "crps_ens")
md += "\n### Paired difference ours − ClimODE, mean [95% bootstrap CI over ICs]\n\n| 변수 | 지표 | " + " | ".join(f"{int(h)//24} d" for h in L) + " |\n|---|---|" + "---:|" * len(L) + "\n"
for v, n, u, f, fd in V:
    for met, ff in (("rmse", fd), ("acc", "{:.3f}")):
        c = []
        for h in L:
            d = S[D][h][met][v]; ci = d["ci95"]
            sig = ci[1] < 0 or ci[0] > 0
            t = ff.format(d["mean"]) + f" [{ff.format(ci[0])}, {ff.format(ci[1])}]"
            c.append(t if sig else t + " n.s.")
        md += f"| {n} | {met} | " + " | ".join(c) + " |\n"
md += "\n### Spread / RMSE (1 = calibrated)\n\n" + H
for v, n, *_ in V:
    for k, nm in ((O, "ours"), (C, "ClimODE")):
        md += f"| {n} | {nm} | " + " | ".join("{:.2f}".format(S[k][h]["spread"][v]["mean"] / S[k][h]["rmse"][v]["mean"]) for h in L) + " |\n"
open(out, "w").write(md); print(md)
