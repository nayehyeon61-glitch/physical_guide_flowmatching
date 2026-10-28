"""Step 18: is horizontal resolution the ceiling of the strong-form physics, and does real 2.8 deg data
add anything a downscaler could learn beyond interpolation?

Three sources of the same 2016 dates (every 8th day, 4 times a day, consecutive 6 h triples):
  coarse  : 5.625 deg KAI files (32 x 64), the grid of steps 11-17
  fine    : true WB2 2.8125 deg (64 x 128, conservative regridding of ERA5), written by wb2_to_kai.py
  interp  : the 5.625 deg state interpolated to the 2.8125 deg grid (FFT zero padding in longitude,
            cubic spline in latitude; orography from the fine file), i.e. what a downscaler sees as input
            and the "no new information" control of a super-resolution model

On ERA5 states x with the centred-difference tendency xdot the strong residual is R1 = xdot + A_h(x);
the rest-state residual R0 = A_h(x) is minus the discrete operator's own tendency and D = R1 - R0 = xdot
the tendency term. Per prognostic equation (u, v, T, q) and source:
  rms(D), rms(R0), rms(R1), corr(-R0, D) (how well the discrete operator predicts the true tendency),
  skill = 1 - <R1^2>/<D^2> (the operator's tendency as a predictor of xdot: 1 = exact, 0 = useless),
and the step 16 zonal spectrum |D_k|^2 / |R1_k|^2 (above 1 the operator sees the tangent at that scale).
  fine >> interp : the fine data carry information the coarse state does not have (super-resolution
                   can add it only by learning it from data)
  fine ~  interp : interpolation is enough, the gain is the finer stencil, not new information
  both ~ coarse  : horizontal resolution is not the physics bottleneck at these scales

    python scripts/step18_resolution_test.py --coarse /home/claude/era5/kai6h --fine /home/claude/era5_hres/kai6h \
        --stats runs/step16/stats --out docs/step18 --threads 2
"""

import argparse
import json
import os
import sys

import matplotlib
import netCDF4 as nc
import numpy as np
import torch
from scipy.interpolate import CubicSpline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pgcfm import contract  # noqa: E402
from pgcfm.coupling import LEVEL_DYN, PhysicsConfig, PhysicsGuide  # noqa: E402
from pgcfm.dyncore import StrongPhysics, finite_difference_tendency  # noqa: E402

DYN, OROG = contract.DYNAMIC_CHANNELS, contract.STATIC_CHANNELS
PROG = ("u", "v", "T", "q")
DIAG = ("mass", "hyd")
ALPHAS = (0.0, 0.25, 0.5, 1.0)  # one-step forecast x + alpha G(x) dt: alpha 0 = persistence
KEY_LEVEL = {"u": 500, "v": 500, "T": 850}
COLORS = {"coarse": "#4477AA", "fine": "#EE6677", "interp": "#228833"}
LABEL = {"coarse": "5.625° 원자료", "fine": "2.8° 실제 자료", "interp": "5.625° → 2.8° 보간"}
GROUPS = {"z": "z", "q": "q", "t": "t", "u": "u", "v": "v"}


def read_axes(path):
    with nc.Dataset(path) as d:
        return np.asarray(d["time"][:]), np.asarray(d["latitude"][:]), np.asarray(d["longitude"][:])


def consecutive_triples(times):
    """Start indices i such that times[i], times[i+1], times[i+2] are 6 h apart."""
    i = np.arange(len(times) - 2)
    ok = (times[i + 1] - times[i] == 6) & (times[i + 2] - times[i + 1] == 6)
    return i[ok]


def load_states(path, idx, mean, std):
    """Normalized (n, 67, H, W) float32 for the (sorted) time indices idx."""
    with nc.Dataset(path) as d:
        f = np.asarray(d["fields"][np.asarray(idx)], dtype=np.float32)
    f -= mean[:, None, None]
    f /= (std + contract.KAI_STD_EPS)[:, None, None]
    return torch.from_numpy(f)


def interpolate(x, lat_c, lat_f, n_lon_f):
    """(n, C, Hc, Wc) -> (n, C, Hf, Wf): band-limited (FFT zero padding) in longitude, cubic spline in
    latitude with extrapolation over the half row beyond the coarse end rows. Linear, so it commutes
    with the per-channel z-score."""
    n, C, Hc, Wc = x.shape
    X = torch.fft.rfft(x.double(), dim=-1) * (n_lon_f / Wc)
    X[..., Wc // 2] *= 0.5  # the coarse Nyquist bin becomes an interior bin of the fine spectrum
    Xf = torch.zeros(n, C, Hc, n_lon_f // 2 + 1, dtype=X.dtype)
    Xf[..., : Wc // 2 + 1] = X
    xf = torch.fft.irfft(Xf, n=n_lon_f, dim=-1).numpy()
    yf = CubicSpline(lat_c, xf, axis=-2, extrapolate=True)(lat_f)
    return torch.from_numpy(np.ascontiguousarray(yf, dtype=np.float32))


def physics_for(n_lat, n_lon, lat, mean, std):
    contract.register_latitudes(lat)
    guide = PhysicsGuide(n_lat, n_lon, mean, std, PhysicsConfig(dt_seconds=6 * 3600.0))
    return guide, StrongPhysics(guide, torch.ones(contract.N_DYNAMIC))


def evaluate(name, states, guide, phys, chunk=8):
    """states: (n_triples, 3, 67, H, W) normalized. Weighted sums over all states and the spectra."""
    g = guide.grid(states.device)
    lw = g.latitude_weight("area", pole_cap_deg=guide.cfg.pole_cap_deg).double()
    sums = {q: np.zeros(5) for q in PROG}  # <D^2>, <R0^2>, <R1^2>, <R0 D>, <1>
    diag = {q: np.zeros(2) for q in DIAG}
    step = {q: np.zeros((len(ALPHAS), 2)) for q in PROG[:3]}  # 6 h forecast error^2 (all levels, key level)
    step_pers = {q: np.zeros(2) for q in PROG[:3]}
    dt = phys.dt
    std_si = {q: phys.std[LEVEL_DYN[v]] for q, v in zip(PROG[:3], ("u", "v", "t"))}  # (L, 1, 1) double
    lv = {q: contract.LEVELS_HPA.index(KEY_LEVEL[q]) for q in KEY_LEVEL}
    spec = {q: [0.0, 0.0] for q in PROG[:3]}  # |D_k|^2, |R1_k|^2
    with torch.no_grad():
        for i in range(0, states.shape[0], chunk):
            s = states[i:i + chunk]
            x, orog = s[:, 1, DYN], s[:, 1, OROG]
            f = finite_difference_tendency(s[:, 0, DYN], s[:, 2, DYN])
            R1, m = phys.residuals(x, orog, f)
            R0, _ = phys.residuals(x, orog, torch.zeros_like(x))
            wt = (lw * m).double()
            W = wt.sum().item()
            dlt = (s[:, 2, DYN] - s[:, 1, DYN]).double()  # the forward 6 h increment (normalized)
            for q, v in zip(PROG[:3], ("u", "v", "t")):
                dq = dlt[:, LEVEL_DYN[v]].index_select(-2, g.lat_rows) * std_si[q]  # SI
                G = -R0[q]
                for j, al in enumerate(ALPHAS):
                    err = (dq - al * G * dt) ** 2 * wt
                    step[q][j] += np.array([err.sum().item(), err[:, lv[q]].sum().item()])
                step_pers[q] += np.array([(dq ** 2 * wt).sum().item(), (dq[:, lv[q]] ** 2 * wt[:, lv[q]]).sum().item()])
            for q in PROG:
                D, E, A = R1[q] - R0[q], R1[q], R0[q]
                sums[q] += np.array([(wt * D * D).sum().item(), (wt * A * A).sum().item(),
                                     (wt * E * E).sum().item(), (wt * A * D).sum().item(), W])
                if q in spec:
                    for j, r in enumerate((D, E)):
                        r = torch.where(wt > 0, r, torch.zeros_like(r))
                        pk = (torch.fft.rfft(r, dim=-1).abs() ** 2) * wt.mean(-1, keepdim=True)
                        spec[q][j] = spec[q][j] + pk.sum((0, 1, 2)).numpy() / W
            for q in DIAG:
                diag[q] += np.array([(wt * R1[q] ** 2).sum().item(), W])
    out = {"n_states": int(states.shape[0]), "equations": {}, "diagnostic": {}, "spectrum": {}}
    for q, (d2, a2, e2, ad, w) in sums.items():
        out["equations"][q] = {
            "rms_tendency": float(np.sqrt(d2 / w)), "rms_rest": float(np.sqrt(a2 / w)),
            "rms_residual": float(np.sqrt(e2 / w)),
            "corr_operator_tendency": float(-ad / np.sqrt(a2 * d2)),  # corr(-R0, D)
            "skill": float(1.0 - e2 / d2),
            "rest_over_residual": float(np.sqrt(a2 / e2)), "tendency_over_residual": float(np.sqrt(d2 / e2))}
    for q, (e2, w) in diag.items():
        out["diagnostic"][q] = {"rms_residual": float(np.sqrt(e2 / w))}
    out["one_step"] = {q: {str(al): {"all_levels": float(np.sqrt(step[q][j][0] / step_pers[q][0])),
                                     f"{KEY_LEVEL[q]}hPa": float(np.sqrt(step[q][j][1] / step_pers[q][1]))}
                           for j, al in enumerate(ALPHAS)} for q in step}
    for q, (sd, se) in spec.items():
        out["spectrum"][q] = {"ratio": (sd / se).tolist(), "cumulative": (np.cumsum(sd) / np.cumsum(se)).tolist(),
                              "tendency_power": sd.tolist(), "residual_power": se.tolist()}
    print(f"[{name}] n {out['n_states']}")
    for q, r in out["equations"].items():
        print(f"  {q:4s} rms D {r['rms_tendency']:.3e} R0 {r['rms_rest']:.3e} R1 {r['rms_residual']:.3e} "
              f"corr(-R0,D) {r['corr_operator_tendency']:+.3f} skill {r['skill']:+.3f}")
    for q, r in out["diagnostic"].items():
        print(f"  {q:4s} rms R {r['rms_residual']:.3e}")
    for q, r in out["one_step"].items():
        print(f"  6 h forecast x + alpha G dt, RMSE / persistence ({q}, all levels | {KEY_LEVEL[q]} hPa): "
              + " ".join(f"a={al} {r[str(al)]['all_levels']:.3f}|{r[str(al)][f'{KEY_LEVEL[q]}hPa']:.3f}" for al in ALPHAS))
    return out


def interpolation_gap(xi, xf):
    """Relative RMS of (interp - fine) per variable group and the fine-grid variance beyond k = 32."""
    gap = {}
    for gname, key in GROUPS.items():
        sl = contract.level_slice(key)
        a, b = xi[:, sl].double(), xf[:, sl].double()
        var = ((b - b.mean((-1, -2), keepdim=True)) ** 2).mean()
        gap[gname] = {"rel_rmse": float(torch.sqrt(((a - b) ** 2).mean() / var))}
        pk = torch.fft.rfft(b - b.mean(-1, keepdim=True), dim=-1).abs() ** 2
        tot, high = pk[..., 1:].sum().item(), pk[..., 33:].sum().item()
        gap[gname]["fine_power_beyond_k32"] = float(high / tot)
    return gap


def plot(res, out):
    for f_ in ("WenQuanYi Zen Hei", "Noto Sans CJK KR", "DejaVu Sans"):
        if any(f_ == x.name for x in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f_
            plt.rcParams["axes.unicode_minus"] = False
            break
    plt.rcParams["mathtext.fontset"] = "dejavusans"  # log tick labels need a minus glyph
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.6))
    ax = ax.ravel()
    for p, q in zip(ax[:3], ("u", "v", "T")):
        for src in ("coarse", "interp", "fine"):
            r = np.asarray(res[src]["spectrum"][q]["ratio"])
            kmax = res["grids"]["coarse"][1] // 2  # modes the coarse state carries (interp has none beyond)
            r = r[:kmax] if src != "fine" else r[:-1]  # drop the Nyquist bin
            k = np.arange(len(r))
            p.plot(k[1:], r[1:], color=COLORS[src], lw=2, marker="o", ms=3, label=LABEL[src])
        p.axhline(1.0, color="k", ls=":", lw=1)
        p.set_yscale("log")
        p.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))  # no minus glyph
        p.grid(alpha=0.3)
        p.set_xlabel("동서 파수 k")
        p.set_title(f"{q}: 경향항 / 잔차 파워 비 (1 위: 연산자가 접선을 봄)")
    ax[0].legend(fontsize=9)
    p = ax[3]
    eqs, srcs = list(PROG), ("coarse", "interp", "fine")
    wdt = 0.26
    for j, src in enumerate(srcs):
        vals = [res[src]["equations"][q]["corr_operator_tendency"] for q in eqs]
        bars = p.bar(np.arange(len(eqs)) + (j - 1) * wdt, vals, wdt, color=COLORS[src], label=LABEL[src])
        for b_, v in zip(bars, vals):
            p.text(b_.get_x() + b_.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    p.set_xticks(np.arange(len(eqs)))
    p.set_xticklabels(eqs)
    p.axhline(0, color="k", lw=0.8)
    p.set_ylim(-0.1, 1.0)
    p.set_ylabel("corr(연산자 경향 -R0, 실제 경향 D)")
    p.set_title("이산 물리 연산자가 실제 경향을 맞히는 정도")
    p.legend(fontsize=9)
    p.grid(alpha=0.3, axis="y")
    fig.suptitle("Step 18: 수평 해상도 상한 시험 (2016, 92 상태, 같은 날짜)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_resolution_test.png"), dpi=130)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--coarse", required=True, help="5.625 deg kai6h folder")
    p.add_argument("--fine", required=True, help="2.8125 deg kai6h folder (wb2_to_kai.py --grid 128x64)")
    p.add_argument("--stats", required=True, help="folder with global_mean.npy / global_std.npy")
    p.add_argument("--year", type=int, default=2016)
    p.add_argument("--max-triples", type=int, default=0, help="0 = all")
    p.add_argument("--out", default="docs/step18")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--plot-only", action="store_true", help="replot from <out>/resolution_test.json")
    a = p.parse_args(argv)
    if a.plot_only:
        with open(os.path.join(a.out, "resolution_test.json")) as f:
            plot(json.load(f), a.out)
        return
    torch.set_num_threads(a.threads)
    mean, std = np.load(os.path.join(a.stats, "global_mean.npy")), np.load(os.path.join(a.stats, "global_std.npy"))
    fine_path, coarse_path = os.path.join(a.fine, f"{a.year}.nc"), os.path.join(a.coarse, f"{a.year}.nc")
    tf, lat_f, lon_f = read_axes(fine_path)
    tc, lat_c, lon_c = read_axes(coarse_path)
    starts = consecutive_triples(tf)
    if a.max_triples:
        starts = starts[: a.max_triples]
    hours = np.stack([tf[starts + j] for j in range(3)], 1)  # (n, 3) hours since Jan 1
    ci = np.searchsorted(tc, hours)
    assert np.array_equal(tc[ci], hours), "coarse file lacks some of the fine file's times"
    print(f"triples {len(starts)} (hours {hours[0].tolist()} .. {hours[-1].tolist()})")
    # states (n, 3, 67, H, W)
    fi = np.stack([starts + j for j in range(3)], 1)
    uf, inv_f = np.unique(fi, return_inverse=True)
    xf = load_states(fine_path, uf, mean, std)[torch.from_numpy(inv_f.reshape(fi.shape))]
    uc, inv_c = np.unique(ci, return_inverse=True)
    xc = load_states(coarse_path, uc, mean, std)[torch.from_numpy(inv_c.reshape(ci.shape))]
    n = xf.shape[0]
    xi = interpolate(xc.reshape(n * 3, *xc.shape[2:]), lat_c, lat_f, len(lon_f)).reshape(n, 3, *xf.shape[2:])
    xi[:, :, OROG] = xf[:, :, OROG]  # the downscaler knows the true fine topography
    gap = interpolation_gap(xi[:, 1], xf[:, 1])
    print("interpolation gap (rel RMSE to the fine field / fine power beyond k=32): "
          + " ".join(f"{g} {v['rel_rmse']:.3f}/{v['fine_power_beyond_k32']:.3f}" for g, v in gap.items()))
    res = {"year": a.year, "n_triples": int(n), "interpolation_gap": gap,
           "grids": {"coarse": [len(lat_c), len(lon_c)], "fine": [len(lat_f), len(lon_f)]}}
    gc, pc = physics_for(len(lat_c), len(lon_c), lat_c, mean, std)
    res["coarse"] = evaluate("coarse", xc, gc, pc)
    gf, pf = physics_for(len(lat_f), len(lon_f), lat_f, mean, std)
    res["fine"] = evaluate("fine", xf, gf, pf)
    res["interp"] = evaluate("interp", xi, gf, pf)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "resolution_test.json"), "w") as f:
        json.dump(res, f, indent=1)
    plot(res, a.out)
    print("cumulative ratio k<=8: " + " | ".join(
        f"{s} " + " ".join(f"{q} {res[s]['spectrum'][q]['cumulative'][8]:.2f}" for q in ("u", "v", "T"))
        for s in ("coarse", "interp", "fine")))


if __name__ == "__main__":
    main()
