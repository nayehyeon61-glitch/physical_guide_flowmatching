"""Step 3 small experiments: primitive-equation residuals on analytic fields.

Prints the numbers quoted in docs/step3_pinn_residual.md and writes them to
experiments/step3_residual.json. CPU, about a minute.

    python -m scripts.step3_residual_experiment [--pinn-steps 1500]
"""

import argparse
import json
import math
import os
import time

import torch

from pgcfm import contract
from pgcfm.physics import analytic, grid as G, pinn, primitive

SC = primitive.ResidualScales()
EQS = ("u", "v", "T", "mass", "hyd", "q")


def rel_max(R):
    return {k: float((v.abs().max() / SC.of(k)).item()) for k, v in R.items()}


def bands(R, g, edges=(0, 60, 80, 87, 90)):
    """max over rows of the (level, lon) RMS residual, per |lat| band, in units of b_j."""
    lat = (g.lat[g.lat_rows] * 180 / math.pi).abs()
    out = {}
    for k, v in R.items():
        rms = G.residual_rms_by_lat(v, g) / SC.of(k)
        out[k] = {f"{lo}-{hi}": float(rms[(lat >= lo) & (lat < hi)].max())
                  for lo, hi in zip(edges[:-1], edges[1:]) if ((lat >= lo) & (lat < hi)).any()}
    return out


def e1_autograd():
    torch.manual_seed(0)
    X = pinn.sample_collocation(4000, pole_cap_deg=89.0)
    out = {}
    for name, kw in [("alpha=0", dict(alpha=0.0)), ("alpha=pi/4", dict(alpha=math.pi / 4)),
                     ("alpha=pi/2", dict(alpha=math.pi / 2)),
                     ("negative: Phi balance x0.9", dict(alpha=math.pi / 4, phi_scale=0.9)),
                     ("rest state U=0 (trivial solution)", dict(U0=0.0, U1=0.0))]:
        R = pinn.autograd_residuals(lambda *c: analytic.solid_body(*c, **kw), *X, create_graph=False)
        out[name] = rel_max(R)
    return out


def e2_grid_discretization():
    out = {}
    for lon, pf, vert in (("centered", None, "p"), ("centered", None, "logp"),
                          ("spectral", None, "logp"), ("spectral", 60.0, "logp")):
        g = G.SphereGrid(lon_method=lon, vertical=vert, polar_filter_deg=pf)
        for alpha, an in ((0.0, "0"), (math.pi / 2, "pi/2")):
            F, T = analytic.on_grid(g, alpha=alpha, with_tendencies=True)
            R = G.strong_residuals(F, g, tendencies=T)
            out[f"lon={lon},filter={pf},vert={vert},alpha={an}"] = bands(R, g)
    g = G.SphereGrid()
    F, T = analytic.on_grid(g, alpha=math.pi / 2, with_tendencies=True)
    R = G.strong_residuals(F, g, tendencies=T)
    hyd = R["hyd"].abs().amax(dim=(-1, -2))
    out["hyd_max_by_level_K"] = {str(l): float(h) for l, h in zip(contract.LEVELS_HPA, hyd)}
    return out


def e3_omega_diagnosis():
    g = G.SphereGrid()
    F, T = analytic.on_grid(g, alpha=math.pi / 2, with_tendencies=True)
    F = {k: v for k, v in F.items() if k != "omega"}  # as on ERA5: no omega
    om = G.diagnose_omega(F, g)
    rows = om.index_select(-2, g.lat_rows)
    R = G.strong_residuals(F, g, tendencies=T)
    return {"omega_diag_max_Pa_s": float(rows.abs().max()), "residuals": rel_max(R)}


def e4_pole_conditioning():
    out = {}
    for dtype in (torch.float32, torch.float64):
        row = {}
        for lat in (80.0, 89.0, 89.9, 89.99, 89.999):
            n = 64
            lam = torch.linspace(0, 2 * math.pi, n + 1, dtype=dtype)[:-1]
            phi = torch.full((n,), math.radians(lat), dtype=dtype)
            p = torch.full((n,), 5.0e4, dtype=dtype)
            s = torch.zeros(n, dtype=dtype)
            R = pinn.autograd_residuals(lambda *c: analytic.solid_body(*c, alpha=math.pi / 2), lam, phi, p, s,
                                        create_graph=False)
            row[str(lat)] = {k: rel_max(R)[k] for k in ("u", "v", "mass")}
        out[str(dtype).replace("torch.", "")] = row
    return out


NOISE = dict(u=0.5, v=0.5, T=0.3, Phi=30.0)


def noisy(F, sigma=NOISE, seed=0):
    gen = torch.Generator().manual_seed(seed)
    out = dict(F)
    for k, sd in sigma.items():
        out[k] = F[k] + sd * torch.randn(F[k].shape, generator=gen, dtype=F[k].dtype)
    return out


def e5_noise_amplification():
    """Generated endpoints carry grid-scale error: white noise with NOISE std devs (SI)."""
    out = {"sigma": NOISE}
    sel = [0.0, 45.0, 75.0, 84.0, 87.0, 88.5]
    for name, kw in (("centered, no filter", dict(lon_method="centered", polar_filter_deg=None)),
                     ("spectral + polar filter 60", dict())):
        g = G.SphereGrid(**kw)
        F, T = analytic.on_grid(g, alpha=0.0, with_tendencies=True)
        R = G.strong_residuals(noisy(F), g, tendencies=T)
        lat = (g.lat[g.lat_rows] * 180 / math.pi)
        rms = {k: G.residual_rms_by_lat(v, g) / SC.of(k) for k, v in R.items()}
        res = {"rms_by_lat": {k: {str(L): float(rms[k][(lat - L).abs().argmin()]) for L in sel}
                              for k in ("u", "v", "T", "mass")}}
        # loss share of a latitude band divided by its area share (1 = fair)
        area = torch.cos(g.lat[g.lat_rows])
        alat = lat.abs()
        for mode in ("none", "area", "metric"):
            w = g.latitude_weight(mode).squeeze(-1)
            r = {}
            for k in ("u", "v", "T", "mass", "hyd"):
                r2 = (R[k] ** 2).mean(dim=(0, 2)) * w
                r[k] = {b: float((r2[m].sum() / r2.sum()) / (area[m].sum() / area.sum()))
                        for b, m in (("60-80", (alat >= 60) & (alat < 80)), ("80-90", alat >= 80))}
            res[f"share_over_area[{mode}]"] = r
        out[name] = res
    return out


def e6_time_integrated():
    """Eq. (18) with the linear bridge (h_psi = 0) between exact endpoints."""
    out = {}
    g = G.SphereGrid()
    dt = contract.CONST.dt_seconds
    for alpha, an in ((0.0, "0"), (math.pi / 2, "pi/2")):
        F0 = analytic.on_grid(g, s=0.0, alpha=alpha)
        F1 = analytic.on_grid(g, s=dt, alpha=alpha)
        for n_sub in (2, 8):
            R = G.time_integrated_residuals(F0, F1, g, dt=dt, n_sub=n_sub)
            out[f"alpha={an},m=1,n_sub={n_sub}"] = {k: rel_max(R)[k] for k in ("u", "v", "T", "q", "hyd")}
    for m in (1, 4, 8, 16):
        for frac in (1.0, 0.25):
            kw = dict(alpha=0.0, q_wavenumber=m)
            F0 = analytic.on_grid(g, s=0.0, **kw)
            F1, T1 = analytic.on_grid(g, s=frac * dt, with_tendencies=True, **kw)
            R = G.time_integrated_residuals(F0, F1, g, dt=frac * dt, n_sub=8)
            scale = float(T1["q"].abs().max())  # max |d_s q|: residual relative to the true tendency
            out[f"tracer m={m},dt={frac}day"] = {
                "q_resid_over_b": rel_max(R)["q"],
                "q_resid_over_max_tendency": float(R["q"].abs().max()) / scale}
    return out


def e7_unit_scaling():
    g = G.SphereGrid()
    F, T = analytic.on_grid(g, alpha=0.0, with_tendencies=True)
    R = G.strong_residuals(noisy(F, seed=1), g, tendencies=T)
    raw = {}
    for k, v in R.items():
        if k == "hyd":
            v = v * contract.CONST.r_dry / g.p3()  # back from K to m2 s-2 Pa-1
        raw[k] = float((v ** 2).mean())
    scaled = {k: float(((v / SC.of(k)) ** 2).mean()) for k, v in R.items()}
    est = {k: float(v.pow(2).mean().sqrt()) for k, v in R.items()}
    return {"raw_SI_mse": raw, "scaled_mse_default_b": scaled, "rms_SI(=data-driven b_j)": est}


def e8_pinn_fit(steps=1500, seed=0):
    """Fit the coordinate PINN to sparse observations, with and without the physics loss."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    kw = dict(alpha=math.pi / 2)
    fn = lambda *c: analytic.solid_body(*c, **kw)
    Xu = pinn.sample_collocation(400, generator=gen, s_range=(0.0, 0.0))
    tgt = {k: v.detach() for k, v in fn(*Xu).items() if k in ("u", "v", "T", "Phi")}
    Xf = pinn.sample_collocation(1500, generator=gen, s_range=(0.0, 0.0))
    Xtest = pinn.sample_collocation(2000, generator=gen, s_range=(0.0, 0.0))
    truth = fn(*Xtest)
    lb, ub = [math.log(5e3), -1.0], [math.log(1e5), 1.0]
    out = {}
    phys = {"u": 1.0, "v": 1.0, "T": 1.0, "mass": 1.0, "hyd": 1.0, "q": 0.0}
    for name, w in (("data only", {k: 0.0 for k in EQS}),
                    ("data + physics, gamma=1", phys),
                    ("data + physics, gamma=1e-4", {k: 1e-4 * v for k, v in phys.items()})):
        torch.manual_seed(seed)
        net = pinn.PhysicsInformedNN([5, 64, 64, 64, 7], lb, ub, dtype=torch.float64)
        t0 = time.time()
        hist = net.train_model(Xu, tgt, Xf, n_adam=steps, lr=2e-3, weights=w)
        F, R = net.predict(*Xtest)
        out[name] = {
            "seconds": round(time.time() - t0, 1),
            "final_data_loss": hist[-1]["data"],
            "test_field_rmse": {k: float(((F[k] - truth[k]) ** 2).mean().sqrt()) for k in ("u", "v", "T", "Phi")},
            "test_residual_rms_over_b": {k: float(((R[k] / SC.of(k)) ** 2).mean().sqrt()) for k in ("u", "v", "mass", "hyd")},
        }
    return out


def e9_gradient_flow():
    """d L_phys / d endpoint must be finite and non-zero (paper section 5.3)."""
    g = G.SphereGrid()
    F0 = analytic.on_grid(g, s=0.0, alpha=0.0)
    F1 = {k: v.clone().requires_grad_(k in ("u", "v", "T", "Phi")) for k, v in
          analytic.on_grid(g, s=contract.CONST.dt_seconds, alpha=0.0).items() if k != "omega"}
    F0 = {k: v for k, v in F0.items() if k != "omega"}
    R = G.time_integrated_residuals(F0, F1, g, n_sub=4)
    L = primitive.physics_loss(R, SC, area_weight=g.latitude_weight())
    L.backward()
    return {k: {"finite": bool(torch.isfinite(F1[k].grad).all()), "abs_max": float(F1[k].grad.abs().max())}
            for k in ("u", "v", "T", "Phi")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinn-steps", type=int, default=1500)
    ap.add_argument("--out", default="experiments/step3_residual.json")
    args = ap.parse_args()
    res = {}
    for name, fn in (("E1_autograd_exactness", e1_autograd), ("E2_grid_discretization", e2_grid_discretization),
                     ("E3_omega_diagnosis", e3_omega_diagnosis), ("E4_pole_conditioning", e4_pole_conditioning),
                     ("E5_noise_amplification", e5_noise_amplification), ("E6_time_integrated", e6_time_integrated),
                     ("E7_unit_scaling", e7_unit_scaling), ("E9_gradient_flow", e9_gradient_flow),
                     ("E8_pinn_fit", lambda: e8_pinn_fit(args.pinn_steps))):
        t0 = time.time()
        res[name] = fn()
        print(f"== {name} ({time.time() - t0:.1f}s)")
        print(json.dumps(res[name], indent=1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
