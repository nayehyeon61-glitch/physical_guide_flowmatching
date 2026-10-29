"""Step 19: super-resolution (5.625 -> 2.8125 deg) as the OUTPUT stage of the step 13 ensemble.

The owner (2026-10-28 00:03): "conversely, is SR usable in step 13? ... pretrain the SR so it produces high
resolution". Step 18 showed SR cannot help the physics operator; here it is placed after the calibrated
members, where it can only add fine-scale detail, and is scored at 2.8125 deg against the true 2.8 deg
ERA5 (WB2 128x64) with the interpolation of the same members as the control.

Model: pgcfm.sr.SRNet (reduced Chiang et al. 2024: cascaded conv, CBAM residual attention blocks, skip
fusion, pixel shuffle, fine-topography head, global interpolation residual), trained with latitude-weighted
MSE on ERA5 pairs (coarse analysis -> fine analysis) of the 5 ClimODE variables, 2012-2015, validated 2016.

Evaluation (2017-2018):
  analysis : SR of the 5.625 deg analysis vs interpolation, RMSE against the 2.8 deg analysis
  ensemble : the step 13 members (``<ens>_ens_<year>.npz``, ClimODE's 180 ICs x 8 members x 6 leads), one or
             several arms, interpolated vs super-resolved member by member: RMSE of the ensemble mean, spread,
             CRPS at 2.8 deg (ClimODE's min-max normalization, properscoring's estimator computed in torch);
             persistence = interpolated x0; the same members scored at 5.625 deg for reference.
             ``--save-sr`` writes the super-resolved members (``<out>/<arm>_sr_<year>.npz``, float16).

CPU (the 2026-10-28 run, 4 years, 31 min training):
    python scripts/run_step19_sr.py --coarse /home/claude/era5/era5_data --fine /home/claude/era5_hres/era5_data \
        --ens runs/step13/seqfm_dct_pinn_regm_sband_clim_flow --out runs/step19 --threads 2
GPU (the 2.8125 deg files from ``wb2_to_kai.py --grid 128x64 --climode-only`` land in the same era5_data root):
    python scripts/run_step19_sr.py --coarse data/era5_5p625/era5_data --fine data/era5_5p625/era5_data \
        --device auto --train-range 1979 2015 --iters 20000 --batch 32 --chunk 30 --out runs/step13_best/sr \
        --ens runs/step13_best/seqfm_dct_pinn_regm_band runs/step13_best/seqregress_pinn --save-sr
Or as the SR=1 stage of scripts/run_step13_best.sh (docs/RUN_ON_GPU.md 12).
"""

import argparse
import json
import os
import sys
import time

import matplotlib
import numpy as np
import torch
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pgcfm.runs import autocast, device_from, run_record  # noqa: E402
from pgcfm.sr import Interpolator, SRNet  # noqa: E402
from scripts.run_step11_climode import FOLDERS, VARS, climode_minmax  # noqa: E402

LABEL = {"z": "z500", "t": "t850", "t2m": "t2m", "u10": "u10", "v10": "v10"}
UNIT = {"z": "m²/s²", "t": "K", "t2m": "K", "u10": "m/s", "v10": "m/s"}


def truth(root, year, deg):
    """(T, 5, H, W) float32 physical, ClimODE order."""
    out = []
    for v, f in zip(VARS, FOLDERS):
        with xr.open_dataset(os.path.join(root, f, f"{f}_{year}_{deg}deg.nc")) as d:
            out.append(d[v].values.astype(np.float32))
    return np.stack(out, 1)


def statics(root, deg):
    """lat (H,), static (1, 3, H, W): orography z-score, land-sea mask, sin(lat)."""
    with xr.open_dataset(os.path.join(root, "constants", f"constants_{deg}deg.nc")) as d:
        lat = d["lat"].values.astype(np.float64)
        oro, lsm = d["orography"].values.astype(np.float32), d["lsm"].values.astype(np.float32)
    sl = np.broadcast_to(np.sin(np.deg2rad(lat)).astype(np.float32)[:, None], oro.shape)
    s = np.stack([(oro - oro.mean()) / oro.std(), lsm, sl])[None]
    return lat, torch.from_numpy(np.ascontiguousarray(s))


def area_weight(lat):
    w = np.cos(np.deg2rad(lat))
    return torch.from_numpy((w / w.mean()).astype(np.float32))[:, None]


def rmse(a, b, w):
    """(..., H, W) tensors -> (...) latitude-weighted RMSE."""
    return torch.sqrt(((a - b) ** 2 * w).mean((-1, -2)))


def crps_ensemble(ens, obs, dim=1):
    """properscoring.crps_ensemble's estimator, E|X - y| - 1/2 E|X - X'| with the pair mean over all M^2
    pairs, in torch (any device); ens with the member axis at ``dim``, obs without it -> (ens without dim)."""
    e = ens.movedim(dim, 0)
    M = e.shape[0]
    skill = (e - obs[None]).abs().mean(0)
    pair = 0.0
    for i in range(M):
        pair = pair + (e[i][None] - e).abs().sum(0)
    return skill - 0.5 * pair / (M * M)


class SR:
    """The trained net with its normalization, applied to physical (N, 5, 32, 64) fields on ``device``."""

    def __init__(self, net, mean, std, s_lr, s_hr, interp, device):
        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.mean, self.std = mean.to(self.device), std.to(self.device)
        self.s_lr, self.s_hr, self.interp = s_lr.to(self.device), s_hr.to(self.device), interp

    def norm(self, x):
        return (x.to(self.device) - self.mean) / self.std

    def both(self, x_phys, chunk=64):
        """physical (N, 5, 32, 64) -> (interpolated, super-resolved) physical (N, 5, 64, 128) on ``device``."""
        xi, xs = [], []
        with torch.no_grad():
            for i in range(0, x_phys.shape[0], chunk):
                x = self.norm(x_phys[i:i + chunk])
                xin = self.interp(x)
                xi.append(xin * self.std + self.mean)
                xs.append(self.net(x, self.s_lr, self.s_hr, xin) * self.std + self.mean)
        return torch.cat(xi), torch.cat(xs)


def train(a, dev, log):
    lat_c, s_lr = statics(a.coarse, "5.625")
    lat_f, s_hr = statics(a.fine, "2.8125")
    interp = Interpolator(lat_c, lat_f, 64, 128, device=dev)
    t0 = time.time()
    xc = {y: torch.from_numpy(truth(a.coarse, y, "5.625")) for y in a.train_years + [a.val_year]}
    xf = {y: torch.from_numpy(truth(a.fine, y, "2.8125")).half() for y in a.train_years + [a.val_year]}
    allc = torch.cat([xc[y] for y in a.train_years])
    mean = allc.mean((0, 2, 3), keepdim=True)
    std = allc.std((0, 2, 3), keepdim=True)
    n_pairs = len(allc)
    del allc
    log(f"[train] {len(a.train_years)} years, {n_pairs} pairs loaded in {time.time() - t0:.0f}s, device {dev}")
    w_f = area_weight(lat_f).to(dev)
    net = SRNet(hidden=a.hidden, n_blocks=a.blocks)
    sr = SR(net, mean, std, s_lr, s_hr, interp, dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    nv = len(xc[a.val_year])
    val_idx = np.linspace(0, nv - 1, a.n_val).astype(int)
    xv_c, xv_f = xc[a.val_year][val_idx], xf[a.val_year][val_idx].float().to(dev)
    hist, curve, best, t0 = [], [], None, time.time()
    for it in range(1, a.iters + 1):
        ys = rng.choice(a.train_years, a.batch)
        ts = [int(rng.integers(0, len(xc[y]))) for y in ys]
        x = sr.norm(torch.stack([xc[y][t] for y, t in zip(ys, ts)]))
        y = sr.norm(torch.stack([xf[y][t] for y, t in zip(ys, ts)]).float())
        xin = interp(x)
        with autocast(dev, a.amp):
            out = net(x, sr.s_lr, sr.s_hr, xin)
        loss = (((out.float() - y) ** 2) * w_f).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(loss.item())
        if it % max(1, a.iters // 20) == 0:
            log(f"  it {it:5d} loss {np.mean(hist[-50:]):.4f} {time.time() - t0:.0f}s")
        if it % a.val_every == 0 or it == a.iters:
            net.eval()
            xi, xs = sr.both(xv_c)
            net.train()
            r_i, r_s = rmse(xi, xv_f, w_f).mean(0).cpu(), rmse(xs, xv_f, w_f).mean(0).cpu()
            rel = float((r_s / r_i).mean())
            curve.append({"it": it, "rel": rel, "rmse_interp": r_i.tolist(), "rmse_sr": r_s.tolist()})
            log(f"  val it {it}: SR / interp RMSE " + " ".join(f"{LABEL[v]} {s / i:.3f}" for v, s, i in zip(VARS, r_s, r_i))
                + f" | mean {rel:.3f}" + (" *" if best is None or rel < best else ""))
            if best is None or rel < best:
                best = rel
                torch.save({"net": {k: v.cpu() for k, v in net.state_dict().items()}, "mean": mean.cpu(),
                            "std": std.cpu(), "hidden": a.hidden, "blocks": a.blocks},
                           os.path.join(a.out, f"{a.tag}.pt"))
    return {"hist": hist[::10], "val_curve": curve, "best_rel": best, "train_seconds": time.time() - t0,
            "n_params": sum(p.numel() for p in net.parameters()), "n_pairs": n_pairs}


def load_sr(a, dev):
    ck = torch.load(os.path.join(a.out, f"{a.tag}.pt"), map_location="cpu")
    lat_c, s_lr = statics(a.coarse, "5.625")
    lat_f, s_hr = statics(a.fine, "2.8125")
    net = SRNet(hidden=ck["hidden"], n_blocks=ck["blocks"])
    net.load_state_dict(ck["net"])
    net.eval()
    return SR(net, ck["mean"], ck["std"], s_lr, s_hr, Interpolator(lat_c, lat_f, 64, 128, device=dev), dev), lat_c, lat_f


def evaluate(a, dev, log):
    sr, lat_c, lat_f = load_sr(a, dev)
    w_c, w_f = area_weight(lat_c).to(dev), area_weight(lat_f).to(dev)
    mx, mn = climode_minmax(a.coarse)
    scl = torch.from_numpy(((mx - mn)[:, None, None]).astype(np.float32)).to(dev)
    mnt = torch.from_numpy(mn[:, None, None].astype(np.float32)).to(dev)
    out = {"analysis": {}, "ensemble": {}}
    xcs, xfs = {}, {}
    for y in a.test_years:
        xcs[y], xfs[y] = torch.from_numpy(truth(a.coarse, y, "5.625")), torch.from_numpy(truth(a.fine, y, "2.8125"))
        idx = np.arange(0, len(xcs[y]), 4)  # analysis SR, every 4th time
        xi, xs = sr.both(xcs[y][idx])
        xfd = xfs[y][idx].to(dev)
        r_i, r_s = rmse(xi, xfd, w_f).mean(0).cpu(), rmse(xs, xfd, w_f).mean(0).cpu()
        del xi, xs, xfd
        out["analysis"][str(y)] = {"rmse_interp": r_i.tolist(), "rmse_sr": r_s.tolist(), "n": int(len(idx))}
        log(f"[analysis {y}] RMSE interp / SR: " + " ".join(f"{LABEL[v]} {i:.3g}/{s:.3g}" for v, i, s in zip(VARS, r_i, r_s)))
    for arm in a.ens:
        name = os.path.basename(arm)
        out["ensemble"][name] = {}
        for y in a.test_years:
            path = f"{arm}_ens_{y}.npz"
            if not os.path.exists(path):
                log(f"[ensemble {name} {y}] {path} missing, skipped")
                continue
            t0 = time.time()
            z = np.load(path)
            E16, O, X0, ics, leads = z["E_minus_x0"], z["O"], z["X0"], z["ics"], z["leads"]
            n, M, L = E16.shape[:3]
            if a.max_ics:
                n = min(n, a.max_ics)
            acc = {k: 0.0 for k in ("rmse_i", "rmse_s", "rmse_sm", "spread2_i", "spread2_s", "crps_i", "crps_s",
                                    "rmse_pers", "rmse_c", "crps_c", "spread2_c")}
            saved = []
            for c in range(0, n, a.chunk):
                sl = slice(c, min(n, c + a.chunk))
                E = torch.from_numpy(E16[sl].astype(np.float32) + X0[sl][:, None, None]).to(dev)  # (b, M, L, 5, 32, 64)
                b = E.shape[0]
                Of = torch.from_numpy(np.stack([np.stack([xfs[y][i + h // 6] for h in leads]) for i in ics[sl]])).to(dev)
                Oc = torch.from_numpy(O[sl]).to(dev)
                Ei, Es = sr.both(E.reshape(b * M * L, 5, 32, 64))
                Ei, Es = Ei.reshape(b, M, L, 5, 64, 128), Es.reshape(b, M, L, 5, 64, 128)
                _, Esm = sr.both(E.mean(1).reshape(b * L, 5, 32, 64))  # SR of the ensemble mean
                Esm = Esm.reshape(b, L, 5, 64, 128)
                Pi, _ = sr.both(torch.from_numpy(X0[sl]))
                acc["rmse_i"] = acc["rmse_i"] + rmse(Ei.mean(1), Of, w_f).sum(0)
                acc["rmse_s"] = acc["rmse_s"] + rmse(Es.mean(1), Of, w_f).sum(0)
                acc["rmse_sm"] = acc["rmse_sm"] + rmse(Esm, Of, w_f).sum(0)
                acc["rmse_pers"] = acc["rmse_pers"] + rmse(Pi[:, None], Of, w_f).sum(0)
                acc["rmse_c"] = acc["rmse_c"] + rmse(E.mean(1), Oc, w_c).sum(0)
                acc["spread2_i"] = acc["spread2_i"] + (Ei.var(1, unbiased=True) * w_f).mean((-1, -2)).sum(0)
                acc["spread2_s"] = acc["spread2_s"] + (Es.var(1, unbiased=True) * w_f).mean((-1, -2)).sum(0)
                acc["spread2_c"] = acc["spread2_c"] + (E.var(1, unbiased=True) * w_c).mean((-1, -2)).sum(0)
                for key, ens, obs, w in (("crps_i", Ei, Of, w_f), ("crps_s", Es, Of, w_f), ("crps_c", E, Oc, w_c)):
                    cr = crps_ensemble((ens - mnt) / scl, (obs - mnt) / scl, dim=1)
                    acc[key] = acc[key] + (cr * w).mean((-1, -2)).sum(0)
                if a.save_sr:
                    saved.append(Es.half().cpu().numpy())
                del E, Of, Oc, Ei, Es, Esm, Pi
            res = {k: (v / n).cpu().tolist() for k, v in acc.items()}  # (L, 5)
            res["leads"], res["n_ic"], res["seconds"] = np.asarray(leads).tolist(), int(n), time.time() - t0
            out["ensemble"][name][str(y)] = res
            if a.save_sr:
                sp = os.path.join(a.out, f"{name}_sr_{y}.npz")
                np.savez(sp, E_sr=np.concatenate(saved), ics=ics[:n], leads=leads, lat=lat_f, vars=np.asarray(VARS))
                log(f"[ensemble {name} {y}] super-resolved members saved: {sp} ({os.path.getsize(sp) / 1e6:.0f} MB)")
            for j, h in enumerate(leads):
                log(f"[ensemble {name} {y}] {h:3d} h z500 RMSE 2.8deg interp {res['rmse_i'][j][0]:.0f} SR {res['rmse_s'][j][0]:.0f} "
                    f"(5.625 {res['rmse_c'][j][0]:.0f}, pers {res['rmse_pers'][j][0]:.0f}) | t2m {res['rmse_i'][j][2]:.2f}/"
                    f"{res['rmse_s'][j][2]:.2f} ({res['rmse_c'][j][2]:.2f}) | CRPS t2m {res['crps_i'][j][2]:.4f}/"
                    f"{res['crps_s'][j][2]:.4f} ({res['crps_c'][j][2]:.4f}) u10 {res['crps_i'][j][3]:.4f}/{res['crps_s'][j][3]:.4f}")
            log(f"[ensemble {name} {y}] {n} ICs in {res['seconds']:.0f}s")
    return out


def _arms(ens):
    """ensemble results as {arm: {year: ...}}; the 2026-10-28 CPU json had {year: ...} for its single arm."""
    if ens and all(k.isdigit() for k in ens):
        return {"seqfm_dct_pinn_regm_sband_clim_flow": ens}
    return ens


def plot(res, out, tag):
    for f_ in ("WenQuanYi Zen Hei", "Noto Sans CJK KR", "DejaVu Sans"):
        if any(f_ == x.name for x in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f_
            plt.rcParams["axes.unicode_minus"] = False
            break
    arms = _arms(res["evaluation"]["ensemble"])
    for arm, ens in arms.items():
        years = sorted(ens)
        if not years:
            continue
        leads = ens[years[0]]["leads"]
        avg = lambda k: np.mean([np.asarray(ens[y][k]) for y in years], 0)  # noqa: E731
        panels = [("rmse", "z", "z500 RMSE (m²/s²)"), ("rmse", "t2m", "t2m RMSE (K)"),
                  ("crps", "t2m", "t2m CRPS (ClimODE 정규화)"), ("crps", "u10", "u10 CRPS (ClimODE 정규화)")]
        fig, ax = plt.subplots(2, 2, figsize=(11, 7.6))
        for p, (metric, v, title) in zip(ax.ravel(), panels):
            iv = VARS.index(v)
            for key, lab, col, ls in ((f"{metric}_i", "멤버 보간 (2.8° 채점)", "#4477AA", "-"),
                                      (f"{metric}_s", "멤버 SR (2.8° 채점)", "#EE6677", "-"),
                                      (f"{metric}_c", "같은 멤버, 5.625° 채점", "#999999", "--")):
                p.plot(leads, avg(key)[:, iv], color=col, ls=ls, lw=2, marker="o", ms=4, label=lab)
            if metric == "rmse":
                p.plot(leads, avg("rmse_pers")[:, iv], color="k", ls=":", lw=1.5, label="지속 (x0 보간, 2.8°)")
                p.plot(leads, avg("rmse_sm")[:, iv], color="#EE6677", ls="-.", lw=1.5, label="앙상블 평균의 SR")
            p.set_title(title)
            p.set_xlabel("lead (h)")
            p.grid(alpha=0.3)
            p.set_xticks(leads)
        ax[0, 0].legend(fontsize=8)
        fig.suptitle(f"Step 19: 13단계 멤버의 출력단 SR ({tag}, {arm}), ClimODE IC {', '.join(years)}", fontsize=12)
        fig.tight_layout()
        suffix = "" if len(arms) == 1 else f"_{arm}"
        fig.savefig(os.path.join(out, f"fig_sr_{tag}{suffix}.png"), dpi=130)
        plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--coarse", required=True, help="5.625 deg era5_data (ClimODE format)")
    p.add_argument("--fine", required=True, help="era5_data root holding the 2.8125 deg files "
                   "(wb2_to_kai.py --grid 128x64 --climode-only; may be the same root as --coarse)")
    p.add_argument("--ens", nargs="+", default=["runs/step13/seqfm_dct_pinn_regm_sband_clim_flow"],
                   help="prefixes of the saved ensembles (<prefix>_ens_<year>.npz), one or several arms")
    p.add_argument("--out", default="runs/step19")
    p.add_argument("--fig", default="docs/step19")
    p.add_argument("--tag", default="sr")
    p.add_argument("--train-years", type=int, nargs="+", default=[2012, 2013, 2014, 2015])
    p.add_argument("--train-range", type=int, nargs=2, default=None, metavar=("FIRST", "LAST"),
                   help="train years FIRST..LAST inclusive (overrides --train-years)")
    p.add_argument("--val-year", type=int, default=2016)
    p.add_argument("--test-years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--n-val", type=int, default=96)
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--chunk", type=int, default=6, help="ICs per evaluation chunk (6 on CPU, 30 on a 24 GB GPU)")
    p.add_argument("--max-ics", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N or auto")
    p.add_argument("--amp", action="store_true", help="bf16 autocast for the training forward pass on cuda")
    p.add_argument("--save-sr", action="store_true", help="write the super-resolved members <out>/<arm>_sr_<year>.npz "
                   "(float16, ~0.7 GB per arm and year for 180 ICs x 8 members x 6 leads)")
    p.add_argument("--eval-only", action="store_true")
    a = p.parse_args(argv)
    if a.train_range:
        a.train_years = list(range(a.train_range[0], a.train_range[1] + 1))
    torch.set_num_threads(a.threads)
    dev = device_from(a.device)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.fig, exist_ok=True)
    log = lambda s: print(s, flush=True)  # noqa: E731
    res = {"args": vars(a), "run": run_record(dev)}
    jp = os.path.join(a.out, f"{a.tag}.json")
    if a.eval_only and os.path.exists(jp):
        with open(jp) as f:
            res = json.load(f)
        res["args_eval"] = vars(a)
    elif a.eval_only:
        log(f"[eval-only] {jp} missing, using the checkpoint {a.out}/{a.tag}.pt only")
    else:
        res["train"] = train(a, dev, log)
        log(f"[train] done {res['train']['train_seconds']:.0f}s, best SR/interp {res['train']['best_rel']:.3f}")
    res["evaluation"] = evaluate(a, dev, log)
    with open(jp, "w") as f:
        json.dump(res, f, indent=1)
    plot(res, a.fig, a.tag)


if __name__ == "__main__":
    main()
