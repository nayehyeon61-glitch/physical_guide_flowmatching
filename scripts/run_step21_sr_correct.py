"""Step 21: SR / downscaling as a CORRECTOR of the forecast (owner, 2026-10-28 03:08: "the error is a position
error on the manifold, not a departure from it ... I want to try correcting it through SR / downscaling").

Step 19 trained the SR net on analysis -> analysis pairs, so it could only add detail. What the downscaling
paper (Chiang et al. 2024) actually does is "model field -> observed field": bias / position correction and
the resolution increase in one net. This step trains the same net (pgcfm.sr.SRNet) on FORECAST pairs:

    input   member x(t+lead) at 5.625 deg (5 ClimODE vars) + the initial state x0 + a lead channel + statics
    target  the true state at t+lead: 2.8125 deg ERA5 (``--scale 2``, the SR corrector) or
            5.625 deg ERA5 (``--scale 1``, the control: the same corrector without the resolution increase)
    pairs   the 2016 members of ``--train-ens`` (ClimODE's 180 ICs x 8 members x 6 leads; every 5th IC held out)

and tests on 2017-2018 (``--test-ens`` arms): the corrected members are scored at 2.8 deg against the fine
truth (interpolation of the raw members, and the step 19 analysis-trained SR, as controls) and, coarsened
by the 2 x 2 area mean, at 5.625 deg against the coarsened fine truth next to the raw members. The two WB2
grids are not nested (the 2 x 2 mean of the 2.8 deg truth differs from the 5.625 deg truth by 82 m2/s2 in
z500), so the 5.625 deg comparison uses the coarsened fine truth, and the SR corrector's baseline there is
the raw member sent through the same path with the untrained (identity) net: interpolate, then coarsen
(``_ic``); the raw members and the ClimODE 5.625 deg truth numbers are kept as references (``_raw``, ``_c``). The corrector is also applied to the ensemble mean
(``corrmean``). Since an MSE-trained corrector shrinks the members toward their conditional mean, a per lead
and variable inflation alpha = RMSE(corrected mean) / spread(corrected), fitted on the held-out 2016 ICs,
gives ``corrinfl`` (CRPS with the spread restored).

    python scripts/run_step21_sr_correct.py --scale 2 --out runs/step21 --threads 2
    python scripts/run_step21_sr_correct.py --scale 1 --out runs/step21 --threads 2
    python scripts/run_step21_sr_correct.py --plot-only --out runs/step21 --fig docs/step21
"""

import argparse
import json
import os
import sys
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pgcfm.runs import device_from, run_record  # noqa: E402
from pgcfm.sr import Interpolator, SRNet  # noqa: E402
from scripts.run_step11_climode import VARS, climode_minmax  # noqa: E402
from scripts.run_step19_sr import SR, area_weight, crps_ensemble, rmse, statics, truth  # noqa: E402

LABEL = {"z": "z500", "t": "t850", "t2m": "t2m", "u10": "u10", "v10": "v10"}


class Ens:
    """One saved ensemble file: members (n, M, L, 5, 32, 64) as float16 deviations + x0."""

    def __init__(self, prefix, year, max_ics=0):
        z = np.load(f"{prefix}_ens_{year}.npz")
        self.E16, self.O, self.X0, self.ics, self.leads = z["E_minus_x0"], z["O"], z["X0"], z["ics"], z["leads"]
        self.n = len(self.E16) if not max_ics else min(len(self.E16), max_ics)
        self.M, self.L = self.E16.shape[1:3]

    def members(self, sl):
        """physical members (b, M, L, 5, 32, 64) float32 for the IC slice."""
        return torch.from_numpy(self.E16[sl].astype(np.float32) + self.X0[sl][:, None, None])


class Corrector:
    """SRNet on (member, x0, lead) -> true state, at scale 2 (2.8125 deg) or 1 (5.625 deg)."""

    def __init__(self, net, mean, std, s_lr, s_hr, interp, scale, leads, device):
        self.device = torch.device(device)
        self.net, self.scale, self.leads = net.to(self.device), scale, [int(h) for h in leads]
        self.mean, self.std = mean.to(self.device), std.to(self.device)
        self.s_lr, self.s_hr, self.interp = s_lr.to(self.device), s_hr.to(self.device), interp
        self.alpha = None  # (L, 5) spread inflation after the correction, fitted on the held-out ICs

    def norm(self, x):
        return (x.to(self.device) - self.mean) / self.std

    def inputs(self, member, x0, lead_h):
        """member, x0 physical (B, 5, 32, 64); lead_h (B,) hours -> (B, 11, 32, 64) normalized inputs."""
        xm = self.norm(member)
        lead = (torch.as_tensor(lead_h, dtype=torch.float32, device=self.device) / 144.0).view(-1, 1, 1, 1)
        return xm, torch.cat([xm, self.norm(x0), lead.expand(-1, 1, *xm.shape[-2:])], 1)

    def __call__(self, member, x0, lead_h, chunk=64):
        """-> corrected physical fields (B, 5, H_out, W_out) on the device."""
        out = []
        with torch.no_grad():
            for i in range(0, member.shape[0], chunk):
                xm, xin = self.inputs(member[i:i + chunk], x0[i:i + chunk], lead_h[i:i + chunk])
                base = self.interp(xm) if self.scale == 2 else xm
                out.append(self.net(xin, self.s_lr, self.s_hr, base) * self.std + self.mean)
        return torch.cat(out)

    def apply_ens(self, E, X0, chunk=64):
        """E (b, M, L, 5, 32, 64), X0 (b, 5, 32, 64) -> corrected (b, M, L, 5, H, W)."""
        b, M, L = E.shape[:3]
        x0 = X0[:, None, None].expand(b, M, L, -1, -1, -1).reshape(b * M * L, 5, 32, 64)
        lead = torch.tensor(self.leads, dtype=torch.float32).view(1, 1, L).expand(b, M, L).reshape(-1)
        y = self(E.reshape(b * M * L, 5, 32, 64), x0, lead, chunk)
        return y.reshape(b, M, L, 5, *y.shape[-2:])

    def apply_mean(self, E, X0, chunk=64):
        """the corrector on the ensemble mean -> (b, L, 5, H, W)."""
        b, M, L = E.shape[:3]
        mu = E.mean(1).reshape(b * L, 5, 32, 64)
        x0 = X0[:, None].expand(b, L, -1, -1, -1).reshape(b * L, 5, 32, 64)
        lead = torch.tensor(self.leads, dtype=torch.float32).view(1, L).expand(b, L).reshape(-1)
        y = self(mu, x0, lead, chunk)
        return y.reshape(b, L, 5, *y.shape[-2:])


def coarsen(x, w2):
    """(..., 64, 128) -> (..., 32, 64) area-weighted 2 x 2 mean; w2 (32, 2) cos-lat of the two fine rows."""
    s = x.shape
    b = x.reshape(*s[:-2], 32, 2, 64, 2)
    return (b * w2[:, :, None, None]).sum((-3, -1)) / (2 * w2.sum(1))[:, None]


def fine_truth_at(xf, ens, sl):
    """(b, L, 5, 64, 128) fine truth at the ICs' valid times."""
    return torch.from_numpy(np.stack([np.stack([xf[i + h // 6] for h in ens.leads]) for i in ens.ics[sl]]))


def build(a, dev, leads, hidden=None, blocks=None):
    lat_c, s_lr = statics(a.coarse, "5.625")
    lat_f, s_hr = statics(a.fine, "2.8125")
    interp = Interpolator(lat_c, lat_f, 64, 128, device=dev) if a.scale == 2 else None
    net = SRNet(c_in=11, hidden=hidden or a.hidden, n_blocks=blocks or a.blocks, scale=a.scale)
    return net, s_lr, (s_hr if a.scale == 2 else s_lr), interp, lat_c, lat_f


def tag(a):
    return f"corr_s{a.scale}"


def validate(corr, tr, xf16, val_ids, w_out, dev, chunk=6):
    """held-out ICs: RMSE of the corrected mean vs the baseline mean (interp or raw), spread -> rel, alpha."""
    acc = {k: 0.0 for k in ("rmse_c", "rmse_b", "sp2_c")}
    for c in range(0, len(val_ids), chunk):
        ids = val_ids[c:c + chunk]
        E = tr.members(ids).to(dev)
        X0 = torch.from_numpy(tr.X0[ids]).to(dev)
        Y = corr.apply_ens(E, X0)
        if corr.scale == 2:
            T = fine_truth_at(xf16, tr, ids).float().to(dev)
            B = corr.interp(E.mean(1).reshape(-1, 5, 32, 64)).reshape(E.shape[0], tr.L, 5, 64, 128)
        else:
            T = torch.from_numpy(tr.O[ids]).to(dev)
            B = E.mean(1)
        acc["rmse_c"] = acc["rmse_c"] + rmse(Y.mean(1), T, w_out).sum(0)
        acc["rmse_b"] = acc["rmse_b"] + rmse(B, T, w_out).sum(0)
        acc["sp2_c"] = acc["sp2_c"] + (Y.var(1, unbiased=True) * w_out).mean((-1, -2)).sum(0)
    n = len(val_ids)
    r_c, r_b, sp = acc["rmse_c"] / n, acc["rmse_b"] / n, (acc["sp2_c"] / n).sqrt()
    return float((r_c / r_b).mean()), r_c, r_b, (r_c / sp)


def train(a, dev, log):
    tr = Ens(a.train_ens, a.train_year)
    xf16 = torch.from_numpy(truth(a.fine, a.train_year, "2.8125")).half() if a.scale == 2 else None
    O = torch.from_numpy(tr.O)
    mean = O.mean((0, 1, 3, 4)).view(1, 5, 1, 1)
    std = O.std((0, 1, 3, 4)).view(1, 5, 1, 1)
    net, s_lr, s_hr, interp, lat_c, lat_f = build(a, dev, tr.leads)
    corr = Corrector(net, mean, std, s_lr, s_hr, interp, a.scale, tr.leads, dev)
    w_out = area_weight(lat_f if a.scale == 2 else lat_c).to(dev)
    all_ids = np.arange(tr.n)
    val_ids, train_ids = all_ids[all_ids % 5 == 4], all_ids[all_ids % 5 != 4]
    log(f"[train] {a.train_ens} {a.train_year}: {len(train_ids)} train ICs x {tr.M} members x {tr.L} leads, "
        f"{len(val_ids)} held-out ICs, scale {a.scale}, device {dev}")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    hist, curve, best, t0 = [], [], None, time.time()
    ck_path = os.path.join(a.out, f"{tag(a)}.pt")
    for it in range(1, a.iters + 1):
        ic = rng.choice(train_ids, a.batch)
        m = rng.integers(0, tr.M, a.batch)
        l = rng.integers(0, tr.L, a.batch)
        member = torch.from_numpy(tr.E16[ic, m, l].astype(np.float32) + tr.X0[ic])
        x0 = torch.from_numpy(tr.X0[ic])
        lead = torch.from_numpy(tr.leads[l].astype(np.float32))
        if a.scale == 2:
            y = torch.from_numpy(np.stack([xf16[tr.ics[i] + int(tr.leads[j]) // 6].numpy() for i, j in zip(ic, l)])).float()
        else:
            y = torch.from_numpy(tr.O[ic, l])
        xm, xin = corr.inputs(member, x0, lead)
        base = interp(xm) if a.scale == 2 else xm
        out = net(xin, corr.s_lr, corr.s_hr, base)
        loss = (((out - corr.norm(y)) ** 2) * w_out).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(loss.item())
        if it % max(1, a.iters // 20) == 0:
            log(f"  it {it:5d} loss {np.mean(hist[-50:]):.4f} {time.time() - t0:.0f}s")
        if it % a.val_every == 0 or it == a.iters:
            net.eval()
            rel, r_c, r_b, alpha = validate(corr, tr, xf16, val_ids, w_out, dev)
            net.train()
            curve.append({"it": it, "rel": rel, "rmse_corr": r_c.cpu().tolist(), "rmse_base": r_b.cpu().tolist()})
            log(f"  val it {it}: corrected / baseline mean RMSE " + " ".join(
                f"{LABEL[v]} {float(r_c[:, k].mean() / r_b[:, k].mean()):.3f}" for k, v in enumerate(VARS))
                + f" | mean {rel:.3f}" + (" *" if best is None or rel < best else ""))
            if best is None or rel < best:
                best = rel
                corr.alpha = alpha.cpu()
                torch.save({"net": {k: v.cpu() for k, v in net.state_dict().items()}, "mean": mean, "std": std,
                            "hidden": a.hidden, "blocks": a.blocks, "scale": a.scale, "alpha": corr.alpha,
                            "leads": tr.leads.tolist()}, ck_path)
    return {"hist": hist[::10], "val_curve": curve, "best_rel": best, "train_seconds": time.time() - t0,
            "n_params": sum(p.numel() for p in net.parameters()), "n_train_ics": int(len(train_ids)),
            "n_val_ics": int(len(val_ids))}


def load_corrector(a, dev):
    ck = torch.load(os.path.join(a.out, f"{tag(a)}.pt"), map_location="cpu")
    net, s_lr, s_hr, interp, lat_c, lat_f = build(a, dev, ck["leads"], ck["hidden"], ck["blocks"])
    net.load_state_dict(ck["net"])
    net.eval()
    corr = Corrector(net, ck["mean"], ck["std"], s_lr, s_hr, interp, ck["scale"], ck["leads"], dev)
    corr.alpha = ck["alpha"]
    return corr, lat_c, lat_f


def load_analysis_sr(path, a, dev):
    """the step 19 analysis-trained SR (reference at 2.8 deg), or None."""
    if not path or not os.path.exists(path):
        return None
    ck = torch.load(path, map_location="cpu")
    lat_c, s_lr = statics(a.coarse, "5.625")
    lat_f, s_hr = statics(a.fine, "2.8125")
    net = SRNet(hidden=ck["hidden"], n_blocks=ck["blocks"])
    net.load_state_dict(ck["net"])
    net.eval()
    return SR(net, ck["mean"], ck["std"], s_lr, s_hr, Interpolator(lat_c, lat_f, 64, 128, device=dev), dev)


def inflate(Y, alpha):
    """members (b, M, L, 5, H, W) -> mean + alpha (L, 5) x deviations."""
    mu = Y.mean(1, keepdim=True)
    return mu + alpha.to(Y.device)[None, None, :, :, None, None] * (Y - mu)


def evaluate(a, dev, log):
    corr, lat_c, lat_f = load_corrector(a, dev)
    asr = load_analysis_sr(a.analysis_sr, a, dev) if corr.scale == 2 else None
    w_c, w_f = area_weight(lat_c).to(dev), area_weight(lat_f).to(dev)
    w2 = torch.from_numpy(np.cos(np.deg2rad(lat_f)).astype(np.float32)).view(32, 2).to(dev)
    mx, mn = climode_minmax(a.coarse)
    scl = torch.from_numpy(((mx - mn)[:, None, None]).astype(np.float32)).to(dev)
    mnt = torch.from_numpy(mn[:, None, None].astype(np.float32)).to(dev)
    crps = lambda ens, obs, w: (crps_ensemble((ens - mnt) / scl, (obs - mnt) / scl, dim=1) * w).mean((-1, -2)).sum(0)  # noqa: E731
    spread2 = lambda ens, w: (ens.var(1, unbiased=True) * w).mean((-1, -2)).sum(0)  # noqa: E731
    out = {}
    for y in a.test_years:
        xf = torch.from_numpy(truth(a.fine, y, "2.8125"))
        for arm in a.test_ens:
            name = os.path.basename(arm)
            ens = Ens(arm, y, a.max_ics)
            t0 = time.time()
            acc = {}
            add = lambda k, v: acc.__setitem__(k, acc.get(k, 0.0) + v)  # noqa: E731
            for c in range(0, ens.n, a.chunk):
                sl = slice(c, min(ens.n, c + a.chunk))
                E = ens.members(sl).to(dev)
                X0 = torch.from_numpy(ens.X0[sl]).to(dev)
                Oc = torch.from_numpy(ens.O[sl]).to(dev)
                Tf = fine_truth_at(xf, ens, sl).to(dev)
                Tcf = coarsen(Tf, w2)
                b = E.shape[0]
                # raw members at 5.625 deg, against the ClimODE truth (_c) and the coarsened fine truth (_cf)
                add("rmse_raw_c", rmse(E.mean(1), Oc, w_c).sum(0)); add("crps_raw_c", crps(E, Oc, w_c))
                add("rmse_pers_c", rmse(X0[:, None], Oc, w_c).sum(0))
                add("rmse_raw_cf", rmse(E.mean(1), Tcf, w_c).sum(0)); add("crps_raw_cf", crps(E, Tcf, w_c))
                add("rmse_pers_cf", rmse(X0[:, None], Tcf, w_c).sum(0)); add("spread2_raw", spread2(E, w_c))
                Y = corr.apply_ens(E, X0)
                Ym = corr.apply_mean(E, X0)
                Yi = inflate(Y, corr.alpha)
                if corr.scale == 2:
                    Ei = corr.interp(E.reshape(-1, 5, 32, 64)).reshape(b, ens.M, ens.L, 5, 64, 128)
                    Pi = corr.interp(X0)
                    add("rmse_i", rmse(Ei.mean(1), Tf, w_f).sum(0)); add("crps_i", crps(Ei, Tf, w_f))
                    add("spread2_i", spread2(Ei, w_f)); add("rmse_pers_f", rmse(Pi[:, None], Tf, w_f).sum(0))
                    Eic = coarsen(Ei, w2)  # the untrained corrector's path (interpolate, coarsen): the 5.625 deg baseline
                    add("rmse_ic_cf", rmse(Eic.mean(1), Tcf, w_c).sum(0)); add("crps_ic_cf", crps(Eic, Tcf, w_c))
                    add("spread2_ic", spread2(Eic, w_c)); add("rmse_pers_ic_cf", rmse(coarsen(Pi, w2)[:, None], Tcf, w_c).sum(0))
                    del Eic
                    add("rmse_f", rmse(Y.mean(1), Tf, w_f).sum(0)); add("crps_f", crps(Y, Tf, w_f))
                    add("crps_finfl", crps(Yi, Tf, w_f)); add("spread2_f", spread2(Y, w_f))
                    add("rmse_fmean", rmse(Ym, Tf, w_f).sum(0))
                    if asr is not None:
                        _, Ea = asr.both(E.reshape(-1, 5, 32, 64))
                        Ea = Ea.reshape(b, ens.M, ens.L, 5, 64, 128)
                        add("rmse_a", rmse(Ea.mean(1), Tf, w_f).sum(0)); add("crps_a", crps(Ea, Tf, w_f))
                        del Ea
                    Yc, Ymc, Yic = coarsen(Y, w2), coarsen(Ym, w2), coarsen(Yi, w2)
                    del Ei, Pi
                else:
                    Yc, Ymc, Yic = Y, Ym, Yi
                    add("rmse_corr_c", rmse(Yc.mean(1), Oc, w_c).sum(0)); add("crps_corr_c", crps(Yc, Oc, w_c))
                    add("crps_corrinfl_c", crps(Yic, Oc, w_c)); add("rmse_corrmean_c", rmse(Ymc, Oc, w_c).sum(0))
                add("rmse_corr_cf", rmse(Yc.mean(1), Tcf, w_c).sum(0)); add("crps_corr_cf", crps(Yc, Tcf, w_c))
                add("crps_corrinfl_cf", crps(Yic, Tcf, w_c)); add("rmse_corrmean_cf", rmse(Ymc, Tcf, w_c).sum(0))
                add("spread2_corr", spread2(Yc, w_c))
                del E, Y, Ym, Yi, Yc, Ymc, Yic, Tf, Tcf
            res = {k: (v / ens.n).cpu().tolist() for k, v in acc.items()}
            for k in [k for k in res if k.startswith("spread2_")]:
                res["spread_" + k[8:]] = np.sqrt(np.asarray(res.pop(k))).tolist()
            res["leads"], res["n_ic"], res["seconds"] = ens.leads.tolist(), int(ens.n), time.time() - t0
            out.setdefault(name, {})[str(y)] = res
            for j, h in enumerate(ens.leads):
                if corr.scale == 2:
                    log(f"[{name} {y}] {h:3d} h 5.625 (vs coarsened fine) z500 RMSE raw {res['rmse_raw_cf'][j][0]:.0f} interp-coarsen {res['rmse_ic_cf'][j][0]:.0f} corr "
                        f"{res['rmse_corr_cf'][j][0]:.0f} corr(mean) {res['rmse_corrmean_cf'][j][0]:.0f} | CRPS raw "
                        f"{res['crps_raw_cf'][j][0]:.4f} corr {res['crps_corr_cf'][j][0]:.4f} infl {res['crps_corrinfl_cf'][j][0]:.4f}"
                        f" | 2.8 t2m RMSE interp {res['rmse_i'][j][2]:.2f} corr {res['rmse_f'][j][2]:.2f}"
                        + (f" aSR {res['rmse_a'][j][2]:.2f}" if 'rmse_a' in res else ""))
                else:
                    log(f"[{name} {y}] {h:3d} h 5.625 (vs ClimODE truth) z500 RMSE raw {res['rmse_raw_c'][j][0]:.0f} corr "
                        f"{res['rmse_corr_c'][j][0]:.0f} corr(mean) {res['rmse_corrmean_c'][j][0]:.0f} | CRPS raw "
                        f"{res['crps_raw_c'][j][0]:.4f} corr {res['crps_corr_c'][j][0]:.4f} infl {res['crps_corrinfl_c'][j][0]:.4f}"
                        f" | t2m RMSE raw {res['rmse_raw_c'][j][2]:.2f} corr {res['rmse_corr_c'][j][2]:.2f}")
            log(f"[{name} {y}] {ens.n} ICs in {res['seconds']:.0f}s")
    return out


def plot(out_dir, fig_dir, arm):
    for f_ in ("WenQuanYi Zen Hei", "Noto Sans CJK KR", "DejaVu Sans"):
        if any(f_ == x.name for x in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f_
            plt.rcParams["axes.unicode_minus"] = False
            break
    res = {}
    for s in (2, 1):
        p = os.path.join(out_dir, f"corr_s{s}.json")
        if os.path.exists(p):
            with open(p) as f:
                res[s] = json.load(f)["evaluation"].get(arm, {})
    if 2 not in res or not res[2]:
        return
    years = sorted(res[2])
    leads = res[2][years[0]]["leads"]
    avg = lambda s, k: np.mean([np.asarray(res[s][y][k]) for y in years], 0) if s in res and all(k in res[s][y] for y in years) else None  # noqa: E731
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.6))
    iz, it2 = VARS.index("z"), VARS.index("t2m")
    series_5 = [((2, "rmse_raw_cf"), "원래 멤버", "#999999", "-"), ((2, "rmse_ic_cf"), "원래 멤버 보간→2×2 평균 (SR 기준선)", "#999999", "--"),
                ((2, "rmse_corr_cf"), "SR 보정 멤버 (2.8°→5.625°)", "#EE6677", "-"),
                ((2, "rmse_corrmean_cf"), "SR 보정(앙상블 평균)", "#EE6677", "-."), ((1, "rmse_corr_cf"), "5.625° 보정망 (대조군)", "#4477AA", "-"),
                ((1, "rmse_corrmean_cf"), "5.625° 보정망(평균)", "#4477AA", "-.")]
    for p, iv, title in ((ax[0, 0], iz, "z500 RMSE, 5.625° (m²/s²)"), (ax[0, 1], it2, "t2m RMSE, 5.625° (K)")):
        for (s, k), lab, col, ls in series_5:
            v = avg(s, k)
            if v is not None:
                p.plot(leads, v[:, iv], color=col, ls=ls, lw=2, marker="o", ms=4, label=lab)
        v = avg(2, "rmse_pers_cf")
        p.plot(leads, v[:, iv], color="k", ls=":", lw=1.5, label="지속")
        p.set_title(title)
    p = ax[1, 0]
    for (s, k), lab, col, ls in (((2, "crps_raw_cf"), "원래 멤버", "#999999", "-"), ((2, "crps_ic_cf"), "보간→2×2 평균 (SR 기준선)", "#999999", "--"),
                                 ((2, "crps_corr_cf"), "SR 보정", "#EE6677", "-"),
                                 ((2, "crps_corrinfl_cf"), "SR 보정 + 퍼짐 복원", "#EE6677", "--"),
                                 ((1, "crps_corr_cf"), "5.625° 보정망", "#4477AA", "-"), ((1, "crps_corrinfl_cf"), "5.625° 보정망 + 퍼짐 복원", "#4477AA", "--")):
        v = avg(s, k)
        if v is not None:
            p.plot(leads, v[:, iz], color=col, ls=ls, lw=2, marker="o", ms=4, label=lab)
    p.set_title("z500 CRPS, 5.625° (ClimODE 정규화)")
    p = ax[1, 1]
    for k, lab, col, ls in (("rmse_i", "멤버 보간", "#999999", "-"), ("rmse_a", "분석장 학습 SR (19단계)", "#228833", "-"),
                            ("rmse_f", "예보 학습 SR 보정", "#EE6677", "-"), ("rmse_fmean", "SR 보정(평균)", "#EE6677", "-.")):
        v = avg(2, k)
        if v is not None:
            p.plot(leads, v[:, it2], color=col, ls=ls, lw=2, marker="o", ms=4, label=lab)
    v = avg(2, "rmse_pers_f")
    p.plot(leads, v[:, it2], color="k", ls=":", lw=1.5, label="지속 (보간)")
    p.set_title("t2m RMSE, 2.8125° (K)")
    for p in ax.ravel():
        p.set_xlabel("lead (h)")
        p.grid(alpha=0.3)
        p.set_xticks(leads)
        p.legend(fontsize=7)
    fig.suptitle(f"Step 21: 예보→진실로 학습한 SR 보정 ({arm}), 5.625° 진실 = 2.8° 진실의 2×2 평균, ClimODE IC {', '.join(years)}", fontsize=11)
    fig.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "fig_sr_correct.png"), dpi=130)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--coarse", default="/home/claude/era5/era5_data")
    p.add_argument("--fine", default="/home/claude/era5_hres/era5_data")
    p.add_argument("--train-ens", default="runs/step13/seqregress_pinn")
    p.add_argument("--train-year", type=int, default=2016)
    p.add_argument("--test-ens", nargs="+", default=["runs/step13/seqfm_dct_pinn_regm_sband_clim_flow", "runs/step13/seqregress_pinn"])
    p.add_argument("--test-years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--analysis-sr", default="runs/step19/sr.pt", help="step 19 analysis-trained SR (2.8 deg reference)")
    p.add_argument("--scale", type=int, default=2, choices=[1, 2])
    p.add_argument("--out", default="runs/step21")
    p.add_argument("--fig", default="docs/step21")
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--chunk", type=int, default=6)
    p.add_argument("--max-ics", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    dev = device_from(a.device)
    os.makedirs(a.out, exist_ok=True)
    log = lambda s: print(s, flush=True)  # noqa: E731
    if a.plot_only:
        plot(a.out, a.fig, os.path.basename(a.test_ens[0]))
        return
    jp = os.path.join(a.out, f"{tag(a)}.json")
    res = {"args": vars(a), "run": run_record(dev)}
    if a.eval_only and os.path.exists(jp):
        with open(jp) as f:
            res = json.load(f)
    elif not a.eval_only:
        res["train"] = train(a, dev, log)
        log(f"[train] done {res['train']['train_seconds']:.0f}s, best corrected / baseline {res['train']['best_rel']:.3f}")
    res["evaluation"] = evaluate(a, dev, log)
    with open(jp, "w") as f:
        json.dump(res, f, indent=1)
    plot(a.out, a.fig, os.path.basename(a.test_ens[0]))


if __name__ == "__main__":
    main()
