"""Step 10 experiment: the ensemble mean as a deterministic forecast, and the correction ladder
(owner's note, 2026-10-24): does a neural operator reduce spatially organised error that is left
after classical bias / linear (MBM) correction and after a local CNN?

    x_hat_corr(t+k) = x_hat(t+k) + N_phi(x_hat(t+k), x_t, static, k)

x_hat is the ensemble mean of the frozen step 7 serial chain (8 members, Heun 8, Markov rollout
10 days). Each lead is corrected directly (post-hoc, no feedback into the rollout).

Splits (all unseen by the serial chain, which was trained on 2001-2004):
    op_train  2005 (step 7 calibration year) + 2007   operator / CNN training
    cal       2006                                      every classical fit (bias, MBM)
    op_val    2008                                      checkpoint selection of operator / CNN
    test      2021                                      scored once
Daily initial conditions inside each split.

Arms (deterministic outputs):
    mean            ensemble mean
    bias            mean + a_kc (per lead and channel)
    bias_grid       mean + a_kc(h, w) (local climatological bias)
    mbm             mean' = a_kc + b_kc mean (``LinearMeanCalibrator``, least squares)
    mbm_tan         mbm with its change projected on the hydrostatic tangent space (step 9)
    mbm_cnn         mbm + local CNN residual (4 x conv3x3, receptive field 9 cells)
    mbm_fno_mean    mbm + owner's ``EnsembleToDeterministic`` fed the mean only (mean-only ablation)
    mbm_fno_ens     mbm + the same operator with member pooling (MBM-shifted members)
    fno_ens         the operator with member pooling on the raw ensemble (does it absorb MBM?)
    *_tan           a learned arm with its correction projected on the hydrostatic tangent space
The operator is the owner's file (``pgcfm/ens_to_det.py``, official neuraloperator FNO), unchanged.

    python scripts/run_step10_deterministic.py --out runs/step10_det --step7 runs/step7 [--quick]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import metrics  # noqa: E402
from pgcfm.cfm import area_weights, rollout  # noqa: E402
from pgcfm.data import PGCFMSequenceDataset, kai_data_config  # noqa: E402
from pgcfm.ens_to_det import EnsembleToDeterministic, OfficialFNO, channel_mse, mean_only  # noqa: E402
from pgcfm.latent import LatentConfig, LatentPGCFM  # noqa: E402
from pgcfm.physics.primitive import ResidualScales  # noqa: E402
from pgcfm.projection import freeze  # noqa: E402
from pgcfm.spread import KEEP, DCTBands, EnsembleCalibrated  # noqa: E402
from pgcfm.structure import LinearMeanCalibrator, spatial_diff, tangent_calibrated  # noqa: E402
from scripts.run_step4_pgcfm import member_residuals  # noqa: E402
from scripts.train_pgcfm import DEFAULTS, build  # noqa: E402

LEARNED = ("mbm_cnn", "mbm_fno_mean", "mbm_fno_ens", "fno_ens", "member_chain", "member_chain_mw",
           "mbmx_cnn", "mbmx_fno_ens")


class LocalCNN(nn.Module):
    """Local control with the operator's interface: mean + head(conv3x3 stack([mean, context]))."""

    def __init__(self, channels, context_channels, width=64, layers=4):
        super().__init__()
        c, net = channels + context_channels, []
        for _ in range(layers):
            net += [nn.Conv2d(c, width, 3, padding=1, padding_mode="replicate"), nn.GELU()]
            c = width
        self.net, self.head = nn.Sequential(*net), nn.Conv2d(width, channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, members, context=None):
        mean = members.mean(1)
        return mean + self.head(self.net(torch.cat([mean, context], 1)))


class PersistenceMBM(nn.Module):
    """Classical regression on two predictors per lead and channel: mu' = a + b mu + c x0 (area
    weighted least squares). The control for a learned corrector that gets x0 as context and can
    blend the forecast toward persistence. Every member is shifted by mu' - mu."""

    def __init__(self, C, K):
        super().__init__()
        self.register_buffer("coef", torch.zeros(K, C, 3))
        self.coef[..., 1] = 1.0

    @torch.no_grad()
    def fit(self, ens, obs, x0):
        w = area_weights(ens.shape[-2]).expand(ens.shape[-2:]).sqrt()
        mu = ens.mean(1)
        for k in range(mu.shape[1]):
            for c in range(mu.shape[2]):
                X = torch.stack([torch.ones_like(mu[:, k, c]), mu[:, k, c], x0[:, c]], -1) * w[..., None]
                self.coef[k, c] = torch.linalg.lstsq(X.reshape(-1, 3).double(),
                                                     (obs[:, k, c] * w).reshape(-1, 1).double()).solution[:, 0].float()
        return self

    def forward(self, ens, x0):
        K = ens.shape[2]
        a, b, c = (self.coef[:K, :, i][None, :, :, None, None] for i in range(3))
        mu = ens.mean(1)
        return ens + (a + (b - 1) * mu + c * x0[:, None])[:, None]


class BlendMBM(nn.Module):
    """Step 20: the classical regression on several deterministic means per lead and channel,
    mu' = a + sum_i b_i mu_i + c x0 (area-weighted least squares), e.g. the step 18 G(x) core (good to
    72 h) blended with the regression mean (good beyond) by lead-dependent weights."""

    def __init__(self, C, K, n_mu):
        super().__init__()
        self.register_buffer("coef", torch.zeros(K, C, n_mu + 2))
        self.coef[..., 1] = 1.0

    @torch.no_grad()
    def fit(self, mus, obs, x0):
        """mus: list of (n, K, C, H, W) means, obs (n, K, C, H, W), x0 (n, C, H, W)."""
        w = area_weights(obs.shape[-2]).expand(obs.shape[-2:]).sqrt()
        for k in range(obs.shape[1]):
            for c in range(obs.shape[2]):
                cols = [torch.ones_like(obs[:, k, c])] + [mu[:, k, c] for mu in mus] + [x0[:, c]]
                X = torch.stack(cols, -1) * w[..., None]
                self.coef[k, c] = torch.linalg.lstsq(X.reshape(-1, len(cols)).double(),
                                                     (obs[:, k, c] * w).reshape(-1, 1).double()).solution[:, 0].float()
        return self

    def forward(self, mus, x0):
        """-> (n, K, C, H, W) blended mean."""
        K = mus[0].shape[1]
        out = self.coef[:K, :, 0][None, :, :, None, None] + self.coef[:K, :, -1][None, :, :, None, None] * x0[:, None]
        for i, mu in enumerate(mus):
            out = out + self.coef[:K, :, 1 + i][None, :, :, None, None] * mu
        return out


class MemberChain(nn.Module):
    """Owner's 2026-10-24 path: each member (already member-wise MBM) -> shared local CNN residual ->
    shared FNO residual (official neuraloperator FNO, member-wise, no pooling) -> average.
    Member order invariant; the corrected members are kept so their spread can be checked."""

    def __init__(self, channels, context_channels, width=32, modes=8, cnn_width=32):
        super().__init__()
        self.cnn = LocalCNN(channels, context_channels, width=cnn_width)
        self.fno = OfficialFNO(n_modes=(modes, modes), in_channels=channels + context_channels,
                               out_channels=channels, hidden_channels=width, n_layers=2,
                               positional_embedding="grid", channel_mlp_dropout=0.0)
        last = self.fno.operator.projection.fcs[-1]  # start at the identity: zero last projection layer
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def members(self, members, context):
        b, m = members.shape[:2]
        x = members.flatten(0, 1)
        ctx = context.repeat_interleave(m, 0)
        x = self.cnn(x[:, None], ctx)
        x = x + self.fno(torch.cat([x, ctx], 1))
        return x.view(b, m, *x.shape[1:])

    def forward(self, members, context=None):
        return self.members(members, context).mean(1)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/step10_det")  # runs/step10 belongs to run_step10_baselines.py
    p.add_argument("--step7", default="runs/step7")
    p.add_argument("--arms", nargs="+", default=["mean", "bias", "bias_grid", "mbm", "mbm_tan", "member_mbm", "mbmx", "mbmx_tan",
                                                     *LEARNED])
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--days", type=int, nargs=4, default=[90, 60, 40, 60], help="days of 2007, 2006, 2008, 2021")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.members, a.steps, a.horizon, a.epochs, a.width, a.modes = 3, 2, 2, 2, 8, 4
        a.days = [16, 16, 16, 16]
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())

    root7 = os.path.join(a.step7, "data")
    tr = os.path.join(root7, "train")
    mean_p, std_p = os.path.join(tr, "stats", "global_mean.npy"), os.path.join(tr, "stats", "global_std.npy")
    dcfg = kai_data_config(mean_p, std_p)
    pair_ds = PGCFMSequenceDataset(dcfg, tr, horizon=1)
    n_lat, n_lon = pair_ds.img_shape_x, pair_ds.img_shape_y

    # new periods, generated in child processes (netCDF4 segfaults on files written in-process)
    gen = os.path.join(os.path.dirname(__file__), "make_balanced_era5.py")
    dirs = {"train2005": os.path.join(root7, "calib")}
    for name, year, days, seed in (("train2007", 2007, a.days[0], 700), ("cal", 2006, a.days[1], 610),
                                   ("op_val", 2008, a.days[2], 800), ("test", 2021, a.days[3], 2100)):
        dirs[name] = os.path.join(a.out, "data", name)
        if not os.path.exists(os.path.join(dirs[name], f"{year}.nc")):
            subprocess.run([sys.executable, gen, "--out", dirs[name], "--years", str(year), "--days", str(days),
                            "--n-lat", str(n_lat), "--n-lon", str(n_lon), "--seed", str(seed)], check=True)

    res7 = json.load(open(os.path.join(a.step7, "results.json")))
    a7 = next(v for k, v in res7.items() if k.startswith("args_"))
    lcfg = LatentConfig(k_lat=a7.get("k_lat", 8), k_lon=a7.get("k_lon", 16), latent=a7.get("latent", 64),
                        enc_hidden=tuple(a7.get("enc_hidden", (256, 128, 64))),
                        dec_hidden=tuple(reversed(a7.get("enc_hidden", (256, 128, 64)))),
                        carry_highpass=a7.get("carry_highpass", False), coupling="serial")
    base = build(dict(DEFAULTS, model_channels=32, channel_mult=[1, 2, 2], num_res_blocks=1), n_lat, n_lon,
                 np.load(mean_p), np.load(std_p))
    guide = base.guide
    guide.scales = ResidualScales(**res7["scales"])
    freeze(guide)
    gm = SimpleNamespace(guide=guide)
    serial = LatentPGCFM(lcfg, n_lat, n_lon)
    serial.load_state_dict(torch.load(os.path.join(a.step7, "serial.pt"))["latent"])
    freeze(serial)
    serial.eval()
    chain = EnsembleCalibrated(serial, "serial")
    bands = DCTBands(n_lat, n_lon, lcfg.k_lat, lcfg.k_lon)
    area = area_weights(n_lat).expand(n_lat, n_lon).clamp_min(1e-6).contiguous()  # poles: channel_mse needs > 0

    def split(names, seed):
        """Rollouts of every daily IC of the split's periods, cached (members in float16)."""
        path = os.path.join(a.out, f"ens_{'+'.join(names)}.pt")
        sig = {"step7": os.path.abspath(a.step7), "members": a.members, "steps": a.steps, "horizon": a.horizon,
               "days": a.days, "seed": seed}
        if os.path.exists(path) and not a.force:
            out = torch.load(path)
            if out.get("sig") != sig:  # never score a rollout made with other settings under this run's args
                raise SystemExit(f"{path} was made with {out.get('sig')}, this run asks for {sig}; "
                                 "use another --out or --force")
            return out
        items = []
        for n in names:
            ds = PGCFMSequenceDataset(dcfg, dirs[n], horizon=a.horizon, train=False)
            items += [ds[i] for i in range(len(ds))]
        b = torch.utils.data.default_collate(items)
        chain.layout(a.members, "interleave")
        ens, t = [], time.time()
        with torch.no_grad():
            for i in range(0, len(items), 8):
                ens.append(rollout(chain, b["x0"][i:i + 8], b["static"][i:i + 8], b["doy"][i:i + 8], a.members,
                                   steps=a.steps, seed=seed + i).half())
        out = {"sig": sig, "ens": torch.cat(ens), "targets": b["targets"], "x0": b["x0"], "static": b["static"], "doy": b["doy"]}
        torch.save(out, path)
        print(f"[data] {names}: {len(items)} ICs, rollout {time.time() - t:.0f}s", flush=True)
        return out

    S = {"op_train": split(["train2005", "train2007"], 100), "cal": split(["cal"], 200),
         "op_val": split(["op_val"], 300), "test": split(["test"], 400)}
    clim = S["op_train"]["targets"].mean((0, 1))  # (66, H, W) climatology for ACC / anomaly power
    K = a.horizon

    # classical fits on the calibration split only
    cal_ens = S["cal"]["ens"].float()
    mbm = LinearMeanCalibrator(cal_ens.shape[3], K).fit_validation(cal_ens, S["cal"]["targets"])
    err_cal = S["cal"]["targets"] - cal_ens.mean(1)  # (B, K, 66, H, W)
    bias_kc = (err_cal * area).mean((0, 3, 4)) / area.mean()  # (K, 66)
    bias_grid = err_cal.mean(0)  # (K, 66, H, W)
    # member-by-member regression x_m' = a_kc + b_kc x_m (each member regressed on truth, pooled)
    M = cal_ens.shape[1]
    mmbm = LinearMeanCalibrator(cal_ens.shape[3], K).fit_validation(
        cal_ens.flatten(0, 1)[:, None], S["cal"]["targets"].repeat_interleave(M, 0))
    mbmx = PersistenceMBM(cal_ens.shape[3], K).fit(cal_ens, S["cal"]["targets"], S["cal"]["x0"])
    del cal_ens

    def member_mbm(ens):
        return mmbm(ens.flatten(0, 1)[:, None])[:, 0].view_as(ens)

    def classical(name, ens, x0=None):
        mu = ens.mean(1)
        if name == "mbmx":
            return mbmx(ens, x0).mean(1)
        if name == "mean":
            return mu
        if name == "bias":
            return mu + bias_kc[None, :, :, None, None]
        if name == "bias_grid":
            return mu + bias_grid[None]
        if name == "member_mbm":
            return member_mbm(ens).mean(1)
        return mbm(ens).mean(1)  # mbm shifts every member by the same mean change

    # learned arms: sample = (IC, lead); input members (MBM-shifted unless fno_ens), context = x0, static, lead
    def tensors(d, shift):
        """(B, M, K, C_keep, H, W) members in float16, MBM applied in chunks (memory)."""
        out = []
        for i in range(0, d["ens"].shape[0], 8):
            ens = d["ens"][i:i + 8].float()
            ens = {"mbm": mbm, "member": member_mbm, "none": lambda e: e,
                   "mbmx": lambda e: mbmx(e, d["x0"][i:i + 8])}[shift](ens)
            out.append(ens[:, :, :, KEEP].half())
        return torch.cat(out)

    def context(d, idx, k):
        lead = torch.full((len(idx), 1, n_lat, n_lon), (k + 1) / K)
        return torch.cat([d["x0"][idx][:, KEEP], d["static"][idx], lead], 1)

    def make_model(name):
        torch.manual_seed(7)
        C, Q = len(KEEP), len(KEEP) + 2
        if name.endswith("_cnn"):
            return LocalCNN(C, Q)
        if name.startswith("member_chain"):
            return MemberChain(C, Q, width=a.width, modes=a.modes)
        return EnsembleToDeterministic(C, context_channels=Q, width=a.width, modes=a.modes)

    def predict(model, name, d, ens):
        """ens (B, M, K, C_keep, H, W) -> (B, K, C_keep, H, W)."""
        model.eval()
        out = torch.empty(ens.shape[0], K, *ens.shape[3:])
        with torch.no_grad():
            for k in range(K):
                for i in range(0, ens.shape[0], 16):
                    idx = torch.arange(i, min(i + 16, ens.shape[0]))
                    x = ens[idx, :, k].float()
                    if name == "mbm_fno_mean":
                        x = mean_only(x)
                    out[idx, k] = model(x, context(d, idx, k))
        return out

    def shift_of(name):
        return ("none" if name == "fno_ens" else "member" if name.startswith("member_")
                else "mbmx" if name.startswith("mbmx_") else "mbm")

    def train(name):
        path = os.path.join(a.out, f"{name}.pt")
        model = make_model(name)
        shift = shift_of(name)
        tr_ens, va_ens = tensors(S["op_train"], shift), tensors(S["op_val"], shift)
        n_params = sum(p.numel() for p in model.parameters())
        if os.path.exists(path) and not a.force:
            ck = torch.load(path, weights_only=False)
            model.load_state_dict(ck["state"])
            return model, ck["info"], va_ens
        dtr, dva = S["op_train"], S["op_val"]
        ytr, yva = dtr["targets"][:, :, KEEP], dva["targets"][:, :, KEEP]
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        pairs = torch.cartesian_prod(torch.arange(tr_ens.shape[0]), torch.arange(K))
        best, best_state, hist, t = float("inf"), None, [], time.time()
        for ep in range(a.epochs):
            model.train()
            tl = []
            for ids in torch.randperm(len(pairs)).split(a.batch):
                i, k = pairs[ids, 0], pairs[ids, 1]
                x = tr_ens[i, :, k].float()
                if name == "mbm_fno_mean":
                    x = mean_only(x)
                ctx = torch.cat([dtr["x0"][i][:, KEEP], dtr["static"][i],
                                 ((k + 1).float() / K)[:, None, None, None].expand(len(i), 1, n_lat, n_lon)], 1)
                if name == "member_chain_mw":  # member-wise MSE (each member regressed on truth)
                    xm = model.members(x, ctx)
                    loss = channel_mse(xm.flatten(0, 1), ytr[i, k].repeat_interleave(xm.shape[1], 0), area).mean()
                else:  # only the averaged output is compared with the truth
                    loss = channel_mse(model(x, ctx), ytr[i, k], area).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl.append(float(loss))
            sched.step()
            pv = predict(model, name, dva, va_ens)
            vl = float(channel_mse(pv.flatten(0, 1), yva.flatten(0, 1), area).mean())
            hist.append({"epoch": ep, "train_mse": float(np.mean(tl)), "val_mse": vl})
            print(f"[{name}] ep {ep} train {np.mean(tl):.4f} val {vl:.4f} ({time.time() - t:.0f}s)", flush=True)
            if vl < best:
                best, best_state = vl, {k_: v.clone() for k_, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        info = {"history": hist, "best_val_mse": best, "n_params": n_params, "train_seconds": time.time() - t}
        torch.save({"state": model.state_dict(), "info": info}, path)
        return model, info, va_ens

    def full(d, pred_keep, ref):
        """Put the KEEP channels into a full 66-channel state (TISR from ``ref``)."""
        out = ref.clone()
        out[:, :, KEEP] = pred_keep
        return out

    def scores(pred, d):
        """pred (B, K, 66, H, W) -> per-lead deterministic scores on the KEEP channels."""
        o = d["targets"]
        pk, ok, ck = pred[:, :, KEEP], o[:, :, KEEP], clim[KEEP]
        r = {"lead_rmse": metrics._wmean((pk - ok) ** 2).mean((0, 2)).sqrt().tolist(),
             "lead_mae": metrics._wmean((pk - ok).abs()).mean((0, 2)).tolist()}
        fa, oa = pk - ck, ok - ck
        acc = metrics._wmean(fa * oa) / (metrics._wmean(fa ** 2) * metrics._wmean(oa ** 2)).sqrt().clamp_min(1e-12)
        r["lead_acc"] = acc.mean((0, 2)).tolist()
        cen = lambda f: f - metrics._wmean(f)[..., None, None]  # noqa: E731
        r["lead_pattern_rmse"] = metrics._wmean((cen(pk) - cen(ok)) ** 2).mean((0, 2)).sqrt().tolist()
        r["lead_grad_rmse"] = [float(metrics._wmean((spatial_diff(pk[:, k]) - spatial_diff(ok[:, k])) ** 2)
                                     .mean().sqrt()) for k in range(K)]
        r["lead_band_amp"] = [(bands.power(fa[:, k]) / bands.power(oa[:, k])).tolist() for k in range(K)]
        r["lead_err_band"] = [bands.power(pk[:, k] - ok[:, k]).tolist() for k in range(K)]
        with torch.no_grad():
            r["lead_hyd"] = member_residuals(gm, d["x0"], pred[:, None], d["static"], d["doy"])["hyd"]
        return r

    res_path = os.path.join(a.out, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) and not a.force else {}
    res.setdefault("runs", []).append(vars(a))  # every invocation; each arm also keeps its own args
    res["n_ics"] = {k: int(v["x0"].shape[0]) for k, v in S.items()}
    res["mbm_coef"] = {"a": mbm.a.tolist(), "b": mbm.b.tolist()}

    def save():
        with open(res_path + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(res_path + ".tmp", res_path)

    if "truth_hyd" not in res:
        d = S["test"]
        res["truth_hyd"] = member_residuals(gm, d["x0"], d["targets"][:, None], d["static"], d["doy"])["hyd"]
        res["persistence"] = scores(d["x0"][:, None].expand_as(d["targets"]).clone(), d)
        save()

    def report(name):
        r = res[name]
        pick = lambda v: " / ".join(f"{v[i]:.3f}" for i in sorted({0, min(4, K - 1), K - 1}))  # noqa: E731
        print(f"[{name}] rmse {pick(r['lead_rmse'])} (mean {np.mean(r['lead_rmse']):.4f}) mae {pick(r['lead_mae'])} "
              f"acc {pick(r['lead_acc'])} pattern {pick(r['lead_pattern_rmse'])} grad {pick(r['lead_grad_rmse'])} "
              f"hyd {pick(r['lead_hyd'])}", flush=True)

    d = S["test"]
    ens_test = d["ens"].float()
    mu_test = ens_test.mean(1)
    for name in a.arms:
        if name in LEARNED:
            if f"{name}" in res and f"{name}_tan" in res and not a.force:
                continue
            model, info, _ = train(name)
            te_ens = tensors(d, shift_of(name))
            pred = full(d, predict(model, name, d, te_ens), mu_test)
            res[name] = scores(pred, d)
            res[name]["info"], res[name]["args"] = info, vars(a)
            if name.startswith("member_chain"):  # do the corrected members still form an ensemble?
                with torch.no_grad():
                    sp_in, sp_out = [], []
                    for k in range(K):
                        x = te_ens[:, :, k].float()
                        xm = torch.cat([model.members(x[j:j + 8], context(d, torch.arange(j, min(j + 8, len(x))), k))
                                        for j in range(0, len(x), 8)])
                        sp_in.append(float(metrics._wmean(x.var(1)).mean().sqrt()))
                        sp_out.append(float(metrics._wmean(xm.var(1)).mean().sqrt()))
                res[name]["lead_spread_in"], res[name]["lead_spread_out"] = sp_in, sp_out
                print(f"[{name}] member spread in {sp_in[0]:.3f}/{sp_in[-1]:.3f} out {sp_out[0]:.3f}/{sp_out[-1]:.3f}")
            report(name)
            # the whole change from the raw mean (MBM + correction) projected on the hydrostatic tangent space
            tan = tangent_calibrated(mu_test[:, None], pred[:, None], guide, d["x0"], d["static"], d["doy"])[:, 0]
            res[f"{name}_tan"] = scores(tan, d)
            res[f"{name}_tan"]["args"] = vars(a)
            report(f"{name}_tan")
        else:
            if name in res and not a.force:
                continue
            if name in ("mbm_tan", "mbmx_tan"):
                pred = tangent_calibrated(mu_test[:, None], classical(name[:-4], ens_test, d["x0"])[:, None], guide,
                                          d["x0"], d["static"], d["doy"])[:, 0]
            else:
                pred = classical(name, ens_test, d["x0"])
            res[name] = scores(pred, d)
            res[name]["args"] = vars(a)
            report(name)
        save()
    return res


if __name__ == "__main__":
    main()
