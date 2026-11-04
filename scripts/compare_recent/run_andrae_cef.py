"""Andrae et al. (ICLR 2025) "Continuous Ensemble Weather Forecasting with Diffusion models", public 5.625 deg
checkpoints (github.com/martinandrae/Continuous-Ensemble-Forecasting, models/), on our 179 2020 ICs.

predict.py's Algorithm 1-3 loop is reproduced unchanged (EDMPrecond, deterministic Heun sampler, 20 steps,
sigma 80 -> 0.03, rho 7, alpha = 1 fixed noise over the direct leads); only the data path differs:
  * inputs are WB2 64x32 conservative ERA5 (data/wb2_5var), not WB1's bilinear 5.625 deg files (TUM server
    unreachable here): the same truth as every other model in this comparison
  * normalisation mean / std per variable over 1979-2018 from the 6-hourly WB2 data (upstream: hourly WB1;
    create_dataset.py) -- a mild input shift for the network, like ClimODE's in docs/results_2026_11
  * static fields orography + land-sea mask, min-max scaled as upstream utils.ERA5Dataset
  * the AR conditioning time -24 h is available every 6 h, so 6-hourly data suffices for the 24 h and 6 h models

    python scripts/compare_recent/run_andrae_cef.py --repo /path/Continuous-Ensemble-Forecasting \
        --model iterative-24h --t-direct 24 --t-iter 24 --n-ens 50 --out runs/step27/andrae
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import common as C

MAX_HORIZON = 240


def norm_stats():
    f = os.path.join(C.DATA, "meanstd_1979_2018.json")
    if not os.path.exists(f):
        s, s2, n = np.zeros(5), np.zeros(5), 0
        for y in range(1979, 2019):
            t = np.asarray(C.truth_year(y), np.float64)
            s += t.sum((0, 2, 3)); s2 += (t ** 2).sum((0, 2, 3)); n += t[:, 0].size
        m = s / n
        json.dump({"mean": m.tolist(), "std": np.sqrt(s2 / n - m ** 2).tolist()}, open(f, "w"))
    d = json.load(open(f))
    return np.array(d["mean"], np.float32), np.array(d["std"], np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--model", default="iterative-24h")
    p.add_argument("--t-direct", type=int, default=24)
    p.add_argument("--t-iter", type=int, default=24)
    p.add_argument("--n-ens", type=int, default=50)
    p.add_argument("--n-ic", type=int, default=0, help="first n ICs only (smoke)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(C.R, "runs", "step27", "andrae"))
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    sys.path.insert(0, a.repo)
    from diffusion_networks import EDMPrecond  # noqa: E402
    from sampler import heun_sampler  # noqa: E402

    cfg = json.load(open(os.path.join(a.repo, "models", a.model, "config.json")))
    cond = cfg["conditioning_times"]
    assert len(cond) == 2 and cfg["model"] in ("autoregressive", "continuous"), cfg
    time_emb = 1 if cfg["model"] == "continuous" else 0
    net = EDMPrecond(filters=cfg["filters"], img_channels=3 * 5 + 2, out_channels=5, img_resolution=64,
                     time_emb=time_emb, sigma_data=1, sigma_min=0.02, sigma_max=88)
    net.load_state_dict(torch.load(os.path.join(a.repo, "models", a.model, "best_model.pth"), map_location="cpu"))
    net.eval()
    mean, std = norm_stats()
    st = C.static()
    stat = np.stack([st["orography"], st["lsm"]]).astype(np.float32)
    lo, hi = stat.min((1, 2)), stat.max((1, 2))
    stat = (stat - lo[:, None, None]) / np.where(hi - lo == 0, 1, hi - lo)[:, None, None]
    stat = torch.from_numpy(stat)

    ics = C.ics_2020()
    if a.n_ic:
        ics = ics[:a.n_ic]
    full = np.concatenate([np.asarray(C.truth_year(2019)), np.asarray(C.truth_year(2020))])
    off = len(C.truth_year(2019))
    nrm = lambda x: torch.from_numpy((x - mean[:, None, None]) / std[:, None, None])  # noqa: E731
    direct = np.arange(a.t_direct, a.t_iter + 1, a.t_direct)  # direct leads within one iteration
    n_iter = max(C.LEADS) // a.t_iter
    E, nd = a.n_ens, len(direct)
    want = [(h // a.t_iter - 1, list(direct).index(a.t_iter)) for h in C.LEADS]  # every saved lead is an iteration end
    ens = np.zeros((len(ics), E, len(C.LEADS), 5, 32, 64), np.float32)
    t0 = time.time()
    for e, i0 in enumerate(ics):
        x0 = nrm(full[off + i0]); xm = nrm(full[off + i0 + cond[1] // 6])
        cl = torch.cat([x0, xm, stat])[None].repeat(E * nd, 1, 1, 1)  # (E*nd, 17, H, W), member-major
        tl = torch.from_numpy(np.tile(direct, E).astype(np.float32)) / MAX_HORIZON
        outs = []
        with torch.no_grad():
            for it in range(n_iter):
                z = torch.randn(E, 1, 5, 32, 64).repeat(1, nd, 1, 1, 1).reshape(E * nd, 5, 32, 64)  # alpha = 1
                pred = heun_sampler(net, z, cl, tl, sigma_max=80, sigma_min=0.03, rho=7, num_steps=20,
                                    S_churn=2.5, S_min=0.75, S_max=80, S_noise=1.05).view(E, nd, 5, 32, 64)
                outs.append(pred)
                init = pred[:, -1]
                k = np.argwhere(direct == -cond[1])
                assert k.size, (direct, cond)
                k = k[0][0] + 1
                prev0 = cl.view(E, nd, -1, 32, 64)[:, 0, :5]
                earlier = prev0 if k == nd else pred[:, -(k + 1)]
                cl = torch.cat([init, earlier, stat[None].repeat(E, 1, 1, 1)], 1).repeat_interleave(nd, 0)
        for j, (it, d) in enumerate(want):
            ens[e, :, j] = outs[it][:, d].numpy() * std[:, None, None] + mean[:, None, None]
        if e % 10 == 0 or e == len(ics) - 1:
            print(f"IC {e + 1}/{len(ics)} {time.time() - t0:.0f} s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    tag = f"andrae_{a.model}" + (f"_n{a.n_ic}" if a.n_ic else "")
    np.savez_compressed(os.path.join(a.out, tag + "_ens.npz"), ens=ens.astype(np.float16), ics=np.array(ics))
    meta = {"model": a.model, "repo": "github.com/martinandrae/Continuous-Ensemble-Forecasting", "members": E,
            "t_direct": a.t_direct, "t_iter": a.t_iter, "seconds": time.time() - t0}
    full_ics = C.ics_2020()
    ok = np.isin(full_ics, ics)
    pad = lambda v: np.where(ok[:, None, None], np.pad(v, ((0, len(full_ics) - len(v)), (0, 0), (0, 0)), constant_values=np.nan), np.nan)  # noqa: E731
    C.save(tag, {k: pad(v) for k, v in C.score(ens, ics=ics).items()}, os.path.join(a.out, tag + ".npz"), meta)
    if E > 8:
        C.save(tag + "_m8", {k: pad(v) for k, v in C.score(ens[:, :8], ics=ics).items()},
               os.path.join(a.out, tag + "_m8.npz"), dict(meta, members=8))


if __name__ == "__main__":
    main()
