"""Is a tendency core (or KAI) short of data or short of capacity? The generalization gap of trained models.

The one-step loss of checkpoints (KAI's validation loss: the relative L2 error of the next-day state, the number
``val`` of the training log) on the days of years the model was trained on and of years it never saw. A model
that is short of data (overfit) is clearly better on its training years than on unseen years; a model that is
short of capacity (or of training steps) is about equally good on both. The answer decides what a longer training
period can give: more years help the first kind directly, the second kind only with more capacity or steps.
``--cores``: tendency cores (their own rollout); ``--kai``: KAI checkpoints (tags kai_...).

    python scripts/kai_gap_diag.py --out runs/kaitend --cores core_gngmz8 core_gngmz8w \
        --years 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 --json runs/kaitend/diag_gap.json

Every day of a year is used (``--days 64``: the 64 days of the training log, to reproduce its number). The
training years of a core are read from its checkpoint. ``--gap-leads K``: the same for every day of the model's own
K-day rollout. Also printed: the normalized RMSE of the 65 scored channels (area weighted, in units of the
training std), whose square is additive over days.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from pgcfm.kai_tend import DYN, OROG, TISR, LpLoss, core_rollout  # noqa: E402
from run_kai_tendency import build, build_core, lat_of, load_years, parser, stats, zscore  # noqa: E402


def main(argv=None):
    p = parser()
    p.add_argument("--cores", nargs="*", default=[])
    p.add_argument("--kai", nargs="*", default=[], help="KAI checkpoints (tags kai_...), the same loss of their rollout")
    p.add_argument("--years", type=int, nargs="+", required=True)
    p.add_argument("--days", type=int, default=0, help="0: every day; 64: the days of the training log's val")
    p.add_argument("--gap-leads", type=int, default=1, help="days of the model's own rollout (the loss of every lead)")
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    K = a.gap_leads
    torch.set_num_threads(a.threads)
    mean, std, inc, _ = stats(a)
    data = {y: torch.from_numpy(zscore(np.asarray(x), mean, std)) for y, x in load_years(a, a.years).items()}
    H = next(iter(data.values())).shape[-2]
    lat = lat_of(os.path.join(a.data, f"{a.val_year}.nc"), H, a.era5)
    w = torch.as_tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32)
    w = (w / w.mean()).view(1, 1, -1, 1)
    dyn = torch.tensor(DYN)
    sc = torch.tensor([c for c in DYN if c not in (TISR, OROG)])
    lp = LpLoss(d=2, p=2, size_average=True, reduction=True)
    out = {}
    for tag in list(a.cores) + list(a.kai):
        ck = torch.load(os.path.join(a.out, f"{tag}.pt"), map_location="cpu", weights_only=False)
        tr = list(ck.get("args", {}).get("train_years", a.train_years))
        if tag in a.kai:
            core = build(a, "kai", inc, None)
            core.load_state_dict(ck["state"], strict=False)
            core.eval()
            W, hist = 0, False
        else:
            core = build_core(a, mean, std, inc, lat, tag=tag)
            W = int(core.opts.get("mem", 0))
            hist = bool(core.opts.get("hist")) or W > 0
        res = {}
        for y in a.years:
            z = data[y]
            last = len(z) - K
            if a.days:
                idx = np.arange(1 if hist else 0, last, max(1, (last - 1) // a.days))[:a.days]
            else:
                idx = np.arange(1 if hist else 0, last)
            rel, se = [], torch.zeros(K, len(sc))
            with torch.no_grad():
                for i in range(0, len(idx), a.val_batch):
                    j = idx[i:i + a.val_batch]
                    x = z[j]
                    tgt = torch.stack([z[j + k] for k in range(1, K + 1)], 1)             # (B, K, 67, H, W)
                    pv = z[j - 1] if hist else None
                    if W:  # as in the training's val: before the first day of the year that day itself
                        pv = torch.stack([z[np.maximum(j - k, 0)] for k in range(W, 0, -1)], 1)
                    if tag in a.kai:
                        pred = core.rollout(x, torch.tensor(j), K, tgt[:, :, TISR])
                    else:
                        yk = core_rollout(core, x[:, dyn], torch.tensor(j), K, tgt[:, :, TISR],
                                          prev=None if pv is None else pv.index_select(-3, dyn))
                        pred = tgt.index_copy(2, dyn, yk)                                 # the static channel: the truth's
                    d = (pred - tgt).flatten(2)
                    rel.append(d.norm(dim=2) / tgt.flatten(2).norm(dim=2))                # (B, K)
                    if i == 0:
                        assert abs(float(lp(pred[:, 0], tgt[:, 0])) - float(rel[-1][:, 0].mean())) < 1e-5
                    se += (((pred - tgt)[:, :, sc] ** 2) * w).mean((3, 4)).sum(0)
            rel = torch.cat(rel).numpy()
            res[y] = dict(loss=rel.mean(0).tolist(), se=(rel.std(0, ddof=1) / np.sqrt(len(rel))).tolist(), n=len(rel),
                          nrmse=torch.sqrt(se / len(rel)).mean(1).tolist(), trained=y in tr)
        part = lambda key, seen: np.mean([v[key] for v in res.values() if v["trained"] == seen], 0)  # noqa: E731
        both = any(v["trained"] for v in res.values()) and not all(v["trained"] for v in res.values())
        gap = (part("loss", False) / part("loss", True) - 1) if both else np.full(K, np.nan)
        gap2 = (part("nrmse", False) / part("nrmse", True) - 1) if both else np.full(K, np.nan)
        out[tag] = dict(train_years=tr, iters=ck.get("args", {}).get("iters"), params=sum(q.numel() for q in core.parameters()),
                        years={str(y): v for y, v in res.items()}, gap=gap.tolist(), gap_nrmse=gap2.tolist())
        print(f"{tag}: trained on {tr[0]}-{tr[-1]} ({len(tr)} y), {out[tag]['params']} parameters")
        print("  year     " + " ".join(f"{y:>7d}" for y in a.years))
        print("  seen     " + " ".join(f"{'yes' if res[y]['trained'] else 'no':>7s}" for y in a.years))
        for k in range(K):
            print(f"  day {k + 1} loss   " + " ".join(f"{res[y]['loss'][k]:7.4f}" for y in a.years)
                  + f"   unseen / seen - 1: {100 * gap[k]:+.1f} %")
            print(f"  day {k + 1} nrmse  " + " ".join(f"{res[y]['nrmse'][k]:7.4f}" for y in a.years)
                  + f"   unseen / seen - 1: {100 * gap2[k]:+.1f} %")
        print(f"  standard error of a year's day-1 loss about {np.mean([v['se'][0] for v in res.values()]):.4f}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
