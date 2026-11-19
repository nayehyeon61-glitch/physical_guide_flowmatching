"""Gains of models over one reference, channel by channel (z500, t850, t2m, u850, ...), from the score files and
their raw arrays (scripts/run_kai_tendency.py --score). Every change is a gain of A over the reference: + = A better.

    python scripts/eval_kai_channel_gains.py --ref blend_core_gngmz8@kai_ro \
        --chan z500,z850,t850,t2m,msl,u850,v850,wind850,u250,v250,wind250 \
        stage:loop_core_gngmz8@kai_ro stage:stcg stage_r5:stcg_r5 stage_hot:stcg_hot stage_hot5:stcg_hot5 > out.md

A model is <score>:<arm>, read from runs/kaitend/score_<score>.json and score_<score>_raw.npz; the reference arm
must be in the same score file (paired over the same ICs). Per channel and lead:
  RMSE gain %     1 - RMSE_A / RMSE_ref (the channel's nRMSE gain: the normalization cancels), with the 95 %
                  interval of a paired bootstrap over the ICs (500 draws)
  information / amplitude
                  the breakdown's split of the MSE (scripts/eval_kai_breakdown.py, terms2) for this one channel,
                  as a share of the gain: information + amplitude + bias = the gain
  ACC, activity   ACC of the anomalies and the pattern amplitude of the forecast anomaly over the truth's
windNNN is the wind vector at NNN hPa: RMSE = sqrt(MSE_u + MSE_v), and the terms of u and v add (no ACC).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_kai_breakdown import terms2  # noqa: E402

SHORT = lambda k: k.replace("_core_gngmz8", "")  # noqa: E731
Q = [0, 2, 4, 6, 9]


def boot(va, vb, seed):
    rng, N = np.random.default_rng(seed), va.shape[0]
    bs = [100 * (1 - np.sqrt(va[i].mean(1)) / np.sqrt(vb[i].mean(1))) for i in
          (rng.integers(0, N, (50, N)) for _ in range(10))]
    return np.percentile(np.concatenate(bs), [2.5, 97.5], axis=0)


def gains(out, sc, k, ref, chans):
    raw = np.load(f"{out}/score_{sc}_raw.npz")
    names = json.load(open(f"{out}/score_{sc}.json"))["channels"]
    ma, mb = raw[f"{k}__mse"].astype(np.float64), raw[f"{ref}__mse"].astype(np.float64)
    sa, sb = terms2(raw, k), terms2(raw, ref)
    res = {}
    for c in chans:
        ix = [names.index(f"u{c[4:]}"), names.index(f"v{c[4:]}")] if c.startswith("wind") else [names.index(c)]
        va, vb = ma[..., ix].sum(-1), mb[..., ix].sum(-1)  # (N, L)
        ra, rb = np.sqrt(va.mean(0)), np.sqrt(vb.mean(0))
        g = 100 * (1 - ra / rb)
        part = 100 * (sb[:, ix].sum(1) - sa[:, ix].sum(1)) / ((ra + rb) * rb)[:, None]  # (L, 3) bias / info / amp
        lo, hi = boot(va, vb, len(res))
        e = dict(gain=g, lo=lo, hi=hi, bias=part[:, 0], info=part[:, 1], amp=part[:, 2])
        assert np.abs(part.sum(1) - g).max() < 1e-3, (k, c)
        if len(ix) == 1:
            e.update(accA=raw[f"{k}__acc"].mean(0)[:, ix[0]], accB=raw[f"{ref}__acc"].mean(0)[:, ix[0]],
                     actA=raw[f"{k}__act"].mean(0)[:, ix[0]], actB=raw[f"{ref}__act"].mean(0)[:, ix[0]])
        res[c] = e
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--ref", required=True)
    p.add_argument("--chan", default="z500,z850,t850,t2m,msl,u850,v850,wind850,u250,v250,wind250")
    p.add_argument("--out", default="runs/kaitend")
    p.add_argument("--json")
    p.add_argument("--compact", action="store_true", help="two short tables: variable x model gains, and day 10")
    a = p.parse_args()
    chans = a.chan.split(",")
    R = {k: gains(a.out, sc, k, a.ref, chans) for sc, k in (m.split(":") for m in a.models)}
    f = lambda v, d=1, s=1: " / ".join(f"{s * v[q]:+.{d}f}".replace("-", "−") for q in Q)  # noqa: E731
    ref = SHORT(a.ref)
    if a.compact:
        ms = list(R)
        print("| 변수 | " + " | ".join(f"`{SHORT(k)}`" for k in ms) + " |")
        print("| --- |" + " --- |" * len(ms))
        for c in chans:
            print(f"| {c} | " + " | ".join(f(R[k][c]["gain"]) for k in ms) + " |")
        print("\n| 변수 | " + " | ".join(f"`{SHORT(k)}`" for k in ms) + " | 기준 |")
        print("| --- |" + " --- |" * (len(ms) + 1))
        for c in chans:
            e0 = R[ms[0]][c]
            cell = lambda e: (f"{e['gain'][9]:+.1f} ({e['info'][9]:+.1f} / {e['amp'][9]:+.1f})"  # noqa: E731
                              + (f", ACC {e['accA'][9]:.3f}, 활동도 {e['actA'][9]:.2f}" if "accA" in e else "")).replace("-", "−")
            base = f"ACC {e0['accB'][9]:.3f}, 활동도 {e0['actB'][9]:.2f}" if "accB" in e0 else ""
            print(f"| {c} | " + " | ".join(cell(R[k][c]) for k in ms) + f" | {base} |")
        return
    for c in chans:
        e0 = next(iter(R.values()))[c]
        act = f", 10일 활동도 {e0['actB'][9]:.2f}" if "actB" in e0 else ""
        print(f"\n#### {c} (기준 `{ref}`{act})\n")
        print("| A | RMSE 이득 1 / 3 / 5 / 7 / 10일 | 정보 항 | 진폭 항 | 95% 구간 (10일) | ACC 차이 1 / 3 / 5 / 7 / 10일"
              " | ACC 10일 A / 기준 | 활동도 10일 |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for k, r in R.items():
            e = r[c]
            ci = f"{e['lo'][9]:+.1f} ~ {e['hi'][9]:+.1f}".replace("-", "−")
            if "accA" in e:
                acc = (f"{f(e['accA'] - e['accB'], 3)} | {e['accA'][9]:.3f} / {e['accB'][9]:.3f} | {e['actA'][9]:.2f}")
            else:
                acc = " |  | "
            print(f"| `{SHORT(k)}` | {f(e['gain'])} | {f(e['info'])} | {f(e['amp'])} | {ci} | {acc} |")
    if a.json:
        json.dump({k: {c: {q: np.round(v, 5).tolist() for q, v in e.items()} for c, e in r.items()} for k, r in R.items()},
                  open(a.json, "w"))


if __name__ == "__main__":
    main()
