"""The Kalman correction of a forecast's OUTPUT toward KAI's forecast at every lead (owner 2026-11-17 22:49 KST: "the
blend is day 0-4 only; can the full 0-10 day blend be checked?"). From cached forecasts only:

    x_d = x_f,d + K_d (z_d - x_f,d),   K_d = (S - c) / (S + R - 2 c)

with S, R, c = the second moments of the forecast's error, KAI's error and their product in the FIT year, per lead,
channel and latitude (the formula of the island's gain in scripts/diag_island_kalman.py; nothing is fed back to a
core). Writes fc_<name>_<test year>.npy for scripts/diag_stage_filter_light.py and a json with the gains.

    python scripts/make_full_blend.py --out runs/kaitend --base ARM --meas kai --name NAME [--days 1 2 ...]

ARM and KAI must be cached for the fit year and the test year (fc_<spec>_<year>.npy; KAI's by
scripts/diag_island_kalman.py --kai-fc).
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_kai_tendency as R  # noqa: E402
from pgcfm.kai_tend import DYN, TISR  # noqa: E402

p = R.parser()
p.add_argument("--base", required=True, help="cache spec of the forecast"); p.add_argument("--meas", default="kai", help="cache spec of KAI's forecast")
p.add_argument("--name", required=True)
p.add_argument("--fit-year", type=int, default=None, help="year of the error statistics (default: --val-year)")
p.add_argument("--days", type=int, nargs="*", default=None, help="corrected leads (default: all)")
p.add_argument("--json", default=None)
a = p.parse_args()
fit = a.fit_year or a.val_year
tr = R.load_years(a, a.train_years); mean, std, inc, clim = R.stats(a, tr); del tr
fyr = R.load_years(a, [fit])[fit]   # a fine grid stays on disk (1.5 deg), normalized on access
fy = R.ZYear(fyr, mean, std, as_torch=True) if isinstance(fyr, np.memmap) else torch.from_numpy(R.zscore(fyr, mean, std))
lat = R.lat_of(os.path.join(a.data, f"{fit}.nc"), fy.shape[-2], a.era5)
L = a.leads
days = sorted(a.days) if a.days else list(range(1, L + 1))
w = np.cos(np.deg2rad(lat)); w = (w / w.mean()).reshape(1, -1)
sc = [i for i, c in enumerate(DYN) if c != TISR]
keep = np.ones((len(DYN), 1)); keep[DYN.index(TISR)] = 0


def fc(spec, year):
    return np.load(os.path.join(a.out, f"fc_{spec.replace('>', '-').replace('@', '_')}_{year}.npy"), mmap_mode="r")


A1, K1, A2, K2 = fc(a.base, fit), fc(a.meas, fit), fc(a.base, a.test_year), fc(a.meas, a.test_year)
ics = np.arange(0, len(fy) - L, a.ic_stride)
assert A1.shape == K1.shape and A1.shape[:2] == (len(ics), L), (A1.shape, K1.shape, len(ics))
assert A2.shape == K2.shape, (A2.shape, K2.shape)
path = os.path.join(a.out, f"fc_{a.name}_{a.test_year}.npy")
out = np.lib.format.open_memmap(path[:-4] + ".part.npy", mode="w+", dtype=np.float16, shape=A2.shape)
for i in range(0, len(A2), 16):     # the leads that are not corrected stay the forecast's (block by block: 4.6 GB at 1.5 deg)
    out[i:i + 16] = A2[i:i + 16]
am = lambda v: (v * w).mean(1)[sc].mean()  # noqa: E731  area mean, mean of the scored channels
rows = {}
for d in days:
    S = Rv = c = 0.0
    for i in range(0, len(ics), 16):   # the second moments of the two errors, 16 ICs at a time
        T = fy[ics[i:i + 16] + d][:, DYN].numpy().astype(np.float64)
        ef, ek = np.asarray(A1[i:i + 16, d - 1], np.float64) - T, np.asarray(K1[i:i + 16, d - 1], np.float64) - T
        S, Rv, c = S + (ef ** 2).sum((0, -1)), Rv + (ek ** 2).sum((0, -1)), c + (ef * ek).sum((0, -1))
    nw = len(ics) * A1.shape[-1]
    S, Rv, c = S / nw, Rv / nw, c / nw                                                       # (66, H)
    K = np.clip((S - c) / np.maximum(S + Rv - 2 * c, 1e-12), 0, 1) * keep
    Pa = (1 - K) ** 2 * S + K ** 2 * Rv + 2 * K * (1 - K) * c
    for i in range(0, len(A2), 16):
        x = np.asarray(A2[i:i + 16, d - 1], np.float32)
        out[i:i + 16, d - 1] = (x + K[None, :, :, None].astype(np.float32) * (np.asarray(K2[i:i + 16, d - 1], np.float32) - x)).astype(np.float16)
    rows[d] = dict(K=float(am(K)), rho=float(am(c / np.sqrt(np.maximum(S * Rv, 1e-24)))),
                   rms_arm=float(np.sqrt((S * w).mean(1))[sc].mean()), rms_kai=float(np.sqrt((Rv * w).mean(1))[sc].mean()),
                   rms_expected=float(np.sqrt((Pa * w).mean(1))[sc].mean()))
    r = rows[d]
    print(f"day {d:2d}: K {r['K']:.3f} rho {r['rho']:.3f} rms in {fit}: forecast {r['rms_arm']:.3f} KAI {r['rms_kai']:.3f} "
          f"expected {r['rms_expected']:.3f} ({100 * (1 - r['rms_expected'] / r['rms_arm']):+.1f} %)", flush=True)
out.flush()
del out
os.replace(path[:-4] + ".part.npy", path)
print("saved", path)
json.dump(dict(base=a.base, meas=a.meas, fit_year=fit, days=days, gains=rows),
          open(a.json or os.path.join(a.out, f"full_blend_{a.name}.json"), "w"), indent=1)
