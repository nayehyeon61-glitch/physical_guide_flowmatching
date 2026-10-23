"""Member-wise rollout of a trained PG-CFM / CFM checkpoint into KAI forecast.nc.

Layout (what scripts/evaluate.py of the evaluation step reads):
    forecast (initial_condition, lead_time, member, channel, lat, lon)
de-normalized ERA5 units (exact inverse, std + 1e-8), all 67 channels in
contract order, lead_time 0 = the initial condition itself,
initial_condition = time index into the test-year file. ``--det`` also
writes the zero-noise (eps = 0) run with the same ICs and no member dim.

    python scripts/infer_pgcfm.py --ckpt runs/pgcfm/joint.pt --test-file /data/test/2020.nc \
        --stats-dir /data/KAI-New/stats --out runs/pgcfm/forecast.nc --leads 10 --members 16 --det

A step 6 checkpoint (``train_projection.py``, ``proj.pt``) is rolled out through
the frozen CFM followed by the projection ODE. A step 7 checkpoint
(``train_latent.py``, ``latent.pt``) is rolled out as encoder -> frozen latent
PINN + latent CFM -> decoder, re-encoded every day.
"""

import argparse
import os
import sys

import netCDF4 as nc
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import contract  # noqa: E402
from pgcfm.cfm import TISR_DYN, rollout  # noqa: E402
from pgcfm.coupling import deterministic_rollout  # noqa: E402
from pgcfm.data import assemble_full_state  # noqa: E402
from scripts.train_pgcfm import build  # noqa: E402


def open_out(path, n_ic, K, M, H, W, lat, lon, ics):
    ds = nc.Dataset(path, "w")
    ds.createDimension("initial_condition", n_ic)
    ds.createDimension("lead_time", K + 1)
    if M:
        ds.createDimension("member", M)
    ds.createDimension("channel", contract.N_CHANNELS)
    ds.createDimension("lat", H)
    ds.createDimension("lon", W)
    dims = ("initial_condition", "lead_time") + (("member",) if M else ()) + ("channel", "lat", "lon")
    v = ds.createVariable("forecast", "f4", dims, zlib=True, complevel=1,
                          chunksizes=(1, 1) + ((1,) if M else ()) + (contract.N_CHANNELS, H, W))
    ds.createVariable("initial_condition", "i8", ("initial_condition",))[:] = ics
    lt = ds.createVariable("lead_time", "i8", ("lead_time",))
    lt.units = "days"
    lt[:] = np.arange(K + 1)
    # integer channel ids as in KAI src/inference.py (scripts/evaluate.py requires 0..66)
    ch = ds.createVariable("channel", "i4", ("channel",))
    ch[:] = np.arange(contract.N_CHANNELS)
    ch.names = " ".join(contract.CHANNEL_NAMES)
    ds.createVariable("lat", "f8", ("lat",))[:] = lat
    ds.createVariable("lon", "f8", ("lon",))[:] = lon
    return ds, v


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ics", type=int, nargs="*", help="time indices (default: every --ic-stride days)")
    p.add_argument("--ic-stride", type=int, default=7)
    p.add_argument("--leads", type=int, default=10)
    p.add_argument("--members", type=int, default=16)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--det", action="store_true", help="also write the eps = 0 run (<out>_det.nc)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    dev = torch.device(a.device)

    mean = np.load(os.path.join(a.stats_dir, "global_mean.npy")).astype(np.float64)
    std = np.load(os.path.join(a.stats_dir, "global_std.npy")).astype(np.float64)
    state = torch.load(a.ckpt, map_location=dev)
    with nc.Dataset(a.test_file) as src:
        T = src["fields"].shape[0]
        lat, lon = np.array(src["latitude"][:]), np.array(src["longitude"][:])
        time = src["time"]
        dates = nc.num2date(time[:], units=time.units, calendar=getattr(time, "calendar", "standard"))
        doy_all = np.array([d.timetuple().tm_yday - 1 for d in dates])
        H, W = src["fields"].shape[-2:]
        model = build(state["config"], H, W, mean, std).to(dev)
        if "projection" in state:  # step 6 checkpoint (train_projection.py): frozen CFM + projection ODE
            from pgcfm.projection import ODEProjection
            from scripts.train_projection import projection_config
            model.cfm = ODEProjection(model.cfm, projection_config(state["projection"]), H, W).to(dev)
        model.load_state_dict(state["model"])
        if "latent_config" in state:  # step 7 checkpoint (train_latent.py): DCT encoder -> latent CFM -> decoder
            from pgcfm.latent import LatentConfig, LatentPGCFM
            lc = {k: tuple(v) if isinstance(v, list) else v for k, v in state["latent_config"].items()}
            model.cfm = LatentPGCFM(LatentConfig(**lc), H, W).to(dev)
            model.cfm.load_state_dict(state["latent"])
        model.eval()
        ics = a.ics if a.ics else list(range(0, T - a.leads, a.ic_stride))
        K = a.leads
        out, var = open_out(a.out, len(ics), K, a.members, H, W, lat, lon, ics)
        det = open_out(a.out.replace(".nc", "_det.nc"), len(ics), K, 0, H, W, lat, lon, ics) if a.det else None
        norm = lambda f: contract.normalize(f, mean, std).astype(np.float32)  # noqa: E731
        denorm = lambda f: contract.denormalize(f, mean, std, exact_inverse=True)  # noqa: E731
        for n, t0 in enumerate(ics):
            seq = norm(np.array(src["fields"][t0:t0 + K + 1], dtype=np.float64))
            dyn = torch.from_numpy(seq[:, contract.DYNAMIC_CHANNELS]).to(dev)
            static = torch.from_numpy(seq[0, contract.STATIC_CHANNELS]).to(dev)[None]
            doy = torch.from_numpy(doy_all[t0:t0 + K + 1])[None].to(dev)
            # forcing mode: tisr_{k+1} is known in advance (astronomical), taken from the file
            tisr = dyn[None, 1:, TISR_DYN:TISR_DYN + 1] if model.cfm.cfg.tisr_mode == "forcing" else None
            ens = rollout(model.cfm, dyn[None, 0], static, doy, a.members, tisr=tisr,
                          steps=a.steps, seed=a.seed + t0)[0]  # (M,K,66,H,W)
            full = assemble_full_state(ens, static[0]).cpu().numpy()
            ic = assemble_full_state(dyn[0], static[0]).cpu().numpy()
            var[n, 0] = denorm(np.broadcast_to(ic, (a.members,) + ic.shape))
            for k in range(K):
                var[n, k + 1] = denorm(full[:, k])
            if det:
                d = deterministic_rollout(model.cfm, dyn[None, 0], static, doy, tisr, a.steps)[0]
                d = assemble_full_state(d, static[0]).cpu().numpy()
                det[1][n, 0] = denorm(ic)
                det[1][n, 1:] = denorm(d)
            print(f"IC {t0}: done", flush=True)
        out.close()
        if det:
            det[0].close()


if __name__ == "__main__":
    main()
