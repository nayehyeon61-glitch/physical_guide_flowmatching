"""A small balanced toy atmosphere for testing the evaluation suite.

Unlike ``scripts/make_synthetic_era5.py`` (independent random channels), the
fields here are tied together the way the diagnostics expect:

* temperature = seasonal/meridional base + 3 vertical modes of AR(1) anomalies,
* geopotential from msl and T by the discrete hypsometric equation (exact),
* winds = geostrophic wind of that geopotential (tapered at the equator)
  + a small ageostrophic AR(1) part,
* q, t2m, sp, tcwv, tisr, skt diagnosed from the above.

The latent state advects eastward and decays (AR(1)), so a model that knows the
dynamics beats persistence and climatology, and a member-wise stochastic
propagation of the latent state is a calibrated ensemble by construction.
The failure-mode "models" in ``MODELS`` each break one property on purpose,
so the suite can be checked for which metric catches which weakness.

Values are plausible, not realistic: use this only to test the evaluation code.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from .. import contract

C = contract.CONST
H, W = contract.N_LAT, contract.N_LON
LAT = np.deg2rad(contract.LATITUDES)[:, None]
P = contract.LEVELS_PA  # 50 -> 1000 hPa
N_LAT_MODES = 6  # 3 temperature modes, msl, 2 ageostrophic wind fields

# Vertical structure of the three temperature modes (rows: mode, cols: level).
_s = np.log(P / P[-1])
VERT_MODES = np.stack([np.ones_like(_s), 1 + _s / 3, np.cos(np.pi * _s / _s[0])])
LATENT_AMP = np.array([3.0, 2.0, 1.5, 700.0, 1.5, 1.5])  # K, K, K, Pa, m/s, m/s
SHIFT = np.array([4, 3, 5, 4, 2, 2])  # eastward grid cells per day
RHO = 0.85


def smooth_noise(rng: np.random.Generator, n: int, sigma=(3.0, 4.0)) -> np.ndarray:
    """n unit-variance fields, Gaussian-filtered white noise (periodic in lon)."""
    out = np.empty((n, H, W))
    for i in range(n):
        f = gaussian_filter(rng.standard_normal((H, W)), sigma, mode=("nearest", "wrap"))
        out[i] = f / f.std()
    return out


def orography() -> np.ndarray:
    la, lo = np.meshgrid(contract.LATITUDES, contract.LONGITUDES, indexing="ij")
    orog = np.zeros((H, W))
    for c_lat, c_lon, h in [(30, 90, 5000), (-20, 290, 4000), (45, 260, 2500), (-75, 0, 3000)]:
        orog += h * np.exp(-(((la - c_lat) / 12) ** 2 + ((lo - c_lon) / 15) ** 2))
    return orog * C.gravity


OROG = orography()


def _ddx(f):
    cos = np.cos(LAT)
    d = (np.roll(f, -1, -1) - np.roll(f, 1, -1)) / (2 * np.deg2rad(360.0 / W) * C.earth_radius)
    return np.where(cos > 1e-6, d / np.maximum(cos, 1e-6), 0.0)


def _ddy(f):
    return np.gradient(f, np.deg2rad(180.0 / (H - 1)) * C.earth_radius, axis=-2)


def tisr(doy: float) -> np.ndarray:
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (doy + 284) / 365.0)
    phi = LAT[:, 0]
    h0 = np.arccos(np.clip(-np.tan(phi) * np.tan(decl), -1, 1))
    q = (1361.0 / np.pi) * (h0 * np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.sin(h0))
    return np.repeat((q * 86400.0)[:, None], W, axis=1)


def decode(latent: np.ndarray, doy: float) -> np.ndarray:
    """Latent (6, H, W) in physical units -> 67-channel KAI state."""
    idx = contract.CHANNEL_INDEX
    out = np.empty((contract.N_CHANNELS, H, W))
    season = np.sin(2 * np.pi * (doy - 80) / 365.0)
    pr = (P / 1e5)[:, None, None]
    t_mean = contract.REFERENCE_MEAN[contract.level_slice("t")][:, None, None]
    T = (t_mean + 18 * (np.cos(LAT) ** 2 - 0.5) * pr + 6 * season * np.sin(LAT) * pr
         + np.einsum("jk,jhw->khw", VERT_MODES, latent[:3]))
    q_mean = contract.REFERENCE_MEAN[contract.level_slice("q")][:, None, None]
    q = q_mean * 1.6 * np.cos(LAT) ** 2 * np.exp(0.08 * (T - t_mean))
    Tv = T * (1 + 0.608 * q)
    msl = 101325.0 + latent[3]
    z = np.empty_like(T)
    z[-1] = C.r_dry * Tv[-1] * np.log(msl / P[-1])
    for i in range(len(P) - 2, -1, -1):  # integrate upward, discrete hypsometric
        z[i] = z[i + 1] + C.r_dry * 0.5 * (Tv[i] + Tv[i + 1]) * np.log(P[i + 1] / P[i])
    f = 2 * C.omega * np.sin(LAT)
    taper = 1 - np.exp(-((np.rad2deg(LAT) / 12.0) ** 4))
    f_safe = np.where(np.abs(f) < 1e-12, 1e-12, f)
    u = -_ddy(z) / f_safe * taper + latent[4]
    v = _ddx(z) / f_safe * taper + latent[5]
    for name, val in (("z", z), ("q", q), ("t", T), ("u", u), ("v", v)):
        out[contract.level_slice(name)] = val
    out[idx["t2m"]] = T[-1] - 1.0
    out[idx["msl"]] = msl
    out[idx["sp"]] = msl * np.exp(-OROG / (C.r_dry * Tv[-1]))
    dp = np.gradient(P)[:, None, None]
    out[idx["tcwv"]] = (q * dp).sum(0) / C.gravity
    out[idx["orog"]] = OROG
    out[idx["tisr"]] = tisr(doy)
    out[idx["skt"]] = T[-1] + 0.5
    return out.astype(np.float32)


def advance(latent: np.ndarray, noise: np.ndarray, noise_scale: float = 1.0) -> np.ndarray:
    """One day: shift east, decay, add AR(1) innovation."""
    out = np.empty_like(latent)
    for j in range(N_LAT_MODES):
        out[j] = RHO * np.roll(latent[j], SHIFT[j], axis=-1)
    return out + noise_scale * np.sqrt(1 - RHO ** 2) * LATENT_AMP[:, None, None] * noise


@dataclass
class World:
    init_times: np.ndarray  # datetime64[D] (N,)
    init_doy: np.ndarray  # (N,)
    ic_latent: np.ndarray  # (N, 6, H, W)
    truth_latent: np.ndarray  # (N, L+1, 6, H, W), index 0 = IC
    truth: np.ndarray  # (N, L+1, 67, H, W)


def make_world(n_ic=3, n_lead=6, year=2020, seed=0) -> World:
    rng = np.random.default_rng(seed)
    doys = np.linspace(15, 320, n_ic).round().astype(int)
    lat0 = np.stack([LATENT_AMP[:, None, None] * smooth_noise(rng, N_LAT_MODES) for _ in range(n_ic)])
    traj = np.empty((n_ic, n_lead + 1, N_LAT_MODES, H, W))
    traj[:, 0] = lat0
    for k in range(n_lead):
        for n in range(n_ic):
            traj[n, k + 1] = advance(traj[n, k], smooth_noise(rng, N_LAT_MODES))
    truth = np.stack([np.stack([decode(traj[n, k], doys[n] + k) for k in range(n_lead + 1)]) for n in range(n_ic)])
    init = np.datetime64(f"{year}-01-01") + (doys - 1).astype("timedelta64[D]")
    return World(init, doys, lat0, traj, truth)


def climatology_state(doy: float) -> np.ndarray:
    return decode(np.zeros((N_LAT_MODES, H, W)), doy)


# ----------------------------------------------------------------------
# Failure-mode models. Each returns (N, L, M, 67, H, W) for leads 1..L.
# ----------------------------------------------------------------------
def _ensemble(world: World, n_members: int, seed: int, step, post=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N, L1 = world.truth.shape[:2]
    out = np.empty((N, L1 - 1, n_members, contract.N_CHANNELS, H, W), dtype=np.float32)
    for n in range(N):
        for m in range(n_members):
            lat = world.ic_latent[n].copy()
            for k in range(L1 - 1):
                lat = step(lat, rng)
                x = decode(lat, world.init_doy[n] + k + 1)
                out[n, k, m] = post(x, rng) if post else x
    return out


def perfect(world, M=8, seed=1):
    return _ensemble(world, M, seed, lambda z, r: advance(z, smooth_noise(r, N_LAT_MODES)))


def zero_noise(world, seed=1):
    """CFM with epsilon = 0 analogue: the conditional-mean path, one member."""
    return _ensemble(world, 1, seed, lambda z, r: advance(z, 0.0))


def underdispersive(world, M=8, seed=1):
    return _ensemble(world, M, seed, lambda z, r: advance(z, smooth_noise(r, N_LAT_MODES), 0.3))


def quasi_static(world, M=8, seed=1):
    """Moves 10% of the true tendency per step (climate_diffusion failure, PR #2)."""
    def step(z, r):
        return z + 0.1 * (advance(z, 0.0) - z) + 0.3 * np.sqrt(1 - RHO ** 2) * LATENT_AMP[:, None, None] * smooth_noise(r, N_LAT_MODES)
    return _ensemble(world, M, seed, step)


def unbalanced(world, M=8, seed=1):
    """Calibrated-looking spread from channel-wise iid noise instead of the dynamics.

    Mimics a generator whose per-channel Gaussian epsilon leaks into the output:
    z, t, u, v get independent perturbations, breaking hydrostatic and
    geostrophic balance while marginal spread stays reasonable.
    """
    def post(x, r):
        x = x.copy()
        for name, amp in (("z", 150.0), ("t", 1.2), ("u", 3.0), ("v", 3.0)):
            sl = contract.level_slice(name)
            x[sl] += amp * smooth_noise(r, contract.N_LEVELS, sigma=(1.5, 2.0))
        return x
    return _ensemble(world, M, seed, lambda z, r: advance(z, smooth_noise(r, N_LAT_MODES), 0.6), post)


def blurry(world, M=8, seed=1):
    """Perfect members with the small scales smoothed away (spectral deficit)."""
    def post(x, r):
        dyn = contract.DYNAMIC_CHANNELS
        x = x.copy()
        x[dyn] = gaussian_filter(x[dyn], (0, 2.5, 2.5), mode=("nearest", "nearest", "wrap"))
        return x
    return _ensemble(world, M, seed, lambda z, r: advance(z, smooth_noise(r, N_LAT_MODES)), post)


MODELS = {
    "perfect": perfect,
    "zero-noise": zero_noise,
    "underdispersive": underdispersive,
    "quasi-static": quasi_static,
    "unbalanced": unbalanced,
    "blurry": blurry,
}
