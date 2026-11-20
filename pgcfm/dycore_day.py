"""One day of the dry primitive equations, integrated by NeuralGCM's dynamical core (dinosaur), on KAI's daily
5.625 deg contract (docs/kai_dycore_integration.md).

Why: the tendency core's physical terms G(x) (pgcfm.dyncore.OperatorFeatures) are the INSTANTANEOUS tendency of the
pressure-level operator at the day's start, never integrated. Here the same four equations (momentum u, v,
thermodynamic T, moisture q as a tracer; continuity and hydrostatic balance are built into the dycore) are
integrated in time over the day by a published, stable solver: spectral T21 on the data's own 64 x 32 equiangular
grid, sigma levels, semi-implicit IMEX RK (imex_rk_sil3), hyperdiffusion, as in dinosaur's notebook
``weather_forecast_on_era5`` (Kochkov et al. 2024). No learned parameters, no radiation, latent heat or friction.

The day's integrated increment of the 12 pressure levels

    D(x) - x  :=  P(S_24h) - P(S_0)        S_0 = the state x interpolated to sigma and truncated to T21,
                                            S_24h = S_0 integrated 24 h, P = back to the pressure levels

(the round trip of the start state is subtracted, so the interpolation and truncation errors cancel) is returned in
SI units per second for u, v, T, q (the 48 channels of G, so that the two hints are the same size) and, for the
forecast-base form, also for z (12 levels) and sp.

The core is torch, the dycore JAX: the dycore runs without gradient on numpy arrays (a hint and a base, like G).

A finer data grid (KAI's 1.5 deg, 121 x 240 with the poles; the Kalman island + version, docs/kai_island_plus.md):
the state is averaged conservatively onto the dycore's own coarse grid (T21 on 64 x 32, the 5.625 deg grid, or T42 on
128 x 64 with ``coarse=64``), integrated there as above, and the day's increment is interpolated back (bilinear in
latitude and periodic longitude). The hint then carries the same large-scale, 24 h integrated tendency as on the
5.625 deg grid, at the cost of one T21 day per state (0.37 s per 8 states on 4 CPU threads); the fine scales are
KAI's and the core's.
"""

import functools
import os
from typing import Dict, Optional

import numpy as np

# JAX on the CPU uses its own thread pool next to torch's; the core and the dycore run one after the other

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from dinosaur import coordinate_systems, primitive_equations, scales, sigma_coordinates  # noqa: E402
from dinosaur import spherical_harmonic, time_integration, vertical_interpolation, filtering  # noqa: E402

from . import contract  # noqa: E402

units = scales.units
LEVELS = np.asarray(contract.LEVELS_HPA, dtype=np.float64)  # 50 .. 1000 hPa (top first, as dinosaur wants)
OUT_VARS = ("u", "v", "t", "q", "z")
# sigma -> pressure with constant extrapolation: a pressure level below the ground (700 hPa over Antarctica) takes the
# lowest sigma layer's value (dinosaur's default returns NaN there); the increment is the difference of two such values
_CONST = vertical_interpolation.vectorize_vertical_interpolation(vertical_interpolation.interp)


def _cell_edges(c, lo, hi):
    """Edges of the cells around ascending centres c, the outer ones clipped to [lo, hi]."""
    c = np.asarray(c, np.float64)
    return np.clip(np.concatenate([[c[0] - (c[1] - c[0]) / 2], (c[1:] + c[:-1]) / 2, [c[-1] + (c[-1] - c[-2]) / 2]]), lo, hi)


def conservative_matrix(src, dst, periodic=False, sine=False):
    """(len(dst), len(src)): the area-weighted average of the source cells over every destination cell (rows sum to
    1). Latitudes with ``sine`` (area = d sin(lat)); longitudes ``periodic`` (360 deg)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    if periodic:
        ds, dd = 360.0 / len(src), 360.0 / len(dst)
        es = np.stack([src - ds / 2, src + ds / 2], 1)
        ed = np.stack([dst - dd / 2, dst + dd / 2], 1)
        M = np.zeros((len(dst), len(src)))
        for sh in (-360.0, 0.0, 360.0):
            lo = np.maximum(ed[:, None, 0], es[None, :, 0] + sh)
            hi = np.minimum(ed[:, None, 1], es[None, :, 1] + sh)
            M += np.clip(hi - lo, 0, None)
    else:
        es, ed = _cell_edges(src, -90.0, 90.0), _cell_edges(dst, -90.0, 90.0)
        f = (lambda e: np.sin(np.deg2rad(e))) if sine else (lambda e: e)
        lo = np.maximum(ed[:-1, None], es[None, :-1])
        hi = np.minimum(ed[1:, None], es[None, 1:])
        M = np.clip(f(np.maximum(hi, lo)) - f(lo), 0, None)
    return M / M.sum(1, keepdims=True)


def linear_matrix(src, dst, periodic=False):
    """(len(dst), len(src)): linear interpolation from the source centres to the destination centres (constant
    beyond the outer source centres; ``periodic``: longitudes over 360 deg)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    M = np.zeros((len(dst), len(src)))
    if periodic:
        d = 360.0 / len(src)
        t = ((dst - src[0]) % 360.0) / d
        i0 = np.floor(t).astype(int) % len(src)
        r = t - np.floor(t)
        M[np.arange(len(dst)), i0] += 1 - r
        M[np.arange(len(dst)), (i0 + 1) % len(src)] += r
        return M
    for k, v in enumerate(dst):
        if v <= src[0]:
            M[k, 0] = 1.0
        elif v >= src[-1]:
            M[k, -1] = 1.0
        else:
            i = int(np.searchsorted(src, v)) - 1
            r = (v - src[i]) / (src[i + 1] - src[i])
            M[k, i], M[k, i + 1] = 1 - r, r
    return M


class CoarseGrid:
    """A fine data grid (lat, lon) <-> the dycore's coarse equiangular grid of ``n_lat`` x 2 n_lat cells without
    poles (centres -90 + (i + 1/2) 180 / n_lat, 0 + j 360 / (2 n_lat)): conservative averages down, linear back."""

    def __init__(self, lat, lon, n_lat=32):
        lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        self.flip = lat[0] > lat[-1]            # the dycore wants S -> N
        latf = lat[::-1] if self.flip else lat
        self.lat_c = -90.0 + (np.arange(n_lat) + 0.5) * 180.0 / n_lat
        self.lon_c = np.arange(2 * n_lat) * 360.0 / (2 * n_lat)
        self.down_lat = conservative_matrix(latf, self.lat_c, sine=True)          # (Hc, H)
        self.down_lon = conservative_matrix(lon, self.lon_c, periodic=True)       # (Wc, W)
        self.up_lat = linear_matrix(self.lat_c, latf)                              # (H, Hc)
        self.up_lon = linear_matrix(self.lon_c, lon, periodic=True)                # (W, Wc)
        self.shape = (n_lat, 2 * n_lat)

    def down(self, x):
        """(..., H, W) fine -> (..., Hc, Wc)."""
        x = x[..., ::-1, :] if self.flip else x
        return np.einsum("ah,...hw,bw->...ab", self.down_lat, x, self.down_lon, optimize=True)

    def up(self, x):
        """(..., Hc, Wc) coarse -> (..., H, W)."""
        y = np.einsum("ha,...ab,wb->...hw", self.up_lat, x, self.up_lon, optimize=True)
        return y[..., ::-1, :] if self.flip else y


class DinosaurDay:
    """``__call__(fields)`` -> increments over ``hours`` of the dry primitive equations.

    fields: dict of numpy arrays in SI on KAI's grid, lat S -> N, lon 0 .. 354.375:
        u, v, t, q, z: (B, 12, H, W) on 50 .. 1000 hPa;  sp: (B, H, W) [Pa]
    returns: dict u, v, t, q, z (B, 12, H, W) and sp (B, H, W): increment over the day [unit] (not per second).
    """

    def __init__(self, orog_geopot: np.ndarray, layers: int = 16, dt_minutes: float = 30.0, hours: float = 24.0,
                 dfi_hours: float = 0.0, tau_hours: Optional[float] = None, max_wavenumber: int = 21,
                 sp_mode: str = "z"):
        H, W = orog_geopot.shape
        # sp_mode "z": the start's surface pressure is where the data's geopotential levels meet the MODEL's (T21,
        # filtered) orography (dinosaur's get_surface_pressure), so the start is in hydrostatic balance with the
        # dycore's own lower boundary; "data": ERA5's sp (consistent with the real orography, not the model's: the
        # dycore then adjusts sp over the mountains in the first hours and every level's z gets a fixed offset)
        self.sp_mode = sp_mode
        assert (H, W) in ((32, 64), (64, 128)), "T21 on 64 x 32 or T42 on 128 x 64 (fine grids: DycoreFeatures' CoarseGrid)"
        self.grid = spherical_harmonic.Grid.construct(max_wavenumber, H // 2, latitude_spacing="equiangular")
        assert self.grid.nodal_shape == (W, H), self.grid.nodal_shape
        self.coords = coordinate_systems.CoordinateSystem(self.grid, sigma_coordinates.SigmaCoordinates.equidistant(layers))
        self.specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
        self.pcoords = vertical_interpolation.PressureCoordinates(LEVELS)
        self.ref_temps = self.specs.nondimensionalize(250.0 * units.degK) * np.ones(layers)
        g = 9.80616
        orog_m = (orog_geopot / g).T[None]  # (1, W, H) nodal order: lon, lat
        orog = self.grid.to_modal(self.specs.nondimensionalize(orog_m * units.m))
        self.orography = filtering.exponential_filter(self.grid, order=2)(orog)
        self.eq = primitive_equations.PrimitiveEquations(self.ref_temps, self.orography, self.coords, self.specs)
        dt = self.specs.nondimensionalize(dt_minutes * units.minute)
        if tau_hours is None:  # dinosaur's notebook scaling of the hyperdiffusion time with resolution
            tau_hours = 8.6 / (2.4 ** np.log2(H / 128))
        tau = self.specs.nondimensionalize(tau_hours * units.hour)
        filt = time_integration.horizontal_diffusion_step_filter(self.grid, dt=dt, tau=tau, order=2)
        step = time_integration.step_with_filters(time_integration.imex_rk_sil3(self.eq, dt), [filt])
        n = int(round(hours * 60.0 / dt_minutes))
        self.n_steps, self.hours = n, hours
        self.integrate = time_integration.repeated(step, n)
        self.dfi = None
        if dfi_hours > 0:
            span = self.specs.nondimensionalize(dfi_hours * units.hour)
            self.dfi = time_integration.digital_filter_initialization(
                equation=self.eq, ode_solver=time_integration.imex_rk_sil3, filters=[filt], time_span=span,
                cutoff_period=span, dt=dt)
        self._fn = jax.jit(jax.vmap(self._one))

    # ---------------------------------------------------------------- one state (vmapped over the batch)
    def _nd(self, x, unit):
        return self.specs.nondimensionalize(x * unit)

    def _to_state(self, u, v, t, q, sp_pa):
        sp_hpa = sp_pa / 100.0
        nodal = vertical_interpolation.interp_pressure_to_sigma(
            {"u": u, "v": v, "t": t, "q": q}, self.pcoords, self.coords.vertical, sp_hpa)
        vor, div = spherical_harmonic.uv_nodal_to_vor_div_modal(self.grid, nodal["u"], nodal["v"])
        tv = self.grid.to_modal(nodal["t"] - self.ref_temps[:, None, None])
        lsp = self.grid.to_modal(jnp.log(sp_pa / self._nd_pa))
        return primitive_equations.State(vorticity=vor, divergence=div, temperature_variation=tv,
                                         log_surface_pressure=lsp, tracers={"q": self.grid.to_modal(nodal["q"])})

    def _to_pressure(self, s):
        """State -> u, v, T, q, z on the 12 pressure levels (nondimensional), sp (hPa)."""
        uu, vv = spherical_harmonic.vor_div_to_uv_nodal(self.grid, s.vorticity, s.divergence)
        t = self.grid.to_nodal(s.temperature_variation) + self.ref_temps[:, None, None]
        z = primitive_equations.get_geopotential_on_sigma(
            t, nodal_orography=self.grid.to_nodal(self.orography), sigma=self.coords.vertical,
            gravity_acceleration=self.specs.gravity_acceleration, ideal_gas_constant=self.specs.ideal_gas_constant)
        q = self.grid.to_nodal(s.tracers["q"])
        sp_hpa = jnp.exp(self.grid.to_nodal(s.log_surface_pressure)) * self._nd_pa / 100.0
        p = vertical_interpolation.interp_sigma_to_pressure({"u": uu, "v": vv, "t": t, "q": q, "z": z},
                                                            self.pcoords, self.coords.vertical, sp_hpa, _CONST)
        return p, sp_hpa

    def _one(self, u, v, t, q, z, sp_pa):
        """Nondimensional nodal inputs (12, W, H), sp in Pa (1, W, H) -> increments (nondimensional; sp in hPa)."""
        if self.sp_mode == "z":
            sp_pa = 100.0 * vertical_interpolation.get_surface_pressure(
                self.pcoords, z, self.grid.to_nodal(self.orography), self.specs.gravity_acceleration)
        s0 = self._to_state(u, v, t, q, sp_pa)
        p0, sp0 = self._to_pressure(s0)
        s1 = self.integrate(self.dfi(s0) if self.dfi is not None else s0)
        p1, sp1 = self._to_pressure(s1)
        out = {k: p1[k] - p0[k] for k in p0}
        out["sp"] = sp1 - sp0
        return out

    @functools.cached_property
    def _nd_pa(self):
        # log surface pressure in dinosaur is the log of the nondimensional pressure
        return 1.0 / float(self.specs.nondimensionalize(1.0 * units.pascal))

    _UNIT = {"u": units.m / units.s, "v": units.m / units.s, "t": units.degK, "q": units.dimensionless,
             "z": units.m ** 2 / units.s ** 2}

    def __call__(self, f: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        tr = lambda a: np.swapaxes(np.asarray(a, np.float64), -1, -2)  # (.., H, W) <-> (.., W, H)  # noqa: E731
        args = [self._nd(tr(f[k]), self._UNIT[k]) for k in ("u", "v", "t", "q", "z")]
        sp = tr(f["sp"])[:, None]
        out = jax.device_get(self._fn(*[jnp.asarray(a, jnp.float32) for a in args], jnp.asarray(sp, jnp.float32)))
        res = {k: tr(np.asarray(self.specs.dimensionalize(np.asarray(out[k], np.float64), self._UNIT[k]).magnitude))
               for k in OUT_VARS}
        res["sp"] = tr(np.asarray(out["sp"], np.float64))[:, 0] * 100.0
        return res


class DycoreFeatures:
    """The integrated physical terms of the tendency core: the same interface as pgcfm.dyncore.OperatorFeatures
    (``__call__(x)`` -> (B, 48, H, W) hint, ``scale``, ``estimate_scale``, ``to``), but the 48 channels are the
    dycore's 24 h INTEGRATED increment of u, v, T, q on the 12 levels (per day, SI), divided by its RMS on ERA5,
    clamped at ``clamp``, instead of the instantaneous operator tendency G(x).

    ``increment(x)``: the same integration as a normalized increment of the 66 dynamic channels (z, q, t, u, v on
    the levels and sp from the dycore; msl takes the sp increment (hydrostatic reduction at fixed temperature);
    t2m, tisr, tcwv, skt: zero), for the forecast-base form (the core corrects x + D(x) - x). Clamped at
    ``inc_clamp`` channel standard deviations, NaN -> 0 (a drifted rollout state can break the dry dycore: a free
    run of it alone gave NaN at day 4, docs/kai_dycore_integration.md).

    x: normalized (B, 66, H, W) dynamic channels; mean, std: (67,) of the full contract. One-entry cache: the
    hint and the increment of the same state tensor come from one integration."""
    daily = True  # a daily quantity: the core holds it over the day's substeps (DynCore.day_hint)


    def __init__(self, mean, std, orog_geopot, scale=None, clamp: float = 5.0, inc_clamp: float = 3.0,
                 layers: int = 12, dt_minutes: Optional[float] = None, hours: float = 24.0, lat=None, lon=None,
                 coarse: Optional[int] = None):
        import torch
        self.torch = torch
        self.mean = np.asarray(mean, np.float64)
        self.std = np.asarray(std, np.float64) + 1e-12
        self.dyn = np.asarray(contract.DYNAMIC_CHANNELS)
        orog = np.asarray(orog_geopot, np.float64)
        # another grid than the dycore's own (5.625 deg, 32 x 64): integrated on a coarse grid of ``coarse`` latitudes
        # (default 32 = T21; 64 = T42) and interpolated back (CoarseGrid)
        self.cg = None
        if orog.shape != (32, 64) or (coarse or 32) != 32:
            assert lat is not None and lon is not None, "a fine grid: the latitudes and longitudes of the data"
            self.cg = CoarseGrid(lat, lon, coarse or 32)
            orog = self.cg.down(orog)
        n_c = orog.shape[0]
        mw = {32: 21, 64: 42}[n_c]
        dt = dt_minutes or {32: 90.0, 64: 45.0}[n_c]
        self.day = DinosaurDay(orog, layers=layers, dt_minutes=dt, hours=hours, max_wavenumber=mw)
        self.scale, self.clamp, self.inc_clamp = scale, clamp, inc_clamp
        self.key, self.val = None, None
        self.store, self.hits, self.pad = {}, 0, 8
        self.noise_scale, self.noise_gen = None, None  # (1, 66, 1, 1) normalized amplitude of the input noise
        ci = contract.CHANNEL_INDEX
        self.full_of = {v: contract.level_slice(v) for v in ("z", "q", "t", "u", "v")}
        self.sp, self.msl = ci["sp"], ci["msl"]
        self.n_nan = 0  # states whose integration gave a non-finite value (counted, set to 0)

    def to(self, device):
        return self

    def _integrate(self, x):
        """x (B, 66, H, W) torch normalized -> (raw48 SI per day (B, 48, H, W) numpy, inc66 normalized numpy)."""
        x = x.detach().cpu()
        if self.noise_scale is not None:
            # noise BEFORE the equations (owner 2026-11-15): the dycore integrates x + a xi, so the hint carries what
            # the equations make of the perturbation in a day (unbalanced parts radiate away, unstable flow grows it)
            F = self.torch.nn.functional
            xi = self.torch.randn(x.shape, generator=self.noise_gen)
            xi = F.avg_pool2d(F.pad(F.pad(xi, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1), mode="reflect"), 3, 1) * 3.0
            x = x + self.noise_scale * xi
        xn = x.double().numpy()
        B = xn.shape[0]
        full = np.zeros((B, len(self.mean)) + xn.shape[-2:])
        full[:, self.dyn] = xn
        phys = full * self.std[None, :, None, None] + self.mean[None, :, None, None]
        if self.cg is not None:   # a fine grid: the state's conservative average on the dycore's grid
            phys = self.cg.down(phys)
        f = {v: phys[:, self.full_of[v]] for v in self.full_of}
        f["sp"] = phys[:, self.sp]
        r = self.day(f)
        self.last_raw_c = None
        if self.cg is not None:   # the day's increment back on the data's grid (a failed state stays non-finite)
            self.last_raw_c = np.concatenate([r[v] for v in ("u", "v", "t", "q")], 1)
            r = {k: self.cg.up(v) for k, v in r.items()}
        bad = ~np.all(np.isfinite(np.concatenate([r[k].reshape(B, -1) for k in r], 1)), 1)
        self.n_nan += int(bad.sum())
        self.last_bad = bad
        inc = np.zeros((B, len(self.mean)) + xn.shape[-2:])
        for v, sl in self.full_of.items():
            inc[:, sl] = r[v]
        inc[:, self.sp] = r["sp"]
        inc[:, self.msl] = r["sp"]
        inc = np.nan_to_num(inc / self.std[None, :, None, None], nan=0.0, posinf=0.0, neginf=0.0)
        inc[bad] = 0.0
        inc = np.clip(inc, -self.inc_clamp, self.inc_clamp)
        raw = np.concatenate([r[v] for v in ("u", "v", "t", "q")], 1)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw[bad] = 0.0
        if self.last_raw_c is not None:
            self.last_raw_c = np.nan_to_num(self.last_raw_c, nan=0.0, posinf=0.0, neginf=0.0)
            self.last_raw_c[bad] = 0.0
        return raw, inc[:, self.dyn]

    def _get(self, x):
        if self.key is not x:
            self.key, self.val = x, (self._lookup(x) if self.store else self._integrate(x))
        return self.val

    # ---- a store of the integrated hint of DATA states (a memory core warms its memory up on the W days before
    # the IC: data states, the same every time they are drawn; only the rollout's own states need a new integration)
    @staticmethod
    def _keys(x):
        a = np.ascontiguousarray(x.detach().cpu().float().numpy())
        return [hash(r.tobytes()) for r in a]

    def prefill(self, states, chunk: int = 8):
        """Integrate the data ``states`` (N, 66, H, W) once and keep their hint (48 channels, float32) by content."""
        for i in range(0, states.shape[0], chunk):
            xs = states[i:i + chunk]
            raw, _ = self._integrate(xs)
            if self.cg is not None:   # a fine grid: the coarse hint is kept (5.6 MB a state at 1.5 deg, 0.4 MB coarse)
                raw = self.last_raw_c
            for k, r in zip(self._keys(xs), raw):
                self.store[k] = r.astype(np.float32)

    def _lookup(self, x):
        keys = self._keys(x)
        miss = [i for i, k in enumerate(keys) if k not in self.store]
        raw = np.zeros((len(keys), 48) + tuple(x.shape[-2:]))
        for i, k in enumerate(keys):
            if k in self.store:
                raw[i] = self.store[k] if self.cg is None else self.cg.up(self.store[k])
        self.hits += len(keys) - len(miss)
        if miss:
            n, B = len(miss), self.pad
            idx = miss + [miss[-1]] * ((-n) % B)  # fixed batch sizes: every new shape is a new JAX compilation
            # detached before indexing: an index op on a state with gradient would be recorded (and saved) on a
            # miss but not on a cache hit, and a checkpointed substep would then save a different number of tensors
            raw[miss] = self._integrate(x.detach()[idx])[0][:n]
        return raw, None  # no increment from the store: the forecast-base form does not use it

    def estimate_scale(self, states, chunk: int = 8):
        sq, n = 0.0, 0
        for i in range(0, states.shape[0], chunk):
            raw, _ = self._integrate(states[i:i + chunk])
            sq = sq + (raw ** 2).sum((0, 2, 3))
            n += raw.shape[0] * raw.shape[2] * raw.shape[3]
        return self.torch.from_numpy(np.sqrt(sq / n)).clamp_min(1e-30).view(1, -1, 1, 1).float()

    def __call__(self, x):
        raw, _ = self._get(x)
        g = self.torch.from_numpy(raw).to(device=x.device, dtype=x.dtype) / self.scale.to(device=x.device, dtype=x.dtype)
        return g.clamp(-self.clamp, self.clamp)

    def increment(self, x):
        _, inc = self._get(x)
        return self.torch.from_numpy(inc).to(device=x.device, dtype=x.dtype)


class ComboFeatures:
    """The core's feature inputs side by side: [G(x) (instantaneous operator, optional), the integrated hint
    (dycore, optional), ``nz`` noise channels]. Noise: white N(0, 1) per channel and grid point, new at every call
    (= every day and member) while ``noise_on``; zeros otherwise (a deterministic run of the same core)."""

    @property
    def daily(self):
        """The dycore hint alone (no instantaneous G(x), no noise channels) is a daily quantity."""
        return self.op is None and self.dy is not None and not self.nz

    def __init__(self, op=None, dy=None, nz: int = 0, seed: int = 0, hnz: float = 0.0):
        import torch
        self.torch, self.op, self.dy, self.nz, self.noise_on = torch, op, dy, int(nz), False
        # hnz > 0 (owner 2026-11-15, "noise inside the equations"): the nz = 48 noise channels are not white noise but
        # what the dycore's 24 h integration makes of a perturbation of its start:
        # delta = hint(x + a xi) - hint(x), a = hnz_scale (1, 66, 1, 1). Zeros when the noise is off.
        self.hnz, self.hnz_scale = float(hnz), None
        if self.hnz and (dy is None or self.nz != 48):
            raise ValueError("hnz needs the dycore hint and nz = 48")
        self.gen = torch.Generator().manual_seed(seed)
        self.key, self.val = None, None
        self.ext = []  # (B, nz, H, W) tensors for the coming calls instead of noise (see __call__)

    @property
    def n_nan(self):
        return 0 if self.dy is None else self.dy.n_nan

    def to(self, device):
        if self.op is not None:
            self.op.to(device)
        return self

    def set_scale(self, gscale):
        """Split the core's buffer (1, C, 1, 1) back to the parts (a loaded core brings its own scales)."""
        i = 0
        for f in (self.op, self.dy):
            if f is not None:
                n = 48 if f is self.dy else gscale.shape[1] - self.nz - (48 if self.dy is not None else 0)
                f.scale = gscale[:, i:i + n]
                i += n

    def scale_buffer(self):
        parts = [f.scale for f in (self.op, self.dy) if f is not None]
        if self.nz:
            parts.append(self.torch.ones(1, self.nz, 1, 1))
        return self.torch.cat([q.cpu().float() for q in parts], 1)

    def __call__(self, x):
        if self.key is not x:
            parts = [f(x) for f in (self.op, self.dy) if f is not None]
            if self.nz and self.ext:
                # channels made beforehand for this call, one entry per coming call (the second-moment hint of a
                # two-stage forecast, scripts/precompute_m2.py); calls without an entry get zeros below
                z = self.ext.pop(0)
            elif self.nz and self.hnz and self.noise_on:
                F = self.torch.nn.functional
                xc = x.detach().cpu().float()
                xi = self.torch.randn(xc.shape, generator=self.gen)
                xi = F.avg_pool2d(F.pad(F.pad(xi, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1), mode="reflect"), 3, 1) * 3.0
                raw0 = self.dy._get(x)[0]
                raw1 = self.dy._integrate(xc + self.hnz_scale * xi)[0]
                d = self.torch.from_numpy(raw1 - raw0).float() / self.dy.scale.cpu().float() / self.hnz  # per unit of perturbation
                d[self.torch.from_numpy(np.asarray(self.dy.last_bad))] = 0.0  # a failed integration: no noise signal
                z = d.clamp(-self.dy.clamp, self.dy.clamp)
            elif self.nz:
                z = (self.torch.randn((x.shape[0], self.nz) + tuple(x.shape[-2:]), generator=self.gen)
                     if self.noise_on and not self.hnz else self.torch.zeros((x.shape[0], self.nz) + tuple(x.shape[-2:])))
            if self.nz:
                parts.append(z.to(x.dtype).to(x.device))
            self.key, self.val = x, self.torch.cat(parts, 1)
        return self.val
