"""Step 4: PG-CFM, the CFM transition coupled to the primitive-equation guide.

Ties the two components together as the paper prescribes (secs. 5.3, 6.1,
appendix C):

  * ``cfm.CFMTransition`` (torchcfm path + U-Net, step 2) generates the
    endpoint y_hat = x_{k+1} with a differentiable Heun integration, so the
    physics loss reaches theta: L_phys -> B_psi -> y_hat -> theta (no detach).
  * ``PhysicalBridge`` is the PINN path of eq. (9) between x_k and y_hat,
        B_psi(s) = (1 - a) I(x) + a I(y_hat) + a (1 - a) h_psi(a; x, y_hat, c),
    a = s / dt. It is exact at both endpoints by construction.
  * The residuals are those of step 3 (``physics.primitive`` through the
    finite-difference operators of ``physics.grid``) in the time-integrated
    form of eq. (18) for momentum, T and q, and as spatial diagnostics for
    mass continuity and hydrostatics.
  * ``Closure`` C_chi(x_k, c_k) is a low-capacity effective source,
    pre-trained on observed pairs and then frozen (sec. 5.2).

Choice that differs from the paper text: h_psi is a gridded CNN (one output
field per channel, conditioned on a) instead of a coordinate MLP, and the
spatial derivatives are finite differences on the KAI grid. A coordinate MLP
over 29k points x 12 levels per sample is too slow to run inside the CFM
training loop; ``physics.pinn`` keeps the coordinate PINN for the ablation.
With the time-integrated residual only A_j(B(s)) at the Simpson nodes is
needed, never d_s h_psi, so the gridded bridge loses nothing there.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from . import contract
from .cfm import TISR_DYN, CFMTransition, calendar_features
from .physics import grid as pgrid
from .physics import primitive

# Upper-air prognostic groups (in the 66 dynamic channels, orog removed).
_DYN = contract.DYNAMIC_CHANNELS
LEVEL_VARS = ("z", "q", "t", "u", "v")
LEVEL_DYN = {v: [_DYN.index(c) for c in range(*contract.level_slice(v).indices(contract.N_CHANNELS))]
             for v in LEVEL_VARS}
BRIDGE_DYN = [i for v in LEVEL_VARS for i in LEVEL_DYN[v]]  # 60 channels h_psi acts on
SP_DYN = _DYN.index(contract.CHANNEL_INDEX["sp"])
CLOSURE_EQS = ("u", "v", "T", "q")


@dataclass
class PhysicsConfig:
    """Weights of the joint objective eq. (12) and of the physics loss eq. (10)."""

    lambda_p: float = 0.1  # lambda_P, final value of the ramp
    ramp_iters: int = 500  # linear ramp 0 -> lambda_p (appendix C: "gradually")
    lambda_h: float = 1e-2  # lambda_H, bridge regularity ||h_psi||^2
    lambda_s: float = 0.0  # lambda_S, energy score fine-tuning (M = 2)
    gamma: Dict[str, float] = field(default_factory=lambda: {
        "u": 1.0, "v": 1.0, "T": 1.0, "mass": 1.0, "hyd": 1.0, "q": 0.0})  # q > 0: + moisture
    n_sub: int = 4  # Simpson intervals in eq. (18)
    sample_steps: int = 2  # Heun steps of the differentiable endpoint (NFE = 2x)
    control: str = "physics"  # "smooth": non-physical Laplacian penalty in place of L_phys (step 4 control)
    grad_ratio: float = 0.0  # > 0: cap ||grad_theta phys|| at grad_ratio x ||grad_theta cfm|| (step 4 fix)
    pole_cap_deg: float = 85.0  # residual rows poleward are dropped (docs/step3, E5)
    lat_weight: str = "area"
    virtual: bool = True  # T_v = T (1 + 0.608 q)


class _Embed(nn.Module):
    """Sinusoidal embedding of a scalar in [0, 1] (bridge time a)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=a.device, dtype=a.dtype) / half)
        ang = a[:, None] * freqs[None] * 1000.0
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def _conv(cin: int, cout: int, k: int = 3) -> nn.Conv2d:
    # circular in longitude is applied by the caller through padding
    return nn.Conv2d(cin, cout, k, padding=0)


class _PeriodicConv(nn.Module):
    """3x3 conv, circular in longitude and replicate in latitude."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = _conv(cin, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (1, 1, 0, 0), mode="circular")
        x = nn.functional.pad(x, (0, 0, 1, 1), mode="replicate")
        return self.conv(x)


class PhysicalBridge(nn.Module):
    """h_psi of eq. (9) on the grid, in normalized units of the 60 level channels.

    Input: x_k, y_hat (66 normalized each), orography, calendar, a. Output is
    multiplied by a (1 - a) by the caller, so the endpoints are untouched.
    The last layer starts at zero, i.e. training starts from the linear
    bridge that step 3 uses.
    """

    def __init__(self, hidden: int = 32, emb: int = 16):
        super().__init__()
        cin = 2 * contract.N_DYNAMIC + 1 + 2
        self.embed = _Embed(emb)
        self.inp = _PeriodicConv(cin, hidden)
        self.film = nn.Linear(emb, 2 * hidden)
        self.mid = _PeriodicConv(hidden, hidden)
        self.out = _PeriodicConv(hidden, len(BRIDGE_DYN))
        nn.init.zeros_(self.out.conv.weight)
        nn.init.zeros_(self.out.conv.bias)

    def forward(self, a: torch.Tensor, x: torch.Tensor, y: torch.Tensor, static: torch.Tensor,
                doy: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        cal = calendar_features(doy).to(x)[:, :, None, None].expand(B, 2, H, W)
        h = torch.nn.functional.silu(self.inp(torch.cat([x, y, static, cal], dim=1)))
        sc, sh = self.film(self.embed(a)).chunk(2, dim=-1)
        h = h * (1 + sc[:, :, None, None]) + sh[:, :, None, None]
        h = torch.nn.functional.silu(self.mid(h))
        return self.out(h)


class Closure(nn.Module):
    """C_chi(x_k, c_k): low-capacity effective sources C_u, C_v, C_T, C_q (sec. 5.2).

    Never sees the generated member or its noise. Outputs are in units of the
    residual scales b_j (so 1 = one typical residual), per level.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        cin = contract.N_DYNAMIC + 1 + 2
        self.net = nn.Sequential(_PeriodicConv(cin, hidden), nn.SiLU(),
                                 nn.Conv2d(hidden, len(CLOSURE_EQS) * contract.N_LEVELS, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, static: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        cal = calendar_features(doy).to(x)[:, :, None, None].expand(B, 2, H, W)
        c = self.net(torch.cat([x, static, cal], dim=1))
        return c.view(B, len(CLOSURE_EQS), contract.N_LEVELS, H, W)


class PhysicsGuide(nn.Module):
    """L_phys of eq. (10) for a pair (x_k, y) through the bridge B_psi.

    Works in SI units: the normalized dynamic state is de-normalized with the
    exact KAI inverse (``std + 1e-8``), orography is re-inserted, and the
    step 3 grid operators give the residuals. Everything stays in the graph.
    """

    def __init__(self, n_lat: int, n_lon: int, mean, std, cfg: PhysicsConfig = PhysicsConfig(),
                 scales: primitive.ResidualScales = primitive.ResidualScales(),
                 bridge: Optional[PhysicalBridge] = None, closure: Optional[Closure] = None):
        super().__init__()
        self.cfg, self.scales = cfg, scales
        self.n_lat, self.n_lon = n_lat, n_lon
        self.bridge = bridge if bridge is not None else PhysicalBridge()
        self.closure = closure if closure is not None else Closure()
        mean = torch.as_tensor(mean, dtype=torch.float64)
        std = torch.as_tensor(std, dtype=torch.float64) + contract.KAI_STD_EPS
        self.register_buffer("mean", mean.view(-1, 1, 1))
        self.register_buffer("std", std.view(-1, 1, 1))
        self._grid = None

    def grid(self, device) -> pgrid.SphereGrid:
        if self._grid is None or self._grid.lat.device != torch.device(device):
            lat = torch.linspace(-90.0, 90.0, self.n_lat).tolist()
            lon = (torch.arange(self.n_lon) * (360.0 / self.n_lon)).tolist()
            self._grid = pgrid.SphereGrid(lat, lon, device=device)
        return self._grid

    # -- units --------------------------------------------------------------
    def to_si(self, dyn: torch.Tensor, static: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Normalized (B,66,H,W) + orog (B,1,H,W) -> step 3 field dict in SI (float64)."""
        from .data import assemble_full_state
        full = assemble_full_state(dyn, static).double()
        return pgrid.fields_from_state(full * self.std + self.mean)

    def _h_si(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Bridge correction (B,60,H,W), normalized -> SI increments per level variable."""
        name = {"z": "Phi", "q": "q", "t": "T", "u": "u", "v": "v"}
        out = {}
        for j, v in enumerate(LEVEL_VARS):
            idx = [contract.DYNAMIC_CHANNELS[i] for i in LEVEL_DYN[v]]
            s = self.std[idx]
            out[name[v]] = h[:, j * contract.N_LEVELS:(j + 1) * contract.N_LEVELS].double() * s
        return out

    def closures(self, x, static, doy) -> primitive.Closures:
        c = self.closure(x, static, doy).double().index_select(-2, self.grid(x.device).lat_rows)
        terms = {}
        for j, eq in enumerate(CLOSURE_EQS):
            terms[eq] = c[:, j] * self.scales.of(eq)
        return primitive.Closures(terms)

    # -- residuals ----------------------------------------------------------
    def residuals(self, x: torch.Tensor, y: torch.Tensor, static: torch.Tensor, doy: torch.Tensor,
                  use_bridge: bool = True, use_closure: bool = True
                  ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Eq. (18) along B_psi. Returns (residuals on lat rows, mean ||h_psi||^2).

        doy is the calendar of x_k. With ``use_bridge=False`` the path is the
        linear bridge (step 3's independent diagnostic).
        """
        cfg, g = self.cfg, self.grid(x.device)
        if cfg.n_sub < 2 or cfg.n_sub % 2:
            raise ValueError("Simpson rule needs an even n_sub >= 2")
        F0, F1 = self.to_si(x, static), self.to_si(y, static)
        C = self.closures(x, static, doy) if use_closure else None
        keys = ("u", "v", "T", "Phi", "q")
        w = torch.ones(cfg.n_sub + 1, dtype=torch.float64)
        w[1:-1:2], w[2:-1:2] = 4.0, 2.0
        w = (w / w.sum()).tolist()
        acc, reg = None, x.new_zeros(())
        for i in range(cfg.n_sub + 1):
            a = i / cfg.n_sub
            Fs = {k: (1.0 - a) * F0[k] + a * F1[k] for k in keys}
            if use_bridge and 0 < i < cfg.n_sub:
                h = self.bridge(torch.full((x.shape[0],), a, device=x.device, dtype=x.dtype),
                                x, y, static, doy)
                reg = reg + (h ** 2).mean() / (cfg.n_sub - 1)
                for k, dk in self._h_si(h).items():
                    Fs[k] = Fs[k] + a * (1.0 - a) * dk
            A = pgrid.strong_residuals(Fs, g, closures=C, virtual=cfg.virtual)
            acc = {k: w[i] * v for k, v in A.items()} if acc is None else \
                {k: acc[k] + w[i] * v for k, v in A.items()}
        dt = contract.CONST.dt_seconds
        for n in primitive.PROGNOSTIC:
            if n in acc:
                acc[n] = acc[n] + ((F1[n] - F0[n]) / dt).index_select(-2, g.lat_rows)
        return acc, reg

    def mask(self, x: torch.Tensor, y_obs: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Valid collocation points, fixed by the OBSERVED pair (sec. 5.3).

        A generated surface pressure can therefore not remove collocation
        points. Levels below either observed surface are excluded.
        """
        g = self.grid(x.device)
        sp0 = self.to_si(x, static)["sp"]
        sp1 = self.to_si(y_obs, static)["sp"]
        return pgrid.above_ground_mask(torch.minimum(sp0, sp1), g)

    def loss(self, R: Dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        g = self.grid(mask.device)
        lw = g.latitude_weight(self.cfg.lat_weight, pole_cap_deg=self.cfg.pole_cap_deg)
        R = {k: v for k, v in R.items() if self.cfg.gamma.get(k, 0.0) > 0}
        return primitive.physics_loss(R, self.scales, self.cfg.gamma, area_weight=lw, mask=mask)

    def per_equation(self, R: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, float]:
        """Normalized RMS of each residual (independent diagnostic, sec. 6.3)."""
        g = self.grid(mask.device)
        lw = g.latitude_weight(self.cfg.lat_weight, pole_cap_deg=self.cfg.pole_cap_deg)
        out = {}
        for k, r in R.items():
            sq = (r / self.scales.of(k)) ** 2
            wt = lw * mask
            sq = torch.where(wt > 0, sq, torch.zeros_like(sq))
            out[k] = float(((sq * wt).sum() / wt.sum()).sqrt())
        return out


def estimate_scales(guide: PhysicsGuide, batches, floor_frac: float = 0.1) -> primitive.ResidualScales:
    """b_j = RMS of the linear-bridge, no-closure residual on observed pairs (sec. 5.1),
    floored at ``floor_frac`` x the synoptic default scale.

    The floor is needed: a constraint the data satisfy almost exactly gets
    b_j -> 0 (R_mass with omega diagnosed from continuity is zero by
    construction; hydrostatic balance on balanced data), and then any
    generated endpoint has a normalized residual of 1e4-1e5 that swamps
    L_CFM (step 4 smoke test: L_phys ~ 1e9).
    """
    sums, n = {}, 0
    with torch.no_grad():
        for b in batches:
            x, y = b["x0"], b["targets"][:, 0]
            R, _ = guide.residuals(x, y, b["static"], b["doy"][:, 0], use_bridge=False, use_closure=False)
            m = guide.mask(x, y, b["static"])
            g = guide.grid(x.device)
            lw = g.latitude_weight(guide.cfg.lat_weight, pole_cap_deg=guide.cfg.pole_cap_deg)
            for k, r in R.items():
                wt = lw * m
                sq = torch.where(wt > 0, r ** 2, torch.zeros_like(r))
                sums[k] = sums.get(k, 0.0) + float((sq * wt).sum() / wt.sum())
            n += 1
    rms = {k: math.sqrt(v / n) for k, v in sums.items()}
    ref = primitive.ResidualScales()
    raw = {"mom": math.sqrt(0.5 * (rms["u"] ** 2 + rms["v"] ** 2)), "T": rms["T"],
           "mass": rms["mass"], "hyd": rms["hyd"], "q": rms.get("q", ref.q)}
    return primitive.ResidualScales(**{k: max(v, floor_frac * getattr(ref, k)) for k, v in raw.items()})


class PGCFM(nn.Module):
    """CFM transition + physics guide. Inference is the plain CFM rollout."""

    def __init__(self, cfm: CFMTransition, guide: PhysicsGuide):
        super().__init__()
        self.cfm, self.guide = cfm, guide

    def _tisr_next(self, y: torch.Tensor) -> Optional[torch.Tensor]:
        return y[:, TISR_DYN:TISR_DYN + 1] if self.cfm.cfg.tisr_mode == "forcing" else None

    # -- appendix C, step 3 -------------------------------------------------
    def pretrain_loss(self, batch, closure_reg: float = 1e-2) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Bridge + closure on OBSERVED pairs: L_phys + lambda_H ||h||^2 + closure size/roughness."""
        x, y, st = batch["x0"], batch["targets"][:, 0], batch["static"]
        doy = batch["doy"][:, 0]
        R, reg = self.guide.residuals(x, y, st, doy)
        m = self.guide.mask(x, y, st)
        lp = self.guide.loss(R, m)
        c = self.guide.closure(x, st, doy)
        rough = (c[..., 1:] - c[..., :-1]).pow(2).mean() + (c[..., 1:, :] - c[..., :-1, :]).pow(2).mean()
        loss = lp + self.guide.cfg.lambda_h * reg + closure_reg * (c.pow(2).mean() + rough)
        return loss, {"phys": lp.item(), "reg": reg.item(), "closure_rms": c.detach().pow(2).mean().sqrt().item()}

    # -- appendix C, step 4: eq. (12) ---------------------------------------
    def joint_loss(self, batch, it: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        terms, logs = self.joint_terms(batch, it)
        return sum(terms.values()), logs

    def joint_terms(self, batch, it: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """The weighted terms of eq. (12): {"cfm", "phys" (lambda_P L_phys + lambda_H reg), "es"}."""
        cfg = self.guide.cfg
        x, y, st = batch["x0"], batch["targets"][:, 0], batch["static"]
        doy0, doy1 = batch["doy"][:, 0], batch["doy"][:, 1]
        l_cfm = self.cfm.loss(x, y, st, doy1)
        lam = cfg.lambda_p * min(1.0, (it + 1) / max(1, cfg.ramp_iters))
        logs = {"cfm": l_cfm.item(), "lambda_p": lam}
        terms = {"cfm": l_cfm}
        if lam > 0 or cfg.lambda_s > 0:
            # Fresh noise, differentiable endpoint: L_phys -> B_psi -> y_hat -> theta.
            n_mem = 2 if cfg.lambda_s > 0 else 1
            xs = x.repeat(n_mem, 1, 1, 1)
            sts, d1 = st.repeat(n_mem, 1, 1, 1), doy1.repeat(n_mem)
            tn = self._tisr_next(y)
            tn = tn.repeat(n_mem, 1, 1, 1) if tn is not None else None
            y_hat = self.cfm.sample(xs, sts, d1, tn, steps=cfg.sample_steps, grad=True)
            if lam > 0 and cfg.control == "smooth":
                # Control arm: same lambda, same differentiable endpoint, but a purely
                # spatial-smoothness penalty normalized by the truth's roughness.
                l_s = (laplacian(y_hat) ** 2).mean() / (laplacian(y) ** 2).mean().detach().clamp_min(1e-12)
                terms["phys"] = lam * l_s
                logs.update(phys=l_s.item())
            elif lam > 0:
                R, reg = self.guide.residuals(xs, y_hat, sts, doy0.repeat(n_mem))
                m = self.guide.mask(x, y, st).repeat(n_mem, 1, 1, 1)
                l_phys = self.guide.loss(R, m)
                terms["phys"] = lam * l_phys + cfg.lambda_h * reg
                logs.update(phys=l_phys.item(), reg=reg.item())
            if cfg.lambda_s > 0:
                es = energy_score(y_hat.view(n_mem, *x.shape), y, self.cfm.w_area)
                terms["es"] = cfg.lambda_s * es
                logs["es"] = es.item()
        return terms, logs


def laplacian(x: torch.Tensor) -> torch.Tensor:
    """5-point grid Laplacian, periodic in longitude, replicate in latitude (normalized units)."""
    xp = nn.functional.pad(nn.functional.pad(x, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1), mode="replicate")
    return xp[..., 1:-1, 2:] + xp[..., 1:-1, :-2] + xp[..., 2:, 1:-1] + xp[..., :-2, 1:-1] - 4 * x


def energy_score(ens: torch.Tensor, obs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Eq. (13), fair pairwise estimator with the area-weighted norm W. ens (M,B,C,H,W)."""
    M = ens.shape[0]

    def norm(d):
        return ((d ** 2) * w).mean(dim=(-3, -2, -1)).clamp_min(1e-12).sqrt()

    skill = torch.stack([norm(ens[m] - obs) for m in range(M)]).mean(0)
    pair = sum(norm(ens[m] - ens[n]) for m in range(M) for n in range(M) if m != n) / (2 * M * (M - 1))
    return (skill - pair).mean()


@torch.no_grad()
def deterministic_rollout(model: CFMTransition, x0, static, doy, tisr=None, steps: int = 8) -> torch.Tensor:
    """Zero-noise (eps = 0) rollout, the baseline the evaluation suite asks for. (B,K,66,H,W)."""
    K = doy.shape[1] - 1
    x, out = x0, []
    for k in range(K):
        f = tisr[:, k] if tisr is not None else None
        eps = torch.zeros((x.shape[0], len(model.gen_idx)) + x.shape[-2:], device=x.device, dtype=x.dtype)
        x = model.sample(x, static, doy[:, k + 1], f, steps=steps, eps=eps)
        out.append(x)
    return torch.stack(out, dim=1)


class HydrostaticProjection(nn.Module):
    """Hard hydrostatic constraint: T on pressure levels diagnosed from the generated z.

    T = -(d Phi / d ln p) / (R_d (1 + 0.608 q)), with the same centred d/d ln p
    as the residual, so R_hyd = 0 by construction instead of by penalty
    (step 4 proposal, docs/step4_pgcfm.md W4). Differentiable; wraps a
    ``CFMTransition`` and keeps its sampling interface.
    """

    def __init__(self, cfm: CFMTransition, guide: PhysicsGuide):
        super().__init__()
        self.cfm, self.guide = cfm, guide
        self.cfg, self.gen_idx, self.w_area = cfm.cfg, cfm.gen_idx, cfm.w_area

    def project(self, dyn: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        g = self.guide
        F = g.to_si(dyn, static)
        grid = g.grid(dyn.device)
        dphi_dlnp = grid.d_p(F["Phi"]) * grid.p3()
        T = -dphi_dlnp / (contract.CONST.r_dry * (1.0 + 0.608 * F["q"]))
        idx = [contract.DYNAMIC_CHANNELS.index(c) for c in range(*contract.level_slice("t").indices(contract.N_CHANNELS))]
        full_idx = [contract.DYNAMIC_CHANNELS[i] for i in idx]
        t_norm = ((T - g.mean[full_idx]) / g.std[full_idx]).to(dyn.dtype)
        out = dyn.clone()
        out[:, idx] = t_norm
        return out

    def sample(self, x, static, doy_next, tisr_next=None, steps=8, generator=None, grad=False, eps=None):
        y = self.cfm.sample(x, static, doy_next, tisr_next, steps=steps, generator=generator, grad=grad, eps=eps)
        with torch.set_grad_enabled(grad):
            return self.project(y, static)
