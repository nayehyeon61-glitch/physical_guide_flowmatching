"""Step 8: calibrate the ensemble distribution of the serial chain instead of its members
(owner's under-dispersion hypotheses, 2026-10-24).

    DCT encoder -> latent CFM members -> PINN Q per member -> decoder/IDCT   (frozen, step 7 serial)
                -> one of the ensemble maps below -> calibrated members

H1: the operator should learn the distribution (spread, covariance, its time evolution), not
correct each member. H2: the PINN Q contracts physically admissible diversity along with the
inadmissible part. The maps (``EnsembleCalibrated.mode``; mean mu and deviations delta over
members, ``pre`` = D(z_cfm) before Q, ``post`` = D(Q(z_cfm)) after Q):

    "serial"           post                                      the step 7 serial chain
    "inflate"          mu_post + alpha delta_post                A: multiplicative inflation (EnKF)
    "band"             mu_post + sum_b alpha_b P_b delta_post    4: inflation per DCT band P_b
    "noQ"              pre                                       3: Q bypassed at inference
    "noQ_hyd"          HydrostaticProjection(pre)                3: hard hydrostatic T <- Phi instead of Q
    "qmean"            mu_post + delta_pre                       Q corrects the mean only
    "tangent"          mu_post + P_T delta_pre                   C: CFM diversity kept along the
                                                                    hydrostatic tangent space
    "tangent_inflate"  mu_post + alpha P_T delta_post            C + A: inflate admissible directions,
                         + (I - P_T) delta_post                     leave the normal part as it is

Every map keeps the ensemble mean of its source (``noQ``: the CFM's own mean). They run inside the
Markov rollout (calibrated members are fed back), as inflation does in ensemble data assimilation.

``CovarianceOperator`` (B) is the learned version: a neuraloperator FNO maps (mean, log spread,
condition) to a bounded log-scale field and a linear ``SpectralConv`` mixes the deviation field,
so it transforms the covariance (Sigma -> A Sigma A^T) and cannot move the mean.

P_T: the hydrostatic residual is linear in (Phi, T, q) to 1e-4 and local to a grid column, so
J = dR_hyd/dy is built exactly per column from one vjp per level and
P_T = I - J^T (J J^T)^-1 J. Only the diagnostic constraint (hydrostatics) defines the tangent
space; the prognostic residuals pin the tendency itself, so their null space would remove almost
all diversity, including the dynamical uncertainty the ensemble is meant to carry.
"""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.models import FNO

from . import contract, metrics
from .cfm import TISR_DYN
from .coupling import HydrostaticProjection
from .ens_operator import N_COND, MemberGrouped, OperatorCorrected
from .latent import dct, idct

KEEP = [i for i in range(contract.N_DYNAMIC) if i != TISR_DYN]
MODES = ("serial", "inflate", "band", "noQ", "noQ_hyd", "qmean", "tangent", "tangent_inflate")


def serial_members(m, x, static, doy_next, steps=8, generator=None, eps=None):
    """Decoded members of a serial ``LatentPGCFM`` before and after Q: (D(z_cfm), D(Q(z_cfm)))."""
    z0 = m.encode(x, static, (doy_next - 1) % 366)
    zc = m.sample_latent(x, static, doy_next, steps, generator, eps, refine=False)
    return m.decode(zc, x), m.decode(m.serial(zc, z0, doy_next), x)


class DCTBands:
    """Partition of the kept k_lat x k_lon DCT-II block into ``n_bands`` by normalized index
    radius, plus one more band (index ``n_bands``) for every mode beyond the cut."""

    def __init__(self, n_lat: int, n_lon: int, k_lat: int, k_lon: int, n_bands: int = 3):
        i = torch.arange(n_lat, dtype=torch.float32)[:, None] / k_lat
        j = torch.arange(n_lon, dtype=torch.float32)[None, :] / k_lon
        r = torch.sqrt(i ** 2 + j ** 2) / 2 ** 0.5
        idx = torch.clamp((r * n_bands).long(), max=n_bands - 1)
        idx[k_lat:, :] = n_bands
        idx[:, k_lon:] = n_bands
        self.idx, self.n = idx, n_bands

    def scale(self, d: torch.Tensor, alphas: Sequence[float]) -> torch.Tensor:
        """Deviations scaled by alphas[b] in band b (modes beyond the cut unchanged)."""
        f = torch.as_tensor([*alphas, 1.0], dtype=d.dtype, device=d.device)[self.idx.to(d.device)]
        return idct(dct(d) * f)

    def power(self, d: torch.Tensor) -> torch.Tensor:
        """Mean square DCT power per band, (..., C, H, W) -> (n_bands + 1,), last = beyond the cut."""
        p = dct(d) ** 2
        idx = self.idx.to(d.device)
        return torch.stack([p[..., idx == b].mean() for b in range(self.n + 1)])


class HydrostaticTangent:
    """Projection onto the tangent space of the hydrostatic constraint at a state y0 (transition
    from x): P_T v = v - J^T (J J^T + ridge)^-1 J v per grid column, J = d R_hyd(x, y) / d y.
    Columns without a valid level (poles, below ground) are left unchanged."""

    def __init__(self, guide):
        self.guide = guide

    def jacobian(self, x, y0, static, doy) -> torch.Tensor:
        """(N, H, W, L, C): one row per hydrostatic level, built with one vjp per level."""
        g = self.guide
        mask = g.mask(x, y0, static)

        def f(y):
            R, _ = g.residuals(x, y, static, doy, use_bridge=False, use_closure=False)
            return R["hyd"] * mask

        with torch.enable_grad():
            out, vjp = torch.func.vjp(f, y0.detach())
            L = out.shape[1]
            Js = []
            for lev in range(L):
                w = torch.zeros_like(out)
                w[:, lev] = 1.0
                Js.append(vjp(w)[0])  # (N, C, H, W): row `lev` of every column's J
        return torch.stack(Js, dim=-1).permute(0, 2, 3, 4, 1)  # (N, H, W, L, C); pole rows stay 0

    @staticmethod
    def project(J: torch.Tensor, v: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
        """P_T v for v (N, C, H, W) or (N, M, C, H, W) (same J for every member)."""
        member_axis = v.dim() == 5
        vc = v.permute(0, 1, 3, 4, 2) if member_axis else v.permute(0, 2, 3, 1)  # (..., H, W, C)
        Jd = J.double()
        A = Jd @ Jd.transpose(-1, -2)
        A = A + (ridge * A.diagonal(dim1=-2, dim2=-1).mean() + 1e-12) * torch.eye(A.shape[-1], dtype=A.dtype,
                                                                                    device=A.device)
        Jx = J[:, None] if member_axis else J
        Ax = A[:, None] if member_axis else A
        jv = (Jx.double() @ vc.double()[..., None])
        lam = torch.linalg.solve(Ax, jv)
        out = vc.double() - (Jx.double().transpose(-1, -2) @ lam)[..., 0]
        out = out.to(v.dtype)
        return out.permute(0, 1, 4, 2, 3) if member_axis else out.permute(0, 3, 1, 2)


class EnsembleCalibrated(MemberGrouped):
    """Frozen serial chain + an ensemble map (module docstring table)."""

    def __init__(self, base, mode: str, guide=None, alpha: float = 1.0, band_alpha: Optional[Sequence[float]] = None,
                 bands: Optional[DCTBands] = None, members: int = 1, order: str = "repeat"):
        super().__init__(base, members, order)
        if mode not in MODES:
            raise ValueError(mode)
        if mode in ("noQ_hyd", "tangent", "tangent_inflate") and guide is None:
            raise ValueError(f"mode {mode} needs the physics guide")
        self.mode, self.guide, self.alpha, self.band_alpha, self.bands = mode, guide, alpha, band_alpha, bands
        self.tangent = HydrostaticTangent(guide) if guide is not None else None
        self.hyd = HydrostaticProjection(base, guide) if guide is not None else None

    def sample(self, x, static, doy_next, tisr_next=None, steps=8, generator=None, grad=False, eps=None):
        with torch.set_grad_enabled(grad):
            pre, post = serial_members(self.base, x, static, doy_next, steps, generator, eps)
            if self.mode == "serial":
                return post
            if self.mode == "noQ":
                return pre
            if self.mode == "noQ_hyd":
                return self.hyd.project(pre, static)
            P, Q = self._group(pre), self._group(post)
            mu = Q.mean(1, keepdim=True)
            dq, dp = Q - mu, P - P.mean(1, keepdim=True)
            if self.mode == "inflate":
                out = mu + self.alpha * dq
            elif self.mode == "band":
                out = mu + self.bands.scale(dq, self.band_alpha)
            elif self.mode == "qmean":
                out = mu + dp
            else:  # tangent space at the corrected mean; J is linear to 1e-4, so the first member's
                # x_k stands for all (it only enters through the q coupling and the mask)
                xg, sg = self._group(x)[:, 0], self._group(static)[:, 0]
                dg = self._group((doy_next - 1) % 366)[:, 0]
                J = self.tangent.jacobian(xg, mu[:, 0], sg, dg)
                if self.mode == "tangent":
                    out = mu + HydrostaticTangent.project(J, dp)
                else:
                    t = HydrostaticTangent.project(J, dq)
                    out = mu + self.alpha * t + (dq - t)
            return self._ungroup(out)


class CovarianceOperator(nn.Module):
    """B: an operator on the ensemble distribution. With mean mu and deviations d over members,

        d' = exp(1.5 tanh(gate * G(mu, log sd, cond))) * (d + gate_mix * S(d)),   out = mu + d'

    G: neuraloperator FNO (latitude padded), the spatially varying log spread change per channel;
    S: a linear spectral convolution (no bias), a learned spatial correlation operator applied to
    every member's deviation field. d' is linear in d, so the map is Sigma -> A Sigma A^T with
    A = diag(scale) (I + gate_mix S), and the ensemble mean never moves (no drift, unlike the
    member-wise operator of step 7). Both gates start at 0 (identity)."""

    def __init__(self, C: int = contract.N_DYNAMIC, K: int = N_COND, n_modes=(12, 12), hidden: int = 32,
                 lat_padding: float = 0.25):
        super().__init__()
        pad = [lat_padding, 0.0] if lat_padding > 0 else None
        self.scale_op = FNO(n_modes=n_modes, in_channels=2 * C + K, out_channels=C, hidden_channels=hidden,
                            domain_padding=pad)
        self.mix = SpectralConv(C, C, n_modes, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))
        self.gate_mix = nn.Parameter(torch.zeros(1))

    def forward(self, U: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, M, C, H, W = U.shape
        mu = U.mean(1, keepdim=True)
        d = U - mu
        logsd = 0.5 * torch.log(d.pow(2).mean(1) + 1e-6)
        g = self.scale_op(torch.cat([mu[:, 0], logsd, cond.mean(1)], dim=1))
        mixed = self.mix(d.reshape(B * M, C, H, W)).reshape(B, M, C, H, W)
        s = torch.exp(1.5 * torch.tanh(self.gate * g))[:, None]
        return mu + s * (d + self.gate_mix * mixed)


class TangentOperatorCorrected(OperatorCorrected):
    """B inside the admissible space: base members -> operator -> the operator's deviations from
    the ensemble mean projected onto the hydrostatic tangent space at that mean (P_T), so the
    learned covariance change cannot add hydrostatic imbalance."""

    def __init__(self, base, op: nn.Module, guide, members: int = 1, order: str = "repeat", band=None):
        super().__init__(base, op, members, order, band)
        self.tangent = HydrostaticTangent(guide)

    def correct(self, x, y, static, doy_next):
        out = self._group(super().correct(x, y, static, doy_next))
        mu = out.mean(1, keepdim=True)
        J = self.tangent.jacobian(self._group(x)[:, 0], mu[:, 0], self._group(static)[:, 0],
                                  self._group((doy_next - 1) % 366)[:, 0])
        return self._ungroup(mu + HydrostaticTangent.project(J, out - mu))


# -- diagnostics ------------------------------------------------------------------

def ensemble_scores(e: torch.Tensor, o: torch.Tensor, bands: Optional[DCTBands] = None) -> Dict[str, list]:
    """e (M, B, K, C, H, W) members, o (B, K, C, H, W) truth (TISR already dropped) -> per lead:
    CRPS, ensemble-mean RMSE, spread, SSR, range coverage (nominal (M-1)/(M+1)), member amplitude,
    and with ``bands`` the spread / error power ratio per DCT band."""
    M, K = e.shape[0], o.shape[1]
    mu = e.mean(0)
    rmse = metrics.ens_mean_rmse(e, o).pow(2).mean(dim=(0, 2)).sqrt()
    spread = metrics.ens_spread(e).pow(2).mean(dim=(0, 2)).sqrt()
    lo, hi = e.min(0).values, e.max(0).values
    inside = ((o >= lo) & (o <= hi)).float()
    out = {
        "lead_crps": [float(metrics.crps(e[:, :, k], o[:, k]).mean()) for k in range(K)],
        "lead_rmse": rmse.tolist(),
        "lead_spread": spread.tolist(),
        "lead_ssr": (((M + 1) / M) ** 0.5 * spread / rmse).tolist(),
        "lead_coverage": metrics._wmean(inside).mean(dim=(0, 2)).tolist(),
        "coverage_nominal": (M - 1) / (M + 1),
        "lead_member_amp": metrics._wmean(e ** 2).mean(dim=(0, 1, 3)).sqrt().tolist(),
    }
    if bands is not None:
        dev, err = e - mu, mu - o
        out["lead_band_ratio"] = [(bands.power(dev[:, :, k]) * M / (M - 1) / bands.power(err[:, k])).tolist()
                                  for k in range(K)]
    return out


@torch.no_grad()
def contraction_report(m, guide, batches, members: int = 8, steps: int = 8, seed: int = 6) -> Dict[str, float]:
    """H2 check, one step: how much member variance Q removes, split into the hydrostatic tangent
    space and its normal complement at the corrected mean. Variances are area weighted over the
    66 - TISR channels; ``*_hyd`` is the RMS hydrostatic residual (K) of the members."""
    tan = HydrostaticTangent(guide)
    g = torch.Generator().manual_seed(seed)
    acc: Dict[str, list] = {}
    for b in batches:
        x, st, d = b["x0"], b["static"], b["doy"]
        B = x.shape[0]
        rep = lambda t: t.repeat(members, *([1] * (t.dim() - 1)))  # noqa: E731
        pre, post = serial_members(m, rep(x), rep(st), rep(d[:, 1]), steps, g)
        P, Q = pre.view(members, B, *x.shape[1:]).transpose(0, 1), post.view(members, B, *x.shape[1:]).transpose(0, 1)
        mu = Q.mean(1)
        J = tan.jacobian(x, mu, st, d[:, 0])
        for name, E in (("pre", P), ("post", Q)):
            dev = E - E.mean(1, keepdim=True)
            t = HydrostaticTangent.project(J, dev)
            for part, v in (("total", dev), ("tangent", t), ("normal", dev - t)):
                acc.setdefault(f"{name}_var_{part}", []).append(float(metrics._wmean(v[:, :, KEEP] ** 2).mean()))
            R, _ = guide.residuals(rep(x), E.transpose(0, 1).reshape(-1, *x.shape[1:]), rep(st), rep(d[:, 0]),
                                   use_bridge=False, use_closure=False)
            mk = guide.mask(rep(x), rep(b["targets"][:, 0]), rep(st))
            acc.setdefault(f"{name}_hyd", []).append(guide.per_equation({"hyd": R["hyd"]}, mk)["hyd"])
            err = E.mean(1) - b["targets"][:, 0]
            acc.setdefault(f"{name}_mean_rmse", []).append(float(metrics._wmean(err[:, KEEP] ** 2).mean().sqrt()))
    out = {k: sum(v) / len(v) for k, v in acc.items()}
    for part in ("total", "tangent", "normal"):
        out[f"kept_{part}"] = out[f"post_var_{part}"] / max(out[f"pre_var_{part}"], 1e-12)
    return out


# -- fitting (calibration period only) ----------------------------------------------

def fit_scalar(crps_of, grid: Sequence[float]):
    """alpha minimizing ``crps_of(alpha)`` (mean CRPS over leads of a calibration rollout)."""
    scores = {float(al): crps_of(al) for al in grid}
    return min(scores, key=scores.get), scores


def fit_bands(crps_of, alpha0: float, n_bands: int, factors: Sequence[float] = (0.6, 0.8, 1.25, 1.6)):
    """One coordinate sweep over the DCT bands from [alpha0] * n_bands; ``crps_of(list)``."""
    best_a = [alpha0] * n_bands
    best = crps_of(best_a)
    trace = [(list(best_a), best)]
    for b in range(n_bands):
        for f in factors:
            cand = list(best_a)
            cand[b] = alpha0 * f
            c = crps_of(cand)
            trace.append((cand, c))
            if c < best:
                best, best_a = c, cand
    return best_a, trace


def fit_posthoc(ens: torch.Tensor, obs: torch.Tensor, grid=None):
    """Per-lead alpha_k for scaling a finished rollout. ens (B,M,K,C,H,W), obs (B,K,C,H,W),
    TISR already dropped."""
    grid = grid if grid is not None else [1.0 + 0.1 * i for i in range(51)]
    alphas = []
    for k in range(ens.shape[2]):
        e = ens[:, :, k].transpose(0, 1)
        mu = e.mean(0, keepdim=True)
        o = obs[:, k:k + 1]
        s = {al: float(metrics.crps((mu + al * (e - mu))[:, :, None], o).mean()) for al in grid}
        alphas.append(float(min(s, key=s.get)))
    return alphas


def posthoc_scale(ens: torch.Tensor, alphas: Sequence[float]) -> torch.Tensor:
    """ens (B,M,K,...) -> mean + alpha_k (member - mean) at every lead k."""
    mu = ens.mean(1, keepdim=True)
    a = torch.as_tensor(list(alphas), dtype=ens.dtype, device=ens.device).view(1, 1, -1, *[1] * (ens.dim() - 3))
    return mu + a * (ens - mu)


@torch.no_grad()
def rollout_contraction(m, guide, batch, members: int = 8, steps: int = 8, leads: Sequence[int] = (1, 3, 6),
                        seed: int = 8) -> Dict[str, Dict[str, float]]:
    """H2 across time: members of the serial rollout differ in x_k, so Q also sees different
    initial states. After ``lead`` serial steps, one more step is drawn with each member's own x_k;
    reports the incoming variance (x_k), the variance before Q (D(z_cfm)) and after Q, and the
    tangent / normal split of both at the corrected mean. ``kept`` = after / before Q."""
    tan = HydrostaticTangent(guide)
    g = torch.Generator().manual_seed(seed)
    x0, st, doy = batch["x0"], batch["static"], batch["doy"]
    B = x0.shape[0]
    rep = lambda t: t.repeat(members, *([1] * (t.dim() - 1)))  # noqa: E731
    x, s = rep(x0), rep(st)
    out = {}
    for k in range(max(leads) + 1):
        dn = rep(doy[:, k + 1])
        pre, post = serial_members(m, x, s, dn, steps, g)
        if k in leads:
            grp = lambda t: t.view(members, B, *t.shape[1:]).transpose(0, 1)  # noqa: E731
            P, Q, X = grp(pre), grp(post), grp(x)
            mu = Q.mean(1)
            J = tan.jacobian(X[:, 0], mu, st, doy[:, k])
            r = {"var_in": float(metrics._wmean((X - X.mean(1, keepdim=True))[:, :, KEEP] ** 2).mean())}
            for name, E in (("pre", P), ("post", Q)):
                dev = E - E.mean(1, keepdim=True)
                t = HydrostaticTangent.project(J, dev)
                r[f"{name}_var"] = float(metrics._wmean(dev[:, :, KEEP] ** 2).mean())
                r[f"{name}_normal_frac"] = float(metrics._wmean((dev - t)[:, :, KEEP] ** 2).mean()) / max(
                    r[f"{name}_var"], 1e-12)
            r["kept"] = r["post_var"] / max(r["pre_var"], 1e-12)
            out[f"lead{k}"] = r
        x = post
    return out
