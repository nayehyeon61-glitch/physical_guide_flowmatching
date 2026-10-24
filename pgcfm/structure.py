"""Step 9: classical spread calibration first, then an operator for what a spread calibration
cannot change (owner's classical1-4 notes, 2026-10-24).

Revised hypothesis (classical3): the neural operator corrects the spatial and inter-variable
error structure left in the fields after the PINN, and its effect goes beyond what a plain
ensemble spread calibration gives. Test: ``serial + calibration`` vs ``serial + operator +
calibration``, each calibration fitted separately with the same procedure on the same
validation period (classcal2 / classical4's four comparisons).

Classical calibration (classical1): with members x^(k), mean mu and anomalies a^(k) = x^(k) - mu,

    x~^(k)_c = mu_c + s_c a^(k)_c,        s_c >= 0 per channel,

which keeps the ensemble mean, multiplies each channel's (population) spread by s_c and, with
s_c > 0, keeps every correlation between members' deviations (in space and across variables).
``SpreadCalibrator`` fits s_c (or s_kc per lead) on a validation set by empirical CRPS, applied
after the rollout. ``LoopInflated`` is the in-loop variant (step 8 ``inflate``), for any base.

``StructureOperator``: x~^(k) = x^(k) + N_phi(x^(k), c), c = [ensemble mean, ensemble log spread,
orography, calendar], a neuraloperator FNO shared by all members (permutation equivariant);
without ``use_stats`` it is member-wise (the classical3 distinction: only with the ensemble
statistics in the input can it be said to use the ensemble). ``StructureCorrected`` runs it on
the serial chain and, with ``preserve_spread``, rescales the corrected deviations so each
channel's area-mean ensemble variance equals the input's: the operator may move the mean, move
spread between regions and change the deviation patterns and cross-variable correlations, but
the global amplitude per channel is left to the calibrator.

``structure_scores``: what a positive channel scale cannot change, measured per lead -- the
energy score, CRPS / ensemble-mean RMSE of spatial differences, the variogram score of
neighbouring grid points, the mismatch between the cross-variable correlation of member
deviations and of the actual errors (all pairs and physically related pairs), the same for the
lag-1 spatial correlation, and member / truth power per DCT band (smoothing).
"""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from neuralop.models import FNO

from . import contract, metrics
from .cfm import area_weights
from .coupling import energy_score
from .ens_operator import N_COND, MemberGrouped, OperatorCorrected
from .spread import KEEP

CANDIDATES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)


def rescale_ensemble_spread(members: torch.Tensor, channel_scales: torch.Tensor) -> Dict[str, torch.Tensor]:
    """classical1: members (B, K, C, H, W) (K = members), channel_scales (C,) >= 0."""
    assert members.ndim == 5
    B, K, C, H, W = members.shape
    assert K >= 2
    assert channel_scales.shape == (C,)
    assert torch.isfinite(members).all()
    assert torch.isfinite(channel_scales).all()
    assert (channel_scales >= 0).all()
    mean_keepdim = members.mean(dim=1, keepdim=True)
    anomalies = members - mean_keepdim
    spread_before = anomalies.square().mean(dim=1).sqrt()  # population (denominator K)
    scale = channel_scales.to(members).reshape(1, 1, C, 1, 1)
    calibrated_members = mean_keepdim + scale * anomalies
    calibrated_mean = calibrated_members.mean(dim=1, keepdim=True)
    calibrated_anomalies = calibrated_members - calibrated_mean
    spread_after = calibrated_anomalies.square().mean(dim=1).sqrt()
    return {"mean": mean_keepdim.squeeze(1), "anomalies": anomalies, "calibrated_members": calibrated_members,
            "spread_before": spread_before, "spread_after": spread_after}


def _pair_abs_sum(a: torch.Tensor) -> torch.Tensor:
    """sum_{i,j} |a_i - a_j| over the member axis 0, by sorting (memory O(M))."""
    M = a.shape[0]
    s = a.sort(dim=0).values
    w = (2 * torch.arange(M, device=a.device, dtype=a.dtype) - M + 1).view(M, *[1] * (a.dim() - 1))
    return 2 * (w * s).sum(0)


def crps_parts(e: torch.Tensor, o: torch.Tensor):
    """e (M, ...), o (...) -> (mean |e - o|, sum_{i,j} |e_i - e_j|), elementwise."""
    return (e - o).abs().mean(0), _pair_abs_sum(e)


def crps_from_parts(skill, pair, M: int, criterion: str = "empirical"):
    """empirical: E|X - y| - 1/2 E|X - X'| with the M^2 pairs (diagonal included); fair: M(M - 1) pairs."""
    den = M * M if criterion == "empirical" else M * (M - 1)
    return skill - 0.5 * pair / den


def lead_crps(e: torch.Tensor, o: torch.Tensor, criterion: str = "empirical") -> list:
    """Area-weighted CRPS per lead, e (M, B, K, C, H, W), o (B, K, C, H, W). Use "empirical" (the
    exact CRPS of the members' empirical distribution) for a recalibrated ensemble: after
    mu + s (x - mu) the members are no longer independent draws, and the fair estimator then
    keeps rewarding s beyond the calibrated scale (``SpreadCalibrator`` docstring)."""
    w = area_weights(o.shape[-2], o.device)
    out = []
    for k in range(o.shape[1]):
        sk, pr = crps_parts(e[:, :, k], o[:, k])
        out.append(float((crps_from_parts(sk, pr, e.shape[0], criterion) * w).mean()))
    return out


class SpreadCalibrator(nn.Module):
    """Post-hoc channel-wise spread calibration of a finished rollout (classical1/4).
    ``per_lead``: one scale per (lead, channel) instead of per channel. Fit on validation only.

    Criterion: for a perfect ensemble (members and truth exchangeable), the expected empirical
    CRPS of mu + s (x - mu) is minimal at s = 1 for any M: the population anomalies shrink by
    (M - 1) / M and the M^2-pair term shrinks by the same factor. The fair CRPS (M(M - 1) pairs)
    is minimal at s > 1 (about 1.3 for M = 8), so fitting with it over-disperses."""

    def __init__(self, C: int = contract.N_DYNAMIC, per_lead: bool = False, leads: int = 1):
        super().__init__()
        self.per_lead = per_lead
        self.register_buffer("scales", torch.ones(leads if per_lead else 1, C))

    @torch.no_grad()
    def fit_validation(self, ens: torch.Tensor, obs: torch.Tensor, candidates: Sequence[float] = CANDIDATES,
                       criterion: str = "empirical") -> Dict[str, object]:
        """ens (B, M, K, C, H, W), obs (B, K, C, H, W). CRPS is separable over channels and leads
        and the pair term of mu + s a is s times that of a, so the search is exact and needs one
        rollout. Returns the CRPS table (S, K, C) and how many scales sit on the grid edge."""
        M, K = ens.shape[1], ens.shape[2]
        w = area_weights(ens.shape[-2], ens.device)
        cand = torch.as_tensor(list(candidates), dtype=ens.dtype)
        table = torch.zeros(len(cand), K, ens.shape[3])
        for k in range(K):
            e = ens[:, :, k].transpose(0, 1)  # (M, B, C, H, W)
            mu = e.mean(0, keepdim=True)
            a = e - mu
            pair = _pair_abs_sum(a)
            for i, s in enumerate(cand.tolist()):
                skill = (mu + s * a - obs[:, k]).abs().mean(0)
                table[i, k] = (crps_from_parts(skill, s * pair, M, criterion) * w).mean(dim=(-2, -1)).mean(0)
        crit = table if self.per_lead else table.mean(1, keepdim=True)
        idx = crit.argmin(0)  # (K or 1, C)
        self.scales = cand[idx].to(self.scales)
        edge = int(((idx == 0) | (idx == len(cand) - 1)).sum())
        return {"table": table, "edge": edge, "criterion": criterion}

    def forward(self, ens: torch.Tensor) -> torch.Tensor:
        """ens (B, M, K, C, H, W) -> mean + s (ens - mean)."""
        K = ens.shape[2]
        s = self.scales.to(ens)
        s = (s if self.per_lead else s.expand(K, -1))[:K].view(1, 1, K, -1, 1, 1)
        mu = ens.mean(1, keepdim=True)
        return mu + s * (ens - mu)


class LinearMeanCalibrator(nn.Module):
    """Classical mean correction (MOS-style): mu' = a_kc + b_kc mu per lead and channel, least
    squares (area weighted) on validation; every member is shifted by mu' - mu, so the deviations
    are unchanged. The control for an operator that lowers the ensemble-mean error only by damping
    the mean toward the climatology."""

    def __init__(self, C: int = contract.N_DYNAMIC, leads: int = 1):
        super().__init__()
        self.register_buffer("a", torch.zeros(leads, C))
        self.register_buffer("b", torch.ones(leads, C))

    @torch.no_grad()
    def fit_validation(self, ens: torch.Tensor, obs: torch.Tensor):
        """ens (B, M, K, C, H, W), obs (B, K, C, H, W)."""
        w = area_weights(ens.shape[-2], ens.device).expand(ens.shape[-2:])
        mu = ens.mean(1)
        dims = (0, 3, 4)
        sw = w.sum() * mu.shape[0]
        mm, mo = (mu * w).sum(dims) / sw, (obs * w).sum(dims) / sw
        cov = ((mu - mm[None, :, :, None, None]) * (obs - mo[None, :, :, None, None]) * w).sum(dims)
        var = ((mu - mm[None, :, :, None, None]) ** 2 * w).sum(dims)
        self.b = (cov / var.clamp_min(1e-12)).to(self.b)
        self.a = (mo - self.b * mm).to(self.a)
        return self

    def forward(self, ens: torch.Tensor) -> torch.Tensor:
        K = ens.shape[2]
        a, b = self.a[:K].to(ens), self.b[:K].to(ens)
        mu = ens.mean(1, keepdim=True)
        return ens + a[None, None, :, :, None, None] + (b[None, None, :, :, None, None] - 1) * mu


@torch.no_grad()
def tangent_calibrated(ens: torch.Tensor, calibrated: torch.Tensor, guide, x0, static, doy) -> torch.Tensor:
    """ens + P_T (calibrated - ens) per lead: only the hydrostatic tangent part of a calibration's
    change is kept (classical1: physics projection after the spread change). J is taken at each
    lead's ensemble mean (transition from the previous lead's mean). A spread change has zero member
    mean, so the mean is kept; a mean shift (``LinearMeanCalibrator``) keeps only its tangent part.
    ens, calibrated (B, M, K, 66, H, W), x0 (B, 66, H, W), doy (B, K+1)."""
    from .spread import HydrostaticTangent
    tan = HydrostaticTangent(guide)
    inc, out, prev = calibrated - ens, ens.clone(), x0
    for k in range(ens.shape[2]):
        mu = ens[:, :, k].mean(1)
        J = tan.jacobian(prev, mu, static, doy[:, k])
        out[:, :, k] += HydrostaticTangent.project(J, inc[:, :, k])
        prev = mu
    return out


class LoopInflated(MemberGrouped):
    """In-loop multiplicative inflation of any member transition: mu + alpha (y - mu), fed back
    (step 8 ``inflate`` for a base other than the serial chain)."""

    def __init__(self, base, alpha: float = 1.0, members: int = 1, order: str = "repeat"):
        super().__init__(base, members, order)
        self.alpha = alpha

    def sample(self, x, static, doy_next, tisr_next=None, steps=8, generator=None, grad=False, eps=None):
        y = self.base.sample(x, static, doy_next, tisr_next, steps=steps, generator=generator, eps=eps)
        Y = self._group(y)
        mu = Y.mean(1, keepdim=True)
        return self._ungroup(mu + self.alpha * (Y - mu))


class StructureOperator(nn.Module):
    """x~ = x + gate N_phi([x, mu, log sd, cond]) (or [x, cond] without ``use_stats``), one network
    for every member. Gate starts at 0 (identity).

    ``net="fno"``: neuraloperator FNO on the lat-lon rectangle (latitude padded so the FFT does not
    join the poles). ``net="sfno"``: neuraloperator SFNO (spherical harmonic transform from
    torch-harmonics, equiangular grid with poles), the operator of Bonev et al. (2023) for the
    sphere; ``n_modes`` is then (l_max, 2 m_max)."""

    def __init__(self, C: int = contract.N_DYNAMIC, K: int = N_COND, n_modes=(12, 12), hidden: int = 32,
                 lat_padding: float = 0.25, use_stats: bool = True, net: str = "fno"):
        super().__init__()
        self.use_stats = use_stats
        c_in = (3 * C if use_stats else C) + K
        if net == "sfno":
            from neuralop.models import SFNO  # needs torch-harmonics
            self.net = SFNO(n_modes=n_modes, in_channels=c_in, out_channels=C, hidden_channels=hidden)
        elif net == "fno":
            pad = [lat_padding, 0.0] if lat_padding > 0 else None
            self.net = FNO(n_modes=n_modes, in_channels=c_in, out_channels=C, hidden_channels=hidden,
                           domain_padding=pad)
        else:
            raise ValueError(net)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, U: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, M, C, H, W = U.shape
        parts = [U, cond]
        if self.use_stats:
            mu = U.mean(1, keepdim=True)
            logsd = 0.5 * torch.log((U - mu).pow(2).mean(1, keepdim=True) + 1e-6)
            parts += [mu.expand_as(U), logsd.expand_as(U)]
        corr = self.net(torch.cat(parts, dim=2).flatten(0, 1)).reshape(B, M, C, H, W)
        return U + self.gate * corr


def match_spread(V: torch.Tensor, U: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rescale the deviations of V (B, M, C, H, W) so each (sample, channel) area-mean ensemble
    variance equals U's; the mean of V is kept."""
    mv, mu = V.mean(1, keepdim=True), U.mean(1, keepdim=True)
    var_v = metrics._wmean((V - mv).pow(2).mean(1))  # (B, C)
    var_u = metrics._wmean((U - mu).pow(2).mean(1))
    s = torch.sqrt((var_u + eps) / (var_v + eps))[:, None, :, None, None]
    return mv + s * (V - mv)


class StructureCorrected(OperatorCorrected):
    """Serial chain -> ``StructureOperator`` -> DCT band limit (``band``) -> optional spread match."""

    def __init__(self, base, op: nn.Module, members: int = 1, order: str = "repeat", band=None,
                 preserve_spread: bool = True):
        super().__init__(base, op, members, order, band)
        self.preserve_spread = preserve_spread

    def correct(self, x, y, static, doy_next):
        out = super().correct(x, y, static, doy_next)
        if not self.preserve_spread:
            return out
        return self._ungroup(match_spread(self._group(out), self._group(y)))


# -- diagnostics ------------------------------------------------------------------

def spatial_diff(f: torch.Tensor) -> torch.Tensor:
    """(..., C, H, W) -> (..., 2C, H, W): longitude difference (periodic) and latitude difference
    (last row 0)."""
    dx = f - f.roll(1, dims=-1)
    dy = torch.nn.functional.pad(f[..., 1:, :] - f[..., :-1, :], (0, 0, 0, 1))
    return torch.cat([dx, dy], dim=-3)


def physical_pairs(channels: Sequence[int]):
    """``channels``: full KAI channel ids. Index pairs (into ``channels``) of physically linked
    fields: (z, t), (u, v), (z, u), (z, v) at one level, and every upper-air variable at adjacent levels."""
    pos = {c: i for i, c in enumerate(channels)}
    ci, L = contract.CHANNEL_INDEX, contract.LEVELS_HPA
    pairs = []
    for i, p in enumerate(L):
        for a, b in (("z", "t"), ("u", "v"), ("z", "u"), ("z", "v")):
            pairs.append((ci[f"{a}{p}"], ci[f"{b}{p}"]))
        if i + 1 < len(L):
            for v in contract.UPPER_AIR_VARS:
                pairs.append((ci[f"{v}{p}"], ci[f"{v}{L[i + 1]}"]))
    pairs = [(pos[a], pos[b]) for a, b in pairs if a in pos and b in pos]
    return torch.tensor(pairs).T


def _corr(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """z (N, C, H, W) zero-mean samples -> (C, C) area-weighted correlation."""
    zw = z * w.sqrt()
    f = zw.permute(1, 0, 2, 3).reshape(z.shape[1], -1)
    cov = f @ f.T
    d = cov.diagonal().clamp_min(1e-12).sqrt()
    return cov / (d[:, None] * d[None])


def _lag_corr(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """z (N, C, H, W) -> (2, C) lag-1 correlation in longitude and latitude."""
    wf = w.expand(z.shape[-2:])

    def c(a, b, ww):
        return (a * b * ww).sum(dim=(0, 2, 3)) / ((a * a * ww).sum(dim=(0, 2, 3)) * (b * b * ww).sum(dim=(0, 2, 3))
                                                   ).clamp_min(1e-12).sqrt()

    return torch.stack([c(z, z.roll(1, dims=-1), wf), c(z[..., 1:, :], z[..., :-1, :], wf[1:])])


@torch.no_grad()
def structure_scores(e: torch.Tensor, o: torch.Tensor, bands=None, channels: Optional[Sequence[int]] = None
                     ) -> Dict[str, list]:
    """e (M, B, K, C, H, W) members, o (B, K, C, H, W) truth; ``channels``: the full KAI channel id of
    each of the C channels (default: the dynamic channels without TISR, ``spread.KEEP``). Per lead."""
    K, C, H = o.shape[1], o.shape[2], o.shape[-2]
    w = area_weights(H, e.device)
    if channels is None and C == len(KEEP):
        channels = [contract.DYNAMIC_CHANNELS[d] for d in KEEP]
    pairs = physical_pairs(channels) if channels is not None else None
    off = ~torch.eye(C, dtype=torch.bool)
    out: Dict[str, list] = {k: [] for k in ("lead_es", "lead_grad_crps", "lead_grad_rmse", "lead_vs", "lead_xcorr_err",
                                            "lead_xcorr_err_phys", "lead_lagcorr_err", "lead_band_amp")}
    for k in range(K):
        ek, ok = e[:, :, k], o[:, k]
        out["lead_es"].append(float(energy_score(ek, ok, w)))
        ge, go = spatial_diff(ek), spatial_diff(ok)
        out["lead_grad_crps"].append(float(metrics.crps(ge, go).mean()))
        out["lead_grad_rmse"].append(float(metrics._wmean((ge.mean(0) - go) ** 2).mean().sqrt()))
        # variogram score of order 1/2 over neighbouring points (Scheuerer and Hamill 2015)
        vs = (go.abs().sqrt() - ge.abs().sqrt().mean(0)) ** 2
        out["lead_vs"].append(float(metrics._wmean(vs).mean()))
        mu = ek.mean(0)
        dev, err = (ek - mu).flatten(0, 1), mu - ok
        rd, re = _corr(dev, w), _corr(err, w)
        out["lead_xcorr_err"].append(float((rd - re)[off].abs().mean()))
        if pairs is not None:
            out["lead_xcorr_err_phys"].append(float((rd - re)[pairs[0], pairs[1]].abs().mean()))
        out["lead_lagcorr_err"].append(float((_lag_corr(dev, w) - _lag_corr(err, w)).abs().mean()))
        if bands is not None:
            out["lead_band_amp"].append((bands.power(ek) / bands.power(ok)).tolist())
    return {k: v for k, v in out.items() if v}
