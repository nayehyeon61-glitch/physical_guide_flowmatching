"""Where to put the tendency core in KAI-alpha (docs/kai_tendency.md).

Two models, trained separately:

* the tendency core: pgcfm.dyncore.DynCore of step 16, unchanged (ClimODE's ResNet as dx/dt = f(x), one Heun
  step, tisr forcing, PositiveMass, rollout training), here with one step = one day on KAI's daily means;
* KAI-alpha: third_party/kai (InceptionNeXt, full next-day state, single-step training, relative L2 loss).

The core is trained first and frozen. Its one-day forecast c(x) and tendency tau = (c(x) - x) / s
(s = std of the daily increment per channel) then enter KAI at ONE position:

    inject  in        tau as 67 extra input channels of the stem
            s1 .. s4  tau added to the features after stage 1 .. 4 through a zero-initialized 1x1 conv
                      (s4 = just before KAI's head)
    form    full      KAI's own output: the full next state                       (original KAI)
            res       x + s * net(.)        zero head = persistence               (control, no core)
            core      c(x) + s * net(.)     zero head = the core's forecast       (the core at the output)
            gate      k(x) + g * (c(x) - k(x))   k = KAI's full next state, g = one weight per channel,
                      zero g = the original KAI: KAI keeps its own form and its trained weights (--init kai),
                      the core is mixed in at the output every step (the trained version of the blend)

A further position needs no training: a lead-wise blend of the two finished forecasts
(scripts/run_kai_tendency.py, ``blend``; ``loop`` = the same weights applied every step and fed back).

The core inside every stage (docs/kai_stage_core.md, owner 2026-11-18: the core is to act in each of KAI's stages
instead of being mixed with KAI's finished output):

    inject  all       tau added at the ENTRY of each of the four stages (after the stage's stem / transition
                      layer, before its blocks), each through its own zero-initialized 1x1 conv, so every stage's
                      blocks process the core's tendency at that stage's width
    form    sgate     k(x) + g(x) * (c(x) - k(x)),  g = clamp(g0 + 1x1 conv of KAI's last-stage features, 0, 1):
                      the output mix of ``gate`` with a weight per channel AND grid point that KAI computes itself;
                      g0 = the loop's fixed weights (--gate-init), the conv starts at zero: the model starts as the
                      loop and learns where to depart from its fixed weights

            dec       the core INSIDE every stage (owner 2026-11-18 22:50 KST: "the variables split into the
                      stage's channels are merged again, put into our tendency core, split again and blended in
                      every stage"; 23:14 KST: "merge them correctly, only then the core has an effect").
                      At the entry of stage i (48 / 96 / 192 / 288 channels at KAI's size):
                          merge   x_i = x + a_i * s * D_i(h_i)   the physical state the stage has reached
                          core    tau_i = (c(x_i) - x_i) / s     the frozen core's own one-day step from x_i
                          split   e_i = P_i(tau_i)               P_i = 1x1 conv, 67 -> stage channels
                          blend   h_i <- h_i + e_i               then the stage's blocks
                      The merge. KAI's split is not invertible (the stem maps 67 variables to FEWER channels and
                      normalizes them per grid point), so the channels alone give the variables back with an error
                      of more than one day's change (measured, docs/kai_stage_core.md): the core would compute
                      its tendency from that error. A correct merge therefore keeps the variables themselves, the
                      input x, and takes from the channels what they add to it: D_i, of the form of KAI's own
                      merge to the variables (its head: 3x3 conv, GELU, 1x1 conv), is the stage's estimate of the
                      day's change (x(t+1) - x(t)) / s. Stage i stands at the part a_i = (i - 1) / 4 of the day
                      (MERGE_AT), so its state is the input plus that part of the estimated change: stage 1 is
                      asked at the input itself (the exact merge of the stem's split), stage 4 three quarters on.
                      D_i is trained ONLY by its error against the data (the trainer's merge loss on
                      ``merge_pred``; its input and its output are detached): the merged state is the best
                      estimate of a real state, not whatever state makes the core's answer useful, and KAI's own
                      weights see the forecast loss only, as in every other arm.
                      D_i (last layer) and P_i start at zero: every stage's state starts as the input x and the
                      model as the KAI it was initialized from; ``all`` is the case a_i = 0 (the core at x only).
                      A linear P_i of the core's increment is the blend of the stage's state with the core's
                      forecast from it, the weight being part of P_i (c - x_i is what a weight would multiply).

    arms    stg   = (full, all)    the core's tendency of x in the four stages, KAI's own output
            stgg  = (sgate, all)   the same and the learned output mix
            outg  = (sgate, none)  the learned output mix alone (the control of the stages)
            stc   = (full, dec)    the core inside the four stages (merge, core, split, blend), KAI's own output
            stcg  = (sgate, dec)   the same and the learned output mix

Memory (the owner's Mori-Zwanzig reading, 2026-11-10): the daily 5.625 deg state is a projection of the
atmosphere, so its exact equation is  dx/dt = Markov term f(x) + memory integral over the past + noise.
DynCore is the Markov term alone. ``make_core(hist=True)`` adds one lag (the observed trend),
``make_core(mem=W)`` a ConvGRU state m that integrates the past (W days before the IC, then the rollout's own
states) and enters the tendency as input before every step: dx/dt = f(x, m).
"""

import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from . import contract

_KAI_ROOT = os.path.join(os.path.dirname(__file__), "..", "third_party", "kai")
if _KAI_ROOT not in sys.path:
    sys.path.insert(0, _KAI_ROOT)
from src.models import KAI  # noqa: E402  (vendored KAI-alpha backbone)
from src.utils import LpLoss  # noqa: E402

TISR = contract.CHANNEL_INDEX["tisr"]
OROG = contract.CHANNEL_INDEX["orog"]
DYN = contract.DYNAMIC_CHANNELS
TISR_DYN = DYN.index(TISR)

# arm -> (form, inject)
ARMS = {"kai": ("full", "none"), "in": ("full", "in"), "s1": ("full", "s1"), "s2": ("full", "s2"),
        "s3": ("full", "s3"), "s4": ("full", "s4"), "res": ("res", "none"), "inres": ("res", "in"),
        "out": ("core", "none"), "inout": ("core", "in"), "gate": ("gate", "none"), "ingate": ("gate", "in"),
        # control of a history core: KAI itself gets the observed trend (x - x_prev) / s as input, no core
        "kaih": ("full", "hist"),
        # control of ``out``: the frozen first stage is not the tendency core but a trained KAI (arm res)
        "stack": ("core", "none"),
        # the core in every stage (inject all) and / or a state-dependent output mix (form sgate)
        "stg": ("full", "all"), "stgg": ("sgate", "all"), "outg": ("sgate", "none"),
        # the core inside every stage: the stage's features merged to the variables, the core, split again (inject dec)
        "stc": ("full", "dec"), "stcg": ("sgate", "dec")}
N_STAGES = 4
# inject dec: where on the way from today's state x(t) to tomorrow's x(t+1) each stage's merged state stands
MERGE_AT = (0.0, 0.25, 0.5, 0.75)


def needs_core(arm: str) -> bool:
    form, inject = ARMS[arm]
    return form in ("core", "gate", "sgate") or inject not in ("none", "hist")


# ------------------------------------------------------------------------------------- the tendency core
def stateless_norm(module: nn.Module) -> nn.Module:
    """Replace every BatchNorm2d by GroupNorm(1, C) and switch Dropout off, in place.

    ClimODE's ResidualBlock (the core's trunk) has BatchNorm + Dropout(0.1). Trained with batch statistics
    and run with running statistics, the core's tendency gets a systematic offset that a rollout integrates:
    on 2017 the global mean of z500 drifts by -0.14 sigma in 10 days (z1000 -0.47, q700 +0.29) with running
    statistics and by +0.004 with batch statistics, same weights (docs/kai_tendency.md). GroupNorm has no
    batch statistics, so the trained and the deployed function are the same."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(1, child.num_features))
        elif isinstance(child, nn.Dropout):
            setattr(module, name, nn.Identity())
        else:
            stateless_norm(child)
    return module


def _hist_core_class():
    """DynCore with the observed trend as input: f(x, x - x_prev). ``prev`` = the state one step before the
    step's start (data at the IC, the model's own previous state afterwards); None = no trend (zeros)."""
    from .dyncore import DynCore, Net, _NetCfg, N_DYN

    class HistCore(DynCore):
        hist = True

        def __init__(self, cfg, static, t6, mean, std, lat_deg):
            super().__init__(cfg, static, t6, mean, std, lat_deg)
            self.net = Net(2 * N_DYN + static.shape[0] + 4, N_DYN, _NetCfg(cfg.hidden, cfg.blocks))
            self.prev, self.trend = None, None

        def _trunk(self, x, hours, g=None):
            from .dyncore import time_features
            B, _, H, W = x.shape
            return self.net.trunk(torch.cat([x, self.trend, self.static.expand(B, -1, -1, -1),
                                             time_features(hours, H, W)], 1))

        def step(self, x, hours, tisr_next):
            # the trend of the step's start, in units of the increment std; the same for both Heun evaluations
            self.trend = torch.zeros_like(x) if self.prev is None else (x - self.prev) / self.t6 * self.keep
            y = super().step(x, hours, tisr_next)
            self.prev = x
            return y
    return HistCore


class ConvGRUCell(nn.Module):
    """The standard convolutional GRU cell (Ballas et al. 2016): gates and candidate are 3x3 convolutions of
    [input, state]; padding as in the core's trunk (longitude circular, latitude reflect). The state is a leaky
    time integral of its inputs with a learned, state-dependent kernel: m <- (1 - z) m + z n."""

    def __init__(self, cin: int, hid: int):
        super().__init__()
        self.hid = hid
        self.gates = nn.Conv2d(cin + hid, 2 * hid, 3)
        self.cand = nn.Conv2d(cin + hid, hid, 3)

    @staticmethod
    def pad(x):
        return F.pad(F.pad(x, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1), mode="reflect")

    def forward(self, u, m=None):
        if m is None:
            m = u.new_zeros(u.shape[0], self.hid, *u.shape[-2:])
        z, r = torch.sigmoid(self.gates(self.pad(torch.cat([u, m], 1)))).chunk(2, 1)
        n = torch.tanh(self.cand(self.pad(torch.cat([u, r * m], 1))))
        return (1 - z) * m + z * n


class LongTerm(nn.Module):
    """The long-memory term of the accumulated memory: two 3x3 convolutions of the accumulators z (padding as in
    the trunk), zero-initialized output, so a new core starts without it."""

    def __init__(self, cin: int, hid: int, cout: int):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(cin, hid, 3), nn.Conv2d(hid, cout, 3)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, z):
        return self.c2(ConvGRUCell.pad(F.silu(self.c1(ConvGRUCell.pad(z)))))


class UNetTrunk(nn.Module):
    """A multi-scale trunk for the tendency core + (docs/kai_tendency_plus_arch.md): the core's own residual blocks
    (ClimODE's ResidualBlock: two 3x3 convolutions, longitude circular, latitude reflect) on three grids, the data's
    and its 2x and 4x coarsening (1.5 / 3 / 6 deg), joined by skip connections.

    Why: the plain trunk keeps the 5.625 deg design (8 convolutions = 17 x 17 cells) on a grid with cells 3.75 times
    smaller, so one evaluation of the tendency sees 2,800 km instead of 10,000 km. The coarsest level here has the
    cell size the 5.625 deg core was built for; the finest keeps the data's grid for what only it resolves.

    ``dims`` = widths of the levels, fine to coarse; ``reps`` = residual blocks per level on the way down (the way
    up has one block per level, after the concatenation with the skip). Coarsening = 2 x 2 mean (an odd number of
    rows, the 121 rows with the poles, is first extended by one reflected row); refining = bilinear to the skip's
    size, periodic in longitude. The output has dims[0] channels on the data's grid."""

    def __init__(self, c_in: int, dims=(64, 128, 192), reps: int = 2):
        super().__init__()
        from model_utils import ResidualBlock  # third_party/climode, on the path through pgcfm.core
        self.dims = tuple(int(d) for d in dims)
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        c = c_in
        for d in self.dims:
            self.down.append(nn.Sequential(ResidualBlock(c, d), *[ResidualBlock(d, d) for _ in range(reps - 1)]))
            c = d
        for hi, lo in zip(self.dims[-2::-1], self.dims[:0:-1]):  # coarse to fine: (width of the skip, width below)
            self.up.append(ResidualBlock(hi + lo, hi))

    @staticmethod
    def coarsen(x):
        if x.shape[-2] % 2:
            x = F.pad(x, (0, 0, 0, 1), mode="reflect")
        return F.avg_pool2d(x, 2)

    @staticmethod
    def refine(x, size):
        H, W = size
        f = W // x.shape[-1]  # the longitudes are halved exactly
        x = F.pad(x, (1, 1, 0, 0), mode="circular")
        return F.interpolate(x, size=(H, W + 2 * f), mode="bilinear", align_corners=False)[..., f:-f]

    def forward(self, x):
        x, skips = x.float(), []
        for i, blk in enumerate(self.down):
            x = blk(x if i == 0 else self.coarsen(x))
            skips.append(x)
        for blk, skip in zip(self.up, skips[-2::-1]):
            x = blk(torch.cat([skip, self.refine(x, skip.shape[-2:])], 1))
        return x


class UNetNet(nn.Module):
    """The multi-scale trunk + the zero-initialized linear head (pgcfm.core.Net with a UNetTrunk)."""

    def __init__(self, c_in: int, c_out: int, dims=(64, 128, 192), reps: int = 2):
        super().__init__()
        self.trunk = UNetTrunk(c_in, dims, reps)
        self.head = nn.Conv2d(self.trunk.dims[0], c_out, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.trunk(x))


PACC_TAU = (2.0, 10.0)  # days: the bounds of the learned kernel time of the accumulated physical tendency


class MemoryKernel(nn.Module):
    """The memory kernel network: the memory integral of the Mori-Zwanzig equation in its discrete form (the sum
    over the lags of the discrete generalized Langevin equation, cf. Lin, Tian, Perez & Livescu 2023; the
    distributed-delay closure of Gupta & Lermusiaux 2021), taken over the memory term's own past:

        M(t) = sum_{s = 0}^{S - 1} K_s(x_t, m_t) * m(t - s),      K_s = k0[s] * (1 + tanh(net([x_t, m_t]))_s)

    m(t - s) = the ConvGRU memory state of s days ago (``mem_ch`` channels), k0 (S, mem_ch) = the kernel itself: one
    weight per lag and memory channel, learned, starting from exp(-s / tau0) (sum over the lags = 1), and the
    network's S gain maps make it depend on the present state and memory (zero-initialized: a new kernel is k0).
    The kernel is in the open (k0 by lag) instead of inside the GRU's gates, and a lag of s days has a direct path
    to the tendency whatever the gates forget. ``static`` (diagnostic) = the kernel without its network (k0)."""

    def __init__(self, cin: int, mem_ch: int, lags: int, hid: int = 24, tau0: float = 4.0):
        super().__init__()
        self.lags, self.static = int(lags), False
        k = torch.exp(-torch.arange(self.lags, dtype=torch.float32) / tau0)
        self.k0 = nn.Parameter((k / k.sum()).view(-1, 1).repeat(1, mem_ch))
        self.c1, self.c2 = nn.Conv2d(cin + mem_ch, hid, 3), nn.Conv2d(hid, self.lags, 3)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def gain(self, x, m):
        """(B, S, H, W): the state-dependent factor of the kernel at every lag and grid point."""
        if self.static:
            return x.new_ones(x.shape[0], self.lags, *x.shape[-2:])
        h = F.silu(self.c1(ConvGRUCell.pad(torch.cat([x, m], 1))))
        return 1 + torch.tanh(self.c2(ConvGRUCell.pad(h)))

    def forward(self, x, past):
        """past = [m(t), m(t - 1), ...] (newest first, at most S states; days before the start are absent)."""
        g = self.gain(x, past[0])
        return sum(self.k0[s].view(1, -1, 1, 1) * g[:, s:s + 1] * ms for s, ms in enumerate(past))


def _mem_core_class():
    """DynCore with a memory term in the Mori-Zwanzig sense: dx/dt = f(x, m), m = ConvGRU state.

    The memory is updated BEFORE every time step with the state the step starts from and its observed trend,
    m_t = GRU(m_{t-1}, [x_t, (x_t - x_{t-1}) / s]), and enters both Heun evaluations of the step as input.
    ``reset(history)`` runs the GRU over the W days before the IC (data); in the rollout it goes on with the
    core's own states. The Markov core is the case m = const, the history core (one lag) the case W = 1
    without recurrence (cf. the LSTM memory closures of Ma, Wang & E 2018 and Wang, Ripamonti & Hesthaven 2020).

    ``mz`` = the Mori-Zwanzig form on a sequence (owner, 2026-11-10: the loss has to be taken over a sequence in the
    form of the equation, not on single terms). The W days before the IC are not only read: the memory starts
    EMPTY at the first day of the sequence (the memory integral runs from 0 to t), every day of the sequence is
    forecast from the day before with the memory of the days up to it (the first forecast is the Markov term
    alone), and the error of each forecast, the innovation e_t = (x_t - forecast of x_t) / s, is the third input
    of the memory: m_t = GRU(m_{t-1}, [x_t, trend_t, e_t]) (the memory term is what the past forecasts missed).
    ``reset`` keeps these W forecasts in ``seq`` for the sequence loss (scripts/run_kai_tendency.py); in the free
    rollout the state is the core's own forecast, so e = 0.

    With ``cfg.gfeat`` (owner, 2026-11-10: the physical terms need memory terms next to them) the trunk also
    gets the operator's tendency G(x) (48 physical terms, step 18) and the memory integrates the physical terms:
    m_t = GRU(m_{t-1}, [x_t, trend_t, (e_t,) G(x_t), G(x_{t-1})]), dx/dt = f(x, G(x), m).

    ``acc`` = time constants in days, e.g. (3, 5) (owner, 2026-11-11: the memory has to hold the past of 3 to 5
    days for certain, as an accumulation structure with its own term and loss). The trained GRU uses one day of
    past, so next to it the same inputs are accumulated with FIXED exponential kernels,
    z_k(t) = exp(-1/tau_k) z_k(t-1) + (1 - exp(-1/tau_k)) tanh(enc_k([x_t, trend_t, e_t, G_t, G_{t-1}])),
    the Markovian embedding of a memory kernel that is a sum of exponentials: z_k is the time integral of the past
    with an e-folding time of tau_k days whatever the training does. The accumulators enter as their own term,
    dx/dt = f(x, G(x), m) + g(z_3, z_5), the long-memory term (LongTerm; ``acc_off`` switches it off, a diagnostic).
    The accumulators start empty with the sequence and are filled over the W days before the IC (W >= 8 for
    tau = 5 d), in the rollout they go on with the states the core is stepped along.

    ``kern`` = S > 0 (owner, 2026-11-11: a memory kernel network that takes the memory term next to the G core as
    its input and accumulates and strengthens its effect): the memory states of the last S days are kept and the
    kernel network (MemoryKernel) sums them over the lags, M(t) = sum_s K_s(x_t, m_t) m(t - s); M enters the
    long-memory term, dx/dt = f(x, G(x), m) + g(M) (with ``acc``: g(z_3, z_5, M)). The past starts empty with the
    sequence, like the memory. ``kernel.static`` = the kernel without its network (diagnostic).

    ``pacc`` = time constants in days, e.g. (3, 5), with ``cfg.gfeat`` (owner, 2026-11-11: the core is a tendency
    structure that takes the physical terms in, so the kernel formula belongs IN the memory term, and the physical
    tendency accumulated in time reinforces the memory term). The memory of the discrete Mori-Zwanzig equation is
    the recursion h_(n+1) = A_n h_n + c_n (Venturi & Li 2023, eqs. 92-97): the kernel is the powers of a contraction
    A (it fades exponentially, their theorem 6) and the source c is what the step of the resolved terms leaves in
    the unresolved part. Here the resolved terms are the 48 physical terms G(x), and per time scale k
        P_k(t) = a_k P_k(t-1) + (1 - a_k) G(x_t)          the physical tendency accumulated in time
        D_k(t) = a_k D_k(t-1) + (1 - a_k) trend_t         the observed trend of the same channels, the same kernel
        E_k(t) = b_k E_k(t-1) + (1 - b_k) e_t             (mz) the innovations, all channels
        a_k = exp(-1 / tau_k),  tau_k in [PACC_TAU[0], PACC_TAU[1]] days, LEARNED per physical term (b_k: per
        channel) from pacc[k]
    (D_k - P_k = what the physical terms missed over the window; E_k = what the forecasts missed, where the
    diagnostic scripts/diag_kai_longmem.py --phys finds most of the information of the accumulated past) and they
    are inputs of the memory, m_t = GRU(m_(t-1), [x_t, trend_t, e_t, G_t, G_(t-1), P_k, D_k, E_k]); the tendency
    stays dx/dt = f(x, G(x), m). The kernel cannot fall back to one day (tau >= PACC_TAU[0]); P and D are fed in
    a rollout as well (G and the trend of the core's own states), E fades there with its kernel (e = 0).
    ``pacc_off`` (diagnostic) = "all" / "phys" (P, D) / "innov" (E) held at zero."""
    from .dyncore import DynCore, Net, _NetCfg, G_EQS, N_DYN, n_gfeat
    from .core import time_features
    from .coupling import LEVEL_DYN
    eq_idx = sum((LEVEL_DYN[{"T": "t"}.get(e, e)] for e in G_EQS), [])  # the channel of every physical term

    class MemCore(DynCore):
        hist = True  # the caller has to hand over the past

        def __init__(self, cfg, static, t6, mean, std, lat_deg, mem=8, mem_ch=24, mz=False, nomem=False,
                     acc=(), acc_ch=12, kern=0, kern_ch=24, pacc=(), mem_enc=0, era=False):
            super().__init__(cfg, static, t6, mean, std, lat_deg)
            self.mem, self.mem_ch, self.mz = int(mem), int(mem_ch), bool(mz)
            self.nomem = bool(nomem)  # control: the same sequence training, the memory state stays zero
            ng = n_gfeat(cfg)
            # ``era``: the time coordinate of the training period as one more input channel (see make_core);
            # the caller sets ``self.era`` (B,) for a training batch, None = the end of the period (a forecast)
            self.era_in, self.era = bool(era), None
            self.net = Net(N_DYN + mem_ch + static.shape[0] + 4 + ng + int(self.era_in), N_DYN,
                           _NetCfg(cfg.hidden, cfg.blocks))
            self.pacc, self.PD, self.pacc_off = tuple(float(t) for t in pacc), None, False
            if self.pacc:  # the physical tendency accumulated in time, a kernel per physical term, in the memory
                if nomem or not ng:
                    raise ValueError("the accumulated physical tendency needs the memory and the physical terms")
                if getattr(cfg, "gsplit", False):
                    raise ValueError("the accumulated physical tendency is defined for the 48 terms (no gsplit)")
                lo, hi = PACC_TAU
                if not all(lo < t < hi for t in self.pacc):
                    raise ValueError(f"time constants of the accumulated physical tendency: {lo} < tau < {hi} days")
                p0 = (torch.tensor(self.pacc) - lo) / (hi - lo)
                p0 = torch.log(p0 / (1 - p0)).view(-1, 1)
                self.pacc_raw = nn.Parameter(p0.repeat(1, ng))  # the kernel time of every physical term
                if mz:
                    self.pacc_raw_e = nn.Parameter(p0.repeat(1, N_DYN))  # ... of every channel's innovation
                self.register_buffer("eq_idx", torch.tensor(eq_idx))
            self.pacc_ch = len(self.pacc) * (2 * ng + (N_DYN if mz else 0))
            self.enc = nn.Conv2d((3 if mz else 2) * N_DYN + 2 * ng + self.pacc_ch, mem_ch, 1)
            if mem_enc:  # the tendency core + at 1.5 deg: the 294 inputs of the memory are not projected to its
                # channels by one linear map but through a hidden layer of ``mem_enc`` channels (per grid point)
                c = self.enc.in_channels
                self.enc = nn.Sequential(nn.Conv2d(c, int(mem_enc), 1), nn.GroupNorm(1, int(mem_enc)), nn.GELU(),
                                         nn.Conv2d(int(mem_enc), mem_ch, 1))
                self.enc.in_channels = c
            self.gru = ConvGRUCell(mem_ch, mem_ch)
            self.acc, self.z, self.acc_off = tuple(float(t) for t in acc), None, False
            self.kern, self.past, self.M = int(kern), [], None
            if (self.acc or self.kern) and nomem:
                raise ValueError("the no-memory control has no long-memory term")
            if self.acc:  # the accumulated memory: fixed exponential kernels of tau days, their own term
                self.enc_acc = nn.Conv2d(self.enc.in_channels, int(acc_ch) * len(self.acc), 1)
                lam = torch.exp(-1.0 / torch.tensor(self.acc)).repeat_interleave(int(acc_ch))
                self.register_buffer("lam", lam.view(1, -1, 1, 1))
            if self.kern:  # the memory kernel network over the memory states of the last ``kern`` days
                self.kernel = MemoryKernel(N_DYN, mem_ch, self.kern, hid=int(kern_ch))
            if self.acc or self.kern:
                self.long = LongTerm(int(acc_ch) * len(self.acc) + (mem_ch if self.kern else 0), cfg.hidden, N_DYN)
            self.m, self.prev, self.gprev, self.started = None, None, None, False
            self.fc, self.seq = None, None  # mz: the forecast of the next state; the forecasts of the sequence
            self.freeze_mem = False  # diagnostic: no update of the memory after the IC (the rollout's own states)

        def reset(self, history=None, hours0=None, x0=None):
            """history (B, W, 66, H, W): the states before the IC, oldest first; None = an empty memory.
            mz: ``hours0`` (B,) = the IC's time, ``x0`` = the IC (its tisr is the forcing of the last forecast)."""
            self.m, self.prev, self.gprev, self.started, self.fc, self.seq = None, None, None, False, None, None
            self.z, self.past, self.M, self.PD = None, [], None, None
            if history is None:
                return
            W = history.shape[1]
            if not self.mz:
                for k in range(W):
                    self.remember(history[:, k])
                return
            out = []
            for k in range(W):  # teacher forcing over the sequence: day k -> forecast of day k + 1
                nxt = history[:, k + 1] if k + 1 < W else x0
                xk = history[:, k]  # one tensor for the memory and the step (the operator's G is cached per tensor)
                self.remember(xk)
                self.fc = self._advance(xk, hours0 - (W - k) * 24.0, None if nxt is None else nxt[:, TISR_DYN])
                out.append(self.fc)
            self.seq = torch.stack(out, 1)

        def remember(self, x):
            if self.nomem:
                self.m, self.prev = x.new_zeros(x.shape[0], self.mem_ch, *x.shape[-2:]), x
                return
            trend = torch.zeros_like(x) if self.prev is None else (x - self.prev) / self.t6 * self.keep
            parts = [x, trend]
            if self.mz:  # the innovation: zero at the start of a sequence and on the core's own states
                parts.append(torch.zeros_like(x) if self.fc is None
                             else (x.detach() - self.fc.detach()) / self.t6 * self.keep)
            if self.cfg.gfeat:
                # the memory terms of the physical terms: the memory integrates the operator's tendency G of the
                # state now and of the day before (next to the observed trend, so what the physical terms missed
                # over the last day, trend - G_prev, is a linear combination of its inputs)
                g = self.features(x)
                parts += [g, torch.zeros_like(g) if self.gprev is None else self.gprev]
                self.gprev = g
                if self.pacc:  # the kernel in the memory term: the time integral of the physical tendency,
                    K = len(self.pacc)  # of the trend of its channels and of the innovations
                    a = torch.exp(-1.0 / self.pacc_tau()).view(1, -1, 1, 1)  # (1, K * 48, 1, 1)
                    src, ret = [g.repeat(1, K, 1, 1), trend[:, self.eq_idx].repeat(1, K, 1, 1)], [a, a]
                    if self.mz:
                        src.append(parts[2].repeat(1, K, 1, 1))
                        ret.append(torch.exp(-1.0 / self.pacc_tau(innov=True)).view(1, -1, 1, 1))
                    src, ret = torch.cat(src, 1), torch.cat(ret, 1)
                    self.PD = (1 - ret) * src if self.PD is None else ret * self.PD + (1 - ret) * src
                    n = 2 * K * g.shape[1]  # the channels of P and D; the rest is E
                    off = {"all": slice(None), "phys": slice(0, n), "innov": slice(n, None)}.get(self.pacc_off)
                    if off is None:
                        parts.append(self.PD)
                    else:
                        on = torch.ones(1, self.PD.shape[1], 1, 1, dtype=self.PD.dtype, device=self.PD.device)
                        on[:, off] = 0.0
                        parts.append(self.PD * on)
            if self.grad_ckpt and torch.is_grad_enabled() and not self.acc:
                # the tendency core +: the memory update is recomputed in the backward pass as well (its inputs
                # are kept one by one, not their concatenation; the memory before it is read from the closure)
                m0 = self.m
                self.m = torch.utils.checkpoint.checkpoint(lambda *p: self.gru(self.enc(torch.cat(p, 1)), m0),
                                                           *parts, use_reentrant=False)
                v = None
            else:
                v = torch.cat(parts, 1)
                self.m = self.gru(self.enc(v), self.m)
            if self.acc:  # the time integral of the past with the fixed kernels (an empty accumulator = no past)
                u = (1 - self.lam) * torch.tanh(self.enc_acc(v))
                self.z = u if self.z is None else self.lam * self.z + u
            if self.kern:  # the memory integral over the memory's own past, with the kernel of the state now
                self.past = [self.m] + self.past[:self.kern - 1]
                self.M = self.kernel(x, self.past)
            self.prev = x

        def pacc_tau(self, innov=False):
            """(K, 48): the learned e-folding time (days) of the kernel of every physical term; ``innov``:
            (K, 66), of every channel's innovation."""
            lo, hi = PACC_TAU
            return lo + (hi - lo) * torch.sigmoid(self.pacc_raw_e if innov else self.pacc_raw)

        def _trunk(self, x, hours, g=None):
            B, _, H, W = x.shape
            parts = [x, self.m, self.static.expand(B, -1, -1, -1), time_features(hours, H, W)]
            if self.cfg.gfeat:
                parts.append(self.features(x) if g is None else g)
            if self.era_in:
                e = x.new_ones(B) if self.era is None else self.era.to(x)
                parts.append(e.view(B, 1, 1, 1).expand(B, 1, H, W))
            return self.net.trunk(torch.cat(parts, 1))

        def tendency(self, x, hours, g=None):
            f = super().tendency(x, hours, g)
            long = [v for v in (self.z, self.M) if v is not None]
            if long and not self.acc_off:  # the long-memory term, the same in both stages of the step
                f = f + self.long(torch.cat(long, 1)) * self.t6 * self.keep
            return f

        def _advance(self, x, hours, tisr_next):
            """One day of the solver with the memory of this day (all its Heun substeps).

            ``grad_ckpt`` (the tendency core +, a fine grid with several substeps a day): the activations of a
            substep are not kept for the backward pass but recomputed in it (torch.utils.checkpoint), so a
            sequence of W + K days costs the memory of ONE substep. The substep is then a pure function of its
            inputs: the memory state and the accumulators it reads are handed over as arguments (in the backward
            pass the attributes already hold the values of later days)."""
            if not (self.grad_ckpt and torch.is_grad_enabled()):
                return DynCore.step(self, x, hours, tisr_next)
            names = [k for k in ("m", "z", "M") if getattr(self, k) is not None]
            n = self.n_sub()

            def sub(i):
                def f(x, hours, *state):
                    keep = [getattr(self, k) for k in names]
                    for k, v in zip(names, state):
                        setattr(self, k, v)
                    try:
                        return DynCore.substep(self, x, hours, i, n)
                    finally:
                        for k, v in zip(names, keep):
                            setattr(self, k, v)
                return f
            state = [getattr(self, k) for k in names]
            for i in range(n):
                x = torch.utils.checkpoint.checkpoint(sub(i), x, hours, *state, use_reentrant=False)
            return DynCore.finish(self, x, tisr_next)

        def step(self, x, hours, tisr_next):
            if not (self.freeze_mem and self.started):
                self.remember(x)  # the time integral of the past first, then the prediction in time
            self.started = True
            self.fc = self._advance(x, hours, tisr_next)
            return self.fc
    return MemCore


def make_core(static: torch.Tensor, inc_std: np.ndarray, mean: np.ndarray, std: np.ndarray, lat_deg,
              hidden: int = 48, blocks=(2, 2), norm: str = "bn", substeps: int = 1, stride: int = 1,
              hist: bool = False, mem: int = 0, mem_ch: int = 24, mz: bool = False, gfeat: bool = False,
              nomem: bool = False, acc=(), acc_ch: int = 12, kern: int = 0, kern_ch: int = 24, pacc=(),
              trunk: str = "res", unet_dims=(64, 128, 192), mem_enc: int = 0, era: bool = False,
              gsplit: bool = False, gsource: str = "op", gbase: bool = False, noise: bool = False, nz: int = 0,
              hnz: float = 0.0, ent: float = 0.0, entv: bool = False,
              ent_anti: bool = False, isl: bool = False):
    """Step 16's DynCore on daily data: ``t6`` is the std of the DAILY increment, one solver step = one day.

    ``norm`` "bn" = ClimODE's block as it is; "gn" = stateless_norm (the drift fix). ``substeps`` Heun steps
    per day (DynCore's dt_hours, in units of its nominal 6 h step). ``stride`` > 1 = the long-scale core: one
    solver step = ``stride`` days (``inc_std`` is then the std of the stride-day increment). ``hist`` = the
    field extended to its history: the previous state enters as the observed trend (_hist_core_class).
    ``mem`` = W > 0: the memory core (_mem_core_class), a ConvGRU state of ``mem_ch`` channels warmed up on the
    W days before the IC; ``mz`` = its Mori-Zwanzig sequence form (innovation input, forecasts of the sequence).
    ``nomem`` (with mem and mz) = the control of the memory term: the same sequence training with a zero memory.
    ``acc`` (with mem) = the accumulated memory: time constants in days of fixed exponential accumulators of the
    memory's inputs (``acc_ch`` channels each) and their own long-memory term in the tendency.
    ``kern`` (with mem) = S: the memory kernel network (MemoryKernel, hidden ``kern_ch``) over the memory states of
    the last S days; its sum over the lags enters the long-memory term.
    ``pacc`` (with mem and gfeat) = the kernel in the memory term: the physical tendency G and the trend of its
    channels accumulated in time with a learned kernel time per physical term (starting at these days), as inputs
    of the memory.
    ``gfeat`` = the physical terms: the operator's tendency G(x) as 48 input channels (step 18); the caller attaches
    ``core.features`` (operator_features) and fills ``core.gscale``. With ``mem`` the memory integrates G as well.
    ``trunk`` "unet" = the multi-scale trunk (UNetTrunk) with the widths ``unet_dims`` (fine to coarse) and
    ``blocks[0]`` residual blocks per level, instead of the plain ``hidden`` x ``blocks`` ResNet.
    ``mem_enc`` (with mem) = hidden channels of a two-layer encoder of the memory's inputs (0 = one linear map).
    ``era`` (with mem; the time extension) = the time coordinate of the training period as one more input channel
    of the trunk, -1 at the start of the period and +1 at its end (``core.era`` (B,), set by the training for
    every batch; None = +1, the end of the period, which is what every forecast uses). A core trained on a long
    period can then hold what changed over the period (the trend, the changes of the reanalysis) apart from the
    dynamics instead of averaging over it."""
    from .dyncore import DynCore, DynCoreConfig
    if gsplit and not gfeat:
        raise ValueError("gsplit splits the physical terms of gfeat")
    if gfeat and hist and not mem:
        raise ValueError("the history core has no G(x) input; use mem for a memory of the physical terms")
    if (len(acc) or kern or len(pacc)) and not mem:
        raise ValueError("the accumulated memory and the memory kernel belong to the memory core (mem > 0)")
    if mem_enc and not mem:
        raise ValueError("the memory encoder belongs to the memory core (mem > 0)")
    if era and not mem:
        raise ValueError("the era input is built into the memory core (mem > 0)")
    if trunk not in ("res", "unet"):
        raise ValueError(trunk)
    if trunk == "unet":
        hidden = int(unet_dims[0])  # the width the head (and a long-memory term) reads
    cfg = DynCoreConfig(hidden=hidden, blocks=tuple(blocks), dt_hours=6.0 / substeps, solver="heun", positive=True,
                        gfeat=gfeat, gsplit=gsplit, gextra=(48 if gsource == "both" else 0) + int(nz), entv=bool(entv))
    if entv and (mem or hist or gbase or trunk != "res"):
        raise ValueError("the entropy state is built for the plain core (no memory, history, dycore base or U-Net)")
    args = (cfg, static, torch.as_tensor(inc_std[DYN], dtype=torch.float32), mean[DYN], std[DYN], lat_deg)
    if mem:
        core = _mem_core_class()(*args, mem=mem, mem_ch=mem_ch, mz=mz, nomem=nomem, acc=tuple(acc), acc_ch=acc_ch,
                                 kern=kern, kern_ch=kern_ch, pacc=tuple(pacc), mem_enc=mem_enc, era=era)
    else:
        core = (_hist_core_class() if hist else DynCore)(*args)
    if trunk == "unet":  # the same inputs and head, the multi-scale trunk in place of the plain ResNet
        c_in = core.net.trunk.layer_cnn[0][0].conv1.in_channels
        core.net = UNetNet(c_in, core.net.head.out_channels, unet_dims, int(blocks[0]))
    if norm == "gn":
        stateless_norm(core.net)
    elif norm != "bn":
        raise ValueError(norm)
    core.stride = int(stride)
    # the source of the physical terms (docs/kai_dycore_integration.md): "op" = the instantaneous operator tendency
    # G(x); "dycore" = the 24 h integration of the dry primitive equations by dinosaur (pgcfm.dycore_day); ``gbase``
    # (with "dycore") = that integration is also the base of the forecast step
    # "both" = G(x) AND the integrated hint (96 physical terms); ``nz`` = noise channels next to them (owner
    # 2026-11-15: G + hint + noise as the core's inputs, trained together): new white noise every day and member,
    # the core learns what to make of it (dycore_day.ComboFeatures)
    if (nz or gsource == "both") and (mem or hist or not gfeat):
        raise ValueError("G + hint (+ noise) inputs are built for the plain core with gfeat")
    if gsource not in ("op", "dycore", "both") or (gsource == "dycore" and not gfeat) or (gbase and gsource != "dycore"):
        raise ValueError(f"gsource {gsource} gbase {gbase} (dycore needs gfeat; gbase needs dycore)")
    if gbase and (mem or hist or substeps != 1):
        raise ValueError("gbase is built for the plain core, one step a day")
    core.gsource, core.gbase, core.nz = gsource, bool(gbase), int(nz)
    core.entv = bool(entv)  # a deterministic run with its entropy as a state (dyncore.DynCore: cfg.entv)
    if isl:
        # the island's entropy (owner 2026-11-17: the double core on the memory core, tuned at the 5-day island with
        # the entropy): this core is the SECOND stage of a pair whose first stage has no entropy state (a memory
        # core). Its entropy state then starts from s^2 = isl_scale x exp(head(island state, static, time)): the
        # variance of the first stage's error at the island, per channel and grid point (DynCore._trunk).
        # ``isl_scale`` = the mean squared error per channel (a running mean kept by the trainer), so the head only
        # holds where and when the island is less certain than usual.
        if not entv:
            raise ValueError("the island's entropy is the start of an entropy state (entv)")
        core.isl_head = nn.Sequential(nn.Conv2d(contract.N_DYNAMIC + static.shape[0] + 4, hidden, 3, padding=1),
                                      nn.SiLU(), nn.Conv2d(hidden, contract.N_DYNAMIC, 1))
        nn.init.zeros_(core.isl_head[2].weight)
        nn.init.zeros_(core.isl_head[2].bias)
        core.register_buffer("isl_scale", torch.ones(1, contract.N_DYNAMIC, 1, 1))
        core.register_buffer("isl_n", torch.zeros(()))
        core.isl_v = None
    core.hnz = float(hnz)  # > 0: the noise channels are the dycore's response to a perturbed start (ComboFeatures)
    if ent:  # the entropy term in place of a noise (dyncore.DynCore._f): sigma(x) head, starts at 0.31 increment std
        if mem or hist or gbase:
            raise ValueError("the entropy term is built for the plain core (no memory, history or dycore base)")
        core.ent, core.ent_members, core.ent_eps, core.ent_bias = float(ent), 0, 1e-2, -1.0
        core.ent_head = nn.Conv2d(hidden, contract.N_DYNAMIC, 1)
        nn.init.zeros_(core.ent_head.weight)
        nn.init.zeros_(core.ent_head.bias)
        wl = torch.cos(torch.deg2rad(torch.as_tensor(np.asarray(lat_deg), dtype=torch.float32)))
        core.ent_w = (wl / wl.mean()).view(1, 1, -1, 1)
        core.noise_on = False
        core.ent_anti = bool(ent_anti)  # the members start as symmetric pairs (DynCore.perturb)
    if noise:  # the noise term of the Schroedinger-bridge core: a 1 x 1 head sigma(x) on the trunk (starts at ~0.3)
        if not gbase:
            raise ValueError("the noise term belongs to the dycore-base core (gbase)")
        core.noise_head = nn.Conv2d(hidden, contract.N_DYNAMIC, 1)
        nn.init.zeros_(core.noise_head.weight)
        nn.init.zeros_(core.noise_head.bias)
        core.noise_bias = -1.0  # softplus(-1) = 0.31 daily-increment std
        core.noise_on = False
    return core


class CachedFeatures:
    """One-entry cache of the operator features: the memory update and the solver step of the same day ask for
    G of the same state tensor."""

    def __init__(self, feats):
        self.feats, self.key, self.val = feats, None, None

    def __call__(self, x):
        if self.key is not x:
            self.key, self.val = x, self.feats(x)
        return self.val


def operator_features(mean, std, x_full: torch.Tensor, states: Optional[torch.Tensor] = None, scale=None,
                      device=None, chunk: int = 8, split: bool = False):
    """Step 18's operator features on KAI's daily contract: G(x) = -R0(x), the discrete primitive-equation
    operator's own tendency of (u, v, T, q) on the 12 levels (48 physical terms) at a normalized state of the 66
    dynamic channels. ``x_full`` (>= 1, 67, H, W) normalized: its orography channel is the lower boundary.
    ``states`` (N, 66, H, W): estimate the per-channel scale (RMS of G over the valid points) from them, ``chunk``
    states at a time (anything with .shape[0] and slices that give tensors: a fine grid is read from disk).
    ``split`` (a core with ``gsplit``): the same tendencies in two parts each, 96 physical terms (dyncore.G_PARTS)."""
    from .coupling import PhysicsConfig, PhysicsGuide
    from .dyncore import OperatorFeatures, StrongPhysics
    H, W = x_full.shape[-2:]
    guide = PhysicsGuide(H, W, mean, std, PhysicsConfig(dt_seconds=24 * 3600.0))
    for q in guide.parameters():
        q.requires_grad_(False)
    feats = OperatorFeatures(StrongPhysics(guide, torch.ones(contract.N_DYNAMIC)), x_full[:1, [OROG]].float(), scale,
                             split=split)
    if states is not None:
        feats.scale = feats.estimate_scale(states, chunk)
    if device is not None:  # after the scale estimate (CPU states): the operator follows the core to its device
        feats.to(device)
    return feats


def core_step(core, xd: torch.Tensor, doy: torch.Tensor, tisr_next: Optional[torch.Tensor]) -> torch.Tensor:
    """One day of the core on the 66 dynamic channels (season phase from the day of year). ``core.grad_ckpt``:
    the activations of a substep are recomputed in the backward pass (a memory core does that itself, _advance;
    a history core keeps its trend as an attribute and is stepped as it is)."""
    hours = doy.float() * 24.0
    if getattr(core, "grad_ckpt", False) and torch.is_grad_enabled() and not getattr(core, "hist", False):
        n, x = core.n_sub(), xd
        for i in range(n):
            x = torch.utils.checkpoint.checkpoint(lambda x, h, i=i: core.substep(x, h, i, n), x, hours,
                                                  use_reentrant=False)
        return core.finish(x, tisr_next)
    return core.step(xd, hours, tisr_next)


def core_rollout(core, xd, doy, steps, tisr=None, prev=None):
    """``steps`` solver steps of core.stride days each; tisr (B, steps, H, W) = the forcing at the end of each.
    ``prev`` (history cores): the dynamic channels one step before xd; memory cores: the W steps before xd
    (B, W, 66, H, W), oldest first (None = an empty memory)."""
    x, out, st = xd, [], getattr(core, "stride", 1)
    if getattr(core, "mem", 0):
        core.reset(prev, doy.float() * 24.0, xd)
    elif getattr(core, "hist", False):
        core.prev = prev
    if getattr(core, "entv", False):  # the entropy state starts again with every trajectory
        core.reset_entropy()
    for k in range(steps):
        x = core_step(core, x, doy + k * st, None if tisr is None else tisr[:, k])
        out.append(x)
    return torch.stack(out, 1)


MEM_STATE = ("m", "prev", "gprev", "fc", "z", "M", "PD")  # a memory core's tensors of ONE trajectory


def hand_memory(src, dst):
    """A second memory core goes on with the first one's memory (owner 2026-11-17 17:47 KST: the second core of the
    two-stage forecast should have the memory too). ``dst`` takes over the state of the trajectory ``src`` was stepped
    along: the memory, the state before the last step and its physical terms, the last forecast (and the long-memory
    accumulators). The state ``dst`` is stepped from next is compared with that last forecast, so a correction made
    at the hand-over (the island's Kalman correction) enters the second core's memory as an innovation. No gradient
    goes back to ``src``. Both cores must be of the same memory form."""
    for k in ("mem", "mem_ch", "mz", "acc", "kern", "pacc"):
        assert getattr(src, k, None) == getattr(dst, k, None), f"hand_memory: the cores differ in {k}"
    for k in MEM_STATE:
        v = getattr(src, k, None)
        setattr(dst, k, v.detach() if torch.is_tensor(v) else None)
    dst.past = [v.detach() for v in getattr(src, "past", [])]
    dst.started, dst.seq = True, None


def core_continue(core, x, doy, steps, tisr=None):
    """``steps`` more solver steps from x WITHOUT a new start of the core's memory (after hand_memory, or the same
    core going on): doy = the day of x, tisr (B, steps, H, W) = the forcing at the end of each step."""
    out, st = [], getattr(core, "stride", 1)
    for k in range(steps):
        x = core_step(core, x, doy + k * st, None if tisr is None else tisr[:, k])
        out.append(x)
    return torch.stack(out, 1)


class CoreBase:
    """The frozen tendency core as a map of the full 67-channel state: (x, doy, tisr_next) -> next-day state.
    ``prev`` (history cores): the full state one day before x; None = no trend.

    A memory core keeps its state across the steps, so it can only be stepped along ONE trajectory at a time
    (``stateful=True``, a coupled rollout): ``start(history, doy, x0)`` warms the memory up on the days before the
    IC, then every call advances the memory with the state it is given. In a coupled rollout that is the mixed
    state, so the memory integrates the trajectory actually followed, and a Mori-Zwanzig core sees the difference
    between the mixed state and its own last forecast as the innovation."""

    def __init__(self, core, stateful: bool = False):
        self.mem = getattr(core, "mem", 0)
        if self.mem and not stateful:  # e.g. inside KAI's training, where every batch is a new trajectory
            raise NotImplementedError("a memory core is stepped along one trajectory only (CoreBase(core, stateful=True))")
        dev = next(core.parameters()).device  # the index of the dynamic channels lives where the core does
        self.core, self.dyn = core, torch.tensor(DYN, device=dev)
        self.hist = getattr(core, "hist", False) and not self.mem

    def start(self, history, doy, x0):
        """history (B, W, 67, H, W): the full states of the W days before the IC x0 (B, 67, H, W), oldest first."""
        self.core.reset(None if history is None else history.index_select(-3, self.dyn), doy.float() * 24.0,
                        x0[:, self.dyn])

    def __call__(self, x, doy, tisr_next=None, prev=None):
        if self.hist:
            self.core.prev = None if prev is None else prev[:, self.dyn]
        return x.index_copy(1, self.dyn, core_step(self.core, x[:, self.dyn], doy, tisr_next))  # orog copied

    def at(self, x, doy, tisr_next=None):
        """The core's one-day forecast from ANOTHER state of the same day (the merged variables of a KAI stage,
        inject dec). A memory core reads the memory of the trajectory it is stepped along (the call above, once a
        day) and does not advance it."""
        if self.hist:
            raise NotImplementedError("a history core holds one trend: it is not evaluated at a stage's state")
        xd = x[:, self.dyn]
        y = self.core._advance(xd, doy.float() * 24.0, tisr_next) if self.mem else core_step(self.core, xd, doy, tisr_next)
        return x.index_copy(1, self.dyn, y)


# ------------------------------------------------------------------------------- KAI with the core inside
@dataclass
class KAIPosConfig:
    depths: tuple = (3, 3, 15, 3)        # KAI.yaml / models.py default
    dims: tuple = (48, 96, 192, 288)
    form: str = "full"
    inject: str = "none"

    @classmethod
    def arm(cls, name: str, depths, dims) -> "KAIPosConfig":
        form, inject = ARMS[name]
        return cls(depths=tuple(depths), dims=tuple(dims), form=form, inject=inject)


class KAIPos(nn.Module):
    """x (B, 67, H, W) z-scored, doy (B,), tisr_next (B, H, W) -> next-day state (B, 67, H, W).

    ``core`` is the frozen first stage, a callable (x, doy, tisr_next) -> next state: CoreBase(tendency core),
    or a trained KAI for the control arm ``stack`` (not a submodule: neither trained nor saved with KAI)."""

    def __init__(self, cfg: KAIPosConfig, inc_std: np.ndarray, core=None, gate_init=0.0):
        """``gate_init`` (form sgate): the start g0 of the core's weight in the output mix, a number or one value
        per channel (67); 0 = the model starts as KAI, the loop's weights = it starts as the loop."""
        super().__init__()
        self.cfg = cfg
        n = contract.N_CHANNELS
        self.kai = KAI(in_chans=n * (2 if cfg.inject in ("in", "hist") else 1), out_chans=n,
                       depths=list(cfg.depths), dims=list(cfg.dims))
        if cfg.form in ("res", "core"):
            last = self.kai.core_model.head[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        if cfg.form == "gate":
            self.gate = nn.Parameter(torch.zeros(1, n, 1, 1))
        if cfg.form == "sgate":
            g0 = torch.as_tensor(gate_init, dtype=torch.float32).expand(n) if np.ndim(gate_init) == 0 \
                else torch.as_tensor(gate_init, dtype=torch.float32)
            self.gate = nn.Parameter(g0.clone().view(1, n, 1, 1))
            self.gate_head = nn.Conv2d(cfg.dims[-1], n, 1)
            nn.init.zeros_(self.gate_head.weight)
            nn.init.zeros_(self.gate_head.bias)
        if cfg.inject in ("all", "dec"):
            self.projs = nn.ModuleList(nn.Conv2d(n, d, 1) for d in cfg.dims)
            for q in self.projs:
                nn.init.zeros_(q.weight)
                nn.init.zeros_(q.bias)
            if cfg.inject == "dec":
                # the merge of a stage's channels to the 67 variables, in the form of KAI's own (its head); none
                # where the stage stands at the input itself (MERGE_AT 0: the merge is x)
                self.decs = nn.ModuleDict({str(i): nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GELU(),
                                                                 nn.Conv2d(d, n, 1))
                                           for i, d in enumerate(cfg.dims) if MERGE_AT[i] > 0})
                for q in self.decs.values():
                    nn.init.zeros_(q[-1].weight)
                    nn.init.zeros_(q[-1].bias)
        elif cfg.inject.startswith("s"):
            self.stage = int(cfg.inject[1]) - 1
            self.proj = nn.Conv2d(n, cfg.dims[self.stage], 1)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        self.register_buffer("s", torch.as_tensor(inc_std, dtype=torch.float32).view(1, -1, 1, 1))
        keep = torch.ones(n)
        keep[OROG] = 0
        self.register_buffer("keep", keep.view(1, -1, 1, 1))
        self.register_buffer("dyn", torch.tensor(DYN))
        # inject dec: the channels of a stage's merged state that may leave the input's (not the orography, not the
        # forcing tisr); not saved: a constant of the channel order
        free = keep.clone()
        free[TISR] = 0
        self.register_buffer("free", free.view(1, -1, 1, 1), persistent=False)
        self._core = [core]
        self.last_core = None
        self.ablate = False  # diagnostic: zero the injected tau (the output anchor of form "core" is kept)
        # rollout training: the gradient of a later day's loss has to pass through the frozen core's input
        # (c(x) = x + ...: without it the main path of forms "core" / "gate" is cut). Off = single-step training.
        self.core_grad = False
        # gradient checkpointing per block: KAI keeps the full grid in all four stages, so on a fine grid the
        # stored activations of a training step (about 12 x dim maps per block) are the memory limit
        self.grad_ckpt = False
        self.uses_core = cfg.form in ("core", "gate", "sgate") or cfg.inject not in ("none", "hist")
        if self.uses_core and core is None:
            raise ValueError("this arm needs the frozen tendency core")
        # a memory core (CoreBase(core, stateful=True)) is stepped along one batch of trajectories: ``rollout``
        # warms its memory up on the ``mem`` days before the IC and every step advances it with the model's state
        self.mem = getattr(core, "mem", 0) if self.uses_core else 0
        # inject all / dec, diagnostics: stages whose injection is switched off (0-based), and the size of what every
        # stage receives relative to its features (RMS ratio per call; a list per stage when switched on)
        self.stage_off, self.stage_log, self.gate_log = (), None, None
        # inject dec (a list when switched on): per call, every merging stage's own estimate of the next day
        # x + s D_i(h_i), with the gradient of the merge layers only. The trainer's merge loss and the scoring's
        # check of the merge compare it with the data.
        self.merge_pred = None

    def core_next(self, x, doy, tisr_next, prev=None):
        with torch.set_grad_enabled(self.core_grad and torch.is_grad_enabled()):
            return self._core[0](x, doy, tisr_next, prev=prev)

    def forward(self, x, doy, tisr_next=None, core_state=None, prev=None):
        """``core_state``: evaluate the core on this state instead of x (the open-loop diagnostic of ``rollout``).
        ``prev``: the state one day before (a history core's trend, or arm kaih's own input); None = no trend."""
        cfg, c, tau = self.cfg, None, None
        if self.uses_core:
            xc = x if core_state is None else core_state
            c = self.core_next(xc, doy, tisr_next, prev)
            tau = (c - xc) / self.s
        elif cfg.inject == "hist":
            tau = torch.zeros_like(x) if prev is None else (x - prev) / self.s * self.keep
        if self.ablate and tau is not None:
            tau = torch.zeros_like(tau)
        m = self.kai.core_model
        h = torch.cat([x, tau], 1) if cfg.inject in ("in", "hist") else x
        est = []
        for i in range(N_STAGES):
            h = m.downsample_layers[i](h)
            if cfg.inject in ("all", "dec") and i not in self.stage_off:
                ti = tau
                if cfg.inject == "dec" and MERGE_AT[i] > 0 and not self.ablate:
                    # merge: the input and the part a_i of the day's change this stage's channels hold; the core's
                    # own step from that state (the gradient of a later day passes the core as it does at x)
                    inc = self.decs[str(i)](h.detach()) * self.free
                    if self.merge_pred is not None:
                        est.append(x.detach() + self.s * inc)
                    xi = x + MERGE_AT[i] * self.s * inc.detach()
                    with torch.set_grad_enabled(self.core_grad and torch.is_grad_enabled()):
                        ti = (self._core[0].at(xi, doy, tisr_next) - xi) / self.s
                d = self.projs[i](ti)
                if self.stage_log is not None:
                    self.stage_log[i].append(float((d.detach().pow(2).mean() / h.detach().pow(2).mean()).sqrt()))
                h = h + d
            if self.grad_ckpt and torch.is_grad_enabled() and h.requires_grad:
                for blk in m.stages[i]:  # a block's activations are recomputed in the backward pass
                    h = torch.utils.checkpoint.checkpoint(blk, h, use_reentrant=False)
            else:
                h = m.stages[i](h)
            if cfg.inject.startswith("s") and i == self.stage:
                h = h + self.proj(tau)
        o = m.head(h)
        self.last_core = c
        if est:
            self.merge_pred.append(est)
        return self._output(x, o, c, h)

    def _output(self, x, o, c, h):
        cfg = self.cfg
        if cfg.form == "full":
            return o
        if cfg.form == "gate":
            return o + self.gate * self.keep * (c - o)
        if cfg.form == "sgate":
            # every channel, as the loop: with its weights the orography and tisr are the core's (exact) ones.
            # The value is clamped, the gradient is not (clamp's own gradient is zero AT the bounds, where the
            # loop's clipped weights of many channels start).
            g = self.gate + self.gate_head(h)
            g = g + (g.clamp(0.0, 1.0) - g).detach()
            if self.gate_log is not None:  # the weight actually used: mean per channel of this call
                self.gate_log.append(g.detach().mean((0, 2, 3)))
            return o + g * (c - o)
        return (x if cfg.form == "res" else c) + o * self.s * self.keep

    def rollout(self, x0, doy0, steps, tisr=None, open_loop=False, prev=None):
        """(B, steps, 67, H, W); tisr (B, steps, H, W) = the core's forcing on day 1..steps.

        ``open_loop`` (diagnostic, inject arms): the core runs on its OWN trajectory from x0 and only its tau
        enters KAI, so KAI's output never feeds the core (closed loop = the trained coupling, the default).
        ``prev``: the state one day before x0; later steps use the trajectory's own previous state. With a
        memory core: the ``mem`` states before x0 (B, mem, 67, H, W), oldest first (None = an empty memory)."""
        x, xc, p, out = x0, x0, prev, []
        if self.mem:
            with torch.no_grad():  # the days before the IC are data: no gradient goes into the warm-up
                self._core[0].start(prev, doy0, x0)
            p = None if prev is None else prev[:, -1]
        for k in range(steps):
            y = self(x, doy0 + k, None if tisr is None else tisr[:, k], xc if open_loop else None, prev=p)
            p = xc if open_loop else x  # the state the next step's trend starts from
            if open_loop:
                xc = self.last_core
            x = y
            out.append(x)
        return torch.stack(out, 1)


def config_dict(cfg) -> dict:
    return asdict(cfg)


__all__ = ["ARMS", "ConvGRUCell", "CoreBase", "KAIPos", "KAIPosConfig", "LpLoss", "make_core", "stateless_norm", "core_step", "core_rollout", "needs_core",
           "config_dict", "MERGE_AT", "N_STAGES", "TISR", "OROG", "DYN", "TISR_DYN", "operator_features", "CachedFeatures"]
