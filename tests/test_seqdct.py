import torch
import torch_dct

from pgcfm import contract
from pgcfm.cfm import TISR_DYN
from pgcfm.seqdct import PINNConfig, PINNProjector, SeqDCT, SeqFM, SeqFMConfig, band_spread, lowpass

C, H, W, K = contract.N_DYNAMIC, 32, 64, 3



def test_matrix_dct_matches_torch_dct():
    from pgcfm.seqdct import dct_2d, dct_t, idct_2d, idct_t
    x = torch.randn(2, 5, 3, 8, 16)
    assert torch.allclose(dct_2d(x), torch_dct.dct_2d(x, norm="ortho"), atol=1e-5)
    assert torch.allclose(dct_t(x), torch_dct.dct(x.movedim(1, -1), norm="ortho").movedim(-1, 1), atol=1e-5)
    assert torch.allclose(idct_2d(dct_2d(x)), x, atol=1e-5) and torch.allclose(idct_t(dct_t(x)), x, atol=1e-5)

def test_seq_dct_is_lossless_and_standardized():
    d = torch.randn(5, K, C, H, W) * 0.3 + 0.1
    for spatial in (False, True):
        b = SeqDCT(K, C, H, W, spatial=spatial)
        b.fit([d])
        a = b(d)
        assert torch.allclose(b.inverse(a), d, atol=1e-4)
        assert torch.allclose(a.std(0), torch.ones_like(a[0]), atol=1e-3)
        assert b.noise((2, K, C, H, W)).shape == (2, K, C, H, W)


def test_lowpass_is_a_projection_on_low_modes():
    x = torch.randn(2, 4, H, W)
    y = lowpass(x, 8, 16)
    assert torch.allclose(lowpass(y, 8, 16), y, atol=1e-5)
    c = torch_dct.dct_2d(x - y, norm="ortho")
    assert c[..., :8, :16].abs().max() < 1e-4  # only the other modes are left in x - y


def test_pinn_starts_as_identity_and_changes_only_low_modes():
    P = PINNProjector(PINNConfig(K=K, hidden=32), torch.randn(8, H, W))
    x0, d, h0 = torch.randn(2, C, H, W), torch.randn(2, K, C, H, W), torch.tensor([6.0, 30.0])
    assert torch.equal(P(x0, d, h0), d)  # zero head
    for q in P.net.head.parameters():
        q.data.normal_(0, 0.1)
    out = P(x0, d, h0)
    c = torch_dct.dct_2d(out - d, norm="ortho")
    assert c[..., :8, :16].abs().max() > 0
    assert c[..., 8:, :].abs().max() < 1e-4 and c[..., :, 16:].abs().max() < 1e-4
    assert torch.equal(out[:, :, TISR_DYN], d[:, :, TISR_DYN])  # tisr is a known forcing


def test_band_spread_splits_low_and_high():
    e = torch.zeros(4, 1, 1, 1, H, W)
    e[:, ..., :, :] = torch.randn(4, 1, 1, 1, 1, 1) * torch.cos(torch.pi * (torch.arange(W) + 0.5) / W)
    s = band_spread(e, 8, 16)
    assert s["low"] > 0 and s["high"] < 1e-5 * s["low"]


def test_seq_fm_loss_and_sample_for_both_bases():
    for basis in ("dct", "grid"):
        fm = SeqFM(SeqFMConfig(K=K, basis=basis, model_channels=32, channel_mult=(1, 2)), torch.randn(8, H, W))
        hist, d, h0 = torch.randn(2, 3, C, H, W), torch.randn(2, K, C, H, W) * 0.2, torch.tensor([0.0, 6.0])
        fm.fit_basis([d])  # two samples: a = +-1/sqrt(2) per coefficient
        assert torch.allclose(fm.to_incr(fm.to_coef(d)), d, atol=1e-4)
        loss = fm.loss(hist, d, h0)
        loss.backward()
        assert torch.isfinite(loss) and 1.2 < loss.item() < 1.8  # zero output layer: E|eps - a|^2 = 1.5
        assert fm.sample(hist, h0, steps=1).shape == d.shape



def test_guided_sampling_lowers_the_guided_loss_and_keeps_the_noise_pairing():
    from pgcfm.seqdct import SeqRegressor, hard_constraints  # noqa: F401
    torch.manual_seed(0)
    fm = SeqFM(SeqFMConfig(K=K, basis="dct", model_channels=32, channel_mult=(1, 2)), torch.randn(8, H, W))
    fm.fit_basis([torch.randn(6, K, C, H, W) * 0.2])
    for q in fm.unet.out.parameters():  # the U-Net's output layer starts at zero: v = 0 means no guidance scale
        q.data.normal_(0, 0.05)
    hist, h0 = torch.randn(2, 3, C, H, W), torch.tensor([0.0, 6.0])
    eps = fm.noise(2)
    phys = lambda d: (d[:, :, 0] ** 2).sum()  # noqa: E731  (a stand-in physics loss on channel 0)
    raw = fm.sample(hist, h0, steps=4, eps=eps)
    assert torch.equal(raw, fm.sample(hist, h0, steps=4, eps=eps))  # same eps, same member
    guided = fm.sample(hist, h0, steps=4, eps=eps, physics=phys, lam=0.5, s0=0.5)
    assert phys(guided) < phys(raw)
    assert torch.equal(fm.sample(hist, h0, steps=4, eps=eps, physics=phys, lam=0.0), raw)
    # the PINN inside the ODE: a projector that shrinks channel 0 pulls the member toward its fixed point
    shrink = lambda d: torch.cat([0.5 * d[:, :, :1], d[:, :, 1:]], 2)  # noqa: E731
    inloop = fm.sample(hist, h0, steps=4, eps=eps, project=shrink, mu=1.0, s0=0.5)
    assert phys(inloop) < phys(raw)
    assert torch.allclose(inloop[:, :, 1:], raw[:, :, 1:], atol=0.5 * float(raw[:, :, 1:].abs().max()))
    assert torch.equal(fm.sample(hist, h0, steps=4, eps=eps, project=shrink, mu=0.0), raw)
    # branching from the kept unguided state and chunking the gradient give the same members
    same, state = fm.sample(hist, h0, steps=4, eps=eps, keep_at=0.5)
    assert torch.equal(same, raw) and state[1] == 2
    assert fm.sample(hist, h0, steps=2, eps=eps, keep_at=0.75)[1][1] == 1  # s0 inside the last step
    branch = fm.sample(hist, h0, steps=4, start=state, physics=lambda d, j: phys(d), lam=0.5, s0=0.5, chunk=1)
    assert torch.allclose(branch, guided, atol=1e-5)


def test_regressor_and_hard_constraints():
    from pgcfm.seqdct import SeqRegressor, hard_constraints
    reg = SeqRegressor(SeqFMConfig(K=K, model_channels=32, channel_mult=(1, 2)), torch.randn(8, H, W))
    hist, d, h0 = torch.randn(2, 3, C, H, W), torch.randn(2, K, C, H, W), torch.tensor([0.0, 6.0])
    reg.fit_scale([d])
    reg.errors.fit([d])
    w = torch.ones(1, 1, 1, H, 1)
    loss = reg.loss(hist, d, h0, w)
    loss.backward()
    assert torch.isfinite(loss) and reg.sample(hist, h0).shape == d.shape
    mean, std = torch.zeros(contract.N_CHANNELS).numpy() + 1.0, torch.ones(contract.N_CHANNELS).numpy()
    x0 = torch.zeros(2, C, H, W)
    fixed, frac = hard_constraints(x0, d, mean, std)  # physical q = 1 + x: q < 0 where x < -1
    q = contract.DYNAMIC_CHANNELS.index(contract.CHANNEL_INDEX["q500"])
    assert 0 < frac < 1 and (fixed[:, :, q] >= -1 - 1e-6).all()
    t = contract.DYNAMIC_CHANNELS.index(contract.CHANNEL_INDEX["t500"])
    assert torch.equal(fixed[:, :, t], d[:, :, t])
