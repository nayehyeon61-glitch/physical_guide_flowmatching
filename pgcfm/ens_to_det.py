"""보정 ensemble -> neural operator -> 결정론적 기후장: 학습용 축약 예제.

실제 저장소/MDM/PINN 구현이 아닙니다. 입력은 사용자가 이미 보정한 ensemble.
공식 neuraloperator 2.0.0의 neuralop.models.FNO를 사용합니다.
자체 SpectralConv/FFT 구현이나 대체 fallback은 없습니다.
설치 (PyTorch는 사용 환경에 맞는 CPU/CUDA 빌드를 먼저 설치):
  python -m pip install neuraloperator==2.0.0
실행 (CPU float32 예제):
  python ensemble_to_deterministic.py --test
  python ensemble_to_deterministic.py --demo --epochs 40

입력 members [B,K,C,H,W], 선택적 context [B,Q,H,W], 출력 [B,C,H,W].
공유 FNO로 각 member의 공간 패턴을 비선형 인코딩한 뒤 member 평균 pooling.
Pooling은 원본 평균과 다름: mean(phi(x_k)) != phi(mean(x_k)) 일반적으로.
Member 순서 불변, K 가변. 유한 차원 pooling이 ensemble의 모든 정보를
보존하거나 모든 member 상호작용을 표현한다고 보장하지는 않습니다.

단계: 생성기/E/D/PINN/MDM 고정 -> train으로 이 head 학습 -> validation으로
checkpoint 선택 -> test 1회 평가. MDM은 test target을 보지 않아야 하며,
operator validation/test와 분리한 calibration split 또는 cross-fitting으로
정해야 합니다. 코드의 detach는 고정 ensemble cache 경계를 명시합니다.
실제 ERA5는 시간순 split + 겹치는 forecast window 누출 방지가 필요합니다.

정규화된 입력/target으로 학습한다고 가정. 실제 단위 RMSE는 역정규화 후 평가.
area에는 실제 격자 면적을 넣으세요. 예제는 직사각형, Fourier 주기 경계입니다.
실제 구면용 operator/경계 처리와 실제 물리 방정식은 별도 연결해야 합니다.
physics_fn(prediction, batch_indices, split_name) -> scalar를 지원합니다.
단위/격자/dt는 사용자 책임입니다.
결정론적 단일 출력은 최종 예측분포/확률 보정 성능을 나타내지 않습니다.
두 FNO에는 normalized rectangular grid 위치 encoding을 사용합니다.
이는 실제 위경도/구면 metric을 대신하지 않습니다.
이전 자체 구현과 가중치 구조가 달라 기존 checkpoint는 재학습해야 합니다.

참고 FNO: https://neuraloperator.github.io/dev/theory_guide/fno.html
"""
import argparse
import copy
import io
import json
from importlib.metadata import version
import torch
from torch import nn
from torch.nn import functional as F
try:
    from neuralop.models import FNO
except ImportError as exc:
    raise ImportError("Install official package: python -m pip install neuraloperator==2.0.0") from exc


class OfficialFNO(nn.Module):
    """Thin adapter; all operator computation is the unmodified official FNO.

    neuraloperator 2.0.0 inserts a bare '_metadata' dictionary when its FNO is
    nested in a torch Module. Remove only this constructor metadata from nested
    weight state_dicts so strict PyTorch loading works. Tensor weights and the
    OrderedDict's native PyTorch _metadata attribute are preserved.
    Reconstruct this model with the same constructor arguments before loading.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.operator = FNO(**kwargs)

    def forward(self, x):
        return self.operator(x)

    def state_dict(self, *args, **kwargs):
        weights = super().state_dict(*args, **kwargs)
        weights.pop("_metadata", None)
        return weights


class EnsembleToDeterministic(nn.Module):
    def __init__(self, channels, context_channels=0, width=12, modes=3):
        super().__init__()
        if min(channels, width, modes) < 1 or context_channels < 0:
            raise ValueError("channels/width/modes must be positive; context_channels >= 0")
        self.channels, self.context_channels = channels, context_channels
        self.modes = modes
        # mean/anomaly features are shared across members; no member-ID embeddings.
        # Official FNO includes its own lifting, spectral blocks and projection.
        self.member_encoder = OfficialFNO(
            n_modes=(modes, modes), in_channels=2 * channels,
            out_channels=width, hidden_channels=width, n_layers=2,
            positional_embedding="grid", channel_mlp_dropout=0.0,
        )
        self.global_operator = OfficialFNO(
            n_modes=(modes, modes), in_channels=width + channels + context_channels,
            out_channels=width, hidden_channels=width, n_layers=2,
            positional_embedding="grid", channel_mlp_dropout=0.0,
        )
        self.head = nn.Conv2d(width, channels, 1)
        # Initial forecast = ensemble mean. A learned residual can change that mean.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, members, context=None):
        if members.ndim != 5 or min(members.shape) < 1:
            raise ValueError("members must be nonempty [B,K,C,H,W]")
        if not members.is_floating_point() or not torch.isfinite(members).all():
            raise ValueError("members must be finite floating tensors")
        b, k, c, h, w = members.shape
        if min(h, w) < 2 * self.modes:
            raise ValueError("grid too small for configured modes; reduce modes")
        if c != self.channels:
            raise ValueError("member channel mismatch")
        mean = members.mean(1)                           # [B,C,H,W]
        anomalies = members - mean[:, None]
        features = torch.cat([members, anomalies], 2)   # [B,K,2C,H,W]
        encoded = self.member_encoder(
            features.reshape(b * k, 2 * c, h, w))       # [B*K,D,H,W]
        summary = encoded.reshape(b, k, -1, h, w).mean(1)  # [B,D,H,W]
        inputs = [mean, summary]
        if self.context_channels:
            if context is None or context.shape != (b, self.context_channels, h, w):
                raise ValueError("context must match [B,Q,H,W]")
            if not torch.isfinite(context).all():
                raise ValueError("context must be finite")
            inputs.append(context)
        elif context is not None:
            raise ValueError("configure context_channels before passing context")
        correction = self.head(self.global_operator(torch.cat(inputs, 1)))
        return mean + correction                        # [B,C,H,W]


def channel_mse(pred, target, area):
    if pred.shape != target.shape or pred.ndim != 4 or not torch.isfinite(target).all():
        raise ValueError("pred/target must match finite [B,C,H,W]")
    if area.shape != pred.shape[-2:] or not torch.isfinite(area).all() or (area <= 0).any():
        raise ValueError("area must be finite positive [H,W]")
    return ((pred - target).square() * (area / area.sum())).sum((-2, -1)).mean(0)


def mean_only(members):
    """Ablation: same shape/model/capacity, but remove all ensemble anomalies."""
    return members.mean(1, keepdim=True).expand_as(members)


def fit(model, train, valid, area, epochs=40, batch_size=8, lr=0.003,
        use_ensemble=True, physics_fn=None, physics_weight=0.0, post_step=None):
    """train/valid = (members, target, context_or_None). Returns best-val model.
    post_step(model), if given, runs after every optimizer step (e.g. an operator-norm projection).

    Only final deterministic output is compared with target; no memberwise MSE.
    Physics callback receives batch indices for selecting fixed dt/past/context.
    Signature: physics_fn(pred, indices, split_name) -> differentiable scalar.
    """
    if epochs < 1 or batch_size < 1 or physics_weight < 0:
        raise ValueError("invalid training configuration")
    if physics_weight > 0 and physics_fn is None:
        raise ValueError("physics_weight requires a physics_fn")
    model.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tx, ty, tc = (None if v is None else v.detach() for v in train)
    vx, vy, vc = (None if v is None else v.detach() for v in valid)
    if not use_ensemble:
        tx, vx = mean_only(tx), mean_only(vx)
    best_score, best_state, history = float("inf"), None, []
    for epoch in range(epochs):
        model.train()
        for ids in torch.randperm(tx.shape[0], device=tx.device).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            pred = model(tx[ids], None if tc is None else tc[ids])
            loss = channel_mse(pred, ty[ids], area).mean()
            if physics_weight:
                physical_loss = physics_fn(pred, ids, "train")
                if physical_loss.ndim != 0 or not physical_loss.requires_grad:
                    raise ValueError("physics_fn must return a differentiable scalar")
                loss = loss + physics_weight * physical_loss
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if post_step is not None:
                post_step(model)
        model.eval()
        with torch.no_grad():
            score = float(channel_mse(model(vx, vc), vy, area).mean())
        history.append(score)
        if score < best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("no finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return {"best_val_mse": best_score, "val_mse_history": history}


def synthetic_split(n, seed):
    """Toy fixed ensemble; not actual MDM output and not research evidence."""
    g = torch.Generator().manual_seed(seed)
    context = F.avg_pool2d(torch.randn(n, 2, 8, 8, generator=g), 3, 1, 1)
    target = context + 0.3 * torch.randn(n, 2, 8, 8, generator=g)
    members = context[:, None] + 0.25 + 0.5 * torch.randn(n, 6, 2, 8, 8, generator=g)
    return members, target, None


def demo(epochs):
    torch.set_num_threads(1)
    train, valid, test = [synthetic_split(n, seed) for n, seed in [(24, 10), (12, 20), (12, 30)]]
    area = torch.ones(8, 8)
    report = {"note": "SYNTHETIC ONLY; no MDM/PINN/ERA5 performance claim",
              "neuraloperator_version": version("neuraloperator"),
              "torch_version": torch.__version__}
    with torch.no_grad():
        report["ensemble_mean_rmse"] = channel_mse(test[0].mean(1), test[1], area).sqrt().tolist()
    for name, full in [("mean_only_operator", False), ("ensemble_operator", True)]:
        # Identical initialization and minibatch order for the two ablations.
        torch.manual_seed(7)
        model = EnsembleToDeterministic(2)
        log = fit(model, train, valid, area, epochs=epochs, use_ensemble=full)
        with torch.no_grad():
            x = test[0] if full else mean_only(test[0])
            prediction = model(x)
            report[name] = {"output_shape": list(prediction.shape),
                            "best_val_mse": log["best_val_mse"],
                            "test_rmse": channel_mse(prediction, test[1], area).sqrt().tolist()}
    print(json.dumps(report, indent=2))


def self_test():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    close = torch.testing.assert_close
    x = torch.randn(2, 4, 2, 8, 8, requires_grad=True)
    model = EnsembleToDeterministic(2)
    assert isinstance(model.member_encoder.operator, FNO)
    assert isinstance(model.global_operator.operator, FNO)
    assert type(model.member_encoder.operator).__module__.startswith("neuralop.")
    assert model(x).shape == (2, 2, 8, 8)
    close(model(x), x.mean(1))
    # Nontrivial head to test actual encoder behavior, not just identity init.
    nn.init.normal_(model.head.weight, std=0.1)
    nn.init.normal_(model.head.bias, std=0.1)
    pred = model(x)
    close(model(x[:, [2, 0, 3, 1]]), pred, atol=2e-6, rtol=2e-5)
    close(model(x.repeat_interleave(2, dim=1)), pred, atol=2e-6, rtol=2e-5)
    assert model(x[:, :1]).shape == pred.shape  # K=1 supported
    # Same input mean, different ensemble structures: network can distinguish them.
    assert not torch.allclose(pred, model(mean_only(x)), atol=1e-7, rtol=1e-6)
    pred.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for module in (model.member_encoder, model.global_operator):
        spectral_grads = [p.grad for name, p in module.named_parameters()
                          if "fno_blocks.convs" in name]
        assert spectral_grads and any(g is not None and torch.isfinite(g).all()
                                      and g.abs().sum() > 0 for g in spectral_grads)
    target = torch.zeros(2, 2, 8, 8)
    area = torch.ones(8, 8)
    close(channel_mse(torch.ones_like(target), target, area), torch.ones(2))
    # Context wiring and binary checkpoint round-trip (safe weights-only loader).
    cm = EnsembleToDeterministic(2, context_channels=1)
    context = torch.randn(2, 1, 8, 8)
    assert cm(x.detach(), context).shape == pred.shape
    restored = EnsembleToDeterministic(2)
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    close(restored(x.detach()), pred.detach())
    fit(EnsembleToDeterministic(2), (x, target, None),
        (x.detach(), target, None), area, epochs=2)
    # Fit on cache does not change pre-existing upstream gradients.
    prior_grad = x.grad.clone()
    fit(EnsembleToDeterministic(2), (x, target, None),
        (x.detach(), target, None), area, epochs=1)
    close(x.grad, prior_grad)
    print("PASS: official FNO, shape, initialization, permutation, variable K, state sensitivity, spectral gradients, context, checkpoint, frozen inputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.test:
        self_test()
    if args.demo:
        demo(args.epochs)
    if not args.test and not args.demo:
        parser.print_help()
