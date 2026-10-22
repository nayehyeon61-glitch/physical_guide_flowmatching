# climate_diffusion에서 얻은 교훈: Quasi-static 병목

이 문서는 이전 저장소 [`nayehyeon61-glitch/climate_diffusion`](https://github.com/nayehyeon61-glitch/climate_diffusion)의
커밋된 결과를 다시 읽고, 그 모델이 **"첫 스텝 이후 거의 움직이지 않는(quasi-static)"** 취약점을
정리한 것입니다. 목적은 PG-CFM(`PINN_FlowMatching_NeurIPS_KO.pdf`) 구현이 같은 함정에 빠지지 않도록
설계 조건을 미리 고정하는 것입니다. 새로운 실험은 하지 않았고, 모든 수치는 아래 출처의 JSON/README에서
직접 읽거나 그 값으로 계산한 것입니다.

## 1. 출처

| 출처 (커밋) | 내용 |
|---|---|
| `docs/results/dynamics-summary.json` (`bae217f`, `feature/latent-dynamics-flow`) | latent dynamics flow 4개 run의 lead별 RMSE, AE reconstruction RMSE, persistence/climatology, ensemble spread |
| `docs/results/autoencoder-probe.json` (같은 커밋) | AE 단독 holdout RMSE, PCA floor |
| `docs/results/temporal-120h-era5-run-001/` (`3e79970`, PR #8 head) | 실제 ERA5 6h + RTX 4090, lead-conditioned manifold MoE, 120h member 궤적 |
| `docs/results/physical-recurrent-run-001/` (`6caf8a6`, `results/physical-recurrent-run-001`) | 같은 데이터에서 물리시간 재귀 residual FM, 8 member 120h 궤적 |
| `docs/results/recurrent-smoke-final/` (`6caf8a6`) | 합성 데이터에서 baseline / recurrent / drift-only 비교 |

데이터는 모두 ERA5 msl/t2m/u10/v10, 6시간 간격, `state_dim=2048`(4×16×32), train 통계로 표준화한 좌표입니다.
단일 seed, 대부분 단일 origin(2009-05-19T06Z)입니다.

## 2. 관측된 현상

### 2.1 Latent dynamics: 예측 오차 = AE 복원 오차 (lead와 무관)

`dynamics-summary.json`의 4개 run 모두에서 모델 RMSE는 lead에 거의 의존하지 않고 AE reconstruction RMSE에 붙어 있습니다.

| run | latent | AE recon RMSE | 모델 RMSE 6h / 24h / 마지막 lead | 모델 6h ÷ AE recon | persistence 6h / 24h | ensemble spread |
|---|---:|---:|---|---:|---|---|
| dyn-h24 (144h) | 128 | 0.871 | 0.874 / 0.872 / 0.878 | 1.003 | 0.445 / 0.724 | ≈0.06 |
| dyn-h120 (720h) | 128 | 0.874 | 0.878 / 0.872 / 0.883 | 1.005 | 0.447 / 0.740 | ≈0.09 |
| dyn-h120-ens (+CRPS) | 128 | 0.878 | 0.883 / 0.877 / 0.886 | 1.005 | 0.447 / 0.740 | ≈0.19 |
| fix-h24-l512 | 512 | 0.848 | 0.856 / 0.852 / 0.871 | 1.009 | 0.445 / 0.724 | ≈0.12 |

해석:

- persistence RMSE(6h)는 곧 **실제 6h tendency의 RMS**(≈0.445)입니다. AE 복원 오차(0.85–0.88)는 그 약 **2배**,
  24h tendency(0.72–0.74)보다도 **약 1.2배** 큽니다. 별도로 학습한 최선의 AE probe도 holdout 0.39–0.44로
  6h tendency와 거의 같은 크기(비 0.88–0.98)입니다.
- 즉 decoder를 통과하는 순간 "한 스텝 동안 대기가 실제로 변하는 양"보다 큰 오차가 생깁니다.
  모델이 latent에서 무엇을 하든 출력 오차는 복원 floor(비 1.003–1.009)에서 시작하고,
  persistence는 lead ≈ 30h까지 모델보다 좋습니다.
- latent를 128→512로 늘려도 floor는 0.87→0.85로만 내려갑니다(PCA floor는 128에서 0.40, 512에서 0.17).
  비선형 AE가 PCA floor에도 크게 못 미칩니다.
- spread는 0.06–0.19로 오차(≈0.87)의 7–22%에 그칩니다. 오차의 대부분이 member마다 같은 "복원 편향"이라
  noise로 덮을 수 없는 성분입니다.

### 2.2 Lead-conditioned manifold MoE: 첫 스텝만 움직임

`temporal-120h-era5-run-001`은 validation RMSE 0.880으로 persistence(0.981)·climatology(1.015)보다 좋았지만,
member 궤적의 tendency 진폭 비(예측 tendency RMS ÷ 실제 tendency RMS)는 다음과 같습니다(8 member ensemble-mean
tendency 기준, `members-6h/summary.json`).

| 변수 | +6h (첫 스텝) | +12h…+120h 범위 (중앙값) |
|---|---:|---|
| msl | 2.94 | 0.032–0.073 (0.046) |
| t2m | 1.26 | 0.008–0.032 (0.019) |
| u10 | 3.04 | 0.031–0.068 (0.046) |
| v10 | 2.64 | 0.029–0.061 (0.043) |

첫 스텝에서 한 번 크게 점프(초기 AE 복원 jump가 섞임)한 뒤 실제 변화량의 2–7%만 만듭니다.
RMSE가 기준선보다 좋았던 것은 "움직이지 않는 smooth 평균장"이 RMSE에서 유리하기 때문이며,
dynamics를 학습했다는 증거가 아닙니다. 같은 run의 coverage80 = 0.335, spread/skill = 0.42로 underdispersion도 동반됩니다.

### 2.3 Physical-time recurrent residual FM: 진폭은 늘었지만 RMSE는 persistence보다 나쁨

`physical-recurrent-run-001`은 `q_next = q + Δt·(drift(q) + residual_sample)`의 물리시간 재귀,
20-step 전체 역전파, `origin + decode(q_j) − decode(q_0)` 출력으로 바꿨습니다.

| 변수 | 개별 member 첫 스텝 | 개별 member 이후 범위 | ensemble-mean 이후 범위 (중앙값) |
|---|---|---|---|
| msl | 0.38–0.55 | 0.11–0.78 | 0.114–0.265 (0.152) |
| t2m | 0.08–0.35 | 0.03–1.25 (대부분 ≤0.3) | 0.040–0.133 (0.080) |
| u10 | 0.44–0.62 | 0.13–0.69 | 0.129–0.301 (0.170) |
| v10 | 0.38–0.54 | 0.11–0.57 | 0.109–0.235 (0.152) |

- 진폭은 2.2절보다 3–5배 커졌고 첫 스텝 jump는 사라졌습니다(origin-offset 덕분). 그래도 실제 변화량의 대략
  10–50%만 재현하며, t2m(일변화가 큰 변수)이 가장 약합니다.
- validation RMSE는 **1.048로 persistence(0.981)·climatology(1.015)보다 나쁩니다.** coverage80 0.49, spread/skill 0.72.
- 합성 smoke에서도 recurrent(0.714)가 persistence(0.690)를 못 넘었고, 같은 weights의 **drift-only(0.689)가 recurrent보다
  RMSE가 낮았습니다.** 학습된 residual FM 성분이 평균 정확도에 기여하지 못하고 noise처럼 작동했다는 뜻입니다.

## 3. 취약점 진단

1. **Lossy latent의 해상도가 신호보다 거칠다.** 한 스텝 tendency(6h: 0.445)보다 decoder floor(0.85–0.88)가 크면,
   latent dynamics가 정확해도 물리 공간에서는 구분되지 않습니다. 학습 손실은 floor에 막혀 "가장 그럴듯한 평균장 근처에
   머무르기"로 수렴하고, 이것이 quasi-static으로 보입니다.
2. **절대 상태를 목표로 하면 "정지"가 국소 최적이다.** 다음 상태 전체를 맞추면 손실의 대부분은 x_k 자체(이미 알고 있는 부분)를
   복원하는 데 쓰이고, 작은 tendency는 상대적으로 무시됩니다. 불확실하면 변화량을 0으로 줄이는 것이 MSE에서 유리합니다.
3. **Decoder의 Jacobian이 변화를 눌러 버린다.** `decode(q_j) − decode(q_0)`로 jump를 지워도, latent 변화 δq가 decoder를 거치며
   축소(smoothing)되면 진폭 비는 1보다 작게 남습니다(2.3절의 10–50%).
4. **Spread가 오차 구조와 맞지 않는다.** 오차의 대부분이 member 공통의 결정론적 편향(복원 floor)이므로,
   noise 기반 ensemble spread가 아무리 늘어도 coverage가 오르지 않습니다(0.33–0.49).
5. **평가 지표가 정지를 보상한다.** state RMSE만 보면 smooth한 정지 예측이 기준선을 이길 수 있습니다(2.2절).
   tendency 진폭 비와 persistence 대비 skill을 함께 보지 않으면 이 실패를 놓칩니다.

## 4. PG-CFM 설계 시사점

PG-CFM 초안은 이미 KAI 격자 `[B, 67, 72, 144]` 물리 공간에서 CFM을 수행하고 latent를 쓰지 않습니다(4.3절).
이것은 위 1·3번 취약점을 구조적으로 피하는 선택이므로 **유지**합니다. 그 위에 다음을 설계 조건으로 둡니다.

### 4.1 CFM 목표를 절대 상태가 아니라 표준화 tendency로

초안 식 (3)은 `y = x_{k+1}`입니다. 이를 다음처럼 바꾸는 것을 제안합니다.

```
Δ_k = x_{k+1} − x_k                       # 물리 공간(KAI 정규화 좌표) 24h tendency
y   = (Δ_k − μ_Δ) / σ_Δ                    # train 기간 채널별(가능하면 위도별) tendency 통계
z_τ = (1 − τ) ε + τ y,   u_τ = y − ε
x̂_{k+1} = x_k + μ_Δ + σ_Δ ⊙ z_1
```

- 손실이 "이미 아는 x_k"를 다시 맞추는 데 낭비되지 않고, 신호 전체가 변화량에 집중됩니다(취약점 2).
- σ_Δ로 나누므로 t2m처럼 일변화가 큰 채널과 작은 채널이 같은 스케일로 학습됩니다.
- noise ε가 tendency 공간에 있으므로 ensemble spread가 직접 "변화량의 불확실성"이 됩니다(취약점 4).
  초안의 "noise 크기 임의 확대 금지" 원칙과도 맞습니다.
- PINN residual(식 5–8)은 그대로 역정규화한 x̂_{k+1}(SI 단위)에서 계산하고, 그래디언트가 z_1 → θ로 흐르게 합니다.
  `Ds u ≈ Δu/Δt` 같은 시간 미분항이 곧 tendency이므로, 목표와 물리 손실이 같은 양을 다룹니다.
- 비교 실험으로 `y = x_{k+1}`(초안 그대로)과 tendency 목표를 같은 backbone/예산에서 한 번 돌려 진폭 비를 비교합니다.

### 4.2 Lossy latent/압축 단계 금지, 쓰려면 floor 조건을 먼저 통과

- 기본 설계는 latent 없이 격자 공간 multi-scale CNN/U-Net으로 velocity를 직접 출력합니다.
- 메모리 때문에 압축(AE, PCA, spectral truncation)이 꼭 필요하면 **학습 전에** 다음을 확인합니다:
  `복원 RMSE / RMS(24h tendency) ≤ 0.1` (채널별). climate_diffusion은 이 비가 ≈1.2(24h 기준), ≈2.0(6h 기준)이었습니다.
- 출력 해상도를 줄이는 pooling/coarse grid도 같은 기준으로 검사합니다(이전 run은 18×36 요청이 실제로 4×16×32로 줄어든 혼동이 있었습니다).

### 4.3 Quasi-static을 잡는 진단을 평가 기본값으로

모든 run의 validation 보고서에 다음을 넣습니다.

| 진단 | 정의 | 통과 기준(초안) |
|---|---|---|
| tendency 진폭 비 | lead별·채널별 `RMS(x̂_{k+1} − x̂_k) / RMS(x_{k+1} − x_k)`, member별과 ensemble-mean 둘 다 | member별 0.7–1.3, 1일차 이후에도 유지 |
| persistence skill | `1 − RMSE_model / RMSE_persistence` (lead별) | 1일차부터 > 0 |
| tendency 상관 | lead별 `corr(x̂_{k+1} − x̂_k, x_{k+1} − x_k)` | > 0 이고 lead에 따라 완만하게 감소 |
| spread/skill, coverage80 | 기존과 동일 | spread/skill ≈ 1, coverage ≈ 0.8 |
| drift-only/zero-noise ablation | ε = 0 단일 경로, 그리고 tendency = 0(persistence) | 생성 경로가 둘 다보다 CRPS·RMSE 우수 |

state RMSE만으로 "기준선보다 좋다"고 판단하지 않습니다(취약점 5).

### 4.4 Rollout 학습과 물리 손실의 주의점

- climate_diffusion은 `trajectory-edges 2`(일부 구간만 감독)에서 진폭이 3–8%, 전체 20구간 역전파에서 10–50%였습니다.
  PG-CFM도 1-step teacher forcing만으로 끝내지 말고, 짧은 rollout(예: 2–4일)에서 tendency 진폭 비를 감시합니다.
- 초안 7절 넷째 한계(물리 손실이 spread를 줄일 수 있음)는 이번 underdispersion과 같은 방향의 위험입니다.
  PINN 가중치를 올릴 때마다 진폭 비와 spread/skill이 같이 떨어지는지 확인합니다.
- 6h → 24h(일평균)로 바뀌면 tendency 크기 대비 비율이 달라집니다. 표준화 통계(μ_Δ, σ_Δ)는 반드시 24h 일평균 pair에서 다시 계산합니다.

## 5. 이 정리의 한계

- 모든 근거는 단일 seed, 대부분 단일 origin이며, 원본 ERA5 archive와 checkpoint 재계산은 하지 않았습니다.
- 4변수(msl/t2m/u10/v10), 6h, 2048차원 상태라 PG-CFM의 67채널·24h·72×144와 직접 같지 않습니다.
  "latent floor가 tendency보다 크면 quasi-static이 된다"는 메커니즘 수준의 교훈으로 사용합니다.
- 진폭 비 0.7–1.3 등의 통과 기준은 이 문서의 제안값이며, 첫 CFM 단독 실험에서 다시 조정합니다.
