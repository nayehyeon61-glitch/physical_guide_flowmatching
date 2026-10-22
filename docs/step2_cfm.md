# 2단계: CFM 단독 앙상블 — 구현, 소형 실험, 약점 분석

기준: 논문 초안 `PINN_FlowMatching_NeurIPS_KO.pdf` (PG-CFM) 4장과 6.2절, 부록 C의 순서 "CFM 단독 → 물리 진단 → 결합" 중 첫 단계. PINN은 아직 넣지 않았다.

## 구현한 것

| 파일 | 내용 |
| --- | --- |
| `pgcfm/cfm.py` | 한 번의 24 h 전이 p(x_{k+1} \| x_k, c_k). [torchcfm](https://github.com/atong01/conditional-flow-matching) v1.0.7을 그대로 사용 |
| `pgcfm/metrics.py` | 면적 가중 fair CRPS, 앙상블 평균 RMSE, spread, spread–skill 비 |
| `scripts/run_step2_cfm.py` | 합성 자료 실험: tisr 생성 vs forcing 두 모델 학습, oracle 앙상블과 1–24일 rollout 비교 |
| `scripts/step2_diagnostics.py` | 과분산 원인 진단: Heun step 수, 멤버 편차의 공간 구조, 추가 학습 |
| `tests/test_cfm.py` | 9개 (식 (3) 일치, 미분 가능한 샘플러, 경도 주기성, Markov rollout, 점수 함수) |
| `docs/step2_results/*.json` | 아래 표의 원자료 |

torchcfm에서 가져온 것과 새로 쓴 것:

- **확률 경로와 목표 속도**: `ConditionalFlowMatcher(sigma=0)`에 source=ε, target=y=x_{k+1}을 넣으면 정확히 논문 식 (3) z_τ=(1−τ)ε+τy, u=y−ε이다 (`test_cfm_target_matches_paper_eq3`). 전이마다 새 ε을 뽑는 독립 coupling이다.
- **backbone**: torchcfm의 guided-diffusion `UNetModel` (다중 스케일 CNN + τ sinusoidal embedding, 논문 4.3절). 입력은 z_τ와 조건 [x_k 66, orog 1, 다음 날 달력 sin/cos 2, (forcing 모드) tisr_{k+1} 1]을 채널로 이어 붙인다. 112만 파라미터 (channels 32, mult 1-2-2).
- **새로 쓴 것**: 면적 가중 손실 W (식 (4), cos φ), Heun 적분기, 멤버별 Markov rollout (앙상블 평균을 되먹이지 않음, 6.2절), 경도 주기 패딩.
- **PINN 결합을 위한 API**: `sample(..., grad=True)`는 no_grad 없이 Heun 전 단계의 graph를 유지해 endpoint 손실의 gradient가 θ까지 간다 (`test_sample_is_differentiable_for_physics_loss`). 목표는 tendency가 아닌 z-score 상태 x_{k+1}이다.

### 구현 중 고친 것
1. **경도 주기성** (Codex 리뷰): torchcfm U-Net은 모든 conv를 zero padding하므로 0°와 358.5°가 서로 무관한 이미지 끝이 된다. 모든 Conv2d의 padding을 경도 circular / 위도 replicate로 바꿨다. 경도 roll 등변성 테스트로 확인했고, 수정 전 코드에서는 이 테스트가 실패한다.
2. **합성 tisr의 정의 (1단계 버그)**: 1단계 생성기는 tisr를 24 h 적분(×86400 s)으로 만들어 정규화 값이 약 50σ였다. KAI 통계(평균 1.071e6, 표준편차 6.04e5 J m⁻²)는 **1시간 누적값을 00/06/12/18 UTC에서 평균**한 값과 맞는다(수정한 생성기: 평균 1.13e6, 표준편차 6.9e5). 따라서 KAI의 tisr는 날짜와 위도뿐 아니라 **경도에도 의존**한다(네 시각 표본이 일주기를 aliasing). 이것은 실제 파일로 확인해야 할 추론이다.

## 실험 설정

- 자료: `make_synthetic_era5.py`의 KAI 형식 합성 자료, 25×48 격자, 학습 8년 × 120일 (952쌍), 시험 2020년 37일.
- 합성 자료의 전이는 z-score 공간의 AR(1)이다: a_{k+1}=0.8a_k+0.6η, x=mean+0.5·std·a. η는 저파수(경도 파수 ≤4) 매끄러운 장. 그래서 **참 조건부 분포를 직접 샘플링하는 oracle 앙상블**을 만들 수 있고, "CFM의 분산 오류"와 "문제가 원래 어려운 것"을 구분할 수 있다.
- 학습: Adam 2e-4, cosine, 3000 iter, batch 16, CPU 4코어에서 모드당 약 19분.
- 평가: 시작일 4개 × 16 멤버 × 24일 rollout, Heun 8 step (NFE 16). 점수는 AR 채널(z, t, u, v, t2m, tcwv, msl, skt)의 평균. q(생성기에 abs()가 있어 oracle이 근사), sp, tisr는 따로 본다.

## 결과

### 1–24일 rollout (AR 채널, z-score 단위)

| lead | 모델 | ens-mean RMSE | spread | CRPS | SSR | 멤버 진폭 |
| --- | --- | --- | --- | --- | --- | --- |
| 1일 | oracle | 0.227 | 0.230 | 0.119 | 1.05 | 0.385 |
| | CFM (tisr 생성) | 0.384 | 0.747 | 0.232 | 2.00 | 0.760 |
| | CFM (tisr forcing) | 0.384 | 0.760 | 0.234 | 2.04 | 0.771 |
| | persistence | 0.231 | – | – | – | – |
| 3일 | oracle / 생성 / forcing | 0.346 / 0.437 / 0.442 | 0.333 / 0.824 / 0.837 | 0.188 / 0.262 / 0.265 | 0.99 / 1.94 / 1.95 | |
| 10일 | oracle / 생성 / forcing | 0.382 / 0.432 / 0.433 | 0.387 / 0.845 / 0.845 | 0.204 / 0.262 / 0.262 | 1.04 / 2.02 / 2.01 | |
| 24일 | oracle / 생성 / forcing | 0.402 / 0.446 / 0.446 | 0.387 / 0.848 / 0.846 | 0.214 / 0.267 / 0.267 | 0.99 / 1.96 / 1.96 | 0.386 / 0.848 / 0.846 |

참값의 장 진폭은 모든 lead에서 0.38–0.39, persistence RMSE는 0.231 / 0.385 / 0.504 / 0.558 (1/3/10/24일)이다. SSR = √((M+1)/M)·spread/RMSE, 1이면 보정된 앙상블이다.

### tisr: 생성 대상 vs forcing

| lead | 1일 | 3일 | 5일 | 10일 | 24일 |
| --- | --- | --- | --- | --- | --- |
| 생성: RMSE / spread | 0.29 / 0.86 | 0.58 / 0.89 | 0.77 / 0.89 | 1.00 / 0.91 | 0.97 / 0.91 |
| forcing: RMSE / spread | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

### 과분산 원인 진단 (forcing 모델, 1일, 시작일 12개)

| 조건 | spread | RMSE | CRPS |
| --- | --- | --- | --- |
| oracle | 0.233 | 0.245 | 0.130 |
| Heun S = 1 / 2 / 4 | 0.817 / 0.768 / 0.761 | 0.401 / 0.391 / 0.389 | 0.246 / 0.237 / 0.235 |
| Heun S = 8 / 16 / 32 | 0.762 / 0.762 / 0.762 | 0.390 / 0.391 / 0.391 | 0.236 / 0.236 / 0.236 |
| 학습 6000 iter (2배), S = 8 | 0.748 | 0.387 | 0.233 |

| 멤버 편차(멤버 − 앙상블 평균)의 공간 구조 | CFM | oracle |
| --- | --- | --- |
| 경도 인접 격자 상관 | 0.03 | 0.97 |
| 경도 파수 > 4의 분산 비율 | 78% | 0% |
| 경도 파수 > 12의 분산 비율 | 47% | 0% |

## 약점 분석

### A. 과분산: 앙상블이 "조건부 평균 + 걸러지지 않은 white noise"가 된다 (가장 중요)
1. **SSR ≈ 2, spread는 oracle의 2.2–3.3배**이다. 두 모드 모두 같다. 1일 ens-mean RMSE 0.384는 persistence(0.231)와 oracle(0.227)보다 나쁘다. 즉 이 설정의 CFM은 **1일 예측에서 persistence를 못 이긴다**.
2. **적분기 문제가 아니다**: Heun S ≥ 4에서 값이 수렴한다.
3. **단순 학습 부족만은 아니다**: 학습을 2배로 늘려도 spread는 0.762 → 0.748로 2%만 줄었다.
4. **원인은 공간 구조**: 멤버 편차의 인접 격자 상관이 0.03으로 사실상 white noise이고, 분산의 78%가 참 분포에는 없는 경도 파수 > 4에 있다. 네트워크가 초기 ε의 격자 규모 성분을 지우지 못하고 endpoint까지 남긴다. 그래서 spread가 채널과 무관하게 거의 같다: 채널별 spread/oracle 비가 2.8–3.7배로 고르고, 참 조건부 분산이 작은 sp는 18배다. **CFM이 채널별 조건부 분산을 학습하지 못하고 있다**.
5. **주의**: 합성 자료의 참 잡음은 경도 파수 ≤ 4의 매우 낮은 차원 분포이다. 실제 ERA5의 오차 스펙트럼은 넓고 red하므로 배율은 달라질 것이다. 다만 "작은 자료(952쌍)와 독립 coupling에서 격자 규모 ε이 남는다"는 실패 방식은 실제 자료에서도 측정해야 할 항목이다.

**PINN 결합(3–4단계)에 주는 의미**: 미분 기반 물리 잔차는 white noise를 격자 간격의 역수만큼 증폭한다. 이 상태에서 L_phys를 붙이면 물리 손실의 대부분이 이 잡음을 지우는 데 쓰이고, PG-CFM이 CFM보다 CRPS가 좋아지더라도 **물리 때문이 아니라 단순한 평활화 효과**일 수 있다. 따라서 결합 실험에는 "CFM + 스펙트럼 평활 penalty" 같은 비물리 대조군을 반드시 넣어야 한다(논문 6.3절 표에 추가 제안).

**논문 수정 제안**: 4.1절은 "spread는 학습한 조건부 분포의 결과이며 inflation을 쓰지 않는다"고 쓰지만, 여기서는 반대로 기본 설정이 **과분산**이다. 다음 중 하나를 CFM 기준선 단계에서 비교할 것을 제안한다. (a) 공간 상관이 있는 base noise (자료의 24 h 변화량 스펙트럼에 맞춘 Gaussian 장, 여전히 CFM), (b) 목표를 상태 대신 정규화된 변화량 (x_{k+1}−x_k)/σ_Δ로 두기 (목표 분산이 줄어 ε 제거 부담이 줄어든다), (c) 더 큰 backbone과 자료. 어느 것도 아직 실험하지 않았다.

### B. Rollout drift
6. 24일 동안 발산하지 않는다. 멤버 진폭은 3일 무렵 0.85에서 포화되어 **참 기후 진폭(0.38)의 2.2배로 계속 머문다**. 모든 멤버가 rollout 내내 자료 manifold 밖(잡음이 섞인 상태)에 있다.
7. 학습 입력은 깨끗한 x_k인데 rollout 입력은 잡음이 섞인 생성 상태이다(exposure bias). 그래도 발산하지 않는 것은 네트워크가 입력을 수축시키기 때문으로 보이고, 이것은 합성 자료의 AR 감쇠 0.8이 돕는 것일 수 있다. 감쇠가 약한 실제 대기에서는 drift가 더 클 수 있다.
8. ens-mean RMSE가 3일 이후 persistence보다 좋은 것은 기술이 아니라 기후값으로 수렴한 결과이다(10일 0.433 ≈ oracle 0.382에 근접, CRPS는 여전히 oracle보다 28% 나쁨).

### C. tisr를 생성할 것인가
9. 생성 대상으로 두면 결정적 채널에 0.86–0.91σ의 spread가 생기고, 되먹임 때문에 오차가 0.29σ(1일)에서 **10일에 1.0σ, 즉 기후 표준편차 수준**까지 커진다. 10일 뒤의 tisr 정보는 사실상 사라진다. forcing으로 두면 오차가 정확히 0이다.
10. 이 합성 자료에서는 다른 채널이 tisr에 의존하지 않으므로 다른 채널 점수 차이는 없다(1일 RMSE 0.384 vs 0.384). 실제 자료에서는 tisr가 t2m, skt를 구동하므로 이 결론을 그대로 옮길 수 없다.
11. **권고**: tisr를 forcing 조건으로 두는 것을 기본값으로 바꾼다. 이는 KAI 계약(67채널 모두 예측)과 다르므로 KAI와 비교할 때는 orog와 tisr를 모두 지표에서 제외한다.

### D. 계산량 (전체 121×240 격자, CPU 4코어)
12. 학습 1 iter(batch 2) 약 7.5초, 생성(batch 2, Heun 8 step) 약 32초. `grad=True`로 Heun 4 step을 거쳐 역전파하면 batch 1에 약 48초, 최대 메모리 약 2.9 GB였다(다른 실험과 동시에 측정해 대략적인 값이다). 결합 학습은 step 수에 비례해 메모리가 늘어나므로 GPU와 함께 마지막 몇 step만 미분하거나 adjoint 방식이 필요할 수 있다.

### E. 이 실험의 한계
13. 합성 자료, 952쌍, 25×48 격자, 112만 파라미터, 3000 iter, seed 1개, 시작일 4개(진단 12개). q 채널은 oracle 근사 때문에 비교에서 뺐다. 실제 ERA5 성능에 대한 주장은 하지 않는다.

## 다음 단계로 넘길 결정
- tisr forcing을 기본값으로 할지 (권고: 예).
- PINN 결합 전에 A의 white-noise 잔여를 줄이는 방법 (a)/(b)/(c) 중 무엇을 CFM 기준선에 넣을지.
- 실제 ERA5 파일에서 tisr가 경도에 의존하는지 확인.

재현:
```
python scripts/run_step2_cfm.py --out runs/step2        # 약 45분 (CPU 4코어)
python scripts/step2_diagnostics.py --run runs/step2     # 약 25분
python -m pytest -q
```
