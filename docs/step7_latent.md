# 7단계: DCT latent PG-CFM (encoder 고정 여부, 병렬 vs 직렬, 앙상블 neural operator)

기준: 논문 초안 `PINN_FlowMatching_NeurIPS_KO.pdf`의 PG-CFM, 6단계(PR #7)의 약점(과대분산은 격자 규모 white noise, 물리 gradient가 CRPS gradient의 수천 배, 평활 대조군이 효과 대부분을 재현), 그리고 소유자의 스케치 `dct_encoder.py`, `ensembl_operator.py`(2026-10-23).

## 요청을 이렇게 읽었습니다

1. **stage 1**: dct-encoder → PINN → idct-decoder를 함께 학습하고 PINN을 고정합니다. PINN은 latent에서 한 스텝(24 h)을 진행하는 Raissi식 tanh MLP P(z_k)이고, 물리 손실은 3·4단계의 원시방정식 잔차를 **decode한 쌍** (D(z_k), D(P(z_k)))에 겁니다.
2. **stage 2 (병렬, `coupling="anchor"`)**: encoder → CFM + 고정 PINN → decoder. CFM은 고정 PINN 스텝 주위의 편차를 생성합니다: z_{k+1} = P(z_k) + s·δ, δ ~ CFM(ε | z_k, P(z_k), 달력). 질문은 **이때 encoder를 다시 학습할지, 고정할지**입니다.
3. **직렬 (`coupling="serial"`, 소유자 추가 요청)**: encoder → CFM이 멤버 A, B, C…를 생성 → **각 멤버를 학습 가능한 PINN Q가 보정** → decoder/IDCT. CFM과 Q는 같은 손실로 동시에 학습합니다(L_CFM + decode한 멤버 CRPS + λ_P × 멤버별 물리 잔차, gradient는 Heun sampler를 거쳐 CFM까지).
4. **앙상블 neural operator (소유자 추가 요청)**: 직렬 출력 뒤에 `ensembl_operator.py`의 EnsembleOperator(neuraloperator FNO, 멤버 공유 encoder + 앙상블 평균 pooling + corrector)를 붙여 fair energy score + 물리 잔차로 학습합니다.

## 구현 (기존 구현 재사용)

| 부분 | 가져온 것 | 스케치에서 고친 것 / 새로 쓴 것 |
| --- | --- | --- |
| DCT / IDCT | `torch-dct` (zh217/torch-dct) `dct_2d/idct_2d(norm="ortho")` | 빈 `dct()/idct()` 채움. 채널별 (lat, lon) 2-D DCT-II, 낮은 k_lat×k_lon 모드만 유지, 계수별 표준화 |
| encoder/decoder MLP | Raissi PINN의 `initialize_NN`(Xavier) + tanh (`physics.pinn.PhysicsInformedNN`) | `super().__init__()`, `nn`/`F` import, `X`/`ze` 오타, `layer2` → `self.layer2`. `F.sinh`는 없고 `torch.sinh`는 지수적으로 커져서 Raissi와 같은 tanh로 바꿈. encoder 마지막 층은 선형(latent를 [-1,1]로 누르지 않음, CFM 잡음은 N(0,I)). decoder에 없던 활성화 추가. 조건 c = 지형(DCT) + 달력 |
| latent CFM | torchcfm `ConditionalFlowMatcher(sigma=0)` + torchcfm `MLP` | Heun sampler(2단계 `cfm.heun`) 재사용 |
| 물리 잔차 | 3·4단계 `PhysicsGuide.residuals`(선형 bridge, closure 없음), 물리 gradient 상한(4단계 W2 수정) | `latent._capped_backward` |
| 앙상블 operator | neuraloperator `FNO` | 스케치 그대로 + gate 0 시작, energy score는 fair 추정(대각 제외; 스케치의 `cdist(p,p).mean()`은 대각 0을 포함해서 작은 M에서 퍼짐을 과하게 보상), 선택적 DCT band-limit |

파일: `pgcfm/latent.py`, `pgcfm/ens_operator.py`, `scripts/run_step7_latent.py`(이 문서의 실험, 원시 결과 `docs/step7/`), `scripts/step7_overdispersion.py`(과대분산 가설 두 개), `scripts/train_latent.py` + `configs/latent_era5.yaml`(실제 ERA5, GPU 한 장), `scripts/infer_pgcfm.py`(`latent.pt`도 forecast.nc로), `tests/test_latent.py`. 실행 명령은 `docs/RUN_ON_GPU.md` 8절.

## 실험 설정

6단계와 같은 균형 합성 자료(25×48, 학습 4년 × 90일 = 356쌍, 시험 2020년 4개 초기장 10일 rollout, 멤버 8, Heun 8 step), 같은 b_j, 같은 평가 함수. 그래서 grid CFM 행은 PR #7의 `cfm` 행과 같은 값입니다(1일 CRPS 0.485).

- DCT k_lat×k_lon = 8×16, latent 64, encoder 256-128-64(스케치 그대로), PINN 256-256.
- stage 1: 1500 iter, λ_P = 0.1, 물리 gradient ≤ 0.5 × 데이터 gradient. stage 2: CFM 2000 iter. retrain은 같은 2000 iter 동안 encoder/decoder도 CFM + 복원 손실로 학습. fine-tune은 frozen 뒤 decoder만 decode한 CRPS로 200 iter, lr 1e-5.
- 직렬: CFM + Q 600 iter(Heun 4 step 역전파). operator: 직렬 고정 후 FNO 200 iter, 멤버 4.
- 대조군: `cfm_dct`(grid CFM 멤버를 같은 DCT 모드로 사영, 학습 없음), `cfm_smooth`(6단계 Laplacian gradient flow).

## 결과

### 1. encoder: 고정 vs 재학습 vs decoder fine-tune (병렬)

| arm | CRPS 1일 / 5일 / 10일 | RMSE 1일 / 10일 | SSR 1일 | latent 이동 | 고정 PINN 오차 | hyd 잔차 1일 |
| --- | --- | --- | --- | --- | --- | --- |
| **frozen** | **0.786** / 0.782 / 0.796 | 1.175 / 1.350 | 0.19 | 0 | 1.184 | **1.60** |
| retrain | 0.842 / 0.822 / 0.821 | 1.212 / 1.357 | 0.12 | 86% | 1.219 | 2.20 |
| decoder fine-tune | 0.787 / 0.775 / 0.786 | 1.178 / 1.343 | 0.18 | 0 | 1.186 | 1.61 |
| (carry_highpass) frozen | 0.759 / 0.820 / 0.801 | 1.134 / 1.403 | 0.19 | 0 | 1.138 | 1.43 |
| (carry_highpass) retrain | 0.833 / 0.842 / 0.782 | 1.193 / 1.366 | 0.12 | 84% | 1.195 | 2.20 |

**권고: stage 2에서 encoder/decoder는 고정합니다.** 다시 학습하면 latent가 84–86% 움직이고, stage 1에서 그 latent에 맞춰 학습한 고정 PINN이 어긋납니다(PINN 예측 오차 1.18 → 1.22, 정역학 잔차 1.6 → 2.2). CFM 손실(0.24)이 frozen(1.55)보다 훨씬 작아지는 것은 encoder가 잔차를 작게 만드는 방향으로 latent를 바꿨기 때문이고(latent 붕괴 경향), 그 결과 퍼짐이 SSR 0.12로 줄었습니다. decoder만 낮은 lr로 fine-tune해도 이득은 없습니다(0.787). 이 결론은 carry_highpass 설정에서도 같습니다.

### 2. 병렬 vs 직렬, 그리고 operator

| arm | CRPS 1일 / 5일 / 10일 | RMSE 10일 | SSR 1일 / 10일 | 멤버 진폭 1일 / 10일 | hyd 1일 / 10일 |
| --- | --- | --- | --- | --- | --- |
| persistence | – | 1.548 (1일 0.753) | – | 진실 1.21 / 1.10 | 진실 0.00 |
| grid CFM (PR #7) | **0.485** / 0.622 / 0.635 | 1.267 | 1.24 / 1.14 | 1.42 / 1.43 | 68 / 91 |
| grid CFM + DCT 사영 (대조) | 0.501 / 0.632 / 0.640 | 1.229 | 0.65 / 0.81 | 0.97 / 1.07 | 23 / 31 |
| grid CFM + 평활 flow (대조) | 0.464 / 0.617 / 0.631 | 1.249 | 0.95 / 0.96 | 1.18 / 1.23 | 44 / 59 |
| stage 1 PINN만 (잡음 없음) | 0.885 / 0.987 / 1.039 | 1.421 | 0 | 0.83 / 0.90 | 1.5 / 2.7 |
| 병렬 frozen | 0.786 / 0.782 / 0.796 | 1.350 | 0.19 / 0.45 | 0.83 / 0.91 | 1.5 / 2.5 |
| **직렬 (CFM → PINN)** | 0.739 / 0.684 / 0.645 | **1.155** | 0.27 / 0.55 | 0.68 / 0.70 | **1.2 / 2.1** |
| 직렬, 물리 없음 | 0.722 / 0.654 / **0.635** | 1.201 | 0.37 / 0.74 | 0.86 / 0.91 | 1.6 / 2.8 |
| 직렬 + FNO operator | 0.682 / 2.355 / 55.0 | 46.1 | 1.17 / 9.4 | 1.37 / 386 | 1.7 / 201 |
| 직렬 + FNO, 물리 없음 | 0.665 / 1.483 / 8.98 | 6.15 | 1.17 / 9.7 | 1.37 / 53 | 4.7 / 114 |
| 직렬 + FNO + DCT band-limit | 0.659 / 2.311 / 10.7 | 7.73 | 1.16 / 8.6 | 1.37 / 59 | 1.6 / 56 |

- **직렬이 병렬보다 모든 lead에서 낫습니다**(1일 0.739 vs 0.786, 10일 0.645 vs 0.796). 병렬에서는 PINN이 평균만 정하고 퍼짐은 CFM 혼자 만들지만, 직렬에서는 PINN Q가 멤버 하나하나를 보고 보정합니다. 10일 앙상블 평균 RMSE(1.155)는 이 실험의 모든 arm 중 가장 낮고 grid CFM(1.267)보다도 낮습니다.
- **latent 멤버는 물리적으로 균형 잡혀 있습니다.** 정역학 잔차가 grid CFM의 68–91에서 1.2–2.8로 30–70배 작고, 멤버 편차의 경도 방향 lag-1 상관이 0.95(진실 tendency 0.957, grid CFM 0.16)입니다. 6단계의 white noise 과대분산은 latent에서 사라집니다.
- **operator는 1일만 좋습니다.** 1일 CRPS 0.66–0.68, SSR 1.17로 퍼짐을 맞추지만, 1-step으로만 학습해서 Markov rollout에서 발산합니다(10일 멤버 진폭 53–386). band-limit과 물리 항 모두 발산을 막지 못합니다(W6).

### 3. 과대분산 가설 검증 (소유자 가설 두 개)

grid CFM(PR #7 설정, 1440 iter)을 학습 자료 양만 바꿔 학습했습니다. 시험은 같은 2020 쌍, 1일, 멤버 8입니다.

| 학습 자료 | 시험 CRPS / RMSE / spread / SSR | 학습 쌍에서 RMSE / spread / SSR | 편차 lon lag-1 | 고파수(k 9–24) 퍼짐/진실 tendency 파워 |
| --- | --- | --- | --- | --- |
| 1년 (89쌍) | 0.564 / 1.112 / 1.073 / 1.02 | 0.705 / 1.043 / 1.57 | 0.11 | 4488 |
| 4년 (356쌍) | 0.484 / 0.955 / 1.115 / 1.24 | 0.846 / 1.109 / 1.39 | 0.15 | 4472 |
| 12년 | TWELVE_YEARS |

같은 4년 CFM에 PINN 잔차의 정규화 gradient flow를 세기 η만 바꿔 붙였습니다(대조: 같은 flow를 Laplacian 에너지에).

| η | 물리 flow: spread / SSR / hyd / lag-1 | 평활 flow: spread / SSR / hyd / lag-1 |
| --- | --- | --- |
| 0 (CFM) | 1.115 / 1.24 / 76 / 0.15 | 같음 |
| 0.5 | 1.068 / 1.19 / 34 / 0.17 | 0.800 / 0.93 / 48 / 0.54 |
| 2 | 1.076 / 1.21 / 68 / 0.17 | 0.538 / 0.62 / 24 / 0.83 |
| 8 | 1.093 / 1.21 / 154 / 0.16 | 0.829 / 0.93 / 55 / −0.16 |

- **가설 (1) "자료가 작아서"**: 자료를 늘리면 평균(RMSE 1.11 → 0.96, CRPS 0.56 → 0.48)은 좋아지지만 **퍼짐은 그대로**입니다(1.07 → 1.12, 고파수 파워 4488 → 4472). 학습 쌍에서도 같은 크기로 과대분산합니다(SSR 1.39–1.57). 표본 추정 오차라면 학습 쌍에서는 퍼짐이 작아야 하고 자료가 늘수록 줄어야 하는데 둘 다 아닙니다. 그러니까 과대분산의 크기는 자료 양이 아니라 모델 구조가 정하는 잡음 바닥입니다. 1년에서 SSR이 1에 가까운 것은 RMSE가 나빠졌기 때문이지 퍼짐이 맞아서가 아닙니다. 다만 latent autoencoder에서는 자료 부족이 직접 문제가 됩니다(W2).
- **가설 (2) "PINN의 물리적 고정 효과가 작아서"**: 물리 flow를 16배 세게 해도 퍼짐은 줄지 않고(1.07 → 1.09) white noise 구조(lag-1 0.16)도 그대로입니다. 대신 정역학 잔차는 η가 크면 오히려 커집니다(34 → 154, 정규화 gradient step이 넘침). 같은 크기의 평활 flow는 퍼짐을 바로 줄입니다. 그러니까 문제는 PINN의 세기가 아니라 **방향**입니다. 원시방정식 잔차(손실의 99% 이상이 정역학)는 기둥 방향 T–z 균형만 보고, 수평 격자 규모 잡음은 거의 보지 못합니다. 직렬 결과도 같은 이야기입니다. physics를 빼면 오히려 1일 CRPS가 좋아지고(0.722 vs 0.739), physics는 정역학 잔차와 10일 RMSE를 개선하면서 멤버 진폭을 줄입니다.
- 결론: 과대분산을 줄인 것은 **구조적 band-limit**(DCT latent, DCT 사영 대조군)이고, 물리 penalty도 자료 양도 아닙니다.

## 취약점 (분석해야 할 것)

- **W1. autoencoder 복원 바닥이 24 h tendency보다 큽니다.** 시험 rmse_ae 1.12–1.14 = tendency RMS(0.74)의 1.5배(학습 쌍에서는 0.70 = 0.98배). climate_diffusion이 quasi-static이 된 것과 같은 실패입니다(docs/lessons_climate_diffusion.md). 그래서 latent arm은 모두 1일에서 persistence(0.753)와 grid CFM에 집니다. DCT 절단만의 바닥은 0.42(0.57배)인데, 잘리는 것은 대부분 x_k의 소규모 구조이고 tendency는 0.685/0.74가 남습니다. **carry_highpass**(절단 위 모드를 x_k에서 그대로 복사)로 바닥이 1.03(1.39배)으로 줄고 1일 CRPS가 좋아지지만(병렬 0.786 → 0.759, 직렬 0.739 → 0.726), 5–10일은 나빠집니다(직렬 10일 0.645 → 0.686). 복사된 소규모 구조가 진화하지 않기 때문입니다.
- **W2. 작은 자료 + 큰 MLP = 과적합.** 356쌍에 1.6M 파라미터 encoder. 학습/시험 복원 오차 0.70 / 1.14(carry 0.55 / 1.03). 자료가 작다는 소유자 가설 (1)은 CFM 퍼짐이 아니라 여기서 맞습니다. 실제 ERA5(32년 ≈ 11,600쌍)에서 먼저 `[stage1] floors`를 확인해야 합니다.
- **W3. latent 앙상블은 과소분산입니다**(SSR 1일 0.19–0.37, 10일 0.45–0.74). white noise는 사라졌지만 CFM이 latent 잔차의 분산을 다 만들지 못합니다. 6단계의 과대분산(SSR 1.24)을 반대 방향으로 뒤집은 셈입니다.
- **W4. 물리 항은 진폭을 줄입니다.** 직렬에서 physics를 켜면 정역학 잔차 1.6 → 1.2, 10일 RMSE 1.20 → 1.16이지만 멤버 진폭 0.86 → 0.68(진실 1.21), CRPS 1일 0.722 → 0.739. 4단계 W3, 3단계(정지 상태가 잔차 0인 해)와 같은 약점입니다.
- **W5. encoder 재학습은 고정 PINN과 latent의 일관성을 깨고**(latent 84–86% 이동) CFM 손실을 latent 축소로 줄입니다(SSR 0.12).
- **W6. FNO operator는 1-step 학습만으로는 rollout에서 불안정합니다.** 1일 SSR은 1.17로 맞추지만 5일 진폭 8–16, 10일 53–386. pooling한 앙상블 평균 feature와 residual 보정이 매 스텝 누적됩니다. DCT band-limit은 정역학 잔차 폭주를 줄이지만(201 → 56) 진폭 발산은 막지 못합니다. 6단계의 rollout 시퀀스 손실(curriculum)이나 보정 크기 제한이 필요합니다. FNO는 위도 방향도 주기로 보므로 두 극을 이웃으로 다룹니다(domain padding 미적용).
- **W7. 합성 자료가 매끄럽습니다**(진실 tendency의 lon lag-1 0.957). DCT 절단의 이점이 실제 ERA5보다 크게 나왔을 수 있습니다. 실제 자료에서는 k를 키워 바닥을 다시 재야 합니다.

## 다음 단계 제안

1. 실제 ERA5에서 stage 1만 돌려 `rmse_ae / tendency_rms`를 확인합니다. 1보다 크면 latent 경로는 멈추고 k, latent, encoder 크기부터 조정합니다.
2. 직렬 구조에 6단계의 rollout 시퀀스 손실을 붙이고, operator를 그 안에서 학습합니다(W6).
3. 물리 항은 정역학 hard projection으로 바꾸고 penalty는 운동량·열역학에만 둡니다(4단계 제안, W4).
