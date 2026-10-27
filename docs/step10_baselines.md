# 10단계: 소유자가 준 참고 모델과의 비교 (U-Cast, Scher & Messori, dduq)

## 요청을 이렇게 읽었습니다

소유자가 U-Cast 논문(Rühling Cachay et al., ICML 2026)과 두 저장소(sipposip/ensemble-neural-network-weather-forecasts, cbuelt/dduq)를 주고, "이것을 base로 우리 모델에 맞춰 비교 모델을 받고 비교 실험"하라고 했습니다. 이어서 "구현보다는 skill(search-first)에 맞춰 최대한 참조"하라고 했습니다. 그래서 원본 코드를 가능한 한 그대로 쓰고, 우리 쪽에는 KAI 규약에 연결하는 코드만 두었습니다.

## search-first 점검 (원본을 어떻게 가져왔는가)

| 출처 | 라이선스 | 채택 방식 | 가져온 것 |
| --- | --- | --- | --- |
| Rose-STL-Lab/u-cast (commit f6ca2ae) | Apache-2.0 (U-Net 파일은 NVIDIA CC BY-NC-SA) | **원본 파일 그대로 vendoring** (`third_party/ucast`, 수정 없음) | `DhariwalUNet`, `BaseModel.ensemble_forward`, inference dropout, weighted MAE / fair CRPS loss, Muon, `HybridOptim`, `LinearWarmupCosineAnnealingLR`, 원형 padding 함수 |
| sipposip/ensemble-neural-network-weather-forecasts | GPL-3 | 코드를 복사하지 않고 **알고리즘만 이식** (`pgcfm/baselines.py`) | 무작위 섭동(멤버 간 재표준화), 특이벡터 섭동(절단 Gaussian 계수, ± 쌍), dropout 비율 조정, multitrain(독립 학습 망) |
| cbuelt/dduq | 명시 없음 | 함수 단위 이식 | random field 섭동(`random_field_perturbation`, `tke`, 상수), DRN(`build_emb_model`, `crps_cost_function`, 하이퍼파라미터) |

- 원본에서 바꾼 것: U-Cast의 전역 `torch.nn.functional.conv2d` 몽키패치는 우리 모델의 모든 `nn.Conv2d`를 깨뜨리므로, 같은 함수를 U-Net forward 안에서만 적용합니다. 입력 상태는 우리 Markov 규약대로 1개(U-Cast는 2개)이고, EMA(0.9999)는 짧은 학습이라 쓰지 않았습니다. DRN은 변수·lead마다 모델 하나(650개) 대신 lead마다 다출력 모델 하나이고, 10 epoch 안에 예보에서 출발하도록 평균을 예보 + 보정으로 둡니다.
- Scher & Messori의 특이벡터는 원본처럼 `scipy.sparse.linalg.svds`로 전체 Jacobian을 만드는 대신, Jacobian-벡터 곱의 부분공간 반복으로 같은 선두 벡터를 구합니다(79,200차원). U-Cast의 `AttentionOp`에 functorch 규칙이 없어 `torch.autograd.functional`을 씁니다.
- 쓰지 않은 것: EasyUQ(격자점마다 학습 표본이 16개뿐이라 의미가 없음), IFS 섭동(자료 없음), Perlin 잡음(Gaussian과 같은 계열).

## 실험 설정 (9단계와 같음)

- 학습: 2001–2004년 합성 자료(356쌍), PG-CFM 직렬 사슬과 같은 기간. 검증 2006년(초기장 16개): 섭동 크기, dropout 비율, MBM 보정을 모두 여기서 empirical CRPS로 고릅니다(Scher & Messori도 properscoring의 empirical CRPS로 고름). 시험: 2021년 test16, 2020년 test. 멤버 8개, 10일 rollout.
- 결정론 backbone = U-Cast stage 1: `DhariwalUNet`(채널 32, [1,2,2], block 2, bottleneck attention, dropout 0.1, 1.56M 파라미터), 잔차 정규화 출력, 면적 가중 MAE, Muon(3e-3) + AdamW(3e-4), warmup + cosine, 1,500 iteration(batch 8, 약 34 epoch, 15분). 비교 사슬 PG-CFM은 4.9M 파라미터(그중 4.5M이 DCT autoencoder).
- U-Cast stage 2: fair CRPS, dropout 멤버 M = 2, Muon 7e-3 + AdamW 7e-5, stage 1의 8%(upstream 8/100 epoch) = 120 iteration. 민감도로 4배(480 iteration)도 돌렸습니다. stage 3 DeepEns = stage 2 4개(seed) × dropout 멤버 2개.
- 모든 앙상블 arm에 9단계와 같은 MBM + 정역학 접공간 사영을 2006년에서 맞춰 적용합니다. PG-CFM 행은 `runs/step9/results.json`(같은 시험 집합)에서 읽습니다.

```
python scripts/run_step10_baselines.py --out runs/step10            # 약 3시간 (CPU 4개), multitrain이 2시간
```

## 결과 (test16, 10일 평균 empirical CRPS)

| arm | 조정값 | 평균 | 1 / 5 / 10일 | RMSE | SSR 1/10일 | 변수 간 상관 불일치 | 정역학 1/10일 (K) | 2020 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PG-CFM 직렬 사슬 | - | 0.721 | 0.742 / 0.723 / 0.687 | 1.185 | 0.29 / 0.54 | 0.067 | 1.11 / 2.10 | 0.722 |
| PG-CFM + in-loop | - | 0.649 | 0.693 / 0.638 / 0.630 | 1.140 | 0.51 / 1.12 | 0.071 | 1.43 / 3.55 | 0.658 |
| PG-CFM + MBM + 사영 (best) | - | 0.620 | 0.613 / 0.616 / 0.610 | 1.104 | 0.94 / 0.93 | 0.069 | 1.24 / 2.15 | 0.632 |
| FM 단독 + MBM + 사영 | - | 0.629 | 0.606 / 0.626 / 0.628 | 1.105 | 0.93 / 0.97 | 0.071 | 1.58 / 2.81 | 0.639 |
| det/raw | - | 0.877 | 0.575 / 0.918 / 0.974 | 1.206 | - | - | 0.46 / 4.36 | 0.884 |
| ic_gauss/raw | 1.0 | 0.661 | 0.472 / 0.689 / 0.710 | 1.166 | 1.42 / 0.95 | 0.119 | 69.28 / 140.50 | 0.663 |
| ic_gauss/mos_ck_tan | 1.0 | 0.589 | 0.443 / 0.607 / 0.612 | 1.047 | 0.93 / 0.91 | 0.121 | 71.05 / 143.49 | 0.595 |
| ic_svd/raw | 30.0 | 0.629 | 0.417 / 0.658 / 0.695 | 1.097 | 1.01 / 0.81 | 0.098 | 23.92 / 48.50 | 0.633 |
| ic_svd/mos_ck_tan | 30.0 | 0.575 | 0.415 / 0.599 / 0.605 | 1.034 | 0.91 / 0.97 | 0.098 | 24.75 / 49.98 | 0.582 |
| ic_rfp/raw | 1.0 | 0.640 | 0.454 / 0.669 / 0.701 | 1.137 | 1.28 / 0.64 | 0.067 | 0.51 / 5.29 | 0.648 |
| ic_rfp/mos_ck_tan | 1.0 | 0.575 | 0.439 / 0.595 / 0.599 | 1.045 | 0.93 / 0.97 | 0.073 | 0.52 / 5.32 | 0.587 |
| mc_dropout/raw | 0.7 | 0.675 | 0.461 / 0.712 / 0.729 | 1.092 | 0.35 / 0.43 | 0.099 | 0.48 / 4.05 | 0.686 |
| mc_dropout/mos_ck_tan | 0.7 | 0.572 | 0.414 / 0.595 / 0.603 | 1.033 | 0.89 / 0.92 | 0.100 | 0.48 / 4.19 | 0.581 |
| multitrain/raw | - | 0.643 | 0.468 / 0.671 / 0.676 | 1.092 | 0.33 / 0.58 | 0.080 | 0.44 / 4.34 | 0.650 |
| multitrain/mos_ck_tan | - | 0.571 | 0.416 / 0.591 / 0.599 | 1.040 | 0.93 / 0.95 | 0.082 | 0.44 / 4.50 | 0.580 |
| ucast/raw | - | 0.770 | 0.536 / 0.817 / 0.812 | 1.182 | 0.17 / 0.35 | 0.104 | 0.48 / 5.49 | 0.770 |
| ucast/mos_ck_tan | - | 0.591 | 0.456 / 0.610 / 0.605 | 1.051 | 0.82 / 1.04 | 0.110 | 0.49 / 5.60 | 0.601 |
| ucast_de/raw | - | 0.690 | 0.496 / 0.725 / 0.732 | 1.138 | 0.28 / 0.53 | 0.095 | 0.48 / 5.38 | 0.692 |
| ucast_de/mos_ck_tan | - | 0.583 | 0.437 / 0.604 / 0.607 | 1.047 | 0.88 / 0.96 | 0.098 | 0.48 / 5.53 | 0.594 |
| ucast_long/raw | - | 0.660 | 0.467 / 0.691 / 0.702 | 1.155 | 0.52 / 0.72 | 0.309 | 2.02 / 8.79 | 0.672 |
| ucast_long/mos_ck_tan | - | 0.579 | 0.444 / 0.597 / 0.600 | 1.047 | 0.87 / 0.97 | 0.322 | 2.00 / 8.82 | 0.589 |
| ucast_de_long/raw | - | 0.641 | 0.456 / 0.668 / 0.686 | 1.139 | 0.56 / 0.81 | 0.267 | 3.01 / 12.41 | 0.651 |
| ucast_de_long/mos_ck_tan | - | 0.576 | 0.437 / 0.594 / 0.600 | 1.046 | 0.88 / 0.97 | 0.280 | 2.98 / 12.20 | 0.586 |
| drn/raw | - | 0.588 | 0.424 / 0.600 / 0.630 | 1.105 | 0.90 / 0.97 | 0.877 | 8.22 / 22.65 | 0.592 |

(`/raw` = 보정 없음, `/mos_ck_tan` = MBM + 접공간 사영. 정역학 잔차의 정답 값은 0입니다. det은 단일 예보라 CRPS = MAE입니다. DRN 멤버는 Gaussian의 분위수라 주변분포만 의미가 있고, 상관·정역학 지표는 의미가 없습니다.)

추가 구조 지표 (MBM + 사영, 10일 평균):

| arm | 물리 쌍 상관 불일치 | 공간 lag-1 상관 불일치 | variogram | energy score |
| --- | --- | --- | --- | --- |
| PG-CFM | **0.057** | 0.056 | 0.069 | 0.749 |
| ic_rfp | 0.071 | **0.020** | **0.049** | 0.700 |
| multitrain | 0.071 | 0.035 | 0.052 | **0.695** |
| mc_dropout | 0.077 | 0.098 | 0.070 | 0.694 |
| ucast_de | 0.078 | 0.068 | 0.059 | 0.710 |

## 해석

1. **점수(CRPS·RMSE)는 격자 U-Net 비교 모델이 모두 앞섭니다.** 같은 MBM 보정 뒤 0.571–0.591 대 PG-CFM 0.620입니다. 차이는 거의 1일에 몰려 있습니다(0.41–0.46 대 0.613). 7단계 W1의 AE floor(복원 오차가 하루 변화량의 1.5배)가 그대로 드러난 것입니다. 5일은 0.594–0.610 대 0.616, 10일은 0.599–0.612 대 0.610으로 중기에서는 비슷합니다. RMSE도 1.03–1.05 대 1.10입니다.
2. **물리 정합성은 PG-CFM이 가장 좋습니다.** 10일 정역학 잔차는 PG-CFM 2.15 K입니다. 역학적으로 일관된 앙상블(dropout, multitrain, random field, U-Cast)은 4.2–5.6 K, U-Cast stage 2를 길게 하면 8.8–12.2 K, DRN 22.6 K, 특이벡터 50 K, Gaussian 섭동 143 K입니다. 격자 U-Net은 1일(0.44–0.52 K)에는 PG-CFM(1.24 K)보다 낫지만 rollout이 길어질수록 잔차가 쌓입니다(PG-CFM 1.1 → 2.1, U-Net 0.46 → 4.4).
3. **변수 간 의존 구조도 PG-CFM이 가장 좋습니다**(물리 쌍 0.057, 다음이 0.071). 반대로 공간 구조(lag-1 상관, variogram)는 random field·multitrain이 낫습니다. DCT 절단으로 세부 구조가 매끄러워진 대가입니다.
4. **CRPS로 조정하면 물리를 깨는 섭동이 뽑힙니다.** Gaussian 섭동은 기후 표준편차 크기(1.0)가 선택되고 정역학 잔차가 143 K인데도 CRPS는 0.589로 PG-CFM보다 좋습니다. 특이벡터도 50 K입니다. 5단계에서 본 "CRPS는 물리 불균형에 둔감하다"가 실제 비교 모델에서 재현됐습니다.
5. **U-Cast 고유의 결과.** 논문 말대로 dropout 앙상블은 과소분산입니다(raw SSR 0.17–0.35, DeepEns 0.28–0.53). stage 2를 4배 길게 하면 분산은 커지지만(0.52–0.81) 변수 간 상관 불일치가 0.10 → 0.28–0.31로, 정역학 잔차가 5.5 → 8.8–12.2 K로 나빠집니다. 격자점별 CRPS 학습이 변수 간 구조를 보지 않고 각 채널에 독립적인 퍼짐을 배운 것입니다. MBM 뒤 CRPS는 0.576–0.579로 거의 같습니다.
6. **비교 모델끼리는 차이가 작습니다**(MBM 뒤 0.571–0.591). 순서는 multitrain ≈ mc_dropout ≈ 특이벡터 ≈ random field < DeepEns < Gaussian ≈ U-Cast 단일 ≈ DRN이고, 2020 test도 같습니다. 보정 전에는 특이벡터(0.629)와 random field(0.640)가 가장 좋습니다.

## 가설에 주는 의미

- PG-CFM의 주장은 "CRPS가 더 좋다"가 될 수 없습니다. 증거가 지지하는 주장은 **"중기(5–10일)에서 비슷한 점수로, 물리적으로 정합하고 변수 간 의존 구조가 맞는 앙상블"**입니다. 점수만 보는 비교 모델은 CRPS가 물리 불균형을 거의 벌하지 않아서 좋은 점수를 받습니다(해석 4).
- 단기 점수를 따라잡으려면 AE floor를 먼저 고쳐야 합니다. 후보: DCT 모드 수 증가, carry_highpass, 또는 격자 공간 생성기(U-Cast식 U-Net backbone) 위에 PINN·사영을 두는 구조.
- 공정한 다음 비교는 같은 backbone입니다: U-Cast U-Net으로 flow matching을 하고 PINN 보정 + MBM 사영을 붙이면, "물리 보정의 가치"를 backbone 차이와 분리해서 잴 수 있습니다.

## 취약점

- W1. 합성 자료, 초기장 16개: 0.005 정도의 차이는 잡음입니다. 1일의 0.15–0.2 차이와 정역학 차이는 그보다 훨씬 큽니다.
- W2. 비교 모델 학습이 짧습니다(stage 1 34 epoch, upstream 100 epoch; batch 8 대 48). U-Cast stage 2는 4배 길게 해도 MBM 뒤 점수가 거의 같았습니다.
- W3. dropout 비율 0.7이 탐색 범위의 끝입니다(0.4 → 0.7로 넓혀도 끝). 더 키워도 MBM 뒤 점수는 0.572에서 거의 변하지 않을 것으로 봅니다.
- W4. Scher & Messori 코드는 GPL-3이라 알고리즘을 다시 썼습니다. 원본의 2.5° z500 단일 변수 설정과 달리 66채널에 적용했으므로, 섭동 크기 격자는 우리 자료에 맞게 다시 잡았습니다(특이벡터 1–300, 선택 30).
- W5. DRN은 원본(변수·lead별 단일 출력)과 달리 lead별 다출력이고, 평균을 예보 + 보정으로 둡니다.

## ERA5에서 할 것

KAI 격자(1.5°, 121×240, 극 포함)는 WeatherBench 2의 1.5° 격자와 같습니다. U-Cast는 이 해상도의 학습된 checkpoint를 공개합니다(`hf:salv47/u-cast/ucast.ckpt`, 원본 `run_inference_standalone.py`). ERA5 비교는 이 공개 예보를 그대로 쓰고, 우리 평가 도구(`scripts/evaluate.py`)로 CRPS와 함께 정역학·변수 간 구조 지표를 같이 보고하는 것이 가장 공정합니다. 단, U-Cast는 12시간 간격 6개 변수 13층이고 KAI는 일평균 5개 변수 12층이므로, 공통 변수·층과 일평균 lead로 맞춰야 합니다.
