# 11단계: ClimODE와 그 논문의 비교 모델들 대 PG-CFM (실제 ERA5, ClimODE 평가 프로토콜)

## 요청을 이렇게 읽었습니다

소유자 요청(2026-10-24): ClimODE 논문(Verma, Heinonen, Garg, ICLR 2024, arXiv 2404.10024)과 그 논문의 비교 모델들로 우리 모델을 평가하되, 평가는 https://github.com/dataymeric/ClimODE 를 참고하라. "즉 여러 모델과 저희 모델의 비교".

그래서 (1) ClimODE의 데이터·분할·초기장·점수 정의를 그대로 쓰고, (2) 그 위에서 ClimODE 공개 체크포인트를 원저자 코드로 직접 돌려 같은 초기장에서 짝 비교하고, (3) 논문 Table 6/7의 NODE, ClimaX, FourCastNet(FCN), IFS 수치를 같은 표에 놓았습니다. 우리 모델은 9·10단계의 최선 사슬(고정 DCT latent encoder → latent CFM → 멤버별 PINN Q → IDCT, Markov rollout, 2016년에 맞춘 MBM + spread scale + 정역학 접공간 사영)입니다. **이 프로젝트에서 처음으로 실제 ERA5에서 돌린 결과입니다.**

## 구현 (기존 구현 재사용)

| 부분 | 가져온 것 | 바꾼 것 |
| --- | --- | --- |
| ClimODE 모델·평가 | 원저자 저장소 Aalto-QuML/ClimODE @ e729d23 (MIT) → `third_party/climode` | pandas 호환 한 줄(`"6H"`→`"6h"`). 체크포인트 `ClimODE_global.pt`는 실행 시 내려받음 |
| ClimODE 실행 | upstream `evaluation_global.py`의 루프 | test 속도장 적합(upstream은 `'### Test velocity here'`로 비워 둠), `weights_only=False`, 144 h까지 적분(Table 7처럼). `scripts/run_climode_eval.py` |
| 점수 | upstream `evaluation_rmsd_mm`, `evaluation_acc_mm`, `evaluation_crps_mm` | PG-CFM 쪽은 같은 식을 벡터화(`climode_scores`). 첫 장에서 upstream 함수와 `assert_allclose`로 일치 확인 후 사용 |
| dataymeric/ClimODE | 요청하신 저장소 | `src/utils/evaluation.py`에는 batch RMSE와 zarr 저장만 있고, 공개한 비교 수치(`visualization/data/other_model_*.csv`)는 ClimaX·IFS 값입니다. 그래서 점수는 그 저장소도 따르는 원저자 함수로 계산했습니다 |
| 데이터 | WeatherBench2 공개 zarr `1959-2023_01_10-6h-64x32_equiangular_conservative` | `scripts/wb2_to_kai.py`: KAI 67채널 배치(12 UTC 스냅샷) + ClimODE용 WB1 형식 파일 |
| PG-CFM | 7–10단계 코드 그대로(`LatentPGCFM`, `train_stage1`, `train_serial`, `LinearMeanCalibrator`, `SpreadCalibrator`, `tangent_calibrated`, `PersistenceMBM`) | 극점 없는 격자를 위해 `contract.register_latitudes` 추가(면적 가중·물리 격자). `scripts/run_step11_climode.py` |

## 프로토콜 (ClimODE와 같게 맞춘 것, 다른 것)

| 항목 | ClimODE | 여기서 PG-CFM |
| --- | --- | --- |
| 격자 | 5.625°, 32×64, 극점 없음 | 같음 |
| 기간 | 학습 2006–2015, 검증 2016, 시험 2017–2018 | 같음. 2016은 모든 고전 보정(MBM, spread, u10 진단) 적합에만 사용 |
| 초기장 | 6 h index 2 + 8e (각 해 12 UTC, 이틀마다) | 같은 날짜·시각 |
| 시간 간격 | 연속 ODE, 6 h 출력 | 24 h Markov 전이, 12 UTC 스냅샷 (일평균 아님) |
| 변수 | z500, t850, t2m, u10, v10 | z500, t850, t2m 직접. **u10/v10은 KAI 채널이 아니어서 u/v1000에서 격자점별 1차 회귀(2016 분석장)로 진단** (진단 자체의 RMSE 0.45 / 0.40 m/s) |
| RMSE | IC마다 위도 가중 RMSE → 평균 | 같음 (앙상블 평균) |
| ACC | 각 시험 해의 시간 평균을 기후값으로, 장 평균 제거 후 위도 가중 상관 | 같음 |
| CRPS | Gaussian CRPS, min-max(2006–2018) 정규화 단위, 비가중 격자 평균 | 같음(앙상블 평균·표준편차) + 앙상블 CRPS(properscoring) |
| 정의상 없는 것 | — | KAI 데이터에 skt가 없어 t2m을 복사, tisr은 해석식으로 계산 |

<!-- results below are filled from docs/step11/report.json -->

## 변형 (작은 테스트 → 취약점 → 수정)

| 변형 | 설정 | 왜 |
| --- | --- | --- |
| **A** | 7–10단계 합성 실험의 최선 사슬 그대로 (k 8×16, latent 64, `carry_highpass` 없음) | "현재 최선 모델" 평가 |
| **B** | A + `carry_highpass` (DCT 절단 위 모드를 x_k에서 복사) | `scripts/train_latent.py`의 ERA5 기본 설정이고, 7단계에서 W1 수정으로 제안한 것 |
| **C** | A + `carry_tendency` (새로 추가): x_{k+1} = x_k + D(z_{k+1}) − D(z_k) | B로도 W1이 안 풀려서 넣은 이론 수정. latent 한 걸음이 아무것도 안 하면 정확히 persistence가 되도록, AE 오차가 변화량에만 걸리게 함 |

학습은 모두 같습니다: stage 1 1500 iter → serial(CFM + 멤버별 PINN Q, 물리 gradient cap 0.5) 600 iter, batch 8, CPU. 보정(MBM, s_kc, mbmx, u10 진단)은 2016년 12 UTC 초기장 60개(6일 간격)에 맞췄습니다.

### 첫 작은 테스트: 실제 ERA5에서의 바닥 오차 (2016, 정규화 단위)

| | DCT 절단 | AE 재구성 | AE 변화량 재구성 | 결정론 PINN 1일 / persistence | PINN 변화량 진폭 / 실제 |
| --- | --- | --- | --- | --- | --- |
| 24 h 변화량 RMS = 0.542 | 0.436 (0.81배) | | | | |
| A | | 0.602 (**1.11배**) | | 1.105 | 1.11 |
| B | | 0.404 (0.75배) | | 1.102 | 0.75 |
| C | | 0 (정의상) | 0.527 (**0.97배**) | **0.988** | **0.23** |

합성 데이터에서는 AE 바닥이 학습 데이터에서 0.98배였는데, 실제 ERA5에서는 66채널 × 32×64를 64차원으로 줄이는 것 자체가 하루 변화량을 담지 못합니다. B는 재구성만 좋아지고(고주파를 x_k에서 그대로 베끼므로) 예측은 그대로입니다. C는 처음으로 persistence 아래로 내려가지만, 예측한 변화량의 진폭이 실제의 23%뿐입니다.

## 결과

### 위도 가중 RMSE (z500 m²/s², t850·t2m K), ClimODE 정의

| model | z500 24h | t850 24h | t2m 24h | z500 72h | t850 72h | t2m 72h | z500 144h | t850 144h | t2m 144h |
|---|---|---|---|---|---|---|---|---|---|
| NODE (paper) | 878 | 3.35 | 3.86 |  |  |  |  |  |  |
| ClimaX (paper) | 365 | 2.17 | 2.37 | 687 | 3.17 | 2.87 | 802 | 3.97 | 3.38 |
| FCN (paper) | 333 | 1.83 | 1.68 |  |  |  |  |  |  |
| IFS (paper) | 51 | 0.87 | 1.02 |  |  |  |  |  |  |
| ClimODE (paper) | 193 | 1.55 | 1.40 | 479 | 2.58 | 2.75 | 784 | 3.62 | 3.30 |
| ClimODE (reproduced, 16 ICs) | 235 | 1.37 | 2.76 | 541 | 2.65 | 4.28 | 910 | 4.25 | 5.95 |
| persistence (16 ICs) | 586 | 2.82 | 1.99 | 927 | 4.19 | 3.18 | 1103 | 4.78 | 3.76 |
| PG-CFM raw (360 ICs) | 804 | 3.59 | 3.63 | 837 | 3.74 | 3.80 | 864 | 3.90 | 3.97 |
| PG-CFM noQ (360 ICs) | 793 | 3.54 | 3.55 | 820 | 3.60 | 3.62 | 840 | 3.68 | 3.71 |
| PG-CFM mbm_tan (360 ICs) | 813 | 3.55 | 3.62 | 844 | 3.70 | 3.79 | 868 | 3.86 | 3.96 |
| PG-CFM mbmx_tan (360 ICs) | 542 | 2.49 | 1.68 | 765 | 3.32 | 2.45 | 829 | 3.55 | 2.76 |
| persistence (360 ICs) | 572 | 2.70 | 1.74 | 903 | 3.91 | 2.61 | 1025 | 4.25 | 2.97 |
| climatology (360 ICs) | 1034 | 5.20 | 5.74 | 1034 | 5.20 | 5.75 | 1033 | 5.20 | 5.74 |
| PG-CFM_hp raw (360 ICs) | 814 | 3.69 | 2.46 | 935 | 4.13 | 2.90 | 1003 | 4.32 | 3.20 |
| PG-CFM_hp noQ (360 ICs) | 770 | 3.49 | 2.38 | 884 | 3.94 | 2.79 | 936 | 4.07 | 2.99 |
| PG-CFM_hp mbm_tan (360 ICs) | 769 | 3.54 | 2.45 | 895 | 4.02 | 2.89 | 960 | 4.24 | 3.20 |
| PG-CFM_hp mbmx_tan (360 ICs) | 565 | 2.81 | 1.72 | 829 | 3.75 | 2.51 | 912 | 4.00 | 2.82 |
| PG-CFM_tend raw (360 ICs) | 622 | 3.39 | 1.72 | 1216 | 7.37 | 2.59 | 1935 | 13.14 | 2.93 |
| PG-CFM_tend noQ (360 ICs) | 557 | 2.67 | 1.73 | 893 | 3.91 | 2.60 | 1018 | 4.28 | 2.96 |
| PG-CFM_tend mbm_tan (360 ICs) | 595 | 2.97 | 1.72 | 1118 | 5.38 | 2.58 | 1625 | 7.77 | 2.92 |
| PG-CFM_tend mbmx_tan (360 ICs) | 575 | 2.86 | 1.72 | 934 | 4.69 | 2.58 | 1188 | 6.74 | 2.92 |

- 논문 수치(점선)는 ClimODE 논문 Table 6/7 그대로입니다. ClimODE 논문은 6–36 h에서 IFS를 빼면 가장 좋고, 72/144 h는 ClimaX와만 비교했습니다.
- ClimODE 재현은 CPU 시간 때문에 각 시험 해의 첫 8개 초기장(1월 1–15일, 16개)만 돌렸습니다(144 h 적분이 초기장 쌍당 약 18분). 겨울만이라 오차가 커서, 같은 16개에서의 persistence(z500 24 h 586)를 함께 적었습니다.
- PG-CFM과 persistence/climatology는 360개 전부입니다.

### 같은 초기장 16개에서의 짝 비교: PG-CFM − ClimODE (음수면 PG-CFM이 나음, 95% bootstrap 구간)

| 비교 | z500 24h | t850 24h | t2m 24h | z500 72h | t850 72h | t2m 72h | z500 144h | t850 144h | t2m 144h |
|---|---|---|---|---|---|---|---|---|---|
| ClimODE 재현 RMSE (기준) | 235 | 1.37 | 2.76 | 541 | 2.65 | 4.28 | 910 | 4.25 | 5.95 |
| A raw − ClimODE | 630 [610, 652] | +2.34 | +1.03 | 360 [329, 392] | +1.25 | -0.16 | -4 [-57, 46] | **-0.33** | **-1.81** |
| A mbm_tan − ClimODE | 646 [624, 669] | +2.31 | +1.03 | 377 [343, 412] | +1.25 | -0.14 | 10 [-45, 63] | **-0.33** | **-1.80** |
| A mbmx_tan − ClimODE | 327 [303, 350] | +1.24 | **-0.84** | 267 [236, 298] | +0.90 | **-1.32** | -22 [-62, 16] | **-0.43** | **-2.55** |
| B mbmx_tan − ClimODE | 345 [318, 370] | +1.57 | **-0.80** | 310 [269, 347] | +1.35 | **-1.24** | 52 [3, 97] | +0.14 | **-2.42** |
| C raw − ClimODE | 414 [392, 434] | +2.22 | **-0.79** | 734 [700, 767] | +5.32 | **-1.10** | 1166 [1113, 1214] | +10.06 | **-2.16** |
| C mbmx_tan − ClimODE | 357 [331, 380] | +1.62 | **-0.79** | 430 [390, 470] | +2.33 | **-1.11** | 370 [320, 416] | +2.94 | **-2.19** |

(z500 칸은 평균과 95% 구간, 굵게 = PG-CFM이 유의하게 나음)

### 확률 예측 (ClimODE 정의 Gaussian CRPS, min-max 단위; spread/skill)

| 모델 | z500 24h | z500 72h | z500 144h | t850 24h | t850 72h | t850 144h | spread/RMSE z500 24/72/144h |
|---|---|---|---|---|---|---|---|
| ClimODE 재현 (16 ICs, Gaussian) | 0.0092 | 0.0232 | 0.0451 | 0.0085 | 0.0177 | 0.0346 | 0.83 / 0.32 / 0.09 |
| A mbm_tan (같은 16 ICs) | 0.0372 | 0.0402 | 0.0405 | 0.0237 | 0.0254 | 0.0253 | 0.88 / 0.84 / 0.84 |
| A mbmx_tan (점예측, std≈0) (같은 16 ICs) | 0.0275 | 0.0426 | 0.0482 | 0.0204 | 0.0293 | 0.0320 | — |
| B mbm_tan (같은 16 ICs) | 0.0309 | 0.0352 | 0.0370 | 0.0254 | 0.0281 | 0.0289 | 0.88 / 0.88 / 0.89 |
| C mbm_tan (같은 16 ICs) | 0.0244 | 0.0565 | 0.0967 | 0.0227 | 0.0490 | 0.0725 | 0.91 / 0.76 / 0.63 |

PG-CFM의 spread/RMSE는 360개 전체 기준입니다.

## 취약점 분석 (나예현 논문 가설 기준)

논문의 주장: (i) flow matching의 잡음으로 앙상블(확률적 흐름)을 만들고, (ii) PINN 물리 안내로 그 흐름을 물리적으로 유지해, (iii) Markov rollout p(x_{t+1}|x_t)로 예측력을 높인다.

**V1. latent 병목이 실제 데이터에서 지배적입니다 (W1, climate_diffusion 실패의 재현).** A의 원 사슬은 z500 24 h 805로 persistence(572)보다 나쁘고, ACC가 0.57 → 0.48로 lead에 거의 무관합니다(진폭 0.68 → 0.63). 사슬이 하루 변화를 만들지 못하고 매끈한 준정상 상태를 냅니다. 합성 데이터(7–10단계)는 저차원이라 이 문제가 1.5배 수준으로 가려져 있었습니다. B(고주파 복사)는 단기는 같고 중기는 오히려 나빠집니다(z500 144 h 912 vs 829): 복사된 작은 규모가 그대로 남아 persistence처럼 오차가 자랍니다.

**V2. 지금 PG-CFM의 기술(skill)은 대부분 고전 회귀에서 옵니다.** 최선 arm은 mbmx_tan(a + bμ + c x_0, 2016 적합)입니다. z500의 c(x_0)는 1일 0.75 → 6일 0.29, b(사슬 평균)는 0.25 → 0.74입니다. 짧은 lead는 persistence가, 긴 lead는 사슬의 (기후값 쪽으로 끌린) 평균이 기여합니다. 그래도 합성 데이터와 달리 b가 0이 아니므로, 실제 데이터에서 사슬은 3–6일에 정보를 줍니다(mbmx_tan이 모든 lead에서 persistence보다 나음).

**V3. PINN Q가 실제 데이터에서도 수축시킵니다 (H2 재확인).** A에서 Q를 끈 noQ가 RMSE·ACC가 더 좋습니다(z500 24 h 793 vs 804, ACC 0.585 vs 0.571). 대신 Q는 spread를 3배로 키웁니다(SSR 0.13 → 0.47). 멤버의 정역학 잔차는 정답의 0.19–0.57배로, 실제 대기보다 "더 균형 잡힌"(= 더 매끈한) 장입니다. 물리 잔차가 작다는 것이 물리적으로 맞다는 뜻이 아니라 평활하다는 뜻이 됩니다(3단계의 "정지 상태가 잔차 0인 해" 문제).

**V4. C(변화량 decoding)는 정지 문제를 불안정으로 바꾸고, 불안정의 원인은 PINN Q입니다.** 360개 전체에서 C raw는 24 h z500 622로 A(804)보다 낫지만 rollout 오차가 계속 커져 144 h에서 z500 1935, t850 13.1 K로 기후값(1034, 5.2 K)보다 나빠집니다. 그런데 **Q를 끈 C noQ는 557 / 893 / 1018로 모든 lead에서 persistence(572 / 903 / 1025)와 같거나 조금 낫고 발산하지 않습니다.** 멤버의 정역학 잔차는 정답 대비 0.48배(1일) → 8.6배(6일)로 커집니다(noQ는 1.0 → 1.3배로 유지). 즉 A에서 끌개였던 매끈한 기후 쪽 수축이 없어지자, 단일 step(K = 1)으로만 학습한 Q의 보정이 자기 출력 위에서 쌓여 물리 잔차를 오히려 키웁니다. 물리 안내가 rollout을 안정시킨다는 가설의 정반대이고, 6단계의 sequence(다단계) 손실 없이는 Q를 rollout에 쓰면 안 된다는 뜻입니다.

**V5. 앙상블 보정은 동작합니다.** A mbm_tan의 spread/RMSE는 0.84–0.88로 모든 lead에서 안정적이고, ClimODE의 spread는 lead가 길수록 줄어 144 h에서 오차의 9%(z500 81 vs 910)입니다. 그래서 6일 확률 예측(같은 16개, Gaussian CRPS)은 A mbm_tan이 ClimODE보다 낫습니다(z500 0.0405 vs 0.0451, t850 0.0253 vs 0.0346, 차이의 구간이 모두 0 아래). 하지만 이것은 V2의 고전 보정과 spread scale 덕분이고, 1일 CRPS는 ClimODE가 3–4배 좋습니다(0.0092 vs 0.0372).

**V6. 비교 프로토콜 자체의 취약점 (논문 수치를 그대로 믿으면 안 되는 이유).**
- upstream `evaluation_global.py`는 `model.eval()`을 부르지 않습니다. 그래서 batch norm이 **시험 배치(2017과 2018의 같은 날짜) 통계**를 쓰고 dropout도 켜져 있습니다. 같은 초기장 4개에서 eval 모드로 돌리면 z500 24 h가 200 → 250, 6 h t2m이 0.99 → 2.20으로 나빠집니다. 논문 수치는 시험 데이터의 배치 통계에 기대고 있습니다.
- ACC의 기후값이 "각 시험 해의 시간 평균"이라 계절 변화가 anomaly에 들어갑니다. persistence도 24 h ACC 0.83, t2m 0.94가 나옵니다.
- min-max 정규화의 min/max를 2006–2018(시험 해 포함)에서 계산하고, CRPS는 위도 가중 없이 평균합니다(RMSE와 가중이 다름).
- ClimaX 수치는 출처마다 다릅니다: ClimODE Table 7의 z500 72 h 687 vs ClimaX 논문/dataymeric csv의 244. ClimODE 논문의 ClimaX는 같은 10년 데이터로 다시 학습한 값으로 보입니다.
- u10/v10은 우리 KAI 채널에 없어 u/v1000에서 진단했습니다. 그래서 바람 비교는 참고용입니다.

## 결론과 다음 단계

- 실제 ERA5에서 지금의 PG-CFM은 ClimODE보다 z500·t850 1–3일 예측이 크게 나쁩니다(같은 초기장에서 z500 +327 / +267). 6일에서는 동률(−22, 구간이 0을 포함)이고, t2m은 모든 lead에서 PG-CFM이 낫습니다(ClimODE의 t2m은 persistence보다 나쁨). 확률 예측은 6일에서 PG-CFM(mbm_tan, 보정된 spread)이, 1일에서 ClimODE가 낫습니다. B·C는 A보다 나은 곳이 없습니다(C는 1일 z500만 조금 나음).
- 논문의 "물리 안내로 예측력을 높인다"는 실제 데이터에서 아직 지지되지 않습니다. PINN은 평활·수축 역할을 하고, 기술은 고전 보정에서 옵니다.
- 이론 수정 제안: latent 병목을 없애는 것이 먼저입니다. (1) 물리 공간 grid CFM(2·6단계의 torchcfm U-Net)으로 돌아가 변화량을 목표로 하거나, (2) latent를 쓴다면 변화량 decoding(C) + 다단계 rollout 손실(6단계 sequence loss)로 안정성을 학습해야 합니다. C noQ가 persistence 수준에서 안정적이므로, 다음 작은 테스트는 C + Q의 K = 3–6 sequence 학습이 Q의 발산(V4)을 없애는지 보는 것입니다. PINN 항은 그다음에 "평활 대조군"(4·6단계)과 함께 다시 평가합니다.

파일: `scripts/wb2_to_kai.py`, `scripts/run_climode_eval.py`, `scripts/run_step11_climode.py`, `scripts/report_step11.py`, `third_party/climode/`, `docs/step11/` (report.json, rmse_table.md, rmse_vs_lead.png, 각 run의 results*.json, climode_scores.json).

재현:

    python scripts/wb2_to_kai.py --out data/era5_5p625 --years 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018
    python scripts/run_climode_eval.py --data data/era5_5p625/era5_data --out runs/step11/climode --max-ics 8 --lead-steps 24
    python scripts/run_step11_climode.py --data data/era5_5p625 --out runs/step11                      # A
    python scripts/run_step11_climode.py --data data/era5_5p625 --out runs/step11 --carry-highpass --tag _hp   # B
    python scripts/run_step11_climode.py --data data/era5_5p625 --out runs/step11 --carry-tendency --tag _tend # C
    python scripts/report_step11.py --run runs/step11 --data data/era5_5p625 --out docs/step11 --tags "" _hp _tend
