# 12단계: latent를 뺀 격자 결정론 코어 (ClimODE 데이터·프로토콜)

## 요청을 이렇게 읽었습니다

11단계 비교(`docs/step11_climode.md`) 뒤 소유자가 수정 가설(R1 격자 경향, R4 x_{k−1} 조건, R2 물리 구조 코어, R5 FM은 앙상블 생성기, R3/R6 K=3–6 rollout 손실)을 승인하고 세 가지를 더했습니다(2026-10-25).

- ClimODE처럼 여러 시점(IC−12 h, −6 h, IC)을 입력으로 넣는다.
- 한 번에 생성하지 않고 1 h 간격으로 생성한다.
- 여러 시간으로 생성되는 시퀀스 전체에 대해 보정한다.

또 "우리가 지금 가진 데이터로 처음부터 학습해 봐요"라고 해서, ClimODE 공개 체크포인트(원저자 사전학습) 대신 같은 데이터에서 같은 예산으로 처음부터 학습하는 비교를 만들었습니다. 원래 크기의 ClimODE 학습 명령은 `docs/RUN_ON_GPU.md` 11절에 있습니다(이 CPU 4코어에서는 불가).

이 단계는 같은 날 소유자가 "ClimODE를 따라가는 것이 아니라" 시퀀스 DCT 설계로 방향을 바꾸면서(`docs/step13_sequence.md`) jump24, jump6 두 모드까지만 끝났습니다. ode와 transport는 약 150 iteration에서 멈췄고, `runs/step12/core_ode_train.pt`에서 이어서 돌릴 수 있습니다.

## 구현

| 부분 | 가져온 것 | 바꾼 것 |
| --- | --- | --- |
| 망 | ClimODE `Climate_ResNet_2D` (`third_party/climode/model_function.py`, 위도 reflect·경도 circular padding) | hidden 48, 블록 (2, 2), 0으로 초기화한 1×1 head (시작 = persistence). `pgcfm/core.py` |
| 수송 모드 | ClimODE `pde`: 변수마다 속도장, du/dt = −(v·∇u + u∇·v), dv/dt = 망 | v(0)를 IC마다 최적화(Optim_velocity)하지 않고 세 시점에서 학습된 망으로 냄 |
| 데이터·점수 | 11단계의 WB1 형식 파일과 `climode_scores` (원저자 함수와 일치 확인됨) | 변수별 z-score, 학습 2006–2015 / 검증 2016 / 시험 2017–2018, ClimODE IC 360개 |
| 학습 | | 무작위 IC, 위도 가중 MSE rollout 손실, Adam + cosine, 1500 iteration × batch 8, 100 iteration마다 체크포인트. `scripts/run_step12_core.py` |

모드는 하나씩 바꿉니다.

- **jump24**: 한 장, 24 h 한 번 점프, 24 h 목표만으로 학습 (latent만 뺀 지금의 PG-CFM 전이)
- **jump6**: 세 장(IC−12 h, −6 h, IC), 6 h 점프를 이어 붙이고 6 h 출력 4개(24 h)로 학습
- **ode**: 같은 입력, 1 h Euler 적분 (ClimODE의 적분기)
- **transport**: ClimODE의 수송 구조

## 결과 (시험 2017–2018, IC 360개, CPU 학습 2분 / 8분)

z500 RMSE [m²/s²]

| 선행 | 24 h | 48 h | 72 h | 96 h | 120 h | 144 h |
| --- | --- | --- | --- | --- | --- | --- |
| jump24 (한 장, 24 h) | 352 | 546 | 681 | 776 | 840 | 885 |
| jump6 (세 장, 6 h, 24 h 구간 학습) | **307** | 549 | 736 | 876 | 984 | 1071 |
| persistence | 572 | 801 | 903 | 961 | 1000 | 1025 |
| 이전 PG-CFM raw (11단계) | 804 | | 837 | | | 864 |
| 이전 PG-CFM 최선 mbmx_tan (11단계) | 542 | | 765 | | | 829 |
| ClimODE 논문 (Table 7) | 193 | | 479 | | | 784 |
| ClimaX (ClimODE 논문) | 365 | | 687 | | | 802 |

t850 [K]: jump24 1.74 / 2.99 / 4.08, jump6 1.66 / 3.08 / 4.34 (24 / 72 / 144 h), persistence 2.70 / 3.91 / 4.25.

## 취약점 기준으로 본 것

1. **L1(latent 바닥)이 11단계 격차의 주원인이었습니다.** 같은 한 장·24 h 전이에서 latent만 빼자 24 h z500이 804에서 352로 절반이 됐습니다. CPU 2분 학습으로 ClimaX(365)와 비슷합니다.
2. **세 시점 입력(R4)은 단기를 줄이지만, 24 h 구간 학습은 길게 버티지 못합니다.** jump6는 24 h를 307로 낮췄지만 96 h부터 jump24보다 나빠지고, 144 h(1071)는 persistence(1025)보다 나쁩니다. 학습한 구간 밖으로 이어 붙이면 오차가 쌓입니다(R3의 근거). 13단계가 144 h 시퀀스를 한 번에 생성하는 이유입니다.
3. ClimODE와의 남은 격차(24 h 193 대 307)는 학습량 차이(GPU 300 epoch 대 CPU 8분)와 섞여 있어 이 결과만으로는 구조 차이라고 말할 수 없습니다.
