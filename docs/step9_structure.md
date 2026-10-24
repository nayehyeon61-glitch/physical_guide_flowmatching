# 9단계: classical 분산 보정을 먼저, 그 위에서 operator의 역할 다시 검증

## 요청을 이렇게 읽었습니다

소유자의 classical1~4 메모(2026-10-24)대로 순서를 바꿨습니다. 먼저 학습 없는 고전적 분산 보정(채널별 s_c, 평균 유지)을 제대로 맞춥니다. 그다음 neural operator(NO)의 가설을 "분산 크기를 맞춘다"에서 "분산 보정으로는 바꿀 수 없는 공간·변수 간 오차 구조를 고친다"(classical3)로 바꾸고, 소유자가 제시한 네 비교(기본 / 기본+보정 / operator / operator+보정, 보정은 경로마다 같은 검증 자료와 같은 절차로 따로 맞춤)로 검증합니다. 고정 모델은 7~8단계와 같은 직렬 사슬(`runs/step7/serial.pt`)입니다.

검증할 가설 (classical3):

> Neural operator는 PINN 보정 이후의 기후장에 남은 공간적·변수 간 오차 구조를 수정하며, 그 효과는 단순한 ensemble spread 보정만으로 얻는 개선을 넘어선다.

## 구현 (기존 구현 재사용, search-first)

| 부분 | 가져온 것 | 바꾼 것 |
| --- | --- | --- |
| `rescale_ensemble_spread` | classical1 연습 문제 그대로(모집단 표준편차, [1,1,C,1,1] scale) | 없음. classical1의 검사(평균 보존, s=1 항등, s=0 수축, member 순서, gradient)를 그대로 테스트로 둠 |
| `SpreadCalibrator` | classical4의 `fit_validation(candidates)` | 후보 0.5~6.0. CRPS가 채널·lead별로 분리되고 μ+s·a의 pair 항이 s배라서 rollout 한 번으로 정확히 탐색. 채널별(s_c) / lead·채널별(s_kc) |
| CRPS 선택 기준 | classical4: empirical CRPS | 아래 "선택 기준" 참고. scoringrules의 `nrg` / `fair` 추정량과 소수 10자리까지 일치(테스트) |
| `LinearMeanCalibrator` + s_kc (`mos_ck`) | member-by-member 후처리(MBM, Van Schaeybroeck & Vannitsem 2015; 구현은 Pythie) | 평균 부분은 lead·채널별 최소제곱 a + b·μ, 분산 부분은 위 s_kc를 다시 맞춤. 멤버 간 의존 구조는 그대로 |
| `tangent_calibrated` | 8단계 `HydrostaticTangent` (P_T) | 보정이 바꾼 양 Δ를 lead마다 앙상블 평균의 정역학 접공간에 사영: x + P_T Δ (classical1 마지막 문단의 "보정 후 물리 사영") |
| `LoopInflated` | 8단계 `inflate` | 아무 transition 위에나 in-loop 곱셈 inflation(보정된 멤버를 되먹임) |
| `StructureOperator` | classical4 `EnsembleOperator` 구조, neuraloperator `FNO` / `SFNO`(torch-harmonics 구면 조화 변환) | x̃ = x + gate·N([x, μ, log σ, 조건]); 공유 네트워크라 멤버 순서에 등변. `use_stats=False`면 멤버별 operator(classical3의 구분) |
| `StructureCorrected` | 7단계 `OperatorCorrected`(DCT band 제한) | `preserve_spread`: 보정된 편차를 채널마다 입력의 면적 평균 분산으로 되돌림. operator는 평균, 지역 간 분산 배분, 편차 패턴, 변수 간 상관은 바꿀 수 있지만 채널 전체 진폭은 못 바꿈(진폭은 calibrator 몫) |
| 학습 | 7단계 `train_operator`(fair energy score, curriculum, 상한 있는 물리 항, 보정 크기 벌점) | + `lambda_grad`: 공간 차분장의 energy score(공간 구조 항). curriculum 1→5일, 300 iter, 2005년 |
| `structure_scores` | scoringrules의 energy / variogram score 정의 | lead별: energy score, 공간 차분의 CRPS·평균 RMSE, 이웃 격자 variogram score(p=0.5), 멤버 편차와 실제 오차의 변수 간 상관 불일치(전체 / 물리적으로 묶인 쌍: z–t, u–v, z–u, z–v 같은 층, 인접 층), lag-1 공간 상관 불일치, DCT band별 멤버/정답 power |

search-first 결과(PyPI, GitHub, 웹; GitHub API는 이 세션에서 막혀 있어 웹 페이지로 확인):

- 채택: **scoringrules**(CRPS·energy score 검증), **torch-harmonics**(neuraloperator `SFNO`의 구면 변환).
- 형태만 채택: **MBM**(Pythie, EUPPBench `ESSD-mbm`). 우리 `mos_ck`가 MBM의 제한된 형태라 따로 설치하지 않음.
- 참고: **Roberts (2026, arXiv:2602.15830)**: 멤버를 서로 엮는 보정(MBM, PoET 등)에서는 fair CRPS가 불공정해져 과대분산을 좋게 평가함. 아래 선택 기준 결과와 같은 이야기.
- 다음 후보: **PoET**(ecmwf-lab/poet, MIT): 앙상블 축 attention으로 멤버를 고치는 후처리. IMPROVER(EMOS, ECC)는 iris 기반이라 이 형상에는 무겁고, 우리 보정은 멤버별 affine이라 ECC 순위 재배치가 이미 암묵적으로 유지됨.

파일: `pgcfm/structure.py`, `scripts/run_step9_structure.py`(원시 결과 `docs/step9/results.json`), `scripts/fit_spread.py`(ERA5: classical / classical_lead / classical_tan / mbm / mbm_tan 추가, `--criterion`), `scripts/infer_pgcfm.py --spread`, `tests/test_structure.py`.

## 실험 설정

- 고정 모델: 7단계 직렬 사슬, 7단계 b_j. 멤버 8, Heun 8 step, Markov rollout 10일.
- 기간 셋(직렬 사슬은 모두 본 적 없음): operator 학습 2005년(8단계 보정 연도), **모든 보정은 새 검증 연도 2006년**(10일 rollout 16개)에서만 맞춤, 점수는 2021년 test16(초기장 16개)과 2020년 test(4개).
- operator 7종:

| 이름 | 입력 | spread | band 제한 | 물리 항 | 네트워크 |
| --- | --- | --- | --- | --- | --- |
| op | 멤버 + μ + log σ + 조건 | 자유 | 있음 | 있음 | FNO |
| op_sp | 같음 | 보존 | 있음 | 있음 | FNO |
| op_sp_member | 멤버 + 조건 (앙상블 통계 없음) | 보존 | 있음 | 있음 | FNO |
| op_sp_nophys | op_sp와 같음 | 보존 | 있음 | 없음 | FNO |
| op_sp_full | op_sp와 같음 | 보존 | 없음 | 있음 | FNO |
| op_sfno / op_sp_sfno | op / op_sp와 같음 | 자유 / 보존 | 있음 | 있음 | SFNO (구면) |
| loop_op / loop_op_sp | op / op_sp와 같음, 단 in-loop 보정된 사슬 위에서 학습·실행 | 자유 / 보존 | 있음 | 있음 | FNO |

- 보정: raw(없음), cal_c(채널별 s_c), cal_ck(lead·채널별), cal_c_fair(cal_c를 fair CRPS로 선택), cal_c_tan(cal_c의 변화를 접공간에 사영), loop(in-loop 스칼라 α), mos_ck(MBM형: 선형 평균 보정 + s_kc), mos_ck_tan.

## 선택 기준: empirical CRPS가 맞습니다

멤버와 정답이 같은 분포에서 나온 완벽한 앙상블에 μ + s(x − μ)를 적용하면, empirical CRPS(M² 쌍, 대각 포함)의 기댓값은 **어떤 M에서도 s = 1에서 최소**입니다. 모집단 편차가 (M − 1)/M만큼 줄어드는 것과 M² 쌍 항이 줄어드는 비율이 같기 때문입니다. fair CRPS(M(M − 1) 쌍)는 s > 1(M = 8에서 약 1.3)에서 최소라서, 재보정된(서로 독립이 아닌) 멤버에 쓰면 과대분산을 고릅니다. 합성 검사(`test_spread_calibrator_recovers_scale`)에서 empirical은 정답 배율(2, 1)을 정확히 고르고 fair는 더 크게 고릅니다.

실험에서도 같습니다(test16, base): cal_c_fair는 SSR이 10일에 1.50까지 넘치고 empirical CRPS가 0.687로 cal_c(0.676)보다 나쁩니다. 그래서 9단계의 주 점수는 **empirical CRPS**이고, fair CRPS는 7~8단계와 비교하려고 같이 적습니다. 8단계의 α 선택(fair 기준)도 같은 이유로 조금 과대분산이었습니다(inflate SSR 1.3). ERA5용 `fit_spread.py`는 이제 기본이 `--criterion empirical`입니다.

## 결과 (test16, 2021년 초기장 16개)

### 1. classical 사다리 (base = 직렬 사슬)

| 보정 | empirical CRPS 1/5/10일 | 평균 | fair 평균 | 평균 RMSE | SSR 1/10일 | 정역학 잔차 1/10일 (K) |
| --- | --- | --- | --- | --- | --- | --- |
| 없음 | 0.742 / 0.723 / 0.687 | 0.721 | 0.689 | 1.185 | 0.29 / 0.54 | 1.11 / 2.10 |
| s_c (채널별, post-hoc) | 0.676 / 0.677 / 0.661 | 0.676 | 0.608 | 1.185 | 0.59 / 1.12 | 2.97 / **10.15** |
| s_kc (lead·채널별) | 0.652 / 0.677 / 0.658 | 0.672 | 0.604 | 1.185 | 0.98 / 0.93 | **9.82 / 10.78** |
| s_c, fair 기준 | 0.658 / 0.689 / 0.690 | 0.687 | 0.597 | 1.185 | 0.80 / 1.50 | 4.31 / 14.75 |
| s_c + 접공간 사영 | 0.676 / 0.677 / 0.661 | 0.676 | 0.608 | 1.185 | 0.59 / 1.12 | **1.15 / 2.32** |
| in-loop α = 1.75 | 0.693 / 0.638 / 0.630 | 0.649 | 0.579 | 1.140 | 0.51 / 1.12 | 1.43 / 3.55 |
| MBM형 (평균 a+bμ, s_kc) | 0.613 / 0.616 / 0.610 | **0.620** | 0.557 | **1.105** | 0.94 / 0.94 | 6.83 / 8.73 |
| MBM형 + 접공간 사영 | 0.613 / 0.616 / 0.610 | 0.620 | 0.557 | 1.104 | 0.94 / 0.93 | **1.24 / 2.15** |

정답 자체의 정역학 잔차는 0.00 K입니다(합성 자료가 정역학 균형).

- 채널별 분산 보정만으로 0.721 → 0.676. lead별로 하면 1일이 좋아집니다(0.652).
- **채널별 배율은 정역학 균형을 깹니다**(2.1 → 10 K). R_hyd가 (Φ, T)에 선형이어도 Φ와 T를 다른 배율로 늘리면 J·S·a ≠ 0입니다(classcal2의 경고가 그대로 나옴). 보정이 바꾼 양을 접공간에 사영하면 CRPS는 소수 셋째 자리까지 같고 잔차는 원래 수준(2.3 K)으로 돌아갑니다. 공짜입니다.
- in-loop 스칼라는 post-hoc보다 좋고(0.649) 평균 RMSE도 줄입니다(1.185 → 1.140): 퍼진 멤버를 되먹이면 비선형 사슬을 지나며 평균 궤적 자체가 나아집니다.
- MBM형(평균의 선형 보정 + lead·채널별 분산)이 가장 좋습니다(0.620, RMSE 1.105).

### 2. 네 비교: operator 기여는 classical 보정 뒤에 남는가

empirical CRPS 10일 평균. 같은 열 안에서는 보정 절차가 같습니다.

| 경로 | 없음 | s_c | s_kc | in-loop α | MBM형 |
| --- | --- | --- | --- | --- | --- |
| base | 0.721 | 0.676 | 0.672 | 0.649 | 0.620 |
| op | 0.661 | 0.643 | 0.641 | 0.651 | 0.621 |
| op_sp | 0.701 | 0.655 | 0.653 | 0.647 | 0.618 |
| op_sp_member | 0.682 | 0.650 | 0.646 | 0.643 | 0.618 |
| op_sp_nophys | 0.730 | 0.669 | 0.669 | 0.650 | 0.620 |
| op_sp_full | 0.734 | 0.676 | 0.675 | 0.658 | 0.620 |
| op_sfno | 0.661 | 0.643 | 0.641 | 0.649 | 0.619 |
| op_sp_sfno | 0.701 | 0.654 | 0.651 | 0.641 | 0.618 |
| loop_op* | 0.653 | 0.649 | 0.646 | 0.653 | 0.618 |
| loop_op_sp* | 0.653 | 0.651 | 0.647 | 0.653 | 0.619 |

\* in-loop α = 1.75로 보정한 사슬 위에서 학습·실행한 operator. "없음" 열이 이미 in-loop 보정을 포함하므로 base의 in-loop 열(0.649)과 비교합니다.

operator의 이득(base 대비)은 보정이 강해질수록 줄어듭니다: 보정 없음 0.06 → 채널별 분산 0.03 → in-loop 0.00~0.008 → MBM형 0.00~0.002. **평균의 선형 보정과 lead별 분산 보정을 하고 나면 operator 7종 모두 base와 같습니다.**

평균 RMSE도 같은 이야기입니다: operator는 1.185 → 1.14~1.16으로 줄이지만, 학습 없는 선형 평균 보정이 1.105로 더 줄이고, 그 위에서는 operator가 있든 없든 1.10입니다.

### 3. 구조 지표: operator가 공간·변수 구조를 고쳤는가

test16, 10일 평균. 양수 채널 배율은 상관 지표(변수 간, lag-1)를 바꾸지 못하므로 그 열은 보정과 무관합니다.

| 경로 (보정) | 변수 간 상관 불일치 | 물리 쌍만 | lag-1 공간 상관 불일치 | variogram (s_kc 후) | 공간 차분 CRPS (MBM형 후) |
| --- | --- | --- | --- | --- | --- |
| base | **0.067** | **0.055** | 0.050 | 0.069 | 0.204 |
| op | 0.098 | 0.074 | 0.048 | 0.069 | 0.205 |
| op_sp | 0.076 | 0.060 | 0.052 | 0.068 | 0.204 |
| op_sp_member | 0.070 | 0.057 | 0.052 | 0.068 | 0.204 |
| op_sp_full | 0.074 | 0.062 | **0.046** | **0.066** | **0.202** |
| op_sfno | 0.126 | 0.111 | 0.050 | 0.069 | 0.204 |

- 변수 간 구조는 **나빠집니다**. 자유형 operator는 멤버 편차의 변수 간 상관을 실제 오차의 상관에서 더 멀어지게 만듭니다(0.067 → 0.098, SFNO 0.126). spread 보존판도 조금 나쁩니다.
- 공간 구조는 band 제한을 푼 op_sp_full에서만 약하게 좋아집니다(lag-1 0.050 → 0.046, variogram 0.069 → 0.066). 하지만 CRPS는 나빠서(보정 없음 0.734) 순 효과가 없습니다.
- DCT band power: 모든 경로에서 band 밖(격자 규모) 멤버 power가 정답의 0~2%입니다. operator가 band 제한 없이도 격자 규모 구조를 거의 만들지 않습니다.
- 앙상블 통계를 입력에 넣어도(op_sp) 멤버별 operator(op_sp_member)보다 나은 점이 없습니다. classical3의 기준으로는 "앙상블 수준의 학습"을 주장할 근거가 없습니다.
- 물리 항: 없으면 보정 전 5·10일이 나빠지고(0.735 / 0.711) 정역학 잔차가 2배입니다. classical 보정 후에는 차이가 사라집니다.
- 구면 SFNO는 같은 설정의 FNO와 같습니다(합성 25×48 격자에서는 극 처리 차이가 드러나지 않음).

### 4. classical 먼저, operator는 나머지 (loop_op, loop_op_sp)

소유자가 제안한 새 역할("classical 분산 보정 후 operator는 나머지 보정")을 직접 시험했습니다. 직렬 사슬 → in-loop 보정(α = 1.75, base에서 맞춘 값) → operator 순서로 고정하고, operator만 2005년에 같은 방식으로 학습했습니다.

| 경로 | empirical CRPS 1/5/10일 | 평균 | 평균 RMSE | 변수 간 상관 불일치 | 정역학 1/10일 (K) |
| --- | --- | --- | --- | --- | --- |
| base + in-loop | 0.693 / 0.638 / 0.630 | 0.649 | 1.140 | 0.071 | 1.43 / 3.55 |
| loop_op_sp | 0.653 / 0.641 / 0.637 | 0.653 | 1.149 | 0.076 | 1.58 / 2.75 |
| loop_op | 0.651 / 0.650 / 0.643 | 0.653 | 1.148 | 0.083 | 1.59 / 2.78 |

- operator가 1일을 0.04 낮추지만(0.693 → 0.65) 5·10일을 0.003~0.013 올려서, 평균은 나아지지 않습니다(0.653 vs 0.649).
- 그 위에 다시 맞춘 in-loop α는 1.0입니다. 분산 크기는 이미 맞아 있어 operator가 할 분산 일이 남지 않았습니다.
- MBM형을 더하면 0.618~0.619로 base+MBM형(0.620)과 같습니다. MBM형 + 접공간 사영도 모든 경로에서 0.618~0.620이고 정역학 잔차는 1.2~1.9 / 1.9~2.7 K입니다.
- 결론: 합성 자료에서는 classical 보정 뒤에 operator가 고칠 "나머지"가 CRPS로 보이지 않습니다. 1일의 이득은 lead별 분산 보정(s_kc, MBM형)으로도 똑같이 얻어집니다.

### 5. classical 보정 뒤 flow matching·PINN의 역할 (base_noQ, base_nophys)

classical 보정이 operator의 이득을 지웠으니, PINN의 이득도 지우는지 확인했습니다. 같은 보정을 세 사슬에 적용합니다: base(직렬 사슬), base_noQ(같은 사슬에서 PINN Q를 추론 때 건너뜀, flow matching만), base_nophys(물리 항 없이 학습한 직렬 사슬, `serial_nophys.pt`). 보정은 모두 2006년에 사슬별로 따로 맞췄습니다.

| 보정 | base | base_noQ (FM만) | base_nophys |
| --- | --- | --- | --- |
| raw | 0.721 | 0.796 | 0.743 |
| s_kc (post-hoc 분산) | 0.672 | 0.738 | 0.714 |
| in-loop (α 1.75 / 1.5 / 1.5) | 0.649 | 0.694 | 0.679 |
| MBM형 | 0.620 | 0.629 | 0.625 |
| MBM형 + 접공간 사영 | 0.620 | 0.629 | 0.625 |
| 2020 test, MBM형 + 사영 | 0.632 | 0.639 | 0.632 |
| MBM형 + 사영 1/5/10일 | 0.613 / 0.616 / 0.610 | 0.606 / 0.626 / 0.628 | 0.615 / 0.621 / 0.622 |
| 평균 RMSE (MBM형) | 1.105 | 1.105 | 1.104 |
| 변수 간 상관 불일치 (MBM형 + 사영) | 0.069 | 0.071 | 0.091 |
| 정역학 1/10일 (K, MBM형 + 사영) | 1.24 / 2.15 | 1.58 / 2.81 | 1.70 / 3.05 |

- PINN의 CRPS 이득은 raw 0.075에서 분산 보정 후 0.066, in-loop 0.045, MBM형 후 0.009로 줄어듭니다. MBM형 후 평균 RMSE가 세 사슬 모두 1.105로 같습니다. PINN이 주던 평균 개선은 대부분 lead·채널별 선형 보정(a + bμ)으로 재현되는 종류였습니다.
- 그래도 MBM형 뒤에 남는 것이 있습니다. 5·10일 CRPS(0.616 / 0.610 vs FM만 0.626 / 0.628), 정역학 정합성(10일 2.15 vs 2.81 K), 변수 간 의존 구조(물리 항 학습 0.069 vs 물리 항 없음 0.091)입니다. 1일은 FM만이 조금 낫습니다(0.606 vs 0.613).
- 물리 항 자체(base vs base_nophys)의 CRPS 기여는 MBM형 후 0.005(test16), 0.000(2020 test)입니다. 물리 항의 값은 점수가 아니라 정합성과 의존 구조에서 나옵니다.
- in-loop(롤아웃 안에서 보정)에서는 PINN 이득이 0.045로 크게 남습니다. 사후 통계 보정을 쓸 수 없는 상황(검증 기간 없는 새 기후 시나리오, 롤아웃 중 보정)에서 PINN이 주 역할을 합니다.
- 해석: flow matching = 시나리오 생성(가장 좋은 의존 구조를 가진 편차), PINN = 물리 정합 평균 보정 + 사영의 기준이 되는 접공간, classical = 진폭과 선형 편향. PINN의 역할은 남지만 "점수를 크게 올린다"에서 "물리적으로 정합된 보정과 그 접공간을 준다"로 바뀝니다.

## 가설 판정

| 가설 | 판정 (합성 자료) | 근거 |
| --- | --- | --- |
| classical 분산 보정만으로 과소분산이 해결된다 | 대부분 예 | SSR 0.29~0.54 → 0.93~0.98, CRPS 0.721 → 0.672(post-hoc), 0.620(MBM형) |
| post-hoc 분산 보정은 물리 정합성을 보존한다 | 아니오, 사영하면 예 | 정역학 2.1 → 10 K; 접공간 사영으로 2.3 K, CRPS 동일 |
| NO는 분산 보정을 넘는 개선을 준다 (classical3) | post-hoc 분산만 상대로는 예, 평균+분산 classical 상대로는 아니오 | 0.676 vs 0.643~0.655, MBM형 후 0.620 vs 0.618~0.621 |
| PINN은 classical 보정 뒤에도 역할이 남는다 | 예, 줄어든 채로 | MBM형 후 0.620 vs FM만 0.629, 5·10일 0.010~0.018; 정역학 2.15 vs 2.81 K; in-loop 0.649 vs 0.694 |
| 물리 항은 점수를 올린다 | MBM형 후에는 거의 아니오 | 0.620 vs 0.625(test16), 0.632 vs 0.632(2020); 의존 구조 0.069 vs 0.091 |
| 그 개선은 공간·변수 간 구조의 개선이다 | 아니오 | 변수 간 상관 불일치 악화, 공간 지표는 평균 개선을 따라갈 뿐 |
| classical 보정 사슬 위의 operator가 나머지를 고친다 (새 역할) | 합성에서는 아니오 | in-loop 위 0.653 vs 0.649, MBM형 위 0.618 vs 0.620 |
| 앙상블 통계 입력이 필요하다 | 아니오 | op_sp ≈ op_sp_member |
| fair CRPS로 보정 배율을 고르면 된다 | 아니오 | 재보정된 멤버에서는 과대분산을 고름; empirical이 맞음 (Roberts 2026과 같은 결론) |

## 취약점 (분석해야 할 것)

- W1. 합성 자료의 오차 구조가 단순할 수 있습니다. 직렬 사슬의 오차가 대부분 선형 평균 편향 + 진폭 부족이라면, classical이 다 가져가는 것이 당연합니다. ERA5에서 MBM 뒤에 남는 오차(상태 의존 편향, 흐름 의존 분산, 전선·저기압의 위치 오차)가 있는지가 operator의 존재 이유를 결정합니다.
- W2. 평가 초기장이 16개라 0.002~0.005 차이는 잡음 수준입니다. 2020년 test(초기장 4개)도 순서가 같습니다: MBM형 후 base 0.632, operator 6종 0.628~0.632; s_kc 후 base 0.676, operator 0.650~0.665.
- W3. post-hoc MBM은 정역학 잔차를 7~9 K로 키웁니다. 쓰려면 접공간판(`mbm_tan`)이 필요합니다.
- W4. operator 학습 손실(energy score + 공간 차분 ES)이 변수 간 의존 구조를 직접 겨냥하지 않습니다. variogram score나 물리 쌍의 상관 손실을 넣어야 classical3의 가설을 공정하게 시험할 수 있습니다.
- W5. AE floor(7단계 W1)가 그대로라 1일 점수의 바닥이 있습니다.

## 다음 단계 제안

1. ERA5에서 `fit_spread.py`로 classical 사다리(classical → classical_lead → mbm → mbm_tan)를 먼저 맞추고, MBM 뒤 잔차의 구조(상태 의존성, 공간 상관)를 진단합니다.
2. operator는 MBM(또는 in-loop) 보정을 고정한 사슬 위에서, 보정 후 잔차만 목표로 학습합니다(loop_op의 ERA5판). 손실에 variogram score / 변수 쌍 의존 구조를 넣고, 판정은 이 문서의 구조 지표로 합니다.
3. 앙상블 축의 상호작용이 필요하면 PoET식 attention을 시험합니다(현재 앙상블 통계 입력은 효과 없음).

## ERA5 명령어 (GPU)

`docs/RUN_ON_GPU.md` 10절.
