# 17단계: 구면 천수방정식 시험대 — "물리를 직접 배우는" 길의 검증

소유자(2026-10-27 11:50): "maziarraissi/PINNs 구조로 직접 물리를 학습시키는 방향과 torchdiffeq로 neural ODE dx/dt 구조로 물리를 학습시켜 ensemble을 물리적으로 보정하는 방향은? 이전 방향은 물리 손실보다 자료 기반 보정 같다." 12:35: "이걸로도 한번 실험 진행 해보죠" (제안한 2-D 천수방정식 시험).

## 요청을 이렇게 읽었습니다

ERA5에서는 두 가지가 섞여 있어 "물리 손실이 왜 동역학을 못 가르치는가"를 가를 수 없습니다. (i) 이산 연산자의 절단오차(5.625°·12층), (ii) 닫힘 결손(비단열·마찰·습윤 과정이 방정식에 없음). 방정식이 완전하고 연산자가 정확한 작은 세계에서 같은 네 가지 학습법을 돌리면 두 원인을 분리할 수 있습니다. 질문은 셋입니다.

1. **물리 잔차 손실**(16단계의 λ_p 항)은 연산자가 정확할 때 동역학을 가르치는가, 그리고 닫힘이 빠지면 어떻게 되는가.
2. **Raissi 좌표 PINN**(자동미분, 절단오차 없음)은 초기 자료 + 물리만으로 미래를 풀 수 있는가(순방향 문제).
3. **torchdiffeq neural ODE**는 같은 자료에서 Heun rollout과 다른 것을 배우는가(솔버의 문제인가, 교사의 문제인가).

## 1. 시험대 (기존 구현 재사용)

| 부품 | 출처 | 용도 |
| --- | --- | --- |
| 구면 천수방정식 스펙트럴 솔버 (`ShallowWaterSolver`, Galewsky 제트, 무작위 초기장, AB3, 와도·발산 hyperdiffusion) | torch-harmonics `examples/shallow_water_equations.py` (수정 없음) | 진실 생성, 정확한 우변 F(x) |
| 경향망 trunk (`Climate_ResNet_2D`) | ClimODE (`pgcfm.core.Net`, vendored) | 격자 학습기 f_θ |
| `odeint` | rtqichen/torchdiffeq (ClimODE의 솔버) | neural ODE arm |
| 좌표 PINN (`PhysicsInformedNN` 포트: tanh MLP, Xavier, Adam → L-BFGS, 접벡터 바람) | maziarraissi/PINNs (`third_party/pinns`, `pgcfm/physics/pinn.py`) | Raissi arm |

새 코드 `pgcfm/swe.py`: `World`(격자↔스펙트럴, Galewsky(umax, lon0, bump), `tendency_spec(closure)`), `TendencyNet`, `rollout_heun`/`rollout_node`, `strong_residual`, `SWEPinn`(이류형 SWE 잔차를 (λ, φ, t) 자동미분으로), 테스트 `tests/test_swe.py`(Williamson case 2 정상 해의 잔차 0, 솔버 첫 step과 경향 일치 등 5개).

**두 설정.**

- **full**: 진실이 학습기 해상도(T21, 64×128)에 삽니다. 물리는 솔버의 hyperdiffusion만 빼고 완전합니다. 그 hyperdiffusion이 이 세계의 "닫힘"이고 크기를 잽니다.
- **coarse**: 진실은 T42(128×256), 학습기는 그것의 T21 절단만 봅니다. 잘린 모드와의 상호작용이 빠진 닫힘입니다. ERA5 5.625°의 상황입니다.

**자료.** 학습 10 + 검증 2 궤적: Galewsky 제트를 흔든 초기장(umax 60–90 m/s, 융기 경도 무작위, 진폭 0.5–1.5×), 제트가 깨지도록 96 h 돌린 뒤 72 h를 1 h 간격으로 기록. 무작위 초기장 궤적 4개 추가. 시험: 표준 Galewsky(t0 = 96, 120, 144, 168 h) 4창 + 무작위 2창, 24 h 예보.

**arm.**

| arm | 학습 | 무엇을 가르는가 |
| --- | --- | --- |
| data | rollout MSE (K = 3 h), Heun 1 h | 16단계 자료항만 core |
| node | rollout MSE, `odeint` rk4 1 h | 솔버만 바꿈 (소유자의 torchdiffeq 길) |
| pi | data + 1·\|f_θ(x) − F_learn(x)\|²/b² (rollout 상태) | 16단계 물리항, 정확한 연산자 |
| pi_only | 잔차만, 자료 상태에서 (궤적 없음) | "물리를 직접 배움" |
| raissi fwd | t = 0 격자 자료 + [0, T]의 배치점 물리 (T = 6, 24 h) | 소유자의 Raissi 길, 순방향 문제 |
| raissi fit | 위 + 매시간 자료 1024점 | PINN의 표현력 대조 (보간 문제) |

**측정.** lead 1–24 h의 Φ·바람 RMSE(persistence 점선), 에너지 비, 진실 상태에서의 접벡터 잔차 \|f_θ(x) − F_true\|/b (F_true = 절단 진실의 정확한 경향, hyperdiffusion 포함; b = 그 RMS), 그리고 **연산자의 닫힘 결손** \|F_learn(x) − F_true\|/b (물리항이 볼 수 없는 부분; full에서는 hyperdiffusion, coarse에서는 절단까지).

## 2. 가설과 기각 기준

- H17a: 연산자가 정확할수록(닫힘 결손 ↓) pi가 data를 이기고, coarse에서는 진다. 기각: full에서도 pi ≤ data.
- H17b: pi_only는 닫힘 결손만큼 틀린 접벡터를 배워 24 h에 에너지가 쌓이거나 발산한다(16단계 arm 3의 정지 대기와 짝). 기각: pi_only가 data에 근접.
- H17c: Raissi fwd는 6 h 안에서도 persistence를 못 이기거나 24 h에서 무너진다; fit은 맞춘다(표현력은 있음). 기각: fwd가 24 h까지 data 수준.
- H17d: node = data (솔버는 교사가 아님). 기각: node가 유의하게 다름.

## 3. 결과

(실행 중: `runs/step17/coarse.json`, `full.json`, 그림 `docs/step17/fig_step17.png`.)

## 4. 약점

(결과 뒤에 씁니다.)
