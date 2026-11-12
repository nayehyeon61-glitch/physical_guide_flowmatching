# Tendency core ++: 확장 core 둘을 쓰는 모델 (2026-11-12)

`docs/kai_tendency.md` 7.9절의 가장 좋은 모델은 **미세조정 KAI + 따로 학습한 core 둘(seed 0, 1)의 되먹임 연결**입니다. 1.5° + 버전(`docs/kai_tendency_plus_1p5_daily.md`)은 이 core를 5.625°의 크기 그대로(hidden 48, 259,842개) 격자점이 14배인 격자에 올렸고, 크기 실험(`scripts/run_kai_plus_arch.sh`, commit `aab72ef`)이 더 큰 core를 만들었습니다. **++ 버전은 이 확장 core 둘로 같은 two core 모델을 만드는 것**입니다. 실행은 `scripts/run_kai_plusplus.sh` 하나입니다.

아직 실제 크기의 결과는 없습니다. CPU에서 작은 변형으로 모든 단계가 도는지만 확인했습니다(6절).

## 1. + 버전과 ++ 버전

| 항목 | + 버전 | ++ 버전 |
| --- | --- | --- |
| KAI | 7.57 M, 20,000 iteration + 미세조정 1,000 iteration | 같음 (같은 폴더의 `kai.pt`, `kai_ro.pt`를 그대로 씀) |
| core의 구조 | 5.625°의 network: trunk 48, 기억 24채널, 259,842개 | 확장 core: 넓은 trunk 또는 다중 규모 trunk, 기억 48채널, 기억 입력의 2층 encoder |
| core의 수 | 하나 (seed 0). seed twin은 크기 실험에서 학습 중 | **둘**: 확장 core(seed 0)와 그 seed twin(seed 1) |
| 궤적 모델 | 미세조정 KAI + core 하나 되먹임 (`loop_core_plus@kai_ro`) | 미세조정 KAI + 확장 core 둘 되먹임 (`loop_A+B@kai_ro`) |
| RMSE가 가장 작은 변형 | 선행일과 채널마다 섞는 혼합 (`blend_core_plus@kai_ro`) | 세 예보의 혼합 (`blend_A+B@kai_ro`) |
| 물리항 G(x) 48개, ConvGRU 기억 8일, MZ loss, 하루 4 substep | 같음 | 같음 |

core의 입력, loss, 학습 일정(6,000 iteration, batch 8), 자료는 + 버전과 같습니다. 바뀌는 것은 core의 network와 core의 수뿐입니다.

## 2. 확장 core의 변형

`ARCH`로 고릅니다. 크기 실험과 같은 이름, 같은 옵션이라서 크기 실험이 이미 학습한 core(`core_plus_w128.pt` 등)는 다시 학습하지 않고 그대로 씁니다.

| `ARCH` | 구조 | parameter | core 둘 (KAI 7.57 M 대비) | 12 iteration + 검증 시간 |
| --- | --- | --- | --- | --- |
| (+ 버전) | trunk 48, 기억 24 | 259,842 | 0.52 M (+7 %) | 88 s |
| `w96` | trunk 96, 기억 48, encoder 128 | 925,218 | 1.85 M (+24 %) | 114 s |
| `w128` (기본) | trunk 128, 기억 48, encoder 128 | 1,435,394 | 2.87 M (+38 %) | 143 s |
| `u64` | 다중 규모 trunk 64 / 128 / 192 (1.5 / 3 / 6°), 기억 48, encoder 128 | 2,883,522 | 5.77 M (+76 %) | 110 s |
| `u96` | 다중 규모 trunk 96 / 160 / 224, 기억 48, encoder 128 | 4,330,530 | 8.66 M (+114 %) | 150 s |

시간은 크기 실험을 돌린 쪽에서 GPU 한 장으로 잰 값입니다(commit `aab72ef`). + 버전의 core 학습이 6,000 iteration에 11.8시간이었으므로, 같은 비율이면 `w128`과 `u96`은 core 하나에 19~20시간쯤입니다(추정).

기본을 `w128`로 둔 것은 크기 실험에서 지금 학습 중인 넓은 trunk이기 때문이고, 어느 변형이 더 좋은지는 아직 모릅니다. **크기 실험의 결과가 나오면 이긴 변형을 `ARCH`로 주면 됩니다.**

## 3. 명령어 (GPU 서버)

+ 버전을 돌린 폴더(`runs/kaiplus_1p5_daily`, `data/era5_1p5_daily`)에서 그대로 돌립니다.

```bash
git pull
# 기본: 넓은 core(w128)와 그 seed twin. 없는 것만 학습하고(끊기면 이어서), 끝나면 채점
bash scripts/run_kai_plusplus.sh

# 다중 규모 core로
ARCH=u96 bash scripts/run_kai_plusplus.sh

# seed twin 대신 서로 다른 확장 core 둘. 크기 실험이 둘 다 학습했으면 학습 없이 채점만 합니다
ARCH=w128 ARCH2=u96 SC=pp_w128_u96 bash scripts/run_kai_plusplus.sh

# GPU를 골라서 (크기 실험이 다른 GPU에서 도는 동안)
CUDA_VISIBLE_DEVICES=1 bash scripts/run_kai_plusplus.sh

# 학습은 다른 GPU에서 돌리고, 여기서는 checkpoint를 기다렸다가 채점만
TRAIN=0 WAIT=1 bash scripts/run_kai_plusplus.sh

# 명령만 출력
DRYRUN=1 bash scripts/run_kai_plusplus.sh
```

- 학습하는 것: `core_plus_<ARCH>.pt`(seed 0)와 `core_plus_<ARCH>_s1.pt`(seed 1). 이미 있으면 건너뜁니다.
- **같은 core를 두 곳에서 학습하지 않습니다.** 크기 실험이 `core_plus_w128`을 학습하는 중이면(같은 서버에 그 학습 process가 있거나, 폴더의 `core_plus_w128.part` 또는 `.log`가 최근 2시간 안에 바뀌었으면) ++ driver는 seed twin을 먼저 학습하고, 그다음 `core_plus_w128.pt`가 생기기를 기다립니다. 두 학습이 한 폴더에서 같은 이름으로 돌면 서로의 중간 저장을 덮어쓰기 때문입니다. 멈춘 학습을 기다리지 않고 여기서 바로 이으려면 `FORCE=1`을 줍니다.
- 반대 방향의 확인은 없습니다. ++ driver가 학습하는 core(예: `core_plus_w128_s1`)를 다른 명령으로 동시에 시작하지 마세요.
- GPU는 `CUDA_VISIBLE_DEVICES`로 고릅니다. 크기 실험과 같은 GPU에 올리면 메모리는 들어가도(변형 하나에 12.6~17.8 GB) 둘 다 느려집니다.
- 폴더에 `core_plus_s1.pt`(+ 버전 core의 seed twin, 크기 실험의 `s1`)가 있으면 **+ 버전의 two core 모델**도 함께 채점해서 나란히 놓습니다.
- 채점은 예보 파일을 디스크에 잠시 씁니다(core나 KAI 하나의 예보가 약 9 GB, 모두 60 GB 안팎).

## 4. 채점에서 나오는 비교

`score_pp.json`, `score_pp.log`(pairs 줄: 부트스트랩 95 % 구간), `breakdown_pp/breakdown_pp.txt`(정확한 정보 / 진폭 분해), `variable_table_pp.md`(변수별 표)에 아래 쌍이 나옵니다. A = 확장 core(seed 0), B = 둘째 core입니다.

| 쌍 | 묻는 것 |
| --- | --- |
| `loop_A+B@kai_ro : kai_ro` | ++ 모델이 미세조정 KAI보다 얼마나 좋은가 |
| `loop_A+B@kai_ro : loop_core_plus@kai_ro` | ++ 모델이 + 모델보다 얼마나 좋은가 |
| `loop_A+B@kai_ro : loop_core_plus+core_plus_s1@kai_ro` | 같은 구조(core 둘)끼리: 확장 core의 몫만 (`core_plus_s1.pt`가 있을 때) |
| `loop_A+B@kai_ro : loop_A@kai_ro` | 확장 core에서도 둘이 하나보다 좋은가 |
| `loop_A@kai_ro : loop_core_plus@kai_ro`, `A : core_plus` | core 하나끼리: 확장 core 대 + core (되먹임, 단독) |
| `loop_B@kai_ro : loop_A@kai_ro`, `B : A` | seed 차이의 크기 (변형이 넘어야 할 잡음) |
| `blend_A+B@kai_ro : kai_ro`, `: kai`, `: blend_core_plus@kai_ro` | 혼합(RMSE가 가장 작은 변형)의 이득 |
| `blend_A+B@kai_ro : blend_climatology@kai_ro` | 혼합이 미세조정 KAI를 기후값 쪽으로 줄인 것보다 좋은가 (단순 감쇠 대조) |
| `blend_A+B@kai_ro : loop_A+B@kai_ro` | 혼합 대 되먹임 |

## 5. 볼 것 (취약점 기준)

1. **용량이 정말 모자랐는가.** + 버전의 core는 1.5°에서 단독으로 KAI보다 1일에 64.6 % 나빴습니다. 확장 core 단독(`A : core_plus`)이 이 차이를 얼마나 줄이는지가 첫 확인입니다.
2. **seed 차이보다 큰가.** 5.625°에서 seed만 바꾼 차이는 0.1~0.4 %였습니다. `B : A`와 `loop_B : loop_A`가 1.5°의 잡음 크기이고, 확장의 이득은 이보다 커야 합니다.
3. **이득이 정보인가 진폭인가.** + 버전에서 core 하나의 10일 이득 +3.5 % 가운데 정보는 +0.5 %였습니다. `breakdown_pp.txt`의 pattern(information) 줄과 ACC로 봅니다. 혼합은 단순 감쇠 대조(`blend_climatology`)보다 좋아야 core의 몫이라 할 수 있습니다.
4. **둘의 이득이 앙상블인가 용량인가.** 5.625°에서 core 둘의 이득(+1.7 / +1.3 / +1.0 / +0.6 / 0.0 %)은 따로 학습한 두 core의 평균 효과였습니다. 확장 core는 이미 용량이 크므로 둘의 이득이 줄 수도 있습니다(`loop_A+B : loop_A`).
5. **공정성.** ++ 모델은 KAI에 parameter를 24~114 % 더합니다(+ 버전 core 하나는 3.4 %). 같은 크기의 KAI는 학습하지 않았으므로, "같은 parameter에서 더 좋다"는 말은 아직 할 수 없습니다.
6. **loss 가중.** sp, skt, z50의 큰 이득은 일부가 KAI와 core의 loss 가중 차이에서 옵니다(+ 문서 8.5절). ++에서도 그대로입니다.

## 6. 시험 상태

- **CPU 작은 시험: 통과** (2026-11-12 19:40~20:29 KST, `SMOKE=1`, 24일 자료, 작은 KAI, 작은 변형: trunk 24, 기억 12채널, encoder 24, 2 iteration). 확장 core 둘(seed 0, 1)의 학습, 채점(예보 34종, 쌍 19개와 부트스트랩 구간), 정보 / 진폭 분해(쌍 16개), 변수별 표까지 한 번에 끝났습니다. 폴더에 `core_plus_s1.pt`가 있어서 + 버전의 two core 모델과 단순 감쇠 대조도 함께 채점됐습니다. seed twin은 첫 core와 다른 예보를 냈습니다(seed가 적용됨).
- 이 시험의 수치에는 의미가 없습니다. 2 iteration만 학습한 core는 persistence와 거의 같습니다. 도는지만 본 것입니다.
- 같은 core를 두 번 학습하지 않는 확인(3절)은 학습을 흉내 낸 가짜 실행으로 일곱 경우를 시험했습니다: 아무것도 없을 때, 다른 곳에서 학습 중(중간 저장이 최근 것), 멈춘 학습(3시간 전), 같은 서버의 process, `FORCE=1`, 둘 다 학습 중, `ARCH2`.
- ++ driver로 돌려 본 변형은 넓은 trunk(`w*`)뿐입니다. 다중 규모 trunk(`u*`)는 같은 옵션을 크기 실험 쪽에서 GPU로 학습하고 있지만, ++ driver를 거친 작은 시험은 하지 않았습니다.
- 실제 크기(1.5°, GPU)에서는 아직 돌리지 않았습니다. 학습 시간(2절)과 디스크(3절)는 추정입니다.
