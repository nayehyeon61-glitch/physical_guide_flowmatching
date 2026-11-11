# Tendency core +: KAI의 해상도(1.5°)에서 학습하는 tendency core (2026-11-11)

## 요청을 이렇게 읽었습니다

소유자 요청(2026-11-11 14:41 KST): "tendency core에 대해서 high resolution, 지금은 5.625에서 KAI 수준의 resolution 정도로 학습하는 구조의 version, 즉 + version을 만들어 주세요."

- **KAI 수준의 해상도**는 KAI가 실제로 쓰는 자료 격자인 1.5°(위도 121 × 경도 240, 극 행 포함, 남에서 북)로 읽었습니다. `pgcfm/contract.py`의 KAI 격자가 이것입니다.
- **+ version**은 `docs/kai_tendency.md`의 tendency core(물리항 G(x) + GRU 기억항 + Mori-Zwanzig sequence loss)와 같은 부품을 그 격자에서 학습할 수 있게 만든 구조입니다. 새 가설을 넣은 것이 아니라, 5.625° 구조를 1.5°로 그대로 올리면 깨지는 곳을 고친 것입니다.
- 이 문서에 있는 것: 구현, 5.625°의 결과가 그대로인지 확인, 실제 1.5° ERA5의 일부(해마다 24일)로 한 작은 시험, GPU에서 돌리는 명령.
- 이 문서에 없는 것: 1.5° 전체 학습과 그 예측 성능. 이 환경은 CPU 4코어, 메모리 14 GB이고 GPU가 없습니다. **1.5°에서의 성능 수치는 아직 없습니다.**

## 1. 5.625° 구조를 1.5°로 올리면 깨지는 곳

격자점은 2,048개에서 29,040개로 14.2배, 격자 간격은 3.75배 촘촘해집니다.

| 취약점 | 5.625° | 1.5°에서 생기는 일 | + version의 처리 |
| --- | --- | --- | --- |
| 하루에 solver 한 step | 한 Heun step이 보는 범위 16칸, 제트(60 m/s)의 하루 이동 12칸 | 하루 이동이 44칸이라 16칸을 넘음. 공기가 온 곳을 망이 볼 수 없음 | 하루를 Heun 4 step으로 나눔(`--core-substeps 4`). substep 하나에 11칸 |
| 극 | 극 행이 없음 | 물리항 G(x)는 1/cos φ를 포함해 극에서 정의되지 않고, 극 옆 행(88.5°)에서 38배로 커짐 | 극 행의 G는 0. G의 scale은 85° 안쪽의 면적 가중 RMS |
| 학습 메모리 | 표본 하나에 약 0.7 GB (8 + 4일 sequence) | 표본 하나에 약 37 GB | substep과 기억항 갱신을 gradient checkpointing(`--core-ckpt`). 표본 하나에 약 1.6 GB |
| 자료 크기 | 한 해 0.2 GB, 전부 메모리에 올림 | 한 해 2.84 GB, 6년 17 GB | 디스크에 둔 채 batch마다 읽고 정규화. 통계도 나눠서 계산 |
| 채점 메모리 | 한 모델의 예측 0.65 GB | 한 모델의 예측 9.3 GB | 초기장을 묶음으로 나눠 계산, 예측은 임시 파일 |
| 장치 | CPU | CPU로는 학습할 수 없음 | `--device cuda` |
| 자료 변환 | 6시간 파일에서 일평균 | 6시간 파일이 한 해 11 GB(압축 전) | 일평균을 바로 씀(`--daily-mean`) |

하루 이동 칸 수는 제가 계산한 값입니다(위도 45°에서 60 m/s는 하루 5,184 km, 한 칸은 5.625°에서 443 km, 1.5°에서 118 km). 한 번의 f 평가가 보는 범위는 3 × 3 convolution 8개라서 8칸이고, Heun step은 f를 두 번 평가하므로 16칸입니다.

## 2. 구조

하루의 계산은 5.625° core와 같고, solver만 하루를 n = 4로 나눕니다.

```
m(t)      = GRU( m(t − 1일), [x, 관측 경향, innovation, G(x), G(x 전날)] )     하루에 한 번
x_(i+1)   = x_i + (Δ / 2) · ( f(x_i) + f(x_i + Δ · f(x_i)) ),   Δ = 1 / n 일,  i = 0 … n − 1
f(x)      = Net( x, m(t), G(x), 지형과 위치, 계절 )                              망의 출력 단위는 일 증분의 표준편차
x(t + 1일) = PositiveMass( x_n, 다음 날의 일사 )
```

- 기억 m은 하루 동안 고정이고, 물리항 G(x)는 substep마다 다시 계산합니다(Heun의 두 번째 평가는 같은 G를 씁니다).
- 학습 loss는 그대로입니다: 8일 sequence를 하루씩 예측한 오차 + rollout 오차 + 잔차의 lag 1, 2 상관 벌점(Mori-Zwanzig 형태).
- 망은 5.625°와 같은 크기(hidden 48, 블록 (2, 2))가 기본값입니다. 격자점이 14배인데 parameter 수는 같으므로, 작은 규모를 다 담지 못할 수 있습니다. `PLUS_HIDDEN`, `PLUS_BLOCKS`로 키울 수 있습니다(6절).
- 극 행: convolution의 위도 방향 padding은 5.625°와 같이 reflect입니다. 극 행의 물리항만 0으로 둡니다.

## 3. 구현

| 파일 | 바뀐 것 |
| --- | --- |
| `pgcfm/dyncore.py` | `DynCore.step`을 `substep`과 `finish`로 나눔(값은 같음). `OperatorFeatures`: 극 행이 있는 격자에서 G를 전체 행으로 채우고(극 행 0), scale을 면적 가중으로 추정. `to(device)` |
| `pgcfm/kai_tend.py` | `MemCore._advance`: substep마다 checkpoint, 기억 상태를 인자로 넘김. `remember`: GRU 갱신도 checkpoint. `core_step`: 기억항이 없는 core도 substep마다 checkpoint. `operator_features(..., chunk)`. `KAIPos.grad_ckpt`: KAI의 블록마다 checkpoint(4.4절) |
| `scripts/run_kai_tendency.py` | `--device`, `--lazy`, `--core-ckpt`, `--kai-ckpt`, `--score-batch`, `--score-chunk`, `--val-batch`. 한 해가 1 GB를 넘으면 자료를 디스크에 둠(`ZYear`, `Lazy`, `stats_stream`). 채점을 초기장 묶음으로 나눔. GPU에서는 학습 log에 그때까지의 최대 GPU 메모리(`gpu_gb`)를 적음 |
| `scripts/wb2_to_kai.py` | `--grid 240x121`(WeatherBench2의 1.5° 극 포함 격자), `--daily-mean`, `--days` |
| `scripts/local_era5_to_kai.py` | `--grid 1.5`(극 포함), `--daily-mean`, `--days`. 0.25° 시간별 archive에서 면적 보존 평균 |
| `scripts/run_kai_plus.sh` | 자료, KAI, rollout 미세조정 KAI, core +, 채점, 항별 분해, 변수별 표를 차례로 실행. 끝난 단계는 건너뛰고, 끊긴 학습은 이어서 함 |
| `scripts/kai_step_memory.py` | KAI 학습 step 하나의 메모리, 시간, 기울기를 재는 작은 도구(4.4절) |
| `scripts/eval_kai_breakdown.py`, `scripts/kai_variable_table.py` | 촘촘한 격자의 규모 분해: 작은 규모를 k10~32(5.625°도 푸는 규모)와 k33 이상(촘촘한 격자에만 있는 규모)으로 나눔. 선행일이 짧은 채점도 처리. 5.625° 출력은 그대로 |

## 4. 확인한 것

### 4.1 5.625°의 결과가 그대로인가

| 시험 | 결과 |
| --- | --- |
| core 학습 4 iteration: 바꾸기 전 코드 대 새 코드 | 가중치와 log가 bit 단위로 같음 |
| 새 코드: 기본 대 `--lazy` 대 `--core-ckpt` | bit 단위로 같음 |
| substep 2: checkpoint 없음 대 `--core-ckpt --lazy` | 가중치가 bit 단위로 같음 |
| 기억항 없는 core, substep 2: checkpoint 없음 대 있음 | bit 단위로 같음 |
| 누적 kernel core(`--core-pacc`): checkpoint 없음 대 있음 | bit 단위로 같음 |
| 채점(KAI, core 2개, 연결, 세 가지 blend): 바꾸기 전 대 새 코드 | 저장한 배열 105개가 모두 같음 |
| 채점: 기본 경로 대 묶음 경로(`--lazy --score-chunk 7 --score-batch 5`) | 채점하는 65채널에서 상대 차이 0.006 % 이하 |

묶음 경로의 차이는 더하는 순서가 달라서 생기는 반올림입니다. 채점하지 않는 상수 채널(지형)의 활동도만 값이 달랐습니다(0으로 나누는 값).

### 4.2 1.5° 작은 시험 (실제 ERA5, 해마다 24일, CPU 2 thread)

| 단계 | 설정 | 결과 | 최대 메모리 | 시간 |
| --- | --- | --- | --- | --- |
| KAI 학습 | 시험 크기 0.43 M, batch 2, 4 iteration | 통과 | 4.2 GB | 47초 |
| core + 학습 | 물리항 + GRU + MZ, 259,842 parameter, substep 4, checkpoint, batch 1, 8 + 1일과 8 + 2일 sequence, 2 iteration | 통과 | 3.3 GB | iteration당 약 40초, 검증(15일) 약 215초 |
| 채점 | 초기장 4개, 선행 4일, 묶음 경로, core 단독과 KAI 연결 | 통과 | 3.2 GB | 274초 |

GPU용 실행 script(`run_kai_plus.sh`)도 같은 자료로 처음부터 끝까지 돌렸습니다(`SMOKE=1 DEVICE=cpu`): KAI, rollout 미세조정 KAI, core +, 채점(KAI와 미세조정 KAI 위의 연결, 기억 0일과 1일), 항별 분해 그림 6장, 변수별 표가 모두 만들어졌습니다.

2 iteration은 학습이 아니라 경로의 시험입니다. core의 출력층은 0에서 시작하므로 core +의 점수는 지속성과 같아야 하고, 실제로 그렇습니다(nRMSE 1 / 3일 0.411 / 0.591, 지속성 0.412 / 0.594). 극 행을 포함해 모든 값이 유한했습니다. 4 iteration의 KAI는 학습되지 않은 상태입니다(0.921 / 0.988).

core + 학습의 메모리를 단계별로 쟀습니다(batch 1, 자료와 모형을 올린 상태 1.2 GB 포함).

| 단계 | 최대 메모리 |
| --- | --- |
| 물리항의 scale 추정 (상태 2개씩) | 1.3 GB |
| 학습 step, 8 + 1일, 하루 단위 checkpoint (처음 구현) | 4.1 GB |
| 같은 것, substep 단위 checkpoint | 3.0 GB |
| 같은 것, 기억항 갱신까지 checkpoint (지금 구현) | 2.55 GB |
| 지금 구현, 8 + 2일 | 2.62 GB |

여기서 계산하면 하루가 늘 때 표본당 약 0.07 GB, substep 하나의 activation이 표본당 약 0.74 GB입니다. checkpoint가 없으면 하루에 substep 4개와 기억항 갱신이 모두 남아 8 + 4일 sequence에 표본당 약 37 GB가 됩니다(이 값은 재지 못하고 계산한 값입니다). 지금 구현은 같은 sequence에 표본당 약 1.6 GB입니다.

### 4.3 1.5° 자료 변환

`wb2_to_kai.py --grid 240x121 --daily-mean`이 쓴 일평균 3일이, 같은 날의 6시간 파일을 평균한 값과 bit 단위로 같았습니다. `local_era5_to_kai.py`의 1.5° 가중치는 행의 합이 1이고, 상수장과 전구 평균을 보존했습니다(1.5°, 1.40625°, 5.625°).

### 4.4 KAI 학습의 메모리 (원래 크기)

새로 받은 clone에서 실행 script를 처음부터 다시 돌리다가 찾은 문제입니다. rollout 미세조정 단계가 시험 크기의 KAI(0.43 M)로도 6.8 GB를 넘었습니다(batch 2, 3일 rollout, 1.5°). KAI는 네 stage 모두 격자를 줄이지 않아서(stride 1), 학습 step이 저장하는 activation이 격자점 수에 비례합니다. 5.625°에서는 보이지 않던 비용입니다.

원래 크기 (3, 3, 15, 3) / (48, 96, 192, 288), 7,574,213 parameter의 학습 step 하나를 재었습니다(`scripts/kai_step_memory.py`, batch 1, 하루, CPU). 61 × 120은 1.5° 격자점의 약 1/4입니다.

| | 31 × 60 | 61 × 120 | 1.5°(121 × 240)로 환산, 표본과 하루당 |
| --- | --- | --- | --- |
| checkpoint 없음 | 0.41 GB | 1.53 GB | 약 6 GB |
| 블록 단위 checkpoint (`--kai-ckpt`) | 0.26~0.29 GB | 0.52 GB | 약 1.4~2 GB |

3일 rollout의 step은 checkpoint가 있으면 61 × 120에서 0.94 GB(31 × 60에서 0.40 GB)라서 1.5°로 환산하면 표본당 약 3.1~3.7 GB이고, checkpoint가 없으면 하루의 약 3배입니다(31 × 60에서 1.16 GB). batch를 1에서 2로 늘리면 61 × 120에서 0.38 GB가 늘었습니다(checkpoint, 하루).

환산은 두 격자의 차이를 격자점 수에 비례시킨 값이고, 1.5°에서 직접 잰 값이 아닙니다(이 기계에서는 다른 학습과 함께 돌릴 메모리가 없었습니다). 이 값으로 계산하면 checkpoint 없이 batch 8은 약 48 GB, 3일 rollout 미세조정은 batch 8에 약 145 GB라서 80 GB GPU에도 들어가지 않습니다.

그래서 `--kai-ckpt`를 넣었습니다. 블록의 입력만 남기고 블록 안의 activation은 backward에서 다시 계산합니다. 기울기는 checkpoint가 없을 때와 bit 단위로 같았습니다(시험 크기의 2일 rollout, 원래 크기의 31 × 60 모두 최대 차이 0). 대신 forward를 한 번 더 하므로 iteration당 시간이 늘어납니다. 이론상 약 1.3배이고, CPU에서 잰 값은 1.6배였습니다.

실행 script는 `KAI_CKPT=1`이 기본이고, 미세조정의 batch를 따로 둡니다(`RO_BATCH`, 기본 4). 고친 뒤 새 clone에서 `SMOKE=1 DEVICE=cpu`를 다시 돌렸고, 미세조정 단계가 5.4 GB 한도 안에서 통과했습니다(고치기 전에는 6.8 GB에서 멈췄습니다).

## 5. GPU에서 돌리는 법

git을 받는 것부터 학습 시험까지입니다. 새로 받은 clone에서 설치 목록, 명령의 문법, `SMOKE=1`의 전 단계를 CPU로 확인했습니다. **GPU에서는 돌려 보지 못했습니다.**

```bash
# 1. 코드와 환경
git clone https://github.com/nayehyeon61-glitch/physical_guide_flowmatching.git
cd physical_guide_flowmatching
git checkout claude/project-thread-wjjayb
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # nvidia-smi의 CUDA에 맞춰 cu126 / cu128 / cu130
pip install -r requirements.txt
pip install git+https://github.com/patrick-kidger/torchcubicspline.git  # requirements.txt에 없습니다
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # True가 나와야 합니다

# 2. 첫 점검 (몇 분): 해마다 24일(약 0.6 GB)만 내려받아 모든 단계를 작은 망으로 몇 iteration씩
SMOKE=1 bash scripts/run_kai_plus.sh

# 3. 원래 크기의 짧은 학습 시험: 실제 자료(2012~2017, 처음 한 번 내려받음), KAI 200 / 미세조정 20 / core + 40 iteration
KAI_ITERS=200 RO_ITERS=20 CORE_ITERS=40 LOGK=50 LOGC=10 OUT=runs/kaiplus_1p5_test bash scripts/run_kai_plus.sh
grep -h '"it"' runs/kaiplus_1p5_test/{kai,kai_ro,core_plus}.log   # sec = 누적 초, gpu_gb = 최대 GPU 메모리

# 4. 본 학습: KAI 20,000 / 미세조정 1,000 / core + 3,000 iteration, 채점까지
nohup bash scripts/run_kai_plus.sh > runs/kaiplus_1p5.log 2>&1 &
tail -f runs/kaiplus_1p5.log
```

- 2번이 통과하면 장치 처리(GPU로 옮기는 곳)와 설치는 맞습니다. 결과는 `runs/kaiplus_1p5_smoke/`에 따로 쌓이고, 학습되지 않은 망이라 점수는 의미가 없습니다.
- 3번은 본 학습과 같은 망, 같은 batch, 같은 자료입니다. log의 `sec`에서 iteration당 시간을, `gpu_gb`에서 메모리를 읽어 본 학습의 시간을 계산할 수 있습니다. 자료는 `data/era5_1p5/`에 한 번만 받고 4번이 그대로 씁니다(1.5° 일평균 한 해가 약 2.8 GB).
- 같은 명령을 다시 실행하면 끝난 단계는 건너뛰고 끊긴 학습은 이어서 합니다. 실행 전에 명령만 보려면 앞에 `DRYRUN=1`을 붙입니다.
- 가지고 계신 0.25° 시간별 archive에서 자료를 만들려면 `SRC=local ARCHIVE=/lustre/home/mahmed/ERA5/hourly`를 붙입니다(실제 archive로는 확인하지 못했습니다). 일평균이 이미 있으면 `SRC=none DATA=<kaiday가 있는 폴더>`입니다.

기본값: 학습 2012~15, 검증 2016, 시험 2017(5.625° 시험과 같은 해), KAI는 원래 크기 (3, 3, 15, 3) / (48, 96, 192, 288). 결과는 `runs/kaiplus_1p5/`에 쌓입니다: `score.log`, `score_plus.json`, `breakdown/breakdown_plus.txt`, `variable_table_plus.md`, 단계별 log `kai.log`, `kai_ro.log`, `core_plus.log`.

| 환경 변수 | 기본 | 뜻 |
| --- | --- | --- |
| `GRID` | 1.5 | 1.40625도 됩니다(128 × 256, 극 행 없음, 5.625° 한 칸 = 4 × 4칸. `SRC=local`만) |
| `KAI_ITERS`, `KAI_BATCH` | 20000, 8 | KAI 학습 |
| `RO_ITERS`, `RO_BATCH` | 1000, 4 | rollout 미세조정(2~3일의 activation을 저장하므로 batch를 따로 둠) |
| `KAI_CKPT` | 1 | KAI의 블록 단위 checkpoint. 0이면 빠르지만 메모리가 약 3~4배 |
| `SUBSTEPS` | 4 | core +의 하루 Heun step 수 |
| `PLUS_HIDDEN`, `PLUS_BLOCKS` | 48, "2 2" | core +의 폭과 깊이 |
| `CORE_BATCH`, `CORE_ITERS`, `CORE_MEM` | 8, 3000, 8 | core +의 batch, iteration, sequence 길이(일) |
| `GFEAT` | 1 | 0이면 물리항 없이 GRU + MZ만 |
| `LOGK`, `LOGC` | 500, 250 | KAI / core 학습의 log 간격(iteration). log마다 검증 loss를 계산 |
| `TRAIN_YEARS` | "2012 2013 2014 2015" | 학습 연도 |

메모리의 어림입니다. 4.2절과 4.4절의 CPU 측정에서 계산한 값이고, GPU에서 잰 값이 아닙니다.

| 단계 | 설정 | 학습 중 activation 메모리 |
| --- | --- | --- |
| KAI 학습 | batch 8, checkpoint (기본) | 약 12~16 GB |
| KAI 학습 | batch 8, `KAI_CKPT=0` | 약 48 GB |
| 미세조정 (3일 rollout) | batch 4, checkpoint (기본) | 약 12~15 GB |
| 미세조정 (3일 rollout) | batch 4, `KAI_CKPT=0` | 약 72 GB |
| core + | hidden 48, 블록 (2, 2), batch 8 (기본) | 약 13 GB (표본당 1.6 GB) |
| core + | hidden 96, 블록 (2, 2, 2), batch 8 | 약 25 GB (표본당 3.1 GB) |

기본값은 24 GB GPU에 들어가도록 잡았습니다. 메모리 오류가 나면 그 단계의 batch(`KAI_BATCH`, `RO_BATCH`, `CORE_BATCH`)를 절반으로 줄여 같은 명령을 다시 실행하면 됩니다. 80 GB GPU라면 `KAI_CKPT=0 RO_BATCH=2`가 iteration당 더 빠릅니다. core +의 sequence 길이(`CORE_MEM`)는 메모리를 거의 늘리지 않습니다(하루에 표본당 약 0.07 GB).

GPU에서의 iteration당 시간은 재지 못했습니다(CPU 2 thread에서는 core + batch 1에 약 40초였습니다). 3번의 log에서 읽으시면 됩니다.

## 6. 한계와 다음에 확인할 것

1. **GPU에서 실행해 보지 못했습니다.** 장치 처리는 GPU에서 돌린 적이 있는 `scripts/run_step16_dynpinn.py`의 방식을 따랐지만, 이 코드 자체는 CPU에서만 확인했습니다. 먼저 `SMOKE=1 bash scripts/run_kai_plus.sh`를 돌려 보시기를 권합니다. 장치가 섞이는 오류가 있다면 여기서 바로 드러납니다. 메모리의 어림(5절)도 CPU에서 작은 격자로 잰 값을 환산한 것이라, 원래 크기의 첫 log(`gpu_gb`)로 확인해야 합니다.
2. **substep 4는 계산에서 나온 설계값이고, 성능으로 확인한 값이 아닙니다.** `SUBSTEPS=1`, `2`, `4`를 같은 조건으로 학습해서 비교해야 합니다. 4가 2보다 낫지 않으면 2가 시간이 절반입니다.
3. **망의 크기.** 격자점이 14배인데 기본 망은 5.625°와 같습니다. 선행 1~3일의 작은 규모(k10 이상)에서 모자랄 수 있습니다. `PLUS_HIDDEN=96 PLUS_BLOCKS="2 2 2"`와 비교가 필요합니다.
4. **극.** 극 행의 물리항은 0이고, 85° 바깥의 G는 clamp만 되어 들어갑니다. 고위도의 오차를 따로 봐야 합니다. `eval_kai_breakdown.py`의 지역 분해에 극 지역(70° 바깥)을 더하는 것이 다음 일입니다.
5. **1.5°와 1.40625°.** 1.5°는 KAI가 원래 쓰는 격자입니다. 1.40625°는 5.625° 한 칸에 4 × 4칸이 정확히 들어맞아 5.625° 결과와 비교하기 쉽지만 KAI의 격자가 아닙니다. 기본은 1.5°이고, 두 격자 모두 같은 코드로 돕니다.
6. **5.625° 결과와의 비교.** 1.5° 채점은 1.5° 격자에서 합니다. 5.625°의 수치와 나란히 놓으려면 예측을 5.625°로 평균한 뒤 채점하는 단계가 필요하고, 아직 없습니다.

## 재현 (이 문서의 시험)

```bash
# 5.625°: 바꾸기 전 코드와 같은지 (core 학습 4 iteration)
python scripts/run_kai_tendency.py --out runs/eq_new --arm core --core-norm gn --core-gfeat --core-mem 8 --core-mz \
    --iters 4 --log-every 2 --batch 2 --curriculum 1 2 --train-years 2015 --threads 2 [--lazy] [--core-ckpt]
# 1.5°: 해마다 24일을 내려받아 모든 단계를 몇 iteration씩 (CPU에서 약 25분)
SMOKE=1 DEVICE=cpu bash scripts/run_kai_plus.sh
# KAI 학습 step 하나의 메모리와 기울기 (4.4절): checkpoint 없음 / 있음, 경우마다 따로 실행
python scripts/kai_step_memory.py --grid 61 120 --ckpt 0 --save g0.pt
python scripts/kai_step_memory.py --grid 61 120 --ckpt 1 --save g1.pt
```
