# KAI 내부에 tendency core 넣기: stage별 적용 (2026-11-18)

## 요청을 이렇게 읽었습니다

소유자 요청(2026-11-18, 시각은 KST):

1. "KAI와 tendency core는 지금 blending하는 구조다. 이번에는 KAI 내부에 tendency core를 삽입해서 효과를 본다. 특히 stage별 적용." 참고 자료는 KAI-α 구조 연구 노트(PDF)입니다.
2. (22:50) "stage마다 들어가는데 현재 48 이후 96, 192..의 구조다. 이때 분할된 변수를 다시 합쳐 우리의 tendency core에 넣고, 다시 분할하여서 각 stage마다 blend되는 구조로."
3. (23:14) "나눠지는 변수를 다시 합치는 거니 올바르게 합쳐 달라. 그래야 tendency core의 효과가 나온다."

- **지금의 연결**은 두 모델의 끝난 출력을 섞습니다. `blend`는 두 예보를 선행일 × 채널 가중으로 섞고, `loop`은 1일 가중을 매일 적용해 되먹입니다. 가중은 검증 연도에서 최소제곱으로 정한 고정값이고 학습되지 않습니다. core는 KAI의 계산에 들어가지 않습니다.
- **이번 구조**는 KAI의 네 stage 각각에서 그 stage의 채널을 변수 67개로 합치고, 그 상태를 tendency core에 넣고, core가 낸 경향을 다시 그 stage의 채널로 나눠 stage에 섞습니다(`stc`, `stcg`).
- core는 entropy 두 core(2단 예보)가 아니라 그 전의 tendency core 기준입니다. 코드는 core의 종류를 가리지 않습니다(plain core, 기억 core +, 다중 규모 core 모두 `--core-tag`로 지정).
- 처음에는 core가 입력 x에서 낸 하루 경향 하나를 네 stage에 똑같이 넣는 구조(`stg`, `stgg`)를 만들었습니다. 소유자의 2번 말로 이것은 요청한 구조가 아님이 분명해졌고, 지금은 새 구조의 **대조**로 남겨 두었습니다. 둘의 차이가 "stage마다 합쳐서 core에 넣는 것"의 몫입니다.

## 1. 구조

PDF의 KAI-α: stem → stage 1(48채널, 블록 3) → stage 2(96, 3) → stage 3(192, 15) → stage 4(288, 3) → head. 네 stage 모두 같은 격자(1.5°에서 121 × 240)이고 stage 사이의 전환층은 채널만 바꿉니다.

```
stage i의 입구 (전환층 뒤, 블록 앞; 채널 수 48 / 96 / 192 / 288), 특징 h_i

  합치기   x_i = x + a_i · s · D_i(h_i)      변수 67개의 상태        a = (0, 1/4, 1/2, 3/4)
  core     τ_i = ( c(x_i) − x_i ) / s        얼린 tendency core가 x_i에서 낸 하루 경향
  분할     e_i = P_i(τ_i)                    67 → stage의 채널 수
  blend    h_i ← h_i + e_i                   그 뒤에 stage i의 블록들

x ─► stem ─[stage 1 입구]─► stage 1 ─► 전환 ─[stage 2 입구]─► stage 2 ─► 전환 ─[3]─► stage 3 ─► 전환 ─[4]─► stage 4 ─► head ─► k(x)

출력:  stc    x(t+1) = k(x)
       stcg   x(t+1) = k(x) + g(x) ⊙ ( c(x) − k(x) ),   g(x) = clamp(g0 + G·h4, 0, 1)
```

x는 오늘의 상태, s는 채널별 하루 변화의 표준편차, c(·)는 core의 하루 예보입니다.

| 부품 | 내용 | 시작값 | 무엇으로 학습되는가 |
| --- | --- | --- | --- |
| `D_i` (합치기) | stage의 채널 → 변수 67개. KAI가 자기 채널을 변수로 합치는 방식(head: 3×3 conv, GELU, 1×1 conv)과 같은 형태. stage 2, 3, 4에 하나씩 | 마지막 층 0 | **자료와의 오차로만** (1.2절) |
| `P_i` (분할) | 변수 67개 → stage의 채널. 1×1 conv, stage마다 하나 | 0 | 예보 loss |
| `g0`, `G` (출력 혼합, `stcg`) | 채널별 기본 가중 67개와, stage 4 특징에서 채널 × 격자점별 변화를 내는 1×1 conv | loop의 고정 가중, 0 | 예보 loss |

- 새 층이 모두 0에서 시작하므로 `stc`의 시작 상태는 초기화에 쓴 KAI와 같고, `stcg`의 시작 상태는 **기존 loop 모델과 정확히 같습니다**. 학습이 거기서 벗어난 만큼이 이 구조의 효과입니다.
- blend는 더하기로 썼습니다. `c(x_i) − x_i`는 stage의 상태와 core 예보를 섞을 때 가중이 곱해지는 바로 그 양이고, 가중은 `P_i`에 들어 있습니다. 실제로 섞인 크기는 채점의 `stage_ratio`(넣은 것의 크기 / stage 특징의 크기)로 봅니다.
- core는 얼려 둡니다. rollout 학습에서 다음 날 loss의 기울기는 core를 지나 앞날의 상태로 갑니다(`CORE_GRAD=1`).
- 기억 core(core +)도 stage 안에서 쓸 수 있습니다. 기억은 하루에 한 번, 궤적의 상태로만 갱신되고 stage에서의 호출은 그 기억을 읽기만 합니다.

### 1.2 올바르게 합치기

core는 물리 변수 67개의 상태에서 하루 경향을 계산하도록 학습된 모델입니다. 합친 것이 실제 상태가 아니면 core가 내는 것은 경향이 아닙니다. 그래서 먼저 stage의 채널에서 변수가 얼마나 정확히 되돌려지는지 쟀습니다(학습된 5.625° `kai`, 합치는 층만 학습, 시험은 2016년; 오차는 채널별 RMSE를 하루 변화의 표준편차 s로 나눠 65개 채널에서 평균).

| 위치 | 채널 | 채널만으로 x(t)를 복원: 1×1 conv | KAI의 head 형태 | 입력을 두고 하루 변화를 추정, x + s·D(h) → x(t+1): 1×1 conv | KAI의 head 형태 |
| --- | ---: | ---: | ---: | ---: | ---: |
| stage 1 입구 | 24 | 1.20 | 1.01 | 0.99 | 0.86 |
| stage 2 입구 | 48 | 1.37 | 0.86 | 0.98 | 0.84 |
| stage 3 입구 | 64 | 1.24 | 0.84 | 0.97 | 0.80 |
| stage 4 입구 | 96 | 1.18 | 0.74 | 0.95 | 0.71 |
| stage 4 뒤 | 96 | 1.04 | 0.73 | 0.90 | 0.69 |

기준: 오늘 상태를 내일 상태로 쓰면(persistence) 1.03, 이 KAI의 하루 예보는 0.74입니다. 1×1 conv 열은 격자점별 최소제곱(2015년 120일로 맞춤), head 형태 열은 2012~2015년으로 1,500 iteration 학습한 값입니다.

읽는 법:

1. **채널만으로는 변수를 되돌릴 수 없습니다.** 어느 stage에서든 복원 오차가 하루 변화의 0.7~1.4배입니다. KAI의 stem은 변수 67개를 더 적은 채널(48, 이 실험의 크기에서는 24)로 줄이고 격자점마다 정규화하므로 분할이 되돌릴 수 있는 변환이 아닙니다. 이 복원을 core에 넣으면 core는 하루 변화만 한 크기의 복원 오차에서 경향을 계산하게 됩니다.
2. **변수 자체는 입력에 그대로 있습니다.** 올바른 합치기는 입력 x를 그대로 두고, stage의 채널이 거기에 더한 것만 채널에서 꺼내는 것입니다. stage의 채널이 더한 것은 하루 변화의 추정이고, head 형태의 층으로 꺼내면 stage가 깊어질수록 정확해집니다(0.86 → 0.71, 아무것도 더하지 않으면 1.03).
3. 그래서 `x_i = x + a_i · s · D_i(h_i)` 입니다. `D_i`는 stage i가 계산한 하루 변화 (x(t+1) − x(t)) / s의 추정이고, `a_i`는 stage가 하루의 어디에 서 있는지입니다.

| stage | a_i | 합친 상태가 나타내는 것 | 그 상태의 오차 (위 표에서) | 입력을 그대로 쓸 때 | 채널만 합칠 때 |
| --- | ---: | --- | ---: | ---: | ---: |
| 1 | 0 | 오늘 상태 x(t). stem이 나눈 것의 정확한 합 | 0 | 0 | 1.01 |
| 2 | 1/4 | 하루의 1/4 지점 | 0.21 | 0.26 | 0.86 |
| 3 | 1/2 | 하루의 1/2 지점 | 0.40 | 0.51 | 0.84 |
| 4 | 3/4 | 하루의 3/4 지점 | 0.53 | 0.77 | 0.74 |

`D_i`가 학습되는 방식도 "올바르게"의 일부입니다.

- `D_i`는 **자료에 대한 자기 오차로만** 학습됩니다. 목표는 다음 날의 실제 상태이고(`x + s·D_i(h_i)`와 x(t+1)의 상대 L2, 예보 loss와 같은 식), rollout의 모든 날에 계산합니다.
- `D_i`의 입력(stage의 특징)과 출력(합친 상태)은 예보 loss의 계산 그래프에서 끊었습니다. 예보 loss가 core를 거쳐 `D_i`를 움직일 수 있으면, 합친 상태는 실제 상태가 아니라 "core에게서 쓸모 있는 답을 끌어내는 상태"로 흘러갈 수 있습니다. 거꾸로 `D_i`의 loss는 KAI의 가중을 건드리지 않으므로, KAI 본체가 받는 학습 신호는 다른 모든 모델과 같은 예보 loss 하나입니다.
- 채점은 시험 연도에서 stage마다 합친 상태의 오차를 다시 잽니다(`score.json`의 `stage_merge`: a_i, 합친 상태의 오차, 입력을 그대로 쓸 때의 오차).

정한 가정 하나: **stage의 깊이를 하루 안의 시간으로 읽었습니다**(a = 0, 1/4, 1/2, 3/4; `pgcfm/kai_tend.py`의 `MERGE_AT`). stage 1의 입구는 stem 바로 뒤라 오늘 상태이고, head는 내일 상태를 내므로 그 사이를 네 stage에 고르게 나눴습니다. core +가 1.5°에서 하루를 4 substep으로 적분하는 것과도 맞습니다. 다른 대응(예: 블록 수에 비례)은 `MERGE_AT`만 바꾸면 됩니다.

### 1.3 대조 구조

- `stg`: core가 **입력 x에서** 낸 경향 τ(x) 하나를 네 stage에 넣습니다(`P_i`만 있음). `stc`에서 a_i를 모두 0으로 둔 것과 같습니다. `stc : stg`가 stage마다 합쳐서 core에 넣는 것의 몫입니다.
- `stgg`: `stg` + 학습되는 출력 혼합. `stcg`의 대조입니다.
- `outg`: 학습되는 출력 혼합만 있고 stage에는 아무것도 넣지 않습니다. stage 삽입 전체의 대조입니다.
- 출력 혼합의 가중 g는 값만 [0, 1]로 자르고 기울기는 그대로 둡니다(straight-through). loop의 가중은 여러 채널에서 0이나 1에 붙어 있고 `clamp`의 기울기는 경계에서 0이라, 그대로 두면 그 채널의 가중이 움직이지 않습니다.

### 1.4 전에 실패한 안쪽 삽입과 다른 점

`docs/kai_tendency.md` 1절의 `in`, `s1`~`s4`는 τ를 **한 곳**에 넣고 KAI를 **처음부터 1일 loss로** 학습했습니다. 결과는 "1일은 조금 좋아지지만 rollout에서 무너진다, 망이 τ에 의존하게 되고 core의 오차가 KAI 안에서 증폭된다"였고 `s4`는 발산했습니다. 이번에는 세 가지를 바꿨습니다.

1. 학습된 KAI에서 출발하고 새 경로는 0에서 시작합니다. 처음 상태가 KAI(또는 loop)이므로 τ 없이는 예보를 못 하는 망이 되지 않습니다.
2. 자기 rollout(2~3일)으로 학습합니다. KAI → core → KAI의 되먹임이 loss 안에 있습니다.
3. stage의 입구에 넣습니다. head 바로 앞에 선형으로 들어가는 길(`s4`)이 없습니다.

## 2. 비교 설계

모든 모델은 **같은 `kai`에서 출발한 같은 미세조정**(2~3일 rollout, 같은 iteration, 같은 학습률, 같은 seed)이고, 다른 것은 core가 들어가는 자리뿐입니다.

| 이름 | core의 자리 | 묻는 것 |
| --- | --- | --- |
| `kai_ro` | 없음 | 미세조정 자체의 몫 (대조) |
| `loop_<core>@kai_ro`, `blend_<core>@kai_ro` | 출력에서 고정 가중으로 섞음 (기존 구조) | 기준선 |
| `stc` | **stage 네 곳: 합치기 → core → 분할 → blend** | 새 구조만으로 얼마나 되는가 (`stc : kai_ro`) |
| `stcg` | 위 + 출력, 가중을 KAI가 계산 | 전체 (`stcg : loop`) |
| `stg`, `stgg` | stage 네 곳에 τ(x) 하나 (+ 출력) | stage마다 합쳐서 core에 넣는 것의 몫 (`stc : stg`, `stcg : stgg`) |
| `outg` | 출력, 가중을 KAI가 계산 | stage 삽입의 몫 (`stcg : outg`), 가중 학습의 몫 (`outg : loop`) |

채점에서 함께 나오는 진단:

- `<arm>~notau`: 같은 rollout에서 stage에 넣는 경향을 모두 0으로. 모델이 stage 삽입에 얼마나 기대는가.
- `<arm>~ns1` … `~ns4`: stage 하나의 경향만 끔. **stage별 기여**입니다.
- `score.json`의 `stage_merge`(1.2절), `stage_ratio`(각 stage가 받은 것의 크기 / 그 stage 특징의 크기, RMS), `gates_used`(실제로 쓴 채널별 출력 가중의 평균), `gates`(기본 가중 g0).
- `breakdown_stage/`: 모든 이득을 정보(패턴) 항과 진폭 항으로 나눈 표(`scripts/eval_kai_breakdown.py`). 긴 선행일의 이득이 매끄러워진 몫인지 가릅니다.

부호는 이 문서 전체에서 이득입니다(+ = 좋아짐). `score` log의 `pairs` 줄만 반대 부호(a의 b 대비 nRMSE 변화, 음수가 좋아짐)입니다.

## 3. 명령

```bash
# 1.5° (GPU 서버, + 버전을 돌린 폴더: runs/kaiplus_1p5_daily 에 kai.pt, kai_ro.pt, core_plus.pt 가 있어야 합니다)
git fetch origin && git checkout claude/project-thread-c97r6n && git pull --ff-only origin claude/project-thread-c97r6n
bash scripts/run_kai_stage.sh                                  # core_plus: stcg, stc, stgg, stg, outg 학습 후 채점
ARMS="stcg stc" bash scripts/run_kai_stage.sh                  # 새 구조만 (대조 없이 loop, kai_ro 와 비교)
C=core_plus_u64 SFX=_u64 bash scripts/run_kai_stage.sh         # 다른 core (다중 규모 u64): tag는 stcg_u64, ...
CORE_GRAD=0 bash scripts/run_kai_stage.sh                      # core를 지나는 기울기 없이 (메모리, 시간 절약)
DRYRUN=1 bash scripts/run_kai_stage.sh                         # 명령만 출력

# GPU가 여러 개일 때: 시작 가중과 kai_ro 를 먼저 만들고, 모델을 GPU마다 나눠 학습한 뒤, 한 번에 채점
SCORE=0 ARMS=" " bash scripts/run_kai_stage.sh
CUDA_VISIBLE_DEVICES=0 SCORE=0 ARMS="stcg stg outg" bash scripts/run_kai_stage.sh &
CUDA_VISIBLE_DEVICES=1 SCORE=0 ARMS="stc stgg" bash scripts/run_kai_stage.sh &
wait; bash scripts/run_kai_stage.sh

# 5.625° (CPU, 이 문서의 수치)
python scripts/wb2_to_kai.py --out data/era5_5p625 --daily-mean --no-climode --years 2012 2013 2014 2015 2016 2017
R="python scripts/run_kai_tendency.py --out runs/kaitend"
$R --arm kai --tag kai --log-every 250
$R --arm core --core-norm gn --iters 3000 --log-every 250 --tag core_gn
GRID=5.625 C=core_gn bash scripts/run_kai_stage.sh
```

- driver는 단계마다 결과 파일이 있으면 건너뛰고, 끊긴 학습은 이어서 합니다(`--resume`).
- 0단계는 KAI와 core의 선행일별 혼합 가중을 검증 연도에서 구해 `score_stage0.json`에 둡니다. 출력 혼합의 g0가 여기서 옵니다(`--gate-init score_stage0.json:blend_<core>`).
- 1.5°의 `kai_ro.pt`가 이미 있으면 그대로 씁니다. 새 모델의 미세조정 설정(`RO_ITERS=1000`, `RO_BATCH=4`, `RO_LR=3e-4`)은 그 `kai_ro`의 것과 같습니다.
- 결과: `score_stage.json`, `stage.log`(모델별 nRMSE와 ACC, 쌍별 차이), `breakdown_stage/breakdown_stage.txt`. 학습 log의 `merge`는 합치는 층의 loss입니다.
- **1.5°에서는 아직 돌리지 않았습니다.** 메모리와 시간의 어림: `stc` / `stcg`는 하루에 core를 네 번 부릅니다(x에서 한 번, stage 2, 3, 4의 상태에서 한 번씩). `kai_ro`의 학습(1,000 iteration, 0.35시간, 10 GB)에 core의 순전파와 역전파 네 번이 더해집니다. core를 지나는 기울기는 substep 단위 checkpoint로 계산합니다(`--core-ckpt`). 메모리가 모자라면 `CORE_GRAD=0`이나 `RO_BATCH=2`를 주세요.

## 4. 결과 (5.625°, CPU)

실험이 돌고 있습니다. 끝나면 이 절에 수치를 넣습니다.

## 5. 하지 않은 것, 다음 후보

- **합치는 층을 예보 loss로도 학습**: 처음 만든 판(commit bb0439f)은 `D_i`를 0에서 시작해 예보 loss의 기울기(core를 거쳐 오는)로만 학습했습니다. 합친 상태가 실제 상태라는 보장이 없어 1.2절의 방식으로 바꿨고, 그 판의 실험 수치는 없습니다.
- **합치는 층의 loss로 KAI의 특징도 학습**: 지금은 `D_i`가 KAI의 특징을 읽기만 합니다. 그 loss를 KAI 본체까지 보내면 stage의 채널이 "변수를 나눈 것"에 더 가까워져 합친 상태가 더 정확해질 수 있습니다. 대신 KAI의 학습 신호가 다른 모델과 달라져서 core의 몫만 따로 볼 수 없게 되므로 이번에는 넣지 않았습니다.
- **stage와 하루 안의 시간의 다른 대응**: `MERGE_AT`. core의 걸음을 stage에 맞춰 1/4일로 줄이는 것(합친 상태에서 1/4일만 적분)도 같은 자리의 후보입니다.
- **core 내부 feature를 stage에**: 다중 규모 core(u64)는 1.5 / 3 / 6° 세 격자의 특징(64 / 128 / 192채널)을 갖습니다. 출력 경향 대신 이 특징을 stage 1~4에 넣는 방법입니다. core의 구조에 묶이고 plain core에는 쓸 수 없습니다.
- **core를 함께 학습**: 지금은 core를 얼려 둡니다. KAI 안에서 core까지 미세조정하면 core 단독 예보와의 비교가 깨지므로 넣지 않았습니다.
