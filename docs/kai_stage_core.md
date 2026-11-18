# KAI 내부에 tendency core 넣기: stage별 적용 (2026-11-18)

## 요청을 이렇게 읽었습니다

소유자 요청(2026-11-18): "KAI와 tendency core는 지금 blending하는 구조다. 이번에는 KAI 내부에 tendency core를 삽입해서 효과를 본다. 특히 stage별 적용." 참고 자료는 KAI-α 구조 연구 노트(PDF)입니다.

- **지금의 연결**은 두 모델의 끝난 출력을 섞습니다. `blend`는 두 예보를 선행일 × 채널 가중으로 섞고, `loop`은 1일 가중을 매일 적용해 되먹입니다. 가중은 검증 연도에서 최소제곱으로 정한 고정값이고 학습되지 않습니다. core는 KAI의 계산에 들어가지 않습니다.
- **이번 구조**는 core의 경향을 KAI의 네 stage 각각에 넣고, 출력에서 섞는 가중도 KAI가 자기 특징에서 직접 계산하게 합니다.
- core는 entropy 두 core(2단 예보)가 아니라 그 전의 tendency core 기준입니다. 코드는 core의 종류를 가리지 않습니다(plain core, 기억 core +, 다중 규모 core 모두 `--core-tag`로 지정).
- stage에 무엇을 넣는가는 세 가지로 읽을 수 있어 소유자에게 물었고(하루 경향 / Heun substep별 경향 / core 내부 feature), 답을 기다리는 동안 첫째(하루 경향)로 진행했습니다. 나머지 둘은 5절에 적었습니다.

## 1. 구조

PDF의 KAI-α: stem → stage 1(48채널, 블록 3) → stage 2(96, 3) → stage 3(192, 15) → stage 4(288, 3) → head. 네 stage 모두 같은 격자(1.5°에서 121 × 240)이고 stage 사이의 전환층은 채널만 바꿉니다.

```
x ──► core(얼림) ──► c(x) = core의 하루 예보,  τ = (c(x) − x) / s      s = 채널별 하루 증분의 표준편차

x ─► stem ─(+P1 τ)─► stage 1 ─► 전환 ─(+P2 τ)─► stage 2 ─► 전환 ─(+P3 τ)─► stage 3 ─► 전환 ─(+P4 τ)─► stage 4 ─► head ─► k(x)
                                                                                                    │
                                                                         g(x) = clamp(g0 + G·h4, 0, 1)   (h4 = stage 4의 특징)
출력:  stg   x(t+1) = k(x)
       stgg  x(t+1) = k(x) + g(x) ⊙ ( c(x) − k(x) )
```

| 부품 | 내용 | 시작값 |
| --- | --- | --- |
| `P1..P4` | stage마다 따로 있는 1×1 conv, 67 → stage의 채널 수. τ를 그 stage의 **입구**(전환층 뒤, 블록 앞)에 더합니다. 그 stage의 블록 전부가 core의 경향을 처리합니다 | 0 (처음에는 KAI 그대로) |
| `g0` | 채널별 기본 가중 67개 | loop의 고정 가중(`--gate-init`) |
| `G` | stage 4 특징에서 채널 × 격자점별 가중의 변화를 내는 1×1 conv | 0 |

- `stg`: stage 네 곳에만 넣고 출력은 KAI의 것 그대로입니다. 시작 상태 = 초기화에 쓴 KAI.
- `stgg`: stage 네 곳 + 학습되는 출력 혼합입니다. 시작 상태 = **기존 loop 모델과 정확히 같은 모델**이고, 학습이 거기서 벗어난 만큼이 이 구조의 효과입니다.
- `outg`(대조): 학습되는 출력 혼합만 있고 stage 삽입은 없습니다. `stgg`와의 차이가 stage 삽입의 몫입니다.
- 가중 g는 값만 [0, 1]로 자르고 기울기는 그대로 둡니다(straight-through). loop의 가중은 여러 채널에서 0이나 1에 붙어 있고 `clamp`의 기울기는 경계에서 0이라, 그대로 두면 그 채널의 가중이 움직이지 않습니다.
- 기억 core(core +)도 KAI 안에서 학습에 쓸 수 있게 했습니다. batch마다 초기 시각 전 `mem`일의 자료로 기억을 채우고, rollout에서는 모델 자신의 상태로 기억을 이어 갑니다(채점의 loop과 같은 방식). 전에는 기억 core를 KAI 학습 안에 넣을 수 없었습니다.

### 전에 실패한 안쪽 삽입과 다른 점

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
| `stg` | stage 네 곳 | 안쪽 삽입만으로 얼마나 되는가 (`stg : kai_ro`) |
| `outg` | 출력, 가중을 KAI가 계산 | 가중을 학습하는 것의 몫 (`outg : loop`) |
| `stgg` | stage 네 곳 + 출력, 가중을 KAI가 계산 | 전체 (`stgg : loop`), stage의 몫 (`stgg : outg`) |

채점에서 함께 나오는 진단:

- `<arm>~notau`: 같은 rollout에서 stage에 넣는 τ를 모두 0으로. 모델이 stage 삽입에 얼마나 기대는가.
- `<arm>~ns1` … `~ns4`: stage 하나의 τ만 끔. **stage별 기여**입니다.
- `<arm>~open`(`stg`): core를 자기 궤적에서만 돌리고 τ만 KAI에 넣음. rollout의 손실이 KAI ↔ core 되먹임 때문인지.
- `score.json`의 `stage_ratio`(각 stage가 받은 삽입의 크기 / 그 stage 특징의 크기, RMS)와 `gates_used`(실제로 쓴 채널별 가중의 평균), `gates`(기본 가중 g0).
- `breakdown_stage/`: 모든 이득을 정보(패턴) 항과 진폭 항으로 나눈 표(`scripts/eval_kai_breakdown.py`). 긴 선행일의 이득이 매끄러워진 몫인지 가릅니다.

부호는 이 문서 전체에서 이득입니다(+ = 좋아짐). `score` log의 `pairs` 줄만 반대 부호(a의 b 대비 nRMSE 변화, 음수가 좋아짐)입니다.

## 3. 명령

```bash
# 1.5° (GPU 서버, + 버전을 돌린 폴더: runs/kaiplus_1p5_daily 에 kai.pt, kai_ro.pt, core_plus.pt 가 있어야 합니다)
git fetch origin && git checkout claude/project-thread-c97r6n && git pull --ff-only origin claude/project-thread-c97r6n
bash scripts/run_kai_stage.sh                                  # core_plus, 새 모델 stg / stgg / outg 학습 후 채점
C=core_plus_u64 SFX=_u64 bash scripts/run_kai_stage.sh         # 다른 core (다중 규모 u64): tag는 stg_u64, stgg_u64, outg_u64
ARMS="stgg" bash scripts/run_kai_stage.sh                      # 새 모델 일부만
CORE_GRAD=0 bash scripts/run_kai_stage.sh                      # core를 지나는 기울기 없이 (메모리, 시간 절약)
DRYRUN=1 bash scripts/run_kai_stage.sh                         # 명령만 출력

# 5.625° (CPU, 이 문서의 수치)
python scripts/wb2_to_kai.py --out data/era5_5p625 --daily-mean --no-climode --years 2012 2013 2014 2015 2016 2017
R="python scripts/run_kai_tendency.py --out runs/kaitend"
$R --arm kai --tag kai --log-every 250
$R --arm core --core-norm gn --iters 3000 --log-every 250 --tag core_gn
GRID=5.625 C=core_gn bash scripts/run_kai_stage.sh
```

- driver는 단계마다 결과 파일이 있으면 건너뛰고, 끊긴 학습은 이어서 합니다(`--resume`).
- 0단계는 KAI와 core의 선행일별 혼합 가중을 검증 연도에서 구해 `score_stage0.json`에 둡니다. `stgg`와 `outg`의 g0가 여기서 옵니다(`--gate-init score_stage0.json:blend_<core>`).
- 1.5°의 `kai_ro.pt`가 이미 있으면 그대로 씁니다. 새 모델의 미세조정 설정(`RO_ITERS=1000`, `RO_BATCH=4`, `RO_LR=3e-4`)은 그 `kai_ro`의 것과 같습니다.
- 결과: `score_stage.json`, `stage.log`(모델별 nRMSE와 ACC, 쌍별 차이), `breakdown_stage/breakdown_stage.txt`.
- **1.5°에서는 아직 돌리지 않았습니다.** 메모리와 시간의 어림: 새 모델 하나의 학습은 `kai_ro`의 학습(1,000 iteration, 0.35시간, 10 GB)에 core의 순전파와 역전파가 더해집니다. core를 지나는 기울기는 substep 단위 checkpoint로 계산합니다(`--core-ckpt`). 메모리가 모자라면 `CORE_GRAD=0`이나 `RO_BATCH=2`를 주세요.

## 4. 결과 (5.625°, CPU)

실험이 돌고 있습니다. 끝나면 이 절에 수치를 넣습니다.

## 5. 하지 않은 것, 다음 후보

- **substep별 경향을 stage에 대응**: core +는 1.5°에서 하루를 Heun 4 substep으로 적분합니다. substep k까지 적분한 경향 (x_k − x) / s를 stage k에 넣으면 KAI의 깊이가 core의 시간 적분과 나란히 갑니다. 5.625°의 core는 하루 1 substep이라 이 실험에서는 하루 경향과 같아져서 넣지 않았습니다.
- **core 내부 feature를 stage에**: 다중 규모 core(u64)는 1.5 / 3 / 6° 세 격자의 특징(64 / 128 / 192채널)을 갖습니다. 출력 경향 대신 이 특징을 stage 1~4에 넣는 방법입니다. core의 구조에 묶이고 plain core에는 쓸 수 없습니다.
- **core를 함께 학습**: 지금은 core를 얼려 둡니다. KAI 안에서 core까지 미세조정하면 core 단독 예보와의 비교가 깨지므로 넣지 않았습니다.
