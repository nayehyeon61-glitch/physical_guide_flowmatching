# 15단계: 대용량 데이터 학습 체계 (broad run 읽기와 학습 루프 정비)

## 요청을 이렇게 읽었습니다

소유자가 2026-10-27 01:55에 올린 커밋 2dd09f2(`docs/step13_broad/` 보고서 PDF와 그림 두 장, `scripts/report_step13_broad.py`, `scripts/plot_step13_broad.py`)는 1979–2015년 37년을 GPU에서 학습한 broad run의 결과 정리입니다. 요청은 (1) 그 결과에서 현재의 문제와 해결책을 정리하고, (2) 대용량 데이터 학습이 체계적으로 돌도록 학습 코드를 고치는 것입니다. 이 문서가 (1)이고, 코드가 (2)입니다. 새 예보 결과는 없으므로 lead-time 그림은 없고, 대신 학습 곡선 그림(`fig_training.png`)이 보고서에 추가됩니다.

중요한 사실 하나: 커밋 2dd09f2에는 보고서 스크립트만 있고, broad run을 GPU에서 돌리기 위해 소유자 쪽에서 고친 학습 코드(장치 옵션 등)는 git에 없습니다. 리포지토리의 13단계 스크립트는 2dd09f2 시점에 CPU 전용이었습니다. 따라서 보고서의 숫자는 현재 리포지토리만으로는 재현되지 않습니다. 이 단계에서 장치 옵션을 리포지토리 쪽에 넣었으니, 소유자 쪽 수정과 다른 점이 있으면 알려 주셔야 합니다(특히 bf16 사용 여부, batch 구성, 검증 유무).

## 1. broad run 결과 읽기

보고서(`docs/step13_broad/step13_broad_report.pdf`)와 `score_best.json`에서 읽은 핵심 숫자입니다. 테스트는 ClimODE와 같은 360 IC(2017–2018), 8 member.

| 모델 (z500 RMSE 24 / 72 / 144 h, m²/s²) | 24 | 72 | 144 | t850 144 h (K) |
| --- | --- | --- | --- | --- |
| ClimODE 논문 | 193 | 479 | 784 | 3.4 |
| 13단계 regm_band, 학습 2012–2015 (4년) | 244 | 522 | 778 | 3.45 |
| **broad regm_band, 학습 1979–2015 (37년)** | **170** | **423** | **741** | **3.21** |
| broad 회귀 앙상블 (seqregress + PINN) | 170 | 427 | 789 | - |
| broad FM 단독 (seqfm_dct raw) | 371 | - | 935 | - |

- 학습 기간을 4년에서 37년으로 늘린 것이 이 프로젝트에서 가장 큰 단일 개선입니다(z500 24 h 244 → 170, 144 h 778 → 741). 모델 구조는 그대로입니다.
- 세 guide arm(band / flow / flowop)의 5변수 평균 fair CRPS는 0.1595 / 0.1585 / 0.1585로 0.6 % 안에서 같고, 모두 회귀 앙상블보다 전 변수·전 lead에서 유의하게 낮습니다(bootstrap CI 상한 < 0). z500 144 h CRPS는 0.0310 vs 0.0338(−8 %).
- FM 단독은 여전히 회귀보다 훨씬 나쁘고(z500 24 h 371 vs 170) SSR ≈ 2(과분산). 작은 스케일(k13–32) 파워는 guide 5.0, 회귀 11.5(1이 완벽).
- flow-dependence는 약합니다. spread–|오차| 사례 상관이 band 0.01, flow 0.08–0.11, flowop 0.04–0.07.
- FM 앙상블 위의 연산자 평균(operator_mean_only) z500 334 / 600 / 810은 회귀 평균 170 / 427 / 789에 단기에서 크게 뒤집니다. guide arm이 회귀 평균을 쓰는 이유입니다.
- day-6 하위 tercile BSS −0.17(음수).

### 결과 측면의 현재 문제

1. FM의 과분산과 작은 스케일 과대 파워를 guide(Γ)가 사후에 눌러 주고 있습니다. FM 자체가 좋아져야 guide의 몫이 줄어듭니다.
2. 40000 it이라는 학습량은 검증 없이 정해졌습니다. 학습 곡선도, early stopping도, best checkpoint도 없었으므로 FM이 덜 학습된 것인지 과학습인지 알 수 없습니다. 37년 데이터에서 batch 16 × 40000 it = 640k 창은 전체 창(37년 × 1460 시작점 ≈ 54k)의 약 12 epoch입니다.
3. 보고서의 표 가운데 기본 모델, flow-dependence, operator ablation 표는 로그에서 손으로 옮긴 숫자였습니다. run을 다시 하면 보고서가 어긋납니다.
4. GPU용 코드가 git에 없으므로 broad run은 지금 리포지토리로 재현되지 않습니다.

## 2. 학습 파이프라인의 문제와 고친 것

| 문제 (2dd09f2 기준) | 영향 | 고친 것 |
| --- | --- | --- |
| `load_years`가 연도 전체를 RAM에 올림 (연 0.8 GB, 37년 ≈ 30 GB) | 15 GB 기계에서 broad 학습 불가, GPU 기계도 RAM 낭비 | `pgcfm/runs.py` `load_years(..., cache=DIR)`: 정규화한 연도를 float32 `.npy`로 한 번 만들고 memmap으로 읽음. `stats.npz` 지문이 다르면 거부 |
| 스크립트가 CPU 고정 (`torch.set_num_threads`, 장치 옵션 없음) | GPU 학습은 리포지토리 밖에서만 가능 | `--device cpu/cuda/cuda:N/auto`, `--amp`(cuda bf16 autocast). 잡음은 생성기(CPU)에서 뽑은 뒤 장치로 옮기므로 CPU와 GPU에서 같은 member가 나옴 (`pgcfm/seqdct.py`) |
| 반복수 고정, 검증 없음, best checkpoint 없음 | 학습량을 눈감고 정함 | `--val-every N --n-val-train M --val-members K --patience P --select best`: 고정 검증 창의 손실(FM은 선택적으로 4 member z500 RMSE)을 재고 `<tag>.pt.best`를 보관, 끝에 best를 선택. `<tag>.pt_metrics.csv`에 학습 곡선 |
| PINN과 seqfm에 각각 손으로 쓴 학습 루프, torch 생성기 상태는 체크포인트에 없음 | 재개하면 잡음 순서가 바뀜 | 공용 `pgcfm.runs.fit()` 하나. `.train` 체크포인트에 numpy 상태 + torch/cuda/생성기 상태 + 손실 이력 + best 기록 |
| 시간 예산 없음 | job 시간 제한이 있는 GPU 큐에서 죽으면 마지막 100 it 체크포인트에서 손으로 재시작 | `--max-hours H`: 넘으면 저장하고 종료 코드 75. `run_step13_best.sh`가 75를 받으면 그대로 멈추고, 같은 명령을 다시 돌리면 이어서 함. `FitConfig.max_steps`는 job array용 |
| 데이터 적재가 학습 스레드에서 직렬로 일어남 | GPU가 CPU 창 복사를 기다림 | `Prefetch`: 창 선택(난수)은 호출 스레드, 창 복사는 작업 스레드. 난수 순서는 prefetch 깊이와 무관하고, `close()`가 뽑아 둔 창만큼 난수를 되감아 학습 뒤의 다른 추정(회귀 오차 적합 등)도 prefetch 유무와 무관 |
| `run_step13_best.sh`에 학습 연도·반복·장치 손잡이 없음 | broad run을 스크립트 밖에서 손으로 조립 | `PROFILE=small/broad/smoke` + 환경 변수 덮어쓰기, 단계마다 `$OUT/manifest.jsonl`(명령, commit, 시작·끝, 종료 코드) |
| 보고서 표가 로그에서 손으로 옮겨짐 | 재실행 시 어긋남 | `scripts/report_step13_broad.py --runs DIR`: 모든 표를 json에서 읽음, 학습 기록 표와 학습 곡선 그림 추가, 한글 글꼴 후보 목록 |
| seqfm의 physics scale 추정 표본이 학습 뒤 난수 상태에서 나옴 | 학습을 했는지, 가중치를 읽었는지, 재개했는지에 따라 잔차 진단 값이 달라짐 | 전용 스트림 `default_rng(seed)`. 가중치를 읽어 생성하는 경로(배포된 재현 경로)와 비트 단위로 같음 |
| tercile 기후 연도가 학습 연도와 겹침 (2006–2015) | broad 프로파일에서 기후가 학습 데이터에서 나옴 | 데이터 단계가 학습 연도를 뺀 2006–2015만 기후용으로 받음 (겹치면 그대로; 약점 6 참조) |

고치지 않은 것은 6절에 있습니다.

## 3. 바뀐 코드

- `pgcfm/runs.py` (새 파일): `device_from`, `autocast`, `run_record`(commit, dirty, torch 버전, 장치 이름, 호스트, 스레드, 시작 시각), `YearArray` / `load_years` / `tendency_std`(memmap 연도, 청크 단위 경향 표준편차), `Prefetch`, `FitConfig`, `fit`.
  `fit`은 `loss_fn(batch) -> loss | (loss, {부분 손실})`를 받아 `.train`(재개용), `.best`(검증 최저), `_metrics.csv`(it, seconds, lr, train_loss, val_loss, 부분 손실)를 쓰고 `{hist, val, best_it, best_val, stopped, seconds, start_it}`를 돌려줍니다. `stopped`는 `iters` / `patience` / `budget`.
- `scripts/run_step13_pinn.py`, `scripts/run_step13_seqfm.py`: 학습 루프를 `fit`으로 옮기고 위 옵션을 받습니다. json에 `run`(장치·commit), `val_curve`, `best_it`, `best_val`, `stopped`가 추가됩니다. 생성은 `--ic-chunk`(기본 4, 저장된 앙상블을 이어 만들 때는 같은 값 유지)로 묶고, 테스트 연도도 캐시에서 읽습니다.
- `pgcfm/seqdct.py`: `SeqDCT.noise`, `SeqRegressor.sample`, `SeqFM.noise`, `hard_constraints`가 장치를 따릅니다.
- `scripts/run_step13_best.sh`: 프로파일, manifest, 종료 코드 75 처리, 학습·검증·테스트 연도 인수.
- `scripts/report_step13_broad.py`: json 기반으로 다시 씀(위 표).
- `tests/test_runs.py`: 재개 동일성, prefetch 불변성, best 선택과 patience, memmap 캐시 = RAM 적재.

## 4. 실행 방법

```bash
# 4년, CPU (전과 같은 설정, 결과도 같음)
bash scripts/run_step13_best.sh

# 37년, GPU: PINN 8000 it batch 8, FM/회귀 40000 it batch 16, 1000 it마다 검증(4 member z500 RMSE),
# memmap 캐시($OUT/cache, 37년 × 0.8 GB ≈ 30 GB 디스크), DEVICE=auto
PROFILE=broad DATA=/path/era5 OUT=runs/step13_broad2 bash scripts/run_step13_best.sh

# job 시간 제한이 있으면: 11시간마다 저장하고 종료 코드 75, 같은 명령을 다시 돌리면 이어서 함
MAX_HOURS=11 PROFILE=broad DATA=/path/era5 OUT=runs/step13_broad2 bash scripts/run_step13_best.sh

# patience: 검증 5번 개선 없으면 멈추고 best를 선택
PATIENCE=5 PROFILE=broad ... ; AMP=1 로 bf16 (아래 6절, 검증 안 됨)

# 보고서 (그림 두 장 + json 기반 PDF)
python scripts/plot_step13_broad.py --scores runs/step13_broad2/score_best.json --out docs/step13_broad
python scripts/report_step13_broad.py --runs runs/step13_broad2 --out docs/step13_broad
```

단계 하나만 따로 돌릴 때(예: FM만 다시):

```bash
python scripts/run_step13_seqfm.py --kai $K --era5 $E --out $OUT --basis dct --pinn $OUT/pinn_project.pt \
  --save-ens --resume-ens --train-years $(seq -s ' ' 1979 2015) --val-year 2016 --test-years 2017 2018 \
  --device auto --cache $OUT/cache --iters 40000 --batch 16 --val-every 1000 --val-members 4 --max-hours 11
```

`$OUT/manifest.jsonl`에 단계별 명령과 commit이 남고, 각 json의 `run`에 장치와 commit이 남으므로 "어느 코드로 어느 장치에서 돌렸는가"가 결과 파일 안에 있습니다.

## 5. 검증 (작은 실험)

모두 이 컨테이너(CPU 4코어)에서 했습니다.

1. 단위 테스트 `tests/test_runs.py` 4개 통과: 17 step에서 예산으로 멈추고 다른 seed로 재개해도 손실 이력과 검증이 중단 없는 run과 같음; prefetch 0과 3에서 batch와 학습 뒤 난수 스트림이 같음; best 상태가 선택되고 patience로 멈춤; memmap 캐시가 RAM 적재와 같고 다른 통계의 캐시는 거부.
2. `--quick` 회귀 비교(2012–2015 데이터, 3 it)를 2dd09f2 코드와 새 코드로 각각 돌려 비교:
   - 1단계 PINN: gate 84개 전부 상대 1e-5 안에서 같고, 손실 이력 같음. 가중치 차이 4e-6(경향 표준편차를 double로 누적한 영향). `--prefetch 0`, `--cache`, `--val-every 2`로도 같음. balance 사영(pibal)도 같음(데이터 항 1e-7).
   - 2단계 FM: 학습 가중치가 비트 단위로 같음. raw 앙상블(2017, 2018) 비트 단위로 같음.
   - 2단계 회귀: 처음에는 가중치가 0.8 달랐는데, 원인은 두 가지였습니다. (a) `fit`에 가중치 제너레이터를 넘겨 두 번째 step부터 gradient clip이 안 걸림(수정: 리스트로 보관), (b) prefetch가 뽑아 둔 창 2개만큼 난수가 더 소비되어 학습 뒤 오차 적합(`fit_errors`) 표본이 달라짐(수정: `Prefetch.close()` 되감기). 수정 뒤 가중치와 raw 앙상블이 비트 단위로 같음.
   - 가중치를 읽어 생성만 하는 경로: 옛 스크립트와 새 스크립트의 json 96개 값과 npz 5개가 비트 단위로 같음.
   - 남는 차이: 잔차 진단(`residual`, `truth_residual`)이 상대 5e-4 정도 다릅니다. scale 추정 표본이 전용 스트림에서 나오게 바꿨기 때문이고, 배포된 "가중치를 읽어 재현" 경로와는 같습니다. PINN을 거친 arm은 PINN 가중치 4e-6 차이만큼(z500 기준 최대 0.1 m²/s²) 다릅니다.
3. `PROFILE=smoke`(2 it, 연 6 IC)로 전체 체인(데이터 → PINN → FM → 회귀 → guide → score → flowdep)을 새 `run_step13_best.sh`로 실행: 성공. 데이터(건너뜀) → PINN(2 it + gate) → FM(2 it, 2016/2017/2018 각 6 IC) → 회귀 → regm guide(band/flow/flowop) → aband → score → flowdep → score_sigma → fig_flowdep가 모두 종료 코드 0, 02:33–02:44(11분). `manifest.jsonl`에 9단계(데이터 두 단계는 건너뜀)가 기록되고, `pinn_project.pt_metrics.csv`, `seqfm_dct.pt_metrics.csv`, `seqregress.pt_metrics.csv`가 만들어지며, 그 출력 폴더에서 json 기반 보고서 PDF가 만들어짐(학습 곡선 그림 포함).
4. 보고서 스크립트: `runs/step13`(4년 run, 옛 json)과 quick run(새 json, 학습 곡선)에서 PDF가 만들어짐. 한글은 `wqy-zenhei`로 렌더링(DroidSansFallback이 없는 기계).

속도: quick PINN 학습이 옛 코드 2.5분 → 새 코드 38초였지만 페이지 캐시 효과가 섞여 있어 의미 있는 차이로 보지 않습니다. GPU에서의 prefetch 효과는 여기서 잴 수 없었습니다.

## 6. 남은 약점

1. **GPU와 bf16은 여기서 검증하지 못했습니다** (CPU 컨테이너). `--device cuda`는 코드 경로만 확인했고, `--amp`는 fp32와 비교하지 않았습니다. 소유자 기계에서 broad 프로파일을 한 번 돌려 2dd09f2 보고서의 숫자(회귀 z500 170 / 427 / 789)와 맞는지 보는 것이 첫 번째 확인입니다.
2. 보정(guide)과 채점은 여전히 CPU이고 `fit` 밖에 있습니다. guide의 spread field 학습(ClimODE NLL)은 재개가 안 됩니다. 37년 run에서도 보정은 2016년 한 해이므로 시간은 괜찮지만(소유자 run에서도 CPU), 체계는 학습과 다릅니다.
3. 앙상블은 연도마다 npz 하나(228 MB)이고 생성 재개는 연도 단위입니다. 360 IC × 8 member가 넘는 테스트(예: 더 많은 member, 더 긴 lead)에는 IC 청크 단위 저장이 필요합니다.
4. 단일 프로세스, 단일 GPU입니다. DDP는 넣지 않았습니다(`fit`의 체크포인트와 난수 구조는 rank 0 저장으로 확장할 수 있지만 검증하지 않은 코드는 넣지 않았습니다).
5. FM의 검증 지표는 고정 창의 학습 손실(+ 선택적으로 4 member z500 RMSE)입니다. CRPS가 아니므로 early stopping이 보정 뒤의 점수를 최적화하지는 않습니다. best 선택과 보정이 같은 2016년을 쓰므로 약한 선택 편향이 있습니다. 깨끗하게 하려면 선택용 연도(예: 2015)를 학습에서 빼야 하는데, 이는 broad 프로파일의 `TRAIN_YEARS`/`VAL_YEAR`로 지정할 수 있지만 기본값은 소유자의 설정(2016 겸용)을 따랐습니다.
6. broad 프로파일에서는 tercile 기후 연도(2006–2015)가 모두 학습 연도입니다. 기후는 학습 데이터에서 나와도 테스트 누출은 아니지만, 4년 run과 BSS를 비교할 때 기후가 다릅니다.
7. 37년 캐시는 디스크 30 GB입니다. 압축하지 않았습니다(memmap은 비압축이어야 하므로). 디스크가 모자라면 `CACHE=0`으로 RAM 적재(30 GB RAM 필요).
8. `report_step13_broad.py`의 2절 결론 문장은 숫자를 json에서 채우지만 문장 틀은 고정이고, 9절 다음 단계는 2026-10-27 run 기준 소유자의 판단을 그대로 둔 것입니다.

## 7. 비교 실험: ClimODE 연도 vs 37년 vs 긴 학습 (소유자 GPU용)

소유자 질문(2026-10-27 02:51, "epoch를 늘려보면? ClimODE와 같은 기준으로 학습하려면?")에 대한 답을 실행 가능한 형태로 넣었습니다.

- 학습량: broad run(40000 it × batch 16, 24 step 창)은 37년 창 약 5.4만 개의 약 12 epoch이고, 6시간 step 표본으로 1,540만 개입니다. ClimODE 300 epoch(10년 × 1460 step = 440만 개)의 3.5배이므로 "ClimODE만큼 덜 돌렸다"는 아닙니다. 다만 4년 run의 회귀 손실은 3000 it 끝에서도 떨어지고 있었고(5분위 평균 0.57 → 0.36), broad run에는 학습 곡선이 없어 포화 여부를 모릅니다.
- ClimODE 프로토콜(`third_party/climode/train_global.py`): 학습 2006–2015(원저자 코드는 2016 포함), 검증 2016, 테스트 2017–2018, 5변수, AdamW 5e-4 + cosine 300 epoch, batch = 연속 8 step × 10년, Gaussian NLL + 분산 벌점 + L2. 변수와 손실까지 맞추면 ClimODE 자체를 학습하는 것이므로(`docs/RUN_ON_GPU.md` 11절), 여기서는 **데이터 연도만** 맞춥니다.

| 프로파일 | 학습 연도 | FM / 회귀 | 검증 / patience | 무엇을 보나 |
| --- | --- | --- | --- | --- |
| `climode` | 2006–2015 | 40000 it × 16 | 1000 it마다 / 0 | ClimODE와 같은 데이터에서의 공정 비교 |
| `broad` | 1979–2015 | 40000 it × 16 | 1000 it마다 / 0 | 소유자 2026-10-27 run 재현 |
| `broad_long` | 1979–2015 | 120000 it × 16 | 1000 it마다 / 10 | 학습량을 늘리면 좋아지나 (best 선택) |

```bash
# 두 프로파일을 차례로 돌리고, 각자 보고서 + 소유자 broad run과 나란히 비교
DATA=/path/era5_5p625 BROAD=runs/step13_broad bash scripts/run_step15_compare.sh
# 시간 제한 job: MAX_HOURS=11 을 붙이고, 종료 코드 75면 같은 명령을 다시 실행
# 하나만: ONLY=climode bash scripts/run_step15_compare.sh
```

출력: `docs/step15/compare.md`(설정·best it·종료 이유, z500/t850 RMSE, 5변수 CRPS, SSR, 첫 run 대비 차이), `docs/step15/fig_compare.png`(2×2 lead별 z500 RMSE, t850 RMSE, z500 CRPS, z500 spread, persistence 점선, ClimODE 논문 값), `docs/step15/fig_compare_training.png`(FM·회귀의 학습·검증 손실), `docs/step15/<profile>/step13_broad_report.pdf`.

읽는 법과 주의:
- `climode`와 `broad`의 차이 = 27년 추가 데이터의 몫. 논문에서 ClimODE와 비교할 숫자는 `climode`입니다.
- `broad_long`의 cosine 스케줄은 120000 it 전체에 걸리므로, patience로 일찍 멈추면 학습률이 아직 높은 상태의 best가 선택됩니다. 그 경우 "더 길게"가 아니라 "같은 길이에서 검증 선택"의 효과에 가깝습니다(`stopped`와 `best_it`이 표에 나옴).
- FM의 과분산(SSR ≈ 2)은 학습량보다 손실·잡음 구조의 문제로 보이므로, epoch로 풀리지 않을 가능성이 큽니다. 회귀 평균의 단기 오차가 주로 움직일 것으로 예상합니다.
- best 선택과 보정이 같은 2016년을 씁니다(6절 약점 5).
- 디스크: memmap 캐시 climode 약 8 GB, broad_long 약 30 GB(통계가 학습 연도마다 달라 공유 불가), 앙상블 arm-연도당 228 MB.
- 비교 스크립트는 여기서 smoke 출력 두 개로만 시험했습니다. 세 프로파일의 실제 결과는 아직 없습니다.
