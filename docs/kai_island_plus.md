# 칼만 섬 + 버전: 1.5°에서 10년, 32년 학습 (2026-11-20)

요청(2026-11-20 12:30 KST): "이제 이 칼만 섬에 대해서도 1.5도 기준 gpu에서 10, 32년도 학습시킬 수 있도록 + 버전 작성"

5.625°에서 가장 좋았던 모델 D3(2026-11-18, 아래 1절)를 KAI의 격자인 1.5°(121 × 240, 극 행 포함)로 옮기고, GPU 서버에서 10년(2006~2015) 또는 32년(1980~2011)으로 처음부터 끝까지 학습하고 채점하는 명령 하나로 묶었습니다. 검증 연도 2016, 시험 연도 2017은 5.625°, + 버전과 같습니다.

- 이 문서에 있는 것: 무엇을 옮겼는지, 1.5°에서 바꾼 것, 명령, 반복 수와 예상 시간, 결과 파일, 주의할 점, 시험 상태.
- 이 문서에 없는 것: 1.5°에서의 성능 수치. 이 환경에는 GPU가 없어서 CPU의 작은 시험(해마다 48일, 작은 망)으로 흐름만 확인했습니다.

## 1. 옮기는 모델 (5.625°의 D3)

| 단계 | 5.625°의 이름 | 1.5° 실행의 이름 | 하는 일 |
| --- | --- | --- | --- |
| KAI, 미세조정 KAI | `kai_x8`, `kai_x8ro` | `kai`, `kai_ro` (폴더마다) | 섬의 측정값. + driver(`run_kai_plus.sh`)가 학습 |
| 첫 core | `core_gndmz8` | `core_m8d` | 8일 기억 + Mori-Zwanzig sequence loss + dycore 24시간 적분 힌트 |
| 칼만 표 | `kal_tab_core_gndmz8_5_2016_s5_d1234o_kaix8ro.npz` | `kal_tab_core_m8d_5_2016_s5_d1234o_kairo.npz` | 검증 연도에서 채널과 위도마다 K = (S − c) / (S + R − 2c), 1~4일 출력 보정과 5일의 섬 |
| 둘째 core | `core_s2memx` | `core_m8d_s2` | 얼린 첫 core와 고정 칼만 섬 뒤에서 학습, 섬에서 첫 core의 기억을 넘겨받음 |
| 섬 실행 | B2 | `isn-core_m8d+core_m8d_s2-Kcorrd1234okairo-err0-m1` | 1~4일 출력을 KAI 쪽으로 보정, 5일에 섬, 6~10일 둘째 core |
| 행렬 섬 | `+incfull` | `...+incfull...` (B3) | 검증 연도 사례로 섬의 보정을 변수 간 × 공간 3 × 3 행렬로 |
| 늦은 KAI 보정 | D2, D3 | `fb10-core_m8d+core_m8d_s2-kairo`, `fb10-core_m8d+core_m8d_s2inc-kairo` | 모든 선행일을 검증 연도의 이득으로 KAI 쪽으로 보정 |
| 단순 혼합 (대조군) | S | `fb10-core_m8d-kairo` | 첫 core 하나 + 모든 선행일에서 KAI와 혼합. 섬 구조의 몫은 이것과 비교 |

5.625°(학습 2012~2015, 시험 2017, 119개 초기장)에서의 수치: D3 10일 nRMSE .454, ACC .378. D3 대 D2는 같은 smoothing에서 5~10일 +0.2 ~ +0.3 %. 섬 구조(D) 대 단순 혼합(S)은 같은 smoothing에서 6~10일 +0.5 ~ +1.1 %(2016년을 시험으로 바꿔도 +0.9 ~ +1.7 %).

## 2. 1.5°에서 바꾼 것

| 부분 | 5.625° | 1.5°에서의 처리 |
| --- | --- | --- |
| dycore 적분 힌트 | 자료 격자(64 × 32)가 곧 T21 격자, 하루에 solver 한 step | 상태를 T21 격자(64 × 32)로 보존 평균해서 24시간 적분하고, 증분을 1.5°로 선형 보간해서 돌려줌. `DYC_GRID=64`이면 T42(128 × 64, dt 45분). **하루에 한 번** 그날 시작 상태로 적분하고, 하루의 Heun substep 4개가 같은 힌트를 씀(기억 m을 하루 동안 고정하는 것과 같은 방식. 5.625°는 하루 한 step이라 같은 뜻이고, substep마다 적분하면 비용이 4배). dycore는 JAX로 CPU에서 돌고(`JAX_PLATFORMS=cpu`), 학습 시작 때 학습 연도의 모든 날을 미리 적분해서 거친 격자로 보관(하루 0.4 MB) |
| 학습 메모리 절약 | core +만 썼음 | gradient checkpointing(`--core-ckpt`)이 적분 힌트의 캐시와 충돌하던 오류를 고침(캐시를 놓친 상태에서만 index 연산이 기록되어, 다시 계산할 때 저장한 tensor 수가 달라짐). 둘째 core도 이것으로 1.5°에서 학습할 수 있음 |
| 칼만 섬 실행 (`diag_island_kalman.py`) | 모든 예보를 메모리에 | 자료는 디스크에 둔 채 읽을 때 정규화, 예보는 묶음마다 디스크에 씀(예보 하나 119 × 10일 × 66채널 = 4.6 GB, float16), `--device cuda`. `--no-central`, `--no-runs`, `--table-only` 추가 |
| 행렬 섬 (`island_matrix_gain.py`, 새 파일) | `diag_island_ensemble.py`가 한 번에 | 위도 행마다 정규방정식을 모으는 방식. 3 × 3 stencil의 간격은 격자 4칸(6°, 5.625°의 한 칸과 비슷), 이웃 행 묶음도 ±4행 |
| 늦은 KAI 보정 (`make_full_blend.py`) | 한 번에 | 초기장 16개씩 나눠 통계를 모으고, 결과를 묶음마다 디스크에 씀 |
| 채점 (`diag_stage_filter_light.py`) | 시험 연도를 메모리에 | 읽을 때 정규화, 필요한 날의 기후값만 |

5.625°에서 바꾼 스크립트 넷이 이전 결과를 그대로 내는 것을 확인했습니다(7절).

## 3. 명령어 (GPU 서버)

서버의 clone은 지금 `claude/project-thread-wjjayb`에 있습니다. 이 버전은 `claude/project-thread-f8ks9d`에 있고, 그 가지에 wjjayb의 내용이 모두 들어 있습니다. `runs/`와 `data/`는 git이 관리하지 않으므로 가지를 바꿔도 그대로 있습니다.

```bash
cd physical_guide_flowmatching
git fetch origin claude/project-thread-f8ks9d
git checkout -B claude/project-thread-f8ks9d FETCH_HEAD     # 이후 갱신: git pull --ff-only origin claude/project-thread-f8ks9d

# dycore(JAX, CPU판)와 dinosaur. 한 번만
pip install jax "git+https://github.com/neuralgcm/dinosaur.git"
python -c "import jax, dinosaur, torch; print(jax.__version__, torch.cuda.is_available())"

# 0. 작은 시험: 모든 단계를 몇 iteration씩 (lustre에서 2015~2017년 48일씩 변환, 30분 안팎 예상)
SMOKE=1 bash scripts/run_kai_island_plus.sh

# 1. 10년 (2006~2015): 자료 변환(없는 해만), KAI, 미세조정 KAI, 첫 core, 칼만 표, 둘째 core, 섬 실행, 행렬 섬, 늦은 보정, 채점
YEARS=10 nohup bash scripts/run_kai_island_plus.sh > runs/island_plus_10y.out 2>&1 &

# 2. 32년 (1980~2011): ++ 32년 명령의 폴더(runs/kaiplus_1p5_daily_32y)에 KAI가 있으면 그대로 씀
YEARS=32 nohup bash scripts/run_kai_island_plus.sh > runs/island_plus_32y.out 2>&1 &

# 진행 보기
tail -f runs/island_plus_10y.out                               # 지금 어느 단계인지
tail -n 3 runs/kaiplus_1p5_daily_10y/core_m8d.log              # 학습 log: loss, val, sec(누적 초), gpu_gb
```

- 끊기면 같은 명령을 다시 실행합니다. 끝난 단계는 건너뛰고, 학습은 마지막 기록 지점에서 이어집니다. **반복 수는 시작한 뒤에 바꾸지 마세요**(학습률 일정이 처음 반복 수로 정해집니다).
- GPU가 둘이면 `CUDA_VISIBLE_DEVICES=1 YEARS=32 ...`처럼 나눕니다. 두 실행이 2006~2011년 자료를 같이 쓰므로, **32년 명령은 10년 실행의 `kai.log`가 생긴 뒤(자료 변환이 끝난 뒤)에 시작**합니다. 그러면 32년 실행은 1980~2005년만 변환합니다.
- 32년 자료가 처음이면 먼저 1980년이 같은 모양인지 확인합니다(몇 초, NOT OK나 WARNING이 나오면 멈추고 출력을 보내 주세요): `python scripts/local_era5_to_kai.py --src /lustre/home/mahmed/ERA5_0p25_DAILY --src-daily --check --years 1980 --grid 1.5 --out data/era5_1p5_daily`
- `YEARS=4`는 + 실행의 폴더(`runs/kaiplus_1p5_daily`)에서 이미 학습된 4년 KAI를 그대로 쓰고 core 둘만 학습합니다(하루 안팎 예상). 같은 1.5°에서 연도의 효과를 볼 기준점이 되고, GPU에서 흐름을 먼저 확인하는 가장 싼 실행입니다.
- `STOP=1`(첫 core까지), `STOP=3`(둘째 core까지)처럼 중간에서 멈출 수 있습니다. `DRYRUN=1`은 명령만 출력합니다.
- 다른 설정: `C1_ITERS`, `C2_ITERS`(반복 수), `DYC_GRID=64`(dycore를 T42로), `TRAIN_YEARS="..." OUT=...`(다른 기간, 자기 폴더에서), `VAL_YEAR`, `TEST_YEAR`, `IC_STRIDE`(채점 초기장 간격, 기본 3일 = 119개).

## 4. 반복 수와 예상 시간

반복 수는 + 버전과 같은 규칙(`TIME=1`)으로 **4년 실행과 자료를 도는 횟수(epoch)가 같도록** 연도 / 4배로 잡았습니다. 5.625°의 시간 확장 시험에서 연도만 늘리면 이득이 작고 반복 수를 같이 늘려야 했기 때문입니다.

| 단계 | 4년 | 10년 | 32년 | 예상 시간 (L40S 한 장, 10년 / 32년) |
| --- | --- | --- | --- | --- |
| 자료 변환 | 있음 | 2006~2011 (6년) | 1980~2011 | 약 1시간 / 약 5시간 (+ 실행: 6년에 53분) |
| KAI (batch 8) | 20,000 | 50,000 | 160,000 | 약 15시간 / 약 2.0일 (1.09초/it). 32년은 ++ 폴더에 있으면 0 |
| 미세조정 KAI | 1,000 | 1,000 | 1,000 | 약 0.4시간 |
| 첫 core `core_m8d` (batch 8) | 3,000 | 7,500 | 24,000 | 약 19시간 / 약 2.5일 (약 9초/it) |
| 칼만 표와 첫 core 예보 (두 해) | | | | 약 1 ~ 2시간 |
| 둘째 core `core_m8d_s2` (batch 4 × 2) | 1,200 | 3,000 | 9,600 | 약 17시간 / 약 2.2일 (약 20초/it) |
| 섬 실행 4번, 행렬 섬, 늦은 보정 3개, 채점 3개 | | | | 약 3 ~ 5시간 |
| 합계 | | | | **약 2.5일 / 약 7일** (32년 KAI가 ++ 폴더에 있으면 약 5일) |

모두 **추정**이고 GPU에서 잰 적이 없습니다. 근거:

- 첫 core는 + 실행의 core +(같은 크기, 같은 기억 구조, L40S에서 7.07초/it, 12.6 GB)에 dycore 힌트를 더한 것입니다. dycore는 거친 격자(64 × 32)에서 CPU로 돌아서 1.5°에서도 5.625°와 비용이 같습니다(이 환경의 CPU에서 상태 하나에 약 0.18초). 자료의 날은 학습 시작 때 미리 적분해 두고, rollout의 둘째 날부터만 그때그때 적분하므로 iteration당 2 ~ 3초를 더해 약 9초로 잡았습니다.
- 둘째 core는 iteration마다 첫 core로 5일(gradient 없이)과 둘째 core로 최대 5일을 가므로, 그때그때 적분하는 상태가 첫 core의 약 5배입니다. 망 계산과 합쳐 약 20초로 잡았습니다.
- 작은 시험(CPU)에서 첫 core의 iteration 시간은 힌트를 하루 한 번으로 바꾼 뒤 420초에서 195초로 줄었습니다. 이 시간은 작은 망이라 dycore가 대부분입니다.

줄이는 방법(반복 수를 절반으로, 약 16 epoch): `YEARS=32 C1_ITERS=12000 C2_ITERS=4800 ...`이면 core 둘이 약 2.4일입니다. 5.625°의 core 크기 시험에서는 연도와 함께 반복 수를 늘렸을 때 이득이 났으므로 기본값은 epoch을 맞춘 값으로 두었습니다.

## 5. 결과 파일 (보내 주시면 읽겠습니다)

폴더(`runs/kaiplus_1p5_daily_10y`, `runs/kaiplus_1p5_daily_32y`)에 생깁니다.

| 파일 | 내용 |
| --- | --- |
| `table_island_plus.md` | 2017년의 1 / 3 / 5 / 7 / 10일: 65채널 nRMSE, ACC, 활동도(예보 anomaly 크기 / 관측). 미세조정 KAI, D3, D2, S, B2, 첫 core. 같은 smoothing에서 KAI 대비 이득(95 % 구간), **D3 대 D2(행렬 섬의 몫), D3와 D2 대 S(섬 구조의 몫)**, 다섯 변수(t2m, t850, z500, u850, msl)의 RMSE / ACC를 물리 단위로 |
| `fig_island_plus.png` | 2 × 2 그림: 선행일별 nRMSE, ACC, 활동도, KAI 대비 같은 smoothing의 이득 |
| `score_island_vs_kai.json`, `score_island_d3.json`, `score_island_vs_s.json` | 위 표의 원본 수치 |
| `island_*.json`, `island_matrix_full.json`, `*.log` | 칼만 이득, 행렬 섬의 λ와 교차검증, 학습 log |

`*.npy`(예보, 한 파일 4.6 GB)는 보내지 않아도 됩니다. 위의 `.md`, `.png`, `.json`, `.log`를 보내 주시면 됩니다.

## 6. 주의할 점 (취약점 기준)

1. **이 크기로 돌려 본 적이 없습니다.** CPU의 작은 시험(해마다 48일, 작은 망, 몇 iteration)으로 모든 단계가 이어지는 것만 확인했습니다. 첫 실제 숫자는 첫 core log의 `sec`(첫 기록 지점의 누적 초)입니다. 100 iteration에 900초 안팎이면 위의 추정과 맞고, 훨씬 크면 자료 읽기나 dycore가 병목입니다.
2. **dycore는 CPU에서 돕니다.** 학습 시작 때 학습 연도의 모든 날을 적분해 둡니다(10년 약 15분, 32년 약 40분, 메모리 1.5 GB / 4.7 GB 추정). 그동안 학습 log에는 `dycore_prefill_states` 한 줄만 찍힙니다. rollout 둘째 날부터의 힌트와 섬 실행의 힌트는 그때그때 적분합니다.
3. **디스크**: 폴더 하나에 예보 캐시 11개 × 4.6 GB = 약 51 GB, 행렬 섬 증분 둘 약 0.9 GB, 칼만 표의 사례 약 0.3 GB. 자료는 1.5° 일평균 한 해 2.84 GB(32년 약 91 GB, 10년 약 28 GB).
4. **KAI의 질이 섬의 측정값입니다.** 10년 실행은 KAI도 10년으로 새로 학습합니다. 5.625°에서 강한 KAI(같은 4년, 반복 8배)로 바꿨을 때 섬 모델의 10일 nRMSE가 .470에서 .455로 좋아졌습니다. KAI가 약하면 섬의 이득도 작아집니다.
5. **검증 연도 하나로 정하는 값이 셋입니다**: 칼만 표, 행렬 섬, 늦은 보정. 모두 2016년에서 정하고 2017년에서 채점합니다. 5.625°에서 두 해를 바꿔도 결론이 같았지만 1.5°에서는 확인하지 않았습니다.
6. **한 seed입니다.** 5.625°에서 D3 대 D2의 차이는 +0.2 % 안팎이라 seed 차이와 비슷한 크기일 수 있습니다. 섬 구조 대 단순 혼합(+0.5 ~ +1.1 %)이 더 믿을 만한 비교입니다.

## 7. 시험 상태

- **5.625°에서 이전 결과 그대로**: 칼만 섬 실행(차이 0.0), 행렬 섬 증분(상대 차이 5.6e-6, 84초), 늦은 보정(차이 0.0), 채점(같은 수치). 2016년 B2 캐시는 11-18의 연도 교환 시험이 덮어쓴 것이라 같은 입력끼리 비교했습니다.
- **1.5° dycore 경로**: 1.5° 상태를 T21로 내려 적분한 증분이 원래 T21 자료로 적분한 것과 상관 .97 ~ .98, NaN 없음.
- **1.5° CPU 작은 시험**(실제 ERA5를 WB2에서 받아 2015 ~ 2017년 해마다 48일, 작은 KAI와 작은 core, iteration 2번, 7개 초기장, 7일): 모든 단계가 처음부터 끝까지 이어지고 표와 그림이 나옴(약 45분). 다시 실행하면 끝난 단계를 모두 건너뜀. 이 시험에서 찾아 고친 것 셋:
  1. 메모리 절약(gradient checkpointing)과 적분 힌트의 캐시가 충돌해 첫 core 학습이 둘째 iteration에서 멈춤(5.625°에서 둘째 core를 `--core-ckpt`로 학습하려다 실패했던 것도 같은 원인으로 보입니다. 추정)
  2. 늦은 보정이 디스크의 자료를 읽는 경로에서 멈춤
  3. 같은 smoothing 비교가 극 근처에서 터짐(D3 대 S의 1일이 +91 %): 위도 행을 묶는 합을 float32로 하면 북극 쪽 행의 작은 power가 사라져 0이 되고, 비율이 1억 배가 됨. float64로 바꾼 뒤 −0.3 %. 5.625°의 수치는 소수 여섯째 자리까지 같음
- **고친 뒤 5.625°를 다시 확인**: 칼만 섬 실행(초기장 8개, 차이 0.0), 같은 smoothing 비교(D3 대 D2, 차이 3e-7 %).
- **dry run**: `YEARS=10`(첫 core 7,500 it, 둘째 core 3,000 it), `YEARS=32`(24,000 / 9,600 it), 폴더 이름, 명령 줄.
- GPU에서는 돌려 보지 못했습니다.

## 재현 (이 문서의 시험)

```bash
# 5.625°: 이전 결과와 같은지 (runs/kaitend의 기존 캐시로)
python scripts/island_matrix_gain.py --out runs/kaitend ... --core-fc core_gndmz8 --meas kai_x8ro --table <kal_tab> --name chk
# 1.5° 작은 시험 (이 환경: WB2에서 받은 48일 자료)
SMOKE=1 DEVICE=cpu SRC=wb2 DATA=data/era5_1p5_smoke OUT=runs/kaiisl_1p5_smoke bash scripts/run_kai_island_plus.sh
DRYRUN=1 YEARS=10 bash scripts/run_kai_island_plus.sh
```
