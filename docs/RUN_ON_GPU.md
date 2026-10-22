# GPU 서버에서 PG-CFM 학습·평가하기

이 문서의 명령은 모두 저장소 루트에서 실행합니다. 데이터 경로는 `KAI_ERA5_IO_FORMAT` 기준(1.5°, 121×240, 67채널, 연도별 `.nc`)이며 `configs/pgcfm_era5.yaml`에서 바꿉니다.

## 0. 코드 받기

```bash
git clone https://github.com/nayehyeon61-glitch/physical_guide_flowmatching.git
cd physical_guide_flowmatching
git checkout claude/project-thread-bessib     # 1–5단계 PR이 모두 합쳐진 통합 브랜치
pip install -r requirements.txt               # torch는 서버 CUDA에 맞는 wheel을 먼저 설치
python -m pytest -q                           # CPU에서 수 분, 전부 통과해야 함
```

PR을 main에 합칠 때 순서는 #1(데이터 계약) → #2, #3, #4, #5(서로 독립, 모두 #1 위) → 이 PR(4단계 통합)입니다. 이 브랜치에는 이미 #1–#5가 merge되어 있으므로, 순서대로 merge하면 이 PR의 diff는 4단계 파일만 남습니다.

## 1. 경로 설정

`configs/pgcfm_era5.yaml`의 세 줄을 서버 경로로 바꿉니다.

```yaml
train_dir: /path/to/era-data/train     # 1980–2011 연도별 .nc
valid_dir: /path/to/era-data/valid     # 2012
stats_dir: /path/to/KAI-New/stats      # global_mean.npy, global_std.npy (67,)
```

시험 연도(2020) 파일은 학습에 쓰지 않고 추론과 평가에만 씁니다.

## 2. 실제 ERA5 잔차 크기 측정 (3단계, 학습 전에 한 번)

```bash
python -m scripts.step3_era5_residual_stats --data-dir /path/to/era-data/train \
    --years 2010 2011 --max-pairs 60 --out runs/era5_residual_stats.json
```

출력의 방정식별 RMS가 b_j입니다. 학습 스크립트도 시작할 때 관측 쌍 50 batch로 b_j를 다시 추정하지만(하한 = 기본 규모 × 0.1), 두 값이 크게 다르면 yaml의 `scales:`에 이 값을 직접 넣으세요. **mass 잔차는 ω를 연속식으로 진단하므로 관측에서 거의 0입니다.** 하한이 없으면 b_mass → 0이 되어 손실이 폭주합니다(`docs/step4_pgcfm.md` W1).

## 3. 학습

GPU 한 장:

```bash
python scripts/train_pgcfm.py --config configs/pgcfm_era5.yaml --out runs/pgcfm
```

한 노드 GPU 여러 장 (`batch`는 GPU당 값):

```bash
torchrun --standalone --nproc_per_node 4 scripts/train_pgcfm.py --config configs/pgcfm_era5.yaml --out runs/pgcfm
```

같은 backbone, 같은 θ 갱신 횟수의 CFM 단독 기준선 (비교에 꼭 필요):

```bash
torchrun --standalone --nproc_per_node 4 scripts/train_pgcfm.py --config configs/pgcfm_era5.yaml \
    --out runs/cfm --phases cfm cfm_more
```

단계(부록 C): `cfm`(CFM 단독, 기본 100k iter) → `guide`(관측 쌍으로 bridge h_ψ + closure 사전학습, θ 고정) → `joint`(식 12, λ_P를 `ramp_iters` 동안 0에서 올림, closure 고정). 각 단계가 끝나면 `runs/pgcfm/<단계>.pt`를 저장합니다.

- 중단 후 재시작: 같은 명령에 `--resume`을 붙이면 마지막으로 끝난 단계 다음부터 이어갑니다(단계 중간 재시작은 아직 없음).
- 값 바꾸기: `--set lambda_p=0.3 batch=4 grad_ratio=0.5` 처럼 yaml 값을 덮어씁니다.
- CFM 단독 체크포인트를 PG-CFM에 재사용: `cp runs/cfm/cfm.pt runs/pgcfm/cfm.pt` 후 `--resume`.
- 로그: `runs/pgcfm/log.jsonl` (100 iter마다 손실, `val_every`마다 1일 CRPS·RMSE·SSR·tendency 비·독립 물리 잔차).

메모리: 121×240에서 `joint`는 미분 가능한 Heun(2 step = U-Net 4회)과 Simpson 5 node의 float64 잔차를 한 graph에 담습니다. OOM이면 `batch`를 줄이거나 `sample_steps=1`, `n_sub=2`로 내리세요.

## 4. 추론 (forecast.nc)

```bash
python scripts/infer_pgcfm.py --ckpt runs/pgcfm/joint.pt --test-file /path/to/era-data/test/2020.nc \
    --stats-dir /path/to/KAI-New/stats --out runs/pgcfm/forecast.nc --leads 10 --members 16 --ic-stride 7 --det
python scripts/infer_pgcfm.py --ckpt runs/cfm/cfm_more.pt --test-file /path/to/era-data/test/2020.nc \
    --stats-dir /path/to/KAI-New/stats --out runs/cfm/forecast.nc --leads 10 --members 16 --ic-stride 7 --det
```

`forecast (initial_condition, lead_time, member, channel, lat, lon)`, ERA5 단위, 67채널, lead 0 = 초기장. `--det`는 같은 초기장의 ε=0 실행을 `forecast_det.nc`로 씁니다.

## 5. 평가 (5단계, NeuralGCM/WeatherBench2 기준)

```bash
python scripts/evaluate.py climatology --data-dir /path/to/era-data/train --years 1980-2011 --out runs/clim_1980_2011.nc
python scripts/evaluate.py score --data-dir /path/to/era-data/test --year 2020 \
    --forecast PG-CFM=runs/pgcfm/forecast.nc --forecast "PG-CFM eps=0"=runs/pgcfm/forecast_det.nc \
    --forecast CFM=runs/cfm/forecast.nc --forecast "CFM eps=0"=runs/cfm/forecast_det.nc \
    --climatology runs/clim_1980_2011.nc --out runs/eval/2020
```

## 6. 직접 확인할 것 (체크리스트)

1. `pytest`가 서버 환경에서 모두 통과하는지.
2. 2단계의 b_j(실제 ERA5)가 학습 로그 첫 줄의 `residual scales b_j`와 같은 자릿수인지. mass·hyd가 하한값(1e-6, 0.1)에 붙어 있다면 그 식이 관측에서 사실상 0이라는 뜻입니다.
3. `cfm` 단계 끝의 val: `ssr`이 1 근처인지, `tendency_ratio`가 1보다 많이 크지 않은지(합성 실험에서 CFM 단독은 1일 과대분산 SSR 1.25).
4. `joint` 단계 로그에서 `cfm` 손실이 `cfm` 단계 끝 값보다 크게 오르지 않는지. 합성 실험에서 1.34 → 1.76으로 올랐고 1일 CRPS가 26% 나빠졌습니다. 오르면 `grad_ratio`를 켜세요(docs/step4 W2).
5. `val res_gen_*`(생성 멤버의 독립 잔차)가 `res_obs_*`(관측 쌍)에 가까워지는지. 가장 큰 항이 무엇인지(합성에서는 hyd가 손실의 95% 이상).
6. 평가에서 PG-CFM과 CFM 단독을 같은 초기장으로 비교: CRPS, 앙상블 평균 RMSE, spread-skill, ε=0 기준선, tendency 진폭 비, 물리 균형 진단(정역학·지균).
7. 멤버 진폭(`lead_member_amp`에 해당하는 spread와 스펙트럼)이 진실보다 작아지지 않는지. PG-CFM이 장기 lead에서 이기는 이유가 진폭 축소(기후값으로 수렴)인지 확인해야 합니다(docs/step4 W3).
