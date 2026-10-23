# GPU 서버에서 PG-CFM 학습·평가하기

이 문서의 명령은 모두 저장소 루트에서 실행합니다. 데이터 경로는 `KAI_ERA5_IO_FORMAT` 기준(1.5°, 121×240, 67채널, 연도별 `.nc`)이며 `configs/pgcfm_era5.yaml`에서 바꿉니다.

## 0. 코드 받기

```bash
git clone https://github.com/nayehyeon61-glitch/physical_guide_flowmatching.git
cd physical_guide_flowmatching
git checkout claude/project-thread-d5l3lg     # 1–7단계가 모두 들어 있는 최신 브랜치 (4단계 통합만: claude/project-thread-bessib)
pip install -r requirements.txt               # torch는 서버 CUDA에 맞는 wheel을 먼저 설치
python -m pytest -q                           # CPU에서 수 분, 전부 통과해야 함
```

PR을 main에 합칠 때 순서는 #1(데이터 계약) → #2, #3, #4, #5(서로 독립, 모두 #1 위) → #6(4단계 통합) → #7(6단계 사영) → 7단계 PR(DCT latent)입니다. 각 PR은 앞 PR 브랜치 위에 쌓여 있으므로, 순서대로 merge하면 각 PR의 diff에는 그 단계 파일만 남습니다.

| 단계 | PR | 브랜치 | 내용 |
|---|---|---|---|
| 1 | #1 | claude/project-thread-owgdcu | KAI ERA5 데이터 계약 (66+1채널, 정규화, 연도별 .nc) |
| — | #2 | claude/project-thread-n5vc19 | climate_diffusion 교훈 문서 |
| 3 | #3 | claude/project-thread-smm6qt | NeuralGCM 4식 PINN 잔차, 극점 처리 |
| 2 | #4 | claude/project-thread-cbs0gw | CFM 단독 앙상블 (torchcfm) |
| 5 | #5 | claude/project-thread-0d48hq | NeuralGCM/WB2 평가 스위트 |
| 4 | #6 | claude/project-thread-bessib | CFM + PINN 결합 PG-CFM, GPU 학습 스크립트 |
| 6 | #7 | claude/project-thread-z37s9z | PINN 고정 → ODE 사영 → rollout 시퀀스 손실 |
| 7 | 이 PR | claude/project-thread-d5l3lg | DCT latent: encoder→PINN→decoder, 병렬/직렬 CFM, FNO 앙상블 operator |

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

## 7. 6단계: PINN 학습·동결 → CFM 결과의 ODE 사영 → 시퀀스 손실 (`docs/step6_projection.md`)

3절의 CFM 단독 체크포인트(`runs/cfm/cfm.pt`)에서 시작합니다. CFM은 고정되고, PINN(bridge + closure)을 수렴할 때까지 학습해 고정한 뒤, 사영 ODE g_φ만 rollout CRPS로 학습합니다.

```bash
pip install torchdiffeq                       # requirements.txt에 추가됨
# (a) 사영 ODE, CRPS 시퀀스 손실만 (λ_P = 0)
torchrun --standalone --nproc_per_node 4 scripts/train_projection.py --config configs/projection_era5.yaml \
    --cfm-ckpt runs/cfm/cfm.pt --out runs/proj_crps
# (b) + 동결 PINN 잔차 (physics gradient ≤ 0.5 × CRPS gradient). 1단계 PINN은 (a)의 것을 재사용
mkdir -p runs/proj_phys && cp runs/proj_crps/pinn.pt runs/proj_phys/
torchrun --standalone --nproc_per_node 4 scripts/train_projection.py --config configs/projection_era5.yaml \
    --cfm-ckpt runs/cfm/cfm.pt --out runs/proj_phys --resume --set lambda_p=0.1
# (c) 비물리 평활 대조군 (같은 λ, 물리 없음)
mkdir -p runs/proj_smooth && cp runs/proj_crps/pinn.pt runs/proj_smooth/
torchrun --standalone --nproc_per_node 4 scripts/train_projection.py --config configs/projection_era5.yaml \
    --cfm-ckpt runs/cfm/cfm.pt --out runs/proj_smooth --resume --set lambda_p=0.1 control=smooth
```

- 1단계 로그: `[pinn] ... val_phys ... rel` — 상대 개선이 `pinn_tol`(1%) 미만인 window가 `pinn_patience`번 이어지면 멈추고 `pinn.pt`를 씁니다. `log.jsonl`의 `stage: pinn` 줄에 기록됩니다.
- 2단계 로그: `[proj] ... crps crps_d1 crps_d2 ... horizon` (+ `phys grad_crps grad_phys phys_scale`). curriculum `[1, 2, 3, 5]`일이 `proj_iters`를 4등분합니다.
- 메모리: 121×240에서 5일 rollout을 CFM까지 역전파(`bptt: true`)하면 GPU당 batch 2 × 2 members가 한계일 수 있습니다. OOM이면 `--set bptt=false`(CFM 사이 gradient 끊음), `curriculum=[1,2,3]`, `cfm_steps`는 추론과 같게 유지하세요.

추론과 평가는 4·5절과 같습니다 (`proj.pt`면 CFM 뒤에 사영 ODE가 자동으로 붙습니다):

```bash
python scripts/infer_pgcfm.py --ckpt runs/proj_crps/proj.pt --test-file /path/to/era-data/test/2020.nc \
    --stats-dir /path/to/KAI-New/stats --out runs/proj_crps/forecast.nc --leads 10 --members 16 --ic-stride 7 --det
python scripts/evaluate.py score --data-dir /path/to/era-data/test --year 2020 \
    --forecast CFM=runs/cfm/forecast.nc --forecast "proj CRPS"=runs/proj_crps/forecast.nc \
    --forecast "proj +PINN"=runs/proj_phys/forecast.nc --forecast "proj +smooth"=runs/proj_smooth/forecast.nc \
    --climatology runs/clim_1980_2011.nc --out runs/eval/2020_step6
```

확인할 것: (1) 사영 후 1일 SSR이 1에 가까워지는지와 함께 멤버 진폭·스펙트럼이 진실보다 작아지지 않는지(압축이 진폭 축소로 가는지), (2) `proj +PINN`이 `proj +smooth`보다 물리 균형 진단(정역학·지균)에서 나은지, (3) 5일 이후 lead에서 curriculum 밖(5일 초과)으로도 안정한지.

## 8. 7단계: DCT latent PG-CFM (`docs/step7_latent.md`)

DCT encoder → latent PINN → IDCT decoder를 먼저 학습하고(stage 1, PINN 고정), 그 latent에서 CFM을 학습합니다(stage 2, encoder/decoder 고정). 기본값은 합성 실험에서 가장 좋았던 **직렬** 결합입니다: CFM이 멤버 A, B, C…를 만들고, 학습 가능한 PINN Q가 멤버마다 보정한 뒤 decoder로 갑니다.

```bash
pip install torch-dct==0.1.6 neuraloperator   # requirements.txt에 추가됨
# (a) 직렬 (기본): stage 1 → CFM + 멤버별 PINN 동시 학습
python scripts/train_latent.py --config configs/latent_era5.yaml --out runs/latent_serial
# (b) 병렬 비교군: 같은 stage 1을 재사용, P(z_k) + CFM, encoder/decoder 고정
mkdir -p runs/latent_anchor && cp runs/latent_serial/stage1.pt runs/latent_anchor/
python scripts/train_latent.py --config configs/latent_era5.yaml --out runs/latent_anchor --resume --set coupling=anchor
# (c) physics 없는 직렬 (stage 1·2 모두 λ_P = 0: PINN 항이 무엇을 더하는지)
python scripts/train_latent.py --config configs/latent_era5.yaml --out runs/latent_serial_nophys --set lambda_p=0
```

- (b)의 stage1.pt 재사용은 `coupling`만 다른 같은 설정일 때만 됩니다(직렬 PINN Q의 가중치는 stage 1에 없으므로 `--resume`이 encoder/decoder/PINN만 읽음). 문제가 생기면 (b)를 처음부터 돌리세요.
- **stage 1이 끝나면 먼저 로그의 `[stage1] floors`를 보세요.** `rmse_ae`가 `tendency_rms`보다 크면(합성에서 1.4–1.5배) 그 latent로는 persistence를 이길 수 없습니다. 이 경우 `k_lat/k_lon`, `latent`, `enc_hidden`을 키우거나 데이터를 늘리세요(W1, W2).
- 메모리: encoder 첫 층은 66 × k_lat × k_lon 입력입니다(기본 24×48 → 76k × 1024). 직렬은 Heun `serial_steps`번을 역전파합니다. OOM이면 `serial_steps=2`, `batch=4`.
- 추론·평가는 4·5절과 같습니다(`latent.pt`를 넣으면 자동으로 latent 경로):

```bash
python scripts/infer_pgcfm.py --ckpt runs/latent_serial/latent.pt --test-file /path/to/era-data/test/2020.nc \
    --stats-dir /path/to/KAI-New/stats --out runs/latent_serial/forecast.nc --leads 10 --members 16 --ic-stride 7 --det
python scripts/evaluate.py score --data-dir /path/to/era-data/test --year 2020 \
    --forecast CFM=runs/cfm/forecast.nc --forecast "latent serial"=runs/latent_serial/forecast.nc \
    --forecast "latent anchor"=runs/latent_anchor/forecast.nc --forecast "serial no phys"=runs/latent_serial_nophys/forecast.nc \
    --climatology runs/clim_1980_2011.nc --out runs/eval/2020_step7
```

합성 데이터 실험 재현 (CPU, 약 3시간): `python scripts/run_step7_latent.py --out runs/step7`, 과분산 가설 검증: `python scripts/step7_overdispersion.py --out runs/step7_od`.

FNO 앙상블 operator(`pgcfm/ens_operator.py`)는 1일 점수는 좋지만 rollout에서 발산하므로(W6) GPU 학습 스크립트에는 아직 넣지 않았습니다.
