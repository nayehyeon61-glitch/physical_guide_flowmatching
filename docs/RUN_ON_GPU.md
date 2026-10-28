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

## 9. 8단계: 앙상블 분포 보정 (`docs/step8_spread.md`)

8절의 직렬 체크포인트(`latent.pt`, `coupling=serial`)를 그대로 쓰고, 학습 없이 멤버 분포만 보정합니다. 검증 기간(학습에 쓰지 않은 해)의 초기장 절반으로 α를 맞추고, 나머지 절반에서 보정하지 않은 직렬과 비교한 표와 H2 점검(Q 전후 분산, 정역학 접공간/법선 성분)을 출력합니다.

```bash
# (1) 보정 적합: inflate(α), band(α_b), posthoc(lead별 α_k), tangent_inflate(허용 방향만 α)
python scripts/fit_spread.py --ckpt runs/latent_serial/latent.pt --valid-dir /path/to/era-data/valid \
    --stats-dir /path/to/KAI-New/stats --out runs/latent_serial/calib.json --n-ics 32 --ic-stride 7 --members 8
# 합성 결과의 기본 추천은 tangent_inflate: --mode tangent_inflate 를 붙이면 점수와 상관없이 그것을 씁니다
# (2) 보정한 예보
python scripts/infer_pgcfm.py --ckpt runs/latent_serial/latent.pt --spread runs/latent_serial/calib.json \
    --test-file /path/to/era-data/test/2020.nc --stats-dir /path/to/KAI-New/stats \
    --out runs/latent_serial/forecast_calib.nc --leads 10 --members 16 --ic-stride 7
# (3) 평가: 보정 전후 비교
python scripts/evaluate.py score --data-dir /path/to/era-data/test --year 2020 \
    --forecast "latent serial"=runs/latent_serial/forecast.nc --forecast "serial + calib"=runs/latent_serial/forecast_calib.nc \
    --climatology runs/clim_1980_2011.nc --out runs/eval/2020_step8
```

- 로그의 `[score]` 줄이 검증 후반부에서 각 보정의 CRPS / SSR / 정역학 잔차(1·5·10일)입니다. `[H2]` 줄의 `kept_*`가 1보다 작으면 Q가 같은 초기장에서도 분산을 줄이는 것입니다(합성에서는 1.57로 반대).
- 비용: `tangent_inflate`는 스텝마다 정역학 잔차의 vjp를 층 수만큼(13층이면 12번) 돌립니다. 느리면 `--no-tangent`. GPU 메모리가 부족하면 `--chunk 2`.
- calib.json의 `candidates`에 다른 보정의 값이 모두 있으므로, `mode`만 바꿔 다시 추론해 볼 수 있습니다.

합성 데이터 실험 재현 (CPU, 약 1.5시간, 7단계 결과 필요): `python scripts/run_step7_latent.py --out runs/step7 --arms lat_serial lat_serial_nophys` 후 `python scripts/run_step8_spread.py --out runs/step8 --step7 runs/step7`.

## 10. 9단계: classical 분산 보정 사다리 (`docs/step9_structure.md`)

9절과 같은 `fit_spread.py`가 이제 classical 보정도 함께 맞춥니다. 선택 기준의 기본값이 empirical CRPS로 바뀌었습니다(재보정된 멤버에는 fair CRPS가 과대분산을 고르기 때문). 추가된 후보는 `classical`(채널별 s_c), `classical_lead`(lead·채널별 s_kc), `classical_tan`(s_c의 변화를 정역학 접공간에 사영), `mbm`(lead·채널별 선형 평균 보정 a + b·μ와 s_kc), `mbm_tan`(mbm의 변화를 접공간에 사영)입니다. 모두 rollout이 끝난 뒤 적용합니다.

```bash
# (1) 적합: 8단계 후보 + classical 사다리, empirical CRPS 기준
python scripts/fit_spread.py --ckpt runs/latent_serial/latent.pt --valid-dir /path/to/era-data/valid \
    --stats-dir /path/to/KAI-New/stats --out runs/latent_serial/calib.json --n-ics 32 --ic-stride 7 --members 8
# 합성 결과의 추천: 물리 정합성까지 지키려면 --mode mbm_tan (또는 classical_tan)
# (2) 보정한 예보: calib.json의 mode만 바꿔 다른 후보도 바로 추론 가능
python scripts/infer_pgcfm.py --ckpt runs/latent_serial/latent.pt --spread runs/latent_serial/calib.json \
    --test-file /path/to/era-data/test/2020.nc --stats-dir /path/to/KAI-New/stats \
    --out runs/latent_serial/forecast_mbm.nc --leads 10 --members 16 --ic-stride 7
```

- `[score]` 줄의 `empirical` 값으로 비교하세요. `[fit] classical ... on the grid edge`가 0이 아니면 후보(0.5~6.0) 밖이 최적인 채널이 있다는 뜻입니다.
- `mbm`의 평균 보정은 lead별로 맞추므로 추론의 `--leads`가 적합한 `--leads`보다 길면 안 됩니다.
- 비용: `*_tan`은 lead마다 정역학 vjp 12번(`--no-tangent`로 끔).

합성 데이터 실험 재현 (CPU, 약 4시간, 7·8단계 결과 필요): `python scripts/run_step9_structure.py --out runs/step9 --step7 runs/step7`.

## 11. 12단계: ClimODE를 저희 데이터로 처음부터 학습 (`docs/step12_core.md`)

공개 체크포인트 대신 같은 데이터로 원본 크기 ClimODE를 다시 학습할 때 씁니다. CPU에서는 한 iteration에 수십 분이 걸려 GPU에서만 돌릴 수 있습니다.
`third_party/climode/train_global.py`는 원저자 코드(@ e729d23)를 바꾸지 않은 사본입니다.

```bash
python scripts/wb2_to_kai.py --out data/era5_5p625 --years 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018
cd data/era5_5p625 && mkdir -p Models
python ../../third_party/climode/train_global.py --scale 0 --batch_size 8 --spectral 0 --solver euler   # 300 epoch, 검증 손실 최저 epoch를 Models/에 저장
cd ../..
python scripts/run_climode_eval.py --data data/era5_5p625/era5_data --out runs/climode_ours \
    --ckpt data/era5_5p625/Models/ClimODE_global_euler_0_model_<epoch>.pt --lead-steps 24
```

- 원저자 스크립트는 학습 구간을 `slice('2006','2016')`로 잡아 검증 해 2016이 학습에 들어갑니다(원문 그대로 둠).
- 평가는 11단계와 같은 함수와 초기장입니다. `--max-ics`를 빼면 시험 해당 180개씩 360개 전부를 돕니다.

## 12. 19단계: 출력단 super-resolution (5.625° → 2.8125°, `docs/step19_sr_output.md`)

13단계 멤버 뒤에 붙는 SR 망(`pgcfm/sr.py`, Chiang et al. 2024 축소판)을 GPU에서 학습하고, 저장된 arm을 멤버별로 SR해 2.8125° 진실(WB2 128×64, 5변수)로 채점합니다. CPU 실행(4년, 31분)과 같은 코드이고 `--device`만 다릅니다.

```bash
# (1) 2.8125° 진실: ClimODE 5변수만, 5.625° 파일과 같은 era5_data 루트에 *_2.8125deg.nc 로 들어갑니다 (한 해 240 MB; 37년 8.9 GB)
python scripts/wb2_to_kai.py --out data/era5_5p625 --grid 128x64 --climode-only --years $(seq 1979 2018)
# (2) SR 학습 + 저장된 arm 채점 (한 번에). --ens 는 여러 arm 가능, --save-sr 은 SR한 멤버를 npz로 (arm·해마다 0.7 GB)
python scripts/run_step19_sr.py --coarse data/era5_5p625/era5_data --fine data/era5_5p625/era5_data --device auto \
    --train-range 1979 2015 --val-year 2016 --test-years 2017 2018 --iters 20000 --batch 32 --chunk 30 \
    --ens runs/step13_best/seqregress_pinn runs/step13_best/seqfm_dct_pinn_regm_band \
          runs/step13_best/seqfm_dct_pinn_regm_sband_clim_flow \
    --out runs/step13_best/sr --fig runs/step13_best/fig --save-sr
# 또는 best 스크립트의 한 단계로: 위 두 명령을 $OUT/sr 에, 채점된 arm 전부에 대해 실행합니다
SR=1 PROFILE=broad bash scripts/run_step13_best.sh          # SR_TRAIN_YEARS="2006 ... 2015" 로 학습 연수를 줄일 수 있음
```

- 학습 데이터는 호스트 메모리에 올립니다: 한 해에 coarse 60 MB + fine 120 MB(float16), 37년이면 약 6.6 GB. 141k 파라미터라 GPU 메모리는 배치 32에서 1 GB 미만입니다. `--amp`(bf16)는 선택.
- 채점은 `--chunk` 초기장씩 GPU에서 합니다(한 chunk = chunk × 8 멤버 × 6 lead × 5변수 × 64 × 128 float32 ≈ chunk × 63 MB, 보간·SR·평균 세 벌). 24 GB GPU면 30, CPU면 6.
- CRPS는 properscoring과 같은 추정량(E|X−y| − ½ E|X−X'|, 쌍 평균 M²)을 torch로 계산합니다(차이 1e-6 확인). 눈금은 ClimODE 최소·최대 정규화(5.625° 2006~2018)입니다.
- 결과: `sr.json`의 `evaluation.analysis[해]`(분석장 보간/SR RMSE), `evaluation.ensemble[arm][해]`(lead × 변수: `rmse_i`/`rmse_s`/`rmse_sm`(평균의 SR)/`rmse_pers`/`rmse_c`(5.625°), `spread2_*`, `crps_*`), `train.val_curve`. 그림 `fig_sr_sr_<arm>.png`(arm이 하나면 `fig_sr_sr.png`): z500·t2m RMSE, t2m·u10 CRPS, 보간 파랑 / SR 빨강 / 5.625° 회색 점선 / 지속 점선.
- 이미 학습한 망으로 다른 arm만 채점: `--eval-only --out <sr.pt 가 있는 폴더> --ens <새 arm ...>` (json의 `evaluation` 만 다시 씀).
- 읽는 법(CPU 결과, `docs/step19_sr_output.md`): 2.8°에서 t2m RMSE·CRPS만 내려가고(24 h −10 %, 144 h −3 %), z500·t850은 ±0.1 %, 바람은 퍼짐만 커집니다. GPU 37년 학습에서 바뀔 수 있는 것은 SR 망의 용량(검증 비 0.76이 평평해진 지점)과 멤버의 2.8° 오차이며, SR이 5.625° 점수를 바꾸지는 않습니다.

## 13. 20단계: blend_band를 ClimODE 연도로 GPU에서 (`docs/step20_pinn_score.md`)

소유자 요청(2026-10-28 04:25): "ClimODE와 비교하게 GPU에서 37년이 아닌 ClimODE 정도의 데이터로". `PROFILE=climode`(학습 2006~2015, 검증 2016, 시험 2017~2018 ClimODE IC 360개)에 `BLEND=1`을 붙이면 13단계 체인 끝에 20단계 단계(G(x) 코어 학습 → 코어 예측 → 혼합 보정 → 채점·energy score·그림)가 이어집니다.

```bash
# 한 번에: 자료 → PINN → FM → 회귀 → regm_band(+σ 장, 대역×셀) → 채점 → [BLEND] G(x) 코어 → 예측 → blend_band → 채점
BLEND=1 PROFILE=climode DATA=data/era5_5p625 OUT=runs/climode_blend bash scripts/run_step13_best.sh
# 더 가볍게: 퍼짐장 arm 없이 (regm_band와 blend_band만)
BLEND=1 SIGMA=0 PROFILE=climode OUT=runs/climode_blend bash scripts/run_step13_best.sh
# 이미 13단계 결과가 있는 폴더면 그 OUT을 가리키면 됩니다: 있는 단계는 건너뛰고 BLEND 단계만 돕니다
BLEND=1 PROFILE=climode OUT=runs/step13_climode bash scripts/run_step13_best.sh
```

BLEND 단계가 실제로 실행하는 명령(손으로 돌릴 때; `$K=$DATA/kai6h`, `$E=$DATA/era5_data`, `$OUT`은 13단계 결과 폴더):

```bash
# (1) G(x) 특징 코어: 66채널 + G(x) 48채널 → ResNet 경향 → Heun 6 h, rollout 손실 1→2→4→8 step 커리큘럼, 물리 손실 없음
python scripts/run_step16_dynpinn.py --kai $K --era5 $E --out $OUT --tag dyn_gfeat --gfeat --lambda-p 0 \
    --train-years $(seq -s ' ' 2006 2015) --val-year 2016 --K 8 --curriculum 1 2 4 8 --val-K 24 --val-chunk 4 \
    --iters 20000 --batch 16 --n-val 16 --n-val-train 8 --val-every 1000 --device auto --cache $OUT/cache --threads 4
# (2) 코어 예측: ClimODE IC(2 + 8e)에서 24..144 h, 검증 해와 시험 해 → $OUT/dyn_gfeat_pred.npz, dyn_gfeat_pred_2016.npz
python scripts/run_step16_predict.py --kai $K --era5 $E --run $OUT --tag dyn_gfeat --val-year 2016 --test-years 2017 2018
# (3) 혼합 보정: μ* = a + b·코어 + c·회귀 평균 + d·x(t) (lead × 변수, 2016 최소제곱) + FM·PINN 편차 × 대역 Γ
python scripts/run_step13_guide.py --ens $OUT/seqfm_dct_pinn --era5 $E --out $OUT --core $OUT/dyn_gfeat \
    --reg $OUT/seqregress_pinn --reg-mbm --blend --val-year 2016 --test-years 2017 2018 --save-arms blend_band
# (4) 채점(회귀 기준 부트스트랩), energy score, 2×3 그림
python scripts/score_step13.py --era5 $E --out $OUT/score_blend.json --ref $OUT/seqregress_pinn \
    --arms $OUT/seqregress_pinn $OUT/seqfm_dct_pinn_regm_band $OUT/seqfm_dct_pinn_blend_band
python scripts/step20_energy_score.py --era5 $E --out $OUT/energy_blend.json \
    --arms $OUT/seqregress_pinn $OUT/seqfm_dct_pinn_regm_band $OUT/seqfm_dct_pinn_blend_band
python scripts/plot_step13.py --era5 $E --scores $OUT/score_blend.json --figures fig_step20 --out $OUT/fig --step "Step 20"
```

- 시간(추정): 코어 20000 × 16은 GPU에서 1~2시간(CPU 4년 4000 × 4가 76분; G(x)는 substep마다 한 번 계산해 Heun 두 단계가 공유). 예측 수 분, 보정 약 30분(CPU, 연산자 적합 포함), 채점 5분. `CORE_ITERS`, `CORE_BATCH`, `CORE_VAL_EVERY`로 바꿀 수 있고, 트레이너는 15단계의 재개(`.train`), best 선택, `MAX_HOURS` 예산을 그대로 씁니다.
- 읽는 법: `score_blend.json`의 `arms[...]["rmse"|"crps"]`(lead 6 × 변수 5)와 `bootstrap_vs_seqregress_pinn`; 혼합 계수는 `guide_seqfm_dct_pinn_blend_band.json`의 `blend_coef`(순서 1, 코어, 회귀, x₀); ClimODE 논문 수치는 그림의 점으로 들어갑니다(`docs/step11/climode_paper_tables.json`: z500 193/479/784).
- 공정한 비교를 위해 적어 둘 것: ClimODE는 단일 결정론 모델(논문은 2006~2016 학습, 검증 해가 학습에 포함), blend_band의 평균은 두 모델의 lead별 가중 평균 + x(t) 항입니다. 단일 모델 비교는 `dyn_gfeat` 자체의 예측(`dyn_gfeat_pred.json`의 RMSE)과 회귀 평균으로 합니다. 이득의 귀속(물리 특징 G(x) 대 두 모델 혼합)은 같은 혼합을 `--gfeat` 없이 학습한 코어(`--tag dyn_lp0`, `--core $OUT/dyn_lp0`)로 한 번 더 돌려 가립니다(CPU 4년 결과: 20단계 문서 2.2).
