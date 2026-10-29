# GPU 서버에서 처음부터: 클론 → 데이터 → 학습 → 채점 (best 모델 blend_band)

소유자 요청(2026-10-28 05:24): "실험할 수 있도록 git에서 받는 것부터 학습까지의 명령어". 아래를 위에서부터 순서대로 치면
ClimODE 연도(학습 2006~2015, 검증 2016, 시험 2017~2018)로 13단계 사슬과 20단계 best 모델(blend_band)까지 한 번에 돕니다.
학습 구조는 `docs/step20/fig_system_blend.png`, 결과 해석은 `docs/step20_pinn_score.md`, 단계별 명령은 `docs/RUN_ON_GPU.md` 13절입니다.

## 요약 (2026-10-29): blend_band(G(x) 코어 혼합)를 GPU로 학습하는 최신 명령

소유자 요청(2026-10-29 01:25): "g core로 나온 188, 즉 best 모델을 학습할 최신 명령어"와 "ClimODE 수준으로 학습시킨 명령어".
1~2절(클론, 환경)을 마친 뒤 아래를 위에서부터 칩니다. 셋 다 같은 `DATA` 폴더를 쓰고(있는 해는 다시 받지 않음) 결과는 각자의 `OUT`에 씁니다.

```bash
mkdir -p runs
# (0) GPU 첫 점검 (15~20분, 수치는 의미 없음). cuda 경로는 GPU 없는 곳에서 만들었으므로 이것이 첫 GPU 실행입니다.
SMOKE=1 BLEND=1 DEVICE=cuda DATA=data/era5_5p625 bash scripts/run_step13_best.sh

# (A) 188 모델 그대로: CPU 실행과 같은 설정 (학습 2012~2015, PINN 1000 × 4, FM·회귀 3000 × 4, G(x) 코어 4000 × 4)
nohup env BLEND=1 SIGMA=0 DEVICE=cuda DATA=data/era5_5p625 OUT=runs/blend_4y \
    bash scripts/run_step13_best.sh > runs/blend_4y.log 2>&1 &

# (B) ClimODE 수준: ClimODE의 학습 연도 (학습 2006~2015, 검증 2016, 시험 2017~2018) + 37년 실행과 같은 반복 수
#     (PINN 8000 × 8, FM·회귀 40000 × 16, G(x) 코어 20000 × 16, 1000 반복마다 검증해 best 선택, memmap 캐시)
nohup env BLEND=1 SIGMA=0 PROFILE=climode DEVICE=cuda DATA=data/era5_5p625 OUT=runs/blend_climode \
    bash scripts/run_step13_best.sh > runs/blend_climode.log 2>&1 &
```

- (A)는 짧으니 먼저 돌리고, 끝난 뒤 (B)를 돌리세요 (같은 GPU에서 동시에 돌리지 않기).
- (A)의 기준값(CPU, 시험 IC 360개): blend_band z500 RMSE 188 / 451 / 723 (24 / 72 / 144 h), 5변수 CRPS 0.02449.
  GPU 부동소수점과 정규화 통계(내려받은 파일에서 다시 계산)가 달라 비트 단위로 같지는 않지만 가까워야 합니다. 크게 다르면 GPU 경로의 버그이니 로그를 보내 주세요.
- (B)의 비교 대상: (A), ClimODE 논문 193 / 479 / 784, 소유자 GPU 37년 regm_band 170 / 423 / 741.
  반복 수가 37년 실행과 같으므로 (B)의 regm_band 대 37년 regm_band는 데이터 양(10년 대 37년)만 다른 비교입니다 (24단계의 "데이터 대 반복 수" 혼동을 풉니다).
- `SIGMA=0`은 blend_band에 쓰지 않는 퍼짐장 팔(regm_flow, regm_flowop, regm_sband_clim_flow)을 빼서 CPU 보정 시간을 약 1시간 줄입니다. regm_band와 blend_band는 그대로 나옵니다.
- GPU에서 도는 단계: pinn_project, seqfm_dct(학습과 멤버 생성), seqregress, core_gfeat. CPU에서 도는 단계: core_predict(약 6분), guide_regm·guide_blend(약 30분씩), 채점.
- 결과: `$OUT/score_blend.json`의 `arms["seqfm_dct_pinn_blend_band"]`, 그림 `$OUT/fig/fig_step20.png`, 혼합 계수 `$OUT/guide_seqfm_dct_pinn_blend_band.json`, 코어 단독 `$OUT/dyn_gfeat_pred.json`.
- 선택: job 시간 제한이 있으면 `MAX_HOURS=11`(저장 후 종료 코드 75, 같은 명령으로 재개). `CLIMODE=1 CLIMODE_ICS=180`은 ClimODE 체크포인트를 시험 IC 360개 전부에서 돌려 짝 비교. `PROFILE=broad`는 같은 명령을 37년으로. ClimODE의 학습량(300 epoch, 우리 batch 16으로 약 11000 반복)까지 맞추려면 (B)에 `GEN_ITERS=11000`. `AMP=1`은 GPU에서 검증되지 않았으니 끄고 시작하세요.
- 보내 주실 파일: `score_blend.json`, `energy_blend.json`, `guide_seqfm_dct_pinn_blend_band.json`, `dyn_gfeat.json`, `*_metrics.csv`, `manifest.jsonl`, `*.log`.

## 0. 준비물

| 항목 | 필요한 것 |
| --- | --- |
| GPU | NVIDIA 1장, 메모리 16 GB 이상 권장 (FM·회귀 40000 반복 × batch 16, 코어 20000 × 16) |
| RAM | 32 GB 이상 권장 (학습 연도를 memmap 캐시로 읽음; 캐시 없이면 10년 ≈ 8 GB를 RAM에 올림) |
| 디스크 | 40 GB (6시간 데이터 13년 ≈ 14 GB, 캐시 ≈ 10 GB, 앙상블 npz ≈ 4 GB, 체크포인트·로그 < 1 GB) |
| 네트워크 | `storage.googleapis.com` (WeatherBench2 zarr) 접속 |
| Python | 3.10 이상 (3.13 + torch 2.14에서 확인) |

cuda 경로(`--device auto`, `CACHE=1`)는 GPU가 없는 환경에서 만들었으므로 4절의 스모크가 첫 GPU 검증입니다.

## 1. 코드

```bash
git clone https://github.com/nayehyeon61-glitch/physical_guide_flowmatching.git
cd physical_guide_flowmatching
git checkout claude/project-thread-43gbu8      # PR #10: 11~24단계 (best 모델 blend_band 포함)
```

## 2. 환경

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# 서버 CUDA에 맞는 torch를 먼저: nvidia-smi 의 CUDA 버전에 맞춰 cu126 / cu128 / cu130 중 하나
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 두 번째가 True 여야 함
python -m pytest -q                                                             # CPU에서 수 분, 전부 통과
```

## 3. 데이터

따로 받을 필요가 없습니다. `run_step13_best.sh`의 첫 단계가 WeatherBench2 zarr에서 필요한 해만 받아
`$DATA/kai6h/<해>.nc`(66채널 KAI 6시간, 해마다 약 1 GB)와 `$DATA/era5_data/`(ClimODE 5변수, 채점·기후값)를 만듭니다.
이미 있는 해는 건너뛰므로 스모크와 본 실행이 같은 `DATA` 폴더를 써도 됩니다. 미리 받아 두려면:

```bash
python scripts/wb2_to_kai.py --out data/era5_5p625 --all-hours --years $(seq -s ' ' 2006 2018)
```

## 4. 스모크 (GPU·데이터·사슬 확인, 반복 2회, 해마다 초기장 6개)

```bash
SMOKE=1 BLEND=1 DEVICE=auto DATA=data/era5_5p625 bash scripts/run_step13_best.sh
```

- 2012~2018년 6시간 자료(약 7 GB)와 2006~2011년 일 자료(기후값)를 받은 뒤 약 15~20분. `runs/step13_smoke/fig/fig_step20.png`가 생기면 통과입니다. 수치는 의미 없습니다.
- 실패하면 `runs/step13_smoke/<단계>.log`의 끝을 보면 됩니다. 단계 이름은 `runs/step13_smoke/manifest.jsonl`에 순서대로 적힙니다.

## 5. 본 실행 (ClimODE 연도, best 모델까지)

```bash
mkdir -p runs
nohup env BLEND=1 PROFILE=climode DATA=data/era5_5p625 OUT=runs/climode_blend \
    bash scripts/run_step13_best.sh > runs/climode_blend.log 2>&1 &
tail -f runs/climode_blend.log
```

단계 순서와 반복 수 (`PROFILE=climode`):

| 단계 | 하는 일 | 반복 × batch | 장치 |
| --- | --- | --- | --- |
| data_6h | 2006~2011년 6시간 자료 추가 (스모크가 2012~2018을 받아 둠) | | 네트워크 |
| pinn_project | 1단계 PINN 사영기, 정규화 `stats/` | 8000 × 8 | GPU |
| seqfm_dct | 시퀀스 FM 학습 + 2016~2018 멤버 8개 생성 (동결 PINN) | 40000 × 16 | GPU |
| seqregress | 같은 U-Net 회귀 + 통계 오차 멤버 | 40000 × 16 | GPU |
| guide_regm | regm_band(+ regm_flow, regm_flowop 퍼짐장) 보정, 2016 | | CPU, 약 30분 |
| aband | regm_sband_clim_flow (대역 × 셀) | | CPU, 약 15분 |
| core_gfeat | G(x) 특징 ODE 코어 (18단계) | 20000 × 16 | GPU |
| core_predict | 코어로 ClimODE 초기장 24~144 h 예측 (2016~2018) | | GPU |
| guide_blend | μ* = a + b·코어 + c·회귀 + d·x(t) + FM 편차 × 대역 Γ → **blend_band** | | CPU, 약 30분 |
| score_blend, energy_blend, fig_blend | 채점(회귀 기준 부트스트랩), energy score, 2×3 그림 | | CPU, 약 10분 |
| score, flowdep, score_sigma, fig_flowdep | 13단계 팔의 채점과 퍼짐 진단 | | CPU, 약 10분 |

- GPU 시간은 서버마다 다릅니다. 같은 반복 수의 37년 broad 실행을 돌리신 서버라면 그때의 `manifest.jsonl`(start, end)로 가늠하면 됩니다.
- **중단되면 같은 명령을 다시** 치면 됩니다. 산출물이 있는 단계는 건너뛰고, 학습 단계는 `<이름>.pt.train` 체크포인트에서 이어 갑니다.
- job 시간 제한이 있으면 `MAX_HOURS=11`을 붙입니다. 11시간마다 저장하고 종료 코드 75로 끝나며, 같은 명령을 다시 실행하면 이어집니다.
- 가볍게: `SIGMA=0` (퍼짐장 팔 없이 regm_band와 blend_band만, guide 30분 → 10분). `AMP=1`은 bf16 autocast인데 GPU에서 검증한 적이 없으니 먼저 끄고 돌리세요.
- 한 단계만 다시: 그 단계의 산출물을 지우고 같은 명령 (예: 코어 반복 수를 바꾸려면 `rm runs/climode_blend/dyn_gfeat*` 뒤 `CORE_ITERS=40000 ...`).
- 덤: `CONTROL=1` 회귀 편차 대조군, `SR=1` 2.8125° 출력단 SR (`docs/RUN_ON_GPU.md` 12절).
- **ClimODE 체크포인트와 짝 비교** (`docs/step22_climode_paired.md`): `CLIMODE=1`을 붙이면 저자들의 GPU 학습 체크포인트를 저자 코드로 같은 IC에서 돌려 모든 팔(BLEND=1이면 blend_band와 코어도)을 짝지어 채점하고 `fig/fig_climode_paired.png`를 그립니다. 기본은 각 해 첫 8개 IC(CPU에서 쌍당 약 18분)이고, GPU에서는 `CLIMODE_ICS=180`으로 시험 IC 전체(360개)를 돌리세요. 부분집합은 `CLIMODE_STRIDE=20 CLIMODE_OFFSET=10`(계절을 가로지른 9쌍)처럼 고릅니다.

## 6. 결과 읽기

| 파일 | 내용 |
| --- | --- |
| `runs/climode_blend/score_blend.json` | `arms[<arm>]["rmse"\|"crps"\|"ssr"]` (lead 6 × 변수 5: z, t, t2m, u10, v10), `bootstrap_vs_seqregress_pinn`; arm = seqregress_pinn, seqfm_dct_pinn_regm_band, seqfm_dct_pinn_blend_band, seqfm_dct_pinn_regm_sband_clim_flow |
| `runs/climode_blend/fig/fig_step20.png` | lead별 2×3 선 그림 (RMSE, CRPS, 퍼짐; ClimODE 논문 수치는 점, 지속성은 점선) |
| `runs/climode_blend/guide_seqfm_dct_pinn_blend_band.json` | `blend_coef` 혼합 계수 (순서 1, 코어, 회귀, x₀; lead × 변수) |
| `runs/climode_blend/energy_blend.json` | 5변수·z–t 쌍 fair energy score |
| `runs/climode_blend/score_best.json`, `fig/fig_flowdep.png` | 13단계 팔 (regm_band, flow, flowop, sband) 채점과 퍼짐–오차 진단 |
| `runs/climode_blend/dyn_gfeat_pred.json` | 코어 단독(단일 모델) RMSE: ClimODE와 1:1 비교용 |
| `runs/climode_blend/manifest.jsonl` | 단계별 명령, 커밋, 시작·끝, 종료 코드 |

표와 PDF 요약: `python scripts/report_step13_broad.py --runs runs/climode_blend --out docs/step13_climode` (reportlab 필요).

비교 기준 (CPU, 학습 2012~2015, 시험 2017~2018 초기장 360개, z500 RMSE 24/72/144 h):

| 모델 | z500 RMSE | 5변수 CRPS |
| --- | --- | --- |
| regm_band (CPU 4년) | 244 / 522 / 778 | 0.02580 |
| **blend_band (CPU 4년)** | **188 / 451 / 723** | **0.02449** |
| regm_band (소유자 GPU 37년) | 170 / 423 / 741 | |
| ClimODE (논문, 단일 모델) | 193 / 479 / 784 | |

## 7. 귀속 대조군 (논문에 같이 넣을 것)

G(x) 특징의 몫을 가리려면 같은 혼합을 데이터만의 코어(`--gfeat` 없음)로 한 번 더 돌립니다. CPU 4년 결과는 두 혼합이 같았습니다
(`docs/step20_pinn_score.md` 2.2). 본 실행이 끝난 뒤:

```bash
O=runs/climode_blend; K=data/era5_5p625/kai6h; E=data/era5_5p625/era5_data
python scripts/run_step16_dynpinn.py --kai $K --era5 $E --out $O --tag dyn_lp0 --lambda-p 0 \
    --train-years $(seq -s ' ' 2006 2015) --val-year 2016 --K 8 --curriculum 1 2 4 8 --val-K 24 --val-chunk 4 \
    --iters 20000 --batch 16 --n-val 16 --n-val-train 8 --val-every 1000 --device auto --cache $O/cache --threads 4
python scripts/run_step16_predict.py --kai $K --era5 $E --run $O --tag dyn_lp0 --val-year 2016 --test-years 2017 2018
python scripts/run_step13_guide.py --ens $O/seqfm_dct_pinn --era5 $E --out $O/ctrl_lp0 --core $O/dyn_lp0 \
    --reg $O/seqregress_pinn --reg-mbm --blend --val-year 2016 --test-years 2017 2018 --save-arms blend_band
# 같은 arm 이름이 겹치지 않게 이름을 바꿔 연결한 뒤 G(x) 혼합을 기준으로 짝 부트스트랩
for y in 2017 2018; do ln -s seqfm_dct_pinn_blend_band_ens_$y.npz $O/ctrl_lp0/seqfm_dct_pinn_blend_lp0_band_ens_$y.npz; done
python scripts/score_step13.py --era5 $E --out $O/ctrl_lp0/score_vs.json --ref $O/seqfm_dct_pinn_blend_band \
    --arms $O/ctrl_lp0/seqfm_dct_pinn_blend_lp0_band
```

`score_vs.json`의 `bootstrap_vs_seqfm_dct_pinn_blend_band`가 음수(G(x) 혼합이 더 나쁨)면 물리 특징의 몫은 0이고 이득은 두 모델 혼합입니다.

## 8. 정리하고 공유할 것

- `runs/climode_blend/score_blend.json`, `energy_blend.json`, `guide_seqfm_dct_pinn_blend_band.json`, `manifest.jsonl`, `fig/*.png`, `*.json`(학습 기록)만 있으면 모든 표와 그림을 다시 만들 수 있습니다. 앙상블 npz(해마다 arm당 228 MB)는 공유하지 않아도 됩니다.
- 체크포인트(`pinn_project.pt`, `seqfm_dct.pt`, `seqregress.pt`, `dyn_gfeat.pt`, `stats/`)를 다른 `OUT`에 넣으면 학습을 건너뛰고 생성·보정·채점만 다시 돕니다.

## 9. NeuralGCM 공개 체크포인트와 비교 (선택, 23단계)

NeuralGCM(Kochkov et al. 2024)의 체크포인트는 `gs://neuralgcm/models/v1/`(CC BY-SA 4.0)에 있고 학습 기간이 1979~2017(2.8°·1.4° 결정론) / 1979~2019(0.7°, 1.4° 확률론)라
우리 시험 해 2017~18이 그쪽 학습 자료입니다. 그래서 이 비교만 **시험 해 2020**으로 옮깁니다. 아무것도 다시 학습하지 않습니다:
NeuralGCM은 체크포인트 그대로 추론, 우리는 5절의 가중치와 2016년 보정 상수를 고정한 채 2020년 멤버만 새로 만듭니다.
비교 변수는 두 모델이 공유하는 z500·t850뿐이고(NeuralGCM에는 t2m·u10·v10이 없음), NeuralGCM 출력은 우리 5.625° 격자로 보존 재격자합니다.

```bash
pip install neuralgcm dinosaur-dycore gcsfs            # JAX는 GPU판으로: pip install -U "jax[cuda12]"
python scripts/wb2_to_kai.py --out data/era5_5p625 --all-hours --years 2020
O=runs/climode_blend; N=runs/climode_blend/ngcm; mkdir -p $N
for f in pinn_project.pt pinn_project.json seqfm_dct.pt seqregress.pt dyn_gfeat.pt dyn_gfeat.json stats \
         seqfm_dct_pinn_ens_2016.npz seqregress_pinn_ens_2016.npz; do ln -s ../$f $N/$f; done
# 2020년 멤버 (고정 가중치): FM, 회귀, G(x) 코어, 보정(regm_band, blend_band)
K=data/era5_5p625; E=data/era5_5p625/era5_data
C="--kai $K --era5 $E --out $N --pinn $N/pinn_project.pt --save-ens --resume-ens --train-years 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 --val-year 2016 --test-years 2020 --device cuda --val-every 0 --patience 0 --max-hours 0 --val-members 0"
python scripts/run_step13_seqfm.py $C --basis dct
python scripts/run_step13_seqfm.py $C --model regress
python scripts/run_step16_predict.py --kai $K --era5 $E --run $N --tag dyn_gfeat --val-year 2016 --test-years 2020
python scripts/run_step13_guide.py --ens $N/seqfm_dct_pinn --era5 $E --out $N --core $N/dyn_gfeat --reg $N/seqregress_pinn \
    --reg-mbm --blend --val-year 2016 --test-years 2020 --save-arms regm_band blend_band
# NeuralGCM: 2020년 ClimODE 규약 IC 전부(180개), ARCO ERA5 0.25°·37층을 내려받아 모델 격자로 재격자
python scripts/run_neuralgcm_eval.py --era5 $E --out $N/neuralgcm_2_8 --model v1/deterministic_2_8_deg --year 2020
python scripts/run_neuralgcm_eval.py --era5 $E --out $N/neuralgcm_ens --model v1/stochastic_1_4_deg --year 2020 --members 8
# 같은 IC에서 짝 비교 (각 체크포인트마다)
for R in neuralgcm_2_8 neuralgcm_ens; do
python scripts/compare_step13_climode.py --era5 $E --climode $N/$R/neuralgcm_pred.npz --ref-name "NeuralGCM ($R)" \
    --arms $N/seqregress_pinn $N/seqfm_dct_pinn_regm_band $N/seqfm_dct_pinn_blend_band --core $N/dyn_gfeat --out $N/${R}_paired.json
python scripts/plot_step13.py --era5 $E --scores $O/score_best.json --years 2020 --figures none --paired $N/${R}_paired.json --out $N/fig_$R
done
```

`--train-years`는 5절에서 쓴 것과 같아야 합니다(정규화 통계가 거기서 옵니다). CPU에서는 `--cpu --threads 4`로 같은 명령이 돌고 2.8° 결정론만 현실적입니다.
학습 자료·해상도·연산 자원이 다르니(NeuralGCM: 0.25°·37층 ERA5 39년, TPU; 우리: 5.625°·12층) 결과는 "이긴다"가 아니라 상한 참조로 적습니다(`docs/step23_neuralgcm.md`).
