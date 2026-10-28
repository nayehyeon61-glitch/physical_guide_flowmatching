# GPU 서버에서 처음부터: 클론 → 데이터 → 학습 → 채점 (best 모델 blend_band)

소유자 요청(2026-10-28 05:24): "실험할 수 있도록 git에서 받는 것부터 학습까지의 명령어". 아래를 위에서부터 순서대로 치면
ClimODE 연도(학습 2006~2015, 검증 2016, 시험 2017~2018)로 13단계 사슬과 20단계 best 모델(blend_band)까지 한 번에 돕니다.
학습 구조는 `docs/step20/fig_system_blend.png`, 결과 해석은 `docs/step20_pinn_score.md`, 단계별 명령은 `docs/RUN_ON_GPU.md` 13절입니다.

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
git checkout claude/project-thread-43gbu8      # PR #10: 11~21단계 (best 모델 blend_band 포함)
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
- 덤: `CONTROL=1` 회귀 편차 대조군, `CLIMODE=1` ClimODE 공개 체크포인트 짝 비교(초기장 쌍당 약 18분), `SR=1` 2.8125° 출력단 SR (`docs/RUN_ON_GPU.md` 12절).

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
