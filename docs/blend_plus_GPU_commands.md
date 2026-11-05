# blend_band+ 학습·시험 명령어 (GPU 서버, lustre ERA5 사용)

브랜치 `claude/project-thread-v2zibi`. 이 브랜치는 `claude/project-thread-43gbu8`(step 26까지)에 step 27 비교 코드를 얹은 것입니다.

ERA5는 `/lustre/home/mahmed/ERA5_0p25_DAILY`에서 **읽기만** 합니다. 0.25°를 5.625°(64x32)로 conservative 평균해서 `data/era5_5p625`에 씁니다.
- 연산은 WB2 64x32와 같은 정확한 면적 가중 평균입니다. 그래도 원자료가 달라 docs/results_2026_11 점수와 소수점 차이가 날 수 있습니다.
- 데이터가 이미 있는 연도는 각 단계가 건너뜁니다. 그래서 WB2를 다시 내려받지 않습니다.

## 0. 코드 받기와 환경

```bash
git clone https://github.com/nayehyeon61-glitch/physical_guide_flowmatching.git
cd physical_guide_flowmatching
git checkout claude/project-thread-v2zibi
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128     # 서버 CUDA에 맞춰 cu12x 선택
pip install -r requirements.txt
pip install git+https://github.com/patrick-kidger/torchcubicspline.git    # requirements.txt에 없음 (ClimODE 코드가 필요)
mkdir -p logs
```

## 1. 데이터: lustre 0.25° → 5.625°

먼저 파일 구조를 확인합니다. 하루 파일 하나에 00/06/12/18 UTC, 13개 층, z q t u v u10 v10 msl tcwv sp skt t2m tisr가 있어야 합니다.

```bash
SRC=/lustre/home/mahmed/ERA5_0p25_DAILY
python scripts/local_era5_to_kai.py --src $SRC --out data/era5_5p625 --check --years 2016
#   파일이 연도 폴더 안에 있으면:  --pattern "{year}/{ymd}.nc"
#   "NOT OK"가 나오면 그 출력을 그대로 보내 주세요. 특히 일평균만 있으면 6시간 모델에 쓸 수 없습니다.
```

OK가 나오면 변환합니다. 범위는 학습 1979~2015, 검증 2016, 시험 2017~18, hold-out 2020이며, 2006~2015는 기후값으로도 씁니다.

```bash
nohup python scripts/local_era5_to_kai.py --src $SRC --out data/era5_5p625 \
    --years $(seq 1979 2020) --workers 20 > logs/data.log 2>&1 &
```

## 2. 기본 모델 blend_band (PROFILE=ngcm, 1979~2015) 학습과 2017~18 시험

이미 `runs/blend_ngcm`에 학습된 결과가 있으면 2단계는 건너뛰어도 됩니다. 끝난 단계는 자동으로 건너뜁니다.

```bash
export DATA=data/era5_5p625 DEVICE=cuda
nohup env BLEND=1 SIGMA=0 PROFILE=ngcm OUT=runs/blend_ngcm bash scripts/run_step13_best.sh > logs/base.log 2>&1 &
```

## 3. 2020 hold-out 멤버 (고정 가중치, 재학습 없음)

```bash
nohup env SKIP_TRAIN=1 SKIP_NGCM=1 OUT=runs/blend_ngcm bash scripts/run_ngcm_vs_neuralgcm.sh > logs/y2020.log 2>&1 &
```

## 4. ClimODE 2020 예측 (짝 비교 기준, 공개 체크포인트 추론만)

```bash
python scripts/run_climode_2020.py runs/blend_ngcm/ngcm2020/climode2020 --lead-steps 24 --threads 8 > logs/climode2020.log 2>&1
```

## 5. + 모델 학습과 시험 (2017~18 + 2020 hold-out)

```bash
DRYRUN=1 BASE=runs/blend_ngcm OUT=runs/blend_plus bash scripts/run_blend_plus.sh     # 명령 확인
nohup env BASE=runs/blend_ngcm OUT=runs/blend_plus bash scripts/run_blend_plus.sh > logs/plus.log 2>&1 &
#   시간 제한이 있는 작업이면 MAX_HOURS=<h> 를 주고, 같은 명령을 다시 실행하면 이어서 학습합니다 (exit 75).
```

결과 파일:
- `runs/blend_plus/score_plus.json`: 2017~18 결과
- `runs/blend_plus/y2020/score_plus_2020.json`, `runs/blend_plus/y2020/climode_paired_plus_2020.json`: 2020 hold-out 결과
- `runs/blend_plus/fig/fig_step26.png`: 그림

## 6. 약점 점검 (재학습 없음)

```bash
nohup bash scripts/run_plus_weakness.sh > logs/weak.log 2>&1 &
```

결과는 `runs/blend_plus/weak2020/`에 쌓입니다. 각 단계(W1~W4)의 내용은 `docs/step27_recent_models.md` 5절에 있습니다.

## 보내 주실 파일 (모두 작음)

- `runs/blend_plus/*.json`, `runs/blend_plus/fig/`
- `runs/blend_plus/y2020/*.json`
- `runs/blend_plus/weak2020/*.json`, `runs/blend_plus/weak2020/*.png`, `runs/blend_plus/weak2020/step27/`
- 학습 시간: `runs/blend_plus/plus_long.json`, `runs/blend_plus/plus_short.json`
