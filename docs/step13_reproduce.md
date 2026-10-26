# 13단계 best model(regm_band) 직접 학습하고 실행하기

보고 모델 regm_band(`docs/step13/fig_system_best.png`)를 데이터 내려받기부터 점수까지 다시 만드는 방법입니다. 결과와 해석은 `docs/step13_sequence.md` 4.2절과 5절에 있습니다.

## 1. 코드와 환경

    git clone https://github.com/nayehyeon61-glitch/physical_guide_flowmatching.git
    cd physical_guide_flowmatching
    git checkout claude/project-thread-43gbu8      # PR #10 (아직 병합 전, #9 위에 쌓임)
    pip install -r requirements.txt

- Python 3.13, torch 2.14에서 확인했습니다.
- 13단계 스크립트는 CPU 전용입니다(`--threads`로 코어 수 지정). 4코어, 15 GB 메모리에서 잰 시간이 아래 표에 있습니다. 학습 중에는 메모리를 약 7.6 GB 쓰므로 두 단계를 동시에 돌리지 마세요.
- 디스크는 약 10 GB가 필요합니다(데이터 6 GB, 앙상블 npz 약 2.7 GB, 체크포인트 0.1 GB).
- 데이터를 내려받으려면 `storage.googleapis.com`에 접속할 수 있어야 합니다.

## 2. 한 번에 실행

    bash scripts/run_step13_best.sh

환경 변수로 바꿀 수 있는 것:

| 변수 | 기본값 | 뜻 |
| --- | --- | --- |
| `DATA` | `data/era5_5p625` | ERA5 폴더 (`kai6h/`, `era5_data/`가 생김) |
| `OUT` | `runs/step13_best` | 체크포인트, 앙상블, 점수 |
| `THREADS` | 4 | CPU 스레드 |
| `CONTROL=1` | 끔 | 논문 대조군: 같은 평균과 Γ를 회귀 자신의 편차에 씌운 팔도 만들고 함께 채점 |
| `CLIMODE=1` | 끔 | ClimODE 공개 체크포인트를 시험 해마다 첫 8개 초기장에서 돌리고 같은 초기장으로 짝 비교 (초기장 쌍당 약 18분, 모두 약 2.5시간) |
| `SMOKE=1` | 끔 | 반복 2회, 해마다 초기장 6개: 전체 사슬이 돌아가는지만 확인 (데이터가 있으면 약 15분, 수치는 의미 없음) |

출력이 이미 있는 단계는 건너뜁니다. 중간에 멈추면 같은 명령을 다시 실행하면 이어서 돌아갑니다. 학습 단계는 50–100 반복마다 `.train` 체크포인트도 남깁니다. 단계마다 로그가 `$OUT/<단계>.log`에 남고, 실패하면 그 로그의 마지막 20줄을 보여 줍니다.

## 3. 단계별 내용 (스크립트가 하는 일)

| 단계 | 명령 | 만드는 것 | CPU 4코어 시간 |
| --- | --- | --- | --- |
| 0. 데이터 | `wb2_to_kai.py --all-hours --years 2012 … 2018`, 그리고 `--years 2006 … 2011` | `kai6h/<해>.nc` (KAI 66채널, 6시간), `era5_data/` (ClimODE 5변수, 채점·기후값) | 네트워크에 따라 수십 분–수 시간 |
| 1. PINN | `run_step13_pinn.py` (project 목적, 1000 반복) | `pinn_project.pt/.json`, 정규화 `stats/` | 47분 |
| 2a. FM | `run_step13_seqfm.py --basis dct --pinn … --save-ens` (3000 반복, 멤버 8) | `seqfm_dct.pt`, `seqfm_dct_pinn_ens_{2016,2017,2018}.npz` | 학습 70분 + 생성 해마다 30–50분 |
| 2b. 회귀 | `run_step13_seqfm.py --model regress --pinn … --save-ens` | `seqregress.pt`, `seqregress_pinn_ens_*.npz` | 학습 20분 + 생성 해마다 약 5분 |
| 3. 보정 | `run_step13_guide.py --ens seqfm_dct_pinn --reg seqregress_pinn --reg-mbm --save-arms regm_band` | `seqfm_dct_pinn_regm_band_ens_{2017,2018}.npz`, `guide_*.json` (Γ, 대역 에너지) | 약 26분 |
| 4. 채점 | `score_step13.py --ref seqregress_pinn --arms seqregress_pinn seqfm_dct_pinn_regm_band` | `score_best.json` | 약 5분 |

학습은 2012–2015년, 보정(a, b, c와 Γ)은 2016년, 시험은 2017–2018년 ClimODE 초기장 360개입니다. 데이터 내려받기를 빼고 약 5–6시간입니다.

## 4. 학습을 건너뛰고 받은 가중치로 실행

프로젝트 파일의 `step13/step13_checkpoints.zip`(95 MB)에 이 문서의 수치를 낸 가중치가 있습니다. `pinn_project.pt`, `pinn_project.json`, `seqfm_dct.pt`, `seqregress.pt`, `stats/`입니다. 출력 폴더에 풀고 실행하면 세 학습 단계가 가중치를 불러오고, 생성·보정·채점만 돌아갑니다(약 3시간).

    mkdir -p runs/step13_best && unzip step13_checkpoints.zip -d runs/step13_best
    bash scripts/run_step13_best.sh

생성 잡음은 고정 seed에서 해마다 같은 순서로 나오므로, 같은 가중치로 모든 초기장을 돌리면 이 문서와 같은 앙상블이 나옵니다. 받은 가중치로 돌린 확인 실행에서 2016년 FM과 회귀 멤버가 원래 실행과 비트 단위로 같았습니다. `SMOKE=1`처럼 초기장 수를 줄이면 다음 해의 잡음 순서가 바뀌어 같지 않습니다. 처음부터 학습하면 seed는 같지만 스레드 수나 라이브러리 버전에 따라 학습 결과가 조금 달라질 수 있습니다.

## 5. 결과 확인

`$OUT/score.log`의 마지막 줄들에 팔마다 z500 CRPS, SSR, 작은 규모 파워, 멤버 z–t 상관이 나옵니다. 이 문서의 수치(360개 초기장)는 다음과 같습니다.

| 팔 | z500 CRPS 24 / 72 / 144 h | z500 RMSE 24 / 72 / 144 h | SSR z500 | 멤버 z–t 상관 144 h |
| --- | --- | --- | --- | --- |
| `seqregress_pinn` (회귀 + 통계 오차) | 0.0109 / 0.0230 / 0.0351 | 244 / 528 / 817 | 1.06–1.34 | 0.41 |
| `seqfm_dct_pinn_regm_band` (보고 모델) | 0.0108 / 0.0226 / 0.0327 | 244 / 522 / 778 | 1.13–1.21 | 0.51 |

`score_best.json`의 `bootstrap_vs_seqregress_pinn`에는 회귀 대비 CRPS 차이와 95% 구간이 lead·변수마다 있습니다. 그래프는 기존 그림 스크립트로 그릴 수 있습니다(`docs/step13_sequence.md` 6절의 명령에 `--scores $OUT/score_best.json`).

## 6. 알아 둘 것

- 저장된 앙상블에는 ClimODE의 5변수(z500, t850, t2m, u10, v10)만 24 h 간격으로 들어 있습니다. 66채널 전체 멤버가 필요하면 `run_step13_seqfm.py`에서 저장할 변수를 늘려야 합니다.
- 보정 단계는 매번 2016년에서 a, b, c와 Γ를 닫힌 형으로 다시 맞춥니다(학습된 가중치가 따로 없음). 다른 해에 적용하려면 그 해의 FM·회귀 앙상블을 같은 방법으로 만든 뒤 `--test-years`를 바꾸면 됩니다.
- GPU에서 더 오래 학습하려면(약점 1번, FM 학습량) 학습 스크립트에 장치 옵션을 먼저 넣어야 합니다. 라이브러리(`pgcfm/seqdct.py`)는 대부분 입력 텐서의 장치를 따르지만, 스크립트가 CPU로 고정되어 있습니다.
