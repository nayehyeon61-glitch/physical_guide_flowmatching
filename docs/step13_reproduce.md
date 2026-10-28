# 13단계 best model 직접 학습하고 실행하기

보고 모델 regm_band(`docs/step13/fig_system_best.png`)와 그 위에 흐름 조건 퍼짐장을 더한 regm_flow, regm_flowop(`docs/step13/fig_system_sigma.png`), 그리고 대역 × 셀 기후 배율에 regm_flow의 사례 배율을 곱한 regm_sband_clim_flow를 데이터 내려받기부터 점수까지 다시 만드는 방법입니다. 확률 점수가 가장 좋은 것은 regm_sband_clim_flow입니다. 결과와 해석은 `docs/step13_sequence.md` 4.2, 4.3, 4.4, 5절에 있습니다.

## 1. 코드와 환경

    git clone https://github.com/nayehyeon61-glitch/physical_guide_flowmatching.git
    cd physical_guide_flowmatching
    git checkout claude/project-thread-43gbu8      # PR #10 (아직 병합 전, #9 위에 쌓임)
    pip install -r requirements.txt

- Python 3.13, torch 2.14에서 확인했습니다.
- 13단계 스크립트는 기본이 CPU입니다(`--threads`로 코어 수 지정). GPU, 37년 학습, 검증과 재개는 15단계(`docs/step15_training_system.md`, `PROFILE=broad`)를 보세요. 4코어, 15 GB 메모리에서 잰 시간이 아래 표에 있습니다. 학습 중에는 메모리를 약 7.6 GB 쓰므로 두 단계를 동시에 돌리지 마세요.
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
| `SIGMA=0` | 켬 (1) | 흐름 조건 퍼짐장 팔(regm_flow, regm_flowop, regm_sband_clim_flow)을 빼고 regm_band만 만듦 |
| `SBAND=0` | 켬 (1) | regm_sband_clim_flow를 빼고 만듦 (약 15분 절약) |
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
| 3. 보정 | `run_step13_guide.py --ens seqfm_dct_pinn --reg seqregress_pinn --reg-mbm --flow-sigma --flow-op --sigma-full-epochs 30 --save-arms regm_band regm_flow regm_flowop` | `seqfm_dct_pinn_regm_{band,flow,flowop}_ens_{2017,2018}.npz`, `guide_*.json` (Γ, 대역 에너지, σ 학습 기록) | 약 1시간 10분 (`SIGMA=0`이면 약 26분) |
| 3b. 대역 × 셀 | `run_step13_aband.py --ens seqfm_dct_pinn --reg seqregress_pinn --save-arms regm_sband_clim_flow` (`SIGMA=1`, `SBAND=1`일 때) | `seqfm_dct_pinn_regm_sband_clim_flow_ens_{2017,2018}.npz`, `aband_seqfm_dct_pinn.json` (소유자의 사례 적응 대역·대역 × 셀 망과 대조군 점수) | 약 15분 |
| 4. 채점 | `score_step13.py --ref seqregress_pinn --arms seqregress_pinn seqfm_dct_pinn_regm_band …regm_flow …regm_flowop …regm_sband_clim_flow` | `score_best.json` | 약 5분 |
| 5. 흐름 의존성 | `diag_step13_flowdep.py --leads 24 48 72 96 120 144` | `flowdep.json` (퍼짐 변동계수, 퍼짐 오분위별 오차) | 1분 |
| 6. 퍼짐장 비교 | `score_step13.py --ref …regm_band`, `plot_step13_flowdep.py` (`SIGMA=1`일 때) | `score_sigma.json`, `fig/fig_flowdep.png` | 약 4분 |
| 7. 출력단 SR (`SR=1`일 때) | `wb2_to_kai.py --grid 128x64 --climode-only`, `run_step19_sr.py --ens <채점한 arm 전부>` | `era5_data/*/*_2.8125deg.nc`, `sr/sr.json`, `sr/sr.pt`, `fig/fig_sr_sr_<arm>.png` | 2.8° 자료 받기 + 학습 31분(4년, CPU) / GPU 수 분 + arm·해마다 채점 CPU 15분 |

학습은 2012–2015년, 보정(a, b, c, Γ, σ 장)은 2016년, 시험은 2017–2018년 ClimODE 초기장 360개입니다. 데이터 내려받기를 빼고 약 6–7시간입니다.

## 4. 학습과 시험을 나눠서 실행하기

한 번에 돌리는 대신 단계마다 직접 실행할 수 있습니다. 스크립트가 하는 것과 같은 명령입니다.

    D=data/era5_5p625; O=runs/step13_best; K=$D/kai6h; E=$D/era5_data

    # 데이터 (2012-2018 6시간 KAI + ClimODE 5변수, 2006-2011은 기후값용)
    python scripts/wb2_to_kai.py --out $D --all-hours --years 2012 2013 2014 2015 2016 2017 2018
    python scripts/wb2_to_kai.py --out $D --years 2006 2007 2008 2009 2010 2011

    # 학습 (2012-2015 학습, 2016 검증)
    python scripts/run_step13_pinn.py --kai $K --era5 $E --out $O                       # 1단계 PINN, 고정용
    python scripts/run_step13_seqfm.py --kai $K --era5 $E --out $O --basis dct \
        --pinn $O/pinn_project.pt --save-ens --resume-ens                                # 2단계 FM + 2016-2018 앙상블
    python scripts/run_step13_seqfm.py --kai $K --era5 $E --out $O --model regress \
        --pinn $O/pinn_project.pt --save-ens --resume-ens                                # 회귀 (평균의 출처)

    # 보정 (2016에 맞추고 2017-2018에 적용): regm_band, regm_flow, regm_flowop
    python scripts/run_step13_guide.py --ens $O/seqfm_dct_pinn --era5 $E --out $O --reg $O/seqregress_pinn \
        --reg-mbm --flow-sigma --flow-op --sigma-full-epochs 30 --save-arms regm_band regm_flow regm_flowop
    # 대역 × 셀 기후 배율 × regm_flow 사례 배율 (위 실행이 남긴 flow_sigma_seqfm_dct_pinn.pt를 씀)
    python scripts/run_step13_aband.py --ens $O/seqfm_dct_pinn --reg $O/seqregress_pinn --era5 $E --out $O \
        --save-arms regm_sband_clim_flow

    # 시험 (2017-2018 ClimODE 초기장 360개)
    python scripts/score_step13.py --era5 $E --out $O/score_best.json --ref $O/seqregress_pinn \
        --arms $O/seqregress_pinn $O/seqfm_dct_pinn_regm_band $O/seqfm_dct_pinn_regm_flow $O/seqfm_dct_pinn_regm_flowop \
        $O/seqfm_dct_pinn_regm_sband_clim_flow
    python scripts/diag_step13_flowdep.py --runs $O --leads 24 48 72 96 120 144 \
        --arms seqfm_dct_pinn seqfm_dct_pinn_regm_band seqfm_dct_pinn_regm_flow seqfm_dct_pinn_regm_flowop \
        seqfm_dct_pinn_regm_sband_clim_flow

`run_step13_seqfm.py`는 한 번 실행에서 학습과 앙상블 생성을 함께 합니다. 출력 폴더에 `seqfm_dct.pt`(회귀는 `seqregress.pt`)가 있으면 학습을 건너뛰고 그 가중치로 생성만 하므로, 학습만 다시 하고 싶으면 그 파일을 지우고, 시험만 하고 싶으면 아래 5절처럼 가중치를 넣어 두면 됩니다. 보정 단계는 매번 2016년에서 a, b, c, Γ, σ 장을 다시 맞춥니다. σ 장 망의 가중치는 기록용으로 `flow_sigma_<앙상블>.pt`, `flow_op_<앙상블>.pt`에 남습니다(6절).

## 5. 학습을 건너뛰고 받은 가중치로 실행

프로젝트 파일의 `step13/step13_checkpoints.zip`(102 MB)에 이 문서의 수치를 낸 가중치가 있습니다. 학습 단계의 `pinn_project.pt`, `pinn_project.json`, `seqfm_dct.pt`, `seqregress.pt`, `stats/`와, 6절의 신경연산자 가중치 `operator_seqfm_dct_pinn.pt`, `operator_mean_only_seqfm_dct_pinn.pt`, `flow_sigma_seqfm_dct_pinn.pt`, `flow_op_seqfm_dct_pinn.pt`입니다. 출력 폴더에 풀고 실행하면 세 학습 단계가 가중치를 불러오고, 생성·보정·채점만 돌아갑니다(약 4시간, `SIGMA=0`이면 약 3시간). 연산자와 σ 장 가중치는 실행 스크립트가 다시 학습하므로 불러 쓰려면 6절의 코드나 `run_step13_operator.py --load-existing`을 씁니다. 이 네 파일을 만든 재학습의 점수도 regm_flow 0.17877, regm_flowop 0.17896(정규화 fair CRPS)으로 원래 실행과 같았습니다.

    mkdir -p runs/step13_best && unzip step13_checkpoints.zip -d runs/step13_best
    bash scripts/run_step13_best.sh

생성 잡음은 고정 seed에서 해마다 같은 순서로 나오므로, 같은 가중치로 모든 초기장을 돌리면 이 문서와 같은 앙상블이 나옵니다. 받은 가중치로 돌린 확인 실행에서 2016년 FM과 회귀 멤버가 원래 실행과 비트 단위로 같았습니다. `SMOKE=1`처럼 초기장 수를 줄이면 다음 해의 잡음 순서가 바뀌어 같지 않습니다. 처음부터 학습하면 seed는 같지만 스레드 수나 라이브러리 버전에 따라 학습 결과가 조금 달라질 수 있습니다.

## 6. 신경연산자 실험

소유자의 연산자 `EnsembleToDeterministic`(`pgcfm/ens_to_det.py`, 공식 neuraloperator FNO)은 두 곳에서 학습됩니다. 둘 다 2–4절의 앙상블(`seqfm_dct_pinn_ens_*.npz`, `seqregress_pinn_ens_*.npz`)이 출력 폴더에 있어야 합니다.

| 실험 | 명령 | 학습 방향과 손실 | 저장되는 가중치 | 시간 |
| --- | --- | --- | --- | --- |
| 3단계: 앙상블 → 결정론 예보 | `run_step13_operator.py --ens $O/seqfm_dct_pinn` (회귀 앙상블은 `--ens $O/seqregress_pinn`) | 출력 = 앙상블 평균 + 보정(헤드 0 초기화). 위도 가중 MSE, Adam 3e-3, 배치 8, 40 epoch, 2016년 마지막 33개 초기장으로 체크포인트. 대조군: 같은 모델에 평균만 입력 | `operator_<앙상블>.pt`, `operator_mean_only_<앙상블>.pt` | 약 45분 |
| 4.3: 연산자 σ 장 | `run_step13_guide.py … --reg-mbm --flow-op --sigma-full-epochs 30 --save-arms regm_band regm_flowop` | 위와 같은 사전학습을 regm_band 멤버에서 한 뒤, 출력 헤드 앞 특징에 새 σ 헤드: ClimODE 가우시안 NLL, 헤드만 30 epoch, 그다음 전체를 lr 1e-4로 30 epoch | `flow_op_<앙상블>.pt` (사전학습 연산자, 미세조정한 OperatorSigma, ρ) | 약 1시간 |

    O=runs/step13_best; E=data/era5_5p625/era5_data
    python scripts/run_step13_operator.py --ens $O/seqfm_dct_pinn --era5 $E --out $O --width 32 --modes 8 --epochs 40
    python scripts/run_step13_guide.py --ens $O/seqfm_dct_pinn --era5 $E --out $O --reg $O/seqregress_pinn --reg-mbm \
        --flow-op --sigma-full-epochs 30 --save-arms regm_band regm_flowop
    python scripts/score_step13.py --era5 $E --out $O/score_op.json --ref $O/seqfm_dct_pinn_regm_band \
        --arms $O/seqfm_dct_pinn_regm_band $O/seqfm_dct_pinn_regm_flowop

첫 명령의 결과는 `operator_seqfm_dct_pinn.json`과 로그 마지막 줄들(앙상블 평균, a + b·μ + c·x0, 연산자, 평균만 입력, persistence의 z500·t850 RMSE)입니다. 이 문서의 값은 z500 144 h 앙상블 평균 997, a + b·μ + c·x0 912, 연산자 839, 평균만 835입니다. 같은 앙상블로 처음부터 다시 학습한 결과도 lead마다 똑같았습니다(seed 고정, 4스레드).

바꿔 볼 수 있는 것:
- `--width`, `--modes`, `--epochs`(사전학습), `--sigma-epochs`(σ 헤드만), `--sigma-full-epochs`(전체 미세조정). `--quick`은 몇 분짜리 점검입니다.
- 입력: σ 버전의 맥락은 `run_step13_guide.py`의 `op_context`(x(t), lead, 연중·하루 시각), 3단계는 `run_step13_operator.py`의 `samples`(x(t), lead)에서 만듭니다. 지형·해륙 같은 정적 장은 아직 넣지 않았습니다.
- 손실: `pgcfm/ens_to_det.py`의 `fit`에는 물리 손실 자리(`physics_fn`, `physics_weight`)가 있지만 지금은 0입니다.

저장한 가중치 다시 불러오기:

    import torch
    from pgcfm.ens_to_det import EnsembleToDeterministic
    from pgcfm.flow_sigma import OperatorSigma
    d = torch.load("runs/step13_best/operator_seqfm_dct_pinn.pt", weights_only=False)
    op = EnsembleToDeterministic(**d["config"]); op.load_state_dict(d["state_dict"])  # 입력은 (x - norm_mean) / norm_std
    f = torch.load("runs/step13_best/flow_op_seqfm_dct_pinn.pt", weights_only=False)
    sig = OperatorSigma(EnsembleToDeterministic(**f["config"]), f["operator_sigma"]["sigma_clim"])
    sig.load_state_dict(f["operator_sigma"])  # sig(members, context, lead_idx) -> σ, 배율은 σ / f["rho"][lead_idx]

## 7. 결과 확인

`$OUT/score.log`의 마지막 줄들에 팔마다 z500 CRPS, SSR, 작은 규모 파워, 멤버 z–t 상관이 나옵니다. 이 문서의 수치(360개 초기장)는 다음과 같습니다.

| 팔 | z500 CRPS 24 / 72 / 144 h | z500 RMSE 24 / 72 / 144 h | SSR z500 | 멤버 z–t 상관 144 h |
| --- | --- | --- | --- | --- |
| `seqregress_pinn` (회귀 + 통계 오차) | 0.0109 / 0.0230 / 0.0351 | 244 / 528 / 817 | 1.06–1.34 | 0.41 |
| `seqfm_dct_pinn_regm_band` (보고 모델) | 0.0108 / 0.0226 / 0.0327 | 244 / 522 / 778 | 1.13–1.21 | 0.51 |
| `seqfm_dct_pinn_regm_flow` (+ ClimODE 망 σ 장) | 0.0108 / 0.0226 / 0.0326 | 244 / 522 / 778 | 1.14–1.23 | 0.51 |
| `seqfm_dct_pinn_regm_flowop` (+ 연산자 σ 장, 30 epoch) | 0.0108 / 0.0226 / 0.0327 | 244 / 522 / 778 | 1.14–1.22 | 0.51 |
| `seqfm_dct_pinn_regm_sband_clim_flow` (대역 × 셀 기후 배율 × regm_flow 배율) | 0.0107 / 0.0225 / 0.0326 | 244 / 522 / 778 | 1.19–1.26 | 0.49 |

5변수 CRPS는 regm_band 0.02580, regm_flow 0.02564(−0.63 %), regm_flowop 0.02566(−0.54 %), regm_sband_clim_flow 0.02550(−1.17 %)입니다. 퍼짐장의 효과는 CRPS보다 `flowdep.json`에서 잘 보입니다. 퍼짐이 가장 큰 사례 20%와 가장 작은 20%의 오차 비(`bin_rmse` 마지막 / 처음)가 z500 24 h에서 regm_band 1.05, regm_flow 1.29, regm_flowop 1.18입니다.

`score_best.json`의 `bootstrap_vs_seqregress_pinn`에는 회귀 대비 CRPS 차이와 95% 구간이 lead·변수마다 있습니다. 그래프는 기존 그림 스크립트로 그릴 수 있습니다(`docs/step13_sequence.md` 6절의 명령에 `--scores $OUT/score_best.json`).

## 8. 알아 둘 것

- 저장된 앙상블에는 ClimODE의 5변수(z500, t850, t2m, u10, v10)만 24 h 간격으로 들어 있습니다. 66채널 전체 멤버가 필요하면 `run_step13_seqfm.py`에서 저장할 변수를 늘려야 합니다.
- 보정 단계는 매번 2016년에서 a, b, c와 Γ를 닫힌 형으로, σ 장 망은 NLL로 다시 맞춥니다(학습된 가중치가 따로 없음). 다른 해에 적용하려면 그 해의 FM·회귀 앙상블을 같은 방법으로 만든 뒤 `--test-years`를 바꾸면 됩니다.
- σ 장 망은 seed가 고정되어 있지만, 스레드 수에 따라 학습 결과가 조금 달라질 수 있습니다. 이 문서의 regm_flow, regm_flowop 수치는 4스레드 실행입니다.
- 출력단 SR(19단계)은 `SR=1`로 같은 스크립트에 붙습니다. 채점된 arm 전부를 멤버별로 2.8125°로 올려 2.8° 진실로 채점하고(`$OUT/sr/sr.json`), `SR_SAVE=1`이면 SR한 멤버도 저장합니다. GPU 설정과 결과 읽는 법은 `docs/RUN_ON_GPU.md` 12절.
- GPU에서 더 오래 학습하려면 `--device cuda`(또는 `PROFILE=broad`)를 쓰세요. 15단계에서 장치 옵션, memmap 캐시, 검증과 best 선택, 시간 예산 재개가 들어갔습니다(`docs/step15_training_system.md`).
