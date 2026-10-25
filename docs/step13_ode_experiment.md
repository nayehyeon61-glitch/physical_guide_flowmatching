# Step13 모델에서 ODE 보정 비교

기준은 사용자 PR #10의 `276a61f6f963c7eb231931733c6629b363e986f4`이며,
실험 브랜치는 `experiment/step13-spherical-ode`입니다. 기존 로컬 ODE 작업은
Git stash와 `/workspace/setup-backups/ode-prototype-before-step13`에 보존한 뒤
새 브랜치에 가져왔습니다.

## 실행 범위

실제 `runs/step13` 체크포인트와 ERA5 파일은 이 컨테이너에 없었습니다.
공개 WeatherBench2 ERA5의 `storage.googleapis.com` 요청도 네트워크 정책에서
403으로 차단되었습니다. 따라서 **합성 데이터에서 원래 모델의 학습·평가
파이프라인을 실행**했습니다. 실제 ERA5 성능이나 물리적 예측 개선을 입증한
실험이 아닙니다. 필요한 도메인은 환경 설정 초안에 추가했으며 runtime 허용은
별도로 적용되어야 합니다.

| 항목 | 실행 설정 |
| --- | --- |
| 모델 | 원래 `PINNProjector`와 `SeqFM`, 새로 학습 |
| 변수 | KAI 66 dynamic 채널, orography는 static |
| 격자 | 전지구 32×64, 5.625°, cell centre |
| 예측 | `K=60`, 6시간 간격, 360시간 = 15일 |
| 합성 자료 | split별 독립 seed, 각 96개 시점; 관측 ERA5 아님 |
| split | 2014 라벨 train / 2015 validation / 2016 test, 중복 없음 |
| CPU 예산 | PINN 20회·hidden 8, FM 40회·model_channels 32·단일 scale |
| 평가 표본 | test 초기조건 1개, member 2개, seed 0 |
| 보정 비교 | FM / 기존 PINN / 수송 ODE / nudging-only / persistence |

동일한 FM sample을 모든 보정 arm에 사용합니다. 정규화·DCT statistics·물리
residual scale·climatology는 training split에서만 계산합니다. 하이퍼파라미터를
test 점수로 선택하지 않았습니다. 이 예산과 표본 수는 실행 검증용이며,
통계적 우열이나 충분히 학습한 모델의 성능을 판단하기에 부족합니다.

첫 pilot의 PINN은 20회에 한 쌍씩만 선택해 일부 lead가 물리 손실에 포함되지
않았습니다. 최종 v2에서는 회당 3개 pair를 순환 선택해 60개 transition을
모두 다룹니다. pilot은 `runs/step13/ode_experiment`, 최종 실행은
`runs/step13/ode_experiment_v2`에 별도로 보존합니다.

## 원래 모델에 연결한 방법

`SeqFM.sample`의 반환값은 이미 시간·공간 IDCT와 lead scaling을 복원한
normalized increment `[B*M,K,66,H,W]`입니다. decoder/IDCT를 다시 적용하지
않습니다. 물리 상태를 만들 때 정확한 KAI 역변환인 `std + KAI_STD_EPS`를
사용합니다. 기존의 `K=24` 가중치를 `K=60`에 억지로 load하지 않습니다.

새 `pgcfm/ode_corrector.py`는 12개 pressure level의 **q/T 24채널**을 보정합니다.
나머지 z/u/v/surface 및 TISR은 원래 FM 결과를 그대로 유지합니다. 수송풍은
각 pressure level에 대응하는 FM의 u/v를 6시간 사이에서 보간하며, 실제 미래
관측을 사용하지 않습니다. 바람 cap은 ±80m/s이며 clipping 비율을 기록합니다.

보정은 관측 초기 상태에서 출발해 매 구간의 직전 보정 상태를 이어갑니다.
FM의 전체 미래 경로는 piecewise-linear reference로 사용합니다. ODE는

\[
\dot q=-V\cdot\nabla_s q+\lambda(r_{FM}(t)-q)
\]

형식의 **수평 scalar transport + reference nudging**입니다. 기존
`torchdiffeq.odeint` RK4가 적분하며, 새 solver는 만들지 않았습니다.
내부 step은 요청한 300초와 수송/nudging bound 중 작은 값으로 정합니다.
각 pressure level의 수직 수송·가열·응결·완전한 운동량 방정식이나 학습된
source closure를 포함한 대기 core가 아닙니다. 이 실험에서 학습한 것은 원래
PINN과 FM이고, ODE transport 자체를 별도로 pretrain하지는 않았습니다.

`transport_strength=0`인 nudging-only arm이 반드시 필요합니다. ODE 대 FM의
오차 감소가 smoothing/lag/spread 감소 때문일 수 있으므로, 추가 수송 효과는
nudging-only와 비교해야 합니다. q/tcwv nonnegativity는 기존
`hard_constraints`로 모든 생성 arm에 공통 적용합니다.

원래 `pgcfm/seqdct.py`, `pgcfm/ens_guide.py`, `pgcfm/ens_to_det.py`는 변경하지
않았습니다. 특히 소유자 operator는 사용·재학습·수정하지 않았습니다.

## 재실행

Python 3.12 / PyTorch 2.9.1 CPU 환경에서 실행했습니다.

```bash
cd /workspace/physical_guide_flowmatching
export UV_CACHE_DIR=/workspace/.cache/uv
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export XDG_CACHE_HOME=/workspace/.cache
uv pip install --python /workspace/.venvs/physical-guide/bin/python -e '.[dev,dct,step13]'

/workspace/.venvs/physical-guide/bin/python scripts/make_step13_synthetic6h.py \
  --out data/step13_synthetic6h --steps 96 --years 2014 2015 2016 --seed 43

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/workspace/.venvs/physical-guide/bin/python scripts/run_step13_ode_experiment.py \
  --kai data/step13_synthetic6h/kai6h \
  --era5 data/step13_synthetic6h/era5_data \
  --data-kind synthetic --provenance data/step13_synthetic6h/manifest.json \
  --out runs/step13/ode_experiment_v2 \
  --K 60 --pinn-iters 20 --pinn-pairs 3 --fm-iters 40 \
  --members 2 --n-eval 1 --threads 4 --eval-tag comparison
```

기존 출력을 덮어쓰지 않도록 새 `--out` 또는 `--eval-tag`를 지정합니다.
checkpoint 재사용은 `--reuse-checkpoints`를 명시하고, 학습 설정과 자료 SHA가
일치할 때만 허용합니다. 데이터를 다시 생성할 때도 다른 `--out`을 쓰거나,
기존 파일을 바꾸려는 의도가 있을 때만 `--overwrite`를 지정하십시오.

`runs/`와 `data/`는 Git 대상에서 제외합니다. checkpoint, 전체 66채널 ensemble
NPZ, 정규화 통계, loss 이력, source SHA, 실행 log는 로컬 artifact로 보존합니다.

## 결과와 해석

최종 수치·그래프·CSV는 [합성 실행 결과](step13_ode_results/synthetic/README.md)에
저장했습니다. 관련 테스트 **72개가 통과**했고 PINN pair coverage는 60/60입니다.
15일의 전체 66채널 normalized RMSE는 FM 1.798226, PINN 1.798182,
ODE 1.796036, nudging-only 1.796472, persistence 1.213174였습니다.
ODE의 FM 대비 약 0.12% 감소 중 상당 부분이 nudging-only에서도 나타납니다.
수송의 추가 감소는 약 0.024%이고 q850에서는 nudging-only보다 오히려
조금 나빴습니다. 따라서 이 작은 표본에서 물리 수송의 유의미한 예측 이점을
확인했다고 해석하지 않습니다. 짧게 학습한 FM은 이 합성 자료의 persistence
보다 전체 오차가 컸습니다.

최종 실행은 약 153초였으며 checkpoint와 전체 ensemble은 약 353 MiB입니다.
별도 CPU 테스트가 실행 일부와 겹쳤으므로 JSON의 arm별 wall time은 통제된
속도 benchmark가 아닙니다. ODE는 4,320개 내부 step, 17,280번 RHS 평가를
완료했고, 이 합성 자료에서는 wind clipping이 발생하지 않았습니다.

실제 ERA5를 확보하면 동일한 비교를 충분한 train/validation/test
기간과 많은 IC/member, 여러 seed로 반복해야 합니다. 그 전에는 합성 데이터의
RMSE/CRPS나 primitive residual을 실제 기후 성능·물리 정합성으로 해석하지
않습니다.
