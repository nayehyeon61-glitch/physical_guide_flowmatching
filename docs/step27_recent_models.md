# Step 27: ClimODE 외 최신 모델과 비교 (2020 hold-out)

소유자 요청 (2026-11-04): "climate ode 말고 그 수준에서 최신 비교가능한 모델들과도 비교", "25~26년 연구 모델로".

## 1. 같은 IC, 같은 정답, 같은 점수식

- IC: `docs/results_2026_11`과 같은 2020년 179개 (12 UTC 2일 간격, 2020-08-30 제외).
- 정답: WB2 ERA5 64x32 conservative. 저희 `data/era5_5p625`를 만든 zarr와 같습니다.
- 점수: `scripts/compare_recent/common.py`. 기존 `compare_step13_climode.py`와 같은 식입니다.
  - persistence의 RMSE, ACC, CRPS가 기존 json과 소수 셋째 자리까지 같아 검증했습니다.
  - CRPS는 ClimODE min-max 단위(2006~2018)로 계산한 뒤 (max - min)을 곱해 물리 단위로 바꿨습니다.
- 앙상블은 모두 8멤버(저희와 같은 수)로 점수를 냈습니다. 50멤버 점수도 `per_ic/*.npz`에 있습니다.

## 2. WeatherBench2 공개 예측 (상위 기준선, 실행 없이 내려받기만)

`python scripts/compare_recent/wb2_forecasts.py` → `docs/step27/per_ic/<model>.npz`.
각 파일은 IC별 rmse, acc, spread, crps_ens, crps_w 값이며 모양은 (179, 6 lead, 5 var)입니다.

표: [`step27/table_2020.md`](step27/table_2020.md), 그림: `step27/fig_recent_2020.png`

이 모델들은 모두 0.25°~1°, 여러 층의 초기장을 씁니다. 저희는 5.625°, 12층입니다.
step 23의 같은 정보량 시험(NeuralGCM을 5.625° IC에서 출발)에서 24 h 격차의 81 %가 IC 정보에서 나왔습니다.
그래서 이 표는 같은 체급 비교가 아니라 상한선입니다.

## 3. 5.625° 급 연구 모델 (2024~2026)

| 모델 | 공개 상태 | 논문 수치 (시험 해) |
|---|---|---|
| Andrae et al. 2025 (ICLR), 확산 앙상블 | 코드와 체크포인트 공개 | z500 120 h RMSE 544, CRPS 243 (2018) |
| CoDiCast 2025 (IJCAI) | 코드만 공개, 확산 가중치 없음 | z500 24/72/144 h 187 / 452 / 758 (2017~18) |
| WeatherODE 2024~25 | 코드만 공개 | z500 24/72 h 115 / 325 (2017~18) |
| ClimPhyDM 2026.08 | 코드 없음 | 그림만 있음 |
| FlowCast-ODE 2025.09 (1°) | 코드 없음 | 그림만 있음 |

저희 37년 모델의 2017~18 값은 144 / 355 / 618입니다.
CoDiCast와 ClimODE보다는 좋고, WeatherODE보다는 24~72 h에서 나쁩니다.

Andrae 모델은 `scripts/compare_recent/run_andrae_cef.py`로 2020 IC에서 재현할 예정이며, 외부 코드 실행 승인을 기다리고 있습니다.

## 4. blend_band+ 넣기

서버에서 `run_blend_plus.sh`가 끝나면 아래처럼 실행합니다.

```
python scripts/compare_recent/report.py --extra runs/blend_plus/y2020/climode_paired_plus_2020.json:plus_blend_band:"ours blend_band+"
```
