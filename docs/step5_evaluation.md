# 5단계: NeuralGCM 기준 평가 스위트와 그림 — 구현, 작은 테스트, 약점 분석

기준: NeuralGCM 논문 (Kochkov et al. 2024, Nature 632, `s41586-024-07744-y (4).pdf`)의 평가 절차.
NeuralGCM은 WeatherBench2(WB2) 프로토콜을 따르므로, 지표는 직접 짜지 않고
[google-research/weatherbench2](https://github.com/google-research/weatherbench2)의 코드를 vendoring해서 그대로 호출합니다.

## 1. NeuralGCM 평가 항목과 구현 대응

| NeuralGCM 항목 | 위치 (논문) | 이 저장소 구현 | WB2 재사용 |
| --- | --- | --- | --- |
| RMSE (ensemble-mean) | Fig. 2a | `scores.score_forecast` → `rmse` | `metrics.MSE` |
| RMSB (시간평균 bias의 공간 RMS) | Fig. 2c | `rmsb` | `metrics.SpatialBias` |
| CRPS | Fig. 2e | `crps` (결정론 예측은 CRPS = MAE) | `metrics.CRPS`, `metrics.MAE` |
| Spread-skill ratio | Fig. 2g | `ssr` = √((M+1)/M) · spread / RMSE | `metrics.EnsembleVariance` |
| RMSE·CRPS 상대 점수 (%) | Fig. 2 세로축 | `scores.skill_vs` (climatology, persistence 기준) | — |
| 공간 지도 (RMSE, bias, CRPS, SSR) | Fig. 2b,d,f,h | `scores.spatial_maps` | `SpatialMSE`, `SpatialBias`, `SpatialCRPS`, `SpatialEnsembleVariance` |
| ACC (WB2 headline) | WB2 | `acc` (day-of-year climatology 기준) | `metrics.ACC` |
| Rank histogram | 보정 확인 | `scores.rank_histogram` | `metrics.RankHistogram` |
| Power spectra (blurriness) | Supp. Fig. 17–18 | `physics.zonal_spectrum`, `spectral_ratio` | `derived_variables.ZonalEnergySpectrum` |
| Geostrophic / ageostrophic wind 균형 | Ext. Data Fig. 3 | `physics.geostrophic_balance` (20–70°) | `GeostrophicWindSpeed`, `AgeostrophicWindSpeed` |
| Water budget | "Water budget" 절 | `physics.global_budgets` (전구 tcwv, sp, 질량가중 T) | — |
| 정역학 균형 (NeuralGCM 4번째 식) | Methods, 원시방정식 | `physics.hydrostatic_residual` | — |
| **추가**: tendency 진폭 비 | PR #2 교훈 | `physics.tendency_amplitude_ratio` | — |

Headline 변수는 NeuralGCM Fig. 2와 같게 z500, t850, q700, u850, v850, t2m, msl이고, 영역은 global / extratropics(|φ|>20°) / tropics입니다.
가중치는 WB2의 cell-area 위도 가중치(극점 포함 격자에서도 정확)입니다.

기준선(자동 추가): **persistence**, **climatology**(day-of-year, WB2 `compute_daily_stat_fast`, 61일 창).
모델 쪽 기준선: **zero-noise**(ε = 0 단일 경로), KAI 결정론 예측은 `--forecast`로 넣습니다.

## 2. 사용법

```bash
# 1회: 학습 기간 climatology (cluster)
python scripts/evaluate.py climatology --data-dir <era5 dir> --years 1980-2011 --out clim_1980_2011.nc
# 채점
python scripts/evaluate.py score --data-dir <era5 dir> --year 2020 \
    --forecast KAI=kai/forecast.nc --forecast PG-CFM=pgcfm/ensemble.nc \
    --forecast "PG-CFM eps=0"=pgcfm/zero_noise.nc --climatology clim_1980_2011.nc --out eval/2020
```

예측 파일은 KAI `src/inference.py` 형식(`forecast(initial_condition, lead_time, channel, lat, lon)`, 역정규화, lead 0 = IC)이며,
PG-CFM ensemble은 `lead_time` 뒤에 `member` 차원을 추가합니다. 출력: `scores.csv`(model, region, metric, variable, lead_days, value),
`physics.csv`, `figures/fig1–7`.

> 주의: upstream KAI의 원본 `inference.py`로 만든 forecast.nc는 lead가 하루 밀려 있습니다(PR #1에서 수정). 수정 전 파일은 다시 만들어야 합니다.

## 3. 작은 테스트: 균형 잡힌 toy 대기 + 실패 모드 6종

`pgcfm/eval/synthetic.py`: 1.5° 121×240, 67채널. T는 3개 연직 모드의 AR(1) 이상치(동쪽 이류 + 감쇠), z는 msl과 T에서
이산 측고 방정식으로 정확히 적분, 바람은 그 z의 지균풍(적도 taper) + 작은 비지균 성분, q·tcwv·sp 등은 진단합니다.
진짜 대기는 아니지만 **정역학·지균 균형이 성립하고, 잠재 상태를 member별로 확률 전파하면 보정된 ensemble이 됨**을 알고 있는 세계입니다.

모델 (각각 한 가지 성질만 일부러 망가뜨림):

| 모델 | 무엇을 흉내내나 |
| --- | --- |
| perfect | 참 동역학 + 새 noise (보정된 CFM의 이상형) |
| zero-noise | ε = 0 경로 (조건부 평균) |
| underdispersive | noise × 0.3 (물리 손실이 spread를 줄이는 경우, 논문 7절 한계 4) |
| quasi-static | 한 step에 참 tendency의 10%만 이동 (climate_diffusion 실패, PR #2) |
| unbalanced | 채널별 iid noise로 spread를 만듦 (z, t, u, v 섭동이 서로 독립 → 균형 붕괴) |
| blurry | perfect member를 공간 평활 (소규모 에너지 손실) |

실행: `python scripts/eval_synthetic_demo.py --n-ic 3 --leads 6 --members 8` (CPU 약 3분). 결과는 `docs/eval/synthetic/`.

### 결과 (z500, global, lead 6일 / 물리 진단은 lead 평균)

| 모델 | RMSE | ACC | CRPS | SSR | tendency 비 (1일 → 6일) | 정역학 잔차 RMS (m²/s²) | ageo/geo (500 hPa) | 소규모 에너지 비 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| truth | – | – | – | – | 1 | 0.003 | 0.10 | 1 |
| perfect | 925 | 0.26 | **493** | **1.01** | 0.99 → 1.01 | 0.003 | 0.09 | 1.07 |
| zero-noise | **863** | **0.36** | 694 | – | 0.78 → 0.36 | 0.003 | 0.09 | 0.39 |
| underdispersive | 870 | 0.35 | 577 | 0.32 | 0.80 → 0.45 | 0.003 | 0.09 | 0.46 |
| quasi-static | 1227 | 0.00 | 819 | 0.30 | 0.20 → 0.19 | 0.003 | 0.10 | 0.53 |
| unbalanced | 890 | 0.31 | 512 | 0.65 | 0.89 → 0.74 | **221** | **0.43** | 1.30 |
| blurry | 907 | 0.26 | 499 | 0.83 | 0.88 → 0.67 | 0.039 | 0.13 | **0.15** |
| persistence | 1325 | 0.01 | 1054 | – | 0 | – | – | – |
| climatology | 927 | – | 745 | – | – | – | – | – |

그림: `docs/eval/synthetic/figures/`
- `fig1_scores_vs_lead.png` RMSE / CRPS / SSR / ACC vs lead (NeuralGCM Fig. 2 배치)
- `fig2_scorecard_crps_vs_persistence.png`, `fig3_scorecard_rmse_vs_climatology.png` 상대 점수 scorecard
- `fig4_rank_histograms.png`, `fig5_z500_spectra.png`, `fig6_physical_consistency.png`(tendency 비, 정역학, 지균, 질량), `fig7_maps.png`

## 4. 지표가 드러낸 약점 (평가 체계 자체 + PG-CFM에 대한 함의)

### A. RMSE·ACC는 나쁜 ensemble을 더 좋게 평가한다
1. zero-noise와 underdispersive가 RMSE·ACC에서 **perfect보다 좋습니다** (z500 6일 RMSE 863·870 vs 925, ACC 0.36 vs 0.26).
   조건부 평균은 RMSE 최적이기 때문입니다. CRPS로 보면 순서가 바로잡힙니다 (493 < 577 < 694).
   → PG-CFM의 주 지표는 **CRPS + SSR**이고, RMSE는 ensemble-mean으로만, 결정론 모델과는 CRPS(=MAE)로 비교합니다. NeuralGCM이 Fig. 2에서 같은 이유로 CRPS를 주 지표로 씁니다.
2. 같은 이유로 "PINN을 넣었더니 RMSE가 내려갔다"는 spread 붕괴(underdispersion)의 신호일 수 있습니다. 반드시 SSR과 tendency 비를 같이 봐야 합니다.

### B. CRPS와 SSR은 물리 균형 붕괴와 blur를 거의 못 본다 (가장 중요한 결과)
3. **unbalanced**는 CRPS가 perfect보다 4%만 나쁘고 RMSE는 오히려 좋지만, 정역학 잔차가 7만 배(221 vs 0.003 m²/s²), 비지균/지균 비가 4배(0.43 vs 0.10)입니다.
   즉 skill 점수만으로는 "physics guide가 효과가 있다"를 **보일 수도, 반박할 수도 없습니다**.
   → 논문 주장(물리 구성으로 예측력 향상)을 검증하려면 정역학·지균 진단과 PINN residual을 **skill과 같은 급의 결과**로 보고해야 합니다. 이 스위트는 그 표를 자동으로 만듭니다.
4. **blurry**는 CRPS 1% 차이로 사실상 구별이 안 되고, 스펙트럼 비(0.15)에서만 잡힙니다. NeuralGCM이 스펙트럼을 따로 보고하는 이유와 같습니다.
5. CFM의 ε은 채널별 iid Gaussian입니다. 학습이 덜 된 velocity field는 그 독립 noise를 출력에 남길 수 있고, 그것이 정확히 unbalanced 모드입니다. PG-CFM에서 PINN 손실이 막아야 하는 실패가 이것이므로, **CFM 단독 vs PG-CFM의 정역학 잔차 비교**가 핵심 ablation입니다.

### C. 진단끼리 서로 보완해야 구분된다
6. underdispersive와 quasi-static은 SSR이 둘 다 약 0.3으로 같습니다. tendency 비(0.45 vs 0.19)와 ACC(0.35 vs 0.00)를 같이 봐야 원인(spread 부족 vs 움직이지 않음)이 갈립니다. rank histogram은 둘 다 U자형입니다.
7. zero-noise의 tendency 비가 lead에 따라 0.78 → 0.36으로 떨어집니다. 결정론 경로는 lead가 갈수록 평활해지므로, 이 지표는 "ε = 0 기준선"과 "ensemble member"를 반드시 분리해서 봐야 합니다(스위트는 member별 비의 평균을 씀).

### D. 평가 설계의 약점 (실 ERA5에서 주의)
8. **표본 수**: toy에서 IC 3개로는 bias 부호가 lead마다 바뀌고 perfect 모델의 sp drift가 ±40 Pa입니다. NeuralGCM은 732개 IC를 씁니다. 2020 test에서 최소 50–100개 IC(서로 3일 이상 간격)를 권장합니다.
9. **정역학·지균 기준값**: ERA5 일평균은 12개 기압면 이산화 오차와 일평균 비선형성 때문에 잔차가 0이 아닙니다. 절대값이 아니라 **같은 날 ERA5 대비 비**로 판정해야 합니다(스위트는 truth 줄을 같이 출력).
10. **Climatology**: 1980–2011 climatology로 2020을 평가하면 온난화 추세만큼 climatology에 bias가 생겨 ACC와 "climatology 대비 %"가 부풀 수 있습니다. 최근 연도로 창을 옮기거나 추세 제거가 필요합니다. (toy의 climatology도 비선형 때문에 z500 bias −40 m²/s² 정도가 있습니다.)
11. **질량 보존**: KAI 채널에는 건조 질량이 따로 없어 sp(수증기 포함)로 대신합니다. 전구 평균 drift는 truth의 하루 변동과 비교해서만 의미가 있습니다. toy는 질량을 보존하지 않으므로 이 패널은 toy에서 판정에 쓰지 않았습니다.
12. **NeuralGCM에서 빠진 것**: 열대저기압 추적, P−E water budget(KAI에 강수 채널 없음), 확률 climatology 대비 CRPS(WB2 `GaussianCRPS`로 추가 가능, climatology std 필요), 장기(AMIP) 안정성. 20/40일 rollout 안정성은 같은 스위트에 lead만 늘려 돌리면 됩니다.
13. **스펙트럼**: 위도별 파수 그대로 30–60° 평균, 파장은 45° 기준으로 표기합니다(WB2의 파장 보간은 생략). 1000–3000 km 대역 비만 요약에 씁니다(1.5° 격자 Nyquist ≈ 236 km보다 충분히 큼).

## 5. 다음 단계
- 통합 PG-CFM 모델이 나오면 KAI 형식 ensemble 파일 + ε = 0 파일 + KAI 결정론 파일로 2020을 채점합니다.
- 보고 기본표: CRPS·SSR(headline 7변수) + tendency 비 + 정역학 잔차 비(ERA5 대비) + ageo/geo 비 + 1000–3000 km 스펙트럼 비.
- 위 3·5번에 따라 논문 실험 절에 "CFM 단독 vs PG-CFM의 물리 균형 진단" 표를 추가할 것을 제안합니다.

## 출처
- WB2 commit과 라이선스: `third_party/weatherbench2/UPSTREAM.md`.
- toy 결과 수치: `docs/eval/synthetic/scores.csv`, `physics.csv` (seed 0, 재현 가능).
