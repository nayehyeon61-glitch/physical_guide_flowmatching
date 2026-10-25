# ERA5 · ClimODE 격자 · 15일 예측 설정

사용자가 확정한 조건은 **기존 FM/encoder/decoder 유지**, ERA5,
ClimODE와 같은 **전지구 5.625° 격자**, **15일 최종 예측 길이**입니다.
출력 간격은 ClimODE의 기본 데이터 처리에 맞춰 **6시간을 권장 기본값**으로
두었습니다. 6시간 출력은 사용자가 별도로 지정한 조건은 아니며 변경할 수
있습니다. 15일은 데이터 sampling 간격이나 solver 내부 step을 뜻하지 않습니다.

## 확인한 ClimODE 기준

| 항목 | 기본 전지구 설정 |
| --- | --- |
| 데이터 | ERA5 / WeatherBench 5.625° |
| 공간 shape | latitude 32 × longitude 64 |
| 위도 | −87.1875° … +87.1875°, 남→북 |
| 경도 | 0° … 354.375°, 동쪽 방향·주기 경계 |
| 변수 순서 | `z500, t850, t2m, u10, v10` |
| 단위 | `z500`: m²/s², 온도: K, 바람: m/s |
| 시간 자료 | 6시간 resampling |
| 이번 목표 | 15일 = 1,296,000초 |
| 6시간 출력 | 초기값 포함 61개 시점, 미래 예측 60개 |

출처는 [ClimODE README](https://github.com/Aalto-QuML/ClimODE/blob/main/README.md),
[`train_global.py`](https://github.com/Aalto-QuML/ClimODE/blob/main/train_global.py),
[`utils.py`](https://github.com/Aalto-QuML/ClimODE/blob/main/utils.py),
[WeatherBench 격자 생성](https://github.com/pangeo-data/WeatherBench/blob/master/src/regrid.py)입니다.
검토한 ClimODE commit은 `e729d23e8799ce0e075699e76d60227d848d8d0c`입니다.
기본 short-range 평가 설정과 이번 15일 목표는 다릅니다. 이 격자·solver를
공유한다고 15일 성능이 논문에서 보장되는 것은 아닙니다.

## 기존 모델에 연결할 위치

```text
관측 이력 → DCT → Encoder → FM ensemble
                            ↓
                   Decoder → IDCT → 역정규화
                            ↓
                   앙상블별 condition c_e
                            ↓
관측 초기값 x(t0) → 구면 transport + learned flux/source → odeint → x_e(t1)
```

FM이 만든 것이 미래 `t1`의 상태라면 condition으로 사용합니다. 그 상태를
초기조건으로 다시 15일 적분하지 않습니다. 기존 FM의 생성 시간 `τ`와 물리
시간 `t`도 분리합니다. 물리 연산은 임의 latent가 아닌 실제 단위의 구면
격자 상태에 적용합니다.

새 solver는 구현하지 않았습니다. `torchdiffeq.odeint`의 RK4 또는 dopri5를
사용합니다. 추가한 것은 구면 공간 연산자와 ensemble/forecast 연결입니다.

## 물리식의 범위와 변수별 budget

`SphericalTransport2D`는 다음 두 scalar transport 형식을 제공합니다.

\[
\begin{aligned}
\text{conservative:}\quad&\dot q=-\nabla_s\cdot(Vq)+K\Delta_s q+R_\theta,\\
\text{advective:}\quad&\dot q=-\nabla_s\cdot(Vq)+q\nabla_s\cdot V+K\Delta_s q+R_\theta.
\end{aligned}
\]

두 번째 형식은 divergent flow에서도 상수 scalar가 유지되도록 합니다.
`SphericalFluxSourceClosure2D`의 learned tendency는

\[
R_\theta=-\nabla_s\cdot J_\theta(q,c_e)+S_\theta(q,c_e)
\]

입니다. 구면 cell area와 face 길이를 사용하는 flux는 면적 적분 budget을
보존하지만, source는 그 budget을 바꿀 수 있습니다. 온도·geopotential에는
source나 실제 forcing이 필요할 수 있습니다. `source_scale=0`을 물리적
정답으로 가정하거나 모든 채널의 면적 평균을 강제로 고정하지 마십시오.

`t2m/t850/z500/u10/v10` 전체가 이 scalar transport만으로 닫힌 대기 시스템이
되는 것은 아닙니다. 특히 z500은 geopotential이고, 10m 바람으로 850hPa 기온과
500hPa geopotential을 정확히 수송한다고 해석할 수 없습니다. ClimODE의
augmented transport velocities도 각 채널의 learned field이며 관측 u10/v10와
동일하지 않습니다. 여기서 제공하는 모델 수준은 **physics-inspired 구면
transport + 학습 tendency**입니다. NeuralGCM 같은 primitive-equation core는
별도 수준의 대기역학 구현입니다.

`velocity(t,state)->(u,v)` 인터페이스로 매 RHS 평가마다 member/channel별
transport velocity를 갱신할 수 있습니다. 출력 shape은 state로 broadcast되는
5차원 tensor이며 단위는 m/s입니다. 예컨대 velocity head가 `2*C` 채널을
출력하고 `vmax*tanh(raw)`로 bound를 적용할 수 있습니다. 이 head는 기존 모델
조건을 받아 별도로 학습해야 하며, `velocity_bound_m_s`도 지정해야 합니다.
관측 바람을 고정해서 15일 동안 그대로 사용하는 것이 기본 대기 예측이라고
가정하지 않습니다. learned velocity의 크기 bound는 그 network Jacobian의
stiffness까지 보장하지 않습니다.

## 좌표와 전처리

`prepare_era5_state(fields, latitude, longitude)`는 실제 tensor와 좌표를 함께
재배열합니다. 위도 내림차순, 경도 `[-180,180)`도 canonical 순서로 바꾸지만,
다른 격자의 데이터는 보간하지 않고 오류로 알립니다. NetCDF에서 pressure
level 선택과 `z/t` 등의 변수 이름을 canonical 이름으로 매핑하는 부분은
명시적으로 수행해야 합니다. `z500` geopotential과 geopotential height는
단위가 다르며 자동 변환하지 않습니다.

기존 모델의 전처리 순서를 그대로 역변환하십시오. 공식 ClimODE loader는
min-max를 쓰고 일부 변수 이름이 `mean/std`여도 실제 값은 min/max일 수
있습니다. 원래 모델이 min-max라면 `channel_mean=min`,
`channel_scale=max-min`으로 `decode_ensemble_forecast`를 호출할 수 있습니다.
z-score라면 mean/std를 사용합니다. 정규화 통계는 train split에서만 계산하고
validation/test에 그대로 적용하십시오.

DCT coefficient 자체를 정규화했다면 정규화 역변환은 IDCT 전입니다. 또 DCT의
경계 가정과 전지구 경도 주기 경계는 자동으로 일치하지 않습니다. 이번 변경은
물리 연산자의 경계 처리를 수정하며 원래 DCT 모델을 재학습하거나 변경하지는
않습니다.

## 구면의 공간 연산

셀 면적은

\[
A_j=R^2\Delta\lambda(\sin\phi_{j+1/2}-\sin\phi_{j-1/2})
\]

입니다. 경도만 periodic이고, 위도는 `torch.roll`로 연결하지 않습니다.
극점의 normal integrated flux는 0입니다. 공간 평균 대신
`area_weighted_mean`을 사용하십시오.

구면 closure의 CNN도 경도만 circular padding을 사용합니다. 위도는 replicate
padding이며, 이는 국소 근사입니다. 완전한 구면 회전 등변 convolution이나
vector의 극점 basis 처리를 구현한 것은 아닙니다.

## 15일 적분과 메모리

```python
from physical_guide_flowmatching import (
    GlobalLatLonGrid, SphericalTransport2D, SphericalFluxSourceClosure2D,
    ODESolverConfig, forecast_times_seconds, integrate_windowed,
)

grid = GlobalLatLonGrid()  # 32 latitude × 64 longitude

# 한 번 생성하고 optimizer에 등록할 공유 module.
# *_from_train은 실제 훈련 자료에서 정할 채널별 단위 scale입니다.
closure = SphericalFluxSourceClosure2D(
    channels=C, grid=grid, condition_channels=D,
    input_scale=state_scales_from_train,
    flux_scale=flux_scales_from_train,
    source_scale=source_scales_from_train,
).to(device=observed_initial.device, dtype=observed_initial.dtype)

# 이 아래는 batch마다 구성합니다. c_e는 기존 FM에서 계산한 condition입니다.
# 동적 velocity_model도 현재 batch의 condition에 연결되어 있어야 합니다.
rhs = SphericalTransport2D(
    grid, transport_form="advective",
    velocity=velocity_model,
    velocity_bound_m_s=velocity_model_speed_bound,
    closure=closure.conditioned_on(c_e),
).to(device=observed_initial.device, dtype=observed_initial.dtype)
x0 = observed_initial[:, None].expand(B, E, C, 32, 64)
times = forecast_times_seconds(15, 6, reference=x0)
step = min(120.0, rhs.explicit_step_limit_seconds())
config = ODESolverConfig(method="rk4", step_seconds=step)

# 학습: detach하지 않고 출력 구간 사이의 gradient도 유지합니다.
trajectory = integrate_windowed(
    rhs, x0, times, config, checkpoint_segments=True,
)
# shape: [61, B, E, C, 32, 64]
day_15_prediction = trajectory[-1]
```

추론에서는 이 호출을 `with torch.no_grad():`로 감쌉니다.

`closure`와 공유 velocity head를 training loop 안에서 매번 새로 초기화하지
마십시오. 위 예제의 `velocity_model`, scale, 원래 encoder/decoder/FM은 실제
기존 모델과 데이터에서 제공해야 합니다. 새 closure의 마지막 convolution은
0으로 초기화되므로 upstream FM gradient는 이 layer가 학습된 뒤 생깁니다.
먼저 짧은 trajectory에서 closure/velocity를 pretrain하고 joint fine-tune하는
순서를 권합니다.

15일이나 6시간은 내부 RK4 step이 아닙니다. 최북단의 경도 방향 cell 폭은 약
30.7km라서 강한 바람이면 수분 이하 step이 필요할 수 있습니다. `120초`도
항상 안전하다는 뜻이 아니므로 계산된 bound와 비교하고, learned tendency가
포함된 상태에서 `h`와 `h/2` 결과를 다시 확인해야 합니다.

`integrate_windowed`는 출력 구간마다 기존 solver를 호출하고, 직전 예측을
다음 초기값으로 사용합니다. 관측으로 재설정하거나 시간축을 0으로 되돌리지
않습니다. checkpoint는 backward 때 구간을 재계산해 activation 메모리를
줄이며, forward step 수 자체는 줄이지 않습니다. FM condition은 적분 전에
고정하고 RHS 안에서 dropout/무작위 재샘플링을 하지 마십시오.

학습은 6h/24h → 72h → 7d → 15d 식의 curriculum을 검토하고, 실제 validation
성능과 안정성에 따라 늘리십시오. 15일 direct-autograd가 항상 메모리에 맞는
것은 아니며 checkpoint도 출력 tensor 자체의 메모리는 유지합니다.

기존 `correct_forecast`의 reference-nudging도 사용할 수 있지만 spread 축소가
있습니다. 구면 tracer의 평균 보존을 요청하면 이제 cell-area 가중 평균을
자동 사용합니다. ERA5 온도/geopotential에는 근거 없이 평균 보존을 요청하지
말고, source/forcing을 포함한 실제 budget을 평가하십시오.

## 실행한 검증과 남은 작업

```bash
/workspace/.venvs/physical-guide/bin/python -m pytest -q
/workspace/.venvs/physical-guide/bin/python examples/era5_15day.py
```

합성 5채널·2앙상블의 `32×64` 격자에서 15일/61시점 실제 적분, 정확한 이산
수송해 비교, step 절반 비교를 실행했습니다. 이 예제는 매끄러운 prescribed
solid-body flow라 내부 step 1800초를 사용할 수 있었습니다. 이 값을 실제
ERA5의 강한 바람·learned tendency에 일반화해서는 안 됩니다.

이 문서 작성 이후 사용자 Step13 브랜치의 원래 모델을 가져와
`pgcfm/ode_corrector.py`로 연결하고, 합성 자료에서 K60 학습·비교를 실행했습니다.
현재 실험은 [Step13 ODE 실험 문서](step13_ode_experiment.md)를 참고하십시오.
실제 ERA5 파일과 기존에 충분히 학습한 checkpoint는 아직 없으므로 실제 ERA5
성능 검증은 남아 있습니다. 필요한 평가는 lead별 area-weighted RMSE/ACC,
CRPS 또는 energy score, spread–skill, rank/reliability, 물리 budget·장기 발산입니다.
