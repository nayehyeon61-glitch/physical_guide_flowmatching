# physical_guide_flowmatching

이 파일은 Step13 소스를 가져오기 전 초기 ODE prototype README의 보존본입니다.
현재 원래 모델과의 연결·실행 결과는 [Step13 실험 문서](step13_ode_experiment.md)를
참고하십시오.

FM이 생성한 앙상블에 물리 시간 적분을 연결하는 PyTorch 모듈입니다. 시간
적분기는 **기존 `torchdiffeq==0.2.5`**를 사용합니다. 새 ODE solver를 구현하지
않습니다. **Cartesian 추적자 예제**와 **ERA5 전지구 구면 transport·source
모듈**을 제공하며, 완성된 대기 dynamical core는 아닙니다.

## 설치 및 실행

클라우드 환경에서 확인한 조합은 Python 3.12, PyTorch 2.9.1 CPU입니다.

```bash
cd /workspace/physical_guide_flowmatching
export UV_CACHE_DIR=/workspace/.cache/uv
uv venv --allow-existing /workspace/.venvs/physical-guide --python python3.12
uv pip install --python /workspace/.venvs/physical-guide/bin/python \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.1+cpu'
uv pip install --python /workspace/.venvs/physical-guide/bin/python -e '.[dev,dct]'

/workspace/.venvs/physical-guide/bin/python -m pytest -q
/workspace/.venvs/physical-guide/bin/python examples/ensemble_correction.py
/workspace/.venvs/physical-guide/bin/python examples/pretrain_closure.py --steps 80
```

GPU 환경에서는 하드웨어에 맞는 공식 PyTorch wheel을 먼저 설치하십시오.
모듈은 입력 tensor의 device를 사용합니다. CPU에서 검증했으며 CUDA 실행은
검증하지 않았습니다. `dct` extra는 `torch-dct` 기반 IDCT 연결 테스트에
필요합니다. 테스트 외 핵심 ODE 모듈은 이 패키지를 직접 요구하지 않습니다.

## 두 가지 사용 경로

ERA5/ClimODE와 같은 전지구 **5.625°, 32×64 격자·15일 예측**에는 새로 추가한
구면 모듈을 사용하십시오. 설정·변수·기존 모델 연결은
[ERA5 15일 설계](docs/era5_15day_ko.md)에 정리했습니다. 전지구 격자와 15일
예측 길이는 사용자가 확정한 조건입니다. 권장 기본값인 6시간 출력을 쓰면
초기값 포함 61개 시점이며, 내부 solver step과 다릅니다.

```bash
/workspace/.venvs/physical-guide/bin/python examples/era5_15day.py
```

`GlobalLatLonGrid`, `SphericalTransport2D`, `SphericalFluxSourceClosure2D`,
`prepare_era5_state`, `forecast_times_seconds`, `integrate_windowed`가 해당
인터페이스입니다. 구면 cell area, 경도만 periodic인 경계, 선택적 source,
분할 적분/checkpoint를 지원합니다. 기온·geopotential에 추적자 보존식을
일괄 적용하지 마십시오.

1. **권장: 물리 RHS + 앙상블별 학습 flux를 실제 시간으로 적분.** FM latent나
   미래 예측을 closure의 condition으로 넣고, 초기 상태는 관측 `x(t0)`를
   사용합니다. 또는 FM이 생성한 것이 실제로 `t0`의 초기 앙상블일 때만 그
   결과를 적분 초기조건으로 사용합니다.
2. **기존 미래 예측을 참조하는 실험적 보정.** `correct_forecast`는 관측
   초기값에서 시작하여, FM 미래 예측으로 향하는 reference에 nudging합니다.
   이는 추가 forcing을 갖는 약한 제약 모델이며 hard physical projection이
   아닙니다. 앙상블 spread가 줄어들 수 있습니다.

**FM이 이미 `t1`을 예측했다면 그 결과를 다시 `t0→t1` 시간만큼 적분하지
마십시오.** FM 생성 시간 `τ`와 물리 시간 `t`도 서로 다른 변수입니다.

Tensor 계약은 `[batch, ensemble, channel, height, width]`, 물리 시간은 경과
초, 격자 간격은 m입니다. 물리 RHS에는 IDCT와 정규화 역변환을 마친 물리량을
넣습니다. 전지구 위경도 격자에 이 Cartesian stencil을 그대로 적용하지
마십시오.

설계·수식·기존 DCT 모델 연결·pretraining·실제 기후 적용 조건은
[한국어 설계 문서](docs/design_ko.md)를 참고하십시오.

## 검증 범위

테스트는 정확해와의 비교, 시간 step 수렴, 각 앙상블의 공간 합 보존과 독립성,
FM condition 및 초기값·학습 closure로의 역전파, DCT/IDCT와 물리 단위 복원을
확인합니다. 예제의 오차 감소는 **제조한 합성 데이터**에서의 결과입니다.
실제 기후 데이터의 예측 성능, 확률 예측 calibration, 장기 안정성은 아직
검증하지 않았습니다. 저장소에는 원래 FM/encoder/decoder 구현이나 학습된
checkpoint가 없으므로 이 모듈을 기존 모델에 연결하는 부분은 해당 모델에서
제공해야 합니다.
