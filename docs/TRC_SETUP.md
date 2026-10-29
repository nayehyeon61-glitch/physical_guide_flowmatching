# TRC(TPU Research Cloud)로 돌리기: 신청부터 이쪽 클라우드 세션에서 직접 조작하기까지

소유자 질문(2026-10-28 07:17): TRC로 가능한가, 이쪽 클라우드 세션에서 직접 돌릴 수 있는가.
답: TRC가 승인되면 TPU VM은 SSH가 되는 일반 리눅스 VM이라 Colab보다 훨씬 낫습니다. 이쪽 세션에서 직접 조작하려면
(1) Google Cloud 인증 정보와 (2) 나가는 네트워크 허용이 있는 **클라우드 환경**을 프로젝트에 추가해야 합니다. 지금 프로젝트는 설정이 없는 기본 환경이라 둘 다 없습니다.
아래 A~C를 순서대로 하면 그 뒤에 시작하는 세션에서 Claude가 gcloud로 TPU VM을 만들고 명령을 직접 돌립니다. SSH가 그 환경에서 실제로 되는지는 D에서 처음 확인합니다.

## A. TRC 신청 (하루~며칠)

1. Google Cloud 계정으로 **프로젝트**를 하나 만듭니다(콘솔 https://console.cloud.google.com → 프로젝트 만들기, 예: `pgcfm-trc`). 프로젝트 ID를 적어 둡니다.
2. **결제 계정**을 연결합니다(TPU는 무료지만 VM 디스크·저장소 요금이 소액 나오고, 결제 계정이 없으면 TPU VM을 만들 수 없습니다). 무료 체험 크레딧이 있으면 그걸로 충분합니다.
3. https://sites.research.google/trc 에서 신청합니다. 양식에는 연구 제목·설명·소속·예상 사용 기간을 적습니다. 연구 설명은 이 저장소의 `docs/paper_hypothesis.md`와 20단계 결과(다중 모델 lead별 혼합 + 흐름 매칭 편차, ClimODE 체크포인트 대비 z500 −15~−18 %)를 두세 문단으로 요약하면 됩니다(필요하면 Claude가 초안을 씁니다).
4. 승인 메일이 오면 거기 적힌 **TPU 종류·영역(zone)·수량**(예: `v3-8` ×5 in `europe-west4-a`, `v2-8` ×5 in `us-central1-f`, 선점형 100개, 최근에는 `v4`·`v5e`도)을 적어 둡니다. 메일의 안내대로 프로젝트 ID를 회신해야 할당량이 그 프로젝트에 들어갑니다. 기간은 보통 30일이고 연장 신청이 됩니다.

## B. Google Cloud 쪽 준비 (콘솔 또는 Cloud Shell, 10분)

Cloud Shell(콘솔 오른쪽 위 `>_`)에서 아래를 실행합니다. `PROJECT`와 `ZONE`은 A에서 받은 값입니다.

```bash
PROJECT=pgcfm-trc; ZONE=europe-west4-a
gcloud config set project $PROJECT
gcloud services enable tpu.googleapis.com compute.googleapis.com iap.googleapis.com storage.googleapis.com
# Claude가 쓸 서비스 계정: TPU VM 만들기·지우기, IAP 터널로 SSH, 저장소 읽기·쓰기
gcloud iam service-accounts create claude-tpu --display-name "Claude TPU runner"
SA=claude-tpu@$PROJECT.iam.gserviceaccount.com
for R in roles/tpu.admin roles/compute.instanceAdmin.v1 roles/iap.tunnelResourceAccessor roles/iam.serviceAccountUser roles/storage.objectAdmin roles/serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding $PROJECT --member serviceAccount:$SA --role $R --quiet > /dev/null
done
# 키 파일 (한 번만, 다운로드 후 Cloud Shell에서 지움)
gcloud iam service-accounts keys create claude-tpu-key.json --iam-account $SA
cloudshell download claude-tpu-key.json && rm claude-tpu-key.json
# 결과 저장용 버킷 (선택: 결과·체크포인트를 VM 밖에 두면 VM을 지워도 남습니다)
gsutil mb -l ${ZONE%-*} gs://$PROJECT-runs
```

키 파일(`claude-tpu-key.json`)은 **채팅에 붙이지 말고** C-4에서 비밀값으로만 넣습니다. 할당량이 들어왔는지는 `gcloud compute tpus tpu-vm create`를 한 번 해 보면 압니다(D-2의 명령을 Cloud Shell에서 그대로 실행; 되면 바로 `delete`).

## C. 이쪽 클라우드 환경 추가 (5분)

1. [Project settings](#project-settings/environment) → **Cloud environment** 메뉴 → **Add cloud environment**. 이름은 `pgcfm-trc`처럼 아무거나.
2. **Network access**: 가장 간단한 건 **Full**입니다. 제한하려면 **Limited**에서 Allowed domains에 다음을 넣고 *Allow package managers*는 켜 둡니다.
   `googleapis.com`, `*.googleapis.com`, `tunnel.cloudapis.com`, `accounts.google.com`, `oauth2.googleapis.com`, `storage.googleapis.com`, `github.com`.
   SSH(22번 포트)는 Limited에서 안 나갈 수 있어서 Claude는 IAP 터널(HTTPS)로 접속을 시도합니다. 그것도 막히면 Full로 바꿔야 합니다.
3. **Setup script**(새 세션마다 실행): gcloud는 이미 있고 `ssh` 명령만 없어서 이것만 넣습니다.
   ```bash
   apt-get update -qq && apt-get install -y -qq openssh-client > /dev/null
   ```
4. **환경 변수**: `GCP_PROJECT`(프로젝트 ID), `TPU_ZONE`(예 `europe-west4-a`), `TPU_TYPE`(예 `v3-8`), `TPU_NAME`(예 `pgcfm-1`), `GCS_BUCKET`(예 `pgcfm-trc-runs`).
5. 저장한 뒤 메뉴에서 그 환경 옆 **톱니바퀴 → Network secrets**(구버전 앱은 *API credentials*)에 비밀값 `GCP_SA_KEY_JSON` = 키 파일의 내용 전체(JSON 한 덩어리)를 넣습니다. 저장 전에는 이 칸이 안 보입니다. Claude는 세션에서 이 변수 이름 하나만 읽습니다.
6. 프로젝트의 환경이 방금 만든 것으로 선택됐는지 확인합니다. **그 뒤에 시작하는 세션**부터 적용되니, 이 스레드에 "환경 넣었습니다"라고 써 주시면 Claude가 새 세션에서 D를 시작합니다.

## D. 그 뒤 Claude가 하는 일 (세션 안에서)

1. 인증: `echo "$GCP_SA_KEY_JSON" > key.json && gcloud auth activate-service-account --key-file key.json && gcloud config set project $GCP_PROJECT`
2. TPU VM 만들기(할당량 확인): `gcloud compute tpus tpu-vm create $TPU_NAME --zone $TPU_ZONE --accelerator-type $TPU_TYPE --version tpu-ubuntu2204-base`
   (v4는 `--version tpu-vm-v4-base`, v5e는 `--version v2-alpha-tpuv5-lite`; 선점형은 `--preemptible`).
3. 접속 확인: `gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $TPU_ZONE --tunnel-through-iap --command 'nproc; free -g'`
   여기서 막히면(터널 불가) 그 사실과 대안(E)을 알려 드립니다.
4. VM 안에서: 저장소 복제 → `pip install -r requirements.txt neuralgcm gcsfs` → `python -c "import jax; print(jax.devices())"`로 TPU 확인 →
   NeuralGCM 세 체크포인트를 2020년 IC 180개 전부로(`docs/GPU_QUICKSTART.md` 9절), 우리 파이프라인은 호스트 CPU(v3-8: 96코어, 335 GB)로 `PROFILE=climode BLEND=1 CLIMODE=1 CLIMODE_ICS=180 DEVICE=cpu THREADS=96`.
   결과(json·png)는 `gsutil cp`로 버킷에 올리고 문서와 PR에 반영합니다. 긴 실행은 VM 안에서 `nohup`로 띄워 두고 세션은 끊어도 됩니다.
5. 다 끝나면 `gcloud compute tpus tpu-vm delete $TPU_NAME --zone $TPU_ZONE`(디스크 요금 중단). 선생님이 직접 지우셔도 됩니다.

## E. 대안: 선생님 컴퓨터로 접속

C가 번거롭거나 D-3에서 터널이 막히면, 선생님 컴퓨터에 gcloud를 설치해 `gcloud auth login`을 해 두고 이 스레드에 그 컴퓨터를 연결해 주세요. Claude가 거기서 D-2~D-5를 그대로 실행합니다. 이 경우 서비스 계정 키는 필요 없습니다.

## F. 비용과 주의

- TPU 사용은 TRC 기간 동안 무료. VM 부팅 디스크 100 GB 약 월 10달러, 버킷 수십 GB 소액, 외부 전송(결과 내려받기) 소액.
- TRC는 기간이 끝나면 할당량이 사라지고 그때부터 TPU VM은 유료(v3-8 시간당 8달러 수준)이니 끝나기 전에 VM을 지웁니다. 선점형은 언제든 중단될 수 있어 긴 학습은 체크포인트 재개(`MAX_HOURS`, 같은 명령 재실행)로 돌립니다.
- 키 파일은 환경 비밀값에만 넣고 저장소·채팅·노트북에 넣지 않습니다. 쓰임이 끝나면 `gcloud iam service-accounts keys delete`로 폐기합니다.
