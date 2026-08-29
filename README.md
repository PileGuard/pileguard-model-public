# PileGuard Model

산란계 군집 압사 위험을 감지하고 시계열 위험 특징을 계산하는 PileGuard Adaptive 모델 저장소입니다.

이번 제출 범위는 다음 다섯 가지입니다.

1. MSU Poultry Piling Dataset 공식 분할을 이용한 Pile/Non-pile 감지 모델 재현
2. NESTLER 영상의 개체 위치·밀도·광류·방향 특징 추출
3. PIO 실제 육계사 YOLO 주석의 개체수·면적밀도·국소 집중도 특징 추출
4. AI Hub 공식 국내 산란계 군집 이미지의 개체 검출·밀도 특징 전이 검증
5. 문헌 기반 Digital Twin 시나리오의 위험곡선과 경보 데모

## Environment

Python 3.12를 사용합니다. PyTorch의 macOS 권장 Python 범위와 현재 시스템 Python 3.14의 패키지 호환성을 고려해 학습 환경을 고정했습니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[train,video,detection,dev]"
```

## Data

원본 데이터와 모델 가중치는 Git에 올리지 않습니다. 특히 AI Hub 자료는 원본뿐 아니라 이미지별 파일명·예측·오류 목록도 공개 저장소에서 제외하고, 출처가 명시된 집계 지표와 그래프만 공개합니다. 상세 진단자료는 이용 승인을 받은 로컬 환경에서만 생성·보관합니다. 저장소 밖의 데이터 디렉터리를 `PILEGUARD_DATA_ROOT`로 지정합니다.

현재 작업공간 구조에서는 다음처럼 확인할 수 있습니다.

```bash
export PILEGUARD_DATA_ROOT="../data"
pileguard-check-data
```

패키지를 설치하기 전에는 다음 명령을 사용합니다.

```bash
PYTHONPATH=src python -m pileguard.data_inventory --data-root ../data
```

정상 데이터 구성은 다음과 같습니다.

- MSU: Train 7,562장 / Validation 853장 / Test 2,848장
- NESTLER: 주석 JSON 6개
- PIO: Train 1,035장 / Validation 452장과 대응 YOLO 라벨

## Domestic laying-hen data intake

농장에서 연속 사건영상을 직접 확보하는 경우 모델 실행 전에 출처 승인, 가명화, SHA-256, 영상 품질, 2인 사건 라벨 검수와 농장 단위 split을 감사합니다. 아래 도구는 민감한 현장영상 반입 준비 기술이며, AI Hub 공개 정지영상 감사와는 별개입니다.

```bash
pileguard-audit-domestic-data --data-root ../data
```

세부 폴더·CSV 규격과 제공 요청 항목은 [`docs/domestic-data-intake.md`](docs/domestic-data-intake.md)에 있습니다. 감사 결과는 민감한 농장정보가 Git에 들어오지 않도록 `outputs/domestic-data-intake/audit.json`에만 저장됩니다. `ready_for_model_validation=true`와 별도의 독립 평가 결과가 모두 없으면 국내 현장 사건 성능을 주장하지 않습니다.

## AI Hub Korean laying-hen validation

[AI Hub 지능형 스마트 축사 데이터(육계, 산란계)](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=575)의 공식 Validation 중 산란 초기·중기·후기 `군집` 원천·라벨 TAR 6개를 사용합니다. 외부 파일공유 링크는 사용하지 않으며 원본은 Git에 올리지 않습니다. AI Hub가 1GB 단위로 나눈 내부 `tar.gz.part*`를 디스크에 다시 풀지 않고 연결해 읽고, 경로 탈출·링크·조각 누락·중복·PNG 해상도·라벨/이미지 대응을 검사합니다.

```bash
export PILEGUARD_AIHUB_ARCHIVE_DIR="$HOME/Downloads"

# 일부 이미지로 연결 확인
pileguard-evaluate-aihub-domestic --max-images 12

# Validation 전체 평가
pileguard-evaluate-aihub-domestic
```

전체 4,546장·닭 bbox 106,023개를 PIO 해외 육계 검출기에 적용했습니다. 같은 Validation에서 비교한 신뢰도 0.05·0.10·0.15·0.25 중 0.25의 중심거리 F1이 0.2808로 가장 높았고, precision 0.2469·recall 0.3256, IoU 0.5 F1 0.0234, 개체수 MAE 17.45마리였습니다. 산란 단계별 중심거리 F1은 초기 0.1921, 중기 0.2967, 후기 0.3077입니다.

개체수 상관계수는 0.2556으로 낮지만 bbox 면적합과 중심 분산의 상관계수는 각각 0.7144, 0.7670으로 장면 간 일부 공간 변화는 보존됐습니다. 모니터링 진입 기준(center F1 0.5)을 통과하지 못했으므로 기존 검출기는 자동 차단하며, 다음 단계는 AI Hub 국내 산란계 라벨을 사용한 미세조정입니다. 임계값 선택과 보고에 동일 Validation을 사용했으므로 독립 Test 결과가 아닙니다.

이 `군집` 데이터는 국내 산란계 정지영상과 개체 bbox를 제공하지만 연속 사고영상, 압사·파일링 사건 구간, 1·3·5분 전 라벨은 제공하지 않습니다. 따라서 국내 개체 검출·정적 밀도 특징 전이 검증으로만 해석합니다. 상세 결과는 [`summary.json`](artifacts/aihub-domestic-validation/summary.json)에 있습니다.

## AI Hub laying-hen detector fine-tuning

공식 Training 36,423장·Validation 4,546장을 원본 TAR에서 순차 검사·YOLO 라벨 변환합니다. 면적 0 bbox 2개와 같은 이미지 안의 완전 중복 bbox 533개(Training 498개, Validation 35개)는 감사 결과에 기록하고 YOLO 학습 라벨에서만 제외합니다.

SHA-256 감사 결과 공식 분할 안에서 교차 이미지 180건·분할 내부 중복 703건을 확인했습니다. 성능을 부풀리지 않도록 원본은 보존하고, Validation에 이미 존재하는 Training 복사본을 제외한 결과 Training 35,548장·Validation 4,538장을 사용합니다.

```bash
export PILEGUARD_DATA_ROOT="../data"

# 압축 검사·중복 제외 manifest 생성
pileguard-prepare-aihub-detection

# 8장씩으로 전체 파이프라인 확인
pileguard-train-aihub --smoke

# PIO 검출기로부터 전체 데이터 미세조정
pileguard-train-aihub

# 기존 PIO 결과와 같은 공식 Validation 4,546장에서 전체 재평가
pileguard-evaluate-aihub-domestic \
  --config configs/aihub_finetuned_validation.toml \
  --archive-dir ../data/raw/aihub_laying_hen_v1/validation

# 임계값·산란단계별 오류 감사
pileguard-audit-aihub-finetuned
```

전체 미세조정 기준은 seed 2026, MPS, 480px, batch 16, 1 epoch이며 55분 44초가 걸렸습니다. 중복을 제외한 Validation에서 precision 0.7736, recall 0.7048, mAP50 0.8017, mAP50-95 0.5136을 기록했습니다. 학습 기록은 [`training_summary.json`](artifacts/aihub-laying-hen-yolo26n/training_summary.json)에 저장합니다.

기존 PIO 해외 육계 검출기와 같은 공식 Validation 4,546장을 사용한 재평가에서 신뢰도 0.25의 중심거리 F1은 0.2808에서 0.7450, IoU 0.5 F1은 0.0234에서 0.7293으로 향상되었고 개체수 MAE는 17.45마리에서 5.75마리로 감소했습니다. 단계별 중심거리 F1은 초기 0.5842, 중기 0.7954, 후기 0.7991입니다. 설정한 개체 검출 모니터링 연동 기준(center F1 0.5)은 통과했습니다. 상세 비교는 [`validation/summary.json`](artifacts/aihub-laying-hen-yolo26n/validation/summary.json)에 저장합니다.

다만 학습·체크포인트 선택과 재평가에 같은 공식 Validation이 사용되어 이 수치는 독립 Test 성능이 아닙니다. 또한 데이터셋에 연속 사건 영상과 압사·파일링 사건 라벨이 없으므로 국내 산란계 개체 검출·정적 밀도 특징 성능으로만 해석하며 현장 압사 위험 예측 성능으로는 주장하지 않습니다. 데이터셋 감사 결과는 [`summary.json`](artifacts/aihub-laying-hen-dataset/summary.json)에 기록합니다.

### Fine-tuned detector error audit

신뢰도 0.05–0.70의 12개 임계값을 공식 Validation 전체에서 비교했습니다. 위치 모니터링을 기준으로 중심거리 F1이 가장 높은 단일 임계값은 0.20이며 F1 0.7467입니다. 개체수 MAE는 임계값 0.25에서 5.75마리로 가장 낮아, 위치 검출 최적값과 개체수 최적값이 다른 교환관계를 기록했습니다.

임계값 0.20에서 산란 초기 중심거리 F1은 0.6265로 중기 0.7857·후기 0.7873보다 낮습니다. 초기 이미지의 81.8%에서 개체수를 과소 검출했고 평균 bias는 -7.01마리인 반면, 중기·후기는 각각 +4.82·+5.17마리의 과대 검출 편향이 있었습니다. 따라서 0.20은 위치 모니터링용 전역 임계값으로만 사용하고, 예측 개체수를 절대 밀도로 사용하기 전에는 산란단계·농장별 보정이 필요합니다. 단계별 최적 임계값은 같은 Validation에서 선택했으므로 진단용으로만 기록하고 배포 정책으로는 사용하지 않습니다. 상세 결과는 [`error-audit/summary.json`](artifacts/aihub-laying-hen-yolo26n/error-audit/summary.json)에 저장합니다.

## Fine-tuned detector video risk pipeline

AI Hub 산란계 미세조정 검출기의 예측 bbox에서 개체수·면적밀도·근접거리·국소집중도를 계산하고 optical flow 속도·수렴도·방향 일관성과 결합해 카메라 상대 위험지수를 영상 프레임별로 생성합니다. Track ID가 없는 일반 영상에서 추적 특징은 0이 아닌 결측으로 보존하며, 사용 가능한 근거가 4종 미만이면 위험값을 `unavailable`로 차단합니다.

```bash
pileguard-run-video-risk \
  --videos "../data/extracted/nestler_v1/poultry_behaviour_dataset/chickens/job_000004/job_000004_400frames.mp4" \
  --device auto
```

복수 영상은 `--videos` 뒤에 경로를 연속해 지정합니다. NESTLER 공식 산란계 영상 6개(Bulgaria 3개, Rwanda 3개) 2,392프레임을 MPS에서 처리한 기능시험에서 모든 프레임에 검출 결과가 있었고, 60프레임 교정 후 위험근거 가용률은 100%였습니다. 영상별 전체 파이프라인 처리속도는 약 45–51 FPS였지만 Apple MPS 기능시험 수치이며 Edge 장치 지연시간으로 일반화하지 않습니다.

`job_000007`에서 경고 상태가 4프레임 출력되었으나, NESTLER에 압사·파일링 사건 정답이 없으므로 사고 적중·오경보·선행시간으로 해석하지 않습니다. 예측 개체수도 산란단계·농장별 보정 전에는 절대밀도로 사용하지 않습니다. 상세 요약과 위험 타임라인은 [`summary.json`](artifacts/aihub-video-risk/summary.json)과 [`risk_timelines.png`](artifacts/aihub-video-risk/risk_timelines.png)에 저장합니다.

## MSU baseline training

MSU 공식 Train/Validation/Test 디렉터리를 변경하지 않습니다. 이 단계에서는 Train과 Validation만 사용하며 Test 평가는 별도 평가 단계에서 한 번만 수행합니다.

```bash
source .venv/bin/activate
export PILEGUARD_DATA_ROOT="../data"

# 빠른 파이프라인 확인
pileguard-train-msu --smoke --device auto

# ResNet18 기준모델 학습
pileguard-train-msu --config configs/msu_resnet18.toml --device auto
```

최적 validation F1 체크포인트는 `weights/msu-resnet18/best.pt`에 저장되며 Git에 포함되지 않습니다. 학습 이력과 그래프는 `artifacts/msu-resnet18/`에 저장됩니다.

기준 실행(seed 2026, MPS)의 최고 validation 결과는 epoch 9에서 accuracy 0.9883, precision 0.9736, recall 0.9885, F1 0.9810입니다. 상세 이력은 [`training_history.json`](artifacts/msu-resnet18/training_history.json), 곡선은 [`training_curves.png`](artifacts/msu-resnet18/training_curves.png)에서 확인할 수 있습니다. Test split 성능은 이 수치에 포함하지 않았습니다.

## MSU test evaluation

학습이 끝난 뒤 공식 Test split을 고정 임계값 0.5로 한 번 평가합니다. 이 과정에서 임계값이나 모델을 Test 결과에 맞춰 조정하지 않습니다.

```bash
pileguard-evaluate-msu --checkpoint weights/msu-resnet18/best.pt --device auto
```

평가 지표, 전체 예측, FP/FN 목록, PR 곡선, 혼동행렬과 고신뢰 오류 모음은 `artifacts/msu-resnet18/test/`에 저장됩니다. 원본 데이터 경로와 이미지가 포함된 예측 CSV 및 오류 모음은 로컬 분석 자료로만 유지하고 Git에는 포함하지 않습니다.

기준 실행의 공식 Test 결과는 accuracy 0.9810, precision 0.9959, recall 0.9616, F1 0.9785, PR-AUC 0.9984입니다. 혼동행렬은 TN 1,567 / FP 5 / FN 49 / TP 1,227이며, FN 49건 중 29건이 `ch16` 채널에 집중되었습니다. 카메라·채널별 지표는 [`metrics.json`](artifacts/msu-resnet18/test/metrics.json)에 함께 기록됩니다.

## NESTLER video features

[NESTLER 데이터셋](https://doi.org/10.5281/zenodo.20924893)이 제공한 추적 bounding box와 실제 400-frame 영상을 결합해 프레임별 군집 밀도, 추적 중심 이동, ROI optical flow 특징을 추출합니다. 현재 단계에서는 검출기 성능과 움직임 특징을 분리하기 위해 제공 주석 bbox를 사용하며, 배포 단계에서 동일 출력 형식의 YOLO 검출기로 교체할 수 있습니다.

```bash
pileguard-extract-nestler --data-root ../data
```

결과는 `artifacts/nestler-features/`의 프레임별 CSV, 요약 JSON, 15-frame 이동평균 그래프로 저장됩니다.

기준 실행에서는 6개 clip의 2,392프레임을 처리했으며 bbox 주석 유효 프레임은 2,309개입니다. 결측 83프레임은 객체가 없는 것으로 처리하지 않고 특징 결측으로 보존합니다. 촬영 환경별 평균 bbox 면적비는 Bulgaria 0.0423, Rwanda 0.0142로 차이가 커서, 후속 위험도 모델은 단일 절대 임계값 대신 카메라별 기준선 대비 변화량을 사용해야 합니다. 상세 분포와 주석 커버리지는 [`summary.json`](artifacts/nestler-features/summary.json)에 기록됩니다.

## NESTLER detector transfer audit

PIO 육계 데이터로 학습한 검출기를 NESTLER 6개 실제 축사 영상 2,392프레임에 적용하고, 제공 추적 bbox 19,861개와 교차 데이터셋 전이 성능을 비교합니다. NESTLER bbox는 포즈·스켈레톤 트래커 범위라 PIO 몸체 박스와 정의가 다르므로 IoU 0.5와 정규화 중심거리 0.05 일치를 함께 기록합니다.

```bash
pileguard-evaluate-nestler-transfer \
  --data-root ../data \
  --weights weights/pio-yolo26n/best.pt
```

신뢰도 0.05에서 중심거리 precision 0.1421, recall 0.1456, F1 0.1438이며 엄격한 IoU F1은 0.0043입니다. 프레임별 개체수 상관계수도 -0.3897로, 총 검출 수가 비슷하더라도 시간 변화는 보존되지 않았습니다. Bulgaria 중심거리 F1은 0.0368, Rwanda는 0.2239로 촬영환경별 차이도 큽니다.

설정된 모니터링 진입 기준(center F1 0.5)을 통과하지 못해 예측 bbox 기반 위험 알림 연결은 자동 차단됩니다. 이 결과는 Piling 사건 성능이 아니라 도메인 전이 실패 감사이며, 다음 단계는 NESTLER 클립 단위 분할을 이용한 미세조정입니다. 상세 결과는 [`summary.json`](artifacts/nestler-detector-transfer/summary.json)에 저장됩니다.

## NESTLER detection dataset

NESTLER 6개 clip을 프레임 단위가 아닌 영상 단위로 train/validation/test에 배정합니다. 각 split에 Bulgaria·Rwanda clip을 하나씩 두어 촬영 환경을 모두 포함하고, 같은 영상의 인접 프레임이 학습과 평가에 동시에 들어가는 누수를 차단합니다.

```bash
pileguard-prepare-nestler-detection --data-root ../data
```

결측 bbox 주석 프레임은 닭이 없는 negative로 변환하지 않고 제외합니다. 이 데이터의 단일 클래스는 닭 몸체가 아니라 NESTLER 포즈·스켈레톤 tracker 영역이며, PIO 몸체 bbox와 같은 라벨로 혼합하지 않습니다. 원본 프레임·라벨·절대경로 manifest는 Git에서 제외된 `outputs/`에 생성되고, 커밋 가능한 통계만 `artifacts/nestler-detection-dataset/summary.json`에 저장됩니다.

## NESTLER tracker-region detector fine-tuning

PIO 검출기 체크포인트를 일반 영상 특징 초기값으로만 사용하고, NESTLER train 757프레임·8,002개 tracker-region으로 YOLO26n을 미세조정합니다. 체크포인트 선택은 validation 798프레임에서만 하며 test 754프레임은 학습 단계에서 평가하지 않습니다.

```bash
pileguard-train-nestler
```

기준 실행(seed 2026, CPU, 5 epochs, 480px)의 validation 결과는 precision 0.5828, recall 0.3021, mAP50 0.2870, mAP50-95 0.1137입니다. 이 수치는 해외 NESTLER tracker-region 검출 성능이며 Piling 사건 예측 성능이 아닙니다. 국내 산란계 농장 성능은 아직 검증되지 않았습니다. 상세 기록은 [`training_summary.json`](artifacts/nestler-tracker-yolo26n/training_summary.json)에 저장됩니다.

## NESTLER tracker-region independent test

학습·체크포인트 선택에서 제외한 `job_000008`·`job_000011` 754프레임을 사전 고정한 기준(mAP50 0.5, recall 0.5)으로 최종 평가합니다. 결과 생성 후 기본 명령은 재실행을 거부하며, 이 test 결과를 사용한 추가 튜닝은 금지합니다.

```bash
pileguard-evaluate-nestler-test
```

독립 test 결과는 precision 0.4570, recall 0.2023, mAP50 0.1677, mAP50-95 0.0754로 사전 기준을 통과하지 못했습니다. Bulgaria `job_000008`은 mAP50 0.6072·recall 0.8169이지만 Rwanda `job_000011`은 mAP50 0.0782·recall 0.1074로 환경 간 격차가 큽니다. 따라서 현 체크포인트는 연구용으로만 보존하고 모니터링 연결을 차단합니다. 국내 산란계 농장 데이터 검증도 별도로 필요합니다. 상세 결과와 촬영지별 지표는 [`metrics.json`](artifacts/nestler-tracker-yolo26n/test/metrics.json)에 저장됩니다.

## NESTLER validation slice audit

동결한 test를 추가 튜닝에 사용하지 않기 위해 validation 798프레임만 촬영지와 주석 개체수 구간으로 분해합니다.

```bash
pileguard-audit-nestler-validation
```

Bulgaria `job_000007`의 mAP50은 0.4268이지만 Rwanda `job_000010`은 0.0772로 격차가 0.3496입니다. 프레임당 주석 개체수별 mAP50은 저밀도(0–5)가 0.0834, 중밀도(6–10)가 0.2517, 고밀도(11 이상)가 0.5232입니다. 다음 학습 설계는 test가 아닌 train/validation에서 Rwanda 촬영환경과 저밀도 작은 개체 샘플을 균형화해야 합니다. 상세 결과는 [`summary.json`](artifacts/nestler-tracker-yolo26n/validation-audit/summary.json)에 저장됩니다.

## NESTLER train-only density balancing

Validation 분석을 근거로 train 757프레임을 저·중·고밀도별 398개 항목씩 재표본화해 총 1,194개 학습 항목으로 2차 미세조정합니다. 원본 validation 798프레임은 학습에 추가하지 않고, 동결 test manifest는 학습 YAML에서 제거합니다.

```bash
pileguard-train-nestler-balanced
pileguard-audit-nestler-validation \
  --config configs/nestler_balanced_validation_audit.toml \
  --weights weights/nestler-balanced-yolo26n/best.pt
```

전체 validation은 precision 0.6122, recall 0.3314, mAP50 0.3289, mAP50-95 0.1340으로 기존 모델보다 각각 0.0294, 0.0293, 0.0419, 0.0203 개선됐습니다. 그러나 Rwanda clip의 mAP50은 0.0772에서 0.0688로, 저밀도 구간은 0.0834에서 0.0619로 하락했습니다. 두 구간의 recall은 각각 0.0028, 0.0349 증가했지만 핵심 도메인 약점은 해결되지 않았습니다. 따라서 이 모델도 연구 후보로만 보존하며 새 독립 데이터와 국내 산란계 현장 데이터가 확보되기 전에는 배포 성능을 주장하지 않습니다. 기존 test 결과는 이전 체크포인트에만 유효하므로 이 변경 모델에는 재사용하지 않았습니다. 상세 내용은 [`training_summary.json`](artifacts/nestler-balanced-yolo26n/training_summary.json)과 validation [`summary.json`](artifacts/nestler-balanced-yolo26n/validation-audit/summary.json)에 저장됩니다.

## PIO static density features

[PIO 데이터셋](https://doi.org/10.5281/zenodo.16686320)의 공식 Train·Validation 이미지와 YOLO 라벨 쌍을 검사하고, 이미지별 개체수·clipped bbox 면적합·중심 분산·최근접거리·4×4 격자 최대 집중도를 계산합니다.

```bash
pileguard-extract-pio --data-root ../data
```

결과는 `artifacts/pio-features/`의 이미지별 CSV, 전체·분할·환경·주령별 요약 JSON과 특징 그래프로 저장됩니다. PIO에는 Piling 사건 라벨이 없으므로 이 결과는 실제 축사 정지영상에서 밀도 특징 추출이 작동하는지 확인하는 용도이며 군집 사고 예측성능으로 해석하지 않습니다.

## PIO YOLO detection baseline

PIO 공식 Train 1,035장과 Validation 452장을 사용해 YOLO26n 닭 개체 검출 기준모델을 학습합니다. 머신별 경로가 커밋되지 않도록 실행 시점에 절대경로 manifest와 dataset YAML을 `outputs/`에 생성합니다.

```bash
# 4장씩으로 학습·평가 파이프라인 확인
pileguard-train-pio --data-root ../data --smoke

# 전체 학습과 독립 Validation 평가
pileguard-train-pio --data-root ../data
pileguard-evaluate-pio --data-root ../data \
  --weights weights/pio-yolo26n/best.pt
```

기준 실행(seed 2026, CPU, 5 epochs, 320 px)의 공식 Validation 결과는 precision 0.7079, recall 0.6132, mAP50 0.6362, mAP50-95 0.2823입니다. 환경별 mAP50은 commercial 0.6433, prototype 0.5921이며, 주차별 mAP50은 1주 0.3052에서 6주 0.7939로 차이가 큽니다. 어린 개체·작은 개체 구간의 개선이 후속 과제입니다. 상세 수치는 [`training_summary.json`](artifacts/pio-yolo26n/training_summary.json)과 [`metrics.json`](artifacts/pio-yolo26n/validation/metrics.json)에 저장됩니다.

이 검증은 해외 육계 개체 검출에 한정됩니다. PIO에는 Piling 사건 라벨이 없으므로 국내 산란계 현장의 압사 위험 예측이나 일반화 성능을 의미하지 않습니다. 또한 검출 실험 의존성인 Ultralytics는 AGPL-3.0 또는 Enterprise 라이선스로 제공되므로 상용 배포 전 라이선스 검토가 필요합니다.

## PIO predicted density features

학습된 YOLO 체크포인트의 예측 bbox를 기존 밀도 특징 계산기로 연결합니다. 공식 Validation 정답 bbox는 모델 입력에 사용하지 않고, 예측 특징을 만든 뒤 개체수·bbox 면적·중심 분산·최근접거리·격자 집중도 오차를 감사하는 용도로만 사용합니다.

```bash
pileguard-extract-pio-predicted \
  --data-root ../data \
  --weights weights/pio-yolo26n/best.pt
```

Validation 452장의 개선 실행에서 예측 bbox 73,301개와 유효 정답 bbox 73,857개를 비교했습니다. 480px 추론과 주령별 신뢰도(`1주 0.15`, `2~3주 0.20`, `4~6주 0.25`)를 적용한 결과, 개체수 MAE는 24.77마리, 정답 평균 대비 MAE는 15.16%, Pearson 상관계수는 0.9670입니다. bbox 면적합과 중심 분산의 상관계수는 각각 0.9787, 0.9335로 장면 간 밀도 변화 방향도 보존되었습니다.

기존 320px·신뢰도 0.15 대비 개체수 MAE는 28.87% 감소했고, 1주차 상대 MAE는 57.15%에서 28.37%로 감소했습니다. 주령별 설정은 같은 Validation에서 선택했으므로 독립 Test 성능이 아니라 PIO 내부 보정 결과입니다. 실시간 입력에는 파일명 대신 농장 관리정보의 입식 주령이 필요하며 국내 산란계 데이터에서 해상도와 임계값을 다시 보정해야 합니다. 상세 결과는 [`summary.json`](artifacts/pio-predicted-features/summary.json), [`threshold_sweep.json`](artifacts/pio-predicted-features/threshold_sweep.json), [`resolution_threshold_sweep.json`](artifacts/pio-predicted-features/resolution_threshold_sweep.json)에 저장됩니다.

## Digital Twin risk scenarios

제안서에 정의한 사회적 유인형, 집단이동 수렴형, 직원 추종·관측외형의 세 가지 합성 전조 시나리오를 생성합니다. NESTLER 특징에서 카메라별 median/IQR 기준선을 만들고, 정규화된 전조 근거로 0–100 위험지수와 hysteresis 경보를 계산합니다.

```bash
pileguard-simulate-risk
```

위험지수는 사건 발생확률이 아닌 데모용 합성 지수입니다. 시나리오의 시간척도, 국내 발생빈도와 예측성능은 실제 농장 사건영상으로 검증되지 않았습니다. 결과는 `artifacts/digital-twin/`에 저장됩니다.

## Integrated monitoring demo

NESTLER 프레임 특징을 초기 60프레임의 카메라별 median/IQR 기준선에 맞춰 정규화하고, Digital Twin과 동일한 위험지수·경보 로직으로 연결합니다. NESTLER에는 군집 압사 사건 라벨과 직원·문 맥락이 없으므로 결과는 이상징후 데모이며 사고 검출 성능이 아닙니다.

```bash
# 저장된 특징으로 위험 타임라인 재현
pileguard-run-demo

# 원본 NESTLER 영상의 특징 추출부터 한 번에 실행
pileguard-run-demo --extract --data-root ../data
```

결과는 `artifacts/integrated-demo/`에 저장됩니다. bbox 또는 움직임 특징이 부족한 프레임은 정상으로 간주하지 않고 `unavailable`로 표시합니다.

## Submission documents

- [`알고리즘 구현 및 분석`](docs/submission/IMPLEMENTATION_REPORT.md): 제안서의 미작성 구현·분석 부분을 교체할 실제 결과 본문
- [`제출 체크리스트`](docs/submission/SUBMISSION_CHECKLIST.md): 공고 기준 제출서류, 코드 제출 해석과 마감 전 확인사항
- [`평가기준 대응표`](docs/submission/EVIDENCE_MATRIX.md): 100점 평가항목과 코드·결과 증빙 연결

## Third-party data and licenses

공개데이터의 저작자·DOI·라이선스, AI Hub 공개 제외 범위와 선택 의존성 고지는
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)에서 확인할 수 있습니다.

## Project layout

```text
src/pileguard/       모델·데이터·평가 코드
configs/             학습 및 추론 설정
scripts/             재현 가능한 실행 명령
tests/               단위·스모크 테스트
artifacts/           Git에 포함할 소형 지표·그래프
outputs/             Git에 포함하지 않는 실행 결과
weights/             Git에 포함하지 않는 모델 가중치
```
