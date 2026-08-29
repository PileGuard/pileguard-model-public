# 국내 산란계 데이터 반입 규격

이 규격은 농장 소유자에게 직접 제공받거나 사용 승인을 받은 국내 산란계 영상만 대상으로 합니다. 공개 출처가 불명확한 링크, 파일공유 사이트, 실행파일 또는 압축파일은 반입하지 않습니다.

## 디렉터리

원본 데이터는 Git 저장소 밖 `PILEGUARD_DATA_ROOT/incoming/domestic_v1/`에 둡니다.

```text
domestic_v1/
├── clips.csv
├── events.csv
└── videos/
    └── <가명 clip_id>.mp4
```

`clips.csv`와 `events.csv`는 [`docs/templates/domestic_clips.csv`](templates/domestic_clips.csv), [`docs/templates/domestic_events.csv`](templates/domestic_events.csv)의 헤더를 그대로 사용합니다. 허용되지 않은 추가 열에는 사람 이름, 전화번호, 농장 실명, 주소, GPS 좌표를 넣지 않습니다.

## 가명화와 승인

- `farm_id`, `house_id`, `camera_id`, `clip_id`, `event_id`는 영문 대문자로 시작하는 3~32자의 가명 ID만 사용합니다.
- 영상에 사람 얼굴, 차량번호, 문서, 음성이 포함되면 제공자가 제거한 사본을 전달해야 합니다.
- 영상은 제공자가 표준 ISO-BMFF MP4/MOV 또는 RIFF AVI로 내보내야 합니다. 전용 카메라 컨테이너는 제공자 측에서 변환하고, 출처 불명 변환 사이트나 프로그램은 사용하지 않습니다.
- `consent_status`는 데이터 소유자의 연구·경진대회 사용 승인을 확인한 `research_approved`만 허용합니다.
- SHA-256은 제공자가 전달한 원본 체크섬을 기록하며, 감사 도구가 실제 파일과 일치하는지 확인합니다.
- 영상·manifest·감사 결과는 Git에 올리지 않습니다.

## 라벨과 분할

- `clip_outcome`: `normal`, `near_piling`, `piling`
- `review_status`: 서로 독립적인 2인 검수 완료(`double_reviewed`) 또는 불일치 조정 완료(`adjudicated`)
- 사건 clip은 `events.csv`에 시작·종료 초와 동일한 사건 유형이 하나 이상 있어야 합니다.
- 정상 clip은 사건 행을 가지면 안 됩니다. 검수하지 않은 영상을 정상으로 간주하지 않습니다.
- 같은 `farm_id`는 하나의 split에만 포함합니다. 프레임이나 clip 단위 무작위 분할은 농장 환경 누수를 일으키므로 금지합니다.
- `train`, `val`, `test` 각각에 최소 `normal`과 `piling` clip이 모두 있어야 검증 준비 완료로 판정합니다.
- `test` 농장은 모델·임계값 선택에 사용하지 않고 한 번의 독립 평가용으로 동결합니다.

## 감사 실행

```bash
pileguard-audit-domestic-data \
  --data-root ../data \
  --config configs/domestic_data_intake.toml
```

도구는 안전한 상대경로, 심볼릭 링크, 파일 시그니처, SHA-256, 중복 영상, 영상 재생 가능 여부, 길이·해상도·FPS, 라벨 범위, 농장 단위 split을 검사합니다. 결과는 Git에서 제외된 `outputs/domestic-data-intake/audit.json`에 저장됩니다. `ready_for_model_validation=true`가 되기 전에는 국내 현장 성능을 주장하지 않습니다.

## 요청해야 할 최소 자료

- 서로 다른 국내 산란계 농장 3곳 이상: `train`, `val`, 독립 `test`에 농장 단위로 분리
- 각 농장의 정상 구간과 군집 밀집·압사 위험 또는 실제 Piling 전후 구간
- 카메라 설치 위치, 해상도, FPS는 가명화된 관리정보로 제공
- 사건 시작·종료 시각에 대한 2인 이상 검수 또는 불일치 조정 기록
- 데이터 소유자의 연구·경진대회 사용 승인과 SHA-256 체크섬

실제 사건영상이 부족하면 정상/이상 탐지 기술 검증까지만 수행하고 사고 예측 성능으로 표현하지 않습니다.
