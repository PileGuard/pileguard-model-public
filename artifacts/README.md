# Artifacts

제안서에 사용할 집계 지표와 그래프를 저장합니다. 원본 데이터, 모델 가중치, AI Hub 이미지별 파일명·예측·오류 목록은 포함하지 않습니다. AI Hub 이미지별 진단 CSV는 이용 승인을 받은 로컬 환경에서만 생성·보관하며 Git에서 제외합니다.

- `msu-resnet18/`: 학습·공식 Test 지표
- 국내 데이터 감사 결과는 농장 정보 보호를 위해 `outputs/domestic-data-intake/`에만 저장
- `aihub-domestic-validation/`: AI Hub 공식 국내 산란계 군집 Validation 전이 집계 지표
- `aihub-laying-hen-yolo26n/`: AI Hub 산란계 미세조정 학습, Validation 재평가, 임계값·오류 감사 집계 지표
- `aihub-video-risk/`: 미세조정 검출 bbox·optical flow·카메라 상대 위험지수 영상 통합 기능시험
- `nestler-features/`: 영상 프레임 특징과 요약
- `nestler-detector-transfer/`: PIO 검출기의 NESTLER 전이 지표와 모니터링 차단 근거
- `nestler-detection-dataset/`: 클립 단위 분할과 tracker-region 라벨 품질 감사
- `nestler-tracker-yolo26n/`: NESTLER tracker-region 검출기 학습·validation 지표
  - `test/`: 잠금한 클립의 독립 test 지표와 배포 차단 결정
  - `validation-audit/`: test 재사용 없이 촬영지·밀도별로 분해한 validation 지표
- `nestler-balanced-yolo26n/`: train 밀도 균형 샘플링 재학습과 validation 비교
  - `validation-audit/`: 동일한 validation 구간의 기존 모델 대비와 test 미사용 기록
- `pio-features/`: PIO 이미지별 정적 밀도 특징과 품질 감사
- `pio-yolo26n/`: PIO YOLO26n 학습·공식 Validation 지표와 그래프
- `pio-predicted-features/`: 예측 bbox 밀도 특징, 정답 대비 오차와 임계값 비교
- `digital-twin/`: 문헌 기반 합성 위험곡선
- `integrated-demo/`: NESTLER 특징 기반 위험 타임라인
