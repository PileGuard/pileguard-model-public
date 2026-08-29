# Configs

학습·평가·특징 추출 설정을 단계별 TOML 파일로 관리합니다. 데이터의 절대 경로와 비밀정보는 커밋하지 않습니다.

- `msu_resnet18.toml`: MSU 분류 기준모델
- `domestic_data_intake.toml`: 국내 산란계 영상 반입·무결성·라벨·농장 분할 감사
- `aihub_domestic_validation.toml`: AI Hub 국내 산란계 군집 Validation 전이 평가
- `aihub_finetuned_validation.toml`: AI Hub 산란계 미세조정 검출기의 동일 Validation 재평가
- `aihub_video_risk.toml`: AI Hub 미세조정 검출기와 카메라 상대 위험지수 영상 통합 실행
- `nestler_features.toml`: NESTLER 영상·추적·광류 특징
- `nestler_detector_transfer.toml`: PIO 검출기의 NESTLER 도메인 전이 감사
- `nestler_detection_dataset.toml`: NESTLER 클립 단위 검출 데이터 분할·추출
- `nestler_tracker_yolo26n.toml`: NESTLER tracker-region YOLO26n 미세조정
- `nestler_tracker_test.toml`: NESTLER tracker-region 독립 test 최종 평가
- `nestler_validation_audit.toml`: test를 건드리지 않는 촬영지·밀도별 validation 감사
- `nestler_balanced_yolo26n.toml`: Rwanda·저밀도 train 균형 샘플링 미세조정
- `nestler_balanced_validation_audit.toml`: 균형 재학습 모델의 validation 구간별 기준 대비
- `pio_features.toml`: PIO YOLO 정지영상 밀도 특징
- `pio_yolo26n.toml`: PIO YOLO26n 개체 검출 학습·평가
- `pio_predicted_features.toml`: PIO 예측 bbox 기반 밀도 특징과 오차 분석
- `digital_twin.toml`: 문헌 기반 합성 시나리오
- `integrated_demo.toml`: NESTLER 특징과 위험지수 통합 데모
