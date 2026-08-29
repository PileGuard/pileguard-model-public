# 알고리즘 부문 평가기준 대응표

공고에 기재된 알고리즘 부문 평가항목 100점을 기준으로 제안서에서 강조할 내용과
검증 근거를 연결했다. 이 표는 자체 예상점수가 아니라 제출자료 누락 방지용이다.

| 평가항목 | 배점 | 제안서에서 강조할 내용 | 구현·근거 | 한계 또는 후속조치 |
|---|---:|---|---|---|
| 필요성·독창성 | 10 | 정지영상 감지에서 형성과정·개입효과 관리로 확장 | Detect–Predict–Adaptive 단계 구조, 카메라 상대 위험도 | Adaptive 자동제어는 미실증 |
| 데이터 다양성 | 10 | MSU·NESTLER·PIO·AI Hub·문헌 시나리오를 역할별로 분리 | 각 데이터셋의 목적·라이선스·DOI 또는 공식 URL | PIO는 육계, AI Hub는 정지영상, ChickenVerse는 핵심 실험에 미사용 |
| 전처리·가공 | 10 | 공식 분할 유지, 카메라 기준선, 결측·경계·중복 처리 | NESTLER 결측 83프레임, PIO 경계초과 박스 2,889개, AI Hub 교차분할 중복 180그룹 제거 | 농장별 ROI·품질 마스크 추가 필요 |
| 데이터 확보·품질 | 10 | 공개데이터 즉시 재현 + 국내 사건영상 수집계획 | 체크섬 기반 분할 누수 감사, 데이터 인벤토리, 수집 필드 정의 | 국내 연속 사건영상은 아직 없음 |
| 학습·분석 | 10 | ResNet18과 국내 산란계 YOLO 재현, 카메라·산란단계별 오류분석 | MSU Test F1 0.9785, AI Hub mAP50 0.8017·중심점 F1 0.7467 | AI Hub 수치는 동일 Validation 선택 결과이며 외부농장 미검증 |
| 테스트·적용 가능성 | 10 | 기능별 검증수준 분리, 자동화된 테스트 | 81 tests, 22 CLI, AI Hub 4,546장 평가, NESTLER 2,392프레임 예측 Box 통합시험 | NESTLER·PIO·AI Hub에는 연속 사고 정답 없음 |
| AI 프로그램 구현 | 5 | 설치 가능한 CLI와 설정 기반 파이프라인 | 분류·검출·평가·오류감사·특징·시뮬레이션·영상 위험 CLI | 1·3·5분 TCN은 국내 사건영상 확보 후 학습 |
| 활용·상용화 | 15 | 기존 CCTV·Edge AI, Shadow Mode, 단계적 활성화 | 로컬 추론 구조, 원본영상 외부전송 최소화 설계 | 농장 RTSP·알림·LED 연계 미구현 |
| 현장문제 해결 | 15 | 야간 순찰 부담, 질식·폐사 위험의 조기 확인 | 정상/관찰/경고/위험 타임라인과 관리자 알림 구조 | 현장 경보효과 미실증 |
| 활용실적·기대효과 | 5 | 공개데이터 재현결과와 실증 KPI 제시 | 사건 Recall·일일 오경보·선행시간·분산시간 정의 | 생산성·폐사율 개선 수치 주장 금지 |

## 증빙 파일 빠른 연결

| 주장 | 근거 파일 |
|---|---|
| 공식 Test Accuracy/Precision/Recall/F1/PR-AUC | `artifacts/msu-resnet18/test/metrics.json` |
| Validation 최적 epoch와 학습곡선 | `artifacts/msu-resnet18/training_history.json`, `training_curves.png` |
| NESTLER 프레임 수·주석 커버리지·특징분포 | `artifacts/nestler-features/summary.json` |
| PIO 이미지·라벨 쌍, 주석 품질, 정적 밀도 특징분포 | `artifacts/pio-features/summary.json`, `image_features.csv` |
| AI Hub 공식 분할·중복·주석 품질 감사 | `artifacts/aihub-laying-hen-dataset/summary.json` |
| 국내 산란계 YOLO 학습지표 | `artifacts/aihub-laying-hen-yolo26n/training_summary.json`, `results.png` |
| 전체 Validation 이전학습 비교·임계값별 성능 | `artifacts/aihub-domestic-validation/summary.json`, `artifacts/aihub-laying-hen-yolo26n/validation/summary.json` |
| 산란단계별 계수오차와 임계값 감사 | `artifacts/aihub-laying-hen-yolo26n/error-audit/summary.json`, `error_audit.png` |
| 예측 Box·광류 기반 영상 위험 기능시험 | `artifacts/aihub-video-risk/summary.json`, `risk_timelines.png` |
| 문헌 기반 3개 시나리오 경보 전환 | `artifacts/digital-twin/scenario_summary.json`, `risk_curves.png` |
| 카메라 기준선·결측 품질 게이트·위험 타임라인 | `artifacts/integrated-demo/summary.json`, `risk_timelines.png` |
| 재현 명령과 데이터 비포함 원칙 | `README.md`, `.gitignore` |
| 자동화된 테스트 | `tests/` |

## 발표·문서에서 피해야 할 표현

- NESTLER 위험단계를 실제 Piling 적중 또는 오경보로 부르지 않는다.
- AI Hub Validation을 독립 Test나 국내 현장 사건예측 성능으로 부르지 않는다.
- NESTLER 예측 Box 위험단계를 사고 적중·오경보·선행시간으로 부르지 않는다.
- PIO 육계 밀도 특징을 국내 산란계 또는 Piling 예측성능으로 부르지 않는다.
- 합성 시나리오 시간을 실제 선행시간으로 부르지 않는다.
- 해외 관찰 사건비율을 국내 발생확률로 부르지 않는다.
- MSU 정지영상 성능을 1·3·5분 전조예측 성능으로 확장하지 않는다.
- 평가 루프 처리량을 Edge 장치의 실시간 FPS로 단정하지 않는다.
- LED 개입효과와 생산성 개선을 자체 실증결과로 표현하지 않는다.
