# Third-party notices

이 저장소는 아래 데이터셋의 원본 이미지·영상·라벨과 모델 가중치를 재배포하지
않습니다. 저장소에 포함된 지표·표·그래프·특징 파일은 PileGuard가 2026년에 만든
분석 결과 또는 변형물이며, 원 저작자가 PileGuard를 보증한다는 의미가 아닙니다.

## Public datasets

### MSU Poultry Piling Dataset v2

- Creators: Daniel Morris, Yunfei Long, Benjamin Smith, Janice Siegford
- Institution: Michigan State University
- DOI: <https://doi.org/10.17632/pgp8mj6ms4.2>
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- Use in this repository: aggregate classification metrics and plots produced from the official splits

### NESTLER Poultry Behaviour Analytics Detection Dataset v1

- Creators: RINISOFT and NESTLER Horizon Project
- DOI: <https://doi.org/10.5281/zenodo.20924893>
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- Use in this repository: derived frame features, aggregate detection metrics, and camera-relative risk timelines

### PIO v1

- Creators: Keyla Boniche and Edmanuel Cruz
- DOI: <https://doi.org/10.5281/zenodo.16686320>
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- Use in this repository: derived image features, aggregate object-detection metrics, and plots

## AI Hub dataset 575

- Source: [지능형 스마트 축사 데이터(육계, 산란계)](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=575)
- Provider: AI Hub, Ministry of Science and ICT, and National Information Society Agency
- Terms: [AI Hub 이용정책](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105) 및 데이터 다운로드 시 동의한 조건
- Use in this repository: source-attributed aggregate evaluation metrics and plots only

AI Hub 원본·라벨·모델 가중치·이미지별 파일명·이미지별 예측 또는 오류 목록은
공개 저장소에 포함하지 않습니다. 해당 자료를 이용하려면 AI Hub에서 직접 이용
승인을 받고 원본 데이터를 내려받아야 합니다.

## Software dependency notice

Ultralytics는 학습·검출 실험을 위한 선택 의존성이며 이 저장소에 그 소스나 모델
가중치를 포함하지 않습니다. Ultralytics 사용 및 배포에는 해당 프로젝트가 제시한
AGPL-3.0 또는 Enterprise 조건이 적용될 수 있으므로 상용 배포 전에 별도로 검토해야
합니다.
