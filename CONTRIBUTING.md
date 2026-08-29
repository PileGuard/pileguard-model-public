# Team Workflow

## Branch naming

브랜치 이름은 `종류/작업-내용` 형식을 사용합니다. 작업 내용은 영문 소문자와 하이픈으로 작성합니다.

- `feature/<topic>`: 새 기능 또는 모델 개발
- `fix/<topic>`: 오류 수정
- `experiment/<topic>`: 실험 및 성능 비교
- `docs/<topic>`: 문서 작업

예시: `feature/piling-detector`, `experiment/yolo-baseline`, `docs/model-report`

모든 변경은 작업 브랜치에서 진행하고 Pull Request를 통해 `main`에 병합합니다.
