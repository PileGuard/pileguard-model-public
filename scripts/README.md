# Scripts

학습, 평가, 특징 추출 및 데모 실행 명령을 저장합니다. 모든 명령은 저장소 루트에서 재현 가능해야 합니다.

```bash
pileguard-audit-domestic-data --data-root ../data
pileguard-check-data --data-root ../data
pileguard-extract-pio --data-root ../data
pileguard-train-pio --data-root ../data
pileguard-evaluate-pio --data-root ../data --weights weights/pio-yolo26n/best.pt
pileguard-extract-pio-predicted --data-root ../data --weights weights/pio-yolo26n/best.pt
pileguard-extract-nestler --data-root ../data
pileguard-prepare-nestler-detection --data-root ../data
pileguard-train-nestler
pileguard-evaluate-nestler-test
pileguard-audit-nestler-validation
pileguard-train-nestler-balanced
pileguard-audit-nestler-validation --config configs/nestler_balanced_validation_audit.toml --weights weights/nestler-balanced-yolo26n/best.pt
pileguard-evaluate-nestler-transfer --data-root ../data --weights weights/pio-yolo26n/best.pt
pileguard-simulate-risk
pileguard-run-demo
```
