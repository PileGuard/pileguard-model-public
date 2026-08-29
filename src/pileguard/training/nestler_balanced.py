"""Fine-tune NESTLER using train-only density-balanced sampling."""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path
from typing import Any

from pileguard.data.nestler_balanced import write_balanced_dataset
from pileguard.models.yolo import (
    copy_artifacts,
    copy_best_checkpoint,
    normalize_detection_metrics,
    require_ultralytics,
    write_json,
)
from pileguard.runtime import resolve_device, seed_everything


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def validate_frozen_contract(input_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        name: Path(str(input_config[name]))
        for name in (
            "dataset_summary",
            "baseline_training_summary",
            "validation_audit",
            "frozen_test_metrics",
        )
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"NESTLER {name} not found: {path}")
    baseline = json.loads(paths["baseline_training_summary"].read_text(encoding="utf-8"))
    validation_audit = json.loads(paths["validation_audit"].read_text(encoding="utf-8"))
    frozen_test = json.loads(paths["frozen_test_metrics"].read_text(encoding="utf-8"))
    if baseline.get("test_evaluated") is not False:
        raise ValueError("Baseline training unexpectedly evaluated test")
    if validation_audit.get("test_inference_performed") is not False:
        raise ValueError("Validation audit unexpectedly performed test inference")
    if frozen_test.get("tuning_after_test_prohibited") is not True:
        raise ValueError("Independent test is not marked as frozen")
    return baseline, frozen_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/nestler_balanced_yolo26n.toml")
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    input_config = config["input"]
    balancing_config = config["balancing"]
    training_config = config["training"]
    output_config = config["output"]
    baseline, frozen_test = validate_frozen_contract(input_config)

    seed = int(training_config["seed"])
    dataset_yaml, balance_audit = write_balanced_dataset(
        source_dataset_yaml=Path(str(input_config["dataset_yaml"])),
        dataset_summary_path=Path(str(input_config["dataset_summary"])),
        output_dir=Path(str(output_config["runtime_dir"])),
        low_maximum=int(balancing_config["low_maximum"]),
        medium_maximum=int(balancing_config["medium_maximum"]),
        seed=seed,
    )
    initialization_weights = args.weights or Path(str(input_config["initialization_weights"]))
    if not initialization_weights.is_file():
        raise FileNotFoundError(
            f"NESTLER initialization checkpoint not found: {initialization_weights}"
        )

    seed_everything(seed)
    device = resolve_device(args.device or str(training_config["device"]))
    epochs = int(args.epochs or training_config["epochs"])
    YOLO = require_ultralytics()
    model = YOLO(str(initialization_weights))
    run_root = Path(str(output_config["run_root"]))
    started_at = time.monotonic()
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=int(training_config["image_size"]),
        batch=int(training_config["batch_size"]),
        workers=int(training_config["workers"]),
        patience=int(training_config["patience"]),
        seed=seed,
        deterministic=True,
        device=str(device),
        max_det=int(training_config["max_detections"]),
        close_mosaic=min(int(training_config["close_mosaic"]), epochs),
        optimizer=str(training_config["optimizer"]),
        weight_decay=float(training_config["weight_decay"]),
        project=str(run_root),
        name="train",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    elapsed_seconds = time.monotonic() - started_at
    run_dir = Path(model.trainer.save_dir)
    artifact_dir = Path(str(output_config["artifact_dir"]))
    evidence_files = copy_artifacts(
        run_dir,
        artifact_dir,
        ("results.csv", "results.png", "labels.jpg", "labels_correlogram.jpg"),
    )
    checkpoint = copy_best_checkpoint(
        run_dir / "weights" / "best.pt", Path(str(output_config["checkpoint_dir"]))
    )
    validation_metrics = normalize_detection_metrics(results)
    baseline_metrics = baseline["validation_metrics"]
    summary = {
        "status": "validation-only balanced fine-tune",
        "initialization": "validation-selected NESTLER checkpoint",
        "class_name": "nestler_tracker_region",
        "balance_audit": balance_audit,
        "epochs": epochs,
        "image_size": int(training_config["image_size"]),
        "batch_size": int(training_config["batch_size"]),
        "device": str(device),
        "optimizer_requested": str(training_config["optimizer"]),
        "optimizer_selected": model.trainer.optimizer.__class__.__name__,
        "elapsed_seconds": elapsed_seconds,
        "baseline_validation_metrics": baseline_metrics,
        "validation_metrics": validation_metrics,
        "validation_delta": {
            name: float(validation_metrics[name]) - float(baseline_metrics[name])
            for name in ("precision", "recall", "map50", "map50_95")
        },
        "test_evaluated": False,
        "frozen_test_reused": False,
        "frozen_test_checkpoint": frozen_test["status"],
        "evidence_files": evidence_files,
        "checkpoint": checkpoint.as_posix(),
        "claim_boundary": (
            "This second-stage model is selected on the existing validation clips only. "
            "Frozen test metrics apply to the previous checkpoint and cannot be claimed for "
            "this changed model. New independent and domestic data are required."
        ),
    }
    write_json(artifact_dir / "training_summary.json", summary)
    print(
        f"status={summary['status']} rows={balance_audit['training_manifest_rows']} "
        f"metrics={validation_metrics} delta={summary['validation_delta']} "
        "test_evaluated=False"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
