"""Fine-tune a NESTLER tracker-region detector without touching the test split."""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path
from typing import Any

from pileguard.models.yolo import (
    copy_artifacts,
    copy_best_checkpoint,
    normalize_detection_metrics,
    require_ultralytics,
    write_json,
)
from pileguard.runtime import resolve_device, seed_everything

EXPECTED_CLASS_NAME = "nestler_tracker_region"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def validate_dataset_contract(summary_path: Path, dataset_yaml: Path) -> dict[str, Any]:
    """Reject leaked, semantically incompatible, or incomplete prepared datasets."""

    if not dataset_yaml.is_file():
        raise FileNotFoundError(
            f"Prepared NESTLER dataset not found: {dataset_yaml}. "
            "Run pileguard-prepare-nestler-detection first."
        )
    if not summary_path.is_file():
        raise FileNotFoundError(f"NESTLER dataset audit not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("class_name") != EXPECTED_CLASS_NAME:
        raise ValueError(
            f"Expected class '{EXPECTED_CLASS_NAME}', found {summary.get('class_name')!r}"
        )
    if summary.get("frame_leakage_between_splits") is not False:
        raise ValueError("NESTLER dataset audit does not prove clip-level split isolation")
    if not str(summary.get("missing_bbox_policy", "")).startswith("exclude frame"):
        raise ValueError("Missing NESTLER bbox annotations must be excluded, not made negative")
    if not str(summary.get("independent_test_use", "")).startswith("reserved"):
        raise ValueError("NESTLER test split is not marked as reserved")

    splits = summary.get("splits", {})
    for split in ("train", "val", "test"):
        split_summary = splits.get(split, {})
        if int(split_summary.get("annotated_frames", 0)) < 1:
            raise ValueError(f"NESTLER split '{split}' has no annotated frames")
        if set(split_summary.get("sites", {})) != {"Bulgaria", "Rwanda"}:
            raise ValueError(f"NESTLER split '{split}' must contain both sites")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/nestler_tracker_yolo26n.toml")
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["output"]

    dataset_yaml = Path(str(data_config["dataset_yaml"]))
    dataset_summary_path = Path(str(data_config["dataset_summary"]))
    dataset_summary = validate_dataset_contract(dataset_summary_path, dataset_yaml)
    if str(data_config["class_name"]) != EXPECTED_CLASS_NAME:
        raise ValueError("Training config uses the wrong NESTLER class name")

    initialization_weights = args.weights or Path(
        str(model_config["initialization_weights"])
    )
    if not initialization_weights.is_file():
        raise FileNotFoundError(
            f"PIO initialization checkpoint not found: {initialization_weights}"
        )

    seed = int(training_config["seed"])
    seed_everything(seed)
    device = resolve_device(args.device or str(training_config["device"]))
    epochs = int(args.epochs or training_config["epochs"])
    image_size = int(args.image_size or training_config["image_size"])
    batch_size = int(args.batch_size or training_config["batch_size"])

    YOLO = require_ultralytics()
    model = YOLO(str(initialization_weights))
    run_root = Path(str(output_config["run_root"]))
    started_at = time.monotonic()
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
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
    split_counts = {
        split: {
            "annotated_frames": int(dataset_summary["splits"][split]["annotated_frames"]),
            "boxes": int(dataset_summary["splits"][split]["boxes"]),
        }
        for split in ("train", "val", "test")
    }
    summary = {
        "status": "validation-selected fine-tune",
        "architecture": str(model_config["architecture"]),
        "initialization": "PIO detector checkpoint used for feature initialization only",
        "class_name": EXPECTED_CLASS_NAME,
        "dataset": split_counts,
        "split_unit": "complete NESTLER job/clip",
        "epochs": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "device": str(device),
        "optimizer_requested": str(training_config["optimizer"]),
        "optimizer_selected": model.trainer.optimizer.__class__.__name__,
        "elapsed_seconds": elapsed_seconds,
        "validation_metrics": normalize_detection_metrics(results),
        "test_evaluated": False,
        "evidence_files": evidence_files,
        "checkpoint": checkpoint.as_posix(),
        "claim_boundary": (
            "Validation metrics measure overseas NESTLER pose/skeleton tracker-region "
            "detection only. The independent test clips remain untouched. NESTLER has no "
            "Piling event labels, and domestic Korean farm performance is unknown."
        ),
    }
    write_json(artifact_dir / "training_summary.json", summary)
    print(
        f"status={summary['status']} train={split_counts['train']['annotated_frames']} "
        f"val={split_counts['val']['annotated_frames']} "
        f"metrics={summary['validation_metrics']} test_evaluated=False"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
