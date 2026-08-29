"""Evaluate a trained PIO YOLO checkpoint on the untouched official validation split."""

from __future__ import annotations

import argparse
import shutil
import time
import tomllib
from pathlib import Path
from typing import Any

from pileguard.data.pio_detection import (
    discover_split_images,
    write_dataset_from_images,
    write_runtime_dataset,
)
from pileguard.data_inventory import resolve_data_root
from pileguard.features.pio import parse_image_metadata
from pileguard.models.yolo import normalize_detection_metrics, require_ultralytics, write_json
from pileguard.runtime import resolve_device

EVIDENCE_PLOTS = (
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "F1_curve.png",
    "P_curve.png",
    "PR_curve.png",
    "R_curve.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxPR_curve.png",
    "BoxR_curve.png",
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def copy_evidence_plots(run_dir: Path, artifact_dir: Path) -> list[str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for filename in EVIDENCE_PLOTS:
        source = run_dir / filename
        if source.is_file():
            shutil.copy2(source, artifact_dir / filename)
            copied.append(filename)
    return copied


def build_evaluation_groups(images: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for image in images:
        environment, week = parse_image_metadata(image.stem)
        groups.setdefault(f"environment-{environment}", []).append(image)
        if week is not None:
            groups.setdefault(f"week-{week}", []).append(image)
    return dict(sorted(groups.items()))


def evaluate_groups(
    *,
    model: Any,
    images: list[Path],
    output_dir: Path,
    class_names: list[str],
    image_size: int,
    batch_size: int,
    device: str,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> dict[str, Any]:
    group_results: dict[str, Any] = {}
    for group_name, group_images in build_evaluation_groups(images).items():
        dataset_yaml = write_dataset_from_images(
            train_images=group_images,
            validation_images=group_images,
            output_dir=output_dir / "runtime" / group_name,
            class_names=class_names,
        )
        results = model.val(
            data=str(dataset_yaml),
            split="val",
            imgsz=image_size,
            batch=batch_size,
            workers=0,
            device=device,
            conf=confidence_threshold,
            iou=iou_threshold,
            max_det=max_detections,
            project=str(output_dir),
            name=group_name,
            exist_ok=True,
            plots=False,
            verbose=False,
        )
        group_results[group_name] = {
            "image_count": len(group_images),
            "metrics": normalize_detection_metrics(results),
        }
    return group_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pio_yolo26n.toml"))
    parser.add_argument("--data-root")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    output_config = config["output"]

    device = resolve_device(args.device or str(training_config["device"]))
    data_root = resolve_data_root(args.data_root)
    dataset_root = data_root / str(data_config["dataset_path"])
    run_root = Path(str(output_config["run_root"]))
    run_name = "val-smoke" if args.smoke else "val"
    dataset_yaml, dataset_counts = write_runtime_dataset(
        dataset_root=dataset_root,
        output_dir=run_root / run_name / "runtime",
        train_split=str(data_config["train_split"]),
        validation_split=str(data_config["validation_split"]),
        class_names=list(data_config["class_names"]),
        smoke_limit=4 if args.smoke else None,
        seed=int(training_config["seed"]),
    )
    weights = args.weights or Path(str(output_config["checkpoint_dir"])) / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"PIO YOLO checkpoint not found: {weights}")

    image_size = 320 if args.smoke else int(args.image_size or training_config["image_size"])
    batch_size = 2 if args.smoke else int(args.batch_size or training_config["batch_size"])
    YOLO = require_ultralytics()
    model = YOLO(str(weights))
    started_at = time.monotonic()
    results = model.val(
        data=str(dataset_yaml),
        split="val",
        imgsz=image_size,
        batch=batch_size,
        workers=0,
        device=str(device),
        conf=float(evaluation_config["confidence_threshold"]),
        iou=float(evaluation_config["iou_threshold"]),
        max_det=int(evaluation_config["max_detections"]),
        project=str(run_root),
        name=run_name,
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    elapsed_seconds = time.monotonic() - started_at
    artifact_dir = Path(str(output_config["artifact_dir"])) / "validation"
    if args.smoke:
        artifact_dir = Path("outputs/smoke/pio-yolo26n/validation")
    run_dir = Path(results.save_dir)
    copied_plots = copy_evidence_plots(run_dir, artifact_dir)
    validation_images = discover_split_images(dataset_root, str(data_config["validation_split"]))
    grouped_metrics = {}
    if bool(evaluation_config["grouped"]) and not args.no_groups and not args.smoke:
        grouped_metrics = evaluate_groups(
            model=model,
            images=validation_images,
            output_dir=run_root / "group-validation",
            class_names=list(data_config["class_names"]),
            image_size=image_size,
            batch_size=batch_size,
            device=str(device),
            confidence_threshold=float(evaluation_config["confidence_threshold"]),
            iou_threshold=float(evaluation_config["iou_threshold"]),
            max_detections=int(evaluation_config["max_detections"]),
        )
    metrics = {
        "status": "smoke" if args.smoke else "official-validation",
        "architecture": str(config["model"]["architecture"]),
        "dataset": {"validation_images": dataset_counts["validation_images"]},
        "image_size": image_size,
        "batch_size": batch_size,
        "device": str(device),
        "elapsed_seconds": elapsed_seconds,
        "metrics": normalize_detection_metrics(results),
        "grouped_metrics": grouped_metrics,
        "evidence_plots": copied_plots,
        "claim_boundary": (
            "PIO is an overseas broiler detection dataset without Piling event labels; "
            "metrics validate chicken detection only."
        ),
    }
    write_json(artifact_dir / "metrics.json", metrics)
    print(
        f"status={metrics['status']} val={dataset_counts['validation_images']} "
        f"metrics={metrics['metrics']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
