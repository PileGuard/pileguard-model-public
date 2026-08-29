"""Fine-tune the PIO detector on official Korean AI Hub laying-hen boxes."""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path
from typing import Any

from pileguard.data.pio_detection import select_smoke_images, write_dataset_from_images
from pileguard.data_inventory import resolve_data_root
from pileguard.models.yolo import (
    copy_artifacts,
    copy_best_checkpoint,
    normalize_detection_metrics,
    require_ultralytics,
    write_json,
)
from pileguard.runtime import resolve_device, seed_everything

EXPECTED_CLASS_NAME = "layer-chicken"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def validate_dataset_contract(summary_path: Path) -> dict[str, Any]:
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"AI Hub dataset audit not found: {summary_path}. "
            "Run pileguard-prepare-aihub-detection first."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not str(summary.get("status", "")).startswith("ready"):
        raise ValueError("AI Hub dataset preparation is blocked by its quality audit")
    dataset = summary.get("dataset", {})
    if dataset.get("class_names") != [EXPECTED_CLASS_NAME]:
        raise ValueError("AI Hub dataset audit has an incompatible detection class")
    if dataset.get("official_count_contract", {}).get("ok") is not True:
        raise ValueError("AI Hub official file and box counts were not proven")
    leakage = summary.get("selected_dataset", {})
    leakage_fields = (
        "training_validation_content_overlap_count",
        "training_validation_filename_overlap_count",
    )
    if any(int(leakage.get(field, -1)) != 0 for field in leakage_fields):
        raise ValueError("AI Hub Training/Validation isolation was not proven")
    splits = dataset.get("splits", {})
    for split in ("train", "val"):
        if int(splits.get(split, {}).get("image_count", 0)) < 1:
            raise ValueError(f"AI Hub split '{split}' has no images")
        if int(splits[split].get("usable_box_count", 0)) < 1:
            raise ValueError(f"AI Hub split '{split}' has no usable boxes")
    return summary


def load_selected_images(dataset_root: Path, manifest_name: str) -> list[Path]:
    manifest_path = dataset_root / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Selected AI Hub image manifest not found: {manifest_path}")
    paths = [
        dataset_root / line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Selected AI Hub images are missing: {missing[:5]}")
    missing_labels = [
        path
        for path in paths
        if not dataset_root.joinpath(
            "labels", path.parent.name, f"{path.stem}.txt"
        ).is_file()
    ]
    if missing_labels:
        raise FileNotFoundError(f"Selected AI Hub labels are missing: {missing_labels[:5]}")
    if not paths:
        raise ValueError(f"Selected AI Hub manifest is empty: {manifest_path}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/aihub_laying_hen_yolo26n.toml")
    )
    parser.add_argument("--data-root")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["output"]

    dataset_summary = validate_dataset_contract(Path(str(data_config["dataset_summary"])))
    if list(data_config["class_names"]) != [EXPECTED_CLASS_NAME]:
        raise ValueError("Training config uses the wrong AI Hub class name")
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
    data_root = resolve_data_root(args.data_root)
    dataset_root = data_root / str(data_config["dataset_path"])
    run_name = "smoke" if args.smoke else "train"
    run_root = Path(str(output_config["run_root"]))
    runtime_dir = run_root / run_name / "runtime"
    selected = dataset_summary["selected_dataset"]
    train_images = load_selected_images(dataset_root, str(selected["train_manifest"]))
    validation_images = load_selected_images(
        dataset_root, str(selected["validation_manifest"])
    )
    if args.smoke:
        train_images = select_smoke_images(train_images, limit=8, seed=seed)
        validation_images = select_smoke_images(validation_images, limit=8, seed=seed + 1)
    dataset_yaml = write_dataset_from_images(
        train_images=train_images,
        validation_images=validation_images,
        output_dir=runtime_dir,
        class_names=list(data_config["class_names"]),
    )
    dataset_counts = {
        "train_images": len(train_images),
        "validation_images": len(validation_images),
    }

    epochs = 1 if args.smoke else int(args.epochs or training_config["epochs"])
    image_size = 320 if args.smoke else int(args.image_size or training_config["image_size"])
    batch_size = 2 if args.smoke else int(args.batch_size or training_config["batch_size"])
    YOLO = require_ultralytics()
    model = YOLO(str(initialization_weights))
    started_at = time.monotonic()
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        workers=0 if args.smoke else int(training_config["workers"]),
        patience=int(training_config["patience"]),
        seed=seed,
        deterministic=True,
        device=str(device),
        max_det=int(training_config["max_detections"]),
        close_mosaic=min(int(training_config["close_mosaic"]), epochs),
        optimizer=str(training_config["optimizer"]),
        lr0=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
        cache=False,
        project=str(run_root),
        name=run_name,
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    elapsed_seconds = time.monotonic() - started_at
    run_dir = Path(model.trainer.save_dir)
    artifact_dir = (
        Path("outputs/smoke/aihub-laying-hen-yolo26n")
        if args.smoke
        else Path(str(output_config["artifact_dir"]))
    )
    evidence_files = copy_artifacts(
        run_dir,
        artifact_dir,
        ("results.csv", "results.png", "labels.jpg", "labels_correlogram.jpg"),
    )
    checkpoint_dir = (
        artifact_dir / "weights"
        if args.smoke
        else Path(str(output_config["checkpoint_dir"]))
    )
    checkpoint = copy_best_checkpoint(run_dir / "weights" / "best.pt", checkpoint_dir)
    summary = {
        "status": "smoke" if args.smoke else "validation-selected fine-tune",
        "architecture": str(model_config["architecture"]),
        "initialization": "PIO YOLO26n checkpoint",
        "class_name": EXPECTED_CLASS_NAME,
        "dataset": {
            **dataset_counts,
            "audited_train_images": int(
                dataset_summary["dataset"]["splits"]["train"]["image_count"]
            ),
            "audited_validation_images": int(
                dataset_summary["dataset"]["splits"]["val"]["image_count"]
            ),
        },
        "epochs": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "device": str(device),
        "optimizer_requested": str(training_config["optimizer"]),
        "optimizer_selected": model.trainer.optimizer.__class__.__name__,
        "elapsed_seconds": elapsed_seconds,
        "validation_metrics": normalize_detection_metrics(results),
        "evidence_files": evidence_files,
        "checkpoint": checkpoint.as_posix(),
        "claim_boundary": (
            "Validation metrics measure static Korean laying-hen object detection only. "
            "The dataset has no continuous piling/smothering incident timing, and this "
            "Validation split is not an independent field-farm test."
        ),
    }
    write_json(artifact_dir / "training_summary.json", summary)
    print(
        f"status={summary['status']} train={dataset_counts['train_images']} "
        f"val={dataset_counts['validation_images']} metrics={summary['validation_metrics']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
