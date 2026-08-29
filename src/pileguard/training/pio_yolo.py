"""Train an Ultralytics YOLO26n chicken detector on official PIO splits."""

from __future__ import annotations

import argparse
import time
import tomllib
from pathlib import Path
from typing import Any

from pileguard.data.pio_detection import write_runtime_dataset
from pileguard.data_inventory import resolve_data_root
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pio_yolo26n.toml"))
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one scratch epoch on four train and four validation images.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["output"]

    seed = int(training_config["seed"])
    seed_everything(seed)
    device = resolve_device(args.device or str(training_config["device"]))
    data_root = resolve_data_root(args.data_root)
    dataset_root = data_root / str(data_config["dataset_path"])
    run_root = Path(str(output_config["run_root"]))
    run_name = "smoke" if args.smoke else "train"
    runtime_dir = run_root / run_name / "runtime"
    dataset_yaml, dataset_counts = write_runtime_dataset(
        dataset_root=dataset_root,
        output_dir=runtime_dir,
        train_split=str(data_config["train_split"]),
        validation_split=str(data_config["validation_split"]),
        class_names=list(data_config["class_names"]),
        smoke_limit=4 if args.smoke else None,
        seed=seed,
    )

    epochs = 1 if args.smoke else int(args.epochs or training_config["epochs"])
    image_size = 320 if args.smoke else int(args.image_size or training_config["image_size"])
    batch_size = 2 if args.smoke else int(args.batch_size or training_config["batch_size"])
    use_pretrained = not args.no_pretrained and not args.smoke
    model_source = (
        str(model_config["pretrained_weights"])
        if use_pretrained
        else str(model_config["architecture"])
    )

    YOLO = require_ultralytics()
    model = YOLO(model_source)
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
        project=str(run_root),
        name=run_name,
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    elapsed_seconds = time.monotonic() - started_at
    run_dir = Path(model.trainer.save_dir)
    artifact_dir = Path(str(output_config["artifact_dir"]))
    if args.smoke:
        artifact_dir = Path("outputs/smoke/pio-yolo26n")
    evidence_files = copy_artifacts(
        run_dir,
        artifact_dir,
        ("results.csv", "results.png", "labels.jpg"),
    )
    checkpoint = copy_best_checkpoint(
        run_dir / "weights" / "best.pt",
        Path(str(output_config["checkpoint_dir"])) / ("smoke" if args.smoke else ""),
    )
    summary = {
        "status": "smoke" if args.smoke else "baseline",
        "architecture": str(model_config["architecture"]),
        "pretrained": use_pretrained,
        "dataset": dataset_counts,
        "epochs": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "device": str(device),
        "elapsed_seconds": elapsed_seconds,
        "validation_metrics": normalize_detection_metrics(results),
        "evidence_files": evidence_files,
        "checkpoint": checkpoint.as_posix(),
        "claim_boundary": (
            "PIO is an overseas broiler detection dataset without Piling event labels; "
            "metrics validate chicken detection only."
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
