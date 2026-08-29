"""Extract PIO density features from model predictions and audit them against labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from pileguard.data_inventory import resolve_data_root
from pileguard.features.pio import (
    NormalizedBox,
    PioSample,
    compute_static_features,
    discover_samples,
    parse_image_metadata,
    parse_yolo_label,
)
from pileguard.models.yolo import require_ultralytics
from pileguard.runtime import resolve_device

matplotlib.use("Agg")

COMPARISON_FEATURES = (
    "object_count",
    "bbox_area_ratio",
    "center_spread",
    "mean_nearest_neighbor_distance",
    "max_grid_fraction",
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def normalized_boxes_from_arrays(
    xywhn: np.ndarray,
    class_ids: np.ndarray,
    *,
    expected_class_id: int,
) -> tuple[NormalizedBox, ...]:
    """Convert Ultralytics normalized arrays into the shared feature representation."""

    coordinates = np.asarray(xywhn, dtype=np.float64)
    classes = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    if coordinates.size == 0:
        return ()
    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError(f"Expected Nx4 normalized boxes, got {coordinates.shape}")
    if len(coordinates) != len(classes):
        raise ValueError("Box and class counts do not match")
    if not np.isfinite(coordinates).all():
        raise ValueError("Predicted boxes contain non-finite coordinates")
    if np.any(classes != expected_class_id):
        unexpected = sorted(set(int(value) for value in classes if value != expected_class_id))
        raise ValueError(f"Unexpected predicted classes: {unexpected}")

    boxes: list[NormalizedBox] = []
    for class_id, (center_x, center_y, width, height) in zip(
        classes, coordinates, strict=True
    ):
        if width <= 0 or height <= 0:
            continue
        boxes.append(
            NormalizedBox(
                class_id=int(class_id),
                center_x=float(np.clip(center_x, 0.0, 1.0)),
                center_y=float(np.clip(center_y, 0.0, 1.0)),
                width=float(np.clip(width, 0.0, 1.0)),
                height=float(np.clip(height, 0.0, 1.0)),
            )
        )
    return tuple(boxes)


def boxes_from_result(result: Any, *, expected_class_id: int) -> tuple[NormalizedBox, ...]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return ()
    return normalized_boxes_from_arrays(
        boxes.xywhn.cpu().numpy(),
        boxes.cls.cpu().numpy(),
        expected_class_id=expected_class_id,
    )


def select_confidence_threshold(
    sample: PioSample,
    *,
    default_threshold: float,
    confidence_by_week: dict[int, float],
) -> float:
    """Select an age-aware threshold from known flock metadata when available."""

    _, week = parse_image_metadata(sample.image_id)
    if week is None:
        return default_threshold
    return confidence_by_week.get(week, default_threshold)


def build_comparison_row(
    sample: PioSample,
    predicted_boxes: tuple[NormalizedBox, ...],
    *,
    confidences: Iterable[float],
    expected_class_id: int,
    grid_rows: int,
    grid_columns: int,
) -> dict[str, Any]:
    predicted = compute_static_features(
        predicted_boxes,
        grid_rows=grid_rows,
        grid_columns=grid_columns,
    )
    reference_boxes = parse_yolo_label(
        sample.label_path, expected_class_id=expected_class_id
    ).boxes
    reference = compute_static_features(
        reference_boxes,
        grid_rows=grid_rows,
        grid_columns=grid_columns,
    )
    environment, week = parse_image_metadata(sample.image_id)
    confidence_values = [float(value) for value in confidences]
    row: dict[str, Any] = {
        "image_id": sample.image_id,
        "split": sample.split,
        "environment": environment,
        "week": week,
        "mean_detection_confidence": (
            float(np.mean(confidence_values)) if confidence_values else None
        ),
        **predicted,
    }
    for feature in COMPARISON_FEATURES:
        reference_value = reference[feature]
        predicted_value = predicted[feature]
        row[f"reference_{feature}"] = reference_value
        if reference_value is None or predicted_value is None:
            row[f"error_{feature}"] = None
            row[f"absolute_error_{feature}"] = None
        else:
            error = float(predicted_value) - float(reference_value)
            row[f"error_{feature}"] = error
            row[f"absolute_error_{feature}"] = abs(error)
    return row


def correlation(reference: list[float], predicted: list[float]) -> float | None:
    if len(reference) < 2 or math.isclose(float(np.std(reference)), 0.0):
        return None
    if math.isclose(float(np.std(predicted)), 0.0):
        return None
    return float(np.corrcoef(reference, predicted)[0, 1])


def comparison_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feature_metrics: dict[str, Any] = {}
    for feature in COMPARISON_FEATURES:
        pairs = [
            (float(row[f"reference_{feature}"]), float(row[feature]))
            for row in rows
            if row[f"reference_{feature}"] is not None and row[feature] is not None
        ]
        reference = [pair[0] for pair in pairs]
        predicted = [pair[1] for pair in pairs]
        errors = np.asarray(predicted, dtype=np.float64) - np.asarray(
            reference, dtype=np.float64
        )
        reference_mean = float(np.mean(reference)) if reference else None
        feature_metrics[feature] = {
            "valid_images": len(pairs),
            "reference_mean": reference_mean,
            "predicted_mean": float(np.mean(predicted)) if predicted else None,
            "mae": float(np.mean(np.abs(errors))) if len(errors) else None,
            "mae_over_reference_mean": (
                float(np.mean(np.abs(errors))) / reference_mean
                if len(errors) and reference_mean and not math.isclose(reference_mean, 0.0)
                else None
            ),
            "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            "bias": float(np.mean(errors)) if len(errors) else None,
            "pearson_correlation": correlation(reference, predicted),
        }
    return {
        "image_count": len(rows),
        "predicted_boxes": sum(int(row["object_count"]) for row in rows),
        "reference_boxes": sum(int(row["reference_object_count"]) for row in rows),
        "images_without_predictions": sum(int(row["object_count"] == 0) for row in rows),
        "feature_metrics": feature_metrics,
    }


def summarize_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    environment_names = sorted({str(row["environment"]) for row in rows})
    weeks = sorted({int(row["week"]) for row in rows if row["week"] is not None})
    return {
        **comparison_metrics(rows),
        "groups": {
            "environment": {
                environment: comparison_metrics(
                    [row for row in rows if row["environment"] == environment]
                )
                for environment in environment_names
            },
            "week": {
                str(week): comparison_metrics([row for row in rows if row["week"] == week])
                for week in weeks
            },
        },
        "claim_boundary": (
            "Predictions use an overseas broiler detector and are compared with PIO detection "
            "labels only; this does not validate Piling risk or Korean laying-hen generalization."
        ),
    }


def write_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No predicted PIO feature rows to save")
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    plot_features = (
        ("object_count", "Object count"),
        ("bbox_area_ratio", "Summed bbox area"),
    )
    for axis, (feature, title) in zip(axes[:2], plot_features, strict=True):
        reference = np.asarray([row[f"reference_{feature}"] for row in rows], dtype=float)
        predicted = np.asarray([row[feature] for row in rows], dtype=float)
        limit = float(max(reference.max(initial=0), predicted.max(initial=0)))
        axis.scatter(reference, predicted, s=9, alpha=0.35)
        axis.plot([0, limit], [0, limit], linestyle="--", color="black", linewidth=1)
        axis.set(title=title, xlabel="Reference", ylabel="Predicted")
        axis.grid(alpha=0.2)
    count_errors = [float(row["error_object_count"]) for row in rows]
    axes[2].hist(count_errors, bins=35, color="#0B6E99", alpha=0.8)
    axes[2].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[2].set(title="Count error", xlabel="Predicted - reference", ylabel="Images")
    axes[2].grid(alpha=0.2)
    figure.suptitle("PIO predicted density features vs. official detection labels")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_extraction(
    *,
    config_path: Path,
    data_root_argument: str | None = None,
    weights: Path | None = None,
    output_dir: Path | None = None,
    max_images: int | None = None,
    confidence_threshold: float | None = None,
    image_size: int | None = None,
    device_argument: str | None = None,
) -> Path:
    config = load_config(config_path)
    data_root = resolve_data_root(data_root_argument)
    dataset_root = data_root / str(config["data"]["dataset_path"])
    samples = discover_samples(dataset_root, list(config["data"]["splits"]))
    if max_images is not None:
        if max_images < 1:
            raise ValueError("max_images must be positive")
        samples = samples[:max_images]

    model_config = config["model"]
    inference_config = config["inference"]
    feature_config = config["features"]
    checkpoint = weights or Path(str(model_config["weights"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PIO YOLO checkpoint not found: {checkpoint}")
    device = resolve_device(device_argument or str(model_config["device"]))
    default_confidence = float(
        confidence_threshold
        if confidence_threshold is not None
        else inference_config["confidence_threshold"]
    )
    if not 0 < default_confidence <= 1:
        raise ValueError("confidence_threshold must be in (0, 1]")
    configured_by_week = {
        int(week): float(value)
        for week, value in dict(inference_config.get("confidence_by_week", {})).items()
    }
    confidence_by_week = {} if confidence_threshold is not None else configured_by_week
    if any(not 0 < value <= 1 for value in confidence_by_week.values()):
        raise ValueError("confidence_by_week values must be in (0, 1]")
    selected_image_size = int(image_size or model_config["image_size"])
    if selected_image_size < 32 or selected_image_size % 32:
        raise ValueError("image_size must be a positive multiple of 32")

    YOLO = require_ultralytics()
    model = YOLO(str(checkpoint))
    started_at = time.monotonic()
    sample_order = {
        (sample.split, sample.image_id): index for index, sample in enumerate(samples)
    }
    threshold_groups: dict[float, list[PioSample]] = {}
    for sample in samples:
        threshold = select_confidence_threshold(
            sample,
            default_threshold=default_confidence,
            confidence_by_week=confidence_by_week,
        )
        threshold_groups.setdefault(threshold, []).append(sample)

    rows: list[dict[str, Any]] = []
    for threshold, grouped_samples in sorted(threshold_groups.items()):
        results = model.predict(
            source=[str(sample.image_path) for sample in grouped_samples],
            stream=True,
            imgsz=selected_image_size,
            batch=int(model_config["batch_size"]),
            device=str(device),
            conf=threshold,
            iou=float(inference_config["iou_threshold"]),
            max_det=int(inference_config["max_detections"]),
            save=False,
            verbose=False,
        )
        for sample, result in zip(grouped_samples, results, strict=True):
            predicted_boxes = boxes_from_result(
                result,
                expected_class_id=int(feature_config["expected_class_id"]),
            )
            result_boxes = getattr(result, "boxes", None)
            confidences = (
                result_boxes.conf.cpu().numpy().tolist()
                if result_boxes is not None and len(result_boxes)
                else []
            )
            row = build_comparison_row(
                sample,
                predicted_boxes,
                confidences=confidences,
                expected_class_id=int(feature_config["expected_class_id"]),
                grid_rows=int(feature_config["grid_rows"]),
                grid_columns=int(feature_config["grid_columns"]),
            )
            row["confidence_threshold"] = threshold
            rows.append(row)
    rows.sort(key=lambda row: sample_order[(str(row["split"]), str(row["image_id"]))])

    summary = summarize_comparison(rows)
    summary.update(
        {
            "checkpoint": checkpoint.as_posix(),
            "split": list(config["data"]["splits"]),
            "confidence_threshold": (
                default_confidence if not confidence_by_week else None
            ),
            "confidence_by_week": {
                str(week): value for week, value in sorted(confidence_by_week.items())
            },
            "iou_threshold": float(inference_config["iou_threshold"]),
            "image_size": selected_image_size,
            "device": str(device),
            "elapsed_seconds": time.monotonic() - started_at,
            "threshold_selection": dict(config.get("threshold_selection", {})),
        }
    )
    destination = output_dir or Path(str(config["output"]["artifact_dir"]))
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "predicted_features.csv"
    write_rows(rows, csv_path)
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_comparison_plot(rows, destination / "comparison.png")
    print(
        f"saved images={summary['image_count']} predicted_boxes={summary['predicted_boxes']} "
        f"reference_boxes={summary['reference_boxes']} output={destination}"
    )
    return csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pio_predicted_features.toml")
    )
    parser.add_argument("--data-root")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--device")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_extraction(
        config_path=args.config,
        data_root_argument=args.data_root,
        weights=args.weights,
        output_dir=args.output_dir,
        max_images=args.max_images,
        confidence_threshold=args.confidence,
        image_size=args.image_size,
        device_argument=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
