"""Evaluate the PIO detector on official AI Hub Korean laying-hen images."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from tqdm import tqdm

from pileguard.data.aihub_laying_hen import (
    AihubAnnotation,
    AihubArchivePair,
    dataset_audit,
    discover_archive_pairs,
    iter_inner_files,
)
from pileguard.evaluation.nestler_transfer import (
    ScoredDetection,
    aggregate_rows,
    build_integration_gate,
    detections_from_result,
    match_centers,
    match_detections,
)
from pileguard.features.pio import NormalizedBox, compute_static_features
from pileguard.features.pio_predicted import COMPARISON_FEATURES, comparison_metrics
from pileguard.models.yolo import require_ultralytics
from pileguard.runtime import resolve_device

matplotlib.use("Agg")


@dataclass(frozen=True)
class PredictionRecord:
    annotation: AihubAnnotation
    detections: tuple[ScoredDetection, ...]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def resolve_archive_dir(argument: str | None) -> Path:
    value = argument or os.environ.get("PILEGUARD_AIHUB_ARCHIVE_DIR")
    if not value:
        raise RuntimeError(
            "Set PILEGUARD_AIHUB_ARCHIVE_DIR or pass --archive-dir with the six official TARs."
        )
    return Path(value).expanduser()


def decode_png(payload: bytes, annotation: AihubAnnotation) -> np.ndarray:
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Source file is not a PNG: {annotation.image_id}")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required. Install the video extra with `pip install -e '.[video]'`."
        ) from error
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode source PNG: {annotation.image_id}")
    height, width = image.shape[:2]
    if (width, height) != (annotation.width, annotation.height):
        raise ValueError(
            f"Decoded dimensions differ from label for {annotation.image_id}: "
            f"{width}x{height} != {annotation.width}x{annotation.height}"
        )
    return image


def infer_stage(
    *,
    model: Any,
    pair: AihubArchivePair,
    annotations: dict[str, AihubAnnotation],
    image_size: int,
    batch_size: int,
    device: str,
    minimum_confidence: float,
    inference_iou: float,
    max_detections: int,
    limit: int | None,
) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    seen: set[str] = set()
    batch_images: list[np.ndarray] = []
    batch_annotations: list[AihubAnnotation] = []
    total = min(len(annotations), limit) if limit is not None else len(annotations)
    progress = tqdm(total=total, desc=f"AIHub-{pair.stage}", leave=False)

    def flush() -> None:
        if not batch_images:
            return
        results = list(
            model.predict(
                source=batch_images,
                imgsz=image_size,
                batch=batch_size,
                device=device,
                conf=minimum_confidence,
                iou=inference_iou,
                max_det=max_detections,
                save=False,
                verbose=False,
            )
        )
        if len(results) != len(batch_annotations):
            raise ValueError("Detector did not return one result per AI Hub image")
        records.extend(
            PredictionRecord(annotation, detections_from_result(result))
            for annotation, result in zip(batch_annotations, results, strict=True)
        )
        progress.update(len(results))
        batch_images.clear()
        batch_annotations.clear()

    try:
        for source_name, payload in iter_inner_files(pair.source_archive, suffix=".png"):
            if limit is not None and len(records) + len(batch_images) >= limit:
                break
            annotation = annotations.get(source_name)
            if annotation is None:
                raise ValueError(f"Source PNG has no label in {pair.stage}: {source_name}")
            if source_name in seen:
                raise ValueError(f"Duplicate source PNG in {pair.stage}: {source_name}")
            seen.add(source_name)
            batch_images.append(decode_png(payload, annotation))
            batch_annotations.append(annotation)
            if len(batch_images) >= batch_size:
                flush()
        flush()
    finally:
        progress.close()
    if limit is None and seen != set(annotations):
        missing = sorted(set(annotations) - seen)
        raise ValueError(f"Missing source PNGs in {pair.stage}: {missing[:5]}")
    if not records:
        raise ValueError(f"No AI Hub source images evaluated for {pair.stage}")
    return records


def normalized_prediction_boxes(
    detections: list[ScoredDetection], *, width: int, height: int
) -> tuple[NormalizedBox, ...]:
    return tuple(
        NormalizedBox(
            class_id=0,
            center_x=((detection.box.x1 + detection.box.x2) / 2) / width,
            center_y=((detection.box.y1 + detection.box.y2) / 2) / height,
            width=(detection.box.x2 - detection.box.x1) / width,
            height=(detection.box.y2 - detection.box.y1) / height,
        )
        for detection in detections
    )


def build_image_row(
    record: PredictionRecord,
    *,
    confidence_threshold: float,
    iou_threshold: float,
    center_distance_threshold: float,
    grid_rows: int,
    grid_columns: int,
) -> dict[str, Any]:
    annotation = record.annotation
    selected = [
        detection
        for detection in record.detections
        if detection.confidence >= confidence_threshold
    ]
    predicted_pixel = [detection.box for detection in selected]
    reference_pixel = list(annotation.pixel_boxes)
    iou_tp, iou_fp, iou_fn = match_detections(
        reference_pixel, predicted_pixel, iou_threshold=iou_threshold
    )
    center_tp, center_fp, center_fn = match_centers(
        reference_pixel,
        predicted_pixel,
        frame_width=annotation.width,
        frame_height=annotation.height,
        distance_threshold=center_distance_threshold,
    )
    predicted_features = compute_static_features(
        normalized_prediction_boxes(selected, width=annotation.width, height=annotation.height),
        grid_rows=grid_rows,
        grid_columns=grid_columns,
    )
    reference_features = compute_static_features(
        annotation.normalized_boxes,
        grid_rows=grid_rows,
        grid_columns=grid_columns,
    )
    count_error = len(predicted_pixel) - len(reference_pixel)
    row: dict[str, Any] = {
        "image_id": annotation.image_id,
        "stage": annotation.stage,
        "width": annotation.width,
        "height": annotation.height,
        "annotation_available": True,
        "confidence_threshold": confidence_threshold,
        "reference_count": len(reference_pixel),
        "predicted_count": len(predicted_pixel),
        "count_error": count_error,
        "absolute_count_error": abs(count_error),
        "iou_true_positives": iou_tp,
        "iou_false_positives": iou_fp,
        "iou_false_negatives": iou_fn,
        "center_true_positives": center_tp,
        "center_false_positives": center_fp,
        "center_false_negatives": center_fn,
        "mean_detection_confidence": (
            float(np.mean([item.confidence for item in selected])) if selected else None
        ),
        **predicted_features,
    }
    for feature in COMPARISON_FEATURES:
        reference_value = reference_features[feature]
        predicted_value = predicted_features[feature]
        row[f"reference_{feature}"] = reference_value
        if reference_value is None or predicted_value is None:
            row[f"error_{feature}"] = None
            row[f"absolute_error_{feature}"] = None
        else:
            error = float(predicted_value) - float(reference_value)
            row[f"error_{feature}"] = error
            row[f"absolute_error_{feature}"] = abs(error)
    return row


def summarize_rows(rows: list[dict[str, Any]], *, include_stages: bool) -> dict[str, Any]:
    summary = {
        "localization": aggregate_rows(rows),
        "density_features": comparison_metrics(rows),
    }
    if include_stages:
        summary["stages"] = {
            stage: summarize_rows(
                [row for row in rows if row["stage"] == stage], include_stages=False
            )
            for stage in sorted({str(row["stage"]) for row in rows})
        }
    return summary


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("No AI Hub evaluation rows to save")
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_plot(
    threshold_summaries: dict[float, dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    output_path: Path,
    *,
    detector_label: str,
) -> None:
    from matplotlib import pyplot as plt

    thresholds = sorted(threshold_summaries)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for metric, color in (("precision", "#0B6E99"), ("recall", "#E07A2D"), ("f1", "#2E7D32")):
        axes[0].plot(
            thresholds,
            [
                threshold_summaries[value]["localization"]["matching_metrics"]["center"][metric]
                or 0.0
                for value in thresholds
            ],
            marker="o",
            label=metric,
            color=color,
        )
    axes[0].set(title="Center-match metrics", xlabel="Confidence", ylabel="Score", ylim=(0, 1))
    axes[0].legend()

    axes[1].scatter(
        [row["reference_count"] for row in selected_rows],
        [row["predicted_count"] for row in selected_rows],
        s=8,
        alpha=0.25,
    )
    limit = max(
        max(int(row["reference_count"]) for row in selected_rows),
        max(int(row["predicted_count"]) for row in selected_rows),
    )
    axes[1].plot([0, limit], [0, limit], "k--", linewidth=1)
    axes[1].set(title="Object count", xlabel="Reference", ylabel="Predicted")

    stages = sorted({str(row["stage"]) for row in selected_rows})
    stage_f1 = [
        summarize_rows(
            [row for row in selected_rows if row["stage"] == stage], include_stages=False
        )["localization"]["matching_metrics"]["center"]["f1"]
        for stage in stages
    ]
    axes[2].bar(stages, stage_f1, color="#6A5ACD")
    axes[2].set(title="Center-match F1 by laying stage", xlabel="Stage", ylabel="F1", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(f"{detector_label} on official AI Hub Korean laying hens")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_evaluation(
    *,
    config_path: Path,
    archive_dir_argument: str | None = None,
    weights: Path | None = None,
    output_dir: Path | None = None,
    max_images: int | None = None,
    device_argument: str | None = None,
) -> Path:
    if max_images is not None and max_images < 1:
        raise ValueError("max_images must be positive")
    config = load_config(config_path)
    archive_dir = resolve_archive_dir(archive_dir_argument)
    pairs = discover_archive_pairs(archive_dir)
    annotations_by_stage, audit = dataset_audit(pairs)
    model_config = config["model"]
    detector_label = str(
        model_config.get("detector_label", "PIO YOLO26n overseas broiler baseline")
    )
    evaluation_config = config["evaluation"]
    feature_config = config["features"]
    checkpoint = weights or Path(str(model_config["weights"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PIO YOLO checkpoint not found: {checkpoint}")
    thresholds = sorted(float(value) for value in evaluation_config["confidence_thresholds"])
    minimum_confidence = float(evaluation_config["minimum_confidence"])
    if not thresholds or minimum_confidence > thresholds[0]:
        raise ValueError("minimum_confidence must not exceed evaluated thresholds")
    device = str(resolve_device(device_argument or str(model_config["device"])))
    YOLO = require_ultralytics()
    model = YOLO(str(checkpoint))

    started_at = time.monotonic()
    records: list[PredictionRecord] = []
    remaining = max_images
    for pair in pairs:
        if remaining == 0:
            break
        stage_limit = remaining
        stage_records = infer_stage(
            model=model,
            pair=pair,
            annotations=annotations_by_stage[pair.stage],
            image_size=int(model_config["image_size"]),
            batch_size=int(model_config["batch_size"]),
            device=device,
            minimum_confidence=minimum_confidence,
            inference_iou=float(evaluation_config["inference_iou_threshold"]),
            max_detections=int(evaluation_config["max_detections"]),
            limit=stage_limit,
        )
        records.extend(stage_records)
        if remaining is not None:
            remaining = max(remaining - len(stage_records), 0)

    rows_by_threshold = {
        threshold: [
            build_image_row(
                record,
                confidence_threshold=threshold,
                iou_threshold=float(evaluation_config["iou_threshold"]),
                center_distance_threshold=float(
                    evaluation_config["center_distance_threshold"]
                ),
                grid_rows=int(feature_config["grid_rows"]),
                grid_columns=int(feature_config["grid_columns"]),
            )
            for record in records
        ]
        for threshold in thresholds
    }
    threshold_summaries = {
        threshold: summarize_rows(rows, include_stages=True)
        for threshold, rows in rows_by_threshold.items()
    }
    selected_threshold = max(
        thresholds,
        key=lambda threshold: (
            float(
                threshold_summaries[threshold]["localization"]["matching_metrics"]["center"]["f1"]
            ),
            -float(threshold_summaries[threshold]["localization"]["count_mae"]),
        ),
    )
    selected_rows = rows_by_threshold[selected_threshold]
    selected_summary = summarize_rows(selected_rows, include_stages=True)
    integration_gate = build_integration_gate(
        selected_summary["localization"],
        minimum_center_f1=float(config["quality"]["minimum_center_f1_for_monitoring"]),
    )
    integration_gate["reason"] = str(
        model_config.get(
            "gate_pass_reason"
            if integration_gate["passed"]
            else "gate_fail_reason",
            integration_gate["reason"],
        )
    )
    claim_boundary = str(
        model_config.get(
            "claim_boundary",
            "This validates cross-domain chicken localization and static density feature "
            "transfer on official Korean laying-hen community images. It is not a "
            "piling/smothering event evaluation: images are not continuous incident clips, "
            "actionValue is not used as an incident target, and threshold selection uses the "
            "same Validation images.",
        )
    )
    summary = {
        "source_detector": detector_label,
        "evaluation_dataset": "AI Hub 575 Korean laying-hen community images",
        "evaluation_mode": "partial smoke audit" if max_images is not None else "full Validation",
        "checkpoint": checkpoint.as_posix(),
        "device": device,
        "image_size": int(model_config["image_size"]),
        "dataset_audit": audit,
        "evaluated_images": len(records),
        "selected_confidence_threshold": selected_threshold,
        "threshold_selection_split": "same AI Hub Validation images reported below",
        "threshold_results": {
            str(threshold): result for threshold, result in threshold_summaries.items()
        },
        "selected_result": selected_summary,
        "monitoring_integration_gate": integration_gate,
        "elapsed_seconds": time.monotonic() - started_at,
        "claim_boundary": claim_boundary,
    }
    destination = output_dir or Path(str(config["output"]["artifact_dir"]))
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "image_metrics.csv"
    write_rows(selected_rows, csv_path)
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_plot(
        threshold_summaries,
        selected_rows,
        destination / "transfer_audit.png",
        detector_label=detector_label,
    )
    center_f1 = selected_summary["localization"]["matching_metrics"]["center"]["f1"]
    print(
        f"saved images={len(records)} threshold={selected_threshold:.3f} "
        f"center_f1={center_f1:.4f} gate={integration_gate['decision']} output={destination}"
    )
    return csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/aihub_domestic_validation.toml")
    )
    parser.add_argument("--archive-dir")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--device")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_evaluation(
        config_path=args.config,
        archive_dir_argument=args.archive_dir,
        weights=args.weights,
        output_dir=args.output_dir,
        max_images=args.max_images,
        device_argument=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
