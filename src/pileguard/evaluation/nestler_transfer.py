"""Audit transfer of the PIO detector to official NESTLER video annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from tqdm import tqdm

from pileguard.data_inventory import resolve_data_root
from pileguard.features.nestler import (
    BoundingBox,
    JobInput,
    discover_jobs,
    parse_frame_boxes,
)
from pileguard.models.yolo import require_ultralytics
from pileguard.runtime import resolve_device

matplotlib.use("Agg")


@dataclass(frozen=True)
class ScoredDetection:
    box: BoundingBox
    confidence: float


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def detections_from_arrays(
    xyxy: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    *,
    expected_class_id: int = 0,
) -> tuple[ScoredDetection, ...]:
    coordinates = np.asarray(xyxy, dtype=np.float64)
    scores = np.asarray(confidences, dtype=np.float64).reshape(-1)
    classes = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    if coordinates.size == 0:
        return ()
    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError(f"Expected Nx4 pixel boxes, got {coordinates.shape}")
    if not len(coordinates) == len(scores) == len(classes):
        raise ValueError("Detection box, confidence, and class counts do not match")
    if not np.isfinite(coordinates).all() or not np.isfinite(scores).all():
        raise ValueError("Detections contain non-finite values")
    if np.any(classes != expected_class_id):
        raise ValueError("NESTLER transfer audit expects the single chicken class")

    detections: list[ScoredDetection] = []
    for index, (x1, y1, x2, y2) in enumerate(coordinates):
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            ScoredDetection(
                box=BoundingBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    track_id=-1,
                ),
                confidence=float(scores[index]),
            )
        )
    return tuple(detections)


def detections_from_result(result: Any) -> tuple[ScoredDetection, ...]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return ()
    return detections_from_arrays(
        boxes.xyxy.cpu().numpy(),
        boxes.conf.cpu().numpy(),
        boxes.cls.cpu().numpy(),
    )


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    intersection_width = max(min(first.x2, second.x2) - max(first.x1, second.x1), 0.0)
    intersection_height = max(min(first.y2, second.y2) - max(first.y1, second.y1), 0.0)
    intersection = intersection_width * intersection_height
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(
    reference_boxes: list[BoundingBox],
    predicted_boxes: list[BoundingBox],
    *,
    iou_threshold: float,
) -> tuple[int, int, int]:
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    candidates = sorted(
        (
            (intersection_over_union(reference, predicted), reference_index, predicted_index)
            for reference_index, reference in enumerate(reference_boxes)
            for predicted_index, predicted in enumerate(predicted_boxes)
        ),
        reverse=True,
    )
    matched_reference: set[int] = set()
    matched_predictions: set[int] = set()
    for iou, reference_index, predicted_index in candidates:
        if iou < iou_threshold:
            break
        if reference_index in matched_reference or predicted_index in matched_predictions:
            continue
        matched_reference.add(reference_index)
        matched_predictions.add(predicted_index)
    true_positives = len(matched_reference)
    return (
        true_positives,
        len(predicted_boxes) - true_positives,
        len(reference_boxes) - true_positives,
    )


def normalized_center_distance(
    first: BoundingBox,
    second: BoundingBox,
    *,
    frame_width: int,
    frame_height: int,
) -> float:
    first_x, first_y = first.center
    second_x, second_y = second.center
    return math.hypot(
        (first_x - second_x) / frame_width,
        (first_y - second_y) / frame_height,
    )


def match_centers(
    reference_boxes: list[BoundingBox],
    predicted_boxes: list[BoundingBox],
    *,
    frame_width: int,
    frame_height: int,
    distance_threshold: float,
) -> tuple[int, int, int]:
    if not 0 < distance_threshold <= 1:
        raise ValueError("distance_threshold must be in (0, 1]")
    candidates = sorted(
        (
            (
                normalized_center_distance(
                    reference,
                    predicted,
                    frame_width=frame_width,
                    frame_height=frame_height,
                ),
                reference_index,
                predicted_index,
            )
            for reference_index, reference in enumerate(reference_boxes)
            for predicted_index, predicted in enumerate(predicted_boxes)
        )
    )
    matched_reference: set[int] = set()
    matched_predictions: set[int] = set()
    for distance, reference_index, predicted_index in candidates:
        if distance > distance_threshold:
            break
        if reference_index in matched_reference or predicted_index in matched_predictions:
            continue
        matched_reference.add(reference_index)
        matched_predictions.add(predicted_index)
    true_positives = len(matched_reference)
    return (
        true_positives,
        len(predicted_boxes) - true_positives,
        len(reference_boxes) - true_positives,
    )


def build_frame_row(
    *,
    job: JobInput,
    frame_index: int,
    timestamp_seconds: float,
    annotation_available: bool,
    reference_boxes: list[BoundingBox],
    detections: tuple[ScoredDetection, ...],
    confidence_threshold: float,
    iou_threshold: float,
    center_distance_threshold: float,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    selected = [
        detection for detection in detections if detection.confidence >= confidence_threshold
    ]
    predicted_boxes = [detection.box for detection in selected]
    if annotation_available:
        iou_true_positives, iou_false_positives, iou_false_negatives = match_detections(
            reference_boxes,
            predicted_boxes,
            iou_threshold=iou_threshold,
        )
        center_true_positives, center_false_positives, center_false_negatives = match_centers(
            reference_boxes,
            predicted_boxes,
            frame_width=frame_width,
            frame_height=frame_height,
            distance_threshold=center_distance_threshold,
        )
        count_error: int | None = len(predicted_boxes) - len(reference_boxes)
    else:
        iou_true_positives = iou_false_positives = iou_false_negatives = None
        center_true_positives = center_false_positives = center_false_negatives = None
        count_error = None
    return {
        "job_id": job.job_id,
        "site": job.site,
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "annotation_available": annotation_available,
        "confidence_threshold": confidence_threshold,
        "reference_count": len(reference_boxes) if annotation_available else None,
        "predicted_count": len(predicted_boxes),
        "count_error": count_error,
        "absolute_count_error": abs(count_error) if count_error is not None else None,
        "iou_true_positives": iou_true_positives,
        "iou_false_positives": iou_false_positives,
        "iou_false_negatives": iou_false_negatives,
        "center_true_positives": center_true_positives,
        "center_false_positives": center_false_positives,
        "center_false_negatives": center_false_negatives,
        "mean_detection_confidence": (
            float(np.mean([detection.confidence for detection in selected]))
            if selected
            else None
        ),
    }


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def pearson_correlation(reference: list[float], predicted: list[float]) -> float | None:
    if len(reference) < 2:
        return None
    if math.isclose(float(np.std(reference)), 0.0) or math.isclose(
        float(np.std(predicted)), 0.0
    ):
        return None
    return float(np.corrcoef(reference, predicted)[0, 1])


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    annotated = [row for row in rows if row["annotation_available"]]
    matching_metrics: dict[str, Any] = {}
    for matching in ("iou", "center"):
        true_positives = sum(int(row[f"{matching}_true_positives"]) for row in annotated)
        false_positives = sum(int(row[f"{matching}_false_positives"]) for row in annotated)
        false_negatives = sum(int(row[f"{matching}_false_negatives"]) for row in annotated)
        precision = safe_ratio(true_positives, true_positives + false_positives)
        recall = safe_ratio(true_positives, true_positives + false_negatives)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else 0.0
        )
        matching_metrics[matching] = {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    reference = [float(row["reference_count"]) for row in annotated]
    predicted = [float(row["predicted_count"]) for row in annotated]
    errors = np.asarray(predicted) - np.asarray(reference)
    return {
        "frame_count": len(rows),
        "annotated_frame_count": len(annotated),
        "annotation_coverage": len(annotated) / len(rows) if rows else 0.0,
        "frames_without_predictions": sum(int(row["predicted_count"] == 0) for row in rows),
        "reference_boxes": int(sum(reference)),
        "predicted_boxes_on_annotated_frames": int(sum(predicted)),
        "matching_metrics": matching_metrics,
        "count_mae": float(np.mean(np.abs(errors))) if len(errors) else None,
        "count_rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        "count_bias": float(np.mean(errors)) if len(errors) else None,
        "count_pearson_correlation": pearson_correlation(reference, predicted),
    }


def summarize_threshold(
    rows: list[dict[str, Any]], *, include_groups: bool
) -> dict[str, Any]:
    summary = aggregate_rows(rows)
    if include_groups:
        summary["jobs"] = {
            job_id: aggregate_rows([row for row in rows if row["job_id"] == job_id])
            for job_id in sorted({str(row["job_id"]) for row in rows})
        }
        summary["sites"] = {
            site: aggregate_rows([row for row in rows if row["site"] == site])
            for site in sorted({str(row["site"]) for row in rows})
        }
    return summary


def build_integration_gate(
    selected_summary: dict[str, Any], *, minimum_center_f1: float
) -> dict[str, Any]:
    if not 0 <= minimum_center_f1 <= 1:
        raise ValueError("minimum_center_f1 must be in [0, 1]")
    observed = float(selected_summary["matching_metrics"]["center"]["f1"])
    passed = observed >= minimum_center_f1
    return {
        "minimum_center_f1": minimum_center_f1,
        "observed_center_f1": observed,
        "passed": passed,
        "decision": "allow predicted monitoring" if passed else "block predicted monitoring",
        "reason": (
            "Cross-dataset localization quality meets the configured gate."
            if passed
            else "Detector transfer is too weak; fine-tune before generating risk alerts."
        ),
    }


def evaluate_job(
    *,
    model: Any,
    job: JobInput,
    confidence_thresholds: list[float],
    minimum_confidence: float,
    image_size: int,
    device: str,
    iou_threshold: float,
    center_distance_threshold: float,
    max_detections: int,
    max_frames: int | None,
) -> dict[float, list[dict[str, Any]]]:
    annotation = json.loads(job.annotation_path.read_text(encoding="utf-8"))
    frame_width = int(annotation["frame_width"])
    frame_height = int(annotation["frame_height"])
    fps = float(annotation["fps"])
    frames = annotation["frames"]
    if max_frames is not None:
        frames = frames[:max_frames]
    results: Iterable[Any] = model.predict(
        source=str(job.video_path),
        stream=True,
        imgsz=image_size,
        device=device,
        conf=minimum_confidence,
        iou=0.7,
        max_det=max_detections,
        save=False,
        verbose=False,
    )
    if max_frames is not None:
        results = islice(results, max_frames)

    rows_by_threshold = {threshold: [] for threshold in confidence_thresholds}
    paired = zip(frames, results, strict=True)
    for frame_annotation, result in tqdm(
        paired,
        total=len(frames),
        desc=job.job_id,
        leave=False,
    ):
        annotation_available, reference_boxes = parse_frame_boxes(
            frame_annotation,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        detections = detections_from_result(result)
        frame_index = int(frame_annotation["frame_index"])
        for threshold in confidence_thresholds:
            rows_by_threshold[threshold].append(
                build_frame_row(
                    job=job,
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / fps,
                    annotation_available=annotation_available,
                    reference_boxes=reference_boxes,
                    detections=detections,
                    confidence_threshold=threshold,
                    iou_threshold=iou_threshold,
                    center_distance_threshold=center_distance_threshold,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )
    return rows_by_threshold


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("No NESTLER transfer rows to save")
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_plot(
    threshold_summaries: dict[float, dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    from matplotlib import pyplot as plt

    thresholds = sorted(threshold_summaries)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for metric, color in (
        ("precision", "#0B6E99"),
        ("recall", "#E07A2D"),
        ("f1", "#2E7D32"),
    ):
        axes[0].plot(
            thresholds,
            [
                threshold_summaries[threshold]["matching_metrics"]["center"][metric] or 0.0
                for threshold in thresholds
            ],
            marker="o",
            label=metric,
            color=color,
        )
    axes[0].plot(
        thresholds,
        [
            threshold_summaries[threshold]["matching_metrics"]["iou"]["f1"]
            for threshold in thresholds
        ],
        marker="x",
        linestyle="--",
        color="#616161",
        label="IoU F1",
    )
    axes[0].set(title="Center-match metrics", xlabel="Confidence", ylabel="Score")
    axes[0].legend()
    annotated = [row for row in selected_rows if row["annotation_available"]]
    axes[1].scatter(
        [row["reference_count"] for row in annotated],
        [row["predicted_count"] for row in annotated],
        s=8,
        alpha=0.3,
    )
    count_limit = max(
        max(int(row["reference_count"]) for row in annotated),
        max(int(row["predicted_count"]) for row in annotated),
    )
    axes[1].plot([0, count_limit], [0, count_limit], "k--", linewidth=1)
    axes[1].set(title="Frame counts", xlabel="Reference", ylabel="Predicted")
    selected_summary = summarize_threshold(selected_rows, include_groups=True)
    job_ids = sorted(selected_summary["jobs"])
    axes[2].bar(
        [job_id.removeprefix("job_") for job_id in job_ids],
        [
            selected_summary["jobs"][job_id]["matching_metrics"]["center"]["f1"]
            for job_id in job_ids
        ],
        color="#6A5ACD",
    )
    axes[2].set(title="Center-match F1 by clip", xlabel="Job", ylabel="F1", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("PIO detector transfer to NESTLER tracking annotations")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_evaluation(
    *,
    config_path: Path,
    data_root_argument: str | None = None,
    weights: Path | None = None,
    output_dir: Path | None = None,
    job_ids: list[str] | None = None,
    max_frames: int | None = None,
) -> Path:
    config = load_config(config_path)
    nestler_config = load_config(Path(str(config["input"]["nestler_config"])))
    data_root = resolve_data_root(data_root_argument)
    dataset_root = data_root / str(nestler_config["data"]["dataset_path"])
    jobs = discover_jobs(dataset_root, nestler_config["data"])
    if job_ids:
        requested = set(job_ids)
        jobs = [job for job in jobs if job.job_id in requested]
        missing = requested - {job.job_id for job in jobs}
        if missing:
            raise ValueError(f"Unknown job IDs: {sorted(missing)}")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")

    model_config = config["model"]
    evaluation_config = config["evaluation"]
    checkpoint = weights or Path(str(model_config["weights"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PIO YOLO checkpoint not found: {checkpoint}")
    thresholds = sorted(float(value) for value in evaluation_config["confidence_thresholds"])
    minimum_confidence = float(evaluation_config["minimum_confidence"])
    if not thresholds or minimum_confidence > thresholds[0]:
        raise ValueError("minimum_confidence must not exceed evaluated thresholds")
    device = str(resolve_device(str(model_config["device"])))
    YOLO = require_ultralytics()
    model = YOLO(str(checkpoint))
    started_at = time.monotonic()
    rows_by_threshold = {threshold: [] for threshold in thresholds}
    for job in jobs:
        job_rows = evaluate_job(
            model=model,
            job=job,
            confidence_thresholds=thresholds,
            minimum_confidence=minimum_confidence,
            image_size=int(model_config["image_size"]),
            device=device,
            iou_threshold=float(evaluation_config["iou_threshold"]),
            center_distance_threshold=float(
                evaluation_config["center_distance_threshold"]
            ),
            max_detections=int(evaluation_config["max_detections"]),
            max_frames=max_frames,
        )
        for threshold in thresholds:
            rows_by_threshold[threshold].extend(job_rows[threshold])

    threshold_summaries = {
        threshold: summarize_threshold(rows, include_groups=False)
        for threshold, rows in rows_by_threshold.items()
    }
    selected_threshold = max(
        thresholds,
        key=lambda threshold: (
            float(
                threshold_summaries[threshold]["matching_metrics"]["center"]["f1"]
            ),
            -float(threshold_summaries[threshold]["count_mae"]),
        ),
    )
    selected_rows = rows_by_threshold[selected_threshold]
    selected_summary = summarize_threshold(selected_rows, include_groups=True)
    integration_gate = build_integration_gate(
        selected_summary,
        minimum_center_f1=float(config["quality"]["minimum_center_f1_for_monitoring"]),
    )
    summary = {
        "source_detector": "PIO YOLO26n overseas broiler baseline",
        "checkpoint": checkpoint.as_posix(),
        "image_size": int(model_config["image_size"]),
        "iou_threshold": float(evaluation_config["iou_threshold"]),
        "center_distance_threshold": float(
            evaluation_config["center_distance_threshold"]
        ),
        "selected_confidence_threshold": selected_threshold,
        "threshold_selection_split": "same NESTLER clips reported below",
        "threshold_results": {
            str(threshold): metrics for threshold, metrics in threshold_summaries.items()
        },
        "selected_result": selected_summary,
        "monitoring_integration_gate": integration_gate,
        "elapsed_seconds": time.monotonic() - started_at,
        "claim_boundary": (
            "This is a cross-dataset transfer audit against NESTLER tracking boxes, not a "
            "Piling-event evaluation. Threshold selection and reporting use the same clips."
        ),
    }
    destination = output_dir or Path(str(config["output"]["artifact_dir"]))
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "frame_metrics.csv"
    write_csv(selected_rows, csv_path)
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_plot(threshold_summaries, selected_rows, destination / "transfer_audit.png")
    print(
        f"saved frames={len(selected_rows)} threshold={selected_threshold:.3f} "
        f"center_f1={summary['selected_result']['matching_metrics']['center']['f1']:.4f} "
        f"gate={integration_gate['decision']} output={destination}"
    )
    return csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/nestler_detector_transfer.toml")
    )
    parser.add_argument("--data-root")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jobs", nargs="*")
    parser.add_argument("--max-frames", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_evaluation(
        config_path=args.config,
        data_root_argument=args.data_root,
        weights=args.weights,
        output_dir=args.output_dir,
        job_ids=args.jobs,
        max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
