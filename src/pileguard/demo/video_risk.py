"""Run the fine-tuned laying-hen detector and relative risk demo on video files."""

from __future__ import annotations

import argparse
import csv
import json
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from pileguard.demo.integrated import (
    build_camera_baselines,
    save_risk_timelines,
    score_feature_rows,
    serialize_baselines,
    summarize_results,
)
from pileguard.evaluation.nestler_transfer import detections_from_result
from pileguard.features.nestler import (
    JOB_TO_SITE,
    compute_flow_features,
    compute_spatial_features,
    feature_distributions,
)
from pileguard.models.yolo import require_ultralytics
from pileguard.runtime import resolve_device


@dataclass(frozen=True)
class VideoInput:
    video_id: str
    site: str
    path: Path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def resolve_video_inputs(paths: list[Path]) -> list[VideoInput]:
    if not paths:
        raise ValueError("Pass at least one video with --videos")
    videos: list[VideoInput] = []
    seen_ids: set[str] = set()
    for path in paths:
        resolved = path.expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"Video not found: {resolved}")
        parent_id = resolved.parent.name
        video_id = parent_id if parent_id.startswith("job_") else resolved.stem
        if video_id in seen_ids:
            raise ValueError(f"Duplicate video ID: {video_id}")
        seen_ids.add(video_id)
        videos.append(
            VideoInput(
                video_id=video_id,
                site=JOB_TO_SITE.get(video_id, "Unknown"),
                path=resolved,
            )
        )
    return videos


def missing_tracking_features() -> dict[str, int | None]:
    return {
        "track_match_count": 0,
        "tracked_speed_per_second": None,
        "tracked_direction_x": None,
        "tracked_direction_y": None,
        "tracked_direction_degrees": None,
        "tracked_coherence": None,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def extract_video_features(
    video: VideoInput,
    *,
    model: Any,
    model_config: dict[str, Any],
    video_config: dict[str, Any],
    feature_config: dict[str, Any],
    source_config: dict[str, Any],
    device: str,
    max_frames_override: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stride = int(video_config["frame_stride"])
    if stride < 1:
        raise ValueError("video.frame_stride must be positive")
    configured_max = int(video_config["max_frames"])
    max_frames = max_frames_override if max_frames_override is not None else configured_max
    if max_frames < 1:
        raise ValueError("max_frames must be positive")

    capture = cv2.VideoCapture(str(video.path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video.path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Invalid video metadata: {video.path}")

    sample_fps = fps / stride
    previous_gray: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    source_frame_index = -1
    started_at = time.monotonic()
    progress = tqdm(total=max_frames, desc=video.video_id, leave=False)
    try:
        while len(rows) < max_frames:
            success, frame = capture.read()
            if not success:
                break
            source_frame_index += 1
            if source_frame_index % stride:
                continue
            results = list(
                model.predict(
                    source=frame,
                    imgsz=int(model_config["image_size"]),
                    device=device,
                    conf=float(model_config["confidence_threshold"]),
                    iou=float(model_config["inference_iou_threshold"]),
                    max_det=int(model_config["max_detections"]),
                    save=False,
                    verbose=False,
                )
            )
            if len(results) != 1:
                raise ValueError("Detector did not return exactly one result for one video frame")
            detections = detections_from_result(results[0])
            boxes = [detection.box for detection in detections]
            resized = cv2.resize(
                frame,
                (int(feature_config["resize_width"]), int(feature_config["resize_height"])),
                interpolation=cv2.INTER_AREA,
            )
            current_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            row: dict[str, Any] = {
                "job_id": video.video_id,
                "site": video.site,
                "frame_index": len(rows),
                "source_frame_index": source_frame_index,
                "timestamp_seconds": source_frame_index / fps,
                "detection_count": len(detections),
                "mean_detection_confidence": (
                    float(np.mean([detection.confidence for detection in detections]))
                    if detections
                    else None
                ),
            }
            row.update(
                compute_spatial_features(
                    boxes,
                    frame_width=width,
                    frame_height=height,
                    grid_rows=int(feature_config["grid_rows"]),
                    grid_columns=int(feature_config["grid_columns"]),
                )
            )
            row.update(missing_tracking_features())
            row.update(
                compute_flow_features(
                    previous_gray,
                    current_gray,
                    boxes,
                    original_width=width,
                    original_height=height,
                    fps=sample_fps,
                    flow_config=feature_config,
                )
            )
            rows.append(row)
            previous_gray = current_gray
            progress.update(1)
    finally:
        progress.close()
        capture.release()
    if not rows:
        raise ValueError(f"No frames read from video: {video.path}")
    elapsed = time.monotonic() - started_at
    summary = {
        "video_id": video.video_id,
        "site": video.site,
        "source_filename": video.path.name,
        "source_dataset": str(source_config["dataset"]),
        "source_doi": str(source_config["doi"]),
        "source_license": str(source_config["license"]),
        "fps": fps,
        "sample_fps": sample_fps,
        "frame_stride": stride,
        "frame_width": width,
        "frame_height": height,
        "raw_frame_count": raw_frame_count,
        "frames_processed": len(rows),
        "frames_without_detections": sum(int(row["detection_count"]) == 0 for row in rows),
        "elapsed_seconds": elapsed,
        "processed_frames_per_second": len(rows) / elapsed if elapsed else None,
    }
    return rows, summary


def run_pipeline(
    *,
    config_path: Path,
    video_paths: list[Path],
    output_dir: Path | None = None,
    weights: Path | None = None,
    device_argument: str | None = None,
    max_frames: int | None = None,
) -> Path:
    config = load_config(config_path)
    model_config = config["model"]
    checkpoint = weights or Path(str(model_config["weights"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Laying-hen detector checkpoint not found: {checkpoint}")
    device = str(resolve_device(device_argument or str(model_config["device"])))
    videos = resolve_video_inputs(video_paths)
    YOLO = require_ultralytics()
    model = YOLO(str(checkpoint))

    all_rows: list[dict[str, Any]] = []
    video_summaries: list[dict[str, Any]] = []
    for video in videos:
        rows, video_summary = extract_video_features(
            video,
            model=model,
            model_config=model_config,
            video_config=config["video"],
            feature_config=config["features"],
            source_config=config["source"],
            device=device,
            max_frames_override=max_frames,
        )
        all_rows.extend(rows)
        video_summaries.append(video_summary)

    destination = output_dir or Path(str(config["output"]["artifact_dir"]))
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(all_rows, destination / "frame_features.csv")
    feature_summary = {
        "detector": str(model_config["detector_label"]),
        "checkpoint": checkpoint.as_posix(),
        "device": device,
        "confidence_threshold": float(model_config["confidence_threshold"]),
        "videos": video_summaries,
        "feature_distributions": feature_distributions(all_rows),
        "tracking_features_available": False,
        "source": config["source"],
    }
    (destination / "feature_summary.json").write_text(
        json.dumps(feature_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    calibration = config["calibration"]
    baselines = build_camera_baselines(
        all_rows,
        calibration_frames=int(calibration["frames"]),
        minimum_samples=int(calibration["minimum_samples"]),
        minimum_scale_ratio=float(calibration["minimum_scale_ratio"]),
        optional_features=set(config["quality"]["optional_baseline_features"]),
    )
    risk_rows = score_feature_rows(all_rows, baselines, config)
    write_csv(risk_rows, destination / "risk_timeseries.csv")
    (destination / "camera_baselines.json").write_text(
        json.dumps(serialize_baselines(baselines), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = summarize_results(risk_rows, baselines)
    summary.update(
        {
            "disclaimer": (
                "End-to-end detector-to-risk functional test only. Input videos have no official "
                "piling/smothering incident labels, so alert states are not validated incidents, "
                "false alarms, lead times, or calibrated probabilities."
            ),
            "source_detector": str(model_config["detector_label"]),
            "detector_confidence_threshold": float(
                model_config["confidence_threshold"]
            ),
            "detector_checkpoint": checkpoint.as_posix(),
            "source_videos": [
                video_summary["source_filename"] for video_summary in video_summaries
            ],
            "source": config["source"],
            "tracking_features_available": False,
            "count_calibration_required": True,
            "config": {
                "video": config["video"],
                "calibration": config["calibration"],
                "quality": config["quality"],
                "risk": config["risk"],
            },
        }
    )
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_risk_timelines(
        risk_rows,
        config,
        destination / "risk_timelines.png",
        figure_title="PileGuard fine-tuned detector — camera-relative video risk timelines",
    )
    monitoring_frames = sum(job["monitoring_frame_count"] for job in summary["jobs"].values())
    available_frames = sum(
        job["available_monitoring_frames"] for job in summary["jobs"].values()
    )
    coverage = available_frames / monitoring_frames if monitoring_frames else 0.0
    print(
        f"saved videos={len(videos)} frames={len(all_rows)} "
        f"monitoring_coverage={coverage:.3f} output={destination}"
    )
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/aihub_video_risk.toml"))
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--max-frames", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_pipeline(
        config_path=args.config,
        video_paths=args.videos,
        output_dir=args.output_dir,
        weights=args.weights,
        device_argument=args.device,
        max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
