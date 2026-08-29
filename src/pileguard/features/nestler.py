"""Extract density, tracked motion, and optical-flow features from NESTLER clips."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from tqdm import tqdm

from pileguard.data_inventory import resolve_data_root

matplotlib.use("Agg")

JOB_TO_SITE = {
    "job_000004": "Bulgaria",
    "job_000007": "Bulgaria",
    "job_000008": "Bulgaria",
    "job_000009": "Rwanda",
    "job_000010": "Rwanda",
    "job_000011": "Rwanda",
}
SUMMARY_FEATURES = [
    "object_count",
    "bbox_area_ratio",
    "center_spread",
    "mean_nearest_neighbor_distance",
    "max_grid_fraction",
    "tracked_speed_per_second",
    "tracked_coherence",
    "flow_speed_per_second",
    "flow_p90_speed_per_second",
    "flow_divergence_per_second",
    "flow_convergence_per_second",
    "flow_coherence",
]


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass(frozen=True)
class JobInput:
    job_id: str
    site: str
    video_path: Path
    annotation_path: Path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def discover_jobs(dataset_root: Path, data_config: dict[str, Any]) -> list[JobInput]:
    jobs: list[JobInput] = []
    for job_root in sorted(dataset_root.glob(data_config["job_pattern"])):
        videos = sorted(job_root.glob(data_config["video_pattern"]))
        annotations = sorted(job_root.glob(data_config["annotation_pattern"]))
        if len(videos) != 1 or len(annotations) != 1:
            raise FileNotFoundError(
                f"Expected one 400-frame video and one annotation in {job_root}, "
                f"found videos={len(videos)} annotations={len(annotations)}"
            )
        jobs.append(
            JobInput(
                job_id=job_root.name,
                site=JOB_TO_SITE.get(job_root.name, "Unknown"),
                video_path=videos[0],
                annotation_path=annotations[0],
            )
        )
    if not jobs:
        raise FileNotFoundError(f"No NESTLER jobs found in {dataset_root}")
    return jobs


def parse_boxes(
    rows: list[list[float]], *, frame_width: int, frame_height: int
) -> list[BoundingBox]:
    boxes: list[BoundingBox] = []
    for row in rows:
        if len(row) < 5:
            continue
        x1 = min(max(float(row[0]), 0.0), float(frame_width))
        y1 = min(max(float(row[1]), 0.0), float(frame_height))
        x2 = min(max(float(row[2]), 0.0), float(frame_width))
        y2 = min(max(float(row[3]), 0.0), float(frame_height))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, track_id=int(row[4])))
    return boxes


def parse_frame_boxes(
    frame_annotation: dict[str, Any], *, frame_width: int, frame_height: int
) -> tuple[bool, list[BoundingBox]]:
    """Distinguish a missing bbox annotation from a valid empty detection list."""

    rows = frame_annotation.get("tracks_bbox")
    if rows is None:
        return False, []
    return True, parse_boxes(rows, frame_width=frame_width, frame_height=frame_height)


def missing_spatial_features() -> dict[str, None]:
    return {
        "object_count": None,
        "bbox_area_ratio": None,
        "center_x": None,
        "center_y": None,
        "center_spread": None,
        "mean_nearest_neighbor_distance": None,
        "max_grid_fraction": None,
    }


def compute_spatial_features(
    boxes: list[BoundingBox],
    *,
    frame_width: int,
    frame_height: int,
    grid_rows: int,
    grid_columns: int,
) -> dict[str, float | int | None]:
    if grid_rows < 1 or grid_columns < 1:
        raise ValueError("grid dimensions must be positive")
    if not boxes:
        return {
            "object_count": 0,
            "bbox_area_ratio": 0.0,
            "center_x": None,
            "center_y": None,
            "center_spread": None,
            "mean_nearest_neighbor_distance": None,
            "max_grid_fraction": 0.0,
        }

    centers = np.array([box.center for box in boxes], dtype=np.float64)
    normalized_centers = centers / np.array([frame_width, frame_height])
    mean_center = normalized_centers.mean(axis=0)
    center_distances = np.linalg.norm(normalized_centers - mean_center, axis=1)
    if len(boxes) > 1:
        pairwise = np.linalg.norm(
            normalized_centers[:, None, :] - normalized_centers[None, :, :], axis=2
        )
        np.fill_diagonal(pairwise, np.inf)
        nearest_neighbor_distance = float(pairwise.min(axis=1).mean())
    else:
        nearest_neighbor_distance = None

    grid_counts = np.zeros((grid_rows, grid_columns), dtype=np.int64)
    grid_x = np.minimum((normalized_centers[:, 0] * grid_columns).astype(int), grid_columns - 1)
    grid_y = np.minimum((normalized_centers[:, 1] * grid_rows).astype(int), grid_rows - 1)
    for x_index, y_index in zip(grid_x, grid_y, strict=True):
        grid_counts[y_index, x_index] += 1

    return {
        "object_count": len(boxes),
        "bbox_area_ratio": float(sum(box.area for box in boxes) / (frame_width * frame_height)),
        "center_x": float(mean_center[0]),
        "center_y": float(mean_center[1]),
        "center_spread": float(np.sqrt(np.mean(center_distances**2))),
        "mean_nearest_neighbor_distance": nearest_neighbor_distance,
        "max_grid_fraction": float(grid_counts.max() / len(boxes)),
    }


def compute_tracking_features(
    previous_boxes: dict[int, BoundingBox],
    current_boxes: list[BoundingBox],
    *,
    frame_width: int,
    frame_height: int,
    fps: float,
) -> dict[str, float | int | None]:
    velocities: list[tuple[float, float]] = []
    for box in current_boxes:
        previous = previous_boxes.get(box.track_id)
        if previous is None:
            continue
        previous_x, previous_y = previous.center
        current_x, current_y = box.center
        velocities.append(
            (
                (current_x - previous_x) / frame_width * fps,
                (current_y - previous_y) / frame_height * fps,
            )
        )
    if not velocities:
        return {
            "track_match_count": 0,
            "tracked_speed_per_second": None,
            "tracked_direction_x": None,
            "tracked_direction_y": None,
            "tracked_direction_degrees": None,
            "tracked_coherence": None,
        }

    velocity_array = np.array(velocities)
    speeds = np.linalg.norm(velocity_array, axis=1)
    mean_velocity = velocity_array.mean(axis=0)
    mean_speed = float(speeds.mean())
    return {
        "track_match_count": len(velocities),
        "tracked_speed_per_second": mean_speed,
        "tracked_direction_x": float(mean_velocity[0]),
        "tracked_direction_y": float(mean_velocity[1]),
        "tracked_direction_degrees": float(
            math.degrees(math.atan2(mean_velocity[1], mean_velocity[0]))
        ),
        "tracked_coherence": float(np.linalg.norm(mean_velocity) / max(mean_speed, 1e-12)),
    }


def compute_flow_features(
    previous_gray: np.ndarray | None,
    current_gray: np.ndarray,
    boxes: list[BoundingBox],
    *,
    original_width: int,
    original_height: int,
    fps: float,
    flow_config: dict[str, Any],
) -> dict[str, float | int | None]:
    if previous_gray is None or not boxes:
        return {
            "flow_roi_pixels": 0,
            "flow_speed_per_second": None,
            "flow_p90_speed_per_second": None,
            "flow_direction_x": None,
            "flow_direction_y": None,
            "flow_direction_degrees": None,
            "flow_divergence_per_second": None,
            "flow_convergence_per_second": None,
            "flow_coherence": None,
        }

    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        pyr_scale=float(flow_config["flow_pyramid_scale"]),
        levels=int(flow_config["flow_levels"]),
        winsize=int(flow_config["flow_window_size"]),
        iterations=int(flow_config["flow_iterations"]),
        poly_n=int(flow_config["flow_poly_n"]),
        poly_sigma=float(flow_config["flow_poly_sigma"]),
        flags=0,
    )
    resized_height, resized_width = current_gray.shape
    mask = np.zeros_like(current_gray, dtype=np.uint8)
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    for box in boxes:
        x1 = max(0, min(round(box.x1 * scale_x), resized_width - 1))
        y1 = max(0, min(round(box.y1 * scale_y), resized_height - 1))
        x2 = max(0, min(round(box.x2 * scale_x), resized_width - 1))
        y2 = max(0, min(round(box.y2 * scale_y), resized_height - 1))
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=1, thickness=-1)
    selected = mask.astype(bool)
    if not selected.any():
        return {
            "flow_roi_pixels": 0,
            "flow_speed_per_second": None,
            "flow_p90_speed_per_second": None,
            "flow_direction_x": None,
            "flow_direction_y": None,
            "flow_direction_degrees": None,
            "flow_divergence_per_second": None,
            "flow_convergence_per_second": None,
            "flow_coherence": None,
        }

    normalized_flow_x = flow[..., 0] / resized_width * fps
    normalized_flow_y = flow[..., 1] / resized_height * fps
    normalized_x = normalized_flow_x[selected]
    normalized_y = normalized_flow_y[selected]
    speeds = np.hypot(normalized_x, normalized_y)
    mean_x = float(normalized_x.mean())
    mean_y = float(normalized_y.mean())
    mean_speed = float(speeds.mean())
    divergence = (
        np.gradient(normalized_flow_x, axis=1) * resized_width
        + np.gradient(normalized_flow_y, axis=0) * resized_height
    )
    mean_divergence = float(divergence[selected].mean())
    return {
        "flow_roi_pixels": int(selected.sum()),
        "flow_speed_per_second": mean_speed,
        "flow_p90_speed_per_second": float(np.percentile(speeds, 90)),
        "flow_direction_x": mean_x,
        "flow_direction_y": mean_y,
        "flow_direction_degrees": float(math.degrees(math.atan2(mean_y, mean_x))),
        "flow_divergence_per_second": mean_divergence,
        "flow_convergence_per_second": max(-mean_divergence, 0.0),
        "flow_coherence": float(math.hypot(mean_x, mean_y) / max(mean_speed, 1e-12)),
    }


def extract_job_features(
    job: JobInput,
    *,
    feature_config: dict[str, Any],
    max_frames: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annotation = json.loads(job.annotation_path.read_text(encoding="utf-8"))
    original_width = int(annotation["frame_width"])
    original_height = int(annotation["frame_height"])
    fps = float(annotation["fps"])
    frames = annotation["frames"]
    if max_frames is not None:
        frames = frames[:max_frames]

    capture = cv2.VideoCapture(str(job.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {job.video_path}")
    rows: list[dict[str, Any]] = []
    previous_gray: np.ndarray | None = None
    previous_boxes: dict[int, BoundingBox] = {}
    missing_bbox_frames = 0
    started_at = time.monotonic()
    try:
        for frame_annotation in tqdm(frames, desc=job.job_id, leave=False):
            success, frame = capture.read()
            if not success:
                raise RuntimeError(
                    f"Video ended before annotation frame {frame_annotation['frame_index']}: "
                    f"{job.video_path}"
                )
            resized = cv2.resize(
                frame,
                (int(feature_config["resize_width"]), int(feature_config["resize_height"])),
                interpolation=cv2.INTER_AREA,
            )
            current_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            annotation_available, boxes = parse_frame_boxes(
                frame_annotation,
                frame_width=original_width,
                frame_height=original_height,
            )
            if not annotation_available:
                missing_bbox_frames += 1
            row: dict[str, Any] = {
                "job_id": job.job_id,
                "site": job.site,
                "frame_index": int(frame_annotation["frame_index"]),
                "timestamp_seconds": int(frame_annotation["frame_index"]) / fps,
                "bbox_annotation_available": annotation_available,
            }
            if annotation_available:
                row.update(
                    compute_spatial_features(
                        boxes,
                        frame_width=original_width,
                        frame_height=original_height,
                        grid_rows=int(feature_config["grid_rows"]),
                        grid_columns=int(feature_config["grid_columns"]),
                    )
                )
            else:
                row.update(missing_spatial_features())
            row.update(
                compute_tracking_features(
                    previous_boxes,
                    boxes,
                    frame_width=original_width,
                    frame_height=original_height,
                    fps=fps,
                )
            )
            row.update(
                compute_flow_features(
                    previous_gray,
                    current_gray,
                    boxes,
                    original_width=original_width,
                    original_height=original_height,
                    fps=fps,
                    flow_config=feature_config,
                )
            )
            rows.append(row)
            previous_gray = current_gray
            previous_boxes = (
                {box.track_id: box for box in boxes} if annotation_available else {}
            )
    finally:
        capture.release()

    summary = {
        "job_id": job.job_id,
        "site": job.site,
        "fps": fps,
        "frame_width": original_width,
        "frame_height": original_height,
        "frames_processed": len(rows),
        "bbox_annotation_frames": len(rows) - missing_bbox_frames,
        "missing_bbox_annotation_frames": missing_bbox_frames,
        "bbox_annotation_coverage": (
            (len(rows) - missing_bbox_frames) / len(rows) if rows else 0.0
        ),
        "elapsed_seconds": time.monotonic() - started_at,
    }
    return rows, summary


def feature_distributions(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    distributions: dict[str, dict[str, float | int | None]] = {}
    for feature_name in SUMMARY_FEATURES:
        values = np.array(
            [row[feature_name] for row in rows if row[feature_name] is not None],
            dtype=np.float64,
        )
        distributions[feature_name] = {
            "valid_count": int(values.size),
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std()) if values.size else None,
            "p50": float(np.percentile(values, 50)) if values.size else None,
            "p95": float(np.percentile(values, 95)) if values.size else None,
            "max": float(values.max()) if values.size else None,
        }
    return distributions


def summarize_features(rows: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    job_distributions = {
        job["job_id"]: feature_distributions(
            [row for row in rows if row["job_id"] == job["job_id"]]
        )
        for job in jobs
    }
    sites = sorted({row["site"] for row in rows})
    site_distributions = {
        site: feature_distributions([row for row in rows if row["site"] == site])
        for site in sites
    }
    return {
        "frame_count": len(rows),
        "job_count": len(jobs),
        "jobs": jobs,
        "feature_distributions": feature_distributions(rows),
        "job_feature_distributions": job_distributions,
        "site_feature_distributions": site_distributions,
    }


def write_feature_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No feature rows to save")
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_feature_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    from matplotlib import pyplot as plt

    plotted_features = [
        ("bbox_area_ratio", "BBox area ratio"),
        ("max_grid_fraction", "Max grid fraction"),
        ("tracked_speed_per_second", "Tracked speed / s"),
        ("flow_speed_per_second", "Optical-flow speed / s"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    job_ids = sorted({row["job_id"] for row in rows})
    for axis, (feature_name, title) in zip(axes.flat, plotted_features, strict=True):
        for job_id in job_ids:
            job_rows = [row for row in rows if row["job_id"] == job_id]
            values = np.array(
                [np.nan if row[feature_name] is None else row[feature_name] for row in job_rows],
                dtype=np.float64,
            )
            kernel = np.ones(15) / 15
            valid = np.isfinite(values).astype(float)
            smoothed_values = np.convolve(np.nan_to_num(values), kernel, mode="same")
            smoothed_counts = np.convolve(valid, kernel, mode="same")
            smoothed = np.divide(
                smoothed_values,
                smoothed_counts,
                out=np.full_like(smoothed_values, np.nan),
                where=smoothed_counts > 0,
            )
            axis.plot(
                [row["timestamp_seconds"] for row in job_rows],
                smoothed,
                label=job_id.removeprefix("job_"),
                linewidth=1.2,
            )
        axis.set(title=title, xlabel="Seconds")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(title="Job", ncols=2, fontsize=8)
    figure.suptitle("NESTLER frame-level features (15-frame moving average)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/nestler_features.toml"))
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--jobs", nargs="*")
    return parser


def run_extraction(
    *,
    config_path: Path,
    data_root_argument: str | None = None,
    output_dir: Path | None = None,
    max_frames: int | None = None,
    job_ids: list[str] | None = None,
) -> Path:
    """Run NESTLER extraction as a reusable pipeline stage and return its CSV path."""

    config = load_config(config_path)
    data_root = resolve_data_root(data_root_argument)
    dataset_root = data_root / config["data"]["dataset_path"]
    jobs = discover_jobs(dataset_root, config["data"])
    if job_ids:
        selected_job_ids = set(job_ids)
        jobs = [job for job in jobs if job.job_id in selected_job_ids]
        missing = selected_job_ids - {job.job_id for job in jobs}
        if missing:
            raise ValueError(f"Unknown job IDs: {sorted(missing)}")

    output_dir = output_dir or Path(config["output"]["artifact_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    job_summaries: list[dict[str, Any]] = []
    for job in jobs:
        rows, job_summary = extract_job_features(
            job,
            feature_config=config["features"],
            max_frames=max_frames,
        )
        all_rows.extend(rows)
        job_summaries.append(job_summary)
        print(
            f"job={job.job_id} frames={job_summary['frames_processed']} "
            f"elapsed={job_summary['elapsed_seconds']:.1f}s"
        )

    summary = summarize_features(all_rows, job_summaries)
    write_feature_csv(all_rows, output_dir / "frame_features.csv")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_feature_plot(all_rows, output_dir / "feature_overview.png")
    print(f"saved frames={len(all_rows)} jobs={len(jobs)} output={output_dir}")
    return output_dir / "frame_features.csv"


def main() -> int:
    args = build_parser().parse_args()
    run_extraction(
        config_path=args.config,
        data_root_argument=args.data_root,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        job_ids=args.jobs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
