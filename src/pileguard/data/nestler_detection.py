"""Build a leakage-safe YOLO dataset from official NESTLER tracking boxes."""

from __future__ import annotations

import argparse
import json
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from pileguard.data_inventory import resolve_data_root
from pileguard.features.nestler import BoundingBox, JobInput, discover_jobs, parse_frame_boxes

SPLIT_NAMES = ("train", "val", "test")
LABEL_SEMANTICS = (
    "NESTLER pose/skeleton tracker regions; these are not interchangeable with "
    "PIO whole-chicken boxes"
)


@dataclass(frozen=True)
class JobAudit:
    job_id: str
    site: str
    split: str
    annotated_frames: int
    missing_annotation_frames: int
    empty_annotated_frames: int
    boxes: int


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def validate_split_assignment(
    split_config: dict[str, Any], available_job_ids: set[str]
) -> dict[str, str]:
    """Return job-to-split mapping after checking coverage and clip isolation."""

    assignment: dict[str, str] = {}
    duplicates: set[str] = set()
    for split in SPLIT_NAMES:
        job_ids = split_config.get(split)
        if not isinstance(job_ids, list) or not job_ids:
            raise ValueError(f"Split '{split}' must contain at least one job")
        for raw_job_id in job_ids:
            job_id = str(raw_job_id)
            if job_id in assignment:
                duplicates.add(job_id)
            assignment[job_id] = split
    if duplicates:
        raise ValueError(f"Jobs assigned to multiple splits: {sorted(duplicates)}")

    configured = set(assignment)
    unknown = configured - available_job_ids
    missing = available_job_ids - configured
    if unknown:
        raise ValueError(f"Configured NESTLER jobs not found: {sorted(unknown)}")
    if missing:
        raise ValueError(f"NESTLER jobs missing from split config: {sorted(missing)}")
    return assignment


def yolo_label(box: BoundingBox, *, frame_width: int, frame_height: int) -> str:
    """Convert one clipped tracker box to a normalized single-class YOLO row."""

    center_x = (box.x1 + box.x2) / 2 / frame_width
    center_y = (box.y1 + box.y2) / 2 / frame_height
    width = (box.x2 - box.x1) / frame_width
    height = (box.y2 - box.y1) / frame_height
    return f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def extract_job(
    job: JobInput,
    *,
    split: str,
    output_root: Path,
    image_width: int,
    jpeg_quality: int,
    max_frames: int | None = None,
) -> tuple[JobAudit, list[Path]]:
    """Extract annotated frames and YOLO labels for one complete clip."""

    annotation = json.loads(job.annotation_path.read_text(encoding="utf-8"))
    frame_width = int(annotation["frame_width"])
    frame_height = int(annotation["frame_height"])
    frames = annotation["frames"]
    if max_frames is not None:
        frames = frames[:max_frames]

    if image_width < 1:
        raise ValueError("image width must be positive")
    image_height = round(frame_height * image_width / frame_width)
    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(job.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {job.video_path}")

    image_paths: list[Path] = []
    annotated_frames = 0
    missing_frames = 0
    empty_frames = 0
    box_count = 0
    try:
        for expected_position, frame_annotation in enumerate(frames):
            success, frame = capture.read()
            if not success:
                raise RuntimeError(
                    f"Video ended before frame {expected_position}: {job.video_path}"
                )
            frame_index = int(frame_annotation["frame_index"])
            if frame_index != expected_position:
                raise ValueError(
                    f"Non-sequential frame index in {job.annotation_path}: "
                    f"expected {expected_position}, found {frame_index}"
                )
            available, boxes = parse_frame_boxes(
                frame_annotation,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if not available:
                missing_frames += 1
                continue

            annotated_frames += 1
            empty_frames += int(not boxes)
            box_count += len(boxes)
            stem = f"{job.job_id}_frame_{frame_index:06d}"
            image_path = (image_dir / f"{stem}.jpg").resolve()
            label_path = label_dir / f"{stem}.txt"
            resized = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
            written = cv2.imwrite(
                str(image_path), resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if not written:
                raise RuntimeError(f"Unable to write image: {image_path}")
            label_rows = [
                yolo_label(box, frame_width=frame_width, frame_height=frame_height)
                for box in boxes
            ]
            _write_text(label_path, "\n".join(label_rows) + ("\n" if label_rows else ""))
            image_paths.append(image_path)
    finally:
        capture.release()

    return (
        JobAudit(
            job_id=job.job_id,
            site=job.site,
            split=split,
            annotated_frames=annotated_frames,
            missing_annotation_frames=missing_frames,
            empty_annotated_frames=empty_frames,
            boxes=box_count,
        ),
        image_paths,
    )


def write_dataset_files(output_root: Path, manifests: dict[str, list[Path]]) -> None:
    for split in SPLIT_NAMES:
        rows = [str(path) for path in manifests[split]]
        _write_text(output_root / f"{split}.txt", "\n".join(rows) + "\n")
    yaml_text = (
        f"path: {output_root.resolve()}\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "test: test.txt\n"
        "names:\n"
        "  0: nestler_tracker_region\n"
    )
    _write_text(output_root / "dataset.yaml", yaml_text)


def build_summary(audits: list[JobAudit]) -> dict[str, Any]:
    split_summary: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        rows = [audit for audit in audits if audit.split == split]
        split_summary[split] = {
            "jobs": [audit.job_id for audit in rows],
            "sites": dict(sorted(Counter(audit.site for audit in rows).items())),
            "annotated_frames": sum(audit.annotated_frames for audit in rows),
            "missing_annotation_frames": sum(
                audit.missing_annotation_frames for audit in rows
            ),
            "empty_annotated_frames": sum(audit.empty_annotated_frames for audit in rows),
            "boxes": sum(audit.boxes for audit in rows),
        }
    all_jobs = {audit.job_id: audit.split for audit in audits}
    return {
        "source": "official NESTLER dataset",
        "label_semantics": LABEL_SEMANTICS,
        "class_name": "nestler_tracker_region",
        "split_unit": "complete job/clip",
        "frame_leakage_between_splits": False,
        "missing_bbox_policy": "exclude frame; never convert missing annotation to empty label",
        "independent_test_use": "reserved; do not use for threshold selection or early stopping",
        "job_assignment": dict(sorted(all_jobs.items())),
        "splits": split_summary,
        "jobs": [asdict(audit) for audit in audits],
        "limitations": [
            "Each split contains only one Bulgaria clip and one Rwanda clip.",
            "Tracking regions have different semantics from PIO whole-chicken boxes.",
            "This dataset has no piling-event labels and cannot validate incident prediction.",
            "Domestic Korean farm validation data is not available.",
        ],
    }


def prepare_dataset(
    *,
    config_path: Path,
    data_root_argument: str | None = None,
    output_dir: Path | None = None,
    artifact_dir: Path | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    data_root = resolve_data_root(data_root_argument)
    dataset_root = data_root / str(config["data"]["dataset_path"])
    jobs = discover_jobs(dataset_root, config["data"])
    assignment = validate_split_assignment(config["split"], {job.job_id for job in jobs})
    output_root = output_dir or Path(str(config["output"]["dataset_dir"]))
    summary_root = artifact_dir or Path(str(config["output"]["artifact_dir"]))

    audits: list[JobAudit] = []
    manifests: dict[str, list[Path]] = {split: [] for split in SPLIT_NAMES}
    for job in jobs:
        split = assignment[job.job_id]
        audit, image_paths = extract_job(
            job,
            split=split,
            output_root=output_root,
            image_width=int(config["image"]["width"]),
            jpeg_quality=int(config["image"]["jpeg_quality"]),
            max_frames=max_frames,
        )
        audits.append(audit)
        manifests[split].extend(image_paths)

    for split in SPLIT_NAMES:
        if not manifests[split]:
            raise ValueError(f"No annotated frames generated for split '{split}'")
    write_dataset_files(output_root, manifests)
    summary = build_summary(audits)
    summary["image_width"] = int(config["image"]["width"])
    summary["jpeg_quality"] = int(config["image"]["jpeg_quality"])
    _write_text(
        summary_root / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/nestler_detection_dataset.toml")
    )
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--max-frames", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = prepare_dataset(
        config_path=args.config,
        data_root_argument=args.data_root,
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

