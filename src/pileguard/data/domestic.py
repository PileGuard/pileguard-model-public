"""Audit owner-approved domestic laying-hen videos before model validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pileguard.data_inventory import resolve_data_root

CLIP_COLUMNS = (
    "clip_id",
    "farm_id",
    "house_id",
    "camera_id",
    "captured_at",
    "video_path",
    "split",
    "clip_outcome",
    "review_status",
    "consent_status",
    "sha256",
)
EVENT_COLUMNS = (
    "event_id",
    "clip_id",
    "event_type",
    "start_seconds",
    "end_seconds",
    "review_status",
)


@dataclass(frozen=True)
class VideoMetadata:
    duration_seconds: float
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    manifest: str
    row: int | None
    message: str


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def read_manifest(path: Path, expected_columns: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path.name}")
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise ValueError(
                f"{path.name} columns must exactly match the published privacy-safe schema"
            )
        return [dict(row) for row in reader]


def secure_media_path(dataset_root: Path, relative_value: str) -> Path:
    """Resolve a media path while rejecting absolute, traversal, and symlink inputs."""

    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("video_path must be a non-empty relative path without traversal")
    resolved_root = dataset_root.resolve()
    candidate = resolved_root
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError("video_path must not use symbolic links")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("video_path escapes the domestic dataset root")
    return resolved


def video_signature_matches(path: Path) -> bool:
    with path.open("rb") as input_file:
        header = input_file.read(12)
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".mov"}:
        return len(header) >= 8 and header[4:8] == b"ftyp"
    if suffix == ".avi":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI "
    return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> VideoMetadata:
    """Read basic video metadata only after extension and signature checks pass."""

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - depends on optional installation
        raise RuntimeError("Install the 'video' optional dependency to audit videos") from error

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("video decoder could not open the file")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if not math.isfinite(fps) or not math.isfinite(frame_count) or fps <= 0 or frame_count <= 0:
        raise ValueError("video metadata has non-positive FPS or frame count")
    return VideoMetadata(
        duration_seconds=frame_count / fps,
        width=width,
        height=height,
        fps=fps,
    )


def add_issue(
    issues: list[AuditIssue], code: str, manifest: str, row: int | None, message: str
) -> None:
    issues.append(AuditIssue("error", code, manifest, row, message))


def audit_domestic_dataset(
    *,
    dataset_root: Path,
    config: dict[str, Any],
    video_probe: Callable[[Path], VideoMetadata] = probe_video,
) -> dict[str, Any]:
    """Return a privacy-safe readiness audit without exposing IDs or media paths."""

    data_config = config["data"]
    security = config["security"]
    quality = config["quality"]
    labels = config["labels"]
    clips = read_manifest(dataset_root / str(data_config["clip_manifest"]), CLIP_COLUMNS)
    events = read_manifest(dataset_root / str(data_config["event_manifest"]), EVENT_COLUMNS)
    issues: list[AuditIssue] = []
    identifier = re.compile(str(security["identifier_pattern"]))
    allowed_suffixes = {str(value).lower() for value in security["allowed_video_suffixes"]}
    allowed_outcomes = {str(value) for value in labels["allowed_clip_outcomes"]}
    allowed_events = {str(value) for value in labels["allowed_event_types"]}
    accepted_reviews = {str(value) for value in labels["accepted_review_statuses"]}
    required_splits = {str(value) for value in labels["required_splits"]}
    required_outcomes = {str(value) for value in labels["required_outcomes_per_split"]}

    seen_clips: set[str] = set()
    clip_rows: dict[str, dict[str, str]] = {}
    clip_row_numbers: dict[str, int] = {}
    video_metadata: dict[str, VideoMetadata] = {}
    farm_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    split_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    outcome_counts: Counter[str] = Counter()
    valid_hashes = 0
    seen_hashes: set[str] = set()

    for row_number, row in enumerate(clips, start=2):
        clip_id = row["clip_id"]
        if clip_id in seen_clips:
            add_issue(issues, "duplicate_clip_id", "clips", row_number, "clip_id is duplicated")
        seen_clips.add(clip_id)
        if clip_id not in clip_rows:
            clip_rows[clip_id] = row
            clip_row_numbers[clip_id] = row_number
        for field in ("clip_id", "farm_id", "house_id", "camera_id"):
            if not identifier.fullmatch(row[field]):
                add_issue(
                    issues,
                    "invalid_pseudonymous_id",
                    "clips",
                    row_number,
                    f"{field} does not match the pseudonymous ID policy",
                )
        split = row["split"]
        if split not in required_splits:
            add_issue(issues, "invalid_split", "clips", row_number, "split is not allowed")
        else:
            split_counts[split] += 1
            farm_splits[row["farm_id"]].add(split)
        outcome = row["clip_outcome"]
        if outcome not in allowed_outcomes:
            add_issue(
                issues, "invalid_clip_outcome", "clips", row_number, "clip outcome is not allowed"
            )
        else:
            outcome_counts[outcome] += 1
            if split in required_splits:
                split_outcomes[split][outcome] += 1
        if row["review_status"] not in accepted_reviews:
            add_issue(
                issues,
                "insufficient_clip_review",
                "clips",
                row_number,
                "clip must be double-reviewed or adjudicated",
            )
        if row["consent_status"] != "research_approved":
            add_issue(
                issues,
                "missing_usage_approval",
                "clips",
                row_number,
                "owner research approval is required",
            )
        try:
            captured_at = datetime.fromisoformat(row["captured_at"])
        except ValueError:
            captured_at = None
        if captured_at is None or captured_at.tzinfo is None:
            add_issue(
                issues,
                "invalid_captured_at",
                "clips",
                row_number,
                "captured_at must be an ISO 8601 timestamp with timezone",
            )

        expected_hash = row["sha256"].lower()
        expected_hash_is_valid = re.fullmatch(r"[0-9a-f]{64}", expected_hash) is not None
        if not expected_hash_is_valid:
            add_issue(issues, "invalid_sha256", "clips", row_number, "sha256 must be 64 hex digits")
        try:
            video_path = secure_media_path(dataset_root, row["video_path"])
        except ValueError as error:
            add_issue(issues, "unsafe_video_path", "clips", row_number, str(error))
            continue
        if video_path.suffix.lower() not in allowed_suffixes:
            add_issue(
                issues, "unsupported_video_type", "clips", row_number, "video suffix is not allowed"
            )
            continue
        if not video_path.is_file():
            add_issue(issues, "missing_video", "clips", row_number, "video file does not exist")
            continue
        if video_path.stat().st_size < int(security["minimum_file_bytes"]):
            add_issue(
                issues, "video_too_small", "clips", row_number, "video file is smaller than allowed"
            )
            continue
        if not video_signature_matches(video_path):
            add_issue(
                issues,
                "signature_mismatch",
                "clips",
                row_number,
                "video signature does not match suffix",
            )
            continue
        if not expected_hash_is_valid:
            continue
        if sha256_file(video_path) != expected_hash:
            add_issue(
                issues, "sha256_mismatch", "clips", row_number, "video checksum does not match"
            )
            continue
        valid_hashes += 1
        if expected_hash in seen_hashes:
            add_issue(
                issues,
                "duplicate_video_content",
                "clips",
                row_number,
                "video content is duplicated by another clip row",
            )
            continue
        seen_hashes.add(expected_hash)
        try:
            metadata = video_probe(video_path)
        except (RuntimeError, ValueError) as error:
            add_issue(issues, "video_probe_failed", "clips", row_number, str(error))
            continue
        if not math.isfinite(metadata.duration_seconds) or not math.isfinite(metadata.fps):
            add_issue(
                issues,
                "invalid_video_metadata",
                "clips",
                row_number,
                "video duration and FPS must be finite",
            )
            continue
        video_metadata[clip_id] = metadata
        if metadata.duration_seconds < float(quality["minimum_duration_seconds"]):
            add_issue(issues, "short_video", "clips", row_number, "video duration is below minimum")
        if metadata.width < int(quality["minimum_width"]) or metadata.height < int(
            quality["minimum_height"]
        ):
            add_issue(
                issues, "low_resolution", "clips", row_number, "video resolution is below minimum"
            )
        if not float(quality["minimum_fps"]) <= metadata.fps <= float(quality["maximum_fps"]):
            add_issue(
                issues, "invalid_fps", "clips", row_number, "video FPS is outside allowed range"
            )

    for splits in farm_splits.values():
        if len(splits) > 1:
            add_issue(
                issues,
                "farm_split_leakage",
                "clips",
                None,
                "one farm appears in more than one split",
            )
    missing_splits = required_splits - set(split_counts)
    if missing_splits:
        add_issue(
            issues,
            "missing_required_split",
            "clips",
            None,
            "one or more required splits have no clips",
        )
    for split in sorted(required_splits):
        if required_outcomes - set(split_outcomes[split]):
            add_issue(
                issues,
                "missing_split_outcome",
                "clips",
                None,
                "a required split lacks one or more required clip outcomes",
            )

    event_types_by_clip: dict[str, set[str]] = defaultdict(set)
    seen_events: set[str] = set()
    for row_number, row in enumerate(events, start=2):
        event_id = row["event_id"]
        if not identifier.fullmatch(event_id):
            add_issue(
                issues,
                "invalid_pseudonymous_id",
                "events",
                row_number,
                "event_id does not match the pseudonymous ID policy",
            )
        if event_id in seen_events:
            add_issue(issues, "duplicate_event_id", "events", row_number, "event_id is duplicated")
        seen_events.add(event_id)
        clip_id = row["clip_id"]
        if clip_id not in clip_rows:
            add_issue(
                issues,
                "unknown_event_clip",
                "events",
                row_number,
                "event references an unknown clip",
            )
            continue
        event_type = row["event_type"]
        if event_type not in allowed_events:
            add_issue(
                issues, "invalid_event_type", "events", row_number, "event type is not allowed"
            )
        else:
            event_types_by_clip[clip_id].add(event_type)
        if row["review_status"] not in accepted_reviews:
            add_issue(
                issues,
                "insufficient_event_review",
                "events",
                row_number,
                "event must be double-reviewed or adjudicated",
            )
        try:
            start = float(row["start_seconds"])
            end = float(row["end_seconds"])
        except ValueError:
            add_issue(
                issues, "invalid_event_time", "events", row_number, "event times must be numeric"
            )
            continue
        duration = video_metadata.get(clip_id)
        if start < 0 or end <= start or (duration is not None and end > duration.duration_seconds):
            add_issue(
                issues,
                "invalid_event_time",
                "events",
                row_number,
                "event times are outside the clip",
            )

    for clip_id, row in clip_rows.items():
        outcome = row["clip_outcome"]
        event_types = event_types_by_clip.get(clip_id, set())
        row_number = clip_row_numbers[clip_id]
        if outcome == "normal" and event_types:
            add_issue(
                issues, "normal_clip_has_event", "clips", row_number, "normal clip has an event"
            )
        if outcome in allowed_events and outcome not in event_types:
            add_issue(
                issues,
                "missing_outcome_event",
                "clips",
                row_number,
                "non-normal clip requires a matching event",
            )

    return {
        "status": "domestic data intake audit",
        "ready_for_model_validation": not issues,
        "source_policy": "owner-approved domestic laying-hen data only",
        "privacy": "identifiers and media paths are excluded from this report",
        "counts": {
            "clips": len(clips),
            "events": len(events),
            "farms": len(farm_splits),
            "valid_sha256": valid_hashes,
            "by_split": dict(sorted(split_counts.items())),
            "by_split_outcome": {
                split: dict(sorted(counts.items()))
                for split, counts in sorted(split_outcomes.items())
            },
            "by_clip_outcome": dict(sorted(outcome_counts.items())),
        },
        "farm_split_leakage": any(issue.code == "farm_split_leakage" for issue in issues),
        "issues": [asdict(issue) for issue in issues],
        "claim_boundary": (
            "A passing intake audit proves schema, integrity, media quality, label review, and "
            "farm-level split isolation only. Model performance requires a separate evaluation."
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/domestic_data_intake.toml"))
    parser.add_argument("--data-root")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    data_root = resolve_data_root(args.data_root)
    dataset_root = data_root / str(config["data"]["dataset_path"])
    try:
        report = audit_domestic_dataset(dataset_root=dataset_root, config=config)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Domestic data configuration error: {error}")
        return 2
    output_path = Path(str(config["output"]["audit_path"]))
    write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_model_validation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
