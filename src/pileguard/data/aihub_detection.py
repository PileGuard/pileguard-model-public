"""Prepare official AI Hub laying-hen community images for YOLO training."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pileguard.data.aihub_laying_hen import (
    AihubAnnotation,
    discover_archive_pairs,
    iter_inner_files,
    load_annotations,
)
from pileguard.data_inventory import resolve_data_root

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SPLITS = {"training": "train", "validation": "val"}
EXPECTED_SPLIT_COUNTS = {
    "train": {"image_count": 36_423, "source_box_count": 851_189},
    "val": {"image_count": 4_546, "source_box_count": 106_023},
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def png_dimensions(payload: bytes, *, image_id: str) -> tuple[int, int]:
    if len(payload) < 24 or payload[:8] != PNG_SIGNATURE or payload[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG payload: {image_id}")
    width, height = struct.unpack(">II", payload[16:24])
    if width < 1 or height < 1:
        raise ValueError(f"Invalid PNG dimensions: {image_id}")
    return width, height


def yolo_label(annotation: AihubAnnotation) -> tuple[str, int]:
    """Serialize unique boxes and report exact duplicates removed from one image."""

    lines: list[str] = []
    seen: set[str] = set()
    duplicate_boxes = 0
    for box in annotation.normalized_boxes:
        line = (
            f"{box.class_id} {box.center_x:.8f} {box.center_y:.8f} "
            f"{box.width:.8f} {box.height:.8f}\n"
        )
        if line in seen:
            duplicate_boxes += 1
            continue
        seen.add(line)
        lines.append(line)
    return "".join(lines), duplicate_boxes


def write_if_changed(path: Path, payload: bytes) -> bool:
    """Write generated data atomically, while making reruns inexpensive."""

    if path.is_file() and path.read_bytes() == payload:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return True


def prepare_split(
    *,
    archive_dir: Path,
    dataset_root: Path,
    output_split: str,
) -> tuple[dict[str, Any], dict[str, list[str]], set[str], dict[str, str]]:
    image_dir = dataset_root / "images" / output_split
    label_dir = dataset_root / "labels" / output_split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    stages: dict[str, Any] = {}
    hashes: dict[str, list[str]] = {}
    filenames: set[str] = set()
    output_names: set[str] = set()
    label_hashes: dict[str, str] = {}
    for pair in discover_archive_pairs(archive_dir):
        annotations = load_annotations(pair)
        source_names: set[str] = set()
        stage_hashes: set[str] = set()
        stage_duplicate_boxes = 0
        stage_training_boxes = 0
        for image_id, payload in iter_inner_files(pair.source_archive, suffix=".png"):
            annotation = annotations.get(image_id)
            if annotation is None:
                raise ValueError(f"Source image has no label in {pair.stage}: {image_id}")
            if image_id in source_names:
                raise ValueError(f"Duplicate source image in {pair.stage}: {image_id}")
            source_names.add(image_id)
            filenames.add(image_id)
            output_name = image_id
            if output_name in output_names:
                raise ValueError(f"Duplicate output image basename: {output_name}")
            output_names.add(output_name)

            dimensions = png_dimensions(payload, image_id=image_id)
            if dimensions != (annotation.width, annotation.height):
                raise ValueError(
                    f"PNG/label dimensions differ for {image_id}: "
                    f"PNG={dimensions}, label={(annotation.width, annotation.height)}"
                )
            digest = hashlib.sha256(payload).hexdigest()
            hashes.setdefault(digest, []).append(image_id)
            stage_hashes.add(digest)
            write_if_changed(image_dir / output_name, payload)
            label_text, duplicate_boxes = yolo_label(annotation)
            stage_duplicate_boxes += duplicate_boxes
            stage_training_boxes += len(annotation.normalized_boxes) - duplicate_boxes
            label_payload = label_text.encode("utf-8")
            write_if_changed(label_dir / f"{Path(output_name).stem}.txt", label_payload)
            label_hashes[image_id] = hashlib.sha256(label_payload).hexdigest()
        missing_sources = sorted(set(annotations) - source_names)
        if missing_sources:
            raise ValueError(
                f"Labels without source images in {pair.stage}: {missing_sources[:5]}"
            )
        stages[pair.stage] = {
            "image_count": len(annotations),
            "source_box_count": sum(
                len(item.pixel_boxes) + item.invalid_boxes for item in annotations.values()
            ),
            "usable_box_count": sum(len(item.pixel_boxes) for item in annotations.values()),
            "duplicate_box_count": stage_duplicate_boxes,
            "training_box_count": stage_training_boxes,
            "invalid_box_count": sum(item.invalid_boxes for item in annotations.values()),
            "clipped_box_count": sum(item.clipped_boxes for item in annotations.values()),
            "unique_image_hashes": len(stage_hashes),
        }

    duplicate_hashes = {
        digest: names for digest, names in hashes.items() if len(names) > 1
    }
    summary = {
        "image_count": sum(stage["image_count"] for stage in stages.values()),
        "source_box_count": sum(stage["source_box_count"] for stage in stages.values()),
        "usable_box_count": sum(stage["usable_box_count"] for stage in stages.values()),
        "duplicate_box_count": sum(stage["duplicate_box_count"] for stage in stages.values()),
        "training_box_count": sum(stage["training_box_count"] for stage in stages.values()),
        "invalid_box_count": sum(stage["invalid_box_count"] for stage in stages.values()),
        "clipped_box_count": sum(stage["clipped_box_count"] for stage in stages.values()),
        "duplicate_image_content_count": sum(
            len(names) - 1 for names in duplicate_hashes.values()
        ),
        "stages": stages,
    }
    return summary, hashes, filenames, label_hashes


def label_conflict_count(
    duplicate_groups: Iterable[list[str]], label_hashes: dict[str, str]
) -> int:
    return sum(
        len({label_hashes[name] for name in names}) > 1 for names in duplicate_groups
    )


def write_selection_manifest(
    dataset_root: Path, *, split: str, image_names: set[str]
) -> Path:
    manifest_path = dataset_root / "manifests" / f"{split}-images.txt"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(f"images/{split}/{name}\n" for name in sorted(image_names)),
        encoding="utf-8",
    )
    return manifest_path


def prepare_dataset(
    *,
    raw_root: Path,
    dataset_root: Path,
    summary_path: Path,
    enforce_official_counts: bool = True,
) -> dict[str, Any]:
    split_summaries: dict[str, Any] = {}
    split_hashes: dict[str, dict[str, list[str]]] = {}
    split_filenames: dict[str, set[str]] = {}
    split_label_hashes: dict[str, dict[str, str]] = {}
    for archive_name, output_split in SPLITS.items():
        summary, hashes, filenames, label_hashes = prepare_split(
            archive_dir=raw_root / archive_name,
            dataset_root=dataset_root,
            output_split=output_split,
        )
        split_summaries[output_split] = summary
        split_hashes[output_split] = hashes
        split_filenames[output_split] = filenames
        split_label_hashes[output_split] = label_hashes

    content_overlap = set(split_hashes["train"]) & set(split_hashes["val"])
    filename_overlap = split_filenames["train"] & split_filenames["val"]
    within_split_duplicates = sum(
        split_summaries[split]["duplicate_image_content_count"] for split in SPLITS.values()
    )
    validation_selected = {
        sorted(names)[0] for names in split_hashes["val"].values()
    }
    training_selected = {
        sorted(names)[0]
        for digest, names in split_hashes["train"].items()
        if digest not in content_overlap
    }
    training_selected -= filename_overlap
    train_manifest = write_selection_manifest(
        dataset_root, split="train", image_names=training_selected
    )
    val_manifest = write_selection_manifest(
        dataset_root, split="val", image_names=validation_selected
    )
    selected_train_hashes = {
        digest
        for digest, names in split_hashes["train"].items()
        if any(name in training_selected for name in names)
    }
    selected_val_hashes = {
        digest
        for digest, names in split_hashes["val"].items()
        if any(name in validation_selected for name in names)
    }
    selected_content_overlap = selected_train_hashes & selected_val_hashes
    within_conflicts = sum(
        label_conflict_count(
            (names for names in split_hashes[split].values() if len(names) > 1),
            split_label_hashes[split],
        )
        for split in SPLITS.values()
    )
    cross_conflicts = sum(
        len(
            {
                *(split_label_hashes["train"][name] for name in split_hashes["train"][digest]),
                *(split_label_hashes["val"][name] for name in split_hashes["val"][digest]),
            }
        )
        > 1
        for digest in content_overlap
    )
    ready = (
        bool(training_selected)
        and bool(validation_selected)
        and not selected_content_overlap
        and not (training_selected & validation_selected)
    )
    was_decontaminated = bool(content_overlap or filename_overlap or within_split_duplicates)
    inventory_ok = all(
        all(split_summaries[split][field] == expected for field, expected in counts.items())
        for split, counts in EXPECTED_SPLIT_COUNTS.items()
    )
    ready = ready and (inventory_ok or not enforce_official_counts)
    result = {
        "status": (
            "ready_after_deduplication" if ready and was_decontaminated else
            "ready" if ready else
            "blocked"
        ),
        "source": "AI Hub dataset 575 official Training and Validation community-image subset",
        "license_note": "Use remains subject to the AI Hub dataset terms accepted at download.",
        "dataset": {
            "class_names": ["layer-chicken"],
            "train_split": "train",
            "validation_split": "val",
            "splits": split_summaries,
            "official_count_contract": {
                "enforced": enforce_official_counts,
                "expected": EXPECTED_SPLIT_COUNTS,
                "ok": inventory_ok,
            },
        },
        "raw_leakage_audit": {
            "training_validation_filename_overlap_count": len(filename_overlap),
            "training_validation_content_overlap_count": len(content_overlap),
            "within_split_duplicate_content_count": within_split_duplicates,
            "within_split_duplicate_label_conflict_group_count": within_conflicts,
            "training_validation_label_conflict_group_count": cross_conflicts,
            "per_file_examples_published": False,
        },
        "selected_dataset": {
            "policy": (
                "Keep one deterministic copy per image hash in Validation; exclude every "
                "Training image hash present in Validation; then keep one copy per remaining "
                "Training hash."
            ),
            "train_manifest": train_manifest.relative_to(dataset_root).as_posix(),
            "validation_manifest": val_manifest.relative_to(dataset_root).as_posix(),
            "train_image_count": len(training_selected),
            "validation_image_count": len(validation_selected),
            "excluded_train_image_count": (
                split_summaries["train"]["image_count"] - len(training_selected)
            ),
            "excluded_validation_image_count": (
                split_summaries["val"]["image_count"] - len(validation_selected)
            ),
            "training_validation_content_overlap_count": len(selected_content_overlap),
            "training_validation_filename_overlap_count": len(
                training_selected & validation_selected
            ),
        },
        "quality_policy": (
            "Non-positive official boxes and exact duplicate boxes within an image are recorded "
            "and excluded; clipped boxes remain after clipping to image bounds. Exact duplicate "
            "image content is retained on disk for audit but excluded from the selected training "
            "manifests."
        ),
        "claim_boundary": (
            "These static community images validate laying-hen detection and spatial-density "
            "features only. They do not contain continuous piling/smothering incident timing."
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/aihub_laying_hen_dataset.toml")
    )
    parser.add_argument("--data-root")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    data_root = resolve_data_root(args.data_root)
    raw_root = data_root / str(config["input"]["raw_dataset_path"])
    dataset_root = data_root / str(config["output"]["dataset_path"])
    summary_path = Path(str(config["output"]["summary_path"]))
    summary = prepare_dataset(
        raw_root=raw_root,
        dataset_root=dataset_root,
        summary_path=summary_path,
    )
    train = summary["dataset"]["splits"]["train"]
    validation = summary["dataset"]["splits"]["val"]
    print(
        f"status={summary['status']} train={train['image_count']} "
        f"validation={validation['image_count']} "
        f"invalid_boxes={train['invalid_box_count'] + validation['invalid_box_count']} "
        f"duplicate_boxes={train['duplicate_box_count'] + validation['duplicate_box_count']}"
    )
    return 0 if str(summary["status"]).startswith("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
