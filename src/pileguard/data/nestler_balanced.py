"""Create a train-only density-balanced NESTLER manifest."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from pileguard.evaluation.nestler_validation import density_slice, job_id_from_image


def read_manifest(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(f"NESTLER manifest not found: {path}")
    rows = [Path(row) for row in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError(f"NESTLER manifest is empty: {path}")
    return rows


def label_path_for_image(image_path: Path) -> Path:
    if image_path.parent.parent.name != "images":
        raise ValueError(f"Unexpected NESTLER image layout: {image_path}")
    dataset_root = image_path.parent.parent.parent
    return dataset_root / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def count_boxes(image_path: Path) -> int:
    label_path = label_path_for_image(image_path)
    if not label_path.is_file():
        raise FileNotFoundError(f"NESTLER label not found: {label_path}")
    return len(label_path.read_text(encoding="utf-8").splitlines())


def validate_train_isolation(
    *,
    train_images: list[Path],
    validation_images: list[Path],
    test_images: list[Path],
    dataset_summary: dict[str, Any],
) -> None:
    train_set = set(train_images)
    validation_set = set(validation_images)
    test_set = set(test_images)
    if train_set & validation_set or train_set & test_set or validation_set & test_set:
        raise ValueError("NESTLER source manifests overlap between train, validation, and test")
    expected_train_jobs = set(dataset_summary["splits"]["train"]["jobs"])
    observed_train_jobs = {job_id_from_image(path) for path in train_images}
    if observed_train_jobs != expected_train_jobs:
        raise ValueError("NESTLER train manifest contains a non-train clip")
    if len(train_images) != int(dataset_summary["splits"]["train"]["annotated_frames"]):
        raise ValueError("NESTLER train manifest count does not match the dataset audit")


def balance_density_bins(
    images: list[Path],
    *,
    low_maximum: int,
    medium_maximum: int,
    seed: int,
) -> tuple[list[Path], dict[str, Any]]:
    bins: dict[str, list[Path]] = {
        "density-low": [],
        "density-medium": [],
        "density-high": [],
    }
    for image in images:
        name = density_slice(
            count_boxes(image),
            low_maximum=low_maximum,
            medium_maximum=medium_maximum,
        )
        bins[name].append(image)
    empty_bins = [name for name, rows in bins.items() if not rows]
    if empty_bins:
        raise ValueError(f"Cannot balance empty NESTLER density bins: {empty_bins}")

    target_count = max(len(rows) for rows in bins.values())
    balanced: list[Path] = []
    output_counts: dict[str, int] = {}
    for name, rows in bins.items():
        ordered = sorted(rows)
        repeats, remainder = divmod(target_count, len(ordered))
        expanded = ordered * repeats + ordered[:remainder]
        balanced.extend(expanded)
        output_counts[name] = len(expanded)
    random.Random(seed).shuffle(balanced)

    job_counts = Counter(job_id_from_image(path) for path in balanced)
    audit = {
        "strategy": "oversample every train density bin to the largest source bin",
        "seed": seed,
        "source_unique_images": len(set(images)),
        "training_manifest_rows": len(balanced),
        "target_rows_per_density_bin": target_count,
        "source_density_counts": {name: len(rows) for name, rows in bins.items()},
        "balanced_density_counts": output_counts,
        "balanced_job_counts": dict(sorted(job_counts.items())),
        "validation_images_added": 0,
        "test_images_added": 0,
    }
    return balanced, audit


def write_balanced_dataset(
    *,
    source_dataset_yaml: Path,
    dataset_summary_path: Path,
    output_dir: Path,
    low_maximum: int,
    medium_maximum: int,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    dataset_summary = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
    source_root = source_dataset_yaml.parent
    train_images = read_manifest(source_root / "train.txt")
    validation_images = read_manifest(source_root / "val.txt")
    test_images = read_manifest(source_root / "test.txt")
    validate_train_isolation(
        train_images=train_images,
        validation_images=validation_images,
        test_images=test_images,
        dataset_summary=dataset_summary,
    )
    balanced_images, audit = balance_density_bins(
        train_images,
        low_maximum=low_maximum,
        medium_maximum=medium_maximum,
        seed=seed,
    )
    if set(balanced_images) - set(train_images):
        raise AssertionError("Balanced manifest introduced a non-train image")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = output_dir / "train-balanced.txt"
    validation_manifest = output_dir / "val.txt"
    train_manifest.write_text(
        "\n".join(str(path) for path in balanced_images) + "\n", encoding="utf-8"
    )
    validation_manifest.write_text(
        "\n".join(str(path) for path in validation_images) + "\n", encoding="utf-8"
    )
    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        f"path: {output_dir.resolve()}\n"
        f"train: {train_manifest.resolve()}\n"
        f"val: {validation_manifest.resolve()}\n"
        "names:\n"
        "  0: nestler_tracker_region\n",
        encoding="utf-8",
    )
    audit["validation_manifest_rows"] = len(validation_images)
    audit["test_manifest_in_training_yaml"] = False
    return dataset_yaml, audit

