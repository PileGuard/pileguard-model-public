"""Runtime dataset preparation for the official PIO YOLO splits."""

from __future__ import annotations

import json
import random
from pathlib import Path

from pileguard.data_inventory import IMAGE_SUFFIXES


def discover_split_images(dataset_root: Path, split: str) -> list[Path]:
    image_root = dataset_root / "images" / split
    label_root = dataset_root / "labels" / split
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(f"PIO split is incomplete: {split} in {dataset_root}")

    images = sorted(
        path
        for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    missing_labels = [
        path.name for path in images if not (label_root / f"{path.stem}.txt").is_file()
    ]
    if missing_labels:
        raise ValueError(f"PIO images without labels in {split}: {missing_labels[:5]}")
    if not images:
        raise FileNotFoundError(f"No PIO images found in {image_root}")
    return images


def select_smoke_images(images: list[Path], *, limit: int, seed: int) -> list[Path]:
    if limit < 1:
        raise ValueError("smoke image limit must be positive")
    if len(images) <= limit:
        return images
    generator = random.Random(seed)
    return sorted(generator.sample(images, limit))


def write_runtime_dataset(
    *,
    dataset_root: Path,
    output_dir: Path,
    train_split: str,
    validation_split: str,
    class_names: list[str],
    smoke_limit: int | None = None,
    seed: int = 2026,
) -> tuple[Path, dict[str, int]]:
    """Create an ignored runtime YAML without committing machine-specific paths."""

    if not class_names:
        raise ValueError("At least one class name is required")
    train_images = discover_split_images(dataset_root, train_split)
    val_images = discover_split_images(dataset_root, validation_split)
    if smoke_limit is not None:
        train_images = select_smoke_images(train_images, limit=smoke_limit, seed=seed)
        val_images = select_smoke_images(val_images, limit=smoke_limit, seed=seed + 1)

    dataset_yaml = write_dataset_from_images(
        train_images=train_images,
        validation_images=val_images,
        output_dir=output_dir,
        class_names=class_names,
    )
    return dataset_yaml, {"train_images": len(train_images), "validation_images": len(val_images)}


def write_dataset_from_images(
    *,
    train_images: list[Path],
    validation_images: list[Path],
    output_dir: Path,
    class_names: list[str],
) -> Path:
    """Write Ultralytics manifests and YAML for explicit image subsets."""

    if not train_images or not validation_images:
        raise ValueError("Train and validation image lists must not be empty")
    if not class_names:
        raise ValueError("At least one class name is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = output_dir / "train-images.txt"
    val_manifest = output_dir / "val-images.txt"
    train_manifest.write_text(
        "\n".join(str(path.resolve()) for path in train_images) + "\n", encoding="utf-8"
    )
    val_manifest.write_text(
        "\n".join(str(path.resolve()) for path in validation_images) + "\n",
        encoding="utf-8",
    )

    dataset_yaml = output_dir / "dataset.yaml"
    quoted_names = ", ".join(json.dumps(name) for name in class_names)
    dataset_yaml.write_text(
        f"train: {json.dumps(str(train_manifest.resolve()))}\n"
        f"val: {json.dumps(str(val_manifest.resolve()))}\n"
        f"nc: {len(class_names)}\n"
        f"names: [{quoted_names}]\n",
        encoding="utf-8",
    )
    return dataset_yaml
