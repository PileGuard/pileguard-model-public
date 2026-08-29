"""Extract static density features from the official PIO YOLO annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from pileguard.data_inventory import IMAGE_SUFFIXES, resolve_data_root

matplotlib.use("Agg")

PIO_NAME_PATTERN = re.compile(r"^(?P<environment>[CP])-W(?P<week>[1-6])-(?:V)?\d+$")
ENVIRONMENT_NAMES = {"C": "commercial", "P": "prototype"}
FEATURE_NAMES = (
    "object_count",
    "bbox_area_ratio",
    "center_spread",
    "mean_nearest_neighbor_distance",
    "max_grid_fraction",
)


@dataclass(frozen=True)
class NormalizedBox:
    """One YOLO box represented in normalized image coordinates."""

    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float

    @property
    def clipped_bounds(self) -> tuple[float, float, float, float]:
        return (
            max(self.center_x - self.width / 2, 0.0),
            max(self.center_y - self.height / 2, 0.0),
            min(self.center_x + self.width / 2, 1.0),
            min(self.center_y + self.height / 2, 1.0),
        )

    @property
    def crosses_frame(self) -> bool:
        x1, y1, x2, y2 = self.clipped_bounds
        return (
            not math.isclose(x2 - x1, self.width, abs_tol=1e-12)
            or not math.isclose(y2 - y1, self.height, abs_tol=1e-12)
        )

    @property
    def clipped_area(self) -> float:
        x1, y1, x2, y2 = self.clipped_bounds
        return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


@dataclass(frozen=True)
class PioSample:
    """A paired PIO image and YOLO label file."""

    image_id: str
    split: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class LabelParseResult:
    """Valid boxes and annotation-quality counts for one label file."""

    boxes: tuple[NormalizedBox, ...]
    source_rows: int
    invalid_box_count: int
    clipped_box_count: int


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def discover_samples(dataset_root: Path, splits: list[str]) -> list[PioSample]:
    """Pair image and label files by stem and fail on incomplete official splits."""

    samples: list[PioSample] = []
    for split in splits:
        image_root = dataset_root / "images" / split
        label_root = dataset_root / "labels" / split
        if not image_root.is_dir() or not label_root.is_dir():
            raise FileNotFoundError(f"PIO split is incomplete: {split} in {dataset_root}")
        images = {
            path.stem: path
            for path in image_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        labels = {
            path.stem: path
            for path in label_root.glob("*.txt")
            if path.name != "classes.txt"
        }
        missing_labels = sorted(set(images) - set(labels))
        missing_images = sorted(set(labels) - set(images))
        if missing_labels or missing_images:
            raise ValueError(
                f"Unpaired PIO files in {split}: "
                f"missing_labels={missing_labels[:5]} missing_images={missing_images[:5]}"
            )
        samples.extend(
            PioSample(
                image_id=image_id,
                split=split,
                image_path=images[image_id],
                label_path=labels[image_id],
            )
            for image_id in sorted(images)
        )
    if not samples:
        raise FileNotFoundError(f"No PIO image-label pairs found in {dataset_root}")
    return samples


def parse_yolo_label(path: Path, *, expected_class_id: int = 0) -> LabelParseResult:
    """Parse one PIO label, skipping zero-area boxes while rejecting malformed rows."""

    boxes: list[NormalizedBox] = []
    source_rows = 0
    invalid_box_count = 0
    clipped_box_count = 0
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        source_rows += 1
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed YOLO row at {path}:{line_number}: {raw_line!r}")
        try:
            class_id = int(parts[0])
            center_x, center_y, width, height = map(float, parts[1:])
        except ValueError as error:
            raise ValueError(
                f"Invalid YOLO value at {path}:{line_number}: {raw_line!r}"
            ) from error
        values = (center_x, center_y, width, height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite YOLO value at {path}:{line_number}")
        if class_id != expected_class_id:
            raise ValueError(
                f"Unexpected class {class_id} at {path}:{line_number}; "
                f"expected {expected_class_id}"
            )
        if not 0 <= center_x <= 1 or not 0 <= center_y <= 1:
            raise ValueError(f"YOLO center outside [0, 1] at {path}:{line_number}")
        if width <= 0 or height <= 0:
            invalid_box_count += 1
            continue
        if width > 1 or height > 1:
            raise ValueError(f"YOLO size outside (0, 1] at {path}:{line_number}")
        box = NormalizedBox(class_id, center_x, center_y, width, height)
        clipped_box_count += int(box.crosses_frame)
        boxes.append(box)
    return LabelParseResult(
        boxes=tuple(boxes),
        source_rows=source_rows,
        invalid_box_count=invalid_box_count,
        clipped_box_count=clipped_box_count,
    )


def parse_image_metadata(image_id: str) -> tuple[str, int | None]:
    """Decode official C/P and W1-W6 filename metadata, preserving unknown legacy IDs."""

    match = PIO_NAME_PATTERN.fullmatch(image_id)
    if match is None:
        return "unknown", None
    return ENVIRONMENT_NAMES[match.group("environment")], int(match.group("week"))


def compute_static_features(
    boxes: tuple[NormalizedBox, ...], *, grid_rows: int, grid_columns: int
) -> dict[str, float | int | None]:
    """Compute NESTLER-compatible spatial features without claiming Piling labels."""

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
            "max_grid_count": 0,
            "max_grid_fraction": 0.0,
        }

    centers = np.asarray([(box.center_x, box.center_y) for box in boxes], dtype=np.float64)
    mean_center = centers.mean(axis=0)
    center_distances = np.linalg.norm(centers - mean_center, axis=1)
    if len(boxes) > 1:
        pairwise = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        np.fill_diagonal(pairwise, np.inf)
        nearest_neighbor_distance = float(pairwise.min(axis=1).mean())
    else:
        nearest_neighbor_distance = None

    grid_counts = np.zeros((grid_rows, grid_columns), dtype=np.int64)
    grid_x = np.minimum((centers[:, 0] * grid_columns).astype(int), grid_columns - 1)
    grid_y = np.minimum((centers[:, 1] * grid_rows).astype(int), grid_rows - 1)
    for x_index, y_index in zip(grid_x, grid_y, strict=True):
        grid_counts[y_index, x_index] += 1
    max_grid_count = int(grid_counts.max())
    return {
        "object_count": len(boxes),
        "bbox_area_ratio": float(sum(box.clipped_area for box in boxes)),
        "center_x": float(mean_center[0]),
        "center_y": float(mean_center[1]),
        "center_spread": float(np.sqrt(np.mean(center_distances**2))),
        "mean_nearest_neighbor_distance": nearest_neighbor_distance,
        "max_grid_count": max_grid_count,
        "max_grid_fraction": float(max_grid_count / len(boxes)),
    }


def extract_sample_features(
    sample: PioSample, *, expected_class_id: int, grid_rows: int, grid_columns: int
) -> dict[str, Any]:
    parsed = parse_yolo_label(sample.label_path, expected_class_id=expected_class_id)
    environment, week = parse_image_metadata(sample.image_id)
    return {
        "image_id": sample.image_id,
        "split": sample.split,
        "environment": environment,
        "week": week,
        "source_annotation_count": parsed.source_rows,
        "invalid_annotation_count": parsed.invalid_box_count,
        "clipped_annotation_count": parsed.clipped_box_count,
        **compute_static_features(
            parsed.boxes,
            grid_rows=grid_rows,
            grid_columns=grid_columns,
        ),
    }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def feature_distributions(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    distributions: dict[str, dict[str, float | int | None]] = {}
    for feature in FEATURE_NAMES:
        values = [float(row[feature]) for row in rows if row[feature] is not None]
        distributions[feature] = {
            "valid_count": len(values),
            "mean": float(np.mean(values)) if values else None,
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "max": max(values) if values else None,
        }
    return distributions


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image_count": len(rows),
        "source_annotations": sum(int(row["source_annotation_count"]) for row in rows),
        "valid_boxes": sum(int(row["object_count"]) for row in rows),
        "invalid_boxes": sum(int(row["invalid_annotation_count"]) for row in rows),
        "clipped_boxes": sum(int(row["clipped_annotation_count"]) for row in rows),
        "feature_distributions": feature_distributions(rows),
    }


def summarize_features(rows: list[dict[str, Any]], *, top_count: int) -> dict[str, Any]:
    split_names = sorted({str(row["split"]) for row in rows})
    environments = sorted({str(row["environment"]) for row in rows})
    weeks = sorted({int(row["week"]) for row in rows if row["week"] is not None})
    top_rows = sorted(rows, key=lambda row: int(row["object_count"]), reverse=True)[:top_count]
    return {
        **group_summary(rows),
        "splits": {
            split: group_summary([row for row in rows if row["split"] == split])
            for split in split_names
        },
        "environments": {
            environment: group_summary(
                [row for row in rows if row["environment"] == environment]
            )
            for environment in environments
        },
        "weeks": {
            str(week): group_summary([row for row in rows if row["week"] == week])
            for week in weeks
        },
        "metadata_counts": {
            "environment": dict(Counter(str(row["environment"]) for row in rows)),
            "week": dict(Counter(str(row["week"]) for row in rows)),
        },
        "top_crowded_images": [
            {
                "image_id": row["image_id"],
                "split": row["split"],
                "environment": row["environment"],
                "week": row["week"],
                "object_count": row["object_count"],
                "bbox_area_ratio": row["bbox_area_ratio"],
            }
            for row in top_rows
        ],
        "claim_boundary": (
            "PIO contains broiler detection boxes but no Piling event labels; these values "
            "validate static density-feature extraction only."
        ),
    }


def write_feature_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No PIO feature rows to save")
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_feature_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    from matplotlib import pyplot as plt

    known_rows = [row for row in rows if row["week"] is not None]
    environments = ("commercial", "prototype")
    colors = {"commercial": "#0B6E99", "prototype": "#E07A2D"}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for environment in environments:
        values = [
            int(row["object_count"])
            for row in known_rows
            if row["environment"] == environment
        ]
        axes[0].hist(values, bins=30, alpha=0.55, label=environment, color=colors[environment])
        medians = []
        for week in range(1, 7):
            week_values = [
                float(row["bbox_area_ratio"])
                for row in known_rows
                if row["environment"] == environment and row["week"] == week
            ]
            medians.append(float(np.median(week_values)) if week_values else np.nan)
        axes[1].plot(
            range(1, 7),
            medians,
            marker="o",
            label=environment,
            color=colors[environment],
        )
    axes[0].set(title="PIO box-count distribution", xlabel="Valid boxes / image", ylabel="Images")
    axes[1].set(
        title="Median summed clipped bbox area by week",
        xlabel="Broiler growth week",
        ylabel="Summed normalized bbox area",
        xticks=range(1, 7),
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("PIO static density features (not Piling labels)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pio_features.toml"))
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--splits", nargs="*")
    return parser


def run_extraction(
    *,
    config_path: Path,
    data_root_argument: str | None = None,
    output_dir: Path | None = None,
    max_images: int | None = None,
    splits: list[str] | None = None,
) -> Path:
    config = load_config(config_path)
    data_root = resolve_data_root(data_root_argument)
    dataset_root = data_root / config["data"]["dataset_path"]
    selected_splits = splits or list(config["data"]["splits"])
    samples = discover_samples(dataset_root, selected_splits)
    if max_images is not None:
        if max_images < 1:
            raise ValueError("max_images must be positive")
        samples = samples[:max_images]

    feature_config = config["features"]
    rows = [
        extract_sample_features(
            sample,
            expected_class_id=int(feature_config["expected_class_id"]),
            grid_rows=int(feature_config["grid_rows"]),
            grid_columns=int(feature_config["grid_columns"]),
        )
        for sample in samples
    ]
    summary = summarize_features(rows, top_count=int(feature_config["top_crowded_images"]))
    output_dir = output_dir or Path(config["output"]["artifact_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "image_features.csv"
    write_feature_csv(rows, csv_path)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_feature_plot(rows, output_dir / "feature_overview.png")
    print(
        f"saved images={summary['image_count']} valid_boxes={summary['valid_boxes']} "
        f"invalid_boxes={summary['invalid_boxes']} output={output_dir}"
    )
    return csv_path


def main() -> int:
    args = build_parser().parse_args()
    run_extraction(
        config_path=args.config,
        data_root_argument=args.data_root,
        output_dir=args.output_dir,
        max_images=args.max_images,
        splits=args.splits,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
