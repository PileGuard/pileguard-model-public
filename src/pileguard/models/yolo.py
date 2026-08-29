"""Helpers shared by the optional Ultralytics YOLO experiment commands."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def require_ultralytics() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is required. Install the detection extra with "
            "`pip install -e '.[detection]'`."
        ) from error
    return YOLO


def normalize_detection_metrics(results: Any) -> dict[str, float]:
    raw_metrics = getattr(results, "results_dict", {})
    key_mapping = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "map50",
        "metrics/mAP50-95(B)": "map50_95",
        "fitness": "fitness",
    }
    normalized: dict[str, float] = {}
    for source_key, output_key in key_mapping.items():
        value = raw_metrics.get(source_key)
        if value is not None:
            normalized[output_key] = float(value)
    return normalized


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_best_checkpoint(source: Path, destination_dir: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"Ultralytics best checkpoint was not created: {source}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "best.pt"
    shutil.copy2(source, destination)
    return destination


def copy_artifacts(
    source_dir: Path, destination_dir: Path, filenames: tuple[str, ...]
) -> list[str]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for filename in filenames:
        source = source_dir / filename
        if source.is_file():
            shutil.copy2(source, destination_dir / filename)
            copied.append(filename)
    return copied
