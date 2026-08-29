"""Audit NESTLER validation performance by clip, site, and reference density."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

import matplotlib

from pileguard.evaluation.nestler_yolo import evaluate_group, job_id_from_image
from pileguard.features.nestler import JOB_TO_SITE
from pileguard.models.yolo import require_ultralytics, write_json
from pileguard.runtime import resolve_device

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def load_validation_contract(
    *,
    dataset_yaml: Path,
    dataset_summary_path: Path,
    training_summary_path: Path,
    frozen_test_metrics_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    """Load the validation manifest while proving that final test results stay frozen."""

    paths = (
        (dataset_yaml, "dataset YAML"),
        (dataset_summary_path, "dataset summary"),
        (training_summary_path, "training summary"),
        (frozen_test_metrics_path, "frozen test metrics"),
    )
    for path, label in paths:
        if not path.is_file():
            raise FileNotFoundError(f"NESTLER {label} not found: {path}")
    dataset_summary = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    frozen_test = json.loads(frozen_test_metrics_path.read_text(encoding="utf-8"))
    if dataset_summary.get("frame_leakage_between_splits") is not False:
        raise ValueError("Validation audit requires clip-isolated splits")
    if training_summary.get("class_name") != "nestler_tracker_region":
        raise ValueError("Training summary has incompatible class semantics")
    if frozen_test.get("test_evaluated") is not True:
        raise ValueError("Final test metrics are not frozen")
    if frozen_test.get("tuning_after_test_prohibited") is not True:
        raise ValueError("Final test contract does not prohibit further test tuning")

    manifest = dataset_yaml.parent / "val.txt"
    if not manifest.is_file():
        raise FileNotFoundError(f"NESTLER validation manifest not found: {manifest}")
    images = [Path(row) for row in manifest.read_text(encoding="utf-8").splitlines()]
    expected = int(dataset_summary["splits"]["val"]["annotated_frames"])
    if len(images) != expected or len(set(images)) != expected:
        raise ValueError("NESTLER validation manifest does not match the dataset audit")
    expected_jobs = set(dataset_summary["splits"]["val"]["jobs"])
    observed_jobs = {job_id_from_image(path) for path in images}
    if observed_jobs != expected_jobs:
        raise ValueError("NESTLER validation manifest contains unexpected clips")
    return dataset_summary, training_summary, images


def label_path_for_image(image_path: Path) -> Path:
    """Map outputs/.../images/<split>/<stem>.jpg to its YOLO label path."""

    if image_path.parent.parent.name != "images":
        raise ValueError(f"Unexpected NESTLER image layout: {image_path}")
    dataset_root = image_path.parent.parent.parent
    return dataset_root / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def reference_count(image_path: Path) -> int:
    label_path = label_path_for_image(image_path)
    if not label_path.is_file():
        raise FileNotFoundError(f"NESTLER validation label not found: {label_path}")
    return len(label_path.read_text(encoding="utf-8").splitlines())


def density_slice(count: int, *, low_maximum: int, medium_maximum: int) -> str:
    if low_maximum < 0 or medium_maximum <= low_maximum:
        raise ValueError("Density thresholds must satisfy 0 <= low < medium")
    if count <= low_maximum:
        return "density-low"
    if count <= medium_maximum:
        return "density-medium"
    return "density-high"


def build_slices(
    images: list[Path], *, low_maximum: int, medium_maximum: int
) -> dict[str, list[Path]]:
    slices: dict[str, list[Path]] = {}
    for image in images:
        job_id = job_id_from_image(image)
        slices.setdefault(f"clip-{job_id}", []).append(image)
        density = density_slice(
            reference_count(image),
            low_maximum=low_maximum,
            medium_maximum=medium_maximum,
        )
        slices.setdefault(density, []).append(image)
    return dict(sorted(slices.items()))


def build_findings(slice_metrics: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    clip_rows = {
        name: row for name, row in slice_metrics.items() if name.startswith("clip-")
    }
    if len(clip_rows) >= 2:
        map_values = [float(row["metrics"]["map50"]) for row in clip_rows.values()]
        gap = max(map_values) - min(map_values)
        findings.append(f"validation clip mAP50 gap={gap:.4f}")
    density_rows = {
        name: row for name, row in slice_metrics.items() if name.startswith("density-")
    }
    if density_rows:
        worst_name, worst_row = min(
            density_rows.items(), key=lambda item: float(item[1]["metrics"]["map50"])
        )
        findings.append(
            f"worst density slice={worst_name} "
            f"mAP50={float(worst_row['metrics']['map50']):.4f}"
        )
    findings.append("use validation slices only for the next training design; keep test frozen")
    return findings


def compare_slice_metrics(
    current: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Compare matching validation slices without consulting the frozen test set."""

    if set(current) != set(baseline):
        raise ValueError("Current and baseline validation slices do not match")
    metric_names = ("precision", "recall", "map50", "map50_95")
    return {
        name: {
            metric: float(current[name]["metrics"][metric])
            - float(baseline[name]["metrics"][metric])
            for metric in metric_names
        }
        for name in sorted(current)
    }


def save_slice_plot(slice_metrics: dict[str, Any], output_path: Path) -> None:
    names = list(slice_metrics)
    map50 = [float(slice_metrics[name]["metrics"]["map50"]) for name in names]
    recall = [float(slice_metrics[name]["metrics"]["recall"]) for name in names]
    positions = list(range(len(names)))
    figure, axis = plt.subplots(figsize=(10, 5))
    width = 0.38
    axis.bar([position - width / 2 for position in positions], map50, width, label="mAP50")
    axis.bar([position + width / 2 for position in positions], recall, width, label="recall")
    axis.set_xticks(positions, names, rotation=25, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("score")
    axis.set_title("NESTLER validation-only slice audit")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/nestler_validation_audit.toml")
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    input_config = config["input"]
    evaluation_config = config["evaluation"]
    output_config = config["output"]
    dataset_yaml = Path(str(input_config["dataset_yaml"]))
    dataset_summary, training_summary, images = load_validation_contract(
        dataset_yaml=dataset_yaml,
        dataset_summary_path=Path(str(input_config["dataset_summary"])),
        training_summary_path=Path(str(input_config["training_summary"])),
        frozen_test_metrics_path=Path(str(input_config["frozen_test_metrics"])),
    )
    weights = args.weights or Path(str(input_config["weights"]))
    if not weights.is_file():
        raise FileNotFoundError(f"NESTLER checkpoint not found: {weights}")
    device = str(resolve_device(args.device or str(evaluation_config["device"])))
    density_config = config["density"]
    slices = build_slices(
        images,
        low_maximum=int(density_config["low_maximum"]),
        medium_maximum=int(density_config["medium_maximum"]),
    )

    YOLO = require_ultralytics()
    model = YOLO(str(weights))
    run_root = Path(str(output_config["run_root"]))
    slice_metrics: dict[str, Any] = {}
    for name, slice_images in slices.items():
        metrics = evaluate_group(
            model=model,
            images=slice_images,
            runtime_dir=run_root / "runtime",
            run_root=run_root,
            group_name=name,
            evaluation_config=evaluation_config,
            device=device,
        )
        slice_metrics[name] = {
            "images": len(slice_images),
            "reference_boxes": sum(reference_count(path) for path in slice_images),
            "site": (
                JOB_TO_SITE.get(name.removeprefix("clip-"), "mixed")
                if name.startswith("clip-")
                else "mixed"
            ),
            "metrics": metrics,
        }

    artifact_dir = Path(str(output_config["artifact_dir"]))
    summary = {
        "status": "validation-only slice audit",
        "test_inference_performed": False,
        "frozen_test_status": "independent test finalized; no tuning allowed",
        "validation_metrics": training_summary["validation_metrics"],
        "validation_dataset": {
            "jobs": dataset_summary["splits"]["val"]["jobs"],
            "annotated_frames": len(images),
            "boxes": int(dataset_summary["splits"]["val"]["boxes"]),
        },
        "density_bins": {
            "low": f"0-{int(density_config['low_maximum'])}",
            "medium": (
                f"{int(density_config['low_maximum']) + 1}-"
                f"{int(density_config['medium_maximum'])}"
            ),
            "high": f">={int(density_config['medium_maximum']) + 1}",
        },
        "slices": slice_metrics,
        "findings": build_findings(slice_metrics),
        "claim_boundary": (
            "This audit reuses validation only to design future research. Frozen test metrics "
            "remain final for the current checkpoint and cannot validate a changed model."
        ),
    }
    comparison_config = config.get("comparison")
    if comparison_config:
        baseline_path = Path(str(comparison_config["baseline_validation_audit"]))
        if not baseline_path.is_file():
            raise FileNotFoundError(
                f"NESTLER baseline validation audit not found: {baseline_path}"
            )
        baseline_audit = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_audit.get("test_inference_performed") is not False:
            raise ValueError("Baseline comparison unexpectedly performed test inference")
        summary["baseline_comparison"] = {
            "baseline_artifact": baseline_path.as_posix(),
            "slice_delta": compare_slice_metrics(
                slice_metrics, baseline_audit["slices"]
            ),
            "test_inference_performed": False,
        }
    write_json(artifact_dir / "summary.json", summary)
    save_slice_plot(slice_metrics, artifact_dir / "slice_metrics.png")
    print(json.dumps({name: row["metrics"] for name, row in slice_metrics.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
