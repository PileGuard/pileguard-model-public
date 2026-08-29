"""Finalize the clip-isolated NESTLER detector test evaluation once."""

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from pathlib import Path
from typing import Any

from pileguard.features.nestler import JOB_TO_SITE
from pileguard.models.yolo import normalize_detection_metrics, require_ultralytics, write_json
from pileguard.runtime import resolve_device

EVIDENCE_PLOTS = (
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "F1_curve.png",
    "P_curve.png",
    "PR_curve.png",
    "R_curve.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxPR_curve.png",
    "BoxR_curve.png",
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def load_final_test_contract(
    *, dataset_summary_path: Path, training_summary_path: Path, dataset_yaml: Path
) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    """Verify that test clips were reserved and the manifest matches the audit."""

    for path, label in (
        (dataset_summary_path, "dataset audit"),
        (training_summary_path, "training summary"),
        (dataset_yaml, "dataset YAML"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"NESTLER {label} not found: {path}")

    dataset_summary = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    if dataset_summary.get("frame_leakage_between_splits") is not False:
        raise ValueError("NESTLER test evaluation requires clip-isolated splits")
    if not str(dataset_summary.get("independent_test_use", "")).startswith("reserved"):
        raise ValueError("NESTLER dataset audit does not reserve the test split")
    if training_summary.get("test_evaluated") is not False:
        raise ValueError("Training summary does not prove that test was untouched")
    if training_summary.get("class_name") != "nestler_tracker_region":
        raise ValueError("Training checkpoint has incompatible class semantics")

    test_manifest = dataset_yaml.parent / "test.txt"
    if not test_manifest.is_file():
        raise FileNotFoundError(f"NESTLER test manifest not found: {test_manifest}")
    test_images = [Path(row) for row in test_manifest.read_text(encoding="utf-8").splitlines()]
    expected_count = int(dataset_summary["splits"]["test"]["annotated_frames"])
    if len(test_images) != expected_count or len(set(test_images)) != expected_count:
        raise ValueError(
            f"NESTLER test manifest mismatch: expected {expected_count} unique images, "
            f"found {len(test_images)} rows and {len(set(test_images))} unique paths"
        )
    expected_jobs = set(dataset_summary["splits"]["test"]["jobs"])
    observed_jobs = {job_id_from_image(path) for path in test_images}
    if observed_jobs != expected_jobs:
        raise ValueError(
            f"NESTLER test clip mismatch: expected {sorted(expected_jobs)}, "
            f"found {sorted(observed_jobs)}"
        )
    return dataset_summary, training_summary, test_images


def job_id_from_image(path: Path) -> str:
    marker = "_frame_"
    if marker not in path.stem:
        raise ValueError(f"Unable to parse NESTLER job from image: {path}")
    return path.stem.split(marker, maxsplit=1)[0]


def write_group_dataset(output_dir: Path, images: list[Path]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "images.txt"
    manifest.write_text("\n".join(str(path) for path in images) + "\n", encoding="utf-8")
    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        f"path: {output_dir.resolve()}\n"
        f"train: {manifest.resolve()}\n"
        f"val: {manifest.resolve()}\n"
        "names:\n"
        "  0: nestler_tracker_region\n",
        encoding="utf-8",
    )
    return dataset_yaml


def copy_evidence_plots(run_dir: Path, artifact_dir: Path) -> list[str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for filename in EVIDENCE_PLOTS:
        source = run_dir / filename
        if source.is_file():
            shutil.copy2(source, artifact_dir / filename)
            copied.append(filename)
    return copied


def evaluate_group(
    *,
    model: Any,
    images: list[Path],
    runtime_dir: Path,
    run_root: Path,
    group_name: str,
    evaluation_config: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    dataset_yaml = write_group_dataset(runtime_dir / group_name, images)
    result = model.val(
        data=str(dataset_yaml),
        split="val",
        imgsz=int(evaluation_config["image_size"]),
        batch=int(evaluation_config["batch_size"]),
        workers=int(evaluation_config["workers"]),
        device=device,
        conf=float(evaluation_config["confidence_threshold"]),
        iou=float(evaluation_config["iou_threshold"]),
        max_det=int(evaluation_config["max_detections"]),
        project=str(run_root),
        name=group_name,
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    return normalize_detection_metrics(result)


def evaluate_gate(metrics: dict[str, float], gate_config: dict[str, Any]) -> dict[str, Any]:
    criteria = {
        "map50": {
            "minimum": float(gate_config["minimum_map50"]),
            "actual": float(metrics["map50"]),
        },
        "recall": {
            "minimum": float(gate_config["minimum_recall"]),
            "actual": float(metrics["recall"]),
        },
    }
    for criterion in criteria.values():
        criterion["passed"] = criterion["actual"] >= criterion["minimum"]
    quantitative_passed = all(item["passed"] for item in criteria.values())
    domestic_required = bool(gate_config["domestic_validation_required"])
    return {
        "criteria": criteria,
        "quantitative_passed": quantitative_passed,
        "domestic_validation_required": domestic_required,
        "domestic_validation_available": False,
        "monitoring_integration_allowed": quantitative_passed and not domestic_required,
        "decision": "research only; block monitoring integration",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/nestler_tracker_test.toml")
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device")
    parser.add_argument(
        "--allow-repeat",
        action="store_true",
        help="Reproduce the frozen evaluation without using it for further tuning.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    input_config = config["input"]
    evaluation_config = config["evaluation"]
    output_config = config["output"]
    artifact_dir = Path(str(output_config["artifact_dir"]))
    metrics_path = artifact_dir / "metrics.json"
    if metrics_path.exists() and not args.allow_repeat:
        raise FileExistsError(
            f"Final test metrics already exist: {metrics_path}. "
            "Use --allow-repeat only to reproduce the frozen evaluation."
        )

    dataset_yaml = Path(str(input_config["dataset_yaml"]))
    dataset_summary, training_summary, test_images = load_final_test_contract(
        dataset_summary_path=Path(str(input_config["dataset_summary"])),
        training_summary_path=Path(str(input_config["training_summary"])),
        dataset_yaml=dataset_yaml,
    )
    weights = args.weights or Path(str(input_config["weights"]))
    if not weights.is_file():
        raise FileNotFoundError(f"NESTLER fine-tuned checkpoint not found: {weights}")
    device = str(resolve_device(args.device or str(evaluation_config["device"])))

    YOLO = require_ultralytics()
    model = YOLO(str(weights))
    run_root = Path(str(output_config["run_root"]))
    result = model.val(
        data=str(dataset_yaml),
        split="test",
        imgsz=int(evaluation_config["image_size"]),
        batch=int(evaluation_config["batch_size"]),
        workers=int(evaluation_config["workers"]),
        device=device,
        conf=float(evaluation_config["confidence_threshold"]),
        iou=float(evaluation_config["iou_threshold"]),
        max_det=int(evaluation_config["max_detections"]),
        project=str(run_root),
        name="test-final",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    overall_metrics = normalize_detection_metrics(result)
    evidence_plots = copy_evidence_plots(Path(result.save_dir), artifact_dir)

    grouped_images: dict[str, list[Path]] = {}
    for image in test_images:
        grouped_images.setdefault(job_id_from_image(image), []).append(image)
    per_job: dict[str, Any] = {}
    for job_id, images in sorted(grouped_images.items()):
        per_job[job_id] = {
            "site": JOB_TO_SITE.get(job_id, "Unknown"),
            "images": len(images),
            "metrics": evaluate_group(
                model=model,
                images=images,
                runtime_dir=run_root / "runtime",
                run_root=run_root,
                group_name=f"test-{job_id}",
                evaluation_config=evaluation_config,
                device=device,
            ),
        }

    summary = {
        "status": "independent test finalized",
        "test_evaluated": True,
        "tuning_after_test_prohibited": True,
        "checkpoint_selection": training_summary["status"],
        "class_name": training_summary["class_name"],
        "split_unit": dataset_summary["split_unit"],
        "test_dataset": {
            "jobs": dataset_summary["splits"]["test"]["jobs"],
            "annotated_frames": len(test_images),
            "boxes": int(dataset_summary["splits"]["test"]["boxes"]),
        },
        "image_size": int(evaluation_config["image_size"]),
        "batch_size": int(evaluation_config["batch_size"]),
        "device": device,
        "metrics": overall_metrics,
        "per_job": per_job,
        "gate": evaluate_gate(overall_metrics, config["gate"]),
        "evidence_plots": evidence_plots,
        "claim_boundary": (
            "This final test measures overseas NESTLER pose/skeleton tracker-region "
            "detection only. It does not measure Piling incident prediction, and no "
            "domestic Korean farm validation is available."
        ),
    }
    write_json(metrics_path, summary)
    print(
        f"status={summary['status']} test={len(test_images)} metrics={overall_metrics} "
        f"decision={summary['gate']['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

