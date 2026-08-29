"""Run NESTLER features through camera-relative PileGuard risk monitoring."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from pileguard.features.nestler import run_extraction
from pileguard.models.risk import (
    RiskEvidence,
    baseline_delta,
    compute_mechanism_scores,
    next_alert_state,
    risk_index,
)

matplotlib.use("Agg")

BASELINE_FEATURES = (
    "object_count",
    "bbox_area_ratio",
    "mean_nearest_neighbor_distance",
    "max_grid_fraction",
    "tracked_speed_per_second",
    "tracked_coherence",
    "flow_speed_per_second",
    "flow_convergence_per_second",
    "flow_coherence",
)
MONITORED_EVIDENCE = (
    "density",
    "inflow",
    "proximity",
    "convergence",
    "directional_coherence",
    "corner",
)
FEATURE_MAPPING = {
    "density": "Positive camera-relative change in bbox_area_ratio.",
    "inflow": "Positive object-count and tracked/flow-speed change used as an inflow proxy.",
    "proximity": "Negative camera-relative change in mean nearest-neighbor distance.",
    "convergence": "Positive camera-relative change in optical-flow convergence.",
    "directional_coherence": "Positive tracked or optical-flow coherence change.",
    "corner": "Positive change in the most occupied spatial grid fraction.",
    "context": "Fixed at zero because NESTLER has no worker, door, or blind-spot metadata.",
}


@dataclass(frozen=True)
class RobustBaseline:
    valid_count: int
    median: float
    q25: float
    q75: float
    iqr: float
    effective_iqr: float


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def optional_float(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def load_feature_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"NESTLER feature CSV not found: {path}")
    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"NESTLER feature CSV has no rows: {path}")
    required = {"job_id", "site", "frame_index", "timestamp_seconds", *BASELINE_FEATURES}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"NESTLER feature CSV is missing columns: {sorted(missing)}")
    return rows


def select_rows(
    rows: list[dict[str, Any]],
    *,
    job_ids: list[str] | None = None,
    max_frames: int | None = None,
) -> list[dict[str, Any]]:
    available_jobs = {str(row["job_id"]) for row in rows}
    if job_ids:
        missing = set(job_ids) - available_jobs
        if missing:
            raise ValueError(f"Unknown job IDs: {sorted(missing)}")
        selected_jobs = set(job_ids)
        rows = [row for row in rows if row["job_id"] in selected_jobs]
    rows = sorted(rows, key=lambda row: (str(row["job_id"]), int(row["frame_index"])))
    if max_frames is None:
        return rows
    counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    for row in rows:
        job_id = str(row["job_id"])
        if counts.get(job_id, 0) >= max_frames:
            continue
        selected.append(row)
        counts[job_id] = counts.get(job_id, 0) + 1
    return selected


def build_camera_baselines(
    rows: list[dict[str, Any]],
    *,
    calibration_frames: int,
    minimum_samples: int,
    minimum_scale_ratio: float,
    optional_features: set[str] | None = None,
) -> dict[str, dict[str, RobustBaseline | None]]:
    if calibration_frames < minimum_samples:
        raise ValueError("calibration_frames must be at least minimum_samples")
    optional_features = optional_features or set()
    unknown_optional = optional_features - set(BASELINE_FEATURES)
    if unknown_optional:
        raise ValueError(f"Unknown optional baseline features: {sorted(unknown_optional)}")
    baselines: dict[str, dict[str, RobustBaseline | None]] = {}
    for job_id in sorted({str(row["job_id"]) for row in rows}):
        job_rows = [row for row in rows if row["job_id"] == job_id][:calibration_frames]
        feature_baselines: dict[str, RobustBaseline | None] = {}
        for feature in BASELINE_FEATURES:
            values = [
                value
                for row in job_rows
                if (value := optional_float(row, feature)) is not None
            ]
            if len(values) < minimum_samples:
                if feature in optional_features:
                    feature_baselines[feature] = None
                    continue
                raise ValueError(
                    f"Insufficient calibration values for {job_id}/{feature}: "
                    f"{len(values)} < {minimum_samples}"
                )
            value_array = np.asarray(values, dtype=np.float64)
            median = float(np.median(value_array))
            q25 = float(np.percentile(value_array, 25))
            q75 = float(np.percentile(value_array, 75))
            iqr = q75 - q25
            effective_iqr = max(iqr, abs(median) * minimum_scale_ratio, 1e-6)
            feature_baselines[feature] = RobustBaseline(
                valid_count=len(values),
                median=median,
                q25=q25,
                q75=q75,
                iqr=iqr,
                effective_iqr=effective_iqr,
            )
        baselines[job_id] = feature_baselines
    return baselines


def normalized_change(
    row: dict[str, Any],
    feature: str,
    baseline: RobustBaseline | None,
    *,
    direction: float = 1.0,
    saturation_iqr: float,
) -> tuple[float, bool]:
    value = optional_float(row, feature)
    if value is None or baseline is None:
        return 0.0, False
    return (
        baseline_delta(
            value,
            median=baseline.median,
            interquartile_range=baseline.effective_iqr,
            direction=direction,
            saturation_iqr=saturation_iqr,
        ),
        True,
    )


def row_to_evidence(
    row: dict[str, Any],
    baselines: dict[str, RobustBaseline | None],
    *,
    saturation_iqr: float,
) -> tuple[RiskEvidence, dict[str, bool]]:
    density, density_ok = normalized_change(
        row, "bbox_area_ratio", baselines["bbox_area_ratio"], saturation_iqr=saturation_iqr
    )
    object_excess, object_ok = normalized_change(
        row, "object_count", baselines["object_count"], saturation_iqr=saturation_iqr
    )
    motion_candidates = [
        normalized_change(
            row,
            feature,
            baselines[feature],
            saturation_iqr=saturation_iqr,
        )
        for feature in ("tracked_speed_per_second", "flow_speed_per_second")
    ]
    valid_motion = [value for value, available in motion_candidates if available]
    movement = max(valid_motion, default=0.0)
    if object_ok and valid_motion:
        inflow = (object_excess + movement) / 2
    elif object_ok:
        inflow = object_excess
    else:
        inflow = movement
    inflow_ok = object_ok or bool(valid_motion)
    proximity, proximity_ok = normalized_change(
        row,
        "mean_nearest_neighbor_distance",
        baselines["mean_nearest_neighbor_distance"],
        direction=-1.0,
        saturation_iqr=saturation_iqr,
    )
    convergence, convergence_ok = normalized_change(
        row,
        "flow_convergence_per_second",
        baselines["flow_convergence_per_second"],
        saturation_iqr=saturation_iqr,
    )
    coherence_candidates = [
        normalized_change(row, feature, baselines[feature], saturation_iqr=saturation_iqr)
        for feature in ("tracked_coherence", "flow_coherence")
    ]
    valid_coherence = [value for value, available in coherence_candidates if available]
    directional_coherence = max(valid_coherence, default=0.0)
    coherence_ok = bool(valid_coherence)
    corner, corner_ok = normalized_change(
        row, "max_grid_fraction", baselines["max_grid_fraction"], saturation_iqr=saturation_iqr
    )
    availability = {
        "density": density_ok,
        "inflow": inflow_ok,
        "proximity": proximity_ok,
        "convergence": convergence_ok,
        "directional_coherence": coherence_ok,
        "corner": corner_ok,
    }
    return (
        RiskEvidence(
            density=density,
            inflow=inflow,
            proximity=proximity,
            convergence=convergence,
            directional_coherence=directional_coherence,
            corner=corner,
            context=0.0,
        ),
        availability,
    )


def score_feature_rows(
    rows: list[dict[str, Any]],
    baselines: dict[str, dict[str, RobustBaseline | None]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    calibration_frames = int(config["calibration"]["frames"])
    saturation_iqr = float(config["calibration"]["saturation_iqr"])
    minimum_valid = int(config["quality"]["minimum_valid_evidence"])
    alpha = float(config["risk"]["smoothing_alpha"])
    thresholds = {
        "watch": float(config["risk"]["watch_threshold"]),
        "warning": float(config["risk"]["warning_threshold"]),
        "critical": float(config["risk"]["critical_threshold"]),
    }
    output_rows: list[dict[str, Any]] = []
    for job_id in sorted(baselines):
        job_rows = [row for row in rows if row["job_id"] == job_id]
        smoothed_risk: float | None = None
        alert_state = "normal"
        for position, row in enumerate(job_rows):
            evidence, availability = row_to_evidence(
                row,
                baselines[job_id],
                saturation_iqr=saturation_iqr,
            )
            valid_count = sum(availability.values())
            evidence_available = valid_count >= minimum_valid
            phase = "calibration" if position < calibration_frames else "monitoring"
            scores = compute_mechanism_scores(evidence) if evidence_available else None
            raw_risk = risk_index(scores) if scores is not None else None
            if phase == "calibration":
                row_alert_state = "calibrating"
            elif raw_risk is None:
                row_alert_state = "unavailable"
            else:
                smoothed_risk = (
                    raw_risk
                    if smoothed_risk is None
                    else alpha * raw_risk + (1 - alpha) * smoothed_risk
                )
                alert_state = next_alert_state(
                    alert_state,
                    smoothed_risk,
                    thresholds=thresholds,
                    release_margin=float(config["risk"]["release_margin"]),
                )
                row_alert_state = alert_state
            output_rows.append(
                {
                    "job_id": job_id,
                    "site": row["site"],
                    "frame_index": int(row["frame_index"]),
                    "timestamp_seconds": float(row["timestamp_seconds"]),
                    "phase": phase,
                    **asdict(evidence),
                    "valid_evidence_count": valid_count,
                    "missing_evidence": ";".join(
                        name for name in MONITORED_EVIDENCE if not availability[name]
                    ),
                    "score_social_attraction": (
                        scores.social_attraction if scores is not None else None
                    ),
                    "score_group_convergence": (
                        scores.group_convergence if scores is not None else None
                    ),
                    "score_external_context": (
                        scores.external_context if scores is not None else None
                    ),
                    "risk_raw": raw_risk,
                    "risk_smoothed": (
                        smoothed_risk
                        if phase == "monitoring" and raw_risk is not None
                        else None
                    ),
                    "alert_state": row_alert_state,
                }
            )
    return output_rows


def summarize_results(
    rows: list[dict[str, Any]], baselines: dict[str, dict[str, RobustBaseline | None]]
) -> dict[str, Any]:
    jobs: dict[str, Any] = {}
    for job_id in sorted(baselines):
        job_rows = [row for row in rows if row["job_id"] == job_id]
        monitoring = [row for row in job_rows if row["phase"] == "monitoring"]
        available = [row for row in monitoring if row["risk_smoothed"] is not None]
        first_alert_seconds = {
            state: next(
                (
                    row["timestamp_seconds"]
                    for row in available
                    if row["alert_state"] == state
                ),
                None,
            )
            for state in ("watch", "warning", "critical")
        }
        jobs[job_id] = {
            "site": job_rows[0]["site"],
            "frame_count": len(job_rows),
            "calibration_frame_count": len(job_rows) - len(monitoring),
            "monitoring_frame_count": len(monitoring),
            "available_monitoring_frames": len(available),
            "monitoring_coverage": len(available) / len(monitoring) if monitoring else 0.0,
            "maximum_risk_index": (
                max(row["risk_smoothed"] for row in available) if available else None
            ),
            "first_alert_seconds": first_alert_seconds,
            "alert_frame_counts": {
                state: sum(row["alert_state"] == state for row in monitoring)
                for state in ("normal", "watch", "warning", "critical", "unavailable")
            },
        }
    return {
        "disclaimer": (
            "Camera-relative anomaly demonstration only. NESTLER has no piling-event labels, "
            "so alerts are not validated incident detections or calibrated probabilities."
        ),
        "event_labels_available": False,
        "risk_index_is_probability": False,
        "feature_mapping": FEATURE_MAPPING,
        "jobs": jobs,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_risk_timelines(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    output_path: Path,
    *,
    figure_title: str = "PileGuard integrated demo — camera-relative NESTLER risk timelines",
) -> None:
    from matplotlib import pyplot as plt

    job_ids = sorted({row["job_id"] for row in rows})
    column_count = min(2, len(job_ids))
    row_count = math.ceil(len(job_ids) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(7 * column_count, 3.7 * row_count),
        sharey=True,
        squeeze=False,
    )
    thresholds = [
        ("watch_threshold", "Watch", "#f9a825"),
        ("warning_threshold", "Warning", "#ef6c00"),
        ("critical_threshold", "Critical", "#c62828"),
    ]
    for axis, job_id in zip(axes.flat, job_ids, strict=False):
        job_rows = [row for row in rows if row["job_id"] == job_id]
        times = [row["timestamp_seconds"] for row in job_rows]
        risks = [
            np.nan if row["risk_smoothed"] is None else row["risk_smoothed"] for row in job_rows
        ]
        calibration_rows = [row for row in job_rows if row["phase"] == "calibration"]
        if calibration_rows:
            axis.axvspan(
                times[0],
                calibration_rows[-1]["timestamp_seconds"],
                color="#90a4ae",
                alpha=0.18,
                label="Calibration" if job_id == job_ids[0] else None,
            )
        axis.plot(times, risks, color="#1565c0", linewidth=1.5)
        for threshold_name, label, color in thresholds:
            axis.axhline(
                float(config["risk"][threshold_name]),
                color=color,
                linestyle="--",
                linewidth=1,
                label=label if job_id == job_ids[0] else None,
            )
        axis.set(
            title=f"{job_id} · {job_rows[0]['site']}",
            xlabel="Seconds",
            ylabel="Risk index",
            ylim=(0, 100),
        )
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(job_ids) :]:
        axis.set_visible(False)
    axes.flat[0].legend(ncols=2, fontsize=8, loc="upper left")
    figure.suptitle(figure_title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def serialize_baselines(
    baselines: dict[str, dict[str, RobustBaseline | None]],
) -> dict[str, dict[str, dict[str, float | int] | None]]:
    return {
        job_id: {
            feature: asdict(baseline) if baseline is not None else None
            for feature, baseline in feature_values.items()
        }
        for job_id, feature_values in baselines.items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/integrated_demo.toml"))
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--data-root")
    parser.add_argument("--feature-output-dir", type=Path)
    parser.add_argument("--jobs", nargs="*")
    parser.add_argument("--max-frames", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.extract:
        feature_output_dir = args.feature_output_dir or Path(
            config["input"]["fresh_feature_dir"]
        )
        feature_csv = run_extraction(
            config_path=Path(config["input"]["nestler_config"]),
            data_root_argument=args.data_root,
            output_dir=feature_output_dir,
            max_frames=args.max_frames,
            job_ids=args.jobs,
        )
        rows = load_feature_rows(feature_csv)
    else:
        feature_csv = args.features or Path(config["input"]["feature_csv"])
        rows = select_rows(
            load_feature_rows(feature_csv),
            job_ids=args.jobs,
            max_frames=args.max_frames,
        )
    calibration = config["calibration"]
    baselines = build_camera_baselines(
        rows,
        calibration_frames=int(calibration["frames"]),
        minimum_samples=int(calibration["minimum_samples"]),
        minimum_scale_ratio=float(calibration["minimum_scale_ratio"]),
    )
    risk_rows = score_feature_rows(rows, baselines, config)
    output_dir = args.output_dir or Path(config["output"]["artifact_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(risk_rows, output_dir / "risk_timeseries.csv")
    (output_dir / "camera_baselines.json").write_text(
        json.dumps(serialize_baselines(baselines), indent=2), encoding="utf-8"
    )
    summary = summarize_results(risk_rows, baselines)
    summary["source_feature_csv"] = str(feature_csv)
    summary["config"] = {
        "calibration": config["calibration"],
        "quality": config["quality"],
        "risk": config["risk"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_risk_timelines(risk_rows, config, output_dir / "risk_timelines.png")
    coverage = sum(
        job["available_monitoring_frames"] for job in summary["jobs"].values()
    ) / max(sum(job["monitoring_frame_count"] for job in summary["jobs"].values()), 1)
    print(
        f"saved jobs={len(baselines)} frames={len(risk_rows)} "
        f"monitoring_coverage={coverage:.3f} output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
