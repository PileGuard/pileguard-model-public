"""Generate three literature-derived piling precursor demonstrations."""

from __future__ import annotations

import argparse
import csv
import json
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from pileguard.models.risk import (
    RiskEvidence,
    compute_mechanism_scores,
    next_alert_state,
    risk_index,
)

matplotlib.use("Agg")

EVIDENCE_FIELDS = [
    "density",
    "inflow",
    "proximity",
    "convergence",
    "directional_coherence",
    "corner",
    "context",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def smoothstep(times: np.ndarray, start: float, end: float) -> np.ndarray:
    if end <= start:
        raise ValueError("smoothstep end must be greater than start")
    phase = np.clip((times - start) / (end - start), 0.0, 1.0)
    return phase * phase * (3 - 2 * phase)


def add_noise(values: np.ndarray, rng: np.random.Generator, scale: float = 0.012) -> np.ndarray:
    return np.clip(values + rng.normal(0, scale, size=values.shape), 0.0, 1.0)


def generate_evidence(
    scenario: str, times: np.ndarray, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    base = 0.06
    if scenario == "social_attraction":
        values = {
            "density": base + 0.90 * smoothstep(times, 40, 145),
            "inflow": base + 0.90 * smoothstep(times, 25, 105),
            "proximity": base + 0.88 * smoothstep(times, 50, 135),
            "convergence": base + 0.18 * smoothstep(times, 70, 140),
            "directional_coherence": 0.12 + 0.18 * smoothstep(times, 45, 115),
            "corner": 0.20 + 0.55 * smoothstep(times, 35, 110),
            "context": np.zeros_like(times),
        }
    elif scenario == "group_convergence":
        values = {
            "density": base + 0.78 * smoothstep(times, 55, 140),
            "inflow": base + 0.52 * smoothstep(times, 40, 120),
            "proximity": base + 0.45 * smoothstep(times, 75, 145),
            "convergence": base + 0.92 * smoothstep(times, 25, 110),
            "directional_coherence": 0.12 + 0.85 * smoothstep(times, 20, 95),
            "corner": 0.10 + 0.18 * smoothstep(times, 80, 145),
            "context": np.zeros_like(times),
        }
    elif scenario == "external_context":
        values = {
            "density": base + 0.80 * smoothstep(times, 70, 150),
            "inflow": base + 0.82 * smoothstep(times, 45, 125),
            "proximity": base + 0.68 * smoothstep(times, 80, 150),
            "convergence": base + 0.25 * smoothstep(times, 65, 135),
            "directional_coherence": 0.14 + 0.38 * smoothstep(times, 45, 115),
            "corner": 0.12 + 0.86 * smoothstep(times, 20, 80),
            "context": (times >= 30).astype(np.float64),
        }
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return {name: add_noise(value, rng) for name, value in values.items()}


def simulate_scenario(
    scenario: str,
    *,
    times: np.ndarray,
    risk_config: dict[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    evidence_values = generate_evidence(scenario, times, rng)
    thresholds = {
        "watch": float(risk_config["watch_threshold"]),
        "warning": float(risk_config["warning_threshold"]),
        "critical": float(risk_config["critical_threshold"]),
    }
    alpha = float(risk_config["smoothing_alpha"])
    smoothed_risk: float | None = None
    alert_state = "normal"
    rows: list[dict[str, Any]] = []
    for index, timestamp in enumerate(times):
        evidence = RiskEvidence(
            **{name: float(evidence_values[name][index]) for name in EVIDENCE_FIELDS}
        )
        scores = compute_mechanism_scores(evidence)
        raw_risk = risk_index(scores)
        smoothed_risk = (
            raw_risk if smoothed_risk is None else alpha * raw_risk + (1 - alpha) * smoothed_risk
        )
        alert_state = next_alert_state(
            alert_state,
            smoothed_risk,
            thresholds=thresholds,
            release_margin=float(risk_config["release_margin"]),
        )
        row = {
            "scenario": scenario,
            "timestamp_seconds": float(timestamp),
            **asdict(evidence),
            **{f"score_{name}": value for name, value in asdict(scores).items()},
            "risk_raw": raw_risk,
            "risk_smoothed": smoothed_risk,
            "alert_state": alert_state,
        }
        rows.append(row)
    return rows


def load_camera_baselines(path: Path) -> dict[str, dict[str, dict[str, float | int | None]]]:
    if not path.is_file():
        raise FileNotFoundError(f"NESTLER feature CSV not found: {path}")
    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    baseline_features = [
        "bbox_area_ratio",
        "mean_nearest_neighbor_distance",
        "tracked_speed_per_second",
        "flow_speed_per_second",
        "flow_divergence_per_second",
    ]
    baselines: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for job_id in sorted({row["job_id"] for row in rows}):
        job_rows = [row for row in rows if row["job_id"] == job_id]
        feature_baselines: dict[str, dict[str, float | int | None]] = {}
        for feature in baseline_features:
            values = np.array(
                [float(row[feature]) for row in job_rows if row.get(feature)], dtype=np.float64
            )
            feature_baselines[feature] = {
                "valid_count": int(values.size),
                "median": float(np.median(values)) if values.size else None,
                "q25": float(np.percentile(values, 25)) if values.size else None,
                "q75": float(np.percentile(values, 75)) if values.size else None,
                "iqr": (
                    float(np.percentile(values, 75) - np.percentile(values, 25))
                    if values.size
                    else None
                ),
            }
        baselines[job_id] = feature_baselines
    return baselines


def summarize_scenarios(
    rows: list[dict[str, Any]], scenario_config: dict[str, Any]
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for scenario in scenario_config:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        transition_rows = [
            row
            for index, row in enumerate(scenario_rows)
            if index == 0 or row["alert_state"] != scenario_rows[index - 1]["alert_state"]
        ]
        first_by_state = {
            state: next(
                (
                    row["timestamp_seconds"]
                    for row in scenario_rows
                    if row["alert_state"] == state
                ),
                None,
            )
            for state in ("watch", "warning", "critical")
        }
        summaries[scenario] = {
            **scenario_config[scenario],
            "maximum_risk_index": max(row["risk_smoothed"] for row in scenario_rows),
            "first_alert_seconds": first_by_state,
            "alert_transitions": [
                {
                    "timestamp_seconds": row["timestamp_seconds"],
                    "alert_state": row["alert_state"],
                    "risk_index": row["risk_smoothed"],
                }
                for row in transition_rows
            ],
        }
    return summaries


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_risk_plot(
    rows: list[dict[str, Any]], config: dict[str, Any], output_path: Path
) -> None:
    from matplotlib import pyplot as plt

    scenarios = list(config["scenarios"])
    figure, axes = plt.subplots(len(scenarios), 1, figsize=(11, 9), sharex=True)
    threshold_lines = [
        ("watch_threshold", "Watch", "#f9a825"),
        ("warning_threshold", "Warning", "#ef6c00"),
        ("critical_threshold", "Critical", "#c62828"),
    ]
    for axis, scenario in zip(axes, scenarios, strict=True):
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        axis.plot(
            [row["timestamp_seconds"] for row in scenario_rows],
            [row["risk_smoothed"] for row in scenario_rows],
            color="#1565c0",
            linewidth=2.2,
        )
        for threshold_name, label, color in threshold_lines:
            axis.axhline(
                float(config["risk"][threshold_name]),
                color=color,
                linestyle="--",
                linewidth=1,
                label=label if scenario == scenarios[0] else None,
            )
        axis.set(
            title=config["scenarios"][scenario]["display_name"],
            ylabel="Risk index",
            ylim=(0, 100),
        )
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Synthetic seconds (assumption)")
    axes[0].legend(ncols=3, loc="upper left")
    figure.suptitle("PileGuard Digital Twin — synthetic precursor risk curves")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_evidence_plot(
    rows: list[dict[str, Any]], config: dict[str, Any], output_path: Path
) -> None:
    from matplotlib import pyplot as plt

    scenarios = list(config["scenarios"])
    figure, axes = plt.subplots(len(scenarios), 1, figsize=(11, 9), sharex=True)
    for axis, scenario in zip(axes, scenarios, strict=True):
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        for feature in EVIDENCE_FIELDS:
            axis.plot(
                [row["timestamp_seconds"] for row in scenario_rows],
                [row[feature] for row in scenario_rows],
                label=feature.replace("_", " "),
                linewidth=1.2,
            )
        axis.set(
            title=config["scenarios"][scenario]["display_name"],
            ylabel="Normalized evidence",
            ylim=(0, 1.05),
        )
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Synthetic seconds (assumption)")
    axes[0].legend(ncols=4, fontsize=8)
    figure.suptitle("Digital Twin evidence trajectories")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/digital_twin.toml"))
    parser.add_argument("--nestler-features", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    simulation = config["simulation"]
    times = np.arange(
        0,
        float(simulation["duration_seconds"]) + float(simulation["step_seconds"]),
        float(simulation["step_seconds"]),
    )
    rng = np.random.default_rng(int(simulation["seed"]))
    rows: list[dict[str, Any]] = []
    for scenario in config["scenarios"]:
        rows.extend(
            simulate_scenario(
                scenario,
                times=times,
                risk_config=config["risk"],
                rng=rng,
            )
        )

    nestler_path = args.nestler_features or Path(config["input"]["nestler_features"])
    camera_baselines = load_camera_baselines(nestler_path)
    output_dir = args.output_dir or Path(config["output"]["artifact_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "scenario_timeseries.csv")
    (output_dir / "camera_baselines.json").write_text(
        json.dumps(camera_baselines, indent=2), encoding="utf-8"
    )
    summary = {
        "disclaimer": (
            "Synthetic demonstration index only. Timing, domestic frequency, and predictive "
            "performance are assumptions and are not validated on farm incidents."
        ),
        "risk_index_is_probability": False,
        "simulation": simulation,
        "risk": config["risk"],
        "scenarios": summarize_scenarios(rows, config["scenarios"]),
    }
    (output_dir / "scenario_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_risk_plot(rows, config, output_dir / "risk_curves.png")
    save_evidence_plot(rows, config, output_dir / "evidence_curves.png")
    print(f"saved scenarios={len(config['scenarios'])} rows={len(rows)} output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
