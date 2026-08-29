"""Audit threshold stability and image-level errors for the fine-tuned AI Hub detector."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

DEFAULT_SUMMARY = Path("artifacts/aihub-laying-hen-yolo26n/validation/summary.json")
DEFAULT_IMAGE_METRICS = Path(
    "artifacts/aihub-laying-hen-yolo26n/validation/image_metrics.csv"
)
DEFAULT_OUTPUT = Path("artifacts/aihub-laying-hen-yolo26n/error-audit")
STAGE_ORDER = ("early", "middle", "late")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def load_image_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError("Image metrics CSV is empty")
    return rows


def number(row: dict[str, Any], key: str) -> float:
    value = row[key]
    if value in (None, ""):
        return 0.0
    return float(value)


def center_f1(*, true_positives: float, false_positives: float, false_negatives: float) -> float:
    denominator = (2 * true_positives) + false_positives + false_negatives
    return (2 * true_positives / denominator) if denominator else 0.0


def summarize_error_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No image rows to summarize")
    errors = np.asarray([number(row, "count_error") for row in rows], dtype=np.float64)
    absolute_errors = np.abs(errors)
    true_positives = sum(number(row, "center_true_positives") for row in rows)
    false_positives = sum(number(row, "center_false_positives") for row in rows)
    false_negatives = sum(number(row, "center_false_negatives") for row in rows)
    undercount = int(np.sum(errors < 0))
    exact = int(np.sum(errors == 0))
    overcount = int(np.sum(errors > 0))
    image_count = len(rows)
    return {
        "image_count": image_count,
        "reference_boxes": int(sum(number(row, "reference_count") for row in rows)),
        "predicted_boxes": int(sum(number(row, "predicted_count") for row in rows)),
        "count_error": {
            "mae": float(np.mean(absolute_errors)),
            "median_absolute_error": float(np.median(absolute_errors)),
            "p90_absolute_error": float(np.quantile(absolute_errors, 0.90)),
            "p95_absolute_error": float(np.quantile(absolute_errors, 0.95)),
            "p99_absolute_error": float(np.quantile(absolute_errors, 0.99)),
            "bias": float(np.mean(errors)),
            "undercount_images": undercount,
            "undercount_fraction": undercount / image_count,
            "exact_count_images": exact,
            "exact_count_fraction": exact / image_count,
            "overcount_images": overcount,
            "overcount_fraction": overcount / image_count,
        },
        "center_matching": {
            "true_positives": int(true_positives),
            "false_positives": int(false_positives),
            "false_negatives": int(false_negatives),
            "f1": center_f1(
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
            ),
        },
    }


def threshold_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold_text, result in summary["threshold_results"].items():
        threshold = float(threshold_text)
        stages = result.get("stages")
        if not stages:
            raise ValueError(
                "Threshold results do not contain stage metrics; rerun the fine-tuned evaluation"
            )
        for stage in STAGE_ORDER:
            localization = stages[stage]["localization"]
            center = localization["matching_metrics"]["center"]
            rows.append(
                {
                    "confidence_threshold": threshold,
                    "stage": stage,
                    "precision": center["precision"],
                    "recall": center["recall"],
                    "f1": center["f1"],
                    "count_mae": localization["count_mae"],
                    "count_bias": localization["count_bias"],
                }
            )
    stage_index = {stage: index for index, stage in enumerate(STAGE_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            float(row["confidence_threshold"]),
            stage_index[str(row["stage"])],
        ),
    )


def global_threshold_tradeoff(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows = []
    for threshold_text, result in summary["threshold_results"].items():
        localization = result["localization"]
        rows.append(
            {
                "confidence_threshold": float(threshold_text),
                "center_f1": float(localization["matching_metrics"]["center"]["f1"]),
                "count_mae": float(localization["count_mae"]),
                "count_bias": float(localization["count_bias"]),
            }
        )
    localization_best = max(
        rows, key=lambda row: (row["center_f1"], -row["count_mae"])
    )
    count_best = min(rows, key=lambda row: (row["count_mae"], -row["center_f1"]))
    return {
        "localization_f1_best": localization_best,
        "count_mae_best": count_best,
    }


def best_thresholds_by_stage(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["stage"])].append(row)
    return {
        stage: dict(
            max(
                grouped[stage],
                key=lambda row: (float(row["f1"]), -float(row["count_mae"])),
            )
        )
        for stage in STAGE_ORDER
    }


def worst_cases(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            number(row, "absolute_count_error"),
            number(row, "center_false_negatives") + number(row, "center_false_positives"),
        ),
        reverse=True,
    )
    result: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked[:limit], start=1):
        count_error = int(number(row, "count_error"))
        result.append(
            {
                "rank": rank,
                "image_id": row["image_id"],
                "stage": row["stage"],
                "reference_count": int(number(row, "reference_count")),
                "predicted_count": int(number(row, "predicted_count")),
                "count_error": count_error,
                "absolute_count_error": int(number(row, "absolute_count_error")),
                "error_direction": (
                    "undercount" if count_error < 0 else "overcount" if count_error > 0 else "exact"
                ),
                "center_false_positives": int(number(row, "center_false_positives")),
                "center_false_negatives": int(number(row, "center_false_negatives")),
            }
        )
    return result


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_plot(
    thresholds: list[dict[str, Any]], stage_errors: dict[str, dict[str, Any]], path: Path
) -> None:
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for stage in STAGE_ORDER:
        stage_rows = [row for row in thresholds if row["stage"] == stage]
        axes[0].plot(
            [row["confidence_threshold"] for row in stage_rows],
            [row["f1"] for row in stage_rows],
            marker="o",
            label=stage,
        )
    axes[0].set(
        title="Center-match F1 sensitivity",
        xlabel="Confidence threshold",
        ylabel="F1",
        ylim=(0, 1),
    )
    axes[0].legend()

    percentiles = ("median_absolute_error", "p90_absolute_error", "p95_absolute_error")
    labels = ("Median", "P90", "P95")
    x = np.arange(len(STAGE_ORDER))
    width = 0.22
    for index, (key, label) in enumerate(zip(percentiles, labels, strict=True)):
        axes[1].bar(
            x + (index - 1) * width,
            [stage_errors[stage]["count_error"][key] for stage in STAGE_ORDER],
            width,
            label=label,
        )
    axes[1].set(
        title="Absolute count-error distribution",
        xlabel="Laying stage",
        ylabel="Chickens per image",
        xticks=x,
        xticklabels=STAGE_ORDER,
    )
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("AI Hub fine-tuned detector error audit")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_audit(
    *, summary_path: Path, image_metrics_path: Path, output_dir: Path, worst_limit: int = 30
) -> Path:
    if worst_limit < 1:
        raise ValueError("worst_limit must be positive")
    summary = load_json(summary_path)
    image_rows = load_image_rows(image_metrics_path)
    thresholds = threshold_rows(summary)
    selected_threshold = float(summary["selected_confidence_threshold"])
    csv_threshold = float(image_rows[0]["confidence_threshold"])
    if not np.isclose(selected_threshold, csv_threshold):
        raise ValueError("Summary threshold and image metrics threshold differ")
    if any(not np.isclose(float(row["confidence_threshold"]), csv_threshold) for row in image_rows):
        raise ValueError("Image metrics CSV mixes confidence thresholds")

    stage_errors = {
        stage: summarize_error_rows([row for row in image_rows if row["stage"] == stage])
        for stage in STAGE_ORDER
    }
    overall_errors = summarize_error_rows(image_rows)
    stage_best = best_thresholds_by_stage(thresholds)
    tradeoff = global_threshold_tradeoff(summary)
    weakest_stage = min(
        STAGE_ORDER, key=lambda stage: float(stage_errors[stage]["center_matching"]["f1"])
    )
    cases = worst_cases(image_rows, limit=worst_limit)
    minimum_stage_f1 = min(
        float(stage_errors[stage]["center_matching"]["f1"]) for stage in STAGE_ORDER
    )
    gate_minimum = float(summary["monitoring_integration_gate"]["minimum_center_f1"])
    audit = {
        "source_detector": summary["source_detector"],
        "evaluated_images": len(image_rows),
        "selected_confidence_threshold": selected_threshold,
        "thresholds_evaluated": sorted(
            {float(row["confidence_threshold"]) for row in thresholds}
        ),
        "overall_error_distribution": overall_errors,
        "stage_error_distribution": stage_errors,
        "stage_diagnostic_best_thresholds": stage_best,
        "global_threshold_tradeoff": tradeoff,
        "weakest_stage_at_selected_threshold": weakest_stage,
        "worst_case_count": len(cases),
        "operating_recommendation": {
            "mode": "single global threshold",
            "confidence_threshold": selected_threshold,
            "minimum_stage_center_f1": minimum_stage_f1,
            "all_stages_pass_configured_gate": minimum_stage_f1 >= gate_minimum,
            "raw_count_calibration_required": True,
            "reason": (
                "Use the aggregate center-F1 optimum for localization monitoring. Do not treat "
                "raw predicted count as a calibrated density estimate because count bias changes "
                "by laying stage. Stage-specific thresholds are diagnostic only because they were "
                "selected on the same Validation split."
            ),
        },
        "claim_boundary": (
            "This audit reuses the same official AI Hub Validation split used for checkpoint and "
            "threshold selection. It diagnoses static laying-hen localization errors; it is not "
            "an independent farm test or a piling/smothering incident evaluation."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_output = output_dir / "summary.json"
    summary_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(thresholds, output_dir / "threshold_by_stage.csv")
    write_csv(cases, output_dir / "worst_count_errors.csv")
    save_plot(thresholds, stage_errors, output_dir / "error_audit.png")
    print(
        f"saved images={len(image_rows)} threshold={selected_threshold:.2f} "
        f"weakest_stage={weakest_stage} output={output_dir}"
    )
    return summary_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--image-metrics", type=Path, default=DEFAULT_IMAGE_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worst-limit", type=int, default=30)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_audit(
        summary_path=args.summary,
        image_metrics_path=args.image_metrics,
        output_dir=args.output_dir,
        worst_limit=args.worst_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
