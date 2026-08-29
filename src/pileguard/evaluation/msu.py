"""Evaluate a trained MSU piling detector on the untouched official test split."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from pileguard.data.msu import (
    CLASS_TO_INDEX,
    MSUPilingDataset,
    build_dataloader,
    build_transform,
    discover_samples,
)
from pileguard.data_inventory import resolve_data_root
from pileguard.models.resnet import build_resnet18
from pileguard.runtime import resolve_device, seed_everything

matplotlib.use("Agg")


@dataclass(frozen=True)
class PredictionRecord:
    path: str
    true_label: int
    pile_probability: float
    predicted_label: int

    @property
    def error_type(self) -> str:
        if self.true_label == 0 and self.predicted_label == 1:
            return "false_positive"
        if self.true_label == 1 and self.predicted_label == 0:
            return "false_negative"
        return ""


def compute_binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute fixed-threshold metrics and threshold-independent ranking metrics."""

    if labels.ndim != 1 or probabilities.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must be one-dimensional arrays of equal length")
    if labels.size == 0:
        raise ValueError("at least one prediction is required")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("both binary classes must be present")

    predictions = (probabilities >= threshold).astype(np.int64)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    precision, recall, pr_thresholds = precision_recall_curve(labels, probabilities)
    metrics = {
        "threshold": threshold,
        "sample_count": int(labels.size),
        "negative_count": int((labels == 0).sum()),
        "positive_count": int((labels == 1).sum()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "pr_auc": float(auc(recall, precision)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": {
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        },
    }
    return metrics, (precision, recall, pr_thresholds)


def select_error_records(
    records: list[PredictionRecord], *, top_per_type: int
) -> tuple[list[PredictionRecord], list[PredictionRecord]]:
    """Select the most confident false positives and false negatives."""

    if top_per_type < 1:
        raise ValueError("top_per_type must be positive")
    false_positives = sorted(
        (record for record in records if record.error_type == "false_positive"),
        key=lambda record: record.pile_probability,
        reverse=True,
    )[:top_per_type]
    false_negatives = sorted(
        (record for record in records if record.error_type == "false_negative"),
        key=lambda record: record.pile_probability,
    )[:top_per_type]
    return false_positives, false_negatives


def source_identifier(path: str) -> str:
    """Extract the camera/channel prefix used by MSU filenames."""

    return Path(path).name.split("_", maxsplit=1)[0]


def compute_source_metrics(records: list[PredictionRecord]) -> dict[str, dict[str, Any]]:
    """Summarize fixed-threshold performance per MSU camera/channel."""

    grouped: dict[str, list[PredictionRecord]] = {}
    for record in records:
        grouped.setdefault(source_identifier(record.path), []).append(record)

    results: dict[str, dict[str, Any]] = {}
    for source, source_records in sorted(grouped.items()):
        labels = np.array([record.true_label for record in source_records], dtype=np.int64)
        predictions = np.array(
            [record.predicted_label for record in source_records], dtype=np.int64
        )
        true_negative, false_positive, false_negative, true_positive = confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).ravel()
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        source_precision = (
            float(true_positive / precision_denominator) if precision_denominator else None
        )
        source_recall = float(true_positive / recall_denominator) if recall_denominator else None
        source_f1 = (
            2 * source_precision * source_recall / (source_precision + source_recall)
            if source_precision is not None
            and source_recall is not None
            and source_precision + source_recall > 0
            else None
        )
        results[source] = {
            "sample_count": len(source_records),
            "negative_count": int((labels == 0).sum()),
            "positive_count": int((labels == 1).sum()),
            "accuracy": float((predictions == labels).mean()),
            "precision": source_precision,
            "recall": source_recall,
            "f1": source_f1,
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        }
    return results


def run_inference(
    *,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float,
) -> list[PredictionRecord]:
    model.eval()
    records: list[PredictionRecord] = []
    with torch.inference_mode():
        for inputs, labels, paths in tqdm(dataloader, desc="test inference", leave=False):
            probabilities = torch.softmax(model(inputs.to(device)), dim=1)[:, 1].cpu().numpy()
            for path, label, probability in zip(paths, labels.tolist(), probabilities, strict=True):
                records.append(
                    PredictionRecord(
                        path=path,
                        true_label=int(label),
                        pile_probability=float(probability),
                        predicted_label=int(probability >= threshold),
                    )
                )
    return records


def portable_path(path: str, dataset_root: Path) -> str:
    source = Path(path).resolve()
    try:
        return str(source.relative_to(dataset_root.resolve()))
    except ValueError:
        return str(source)


def write_prediction_csv(
    records: list[PredictionRecord], output_path: Path, dataset_root: Path
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "path",
                "true_label",
                "pile_probability",
                "predicted_label",
                "error_type",
            ],
        )
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["path"] = portable_path(record.path, dataset_root)
            row["error_type"] = record.error_type
            writer.writerow(row)


def save_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, label=f"PR-AUC = {metrics['pr_auc']:.4f}")
    prevalence = metrics["positive_count"] / metrics["sample_count"]
    axis.axhline(prevalence, color="gray", linestyle="--", label="Class prevalence")
    axis.set(
        title="MSU Test Precision-Recall Curve",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_confusion_matrix(metrics: dict[str, Any], output_path: Path) -> None:
    from matplotlib import pyplot as plt

    counts = metrics["confusion_matrix"]
    matrix = np.array(
        [
            [counts["true_negative"], counts["false_positive"]],
            [counts["false_negative"], counts["true_positive"]],
        ]
    )
    figure, axis = plt.subplots(figsize=(5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set(
        title=f"MSU Test Confusion Matrix (threshold={metrics['threshold']:.2f})",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Negative", "Pile"],
        yticklabels=["Negative", "Pile"],
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_error_montage(
    false_positives: list[PredictionRecord],
    false_negatives: list[PredictionRecord],
    output_path: Path,
) -> None:
    columns = 4
    tile_size = (320, 180)
    header_height = 30
    selected = [("FP", record) for record in false_positives] + [
        ("FN", record) for record in false_negatives
    ]
    if not selected:
        return
    rows = (len(selected) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * tile_size[0], rows * (tile_size[1] + header_height)),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (error_type, record) in enumerate(selected):
        column = index % columns
        row = index // columns
        x = column * tile_size[0]
        y = row * (tile_size[1] + header_height)
        with Image.open(record.path) as source:
            tile = ImageOps.fit(source.convert("RGB"), tile_size)
        canvas.paste(tile, (x, y + header_height))
        header_color = "#c62828" if error_type == "FP" else "#1565c0"
        draw.rectangle((x, y, x + tile_size[0], y + header_height), fill=header_color)
        draw.text(
            (x + 7, y + 7),
            f"{error_type}  pile={record.pile_probability:.3f}  {Path(record.path).name}",
            fill="white",
        )
    canvas.save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/msu-resnet18/best.pt"))
    parser.add_argument("--data-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/msu-resnet18/test"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-errors", type=int, default=12)
    parser.add_argument("--max-samples-per-class", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    seed_everything(2026)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["architecture"] != "resnet18":
        raise ValueError(f"Unsupported architecture: {checkpoint['architecture']}")
    if checkpoint["class_to_index"] != CLASS_TO_INDEX:
        raise ValueError("Checkpoint class mapping does not match the MSU dataset mapping")

    data_root = resolve_data_root(args.data_root)
    dataset_path = checkpoint["config"]["data"]["dataset_path"]
    dataset_root = data_root / dataset_path
    test_dataset = MSUPilingDataset(
        discover_samples(
            dataset_root,
            "test",
            max_samples_per_class=args.max_samples_per_class,
            seed=2026,
        ),
        build_transform(
            train=False,
            image_height=int(checkpoint["image_height"]),
            image_width=int(checkpoint["image_width"]),
            grayscale=bool(checkpoint["grayscale"]),
        ),
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=2026,
    )
    model = build_resnet18(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    started_at = time.monotonic()
    records = run_inference(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=args.threshold,
    )
    labels = np.array([record.true_label for record in records], dtype=np.int64)
    probabilities = np.array([record.pile_probability for record in records])
    metrics, (precision, recall, _thresholds) = compute_binary_metrics(
        labels, probabilities, threshold=args.threshold
    )
    metrics.update(
        {
            "split": "official_test",
            "device": str(device),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_best_val_f1_at_0_5": float(checkpoint["best_val_f1_at_0_5"]),
            "elapsed_seconds": time.monotonic() - started_at,
            "source_metrics": compute_source_metrics(records),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_prediction_csv(records, args.output_dir / "predictions.csv", dataset_root)
    errors = [record for record in records if record.error_type]
    write_prediction_csv(errors, args.output_dir / "errors.csv", dataset_root)
    save_pr_curve(precision, recall, metrics, args.output_dir / "pr_curve.png")
    save_confusion_matrix(metrics, args.output_dir / "confusion_matrix.png")
    false_positives, false_negatives = select_error_records(
        records, top_per_type=args.top_errors
    )
    save_error_montage(
        false_positives,
        false_negatives,
        args.output_dir / "top_errors.png",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(
        f"test={metrics['sample_count']} accuracy={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} pr_auc={metrics['pr_auc']:.4f}"
    )
    print(f"saved evaluation={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
