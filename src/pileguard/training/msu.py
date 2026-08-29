"""Train a ResNet18 piling detector on the official MSU dataset splits."""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
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
class EpochMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def calculate_metrics(
    *,
    loss_sum: float,
    sample_count: int,
    true_positive: int,
    true_negative: int,
    false_positive: int,
    false_negative: int,
) -> EpochMetrics:
    accuracy = (true_positive + true_negative) / sample_count
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return EpochMetrics(
        loss=loss_sum / sample_count,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def run_epoch(
    *,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    description: str,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    sample_count = 0
    true_positive = true_negative = false_positive = false_negative = 0

    for inputs, labels, _paths in tqdm(dataloader, desc=description, leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        true_positive += int(((predictions == 1) & (labels == 1)).sum().item())
        true_negative += int(((predictions == 0) & (labels == 0)).sum().item())
        false_positive += int(((predictions == 1) & (labels == 0)).sum().item())
        false_negative += int(((predictions == 0) & (labels == 1)).sum().item())

    return calculate_metrics(
        loss_sum=loss_sum,
        sample_count=sample_count,
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def save_history_plot(history: list[dict[str, Any]], output_path: Path) -> None:
    from matplotlib import pyplot as plt

    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [item["train"]["loss"] for item in history], label="Train")
    axes[0].plot(epochs, [item["val"]["loss"] for item in history], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[0].legend()
    axes[1].plot(epochs, [item["train"]["f1"] for item in history], label="Train")
    axes[1].plot(epochs, [item["val"]["f1"] for item in history], label="Validation")
    axes[1].set(title="Pile F1 at 0.5", xlabel="Epoch", ylabel="F1", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/msu_resnet18.toml"))
    parser.add_argument("--data-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-samples-per-class", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one epoch on eight images per class without downloading pretrained weights.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["output"]

    seed = int(training_config["seed"])
    seed_everything(seed)
    device = resolve_device(args.device)
    data_root = resolve_data_root(args.data_root)
    dataset_root = data_root / data_config["dataset_path"]
    epochs = 1 if args.smoke else int(args.epochs or training_config["epochs"])
    sample_limit = 8 if args.smoke else args.max_samples_per_class
    pretrained = bool(model_config["pretrained"]) and not args.no_pretrained and not args.smoke
    num_workers = 0 if args.smoke else int(training_config["num_workers"])

    train_dataset = MSUPilingDataset(
        discover_samples(dataset_root, "train", max_samples_per_class=sample_limit, seed=seed),
        build_transform(
            train=True,
            image_height=int(data_config["image_height"]),
            image_width=int(data_config["image_width"]),
            grayscale=bool(data_config["grayscale"]),
        ),
    )
    val_dataset = MSUPilingDataset(
        discover_samples(dataset_root, "val", max_samples_per_class=sample_limit, seed=seed),
        build_transform(
            train=False,
            image_height=int(data_config["image_height"]),
            image_width=int(data_config["image_width"]),
            grayscale=bool(data_config["grayscale"]),
        ),
    )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )

    model = build_resnet18(pretrained=pretrained).to(device)
    train_counts = train_dataset.class_counts
    total_train = sum(train_counts.values())
    class_weights = torch.tensor(
        [total_train / (2 * train_counts[index]) for index in range(len(CLASS_TO_INDEX))],
        device=device,
        dtype=torch.float32,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = SGD(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        momentum=float(training_config["momentum"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = StepLR(
        optimizer,
        step_size=int(training_config["lr_step_size"]),
        gamma=float(training_config["lr_gamma"]),
    )

    checkpoint_dir = Path(output_config["checkpoint_dir"])
    artifact_dir = Path(output_config["artifact_dir"])
    if args.smoke:
        checkpoint_dir = checkpoint_dir / "smoke"
        artifact_dir = Path("outputs/smoke/msu-resnet18")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    best_epoch = 0
    started_at = time.monotonic()
    print(
        f"device={device} train={len(train_dataset)} val={len(val_dataset)} "
        f"epochs={epochs} pretrained={pretrained}"
    )
    for epoch in range(1, epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            description=f"train {epoch}/{epochs}",
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                description=f"val {epoch}/{epochs}",
            )
        scheduler.step()
        epoch_record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": asdict(train_metrics),
            "val": asdict(val_metrics),
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch} train_loss={train_metrics.loss:.4f} "
            f"val_loss={val_metrics.loss:.4f} val_f1={val_metrics.f1:.4f} "
            f"val_recall={val_metrics.recall:.4f}"
        )
        if val_metrics.f1 > best_val_f1:
            best_val_f1 = val_metrics.f1
            best_epoch = epoch
            torch.save(
                {
                    "format_version": 1,
                    "architecture": model_config["architecture"],
                    "class_to_index": CLASS_TO_INDEX,
                    "image_height": int(data_config["image_height"]),
                    "image_width": int(data_config["image_width"]),
                    "grayscale": bool(data_config["grayscale"]),
                    "epoch": epoch,
                    "best_val_f1_at_0_5": best_val_f1,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                },
                checkpoint_dir / "best.pt",
            )

    elapsed_seconds = time.monotonic() - started_at
    training_summary = {
        "device": str(device),
        "elapsed_seconds": elapsed_seconds,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_class_counts": train_dataset.class_counts,
        "val_class_counts": val_dataset.class_counts,
        "best_epoch": best_epoch,
        "best_val_f1_at_0_5": best_val_f1,
        "history": history,
    }
    (artifact_dir / "training_history.json").write_text(
        json.dumps(training_summary, indent=2), encoding="utf-8"
    )
    save_history_plot(history, artifact_dir / "training_curves.png")
    print(
        f"saved checkpoint={checkpoint_dir / 'best.pt'} "
        f"history={artifact_dir / 'training_history.json'} elapsed={elapsed_seconds:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
