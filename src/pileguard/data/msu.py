"""MSU Poultry Piling Dataset adapters using the official split directories."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as transform_functional

from pileguard.data_inventory import IMAGE_SUFFIXES

CLASS_TO_INDEX = {"negatives": 0, "positives": 1}
INDEX_TO_CLASS = {index: name for name, index in CLASS_TO_INDEX.items()}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class MSUSample:
    path: Path
    label: int


class CenterCropToAspect:
    """Center-crop an image to a target aspect ratio before resizing."""

    def __init__(self, aspect_ratio: float) -> None:
        if aspect_ratio <= 0:
            raise ValueError("aspect_ratio must be positive")
        self.aspect_ratio = aspect_ratio

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        current_ratio = width / height
        if current_ratio > self.aspect_ratio:
            crop_width = round(height * self.aspect_ratio)
            return transform_functional.center_crop(image, (height, crop_width))
        crop_height = round(width / self.aspect_ratio)
        return transform_functional.center_crop(image, (crop_height, width))


def build_transform(
    *, train: bool, image_height: int, image_width: int, grayscale: bool
) -> transforms.Compose:
    """Build transforms that retain the reference model's wide field of view."""

    operations: list[Callable] = [CenterCropToAspect(image_width / image_height)]
    if grayscale:
        operations.append(transforms.Grayscale(num_output_channels=3))
    operations.append(transforms.Resize((image_height, image_width), antialias=True))
    if train:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


def discover_samples(
    dataset_root: Path,
    split: str,
    *,
    max_samples_per_class: int | None = None,
    seed: int = 2026,
) -> list[MSUSample]:
    """Discover a deterministic subset while preserving official split membership."""

    split_root = dataset_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"MSU split directory not found: {split_root}")

    random_generator = random.Random(seed)
    samples: list[MSUSample] = []
    for class_name, label in CLASS_TO_INDEX.items():
        class_root = split_root / class_name
        paths = sorted(
            path
            for path in class_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise FileNotFoundError(f"No images found in {class_root}")
        if max_samples_per_class is not None and len(paths) > max_samples_per_class:
            paths = sorted(random_generator.sample(paths, max_samples_per_class))
        samples.extend(MSUSample(path=path, label=label) for path in paths)
    return samples


class MSUPilingDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Piling image dataset with explicit label mapping and source paths."""

    def __init__(self, samples: list[MSUSample], transform: Callable) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, sample.label, str(sample.path)

    @property
    def class_counts(self) -> dict[int, int]:
        counts = {index: 0 for index in INDEX_TO_CLASS}
        for sample in self.samples:
            counts[sample.label] += 1
        return counts


def seed_worker(worker_id: int) -> None:
    """Derive deterministic worker seeds from PyTorch's initial seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def build_dataloader(
    dataset: MSUPilingDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
