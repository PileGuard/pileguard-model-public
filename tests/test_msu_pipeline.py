import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from pileguard.data.msu import (
    CLASS_TO_INDEX,
    MSUPilingDataset,
    build_transform,
    discover_samples,
)
from pileguard.models.resnet import build_resnet18


class MSUPipelineTest(unittest.TestCase):
    def test_discovers_classes_with_stable_label_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_root = Path(directory)
            for class_name in CLASS_TO_INDEX:
                class_root = dataset_root / "train" / class_name
                class_root.mkdir(parents=True)
                Image.new("RGB", (64, 36), color="white").save(class_root / "sample.jpg")

            samples = discover_samples(dataset_root, "train")

            self.assertEqual(len(samples), 2)
            self.assertEqual({sample.label for sample in samples}, {0, 1})

    def test_transform_and_model_output_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_root = Path(directory)
            for class_name in CLASS_TO_INDEX:
                class_root = dataset_root / "train" / class_name
                class_root.mkdir(parents=True)
                Image.new("RGB", (192, 108), color="white").save(class_root / "sample.jpg")

            dataset = MSUPilingDataset(
                discover_samples(dataset_root, "train"),
                build_transform(train=False, image_height=64, image_width=128, grayscale=True),
            )
            image, _label, _path = dataset[0]
            model = build_resnet18(pretrained=False)

            output = model(image.unsqueeze(0))

            self.assertEqual(tuple(image.shape), (3, 64, 128))
            self.assertEqual(tuple(output.shape), (1, 2))
            self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
