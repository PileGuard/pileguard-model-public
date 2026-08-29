import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pileguard.data.pio_detection import (
    discover_split_images,
    select_smoke_images,
    write_runtime_dataset,
)
from pileguard.evaluation.pio_yolo import build_evaluation_groups, copy_evidence_plots
from pileguard.models.yolo import normalize_detection_metrics


class PioDetectionTest(unittest.TestCase):
    def create_split(self, root: Path, split: str, count: int) -> None:
        image_root = root / "images" / split
        label_root = root / "labels" / split
        image_root.mkdir(parents=True)
        label_root.mkdir(parents=True)
        for index in range(count):
            stem = f"C-W1-{index:04d}"
            (image_root / f"{stem}.jpg").write_bytes(b"image")
            (label_root / f"{stem}.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
            )

    def test_discover_split_images_rejects_missing_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "train").mkdir(parents=True)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "images" / "train" / "sample.jpg").write_bytes(b"image")

            with self.assertRaisesRegex(ValueError, "without labels"):
                discover_split_images(root, "train")

    def test_smoke_selection_is_deterministic(self) -> None:
        images = [Path(f"{index}.jpg") for index in range(20)]

        first = select_smoke_images(images, limit=4, seed=2026)
        second = select_smoke_images(images, limit=4, seed=2026)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)

    def test_runtime_dataset_uses_ignored_manifests_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pio"
            output = Path(directory) / "runtime"
            self.create_split(root, "train", 6)
            self.create_split(root, "val", 5)

            yaml_path, counts = write_runtime_dataset(
                dataset_root=root,
                output_dir=output,
                train_split="train",
                validation_split="val",
                class_names=["chicken"],
                smoke_limit=3,
            )

            self.assertEqual(counts, {"train_images": 3, "validation_images": 3})
            self.assertEqual(len((output / "train-images.txt").read_text().splitlines()), 3)
            yaml_text = yaml_path.read_text(encoding="utf-8")
            self.assertIn("nc: 1", yaml_text)
            self.assertIn('names: ["chicken"]', yaml_text)

    def test_normalize_detection_metrics_uses_stable_names(self) -> None:
        results = SimpleNamespace(
            results_dict={
                "metrics/precision(B)": 0.91,
                "metrics/recall(B)": 0.82,
                "metrics/mAP50(B)": 0.88,
                "metrics/mAP50-95(B)": 0.57,
                "fitness": 0.60,
            }
        )

        metrics = normalize_detection_metrics(results)

        self.assertEqual(
            metrics,
            {
                "precision": 0.91,
                "recall": 0.82,
                "map50": 0.88,
                "map50_95": 0.57,
                "fitness": 0.60,
            },
        )

    def test_evaluation_groups_preserve_environment_and_week(self) -> None:
        images = [
            Path("C-W1-0001.jpg"),
            Path("C-W2-0002.jpg"),
            Path("P-W1-0003.jpg"),
            Path("K-505.jpg"),
        ]

        groups = build_evaluation_groups(images)

        self.assertEqual(len(groups["environment-commercial"]), 2)
        self.assertEqual(len(groups["environment-prototype"]), 1)
        self.assertEqual(len(groups["environment-unknown"]), 1)
        self.assertEqual(len(groups["week-1"]), 2)
        self.assertEqual(len(groups["week-2"]), 1)

    def test_copy_evidence_plots_only_copies_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            artifact_dir = root / "artifacts"
            run_dir.mkdir()
            (run_dir / "PR_curve.png").write_bytes(b"plot")

            copied = copy_evidence_plots(run_dir, artifact_dir)

            self.assertEqual(copied, ["PR_curve.png"])
            self.assertEqual((artifact_dir / "PR_curve.png").read_bytes(), b"plot")


if __name__ == "__main__":
    unittest.main()
