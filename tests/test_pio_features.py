import tempfile
import unittest
from pathlib import Path

from pileguard.features.pio import (
    NormalizedBox,
    compute_static_features,
    discover_samples,
    parse_image_metadata,
    parse_yolo_label,
    summarize_features,
)


class PioFeatureTest(unittest.TestCase):
    def test_parse_yolo_label_audits_degenerate_and_clipped_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text(
                "0 0.5 0.5 0.2 0.4\n"
                "0 0.01 0.5 0.1 0.2\n"
                "0 0.2 0.2 0.0 0.1\n",
                encoding="utf-8",
            )

            result = parse_yolo_label(path)

            self.assertEqual(result.source_rows, 3)
            self.assertEqual(len(result.boxes), 2)
            self.assertEqual(result.invalid_box_count, 1)
            self.assertEqual(result.clipped_box_count, 1)

    def test_parse_yolo_label_rejects_unexpected_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unexpected class"):
                parse_yolo_label(path)

    def test_compute_static_features_uses_clipped_area_and_grid_density(self) -> None:
        boxes = (
            NormalizedBox(0, 0.01, 0.10, 0.10, 0.20),
            NormalizedBox(0, 0.20, 0.10, 0.10, 0.20),
        )

        features = compute_static_features(boxes, grid_rows=2, grid_columns=2)

        self.assertEqual(features["object_count"], 2)
        self.assertAlmostEqual(float(features["bbox_area_ratio"]), 0.032)
        self.assertEqual(features["max_grid_count"], 2)
        self.assertEqual(features["max_grid_fraction"], 1.0)
        self.assertAlmostEqual(float(features["mean_nearest_neighbor_distance"]), 0.19)

    def test_discover_samples_requires_complete_image_label_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "train").mkdir(parents=True)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "images" / "train" / "C-W1-0001.jpg").write_bytes(b"image")

            with self.assertRaisesRegex(ValueError, "Unpaired PIO files"):
                discover_samples(root, ["train"])

    def test_metadata_parser_preserves_unknown_legacy_ids(self) -> None:
        self.assertEqual(parse_image_metadata("C-W6-V0038"), ("commercial", 6))
        self.assertEqual(parse_image_metadata("P-W2-0042"), ("prototype", 2))
        self.assertEqual(parse_image_metadata("K-505"), ("unknown", None))

    def test_summary_keeps_piling_claim_boundary(self) -> None:
        row = {
            "image_id": "C-W1-0001",
            "split": "train",
            "environment": "commercial",
            "week": 1,
            "source_annotation_count": 2,
            "invalid_annotation_count": 0,
            "clipped_annotation_count": 0,
            "object_count": 2,
            "bbox_area_ratio": 0.1,
            "center_spread": 0.2,
            "mean_nearest_neighbor_distance": 0.3,
            "max_grid_fraction": 0.5,
        }

        summary = summarize_features([row], top_count=1)

        self.assertEqual(summary["valid_boxes"], 2)
        self.assertIn("no Piling event labels", summary["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
