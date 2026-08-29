import tempfile
import unittest
from pathlib import Path

import numpy as np

from pileguard.features.pio import PioSample
from pileguard.features.pio_predicted import (
    build_comparison_row,
    normalized_boxes_from_arrays,
    select_confidence_threshold,
    summarize_comparison,
)


class PioPredictedFeatureTest(unittest.TestCase):
    def test_week_aware_confidence_uses_metadata_and_fallback(self) -> None:
        week_one = PioSample("C-W1-0001", "val", Path("image.jpg"), Path("label.txt"))
        unknown = PioSample("K-505", "val", Path("image.jpg"), Path("label.txt"))

        self.assertEqual(
            select_confidence_threshold(
                week_one,
                default_threshold=0.25,
                confidence_by_week={1: 0.15},
            ),
            0.15,
        )
        self.assertEqual(
            select_confidence_threshold(
                unknown,
                default_threshold=0.25,
                confidence_by_week={1: 0.15},
            ),
            0.25,
        )

    def test_normalized_boxes_from_arrays_validates_shapes_and_classes(self) -> None:
        boxes = normalized_boxes_from_arrays(
            np.asarray([[0.5, 0.5, 0.2, 0.4]]),
            np.asarray([0.0]),
            expected_class_id=0,
        )

        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0].clipped_area, 0.08)
        with self.assertRaisesRegex(ValueError, "Unexpected predicted classes"):
            normalized_boxes_from_arrays(
                np.asarray([[0.5, 0.5, 0.2, 0.4]]),
                np.asarray([1.0]),
                expected_class_id=0,
            )

    def test_comparison_row_keeps_predictions_separate_from_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label = root / "C-W2-0001.txt"
            image = root / "C-W2-0001.jpg"
            label.write_text(
                "0 0.5 0.5 0.2 0.2\n0 0.2 0.2 0.1 0.1\n", encoding="utf-8"
            )
            sample = PioSample("C-W2-0001", "val", image, label)
            predicted = normalized_boxes_from_arrays(
                np.asarray([[0.5, 0.5, 0.2, 0.2]]),
                np.asarray([0]),
                expected_class_id=0,
            )

            row = build_comparison_row(
                sample,
                predicted,
                confidences=[0.8],
                expected_class_id=0,
                grid_rows=4,
                grid_columns=4,
            )

            self.assertEqual(row["object_count"], 1)
            self.assertEqual(row["reference_object_count"], 2)
            self.assertEqual(row["error_object_count"], -1.0)
            self.assertEqual(row["absolute_error_object_count"], 1.0)
            self.assertAlmostEqual(row["mean_detection_confidence"], 0.8)

    def test_summary_reports_feature_error_and_claim_boundary(self) -> None:
        rows = []
        for reference, predicted in ((10, 8), (20, 18), (30, 33)):
            row = {
                "object_count": predicted,
                "reference_object_count": reference,
                "environment": "commercial",
                "week": 1,
            }
            for feature in (
                "bbox_area_ratio",
                "center_spread",
                "mean_nearest_neighbor_distance",
                "max_grid_fraction",
            ):
                row[feature] = float(predicted)
                row[f"reference_{feature}"] = float(reference)
            rows.append(row)

        summary = summarize_comparison(rows)

        self.assertEqual(summary["predicted_boxes"], 59)
        self.assertEqual(summary["reference_boxes"], 60)
        self.assertAlmostEqual(summary["feature_metrics"]["object_count"]["mae"], 7 / 3)
        self.assertEqual(summary["groups"]["week"]["1"]["image_count"], 3)
        self.assertIn("does not validate Piling", summary["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
