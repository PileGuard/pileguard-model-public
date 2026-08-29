import unittest

from pileguard.data.aihub_laying_hen import AihubAnnotation
from pileguard.evaluation.aihub_domestic import (
    PredictionRecord,
    build_image_row,
    summarize_rows,
)
from pileguard.evaluation.nestler_transfer import ScoredDetection
from pileguard.features.nestler import BoundingBox
from pileguard.features.pio import NormalizedBox


class AihubDomesticEvaluationTest(unittest.TestCase):
    def test_builds_localization_and_density_metrics(self) -> None:
        reference_pixel = BoundingBox(10, 10, 30, 30, 0)
        annotation = AihubAnnotation(
            image_id="sample.png",
            stage="early",
            width=100,
            height=100,
            pixel_boxes=(reference_pixel,),
            normalized_boxes=(NormalizedBox(0, 0.2, 0.2, 0.2, 0.2),),
            action_positive_boxes=1,
            clipped_boxes=0,
        )
        record = PredictionRecord(
            annotation,
            (
                ScoredDetection(BoundingBox(10, 10, 30, 30, -1), 0.8),
                ScoredDetection(BoundingBox(60, 60, 80, 80, -1), 0.1),
            ),
        )

        row = build_image_row(
            record,
            confidence_threshold=0.25,
            iou_threshold=0.5,
            center_distance_threshold=0.05,
            grid_rows=4,
            grid_columns=4,
        )
        summary = summarize_rows([row], include_stages=True)

        self.assertEqual(row["predicted_count"], 1)
        self.assertEqual(row["center_true_positives"], 1)
        self.assertEqual(row["error_object_count"], 0)
        self.assertEqual(summary["localization"]["matching_metrics"]["iou"]["f1"], 1.0)
        self.assertEqual(summary["density_features"]["feature_metrics"]["object_count"]["mae"], 0)
        self.assertIn("early", summary["stages"])


if __name__ == "__main__":
    unittest.main()
