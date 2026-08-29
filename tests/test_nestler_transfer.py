import unittest

import numpy as np

from pileguard.evaluation.nestler_transfer import (
    aggregate_rows,
    build_integration_gate,
    detections_from_arrays,
    intersection_over_union,
    match_centers,
    match_detections,
)
from pileguard.features.nestler import BoundingBox


class NestlerTransferTest(unittest.TestCase):
    def test_detection_array_conversion_keeps_confidence(self) -> None:
        detections = detections_from_arrays(
            np.asarray([[1, 2, 11, 12]], dtype=float),
            np.asarray([0.8]),
            np.asarray([0]),
        )

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.8)
        self.assertEqual(detections[0].box.area, 100)

    def test_iou_matching_is_one_to_one(self) -> None:
        reference = [
            BoundingBox(0, 0, 10, 10, 1),
            BoundingBox(20, 20, 30, 30, 2),
        ]
        predicted = [
            BoundingBox(0, 0, 10, 10, -1),
            BoundingBox(1, 1, 9, 9, -1),
            BoundingBox(40, 40, 50, 50, -1),
        ]

        self.assertAlmostEqual(intersection_over_union(reference[0], predicted[0]), 1.0)
        self.assertEqual(
            match_detections(reference, predicted, iou_threshold=0.5),
            (1, 2, 1),
        )
        self.assertEqual(
            match_centers(
                reference,
                predicted,
                frame_width=100,
                frame_height=100,
                distance_threshold=0.05,
            ),
            (1, 2, 1),
        )

    def test_aggregate_rows_reports_domain_transfer_failure(self) -> None:
        rows = [
            {
                "annotation_available": True,
                "predicted_count": 1,
                "reference_count": 2,
                "iou_true_positives": 1,
                "iou_false_positives": 0,
                "iou_false_negatives": 1,
                "center_true_positives": 1,
                "center_false_positives": 0,
                "center_false_negatives": 1,
            },
            {
                "annotation_available": True,
                "predicted_count": 0,
                "reference_count": 2,
                "iou_true_positives": 0,
                "iou_false_positives": 0,
                "iou_false_negatives": 2,
                "center_true_positives": 0,
                "center_false_positives": 0,
                "center_false_negatives": 2,
            },
        ]

        summary = aggregate_rows(rows)

        self.assertEqual(summary["frames_without_predictions"], 1)
        self.assertEqual(summary["count_mae"], 1.5)
        self.assertEqual(summary["matching_metrics"]["center"]["recall"], 0.25)
        self.assertEqual(summary["matching_metrics"]["center"]["f1"], 0.4)

        gate = build_integration_gate(summary, minimum_center_f1=0.5)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["decision"], "block predicted monitoring")


if __name__ == "__main__":
    unittest.main()
