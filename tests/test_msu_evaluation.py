import unittest

import numpy as np

from pileguard.evaluation.msu import (
    PredictionRecord,
    compute_binary_metrics,
    compute_source_metrics,
    select_error_records,
)


class MSUEvaluationTest(unittest.TestCase):
    def test_computes_fixed_threshold_metrics(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.8, 0.7, 0.4])

        metrics, (precision, recall, thresholds) = compute_binary_metrics(
            labels, probabilities, threshold=0.5
        )

        self.assertEqual(metrics["confusion_matrix"]["true_negative"], 1)
        self.assertEqual(metrics["confusion_matrix"]["false_positive"], 1)
        self.assertEqual(metrics["confusion_matrix"]["false_negative"], 1)
        self.assertEqual(metrics["confusion_matrix"]["true_positive"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertGreater(metrics["pr_auc"], 0)
        self.assertEqual(len(precision), len(recall))
        self.assertEqual(len(thresholds), len(precision) - 1)

    def test_selects_most_confident_errors(self) -> None:
        records = [
            PredictionRecord("fp-low.jpg", 0, 0.7, 1),
            PredictionRecord("fp-high.jpg", 0, 0.9, 1),
            PredictionRecord("fn-low.jpg", 1, 0.1, 0),
            PredictionRecord("fn-high.jpg", 1, 0.4, 0),
            PredictionRecord("correct.jpg", 1, 0.9, 1),
        ]

        false_positives, false_negatives = select_error_records(records, top_per_type=1)

        self.assertEqual(false_positives[0].path, "fp-high.jpg")
        self.assertEqual(false_negatives[0].path, "fn-low.jpg")

    def test_computes_source_metrics_when_a_source_has_one_class(self) -> None:
        records = [
            PredictionRecord("test/negatives/cam1_a.jpg", 0, 0.1, 0),
            PredictionRecord("test/negatives/cam1_b.jpg", 0, 0.9, 1),
            PredictionRecord("test/positives/ch16_a.jpg", 1, 0.1, 0),
            PredictionRecord("test/positives/ch16_b.jpg", 1, 0.9, 1),
        ]

        metrics = compute_source_metrics(records)

        self.assertIsNone(metrics["cam1"]["recall"])
        self.assertEqual(metrics["cam1"]["false_positive"], 1)
        self.assertEqual(metrics["ch16"]["recall"], 0.5)
        self.assertEqual(metrics["ch16"]["false_negative"], 1)


if __name__ == "__main__":
    unittest.main()
