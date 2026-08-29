import unittest

from pileguard.evaluation.aihub_error_audit import (
    best_thresholds_by_stage,
    global_threshold_tradeoff,
    summarize_error_rows,
    worst_cases,
)


def image_row(
    *, image_id: str, stage: str, reference: int, predicted: int, tp: int, fp: int, fn: int
) -> dict[str, str]:
    error = predicted - reference
    return {
        "image_id": image_id,
        "stage": stage,
        "confidence_threshold": "0.25",
        "reference_count": str(reference),
        "predicted_count": str(predicted),
        "count_error": str(error),
        "absolute_count_error": str(abs(error)),
        "center_true_positives": str(tp),
        "center_false_positives": str(fp),
        "center_false_negatives": str(fn),
    }


class AihubErrorAuditTest(unittest.TestCase):
    def test_summarizes_count_error_distribution(self) -> None:
        rows = [
            image_row(
                image_id="under.png", stage="early", reference=10, predicted=6, tp=6, fp=0, fn=4
            ),
            image_row(
                image_id="over.png", stage="early", reference=10, predicted=12, tp=9, fp=3, fn=1
            ),
        ]

        summary = summarize_error_rows(rows)

        self.assertEqual(summary["count_error"]["mae"], 3.0)
        self.assertEqual(summary["count_error"]["bias"], -1.0)
        self.assertEqual(summary["count_error"]["undercount_images"], 1)
        self.assertAlmostEqual(summary["center_matching"]["f1"], 30 / 38)
        self.assertEqual(worst_cases(rows, limit=1)[0]["image_id"], "under.png")

    def test_selects_best_diagnostic_threshold_per_stage(self) -> None:
        rows = []
        for stage in ("early", "middle", "late"):
            rows.extend(
                [
                    {
                        "stage": stage,
                        "confidence_threshold": 0.20,
                        "f1": 0.70,
                        "count_mae": 4.0,
                    },
                    {
                        "stage": stage,
                        "confidence_threshold": 0.25,
                        "f1": 0.75,
                        "count_mae": 5.0,
                    },
                ]
            )

        selected = best_thresholds_by_stage(rows)

        self.assertEqual(selected["early"]["confidence_threshold"], 0.25)
        self.assertEqual(selected["middle"]["confidence_threshold"], 0.25)
        self.assertEqual(selected["late"]["confidence_threshold"], 0.25)

    def test_reports_localization_and_count_threshold_tradeoff(self) -> None:
        summary = {
            "threshold_results": {
                "0.2": {
                    "localization": {
                        "matching_metrics": {"center": {"f1": 0.75}},
                        "count_mae": 6.2,
                        "count_bias": 1.0,
                    }
                },
                "0.25": {
                    "localization": {
                        "matching_metrics": {"center": {"f1": 0.74}},
                        "count_mae": 5.7,
                        "count_bias": -1.4,
                    }
                },
            }
        }

        tradeoff = global_threshold_tradeoff(summary)

        self.assertEqual(
            tradeoff["localization_f1_best"]["confidence_threshold"], 0.2
        )
        self.assertEqual(tradeoff["count_mae_best"]["confidence_threshold"], 0.25)


if __name__ == "__main__":
    unittest.main()
