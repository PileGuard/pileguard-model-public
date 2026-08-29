import unittest
from pathlib import Path

from pileguard.evaluation.nestler_validation import (
    build_findings,
    compare_slice_metrics,
    density_slice,
    label_path_for_image,
)


class NestlerValidationAuditTest(unittest.TestCase):
    def test_density_slice_uses_predeclared_boundaries(self) -> None:
        self.assertEqual(density_slice(5, low_maximum=5, medium_maximum=10), "density-low")
        self.assertEqual(
            density_slice(6, low_maximum=5, medium_maximum=10), "density-medium"
        )
        self.assertEqual(
            density_slice(11, low_maximum=5, medium_maximum=10), "density-high"
        )

    def test_label_path_for_image_preserves_split_and_stem(self) -> None:
        image = Path("outputs/fold/images/val/job_000007_frame_000001.jpg")
        self.assertEqual(
            label_path_for_image(image),
            Path("outputs/fold/labels/val/job_000007_frame_000001.txt"),
        )

    def test_label_path_rejects_unexpected_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            label_path_for_image(Path("images/job.jpg"))

    def test_findings_identify_worst_density_slice(self) -> None:
        rows = {
            "clip-job_a": {"metrics": {"map50": 0.7}},
            "clip-job_b": {"metrics": {"map50": 0.2}},
            "density-low": {"metrics": {"map50": 0.6}},
            "density-high": {"metrics": {"map50": 0.1}},
        }
        findings = build_findings(rows)
        self.assertIn("validation clip mAP50 gap=0.5000", findings)
        self.assertIn("worst density slice=density-high mAP50=0.1000", findings)

    def test_compare_slice_metrics_returns_current_minus_baseline(self) -> None:
        current = {
            "density-low": {
                "metrics": {
                    "precision": 0.4,
                    "recall": 0.3,
                    "map50": 0.2,
                    "map50_95": 0.1,
                }
            }
        }
        baseline = {
            "density-low": {
                "metrics": {
                    "precision": 0.3,
                    "recall": 0.1,
                    "map50": 0.15,
                    "map50_95": 0.08,
                }
            }
        }
        delta = compare_slice_metrics(current, baseline)["density-low"]
        self.assertAlmostEqual(delta["precision"], 0.1)
        self.assertAlmostEqual(delta["recall"], 0.2)
        self.assertAlmostEqual(delta["map50"], 0.05)
        self.assertAlmostEqual(delta["map50_95"], 0.02)


if __name__ == "__main__":
    unittest.main()
