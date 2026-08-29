import unittest
from pathlib import Path
from unittest.mock import patch

from pileguard.data.nestler_balanced import balance_density_bins, validate_train_isolation


class NestlerBalancedDataTest(unittest.TestCase):
    @patch("pileguard.data.nestler_balanced.count_boxes")
    def test_balance_density_bins_matches_largest_bin(self, count_boxes) -> None:
        counts = {
            "job_a_frame_000000.jpg": 1,
            "job_a_frame_000001.jpg": 7,
            "job_b_frame_000000.jpg": 12,
            "job_b_frame_000001.jpg": 13,
            "job_b_frame_000002.jpg": 14,
        }
        count_boxes.side_effect = lambda path: counts[path.name]
        images = [Path(name) for name in counts]
        balanced, audit = balance_density_bins(
            images, low_maximum=5, medium_maximum=10, seed=2026
        )
        self.assertEqual(len(balanced), 9)
        self.assertEqual(
            audit["balanced_density_counts"],
            {"density-low": 3, "density-medium": 3, "density-high": 3},
        )
        self.assertEqual(set(balanced), set(images))

    def test_validate_train_isolation_rejects_validation_leakage(self) -> None:
        summary = {
            "splits": {
                "train": {"jobs": ["job_000004"], "annotated_frames": 1},
            }
        }
        image = Path("job_000004_frame_000000.jpg")
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_train_isolation(
                train_images=[image],
                validation_images=[image],
                test_images=[],
                dataset_summary=summary,
            )


if __name__ == "__main__":
    unittest.main()
