import unittest

from pileguard.data.nestler_detection import (
    JobAudit,
    build_summary,
    validate_split_assignment,
    yolo_label,
)
from pileguard.features.nestler import BoundingBox


class NestlerDetectionDataTest(unittest.TestCase):
    def test_validate_split_assignment_rejects_clip_leakage(self) -> None:
        split = {
            "train": ["job_a"],
            "val": ["job_b"],
            "test": ["job_a", "job_c"],
        }
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            validate_split_assignment(split, {"job_a", "job_b", "job_c"})

    def test_validate_split_assignment_requires_complete_coverage(self) -> None:
        split = {"train": ["job_a"], "val": ["job_b"], "test": ["job_c"]}
        with self.assertRaisesRegex(ValueError, "missing from split config"):
            validate_split_assignment(split, {"job_a", "job_b", "job_c", "job_d"})

    def test_yolo_label_normalizes_tracker_box(self) -> None:
        box = BoundingBox(x1=10, y1=20, x2=50, y2=60, track_id=7)
        self.assertEqual(
            yolo_label(box, frame_width=100, frame_height=100),
            "0 0.300000 0.400000 0.400000 0.400000",
        )

    def test_summary_preserves_missing_annotation_policy(self) -> None:
        audits = [
            JobAudit("job_a", "Bulgaria", "train", 9, 1, 0, 20),
            JobAudit("job_b", "Rwanda", "val", 8, 2, 1, 12),
            JobAudit("job_c", "Bulgaria", "test", 10, 0, 0, 30),
        ]
        summary = build_summary(audits)
        self.assertFalse(summary["frame_leakage_between_splits"])
        self.assertIn("exclude frame", summary["missing_bbox_policy"])
        self.assertEqual(summary["splits"]["train"]["annotated_frames"], 9)
        self.assertEqual(summary["splits"]["val"]["empty_annotated_frames"], 1)


if __name__ == "__main__":
    unittest.main()

