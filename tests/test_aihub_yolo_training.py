import json
import tempfile
import unittest
from pathlib import Path

from pileguard.training.aihub_yolo import validate_dataset_contract


def valid_summary() -> dict[str, object]:
    return {
        "status": "ready",
        "dataset": {
            "class_names": ["layer-chicken"],
            "official_count_contract": {"ok": True},
            "splits": {
                "train": {"image_count": 10, "usable_box_count": 100},
                "val": {"image_count": 4, "usable_box_count": 40},
            },
        },
        "selected_dataset": {
            "training_validation_filename_overlap_count": 0,
            "training_validation_content_overlap_count": 0,
        },
    }


class AihubYoloTrainingContractTest(unittest.TestCase):
    def test_accepts_audited_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(valid_summary()), encoding="utf-8")

            summary = validate_dataset_contract(path)

            self.assertEqual(summary["status"], "ready")

    def test_rejects_training_validation_content_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            summary = valid_summary()
            summary["selected_dataset"]["training_validation_content_overlap_count"] = 1
            path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "isolation"):
                validate_dataset_contract(path)


if __name__ == "__main__":
    unittest.main()
