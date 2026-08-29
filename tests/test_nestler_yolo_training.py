import json
import tempfile
import unittest
from pathlib import Path

from pileguard.training.nestler_yolo import validate_dataset_contract


def valid_summary() -> dict:
    split = {
        "annotated_frames": 10,
        "boxes": 20,
        "sites": {"Bulgaria": 1, "Rwanda": 1},
    }
    return {
        "class_name": "nestler_tracker_region",
        "frame_leakage_between_splits": False,
        "missing_bbox_policy": "exclude frame; never convert missing annotation to empty label",
        "independent_test_use": "reserved; do not use for threshold selection",
        "splits": {"train": split, "val": split, "test": split},
    }


class NestlerYoloTrainingTest(unittest.TestCase):
    def write_contract(self, root: Path, summary: dict) -> tuple[Path, Path]:
        dataset_yaml = root / "dataset.yaml"
        dataset_yaml.write_text("names:\n  0: nestler_tracker_region\n", encoding="utf-8")
        summary_path = root / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return summary_path, dataset_yaml

    def test_validate_dataset_contract_accepts_isolated_clips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path, dataset_yaml = self.write_contract(root, valid_summary())
            result = validate_dataset_contract(summary_path, dataset_yaml)
            self.assertEqual(result["splits"]["test"]["annotated_frames"], 10)

    def test_validate_dataset_contract_rejects_frame_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = valid_summary()
            summary["frame_leakage_between_splits"] = True
            summary_path, dataset_yaml = self.write_contract(root, summary)
            with self.assertRaisesRegex(ValueError, "clip-level split isolation"):
                validate_dataset_contract(summary_path, dataset_yaml)

    def test_validate_dataset_contract_requires_reserved_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = valid_summary()
            summary["independent_test_use"] = "used during training"
            summary_path, dataset_yaml = self.write_contract(root, summary)
            with self.assertRaisesRegex(ValueError, "not marked as reserved"):
                validate_dataset_contract(summary_path, dataset_yaml)


if __name__ == "__main__":
    unittest.main()

