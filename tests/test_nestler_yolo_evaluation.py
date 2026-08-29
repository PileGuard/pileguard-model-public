import json
import tempfile
import unittest
from pathlib import Path

from pileguard.evaluation.nestler_yolo import (
    evaluate_gate,
    job_id_from_image,
    load_final_test_contract,
)


class NestlerYoloEvaluationTest(unittest.TestCase):
    def test_job_id_from_image(self) -> None:
        path = Path("job_000008_frame_000123.jpg")
        self.assertEqual(job_id_from_image(path), "job_000008")

    def test_gate_blocks_when_recall_is_below_predeclared_minimum(self) -> None:
        gate = evaluate_gate(
            {"map50": 0.6, "recall": 0.4},
            {
                "minimum_map50": 0.5,
                "minimum_recall": 0.5,
                "domestic_validation_required": True,
            },
        )
        self.assertFalse(gate["quantitative_passed"])
        self.assertFalse(gate["monitoring_integration_allowed"])

    def test_contract_rejects_training_that_touched_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "job_000008_frame_000000.jpg"
            image.touch()
            (root / "test.txt").write_text(str(image) + "\n", encoding="utf-8")
            dataset_yaml = root / "dataset.yaml"
            dataset_yaml.write_text("names:\n  0: nestler_tracker_region\n", encoding="utf-8")
            dataset_summary = {
                "frame_leakage_between_splits": False,
                "independent_test_use": "reserved; final evaluation only",
                "splits": {
                    "test": {
                        "annotated_frames": 1,
                        "jobs": ["job_000008"],
                    }
                },
            }
            training_summary = {
                "test_evaluated": True,
                "class_name": "nestler_tracker_region",
            }
            dataset_summary_path = root / "dataset-summary.json"
            training_summary_path = root / "training-summary.json"
            dataset_summary_path.write_text(json.dumps(dataset_summary), encoding="utf-8")
            training_summary_path.write_text(json.dumps(training_summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "test was untouched"):
                load_final_test_contract(
                    dataset_summary_path=dataset_summary_path,
                    training_summary_path=training_summary_path,
                    dataset_yaml=dataset_yaml,
                )


if __name__ == "__main__":
    unittest.main()

