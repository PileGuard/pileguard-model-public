import io
import json
import struct
import tarfile
import tempfile
import unittest
from pathlib import Path

from pileguard.data.aihub_detection import PNG_SIGNATURE, prepare_dataset, yolo_label
from pileguard.data.aihub_laying_hen import STAGE_METADATA, parse_annotation


def build_inner_tar(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def write_wrapper(path: Path, files: dict[str, bytes]) -> None:
    inner = build_inner_tar(files)
    with tarfile.open(path, mode="w") as archive:
        info = tarfile.TarInfo("dataset.tar.gz.part0")
        info.size = len(inner)
        archive.addfile(info, io.BytesIO(inner))


def annotation(image_id: str, lifecycle: str, *, invalid: bool = False) -> bytes:
    return json.dumps(
        {
            "image": {"fileName": image_id, "width": 100, "height": 80},
            "annotationImageInfo": {"lifeCycle": lifecycle, "action": "community"},
            "annotationObjectInfo": [
                {
                    "category": "layer-chicken",
                    "BBox": [10, 20, 0 if invalid else 30, 20],
                    "actionValue": False,
                    "iscrowd": 0,
                }
            ],
        }
    ).encode()


def png_payload(marker: str) -> bytes:
    return PNG_SIGNATURE + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 100, 80) + marker.encode()


def write_split(root: Path, split: str, *, overlap_payload: bytes | None = None) -> None:
    split_root = root / split
    split_root.mkdir(parents=True)
    for stage, metadata in STAGE_METADATA.items():
        token = str(metadata["filename_token"])
        image_id = f"{split}-{stage}.png"
        source = png_payload(image_id)
        if split == "validation" and stage == "late" and overlap_payload is not None:
            source = overlap_payload
        write_wrapper(
            split_root / f"{token}_군집_라벨.tar",
            {f"labels/{Path(image_id).stem}.json": annotation(
                image_id,
                str(metadata["life_cycle"]),
                invalid=split == "training" and stage == "early",
            )},
        )
        write_wrapper(
            split_root / f"{token}_군집_원천.tar",
            {f"images/{image_id}": source},
        )


class AihubDetectionPreparationTest(unittest.TestCase):
    def test_removes_exact_duplicate_boxes_from_yolo_label(self) -> None:
        payload = json.loads(annotation("sample.png", str(STAGE_METADATA["early"]["life_cycle"])))
        payload["annotationObjectInfo"].append(payload["annotationObjectInfo"][0].copy())
        parsed = parse_annotation(
            json.dumps(payload).encode(), stage="early", member_name="sample.json"
        )

        label, duplicate_boxes = yolo_label(parsed)

        self.assertEqual(duplicate_boxes, 1)
        self.assertEqual(len(label.splitlines()), 1)

    def test_prepares_yolo_dataset_and_records_invalid_box(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_root = root / "raw"
            write_split(raw_root, "training")
            write_split(raw_root, "validation")

            result = prepare_dataset(
                raw_root=raw_root,
                dataset_root=root / "dataset",
                summary_path=root / "summary.json",
                enforce_official_counts=False,
            )

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["dataset"]["splits"]["train"]["image_count"], 3)
            self.assertEqual(result["dataset"]["splits"]["train"]["invalid_box_count"], 1)
            self.assertEqual(result["dataset"]["splits"]["train"]["duplicate_box_count"], 0)
            self.assertNotIn(
                "invalid_box_examples", result["dataset"]["splits"]["train"]
            )
            empty_label = root / "dataset/labels/train/training-early.txt"
            self.assertEqual(empty_label.read_text(), "")
            regular_label = root / "dataset/labels/val/validation-middle.txt"
            self.assertEqual(len(regular_label.read_text().split()), 5)

    def test_blocks_exact_image_content_overlap_between_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_root = root / "raw"
            overlap = png_payload("shared-image")
            write_split(raw_root, "training")
            training_early = raw_root / "training" / "산란초기_군집_원천.tar"
            write_wrapper(
                training_early,
                {"images/training-early.png": overlap},
            )
            write_split(raw_root, "validation", overlap_payload=overlap)

            result = prepare_dataset(
                raw_root=raw_root,
                dataset_root=root / "dataset",
                summary_path=root / "summary.json",
                enforce_official_counts=False,
            )

            self.assertEqual(result["status"], "ready_after_deduplication")
            self.assertEqual(
                result["raw_leakage_audit"]["training_validation_content_overlap_count"], 1
            )
            self.assertFalse(
                result["raw_leakage_audit"]["per_file_examples_published"]
            )
            self.assertNotIn("content_overlap_examples", result["raw_leakage_audit"])
            self.assertEqual(
                result["selected_dataset"]["training_validation_content_overlap_count"], 0
            )
            self.assertEqual(result["selected_dataset"]["train_image_count"], 2)


if __name__ == "__main__":
    unittest.main()
