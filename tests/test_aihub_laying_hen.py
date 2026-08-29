import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from pileguard.data.aihub_laying_hen import (
    AihubArchivePair,
    iter_inner_files,
    load_annotations,
    parse_annotation,
)


def annotation_payload(
    *, image_id: str = "sample.png", bbox: list[int] | None = None
) -> bytes:
    return json.dumps(
        {
            "image": {"fileName": image_id, "width": 100, "height": 80},
            "annotationImageInfo": {
                "lifeCycle": "early-stage-of-egg-laying",
                "action": "community",
            },
            "annotationObjectInfo": [
                {
                    "category": "layer-chicken",
                    "BBox": bbox or [-5, 10, 25, 20],
                    "actionValue": True,
                    "isCrowd": 0,
                }
            ],
        }
    ).encode()


def build_inner_tar(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def write_wrapper(path: Path, inner: bytes, *, split_at: int | None = None) -> None:
    split = split_at or len(inner)
    pieces = (inner[:split], inner[split:]) if split < len(inner) else (inner,)
    offset = 0
    with tarfile.open(path, mode="w") as archive:
        for piece in pieces:
            info = tarfile.TarInfo(f"dataset.tar.gz.part{offset}")
            info.size = len(piece)
            archive.addfile(info, io.BytesIO(piece))
            offset += len(piece)


class AihubLayingHenDataTest(unittest.TestCase):
    def test_streams_contiguous_split_archive_and_parses_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper = root / "label.tar"
            inner = build_inner_tar({"labels/sample.json": annotation_payload()})
            write_wrapper(wrapper, inner, split_at=max(len(inner) // 2, 1))
            pair = AihubArchivePair("early", wrapper, root / "unused.tar")

            annotations = load_annotations(pair)

            self.assertEqual(set(annotations), {"sample.png"})
            sample = annotations["sample.png"]
            self.assertEqual(len(sample.pixel_boxes), 1)
            self.assertEqual(sample.pixel_boxes[0].x1, 0)
            self.assertEqual(sample.clipped_boxes, 1)
            self.assertEqual(sample.action_positive_boxes, 1)

    def test_rejects_gap_in_split_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wrapper = Path(directory) / "broken.tar"
            with tarfile.open(wrapper, mode="w") as archive:
                for name, payload in (("data.tar.gz.part0", b"abc"), ("data.tar.gz.part9", b"def")):
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))

            with self.assertRaisesRegex(ValueError, "incomplete"):
                list(iter_inner_files(wrapper, suffix=".json"))

    def test_rejects_inner_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wrapper = Path(directory) / "unsafe.tar"
            write_wrapper(wrapper, build_inner_tar({"../sample.json": annotation_payload()}))

            with self.assertRaisesRegex(ValueError, "Unsafe archive"):
                list(iter_inner_files(wrapper, suffix=".json"))

    def test_rejects_claim_incompatible_action(self) -> None:
        payload = json.loads(annotation_payload())
        payload["annotationImageInfo"]["action"] = "piling"

        with self.assertRaisesRegex(ValueError, "community"):
            parse_annotation(
                json.dumps(payload).encode(), stage="early", member_name="sample.json"
            )

    def test_records_and_skips_non_positive_official_box(self) -> None:
        annotation = parse_annotation(
            annotation_payload(bbox=[10, 20, 0, 15]),
            stage="early",
            member_name="sample.json",
        )

        self.assertEqual(annotation.invalid_boxes, 1)
        self.assertEqual(annotation.pixel_boxes, ())
        self.assertEqual(annotation.normalized_boxes, ())
        self.assertEqual(annotation.action_positive_boxes, 1)


if __name__ == "__main__":
    unittest.main()
