import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from pileguard.data.domestic import (
    CLIP_COLUMNS,
    EVENT_COLUMNS,
    VideoMetadata,
    audit_domestic_dataset,
    secure_media_path,
)


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class DomesticDataIntakeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "data": {"clip_manifest": "clips.csv", "event_manifest": "events.csv"},
            "security": {
                "allowed_video_suffixes": [".mp4"],
                "minimum_file_bytes": 12,
                "identifier_pattern": r"^[A-Z][A-Z0-9_-]{2,31}$",
            },
            "quality": {
                "minimum_duration_seconds": 10.0,
                "minimum_width": 640,
                "minimum_height": 360,
                "minimum_fps": 10.0,
                "maximum_fps": 120.0,
            },
            "labels": {
                "allowed_clip_outcomes": ["normal", "near_piling", "piling"],
                "allowed_event_types": ["near_piling", "piling"],
                "accepted_review_statuses": ["double_reviewed", "adjudicated"],
                "required_splits": ["train", "val", "test"],
                "required_outcomes_per_split": ["normal", "piling"],
            },
        }

    def test_secure_media_path_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "relative path"):
                secure_media_path(Path(directory), "../outside.mp4")

    def test_secure_media_path_rejects_internal_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.mp4"
            target.write_bytes(b"video")
            (root / "linked.mp4").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                secure_media_path(root, "linked.mp4")

    def test_audit_accepts_reviewed_farm_isolated_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            videos.mkdir()
            clip_rows: list[dict[str, str]] = []
            event_rows: list[dict[str, str]] = []
            split_names = ("train", "val", "test")
            for index in range(1, 7):
                split = split_names[(index - 1) // 2]
                outcome = "normal" if index % 2 else "piling"
                video = videos / f"D{index:02d}.mp4"
                video.write_bytes(b"\x00\x00\x00\x18ftypisom" + bytes([index]))
                clip_rows.append(
                    {
                        "clip_id": f"D{index:02d}",
                        "farm_id": f"F{((index - 1) // 2) + 1:02d}",
                        "house_id": f"H{index:02d}",
                        "camera_id": f"C{index:02d}",
                        "captured_at": "2026-01-01T00:00:00+09:00",
                        "video_path": f"videos/D{index:02d}.mp4",
                        "split": split,
                        "clip_outcome": outcome,
                        "review_status": "double_reviewed",
                        "consent_status": "research_approved",
                        "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                    }
                )
                if outcome == "piling":
                    event_rows.append(
                        {
                            "event_id": f"E{index:02d}",
                            "clip_id": f"D{index:02d}",
                            "event_type": "piling",
                            "start_seconds": "10",
                            "end_seconds": "20",
                            "review_status": "double_reviewed",
                        }
                    )
            write_csv(root / "clips.csv", CLIP_COLUMNS, clip_rows)
            write_csv(root / "events.csv", EVENT_COLUMNS, event_rows)

            report = audit_domestic_dataset(
                dataset_root=root,
                config=self.config,
                video_probe=lambda _: VideoMetadata(30.0, 1280, 720, 30.0),
            )

            self.assertTrue(report["ready_for_model_validation"])
            self.assertEqual(report["counts"]["valid_sha256"], 6)
            self.assertNotIn("F01", str(report))

    def test_audit_does_not_decode_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            videos.mkdir()
            video = videos / "D01.mp4"
            video.write_bytes(b"\x00\x00\x00\x18ftypisom")
            row = {
                "clip_id": "D01",
                "farm_id": "F01",
                "house_id": "H01",
                "camera_id": "C01",
                "captured_at": "2026-01-01T00:00:00+09:00",
                "video_path": "videos/D01.mp4",
                "split": "train",
                "clip_outcome": "normal",
                "review_status": "double_reviewed",
                "consent_status": "research_approved",
                "sha256": "0" * 64,
            }
            write_csv(root / "clips.csv", CLIP_COLUMNS, [row])
            write_csv(root / "events.csv", EVENT_COLUMNS, [])
            config = {
                **self.config,
                "labels": {
                    **self.config["labels"],
                    "required_splits": ["train"],
                    "required_outcomes_per_split": ["normal"],
                },
            }
            probe_calls = 0

            def counting_probe(_: Path) -> VideoMetadata:
                nonlocal probe_calls
                probe_calls += 1
                return VideoMetadata(30.0, 1280, 720, 30.0)

            report = audit_domestic_dataset(
                dataset_root=root, config=config, video_probe=counting_probe
            )

            self.assertEqual(probe_calls, 0)
            self.assertIn("sha256_mismatch", {issue["code"] for issue in report["issues"]})

    def test_audit_rejects_farm_leakage_and_unreviewed_normal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            videos.mkdir()
            rows = []
            for index, split in enumerate(("train", "val", "test"), start=1):
                video = videos / f"D{index:02d}.mp4"
                video.write_bytes(b"\x00\x00\x00\x18ftypisom" + bytes([index]))
                rows.append(
                    {
                        "clip_id": f"D{index:02d}",
                        "farm_id": "F01" if index < 3 else "F03",
                        "house_id": f"H{index:02d}",
                        "camera_id": f"C{index:02d}",
                        "captured_at": "2026-01-01T00:00:00+09:00",
                        "video_path": f"videos/D{index:02d}.mp4",
                        "split": split,
                        "clip_outcome": "normal",
                        "review_status": "single_reviewed" if index == 1 else "double_reviewed",
                        "consent_status": "research_approved",
                        "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                    }
                )
            write_csv(root / "clips.csv", CLIP_COLUMNS, rows)
            write_csv(root / "events.csv", EVENT_COLUMNS, [])

            report = audit_domestic_dataset(
                dataset_root=root,
                config=self.config,
                video_probe=lambda _: VideoMetadata(30.0, 1280, 720, 30.0),
            )

            codes = {issue["code"] for issue in report["issues"]}
            self.assertFalse(report["ready_for_model_validation"])
            self.assertIn("farm_split_leakage", codes)
            self.assertIn("insufficient_clip_review", codes)


if __name__ == "__main__":
    unittest.main()
