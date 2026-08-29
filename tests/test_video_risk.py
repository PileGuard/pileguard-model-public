import tempfile
import unittest
from pathlib import Path

from pileguard.demo.video_risk import missing_tracking_features, resolve_video_inputs


class VideoRiskTest(unittest.TestCase):
    def test_resolves_nestler_video_identity_without_storing_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video_dir = Path(directory) / "job_000004"
            video_dir.mkdir()
            video_path = video_dir / "job_000004_400frames.mp4"
            video_path.touch()

            videos = resolve_video_inputs([video_path])

            self.assertEqual(videos[0].video_id, "job_000004")
            self.assertEqual(videos[0].site, "Bulgaria")

    def test_rejects_duplicate_video_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video_dir = Path(directory) / "job_000004"
            video_dir.mkdir()
            first = video_dir / "first.mp4"
            second = video_dir / "second.mp4"
            first.touch()
            second.touch()

            with self.assertRaisesRegex(ValueError, "Duplicate video ID"):
                resolve_video_inputs([first, second])

    def test_tracking_features_are_explicitly_missing(self) -> None:
        features = missing_tracking_features()

        self.assertEqual(features["track_match_count"], 0)
        self.assertIsNone(features["tracked_speed_per_second"])
        self.assertIsNone(features["tracked_coherence"])


if __name__ == "__main__":
    unittest.main()
