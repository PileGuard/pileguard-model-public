import unittest

from pileguard.demo.integrated import (
    BASELINE_FEATURES,
    build_camera_baselines,
    row_to_evidence,
    score_feature_rows,
)


def feature_row(frame_index: int, value: float, *, missing: bool = False) -> dict[str, object]:
    row: dict[str, object] = {
        "job_id": "job_test",
        "site": "Test",
        "frame_index": frame_index,
        "timestamp_seconds": frame_index / 10,
    }
    for feature in BASELINE_FEATURES:
        row[feature] = "" if missing else value
    return row


class IntegratedDemoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [feature_row(index, 1.0 + index * 0.01) for index in range(4)]
        self.baselines = build_camera_baselines(
            self.rows,
            calibration_frames=3,
            minimum_samples=3,
            minimum_scale_ratio=0.05,
        )

    def test_camera_relative_evidence_increases_above_baseline(self) -> None:
        high_row = feature_row(5, 2.0)

        evidence, availability = row_to_evidence(
            high_row,
            self.baselines["job_test"],
            saturation_iqr=3.0,
        )

        self.assertEqual(evidence.density, 1.0)
        self.assertEqual(evidence.inflow, 1.0)
        self.assertEqual(evidence.convergence, 1.0)
        self.assertTrue(all(availability.values()))

    def test_missing_features_are_reported_unavailable(self) -> None:
        evidence, availability = row_to_evidence(
            feature_row(5, 2.0, missing=True),
            self.baselines["job_test"],
            saturation_iqr=3.0,
        )

        self.assertEqual(evidence.context, 0.0)
        self.assertFalse(any(availability.values()))

    def test_calibration_frames_never_emit_alerts(self) -> None:
        rows = self.rows[:3] + [feature_row(3, 2.0)]
        config = {
            "calibration": {"frames": 3, "saturation_iqr": 3.0},
            "quality": {"minimum_valid_evidence": 4},
            "risk": {
                "smoothing_alpha": 1.0,
                "watch_threshold": 10.0,
                "warning_threshold": 20.0,
                "critical_threshold": 30.0,
                "release_margin": 5.0,
            },
        }

        results = score_feature_rows(rows, self.baselines, config)

        self.assertEqual([row["alert_state"] for row in results[:3]], ["calibrating"] * 3)
        self.assertEqual(results[-1]["alert_state"], "critical")

    def test_missing_monitoring_frame_does_not_reuse_previous_risk(self) -> None:
        rows = self.rows[:3] + [feature_row(3, 2.0), feature_row(4, 2.0, missing=True)]
        config = {
            "calibration": {"frames": 3, "saturation_iqr": 3.0},
            "quality": {"minimum_valid_evidence": 4},
            "risk": {
                "smoothing_alpha": 1.0,
                "watch_threshold": 10.0,
                "warning_threshold": 20.0,
                "critical_threshold": 30.0,
                "release_margin": 5.0,
            },
        }

        results = score_feature_rows(rows, self.baselines, config)

        self.assertEqual(results[-1]["alert_state"], "unavailable")
        self.assertIsNone(results[-1]["risk_smoothed"])

    def test_optional_missing_tracking_baselines_remain_unavailable(self) -> None:
        rows = [feature_row(index, 1.0 + index * 0.01) for index in range(4)]
        for row in rows:
            row["tracked_speed_per_second"] = ""
            row["tracked_coherence"] = ""
        baselines = build_camera_baselines(
            rows,
            calibration_frames=3,
            minimum_samples=3,
            minimum_scale_ratio=0.05,
            optional_features={"tracked_speed_per_second", "tracked_coherence"},
        )

        evidence, availability = row_to_evidence(
            rows[-1], baselines["job_test"], saturation_iqr=3.0
        )

        self.assertIsNone(baselines["job_test"]["tracked_speed_per_second"])
        self.assertTrue(availability["inflow"])
        self.assertTrue(availability["directional_coherence"])
        self.assertGreaterEqual(evidence.inflow, 0.0)


if __name__ == "__main__":
    unittest.main()
