import unittest

from pileguard.models.risk import (
    RiskEvidence,
    baseline_delta,
    compute_mechanism_scores,
    next_alert_state,
    risk_index,
)


class RiskModelTest(unittest.TestCase):
    def test_high_social_evidence_produces_high_index(self) -> None:
        evidence = RiskEvidence(
            density=1.0,
            inflow=1.0,
            proximity=1.0,
            convergence=0.0,
            directional_coherence=0.0,
            corner=1.0,
            context=0.0,
        )

        index = risk_index(compute_mechanism_scores(evidence))

        self.assertAlmostEqual(index, 100.0)

    def test_alert_release_uses_hysteresis(self) -> None:
        thresholds = {"watch": 40.0, "warning": 60.0, "critical": 80.0}

        held = next_alert_state(
            "warning", 57.0, thresholds=thresholds, release_margin=5.0
        )
        released = next_alert_state(
            "warning", 54.0, thresholds=thresholds, release_margin=5.0
        )

        self.assertEqual(held, "warning")
        self.assertEqual(released, "watch")

    def test_camera_baseline_delta_is_clipped(self) -> None:
        self.assertEqual(
            baseline_delta(10, median=10, interquartile_range=2),
            0.0,
        )
        self.assertEqual(
            baseline_delta(16, median=10, interquartile_range=2),
            1.0,
        )
        self.assertEqual(
            baseline_delta(4, median=10, interquartile_range=2, direction=-1),
            1.0,
        )

    def test_rejects_out_of_range_evidence(self) -> None:
        with self.assertRaises(ValueError):
            RiskEvidence(
                density=1.1,
                inflow=0.0,
                proximity=0.0,
                convergence=0.0,
                directional_coherence=0.0,
                corner=0.0,
                context=0.0,
            )


if __name__ == "__main__":
    unittest.main()
