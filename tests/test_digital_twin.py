import unittest

import numpy as np

from pileguard.demo.digital_twin import generate_evidence, simulate_scenario, smoothstep


class DigitalTwinTest(unittest.TestCase):
    def test_smoothstep_has_fixed_endpoints(self) -> None:
        values = smoothstep(np.array([0.0, 5.0, 10.0]), 0, 10)

        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 1.0)
        self.assertGreater(values[1], 0.0)
        self.assertLess(values[1], 1.0)

    def test_every_scenario_stays_in_normalized_range(self) -> None:
        times = np.arange(0, 181)
        for scenario in (
            "social_attraction",
            "group_convergence",
            "external_context",
        ):
            evidence = generate_evidence(scenario, times, np.random.default_rng(2026))
            values_are_normalized = all(
                np.all((values >= 0) & (values <= 1)) for values in evidence.values()
            )
            self.assertTrue(values_are_normalized)

    def test_social_scenario_reaches_warning(self) -> None:
        rows = simulate_scenario(
            "social_attraction",
            times=np.arange(0, 181),
            risk_config={
                "smoothing_alpha": 0.18,
                "watch_threshold": 40.0,
                "warning_threshold": 60.0,
                "critical_threshold": 80.0,
                "release_margin": 5.0,
            },
            rng=np.random.default_rng(2026),
        )

        self.assertIn("warning", {row["alert_state"] for row in rows})
        self.assertGreater(rows[-1]["risk_smoothed"], rows[0]["risk_smoothed"])


if __name__ == "__main__":
    unittest.main()
