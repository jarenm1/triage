"""Exercise evaluation failure accounting against the real CUDA task."""

import unittest

import torch

from rl.tracking import FAILURE_SQUARED_ERROR, evaluate_tracking


class SaturatedMotorPolicy:
    def __init__(self, n):
        actions = torch.tensor([[10.0, -10.0, -10.0, 10.0]], device="cuda").repeat(n, 1)
        self.distribution = torch.distributions.Normal(
            actions, torch.ones_like(actions), validate_args=False
        )

    def forward_eval(self, observations):
        return self.distribution, None, ()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TrackingEvaluationTests(unittest.TestCase):
    def test_crash_cannot_improve_error_or_remove_missed_targets(self):
        n = 4
        report, _ = evaluate_tracking(
            SaturatedMotorPolicy(n), n, 17, 0, schedule="settling"
        )
        self.assertEqual(report["survival_rate"], 0.0)
        self.assertEqual(report["commanded_targets"], 4 * n)
        self.assertEqual(report["settled_commands"], 0.0)
        self.assertEqual(report["settled_fraction"], 0.0)
        self.assertGreater(report["mean_episode_rms_position_error"], 5.5)
        self.assertAlmostEqual(
            report["maximum_position_error"], FAILURE_SQUARED_ERROR**0.5, places=5
        )
        self.assertAlmostEqual(report["action_saturation_fraction"], 1.0)
        self.assertEqual(report["long_steps"], n * 2000)
        self.assertEqual(report["short_steps"], 0)

    def test_failed_tail_uses_original_interruption_schedule(self):
        n = 4
        report, plan = evaluate_tracking(
            SaturatedMotorPolicy(n), n, 17, 0, schedule="mixed"
        )
        expected_short_steps = 0
        for commands in plan.cpu().tolist():
            remaining = 2000
            for _, _, _, duration in commands:
                controls = min(remaining, int(duration))
                if duration <= 100:
                    expected_short_steps += controls
                remaining -= controls
                if remaining == 0:
                    break
        self.assertEqual(report["survival_rate"], 0.0)
        self.assertGreater(report["mean_episode_rms_position_error"], 5.5)
        self.assertGreater(expected_short_steps, 0)
        self.assertEqual(report["short_steps"], expected_short_steps)
        self.assertEqual(report["long_steps"], n * 2000 - expected_short_steps)
        recombined_squared_error = (
            report["short_interval_rms"] ** 2 * report["short_steps"]
            + report["long_interval_rms"] ** 2 * report["long_steps"]
        ) / (n * 2000)
        self.assertAlmostEqual(
            recombined_squared_error, report["pooled_rms_position_error"] ** 2, places=4
        )


if __name__ == "__main__":
    unittest.main()
