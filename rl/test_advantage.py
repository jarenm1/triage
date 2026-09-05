"""Behavioral checks for the adapted learner's episode and horizon boundaries."""

import unittest

import torch

from rl.vendor.torch_pufferl import compute_puff_advantage


class AdvantageTests(unittest.TestCase):
    def devices(self):
        return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    def test_timeout_bootstrap_failure_and_horizon_are_distinct(self):
        for device in self.devices():
            with self.subTest(device=device):
                values = torch.zeros(3, 3, device=device)
                rewards = torch.tensor([[1.0, 2.0, 3.0]] * 3, device=device)
                terminated = torch.zeros_like(values)
                truncated = torch.zeros_like(values)
                truncated[0, 1] = 1
                terminated[1, 1] = 1
                result = compute_puff_advantage(
                    values,
                    rewards,
                    terminated,
                    truncated,
                    torch.full_like(values, 10),
                    torch.full((3,), 4.0, device=device),
                    torch.ones_like(values),
                    torch.empty_like(values),
                    0.5,
                    1.0,
                    1.0,
                    1.0,
                )
                # Timeout bootstraps the final state but stops episode recurrence;
                # failure neither bootstraps nor crosses the episode boundary.
                # The final rollout transition retains its reward and bootstrap.
                expected = torch.tensor(
                    [[4.5, 7.0, 5.0], [2.0, 2.0, 5.0], [3.25, 4.5, 5.0]],
                    device=device,
                )
                torch.testing.assert_close(result, expected)

    def test_importance_and_trace_clips_have_separate_effects(self):
        for device in self.devices():
            with self.subTest(device=device):
                values = torch.tensor([[2.0, 3.0]], device=device)
                zeros = torch.zeros_like(values)
                result = compute_puff_advantage(
                    values,
                    torch.tensor([[1.0, 4.0]], device=device),
                    zeros,
                    zeros,
                    zeros,
                    torch.tensor([6.0], device=device),
                    torch.tensor([[0.5, 3.0]], device=device),
                    torch.empty_like(values),
                    0.5,
                    0.8,
                    1.0,
                    0.25,
                )
                torch.testing.assert_close(
                    result, torch.tensor([[0.65, 4.0]], device=device)
                )


if __name__ == "__main__":
    unittest.main()
