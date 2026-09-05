"""Full-horizon, paired tracking evaluation without rollout host transfers."""

import math

import torch

from rl.environment import TrackingEnv

HORIZON = 2000
FAILURE_SQUARED_ERROR = 37.0625


@torch.no_grad()
def evaluate_tracking(
    policy, n, seed, device, library=None, schedule="mixed", baseline=False
):
    with TrackingEnv(n, seed, HORIZON, device, library, schedule) as env:
        plan = env.command_plan.clone()
        boundaries = plan[:, :, 3].cumsum(1).contiguous()
        times = torch.arange(HORIZON, device=env.device).expand(n, -1).contiguous()
        indices = torch.searchsorted(boundaries, times.float(), right=True)
        short = plan[:, :, 3].gather(1, indices) <= 100
        active = torch.ones(n, dtype=torch.bool, device=env.device)
        error_sum = torch.zeros(n, device=env.device)
        short_error = torch.zeros_like(error_sum)
        long_error = torch.zeros_like(error_sum)
        max_error = torch.zeros_like(error_sum)
        saturation = torch.zeros_like(error_sum)
        observed_steps = torch.zeros_like(error_sum)
        settled = torch.zeros_like(error_sum)
        for step in range(HORIZON):
            observations = env.observations
            if baseline:
                observations = observations.clone()
                observations[:, :3].add_(env.targets)
                observations[:, 2].sub_(1.0)
            distribution, _, _ = policy.forward_eval(observations)
            env.step(distribution.mean.contiguous())
            valid = (
                active
                & (env.terminated == 0)
                & torch.isfinite(env.final_observations).all(1)
            )
            squared = env.final_observations[:, :3].square().sum(1)
            squared = torch.where(valid, squared, FAILURE_SQUARED_ERROR)
            error_sum.add_(squared)
            short_error.add_(torch.where(short[:, step], squared, 0))
            long_error.add_(torch.where(short[:, step], 0, squared))
            max_error = torch.maximum(max_error, squared.sqrt())
            # Saturation describes executed controls in the first episode, including
            # its failure control; the nonexistent tail has no rotor commands.
            saturation.add_(torch.where(active, env.action_saturation, 0))
            observed_steps.add_(active)
            settled.add_(
                (valid & (env.command_finished > 0) & (env.command_settled > 0)).float()
            )
            active = valid & ((env.truncated == 0) | (step == HORIZON - 1))
        short_count = short.sum()
        long_count = n * HORIZON - short_count
        values = (
            torch.stack(
                [
                    active.float().mean(),
                    (error_sum / HORIZON).sqrt().mean(),
                    (error_sum.sum() / (n * HORIZON)).sqrt(),
                    (short_error.sum() / short_count.clamp_min(1)).sqrt(),
                    (long_error.sum() / long_count.clamp_min(1)).sqrt(),
                    max_error.max(),
                    saturation.sum() / observed_steps.sum().clamp_min(1),
                    settled.sum(),
                    short_count.float(),
                    long_count.float(),
                ]
            )
            .cpu()
            .tolist()
        )
        result = dict(
            zip(
                [
                    "survival_rate",
                    "mean_episode_rms_position_error",
                    "pooled_rms_position_error",
                    "short_interval_rms",
                    "long_interval_rms",
                    "maximum_position_error",
                    "action_saturation_fraction",
                    "settled_commands",
                    "short_steps",
                    "long_steps",
                ],
                values,
            )
        )
        if not result["short_steps"]:
            result["short_interval_rms"] = None
        if not result["long_steps"]:
            result["long_interval_rms"] = None
        result.update(
            episodes=n,
            seed=seed,
            schedule=schedule,
            horizon=HORIZON,
            failure_squared_error=FAILURE_SQUARED_ERROR,
            policy="frozen_hover_checkpoint" if baseline else "tracking_checkpoint",
            saturation_denominator="executed first-episode rotor controls",
        )
        if schedule == "settling":
            result["commanded_targets"] = 4 * n
            result["settled_fraction"] = result["settled_commands"] / (4 * n)
            result["settling_criterion"] = (
                "last 50 consecutive steps of each fixed-500 command: distance <=0.2m and speed <=0.2m/s"
            )
        return result, plan


@torch.no_grad()
def evaluate_suites(
    policy, baseline_policy, n, seed, device, library=None, schedule=None
):
    report = {"episodes_per_suite": n, "seed": seed, "suites": {}}
    for suite in (schedule,) if schedule else ("settling", "mixed"):
        baseline, baseline_plan = evaluate_tracking(
            baseline_policy, n, seed, device, library, suite, True
        )
        evaluation, plan = evaluate_tracking(policy, n, seed, device, library, suite)
        if not torch.equal(plan, baseline_plan):
            raise RuntimeError(
                "Paired tracking policies received different command plans"
            )
        gates = {"survival": evaluation["survival_rate"] >= 0.95}
        if suite == "settling":
            gates["settled_targets"] = evaluation["settled_fraction"] >= 0.90
        else:
            baseline_rms = baseline["mean_episode_rms_position_error"]
            rms = evaluation["mean_episode_rms_position_error"]
            evaluation["rms_baseline_ratio"] = (
                rms / baseline_rms if baseline_rms else None
            )
            gates["rms_improvement"] = math.isfinite(rms) and rms <= 0.75 * baseline_rms
        report["suites"][suite] = {
            "baseline": baseline,
            "evaluation": evaluation,
            "identical_command_plans": True,
            "gates": gates,
            "passed": all(gates.values()),
        }
    report["passed"] = all(suite["passed"] for suite in report["suites"].values())
    report["complete_acceptance_suites"] = schedule is None
    return report
