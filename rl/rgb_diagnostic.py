"""Measure RGB policy dependence on correctly paired visual observations."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rl.policy import RGBPolicy, RecurrentRGBPolicy
from rl.rgb_checkpoint import load_checkpoint
from rl.rgb_environment import RGBEnv


MODES = ("normal", "shuffle", "blank")


def transformed_observations(observations, mode):
    if mode == "normal":
        return observations
    if mode == "shuffle":
        return torch.roll(observations, shifts=1, dims=0)
    if mode == "blank":
        return torch.zeros_like(observations)
    raise ValueError(f"unknown diagnostic mode: {mode}")


@torch.inference_mode()
def run_mode(args, mode):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        torch.cuda.manual_seed_all(args.seed)
    with RGBEnv(
        n=args.num_envs,
        seed=args.seed,
        max_steps=args.max_steps,
        width=args.width,
        height=args.height,
        device=args.device,
        library=args.library,
    ) as env:
        policy_cls = RecurrentRGBPolicy if args.recurrent else RGBPolicy
        policy = policy_cls(env.observation_shape).to(env.device)
        load_checkpoint(args.checkpoint, policy=policy, env=env)
        reward_sum = torch.zeros((), device=env.device)
        action_sum = torch.zeros(env.action_size, device=env.device)
        action_abs_sum = torch.zeros(env.action_size, device=env.device)
        terminated = torch.zeros((), device=env.device)
        truncated = torch.zeros((), device=env.device)
        completed = torch.zeros((), device=env.device)
        completed_return_sum = torch.zeros((), device=env.device)
        completed_length_sum = torch.zeros((), device=env.device)
        state = policy.initial_state(args.num_envs, env.device)
        prev_action = torch.zeros(args.num_envs, env.action_size, device=env.device)
        for _ in range(args.steps):
            observations = transformed_observations(env.observations, mode)
            distribution, _, state = policy.forward_eval(
                observations, state, prev_action
            )
            actions = distribution.mean
            action_sum += actions.sum(dim=0)
            action_abs_sum += actions.abs().sum(dim=0)
            env.step(actions)
            done = (env.terminated + env.truncated) > 0
            if getattr(policy, "recurrent", False):
                reset = done.unsqueeze(-1).float()
                state = state * (1.0 - reset)
                prev_action = actions * (1.0 - reset)
            reward_sum += env.rewards.sum()
            terminated += env.terminated.sum()
            truncated += env.truncated.sum()
            completed += done.sum()
            completed_return_sum += (env.completed_returns * done).sum()
            completed_length_sum += (env.completed_lengths * done).sum()
        count = float(args.steps * args.num_envs)
        episode_count = completed.item()
        return {
            "mode": mode,
            "steps": args.steps,
            "environments": args.num_envs,
            "mean_reward_per_step": reward_sum.item() / count,
            "terminated": int(terminated.item()),
            "truncated": int(truncated.item()),
            "completed_episodes": int(episode_count),
            "mean_completed_return": (
                completed_return_sum.item() / episode_count if episode_count else None
            ),
            "mean_completed_length": (
                completed_length_sum.item() / episode_count if episode_count else None
            ),
            "mean_action": (action_sum / count).cpu().tolist(),
            "mean_abs_action": (action_abs_sum / count).cpu().tolist(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library")
    parser.add_argument("--recurrent", action="store_true")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.steps <= 0 or args.max_steps <= 0:
        parser.error("num-envs, steps and max-steps must be positive")
    results = [run_mode(args, mode) for mode in args.modes]
    print(
        json.dumps(
            {
                "status": "ok",
                "checkpoint": str(args.checkpoint),
                "results": results,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
