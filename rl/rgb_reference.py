"""Privileged-state reference controller for the E001 course.

Steers toward the true wall-gap center using simulator state (gap_centers,
vehicle_xs) rather than rendered observations. Proves the course is solvable
and gives the RGB policy a diagnostic ceiling: if the reference succeeds and
the RGB policy fails, the gap is a vision problem, not a task problem.
"""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rl.rgb_environment import RGBEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library")
    parser.add_argument("--forward", type=float, default=0.8)
    parser.add_argument("--gain", type=float, default=1.5)
    args = parser.parse_args()

    with RGBEnv(
        n=args.num_envs,
        seed=args.seed,
        max_steps=args.max_steps,
        device=args.device,
        library=args.library,
    ) as env:
        episodes = []
        for step in range(args.steps):
            error = env.gap_centers - env.vehicle_xs
            actions = torch.stack(
                [
                    torch.full_like(error, args.forward),
                    (args.gain * error).clamp(-1.0, 1.0),
                ],
                dim=1,
            )
            env.step(actions)
            done = (env.terminated + env.truncated) > 0
            successes = env.successes.detach().to("cpu").numpy()
            terminated = env.terminated.detach().to("cpu").numpy()
            truncated = env.truncated.detach().to("cpu").numpy()
            returns = env.completed_returns.detach().to("cpu").numpy()
            lengths = env.completed_lengths.detach().to("cpu").numpy()
            counts = env.episode_counts.detach().to("cpu").numpy()
            done_np = done.detach().to("cpu").numpy()
            for i in range(args.num_envs):
                if done_np[i]:
                    episodes.append(
                        {
                            "environment": i,
                            "episode": int(counts[i]),
                            "success": bool(successes[i] > 0.5),
                            "terminated": bool(terminated[i] > 0.5),
                            "truncated": bool(truncated[i] > 0.5),
                            "return": float(returns[i]),
                            "length": int(lengths[i]),
                        }
                    )

    total = len(episodes)
    successes = sum(1 for e in episodes if e["success"])
    collisions = sum(1 for e in episodes if e["terminated"] and not e["success"])
    timeouts = sum(1 for e in episodes if e["truncated"])
    print(
        json.dumps(
            {
                "status": "ok",
                "controller": "privileged_gap_steering",
                "seed": args.seed,
                "episodes": total,
                "successes": successes,
                "collisions": collisions,
                "timeouts": timeouts,
                "success_rate": successes / total if total else None,
                "mean_return": (
                    sum(e["return"] for e in episodes) / total if total else None
                ),
                "mean_length": (
                    sum(e["length"] for e in episodes) / total if total else None
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
