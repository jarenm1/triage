"""Evaluate a saved RGB checkpoint on a held-out seed; writes the E001 artifact set."""

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rl.policy import RGBPolicy, RecurrentRGBPolicy
from rl.rgb_checkpoint import load_checkpoint
from rl.rgb_environment import RGBEnv
from rl.rgb_replay import canonical, sha256_file, source_info, utc_now, write_json

SCHEMA = "E001/1-rgb-eval-v1"
ROOT = Path(__file__).resolve().parents[1]


def evaluate(args, output):
    source = source_info()
    native = Path(args.library) if args.library else ROOT / "target/release/librgb_env.so"
    native = native.resolve()
    started = utc_now()
    episodes = []
    steps_run = 0
    with RGBEnv(
        n=args.num_envs,
        seed=args.seed,
        max_steps=args.max_steps,
        width=args.width,
        height=args.height,
        device=args.device,
        library=str(native),
    ) as env:
        policy_cls = RecurrentRGBPolicy if args.recurrent else RGBPolicy
        policy = policy_cls(env.observation_shape).to(env.device)
        checkpoint = load_checkpoint(args.checkpoint, policy=policy, env=env)
        state = policy.initial_state(args.num_envs, env.device)
        prev_action = torch.zeros(args.num_envs, env.action_size, device=env.device)
        with torch.inference_mode():
            for step in range(args.steps):
                distribution, _, next_state = policy.forward_eval(
                    env.observations, state, prev_action
                )
                actions = distribution.mean
                env.step(actions)
                steps_run = step + 1
                done = (env.terminated + env.truncated) > 0
                successes = env.successes.detach().to("cpu").numpy()
                terminated = env.terminated.detach().to("cpu").numpy()
                truncated = env.truncated.detach().to("cpu").numpy()
                completed_returns = env.completed_returns.detach().to("cpu").numpy()
                completed_lengths = env.completed_lengths.detach().to("cpu").numpy()
                episode_counts = env.episode_counts.detach().to("cpu").numpy()
                done_np = done.detach().to("cpu").numpy()
                for env_index in range(args.num_envs):
                    if not done_np[env_index]:
                        continue
                    episodes.append(
                        {
                            "step": step,
                            "environment": int(env_index),
                            "episode": int(episode_counts[env_index]),
                            "success": bool(successes[env_index] > 0.5),
                            "terminated": bool(terminated[env_index] > 0.5),
                            "truncated": bool(truncated[env_index] > 0.5),
                            "return": float(completed_returns[env_index]),
                            "length": int(completed_lengths[env_index]),
                        }
                    )
                if getattr(policy, "recurrent", False):
                    reset = done.unsqueeze(-1).float()
                    state = next_state * (1.0 - reset)
                    prev_action = actions * (1.0 - reset)
                if args.episodes and len(episodes) >= args.episodes:
                    break

    (output / "episodes.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in episodes),
        encoding="utf-8",
    )

    total = len(episodes)
    successes = sum(1 for e in episodes if e["success"])
    collisions = sum(1 for e in episodes if e["terminated"] and not e["success"])
    timeouts = sum(1 for e in episodes if e["truncated"])
    returns = [e["return"] for e in episodes]
    lengths = [e["length"] for e in episodes]
    evaluation = {
        "schema": SCHEMA,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "seed": args.seed,
        "split": args.split,
        "episodes": total,
        "successes": successes,
        "collisions": collisions,
        "timeouts": timeouts,
        "success_rate": successes / total if total else None,
        "collision_rate": collisions / total if total else None,
        "timeout_rate": timeouts / total if total else None,
        "mean_return": sum(returns) / total if total else None,
        "mean_length": sum(lengths) / total if total else None,
        "steps_run": steps_run,
        "environments": args.num_envs,
        "source": source,
        "native_library_sha256": sha256_file(native),
        "started_utc": started,
        "finished_utc": utc_now(),
    }
    write_json(output / "evaluation.json", evaluation)
    write_json(
        output / "config.json",
        {
            "schema": SCHEMA,
            "task": "visual_nav_v0",
            "split": args.split,
            "seed": args.seed,
            "batch": args.num_envs,
            "resolution": [args.height, args.width],
            "max_steps": args.max_steps,
            "policy": "RecurrentRGBPolicy" if args.recurrent else "RGBPolicy",
            "action": {"shape_per_environment": [2], "names": ["forward", "lateral"], "range": [-1.0, 1.0]},
            "reward": {"step": -0.002, "forward_velocity": 0.08, "collision": -10.0, "success": 1.0},
        },
    )
    write_json(
        output / "seeds.json",
        {
            "root_seed": args.seed,
            "named_streams": {
                "environment_reset": "native reset seed, per-environment slot order",
                "appearance": "seeded per-env color hash",
                "geometry": "seeded per-(env, episode) obstacle jitter and wall gap",
                "policy": "checkpoint weights; deterministic mean actions",
            },
        },
    )
    write_json(
        output / "manifest.json",
        {
            "schema": SCHEMA,
            "experiment": "E001/1",
            "run": output.name,
            "status": "completed",
            "source": source,
            "hardware": {
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "device": str(env.device),
                "gpu": torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else None,
            },
            "native_library": str(native),
            "native_library_sha256": sha256_file(native),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": evaluation["checkpoint_sha256"],
            "episodes": "episodes.jsonl",
            "evaluation": "evaluation.json",
            "started_utc": started,
            "finished_utc": evaluation["finished_utc"],
        },
    )
    checksum_lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifacts.sha256":
            checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(output)}")
    checksum_text = "\n".join(checksum_lines) + "\n"
    (output / "artifacts.sha256").write_text(checksum_text, encoding="utf-8")
    evaluation["artifact_set_sha256"] = hashlib.sha256(checksum_text.encode()).hexdigest()
    print(json.dumps(evaluation, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="development", choices=["development", "final"])
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=0, help="stop after N episodes (0 = all steps)")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library")
    parser.add_argument("--recurrent", action="store_true")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.steps <= 0 or args.max_steps <= 0:
        parser.error("num-envs, steps and max-steps must be positive")
    output = args.output
    if output.exists():
        parser.error(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    evaluate(args, output)


if __name__ == "__main__":
    main()
