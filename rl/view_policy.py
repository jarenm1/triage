"""Run a saved policy with optional asynchronous recording/live visualization.

Example: python rl/view_policy.py --task tracking \
  --checkpoint cuda/build/tracking-final-seed11.pt --steps 10000 \
  --environment-ids 0,7,31 --record /tmp/tracking.ndjson \
  --listen 127.0.0.1:9876 --realtime
Repeat without --realtime and with --disabled to measure the headless baseline.
"""

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rl.environment import HoverEnv, TrackingEnv
from rl.snapshot import SnapshotProducer
from rl.train import load_policy, seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("hover", "tracking"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--environment-ids", default="0")
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument(
        "--scenario", choices=("trained", "long-flight-v1"), default="trained"
    )
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--record")
    parser.add_argument("--listen", help="localhost TCP listener, e.g. 127.0.0.1:9876")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="pace at 100 control steps/s outside simulation",
    )
    parser.add_argument(
        "--disabled",
        action="store_true",
        help="identical rollout without staging/IO for baseline comparison",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library")
    args = parser.parse_args()
    try:
        ids = [int(value) for value in args.environment_ids.split(",")]
    except ValueError:
        parser.error("--environment-ids must be comma-separated integers")
    if args.batch <= 0 or args.steps <= 0 or args.sample_every <= 0:
        parser.error("batch, steps and sample-every must be positive")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    if not 0 <= args.seed < 2**64:
        parser.error("seed must fit uint64")
    if (
        not 1 <= len(ids) <= 64
        or len(set(ids)) != len(ids)
        or any(i < 0 or i >= args.batch for i in ids)
    ):
        parser.error("select 1..64 unique IDs within the batch")
    if not 2 <= args.slots <= 16:
        parser.error("slots must be in [2,16]")
    if not args.disabled and not (args.record or args.listen):
        parser.error("provide --record and/or --listen (or --disabled for baseline)")
    if args.scenario != "trained" and args.task != "tracking":
        parser.error("long-flight-v1 requires a tracking checkpoint")
    seed_all(args.seed, args.device)
    policy, checkpoint = load_policy(args.checkpoint, args.task, args.device)
    policy.requires_grad_(False)
    header = {
        "version": 1,
        "scene_version": 1,
        "frame": "ENU_FLU",
        "quaternion": "wxyz",
        "task": args.task if args.scenario == "trained" else "tracking-long-flight-v1",
        "seed": args.seed,
        "control_dt": 0.01,
        "sample_every": args.sample_every,
        "environment_ids": ids,
        "checkpoint": str(Path(args.checkpoint)),
        "camera": {"vertical_fov_radians": 0.87266463, "near": 0.1, "far": 80.0},
    }
    env_type = TrackingEnv if args.task == "tracking" else HoverEnv
    producer = None
    scenario_options = (
        {"schedule": "long-flight-v1"} if args.scenario != "trained" else {}
    )
    with env_type(
        n=args.batch,
        seed=args.seed,
        device=args.device,
        max_steps=6000 if scenario_options else checkpoint["task"]["max_steps"],
        library=args.library,
        **scenario_options,
    ) as env:
        with torch.inference_mode():
            for _ in range(args.warmup):
                distribution, _ = policy(env.observations)
                env.step(distribution.mean.contiguous())
        env.reset(args.seed)
        env.stream.synchronize()
        if not args.disabled:
            producer = SnapshotProducer(
                env, header, record=args.record, listen=args.listen, slots=args.slots
            )
        env.stream.synchronize()
        start = time.perf_counter()
        pacing_seconds = 0.0
        try:
            with torch.inference_mode():
                if producer:
                    producer.submit(0)
                for step in range(1, args.steps + 1):
                    distribution, _ = policy(env.observations)
                    env.step(distribution.mean.contiguous())
                    if producer:
                        producer.poll()
                        if step % args.sample_every == 0:
                            producer.submit(step)
                    if args.realtime:
                        delay = start + step * 0.01 - time.perf_counter()
                        if delay > 0:
                            before = time.perf_counter()
                            time.sleep(delay)
                            pacing_seconds += time.perf_counter() - before
            env.stream.synchronize()
            rollout_seconds = time.perf_counter() - start
        finally:
            drain_start = time.perf_counter()
            if producer:
                producer.close()
            drain_seconds = time.perf_counter() - drain_start
    report = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "scenario": args.scenario,
        "disabled": args.disabled,
        "batch": args.batch,
        "steps": args.steps,
        "warmup": args.warmup,
        "seed": args.seed,
        "sample_every": args.sample_every,
        "environment_ids": ids,
        "rollout_seconds": rollout_seconds,
        "pacing_seconds": pacing_seconds,
        "drain_seconds": drain_seconds,
        "environment_steps_per_second": args.batch * args.steps / rollout_seconds,
        "control_calls_per_second": args.steps / rollout_seconds,
        "snapshot_stats": producer.stats if producer else None,
    }
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
