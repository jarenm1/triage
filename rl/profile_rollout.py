"""Trace a real rollout and reject host transfers or waits inside collection."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rl.environment import HoverEnv
from rl.policy import HoverPolicy
from rl.vendor.torch_pufferl import PuffeRL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument(
        "--trace", type=Path, default=Path("cuda/build/hover-rollout.json")
    )
    args = parser.parse_args()
    if args.num_envs <= 0 or args.horizon <= 0:
        parser.error("num-envs and horizon must be positive")
    config = dict(
        horizon=args.horizon,
        total_timesteps=args.num_envs * args.horizon,
        minibatch_size=args.horizon,
        learning_rate=0.0025,
        beta1=0.9,
        eps=1e-8,
    )
    policy = HoverPolicy().cuda()
    with HoverEnv(args.num_envs) as env:
        learner = PuffeRL(config, env, policy)
        learner.rollouts()
        learner.rollouts()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as trace:
            with torch.profiler.record_function("hover_rollout"):
                learner.rollouts()
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        trace.export_chrome_trace(str(args.trace))
    events = json.loads(args.trace.read_text())["traceEvents"]
    scope = next(
        e for e in events if e.get("name") == "hover_rollout" and e.get("ph") == "X"
    )
    start, end = scope["ts"], scope["ts"] + scope["dur"]
    runtime = Counter(
        e["name"]
        for e in events
        if e.get("ph") == "X"
        and start <= e.get("ts", -1) < end
        and (
            e.get("cat") == "cuda_runtime" or "_local_scalar_dense" in e.get("name", "")
        )
    )
    copies = Counter(e["name"] for e in events if e.get("cat") == "gpu_memcpy")
    kernels = sum(e.get("cat") == "kernel" for e in events)
    forbidden = {
        name: count
        for name, count in runtime.items()
        if "Synchronize" in name
        or "_local_scalar_dense" in name
        or name.startswith(("cudaMalloc", "cudaFree"))
        or name == "cudaMemcpy"
    }
    host_copies = {
        name: count for name, count in copies.items() if "Device -> Device" not in name
    }
    passed = kernels > 0 and not forbidden and not host_copies
    report = dict(
        passed=passed,
        environments=args.num_envs,
        horizon=args.horizon,
        kernel_activities=kernels,
        rollout_runtime_calls=runtime,
        device_copy_activities=copies,
        forbidden_calls=forbidden,
        host_copy_activities=host_copies,
        trace=str(args.trace),
        scope="warmed policy sampling + native environment + rollout writes; excludes learner update, logging, profiler setup/teardown",
    )
    args.trace.with_suffix(".summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
