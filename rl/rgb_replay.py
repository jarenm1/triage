"""Record a reproducible fixed-action RGB environment run."""

import argparse
import datetime as dt
import hashlib
import json
import platform
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from rl.rgb_environment import RGBEnv


SCHEMA = "E000/2-rgb-replay-v1"
ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def write_json(path, value):
    path.write_bytes(canonical(value) + b"\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_info():
    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def png_chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(
        ">I", zlib.crc32(kind + payload) & 0xFFFFFFFF
    )


def write_png(path, rgb):
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("RGB sample must have shape [height,width,3]")
    height, width, _ = rgb.shape
    rows = b"".join(b"\x00" + row.tobytes() for row in rgb)
    payload = b"\x89PNG\r\n\x1a\n"
    payload += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += png_chunk(b"IDAT", zlib.compress(rows, 6))
    payload += png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def latest_rgb(observations):
    array = observations[0, :, :, -3:].detach().to("cpu").numpy()
    return np.clip(array, 0, 255).astype(np.uint8)


def fixed_action(step, n, device):
    forward = 0.70
    lateral = 0.55 if (step // 16) % 2 == 0 else -0.55
    action = torch.tensor([forward, lateral], dtype=torch.float32, device=device)
    return action.expand(n, -1).clone(), [forward, lateral]


def start_memory_sampler(path, device):
    try:
        stream = path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            [
                "nvidia-smi",
                f"--id={device}",
                "--query-gpu=timestamp,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "--loop-ms=20",
            ],
            stdout=stream,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return process, stream
    except (FileNotFoundError, OSError):
        return None, None


def stop_memory_sampler(process, stream):
    if process is not None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if stream is not None:
        stream.close()


def memory_summary(path):
    used = []
    total = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 3:
                continue
            try:
                used.append(int(fields[-2]))
                total.append(int(fields[-1]))
            except ValueError:
                continue
    return {
        "available": bool(used),
        "sample_count": len(used),
        "peak_used_mib": max(used) if used else None,
        "device_total_mib": max(total) if total else None,
    }


def task_config(args):
    return {
        "schema": SCHEMA,
        "task": "visual_nav_v0_diagnostic",
        "batch": args.num_envs,
        "resolution": [args.height, args.width],
        "max_steps": args.max_steps,
        "observation": {
            "layout": "NHWC",
            "shape_per_environment": [args.height, args.width, 12],
            "history_frames": 4,
            "channels": "RGB",
            "terminal_observation": "pre_autoreset_history",
        },
        "action": {
            "shape_per_environment": [2],
            "names": ["forward", "lateral"],
            "range": [-1.0, 1.0],
            "controller": "fixed_action_v1: forward=0.70; lateral alternates +0.55/-0.55 every 16 steps",
        },
        "dynamics": {
            "control_dt_s": 0.05,
            "max_speed": 0.75,
            "vehicle_radius": 0.30,
            "velocity_response": 0.35,
        },
        "reward": {
            "step": -0.002,
            "forward_velocity": 0.08,
            "collision": -1.0,
            "success": 1.0,
        },
        "camera": {
            "vertical_fov_degrees": 60.0,
            "near": 0.05,
            "far": 14.0,
            "eye_offset": [0.0, 2.0, 4.5],
            "target_offset": [0.0, 1.0, -2.0],
        },
        "obstacles": [
            {"center": [-1.4, 0.8, -2.4], "scale": [0.9, 1.6, 0.8]},
            {"center": [1.1, 1.1, -4.2], "scale": [1.5, 2.2, 0.9]},
            {"center": [-0.1, 0.55, -6.1], "scale": [2.5, 1.1, 0.7]},
        ],
    }


def record_run(args, output):
    source = source_info()
    native = Path(args.library) if args.library else ROOT / "target/release/librgb_env.so"
    native = native.resolve()
    started = utc_now()
    memory_path = output / "memory.csv"
    memory_process, memory_stream = start_memory_sampler(memory_path, args.device)
    metrics = []
    episodes = []
    actions = []
    checksums = []
    reset_checksum = None
    try:
        with RGBEnv(
            n=args.num_envs,
            seed=args.seed,
            max_steps=args.max_steps,
            width=args.width,
            height=args.height,
            device=args.device,
            library=str(native),
        ) as env:
            for step in range(args.warmup):
                warmup_actions, _ = fixed_action(step, args.num_envs, env.device)
                env.step(warmup_actions)
            env.reset(args.seed)
            reset_checksum = env.checksum()
            write_png(output / "sample-reset-env-000.png", latest_rgb(env.observations))

            for step in range(args.steps):
                action_tensor, action_values = fixed_action(step, args.num_envs, env.device)
                actions.append({"step": step, "action": action_values})
                env.step(action_tensor)
                timing = dict(env.last_step_timings)
                checksum = env.checksum()
                checksums.append(checksum)
                rewards = env.rewards.detach().to("cpu").numpy()
                terminated = env.terminated.detach().to("cpu").numpy()
                truncated = env.truncated.detach().to("cpu").numpy()
                done = (terminated + truncated) > 0.5
                completed_returns = env.completed_returns.detach().to("cpu").numpy()
                completed_lengths = env.completed_lengths.detach().to("cpu").numpy()
                episode_counts = env.episode_counts.detach().to("cpu").numpy()
                metrics.append(
                    {
                        "step": step,
                        "envs": args.num_envs,
                        "checksum_fnv1a64": f"{checksum:016x}",
                        "reward_sum": float(rewards.sum()),
                        "terminated": int(terminated.sum()),
                        "truncated": int(truncated.sum()),
                        "completed": int(done.sum()),
                        "timing_ms": timing,
                    }
                )
                for env_index in np.flatnonzero(done):
                    episodes.append(
                        {
                            "step": step,
                            "environment": int(env_index),
                            "episode": int(episode_counts[env_index]),
                            "terminated": bool(terminated[env_index] > 0.5),
                            "truncated": bool(truncated[env_index] > 0.5),
                            "return": float(completed_returns[env_index]),
                            "length": int(completed_lengths[env_index]),
                        }
                    )

            write_png(output / "sample-terminal-env-000.png", latest_rgb(env.final_observations))
            write_png(output / "sample-autoreset-env-000.png", latest_rgb(env.observations))
    finally:
        stop_memory_sampler(memory_process, memory_stream)

    (output / "metrics.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in metrics),
        encoding="utf-8",
    )
    (output / "episodes.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in episodes),
        encoding="utf-8",
    )
    (output / "actions.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in actions),
        encoding="utf-8",
    )

    total_ms = [item["timing_ms"]["total_ms"] for item in metrics]
    stage_names = [
        "action_stage_ms",
        "native_step_ms",
        "dynamics_ms",
        "render_readback_ms",
        "history_ms",
        "observation_copy_ms",
        "total_ms",
    ]
    timing_summary = {
        name: {
            "mean": float(np.mean([item["timing_ms"][name] for item in metrics])),
            "p50": float(np.percentile([item["timing_ms"][name] for item in metrics], 50)),
            "p95": float(np.percentile([item["timing_ms"][name] for item in metrics], 95)),
        }
        for name in stage_names
    }
    memory = memory_summary(memory_path)
    trace_payload = {
        "reset_checksum": reset_checksum,
        "checksums": checksums,
        "actions": actions,
        "episodes": episodes,
    }
    trace_digest = hashlib.sha256(canonical(trace_payload)).hexdigest()
    summary = {
        "schema": SCHEMA,
        "status": "ok",
        "run": output.name,
        "source": source,
        "envs": args.num_envs,
        "steps": args.steps,
        "warmup": args.warmup,
        "resolution": [args.width, args.height],
        "seed": args.seed,
        "reset_checksum_fnv1a64": f"{reset_checksum:016x}",
        "final_checksum_fnv1a64": f"{checksums[-1]:016x}",
        "trace_sha256": trace_digest,
        "timing_ms": timing_summary,
        "environment_frames_per_second": args.num_envs * args.steps / (sum(total_ms) / 1_000.0),
        "rendered_rgba_bytes": args.num_envs * args.steps * args.width * args.height * 4,
        "episodes": len(episodes),
        "memory": memory,
        "native_library_sha256": sha256_file(native),
        "started_utc": started,
        "finished_utc": utc_now(),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "config.json", task_config(args))
    write_json(
        output / "seeds.json",
        {
            "root_seed": args.seed,
            "named_streams": {
                "environment_reset": "native reset seed, per-environment slot order",
                "appearance": "deterministic env-index/index color hash in native renderer",
                "fixed_action": "fixed_action_v1",
            },
        },
    )
    write_json(
        output / "manifest.json",
        {
            "schema": SCHEMA,
            "experiment": "E000/2",
            "run": output.name,
            "status": "completed",
            "source": source,
            "hardware": {
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "device": str(torch.device("cuda", args.device) if torch.cuda.is_available() else torch.device("cpu")),
                "gpu": torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else None,
            },
            "native_library": str(native),
            "native_library_sha256": sha256_file(native),
            "configuration": "config.json",
            "seeds": "seeds.json",
            "metrics": "metrics.jsonl",
            "episodes": "episodes.jsonl",
            "trace": "actions.jsonl plus checksum sequence in summary.json",
            "profile": "metrics.jsonl and memory.csv",
            "started_utc": started,
            "finished_utc": summary["finished_utc"],
        },
    )
    (output / "report.md").write_text(
        "\n".join(
            [
                f"# {output.name}",
                "",
                "Fixed-action RGB replay and staged full-loop profile for E000/2.",
                "",
                f"- Source revision: `{source['revision']}`; dirty: `{source['dirty']}`",
                f"- Environments: {args.num_envs}; resolution: {args.width}x{args.height}; measured steps: {args.steps}",
                f"- Trace SHA-256: `{trace_digest}`",
                f"- Mean total step: {timing_summary['total_ms']['mean']:.3f} ms",
                f"- Throughput: {summary['environment_frames_per_second']:.1f} environment-frames/s",
                f"- Peak sampled device memory: {memory['peak_used_mib']} MiB",
                "",
                "The trace records fixed commands, observation checksums, terminal outcomes, and selected reset, terminal, and autoreset frames.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifacts.sha256":
            checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(output)}")
    checksum_text = "\n".join(checksum_lines) + "\n"
    (output / "artifacts.sha256").write_text(checksum_text, encoding="utf-8")
    summary["artifact_set_sha256"] = hashlib.sha256(checksum_text.encode()).hexdigest()
    print(json.dumps(summary, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library")
    args = parser.parse_args()
    if min(args.num_envs, args.steps, args.max_steps, args.width, args.height) <= 0:
        parser.error("environment count, steps, max-steps, width and height must be positive")
    output = Path(args.output).resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.mkdir(parents=True)
    record_run(args, output)


if __name__ == "__main__":
    main()
