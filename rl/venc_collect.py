"""Collect rollouts for visual-encoder training.

Runs RGBEnv at the encoder's native resolution with a scripted policy:
the privileged reference controller (steer to gap) plus noise and random
segments for coverage. Stores per-step:

  frames   (N,3,H,W) uint8   latest frame only (not the 4-frame history)
  poses    (N,4)             x, z, vx, vz
  actions  (N,2)
  gap_x    (N,)              privileged gap center
  wall_d   (N,)              distance to wall plane
  occ      (N,32)            analytic free-space profile across image columns
  env_id   (N,)              env index (for sequence slicing)
  done     (N,)              episode ended this step

Usage:
  python -m rl.venc_collect --output target/venc-data/train --seed 11 \
      --num-envs 64 --steps 1500 --width 128 --height 96
"""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rl.rgb_environment import RGBEnv

# scene constants mirrored from apps/rgb-env/src/lib.rs
WALL_Z = -5.2
WALL_HALF_Z = 0.15
WALL_TOP = 1.2
CORRIDOR_HALF = 4.5
GAP_HALF = 1.5
CAMERA_OFFSET = (0.0, 2.0, 4.5)
CAMERA_TARGET = (0.0, 1.0, -2.0)
FOV_RAD = 1.0471975512  # 60 deg
OCC_COLS = 32


def occupancy_profile(x, z, gap_x, cols=OCC_COLS, aspect=128 / 96):
    """Per-column free-space profile at the image's middle row.

    For each column, cast the camera ray, intersect the wall plane, and mark
    occupied where the ray hits wall geometry. Returns (B, cols) float in {0,1}.
    """
    eye = torch.stack(
        [x, torch.full_like(x, CAMERA_OFFSET[1]), z + CAMERA_OFFSET[2]], dim=-1
    )
    fwd = torch.tensor(
        [CAMERA_TARGET[0], CAMERA_TARGET[1] - CAMERA_OFFSET[1],
         CAMERA_TARGET[2] - CAMERA_OFFSET[2]],
        device=x.device,
    )
    fwd = fwd / fwd.norm()
    right = torch.linalg.cross(fwd, torch.tensor([0.0, 1.0, 0.0], device=x.device))
    right = right / right.norm()

    tan_half = torch.tan(torch.tensor(FOV_RAD / 2, device=x.device))
    ndc = torch.linspace(-1 + 1 / cols, 1 - 1 / cols, cols, device=x.device)
    dirs = fwd[None, :] + (ndc * tan_half * aspect)[:, None] * right[None, :]
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)  # (cols, 3)
    b = x.shape[0]
    dz = dirs[:, 2].unsqueeze(0).expand(b, -1)
    t = (WALL_Z - eye[:, 2:3]) / dz.clamp(max=-1e-3)
    hit_x = eye[:, 0:1] + t * dirs[:, 0].unsqueeze(0).expand(b, -1)
    hit_y = eye[:, 1:2] + t * dirs[:, 1].unsqueeze(0).expand(b, -1)
    in_gap = (hit_x - gap_x[:, None]).abs() < GAP_HALF
    in_wall_x = hit_x.abs() < CORRIDOR_HALF
    in_wall_y = (hit_y > 0.0) & (hit_y < WALL_TOP)
    hits_wall = (t > 0) & in_wall_x & in_wall_y & ~in_gap
    return hits_wall.float()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=96)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--library")
    p.add_argument("--noise", type=float, default=0.4)
    p.add_argument("--random-frac", type=float, default=0.2)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=False)

    with RGBEnv(
        n=args.num_envs,
        seed=args.seed,
        max_steps=args.steps + 1,
        width=args.width,
        height=args.height,
        device=args.device,
        library=args.library,
    ) as env:
        frames, poses, actions, gaps, walld, occs, env_ids, dones = (
            [], [], [], [], [], [], [], []
        )
        random_mask = (torch.rand(env.n) < args.random_frac).to(env.device)
        for step in range(args.steps):
            # reference controller + noise; some envs act randomly
            lateral = (env.gap_centers - env.vehicle_xs).clamp(-1, 1) * 1.5
            act = torch.stack(
                [torch.full_like(lateral, 0.8), lateral], dim=-1
            )
            act = act + torch.randn_like(act) * args.noise
            rand_act = torch.rand(env.n, 2, device=act.device) * 2 - 1
            act = torch.where(random_mask[:, None], rand_act, act).clamp(-1, 1)

            env.step(act)

            pose = env.vehicle_poses  # x, z, vx, vz
            gap = env.gap_centers
            frames.append(env.observations[:, -3:].cpu())
            poses.append(pose.cpu())
            actions.append(act.cpu())
            gaps.append(gap.cpu())
            walld.append((WALL_Z - pose[:, 1]).cpu())
            occs.append(
                occupancy_profile(
                    pose[:, 0], pose[:, 1], gap, aspect=args.width / args.height
                ).cpu()
            )
            env_ids.append(torch.arange(env.n))
            dones.append((env.terminated + env.truncated).cpu() > 0)

            # re-randomize the random subset occasionally
            if step % 25 == 0:
                random_mask = (torch.rand(env.n) < args.random_frac).to(env.device)

        data = {
            "frames": torch.stack(frames).transpose(0, 1).reshape(-1, 3, args.height, args.width),
            "poses": torch.stack(poses).transpose(0, 1).reshape(-1, 4),
            "actions": torch.stack(actions).transpose(0, 1).reshape(-1, 2),
            "gap_x": torch.stack(gaps).transpose(0, 1).reshape(-1),
            "wall_d": torch.stack(walld).transpose(0, 1).reshape(-1),
            "occ": torch.stack(occs).transpose(0, 1).reshape(-1, OCC_COLS),
            "env_id": torch.stack(env_ids).transpose(0, 1).reshape(-1),
            "done": torch.stack(dones).transpose(0, 1).reshape(-1),
        }
        # reshape: (env, step, ...) -> flat, keeping env-major order so
        # contiguous runs per env are recoverable via env_id + done
        path = out_dir / "rollouts.pt"
        torch.save(data, path)
        n = data["frames"].shape[0]
        print(f"saved {n} frames -> {path}")
        print(f"episodes ended: {int(data['done'].sum())}")


if __name__ == "__main__":
    main()
