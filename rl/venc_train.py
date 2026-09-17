"""Train and evaluate visual-encoder ablation variants A-H.

Loads rollouts collected by rl.venc_collect, samples contiguous windows,
and trains with auxiliary objectives:

  probe   MSE on [gap_x, wall_d, vx, vz] (normalized)
  occ     BCE on the 32-column free-space profile
  future  MSE on predicted next-frame features (stop-grad target)

Variants (one config each):

  A  frame-independent CNN, pooled probes
  B  + spatial latent map probes
  C  + ConvGRU temporal fusion
  D  + motion conditioning
  E  + ego-motion warp
  F  + diff channel
  G  + future-feature loss
  H  + tiny causal transformer over g history

Usage:
  python -m rl.venc_train --variant g --data target/venc-data/train \
      --val target/venc-data/val --steps 4000
  python -m rl.venc_train --variant g --bench-only --checkpoint ...
"""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch import nn

from rl.visual_encoder import VisualEncoder, VisualEncoderConfig

CONTROL_DT = 0.05
PROBE_SCALE = torch.tensor([4.5, 10.0, 0.75, 0.75])  # gap_x, wall_d, vx, vz
FUTURE_DIM = 64 * 48  # latent_channels * 6 * 8

VARIANTS = {
    "a": dict(spatial_probe=False, temporal=False, warp=False,
              motion_condition=False, diff_signal=False, future_head=False),
    "b": dict(temporal=False, warp=False, motion_condition=False,
              diff_signal=False, future_head=False),
    "c": dict(warp=False, motion_condition=False, diff_signal=False,
              future_head=False),
    "d": dict(warp=False, diff_signal=False, future_head=False),
    "e": dict(diff_signal=False, future_head=False),
    "f": dict(future_head=False),
    "g": dict(),
    "h": dict(global_temporal=True),
}


class RolloutData:
    """Flat rollout storage -> contiguous per-env segments -> windows."""

    def __init__(self, path, device):
        d = torch.load(path, weights_only=False)
        self.device = device
        self.frames = d["frames"]          # (N,3,H,W) uint8, env-major
        self.poses = d["poses"]            # (N,4)
        self.actions = d["actions"]        # (N,2)
        self.gap_x = d["gap_x"]
        self.wall_d = d["wall_d"]
        self.occ = d["occ"]                # (N,32)
        self.done = d["done"]
        self.env_id = d["env_id"]
        n = self.frames.shape[0]
        seg_start = torch.ones(n, dtype=torch.bool)
        seg_start[1:] = (self.env_id[1:] != self.env_id[:-1]) | self.done[:-1]
        self.seg_starts = seg_start.nonzero().flatten().tolist() + [n]

    def windows(self, length):
        out = []
        for i in range(len(self.seg_starts) - 1):
            s, e = self.seg_starts[i], self.seg_starts[i + 1]
            for w in range(s, e - length + 1):
                out.append((w, w + length))
        return out

    def batch(self, window_list, idx):
        sel = [window_list[i] for i in idx]
        frames = torch.stack([self.frames[s:e] for s, e in sel])
        poses = torch.stack([self.poses[s:e] for s, e in sel])
        actions = torch.stack([self.actions[s:e] for s, e in sel])
        gap = torch.stack([self.gap_x[s:e] for s, e in sel])
        wall = torch.stack([self.wall_d[s:e] for s, e in sel])
        occ = torch.stack([self.occ[s:e] for s, e in sel])
        dx = torch.diff(poses[..., 0], dim=-1, prepend=poses[:, :1, 0])
        dz = torch.diff(poses[..., 1], dim=-1, prepend=poses[:, :1, 1])
        dx[:, 0] = poses[:, 0, 2] * CONTROL_DT
        dz[:, 0] = poses[:, 0, 3] * CONTROL_DT
        motion = torch.stack(
            [dx, dz, poses[..., 2], poses[..., 3],
             torch.full_like(dx, CONTROL_DT)],
            dim=-1,
        )
        probe_target = torch.stack(
            [gap, wall, poses[..., 2], poses[..., 3]], dim=-1
        ) / PROBE_SCALE
        return {
            "frames": frames.to(self.device),
            "motion": motion.to(self.device),
            "actions": actions.to(self.device),
            "probe": probe_target.to(self.device),
            "occ": occ.to(self.device),
        }


def future_target(enc, frames):
    """Encode next frames detached, downsample to the future head's shape."""
    with torch.no_grad():
        f, _ = enc.encode(frames)
        f = F.adaptive_avg_pool2d(f, (6, 8))
        return F.normalize(f.reshape(f.shape[0], -1), dim=-1)


def evaluate(enc, data, windows, batch=64, device="cuda"):
    enc.eval()
    scale = PROBE_SCALE.to(device)
    mae = torch.zeros(4, device=device)
    iou_n, iou_d = 0.0, 0.0
    fut_n, fut_c = 0.0, 0
    n = 0
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            b = data.batch(windows, list(range(i, min(i + batch, len(windows)))))
            outs, _ = enc.forward_sequence(
                b["frames"][:, :-1], b["motion"][:, :-1], b["actions"][:, :-1]
            )
            pred = outs["probe"] * scale
            tgt = b["probe"][:, :-1] * scale
            mae += (pred - tgt).abs().sum(dim=(0, 1))
            p_occ = (outs["occupancy"] > 0).float()
            t_occ = b["occ"][:, :-1]
            iou_n += (p_occ * t_occ).sum().item()
            iou_d += ((p_occ + t_occ) > 0).float().sum().item()
            if "future_f" in outs:
                tgt_f = future_target(enc, b["frames"][:, -1])
                pred_f = F.normalize(
                    outs["future_f"][:, -1].reshape(-1, FUTURE_DIM), dim=-1
                )
                fut_n += F.mse_loss(pred_f, tgt_f, reduction="sum").item()
                fut_c += pred_f.shape[0]
            n += b["frames"].shape[0] * (b["frames"].shape[1] - 1)
    enc.train()
    return {
        "mae_gap_x": (mae[0] / n).item(),
        "mae_wall_d": (mae[1] / n).item(),
        "mae_vx": (mae[2] / n).item(),
        "mae_vz": (mae[3] / n).item(),
        "occ_iou": iou_n / max(iou_d, 1e-9),
        "future_mse": fut_n / max(fut_c, 1),
    }


def count_macs(enc, frame, state, motion, action):
    macs = [0]

    def conv_hook(mod, inp, out):
        k = mod.weight.shape[2] * mod.weight.shape[3] if mod.weight.dim() == 4 else 1
        macs[0] += out.numel() * (mod.in_channels // mod.groups) * k

    def lin_hook(mod, inp, out):
        macs[0] += out.numel() * mod.in_features

    hooks = []
    for m in enc.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    with torch.no_grad():
        enc.update(frame, state, motion, action)
    for h in hooks:
        h.remove()
    return macs[0]


def benchmark(enc, device, iters=200):
    enc.eval()
    state = enc.initial_state(1, device)
    frame = torch.rand(1, 3, *enc.cfg.input_hw, device=device)
    motion = torch.zeros(1, enc.cfg.motion_dim, device=device)
    action = torch.zeros(1, enc.cfg.action_dim, device=device)
    macs = count_macs(enc, frame, state, motion, action)
    with torch.no_grad():
        for _ in range(20):
            _, state = enc.update(frame, state, motion, action)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            _, state = enc.update(frame, state, motion, action)
        end.record()
        torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return {
        "params": sum(p.numel() for p in enc.parameters()),
        "macs": macs,
        "latency_ms": ms,
        "fps": 1000.0 / ms,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 1e6,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    p.add_argument("--data", type=Path, default=Path("target/venc-data/train"))
    p.add_argument("--val", type=Path, default=Path("target/venc-data/val"))
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--window", type=int, default=9)  # 8 updates + 1 future target
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bench-only", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = VisualEncoderConfig(**VARIANTS[args.variant])
    enc = VisualEncoder(cfg).to(device)
    if args.checkpoint:
        enc.load_state_dict(torch.load(args.checkpoint, weights_only=True))

    if args.bench_only:
        print(json.dumps(benchmark(enc, device), indent=2))
        return

    train = RolloutData(args.data / "rollouts.pt", device)
    val = RolloutData(args.val / "rollouts.pt", device)
    train_windows = train.windows(args.window)
    val_windows = val.windows(args.window)
    print(f"train windows: {len(train_windows)}, val: {len(val_windows)}")

    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=1e-4)
    log = []
    for step in range(args.steps):
        idx = torch.randint(0, len(train_windows), (args.batch,)).tolist()
        b = train.batch(train_windows, idx)
        outs, _ = enc.forward_sequence(
            b["frames"][:, :-1], b["motion"][:, :-1], b["actions"][:, :-1]
        )
        loss_probe = F.mse_loss(outs["probe"], b["probe"][:, :-1])
        loss_occ = F.binary_cross_entropy_with_logits(
            outs["occupancy"], b["occ"][:, :-1]
        )
        loss = loss_probe + loss_occ
        logs = {"probe": loss_probe.item(), "occ": loss_occ.item()}
        if "future_f" in outs:
            tgt_f = future_target(enc, b["frames"][:, -1])
            pred_f = F.normalize(
                outs["future_f"][:, -1].reshape(-1, FUTURE_DIM), dim=-1
            )
            loss_fut = F.mse_loss(pred_f, tgt_f)
            loss = loss + 0.5 * loss_fut
            logs["future"] = loss_fut.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == args.steps - 1:
            metrics = evaluate(enc, val, val_windows[:512], device=device)
            metrics["step"] = step
            metrics.update(logs)
            log.append(metrics)
            print(json.dumps(metrics))

    result = {
        "variant": args.variant,
        "config": VARIANTS[args.variant],
        "bench": benchmark(enc, device),
        "final": log[-1],
        "history": log,
    }
    out = args.output or Path(f"target/venc-results/{args.variant}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    ckpt = out.with_suffix(".pt")
    torch.save(enc.state_dict(), ckpt)
    print(f"saved {out} and {ckpt}")


if __name__ == "__main__":
    main()
