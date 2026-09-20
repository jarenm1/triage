"""Train and evaluate visual-encoder ablation variants.

Loads rollouts collected by rl.venc_collect, samples contiguous windows,
and trains with auxiliary objectives:

  probe   MSE on [gap_x, wall_d, vx, vz] (normalized)
  occ     BCE on the 32-column free-space profile
  future  MSE on predicted next-frame features (stop-grad target)
  consist cosine loss between F(frame) and F(paired-appearance frame)
  domain  DANN: discriminator on F_t can't tell sim from real footage
  distill cosine loss between projected F_t and DINO teacher features

Variants (one config each):

  A  frame-independent CNN, pooled probes
  B  + spatial latent map probes
  C  + ConvGRU temporal fusion
  D  + motion conditioning
  E  + ego-motion warp
  F  + diff channel
  G  + future-feature loss
  H  + tiny causal transformer over g history
  I  G + paired-appearance consistency loss
  J  I + DINO feature distillation
  K  I + domain-adversarial alignment on real footage (--real)
  L  K + DINO feature distillation (--teacher)

Usage:
  python -m rl.venc_train --variant k --data target/venc-data/train-paired \
      --val target/venc-data/val-paired --real target/venc-data/real/frames.pt
"""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rl.visual_encoder import VisualEncoder, VisualEncoderConfig
from rl.venc_augment import augment
from rl.venc_domain import (
    DomainDiscriminator,
    degrade,
    grad_reverse,
    load_teacher,
)

CONTROL_DT = 0.05
PROBE_SCALE = torch.tensor([4.5, 10.0, 0.75, 0.75])  # gap_x, wall_d, vx, vz
FUTURE_DIM = 64 * 48  # latent_channels * 6 * 8
DINO_MEAN = torch.tensor([0.485, 0.456, 0.406])
DINO_STD = torch.tensor([0.229, 0.224, 0.225])

# cfg = VisualEncoderConfig kwargs; consistency/domain/distill are trainer flags
VARIANTS = {
    "a": dict(cfg=dict(spatial_probe=False, temporal=False, warp=False,
                       motion_condition=False, diff_signal=False,
                       future_head=False)),
    "b": dict(cfg=dict(temporal=False, warp=False, motion_condition=False,
                       diff_signal=False, future_head=False)),
    "c": dict(cfg=dict(warp=False, motion_condition=False, diff_signal=False,
                       future_head=False)),
    "d": dict(cfg=dict(warp=False, diff_signal=False, future_head=False)),
    "e": dict(cfg=dict(diff_signal=False, future_head=False)),
    "f": dict(cfg=dict(future_head=False)),
    "g": dict(cfg=dict()),
    "h": dict(cfg=dict(global_temporal=True)),
    "i": dict(cfg=dict(), consistency=True),
    "j": dict(cfg=dict(distill=True), consistency=True, distill=True),
    "k": dict(cfg=dict(), consistency=True, domain=True),
    "l": dict(cfg=dict(distill=True), consistency=True, domain=True,
              distill=True),
    # M: distill on sim+real (positive alignment), disc is eval-only metric
    "m": dict(cfg=dict(distill=True), consistency=True, distill=True,
              distill_real=True, domain_eval=True),
}


class Teacher:
    """Frozen DINOv2/v3 -> patch token map (B,C,h/p,w/p)."""

    def __init__(self, name, device, checkpoint=None, variant="large"):
        self.net, self.patch, self.embed_dim = load_teacher(
            name, device, checkpoint, variant=variant
        )
        self.device = device
        self.mean = DINO_MEAN.reshape(1, 3, 1, 1).to(device)
        self.std = DINO_STD.reshape(1, 3, 1, 1).to(device)

    @torch.no_grad()
    def features(self, frames, chunk=32):
        outs = []
        for i in range(0, frames.shape[0], chunk):
            x = frames[i : i + chunk]
            x = x.float() / 255.0 if x.dtype == torch.uint8 else x
            x = (x - self.mean) / self.std
            h, w = x.shape[2] // self.patch * self.patch, \
                x.shape[3] // self.patch * self.patch
            x = F.interpolate(x, (h, w), mode="bilinear", align_corners=False)
            out = self.net.forward_features(x)["x_prenorm"]
            out = out[:, 1 + getattr(self.net, "n_storage_tokens", 0):]
            gh, gw = h // self.patch, w // self.patch
            outs.append(
                out.reshape(x.shape[0], gh, gw, -1).permute(0, 3, 1, 2)
            )
        return torch.cat(outs)

class RolloutData:
    """Flat rollout storage -> contiguous per-env segments -> windows."""

    def __init__(self, path, device):
        d = torch.load(path, weights_only=False)
        self.device = device
        # frames stored as JPEG byte blobs (bounded RAM, Drive-friendly size)
        self.frames_jpeg = d.get("frames_jpeg")
        self.frames_len = d.get("frames_len")
        self.frames = d.get("frames")            # legacy raw uint8
        self.pframes_jpeg = d.get("pframes_jpeg")
        self.pframes_len = d.get("pframes_len")
        self.pframes = d.get("pframes")
        self.poses = d["poses"]            # (N,4)
        self.actions = d["actions"]        # (N,2)
        self.gap_x = d["gap_x"]
        self.wall_d = d["wall_d"]
        self.occ = d["occ"]                # (N,32)
        self.done = d["done"]
        self.env_id = d["env_id"]
        n = self.poses.shape[0]
        seg_start = torch.ones(n, dtype=torch.bool)
        seg_start[1:] = (self.env_id[1:] != self.env_id[:-1]) | self.done[:-1]
        self.seg_starts = seg_start.nonzero().flatten().tolist() + [n]

    def _decode(self, packed, lens, s, e):
        """JPEG blobs -> (T,3,H,W) uint8."""
        import io
        from PIL import Image
        out = []
        for i in range(s, e):
            n = int(lens[i])
            img = Image.open(io.BytesIO(bytes(packed[i, :n].numpy())))
            out.append(
                torch.from_numpy(np.array(img)).permute(2, 0, 1)
            )
        return torch.stack(out)

    def _frames(self, s, e):
        if self.frames_jpeg is not None:
            return self._decode(self.frames_jpeg, self.frames_len, s, e)
        return self.frames[s:e]

    def _pframes(self, s, e):
        if self.pframes_jpeg is not None:
            return self._decode(self.pframes_jpeg, self.pframes_len, s, e)
        return self.pframes[s:e]

    def windows(self, length):
        out = []
        for i in range(len(self.seg_starts) - 1):
            s, e = self.seg_starts[i], self.seg_starts[i + 1]
            for w in range(s, e - length + 1):
                out.append((w, w + length))
        return out

    def batch(self, window_list, idx):
        sel = [window_list[i] for i in idx]
        frames = torch.stack([self._frames(s, e) for s, e in sel])
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
        out = {
            "frames": frames.to(self.device),
            "motion": motion.to(self.device),
            "actions": actions.to(self.device),
            "probe": probe_target.to(self.device),
            "occ": occ.to(self.device),
        }
        if self.pframes is not None or self.pframes_jpeg is not None:
            out["pframes"] = torch.stack(
                [self._pframes(s, e) for s, e in sel]
            ).to(self.device)
        return out


def future_target(enc, frames):
    with torch.no_grad():
        f, _ = enc.encode(frames)
        f = F.adaptive_avg_pool2d(f, (6, 8))
        return F.normalize(f.reshape(f.shape[0], -1), dim=-1)


def paired_cosine(enc, frames, pframes):
    with torch.no_grad():
        f1, _ = enc.encode(frames)
        f2, _ = enc.encode(pframes)
        return F.cosine_similarity(
            f1.reshape(f1.shape[0], -1), f2.reshape(f2.shape[0], -1), dim=-1
        ).mean()


def evaluate(enc, data, windows, batch=64, device="cuda", disc=None,
             real_frames=None):
    enc.eval()
    scale = PROBE_SCALE.to(device)
    mae = torch.zeros(4, device=device)
    iou_n, iou_d = 0.0, 0.0
    fut_n, fut_c = 0.0, 0
    cos_n, cos_c = 0.0, 0
    dom_correct, dom_n = 0, 0
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
            if "pframes" in b:
                cos_n += paired_cosine(
                    enc,
                    b["frames"][:, :-1].reshape(-1, *b["frames"].shape[2:]),
                    b["pframes"][:, :-1].reshape(-1, *b["pframes"].shape[2:]),
                ).item() * b["frames"].shape[0]
                cos_c += b["frames"].shape[0]
            if disc is not None:
                f_sim = outs["f"].reshape(-1, *outs["f"].shape[2:])
                dom_correct += (disc(f_sim) < 0).sum().item()
                dom_n += f_sim.shape[0]
            n += b["frames"].shape[0] * (b["frames"].shape[1] - 1)
        if disc is not None and real_frames is not None:
            for j in range(0, real_frames.shape[0], 512):
                f_real, _ = enc.encode(real_frames[j : j + 512].to(device))
                dom_correct += (disc(f_real) > 0).sum().item()
                dom_n += f_real.shape[0]
    enc.train()
    return {
        "mae_gap_x": (mae[0] / n).item(),
        "mae_wall_d": (mae[1] / n).item(),
        "mae_vx": (mae[2] / n).item(),
        "mae_vz": (mae[3] / n).item(),
        "occ_iou": iou_n / max(iou_d, 1e-9),
        "future_mse": fut_n / max(fut_c, 1),
        "paired_cos": cos_n / max(cos_c, 1),
        "domain_acc": dom_correct / max(dom_n, 1),
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
    p.add_argument("--teacher-variant", default="large",
                   choices=["small", "base", "large", "giant"],
                   help="lingbot backbone size")
    p.add_argument("--data", type=Path, default=Path("target/venc-data/train"))
    p.add_argument("--val", type=Path, default=Path("target/venc-data/val"))
    p.add_argument("--teacher", choices=["dinov2", "dinov3", "lingbot"],
                   default="dinov2")
    p.add_argument("--real", type=Path,
                   help="real-frame tensor for domain adversarial (frames.pt)")
    p.add_argument("--distill-dim", type=int, default=0,
                   help="teacher embed dim; 0 = auto (384 dino, 1024 lingbot)")
    p.add_argument("--teacher-ckpt", type=Path,
                   help="local .pth for gated teachers (dinov3)")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--window", type=int, default=9)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bench-only", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--consistency-coef", type=float, default=0.5)
    p.add_argument("--distill-coef", type=float, default=0.5)
    p.add_argument("--weak-disc", action="store_true",
                   help="linear discriminator so the encoder can win DANN")
    p.add_argument("--augment", action="store_true",
                   help="photometric augmentation on encoder input")
    p.add_argument("--viz", action="store_true",
                   help="live rerun viewer: input frame, latent PCA, occ, scalars")
    p.add_argument("--viz-every", type=int, default=25,
                   help="log visuals every N steps")

    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    spec = VARIANTS[args.variant]
    cfg = VisualEncoderConfig(**spec["cfg"])
    teacher = (
        Teacher(args.teacher, device, args.teacher_ckpt,
                variant=args.teacher_variant)
        if spec.get("distill")
        else None
    )
    if teacher is not None:
        cfg.distill_dim = args.distill_dim or teacher.embed_dim

    enc = VisualEncoder(cfg).to(device)
    if args.checkpoint:
        enc.load_state_dict(torch.load(args.checkpoint, weights_only=True))

    if args.bench_only:
        print(json.dumps(benchmark(enc, device), indent=2))
        return

    disc = (
        DomainDiscriminator(cfg.latent_channels, weak=args.weak_disc).to(device)
        if spec.get("domain") or spec.get("domain_eval")
        else None
    )
    disc_opt = (
        torch.optim.AdamW(disc.parameters(), lr=1e-3)
        if disc is not None and spec.get("domain_eval")
        else None
    )
    real_frames = None
    if args.real:
        rd = torch.load(args.real, weights_only=False)
        if "frames_jpeg" in rd:
            # decode JPEG blobs once to a CPU uint8 tensor
            import io
            from PIL import Image
            packed, lens = rd["frames_jpeg"], rd["frames_len"]
            imgs = []
            for i in range(packed.shape[0]):
                n = int(lens[i])
                imgs.append(torch.from_numpy(np.array(
                    Image.open(io.BytesIO(bytes(packed[i, :n].numpy())))
                )).permute(2, 0, 1))
            real_frames = torch.stack(imgs)  # CPU; chunks move to GPU
        else:
            real_frames = rd["frames"]

    train = RolloutData(args.data / "rollouts.pt", device)
    val = RolloutData(args.val / "rollouts.pt", device)
    viz = None
    if args.viz:
        from rl.venc_viz import init_viz, log_eval, log_step
        viz = init_viz()
    train_windows = train.windows(args.window)
    val_windows = val.windows(args.window)
    print(f"train windows: {len(train_windows)}, val: {len(val_windows)}")

    params = list(enc.parameters()) + (
        list(disc.parameters()) if disc is not None and spec.get("domain") else []
    )
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    log = []
    for step in range(args.steps):
        idx = torch.randint(0, len(train_windows), (args.batch,)).tolist()
        b = train.batch(train_windows, idx)
        frames_in = b["frames"][:, :-1]
        if args.augment:
            frames_in = augment(
                frames_in.reshape(-1, *frames_in.shape[2:])
            ).reshape(frames_in.shape)
        outs, _ = enc.forward_sequence(
            frames_in, b["motion"][:, :-1], b["actions"][:, :-1]
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

        if spec.get("consistency") and "pframes" in b:
            flat_p = b["pframes"][:, :-1].reshape(-1, *b["pframes"].shape[2:])
            f_p, _ = enc.encode(flat_p)
            f_t = outs["f"].reshape(-1, *outs["f"].shape[2:])
            loss_con = (
                1
                - F.cosine_similarity(
                    f_t.reshape(f_t.shape[0], -1),
                    f_p.reshape(f_p.shape[0], -1),
                    dim=-1,
                ).mean()
            )
            loss = loss + args.consistency_coef * loss_con
            logs["consist"] = loss_con.item()

        if disc is not None and real_frames is not None and spec.get("domain"):
            # sim features vs real features; GRL reverses into enc
            f_sim = outs["f"].reshape(-1, *outs["f"].shape[2:])
            n_real = min(f_sim.shape[0], real_frames.shape[0])
            ridx = torch.randint(0, real_frames.shape[0], (n_real,))
            f_real, _ = enc.encode(real_frames[ridx].to(device))
            logits = torch.cat(
                [
                    disc(grad_reverse(f_sim[:n_real])),
                    disc(grad_reverse(f_real)),
                ]
            )
            labels = torch.cat(
                [torch.zeros(n_real, device=device),
                 torch.ones(n_real, device=device)]
            )
            loss_dom = F.binary_cross_entropy_with_logits(logits, labels)
            loss = loss + args.domain_coef * loss_dom
            logs["domain"] = loss_dom.item()

        if disc is not None and real_frames is not None and spec.get("domain_eval"):
            # eval-only disc: train to distinguish, no gradient into enc.
            # domain_acc ~0.5 then genuinely means the latent is invariant.
            f_sim = outs["f"].reshape(-1, *outs["f"].shape[2:]).detach()
            n_real = min(f_sim.shape[0], real_frames.shape[0])
            ridx = torch.randint(0, real_frames.shape[0], (n_real,))
            with torch.no_grad():
                f_real, _ = enc.encode(real_frames[ridx].to(device))
            logits = torch.cat([disc(f_sim[:n_real]), disc(f_real)])
            labels = torch.cat(
                [torch.zeros(n_real, device=device),
                 torch.ones(n_real, device=device)]
            )
            loss_disc = F.binary_cross_entropy_with_logits(logits, labels)
            disc_opt.zero_grad()
            loss_disc.backward()
            disc_opt.step()
            logs["domain"] = loss_disc.item()

        if teacher is not None and "distill_f" in outs:
            flat_f = b["frames"][:, :-1].reshape(-1, *b["frames"].shape[2:])
            s_feat = outs["distill_f"].reshape(-1, *outs["distill_f"].shape[2:])
            if spec.get("distill_real") and real_frames is not None:
                # distill on real frames too: positive domain alignment
                n_real = min(flat_f.shape[0], real_frames.shape[0])
                ridx = torch.randint(0, real_frames.shape[0], (n_real,))
                r_f, _ = enc.encode(real_frames[ridx].to(device))
                r_feat = enc.distill_proj(r_f)
                flat_f = torch.cat([flat_f, real_frames[ridx].to(device)])
                s_feat = torch.cat([s_feat, r_feat])

            t_feat = teacher.features(flat_f)
            t_feat = F.interpolate(
                t_feat, s_feat.shape[2:], mode="bilinear", align_corners=False
            )
            loss_dis = (
                1
                - F.cosine_similarity(
                    F.normalize(s_feat, dim=1).reshape(s_feat.shape[0], -1),
                    F.normalize(t_feat, dim=1).reshape(t_feat.shape[0], -1),
                    dim=-1,
                ).mean()
            )
            loss = loss + args.distill_coef * loss_dis
            logs["distill"] = loss_dis.item()

        opt.zero_grad()
        loss.backward()
        opt.step()
        if viz is not None and step % args.viz_every == 0:
            real = None
            if real_frames is not None:
                ri = torch.randint(0, real_frames.shape[0], (1,)).item()
                with torch.no_grad():
                    rf, _ = enc.encode(real_frames[ri : ri + 1].to(device))
                real = (real_frames[ri], rf[0])
            log_step(viz, step, outs, b, logs, real=real)
        if step % 200 == 0 or step == args.steps - 1:
            metrics = evaluate(
                enc, val, val_windows[:512], device=device,
                disc=disc, real_frames=real_frames,
            )
            if viz is not None:
                log_eval(viz, step, metrics)
            metrics["step"] = step
            metrics.update(logs)
            log.append(metrics)
            print(json.dumps(metrics))
            # periodic checkpoint: a timeout must not lose the run
            out_path = args.output or Path(
                f"target/venc-results/{args.variant}.json"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(enc.state_dict(), out_path.with_suffix(".pt"))
            out_path.write_text(json.dumps(
                {"variant": args.variant, "config": spec,
                 "final": log[-1], "history": log}, indent=2
            ))

    result = {
        "variant": args.variant,
        "config": spec,
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
