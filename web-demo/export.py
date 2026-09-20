"""Export the visual encoder to ONNX + a demo rollout for the web demo.

Produces in web-demo/:
  model.onnx   update() step: (frame, h, motion, action) -> (h_new, occ, probe, f)
  frames.jpg   sprite sheet of rollout frames (grid of 128x96 tiles)
  meta.json    rollout meta: grid dims, motion/action sequences, PCA matrix

Usage:
  python web-demo/export.py --checkpoint target/venc-results/m-vitb.pt \
      --data target/venc-data/val-fpv/rollouts.pt --frames 240
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn

from rl.visual_encoder import VisualEncoder, VisualEncoderConfig


class UpdateStep(nn.Module):
    """Single-step wrapper with flat tensor inputs/outputs for ONNX."""

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, frame, h, motion, action):
        out, new_state = self.enc.update(
            frame, {"h": h}, motion, action
        )
        return new_state["h"], out["occupancy"], out["probe"], out["f"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path,
                   default=Path("target/venc-data/val-fpv/rollouts.pt"))
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--cols", type=int, default=16)
    p.add_argument("--out", type=Path, default=Path("web-demo"))
    p.add_argument("--real", type=Path,
                   default=Path("target/venc-data/real/frames.pt"),
                   help="real FPV frames for the sim/real toggle")
    args = p.parse_args()
    cfg = VisualEncoderConfig(distill=True, distill_dim=768)


    enc = VisualEncoder(cfg)
    enc.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    enc.eval()

    step = UpdateStep(enc).eval()
    h_lat, w_lat = enc.latent_hw()
    frame = torch.zeros(1, 3, *cfg.input_hw)
    h = torch.zeros(1, cfg.latent_channels, h_lat, w_lat)
    motion = torch.zeros(1, cfg.motion_dim)
    action = torch.zeros(1, cfg.action_dim)
    onnx_path = args.out / "model.onnx"
    torch.onnx.export(
        step, (frame, h, motion, action), onnx_path,
        input_names=["frame", "h", "motion", "action"],
        output_names=["h_new", "occupancy", "probe", "f"],
        opset_version=17,
    )
    # embed weights: ORT-web can't fetch external .data files
    import onnx
    m = onnx.load(onnx_path)
    onnx.save_model(m, onnx_path, save_as_external_data=False)
    data_file = onnx_path.with_suffix(".onnx.data")
    if data_file.exists():
        data_file.unlink()
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    # ---- rollout: one contiguous segment ----
    d = torch.load(args.data, weights_only=False)
    env_id, done = d["env_id"], d["done"]
    # find first segment with >= args.frames steps
    n = env_id.shape[0]
    seg_start = 0
    start, end = 0, min(args.frames, n)
    for i in range(1, n):
        if env_id[i] != env_id[i - 1] or done[i - 1]:
            if i - seg_start >= args.frames:
                start, end = seg_start, seg_start + args.frames
                break
            seg_start = i

    frames = d["frames"][start:end]          # (T,3,96,128) uint8
    poses = d["poses"][start:end]            # (T,4)
    actions = d["actions"][start:end]        # (T,2)
    occ = d["occ"][start:end]                # (T,32)
    t = frames.shape[0]
    print(f"rollout segment: {t} frames")

    # motion: [dx, dz, vx, vz, dt] matching RolloutData.batch
    dt = 0.05
    dx = torch.diff(poses[:, 0], prepend=poses[:1, 0])
    dz = torch.diff(poses[:, 1], prepend=poses[:1, 1])
    dx[0], dz[0] = poses[0, 2] * dt, poses[0, 3] * dt
    motion_seq = torch.stack(
        [dx, dz, poses[:, 2], poses[:, 3], torch.full_like(dx, dt)], dim=-1
    )

    # ---- sprite sheet ----
    from PIL import Image
    th, tw = cfg.input_hw
    cols = args.cols
    rows = (t + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th))
    for i in range(t):
        img = Image.fromarray(frames[i].permute(1, 2, 0).numpy())
        sheet.paste(img, ((i % cols) * tw, (i // cols) * th))
    sheet.save(args.out / "frames.jpg", quality=88)
    print(f"wrote frames.jpg ({(args.out / 'frames.jpg').stat().st_size / 1e3:.0f} KB)")

    # ---- PCA projection for latent viz (fit on this rollout) ----
    with torch.no_grad():
        state = enc.initial_state(1)
        feats = []
        for i in range(t):
            f_t, _ = enc.encode(frames[i : i + 1].float() / 255.0)
            feats.append(f_t[0].reshape(cfg.latent_channels, -1).T)
            out, state = enc.update(
                frames[i : i + 1].float() / 255.0, state,
                motion_seq[i : i + 1], actions[i : i + 1],
            )
        x = torch.cat(feats)  # (T*HW, C)
        x = x - x.mean(0, keepdim=True)
        _, _, v = torch.pca_lowrank(x, q=3)
        pca = v.numpy().astype(np.float32)  # (C,3)

    meta = {
        "frames": t, "cols": cols, "tile_w": tw, "tile_h": th,
        "latent_hw": [h_lat, w_lat], "latent_channels": cfg.latent_channels,
        "motion": motion_seq.tolist(),
        "actions": actions.tolist(),
        "occ_target": occ.tolist(),
        "pca": pca.tolist(),
    }
    (args.out / "meta.json").write_text(json.dumps(meta))
    print(f"wrote meta.json ({(args.out / 'meta.json').stat().st_size / 1e3:.0f} KB)")

    # ---- real FPV frames: same sprite format, zero motion/action ----
    if args.real.exists():
        rd = torch.load(args.real, weights_only=False)
        rframes = rd["frames"][: args.frames]  # (T,3,96,128) uint8
        rt = rframes.shape[0]
        rsheet = Image.new("RGB", (cols * tw, rows * th))
        for i in range(rt):
            img = Image.fromarray(rframes[i].permute(1, 2, 0).numpy())
            rsheet.paste(img, ((i % cols) * tw, (i // cols) * th))
        rsheet.save(args.out / "real.jpg", quality=88)
        rmeta = {
            "frames": rt, "cols": cols, "tile_w": tw, "tile_h": th,
            "latent_hw": [h_lat, w_lat], "latent_channels": cfg.latent_channels,
            "motion": [[0.0] * cfg.motion_dim] * rt,
            "actions": [[0.0] * cfg.action_dim] * rt,
            "occ_target": None,
            "pca": pca.tolist(),
        }
        (args.out / "real.json").write_text(json.dumps(rmeta))
        print(f"wrote real.jpg + real.json ({rt} real frames)")


if __name__ == "__main__":
    main()
