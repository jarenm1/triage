"""Post-hoc evaluation for trained encoder checkpoints.

Adds the occlusion-retention probe on top of the standard metrics: blank N
frames mid-sequence and measure how much probe accuracy degrades — the
property the persistent state exists for.

Usage:
  python -m rl.venc_eval --checkpoint target/venc-results/g.pt --variant g
"""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rl.visual_encoder import VisualEncoder, VisualEncoderConfig
from rl.venc_train import RolloutData, VARIANTS, PROBE_SCALE, evaluate


def occlusion_eval(enc, data, windows, blank_len=4, batch=64, device="cuda"):
    """Blank `blank_len` frames mid-window; report probe MAE after the gap.

    Compares against the same windows unblanked. A recurrent encoder should
    degrade gracefully; a frame-independent one cannot recover.
    """
    enc.eval()
    scale = PROBE_SCALE.to(device)
    mae_blank = torch.zeros(4, device=device)
    mae_ctrl = torch.zeros(4, device=device)
    n = 0
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            b = data.batch(windows, list(range(i, min(i + batch, len(windows)))))
            t = b["frames"].shape[1]
            mid = t // 2
            blanked = b["frames"].clone()
            blanked[:, mid : mid + blank_len] = 0

            outs_b, _ = enc.forward_sequence(
                blanked, b["motion"], b["actions"]
            )
            outs_c, _ = enc.forward_sequence(
                b["frames"], b["motion"], b["actions"]
            )
            tgt = b["probe"] * scale
            last_blank = mid + blank_len - 1
            mae_blank += (
                (outs_b["probe"][:, last_blank] * scale - tgt[:, last_blank])
                .abs().sum(dim=0)
            )
            mae_ctrl += (
                (outs_c["probe"][:, last_blank] * scale - tgt[:, last_blank])
                .abs().sum(dim=0)
            )
            n += b["frames"].shape[0]
    return {
        "occ_mae_gap_x": (mae_blank[0] / n).item(),
        "occ_mae_wall_d": (mae_blank[1] / n).item(),
        "ctrl_mae_gap_x": (mae_ctrl[0] / n).item(),
        "ctrl_mae_wall_d": (mae_ctrl[1] / n).item(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    p.add_argument("--val", type=Path, default=Path("target/venc-data/val"))
    p.add_argument("--window", type=int, default=12)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    sd = torch.load(args.checkpoint, weights_only=True)
    cfg = VisualEncoderConfig(**VARIANTS[args.variant]["cfg"])
    if "distill_proj.weight" in sd:
        cfg.distill_dim = sd["distill_proj.weight"].shape[0]
    enc = VisualEncoder(cfg).to(device)
    enc.load_state_dict(sd)

    val = RolloutData(args.val / "rollouts.pt", device)
    windows = val.windows(args.window)[:512]
    result = {
        "variant": args.variant,
        "standard": evaluate(enc, val, windows, device=device),
        "occlusion": occlusion_eval(enc, val, windows, device=device),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
