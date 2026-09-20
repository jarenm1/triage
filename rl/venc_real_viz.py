"""Log real-FPV frames + encoder latents to rerun from a checkpoint.

Connects to the same rerun app as training (app_id "venc-train") so the
panels appear alongside the live training stream.

Usage:
    python -m rl.venc_real_viz --checkpoint target/venc-results/m-vitb-real23k.pt
"""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from rl.visual_encoder import VisualEncoder, VisualEncoderConfig
from rl.venc_viz import _pca_rgb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--real", type=Path,
                   default=Path("target/venc-data/real/frames.pt"))
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--connect", default="rerun+http://127.0.0.1:9876/proxy")
    args = p.parse_args()

    import rerun as rr
    rr.init("venc-train")
    rr.connect_grpc(args.connect)

    cfg = VisualEncoderConfig(distill=True, distill_dim=768)
    enc = VisualEncoder(cfg)
    enc.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    enc.eval().cuda()

    frames = torch.load(args.real, weights_only=False)["frames"]
    idx = torch.randperm(frames.shape[0])[: args.n]

    with torch.no_grad():
        for i, j in enumerate(idx):
            fr = frames[j]
            f, _ = enc.encode(fr.unsqueeze(0).cuda())
            rr.set_time("sample", sequence=i)
            rr.log("real/frame", rr.Image(fr.permute(1, 2, 0).numpy()))
            rr.log("real/latent_pca", rr.Image(_pca_rgb(f[0])))
    print(f"logged {args.n} real frames from {args.checkpoint}")


if __name__ == "__main__":
    main()
