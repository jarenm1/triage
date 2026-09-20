"""Pack raw real-frame tensors into JPEG-blob format for Drive upload.

Usage:
    python -m rl.venc_pack_real --in target/venc-data/real/frames-256.pt \
        --out target/venc-data/real/real-256.pt
"""

import argparse
import io
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="src", type=Path, required=True)
    p.add_argument("--out", dest="dst", type=Path, required=True)
    args = p.parse_args()

    d = torch.load(args.src, weights_only=False)
    frames = d["frames"]  # (N,3,H,W) uint8
    blobs, lens = [], []
    for i in range(frames.shape[0]):
        b = io.BytesIO()
        Image.fromarray(frames[i].permute(1, 2, 0).numpy()).save(
            b, format="JPEG", quality=88
        )
        blobs.append(b.getvalue())
        lens.append(len(b.getvalue()))
        if i % 2000 == 0:
            print(f"{i}/{frames.shape[0]}", flush=True)
    max_len = max(lens)
    packed = torch.zeros(len(blobs), max_len, dtype=torch.uint8)
    for i, b in enumerate(blobs):
        packed[i, : len(b)] = torch.frombuffer(b, dtype=torch.uint8)
    torch.save(
        {"frames_jpeg": packed, "frames_len": torch.tensor(lens),
         "files": d.get("files")},
        args.dst,
    )
    print(f"saved {len(blobs)} frames -> {args.dst} "
          f"({args.dst.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
