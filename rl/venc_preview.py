"""Visual scene preview: render a grid of env frames to a PNG.

Usage:
    python -m rl.venc_preview --envs 16 --steps 0 --seed 11 --out preview.png
    python -m rl.venc_preview --envs 8 --steps 40 --seed 11 --out preview.png
    python -m rl.venc_preview --envs 8 --steps 40 --seed 11 --paired --out paired.png

--steps N renders after N forward steps (default 0 = initial frame).
--paired shows the variant-1 (alternate appearance) render.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rl.rgb_environment import RGBEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--paired", action="store_true",
                   help="render the variant-1 (alternate appearance) view")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("target/preview.png"))
    args = p.parse_args()

    env = RGBEnv(args.envs, seed=args.seed, width=128, height=96,
                 paired=args.paired)
    env.reset()
    if args.steps > 0:
        actions = torch.zeros(args.envs, 2)
        actions[:, 0] = 0.3  # gentle forward drift
        for _ in range(args.steps):
            env.step(actions)

    obs = env.paired_observations if args.paired else env.observations
    # latest frame of the 4-frame stack: channels 9:12
    frames = obs[:, 9:12].float() / 255.0  # (N,3,H,W)

    n, c, h, w = frames.shape
    cols = min(args.cols, n)
    rows = (n + cols - 1) // cols
    grid = torch.zeros(3, rows * h, cols * w)
    for i in range(n):
        r, cidx = divmod(i, cols)
        grid[:, r * h:(r + 1) * h, cidx * w:(cidx + 1) * w] = frames[i]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    ).save(args.out)
    print(f"saved {args.out} ({n} envs, {args.steps} steps, "
          f"{'paired' if args.paired else 'primary'} view)")


if __name__ == "__main__":
    main()
