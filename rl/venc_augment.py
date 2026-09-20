"""GPU-side photometric augmentation for sim->real domain randomization.

Applied to encoder input frames only (teacher/discriminator see clean or
separately-augmented frames). All ops are batched tensor ops — no PIL, no CPU.
Frames are float (B,3,H,W) in [0,1] or uint8.
"""

import torch
import torch.nn.functional as F


def _rand(b, lo, hi, device):
    return torch.empty(b, 1, 1, 1, device=device).uniform_(lo, hi)


def brightness_contrast(x, b_range=(0.7, 1.3), c_range=(0.7, 1.3)):
    b = _rand(x.shape[0], *b_range, x.device)
    c = _rand(x.shape[0], *c_range, x.device)
    return (x - 0.5) * c + 0.5 + (b - 1.0)


def hue_shift(x, max_shift=0.08):
    """Cheap hue jitter: random channel rotation in RGB space."""
    b = x.shape[0]
    t = torch.rand(b, device=x.device) * 2 - 1  # [-1,1]
    t = (t * max_shift).view(b, 1, 1, 1)
    # rotate channels: r->g->b->r scaled by t
    r, g, bl = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return torch.cat(
        [r + t * (g - r), g + t * (bl - g), bl + t * (r - bl)], dim=1
    )


def gaussian_noise(x, std_range=(0.0, 0.08)):
    std = _rand(x.shape[0], *std_range, x.device)
    return x + torch.randn_like(x) * std


def motion_blur(x, max_kernel=7):
    """Horizontal motion blur via depthwise conv, random kernel size."""
    b = x.shape[0]
    k = int(torch.randint(1, max_kernel + 1, (1,)).item())
    if k <= 1:
        return x
    kernel = torch.ones(3, 1, 1, k, device=x.device) / k
    pad_l = (k - 1) // 2
    pad_r = k - 1 - pad_l
    x = F.pad(x, (pad_l, pad_r, 0, 0), mode="replicate")
    return F.conv2d(x, kernel, groups=3)


def vignette(x, strength_range=(0.0, 0.4)):
    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=x.device),
        torch.linspace(-1, 1, w, device=x.device),
        indexing="ij",
    )
    r = (xx**2 + yy**2).clamp(0, 1)
    s = _rand(b, *strength_range, x.device)
    return x * (1 - s * r)


def jpeg_artifacts(x, block=8, strength_range=(0.0, 0.5)):
    """Block-quantization approximation of JPEG artifacts."""
    b, c, h, w = x.shape
    s = _rand(b, *strength_range, x.device)
    small = F.interpolate(
        x, size=(h // block, w // block), mode="bilinear", align_corners=False
    )
    blocky = F.interpolate(small, size=(h, w), mode="nearest")
    return x * (1 - s) + blocky * s


def augment(x, p=0.8):
    """Apply a random subset of augmentations to a batch of frames.

    x: (B,3,H,W) float in [0,1] or uint8. Returns float in [0,1].
    Each op is applied to the whole batch with per-sample parameters.
    """
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    ops = [brightness_contrast, hue_shift, gaussian_noise, vignette,
           jpeg_artifacts]
    for op in ops:
        if torch.rand(()) < p:
            x = op(x)
    if torch.rand(()) < 0.4:  # motion blur less often — it's strong
        x = motion_blur(x)
    return x.clamp(0, 1)
