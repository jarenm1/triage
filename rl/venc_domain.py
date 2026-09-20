"""Domain-adversarial sim->real alignment for the visual encoder.

Pieces:

  GradReverse      gradient reversal layer (DANN)
  DomainDiscriminator  small conv net: F_t -> {sim, real}
  degrade()        sim-frame augmentation: noise, blur, jitter, JPEG-ish
                   quantization — pushes sim statistics toward real footage
  load_teacher()   DINOv2 (open) or DINOv3 (gated; needs HF token)

Real footage has no pose/action labels, so it participates only through
the discriminator and (optionally) teacher features — never probes.
"""

import torch
import torch.nn.functional as F
from torch import nn


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class DomainDiscriminator(nn.Module):
    """Domain classifier on the latent map F_t.

    `weak=True` uses a linear head on mean-pooled features — deliberately
    underpowered so the encoder can actually win the adversarial game
    against a small real dataset.
    """

    def __init__(self, channels=64, weak=False):
        super().__init__()
        self.weak = weak
        if weak:
            self.net = nn.Linear(channels, 1)
        else:
            self.net = nn.Sequential(
                nn.Conv2d(channels, 64, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(64, 64, 3, 2, 1),
                nn.GELU(),
                nn.Conv2d(64, 1, 1),
            )

    def forward(self, f):
        if self.weak:
            return self.net(f.mean(dim=(2, 3))).squeeze(1)
        return self.net(f).mean(dim=(2, 3)).squeeze(1)  # (B,) logits



def degrade(frames, rng=None):
    """Sim -> pseudo-real degradation. frames uint8 or float [0,1], (B,3,H,W).

    Applies a random subset of: gaussian noise, gaussian blur, brightness/
    contrast jitter, channel shift, coarse quantization (JPEG-ish).
    """
    x = frames.float() / 255.0 if frames.dtype == torch.uint8 else frames.clone()
    b = x.shape[0]
    dev = x.device
    r = torch.rand(b, 5, device=dev)

    # gaussian noise
    sigma = (r[:, 0] * 0.05).reshape(b, 1, 1, 1)
    x = x + torch.randn_like(x) * sigma

    # blur: depthwise 3x3 gaussian applied to a random subset
    k = torch.tensor([1.0, 2.0, 1.0], device=dev)
    k = (k[:, None] @ k[None, :]) / 16.0
    k = k.reshape(1, 1, 3, 3).expand(3, 1, 3, 3)
    blurred = F.conv2d(x, k, padding=1, groups=3)
    w = (r[:, 1] < 0.5).float().reshape(b, 1, 1, 1)
    x = w * blurred + (1 - w) * x

    # brightness/contrast jitter
    gain = 1.0 + (r[:, 2] - 0.5) * 0.5
    bias = (r[:, 3] - 0.5) * 0.2
    x = x * gain.reshape(b, 1, 1, 1) + bias.reshape(b, 1, 1, 1)

    # coarse quantization (JPEG-ish blocking)
    levels = torch.where(r[:, 4] < 0.4, 24.0, 255.0).reshape(b, 1, 1, 1)
    x = torch.round(x.clamp(0, 1) * levels) / levels

    return x.clamp(0, 1)


def load_teacher(name, device, checkpoint=None, variant="large"):
    """name: 'dinov2' | 'dinov3' | 'lingbot'.

    dinov3 needs `checkpoint` (local .pth, gated). lingbot uses the vendored
    lingbot_vision loader; `variant` picks small/base/large/giant.
    Returns (net, patch_size, embed_dim).
    """
    if name == "lingbot":
        from rl.vendor.lingbot_vision.loader import load_pretrained_backbone

        net, embed_dim = load_pretrained_backbone(
            variant=variant, device=device, dtype=torch.float32
        )
        patch = net.patch_size
        return net, patch, embed_dim
    if name == "dinov3":
        net = torch.hub.load(
            "facebookresearch/dinov3", "dinov3_vits16",
            pretrained=False,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        patch, embed_dim = 16, 384
    elif name == "dinov2":
        net = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
        )
        patch, embed_dim = 14, 384
    else:
        raise ValueError(f"unknown teacher {name}")
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, patch, embed_dim

