"""Temporally persistent visual-inertial encoder for the E001 corridor task.

Design: docs/visual-encoder.md. Compress early (stride-8 stem), keep temporal
compute on a small spatial latent (16x12x64 at 128x96 input), align the previous
state with a learned ego-motion warp before recurrent fusion, and expose state
as a first-class API:

    state = enc.initial_state(batch, device)
    state = enc.propagate(state, motion)          # warp only (IMU-rate step)
    out   = enc.update(frame, state, motion, action)  # encode + fuse (camera step)

`motion` is [dx, dz, vx, vz, dt] per env. Ablation flags live on
VisualEncoderConfig; every variant A-H is one config.
"""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class VisualEncoderConfig:
    input_hw: tuple = (96, 128)  # (H, W)
    stem_channels: tuple = (24, 32, 64)
    latent_channels: int = 64
    context_channels: int = 96
    motion_dim: int = 5  # dx, dz, vx, vz, dt
    motion_embed: int = 64
    global_dim: int = 128
    action_dim: int = 2
    # ablation flags
    spatial_probe: bool = True   # B: probes read the spatial map, not just g
    temporal: bool = True        # C: ConvGRU fusion (off = frame-independent)
    motion_condition: bool = True  # D: m_t biases fusion
    warp: bool = True            # E: propagate() aligns H by learned flow
    diff_signal: bool = True     # F: |F_t - H'| input channel
    future_head: bool = True     # G: predict next-frame features (loss flag)
    global_temporal: bool = False  # H: tiny causal transformer over g history
    global_temporal_len: int = 8
    separable_gates: bool = True
    occupancy_hw: tuple = (24, 32)


def _dw_block(c_in, c_out, stride=1):
    """Depthwise-separable residual block."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_in, 3, stride, 1, groups=c_in, bias=False),
        nn.BatchNorm2d(c_in),
        nn.GELU(),
        nn.Conv2d(c_in, c_out, 1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.GELU(),
    )


class Stem(nn.Module):
    """Stride-8 conv stem -> spatial features F_t plus a /16 context map."""

    def __init__(self, cfg: VisualEncoderConfig):
        super().__init__()
        c = cfg.stem_channels
        self.blocks = nn.Sequential(
            nn.Conv2d(3, c[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.GELU(),
            _dw_block(c[0], c[0]),
            nn.Conv2d(c[0], c[1], 3, 2, 1, bias=False),
            nn.BatchNorm2d(c[1]),
            nn.GELU(),
            _dw_block(c[1], c[1]),
            nn.Conv2d(c[1], c[2], 3, 2, 1, bias=False),
            nn.BatchNorm2d(c[2]),
            nn.GELU(),
            _dw_block(c[2], cfg.latent_channels),
        )
        self.context = nn.Sequential(
            nn.Conv2d(cfg.latent_channels, cfg.context_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.context_channels),
            nn.GELU(),
        )

    def forward(self, frame):
        f = self.blocks(frame)
        return f, self.context(f)


class MotionEncoder(nn.Module):
    def __init__(self, cfg: VisualEncoderConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.motion_dim, cfg.motion_embed),
            nn.GELU(),
            nn.Linear(cfg.motion_embed, cfg.motion_embed),
            nn.GELU(),
        )

    def forward(self, motion):
        return self.net(motion)


class LearnedWarp(nn.Module):
    """Predicts a per-cell 2D flow field for H from (H, motion embedding).

    Zero-initialized output layer: identity warp at init. propagate() also
    returns a validity mask (1 where the sampled source cell was in-frame).
    """

    def __init__(self, cfg: VisualEncoderConfig):
        super().__init__()
        c = cfg.latent_channels
        self.flow = nn.Sequential(
            nn.Conv2d(c + cfg.motion_embed, c, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(c, 2, 3, 1, 1),
        )
        nn.init.zeros_(self.flow[-1].weight)
        nn.init.zeros_(self.flow[-1].bias)
        self.motion_proj = nn.Linear(cfg.motion_embed, cfg.motion_embed)

    def forward(self, h, m):
        b, c, hh, ww = h.shape
        mb = self.motion_proj(m).reshape(b, -1, 1, 1).expand(b, -1, hh, ww)
        flow = self.flow(torch.cat([h, mb], dim=1))  # pixels, in latent cells
        ys, xs = torch.meshgrid(
            torch.arange(hh, device=h.device, dtype=h.dtype),
            torch.arange(ww, device=h.device, dtype=h.dtype),
            indexing="ij",
        )
        grid_x = (xs + flow[:, 1]) / max(ww - 1, 1) * 2 - 1
        grid_y = (ys + flow[:, 0]) / max(hh - 1, 1) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)
        warped = F.grid_sample(
            h, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        mask = ((grid_x.abs() <= 1) & (grid_y.abs() <= 1)).unsqueeze(1).to(h.dtype)
        return warped, mask


class ConvGRUCell(nn.Module):
    """ConvGRU with optional depthwise-separable gate convs and a per-channel
    motion bias (FiLM-lite) on the gate pre-activations."""

    def __init__(self, c_in, c_h, motion_embed=0, separable=True):
        super().__init__()
        self.c_h = c_h
        self.motion_embed = motion_embed
        gate_in = c_in + c_h

        def conv(c_in_, c_out_):
            if separable:
                return nn.Sequential(
                    nn.Conv2d(c_in_, c_in_, 3, 1, 1, groups=c_in_),
                    nn.Conv2d(c_in_, c_out_, 1),
                )
            return nn.Conv2d(c_in_, c_out_, 3, 1, 1)

        self.gates = conv(gate_in, 2 * c_h)      # z, r
        self.candidate = conv(gate_in, c_h)      # n
        if motion_embed:
            self.motion_bias = nn.Linear(motion_embed, 3 * c_h)
            nn.init.zeros_(self.motion_bias.weight)
            nn.init.zeros_(self.motion_bias.bias)

    def forward(self, x, h, m=None):
        xh = torch.cat([x, h], dim=1)
        z, r = self.gates(xh).chunk(2, dim=1)
        n_pre = self.candidate(torch.cat([x, r * h], dim=1))
        if self.motion_embed and m is not None:
            bias = self.motion_bias(m).reshape(-1, 3, self.c_h, 1, 1)
            z = z + bias[:, 0]
            r = r + bias[:, 1]
            n_pre = n_pre + bias[:, 2]
        z, r, n = torch.sigmoid(z), torch.sigmoid(r), torch.tanh(n_pre)
        return z * h + (1 - z) * n


class GlobalTemporal(nn.Module):
    """Variant H: tiny causal transformer over pooled g_t history."""

    def __init__(self, cfg: VisualEncoderConfig):
        super().__init__()
        self.len = cfg.global_temporal_len
        layer = nn.TransformerEncoderLayer(
            cfg.global_dim, 4, cfg.global_dim * 2, batch_first=True, norm_first=True
        )
        self.net = nn.TransformerEncoder(layer, 2)
        self.pos = nn.Parameter(torch.zeros(1, self.len, cfg.global_dim))

    def forward(self, g_hist):
        # g_hist: (B, K, D), oldest-first, zero-padded at the front
        k = g_hist.shape[1]
        mask = torch.triu(
            torch.ones(k, k, device=g_hist.device, dtype=torch.bool), diagonal=1
        )
        out = self.net(g_hist + self.pos[:, :k], mask=mask)
        return out[:, -1]


class VisualEncoder(nn.Module):
    def __init__(self, cfg: VisualEncoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or VisualEncoderConfig()
        cfg = self.cfg
        self.stem = Stem(cfg)
        self.motion = MotionEncoder(cfg)
        self.warp_net = LearnedWarp(cfg) if cfg.warp else None

        in_ch = cfg.latent_channels + 1 + (cfg.latent_channels if cfg.diff_signal else 0)
        self.fusion = (
            ConvGRUCell(
                in_ch,
                cfg.latent_channels,
                motion_embed=cfg.motion_embed if cfg.motion_condition else 0,
                separable=cfg.separable_gates,
            )
            if cfg.temporal
            else nn.Conv2d(in_ch, cfg.latent_channels, 1)
        )

        self.global_head = nn.Sequential(
            nn.Linear(cfg.latent_channels + cfg.context_channels, cfg.global_dim),
            nn.GELU(),
            nn.Linear(cfg.global_dim, cfg.global_dim),
        )
        self.global_temporal = GlobalTemporal(cfg) if cfg.global_temporal else None

        # probes (training-only; decode from spatial map when spatial_probe)
        self.probe = nn.Linear(cfg.global_dim, 4)  # gap_x, wall_dist, vx, vz
        occ_w = cfg.occupancy_hw[1]
        self.occupancy = nn.Sequential(
            nn.Conv2d(cfg.latent_channels, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 8, 1),
            nn.GELU(),
        )
        self.occupancy_out = nn.Linear(8 * (cfg.input_hw[1] // 8), occ_w)
        if cfg.future_head:
            self.future = nn.Sequential(
                nn.Linear(cfg.global_dim + cfg.motion_embed + cfg.action_dim, 256),
                nn.GELU(),
                nn.Linear(256, 8 * 6 * cfg.latent_channels),
            )

    def latent_hw(self):
        h, w = self.cfg.input_hw
        return h // 8, w // 8


    # ---- state API ----

    def initial_state(self, batch, device=None):
        h, w = self.latent_hw()
        dev = device or next(self.parameters()).device
        state = {
            "h": torch.zeros(
                batch, self.cfg.latent_channels, h, w, device=dev
            ),
        }
        if self.cfg.global_temporal:
            state["g_hist"] = torch.zeros(
                batch, self.cfg.global_temporal_len, self.cfg.global_dim, device=dev
            )
        return state

    def encode(self, frame):
        """frame: (B,3,H,W) uint8 or float in [0,1] -> (F_t, ctx_t)."""
        if frame.dtype == torch.uint8:
            frame = frame.float() / 255.0
        return self.stem(frame)

    def propagate(self, state, motion):
        """Warp-only step: align H to the current frame given ego-motion."""
        m = self.motion(motion)
        if self.warp_net is not None:
            h_warped, mask = self.warp_net(state["h"], m)
        else:
            h_warped = state["h"]
            mask = torch.ones(
                h_warped.shape[0], 1, *h_warped.shape[2:], device=h_warped.device
            )
        return {**state, "h": h_warped, "mask": mask, "m": m}

    def update(self, frame, state, motion, action=None):
        """Camera step: encode frame, align previous state, fuse."""
        f_t, ctx = self.encode(frame)
        prop = self.propagate(state, motion)
        h_prev, mask, m = prop["h"], prop["mask"], prop["m"]

        parts = [f_t, mask]
        if self.cfg.diff_signal:
            parts.append((F.normalize(f_t, dim=1) - F.normalize(h_prev, dim=1)).abs())
        x = torch.cat(parts, dim=1)

        if self.cfg.temporal:
            h = self.fusion(x, h_prev, m if self.cfg.motion_condition else None)
        else:
            h = self.fusion(x)

        pooled = torch.cat(
            [h.mean(dim=(2, 3)), ctx.mean(dim=(2, 3))], dim=1
        )
        g = self.global_head(pooled)

        new_state = {"h": h}
        if self.cfg.global_temporal:
            hist = torch.cat([state["g_hist"][:, 1:], g.unsqueeze(1)], dim=1)
            g = self.global_temporal(hist)
            new_state["g_hist"] = hist

        occ_feat = self.occupancy(h).mean(dim=2)  # (B,8,W_lat) column features
        out = {
            "h": h,
            "g": g,
            "f": f_t,
            "mask": mask,
            "probe": self.probe(g),
            "occupancy": self.occupancy_out(
                occ_feat.reshape(occ_feat.shape[0], -1)
            ),
        }
        if self.cfg.future_head and action is not None:
            pred = self.future(torch.cat([g, m, action], dim=1))
            out["future_f"] = pred.reshape(-1, self.cfg.latent_channels, 6, 8)
        return out, new_state

    def forward_sequence(self, frames, motions, actions=None, state=None):
        """frames (B,T,3,H,W), motions (B,T,motion_dim), actions (B,T,A).

        Returns stacked outputs and final state. Used for training."""
        b, t = frames.shape[:2]
        if state is None:
            state = self.initial_state(b, frames.device)
        outs = []
        for i in range(t):
            act = actions[:, i] if actions is not None else None
            out, state = self.update(frames[:, i], state, motions[:, i], act)
            outs.append(out)
        return {k: torch.stack([o[k] for o in outs], dim=1) for k in outs[0]}, state
