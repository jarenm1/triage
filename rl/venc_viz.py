"""Live visualization for visual-encoder training via rerun.

Logs, every --viz-every steps:
  input/frame        latest input frame (uint8 RGB)
  latent/pca         spatial latent map -> PCA -> RGB, upscaled
  latent/mean        per-channel mean activation heatmap
  occ/pred           32-column free-space prediction (bar strip)
  occ/target         ground-truth free-space strip
  scalars/*          losses + eval metrics

Usage in venc_train: --viz opens the rerun viewer alongside training.
"""

import numpy as np
import torch


def init_viz(app_id="venc-train", spawn=True):
    import rerun as rr

    rr.init(app_id)
    if spawn:
        try:
            rr.spawn()
        except Exception:
            # no native viewer binary: host a web viewer instead
            uri = rr.serve_grpc()
            rr.serve_web_viewer(connect_to=uri, open_browser=True)
    return rr


def _pca_rgb(feat: torch.Tensor) -> np.ndarray:
    """(C,H,W) feature map -> (H,W,3) uint8 via PCA to 3 components."""
    c, h, w = feat.shape
    x = feat.detach().reshape(c, -1).T.float().cpu()  # (HW, C)
    x = x - x.mean(0, keepdim=True)
    # SVD on HW x C; take top-3 right singular vectors
    _, _, v = torch.pca_lowrank(x, q=3)
    proj = (x @ v).reshape(h, w, 3).numpy()
    # normalize each channel to 0-255
    lo, hi = proj.min(axis=(0, 1)), proj.max(axis=(0, 1))
    img = (proj - lo) / np.maximum(hi - lo, 1e-6)
    return (img * 255).astype(np.uint8)


def _occ_strip(occ: torch.Tensor, width=256, height=24) -> np.ndarray:
    """(32,) occupancy profile -> (height,width,3) bar strip."""
    v = occ.detach().float().cpu().numpy()
    v = np.clip(v, 0.0, 1.0)
    cols = (v * 255).astype(np.uint8)
    strip = np.repeat(cols[None, :], height, axis=0)  # (H,32)
    img = np.repeat(np.repeat(strip, width // 32, axis=1), 1, axis=0)
    return np.stack([img] * 3, axis=-1)


def log_step(rr, step, enc_outs, batch, logs, frame_idx=0, real=None):
    """Log one training step's visuals + scalars."""
    rr.set_time("step", sequence=step)

    # input frame (uint8 RGB)
    frame = batch["frames"][frame_idx, 0]  # (3,H,W) uint8
    rr.log("input/frame", rr.Image(frame.permute(1, 2, 0).cpu().numpy()))

    # spatial latent -> PCA RGB
    f = enc_outs["f"][frame_idx, -1].detach()  # (C,H,W)
    rr.log("latent/pca", rr.Image(_pca_rgb(f)))

    # mean activation heatmap
    heat = f.mean(0).cpu().numpy()
    heat = (heat - heat.min()) / max(heat.max() - heat.min(), 1e-6)
    rr.log("latent/mean", rr.Image((heat * 255).astype(np.uint8)))

    # occupancy: predicted vs target
    occ_pred = torch.sigmoid(enc_outs["occupancy"][frame_idx, -1])
    occ_tgt = batch["occ"][frame_idx, 0]
    rr.log("occ/pred", rr.Image(_occ_strip(occ_pred)))
    rr.log("occ/target", rr.Image(_occ_strip(occ_tgt)))

    for k, v in logs.items():
        rr.log(f"scalars/{k}", rr.Scalars(v))

    # real FPV frame + its latent (domain-gap watch)
    if real is not None:
        rframe, rf = real  # (3,H,W) uint8, (C,h,w) latent
        rr.log("real/frame", rr.Image(rframe.permute(1, 2, 0).cpu().numpy()))
        rr.log("real/latent_pca", rr.Image(_pca_rgb(rf.detach())))


def log_eval(rr, step, metrics):
    rr.set_time("step", sequence=step)
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            rr.log(f"eval/{k}", rr.Scalars(v))
