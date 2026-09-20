# Visual Encoder Web Demo

Self-contained browser demo of the visual-inertial encoder. No backend —
ONNX Runtime Web runs inference client-side (WebGPU, WASM fallback).

## Files

- `index.html`, `main.js` — the demo page (play/scrub through a rollout,
  shows input frame, spatial latent as PCA→RGB, free-space profile, probes)
- `model.onnx` + `model.onnx.data` — exported `update()` step (~1.4 MB)
- `frames.jpg` — sprite sheet of 240 rollout frames (128×96 tiles, 16 cols)
- `meta.json` — motion/action sequences, occupancy targets, PCA matrix
- `export.py` — regenerates all of the above from a checkpoint

## Run locally

```sh
python3 -m http.server 8931 --directory web-demo
# open http://127.0.0.1:8931/
```

## Deploy

Copy `index.html`, `main.js`, `model.onnx`, `model.onnx.data`,
`frames.jpg`, `meta.json` to any static host (Cloudflare Pages, etc.).
The ONNX Runtime Web runtime loads from jsdelivr CDN — vendor it locally
if you want zero external deps.

## Regenerate

```sh
python web-demo/export.py --checkpoint target/venc-results/m-vitb.pt \
    --data target/venc-data/val-fpv/rollouts.pt --frames 240
```
