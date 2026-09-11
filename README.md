# triage

A drone simulator with GPU physics, reinforcement learning, and 3D rendering. Still in development.

It can generate reproducible labelled obstacle scenes, inspect RGB/depth/instance outputs, and train drones to hover and follow target positions.

## Try it

```sh
nix develop
cargo run -p window-demo
```

This opens the rendering demo, not a trained policy. It needs a working graphics driver.

## Generate synthetic data

Inside `nix develop`, with a working graphics driver (CUDA is not required):

```sh
cargo run -p render-smoke -- --generate --seed 42 --count 3 --output target/dataset
cargo run -p render-smoke -- --inspect \
  --scene target/dataset/sample-0000000000/scene.json \
  --output target/replay --verify
```

The output directory must not exist. A completed batch contains a manifest and one bundle per sample: the realized scene, provenance/calibration, sRGB `color.png`, raw linear RGBA8, optical-Z depth, uint32 instance IDs and uint32 semantic classes. Gates have three box parts but one instance ID. Ontology v1 uses background 0, ground 1, gate 2 and obstacle 3.

`--start-index` selects independently reproducible samples; `--width` and `--height` set sensor resolution. `--recipe FILE` loads versioned variation settings and cannot be combined with seed/dimension flags. `--scene-only` generates JSON without a GPU. See `--generate --help`.

Scene schema v2 replaces the earlier single-box v1 schema; older scene versions are rejected. Saved scenes replay without invoking the generator. Color is clipped/quantized LDR, not HDR; exact pixels across different GPUs are not promised.

## Explore the static showcase

```sh
NO_COLOR=true trunk serve --config apps/web-demo/Trunk.toml
NO_COLOR=true trunk build --config apps/web-demo/Trunk.toml --release --locked
```

Serve `apps/web-demo/dist/` over HTTPS or localhost. The page uses the same Rust generator as native captures, with seed/sample controls, calibrated camera reset, scene downloads and RGB/depth/instance views. A checked-in [native capture example](examples/seed-42-sample-0/index.html) remains accessible when WebGPU is unavailable. CUDA training does not run in the browser.

Keep this page and its native example current when changing generation or sensor contracts. The example is seed 42/sample 0 at default 640×480; regenerate it with the native command above and retain its `index.html`.

## Train a drone

Requires Linux and an NVIDIA GPU. Run these from the repository root, inside `nix develop`:

```sh
nu rl/setup.nu
cuda/build/rl-venv/bin/python -m rl.train train --task hover
```

The setup command installs Python dependencies and builds the simulator. Training saves a checkpoint to `cuda/build/hover.pt`.

## Code

- `cuda/` — drone physics
- `rl/` — training and evaluation
- `crates/` — shared scene generation, rendering, inspection and flight playback
- `apps/` — demos

[Contributing](contributing.md) | [Project plan](PLAN.md)

The window demo's embedded [Barnaslingan 02 HDRI](https://polyhaven.com/a/barnaslingan_02) is a [CC0 Poly Haven asset](https://polyhaven.com/license). It is not used in the seeded sensor examples.
