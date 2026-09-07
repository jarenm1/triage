# triage

A drone simulator with GPU physics, reinforcement learning, and 3D rendering. Still in development.

It can train drones to hover and follow target positions, then display their flights live or from recordings.

## Try it

```sh
nix develop
cargo run -p window-demo
```

This opens the rendering demo, not a trained policy. It needs a working graphics driver.

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
- `crates/` — rendering and flight playback
- `apps/` — demos

[Contributing](contributing.md) | [Project plan](PLAN.md)
