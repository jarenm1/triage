# Contributing

Run `nix develop` from the repository root to get the development tools.

Keep changes small. Explain what you changed and why, and list what you tested.

## Check your changes

For Rust:

```sh
cargo fmt --all
cargo test --workspace
```

For physics without a GPU:

```sh
nu cuda/build.nu --cpu-only
```

For CUDA physics (requires an NVIDIA GPU):

```sh
nu cuda/build.nu --release
```

For Python, after running `nu rl/setup.nu`:

```sh
cuda/build/rl-venv/bin/python -m unittest discover -s rl -p 'test_*.py'
```

If you change rendering, open the demo and check it visually. Say which checks you could not run.

Do not commit build files, trained models, or virtual environments.
