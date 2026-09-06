#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    println!(
        "This is a WebGPU browser demo. Run `nix develop --command trunk serve --config apps/web-demo/Trunk.toml` from the repository root, then open http://localhost:8081 in a WebGPU-capable browser."
    );
}
