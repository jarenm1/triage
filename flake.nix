{
  description = "Modern Rust development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    crane.url = "github:ipetkov/crane";
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
      crane,
    }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forEachSystem = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forEachSystem (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
            overlays = [ rust-overlay.overlays.default ];
          };

          rustToolchain = pkgs.rust-bin.stable.latest.default.override {
            extensions = [
              "clippy"
              "rust-analyzer"
              "rust-src"
              "rustfmt"
            ];
            targets = [ "wasm32-unknown-unknown" ];
          };

          craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
        in
        {
          default = craneLib.devShell {
            packages =
              (with pkgs; [
                binaryen
                cargo-audit
                cargo-edit
                cargo-nextest
                nushell
                pkg-config
                tracy_0_11
                trunk
                vulkan-loader
                vulkan-tools
                wasm-bindgen-cli
                wayland
              ])
              ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
                pkgs.cmake
                pkgs.ninja
                pkgs.python312
                pkgs.uv
                pkgs.cudaPackages.cudatoolkit
                pkgs.cudaPackages.cuda_gdb
              ];
            LD_LIBRARY_PATH =
              (pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
                pkgs.vulkan-loader
                pkgs.libxkbcommon
                pkgs.wayland
                pkgs.libx11
                pkgs.libxcursor
                pkgs.libxi
                pkgs.libxrandr
                pkgs.libxcb
              ])
              + pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux ":/run/opengl-driver/lib";
          };
        }
      );

      formatter = forEachSystem (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        pkgs.nixfmt
      );
    };
}
