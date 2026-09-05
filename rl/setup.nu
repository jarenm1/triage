#!/usr/bin/env nu

# Run inside the repository's Nix development shell.
def main [
  --python: string = "python3.12"
] {
  let root = ($env.FILE_PWD | path dirname)
  let environment = ($root | path join "cuda" "build" "rl-venv")
  with-env { UV_PROJECT_ENVIRONMENT: $environment } {
    uv sync --project $env.FILE_PWD --python $python --locked
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
  }
  nu ($root | path join "cuda" "build.nu") --release
  if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
}
