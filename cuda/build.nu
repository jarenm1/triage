#!/usr/bin/env nu

def main [
  --release (-r)  # Build optimized binaries without debug information.
  --skip-tests    # Compile without running the CUDA smoke test.
] {
  let source_dir = ($env.FILE_PWD | path expand)
  let build_dir = ($source_dir | path join "build")
  let build_type = if $release { "Release" } else { "RelWithDebInfo" }

  cmake -S $source_dir -B $build_dir -G Ninja $"-DCMAKE_BUILD_TYPE=($build_type)"
  if $env.LAST_EXIT_CODE != 0 {
    exit $env.LAST_EXIT_CODE
  }

  cmake --build $build_dir
  if $env.LAST_EXIT_CODE != 0 {
    exit $env.LAST_EXIT_CODE
  }

  if not $skip_tests {
    ctest --test-dir $build_dir --output-on-failure
    if $env.LAST_EXIT_CODE != 0 {
      exit $env.LAST_EXIT_CODE
    }
  }
}
