#!/usr/bin/env nu

def main [
  --release (-r)  # Build optimized binaries without debug information.
  --skip-tests    # Compile without running tests.
  --cpu-only      # Build reference physics without a CUDA toolkit or GPU.
] {
  let source_dir = ($env.FILE_PWD | path expand)
  let build_dir = if $cpu_only {
    $source_dir | path join "build" "cpu"
  } else {
    $source_dir | path join "build"
  }
  let build_type = if $release { "Release" } else { "RelWithDebInfo" }
  let enable_cuda = if $cpu_only { "OFF" } else { "ON" }

  cmake -S $source_dir -B $build_dir -G Ninja $"-DCMAKE_BUILD_TYPE=($build_type)" $"-DSIM_CUDA_ENABLE_CUDA=($enable_cuda)"
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
