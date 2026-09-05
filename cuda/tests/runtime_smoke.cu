#include "runtime.cuh"

#include <exception>
#include <iostream>

int main() {
  try {
    const sim_cuda::RuntimeInfo info = sim_cuda::probe_runtime();
    std::cout << "CUDA device " << info.device_ordinal << ": "
              << info.device_name << " (sm_" << info.compute_major
              << info.compute_minor << ", "
              << info.global_memory_bytes / (1024 * 1024) << " MiB)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CUDA runtime probe failed: " << error.what() << '\n';
    return 1;
  }
}
