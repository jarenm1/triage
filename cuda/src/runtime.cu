#include "runtime.cuh"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace sim_cuda {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return;
  }

  std::ostringstream message;
  message << operation << ": " << cudaGetErrorString(status);
  throw std::runtime_error(message.str());
}

__global__ void write_probe_value(int* output) {
  *output = 0x51A;
}

}  // namespace

RuntimeInfo probe_runtime(int device_ordinal) {
  check_cuda(cudaSetDevice(device_ordinal), "cudaSetDevice");

  cudaDeviceProp properties{};
  check_cuda(
      cudaGetDeviceProperties(&properties, device_ordinal),
      "cudaGetDeviceProperties");

  int* device_output = nullptr;
  check_cuda(cudaMalloc(&device_output, sizeof(*device_output)), "cudaMalloc");

  write_probe_value<<<1, 1>>>(device_output);
  const cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    cudaFree(device_output);
    check_cuda(launch_status, "write_probe_value launch");
  }

  int host_output = 0;
  const cudaError_t copy_status = cudaMemcpy(
      &host_output,
      device_output,
      sizeof(host_output),
      cudaMemcpyDeviceToHost);
  const cudaError_t free_status = cudaFree(device_output);
  check_cuda(copy_status, "cudaMemcpy");
  check_cuda(free_status, "cudaFree");

  if (host_output != 0x51A) {
    throw std::runtime_error("CUDA probe kernel returned an invalid value");
  }

  return RuntimeInfo{
      .device_ordinal = device_ordinal,
      .device_name = properties.name,
      .compute_major = properties.major,
      .compute_minor = properties.minor,
      .global_memory_bytes = properties.totalGlobalMem,
  };
}

}  // namespace sim_cuda
