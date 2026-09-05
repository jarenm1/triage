#include "physics.cuh"
#include "physics_model.cuh"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace sim_cuda {
namespace {

__global__ void physics_step_kernel(const DeviceState *current_states,
                                    const DeviceActions *actions,
                                    DeviceState *next_states,
                                    const std::size_t environment_count,
                                    const DeviceVehicleParameters parameters,
                                    const float timestep, const int substeps) {
  const std::size_t environment_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (environment_index >= environment_count) {
    return;
  }

  DeviceState state = current_states[environment_index];
  const DeviceActions action = actions[environment_index];
  for (int substep = 0; substep < substeps; ++substep) {
    state = model::step_substep(state, action, parameters, timestep);
  }
  next_states[environment_index] = state;
}

void check_launch(const cudaError_t status, const char *operation) {
  if (status == cudaSuccess) {
    return;
  }
  std::ostringstream message;
  message << operation << ": " << cudaGetErrorString(status);
  throw std::runtime_error(message.str());
}

} // namespace

void launch_physics_step(const DeviceState *current_states,
                         const DeviceActions *actions, DeviceState *next_states,
                         const std::size_t environment_count,
                         const DeviceVehicleParameters &parameters,
                         const float physics_timestep_seconds,
                         const int substeps, cudaStream_t stream) {
  model::validate_parameters(parameters, physics_timestep_seconds, substeps);
  if (environment_count == 0) {
    return;
  }
  if (current_states == nullptr || actions == nullptr ||
      next_states == nullptr) {
    throw std::invalid_argument(
        "state and action device pointers must not be null");
  }

  constexpr unsigned int threads_per_block = 256;
  const auto block_count = static_cast<unsigned int>(
      (environment_count + threads_per_block - 1) / threads_per_block);
  physics_step_kernel<<<block_count, threads_per_block, 0, stream>>>(
      current_states, actions, next_states, environment_count, parameters,
      physics_timestep_seconds, substeps);
  check_launch(cudaGetLastError(), "physics_step_kernel launch");
}

} // namespace sim_cuda
