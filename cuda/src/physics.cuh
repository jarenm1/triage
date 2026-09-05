#pragma once

#include "physics_types.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>

namespace sim_cuda {

using DeviceState = MultirotorState<float>;
using DeviceActions = RotorActions<float>;
using DeviceVehicleParameters = VehicleParameters<float>;

// Coupled midpoint/RK2 on the supplied stream, without host synchronization.
// Each substep advances physics_timestep_seconds; it must be positive and
// strictly less than twice every motor time constant (not an accuracy bound).
void launch_physics_step(const DeviceState *current_states,
                         const DeviceActions *actions, DeviceState *next_states,
                         std::size_t environment_count,
                         const DeviceVehicleParameters &parameters,
                         float physics_timestep_seconds, int substeps = 1,
                         cudaStream_t stream = nullptr);

} // namespace sim_cuda
