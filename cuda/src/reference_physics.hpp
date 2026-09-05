#pragma once

#include "physics_types.hpp"

namespace sim_cuda {

using ReferenceState = MultirotorState<double>;
using ReferenceActions = RotorActions<double>;
using ReferenceVehicleParameters = VehicleParameters<double>;

ReferenceVehicleParameters make_reference_quad_x();

void validate_reference_parameters(const ReferenceVehicleParameters &parameters,
                                   double physics_timestep_seconds,
                                   int substeps);

// Coupled midpoint/RK2. Each substep advances physics_timestep_seconds;
// it must be positive and strictly less than twice every motor time constant.
// That motor stability bound is not an accuracy or rigid-body stability bound.
ReferenceState step_reference(const ReferenceState &initial_state,
                              const ReferenceActions &actions,
                              const ReferenceVehicleParameters &parameters,
                              double physics_timestep_seconds,
                              int substeps = 1);

} // namespace sim_cuda
