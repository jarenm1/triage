#include "reference_physics.hpp"
#include "physics_model.cuh"

namespace sim_cuda {

ReferenceVehicleParameters make_reference_quad_x() {
  constexpr double arm = 0.17;
  constexpr double thrust_coefficient = 1.91e-6;
  constexpr double torque_coefficient = 2.6e-8;
  constexpr double maximum_speed = 2200.0;
  constexpr double time_constant = 0.03;
  constexpr Vec3<double> thrust_direction{0.0, 0.0, 1.0};

  return {
      .mass = 1.0,
      .inertia_diagonal_b = {0.0082, 0.0082, 0.0148},
      .gravity_w = {0.0, 0.0, -9.80665},
      .rotors = {{
          {{arm, arm, 0.0},
           thrust_direction,
           1.0,
           thrust_coefficient,
           torque_coefficient,
           0.0,
           maximum_speed,
           time_constant},
          {{-arm, arm, 0.0},
           thrust_direction,
           -1.0,
           thrust_coefficient,
           torque_coefficient,
           0.0,
           maximum_speed,
           time_constant},
          {{-arm, -arm, 0.0},
           thrust_direction,
           1.0,
           thrust_coefficient,
           torque_coefficient,
           0.0,
           maximum_speed,
           time_constant},
          {{arm, -arm, 0.0},
           thrust_direction,
           -1.0,
           thrust_coefficient,
           torque_coefficient,
           0.0,
           maximum_speed,
           time_constant},
      }},
  };
}

void validate_reference_parameters(const ReferenceVehicleParameters &parameters,
                                   const double physics_timestep_seconds,
                                   const int substeps) {
  model::validate_parameters(parameters, physics_timestep_seconds, substeps);
}

ReferenceState step_reference(const ReferenceState &initial_state,
                              const ReferenceActions &actions,
                              const ReferenceVehicleParameters &parameters,
                              const double physics_timestep_seconds,
                              const int substeps) {
  validate_reference_parameters(parameters, physics_timestep_seconds, substeps);
  ReferenceState state = initial_state;
  for (int substep = 0; substep < substeps; ++substep) {
    state = model::step_substep(state, actions, parameters,
                                physics_timestep_seconds);
  }
  return state;
}

} // namespace sim_cuda
