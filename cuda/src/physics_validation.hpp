#pragma once

#include "physics_types.hpp"

#include <cmath>
#include <stdexcept>

namespace sim_cuda::model {

template <typename Scalar>
void validate_parameters(const VehicleParameters<Scalar> &parameters,
                         const Scalar timestep, const int substeps) {
  if (!(parameters.mass > Scalar{0}) ||
      !(parameters.inertia_diagonal_b.x > Scalar{0}) ||
      !(parameters.inertia_diagonal_b.y > Scalar{0}) ||
      !(parameters.inertia_diagonal_b.z > Scalar{0})) {
    throw std::invalid_argument("mass and diagonal inertia must be positive");
  }
  if (!(timestep > Scalar{0}) || substeps <= 0) {
    throw std::invalid_argument(
        "physics timestep and substep count must be positive");
  }
  constexpr Scalar sign_tolerance =
      sizeof(Scalar) == sizeof(float) ? Scalar{1e-6F} : Scalar{1e-12};
  for (const RotorParameters<Scalar> &rotor : parameters.rotors) {
    if (!(rotor.time_constant > Scalar{0}) || rotor.minimum_speed < Scalar{0} ||
        rotor.maximum_speed < rotor.minimum_speed ||
        rotor.thrust_coefficient < Scalar{0} ||
        rotor.torque_coefficient < Scalar{0} ||
        std::abs(std::abs(rotor.reaction_torque_sign) - Scalar{1}) >
            sign_tolerance) {
      throw std::invalid_argument("invalid rotor parameters");
    }
    // Explicit midpoint's motor mode is non-decaying at dt/tau == 2 and
    // unstable above it. Substeps each consume timestep; they do not divide it.
    if (!(timestep / rotor.time_constant < Scalar{2})) {
      throw std::invalid_argument(
          "physics timestep must be less than twice every rotor time constant");
    }
  }
}

} // namespace sim_cuda::model
