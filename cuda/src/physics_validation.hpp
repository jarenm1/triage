#pragma once

#include "physics_types.hpp"

#include <cmath>
#include <stdexcept>

#if defined(__CUDACC__)
#define SIM_CUDA_HOST_DEVICE __host__ __device__
#else
#define SIM_CUDA_HOST_DEVICE
#endif

namespace sim_cuda::model {

template <typename Scalar>
SIM_CUDA_HOST_DEVICE constexpr Scalar validation_tolerance() {
  return sizeof(Scalar) == sizeof(float) ? Scalar{1e-5F} : Scalar{1e-10};
}

template <typename Scalar>
SIM_CUDA_HOST_DEVICE bool finite_scalar(const Scalar value) {
#if defined(__CUDA_ARCH__)
  return isfinite(value);
#else
  return std::isfinite(value);
#endif
}

template <typename Scalar>
SIM_CUDA_HOST_DEVICE bool finite_vector(const Vec3<Scalar> &value) {
  return finite_scalar(value.x) && finite_scalar(value.y) &&
         finite_scalar(value.z);
}

template <typename Scalar>
SIM_CUDA_HOST_DEVICE bool unit_squared_norm(const Scalar squared_norm) {
  const Scalar low = Scalar{1} - validation_tolerance<Scalar>();
  const Scalar high = Scalar{1} + validation_tolerance<Scalar>();
  return squared_norm >= low * low && squared_norm <= high * high;
}

template <typename Scalar>
SIM_CUDA_HOST_DEVICE bool valid_state(const MultirotorState<Scalar> &state) {
  const auto &q = state.attitude_wb;
  if (!finite_vector(state.position_w) ||
      !finite_vector(state.linear_velocity_w) ||
      !finite_vector(state.angular_velocity_b) || !finite_scalar(q.w) ||
      !finite_scalar(q.x) || !finite_scalar(q.y) || !finite_scalar(q.z) ||
      !unit_squared_norm(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)) {
    return false;
  }
  for (const Scalar speed : state.rotor_speed) {
    if (!finite_scalar(speed) || speed < Scalar{0}) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
SIM_CUDA_HOST_DEVICE bool
valid_parameters(const VehicleParameters<Scalar> &parameters,
                 const Scalar timestep, const int substeps) {
  const auto &inertia = parameters.inertia_diagonal_b;
  if (!finite_scalar(timestep) || !(timestep > Scalar{0}) || substeps <= 0 ||
      !finite_scalar(parameters.mass) || !(parameters.mass > Scalar{0}) ||
      !finite_vector(inertia) || !(inertia.x > Scalar{0}) ||
      !(inertia.y > Scalar{0}) || !(inertia.z > Scalar{0}) ||
      !finite_vector(parameters.gravity_w)) {
    return false;
  }
  const Scalar maximum = inertia.x > inertia.y
                             ? (inertia.x > inertia.z ? inertia.x : inertia.z)
                             : (inertia.y > inertia.z ? inertia.y : inertia.z);
  const Scalar x = inertia.x / maximum;
  const Scalar y = inertia.y / maximum;
  const Scalar z = inertia.z / maximum;
  const Scalar tolerance = validation_tolerance<Scalar>() * (x + y + z);
  if (x > y + z + tolerance || y > x + z + tolerance || z > x + y + tolerance) {
    return false;
  }
  constexpr Scalar sign_tolerance =
      sizeof(Scalar) == sizeof(float) ? Scalar{1e-6F} : Scalar{1e-12};
  for (const RotorParameters<Scalar> &rotor : parameters.rotors) {
    const auto &direction = rotor.thrust_direction_b;
    const Scalar sign = rotor.reaction_torque_sign < Scalar{0}
                            ? -rotor.reaction_torque_sign
                            : rotor.reaction_torque_sign;
    if (!finite_vector(rotor.position_b) || !finite_vector(direction) ||
        !unit_squared_norm(direction.x * direction.x +
                           direction.y * direction.y +
                           direction.z * direction.z) ||
        !finite_scalar(sign) || sign < Scalar{1} - sign_tolerance ||
        sign > Scalar{1} + sign_tolerance ||
        !finite_scalar(rotor.thrust_coefficient) ||
        rotor.thrust_coefficient < Scalar{0} ||
        !finite_scalar(rotor.torque_coefficient) ||
        rotor.torque_coefficient < Scalar{0} ||
        !finite_scalar(rotor.minimum_speed) ||
        rotor.minimum_speed < Scalar{0} ||
        !finite_scalar(rotor.maximum_speed) ||
        rotor.maximum_speed < rotor.minimum_speed ||
        !finite_scalar(rotor.time_constant) ||
        !(rotor.time_constant > Scalar{0}) ||
        !(timestep / rotor.time_constant < Scalar{2})) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
void validate_parameters(const VehicleParameters<Scalar> &parameters,
                         const Scalar timestep, const int substeps) {
  if (!valid_parameters(parameters, timestep, substeps)) {
    throw std::invalid_argument("invalid vehicle parameters or physics timing");
  }
}

} // namespace sim_cuda::model

#undef SIM_CUDA_HOST_DEVICE
