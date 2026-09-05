#pragma once

#include "physics_model.hpp"

#include <cmath>

#if defined(__CUDACC__)
#define SIM_CUDA_HOST_DEVICE __host__ __device__
#else
#define SIM_CUDA_HOST_DEVICE
#endif

namespace sim_cuda::model {

template <typename Scalar>
SIM_CUDA_HOST_DEVICE Quaternion<Scalar>
integrate_attitude(const Quaternion<Scalar> q,
                   const Vec3<Scalar> angular_velocity_b,
                   const Scalar timestep) {
  const Quaternion<Scalar> derivative =
      attitude_derivative(q, angular_velocity_b);
  Quaternion<Scalar> next{
      q.w + derivative.w * timestep,
      q.x + derivative.x * timestep,
      q.y + derivative.y * timestep,
      q.z + derivative.z * timestep,
  };
  const Scalar inverse_norm =
      Scalar{1} / sqrt(next.w * next.w + next.x * next.x + next.y * next.y +
                       next.z * next.z);
  next.w *= inverse_norm;
  next.x *= inverse_norm;
  next.y *= inverse_norm;
  next.z *= inverse_norm;
  return next;
}

// Preserve the existing split, first-order scheme: exact motor response, then
// endpoint rotor wrench evaluated at the old attitude and angular velocity.
// Translation and rotation are semi-implicit: position uses the new velocity,
// and normalized quaternion Euler uses the new body angular velocity.
// timestep is the duration of each substep, not the total divided by substeps.
template <typename Scalar>
SIM_CUDA_HOST_DEVICE MultirotorState<Scalar> step_substep(
    const MultirotorState<Scalar> &state, const RotorActions<Scalar> &actions,
    const VehicleParameters<Scalar> &parameters, const Scalar timestep) {
  MultirotorState<Scalar> next = state;
  BodyWrench<Scalar> wrench{};

#pragma unroll
  for (std::size_t rotor_index = 0; rotor_index < kRotorCount; ++rotor_index) {
    const RotorParameters<Scalar> &rotor = parameters.rotors[rotor_index];
    const Scalar speed = rotor_response(state.rotor_speed[rotor_index],
                                        actions[rotor_index], rotor, timestep);
    next.rotor_speed[rotor_index] = speed;
    accumulate_rotor_wrench(wrench, rotor, speed);
  }

  const Vec3<Scalar> acceleration_w =
      linear_acceleration_w(state.attitude_wb, wrench.force_b, parameters);
  next.linear_velocity_w =
      add(state.linear_velocity_w, scale(acceleration_w, timestep));
  next.position_w =
      add(state.position_w, scale(next.linear_velocity_w, timestep));

  const Vec3<Scalar> angular_acceleration = angular_acceleration_b(
      state.angular_velocity_b, wrench.torque_b, parameters);
  next.angular_velocity_b =
      add(state.angular_velocity_b, scale(angular_acceleration, timestep));
  next.attitude_wb =
      integrate_attitude(state.attitude_wb, next.angular_velocity_b, timestep);
  return next;
}

} // namespace sim_cuda::model

#undef SIM_CUDA_HOST_DEVICE
