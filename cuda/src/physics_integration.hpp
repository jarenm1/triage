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

// Full-state explicit midpoint: both wrenches use their stage's rotor speeds,
// and body derivatives use that stage's attitude and angular velocity.
// Requires positive motor time constants and timestep / time_constant < 2
// for asymptotic motor stability; accuracy generally needs a much smaller
// ratio.
template <typename Scalar>
SIM_CUDA_HOST_DEVICE MultirotorState<Scalar> step_midpoint(
    const MultirotorState<Scalar> &state, const RotorActions<Scalar> &actions,
    const VehicleParameters<Scalar> &parameters, const Scalar timestep) {
  const Scalar half_step = timestep * Scalar{0.5};
  MultirotorState<Scalar> next{};
  BodyWrench<Scalar> initial_wrench{};
  BodyWrench<Scalar> midpoint_wrench{};
#pragma unroll
  for (std::size_t rotor_index = 0; rotor_index < kRotorCount; ++rotor_index) {
    const RotorParameters<Scalar> &rotor = parameters.rotors[rotor_index];
    const Scalar command =
        rotor.minimum_speed + clamp_action(actions[rotor_index]) *
                                  (rotor.maximum_speed - rotor.minimum_speed);
    const Scalar speed = state.rotor_speed[rotor_index];
    const Scalar midpoint_speed =
        speed + half_step * (command - speed) / rotor.time_constant;
    next.rotor_speed[rotor_index] =
        speed + timestep * (command - midpoint_speed) / rotor.time_constant;
    accumulate_rotor_wrench(initial_wrench, rotor, speed);
    accumulate_rotor_wrench(midpoint_wrench, rotor, midpoint_speed);
  }

  const Vec3<Scalar> midpoint_velocity =
      add(state.linear_velocity_w,
          scale(linear_acceleration_w(state.attitude_wb, initial_wrench.force_b,
                                      parameters),
                half_step));
  const Vec3<Scalar> midpoint_angular_velocity =
      add(state.angular_velocity_b,
          scale(angular_acceleration_b(state.angular_velocity_b,
                                       initial_wrench.torque_b, parameters),
                half_step));
  const Quaternion<Scalar> midpoint_attitude = integrate_attitude(
      state.attitude_wb, state.angular_velocity_b, half_step);
  // Position does not feed any derivative; its midpoint value need not be
  // materialized. Its final derivative is the predicted midpoint velocity.
  next.position_w = add(state.position_w, scale(midpoint_velocity, timestep));
  next.linear_velocity_w =
      add(state.linear_velocity_w,
          scale(linear_acceleration_w(midpoint_attitude,
                                      midpoint_wrench.force_b, parameters),
                timestep));
  next.angular_velocity_b =
      add(state.angular_velocity_b,
          scale(angular_acceleration_b(midpoint_angular_velocity,
                                       midpoint_wrench.torque_b, parameters),
                timestep));
  const Quaternion<Scalar> derivative =
      attitude_derivative(midpoint_attitude, midpoint_angular_velocity);
  Quaternion<Scalar> attitude{
      state.attitude_wb.w + timestep * derivative.w,
      state.attitude_wb.x + timestep * derivative.x,
      state.attitude_wb.y + timestep * derivative.y,
      state.attitude_wb.z + timestep * derivative.z,
  };
  const Scalar inverse_norm =
      Scalar{1} / sqrt(attitude.w * attitude.w + attitude.x * attitude.x +
                       attitude.y * attitude.y + attitude.z * attitude.z);
  next.attitude_wb = {attitude.w * inverse_norm, attitude.x * inverse_norm,
                      attitude.y * inverse_norm, attitude.z * inverse_norm};
  return next;
}

} // namespace sim_cuda::model

#undef SIM_CUDA_HOST_DEVICE
