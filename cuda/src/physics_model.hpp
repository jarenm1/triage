#pragma once

#include "physics_math.hpp"

#if defined(__CUDACC__)
#define SIM_CUDA_HOST_DEVICE __host__ __device__
#else
#define SIM_CUDA_HOST_DEVICE
#endif

namespace sim_cuda::model {

template <typename Scalar>
SIM_CUDA_HOST_DEVICE Scalar clamp_action(const Scalar action) {
  if (action < Scalar{0}) {
    return Scalar{0};
  }
  if (action > Scalar{1}) {
    return Scalar{1};
  }
  return action;
}

template <typename Scalar> struct BodyWrench {
  Vec3<Scalar> force_b{};
  Vec3<Scalar> torque_b{};
};

// Accumulate in rotor-index order, including arm and signed reaction torques.
template <typename Scalar>
SIM_CUDA_HOST_DEVICE void
accumulate_rotor_wrench(BodyWrench<Scalar> &wrench,
                        const RotorParameters<Scalar> &rotor,
                        const Scalar speed) {
  const Scalar speed_squared = speed * speed;
  const Vec3<Scalar> rotor_force =
      scale(rotor.thrust_direction_b, rotor.thrust_coefficient * speed_squared);
  const Vec3<Scalar> arm_torque = cross(rotor.position_b, rotor_force);
  const Vec3<Scalar> reaction_torque = scale(
      rotor.thrust_direction_b,
      rotor.reaction_torque_sign * rotor.torque_coefficient * speed_squared);
  wrench.force_b = add(wrench.force_b, rotor_force);
  wrench.torque_b = add(wrench.torque_b, add(arm_torque, reaction_torque));
}

// Evaluate forces at supplied speeds without advancing the motor or body state.
template <typename Scalar>
SIM_CUDA_HOST_DEVICE BodyWrench<Scalar>
rotor_wrench(const FixedArray<Scalar, kRotorCount> &rotor_speed,
             const VehicleParameters<Scalar> &parameters) {
  BodyWrench<Scalar> wrench{};
#pragma unroll
  for (std::size_t rotor_index = 0; rotor_index < kRotorCount; ++rotor_index) {
    accumulate_rotor_wrench(wrench, parameters.rotors[rotor_index],
                            rotor_speed[rotor_index]);
  }
  return wrench;
}

// Rigid-body velocity derivatives: world-frame translation and body-frame
// rotation. These evaluate the supplied state and never advance it.
template <typename Scalar>
SIM_CUDA_HOST_DEVICE Vec3<Scalar>
linear_acceleration_w(const Quaternion<Scalar> attitude_wb,
                      const Vec3<Scalar> force_b,
                      const VehicleParameters<Scalar> &parameters) {
  const Vec3<Scalar> force_w = rotate_body_to_world(attitude_wb, force_b);
  return add(parameters.gravity_w, scale(force_w, Scalar{1} / parameters.mass));
}

template <typename Scalar>
SIM_CUDA_HOST_DEVICE Vec3<Scalar>
angular_acceleration_b(const Vec3<Scalar> angular_velocity_b,
                       const Vec3<Scalar> torque_b,
                       const VehicleParameters<Scalar> &parameters) {
  const Vec3<Scalar> angular_momentum{
      parameters.inertia_diagonal_b.x * angular_velocity_b.x,
      parameters.inertia_diagonal_b.y * angular_velocity_b.y,
      parameters.inertia_diagonal_b.z * angular_velocity_b.z,
  };
  const Vec3<Scalar> gyroscopic = cross(angular_velocity_b, angular_momentum);
  const Vec3<Scalar> net_torque{
      torque_b.x - gyroscopic.x,
      torque_b.y - gyroscopic.y,
      torque_b.z - gyroscopic.z,
  };
  return {
      net_torque.x / parameters.inertia_diagonal_b.x,
      net_torque.y / parameters.inertia_diagonal_b.y,
      net_torque.z / parameters.inertia_diagonal_b.z,
  };
}

template <typename Scalar>
SIM_CUDA_HOST_DEVICE Quaternion<Scalar>
attitude_derivative(const Quaternion<Scalar> q,
                    const Vec3<Scalar> angular_velocity_b) {
  return {
      Scalar{-0.5} * (q.x * angular_velocity_b.x + q.y * angular_velocity_b.y +
                      q.z * angular_velocity_b.z),
      Scalar{0.5} * (q.w * angular_velocity_b.x + q.y * angular_velocity_b.z -
                     q.z * angular_velocity_b.y),
      Scalar{0.5} * (q.w * angular_velocity_b.y + q.z * angular_velocity_b.x -
                     q.x * angular_velocity_b.z),
      Scalar{0.5} * (q.w * angular_velocity_b.z + q.x * angular_velocity_b.y -
                     q.y * angular_velocity_b.x),
  };
}

} // namespace sim_cuda::model

#undef SIM_CUDA_HOST_DEVICE
