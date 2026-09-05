#pragma once

#include "physics_types.cuh"

#include <cmath>
#include <stdexcept>

#if defined(__CUDACC__)
#define SIM_CUDA_HOST_DEVICE __host__ __device__
#else
#define SIM_CUDA_HOST_DEVICE
#endif

namespace sim_cuda::model {

template <typename Scalar>
SIM_CUDA_HOST_DEVICE Vec3<Scalar> add(const Vec3<Scalar> a,
                                      const Vec3<Scalar> b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

template <typename Scalar>
SIM_CUDA_HOST_DEVICE Vec3<Scalar> scale(const Vec3<Scalar> value,
                                        const Scalar factor) {
  return {value.x * factor, value.y * factor, value.z * factor};
}

template <typename Scalar>
SIM_CUDA_HOST_DEVICE Vec3<Scalar> cross(const Vec3<Scalar> a,
                                        const Vec3<Scalar> b) {
  return {
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x,
  };
}

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

template <typename Scalar>
SIM_CUDA_HOST_DEVICE Vec3<Scalar>
rotate_body_to_world(const Quaternion<Scalar> q, const Vec3<Scalar> value_b) {
  const Vec3<Scalar> imaginary{q.x, q.y, q.z};
  const Vec3<Scalar> first_cross = cross(imaginary, value_b);
  const Vec3<Scalar> second_cross = cross(imaginary, first_cross);
  return add(value_b, add(scale(first_cross, Scalar{2} * q.w),
                          scale(second_cross, Scalar{2})));
}

template <typename Scalar>
SIM_CUDA_HOST_DEVICE Quaternion<Scalar>
integrate_attitude(const Quaternion<Scalar> q,
                   const Vec3<Scalar> angular_velocity_b,
                   const Scalar timestep) {
  const Quaternion<Scalar> derivative{
      Scalar{-0.5} * (q.x * angular_velocity_b.x + q.y * angular_velocity_b.y +
                      q.z * angular_velocity_b.z),
      Scalar{0.5} * (q.w * angular_velocity_b.x + q.y * angular_velocity_b.z -
                     q.z * angular_velocity_b.y),
      Scalar{0.5} * (q.w * angular_velocity_b.y + q.z * angular_velocity_b.x -
                     q.x * angular_velocity_b.z),
      Scalar{0.5} * (q.w * angular_velocity_b.z + q.x * angular_velocity_b.y -
                     q.y * angular_velocity_b.x),
  };
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

template <typename Scalar>
SIM_CUDA_HOST_DEVICE MultirotorState<Scalar> step_substep(
    const MultirotorState<Scalar> &state, const RotorActions<Scalar> &actions,
    const VehicleParameters<Scalar> &parameters, const Scalar timestep) {
  MultirotorState<Scalar> next = state;
  Vec3<Scalar> total_force_b{};
  Vec3<Scalar> total_torque_b{};

#pragma unroll
  for (std::size_t rotor_index = 0; rotor_index < kRotorCount; ++rotor_index) {
    const RotorParameters<Scalar> &rotor = parameters.rotors[rotor_index];
    const Scalar action = clamp_action(actions[rotor_index]);
    const Scalar commanded_speed =
        rotor.minimum_speed +
        action * (rotor.maximum_speed - rotor.minimum_speed);
    const Scalar decay = exp(-timestep / rotor.time_constant);
    const Scalar speed =
        commanded_speed +
        (state.rotor_speed[rotor_index] - commanded_speed) * decay;
    next.rotor_speed[rotor_index] = speed;

    const Scalar speed_squared = speed * speed;
    const Vec3<Scalar> rotor_force = scale(
        rotor.thrust_direction_b, rotor.thrust_coefficient * speed_squared);
    const Vec3<Scalar> arm_torque = cross(rotor.position_b, rotor_force);
    const Vec3<Scalar> reaction_torque = scale(
        rotor.thrust_direction_b,
        rotor.reaction_torque_sign * rotor.torque_coefficient * speed_squared);
    total_force_b = add(total_force_b, rotor_force);
    total_torque_b = add(total_torque_b, add(arm_torque, reaction_torque));
  }

  const Vec3<Scalar> force_w =
      rotate_body_to_world(state.attitude_wb, total_force_b);
  const Vec3<Scalar> acceleration_w =
      add(parameters.gravity_w, scale(force_w, Scalar{1} / parameters.mass));
  next.linear_velocity_w =
      add(state.linear_velocity_w, scale(acceleration_w, timestep));
  next.position_w =
      add(state.position_w, scale(next.linear_velocity_w, timestep));

  const Vec3<Scalar> angular_momentum{
      parameters.inertia_diagonal_b.x * state.angular_velocity_b.x,
      parameters.inertia_diagonal_b.y * state.angular_velocity_b.y,
      parameters.inertia_diagonal_b.z * state.angular_velocity_b.z,
  };
  const Vec3<Scalar> gyroscopic =
      cross(state.angular_velocity_b, angular_momentum);
  const Vec3<Scalar> net_torque{
      total_torque_b.x - gyroscopic.x,
      total_torque_b.y - gyroscopic.y,
      total_torque_b.z - gyroscopic.z,
  };
  const Vec3<Scalar> angular_acceleration{
      net_torque.x / parameters.inertia_diagonal_b.x,
      net_torque.y / parameters.inertia_diagonal_b.y,
      net_torque.z / parameters.inertia_diagonal_b.z,
  };
  next.angular_velocity_b =
      add(state.angular_velocity_b, scale(angular_acceleration, timestep));
  next.attitude_wb =
      integrate_attitude(state.attitude_wb, next.angular_velocity_b, timestep);
  return next;
}

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
  }
}

} // namespace sim_cuda::model

#undef SIM_CUDA_HOST_DEVICE
