#pragma once

#include <cstddef>

namespace sim_cuda {

inline constexpr std::size_t kRotorCount = 4;
#if defined(__CUDACC__)
#define SIM_CUDA_HOST_DEVICE __host__ __device__
#else
#define SIM_CUDA_HOST_DEVICE
#endif

template <typename Value, std::size_t Size> struct FixedArray {
  Value values[Size];

  SIM_CUDA_HOST_DEVICE constexpr Value &operator[](const std::size_t index) {
    return values[index];
  }

  SIM_CUDA_HOST_DEVICE constexpr const Value &
  operator[](const std::size_t index) const {
    return values[index];
  }

  SIM_CUDA_HOST_DEVICE constexpr Value *begin() { return values; }
  SIM_CUDA_HOST_DEVICE constexpr const Value *begin() const { return values; }
  SIM_CUDA_HOST_DEVICE constexpr Value *end() { return values + Size; }
  SIM_CUDA_HOST_DEVICE constexpr const Value *end() const {
    return values + Size;
  }

  constexpr void fill(const Value &value) {
    for (std::size_t index = 0; index < Size; ++index) {
      values[index] = value;
    }
  }
};

enum class RotorIndex : std::size_t {
  front_left = 0,
  rear_left = 1,
  rear_right = 2,
  front_right = 3,
};

template <typename Scalar> struct Vec3 {
  Scalar x;
  Scalar y;
  Scalar z;
};

template <typename Scalar> struct Quaternion {
  Scalar w;
  Scalar x;
  Scalar y;
  Scalar z;
};

template <typename Scalar> struct RotorParameters {
  Vec3<Scalar> position_b;
  Vec3<Scalar> thrust_direction_b;
  Scalar reaction_torque_sign;
  Scalar thrust_coefficient;
  Scalar torque_coefficient;
  Scalar minimum_speed;
  Scalar maximum_speed;
  Scalar time_constant;
};

template <typename Scalar> struct VehicleParameters {
  Scalar mass;
  Vec3<Scalar> inertia_diagonal_b;
  Vec3<Scalar> gravity_w;
  FixedArray<RotorParameters<Scalar>, kRotorCount> rotors;
};

template <typename Scalar> struct MultirotorState {
  Vec3<Scalar> position_w;
  Quaternion<Scalar> attitude_wb;
  Vec3<Scalar> linear_velocity_w;
  Vec3<Scalar> angular_velocity_b;
  FixedArray<Scalar, kRotorCount> rotor_speed;
};

template <typename Scalar> using RotorActions = FixedArray<Scalar, kRotorCount>;

#undef SIM_CUDA_HOST_DEVICE

} // namespace sim_cuda
