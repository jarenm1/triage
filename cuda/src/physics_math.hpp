#pragma once

#include "physics_types.hpp"

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

// attitude_wb maps body-frame vectors into the world frame.
template <typename Scalar>
SIM_CUDA_HOST_DEVICE Vec3<Scalar>
rotate_body_to_world(const Quaternion<Scalar> q, const Vec3<Scalar> value_b) {
  const Vec3<Scalar> imaginary{q.x, q.y, q.z};
  const Vec3<Scalar> first_cross = cross(imaginary, value_b);
  const Vec3<Scalar> second_cross = cross(imaginary, first_cross);
  return add(value_b, add(scale(first_cross, Scalar{2} * q.w),
                          scale(second_cross, Scalar{2})));
}

} // namespace sim_cuda::model

#undef SIM_CUDA_HOST_DEVICE
