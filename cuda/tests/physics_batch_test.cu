#include "physics.cuh"
#include "reference_physics.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPhysicsTimestep = 0.001;
constexpr std::size_t kEnvironmentCount = 10'000;
constexpr int kControlSteps = 100;

void require(const bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_near(const double actual, const double expected,
                  const double tolerance, const std::string &quantity) {
  if (std::abs(actual - expected) > tolerance) {
    std::ostringstream message;
    message << quantity << ": expected " << expected << ", got " << actual
            << " (tolerance " << tolerance << ')';
    throw std::runtime_error(message.str());
  }
}

void check_cuda(const cudaError_t status, const char *operation) {
  if (status == cudaSuccess) {
    return;
  }
  std::ostringstream message;
  message << operation << ": " << cudaGetErrorString(status);
  throw std::runtime_error(message.str());
}

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(const std::size_t count) {
    check_cuda(cudaMalloc(&pointer_, count * sizeof(T)), "cudaMalloc");
  }

  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  T *get() { return pointer_; }
  const T *get() const { return pointer_; }

private:
  T *pointer_ = nullptr;
};

sim_cuda::DeviceVehicleParameters
as_device_parameters(const sim_cuda::ReferenceVehicleParameters &source) {
  sim_cuda::DeviceVehicleParameters destination{
      .mass = static_cast<float>(source.mass),
      .inertia_diagonal_b =
          {
              static_cast<float>(source.inertia_diagonal_b.x),
              static_cast<float>(source.inertia_diagonal_b.y),
              static_cast<float>(source.inertia_diagonal_b.z),
          },
      .gravity_w =
          {
              static_cast<float>(source.gravity_w.x),
              static_cast<float>(source.gravity_w.y),
              static_cast<float>(source.gravity_w.z),
          },
  };
  for (std::size_t index = 0; index < sim_cuda::kRotorCount; ++index) {
    const auto &rotor = source.rotors[index];
    destination.rotors[index] = {
        .position_b =
            {
                static_cast<float>(rotor.position_b.x),
                static_cast<float>(rotor.position_b.y),
                static_cast<float>(rotor.position_b.z),
            },
        .thrust_direction_b =
            {
                static_cast<float>(rotor.thrust_direction_b.x),
                static_cast<float>(rotor.thrust_direction_b.y),
                static_cast<float>(rotor.thrust_direction_b.z),
            },
        .reaction_torque_sign = static_cast<float>(rotor.reaction_torque_sign),
        .thrust_coefficient = static_cast<float>(rotor.thrust_coefficient),
        .torque_coefficient = static_cast<float>(rotor.torque_coefficient),
        .minimum_speed = static_cast<float>(rotor.minimum_speed),
        .maximum_speed = static_cast<float>(rotor.maximum_speed),
        .time_constant = static_cast<float>(rotor.time_constant),
    };
  }
  return destination;
}

sim_cuda::ReferenceVehicleParameters
as_reference_parameters(const sim_cuda::DeviceVehicleParameters &source) {
  sim_cuda::ReferenceVehicleParameters destination{
      .mass = source.mass,
      .inertia_diagonal_b =
          {
              source.inertia_diagonal_b.x,
              source.inertia_diagonal_b.y,
              source.inertia_diagonal_b.z,
          },
      .gravity_w = {source.gravity_w.x, source.gravity_w.y, source.gravity_w.z},
  };
  for (std::size_t index = 0; index < sim_cuda::kRotorCount; ++index) {
    const auto &rotor = source.rotors[index];
    destination.rotors[index] = {
        .position_b = {rotor.position_b.x, rotor.position_b.y,
                       rotor.position_b.z},
        .thrust_direction_b =
            {
                rotor.thrust_direction_b.x,
                rotor.thrust_direction_b.y,
                rotor.thrust_direction_b.z,
            },
        .reaction_torque_sign = rotor.reaction_torque_sign,
        .thrust_coefficient = rotor.thrust_coefficient,
        .torque_coefficient = rotor.torque_coefficient,
        .minimum_speed = rotor.minimum_speed,
        .maximum_speed = rotor.maximum_speed,
        .time_constant = rotor.time_constant,
    };
  }
  return destination;
}

sim_cuda::ReferenceState
as_reference_state(const sim_cuda::DeviceState &source) {
  sim_cuda::ReferenceState destination{
      .position_w = {source.position_w.x, source.position_w.y,
                     source.position_w.z},
      .attitude_wb =
          {
              source.attitude_wb.w,
              source.attitude_wb.x,
              source.attitude_wb.y,
              source.attitude_wb.z,
          },
      .linear_velocity_w =
          {
              source.linear_velocity_w.x,
              source.linear_velocity_w.y,
              source.linear_velocity_w.z,
          },
      .angular_velocity_b =
          {
              source.angular_velocity_b.x,
              source.angular_velocity_b.y,
              source.angular_velocity_b.z,
          },
  };
  for (std::size_t index = 0; index < sim_cuda::kRotorCount; ++index) {
    destination.rotor_speed[index] = source.rotor_speed[index];
  }
  return destination;
}

sim_cuda::ReferenceActions
as_reference_actions(const sim_cuda::DeviceActions &source) {
  sim_cuda::ReferenceActions destination{};
  for (std::size_t index = 0; index < sim_cuda::kRotorCount; ++index) {
    destination[index] = source[index];
  }
  return destination;
}

void populate_inputs(std::vector<sim_cuda::DeviceState> &states,
                     std::vector<sim_cuda::DeviceActions> &actions,
                     const sim_cuda::DeviceVehicleParameters &parameters) {
  const float hover_speed =
      std::sqrt(-parameters.mass * parameters.gravity_w.z /
                (static_cast<float>(sim_cuda::kRotorCount) *
                 parameters.rotors[0].thrust_coefficient));
  const float hover_action = hover_speed / parameters.rotors[0].maximum_speed;

  for (std::size_t environment = 0; environment < states.size();
       ++environment) {
    const float phase = static_cast<float>(environment % 101) / 100.0F;
    const float half_yaw = 0.05F * (phase - 0.5F);
    const float yaw_norm = std::sqrt(1.0F - half_yaw * half_yaw);
    states[environment] = {
        .position_w = {0.1F * phase, -0.2F * phase, 1.0F + phase},
        .attitude_wb = {yaw_norm, 0.0F, 0.0F, half_yaw},
        .linear_velocity_w = {0.02F * phase, -0.01F * phase, 0.0F},
        .angular_velocity_b = {0.01F * phase, -0.015F * phase, 0.005F * phase},
        .rotor_speed = {hover_speed, hover_speed, hover_speed, hover_speed},
    };
    actions[environment] = {
        hover_action + 0.01F * (phase - 0.5F),
        hover_action - 0.008F * (phase - 0.5F),
        hover_action + 0.006F * (phase - 0.5F),
        hover_action - 0.004F * (phase - 0.5F),
    };
  }
}

std::vector<sim_cuda::ReferenceState> run_reference_trajectory(
    const std::vector<sim_cuda::DeviceState> &initial_states,
    const std::vector<sim_cuda::DeviceActions> &actions,
    const sim_cuda::ReferenceVehicleParameters &parameters) {
  std::vector<sim_cuda::ReferenceState> states(initial_states.size());
  for (std::size_t environment = 0; environment < states.size();
       ++environment) {
    states[environment] = as_reference_state(initial_states[environment]);
  }

  for (int step = 0; step < kControlSteps; ++step) {
    for (std::size_t environment = 0; environment < states.size();
         ++environment) {
      states[environment] = sim_cuda::step_reference(
          states[environment], as_reference_actions(actions[environment]),
          parameters, kPhysicsTimestep);
    }
  }
  return states;
}

float run_cuda_trajectory(
    const std::vector<sim_cuda::DeviceState> &initial_states,
    const std::vector<sim_cuda::DeviceActions> &actions,
    const sim_cuda::DeviceVehicleParameters &parameters,
    std::vector<sim_cuda::DeviceState> &result) {
  DeviceBuffer<sim_cuda::DeviceState> state_a(initial_states.size());
  DeviceBuffer<sim_cuda::DeviceState> state_b(initial_states.size());
  DeviceBuffer<sim_cuda::DeviceActions> device_actions(actions.size());
  check_cuda(cudaMemcpy(state_a.get(), initial_states.data(),
                        initial_states.size() * sizeof(sim_cuda::DeviceState),
                        cudaMemcpyHostToDevice),
             "copy initial states");
  check_cuda(cudaMemcpy(device_actions.get(), actions.data(),
                        actions.size() * sizeof(sim_cuda::DeviceActions),
                        cudaMemcpyHostToDevice),
             "copy actions");

  sim_cuda::launch_physics_step(
      state_a.get(), device_actions.get(), state_b.get(), initial_states.size(),
      parameters, static_cast<float>(kPhysicsTimestep));
  check_cuda(cudaDeviceSynchronize(), "synchronize warm-up step");

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "create start event");
  check_cuda(cudaEventCreate(&stop), "create stop event");
  check_cuda(cudaEventRecord(start), "record start event");

  sim_cuda::DeviceState *current = state_a.get();
  sim_cuda::DeviceState *next = state_b.get();
  for (int step = 0; step < kControlSteps; ++step) {
    sim_cuda::launch_physics_step(current, device_actions.get(), next,
                                  initial_states.size(), parameters,
                                  static_cast<float>(kPhysicsTimestep));
    std::swap(current, next);
  }

  check_cuda(cudaEventRecord(stop), "record stop event");
  check_cuda(cudaEventSynchronize(stop), "synchronize stop event");
  float elapsed_milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&elapsed_milliseconds, start, stop),
             "measure elapsed device time");
  check_cuda(cudaEventDestroy(start), "destroy start event");
  check_cuda(cudaEventDestroy(stop), "destroy stop event");

  check_cuda(cudaMemcpy(result.data(), current,
                        result.size() * sizeof(sim_cuda::DeviceState),
                        cudaMemcpyDeviceToHost),
             "copy final states");
  return elapsed_milliseconds;
}

void compare_trajectories(
    const std::vector<sim_cuda::DeviceState> &actual_states,
    const std::vector<sim_cuda::ReferenceState> &expected_states,
    double &maximum_error) {
  auto include_error = [&maximum_error](const double actual,
                                        const double expected) {
    maximum_error = std::max(maximum_error, std::abs(actual - expected));
    require(std::isfinite(actual), "CUDA state contains a non-finite value");
  };

  for (std::size_t environment = 0; environment < actual_states.size();
       ++environment) {
    const auto &actual = actual_states[environment];
    const auto &expected = expected_states[environment];
    include_error(actual.position_w.x, expected.position_w.x);
    include_error(actual.position_w.y, expected.position_w.y);
    include_error(actual.position_w.z, expected.position_w.z);
    include_error(actual.attitude_wb.w, expected.attitude_wb.w);
    include_error(actual.attitude_wb.x, expected.attitude_wb.x);
    include_error(actual.attitude_wb.y, expected.attitude_wb.y);
    include_error(actual.attitude_wb.z, expected.attitude_wb.z);
    include_error(actual.linear_velocity_w.x, expected.linear_velocity_w.x);
    include_error(actual.linear_velocity_w.y, expected.linear_velocity_w.y);
    include_error(actual.linear_velocity_w.z, expected.linear_velocity_w.z);
    include_error(actual.angular_velocity_b.x, expected.angular_velocity_b.x);
    include_error(actual.angular_velocity_b.y, expected.angular_velocity_b.y);
    include_error(actual.angular_velocity_b.z, expected.angular_velocity_b.z);
    for (std::size_t rotor = 0; rotor < sim_cuda::kRotorCount; ++rotor) {
      include_error(actual.rotor_speed[rotor], expected.rotor_speed[rotor]);
    }
    const double quaternion_norm =
        std::sqrt(actual.attitude_wb.w * actual.attitude_wb.w +
                  actual.attitude_wb.x * actual.attitude_wb.x +
                  actual.attitude_wb.y * actual.attitude_wb.y +
                  actual.attitude_wb.z * actual.attitude_wb.z);
    require_near(quaternion_norm, 1.0, 2e-6, "CUDA quaternion norm");
  }
  require(maximum_error < 2e-3, "CPU/CUDA trajectory error exceeds tolerance");
}

void test_cuda_batch() {
  const auto device_parameters =
      as_device_parameters(sim_cuda::make_reference_quad_x());
  const auto reference_parameters = as_reference_parameters(device_parameters);
  std::vector<sim_cuda::DeviceState> initial_states(kEnvironmentCount);
  std::vector<sim_cuda::DeviceActions> actions(kEnvironmentCount);
  populate_inputs(initial_states, actions, device_parameters);

  const auto expected_states =
      run_reference_trajectory(initial_states, actions, reference_parameters);
  std::vector<sim_cuda::DeviceState> actual_states(kEnvironmentCount);
  const float elapsed_milliseconds = run_cuda_trajectory(
      initial_states, actions, device_parameters, actual_states);

  double maximum_error = 0.0;
  compare_trajectories(actual_states, expected_states, maximum_error);

  const double transitions_per_second = static_cast<double>(kEnvironmentCount) *
                                        kControlSteps * 1000.0 /
                                        elapsed_milliseconds;
  std::cout << "physics-only batch: " << kEnvironmentCount << " environments, "
            << kControlSteps << " steps, " << elapsed_milliseconds
            << " ms CUDA device time, " << transitions_per_second
            << " environment transitions/s, maximum CPU/CUDA error "
            << maximum_error << '\n';
}

} // namespace

int main() {
  try {
    test_cuda_batch();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "CUDA physics batch test failed: " << error.what() << '\n';
    return 1;
  }
}
