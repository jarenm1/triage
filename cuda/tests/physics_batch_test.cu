#include "physics.cuh"
#include "reference_physics.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPhysicsTimestep = 0.001;
constexpr std::size_t kEnvironmentCount = 10'000;
constexpr int kControlSteps = 100;
constexpr int kSubsteps = 3;

void require(const bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_near(const double actual, const double expected,
                  const double tolerance, const std::string &quantity) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > tolerance) {
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
    if (count != 0) {
      check_cuda(cudaMalloc(&pointer_, count * sizeof(T)), "cudaMalloc");
    }
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

class Stream {
public:
  Stream() {
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "create non-default stream");
  }
  ~Stream() { cudaStreamDestroy(stream_); }
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  cudaStream_t get() const { return stream_; }

private:
  cudaStream_t stream_ = nullptr;
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
          parameters, kPhysicsTimestep, kSubsteps);
    }
  }
  return states;
}

float run_cuda_trajectory(
    const std::vector<sim_cuda::DeviceState> &initial_states,
    const std::vector<sim_cuda::DeviceActions> &actions,
    const sim_cuda::DeviceVehicleParameters &parameters,
    std::vector<sim_cuda::DeviceState> &result) {
  Stream stream;
  DeviceBuffer<sim_cuda::DeviceState> state_a(initial_states.size());
  DeviceBuffer<sim_cuda::DeviceState> state_b(initial_states.size());
  DeviceBuffer<sim_cuda::DeviceActions> device_actions(actions.size());
  if (!initial_states.empty()) {
    check_cuda(
        cudaMemcpyAsync(state_a.get(), initial_states.data(),
                        initial_states.size() * sizeof(sim_cuda::DeviceState),
                        cudaMemcpyHostToDevice, stream.get()),
        "copy initial states");
    check_cuda(cudaMemcpyAsync(device_actions.get(), actions.data(),
                               actions.size() * sizeof(sim_cuda::DeviceActions),
                               cudaMemcpyHostToDevice, stream.get()),
               "copy actions");
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "create start event");
  check_cuda(cudaEventCreate(&stop), "create stop event");
  check_cuda(cudaEventRecord(start, stream.get()), "record start event");

  sim_cuda::DeviceState *current = state_a.get();
  sim_cuda::DeviceState *next = state_b.get();
  for (int step = 0; step < kControlSteps; ++step) {
    sim_cuda::launch_physics_step(
        current, device_actions.get(), next, initial_states.size(), parameters,
        static_cast<float>(kPhysicsTimestep), kSubsteps, stream.get());
    std::swap(current, next);
  }

  check_cuda(cudaEventRecord(stop, stream.get()), "record stop event");
  check_cuda(cudaEventSynchronize(stop), "synchronize stop event");
  float elapsed_milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&elapsed_milliseconds, start, stop),
             "measure elapsed device time");
  check_cuda(cudaEventDestroy(start), "destroy start event");
  check_cuda(cudaEventDestroy(stop), "destroy stop event");

  if (!result.empty()) {
    check_cuda(cudaMemcpyAsync(result.data(), current,
                               result.size() * sizeof(sim_cuda::DeviceState),
                               cudaMemcpyDeviceToHost, stream.get()),
               "copy final states");
  }
  check_cuda(cudaStreamSynchronize(stream.get()), "synchronize result copy");
  return elapsed_milliseconds;
}

struct TrajectoryErrors {
  double position_m = 0.0;
  double velocity_m_per_s = 0.0;
  double angular_velocity_rad_per_s = 0.0;
  double orientation_rad = 0.0;
  double rotor_speed_rad_per_s = 0.0;
};

TrajectoryErrors compare_trajectories(
    const std::vector<sim_cuda::DeviceState> &actual_states,
    const std::vector<sim_cuda::ReferenceState> &expected_states) {
  require(actual_states.size() == expected_states.size(),
          "trajectory size mismatch");
  TrajectoryErrors errors;
  auto compare = [](const double actual, const double expected,
                    const double tolerance, double &maximum,
                    const char *quantity) {
    require_near(actual, expected, tolerance, quantity);
    maximum = std::max(maximum, std::abs(actual - expected));
  };
  for (std::size_t environment = 0; environment < actual_states.size();
       ++environment) {
    const auto &actual = actual_states[environment];
    const auto &expected = expected_states[environment];
    compare(actual.position_w.x, expected.position_w.x, 2e-4, errors.position_m,
            "position x [m]");
    compare(actual.position_w.y, expected.position_w.y, 2e-4, errors.position_m,
            "position y [m]");
    compare(actual.position_w.z, expected.position_w.z, 2e-4, errors.position_m,
            "position z [m]");
    compare(actual.linear_velocity_w.x, expected.linear_velocity_w.x, 2e-4,
            errors.velocity_m_per_s, "velocity x [m/s]");
    compare(actual.linear_velocity_w.y, expected.linear_velocity_w.y, 2e-4,
            errors.velocity_m_per_s, "velocity y [m/s]");
    compare(actual.linear_velocity_w.z, expected.linear_velocity_w.z, 2e-4,
            errors.velocity_m_per_s, "velocity z [m/s]");
    compare(actual.angular_velocity_b.x, expected.angular_velocity_b.x, 2e-4,
            errors.angular_velocity_rad_per_s, "roll rate [rad/s]");
    compare(actual.angular_velocity_b.y, expected.angular_velocity_b.y, 2e-4,
            errors.angular_velocity_rad_per_s, "pitch rate [rad/s]");
    compare(actual.angular_velocity_b.z, expected.angular_velocity_b.z, 2e-4,
            errors.angular_velocity_rad_per_s, "yaw rate [rad/s]");
    for (std::size_t rotor = 0; rotor < sim_cuda::kRotorCount; ++rotor) {
      compare(actual.rotor_speed[rotor], expected.rotor_speed[rotor], 4e-3,
              errors.rotor_speed_rad_per_s, "rotor speed [rad/s]");
    }
    const double a[] = {actual.attitude_wb.w, actual.attitude_wb.x,
                        actual.attitude_wb.y, actual.attitude_wb.z};
    const double b[] = {expected.attitude_wb.w, expected.attitude_wb.x,
                        expected.attitude_wb.y, expected.attitude_wb.z};
    double norm_a_squared = 0.0;
    double norm_b_squared = 0.0;
    double dot = 0.0;
    for (int component = 0; component < 4; ++component) {
      require(std::isfinite(a[component]) && std::isfinite(b[component]),
              "orientation contains a non-finite value");
      norm_a_squared += a[component] * a[component];
      norm_b_squared += b[component] * b[component];
      dot += a[component] * b[component];
    }
    const double norm_a = std::sqrt(norm_a_squared);
    const double norm_b = std::sqrt(norm_b_squared);
    require_near(norm_a, 1.0, 2e-6, "CUDA quaternion norm");
    require_near(norm_b, 1.0, 1e-12, "reference quaternion norm");
    double chord_squared = 0.0;
    const double sign = dot < 0.0 ? -1.0 : 1.0;
    for (int component = 0; component < 4; ++component) {
      const double difference =
          a[component] / norm_a - sign * b[component] / norm_b;
      chord_squared += difference * difference;
    }
    // The shorter unit-quaternion chord avoids acos cancellation near zero
    // and treats q and -q as the same physical orientation.
    const double angle =
        4.0 * std::asin(std::min(1.0, std::sqrt(chord_squared) / 2.0));
    compare(angle, 0.0, 5e-5, errors.orientation_rad,
            "orientation angle [rad]");
  }
  return errors;
}

void test_cuda_batch(const std::size_t environment_count) {
  const auto device_parameters =
      as_device_parameters(sim_cuda::make_reference_quad_x());
  const auto reference_parameters = as_reference_parameters(device_parameters);
  std::vector<sim_cuda::DeviceState> initial_states(environment_count);
  std::vector<sim_cuda::DeviceActions> actions(environment_count);
  populate_inputs(initial_states, actions, device_parameters);

  const auto expected_states =
      run_reference_trajectory(initial_states, actions, reference_parameters);
  std::vector<sim_cuda::DeviceState> actual_states(environment_count);
  const float elapsed_milliseconds = run_cuda_trajectory(
      initial_states, actions, device_parameters, actual_states);
  const auto errors = compare_trajectories(actual_states, expected_states);
  std::cout << "physics-only batch: " << environment_count << " environments, "
            << kControlSteps << " launches x " << kSubsteps << " substeps, "
            << elapsed_milliseconds << " ms CUDA device time; maximum errors: "
            << "position " << errors.position_m << " m, velocity "
            << errors.velocity_m_per_s << " m/s, angular velocity "
            << errors.angular_velocity_rad_per_s << " rad/s, orientation "
            << errors.orientation_rad << " rad, rotor speed "
            << errors.rotor_speed_rad_per_s << " rad/s\n";
}

void test_cuda_analytic_motion() {
  auto parameters = as_device_parameters(sim_cuda::make_reference_quad_x());
  parameters.gravity_w = {};
  for (auto &rotor : parameters.rotors) {
    rotor.thrust_coefficient = 0.0F;
    rotor.torque_coefficient = 0.0F;
  }
  std::vector<sim_cuda::DeviceState> initial(1);
  initial[0] = {
      .position_w = {1.0F, -2.0F, 3.0F},
      .attitude_wb = {1.0F, 0.0F, 0.0F, 0.0F},
      .linear_velocity_w = {0.5F, -1.0F, 2.0F},
      .angular_velocity_b = {0.0F, 0.0F, 1.7F},
      .rotor_speed = {200.0F, 500.0F, 800.0F, 1100.0F},
  };
  std::vector<sim_cuda::DeviceActions> actions(1);
  actions[0] = {0.8F, 0.6F, 0.4F, 0.2F};
  std::vector<sim_cuda::DeviceState> actual(1);
  run_cuda_trajectory(initial, actions, parameters, actual);

  // Continuous solutions, not another call to the shared stepping model.
  const double duration =
      static_cast<double>(static_cast<float>(kPhysicsTimestep)) *
      kControlSteps * kSubsteps;
  auto expected = as_reference_state(initial[0]);
  expected.position_w.x += 0.5 * duration;
  expected.position_w.y -= duration;
  expected.position_w.z += 2.0 * duration;
  const double half_angle = initial[0].angular_velocity_b.z * duration / 2.0;
  expected.attitude_wb = {std::cos(half_angle), 0.0, 0.0, std::sin(half_angle)};
  for (std::size_t rotor = 0; rotor < sim_cuda::kRotorCount; ++rotor) {
    const double command = static_cast<double>(actions[0][rotor]) *
                           parameters.rotors[rotor].maximum_speed;
    expected.rotor_speed[rotor] =
        command +
        (initial[0].rotor_speed[rotor] - command) *
            std::exp(-duration / parameters.rotors[rotor].time_constant);
  }
  compare_trajectories(actual, {expected});
}

void test_cuda_motor_stability_limit() {
  auto parameters = as_device_parameters(sim_cuda::make_reference_quad_x());
  parameters.rotors[3].time_constant = 0.015625F;
  const float limit = 2.0F * parameters.rotors[3].time_constant;
  DeviceBuffer<sim_cuda::DeviceState> states(1);
  DeviceBuffer<sim_cuda::DeviceActions> actions(1);
  for (const float dt : {limit, std::nextafter(limit, 1.0F),
                         std::numeric_limits<float>::infinity()}) {
    bool rejected = false;
    try {
      sim_cuda::launch_physics_step(states.get(), actions.get(), states.get(),
                                    1, parameters, dt, 100);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected,
            "CUDA launch accepted a non-decaying/unstable motor timestep");
  }
}

} // namespace

int main() {
  try {
    for (const std::size_t count :
         {std::size_t{0}, std::size_t{1}, std::size_t{255}, std::size_t{256},
          std::size_t{257}, kEnvironmentCount}) {
      test_cuda_batch(count);
    }
    test_cuda_analytic_motion();
    test_cuda_motor_stability_limit();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "CUDA physics batch test failed: " << error.what() << '\n';
    return 1;
  }
}
