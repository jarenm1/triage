#include "physics_batch.hpp"
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

template <typename T>
void upload(DeviceBuffer<T> &destination, const std::vector<T> &source,
            cudaStream_t stream) {
  if (!source.empty()) {
    check_cuda(cudaMemcpyAsync(destination.get(), source.data(),
                               source.size() * sizeof(T),
                               cudaMemcpyHostToDevice, stream),
               "upload");
  }
}

template <typename T>
std::vector<T> download(const DeviceBuffer<T> &source, std::size_t count,
                        cudaStream_t stream) {
  std::vector<T> result(count);
  if (count != 0) {
    check_cuda(cudaMemcpyAsync(result.data(), source.get(), count * sizeof(T),
                               cudaMemcpyDeviceToHost, stream),
               "download");
  }
  check_cuda(cudaStreamSynchronize(stream), "wait for download");
  return result;
}

template <typename F> void require_rejected(F operation) {
  bool rejected = false;
  try {
    operation();
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "invalid public batch operation was accepted");
}

void run_cuda_trajectory(
    const std::vector<sim_cuda::DeviceState> &initial_states,
    const std::vector<sim_cuda::DeviceActions> &actions,
    const sim_cuda::DeviceVehicleParameters &parameters,
    std::vector<sim_cuda::DeviceState> &result) {
  Stream stream;
  const auto count = initial_states.size();
  DeviceBuffer<sim_cuda::DeviceState> initial(count), output(count);
  DeviceBuffer<sim_cuda::DeviceVehicleParameters> device_parameters(count);
  DeviceBuffer<sim_cuda::DeviceActions> device_actions(count);
  const std::vector<sim_cuda::DeviceVehicleParameters> parameter_rows(
      count, parameters);
  upload(initial, initial_states, stream.get());
  upload(device_parameters, parameter_rows, stream.get());
  upload(device_actions, actions, stream.get());
  sim_cuda::PhysicsBatch batch(
      {0, count, static_cast<float>(kPhysicsTimestep), kSubsteps},
      {initial.get(), count}, {device_parameters.get(), count}, stream.get());
  for (int step = 0; step < kControlSteps; ++step) {
    batch.step({device_actions.get(), count}, stream.get());
  }
  sim_cuda::ExportBuffers outputs;
  outputs.states = {output.get(), count};
  batch.export_state({}, outputs, stream.get());
  result = download(output, count, stream.get());
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
  run_cuda_trajectory(initial_states, actions, device_parameters,
                      actual_states);
  const auto errors = compare_trajectories(actual_states, expected_states);
  std::cout << "physics-only batch: " << environment_count << " environments, "
            << kControlSteps << " steps x " << kSubsteps << " substeps; "
            << "maximum errors: "
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

void require_vec(const sim_cuda::Vec3<float> &a,
                 const sim_cuda::Vec3<float> &b) {
  require(a.x == b.x && a.y == b.y && a.z == b.z, "vector changed");
}

void require_rotors(const sim_cuda::DeviceRotors &a,
                    const sim_cuda::DeviceRotors &b) {
  for (std::size_t r = 0; r < sim_cuda::kRotorCount; ++r) {
    require_vec(a[r].position_b, b[r].position_b);
    require_vec(a[r].thrust_direction_b, b[r].thrust_direction_b);
    require(a[r].reaction_torque_sign == b[r].reaction_torque_sign &&
                a[r].thrust_coefficient == b[r].thrust_coefficient &&
                a[r].torque_coefficient == b[r].torque_coefficient &&
                a[r].minimum_speed == b[r].minimum_speed &&
                a[r].maximum_speed == b[r].maximum_speed &&
                a[r].time_constant == b[r].time_constant,
            "rotor parameters changed");
  }
}

void require_parameters(const sim_cuda::DeviceVehicleParameters &a,
                        const sim_cuda::DeviceVehicleParameters &b) {
  require(a.mass == b.mass, "mass changed");
  require_vec(a.inertia_diagonal_b, b.inertia_diagonal_b);
  require_vec(a.gravity_w, b.gravity_w);
  require_rotors(a.rotors, b.rotors);
}

void require_state(const sim_cuda::DeviceState &a,
                   const sim_cuda::DeviceState &b) {
  require_vec(a.position_w, b.position_w);
  require_vec(a.linear_velocity_w, b.linear_velocity_w);
  require_vec(a.angular_velocity_b, b.angular_velocity_b);
  require(a.attitude_wb.w == b.attitude_wb.w &&
              a.attitude_wb.x == b.attitude_wb.x &&
              a.attitude_wb.y == b.attitude_wb.y &&
              a.attitude_wb.z == b.attitude_wb.z,
          "attitude changed");
  for (std::size_t r = 0; r < sim_cuda::kRotorCount; ++r) {
    require(a.rotor_speed[r] == b.rotor_speed[r], "motor state changed");
  }
}

void test_batch_contracts() {
  using namespace sim_cuda;
  constexpr std::size_t n = 4;
  constexpr float dt = 0.001F;
  constexpr int substeps = 16;
  Stream first, second, third;
  const auto base = as_device_parameters(make_reference_quad_x());
  std::vector<DeviceState> initial(n);
  std::vector<DeviceActions> actions(n);
  populate_inputs(initial, actions, base);
  std::vector<DeviceVehicleParameters> parameters(n, base);
  for (std::size_t i = 0; i < n; ++i) {
    parameters[i].mass *= 1.0F + 0.25F * i;
    parameters[i].gravity_w.x = 0.5F * i;
    parameters[i].rotors[3].time_constant *= 1.0F + 0.1F * i;
  }
  DeviceBuffer<DeviceState> state_input(n), before(n), reset_snapshot(n),
      after(n);
  DeviceBuffer<DeviceVehicleParameters> parameter_input(n), reset_parameters(n),
      after_parameters(n);
  DeviceBuffer<DeviceActions> action_input(n);
  DeviceBuffer<std::uint8_t> mask_input(n);
  DeviceBuffer<ResetStatus> reset_status(n);
  upload(state_input, initial, first.get());
  upload(parameter_input, parameters, first.get());
  upload(action_input, actions, first.get());
  PhysicsBatch batch({0, n, dt, substeps}, {state_input.get(), n},
                     {parameter_input.get(), n}, first.get());

  std::vector<ReferenceState> expected_before(n);
  for (std::size_t i = 0; i < n; ++i) {
    expected_before[i] = step_reference(
        as_reference_state(initial[i]), as_reference_actions(actions[i]),
        as_reference_parameters(parameters[i]), dt, substeps);
  }
  batch.step({action_input.get(), n}, first.get());
  ExportBuffers output;
  output.states = {before.get(), n};
  batch.export_state({}, output, second.get());

  auto replacement = initial;
  auto replacement_parameters = parameters;
  replacement[0].position_w = {4.0F, 5.0F, 6.0F};
  replacement_parameters[0].mass *= 2.0F;
  replacement_parameters[0].gravity_w = {3.0F, -2.0F, -1.0F};
  replacement[1].rotor_speed[3] = -1.0F;
  replacement_parameters[1].mass *= 3.0F;
  replacement[2].position_w.x = 42.0F;
  replacement_parameters[2].rotors[3].time_constant = dt / 2.0F;
  replacement[3].position_w.x = 99.0F;
  replacement_parameters[3].mass *= 4.0F;
  const std::vector<std::uint8_t> mask{2, 1, 255, 0};
  upload(state_input, replacement, third.get());
  upload(parameter_input, replacement_parameters, third.get());
  upload(mask_input, mask, third.get());
  batch.apply_reset({mask_input.get(), n}, {state_input.get(), n},
                    {parameter_input.get(), n}, {reset_status.get(), n},
                    third.get());
  output.states = {reset_snapshot.get(), n};
  output.parameters = {reset_parameters.get(), n};
  batch.export_state({}, output, first.get());
  batch.step({action_input.get(), n}, second.get());
  output.states = {after.get(), n};
  output.parameters = {after_parameters.get(), n};
  batch.export_state({}, output, third.get());
  // No host wait between step, snapshot, replacement, and the subsequent step.
  const auto final = download(after, n, third.get());
  const auto first_snapshot = download(before, n, second.get());
  const auto replaced = download(reset_snapshot, n, first.get());
  const auto replaced_parameters = download(reset_parameters, n, first.get());
  const auto final_parameters = download(after_parameters, n, third.get());
  const auto statuses = download(reset_status, n, third.get());
  require(statuses == std::vector<ResetStatus>{ResetStatus::applied,
                                               ResetStatus::invalid_state,
                                               ResetStatus::invalid_parameters,
                                               ResetStatus::not_selected},
          "reset statuses");
  compare_trajectories(first_snapshot, expected_before);
  require(std::abs(first_snapshot[0].linear_velocity_w.z -
                   first_snapshot[3].linear_velocity_w.z) > 0.01F,
          "per-environment mass must affect acceleration");
  std::vector<ReferenceState> expected_after(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto &expected_state = i == 0 ? replacement[i] : first_snapshot[i];
    const auto &expected_parameters =
        i == 0 ? replacement_parameters[i] : parameters[i];
    require_state(replaced[i], expected_state);
    require_parameters(replaced_parameters[i], expected_parameters);
    require_parameters(final_parameters[i], expected_parameters);
    expected_after[i] = step_reference(
        as_reference_state(expected_state), as_reference_actions(actions[i]),
        as_reference_parameters(expected_parameters), dt, substeps);
  }
  compare_trajectories(final, expected_after);

  const std::vector<std::uint32_t> indices{3, 0, 3, 4, UINT32_MAX, 1};
  const auto m = indices.size();
  DeviceBuffer<std::uint32_t> selector(m);
  DeviceBuffer<DeviceState> selected(m);
  DeviceBuffer<DeviceVehicleParameters> selected_parameters(m);
  DeviceBuffer<Vec3<float>> positions(m);
  DeviceBuffer<DeviceRotorSpeeds> motors(m);
  DeviceBuffer<DeviceRotors> rotors(m);
  DeviceBuffer<float> masses(m);
  DeviceBuffer<ExportStatus> export_status(m);
  std::vector<DeviceState> sentinel(m, initial[0]);
  std::vector<DeviceVehicleParameters> parameter_sentinel(m, base);
  std::vector<Vec3<float>> position_sentinel(m, {91.0F, 92.0F, 93.0F});
  std::vector<DeviceRotorSpeeds> motor_sentinel(m, {91, 92, 93, 94});
  std::vector<DeviceRotors> rotor_sentinel(m, base.rotors);
  std::vector<float> mass_sentinel(m, -17.0F);
  upload(selector, indices, first.get());
  upload(selected, sentinel, first.get());
  upload(selected_parameters, parameter_sentinel, first.get());
  upload(positions, position_sentinel, first.get());
  upload(motors, motor_sentinel, first.get());
  upload(rotors, rotor_sentinel, first.get());
  upload(masses, mass_sentinel, first.get());
  ExportSelection selection{SelectionKind::indexed, {selector.get(), m}};
  ExportBuffers selected_output;
  selected_output.states = {selected.get(), m};
  selected_output.parameters = {selected_parameters.get(), m};
  selected_output.position_w = {positions.get(), m};
  selected_output.rotor_speed = {motors.get(), m};
  selected_output.mass = {masses.get(), m};
  selected_output.rotors = {rotors.get(), m};
  selected_output.status = {export_status.get(), m};
  batch.export_state(selection, selected_output, first.get());
  const auto selected_states = download(selected, m, first.get());
  const auto selected_params = download(selected_parameters, m, first.get());
  const auto selected_positions = download(positions, m, first.get());
  const auto selected_motors = download(motors, m, first.get());
  const auto selected_rotors = download(rotors, m, first.get());
  const auto selected_masses = download(masses, m, first.get());
  const auto selected_status = download(export_status, m, first.get());
  for (std::size_t i = 0; i < m; ++i) {
    const bool valid = indices[i] < n;
    const auto &s = valid ? final[indices[i]] : sentinel[i];
    const auto &p =
        valid ? final_parameters[indices[i]] : parameter_sentinel[i];
    require(selected_status[i] ==
                (valid ? ExportStatus::exported : ExportStatus::invalid_index),
            "selected export status");
    require_state(selected_states[i], s);
    require_parameters(selected_params[i], p);
    require_vec(selected_positions[i],
                valid ? s.position_w : position_sentinel[i]);
    require_near(selected_masses[i], valid ? p.mass : mass_sentinel[i], 0,
                 "selected mass");
    require_rotors(selected_rotors[i], valid ? p.rotors : rotor_sentinel[i]);
    for (std::size_t r = 0; r < kRotorCount; ++r) {
      require_near(selected_motors[i][r],
                   valid ? s.rotor_speed[r] : motor_sentinel[i][r], 0,
                   "selected motor");
    }
  }

  auto bad = selected_output;
  bad.mass.size = m - 1;
  require_rejected(
      [&] { batch.step({action_input.get(), n - 1}, first.get()); });
  require_rejected([&] {
    batch.apply_reset({mask_input.get(), n - 1}, {state_input.get(), n},
                      {parameter_input.get(), n}, {reset_status.get(), n},
                      first.get());
  });
  require_rejected([&] { batch.export_state(selection, bad, first.get()); });
  bad = selected_output;
  bad.mass = {reinterpret_cast<float *>(selected.get()), m};
  require_rejected([&] { batch.export_state(selection, bad, first.get()); });
  bad = selected_output;
  bad.mass = {reinterpret_cast<float *>(selector.get()), m};
  require_rejected([&] { batch.export_state(selection, bad, first.get()); });
  bad = selected_output;
  bad.status = {};
  require_rejected([&] { batch.export_state(selection, bad, first.get()); });
  require_rejected([&] {
    batch.apply_reset({mask_input.get(), n}, {state_input.get(), n},
                      {parameter_input.get(), n},
                      {reinterpret_cast<ResetStatus *>(state_input.get()), n},
                      first.get());
  });
  const auto unchanged = download(selected, m, first.get());
  const auto unchanged_indices = download(selector, m, first.get());
  const auto unchanged_reset = download(state_input, n, first.get());
  require(unchanged_indices == indices, "rejected export overwrote selector");
  for (std::size_t i = 0; i < m; ++i) {
    require_state(unchanged[i], selected_states[i]);
  }
  for (std::size_t i = 0; i < n; ++i) {
    require_state(unchanged_reset[i], replacement[i]);
  }
  batch.export_state({SelectionKind::indexed, {}}, {}, second.get());

  PhysicsBatch empty({0, 0, dt, substeps}, {}, {}, first.get());
  empty.step({}, second.get());
  empty.apply_reset({}, {}, {}, {}, third.get());
  empty.export_state({}, {}, first.get());
  empty.export_state({SelectionKind::indexed, {}}, {}, second.get());
  empty.export_state(selection, selected_output, first.get());
  const auto empty_status = download(export_status, m, first.get());
  const auto empty_states = download(selected, m, first.get());
  const auto empty_parameters = download(selected_parameters, m, first.get());
  for (std::size_t i = 0; i < m; ++i) {
    require(empty_status[i] == ExportStatus::invalid_index,
            "empty batch indexed export must mark every index invalid");
    require_state(empty_states[i], selected_states[i]);
    require_parameters(empty_parameters[i], selected_params[i]);
  }
  ExportBuffers nonempty;
  nonempty.states = {selected.get(), 1};
  require_rejected([&] { empty.export_state({}, nonempty, first.get()); });
}

void test_constructor_validation() {
  using namespace sim_cuda;
  Stream stream;
  DeviceBuffer<DeviceState> states(1);
  DeviceBuffer<DeviceVehicleParameters> parameters(1);
  const auto base = as_device_parameters(make_reference_quad_x());
  DeviceState valid{};
  valid.attitude_wb.w = 1.0F;
  auto construct = [&](const DeviceState &state,
                       const DeviceVehicleParameters &params, float dt) {
    const std::vector<DeviceState> state_rows{state};
    const std::vector<DeviceVehicleParameters> parameter_rows{params};
    upload(states, state_rows, stream.get());
    upload(parameters, parameter_rows, stream.get());
    PhysicsBatch batch({0, 1, dt, 100}, {states.get(), 1},
                       {parameters.get(), 1}, stream.get());
  };
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (int fault = 0; fault < 6; ++fault) {
    auto state = valid;
    switch (fault) {
    case 0:
      state.position_w.z = nan;
      break;
    case 1:
      state.attitude_wb.w = 0.5F;
      break;
    case 2:
      state.linear_velocity_w.y = nan;
      break;
    case 3:
      state.angular_velocity_b.x = nan;
      break;
    case 4:
      state.rotor_speed[3] = -1.0F;
      break;
    case 5:
      state.rotor_speed[3] = nan;
      break;
    }
    require_rejected([&] { construct(state, base, 0.001F); });
  }
  for (int fault = 0; fault < 12; ++fault) {
    auto params = base;
    auto &last = params.rotors[3];
    switch (fault) {
    case 0:
      params.mass = 0.0F;
      break;
    case 1:
      params.inertia_diagonal_b.y = -1.0F;
      break;
    case 2:
      params.inertia_diagonal_b.z = 10.0F;
      break;
    case 3:
      params.gravity_w.x = nan;
      break;
    case 4:
      last.position_b.y = nan;
      break;
    case 5:
      last.thrust_direction_b.z = 0.5F;
      break;
    case 6:
      last.reaction_torque_sign = 0.0F;
      break;
    case 7:
      last.thrust_coefficient = -1.0F;
      break;
    case 8:
      last.torque_coefficient = nan;
      break;
    case 9:
      last.minimum_speed = -1.0F;
      break;
    case 10:
      last.maximum_speed = last.minimum_speed - 1.0F;
      break;
    case 11:
      last.time_constant = 0.0F;
      break;
    }
    require_rejected([&] { construct(valid, params, 0.001F); });
  }
  auto params = base;
  params.rotors[3].time_constant = 0.015625F;
  const float limit = 2.0F * params.rotors[3].time_constant;
  construct(valid, params, std::nextafter(limit, 0.0F));
  for (float dt : {limit, std::nextafter(limit, 1.0F),
                   std::numeric_limits<float>::infinity()}) {
    require_rejected([&] { construct(valid, params, dt); });
  }
  // Rotor state is physical memory, not a command subject to command limits.
  valid.rotor_speed[3] = base.rotors[3].maximum_speed + 10.0F;
  construct(valid, base, 0.001F);
  require_rejected([&] {
    PhysicsBatch batch({0, 1, 0.001F, 1}, {states.get(), 0},
                       {parameters.get(), 1}, stream.get());
  });
  require_rejected([&] {
    PhysicsBatch batch({0, 1, 0.001F, 1}, {states.get(), 1},
                       {parameters.get(), 0}, stream.get());
  });
  require_rejected([&] {
    PhysicsBatch batch({0, 1, 0.001F, 1}, {&valid, 1}, {parameters.get(), 1},
                       stream.get());
  });
  require_rejected([&] {
    PhysicsBatch batch({0, 1, 0.001F, 1}, {states.get(), 1}, {&base, 1},
                       stream.get());
  });
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
    test_batch_contracts();
    test_constructor_validation();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "CUDA physics batch test failed: " << error.what() << '\n';
    return 1;
  }
}
