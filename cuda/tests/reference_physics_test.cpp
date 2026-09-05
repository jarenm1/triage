#include "physics_model.hpp"
#include "reference_physics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

constexpr double kPhysicsTimestep = 0.001;

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

sim_cuda::ReferenceState identity_state() {
  return {.position_w = {},
          .attitude_wb = {1.0, 0.0, 0.0, 0.0},
          .linear_velocity_w = {},
          .angular_velocity_b = {},
          .rotor_speed = {}};
}

void test_zero_force_drift() {
  auto parameters = sim_cuda::make_reference_quad_x();
  parameters.gravity_w = {};
  auto initial = identity_state();
  initial.position_w = {2.0, -3.0, 4.0};
  initial.linear_velocity_w = {-0.5, 1.25, 2.0};
  const auto next = sim_cuda::step_reference(initial, {}, parameters, 0.01, 50);
  require_near(next.position_w.x, 1.75, 1e-12, "drift x position");
  require_near(next.position_w.y, -2.375, 1e-12, "drift y position");
  require_near(next.position_w.z, 5.0, 1e-12, "drift z position");
  require_near(next.linear_velocity_w.x, -0.5, 1e-12, "drift x velocity");
  require_near(next.linear_velocity_w.y, 1.25, 1e-12, "drift y velocity");
  require_near(next.linear_velocity_w.z, 2.0, 1e-12, "drift z velocity");
}

void test_analytic_convergence() {
  auto parameters = sim_cuda::make_reference_quad_x();
  constexpr double duration = 0.8;
  constexpr double rate = 1.7;
  double previous_rotation_error = 0.0;
  for (const int steps : {20, 40, 80}) {
    const double dt = duration / steps;
    auto initial = identity_state();
    initial.position_w.z = 3.0;
    initial.linear_velocity_w.z = 0.7;
    initial.angular_velocity_b.z = rate;
    const auto next =
        sim_cuda::step_reference(initial, {}, parameters, dt, steps);
    const double exact_position =
        3.0 + 0.7 * duration +
        0.5 * parameters.gravity_w.z * duration * duration;
    require_near(next.position_w.z, exact_position, 1e-12,
                 "continuous free-fall position");
    require_near(next.linear_velocity_w.z,
                 0.7 + parameters.gravity_w.z * duration, 1e-12,
                 "continuous free-fall velocity");
    require_near(next.angular_velocity_b.z, rate, 1e-12, "constant yaw rate");
    require_near(next.attitude_wb.x, 0.0, 1e-12,
                 "constant rotation quaternion x");
    require_near(next.attitude_wb.y, 0.0, 1e-12,
                 "constant rotation quaternion y");
    require_near(next.attitude_wb.w, std::cos(rate * duration / 2.0), 3e-4,
                 "constant rotation quaternion w");
    require_near(next.attitude_wb.z, std::sin(rate * duration / 2.0), 3e-4,
                 "constant rotation quaternion z");
    const double rotation_error =
        std::abs(2.0 * std::atan2(next.attitude_wb.z, next.attitude_wb.w) -
                 rate * duration);
    if (steps != 20) {
      require_near(rotation_error, 0.0, 0.3 * previous_rotation_error + 1e-13,
                   "rotation second-order-or-better refinement");
    }
    previous_rotation_error = rotation_error;
  }
}

void test_symmetric_hover() {
  const auto parameters = sim_cuda::make_reference_quad_x();
  auto initial = identity_state();
  const double hover_speed = std::sqrt(
      -parameters.mass * parameters.gravity_w.z /
      (sim_cuda::kRotorCount * parameters.rotors[0].thrust_coefficient));
  sim_cuda::ReferenceActions actions{};
  for (std::size_t index = 0; index < sim_cuda::kRotorCount; ++index) {
    initial.rotor_speed[index] = hover_speed;
    actions[index] = hover_speed / parameters.rotors[index].maximum_speed;
  }
  const auto next =
      sim_cuda::step_reference(initial, actions, parameters, kPhysicsTimestep);
  require_near(next.linear_velocity_w.z, 0.0, 1e-12, "hover vertical velocity");
  require_near(next.angular_velocity_b.x, 0.0, 1e-12, "hover roll rate");
  require_near(next.angular_velocity_b.y, 0.0, 1e-12, "hover pitch rate");
  require_near(next.angular_velocity_b.z, 0.0, 1e-12, "hover yaw rate");
}

void test_individual_rotors() {
  auto parameters = sim_cuda::make_reference_quad_x();
  parameters.gravity_w = {};
  // Independent quad-X geometry: front-left, rear-left, rear-right,
  // front-right; positive thrust along body z and alternating reaction torque.
  constexpr double roll_sign[] = {1.0, 1.0, -1.0, -1.0};
  constexpr double pitch_sign[] = {-1.0, 1.0, 1.0, -1.0};
  constexpr double yaw_sign[] = {1.0, -1.0, 1.0, -1.0};
  constexpr double speed = 1100.0;
  constexpr double force = 1.91e-6 * speed * speed;
  for (std::size_t rotor = 0; rotor < sim_cuda::kRotorCount; ++rotor) {
    auto initial = identity_state();
    initial.rotor_speed[rotor] = speed;
    const auto wrench =
        sim_cuda::model::rotor_wrench(initial.rotor_speed, parameters);
    const auto acceleration = sim_cuda::model::linear_acceleration_w(
        initial.attitude_wb, wrench.force_b, parameters);
    const auto angular_acceleration = sim_cuda::model::angular_acceleration_b(
        initial.angular_velocity_b, wrench.torque_b, parameters);
    require_near(acceleration.x, 0.0, 1e-12, "single rotor x acceleration");
    require_near(acceleration.y, 0.0, 1e-12, "single rotor y acceleration");
    require_near(acceleration.z, force, 1e-12, "single rotor z acceleration");
    require_near(angular_acceleration.x,
                 roll_sign[rotor] * 0.17 * force / 0.0082, 1e-12,
                 "rotor " + std::to_string(rotor) + " roll acceleration");
    require_near(angular_acceleration.y,
                 pitch_sign[rotor] * 0.17 * force / 0.0082, 1e-12,
                 "rotor " + std::to_string(rotor) + " pitch acceleration");
    require_near(angular_acceleration.z,
                 yaw_sign[rotor] * 2.6e-8 * speed * speed / 0.0148, 1e-12,
                 "rotor " + std::to_string(rotor) + " yaw acceleration");
  }
}

void test_tilted_thrust() {
  auto parameters = sim_cuda::make_reference_quad_x();
  parameters.gravity_w = {};
  auto initial = identity_state();
  constexpr double angle = 0.7;
  initial.attitude_wb = {std::cos(angle / 2.0), 0.0, std::sin(angle / 2.0),
                         0.0};
  initial.rotor_speed.fill(1100.0);
  sim_cuda::ReferenceActions actions{};
  actions.fill(0.5);
  const auto next =
      sim_cuda::step_reference(initial, actions, parameters, kPhysicsTimestep);
  constexpr double impulse = 4.0 * 1.91e-6 * 1100.0 * 1100.0 * kPhysicsTimestep;
  require_near(next.linear_velocity_w.x, impulse * std::sin(angle), 1e-12,
               "tilted thrust world x velocity");
  require_near(next.linear_velocity_w.y, 0.0, 1e-12,
               "tilted thrust world y velocity");
  require_near(next.linear_velocity_w.z, impulse * std::cos(angle), 1e-12,
               "tilted thrust world z velocity");
}

void test_motor_response() {
  auto parameters = sim_cuda::make_reference_quad_x();
  parameters.gravity_w = {};
  for (auto &rotor : parameters.rotors) {
    rotor.thrust_coefficient = 0.0;
    rotor.torque_coefficient = 0.0;
  }
  auto initial = identity_state();
  sim_cuda::ReferenceActions actions{};
  for (std::size_t rotor = 0; rotor < sim_cuda::kRotorCount; ++rotor) {
    initial.rotor_speed[rotor] = 200.0 + 300.0 * rotor;
    actions[rotor] = 0.8 - 0.2 * rotor;
    parameters.rotors[rotor].time_constant = 0.02 + 0.01 * rotor;
  }
  constexpr double duration = 0.04;
  double previous_error = 0.0;
  for (const int substeps : {40, 80, 160}) {
    const double timestep = duration / substeps;
    double error = 0.0;
    const auto next = sim_cuda::step_reference(initial, actions, parameters,
                                               timestep, substeps);
    for (std::size_t rotor = 0; rotor < sim_cuda::kRotorCount; ++rotor) {
      const double command =
          actions[rotor] * parameters.rotors[rotor].maximum_speed;
      const double expected =
          command +
          (initial.rotor_speed[rotor] - command) *
              std::exp(-duration / parameters.rotors[rotor].time_constant);
      require_near(next.rotor_speed[rotor], expected, 0.2,
                   "motor transient rotor " + std::to_string(rotor));
      error = std::max(error, std::abs(next.rotor_speed[rotor] - expected));
    }
    if (substeps != 40) {
      require_near(error, 0.0, 0.3 * previous_error + 1e-12,
                   "motor transient second-order refinement");
    }
    previous_error = error;
  }
}

void test_motor_stability_limit() {
  auto parameters = sim_cuda::make_reference_quad_x();
  // Put the limiting motor last to catch validation of only the first rotor.
  parameters.rotors[3].time_constant = 0.015625;
  const double limit = 2.0 * parameters.rotors[3].time_constant;
  auto actions = sim_cuda::ReferenceActions{};
  actions.fill(0.5);
  const auto stable =
      sim_cuda::step_reference(identity_state(), actions, parameters,
                               1.5 * parameters.rotors[3].time_constant);
  if (!(stable.rotor_speed[3] > 0.0 &&
        stable.rotor_speed[3] < 0.5 * parameters.rotors[3].maximum_speed)) {
    throw std::runtime_error(
        "stable motor command did not contract toward target");
  }
  for (const double dt : {limit, std::nextafter(limit, 1.0),
                          std::numeric_limits<double>::infinity()}) {
    bool rejected = false;
    try {
      sim_cuda::step_reference(identity_state(), {}, parameters, dt, 100);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!rejected) {
      throw std::runtime_error("non-decaying/unstable motor timestep accepted");
    }
  }
}

} // namespace

int main() {
  try {
    test_zero_force_drift();
    test_analytic_convergence();
    test_symmetric_hover();
    test_individual_rotors();
    test_tilted_thrust();
    test_motor_response();
    test_motor_stability_limit();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Reference physics test failed: " << error.what() << '\n';
    return 1;
  }
}
