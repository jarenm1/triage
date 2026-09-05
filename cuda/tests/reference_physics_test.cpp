#include "reference_physics.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

constexpr double kPhysicsTimestep = 0.001;

void require_near(const double actual, const double expected,
                  const double tolerance, const std::string &quantity) {
  if (std::abs(actual - expected) > tolerance) {
    std::ostringstream message;
    message << quantity << ": expected " << expected << ", got " << actual
            << " (tolerance " << tolerance << ')';
    throw std::runtime_error(message.str());
  }
}

sim_cuda::ReferenceState identity_state() {
  return {
      .position_w = {},
      .attitude_wb = {1.0, 0.0, 0.0, 0.0},
      .linear_velocity_w = {},
      .angular_velocity_b = {},
      .rotor_speed = {},
  };
}

void test_free_fall() {
  const auto parameters = sim_cuda::make_reference_quad_x();
  const auto initial = identity_state();
  const sim_cuda::ReferenceActions actions{};
  const auto next =
      sim_cuda::step_reference(initial, actions, parameters, kPhysicsTimestep);

  require_near(next.linear_velocity_w.z,
               parameters.gravity_w.z * kPhysicsTimestep, 1e-12,
               "free-fall vertical velocity");
  require_near(next.position_w.z,
               parameters.gravity_w.z * kPhysicsTimestep * kPhysicsTimestep,
               1e-12, "free-fall vertical position");
  require_near(next.attitude_wb.w, 1.0, 1e-12, "free-fall attitude");
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

void test_motor_response() {
  auto parameters = sim_cuda::make_reference_quad_x();
  parameters.gravity_w = {};
  for (auto &rotor : parameters.rotors) {
    rotor.thrust_coefficient = 0.0;
    rotor.torque_coefficient = 0.0;
  }
  const auto initial = identity_state();
  sim_cuda::ReferenceActions actions{};
  actions.fill(0.5);
  constexpr double timestep = 0.012;

  const auto next =
      sim_cuda::step_reference(initial, actions, parameters, timestep);
  const double command = 0.5 * parameters.rotors[0].maximum_speed;
  const double expected =
      command *
      (1.0 - std::exp(-timestep / parameters.rotors[0].time_constant));
  for (const double rotor_speed : next.rotor_speed) {
    require_near(rotor_speed, expected, 1e-12, "motor step response");
  }
}

} // namespace

int main() {
  try {
    test_free_fall();
    test_symmetric_hover();
    test_motor_response();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Reference physics test failed: " << error.what() << '\n';
    return 1;
  }
}
