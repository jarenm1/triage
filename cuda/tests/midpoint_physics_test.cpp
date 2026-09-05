#include "physics_integration.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

using State = sim_cuda::MultirotorState<double>;
using Actions = sim_cuda::RotorActions<double>;
using Parameters = sim_cuda::VehicleParameters<double>;

void require_near(const double actual, const double expected,
                  const double tolerance, const std::string &quantity) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      !std::isfinite(tolerance) || tolerance < 0.0 ||
      std::abs(actual - expected) > tolerance) {
    std::ostringstream message;
    message << quantity << ": expected " << expected << ", got " << actual
            << " (tolerance " << tolerance << ')';
    throw std::runtime_error(message.str());
  }
}

State identity_state() {
  State state{};
  state.attitude_wb.w = 1.0;
  return state;
}

Parameters symmetric_parameters() {
  Parameters parameters{};
  parameters.mass = 1.0;
  parameters.inertia_diagonal_b = {0.8, 1.0, 1.2};
  constexpr double x[] = {1.0, -1.0, -1.0, 1.0};
  constexpr double y[] = {1.0, 1.0, -1.0, -1.0};
  for (std::size_t index = 0; index < sim_cuda::kRotorCount; ++index) {
    parameters.rotors[index] = {
        .position_b = {x[index], y[index], 0.0},
        .thrust_direction_b = {0.0, 0.0, 1.0},
        .reaction_torque_sign = index % 2 == 0 ? 1.0 : -1.0,
        .thrust_coefficient = 0.25,
        .torque_coefficient = 0.01,
        .minimum_speed = 0.0,
        .maximum_speed = 2.0,
        .time_constant = 0.2,
    };
  }
  return parameters;
}

void require_unit_attitude(const State &state) {
  const auto q = state.attitude_wb;
  require_near(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z, 1.0, 2e-14,
               "unit attitude");
}

State advance(State state, const Actions &actions, const Parameters &parameters,
              const double duration, const int steps) {
  for (int step = 0; step < steps; ++step) {
    state = sim_cuda::model::step_midpoint(state, actions, parameters,
                                           duration / steps);
    require_unit_attitude(state);
  }
  return state;
}

void test_continuous_freefall() {
  auto parameters = symmetric_parameters();
  parameters.gravity_w = {0.0, 0.0, -9.81};
  auto initial = identity_state();
  initial.position_w = {2.0, -1.0, 3.0};
  initial.linear_velocity_w = {0.4, -0.2, 0.7};
  constexpr double duration = 0.8;
  const auto next = advance(initial, {}, parameters, duration, 20);
  require_near(next.position_w.x, 2.0 + 0.4 * duration, 1e-13, "drift x");
  require_near(next.position_w.y, -1.0 - 0.2 * duration, 1e-13, "drift y");
  require_near(next.position_w.z,
               3.0 + 0.7 * duration - 0.5 * 9.81 * duration * duration, 1e-13,
               "continuous freefall position");
  require_near(next.linear_velocity_w.z, 0.7 - 9.81 * duration, 1e-13,
               "continuous freefall velocity");
}

void test_coupled_spinup() {
  const auto parameters = symmetric_parameters();
  auto initial = identity_state();
  constexpr double initial_speed = 0.3;
  constexpr double command = 2.0;
  constexpr double tau = 0.2;
  constexpr double duration = 0.4;
  initial.rotor_speed.fill(initial_speed);
  Actions actions{};
  actions.fill(1.0);
  // With total thrust coefficient / mass = 1, acceleration is exactly
  // [command + (initial_speed - command) exp(-t/tau)] squared.
  const double delta = initial_speed - command;
  const auto velocity_integral = [](const double rate) {
    return -std::expm1(-rate * duration) / rate;
  };
  const auto position_integral = [](const double rate) {
    return duration / rate + std::expm1(-rate * duration) / (rate * rate);
  };
  const double exact_velocity =
      command * command * duration +
      2.0 * command * delta * velocity_integral(1.0 / tau) +
      delta * delta * velocity_integral(2.0 / tau);
  const double exact_position =
      0.5 * command * command * duration * duration +
      2.0 * command * delta * position_integral(1.0 / tau) +
      delta * delta * position_integral(2.0 / tau);
  double previous_position_error = 0.0;
  double previous_velocity_error = 0.0;
  for (const int steps : {40, 80, 160}) {
    const auto next = advance(initial, actions, parameters, duration, steps);
    const double position_error = std::abs(next.position_w.z - exact_position);
    const double velocity_error =
        std::abs(next.linear_velocity_w.z - exact_velocity);
    require_near(next.position_w.z, exact_position, 1e-3,
                 "spinup analytic position");
    require_near(next.linear_velocity_w.z, exact_velocity, 1e-3,
                 "spinup analytic velocity");
    require_near(next.angular_velocity_b.x, 0.0, 1e-13, "symmetric roll");
    require_near(next.angular_velocity_b.y, 0.0, 1e-13, "symmetric pitch");
    require_near(next.angular_velocity_b.z, 0.0, 1e-13, "symmetric yaw");
    if (steps != 40) {
      require_near(position_error, 0.0, 0.3 * previous_position_error + 1e-14,
                   "spinup position second-order refinement");
      require_near(velocity_error, 0.0, 0.3 * previous_velocity_error + 1e-14,
                   "spinup velocity second-order refinement");
    }
    if (steps == 160) {
      require_near(next.position_w.z, exact_position, 1e-5,
                   "refined spinup position");
      require_near(next.linear_velocity_w.z, exact_velocity, 2e-5,
                   "refined spinup velocity");
    }
    previous_position_error = position_error;
    previous_velocity_error = velocity_error;
  }
}

void test_principal_axis_rotation() {
  const auto parameters = symmetric_parameters();
  auto initial = identity_state();
  constexpr double rate = 1.7;
  constexpr double duration = 0.8;
  initial.angular_velocity_b.x = rate;
  double previous_error = 0.0;
  for (const int steps : {20, 40, 80}) {
    const auto next = advance(initial, {}, parameters, duration, steps);
    require_near(next.angular_velocity_b.x, rate, 1e-13, "principal-axis rate");
    require_near(next.angular_velocity_b.y, 0.0, 1e-13, "cross-axis y rate");
    require_near(next.angular_velocity_b.z, 0.0, 1e-13, "cross-axis z rate");
    require_near(next.attitude_wb.w, std::cos(0.5 * rate * duration), 2e-4,
                 "principal-axis quaternion w");
    require_near(next.attitude_wb.x, std::sin(0.5 * rate * duration), 2e-4,
                 "principal-axis quaternion x");
    require_near(next.attitude_wb.y, 0.0, 1e-13, "principal-axis quaternion y");
    require_near(next.attitude_wb.z, 0.0, 1e-13, "principal-axis quaternion z");
    const double angle =
        2.0 * std::atan2(next.attitude_wb.x, next.attitude_wb.w);
    const double error = std::abs(angle - rate * duration);
    require_near(angle, rate * duration, 3e-4, "principal-axis angle");
    if (steps != 20) {
      require_near(error, 0.0, 0.3 * previous_error + 1e-14,
                   "rotation second-order refinement");
    }
    previous_error = error;
  }
}

void test_stable_motor_transient() {
  const auto parameters = symmetric_parameters();
  // RK2's negative-real-axis stability interval is dt/tau in (0, 2).
  // Exercise a ratio of 1.5, not an accuracy-oriented small step. Out-of-range
  // actions also ensure the motor ODE targets the clamped command.
  constexpr double dt = 1.5 * 0.2;
  for (const double action : {-0.5, 1.5}) {
    auto state = identity_state();
    const double target = action < 0.0 ? 0.0 : 2.0;
    state.rotor_speed.fill(2.0 - target);
    Actions actions{};
    actions.fill(action);
    double previous_error = 2.0;
    for (int step = 0; step < 32; ++step) {
      state = sim_cuda::model::step_midpoint(state, actions, parameters, dt);
      require_unit_attitude(state);
      for (const double speed : state.rotor_speed) {
        require_near(speed, 1.0, 1.0, "bounded rotor transient");
        require_near(speed, target, 0.9 * previous_error,
                     "contracting rotor transient");
      }
      previous_error = std::abs(state.rotor_speed[0] - target);
    }
    for (const double speed : state.rotor_speed) {
      require_near(speed, target, 1e-6, "settled rotor transient");
    }
  }
}

} // namespace

int main() {
  try {
    test_continuous_freefall();
    test_coupled_spinup();
    test_principal_axis_rotation();
    test_stable_motor_transient();
    std::cout << "midpoint physics tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "midpoint physics test failed: " << error.what() << '\n';
    return 1;
  }
}
