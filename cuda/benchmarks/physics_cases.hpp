#pragma once

#include "physics_model.hpp"
#include "reference_physics.hpp"

#include <cmath>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace sim_cuda::benchmark {

inline constexpr int control_steps = 100;
inline constexpr double control_dt = 0.01;

struct Scenario {
  std::string_view name;
  ReferenceVehicleParameters parameters;
  ReferenceState initial;
};

inline std::vector<Scenario> scenarios() {
  const auto parameters = make_reference_quad_x();
  ReferenceState initial{};
  initial.position_w = {0.0, 0.0, 10.0};
  initial.attitude_wb = {1.0, 0.0, 0.0, 0.0};
  const double hover_speed =
      std::sqrt(-parameters.mass * parameters.gravity_w.z /
                (kRotorCount * parameters.rotors[0].thrust_coefficient));
  initial.rotor_speed.fill(hover_speed);
  std::vector<Scenario> result{{"hover", parameters, initial},
                               {"motor_reversals", parameters, initial},
                               {"coupled_attitude", parameters, initial},
                               {"torque_free_rotation", parameters, initial}};
  result[1].initial.rotor_speed.fill(200.0);
  auto &coupled = result[2];
  coupled.initial.attitude_wb = {std::cos(0.175), 0.0, std::sin(0.175), 0.0};
  coupled.initial.angular_velocity_b = {3.0, -2.0, 4.0};
  coupled.initial.linear_velocity_w = {1.0, -0.5, 0.2};
  coupled.parameters.inertia_diagonal_b = {0.007, 0.010, 0.017};
  for (std::size_t rotor = 0; rotor < kRotorCount; ++rotor) {
    coupled.parameters.rotors[rotor].time_constant = 0.015 + 0.01 * rotor;
  }
  auto &rotation = result[3];
  rotation.parameters.gravity_w = {};
  rotation.parameters.inertia_diagonal_b = {0.007, 0.010, 0.017};
  rotation.initial.angular_velocity_b = {5.0, -3.0, 8.0};
  rotation.initial.rotor_speed.fill(0.0);
  for (auto &rotor : rotation.parameters.rotors) {
    rotor.thrust_coefficient = 0.0;
    rotor.torque_coefficient = 0.0;
  }
  return result;
}

// Piecewise-constant commands switch only on the common 100 Hz control grid.
// Every integrator therefore sees identical discontinuities, regardless of dt.
inline ReferenceActions actions_for(const Scenario &scenario, const int step) {
  ReferenceActions actions{};
  if (scenario.name == "torque_free_rotation") {
    return actions;
  }
  if (scenario.name == "motor_reversals") {
    actions.fill((step / 10) % 2 == 0 ? 0.8 : 0.2);
    return actions;
  }
  const double hover_action =
      std::sqrt(
          -scenario.parameters.mass * scenario.parameters.gravity_w.z /
          (kRotorCount * scenario.parameters.rotors[0].thrust_coefficient)) /
      scenario.parameters.rotors[0].maximum_speed;
  actions.fill(hover_action);
  if (scenario.name == "coupled_attitude") {
    const double sign = (step / 10) % 2 == 0 ? 1.0 : -1.0;
    actions[0] += 0.03 * sign;
    actions[1] -= 0.02 * sign;
    actions[2] += 0.01 * sign;
    actions[3] -= 0.025 * sign;
  }
  return actions;
}

namespace reference_detail {

inline Quaternion<double> normalized(Quaternion<double> q) {
  const double norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (!std::isfinite(norm) || norm == 0.0) {
    throw std::runtime_error("invalid reference attitude");
  }
  q.w /= norm;
  q.x /= norm;
  q.y /= norm;
  q.z /= norm;
  return q;
}

// Independent RK4 integration of the same physical equations, not repeated
// calls to production stepping. Stage motor speeds feed each stage's wrench.
inline ReferenceState derivative(const ReferenceState &state,
                                 const ReferenceActions &actions,
                                 const ReferenceVehicleParameters &parameters) {
  ReferenceState rate{};
  const auto wrench = model::rotor_wrench(state.rotor_speed, parameters);
  rate.position_w = state.linear_velocity_w;
  rate.linear_velocity_w = model::linear_acceleration_w(
      normalized(state.attitude_wb), wrench.force_b, parameters);
  rate.angular_velocity_b = model::angular_acceleration_b(
      state.angular_velocity_b, wrench.torque_b, parameters);
  rate.attitude_wb =
      model::attitude_derivative(state.attitude_wb, state.angular_velocity_b);
  for (std::size_t rotor = 0; rotor < kRotorCount; ++rotor) {
    const auto &p = parameters.rotors[rotor];
    const double command =
        p.minimum_speed + model::clamp_action(actions[rotor]) *
                              (p.maximum_speed - p.minimum_speed);
    rate.rotor_speed[rotor] =
        (command - state.rotor_speed[rotor]) / p.time_constant;
  }
  return rate;
}

inline ReferenceState add_scaled(const ReferenceState &state,
                                 const ReferenceState &rate, const double dt) {
  ReferenceState result = state;
  result.position_w =
      model::add(state.position_w, model::scale(rate.position_w, dt));
  result.linear_velocity_w = model::add(
      state.linear_velocity_w, model::scale(rate.linear_velocity_w, dt));
  result.angular_velocity_b = model::add(
      state.angular_velocity_b, model::scale(rate.angular_velocity_b, dt));
  result.attitude_wb.w += dt * rate.attitude_wb.w;
  result.attitude_wb.x += dt * rate.attitude_wb.x;
  result.attitude_wb.y += dt * rate.attitude_wb.y;
  result.attitude_wb.z += dt * rate.attitude_wb.z;
  for (std::size_t rotor = 0; rotor < kRotorCount; ++rotor) {
    result.rotor_speed[rotor] += dt * rate.rotor_speed[rotor];
  }
  return result;
}

inline ReferenceState rk4(const ReferenceState &state,
                          const ReferenceActions &actions,
                          const ReferenceVehicleParameters &parameters,
                          const double dt) {
  const auto k1 = derivative(state, actions, parameters);
  const auto k2 =
      derivative(add_scaled(state, k1, dt / 2.0), actions, parameters);
  const auto k3 =
      derivative(add_scaled(state, k2, dt / 2.0), actions, parameters);
  const auto k4 = derivative(add_scaled(state, k3, dt), actions, parameters);
  auto next = add_scaled(state, k1, dt / 6.0);
  next = add_scaled(next, k2, dt / 3.0);
  next = add_scaled(next, k3, dt / 3.0);
  next = add_scaled(next, k4, dt / 6.0);
  next.attitude_wb = normalized(next.attitude_wb);
  return next;
}

} // namespace reference_detail

inline std::vector<ReferenceState> reference_trace(const Scenario &scenario,
                                                   const int substeps) {
  if (substeps <= 0) {
    throw std::invalid_argument("reference substeps must be positive");
  }
  std::vector<ReferenceState> trace;
  trace.reserve(control_steps);
  auto state = scenario.initial;
  const double dt = control_dt / substeps;
  for (int control = 0; control < control_steps; ++control) {
    const auto actions = actions_for(scenario, control);
    for (int substep = 0; substep < substeps; ++substep) {
      state = reference_detail::rk4(state, actions, scenario.parameters, dt);
    }
    trace.push_back(state);
  }
  return trace;
}

} // namespace sim_cuda::benchmark
