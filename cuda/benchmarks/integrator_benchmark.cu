#include "integrator_cases.hpp"
#include "physics_integration.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <system_error>
#include <utility>
#include <vector>

namespace {
using namespace sim_cuda;
namespace cases = sim_cuda::benchmark;
using State = MultirotorState<float>;
using Actions = RotorActions<float>;
using Parameters = VehicleParameters<float>;
constexpr int trials = 5;
constexpr int threads = 256;
// Include the fine-step float32 regime: first-order truncation error can meet
// the accuracy gates only after roundoff has started to matter.
constexpr std::array<int, 12> sweep{1,  2,   4,   8,   16,   32,
                                    64, 128, 256, 512, 1024, 2048};

void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}
void cleanup_check(cudaError_t status, const char *operation) noexcept {
  if (status != cudaSuccess) {
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
  }
}
template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t count) {
    check(cudaMalloc(&data_, count * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() { cleanup_check(cudaFree(data_), "cudaFree"); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  T *get() const { return data_; }

private:
  T *data_ = nullptr;
};
class Stream {
public:
  Stream() {
    check(cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking),
          "create stream");
  }
  ~Stream() { cleanup_check(cudaStreamDestroy(value_), "destroy stream"); }
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  cudaStream_t get() const { return value_; }

private:
  cudaStream_t value_ = nullptr;
};
class Event {
public:
  Event() { check(cudaEventCreate(&value_), "create event"); }
  ~Event() { cleanup_check(cudaEventDestroy(value_), "destroy event"); }
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
  cudaEvent_t get() const { return value_; }

private:
  cudaEvent_t value_ = nullptr;
};

Vec3<float> narrow(Vec3<double> v) {
  return {static_cast<float>(v.x), static_cast<float>(v.y),
          static_cast<float>(v.z)};
}
State narrow(const ReferenceState &s) {
  State out{};
  out.position_w = narrow(s.position_w);
  out.linear_velocity_w = narrow(s.linear_velocity_w);
  out.angular_velocity_b = narrow(s.angular_velocity_b);
  out.attitude_wb = {
      static_cast<float>(s.attitude_wb.w), static_cast<float>(s.attitude_wb.x),
      static_cast<float>(s.attitude_wb.y), static_cast<float>(s.attitude_wb.z)};
  for (std::size_t i = 0; i < kRotorCount; ++i)
    out.rotor_speed[i] = static_cast<float>(s.rotor_speed[i]);
  return out;
}
Parameters narrow(const ReferenceVehicleParameters &p) {
  Parameters out{};
  out.mass = static_cast<float>(p.mass);
  out.inertia_diagonal_b = narrow(p.inertia_diagonal_b);
  out.gravity_w = narrow(p.gravity_w);
  for (std::size_t i = 0; i < kRotorCount; ++i) {
    const auto &r = p.rotors[i];
    out.rotors[i] = {narrow(r.position_b),
                     narrow(r.thrust_direction_b),
                     static_cast<float>(r.reaction_torque_sign),
                     static_cast<float>(r.thrust_coefficient),
                     static_cast<float>(r.torque_coefficient),
                     static_cast<float>(r.minimum_speed),
                     static_cast<float>(r.maximum_speed),
                     static_cast<float>(r.time_constant)};
  }
  return out;
}

__global__ void reset_states(State *states, const State *initial,
                             std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    states[i] = *initial;
}
template <bool Midpoint, bool Record>
__global__ void advance(State *states, const Actions *schedule,
                        Parameters parameters, std::size_t count, int control,
                        int substeps, State *endpoints) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  State state = states[i];
  const Actions action = schedule[control];
  const float dt = static_cast<float>(cases::control_dt / substeps);
  for (int step = 0; step < substeps; ++step) {
    if constexpr (Midpoint)
      state = model::step_midpoint(state, action, parameters, dt);
    else
      state = model::step_semi_implicit(state, action, parameters, dt);
  }
  states[i] = state;
  if constexpr (Record) {
    if (i == 0)
      endpoints[control] = state;
  }
}

struct Errors {
  double position = 0, velocity = 0, angular_rate = 0, attitude = 0, rotor = 0,
         quaternion_norm = 0;
  bool finite = true;
  void accumulate(double &maximum, double value) {
    if (!std::isfinite(value)) {
      finite = false;
      maximum = std::numeric_limits<double>::infinity();
    } else
      maximum = std::max(maximum, value);
  }
  bool passes(double rigid, double motor, double norm) const {
    return finite && position <= rigid && velocity <= rigid &&
           angular_rate <= rigid && attitude <= rigid && rotor <= motor &&
           quaternion_norm <= norm;
  }
};
template <typename A, typename B> double distance(Vec3<A> a, Vec3<B> b) {
  return std::hypot(static_cast<double>(a.x) - b.x,
                    static_cast<double>(a.y) - b.y,
                    static_cast<double>(a.z) - b.z);
}
template <typename Scalar>
std::array<double, 4> quaternion(Quaternion<Scalar> q) {
  return {static_cast<double>(q.w), static_cast<double>(q.x),
          static_cast<double>(q.y), static_cast<double>(q.z)};
}
double norm(const std::array<double, 4> &q) {
  return std::hypot(std::hypot(q[0], q[1]), std::hypot(q[2], q[3]));
}
template <typename Scalar>
Errors compare(const std::vector<MultirotorState<Scalar>> &actual,
               const std::vector<ReferenceState> &reference) {
  if (actual.size() != cases::control_steps ||
      reference.size() != actual.size())
    throw std::runtime_error("reference/endpoint trace has wrong size");
  Errors errors;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const auto &a = actual[i];
    const auto &b = reference[i];
    errors.accumulate(errors.position, distance(a.position_w, b.position_w));
    errors.accumulate(errors.velocity,
                      distance(a.linear_velocity_w, b.linear_velocity_w));
    errors.accumulate(errors.angular_rate,
                      distance(a.angular_velocity_b, b.angular_velocity_b));
    for (std::size_t r = 0; r < kRotorCount; ++r)
      errors.accumulate(
          errors.rotor,
          std::abs(static_cast<double>(a.rotor_speed[r]) - b.rotor_speed[r]));
    auto qa = quaternion(a.attitude_wb), qb = quaternion(b.attitude_wb);
    const double na = norm(qa), nb = norm(qb);
    errors.accumulate(errors.quaternion_norm, std::abs(na - 1));
    errors.accumulate(errors.quaternion_norm, std::abs(nb - 1));
    if (!std::isfinite(na) || !std::isfinite(nb) || na == 0 || nb == 0) {
      errors.accumulate(errors.attitude,
                        std::numeric_limits<double>::infinity());
      continue;
    }
    double dot = 0;
    for (int j = 0; j < 4; ++j) {
      qa[j] /= na;
      qb[j] /= nb;
      dot += qa[j] * qb[j];
    }
    // Chord/antichord atan2 remains accurate at tiny angles, unlike acos(dot).
    std::array<double, 4> difference{}, sum{};
    const double sign = dot < 0 ? -1 : 1;
    for (int j = 0; j < 4; ++j) {
      difference[j] = qa[j] - sign * qb[j];
      sum[j] = qa[j] + sign * qb[j];
    }
    errors.accumulate(errors.attitude,
                      4 * std::atan2(norm(difference), norm(sum)));
  }
  return errors;
}
void print_errors(const Errors &e) {
  std::cout << e.position << ',' << e.velocity << ',' << e.angular_rate << ','
            << e.attitude << ',' << e.rotor << ',' << e.quaternion_norm;
}
struct Timing {
  double minimum, median, maximum;
};
Timing summarize(std::array<double, trials> values) {
  std::sort(values.begin(), values.end());
  return {values.front(), values[trials / 2], values.back()};
}
struct Result {
  bool pass;
  double event_ms, wall_ms;
};

class Workspace {
public:
  explicit Workspace(std::size_t count)
      : states(count), initial(1), schedule(cases::control_steps),
        endpoints(cases::control_steps), count(count),
        blocks(static_cast<unsigned>((count + threads - 1) / threads)) {}
  void reset() {
    reset_states<<<blocks, threads, 0, stream.get()>>>(states.get(),
                                                       initial.get(), count);
    check(cudaGetLastError(), "reset launch");
    check(cudaStreamSynchronize(stream.get()), "reset completion");
  }
  template <bool Midpoint, bool Record>
  void trajectory(const Parameters &p, int substeps) {
    for (int c = 0; c < cases::control_steps; ++c) {
      advance<Midpoint, Record><<<blocks, threads, 0, stream.get()>>>(
          states.get(), schedule.get(), p, count, c, substeps, endpoints.get());
      check(cudaGetLastError(), "advance launch");
    }
  }
  template <bool Midpoint>
  Result measure(const cases::Scenario &scenario, int substeps,
                 const std::vector<ReferenceState> &reference) {
    const auto p = narrow(scenario.parameters);
    reset();
    trajectory<Midpoint, false>(p, substeps);
    check(cudaStreamSynchronize(stream.get()), "warmup completion");
    std::array<double, trials> gpu{}, wall{};
    for (int t = 0; t < trials; ++t) {
      reset();
      const auto before = std::chrono::steady_clock::now();
      check(cudaEventRecord(start.get(), stream.get()), "record start");
      trajectory<Midpoint, false>(p, substeps);
      check(cudaEventRecord(stop.get(), stream.get()), "record stop");
      check(cudaEventSynchronize(stop.get()), "timed completion");
      const auto after = std::chrono::steady_clock::now();
      wall[t] =
          std::chrono::duration<double, std::milli>(after - before).count();
      float elapsed = 0;
      check(cudaEventElapsedTime(&elapsed, start.get(), stop.get()),
            "event elapsed time");
      gpu[t] = elapsed;
      if (!(gpu[t] > 0) || !std::isfinite(gpu[t]) || !(wall[t] > 0) ||
          !std::isfinite(wall[t]))
        throw std::runtime_error("invalid timing interval");
    }
    // Endpoint recording/readback is a separate trajectory, never timed.
    reset();
    trajectory<Midpoint, true>(p, substeps);
    std::vector<State> trace(cases::control_steps);
    check(cudaMemcpyAsync(trace.data(), endpoints.get(),
                          trace.size() * sizeof(State), cudaMemcpyDeviceToHost,
                          stream.get()),
          "read accuracy trace");
    check(cudaStreamSynchronize(stream.get()), "accuracy completion");
    const auto errors = compare(trace, reference);
    const bool pass = errors.passes(1e-3, 0.1, 1e-5);
    const auto g = summarize(gpu), w = summarize(wall);
    const double vector_rate = cases::control_steps * 1000.0 / g.median;
    const double wall_vector_rate = cases::control_steps * 1000.0 / w.median;
    std::cout << "trial," << scenario.name << ','
              << (Midpoint ? "midpoint" : "baseline") << ',' << count << ','
              << substeps << ',' << cases::control_dt / substeps << ','
              << (errors.finite ? (pass ? "PASS" : "FAIL") : "UNSTABLE") << ',';
    print_errors(errors);
    std::cout << ',' << g.minimum << ',' << g.median << ',' << g.maximum << ','
              << w.minimum << ',' << w.median << ',' << w.maximum << ','
              << vector_rate << ',' << vector_rate * count << ','
              << vector_rate * count * substeps << ',' << wall_vector_rate
              << ',' << wall_vector_rate * count << ','
              << wall_vector_rate * count * substeps << '\n';
    return {pass, g.median, w.median};
  }
  Stream stream;
  DeviceBuffer<State> states, initial;
  DeviceBuffer<Actions> schedule;
  DeviceBuffer<State> endpoints;
  Event start, stop;
  std::size_t count;
  unsigned blocks;
};

std::size_t parse_count(int argc, char **argv) {
  if (argc == 1)
    return 10'000;
  if (argc != 3 || std::string_view(argv[1]) != "--environments")
    throw std::runtime_error("usage: sim_cuda_integrator_benchmark "
                             "[--environments N]; 1 <= N <= 10000000");
  std::string_view text(argv[2]);
  std::size_t count = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), count);
  if (text.empty() || result.ec != std::errc{} ||
      result.ptr != text.data() + text.size() || count == 0 ||
      count > 10'000'000)
    throw std::runtime_error(
        "--environments requires a decimal integer in [1,10000000]");
  return count;
}
void metadata(std::size_t count) {
  int device = 0, driver = 0, runtime = 0;
  check(cudaGetDevice(&device), "get device");
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, device), "device properties");
  check(cudaDriverGetVersion(&driver), "driver version");
  check(cudaRuntimeGetVersion(&runtime), "runtime version");
  std::cout << "# gpu=" << properties.name << "; device=" << device
            << "; compute_capability=" << properties.major << '.'
            << properties.minor << "; cuda_driver_api=" << driver
            << "; runtime=" << runtime << "; compiled_cuda=" << CUDART_VERSION
            << '\n';
  std::cout << "# precision=float32; oracle=float64 RK4; nvcc="
            << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__ << '.'
            << __CUDACC_VER_BUILD__;
#ifdef SIM_CUDA_BUILD_PROFILE
  std::cout << "; build_profile=" << SIM_CUDA_BUILD_PROFILE;
#else
  std::cout << "; build_profile=unspecified (provide SIM_CUDA_BUILD_PROFILE)";
#endif
#ifdef NDEBUG
  std::cout << "; NDEBUG=1\n";
#else
  std::cout << "; NDEBUG=0\n";
#endif
  utsname host{};
  if (uname(&host) != 0)
    throw std::runtime_error("uname failed");
  std::cout << "# host_os=" << host.sysname << ' ' << host.release << ' '
            << host.version << "; host_cpu_arch=" << host.machine
            << "; hostname=" << host.nodename << '\n';
  std::ifstream cpu_info("/proc/cpuinfo");
  std::string line;
  std::string cpu_model = "unavailable";
  while (std::getline(cpu_info, line)) {
    if (line.starts_with("model name")) {
      const auto colon = line.find(':');
      if (colon != std::string::npos)
        cpu_model = line.substr(colon + 1);
      break;
    }
  }
  std::ifstream driver_info("/proc/driver/nvidia/version");
  std::string driver_release;
  if (!std::getline(driver_info, driver_release))
    driver_release = "unavailable";
  std::cout << "# cpu_model=" << cpu_model
            << "; driver_release=" << driver_release << '\n';
  std::cout << "# determinism=fixed inputs and action schedule, no RNG or "
               "cross-thread reductions; cross-device bitwise identity not "
               "promised; fast_math=not enabled by project\n";
  std::cout
      << "# clock/power policy=uncontrolled; no clock or power query "
         "performed\n"
      << "# workload=homogeneous replicated batch; one thread/environment; "
         "in-place AoS; threads/block="
      << threads << '\n'
      << "# terms=first-order motor lag, quadratic rotor thrust/reaction "
         "torque, rotor arm torque, gravity, rigid-body gyroscopic coupling, "
         "normalized quaternion attitude\n"
      << "# excluded=contacts/collisions, aerodynamic drag, rotor gyroscopic "
         "dynamics, sensors, observations, rewards, resets/termination policy, "
         "rendering, policy inference, communication\n"
      << "# state_dimension=17 (quaternion stores 4); action_dimension=4; "
         "control_steps="
      << cases::control_steps << "; control_dt_s=" << cases::control_dt
      << "; warmup_full_trajectories=1/configuration; timed_trials=" << trials
      << '\n'
      << "# gates=max endpoint Euclidean position<=0.001m, velocity<=0.001m/s, "
         "angular_rate<=0.001rad/s, sign-invariant attitude<=0.001rad, "
         "component rotor<=0.1rad/s; quaternion_norm_error<=1e-5\n"
      << "# oracle convergence gates=1e-8 for rigid quantities, 1e-6 rotor "
         "speed; quaternion_norm_error<=1e-12\n"
      << "# timing=physics-only 100 control launches plus event/launch "
         "overhead; CUDA-event and wall include device completion; excludes "
         "allocation, upload, reset, accuracy recording/readback, logging\n"
      << "# rates=qualified physics-only, NOT full environment throughput; "
         "vector step=batch control step; transition=one environment control "
         "step; physics update=one environment substep\n"
      << "# owned_device_allocation_bytes="
      << (count + 1 + cases::control_steps) * sizeof(State) +
             cases::control_steps * sizeof(Actions)
      << "; excludes CUDA context/events/driver/code allocations; NOT "
         "whole-GPU peak; sizeof_state="
      << sizeof(State) << '\n';
}
} // namespace

int main(int argc, char **argv) {
  try {
    const auto count = parse_count(argc, argv);
    std::cout << std::setprecision(12);
    metadata(count);
    const auto scenarios = cases::scenarios();
    if (scenarios.empty())
      throw std::runtime_error("no scenarios supplied");
    std::vector<std::vector<ReferenceState>> references;
    references.reserve(scenarios.size());
    std::cout << "# oracle columns: "
                 "record,scenario,coarse_substeps,fine_substeps,status,"
                 "position_m,velocity_m_s,angular_rate_rad_s,attitude_rad,"
                 "rotor_rad_s,quaternion_norm_error\n";
    for (const auto &scenario : scenarios) {
      auto coarse = cases::reference_trace(scenario, 100);
      auto fine = cases::reference_trace(scenario, 200);
      const auto errors = compare(coarse, fine);
      const bool pass = errors.passes(1e-8, 1e-6, 1e-12);
      std::cout << "oracle," << scenario.name << ",100,200,"
                << (pass ? "PASS," : "FAIL,");
      print_errors(errors);
      std::cout << '\n';
      if (!pass)
        throw std::runtime_error(
            "RK4 oracle failed convergence/finiteness/norm certification");
      references.push_back(std::move(fine));
    }
    Workspace workspace(count);
    std::array<std::array<bool, sweep.size()>, 2> all_pass{};
    for (auto &method : all_pass)
      method.fill(true);
    std::array<std::array<double, sweep.size()>, 2> total_gpu{}, total_wall{};
    std::cout
        << "# trial columns: "
           "record,scenario,method,environments,substeps,physics_dt_s,status,"
           "position_m,velocity_m_s,angular_rate_rad_s,attitude_rad,rotor_rad_"
           "s,quaternion_norm_error,event_min_ms,event_median_ms,event_max_ms,"
           "wall_min_ms,wall_median_ms,wall_max_ms,event_vector_steps_s,event_"
           "environment_transitions_s,event_physics_updates_s,wall_vector_"
           "steps_s,wall_environment_transitions_s,wall_physics_updates_s\n";
    for (std::size_t s = 0; s < scenarios.size(); ++s) {
      const auto &scenario = scenarios[s];
      const auto initial = narrow(scenario.initial);
      std::array<Actions, cases::control_steps> schedule{};
      for (int c = 0; c < cases::control_steps; ++c) {
        const auto a = cases::actions_for(scenario, c);
        for (std::size_t r = 0; r < kRotorCount; ++r)
          schedule[c][r] = static_cast<float>(a[r]);
      }
      check(cudaMemcpyAsync(workspace.initial.get(), &initial, sizeof(initial),
                            cudaMemcpyHostToDevice, workspace.stream.get()),
            "upload scenario initial");
      check(cudaMemcpyAsync(workspace.schedule.get(), schedule.data(),
                            sizeof(schedule), cudaMemcpyHostToDevice,
                            workspace.stream.get()),
            "upload action schedule");
      check(cudaStreamSynchronize(workspace.stream.get()),
            "scenario upload completion");
      for (std::size_t k = 0; k < sweep.size(); ++k) {
        const std::array<Result, 2> results{
            workspace.measure<false>(scenario, sweep[k], references[s]),
            workspace.measure<true>(scenario, sweep[k], references[s])};
        for (int method = 0; method < 2; ++method) {
          all_pass[method][k] = all_pass[method][k] && results[method].pass;
          total_gpu[method][k] += results[method].event_ms;
          total_wall[method][k] += results[method].wall_ms;
        }
      }
    }
    std::cout << "# summary columns: "
                 "record,method,substeps,all_scenarios_pass,sum_scenario_event_"
                 "median_ms,sum_scenario_wall_median_ms\n";
    double fastest = std::numeric_limits<double>::infinity();
    int best_method = -1, best_substeps = 0;
    for (int method = 0; method < 2; ++method) {
      for (std::size_t k = 0; k < sweep.size(); ++k) {
        std::cout << "summary," << (method ? "midpoint" : "baseline") << ','
                  << sweep[k] << ',' << (all_pass[method][k] ? "PASS" : "FAIL")
                  << ',' << total_gpu[method][k] << ',' << total_wall[method][k]
                  << '\n';
        if (all_pass[method][k] && total_gpu[method][k] < fastest) {
          fastest = total_gpu[method][k];
          best_method = method;
          best_substeps = sweep[k];
        }
      }
    }
    if (best_method < 0)
      std::cout << "# selection=none meets all scenario gates\n";
    else
      std::cout << "# selection_by_sum_scenario_event_medians="
                << (best_method ? "midpoint" : "baseline")
                << "; substeps=" << best_substeps
                << "; summed_event_ms=" << fastest << '\n';
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0)
      throw std::runtime_error("getrusage failed");
    std::cout << "# host_process_max_rss_kib=" << usage.ru_maxrss
              << "; Linux getrusage high-water, includes oracle and CUDA host "
                 "runtime, NOT device memory\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "integrator benchmark: " << error.what() << '\n';
    return 1;
  }
}
