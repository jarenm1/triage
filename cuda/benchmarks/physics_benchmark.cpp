#include "benchmark_report.hpp"
#include "physics.cuh"
#include "physics_cases.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
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
namespace report = sim_cuda::benchmark::report;
using State = DeviceState;
using Actions = DeviceActions;
using Parameters = DeviceVehicleParameters;
constexpr int trials = 5;
constexpr int production_substeps = 16;
constexpr std::size_t samples = 3;
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
    check(cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(T)),
          "cudaMalloc");
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
  if ((actual.size() != reference.size() &&
       actual.size() != samples * reference.size()) ||
      reference.size() != cases::control_steps)
    throw std::runtime_error("reference/endpoint trace has wrong size");
  Errors errors;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const auto &a = actual[i];
    const auto &b = reference[i / (actual.size() / reference.size())];
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

class Workspace {
public:
  explicit Workspace(std::size_t n)
      : initial(n), states(n), next(n), schedule(n * cases::control_steps),
        endpoints(samples * cases::control_steps), host_initial(n),
        host_schedule(n * cases::control_steps),
        trace(samples * cases::control_steps), count(n) {}

  void upload(const cases::Scenario &scenario) {
    parameters = narrow(scenario.parameters);
    std::fill(host_initial.begin(), host_initial.end(),
              narrow(scenario.initial));
    for (int c = 0; c < cases::control_steps; ++c) {
      Actions action{};
      const auto reference = cases::actions_for(scenario, c);
      for (std::size_t r = 0; r < kRotorCount; ++r)
        action[r] = static_cast<float>(reference[r]);
      std::fill_n(host_schedule.begin() + c * count, count, action);
    }
    check(cudaMemcpyAsync(initial.get(), host_initial.data(),
                          count * sizeof(State), cudaMemcpyHostToDevice,
                          stream.get()),
          "upload initial states");
    check(cudaMemcpyAsync(schedule.get(), host_schedule.data(),
                          host_schedule.size() * sizeof(Actions),
                          cudaMemcpyHostToDevice, stream.get()),
          "upload action schedule");
    check(cudaStreamSynchronize(stream.get()), "upload completion");
  }
  void reset() {
    check(cudaMemcpyAsync(states.get(), initial.get(), count * sizeof(State),
                          cudaMemcpyDeviceToDevice, stream.get()),
          "reset states");
    check(cudaStreamSynchronize(stream.get()), "reset completion");
  }
  void trajectory(int substeps, bool record) {
    State *current = states.get(), *output = next.get();
    const std::array<std::size_t, samples> indices{0, count / 2, count - 1};
    for (int c = 0; c < cases::control_steps; ++c) {
      sim_cuda::launch_physics_step(
          current, schedule.get() + c * count, output, count, parameters,
          static_cast<float>(cases::control_dt / substeps), substeps,
          stream.get());
      std::swap(current, output);
      if (record) {
        for (std::size_t sample = 0; sample < samples; ++sample)
          check(cudaMemcpyAsync(endpoints.get() + c * samples + sample,
                                current + indices[sample], sizeof(State),
                                cudaMemcpyDeviceToDevice, stream.get()),
                "record sampled control endpoint");
      }
    }
  }
  Errors accuracy(int substeps, const std::vector<ReferenceState> &reference) {
    reset();
    trajectory(substeps, true);
    check(cudaMemcpyAsync(trace.data(), endpoints.get(),
                          trace.size() * sizeof(State), cudaMemcpyDeviceToHost,
                          stream.get()),
          "read accuracy trace");
    check(cudaStreamSynchronize(stream.get()), "accuracy completion");
    return compare(trace, reference);
  }
  report::Measurement measure(const cases::Scenario &scenario,
                              const Errors &errors) {
    reset();
    trajectory(production_substeps, false);
    check(cudaStreamSynchronize(stream.get()), "warmup completion");
    std::array<double, trials> gpu{}, wall{};
    for (int t = 0; t < trials; ++t) {
      reset();
      const auto before = std::chrono::steady_clock::now();
      check(cudaEventRecord(start.get(), stream.get()), "record start");
      trajectory(production_substeps, false);
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
    const auto g = summarize(gpu), w = summarize(wall);
    return {std::string(scenario.name),
            count,
            production_substeps,
            errors.passes(1e-3, 0.1, 1e-5),
            g.minimum,
            g.median,
            g.maximum,
            w.minimum,
            w.median,
            w.maximum};
  }
  void memory_accounting() const {
    std::cout << "# environments=" << count
              << "; owned_device_allocation_bytes="
              << (3 * count + samples * cases::control_steps) * sizeof(State) +
                     count * cases::control_steps * sizeof(Actions)
              << "; host_input_and_endpoint_payload_bytes="
              << (count + samples * cases::control_steps) * sizeof(State) +
                     count * cases::control_steps * sizeof(Actions)
              << '\n';
  }

private:
  Stream stream;
  DeviceBuffer<State> initial, states, next;
  DeviceBuffer<Actions> schedule;
  DeviceBuffer<State> endpoints;
  std::vector<State> host_initial;
  std::vector<Actions> host_schedule;
  std::vector<State> trace;
  Event start, stop;
  std::size_t count;
  Parameters parameters{};
};

// FNV-1a over explicitly ordered IEEE scalar bits, least-significant byte
// first. Include both oracle inputs and narrowed device inputs, never object
// padding.
class Fingerprint {
public:
  template <typename T> void scalar(T value) {
    const auto bits = [&] {
      if constexpr (sizeof(T) == 4)
        return std::uint64_t(std::bit_cast<std::uint32_t>(value));
      else
        return std::bit_cast<std::uint64_t>(value);
    }();
    for (std::size_t i = 0; i < sizeof(T); ++i) {
      hash ^= (bits >> (i * 8)) & 0xff;
      hash *= UINT64_C(1099511628211);
    }
  }
  template <typename T> void vec(Vec3<T> v) {
    scalar(v.x);
    scalar(v.y);
    scalar(v.z);
  }
  template <typename T> void state(const MultirotorState<T> &s) {
    vec(s.position_w);
    vec(s.linear_velocity_w);
    vec(s.angular_velocity_b);
    scalar(s.attitude_wb.w);
    scalar(s.attitude_wb.x);
    scalar(s.attitude_wb.y);
    scalar(s.attitude_wb.z);
    for (const auto v : s.rotor_speed)
      scalar(v);
  }
  template <typename T> void parameters(const VehicleParameters<T> &p) {
    scalar(p.mass);
    vec(p.inertia_diagonal_b);
    vec(p.gravity_w);
    for (const auto &r : p.rotors) {
      vec(r.position_b);
      vec(r.thrust_direction_b);
      scalar(r.reaction_torque_sign);
      scalar(r.thrust_coefficient);
      scalar(r.torque_coefficient);
      scalar(r.minimum_speed);
      scalar(r.maximum_speed);
      scalar(r.time_constant);
    }
  }
  std::string str() const {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
  }

private:
  std::uint64_t hash = UINT64_C(14695981039346656037);
};
std::string fingerprint(const cases::Scenario &scenario) {
  Fingerprint f;
  f.parameters(scenario.parameters);
  f.state(scenario.initial);
  f.parameters(narrow(scenario.parameters));
  f.state(narrow(scenario.initial));
  for (int c = 0; c < cases::control_steps; ++c)
    for (const double action : cases::actions_for(scenario, c)) {
      f.scalar(action);
      f.scalar(static_cast<float>(action));
    }
  return f.str();
}

struct Options {
  std::size_t count = 10'000;
  std::string label, output, baseline;
  bool help = false;
};
Options options(int argc, char **argv) {
  Options result;
  std::set<std::string_view> seen;
  for (int i = 1; i < argc; ++i) {
    const std::string_view option(argv[i]);
    if (!seen.insert(option).second)
      throw std::runtime_error("duplicate option: " + std::string(option));
    if (option == "--help") {
      result.help = true;
      continue;
    }
    if (option != "--environments" && option != "--label" &&
        option != "--output" && option != "--baseline")
      throw std::runtime_error("unknown option: " + std::string(option));
    if (++i == argc || std::string_view(argv[i]).empty() ||
        std::string_view(argv[i]).starts_with("--"))
      throw std::runtime_error("missing value for " + std::string(option));
    const std::string_view value(argv[i]);
    if (option == "--environments") {
      const auto parsed = std::from_chars(
          value.data(), value.data() + value.size(), result.count);
      if (parsed.ec != std::errc{} ||
          parsed.ptr != value.data() + value.size() || result.count == 0 ||
          result.count > 100'000)
        throw std::runtime_error(
            "--environments requires a decimal integer in [1,100000]");
    } else if (option == "--label")
      result.label = value;
    else if (option == "--output")
      result.output = value;
    else
      result.baseline = value;
  }
  if ((!result.output.empty() || !result.baseline.empty()) &&
      result.label.empty())
    throw std::runtime_error("--label is required with --output or --baseline");
  if (!result.output.empty() && !result.baseline.empty()) {
    const auto output = std::filesystem::weakly_canonical(result.output);
    const auto baseline = std::filesystem::weakly_canonical(result.baseline);
    if (output == baseline ||
        (std::filesystem::exists(output) && std::filesystem::exists(baseline) &&
         std::filesystem::equivalent(output, baseline)))
      throw std::runtime_error("output must not overwrite the baseline report");
  }
  return result;
}

std::map<std::string, std::string> metadata(std::size_t count) {
  std::map<std::string, std::string> m;
  int device = 0, driver = 0, runtime = 0;
  check(cudaGetDevice(&device), "get device");
  cudaDeviceProp p{};
  check(cudaGetDeviceProperties(&p, device), "device properties");
  check(cudaDriverGetVersion(&driver), "driver version");
  check(cudaRuntimeGetVersion(&runtime), "runtime version");
  std::ostringstream uuid;
  uuid << std::hex << std::setfill('0');
  bool uuid_known = false;
  for (const char byte : p.uuid.bytes) {
    const auto value = static_cast<unsigned char>(byte);
    uuid_known |= value != 0;
    uuid << std::setw(2) << static_cast<unsigned>(value);
  }
  m["suite_version"] = "production-physics-v1";
  m["gpu_uuid"] = uuid_known ? uuid.str() : "unknown";
  m["gpu_name"] = p.name;
  m["gpu_compute_capability"] =
      std::to_string(p.major) + "." + std::to_string(p.minor);
  m["cuda_driver_api"] = std::to_string(driver);
  m["cuda_runtime"] = std::to_string(runtime);
  m["cuda_headers"] = std::to_string(CUDART_VERSION);
#ifdef SIM_CUDA_COMPILER_VERSION
  m["cuda_compiler"] = SIM_CUDA_COMPILER_VERSION;
#else
  m["cuda_compiler"] = "unknown";
#endif
#ifdef SIM_CUDA_HOST_COMPILER_VERSION
  m["host_compiler"] = SIM_CUDA_HOST_COMPILER_VERSION;
#else
  m["host_compiler"] = "unknown";
#endif
#ifdef SIM_CUDA_BUILD_PROFILE
  m["build_profile"] = SIM_CUDA_BUILD_PROFILE;
#else
  m["build_profile"] = "unknown";
#endif
#ifdef NDEBUG
  m["ndebug"] = "1";
#else
  m["ndebug"] = "0";
#endif
  utsname host{};
  const bool host_known = uname(&host) == 0;
  m["host_os"] = host_known ? std::string(host.sysname) + " " + host.release +
                                  " " + host.version
                            : "unknown";
  m["host_cpu_arch"] = host_known ? host.machine : "unknown";
  m["hostname"] = host_known ? host.nodename : "unknown";
  m["host_cpu_model"] = "unknown";
  std::ifstream cpu("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpu, line)) {
    if (line.starts_with("model name")) {
      const auto colon = line.find(':');
      if (colon != std::string::npos)
        m["host_cpu_model"] = line.substr(colon + 1);
      break;
    }
  }
  std::ifstream release("/proc/driver/nvidia/version");
  m["cuda_driver_release"] = std::getline(release, line) ? line : "unknown";
  m["precision"] = "float32 device; float64 RK4 oracle";
  m["base_environments"] = std::to_string(count);
  m["large_environments"] = std::to_string(count * 10);
  std::ostringstream control_dt;
  control_dt << std::setprecision(17) << cases::control_dt;
  m["control_dt_seconds"] = control_dt.str();
  m["control_steps"] = std::to_string(cases::control_steps);
  m["production_substeps"] = std::to_string(production_substeps);
  m["trials"] = std::to_string(trials);
  m["warmup_trajectories"] = "1";
  m["accuracy_gates"] = "position_m=1e-3;velocity_m/s=1e-3;angular_rate_rad/"
                        "s=1e-3;attitude_rad=1e-3;rotor_rad/"
                        "s=0.1;quaternion_norm=1e-5;all_finite";
  m["oracle_certification"] =
      "RK4 S=100 vs 200;rigid=1e-8;rotor=1e-6;quaternion_norm=1e-12";
  m["accuracy_sampling"] = "first,middle,last at every control endpoint";
  m["refinement"] = "motor_reversals "
                    "S=16,32,64;velocity<=max(0.6*previous,3e-5);rotor<=max(0."
                    "6*previous,0.005);all accuracy gates";
  m["layout"] =
      "AoS; homogeneous replicated batch; full control-major per-environment "
      "actions; D2D initial reset; two distinct pingpong state buffers";
  m["state_bytes"] = std::to_string(sizeof(State));
  m["action_bytes"] = std::to_string(sizeof(Actions));
  m["timed_scope"] =
      "public launch_physics_step each control; parameter validation and "
      "launch overhead; completion-inclusive events and wall; no "
      "reset/upload/readback/allocation/logging";
  m["clock_power_policy"] = "uncontrolled; no clock or power query performed";
  m["input_fingerprint"] =
      "FNV1a64; explicit oracle-double and device-float parameter/state/action "
      "scalars; IEEE bits little-endian; no padding";
  return m;
}
void print_measurement(const report::Measurement &r, const Errors &errors) {
  std::cout << "trial," << r.workload << ',' << r.environments << ','
            << r.substeps << ',' << cases::control_dt / r.substeps << ','
            << (r.accuracy_pass ? "PASS," : "FAIL,");
  print_errors(errors);
  std::cout << ',' << r.event_min_ms << ',' << r.event_median_ms << ','
            << r.event_max_ms << ',' << r.wall_min_ms << ',' << r.wall_median_ms
            << ',' << r.wall_max_ms;
  for (const double ms : {r.event_median_ms, r.wall_median_ms}) {
    const double vector_rate = cases::control_steps * 1000.0 / ms;
    std::cout << ',' << vector_rate << ',' << vector_rate * r.environments
              << ',' << vector_rate * r.environments * r.substeps;
  }
  std::cout << '\n';
}
} // namespace

int main(int argc, char **argv) {
  try {
    const auto config = options(argc, argv);
    if (config.help) {
      std::cout << "Usage: sim_cuda_physics_benchmark [--environments N] "
                   "[--label NAME] "
                   "[--output PATH] [--baseline PATH] [--help]\n"
                   "N defaults to 10000, range [1,100000]; larger workload "
                   "uses 10*N.\n"
                   "--label is required for saved reports/comparisons. "
                   "Performance deltas "
                   "are informational; correctness failures return nonzero.\n";
      return 0;
    }
    std::cout << std::setprecision(12);
    report::Run run;
    run.label = config.label;
    run.compatibility = metadata(config.count);
    const auto scenarios = cases::scenarios();
    if (scenarios.size() != 4)
      throw std::runtime_error("expected four physics scenarios");
    for (const auto &scenario : scenarios)
      run.compatibility["scenario_hash." + std::string(scenario.name)] =
          fingerprint(scenario);
    std::cout << "# label="
              << (config.label.empty() ? "unlabeled" : config.label) << '\n';
    for (const auto &[key, value] : run.compatibility)
      std::cout << "# " << key << '=' << value << '\n';
    std::cout
        << "# rates=qualified physics-only, NOT full environment throughput; "
           "vector step=batch control step; transition=one environment control "
           "step; physics update=one environment substep\n"
           "# excluded=contacts/collisions, aerodynamic drag, rotor gyroscopic "
           "dynamics, sensors, observations, rewards, resets/termination "
           "policy, rendering, inference, communication\n"
           "# memory=owned device payload only, excludes CUDA "
           "context/events/driver/code; not whole-GPU peak. Host payload "
           "excludes vector/allocator overhead, oracle storage, runtime and "
           "executable; RSS below includes these and is not device memory.\n"
           "# determinism=fixed inputs, no RNG or cross-thread reductions; no "
           "cross-device bitwise guarantee\n"
           "# spread=min/median/max over five trials; clocks uncontrolled; "
           "repeat measurements before interpreting performance changes\n"
           "# error "
           "columns=position_m,velocity_m_s,angular_rate_rad_s,attitude_rad,"
           "rotor_rad_s,quaternion_norm_error\n"
           "# trial "
           "columns=record,workload,environments,substeps,physics_dt_s,status,"
           "position_m,velocity_m_s,angular_rate_rad_s,attitude_rad,rotor_rad_"
           "s,quaternion_norm_error,event_min_ms,event_median_ms,event_max_ms,"
           "wall_min_ms,wall_median_ms,wall_max_ms,event_vector_steps_s,event_"
           "environment_transitions_s,event_physics_updates_s,wall_vector_"
           "steps_s,wall_environment_transitions_s,wall_physics_updates_s\n";
    std::vector<std::vector<ReferenceState>> references;
    for (const auto &scenario : scenarios) {
      const auto coarse = cases::reference_trace(scenario, 100);
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
    bool correct = true;
    bool refined = false;
    {
      Workspace workspace(config.count);
      workspace.memory_accounting();
      for (std::size_t s = 0; s < scenarios.size(); ++s) {
        workspace.upload(scenarios[s]);
        const auto errors =
            workspace.accuracy(production_substeps, references[s]);
        auto measurement = workspace.measure(scenarios[s], errors);
        if (scenarios[s].name == "motor_reversals") {
          auto previous = errors;
          bool refinement_pass = true;
          for (const int substeps : {32, 64}) {
            const auto finer = workspace.accuracy(substeps, references[s]);
            const bool pass =
                finer.passes(1e-3, 0.1, 1e-5) &&
                finer.velocity <= std::max(0.6 * previous.velocity, 3e-5) &&
                finer.rotor <= std::max(0.6 * previous.rotor, 0.005);
            std::cout << "refinement," << scenarios[s].name << ',' << substeps
                      << ',' << (pass ? "PASS," : "FAIL,");
            print_errors(finer);
            std::cout << '\n';
            refinement_pass &= pass;
            previous = finer;
          }
          measurement.accuracy_pass &= refinement_pass;
          refined = true;
        }
        correct &= measurement.accuracy_pass;
        print_measurement(measurement, errors);
        run.measurements.push_back(std::move(measurement));
      }
    }
    bool large_measured = false;
    for (std::size_t s = 0; s < scenarios.size(); ++s) {
      if (scenarios[s].name != "coupled_attitude")
        continue;
      Workspace workspace(config.count * 10);
      workspace.memory_accounting();
      workspace.upload(scenarios[s]);
      const auto errors =
          workspace.accuracy(production_substeps, references[s]);
      auto measurement = workspace.measure(scenarios[s], errors);
      correct &= measurement.accuracy_pass;
      print_measurement(measurement, errors);
      run.measurements.push_back(std::move(measurement));
      large_measured = true;
    }
    if (!refined || !large_measured || run.measurements.size() != 5)
      throw std::runtime_error("incomplete physics benchmark suite");
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0)
      std::cout << "# host_process_max_rss_kib=" << usage.ru_maxrss
                << "; Linux process high-water; NOT device memory\n";
    else
      std::cout << "# host_process_max_rss_kib=unknown\n";
    if (!config.output.empty())
      report::write_report(config.output, run);
    if (!config.baseline.empty())
      report::compare_report(config.baseline, run, std::cout);
    return correct ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "physics benchmark: " << error.what() << '\n';
    return 1;
  }
}
