#include "hover_env.h"
#include "physics_batch.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
using namespace sim_cuda;
constexpr unsigned threads = 256;
constexpr float timestep = 0.000625f;
constexpr int substeps = 16;
constexpr float pi = 3.14159265358979323846f;
constexpr int plan_commands = 101;
thread_local char last_error[1024]{};

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

class DeviceGuard {
public:
  explicit DeviceGuard(int device) {
    check(cudaGetDevice(&previous_), "cudaGetDevice");
    if (device != previous_) {
      check(cudaSetDevice(device), "cudaSetDevice");
      changed_ = true;
    }
  }
  ~DeviceGuard() noexcept {
    if (changed_)
      cudaSetDevice(previous_);
  }

private:
  int previous_ = 0;
  bool changed_ = false;
};

// One structure is passed by value to kernels; every pointee is preallocated.
struct Buffers {
  float *observations = nullptr;
  float *rewards = nullptr;
  float *terminated = nullptr;
  float *truncated = nullptr;
  float *final_observations = nullptr;
  float *completed_returns = nullptr;
  float *completed_lengths = nullptr;
  ResetStatus *reset_status = nullptr;
  std::uint64_t *episode_counts = nullptr;
  float *current_returns = nullptr;
  float *current_lengths = nullptr;
  DeviceState *states = nullptr;
  DeviceVehicleParameters *parameters = nullptr;
  DeviceActions *commands = nullptr;
  std::uint8_t *reset_mask = nullptr;
  std::uint8_t *invalid_actions = nullptr;
  bool tracking = false;
  int schedule = 0;
  float *targets = nullptr;
  float *final_targets = nullptr;
  float *command_duration = nullptr;
  float *command_elapsed = nullptr;
  float *command_finished = nullptr;
  float *command_settled = nullptr;
  std::uint64_t *command_index = nullptr;
  float *action_saturation = nullptr;
  float *command_plan = nullptr;
  int *elapsed = nullptr;
  int *settled_steps = nullptr;
};

__device__ std::uint64_t mix(std::uint64_t value) {
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__device__ float uniform(std::uint64_t &rng) {
  rng += 0x9e3779b97f4a7c15ULL;
  return static_cast<float>(mix(rng) >> 40) * (1.0f / 16777216.0f);
}

__device__ float symmetric(std::uint64_t &rng, float extent) {
  return (2.0f * uniform(rng) - 1.0f) * extent;
}
// Duration draws use rejection to avoid modulo bias. This stream is separate
// from reset-state sampling and keyed by episode and command, never trajectory.
__device__ unsigned uniform_integer(std::uint64_t &rng, unsigned count) {
  const unsigned threshold = (0u - count) % count;
  unsigned value;
  do {
    rng += 0x9e3779b97f4a7c15ULL;
    value = static_cast<unsigned>(mix(rng));
  } while (value < threshold);
  return value % count;
}

__device__ void select_command(Buffers b, std::size_t i,
                               std::uint64_t command) {
  b.command_index[i] = command;
  const float *entry = b.command_plan + (i * plan_commands + command) * 4;
  for (int axis = 0; axis < 3; ++axis)
    b.targets[3 * i + axis] = entry[axis];
  b.elapsed[i] = b.settled_steps[i] = 0;
}

__device__ void generate_plan(Buffers b, std::size_t i, std::uint64_t seed,
                              std::uint64_t episode) {
  float previous[3] = {0.0f, 0.0f, 1.0f};
  for (int command = 0; command < plan_commands; ++command) {
    std::uint64_t rng = mix(seed ^ 0xa0761d6478bd642fULL) ^
                        mix(i + 0xe7037ed1a0b428dbULL) ^
                        mix(episode + 0x8ebc6af09c88c6e3ULL) ^
                        mix(command + 0x589965cc75374cc3ULL);
    float *entry = b.command_plan + (i * plan_commands + command) * 4;
    float displacement2;
    do {
      entry[0] = symmetric(rng, 1.0f);
      entry[1] = symmetric(rng, 1.0f);
      entry[2] = 1.25f + symmetric(rng, .5f);
      const float dx = entry[0] - previous[0];
      const float dy = entry[1] - previous[1];
      const float dz = entry[2] - previous[2];
      displacement2 = dx * dx + dy * dy + dz * dz;
    } while (displacement2 < .25f || displacement2 > 2.25f);
    if (b.schedule == 1) {
      entry[3] = 500.0f;
    } else {
      const bool short_interval = uniform_integer(rng, 4) == 0;
      entry[3] = short_interval ? 20 + uniform_integer(rng, 81)
                                : 200 + uniform_integer(rng, 301);
    }
    for (int axis = 0; axis < 3; ++axis)
      previous[axis] = entry[axis];
  }
  select_command(b, i, 0);
}

// Matches make_reference_quad_x, expressed on-device to avoid host staging.
__device__ DeviceVehicleParameters quad_parameters() {
  DeviceVehicleParameters p{};
  p.mass = 1.0f;
  p.inertia_diagonal_b = {0.0082f, 0.0082f, 0.0148f};
  p.gravity_w = {0.0f, 0.0f, -9.80665f};
  for (int rotor = 0; rotor < 4; ++rotor) {
    auto &r = p.rotors[rotor];
    r.position_b = {(rotor == 0 || rotor == 3) ? .17f : -.17f,
                    rotor < 2 ? .17f : -.17f, 0.0f};
    r.thrust_direction_b = {0.0f, 0.0f, 1.0f};
    r.reaction_torque_sign = rotor % 2 == 0 ? 1.0f : -1.0f;
    r.thrust_coefficient = 1.91e-6f;
    r.torque_coefficient = 2.6e-8f;
    r.minimum_speed = 0.0f;
    r.maximum_speed = 2200.0f;
    r.time_constant = .03f;
  }
  return p;
}

__device__ float hover_speed(const DeviceVehicleParameters &p) {
  float thrust = 0.0f;
  for (int r = 0; r < 4; ++r)
    thrust += p.rotors[r].thrust_coefficient;
  return sqrtf(-p.mass * p.gravity_w.z / thrust);
}

__device__ DeviceState reset_state(std::uint64_t seed, std::size_t index,
                                   std::uint64_t episode,
                                   const DeviceVehicleParameters &p) {
  std::uint64_t rng = mix(seed) ^ mix(index + 0x9e3779b97f4a7c15ULL) ^
                      mix(episode + 0xd1b54a32d192ed03ULL);
  DeviceState s{};
  s.position_w = {symmetric(rng, .15f), symmetric(rng, .15f),
                  1.0f + symmetric(rng, .15f)};
  const float roll = symmetric(rng, .05f) * .5f;
  const float pitch = symmetric(rng, .05f) * .5f;
  const float yaw = symmetric(rng, pi) * .5f;
  const float cr = cosf(roll), sr = sinf(roll);
  const float cp = cosf(pitch), sp = sinf(pitch);
  const float cy = cosf(yaw), sy = sinf(yaw);
  s.attitude_wb = {cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                   cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy};
  s.linear_velocity_w = {symmetric(rng, .05f), symmetric(rng, .05f),
                         symmetric(rng, .05f)};
  s.angular_velocity_b = {symmetric(rng, .02f), symmetric(rng, .02f),
                          symmetric(rng, .02f)};
  const float speed = hover_speed(p);
  for (int r = 0; r < 4; ++r)
    s.rotor_speed[r] = speed;
  return s;
}

__device__ bool observation(const DeviceState &s,
                            const DeviceVehicleParameters &p, float *out,
                            const float *target = nullptr) {
  out[0] = target ? s.position_w.x - target[0] : s.position_w.x;
  out[1] = target ? s.position_w.y - target[1] : s.position_w.y;
  out[2] = s.position_w.z - (target ? target[2] : 1.0f);
  out[3] = s.linear_velocity_w.x;
  out[4] = s.linear_velocity_w.y;
  out[5] = s.linear_velocity_w.z;
  const auto q = s.attitude_wb;
  out[6] = 1 - 2 * (q.y * q.y + q.z * q.z);
  out[7] = 2 * (q.x * q.y - q.w * q.z);
  out[8] = 2 * (q.x * q.z + q.w * q.y);
  out[9] = 2 * (q.x * q.y + q.w * q.z);
  out[10] = 1 - 2 * (q.x * q.x + q.z * q.z);
  out[11] = 2 * (q.y * q.z - q.w * q.x);
  out[12] = 2 * (q.x * q.z - q.w * q.y);
  out[13] = 2 * (q.y * q.z + q.w * q.x);
  out[14] = 1 - 2 * (q.x * q.x + q.y * q.y);
  out[15] = s.angular_velocity_b.x;
  out[16] = s.angular_velocity_b.y;
  out[17] = s.angular_velocity_b.z;
  for (int r = 0; r < 4; ++r)
    out[18 + r] = s.rotor_speed[r] / p.rotors[r].maximum_speed;
  bool valid = isfinite(q.w) && isfinite(q.x) && isfinite(q.y) && isfinite(q.z);
  for (int j = 0; j < 22; ++j)
    valid = valid && isfinite(out[j]);
  // An invalid terminal state is not a meaningful bootstrap observation. Clear
  // the entire row rather than allowing NaNs through a later masked multiply.
  if (!valid) {
    for (int j = 0; j < 22; ++j)
      out[j] = 0.0f;
    out[6] = out[10] = out[14] = 1.0f;
  }
  return valid;
}

__global__ void initialize(Buffers b, std::size_t n, std::uint64_t seed) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  b.parameters[i] = quad_parameters();
  b.states[i] = reset_state(seed, i, 0, b.parameters[i]);
  b.episode_counts[i] = 0;
  b.current_returns[i] = b.current_lengths[i] = 0;
  b.completed_returns[i] = b.completed_lengths[i] = 0;
  b.rewards[i] = b.terminated[i] = b.truncated[i] = 0;
  b.reset_mask[i] = 1;
  b.invalid_actions[i] = 0;
  b.reset_status[i] = ResetStatus::applied;
  if (b.tracking) {
    generate_plan(b, i, seed, 0);
    for (int axis = 0; axis < 3; ++axis)
      b.final_targets[3 * i + axis] = b.targets[3 * i + axis];
    b.command_duration[i] = b.command_elapsed[i] = 0.0f;
    b.command_finished[i] = b.command_settled[i] = 0.0f;
    b.action_saturation[i] = 0.0f;
  }
  observation(b.states[i], b.parameters[i], b.observations + 22 * i,
              b.tracking ? b.targets + 3 * i : nullptr);
  for (int j = 0; j < 22; ++j)
    b.final_observations[22 * i + j] = b.observations[22 * i + j];
}

__global__ void map_actions(Buffers b, std::size_t n, const float *actions) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const auto &p = b.parameters[i];
  const float speed = hover_speed(p);
  bool invalid = false;
  float cost = 0.0f;
  float saturated = 0.0f;
  for (int r = 0; r < 4; ++r) {
    const float raw = actions[4 * i + r];
    invalid = invalid || !isfinite(raw);
    const float bounded = isfinite(raw) ? tanhf(raw) : 0.0f;
    cost += bounded * bounded;
    if (b.tracking && (isinf(raw) || fabsf(bounded) >= .99f))
      saturated += .25f;
    const auto &rotor = p.rotors[r];
    const float hover = (speed - rotor.minimum_speed) /
                        (rotor.maximum_speed - rotor.minimum_speed);
    b.commands[i][r] = fminf(1.0f, fmaxf(0.0f, hover + .15f * bounded));
  }
  b.invalid_actions[i] = invalid;
  if (b.tracking)
    b.action_saturation[i] = saturated;
  // Reuse the reward slot as stream-local scratch until transition completion.
  b.rewards[i] = .005f * .25f * cost;
}

__global__ void finish_transition(Buffers b, std::size_t n, std::uint64_t seed,
                                  int max_steps) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float *out = b.final_observations + 22 * i;
  const bool finite = observation(b.states[i], b.parameters[i], out,
                                  b.tracking ? b.targets + 3 * i : nullptr);
  const float distance2 = out[0] * out[0] + out[1] * out[1] + out[2] * out[2];
  const float velocity2 = out[3] * out[3] + out[4] * out[4] + out[5] * out[5];
  const float rate2 = out[15] * out[15] + out[16] * out[16] + out[17] * out[17];
  const float tilt = acosf(fminf(1.0f, fmaxf(-1.0f, out[14])));
  const auto position = b.states[i].position_w;
  const bool outside = b.tracking
                           ? fabsf(position.x) > 3.0f ||
                                 fabsf(position.y) > 3.0f || position.z > 3.0f
                           : distance2 > 4.0f;
  const bool failed = !finite || b.invalid_actions[i] || position.z < .05f ||
                      outside || tilt > .8f;
  const float reward = failed ? -1.0f
                              : expf(-2.0f * distance2 - .1f * velocity2 -
                                     .05f * rate2 - .5f * tilt * tilt) -
                                    b.rewards[i];
  const float length = b.current_lengths[i] + 1.0f;
  const float episode_return = b.current_returns[i] + reward;
  const bool timeout = !failed && length >= max_steps;
  const bool done = failed || timeout;
  b.rewards[i] = reward;
  b.terminated[i] = failed ? 1.0f : 0.0f;
  b.truncated[i] = timeout ? 1.0f : 0.0f;
  b.reset_mask[i] = done;
  b.completed_returns[i] = done ? episode_return : 0.0f;
  b.completed_lengths[i] = done ? length : 0.0f;
  b.current_returns[i] = done ? 0.0f : episode_return;
  b.current_lengths[i] = done ? 0.0f : length;
  if (b.tracking) {
    for (int axis = 0; axis < 3; ++axis)
      b.final_targets[3 * i + axis] = b.targets[3 * i + axis];
    const auto command = b.command_index[i];
    const float duration =
        b.command_plan[(i * plan_commands + command) * 4 + 3];
    const int elapsed = ++b.elapsed[i];
    b.settled_steps[i] = !failed && distance2 <= .04f && velocity2 <= .04f
                             ? b.settled_steps[i] + 1
                             : 0;
    // Failure takes precedence, including failure exactly on a boundary.
    const bool finished = !failed && elapsed >= duration;
    b.command_duration[i] = duration;
    b.command_elapsed[i] = elapsed;
    b.command_finished[i] = finished ? 1.0f : 0.0f;
    b.command_settled[i] = finished && b.settled_steps[i] >= 50 ? 1.0f : 0.0f;
    if (finished && !done)
      select_command(b, i, command + 1);
  }
  if (done) {
    const auto episode = ++b.episode_counts[i];
    b.states[i] = reset_state(seed, i, episode, b.parameters[i]);
    if (b.tracking)
      generate_plan(b, i, seed, episode);
  }
}

__global__ void publish_observations(Buffers b, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  observation(b.states[i], b.parameters[i], b.observations + 22 * i,
              b.tracking ? b.targets + 3 * i : nullptr);
}

struct SnapshotSlot {
  Vec3<float> positions[64];
  Quaternion<float> attitudes[64];
  ExportStatus status[64];
  triage_snapshot_vehicle vehicles[64];
};

__global__ void pack_snapshot(Buffers b, const std::uint32_t *ids,
                              std::size_t count, SnapshotSlot *slot) {
  const unsigned j = threadIdx.x;
  if (j >= count)
    return;
  const auto i = ids[j];
  auto &out = slot->vehicles[j];
  out.environment_id = i;
  out.episode_id = b.episode_counts[i];
  const auto p = slot->positions[j];
  const auto q = slot->attitudes[j];
  out.position_w[0] = p.x;
  out.position_w[1] = p.y;
  out.position_w[2] = p.z;
  out.attitude_wb[0] = q.w;
  out.attitude_wb[1] = q.x;
  out.attitude_wb[2] = q.y;
  out.attitude_wb[3] = q.z;
  for (unsigned axis = 0; axis < 3; ++axis)
    out.target_w[axis] =
        b.tracking ? b.targets[3 * i + axis] : (axis == 2 ? 1.0f : 0.0f);
}

class SnapshotRing {
public:
  struct Slot {
    SnapshotSlot *device = nullptr;
    triage_snapshot_vehicle *host = nullptr;
    cudaEvent_t start = nullptr, gathered = nullptr;
    cudaEvent_t copying = nullptr, ready = nullptr;
    std::uint64_t step = 0;
  };
  SnapshotRing(const std::uint32_t *selection, std::size_t count,
               std::size_t capacity, std::size_t batch)
      : count(count), capacity(capacity) {
    if (!selection || count == 0 || count > 64 || capacity < 2 || capacity > 16)
      throw std::invalid_argument("snapshot needs 1..64 IDs and 2..16 slots");
    for (std::size_t i = 0; i < count; ++i) {
      if (selection[i] >= batch)
        throw std::invalid_argument("snapshot environment ID out of range");
      for (std::size_t j = 0; j < i; ++j)
        if (selection[i] == selection[j])
          throw std::invalid_argument("snapshot IDs must be unique");
    }
    try {
      check(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking),
            "create snapshot transfer stream");
      check(cudaMalloc(reinterpret_cast<void **>(&ids), count * sizeof(*ids)),
            "allocate snapshot IDs");
      check(cudaMemcpy(ids, selection, count * sizeof(*ids),
                       cudaMemcpyHostToDevice),
            "upload snapshot IDs");
      for (std::size_t i = 0; i < capacity; ++i) {
        auto &s = slots[i];
        check(cudaMalloc(reinterpret_cast<void **>(&s.device),
                         sizeof(SnapshotSlot)),
              "allocate snapshot slot");
        check(cudaMallocHost(reinterpret_cast<void **>(&s.host),
                             count * sizeof(*s.host)),
              "pin snapshot slot");
        check(cudaEventCreate(&s.start), "create snapshot start");
        check(cudaEventCreate(&s.gathered), "create snapshot gathered");
        check(cudaEventCreate(&s.copying), "create snapshot copying");
        check(cudaEventCreate(&s.ready), "create snapshot ready");
      }
    } catch (...) {
      release();
      throw;
    }
  }
  ~SnapshotRing() { release(); }
  std::size_t count, capacity, head = 0, tail = 0, pending = 0;
  std::uint32_t *ids = nullptr;
  cudaStream_t transfer = nullptr;
  Slot slots[16]{};

private:
  void release() noexcept {
    if (transfer)
      cudaStreamSynchronize(transfer);
    for (auto &s : slots) {
      if (s.device)
        cudaFree(s.device);
      if (s.host)
        cudaFreeHost(s.host);
      if (s.start)
        cudaEventDestroy(s.start);
      if (s.gathered)
        cudaEventDestroy(s.gathered);
      if (s.copying)
        cudaEventDestroy(s.copying);
      if (s.ready)
        cudaEventDestroy(s.ready);
    }
    if (ids)
      cudaFree(ids);
    if (transfer)
      cudaStreamDestroy(transfer);
  }
};

class HoverEnv {
public:
  HoverEnv(int device, std::size_t n, std::uint64_t seed, int max_steps,
           cudaStream_t stream, bool tracking = false, int schedule = 0)
      : device_(device), n_(n), seed_(seed), max_steps_(max_steps) {
    if (n == 0 || n > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument(
          "hover environment count must be in [1, UINT32_MAX]");
    if (max_steps <= 0 || max_steps > (1 << 24))
      throw std::invalid_argument("max_steps must be in [1, 16777216]");
    if (tracking && max_steps > 2000)
      throw std::invalid_argument("tracking max_steps must be in [1, 2000]");
    if (tracking && schedule != 0 && schedule != 1)
      throw std::invalid_argument(
          "tracking schedule must be 0 (mixed) or 1 (fixed500)");
    b_.tracking = tracking;
    b_.schedule = schedule;
    DeviceGuard guard(device_);
    validate_stream(stream);
    try {
      check(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming),
            "create hover event");
      allocate(b_.observations, 22);
      allocate(b_.rewards);
      allocate(b_.terminated);
      allocate(b_.truncated);
      allocate(b_.final_observations, 22);
      allocate(b_.completed_returns);
      allocate(b_.completed_lengths);
      allocate(b_.reset_status);
      allocate(b_.episode_counts);
      allocate(b_.current_returns);
      allocate(b_.current_lengths);
      allocate(b_.states);
      allocate(b_.parameters);
      allocate(b_.commands);
      allocate(b_.reset_mask);
      allocate(b_.invalid_actions);
      if (tracking) {
        allocate(b_.targets, 3);
        allocate(b_.final_targets, 3);
        allocate(b_.command_duration);
        allocate(b_.command_elapsed);
        allocate(b_.command_finished);
        allocate(b_.command_settled);
        allocate(b_.command_index);
        allocate(b_.action_saturation);
        allocate(b_.command_plan, plan_commands * 4);
        allocate(b_.elapsed);
        allocate(b_.settled_steps);
      }
      initialize<<<blocks(), threads, 0, stream>>>(b_, n_, seed_);
      check(cudaGetLastError(), "initialize hover task");
      physics_ = std::make_unique<PhysicsBatch>(
          BatchConfig{device_, n_, timestep, substeps},
          DeviceSpan<const DeviceState>{b_.states, n_},
          DeviceSpan<const DeviceVehicleParameters>{b_.parameters, n_}, stream);
      check(cudaEventRecord(completion_, stream),
            "record hover initialization");
      pending_ = true;
    } catch (...) {
      cudaStreamSynchronize(stream);
      release();
      throw;
    }
  }

  ~HoverEnv() noexcept { release(); }

  void reset(std::uint64_t seed, cudaStream_t stream) {
    DeviceGuard guard(device_);
    begin(stream);
    try {
      initialize<<<blocks(), threads, 0, stream>>>(b_, n_, seed);
      check(cudaGetLastError(), "reset hover task");
      apply_reset(stream);
      finish(stream);
      seed_ = seed;
    } catch (...) {
      recover(stream);
      throw;
    }
  }

  void step(const float *actions, cudaStream_t stream) {
    DeviceGuard guard(device_);
    validate_actions(actions);
    begin(stream);
    try {
      map_actions<<<blocks(), threads, 0, stream>>>(b_, n_, actions);
      check(cudaGetLastError(), "map hover actions");
      physics_->step({b_.commands, n_}, stream);
      ExportBuffers exports{};
      exports.states = {b_.states, n_};
      physics_->export_state({}, exports, stream);
      finish_transition<<<blocks(), threads, 0, stream>>>(b_, n_, seed_,
                                                          max_steps_);
      check(cudaGetLastError(), "finish hover transition");
      apply_reset(stream);
      // Export committed state, not reset candidates: even a rejected reset
      // must not advertise observations inconsistent with PhysicsBatch.
      physics_->export_state({}, exports, stream);
      publish_observations<<<blocks(), threads, 0, stream>>>(b_, n_);
      check(cudaGetLastError(), "publish hover observations");
      finish(stream);
    } catch (...) {
      recover(stream);
      throw;
    }
  }

  void configure_snapshot(const std::uint32_t *ids, std::size_t count,
                          std::size_t slots) {
    DeviceGuard guard(device_);
    if (snapshots_)
      throw std::invalid_argument("snapshot ring already configured");
    snapshots_ = std::make_unique<SnapshotRing>(ids, count, slots, n_);
  }

  void disable_snapshot() {
    DeviceGuard guard(device_);
    wait();
    snapshots_.reset();
  }

  int submit_snapshot(std::uint64_t step, cudaStream_t stream) {
    DeviceGuard guard(device_);
    if (!snapshots_)
      throw std::invalid_argument("snapshot ring is not configured");
    auto &ring = *snapshots_;
    if (ring.pending == ring.capacity)
      return 0;
    begin(stream);
    auto &s = ring.slots[ring.tail];
    try {
      check(cudaEventRecord(s.start, stream), "start snapshot gather");
      ExportBuffers out{};
      out.position_w = {s.device->positions, ring.count};
      out.attitude_wb = {s.device->attitudes, ring.count};
      out.status = {s.device->status, ring.count};
      physics_->export_state({SelectionKind::indexed, {ring.ids, ring.count}},
                             out, stream);
      pack_snapshot<<<1, 64, 0, stream>>>(b_, ring.ids, ring.count, s.device);
      check(cudaGetLastError(), "pack snapshot");
      check(cudaEventRecord(s.gathered, stream), "record snapshot gather");
      // Only the compact slot is read by transfer. Future physics can mutate
      // live state immediately after packing; it never waits for D2H.
      finish(stream);
      check(cudaStreamWaitEvent(ring.transfer, s.gathered, 0),
            "order snapshot transfer");
      check(cudaEventRecord(s.copying, ring.transfer), "start snapshot copy");
      check(cudaMemcpyAsync(s.host, s.device->vehicles,
                            ring.count * sizeof(*s.host),
                            cudaMemcpyDeviceToHost, ring.transfer),
            "copy snapshot");
      check(cudaEventRecord(s.ready, ring.transfer), "record snapshot ready");
      s.step = step;
      ring.tail = (ring.tail + 1) % ring.capacity;
      ++ring.pending;
      return 1;
    } catch (...) {
      recover(stream);
      // Fatal CUDA-error cleanup is outside the nonblocking success path.
      cudaStreamSynchronize(ring.transfer);
      throw;
    }
  }

  int poll_snapshot(triage_snapshot_vehicle *output, std::size_t count,
                    std::uint64_t *step, float *gather_ms, float *copy_ms) {
    DeviceGuard guard(device_);
    if (!snapshots_ || count != snapshots_->count || !output || !step ||
        !gather_ms || !copy_ms)
      throw std::invalid_argument("invalid snapshot poll output");
    auto &ring = *snapshots_;
    if (!ring.pending)
      return 0;
    auto &s = ring.slots[ring.head];
    const auto status = cudaEventQuery(s.ready);
    if (status == cudaErrorNotReady)
      return 0;
    check(status, "query snapshot readiness");
    check(cudaEventElapsedTime(gather_ms, s.start, s.gathered), "time gather");
    check(cudaEventElapsedTime(copy_ms, s.copying, s.ready), "time D2H");
    for (std::size_t i = 0; i < count; ++i)
      output[i] = s.host[i];
    *step = s.step;
    ring.head = (ring.head + 1) % ring.capacity;
    --ring.pending;
    return 1;
  }

  void *buffer(int field) {
    if (field >= TRIAGE_TRACKING_TARGETS && !b_.tracking)
      throw std::invalid_argument(
          "tracking buffers require a tracking environment");
    switch (field) {
    case TRIAGE_HOVER_OBSERVATIONS:
      return b_.observations;
    case TRIAGE_HOVER_REWARDS:
      return b_.rewards;
    case TRIAGE_HOVER_TERMINATED:
      return b_.terminated;
    case TRIAGE_HOVER_TRUNCATED:
      return b_.truncated;
    case TRIAGE_HOVER_FINAL_OBSERVATIONS:
      return b_.final_observations;
    case TRIAGE_HOVER_COMPLETED_RETURNS:
      return b_.completed_returns;
    case TRIAGE_HOVER_COMPLETED_LENGTHS:
      return b_.completed_lengths;
    case TRIAGE_HOVER_RESET_STATUS:
      return b_.reset_status;
    case TRIAGE_HOVER_EPISODE_COUNTS:
      return b_.episode_counts;
    case TRIAGE_HOVER_CURRENT_RETURNS:
      return b_.current_returns;
    case TRIAGE_HOVER_CURRENT_LENGTHS:
      return b_.current_lengths;
    case TRIAGE_TRACKING_TARGETS:
      return b_.targets;
    case TRIAGE_TRACKING_FINAL_TARGETS:
      return b_.final_targets;
    case TRIAGE_TRACKING_COMMAND_DURATION:
      return b_.command_duration;
    case TRIAGE_TRACKING_COMMAND_ELAPSED:
      return b_.command_elapsed;
    case TRIAGE_TRACKING_COMMAND_FINISHED:
      return b_.command_finished;
    case TRIAGE_TRACKING_COMMAND_SETTLED:
      return b_.command_settled;
    case TRIAGE_TRACKING_COMMAND_INDEX:
      return b_.command_index;
    case TRIAGE_TRACKING_ACTION_SATURATION:
      return b_.action_saturation;
    case TRIAGE_TRACKING_COMMAND_PLAN:
      return b_.command_plan;
    default:
      throw std::invalid_argument(
          "unknown environment buffer field; expected an id in [0,19]");
    }
  }

  void wait() {
    DeviceGuard guard(device_);
    if (pending_)
      check(cudaEventSynchronize(completion_), "wait for hover destruction");
  }

private:
  int device_;
  std::size_t n_;
  std::uint64_t seed_;
  int max_steps_;
  Buffers b_{};
  void *allocations_[27]{};
  std::size_t allocation_bytes_[27]{};
  unsigned allocation_count_ = 0;
  std::unique_ptr<PhysicsBatch> physics_;
  std::unique_ptr<SnapshotRing> snapshots_;
  cudaEvent_t completion_ = nullptr;
  bool pending_ = false;

  unsigned blocks() const {
    return static_cast<unsigned>((n_ + threads - 1) / threads);
  }

  template <typename T> void allocate(T *&pointer, std::size_t width = 1) {
    if (n_ > std::numeric_limits<std::size_t>::max() / sizeof(T) / width)
      throw std::invalid_argument("hover allocation size overflow");
    void *allocation = nullptr;
    check(cudaMalloc(&allocation, n_ * width * sizeof(T)),
          "allocate hover buffer");
    allocation_bytes_[allocation_count_] = n_ * width * sizeof(T);
    allocations_[allocation_count_++] = allocation;
    pointer = static_cast<T *>(allocation);
  }

  void validate_stream(cudaStream_t stream) const {
    int stream_device = -1;
    check(cudaStreamGetDevice(stream, &stream_device), "cudaStreamGetDevice");
    if (stream_device != device_)
      throw std::invalid_argument(
          "hover stream belongs to another CUDA device");
  }

  void validate_actions(const float *actions) const {
    if (!actions || reinterpret_cast<std::uintptr_t>(actions) % alignof(float))
      throw std::invalid_argument(
          "hover actions require an aligned nonnull float32 device pointer");
    cudaPointerAttributes attributes{};
    check(cudaPointerGetAttributes(&attributes, actions),
          "inspect hover action pointer");
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device_)
      throw std::invalid_argument(
          "hover actions must reside on the configured CUDA device");
    const auto begin_address = reinterpret_cast<std::uintptr_t>(actions);
    const auto bytes = n_ * 4 * sizeof(float);
    if (begin_address > std::numeric_limits<std::uintptr_t>::max() - bytes)
      throw std::invalid_argument("hover action address range overflows");
    // Task outputs are mutable during a step; disallow overlapping action
    // views.
    for (unsigned j = 0; j < allocation_count_; ++j) {
      const auto start = reinterpret_cast<std::uintptr_t>(allocations_[j]);
      if (begin_address < start + allocation_bytes_[j] &&
          start < begin_address + bytes)
        throw std::invalid_argument(
            "hover actions must not overlap task buffers");
    }
  }

  void begin(cudaStream_t stream) {
    validate_stream(stream);
    if (pending_)
      check(cudaStreamWaitEvent(stream, completion_, 0), "order hover stream");
  }

  void finish(cudaStream_t stream) {
    check(cudaEventRecord(completion_, stream), "record hover completion");
    pending_ = true;
  }

  void apply_reset(cudaStream_t stream) {
    physics_->apply_reset({b_.reset_mask, n_}, {b_.states, n_},
                          {b_.parameters, n_}, {b_.reset_status, n_}, stream);
  }

  void recover(cudaStream_t stream) noexcept {
    // Only CUDA-error cleanup may wait. Preserve ownership/order after partial
    // enqueue, including failures after the last PhysicsBatch operation.
    if (cudaEventRecord(completion_, stream) == cudaSuccess) {
      pending_ = true;
    } else {
      cudaStreamSynchronize(stream);
    }
  }

  void release() noexcept {
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess)
      return;
    if (previous != device_ && cudaSetDevice(device_) != cudaSuccess)
      return;
    if (pending_)
      cudaEventSynchronize(completion_);
    snapshots_.reset();
    physics_.reset();
    for (unsigned j = 0; j < allocation_count_; ++j)
      cudaFree(allocations_[j]);
    allocation_count_ = 0;
    if (completion_)
      cudaEventDestroy(completion_);
    completion_ = nullptr;
    pending_ = false;
    if (previous != device_)
      cudaSetDevice(previous);
  }
};

HoverEnv &environment(void *env) {
  if (!env)
    throw std::invalid_argument("hover environment handle is null");
  return *static_cast<HoverEnv *>(env);
}

void capture_error() noexcept {
  try {
    throw;
  } catch (const std::exception &error) {
    std::snprintf(last_error, sizeof(last_error), "%s", error.what());
  } catch (...) {
    std::snprintf(last_error, sizeof(last_error),
                  "unknown native hover exception");
  }
}
} // namespace

extern "C" void *triage_hover_create(int device, size_t n, uint64_t seed,
                                     int max_steps, void *stream) {
  last_error[0] = '\0';
  try {
    return new HoverEnv(device, n, seed, max_steps,
                        static_cast<cudaStream_t>(stream));
  } catch (...) {
    capture_error();
    return nullptr;
  }
}

extern "C" void *triage_tracking_create(int device, size_t n, uint64_t seed,
                                        int max_steps, int schedule,
                                        void *stream) {
  last_error[0] = '\0';
  try {
    return new HoverEnv(device, n, seed, max_steps,
                        static_cast<cudaStream_t>(stream), true, schedule);
  } catch (...) {
    capture_error();
    return nullptr;
  }
}

extern "C" int triage_hover_reset(void *env, uint64_t seed, void *stream) {
  last_error[0] = '\0';
  try {
    environment(env).reset(seed, static_cast<cudaStream_t>(stream));
    return 0;
  } catch (...) {
    capture_error();
    return -1;
  }
}

extern "C" int triage_hover_step(void *env, const float *actions,
                                 void *stream) {
  last_error[0] = '\0';
  try {
    environment(env).step(actions, static_cast<cudaStream_t>(stream));
    return 0;
  } catch (...) {
    capture_error();
    return -1;
  }
}

extern "C" void *triage_hover_buffer(void *env, int field) {
  last_error[0] = '\0';
  try {
    return environment(env).buffer(field);
  } catch (...) {
    capture_error();
    return nullptr;
  }
}

extern "C" int triage_hover_destroy(void *env) {
  last_error[0] = '\0';
  std::unique_ptr<HoverEnv> owner(static_cast<HoverEnv *>(env));
  try {
    if (owner)
      owner->wait();
    owner.reset();
    return 0;
  } catch (...) {
    capture_error();
    return -1;
  }
}

extern "C" const char *triage_hover_error(void) { return last_error; }

extern "C" int triage_snapshot_configure(void *env, const uint32_t *ids,
                                         size_t count, size_t slots) {
  last_error[0] = '\0';
  try {
    environment(env).configure_snapshot(ids, count, slots);
    return 0;
  } catch (...) {
    capture_error();
    return -1;
  }
}

extern "C" int triage_snapshot_submit(void *env, uint64_t step, void *stream) {
  last_error[0] = '\0';
  try {
    return environment(env).submit_snapshot(step,
                                            static_cast<cudaStream_t>(stream));
  } catch (...) {
    capture_error();
    return -1;
  }
}

extern "C" int triage_snapshot_poll(void *env, triage_snapshot_vehicle *output,
                                    size_t count, uint64_t *step,
                                    float *gather_ms, float *copy_ms) {
  last_error[0] = '\0';
  try {
    return environment(env).poll_snapshot(output, count, step, gather_ms,
                                          copy_ms);
  } catch (...) {
    capture_error();
    return -1;
  }
}

extern "C" int triage_snapshot_disable(void *env) {
  last_error[0] = '\0';
  try {
    environment(env).disable_snapshot();
    return 0;
  } catch (...) {
    capture_error();
    return -1;
  }
}
