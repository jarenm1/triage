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
                            const DeviceVehicleParameters &p, float *out) {
  out[0] = s.position_w.x;
  out[1] = s.position_w.y;
  out[2] = s.position_w.z - 1.0f;
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
  observation(b.states[i], b.parameters[i], b.observations + 22 * i);
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
  for (int r = 0; r < 4; ++r) {
    const float raw = actions[4 * i + r];
    invalid = invalid || !isfinite(raw);
    const float bounded = isfinite(raw) ? tanhf(raw) : 0.0f;
    cost += bounded * bounded;
    const auto &rotor = p.rotors[r];
    const float hover = (speed - rotor.minimum_speed) /
                        (rotor.maximum_speed - rotor.minimum_speed);
    b.commands[i][r] = fminf(1.0f, fmaxf(0.0f, hover + .15f * bounded));
  }
  b.invalid_actions[i] = invalid;
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
  const bool finite = observation(b.states[i], b.parameters[i], out);
  const float distance2 = out[0] * out[0] + out[1] * out[1] + out[2] * out[2];
  const float velocity2 = out[3] * out[3] + out[4] * out[4] + out[5] * out[5];
  const float rate2 = out[15] * out[15] + out[16] * out[16] + out[17] * out[17];
  const float tilt = acosf(fminf(1.0f, fmaxf(-1.0f, out[14])));
  const bool failed = !finite || b.invalid_actions[i] ||
                      b.states[i].position_w.z < .05f || distance2 > 4.0f ||
                      tilt > .8f;
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
  if (done) {
    const auto episode = ++b.episode_counts[i];
    b.states[i] = reset_state(seed, i, episode, b.parameters[i]);
  }
}

__global__ void publish_observations(Buffers b, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  observation(b.states[i], b.parameters[i], b.observations + 22 * i);
}

class HoverEnv {
public:
  HoverEnv(int device, std::size_t n, std::uint64_t seed, int max_steps,
           cudaStream_t stream)
      : device_(device), n_(n), seed_(seed), max_steps_(max_steps) {
    if (n == 0 || n > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument(
          "hover environment count must be in [1, UINT32_MAX]");
    if (max_steps <= 0 || max_steps > (1 << 24))
      throw std::invalid_argument("max_steps must be in [1, 16777216]");
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

  void *buffer(int field) {
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
    default:
      throw std::invalid_argument(
          "unknown hover buffer field; expected an id in [0,10]");
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
  void *allocations_[16]{};
  unsigned allocation_count_ = 0;
  std::unique_ptr<PhysicsBatch> physics_;
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
    const std::size_t sizes[16] = {22 * sizeof(float),
                                   sizeof(float),
                                   sizeof(float),
                                   sizeof(float),
                                   22 * sizeof(float),
                                   sizeof(float),
                                   sizeof(float),
                                   sizeof(ResetStatus),
                                   sizeof(std::uint64_t),
                                   sizeof(float),
                                   sizeof(float),
                                   sizeof(DeviceState),
                                   sizeof(DeviceVehicleParameters),
                                   sizeof(DeviceActions),
                                   sizeof(std::uint8_t),
                                   sizeof(std::uint8_t)};
    for (unsigned j = 0; j < allocation_count_; ++j) {
      const auto start = reinterpret_cast<std::uintptr_t>(allocations_[j]);
      if (begin_address < start + n_ * sizes[j] &&
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
