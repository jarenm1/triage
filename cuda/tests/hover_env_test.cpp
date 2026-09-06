#include "hover_env.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void cuda_check(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
void check(int result) {
  if (result != 0)
    throw std::runtime_error(triage_hover_error());
}
struct Stream {
  cudaStream_t value{};
  Stream() {
    cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
  }
  ~Stream() { cudaStreamDestroy(value); }
};
struct Actions {
  float *value{};
  explicit Actions(std::size_t n) {
    cuda_check(cudaMalloc(&value, n * 4 * sizeof(float)));
    cuda_check(cudaMemset(value, 0, n * 4 * sizeof(float)));
  }
  ~Actions() { cudaFree(value); }
};
struct Environment {
  void *value{};
  Environment(std::size_t n, int limit, cudaStream_t stream,
              bool tracking = false, int schedule = 0) {
    value = tracking ? triage_tracking_create(0, n, 17, limit, schedule, stream)
                     : triage_hover_create(0, n, 17, limit, stream);
    require(value != nullptr, triage_hover_error());
  }
  ~Environment() { triage_hover_destroy(value); }
};
template <typename T>
std::vector<T> download(void *env, int field, std::size_t count,
                        cudaStream_t stream) {
  auto *pointer = triage_hover_buffer(env, field);
  require(pointer != nullptr, triage_hover_error());
  std::vector<T> values(count);
  cuda_check(cudaMemcpyAsync(values.data(), pointer, count * sizeof(T),
                             cudaMemcpyDeviceToHost, stream));
  cuda_check(cudaStreamSynchronize(stream));
  return values;
}
void test_timeout_and_reproducibility() {
  constexpr std::size_t n = 4;
  Stream first, second;
  Actions actions(n);
  Environment env(n, 2, first.value);
  const auto initial = download<float>(env.value, 0, n * 22, first.value);
  check(triage_hover_step(env.value, actions.value, first.value));
  check(triage_hover_step(env.value, actions.value, second.value));
  const auto terminal = download<float>(env.value, 2, n, second.value);
  const auto truncated = download<float>(env.value, 3, n, second.value);
  const auto final_obs = download<float>(env.value, 4, n * 22, second.value);
  const auto reset_obs = download<float>(env.value, 0, n * 22, second.value);
  const auto lengths = download<float>(env.value, 6, n, second.value);
  const auto episodes = download<std::uint64_t>(env.value, 8, n, second.value);
  const auto current_lengths = download<float>(env.value, 10, n, second.value);
  for (std::size_t i = 0; i < n; ++i) {
    require(terminal[i] == 0 && truncated[i] == 1,
            "timeout must only truncate");
    require(lengths[i] == 2 && current_lengths[i] == 0,
            "autoreset must retain final length and clear new episode length");
    require(episodes[i] == 1, "autoreset must advance episode identity once");
    bool different = false;
    for (std::size_t j = 0; j < 22; ++j) {
      const auto index = i * 22 + j;
      require(std::isfinite(final_obs[index]) &&
                  std::isfinite(reset_obs[index]),
              "observations must remain finite");
      different |= final_obs[index] != reset_obs[index];
    }
    require(different, "final observations must not be overwritten by reset");
  }
  check(triage_hover_reset(env.value, 17, first.value));
  require(download<float>(env.value, 0, n * 22, first.value) == initial,
          "same seed reset must reproduce initial observations");
  check(triage_hover_step(env.value, actions.value, second.value));
  const auto one_step = download<float>(env.value, 10, n, second.value);
  require(std::all_of(one_step.begin(), one_step.end(),
                      [](float length) { return length == 1; }),
          "reset must restart episode clocks");
  float host_actions[n * 4]{};
  require(triage_hover_step(env.value, host_actions, first.value) != 0,
          "host actions must be rejected");
  require(download<float>(env.value, 10, n, second.value) == one_step,
          "rejected action input must not advance the environment");
}
void test_failure_is_not_timeout() {
  constexpr std::size_t n = 4;
  Stream stream;
  Actions actions(n);
  Environment env(n, 1000, stream.value);
  std::vector<float> differential(n * 4);
  for (std::size_t i = 0; i < n; ++i) {
    differential[i * 4] = 10;
    differential[i * 4 + 1] = -10;
    differential[i * 4 + 2] = -10;
    differential[i * 4 + 3] = 10;
  }
  cuda_check(cudaMemcpy(actions.value, differential.data(),
                        differential.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  bool observed_failure = false;
  for (int step = 0; step < 300 && !observed_failure; ++step) {
    check(triage_hover_step(env.value, actions.value, stream.value));
    const auto terminal = download<float>(env.value, 2, n, stream.value);
    const auto truncated = download<float>(env.value, 3, n, stream.value);
    const auto rewards = download<float>(env.value, 1, n, stream.value);
    for (std::size_t i = 0; i < n; ++i) {
      if (terminal[i] != 0) {
        require(truncated[i] == 0, "failure must not also truncate");
        require(rewards[i] == -1,
                "failed transition must receive failure reward");
        observed_failure = true;
      }
    }
  }
  require(observed_failure,
          "sustained differential thrust must fail hover envelope");
}

void test_tracking_schedule_and_switches() {
  constexpr std::size_t n = 256, commands = 101;
  Stream stream;
  Actions actions(n);
  Environment env(n, 2000, stream.value, true);
  const auto plan =
      download<float>(env.value, 19, n * commands * 4, stream.value);
  bool short_seen = false, long_seen = false, back_to_back = false;
  for (std::size_t i = 0; i < n; ++i) {
    float previous[3]{0, 0, 1};
    bool previous_short = false;
    for (std::size_t c = 0; c < commands; ++c) {
      const float *target = plan.data() + (i * commands + c) * 4;
      require(std::abs(target[0]) <= 1 && std::abs(target[1]) <= 1 &&
                  target[2] >= .75F && target[2] <= 1.75F,
              "tracking target outside agreed bounds");
      float distance2 = 0;
      for (int j = 0; j < 3; ++j) {
        distance2 += (target[j] - previous[j]) * (target[j] - previous[j]);
        previous[j] = target[j];
      }
      require(distance2 >= .25F - 1e-6F && distance2 <= 2.25F + 1e-6F,
              "successive commanded displacement outside agreed bounds");
      const bool brief = target[3] >= 20 && target[3] <= 100;
      const bool hold = target[3] >= 200 && target[3] <= 500;
      require((brief || hold) && std::floor(target[3]) == target[3],
              "command duration must belong to an agreed integer interval");
      short_seen |= brief;
      long_seen |= hold;
      back_to_back |= previous_short && brief;
      previous_short = brief;
    }
  }
  require(short_seen && long_seen && back_to_back,
          "seeded plans must exercise holds and consecutive interruptions");
  auto previous_targets = download<float>(env.value, 11, n * 3, stream.value);
  bool switched = false, far_from_target = false;
  for (int step = 0; step < 200; ++step) {
    check(triage_hover_step(env.value, actions.value, stream.value));
    const auto targets = download<float>(env.value, 11, n * 3, stream.value);
    const auto final_targets =
        download<float>(env.value, 12, n * 3, stream.value);
    const auto final_obs = download<float>(env.value, 4, n * 22, stream.value);
    const auto obs = download<float>(env.value, 0, n * 22, stream.value);
    const auto rewards = download<float>(env.value, 1, n, stream.value);
    const auto finished = download<float>(env.value, 15, n, stream.value);
    const auto elapsed = download<float>(env.value, 14, n, stream.value);
    const auto duration = download<float>(env.value, 13, n, stream.value);
    const auto episode = download<std::uint64_t>(env.value, 8, n, stream.value);
    require(final_targets == previous_targets,
            "transition must retain its active target before switching");
    for (std::size_t i = 0; i < n; ++i) {
      require(episode[i] == 0,
              "a command switch must not reset the physical episode");
      const float *before = final_obs.data() + 22 * i;
      const float *after = obs.data() + 22 * i;
      for (int j = 0; j < 3; ++j)
        require(std::abs(before[j] + final_targets[3 * i + j] - after[j] -
                         targets[3 * i + j]) < 1e-6F,
                "command switch must not teleport physical position");
      for (int j = 3; j < 22; ++j)
        require(
            before[j] == after[j],
            "command switch must preserve velocity attitude and motor state");
      const float tilt = std::acos(std::clamp(before[14], -1.0F, 1.0F));
      float distance2 = 0, velocity2 = 0, rate2 = 0;
      for (int j = 0; j < 3; ++j) {
        distance2 += before[j] * before[j];
        velocity2 += before[j + 3] * before[j + 3];
        rate2 += before[j + 15] * before[j + 15];
      }
      far_from_target |= distance2 > 4.0F;
      const float expected = std::exp(-2 * distance2 - .1F * velocity2 -
                                      .05F * rate2 - .5F * tilt * tilt);
      require(std::abs(rewards[i] - expected) < 2e-5F,
              "reward must use the target active during the transition");
      require((finished[i] != 0) == (elapsed[i] == duration[i]),
              "command completion must follow its sampled duration");
      switched |= finished[i] != 0;
    }
    previous_targets = targets;
  }
  require(switched, "exercise a real target replacement");
  require(far_from_target,
          "tracking must allow safe flight beyond the old hover error limit");
  require(download<float>(env.value, 19, n * commands * 4, stream.value) ==
              plan,
          "command sequence must not depend on physical trajectory");
  check(triage_hover_reset(env.value, 17, stream.value));
  require(download<float>(env.value, 19, n * commands * 4, stream.value) ==
              plan,
          "seed reset must reproduce the complete command schedule");
  Environment settling(n, 2000, stream.value, true, 1);
  const auto settling_plan =
      download<float>(settling.value, 19, n * commands * 4, stream.value);
  for (std::size_t c = 0; c < n * commands; ++c)
    require(settling_plan[c * 4 + 3] == 500,
            "settling evaluation must allow exactly 500 controls per target");
}

// The timeout releases a broken synchronizing implementation instead of hanging
// the test process. Successful submit/poll must finish while the gate is
// closed.
struct StreamGate {
  std::mutex mutex;
  std::condition_variable condition;
  bool released = false, timed_out = false;
  cudaStream_t stream;
  explicit StreamGate(cudaStream_t stream) : stream(stream) {
    cuda_check(cudaLaunchHostFunc(
        stream,
        [](void *data) {
          auto &gate = *static_cast<StreamGate *>(data);
          std::unique_lock lock(gate.mutex);
          gate.timed_out = !gate.condition.wait_for(
              lock, std::chrono::seconds(5), [&] { return gate.released; });
        },
        this));
  }
  void release() {
    std::lock_guard lock(mutex);
    released = true;
    condition.notify_one();
  }
  ~StreamGate() {
    release();
    cudaStreamSynchronize(stream);
  }
};

void test_snapshot_ordering_and_backpressure() {
  constexpr std::size_t n = 4;
  Stream first, second;
  Actions actions(n);
  Environment env(n, 1, first.value, true);
  const std::uint32_t ids[] = {3, 1};
  check(triage_snapshot_configure(env.value, ids, 2, 2));
  const auto initial = download<float>(env.value, TRIAGE_HOVER_OBSERVATIONS,
                                       n * 22, first.value);
  const auto initial_targets =
      download<float>(env.value, TRIAGE_TRACKING_TARGETS, n * 3, first.value);
  triage_snapshot_vehicle rows[2]{};
  std::uint64_t step = 999;
  float gather_ms, copy_ms;
  // Load/instrument the snapshot kernel before testing steady-state readiness;
  // Compute Sanitizer may synchronize its first instrumented kernel launch.
  require(triage_snapshot_submit(env.value, 0, first.value) == 1,
          "snapshot warm-up must queue");
  cuda_check(cudaDeviceSynchronize());
  require(triage_snapshot_poll(env.value, rows, 2, &step, &gather_ms,
                               &copy_ms) == 1,
          "snapshot warm-up must complete");
  {
    StreamGate gate(first.value);
    require(triage_snapshot_submit(env.value, 0, first.value) == 1,
            "initial snapshot must queue without waiting");
    check(triage_hover_step(env.value, actions.value, second.value));
    require(triage_snapshot_submit(env.value, 1, second.value) == 1,
            "cross-stream post-reset snapshot must queue");
    require(triage_snapshot_submit(env.value, 2, first.value) == 0,
            "full snapshot ring must drop without overwriting");
    require(triage_snapshot_poll(env.value, rows, 2, &step, &gather_ms,
                                 &copy_ms) == 0,
            "snapshot must not expose an unfinished transfer");
    {
      std::lock_guard lock(gate.mutex);
      require(!gate.timed_out,
              "snapshot operations waited for blocked CUDA work");
    }
    gate.release();
  }
  const auto reset = download<float>(env.value, TRIAGE_HOVER_OBSERVATIONS,
                                     n * 22, second.value);
  const auto reset_targets =
      download<float>(env.value, TRIAGE_TRACKING_TARGETS, n * 3, second.value);
  // Mutate live state again before consuming either slot.
  check(triage_hover_step(env.value, actions.value, first.value));
  cuda_check(cudaDeviceSynchronize());
  for (unsigned frame = 0; frame < 2; ++frame) {
    require(triage_snapshot_poll(env.value, rows, 2, &step, &gather_ms,
                                 &copy_ms) == 1,
            "completed snapshots must remain independently readable");
    require(step == frame, "snapshot order must match submission order");
    const auto &observations = frame == 0 ? initial : reset;
    const auto &targets = frame == 0 ? initial_targets : reset_targets;
    for (unsigned j = 0; j < 2; ++j) {
      const auto id = ids[j];
      require(rows[j].environment_id == id && rows[j].episode_id == frame,
              "snapshot selection and post-reset episode must stay paired");
      for (unsigned axis = 0; axis < 3; ++axis) {
        require(rows[j].target_w[axis] == targets[3 * id + axis],
                "snapshot must retain its current commanded target");
        require(std::abs(rows[j].position_w[axis] -
                         (observations[22 * id + axis] +
                          targets[3 * id + axis])) < 1e-6f,
                "snapshot pose must precede subsequent live-state mutations");
      }
      const auto *q = rows[j].attitude_wb;
      const float rotation[] = {
          1 - 2 * (q[2] * q[2] + q[3] * q[3]), 2 * (q[1] * q[2] - q[0] * q[3]),
          2 * (q[1] * q[3] + q[0] * q[2]),     2 * (q[1] * q[2] + q[0] * q[3]),
          1 - 2 * (q[1] * q[1] + q[3] * q[3]), 2 * (q[2] * q[3] - q[0] * q[1]),
          2 * (q[1] * q[3] - q[0] * q[2]),     2 * (q[2] * q[3] + q[0] * q[1]),
          1 - 2 * (q[1] * q[1] + q[2] * q[2])};
      for (unsigned k = 0; k < 9; ++k)
        require(std::abs(rotation[k] - observations[22 * id + 6 + k]) < 1e-6f,
                "snapshot Hamilton quaternion must match physical attitude");
    }
  }
  require(triage_snapshot_submit(env.value, 2, first.value) == 1,
          "consumed snapshot slots must be reusable");
  check(triage_snapshot_disable(env.value));
}
} // namespace
int main() {
  try {
    test_timeout_and_reproducibility();
    test_failure_is_not_timeout();
    test_tracking_schedule_and_switches();
    test_snapshot_ordering_and_backpressure();
    std::cout
        << "Hover and tracking episode, command, safety, reward, and seed "
           "replay contracts passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Hover environment test failed: " << error.what() << '\n';
    return 1;
  }
}
