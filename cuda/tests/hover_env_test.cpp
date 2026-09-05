#include "hover_env.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
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
  Environment(std::size_t n, int limit, cudaStream_t stream) {
    value = triage_hover_create(0, n, 17, limit, stream);
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
} // namespace
int main() {
  try {
    test_timeout_and_reproducibility();
    test_failure_is_not_timeout();
    std::cout << "Hover timeout/failure, final observations, cross-stream "
                 "reset, and seed replay passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Hover environment test failed: " << error.what() << '\n';
    return 1;
  }
}
