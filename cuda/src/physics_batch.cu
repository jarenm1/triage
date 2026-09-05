#include "physics_batch.hpp"
#include "physics_integration.hpp"
#include "physics_validation.hpp"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace sim_cuda {
namespace {

void check_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

class DeviceGuard {
public:
  explicit DeviceGuard(int device) {
    check_cuda(cudaGetDevice(&previous_), "cudaGetDevice");
    if (previous_ != device) {
      check_cuda(cudaSetDevice(device), "cudaSetDevice");
      changed_ = true;
    }
  }
  ~DeviceGuard() noexcept {
    if (changed_) {
      cudaSetDevice(previous_);
    }
  }
  DeviceGuard(const DeviceGuard &) = delete;
  DeviceGuard &operator=(const DeviceGuard &) = delete;

private:
  int previous_ = 0;
  bool changed_ = false;
};

void validate_stream(cudaStream_t stream, int device) {
  int stream_device = -1;
  check_cuda(cudaStreamGetDevice(stream, &stream_device),
             "cudaStreamGetDevice");
  if (stream_device != device) {
    throw std::invalid_argument("stream belongs to another CUDA device");
  }
}

struct AddressRange {
  std::uintptr_t begin = 0;
  std::uintptr_t end = 0;
};

template <typename T>
AddressRange validate_span(DeviceSpan<T> span, std::size_t count, int device) {
  if (span.size != count) {
    throw std::invalid_argument("device span has incorrect element count");
  }
  if (span.size == 0) {
    return {};
  }
  if (span.data == nullptr ||
      span.size > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::invalid_argument("invalid device span pointer or extent");
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(span.data);
  const auto bytes = span.size * sizeof(T);
  if (begin % alignof(T) != 0 ||
      begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    throw std::invalid_argument("misaligned or overflowing device span");
  }
  cudaPointerAttributes attributes{};
  check_cuda(cudaPointerGetAttributes(&attributes, span.data),
             "cudaPointerGetAttributes");
  if (attributes.type != cudaMemoryTypeDevice || attributes.device != device) {
    throw std::invalid_argument("span must use configured-device CUDA storage");
  }
  return {begin, begin + bytes};
}

bool overlaps(AddressRange left, AddressRange right) {
  return left.begin < right.end && right.begin < left.end;
}

class OutputRanges {
public:
  explicit OutputRanges(AddressRange selector) : selector_(selector) {}

  template <typename T>
  void add(DeviceSpan<T> span, std::size_t count, int device,
           bool required = false) {
    if (span.size == 0 && !required) {
      return;
    }
    const auto range = validate_span(span, count, device);
    if (overlaps(range, selector_)) {
      throw std::invalid_argument("export output overlaps selection indices");
    }
    for (std::size_t index = 0; index < size_; ++index) {
      if (overlaps(range, ranges_[index])) {
        throw std::invalid_argument("export outputs overlap");
      }
    }
    ranges_[size_++] = range;
  }

private:
  AddressRange selector_;
  AddressRange ranges_[12]{};
  std::size_t size_ = 0;
};

constexpr unsigned int threads_per_block = 256;
unsigned int block_count(std::size_t count) {
  return static_cast<unsigned int>((count + threads_per_block - 1) /
                                   threads_per_block);
}

__global__ void
initialize_kernel(DeviceState *states, DeviceVehicleParameters *parameters,
                  const DeviceState *initial_states,
                  const DeviceVehicleParameters *initial_parameters,
                  std::size_t count, float timestep, int substeps,
                  std::uint32_t *error) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const auto state = initial_states[index];
  const auto params = initial_parameters[index];
  if (!model::valid_state(state) ||
      !model::valid_parameters(params, timestep, substeps)) {
    atomicExch(error, 1U);
    return;
  }
  states[index] = state;
  parameters[index] = params;
}

__global__ void step_kernel(DeviceState *states,
                            const DeviceVehicleParameters *parameters,
                            const DeviceActions *actions, std::size_t count,
                            float timestep, int substeps) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  auto state = states[index];
  const auto params = parameters[index];
  const auto action = actions[index];
  for (int substep = 0; substep < substeps; ++substep) {
    state = model::step_midpoint(state, action, params, timestep);
  }
  states[index] = state;
}

__global__ void reset_kernel(DeviceState *states,
                             DeviceVehicleParameters *parameters,
                             const std::uint8_t *mask,
                             const DeviceState *supplied_states,
                             const DeviceVehicleParameters *supplied_parameters,
                             ResetStatus *status, std::size_t count,
                             float timestep, int substeps) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  if (mask[index] == 0) {
    status[index] = ResetStatus::not_selected;
    return;
  }
  const auto state = supplied_states[index];
  if (!model::valid_state(state)) {
    status[index] = ResetStatus::invalid_state;
    return;
  }
  const auto params = supplied_parameters[index];
  if (!model::valid_parameters(params, timestep, substeps)) {
    status[index] = ResetStatus::invalid_parameters;
    return;
  }
  states[index] = state;
  parameters[index] = params;
  status[index] = ResetStatus::applied;
}

__global__ void export_kernel(const DeviceState *states,
                              const DeviceVehicleParameters *parameters,
                              std::size_t environment_count,
                              ExportSelection selection, ExportBuffers outputs,
                              std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t destination =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       destination < count; destination += stride) {
    const std::size_t index = selection.kind == SelectionKind::all
                                  ? destination
                                  : selection.indices.data[destination];
    if (index >= environment_count) {
      outputs.status.data[destination] = ExportStatus::invalid_index;
      continue;
    }
    const auto &state = states[index];
    const auto &params = parameters[index];
    if (outputs.states.size)
      outputs.states.data[destination] = state;
    if (outputs.parameters.size)
      outputs.parameters.data[destination] = params;
    if (outputs.position_w.size)
      outputs.position_w.data[destination] = state.position_w;
    if (outputs.attitude_wb.size)
      outputs.attitude_wb.data[destination] = state.attitude_wb;
    if (outputs.linear_velocity_w.size)
      outputs.linear_velocity_w.data[destination] = state.linear_velocity_w;
    if (outputs.angular_velocity_b.size)
      outputs.angular_velocity_b.data[destination] = state.angular_velocity_b;
    if (outputs.rotor_speed.size)
      outputs.rotor_speed.data[destination] = state.rotor_speed;
    if (outputs.mass.size)
      outputs.mass.data[destination] = params.mass;
    if (outputs.inertia_diagonal_b.size)
      outputs.inertia_diagonal_b.data[destination] = params.inertia_diagonal_b;
    if (outputs.gravity_w.size)
      outputs.gravity_w.data[destination] = params.gravity_w;
    if (outputs.rotors.size)
      outputs.rotors.data[destination] = params.rotors;
    if (outputs.status.size)
      outputs.status.data[destination] = ExportStatus::exported;
  }
}

} // namespace

PhysicsBatch::PhysicsBatch(
    const BatchConfig &config, DeviceSpan<const DeviceState> initial_states,
    DeviceSpan<const DeviceVehicleParameters> initial_parameters,
    cudaStream_t stream)
    : config_(config) {
  if (config_.device < 0 ||
      config_.environment_count > std::numeric_limits<std::uint32_t>::max() ||
      !model::finite_scalar(config_.physics_timestep_seconds) ||
      !(config_.physics_timestep_seconds > 0.0F) || config_.substeps <= 0) {
    throw std::invalid_argument("invalid physics batch configuration");
  }
  DeviceGuard guard(config_.device);
  validate_stream(stream, config_.device);
  validate_span(initial_states, config_.environment_count, config_.device);
  validate_span(initial_parameters, config_.environment_count, config_.device);
  try {
    check_cuda(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming),
               "cudaEventCreateWithFlags");
    if (config_.environment_count == 0) {
      return;
    }
    check_cuda(
        cudaMalloc(&states_, config_.environment_count * sizeof(*states_)),
        "cudaMalloc states");
    check_cuda(cudaMalloc(&parameters_,
                          config_.environment_count * sizeof(*parameters_)),
               "cudaMalloc parameters");
    check_cuda(cudaMalloc(&initial_error_, sizeof(*initial_error_)),
               "cudaMalloc initial error");
    check_cuda(
        cudaMemsetAsync(initial_error_, 0, sizeof(*initial_error_), stream),
        "cudaMemsetAsync initial error");
    initialize_kernel<<<block_count(config_.environment_count),
                        threads_per_block, 0, stream>>>(
        states_, parameters_, initial_states.data, initial_parameters.data,
        config_.environment_count, config_.physics_timestep_seconds,
        config_.substeps, initial_error_);
    check_cuda(cudaGetLastError(), "initialize_kernel launch");
    std::uint32_t error = 0;
    check_cuda(cudaMemcpyAsync(&error, initial_error_, sizeof(error),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync initial error");
    record_completion(stream);
    check_cuda(cudaEventSynchronize(completion_), "initial validation wait");
    if (error != 0) {
      throw std::invalid_argument(
          "invalid initial state or vehicle parameters");
    }
    check_cuda(cudaFree(initial_error_), "cudaFree initial error");
    initial_error_ = nullptr;
  } catch (...) {
    // A launch or event-record failure can leave work beyond the last event.
    cudaStreamSynchronize(stream);
    release();
    throw;
  }
}

PhysicsBatch::~PhysicsBatch() noexcept { release(); }

void PhysicsBatch::wait_for_previous(cudaStream_t stream) {
  if (pending_) {
    check_cuda(cudaStreamWaitEvent(stream, completion_, 0),
               "cudaStreamWaitEvent");
  }
}

void PhysicsBatch::record_completion(cudaStream_t stream) {
  check_cuda(cudaEventRecord(completion_, stream), "cudaEventRecord");
  pending_ = true;
}

void PhysicsBatch::release() noexcept {
  int previous = -1;
  if (cudaGetDevice(&previous) != cudaSuccess) {
    return;
  }
  if (previous != config_.device &&
      cudaSetDevice(config_.device) != cudaSuccess) {
    return;
  }
  if (pending_) {
    cudaEventSynchronize(completion_);
  }
  if (initial_error_)
    cudaFree(initial_error_);
  if (parameters_)
    cudaFree(parameters_);
  if (states_)
    cudaFree(states_);
  if (completion_)
    cudaEventDestroy(completion_);
  initial_error_ = nullptr;
  parameters_ = nullptr;
  states_ = nullptr;
  completion_ = nullptr;
  pending_ = false;
  if (previous != config_.device) {
    cudaSetDevice(previous);
  }
}

void PhysicsBatch::step(DeviceSpan<const DeviceActions> actions,
                        cudaStream_t stream) {
  DeviceGuard guard(config_.device);
  validate_stream(stream, config_.device);
  validate_span(actions, config_.environment_count, config_.device);
  if (config_.environment_count == 0) {
    return;
  }
  try {
    wait_for_previous(stream);
    step_kernel<<<block_count(config_.environment_count), threads_per_block, 0,
                  stream>>>(states_, parameters_, actions.data,
                            config_.environment_count,
                            config_.physics_timestep_seconds, config_.substeps);
    check_cuda(cudaGetLastError(), "step_kernel launch");
    record_completion(stream);
  } catch (...) {
    cudaStreamSynchronize(stream);
    throw;
  }
}

void PhysicsBatch::apply_reset(
    DeviceSpan<const std::uint8_t> mask,
    DeviceSpan<const DeviceState> supplied_states,
    DeviceSpan<const DeviceVehicleParameters> supplied_parameters,
    DeviceSpan<ResetStatus> status_output, cudaStream_t stream) {
  DeviceGuard guard(config_.device);
  validate_stream(stream, config_.device);
  const auto count = config_.environment_count;
  const auto mask_range = validate_span(mask, count, config_.device);
  const auto state_range =
      validate_span(supplied_states, count, config_.device);
  const auto parameter_range =
      validate_span(supplied_parameters, count, config_.device);
  const auto status_range = validate_span(status_output, count, config_.device);
  if (overlaps(status_range, mask_range) ||
      overlaps(status_range, state_range) ||
      overlaps(status_range, parameter_range)) {
    throw std::invalid_argument("reset status overlaps input storage");
  }
  if (count == 0) {
    return;
  }
  try {
    wait_for_previous(stream);
    reset_kernel<<<block_count(count), threads_per_block, 0, stream>>>(
        states_, parameters_, mask.data, supplied_states.data,
        supplied_parameters.data, status_output.data, count,
        config_.physics_timestep_seconds, config_.substeps);
    check_cuda(cudaGetLastError(), "reset_kernel launch");
    record_completion(stream);
  } catch (...) {
    cudaStreamSynchronize(stream);
    throw;
  }
}

void PhysicsBatch::export_state(const ExportSelection &selection,
                                const ExportBuffers &outputs,
                                cudaStream_t stream) {
  DeviceGuard guard(config_.device);
  validate_stream(stream, config_.device);
  if (selection.kind != SelectionKind::all &&
      selection.kind != SelectionKind::indexed) {
    throw std::invalid_argument("unknown export selection kind");
  }
  const bool indexed = selection.kind == SelectionKind::indexed;
  const auto count =
      indexed ? selection.indices.size : config_.environment_count;
  const auto selector =
      validate_span(selection.indices, indexed ? count : 0, config_.device);
  OutputRanges ranges(selector);
  ranges.add(outputs.states, count, config_.device);
  ranges.add(outputs.parameters, count, config_.device);
  ranges.add(outputs.position_w, count, config_.device);
  ranges.add(outputs.attitude_wb, count, config_.device);
  ranges.add(outputs.linear_velocity_w, count, config_.device);
  ranges.add(outputs.angular_velocity_b, count, config_.device);
  ranges.add(outputs.rotor_speed, count, config_.device);
  ranges.add(outputs.mass, count, config_.device);
  ranges.add(outputs.inertia_diagonal_b, count, config_.device);
  ranges.add(outputs.gravity_w, count, config_.device);
  ranges.add(outputs.rotors, count, config_.device);
  ranges.add(outputs.status, count, config_.device, indexed);
  if (count == 0) {
    return;
  }
  try {
    wait_for_previous(stream);
    const auto launch_count = count < std::numeric_limits<std::uint32_t>::max()
                                  ? count
                                  : std::numeric_limits<std::uint32_t>::max();
    export_kernel<<<block_count(launch_count), threads_per_block, 0, stream>>>(
        states_, parameters_, config_.environment_count, selection, outputs,
        count);
    check_cuda(cudaGetLastError(), "export_kernel launch");
    record_completion(stream);
  } catch (...) {
    cudaStreamSynchronize(stream);
    throw;
  }
}

} // namespace sim_cuda
