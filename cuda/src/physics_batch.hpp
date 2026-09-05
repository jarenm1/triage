#pragma once

#include "physics_types.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace sim_cuda {

using DeviceState = MultirotorState<float>;
using DeviceActions = RotorActions<float>;
using DeviceVehicleParameters = VehicleParameters<float>;
using DeviceRotorSpeeds = FixedArray<float, kRotorCount>;
using DeviceRotors = FixedArray<RotorParameters<float>, kRotorCount>;

// Non-owning view of contiguous CUDA device elements, not host/managed memory.
// The caller guarantees the declared extent and keeps storage alive and inputs
// unchanged until queued use completes. Nonempty spans use the batch's device.
template <typename T> struct DeviceSpan {
  T *data = nullptr;
  std::size_t size = 0;
};

struct BatchConfig {
  int device;
  // Zero through UINT32_MAX environments.
  std::size_t environment_count;
  float physics_timestep_seconds;
  int substeps;
};

enum class ResetStatus : std::uint8_t {
  not_selected,
  applied,
  invalid_state,
  invalid_parameters,
};

enum class SelectionKind { all, indexed };
struct ExportSelection {
  SelectionKind kind = SelectionKind::all;
  // Used only for indexed selection. Order and duplicates are preserved.
  DeviceSpan<const std::uint32_t> indices;
};

enum class ExportStatus : std::uint8_t { exported, invalid_index };

// Each nonempty field requests a tightly packed output array. Its element count
// must equal the selection size (or batch size for all). Empty fields are
// omitted. Outputs must be disjoint from one another and from selection data.
// Full records are export formats, not aliases to internal storage.
struct ExportBuffers {
  DeviceSpan<DeviceState> states;
  DeviceSpan<DeviceVehicleParameters> parameters;
  DeviceSpan<Vec3<float>> position_w;
  DeviceSpan<Quaternion<float>> attitude_wb;
  DeviceSpan<Vec3<float>> linear_velocity_w;
  DeviceSpan<Vec3<float>> angular_velocity_b;
  DeviceSpan<DeviceRotorSpeeds> rotor_speed;
  DeviceSpan<float> mass;
  DeviceSpan<Vec3<float>> inertia_diagonal_b;
  DeviceSpan<Vec3<float>> gravity_w;
  DeviceSpan<DeviceRotors> rotors;
  // Required for indexed selection, optional for all. Invalid indices leave
  // their data outputs untouched and report invalid_index in this array.
  DeviceSpan<ExportStatus> status;
};

// Fixed-size, single-host-caller owner. Calls on different supplied streams are
// ordered by the batch; inputs must already be ready on the supplied stream.
// Normal step/reset/export do not allocate or synchronize the host. Creation,
// destruction and CUDA-error cleanup may wait. There is no public live alias.
class PhysicsBatch {
public:
  // Initial states/parameters must each contain environment_count elements on
  // config.device. Creation waits for validation and rejects any invalid row.
  // States require finite fields, unit attitude (norm tolerance 1e-5), and
  // nonnegative rotor speeds; actual speeds may exceed command limits.
  // Parameters require finite fields, positive mass/realizable diagonal
  // inertia, unit thrust directions, valid rotor bounds, and
  // timestep/time_constant < 2. Zero-sized batches are valid. Step/reset and
  // empty exports are no-ops after structural validation; nonempty indexed
  // exports report invalid indices. Each substep advances
  // physics_timestep_seconds, not dt/substeps.
  PhysicsBatch(const BatchConfig &config,
               DeviceSpan<const DeviceState> initial_states,
               DeviceSpan<const DeviceVehicleParameters> initial_parameters,
               cudaStream_t stream);
  ~PhysicsBatch() noexcept;

  PhysicsBatch(const PhysicsBatch &) = delete;
  PhysicsBatch &operator=(const PhysicsBatch &) = delete;
  PhysicsBatch(PhysicsBatch &&) = delete;
  PhysicsBatch &operator=(PhysicsBatch &&) = delete;

  const BatchConfig &config() const noexcept { return config_; }

  void step(DeviceSpan<const DeviceActions> actions, cudaStream_t stream);

  // Inputs and status have full batch extent. A nonzero mask byte selects an
  // environment. State and parameters commit together only if both validate.
  // Invalid/unselected rows remain unchanged; no episode/RNG bookkeeping
  // occurs. Invalid state takes precedence if both supplied records are
  // invalid. Status storage must not overlap inputs. Host structural errors
  // throw before launch; row failures are reported on the device without a host
  // readback.
  void
  apply_reset(DeviceSpan<const std::uint8_t> mask,
              DeviceSpan<const DeviceState> supplied_states,
              DeviceSpan<const DeviceVehicleParameters> supplied_parameters,
              DeviceSpan<ResetStatus> status_output, cudaStream_t stream);

  void export_state(const ExportSelection &selection,
                    const ExportBuffers &outputs, cudaStream_t stream);

private:
  BatchConfig config_;
  DeviceState *states_ = nullptr;
  DeviceVehicleParameters *parameters_ = nullptr;
  std::uint32_t *initial_error_ = nullptr;
  cudaEvent_t completion_ = nullptr;
  bool pending_ = false;

  void wait_for_previous(cudaStream_t stream);
  void record_completion(cudaStream_t stream);
  void release() noexcept;
};

} // namespace sim_cuda
