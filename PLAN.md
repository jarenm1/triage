# GPU-Native Multirotor RL Environment and Synthetic Data Simulator

## Architecture, Theoretical Contracts, and Evidence-Gated Roadmap

> **Status:** Living engineering plan. The physical conventions, observable semantics, and measurement definitions below are architectural contracts. Kernel decomposition, storage packing, integrator, ML library, model family, and file formats remain implementation choices until measured.

---

## 1. Mission and Scope

Build a simulation and synthetic-data system for autonomous multirotors that can:

1. **Run state-based RL at high throughput.** Keep simulation state, policy inputs, actions, rewards, and reset work on one GPU during the training critical path.
2. **Render calibrated synthetic sensors.** Produce RGB, metric camera depth, and stable object/instance identifiers through `sim-graphics` for selected environments at an independently configured sensor rate.
3. **Measure sim-to-real transfer.** Compare training and evaluation regimes on labeled, held-out real data. The system can quantify and improve transfer; it cannot guarantee zero-shot transfer.
4. **Generate and search structured scenarios.** Represent scenes and disturbances as validated, replayable data rather than generating raw pixels.
5. **Support learned dynamics only when justified by evidence.** Add bounded force/torque residuals or latent predictive models only if they improve held-out prediction or policy outcomes over analytical baselines.

### Initial product boundary

The first physics implementation is a **single free-flying multirotor rigid body with rotor dynamics and terminal ground contact**. Contact-rich cars, rovers, articulated bodies, wheel/tire models, and general rigid-body contact require different state and solver contracts; they are future vehicle-family adapters, not swappable kernels inside the first multirotor module.

### Explicit non-goals

- Simulating every aerodynamic effect from first principles.
- Rendering a camera for every headless RL environment on every control step.
- Treating synthetic labels as evidence of real-world accuracy.
- Promising portable zero-copy sharing between CUDA and WebGPU resources.
- Committing to RK4, PPO, Dreamer, RSSM, Mamba, ONNX, or a particular dataset container before comparative measurements.
- Making CUDA physics cross-platform. Initial physics acceleration targets NVIDIA GPUs; rendering and recorded replay remain cross-platform goals.

---

## 2. Repository Starting Point

The repository already contains useful rendering infrastructure:

- `sim-graphics`: Rust/`wgpu` renderer with instanced primitives, display views, sensor color, `R32Float` depth, `R32Uint` object IDs, pooled targets, and asynchronous staging-buffer readback.
- `sim-graphics-winit`: native window integration.
- `window-demo` and `render-smoke`: interactive and offscreen examples, including WebAssembly/WebGPU-oriented workspace support.

The current renderer consumes CPU-authored `Frame` vectors and uploads instance data with `wgpu::Queue::write_buffer`. Each sensor view currently owns separate 2D targets. Therefore, GPU-authored transforms, true batched camera rendering, and CUDA/renderer interoperation are roadmap work—not existing capabilities.

---

## 3. Architecture and Seams

```mermaid
flowchart TB
    subgraph ML[Python / ML orchestration]
        POLICY[PyTorch policy and trainer]
        GYM[Gymnasium adapter]
        EVAL[Dataset and transfer evaluation]
        SEARCH[Scenario search and curricula]
    end

    subgraph SIM[Simulation module]
        CONTRACT[MultirotorBatch interface]
        CUDA[CUDA implementation]
        REF[Reference implementation]
    end

    subgraph BRIDGE[RenderSnapshot seam]
        STAGED[Portable staged-copy adapter]
        NATIVE[Optional native CUDA/Vulkan adapter]
    end

    subgraph GFX[Rust / wgpu rendering]
        RENDER[sim-graphics]
        VIEWER[sim-graphics-winit viewer]
        SENSOR[Offscreen sensor outputs]
        READBACK[Asynchronous export ring]
    end

    POLICY <-->|device tensors| CONTRACT
    GYM --> CONTRACT
    CONTRACT --> CUDA
    CONTRACT --> REF
    CUDA -->|selected poses and cameras| STAGED
    CUDA -. feasibility gated .-> NATIVE
    STAGED --> RENDER
    NATIVE -. native only .-> RENDER
    RENDER --> VIEWER
    RENDER --> SENSOR
    SENSOR --> READBACK
    READBACK --> EVAL
    SEARCH --> GYM
```

### 3.1 Simulation seam

The simulation is a deep module. Callers should need only a validated configuration and a small batched interface:

```text
reset(mask?, seed?) -> ResetResult
step(actions)        -> StepResult
snapshot(selection)  -> RenderSnapshot
```

The interface includes tensor shapes and devices, frame conventions, reset behavior, error modes, determinism guarantees, and stream-ordering requirements. Internal kernel count, memory packing, and fusion are implementation details.

The initial ML adapter is PyTorch-specific. A later JAX or framework-neutral adapter may use DLPack or another FFI mechanism, but it must separately define ownership, lifetime, device, and producer/consumer stream synchronization. A tensor pointer alone is not a complete interoperation contract.

### 3.2 CUDA-to-`wgpu` seam

There are two render adapters behind the same logical `RenderSnapshot` interface:

1. **Portable staged-copy adapter — required.** Gather only selected environments, copy asynchronously through bounded staging memory, tolerate one or more frames of latency, and drop visualization frames rather than stall training.
2. **Native external-memory adapter — optional and feasibility-gated.** On a supported NVIDIA/Vulkan platform, a proof of concept may allocate exportable Vulkan memory, import it into CUDA, verify that CUDA and Vulkan select the same physical device, and synchronize ownership with external semaphores. Integrating such resources through `wgpu` requires unsafe, backend-specific HAL access and is not a portable WebGPU feature.

Failure of the native adapter must not block the simulator, viewer, offline renderer, or web replay. Wasm replay consumes recorded data; it does not share CUDA memory.

### 3.3 Scene seam

A versioned scene description owns geometry references, transforms, semantic identities, lights, weather, and scenario parameters. Physics consumes only collision/query representations it explicitly supports. Rendering consumes visual assets. A mesh present in `wgpu` memory does not automatically exist in a CUDA ray-query structure.

---

## 4. Fixed Physical Conventions

These conventions are fixed across reference physics, CUDA physics, logs, tasks, and exported metadata. Renderer-specific coordinates are converted at the `RenderSnapshot` seam.

### 4.1 Frames, units, and attitude

- All physical quantities use SI units: metres, seconds, kilograms, radians, newtons, and newton-metres.
- World frame $W$: right-handed East-North-Up (ENU), with gravity
  $$
  \mathbf g_W = [0, 0, -g]^T, \qquad g \approx 9.80665\ \mathrm{m/s^2}.
  $$
- Body frame $B$: right-handed Forward-Left-Up (FLU), fixed to the vehicle centre of mass.
- Camera optical frame $C$: right-handed $x$ right, $y$ down, $z$ forward. Every camera has calibrated intrinsics and an explicit rigid transform between $B$ and $C$.
- $\mathbf q_{WB} = [w,x,y,z]$ is a scalar-first Hamilton unit quaternion whose active rotation $R_{WB}$ maps body-frame vectors into the world frame.
- $\mathbf v_W$ is world-frame linear velocity. $\boldsymbol\omega_B$ is body-frame angular velocity.
- Exported timestamps use simulation time. Physics advances on a fixed timestep and control uses an integer number of physics substeps. Other simulated sensors use deterministic, timestamped schedules; display and export consumers may run asynchronously.

The renderer currently uses a right-handed, Y-up graphics convention. The render adapter performs one documented basis change from ENU/FLU into renderer coordinates; physics never adopts graphics coordinates implicitly.

### 4.2 Multirotor equations of motion

For position $\mathbf p_W$, velocity $\mathbf v_W$, body angular velocity $\boldsymbol\omega_B$, body inertia $I_B$, total body force $\mathbf F_B$, world-frame external force $\mathbf F_W^{ext}$, and total body torque $\boldsymbol\tau_B$:

$$
\dot{\mathbf p}_W = \mathbf v_W
$$

$$
m\dot{\mathbf v}_W = m\mathbf g_W + R_{WB}\mathbf F_B + \mathbf F_W^{ext}
$$

$$
I_B\dot{\boldsymbol\omega}_B = \boldsymbol\tau_B - \boldsymbol\omega_B \times (I_B\boldsymbol\omega_B)
$$

$$
\dot{\mathbf q}_{WB} = \frac{1}{2}\mathbf q_{WB}\otimes[0,\boldsymbol\omega_B].
$$

Numerical integration must maintain a normalized attitude representation. Quaternion renormalization, exponential-map updates, and integrator selection are implementation choices subject to convergence and stability tests.

### 4.3 Rotor and actuator model

The initial model has $M=4$ rotors in a validated Quad-X configuration, while the equations remain defined for fixed $M$:

$$
\dot\Omega_i = \frac{\Omega_{i,cmd}-\Omega_i}{\tau_i},
\qquad
\mathbf F_{i,B} = k_{f,i}\Omega_i^2\mathbf d_{i,B},
$$

$$
\boldsymbol\tau_{i,B}
= \mathbf r_{i,B}\times\mathbf F_{i,B}
+ s_i k_{m,i}\Omega_i^2\mathbf d_{i,B},
$$

where $\mathbf r_{i,B}$ is rotor position, $\mathbf d_{i,B}$ is its thrust direction, and $s_i\in\{-1,+1\}$ is the signed reaction-torque direction on the body. Rotor ordering, position, direction, physical spin, derived torque sign, limits, and coefficients are named configuration—not undocumented tensor columns.

A normalized action is converted to commanded rotor speed through a monotonic, configurable actuator map. The first implementation may use an affine map. Saturation is part of the model and occurs before integration.

This is a calibrated quadratic rotor model, not blade-element momentum theory. Higher-order inflow, ground effect, battery sag, downwash, and wake interactions are later analytical or learned corrections.

### 4.4 Aerodynamics and environment

The baseline supports wind-relative empirical drag. With $\mathbf v_{rel,W}=\mathbf v_W-\mathbf v_{wind,W}$, drag is evaluated in a documented frame using validated linear and/or quadratic coefficients. The exact parameterization is replaceable; its units, sign, and frame are not.

Phase 1 ground interaction is a termination condition against an analytic plane or height function. Bounce, friction, resting contact, wheels, and general collision response require a contact solver and are outside the initial module.

### 4.5 Time integration

The physical model is fixed; the integrator is not. Phase 0 compares at least a simple baseline and a higher-order method—for example semi-implicit Euler, midpoint/RK2, and RK4 or a Lie-group attitude update—at equal control quality and stability.

The selected method must:

- use a fixed, explicit physics timestep for reproducibility;
- integrate coupled rotor and rigid-body state consistently;
- support an integer number of physics substeps per control step;
- demonstrate the expected convergence trend on smooth trajectories;
- remain stable over the declared parameter and action envelope; and
- define behavior at discontinuities such as saturation, termination, and reset.

RK4 is therefore a candidate, not an architectural requirement.

#### Current integration selection (2026-09-04)

Production CPU and CUDA stepping use coupled explicit midpoint/RK2, including motor speeds at both stages and normalized stage/output attitude. The discarded semi-implicit implementation and its selection experiment are preserved in JJ revision `vpommxpy` (`319431e2`), not in current source. Each supplied physics timestep must be strictly less than twice **every** rotor time constant; validation rejects the non-decaying motor limit and larger steps without silently subdividing time. This is a motor stability condition, not a general rigid-body stability or accuracy guarantee.

The original selection experiment swept 1–2,048 physics substeps per 10 ms control interval against double-precision RK4 traces. Its measurements below are historical evidence for the selection, not a retained alternative implementation or a baseline directly comparable to the current production-path benchmark.

On the RTX 2070 SUPER, a 10,000-environment **homogeneous replicated** batch over 100 control calls passed all four scenarios—hover, motor command reversals, coupled attitude motion, and unequal-inertia torque-free rotation—with midpoint at **16 substeps (0.625 ms)**. Maximum errors across these traces were approximately 0.107 mm position, 0.192 mm/s velocity, 0.000146 rad/s angular rate, 0.0000116 rad attitude, and 0.0422 rad/s rotor speed. The provisional gates are 1 mm, 1 mm/s, 0.001 rad/s, 0.001 rad, and 0.1 rad/s respectively. No tested baseline substep count passed every gate across all four scenarios; fine-step float32 error is not monotonically decreasing.

That experiment's Release-build median device times for the selected configuration were 3.2–6.9 ms per 100-control-call trajectory, depending on scenario, with uncontrolled clocks/power. These are physics-only measurements, not full-environment or training throughput, and do not establish a matched-accuracy speedup ratio when no baseline configuration passes. The retained scenario/parameter envelope is defined in `cuda/benchmarks/physics_cases.hpp`; 0.625 ms is a measured starting point, not a universal timestep default or proof of the broader flight envelope.

#### Current native physics batch

`cuda/src/physics_batch.hpp` defines `PhysicsBatch`, which owns per-environment float32 physical state and vehicle parameters on a fixed CUDA device. Environment count, physics timestep, and substep count are fixed at construction. Initial state and parameters are explicit device arrays; construction waits for validation and rejects the entire creation if any row is invalid. The old raw-buffer launcher is removed.

- `step` borrows full-batch device actions and advances owned state in place with the selected midpoint method.
- `apply_reset` accepts full-batch device mask/state/parameter arrays and writes per-environment statuses. Each selected pair commits together only if both records validate; invalid and unselected rows retain their previous state and parameters. Invalid state takes precedence over invalid parameters. Actual rotor speeds need not lie within command limits.
- `export_state` copies requested typed fields or full records into caller-owned device arrays. Selection is all environments or an ordered device index array, including duplicates. Indexed exports require statuses; invalid IDs leave data outputs untouched. Even an empty batch reports invalid IDs for a nonempty indexed selection.

One serialized host caller owns each batch's operation sequence. A batch-owned event orders every step, replacement, and export across supplied streams; an export therefore observes its position in that sequence, not later mutations. Callers must make borrowed inputs ready on the supplied stream, preserve their contents until consumed, and keep all borrowed allocations alive until queued use completes. Nonempty spans require storage on the configured CUDA device, not host or managed memory; declared allocation extents remain the caller's responsibility. Export outputs cannot overlap each other or their selector, and reset status cannot overlap reset inputs.

Normal operations allocate no storage and do not synchronize the host. Creation, destruction, and CUDA-error cleanup may wait. This batch does not own curriculum, randomization profiles, reset generation, episode counters, RNG state, observations/rewards, or checkpoint semantics; higher layers supply concrete reset data, including data generated on the GPU. It adds no new physics or numerical-failure policy for stepping.

Implementation evidence in JJ change `mkuzstvn`: CUDA-build and CPU-only correctness suites passed, as did two complete production benchmark runs and both motor-refinement levels. Compute Sanitizer reported zero errors for the batch tests and the benchmark at base count 257 (larger case 2,570). A throwaway instrumented smoke submitted 30 step/reset/export operations while a stream was deliberately blocked, checked resulting physical snapshots, and counted zero C++ allocations, CUDA allocation/free calls, and explicit host waits/synchronous copies during those operations. This checks the exercised steady-state path, not opaque driver-internal memory management.

#### Production regression benchmark

`sim_cuda_physics_benchmark` calls the public `PhysicsBatch::step` interface; it has no private stepping kernel or competing integrator. The default workload set times the four scenarios at 10,000 environments plus coupled attitude at 100,000 environments, all at 16 substeps per 10 ms control call. `--environments N` changes the base count; the larger case remains `10*N`. The double-precision RK4 oracle is checked by refinement, and accuracy uses indexed public exports of the first/middle/last environment after each control call, checking every export status. Additional motor-transient refinement at 32 and 64 substeps uses separate fixed-timing batches and is accuracy-only, with float32 error floors.

```sh
nu cuda/build.nu --release
cuda/build/sim_cuda_physics_benchmark --label REVISION --output cuda/build/physics-baseline.tsv
# After rebuilding the revision being compared, on the same machine:
cuda/build/sim_cuda_physics_benchmark --label NEW_REVISION --baseline cuda/build/physics-baseline.tsv --output cuda/build/physics-current.tsv
```

Use actual revision identifiers for the labels. Saved reports include hardware/compiler/configuration metadata and fingerprints of the scenario parameters, initial states, and action schedules. Comparison requires matching metadata and complete workload/count/substep sets; incompatible, malformed, or incomplete reports are rejected. A workload that fails accuracy in either run is ineligible for a timing percentage. Correctness failures return nonzero, but timing changes are informational: report event/wall median deltas and min/max spread, then repeat suspicious measurements rather than applying an arbitrary CI slowdown threshold. Sample ranges are not confidence intervals, and clocks/system load remain uncontrolled.

Both timers cover the public step sequence, including host span checks, stream-ordering events, and launch overhead; wall time additionally includes waiting for completion. Creation, checked masked state/parameter reset, allocation, preuploaded action schedules, exports/readback, and logging are outside the timed interval. The batch is reused across scenarios and reset before each timed trial. Its owned in-place state and full per-environment parameters/actions make old raw-launcher and selection-experiment timings incompatible baselines; reports identify the suite as `production-physics-batch-v2`. Memory reporting gives a workspace device-payload lower bound including owned state/parameters, separately reports host payload and process RSS, and excludes CUDA resources rather than claiming whole-GPU peak memory.

---

## 5. Logical State and Tensor Contracts

The following is the logical schema. The backend may use structure-of-arrays, array-of-structures, fused workspaces, or opaque packed tensors internally. Public fields must not depend on magic column offsets.

| Logical value | Shape | Type | Meaning |
| :--- | :--- | :--- | :--- |
| `position_w` | `[N, 3]` | `float32` | $\mathbf p_W$ in metres |
| `attitude_wb` | `[N, 4]` | `float32` | Unit $\mathbf q_{WB}=[w,x,y,z]$ |
| `linear_velocity_w` | `[N, 3]` | `float32` | $\mathbf v_W$ in m/s |
| `angular_velocity_b` | `[N, 3]` | `float32` | $\boldsymbol\omega_B$ in rad/s |
| `rotor_speed` | `[N, M]` | `float32` | Actual $\Omega_i$ in rad/s |
| `actions` | `[N, M]` | `float32` | Normalized actuator commands in `[0,1]` |
| `wind_velocity_w` | `[N, 3]` | `float32` | Local ambient wind in m/s |
| `sensor_bias` | task-defined | `float32` | Persistent accelerometer/gyro bias state |
| `rng_counter` | implementation-defined | integer | Counter-based stochastic state |
| `elapsed_steps` | `[N]` | integer | Control steps in the current episode |
| `episode_id` | `[N]` | integer | Monotonic episode identity per environment |
| `observations` | `[N, ObsDim]` or dict | `float32` | Task-defined policy observation |
| `rewards` | `[N]` | `float32` | Task reward for the transition |
| `terminated` | `[N]` | `bool` | MDP terminal state, such as crash or task success |
| `truncated` | `[N]` | `bool` | External cutoff, such as a time limit |
| `termination_code` | `[N]` | integer enum | Stable cause for diagnostics |

Vehicle parameters are a validated named bundle: mass, a positive-definite inertia tensor, rotor geometry, physical spin and reaction-torque signs, thrust/torque coefficients, actuator limits and time constants, and aerodynamic coefficients. Per-environment packing is private to the backend so the parameter set can evolve without changing every caller.

### 5.1 Ownership and execution

- State is owned by the simulation module; callers receive documented tensor views or results, not unrestricted mutable aliases.
- Actions must be on the configured device with the declared shape and dtype. Invalid inputs fail before launching work where practical.
- The PyTorch adapter must respect the current CUDA stream or establish explicit producer/consumer events. It must not rely accidentally on the legacy default stream.
- Host synchronization is absent from the headless `step` critical path. Logging that requires scalar host values is rate-limited and explicit.
- Temporary storage is allocated during creation or capacity growth, not during each steady-state step.

### 5.2 Reset and episode semantics

The engine distinguishes `terminated` from `truncated`; there is no ambiguous `done` tensor in the core interface.

Autoreset is a configured semantic mode compatible with the Python adapter:

- **Disabled:** terminal observation is returned and the environment remains ended until reset.
- **Next-step:** terminal observation is returned; reset occurs before consuming that environment's next action.
- **Same-step:** returned observation is the reset observation, while `final_observation`, final episode statistics, and masks preserve the transition that ended.

A fused reward/termination/reset kernel is allowed only if it preserves those observable semantics. Timeout is truncation by default. A crash or arena failure is termination unless a task specification says otherwise.

### 5.3 Reproducibility

Randomization and sensor noise use a counter-based scheme derived from stable identities such as `(base_seed, environment_id, episode_id, stream_id, sample_index)`. Reset ordering or unrelated environments must not silently perturb another environment's random stream.

Exact bitwise equality is required only on the same supported software/hardware path when declared by that backend. Cross-backend verification uses numerical tolerances and distributional tests because floating-point reduction and transcendental implementations may differ.

---

## 6. Sensor Contracts

### 6.1 IMU

At a sensor located at the centre of mass, ideal gyroscope and accelerometer outputs are:

$$
\tilde{\boldsymbol\omega}_B
= \boldsymbol\omega_B + \mathbf b_g + \mathbf n_g,
$$

$$
\tilde{\mathbf f}_B
= R_{BW}(\mathbf a_W-\mathbf g_W) + \mathbf b_a + \mathbf n_a.
$$

The accelerometer measures **specific force**, not world acceleration with gravity added. Sensor-axis misalignment, scale error, saturation, quantization, latency, and an offset lever arm may be introduced later as explicit model terms.

Bias follows a documented stochastic process, such as a discrete random walk:

$$
\mathbf b_{k+1}=\mathbf b_k+\sigma_b\sqrt{\Delta t}\,\boldsymbol\xi_k.
$$

Noise parameters must state whether they are continuous-time densities or per-sample standard deviations so changing the sensor rate does not silently change the physical model.

### 6.2 Range sensing

The first range sensor intersects a configured ray with an analytic ground plane or height field and reports range along the ray. General mesh raycasting is a future geometry-query adapter with its own acceleration structure; it is not delegated implicitly to render meshes.

### 6.3 Visual sensors

Each visual sample records:

- simulation timestamp and camera exposure interval;
- image dimensions;
- camera intrinsics and distortion model;
- camera extrinsics;
- near/far planes and depth convention;
- scene, environment, episode, frame, and randomization identifiers; and
- renderer and asset version metadata.

Canonical outputs:

| Output | Contract |
| :--- | :--- |
| Color | Exported RGB is explicitly tagged with its transfer function; default dataset output is sRGB RGB8, with optional linear/HDR output for sensor modeling. |
| Depth | `float32` metres along optical $+z_C$ (camera z-depth, not Euclidean range). Background is represented by the far value and object ID `0`. |
| Object ID | `uint32`; `0` is background and every nonzero ID maps through per-frame metadata to a stable instance and semantic class. |

“Pixel-perfect” means aligned with the simulator's rendered visibility and label ontology. It does not imply perfect real-world labels. RGB realism additionally depends on assets, materials, lighting, camera response, exposure, motion blur, rolling shutter, distortion, noise, and compression.

### 6.4 Visual scaling rule

Physics count and visual-sensor count are independent:

- $N_{physics}$ may be $10{,}000+$.
- $N_{visual}\ll N_{physics}$ is selected explicitly.
- Visual sensors run at their own rate and may use stale-but-timestamped snapshots.
- Viewer and exporter backpressure drops or delays render work; it never blocks headless training by default.

Raw output bandwidth is unavoidable. At $256\times256$, a four-byte color attachment, `float32` depth, and `uint32` IDs require 12 bytes/pixel. Rendering all three for 10,000 cameras would produce roughly 7.3 GiB per sensor tick before internal attachments or encoding, so “render every environment simultaneously” is not a design target.

---

## 7. Scenario Generation and Sim-to-Real Evaluation

### 7.1 Structured scenarios

Scenarios are versioned, schema-validated data containing asset references, poses, trajectories, weather, lighting, sensor configuration, and randomization distributions. Every realized scenario has a seed and can be replayed without the generator.

Generation proceeds in increasing complexity:

1. deterministic fixtures;
2. parameterized procedural generation;
3. distribution sampling and mutation;
4. optimization or adversarial search over valid parameters; and
5. optional model- or prompt-assisted proposals that still pass schema and feasibility validation.

An LLM is therefore a possible proposal mechanism, not the world model or source of truth.

### 7.2 Dataset protocol

- Define a versioned label ontology before export.
- Split real data by flight, site, and collection session to prevent adjacent-frame leakage.
- Keep a final real test split untouched by scene tuning, model selection, and threshold selection.
- Record provenance and randomization parameters for every synthetic frame.
- Store images/masks in suitable media files or shards and metadata in a versioned manifest. COCO, YOLO, Parquet, or another container may be provided by adapters; no single format must carry every payload.
- Validate exported geometry with simple scenes whose projected boxes, depth, occlusion, and IDs are analytically known.

### 7.3 Transfer evaluation

The evaluation harness measures downstream performance under explicit regimes:

- pretrained model without simulator-specific training;
- synthetic-only training;
- real-only training with a declared label budget;
- synthetic pretraining followed by real fine-tuning; and
- mixed real/synthetic training.

Primary evidence is performance on the untouched real test set using task-appropriate metrics such as class-wise mIoU, boundary F1, mask AP, calibration, and failure slices. A “sim-to-real gap” always names the two regimes and datasets being subtracted; it is not computed by comparing unpaired pixels.

SAM, YOLO segmentation models, DINO-family encoders, or later models are versioned evaluation adapters, not architectural dependencies. Prompting and class mapping are part of each adapter's protocol. Representation distance may diagnose domain shift but is not a substitute for downstream real-data performance.

Scene parameters may be tuned on training/tuning splits. The final test split is evaluated only at release gates. The project claims measured transfer and confidence intervals—not parity or guaranteed zero-shot behavior.

---

## 8. Optional Learned Models

### 8.1 Aerodynamic residual

A learned residual may predict bounded body-frame force and torque corrections:

$$
(\Delta\mathbf F_B,\Delta\boldsymbol\tau_B)
= f_\theta(\text{state},\text{action},\text{context}).
$$

It is admitted only after:

- synchronized real or higher-fidelity force/trajectory data exists;
- train/tuning/test trajectories are separated by flight regime;
- it improves held-out one-step and closed-loop rollout error over the analytical model;
- outputs are bounded or gated outside supported data; and
- batch latency and launch overhead are measured inside the actual simulation loop.

The inference implementation—generated CUDA, a fused framework operation, TorchScript, ONNX, or another path—is selected from measurements. A nominal sub-millisecond claim without batch size and synchronization scope is not an acceptance criterion.

### 8.2 Latent predictive model

A latent dynamics model is a separate research module, not the scenario generator. It is justified only if it improves sample efficiency, planning performance, or total training cost after accounting for model training and exploitation of model error. RSSM, transformer, state-space, or other architectures compete under the same held-out rollout and policy-evaluation protocol.

Latent “steps/sec” must be reported together with batch size, horizon, model size, device, precision, and whether decoding or policy inference is included.

### 8.3 Failure mining

Failure mining searches a bounded, valid scenario parameter space against a frozen policy checkpoint and explicit failure objectives. Results retain seeds and full scenario records. Mined examples enter a curriculum only after deduplication and evaluation on a separate scenario set, preventing the curriculum from degenerating into memorized adversarial cases.

---

## 9. Performance and Measurement Contract

“Steps/sec” is not used without qualification. For $K$ batched control calls, $N$ environments, and elapsed wall time $T$ after warm-up:

$$
\text{vector steps/s}=\frac{K}{T},
\qquad
\text{environment transitions/s}=\frac{KN}{T}.
$$

With $S$ physics substeps per control step:

$$
\text{physics state updates/s}=S\frac{KN}{T}.
$$

At control frequency $f_c$, aggregate real-time factor is:

$$
\mathrm{RTF}=\frac{KN/T}{Nf_c}=\frac{K/T}{f_c}.
$$

Every benchmark records:

- GPU, driver, power/clock policy, CPU, OS, and build profile;
- environment count, timestep, substeps, observation/action dimensions, and enabled model terms;
- precision and deterministic settings;
- warm-up and sample duration;
- CUDA-event device time and end-to-end wall time;
- whether policy inference, reward, reset, randomization, sensors, rendering, readback, and logging are included; and
- peak device and host memory.

Required benchmark tiers:

| Tier | Included |
| :--- | :--- |
| Physics microbenchmark | Dynamics and integration only |
| Full environment | Physics, kinematic sensors, observations, rewards, termination, and configured reset semantics |
| Training loop | Full environment plus policy forward/backward and optimizer |
| Visual pipeline | Selected snapshots, rendering, requested outputs, and separately measured readback/export |

Initial capacity floor: **10,000 state-only environments at a 100 Hz control rate in real time or faster on the RTX 2070 SUPER target**, equivalent to at least 1,000,000 environment transitions/s for the full-environment tier. This is a useful minimum, not a claimed maximum. Maximum throughput is established by Phase 1 measurements.

Rendering throughput is reported in sensor megapixels/s by output set and resolution, not only cameras/s. Viewer cost is reported as frame-time percentiles and measured training-throughput change under a fixed workload. The viewer must not introduce a synchronous dependency into `step`; a universal `<0.5%` same-GPU overhead promise is not credible.

---

## 10. Implementation Roadmap

```mermaid
flowchart TD
    P0["Phase 0: Contracts and feasibility<br/>Reference model, benchmark definitions, interop spike"]
    P1["Phase 1: State-only GPU simulation<br/>CUDA dynamics, sensors, reset semantics, PyTorch binding"]
    P2["Phase 2: RL baseline<br/>Gymnasium adapter, tasks, randomization, reproducible training"]
    P3["Phase 3: Rendering integration<br/>RenderSnapshot adapters, viewer, bounded visual batches, replay"]
    P4["Phase 4: Synthetic data and transfer evaluation<br/>Scenes, calibrated sensors, exporter, held-out real protocol"]
    P5["Phase 5: Evidence-gated research<br/>Failure mining, residual dynamics, latent models"]

    P0 --> P1 --> P2 --> P3 --> P4 --> P5
```

### Phase 0: Contracts and feasibility

**Goal:** Retire theoretical ambiguity and the highest-risk platform assumption before optimizing.

- [ ] **Freeze the v1 physical contract**
  - Encode frame, quaternion, unit, actuator, timing, and termination conventions in types/configuration.
  - Define the valid parameter and action envelope.
- [ ] **Build a reference multirotor model**
  - Favor clarity and `float64` verification over throughput.
  - Cover force/torque allocation, motor lag, rigid-body derivatives, and sensor truth.
- [ ] **Create analytic and convergence fixtures**
  - Free fall, stationary supported IMU, constant rotation, symmetric hover, zero-force drift, and drag decay.
  - Compare integrators by error, stability, and cost before selecting one.
- [ ] **Specify benchmark tooling**
  - Emit the metadata and qualified rates from Section 9.
  - Prevent accidental host synchronization from being hidden outside the measured interval.
- [ ] **Prototype the render transfer seam**
  - Implement or measure a minimal staged path against the current CPU-authored `Frame` interface.
  - Test whether `wgpu` 30's unsafe Vulkan HAL route can safely wrap exportable memory and synchronize with CUDA on the target machine.
  - Record a go/no-go decision for native interop; retain staging as the supported fallback.

**Exit gate:** Reference trajectories and conventions are executable; benchmark terms are unambiguous; rendering has a proven fallback; native interop is classified as supported, experimental, or rejected.

### Phase 1: State-only GPU simulation

**Goal:** Implement the full environment tier without rendering or policy inference.

- [ ] **Implement batched dynamics and selected integration**
  - Couple actuator and rigid-body state correctly.
  - Normalize attitude and detect non-finite or out-of-envelope state.
- [ ] **Implement deterministic reset and randomization**
  - Support masked reset, stable episode IDs, and counter-based random streams.
  - Randomize only named, physically valid parameters.
- [ ] **Implement kinematic sensor state**
  - IMU truth, discrete noise/bias processes, and analytic ground range.
- [ ] **Implement task transition kernels**
  - Observations, rewards, `terminated`, `truncated`, termination codes, and final episode data.
  - Fuse kernels only when profiling shows a benefit and semantics remain unchanged.
- [ ] **Implement the PyTorch adapter**
  - Validate device/dtype/shape, respect CUDA stream ordering, and avoid steady-state allocation and host synchronization.
- [ ] **Verify CPU/GPU equivalence and measure throughput**
  - Compare derivatives and finite trajectories over the declared envelope.
  - Demonstrate the initial 10,000-environment capacity floor or document the measured bottleneck before changing scope.

**Exit gate:** Correctness fixtures pass on reference and CUDA implementations; reset semantics are observable and reproducible; full-environment throughput is reported under the fixed protocol.

### Phase 2: Vectorized RL baseline

**Goal:** Demonstrate that the simulator trains and evaluates a policy correctly.

- [ ] **Implement `DroneVectorEnv`**
  - Expose Gymnasium-compatible `reset()` and five-value `step()` results: observations, rewards, terminations, truncations, and infos.
  - Declare autoreset mode and preserve final observations and episode statistics.
  - Keep a typed tensor-native interface beneath the Python compatibility adapter.
- [ ] **Implement state-based tasks**
  - Begin with hover and target tracking.
  - Add waypoint and state-based gate navigation only after simpler tasks establish correctness.
- [ ] **Implement domain randomization**
  - Sample validated distributions for mass/inertia, rotor coefficients, actuator response, wind, initial state, and sensor characteristics.
  - Version every distribution used in an experiment.
- [ ] **Train a reproducible baseline**
  - Start with a small on-policy baseline; select the library from integration simplicity and profiler evidence.
  - Report success, sample count, wall time, and distribution across multiple seeds.

Initial hover acceptance profile, to be versioned with the task: at least 90% of randomized evaluation episodes remain within the declared position/attitude envelope for a 20-second episode. “Converges in under two minutes” is a stretch measurement on named hardware, not a correctness gate.

**Exit gate:** A saved policy reproducibly meets the versioned hover criterion on held-out randomized seeds, and the end-to-end training profile identifies simulation versus learner cost.

#### Current GPU hover / PufferLib integration

JJ change `mxmnyztp` adds a concrete GPU hover environment and a PufferLib-derived Torch learner. The chosen integration is a narrowly vendored, explicitly adapted learner from [PufferLib 4.0 revision `42f70d6932c30ac977736f861006809c50168ba9`](https://github.com/PufferAI/PufferLib/tree/42f70d6932c30ac977736f861006809c50168ba9), not the full PufferLib installation or its CPU-environment vectorizer. Attribution, the MIT license, and the exact adaptation rationale are retained in `rl/vendor/`. The learner retains Muon, prioritized trajectory-segment replay, clipped policy/value objectives, and V-trace corrections. It uses a two-hidden-layer float32 MLP Gaussian policy; native BF16 learning, recurrent policies, distributed execution, Protein, and dashboards are not included.

`cuda/src/hover_env.h` exposes the task through a C interface; `rl/environment.py` borrows its CUDA buffers through ctypes and the CUDA array interface. Physics remains in `PhysicsBatch`. The task owns GPU observations, rewards, episode clocks/counters, failure/timeout decisions, deterministic reset generation, final observations, and same-step autoreset. Python consumers and native operations use one declared CUDA stream. Policy actions map to `clamp(hover_command + 0.15*tanh(raw_action), 0, 1)` per rotor; the policy learns the offsets, with no scripted stabilizing controller. The 22 observation values are target-relative world position, world velocity, a row-major body-to-world rotation matrix, body angular velocity, and normalized actual rotor speeds.

Task version 1 holds a fixed reference Quad-X at target `(0,0,1)` using a 100 Hz control rate and 16 midpoint substeps. Initial position varies by ±0.15 m per axis, roll/pitch by ±0.05 rad, yaw across a full turn, velocity by ±0.05 m/s, and body rate by ±0.02 rad/s. Motor state starts at equilibrium. Ground crossing, distance beyond 2 m, tilt beyond 0.8 rad, and nonfinite state/actions terminate; the 2,000-control limit truncates instead. Final observations are preserved before reset on every step. The learner bootstraps timeouts from final observations, suppresses bootstrap on failure, stops advantage recurrence at either episode boundary, and retains the final rollout transition.

The target Nix shell includes Python 3.12, uv, and runtime/driver library discovery. `rl/uv.lock` pins Torch 2.7.1+cu128, NumPy 2.2.6, and transitive dependencies. From the repository root:

```sh
nix develop
nu rl/setup.nu
cuda/build/rl-venv/bin/python -m unittest rl.test_advantage
cuda/build/rl-venv/bin/python rl/train.py train --seed 1 --log-every 25 --checkpoint cuda/build/hover-final-seed1.pt
cuda/build/rl-venv/bin/python rl/train.py train --seed 2 --log-every 25 --checkpoint cuda/build/hover-final-seed2.pt
cuda/build/rl-venv/bin/python rl/train.py eval --checkpoint cuda/build/hover-final-seed1.pt --eval-envs 1024 --eval-seed 900000001
cuda/build/rl-venv/bin/python rl/train.py eval --checkpoint cuda/build/hover-final-seed2.pt --eval-envs 1024 --eval-seed 900000002
cuda/build/rl-venv/bin/python rl/profile_rollout.py
```

The default training run uses 1,024 environments, a 64-control rollout horizon, minibatches of 8,192 transitions, replay ratio 2, and 20 million requested transitions (20,054,016 after completing the last rollout). Both independently trained checkpoints loaded successfully with `weights_only=True` and passed fresh held-out evaluation:

| Training seed | Fresh evaluation seed | Successful 20-second episodes | Fixed hover-thrust baseline | Mean episode maximum distance / tilt |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 900000001 | 1,024 / 1,024 | 0 / 1,024 | 0.278 m / 0.088 rad |
| 2 | 900000002 | 1,024 / 1,024 | 0 / 1,024 | 0.309 m / 0.126 rad |

Success requires remaining within 1 m of the target and 0.7 rad tilt for the entire 20 seconds, without failure. Evaluation counts only the first episode per environment, not subsequent autoresets. Checkpoints retain task/learner configuration, seeds, optimizer and RNG state, and evaluation metadata; they support saved-policy evaluation, not exact mid-episode simulator restoration.

All five native CTests passed, including hover timeout/failure separation, final observations, seed replay, and cross-stream stepping. Compute Sanitizer reported zero errors for the hover test. Both advantage regression tests passed on CPU and CUDA. A warmed 1,024-environment, 64-control rollout trace recorded 3,338 GPU kernels and 513 device-to-device copies, with no host-copy activities, scalar readbacks, CUDA allocation/free calls, or host synchronization inside the collection interval. `rl/profile_rollout.py` repeats this check and saves its trace and summary under `cuda/build/`; profiler setup/teardown, learner updates, and logging are explicitly outside that interval. Torch's tensor-standard-deviation Gaussian sampling originally introduced per-step readbacks; equivalent `loc + scale*randn_like(loc)` arithmetic removed them. Device copies and multiple kernel launches remain; no CUDA-graph or maximum-throughput claim is made.

This establishes the first state-based training milestone on the RTX 2070 SUPER, not the complete Phase 1/2 roadmap or a sim-to-real result. Vehicle parameters are fixed; broad domain randomization, wind/drag, noisy sensors, other tasks, a general Gymnasium adapter, and rendering integration remain separate work.

#### Randomized target tracking contract

JJ change `zmowkoop` adds a separately named `tracking` task; `hover` remains the default and retains its task/checkpoint contract. Both share the existing `PhysicsBatch`, direct four-motor action mapping, reward coefficients, initial-state distribution, and 22-value ground-truth observation layout. For tracking, the first three observations are world position relative to the **current** commanded target. Neither future commands nor countdowns enter the policy observation. Vehicle parameters remain fixed; this slice adds no wind, obstacles, noisy sensors, yaw commands, or rendering.

Each episode lasts 2,000 controls at 100 Hz, with 16 midpoint physics substeps per control (`dt = 0.000625 s`). Commands are generated from seed, environment, episode, and command identities, independently of the vehicle trajectory. Candidates are uniform in `x,y ∈ [-1,1] m`, `z ∈ [0.75,1.75] m`, rejected until displacement from the previous commanded target is within `[0.5,1.5] m`; the first previous target is `(0,0,1)`. Each mixed-schedule command independently chooses a 75% long interval, uniform integer `[200,500]` controls, or a 25% short interval, uniform integer `[20,100]`. Consecutive short commands are allowed. A new command replaces the previous one without resetting physical state.

Transition ordering is physics, reward/outcomes/metrics against the preceding target, then command publication for the next observation or episode autoreset. Final observations and `final_targets` describe the preceding transition. Tracking terminates on nonfinite state/actions, tilt beyond 0.8 rad, `x,y` outside ±3 m, or `z` outside `[0.05,3] m—not on target distance. Timeout truncates. Reward remains `exp(-2*distance² - 0.1*speed² - 0.05*body_rate² - 0.5*tilt²) - 0.005*mean(tanh(raw_action)²)`, with failure reward −1.

Acceptance uses two suites, with 1,024 fresh 20-second episodes per suite and policy:

- **Settling:** four fixed-500-control commands per episode. A command settles only when its final 50 consecutive controls have distance ≤0.2 m and speed ≤0.2 m/s. At least 90% of all commanded targets must settle, with at least 95% episode survival. Failed episode tails do not remove targets from the denominator.
- **Mixed/interruption:** the randomized schedule above, at least 95% survival, and mean episode RMS position error ≤75% of a frozen trained hover policy's mean episode RMS. That baseline continues holding `(0,0,1)`; it is not supplied moving-target errors. Each paired evaluation receives an identical pre-generated command plan.

Every episode contributes its full 2,000-control error horizon. Failure and remaining tail controls receive squared error `37.0625 m²`, the worst arena-to-target bound (`4² + 4² + 2.25²`). Reports distinguish mean episode RMS (the gate), pooled RMS, short-/long-interval pooled RMS, maximum error including failure padding, and action saturation. Saturation is the fraction of executed rotor controls with `abs(tanh(raw_action)) >= 0.99`; nonexistent failure-tail controls do not enter that denominator.

`rl/train.py --task tracking` trains the mixed schedule; `--schedule settling|mixed` filters evaluation only. Tracking checkpoints bind the exact 2,000-control task configuration. `--init-checkpoint` loads weights with a fresh optimizer and records their provenance. Checkpoints and evaluation JSON are retained under ignored `cuda/build/`, not embedded in source control.

#### Tracking training and measured acceptance

Tracking seeds 11 and 22 independently initialized from the previously trained hover seeds 1 and 2, respectively, using fresh optimizers. Each trained for 80,019,456 tracking transitions (80 million requested, rounded to the final complete rollout), with learning rate 0.0025 and the existing 1,024-environment / 64-control rollout / 8,192-transition minibatch / replay-ratio-2 configuration. These are hover-initialized tracking policies, not tracking runs from scratch. Development evaluation used 128 episodes on seeds 5000011 and 5000022. Initial 40-million-transition runs at learning rate 0.0005 failed; increasing learning updates and training duration—not changing the task, reward, or gates—produced the accepted policies.

Both saved policies loaded with `weights_only=True` and passed both suites on previously unused evaluation seeds. The paired frozen baseline was `hover-final-seed1.pt`, holding `(0,0,1)`; it survived all baseline episodes.

| Training seed | Fresh evaluation seed | Settled targets | Settling survival | Mixed survival | Mixed mean episode RMS / baseline | Baseline RMS ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 11 | 910000011 | 4,096 / 4,096 (100%) | 1,024 / 1,024 | 997 / 1,024 (97.36%) | 0.5772 / 0.8384 m | 68.84% |
| 22 | 910000022 | 4,033 / 4,096 (98.46%) | 1,021 / 1,024 (99.71%) | 1,011 / 1,024 (98.73%) | 0.5395 / 0.8432 m | 63.99% |

| Policy / suite | Mean episode RMS | Pooled RMS | Short / long interval RMS | Maximum error | Action saturation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 11 / settling | 0.3702 m | 0.3745 m | — / 0.3745 m | 1.5924 m | 0% |
| 22 / settling | 0.3842 m | 0.4192 m | — / 0.4192 m | 6.0879 m | 0% |
| 11 / mixed | 0.5772 m | 0.8084 m | 1.0909 / 0.7885 m | 6.0879 m | 0.000481% |
| 22 / mixed | 0.5395 m | 0.6448 m | 0.9768 / 0.6188 m | 6.0879 m | 0% |

The 6.0879 m maxima include the explicit failure padding; failures were not discarded. All five native CTests and four Python regression tests passed. The new tests cover command bounds/timing and replay, uninterrupted physical state, old-target reward ordering, failure-tail error accounting, missed settling targets, and original-plan short/long classification after autoreset. An independent recorded-state check recomputed the final-50-control settling windows and matched every native finish/settlement flag across 128 × 2,000 controls (486 settled commands on the development policy). Compute Sanitizer reported zero errors. Both original hover checkpoints still achieved 1,024 / 1,024 hover successes on their regression seeds.

Warmed hover and tracking rollout traces each recorded 3,338 kernels and 513 device-to-device copies for 1,024 environments × 64 controls, with no recorded host-copy activities, scalar readbacks, CUDA allocation/free calls, or host synchronization in the collection interval. This checks rollout residency, not learner-update residency or a controlled throughput comparison. The Nix shell also supplies zlib, required for NumPy to import independently of Torch.

After creating the hover checkpoints with the commands above, reproduce the training and acceptance runs inside `nix develop`:

```sh
cuda/build/rl-venv/bin/python rl/train.py train --task tracking --seed 11 --init-checkpoint cuda/build/hover-final-seed1.pt --steps 80000000 --learning-rate 0.0025 --log-every 200 --eval-envs 128 --eval-seed 5000011 --checkpoint cuda/build/tracking-final-seed11.pt --output cuda/build/tracking-final-development-seed11.json
cuda/build/rl-venv/bin/python rl/train.py train --task tracking --seed 22 --init-checkpoint cuda/build/hover-final-seed2.pt --steps 80000000 --learning-rate 0.0025 --log-every 200 --eval-envs 128 --eval-seed 5000022 --checkpoint cuda/build/tracking-final-seed22.pt --output cuda/build/tracking-final-development-seed22.json
cuda/build/rl-venv/bin/python rl/train.py eval --task tracking --checkpoint cuda/build/tracking-final-seed11.pt --baseline-checkpoint cuda/build/hover-final-seed1.pt --eval-envs 1024 --eval-seed 910000011 --output cuda/build/tracking-acceptance-seed11.json
cuda/build/rl-venv/bin/python rl/train.py eval --task tracking --checkpoint cuda/build/tracking-final-seed22.pt --baseline-checkpoint cuda/build/hover-final-seed1.pt --eval-envs 1024 --eval-seed 910000022 --output cuda/build/tracking-acceptance-seed22.json
cuda/build/rl-venv/bin/python -m unittest rl.test_advantage rl.test_tracking
cuda/build/rl-venv/bin/python rl/profile_rollout.py --task tracking
```

This completes the agreed fixed-vehicle target-tracking slice, not the remaining Phase 1/2 roadmap, general Gymnasium compatibility, or sim-to-real validation.

### Phase 3: Rendering integration and replay

**Goal:** Add visualization and bounded visual sensing without coupling renderer cadence to training cadence.

- [x] **Define and implement `RenderSnapshot`**
  - Select environment IDs and include timestamped poses, object identities, cameras, and scene version.
  - Apply the documented ENU/FLU-to-renderer transform once.
- [x] **Ship the staged-copy adapter first**
  - Use bounded double/triple buffering and explicit backpressure.
  - Measure gather, transfer, `queue.write_buffer`, and render costs separately.
- [ ] **Resolve the native interop experiment**
  - If Phase 0 proves safety and value, isolate unsafe Vulkan/CUDA ownership and semaphore logic behind a native-only adapter.
  - Otherwise close the path without burdening the portable renderer.
- [ ] **Extend `sim-graphics` for calibrated sensor batches**
  - Budget camera count, resolution, and output set explicitly.
  - Verify color transfer, optical z-depth, object-ID stability, occlusion, and background semantics.
- [x] **Build the decoupled viewer**
  - Free/orbit and vehicle-mounted views at a display-rate target such as 60 FPS.
  - Drop stale snapshots rather than delaying simulation.
- [x] **Add versioned trajectory recording and replay**
  - Record conventions, schema version, timestamps, scenario identity, and enough state for reproducible, timestamp-aligned visual replay; bit-exact pixels are required only where a renderer/backend explicitly guarantees them.
  - Support native and web replay through the same logical recording schema.

**Exit gate:** A trained trajectory can be viewed live and replayed; the headless step path remains render-independent; visual correctness fixtures pass; throughput impact is measured rather than assumed.

#### Current trained-policy trajectory bridge

JJ change `rtywlmus` connects saved hover/tracking policies to selected-state staging, a native live viewer, and native/WebGPU recording playback. `rl/view_policy.py` uses the existing checkpoint loader and deterministic policy means; it does not change the learner, task, or ordinary headless training path. This completes the fixed-scene trajectory-viewing slice, not batched camera sensing or the entire Phase 3 exit gate. Visual inspection of the new viewer/playback controls is explicitly left to the user.

The optional native snapshot ring exports actual selected `PhysicsBatch` positions and Hamilton quaternions, then packs current targets and episode IDs on the producing stream. A separate nonblocking stream copies only these immutable compact slots to pinned host memory. Completion events gate polling and slot reuse; later physics never waits for the D2H transfer. Selection is 1–64 unique environments; 2–16 slots are allocated once, with four by default. Full rings drop submissions. Python copies ready rows into a bounded worker queue; JSON encoding, recording and localhost TCP writes belong to the worker, not the control call. A slow network client retains at most one partial line and the latest pending snapshot in application storage; it cannot hold a CUDA slot or block stepping. Teardown may drain/wait, but never waits for a viewer. Recording files must not already exist.

`sim-trajectory` owns the version-1 NDJSON schema, validation, timestamp sampling, coordinate conversion, fixed drone/target/ground geometry and cameras for both native and WASM consumers:

- The first line records schema/scene versions, ENU/FLU and scalar-first quaternion conventions, task, seed, checkpoint path, 10 ms control interval, requested sampling cadence, ordered environment selection and camera FOV/near/far calibration. Scene version 1 is illustrative quadrotor geometry and target markers above a ground plane, not a physical collision asset.
- Each subsequent complete line records global control step/time and selected vehicles' environment IDs, episode IDs, world positions, body-to-world quaternions and active targets. These are **current post-autoreset** poses, not terminal-transition poses. Absolute simulation time never resets.
- Replay sample-holds at the recorded timestamps, including dropped samples; it never interpolates across an episode reset. Recordings preserve realized poses rather than promising simulator restoration or bit-identical pixels across devices.
- ENU `(x,y,z)` becomes renderer `(x,z,-y)` through one proper basis rotation. The mounted camera is 0.5 m forward and 0.12 m up in FLU, looking forward with body up; orbit uses world up. These mounts and geometry are part of scene version 1.
- Background ID is `0`, ground is `4,294,967,295`, and vehicle/target IDs are `2*environment_id+1` / `2*environment_id+2`. Shared identity definitions drive both rendered labels and browser metadata.

Inside `nix develop`, after building the CUDA library and creating the existing policy checkpoints:

```sh
# Record 40 simulated seconds, independent of wall-clock playback speed.
cuda/build/rl-venv/bin/python rl/view_policy.py --task tracking --checkpoint cuda/build/tracking-final-seed11.pt --environment-ids 0,7,31 --steps 4000 --record cuda/build/tracking-view.ndjson
cargo run -p window-demo -- --trajectory cuda/build/tracking-view.ndjson

# Live: start the producer, then the viewer in another terminal.
cuda/build/rl-venv/bin/python rl/view_policy.py --task tracking --checkpoint cuda/build/tracking-final-seed11.pt --environment-ids 0,7,31 --steps 6000 --listen 127.0.0.1:9876 --realtime
cargo run -p window-demo -- --live 127.0.0.1:9876

# Native headless geometry/label/depth smoke and diagnostic captures.
cargo run -p render-smoke -- --trajectory cuda/build/tracking-view.ndjson --output target/tracking-view

# Browser: select the same NDJSON file in the supplied page.
trunk serve --config apps/web-demo/Trunk.toml
```

Native controls: Space pauses, R restarts recorded playback, C switches orbit/mounted cameras, N selects the mounted vehicle, arrows orbit and wheel zooms. The live viewer consumes the latest complete snapshot without waiting for network IO and preserves its last pose on disconnect. Start a new viewer to reconnect. Browser loading starts paused; GPU buttons control play/pause and camera mode. Space toggles playback, Home restarts, `[`/`]` seek one second, C switches cameras, N selects the vehicle, and S returns to the showcase. Embedders can call `load_trajectory(ndjson)`, `play()`, `pause()`, `restart()`, `seek(absolute_seconds)` and `set_camera_mode("orbit"|"mounted")`; invalid loads preserve the existing recording. RGB/depth/ID/comparison modes remain available. Browser replay needs no CUDA or Python.

Verification on the RTX 2070 SUPER: all five native CTests, six Rust tests and four Python regressions passed; the release WASM bundle built. The native regression covers blocked-stream nonblocking submission/polling, full-ring drops, cross-stream ordering, selected pose/attitude/target correspondence, post-reset episode identity and independence from later state mutations. Compute Sanitizer reported zero errors after warming its snapshot-kernel instrumentation before the blocked-stream check. Shared tests cover sample-hold through timestamp gaps/resets and physical yaw/mounted-camera basis conversion. A real 25-second tracking rollout recorded all 501 requested samples. A separate 64-environment live stream delivered 4,901 increasing complete snapshots, each matched exactly against the simultaneously recorded frame. A deliberately non-reading client still completed all 5,000 controls while replacing 4,981 network snapshots. The received recording then passed native parsing and six headless RGB/depth/ID renders (first/middle/last, orbit/mounted), including finite optical depth, far-valued background and identity-map checks. No visual sign-off is claimed.

Measured staging impact, **not learner-update or live-viewer frame-rate performance**: three sequential paired runs used tracking seed 10001, checkpoint seed 11, 1,024 environments, 100 warm-up controls followed by reset, 10,000 measured controls, selection `[0,7,31]`, and one sample per five controls. Policy inference, full environment and optional recording/worker activity are inside the wall interval; checkpoint loading, construction, warm-up/reset and final recording drain are outside. Unpaced baseline median was 2.240 s (range 2.188–2.338), versus 2.639 s (2.612–2.674) with staging/recording: about 4.57 versus 3.88 million environment transitions/s, or a 15.1% throughput reduction. All three recording runs retained 2,001 samples without ring/worker drops. Average GPU gather/packing was 5.11–5.21 microseconds per selected snapshot; D2H was 2.12–2.14 microseconds, timed separately with CUDA events. These are uncontrolled-clock Linux/i7-9700K/RTX-2070-SUPER measurements with driver 595.99.02, Release CUDA and float32 physics/policy; ranges are not confidence intervals.

For a three-vehicle, 960×540 RGB/depth/ID headless scene, six debug-build samples measured CPU scene preparation at 11–20 microseconds, upload-plus-render CPU submission at 141–295 microseconds, and GPU-completed wall time at 309–673 microseconds. The latter includes submission/polling and is **not** a GPU timestamp measurement. Separate temporary instrumentation measured the two `queue.write_buffer` CPU calls together at 7.8–36.5 microseconds on the six GPU-only submissions; that measures CPU staging, not isolated GPU transfer time. Explicit render/readback took 65.6–72.6 ms and remains outside interactive rendering. The probe and transport/performance driver were removed after measurement; the CLI retains stage counters/timings and a `--disabled` paired-baseline mode. Concurrent learner-plus-viewer throughput, display latency, visual inspection and the optional native-interoperation decision remain unclaimed.

#### Large-jump playback stress scenario

`rl/view_policy.py --scenario long-flight-v1 --task tracking` loads the **same unchanged tracking checkpoint** but commands abrupt targets `(10,0,2)`, `(10,10,2)`, `(-10,10,2)`, `(-10,-10,2)`, `(10,-10,2)` and `(0,0,2)` in metres, held for 10 seconds each. The route contains 10 m and 20 m horizontal jumps; it is not a smoothed target trajectory or a scripted flight controller. The episode limit is 60 seconds and the arena is widened to horizontal ±20 m and altitude `[0.05,5]` m so the original ±3 m arena does not reject the requested destinations. Tilt, nonfinite-action/state and ground failure checks, motor mapping, physics and same-step autoreset are unchanged. A failure restarts the route at its first target.

This is an out-of-training-distribution stress scenario, not a new training task or acceptance claim. Existing mixed/settling schedules and checkpoint contracts are unchanged. Recordings identify it as `tracking-long-flight-v1` rather than ordinary `tracking`, and native/web readers support that identity.

```sh
cuda/build/rl-venv/bin/python rl/view_policy.py --task tracking --scenario long-flight-v1 --checkpoint cuda/build/tracking-final-seed11.pt --batch 1 --environment-ids 0 --steps 6000 --record cuda/build/tracking-large-jumps.ndjson
cargo run -p window-demo -- --trajectory cuda/build/tracking-large-jumps.ndjson
```

Observed with environment seed 10001 and policy seed 11: all 1,201 requested samples were recorded over 60 simulated seconds, with 240 episode resets. The policy failed on the first 10 m command before reaching later route points; the recording retains those failures rather than substituting a successful or smoothed flight. Five native tests, the two shared replay tests, four Python regressions and the release WASM build passed after adding the scenario.

#### Current sensor-inspection slice

JJ change `uunkvlmu` starts the visual/data path independently of drone training. `sim-inspection` supplies a shared version-1 scene configuration: explicit axis-aligned boxes with stable nonzero uint32 IDs, names, positions, scales and linear colors, plus camera pose, vertical field of view, clipping planes and resolution. Both the native window inspector and headless renderer consume this configuration. Coordinates are explicitly right-handed **Y-up, in metres**, matching the renderer; this is not yet the ENU/FLU physics `RenderSnapshot` adapter.

The inspector displays RGB, optical depth, instance colors, and a separate observer view with the sensor's true near/far frustum, world axes, and object identity callouts. Observer gizmos never enter sensor labels. Camera intrinsics use pixel centers `(column+0.5,row+0.5)` with the image origin at the upper-left edge. Exported extrinsics map world coordinates into optical X-right/Y-down/Z-forward coordinates and are stored column-major.

Interactive presentation uses `GpuInspector` and the existing renderer's frame graph, mesh registry, and transient texture pool. Both scene views use `Renderer::execute_gpu`, which skips readback scheduling. Sensor outputs are copied into persistent sampled GPU textures before the observer graph can reuse transient attachments. Panel coloring, text and callouts are composed on the GPU and presented through a `wgpu` surface. Meshes, pipelines, font atlas and sampled textures persist; sensor textures resize only when sensor dimensions change. Camera input only updates the scene and requests redraw. Explicit export retains the synchronous CPU capture path.

The initial CPU-preview design was rejected after an instrumented debug-build run measured a 792.63 ms median camera-update path: roughly 39 ms sensor capture/readback, 2 ms observer capture, 220 ms CPU panel construction and 532 ms CPU preview resizing. The replacement's warmed, offscreen GPU-completed frame benchmark measured 0.544 ms median over 16 camera updates at 948×1064 output with 640×480 sensors on the RTX 2070 SUPER. Completion waits were benchmark-only; the interactive rendering method does not poll or wait for readback. These are pipeline timings, not desktop input-to-photon latency or a vsync/frame-rate guarantee. The GPU-composited image was inspected offscreen.

Native controls: arrows orbit the camera; `+`/`-` dolly; PageUp/PageDown translate camera height; `[`/`]` change FOV; `S` saves the configuration; `L` reloads; `R` resets the unsaved scene; `E` explicitly exports a capture; Escape closes. A missing window scene file starts the default scene. Invalid reloads report an error and preserve the current valid scene. Repeated window exports use unique subdirectories under `target/inspection/`.

```sh
nix develop
cargo run -p window-demo -- --inspect
# Or edit/reload a specific scene:
cargo run -p window-demo -- --inspect target/my-scene.json
cargo run -p render-smoke -- --inspect --output target/my-capture --verify
cargo run -p render-smoke -- --inspect --scene target/my-capture/scene.json --output target/my-replay
```

Capture destinations must be absent or empty. A complete bundle is published together rather than overwriting a previous capture with partially updated files:

- `scene.json`: the exact versioned scene configuration.
- `metadata.json`: dimensions, layout, intrinsics, extrinsics, object identities and output interpretation.
- `color.rgba8`: top-to-bottom, tightly packed **linear RGBA8 UNORM**, not sRGB; lighting is clipped/quantized by the existing sensor attachment.
- `depth.f32le`: float32 little-endian optical Z-depth in metres; background is the camera's far value.
- `object_ids.u32le`: uint32 little-endian IDs, with background `0`; no palette or 8-bit truncation.
- `preview.png`: diagnostic panels only, including sRGB display conversion, false-color depth, instance palette and observer annotations.

The headless `--verify` command checks all 76,800 pixels of an asymmetric, analytically projected two-box scene: visible IDs, occlusion, optical depths 3.5/5.5 m and far-plane background. It also verifies exact RGB/depth/ID replay after saving and reloading the supplied scene on the current renderer. An independent export check reconstructed all 30,710 foreground pixels in the default 640×480 capture using its calibration; the maximum distance from the labelled box surface was 0.0000273 m, and IDs 11, 257 and 65539 survived export intact. This is same-backend replay evidence, not a promise of bit-identical rendering across devices.

After the GPU presentation cutover, the headless geometry/replay checks still passed and all three raw sensor payloads were byte-identical to the preceding CPU-inspector capture. All four existing rendering/registry unit tests passed, and the unchanged headless demo still rendered its 401-instance scene. Unsupported scene versions are rejected without creating a capture bundle. The temporary latency probe and timing instrumentation were removed after measurement.

This slice establishes inspectable sensor output and scene replay. Procedural scene distributions, large dataset batches, perception-model comparisons, physical camera motion and sim-to-real improvement remain separate milestones.

#### Embeddable WASM canvas showcase

`apps/web-demo` is a canvas-only WebGPU showcase with deterministic courtyard and calibration scenes. RGB, depth, uint32 instance IDs and comparison use the same sensor camera. Gates share an ID across their component meshes; both scenes include ID `4,000,000,001`. GPU-rendered buttons select outputs, switch scenes, reset the view, toggle orbit and zoom. Control state and hit testing live in WASM; the small `app.js` harness supplies DOM pointer/keyboard events and animation scheduling, not a surrounding website UI.

Depth **display colors** auto-range between the nearest and farthest visible foreground samples each frame. A GPU workgroup reduction ignores background IDs and invalid depths; a constant-depth surface uses the palette midpoint. Nearest is red, farthest blue, and background black. The underlying optical-Z depth remains in metres. The overlay is added only after sensor rendering, so its buttons never enter sensor labels. Sensor textures use five bounded resolution tiers up to 960×720; neither normal presentation nor relative-depth reduction reads images back to the CPU.

```sh
nix develop --command trunk build --config apps/web-demo/Trunk.toml --release --locked
nix develop --command trunk serve --config apps/web-demo/Trunk.toml
```

The release bundle is in `apps/web-demo/dist/`. A host imports `web-demo.js`, calls its default WASM initializer, then `create_renderer(canvas)`. Keep `web-demo_bg.wasm` beside the generated JS module; `web-demo.d.ts` supplies TypeScript declarations. `Engine.render(width, height, pixel_ratio, delta_seconds)` draws into the supplied canvas; dimensions are backing pixels, UI pointer coordinates use those same pixels, and camera drag deltas use CSS pixels. Reuse the bundled `app.js` as the minimal event/render-loop example. No global renderer variable or fixed canvas ID is required by the WASM module. Serve via HTTPS or localhost in a WebGPU-enabled browser; CUDA physics and trained policies are not part of this browser demo.

Verification: real browser RGB/depth/ID/comparison output and GPU buttons were inspected. Output selection, scene switching, zoom, orbit start/stop, reset and cancelled GUI drag-out were exercised; selecting outputs or cancelling a GUI gesture did not move the camera. The user also verified the canvas and relative-depth presentation. The shared attachment-sampling change passed the four existing rendering tests and the 76,800-pixel native analytical depth/occlusion and exact scene-replay smoke check.

### Phase 4: Synthetic data and transfer evaluation

**Goal:** Produce auditable datasets and answer whether they improve performance on held-out real data.

#### Seeded obstacle generation and static showcase (2026-09-10)

The first Phase 4 slice is implemented in JJ change `rvsmtnov`. `sim-scene` is the shared, GPU-independent Rust scene/generator module used by native inspection and WASM. Scene schema **v2** replaces the earlier inspection v1 schema: each instance owns a unique nonzero uint32 ID, name, semantic class and one or more world-space axis-aligned box parts. Unsupported versions reject; there are no v1 compatibility aliases. Right-handed Y-up metres and optical X-right/Y-down/Z-forward calibration remain unchanged. A saved realized scene replays without calling the generator.

Generator recipe v1 samples ground, two three-part gates and box obstacles. The complete recipe records seed, image dimensions, inclusive integer distributions for obstacle count/size and gate width/height, obstacle spread and camera jitter. Placement uses shuffled cells with bounded jitter, colors and camera FOV also vary, and each sample has its own seed/index-derived integer RNG. Sample order and batch boundaries do not affect scene data. Gates share one instance ID across all three parts; full-width IDs exercise values above signed int32. Ontology v1 defines background 0, ground 1, gate 2 and obstacle 3. These are bounded geometric scenes, not realistic materials or a general asset/weather generator.

```sh
cargo run -p render-smoke -- --generate --seed 42 --count 3 --output target/dataset
cargo run -p render-smoke -- --generate --seed 42 --start-index 2 --scene-only
cargo run -p render-smoke -- --inspect --scene target/dataset/sample-0000000000/scene.json --output target/replay --verify
NO_COLOR=true trunk build --config apps/web-demo/Trunk.toml --release --locked
```

`--recipe FILE` excludes `--seed`, `--width` and `--height`; duplicate/unknown flags, zero count, invalid dimensions and sample-index overflow reject. Batch output must be absent. One inspector captures all samples into a temporary sibling directory; the dataset manifest and all samples publish together. Linux/Android use no-replace rename; other platforms check before standard rename (a concurrent empty-directory creation can be replaced on other Unix, never an existing populated dataset). This is a synchronous, small-batch exporter, not the planned asynchronous pipeline.

Each sample includes exact `scene.json`, metadata v2 with provenance/calibration/ontology/instance mapping, optical-Z `depth.f32le`, uint32 `object_ids.u32le`, uint32 `semantic_classes.u32le`, raw linear `color.rgba8`, sRGB `color.png`, and diagnostic `preview.png`. PNG conversion uses the same clipped/quantized linear LDR source; it does not restore HDR information. Mask IDs are checked against the scene before publishing. The batch manifest names every sample and its scene/metadata paths.

`apps/web-demo` now wraps the reusable WebGPU canvas in a static showcase, replacing its independent hardcoded scene builder with the shared generator. Seed/sample selection, output modes, calibrated camera reset, instance/class legend, calibration/provenance and realized JSON download are available. Scene downloads reflect the current interacted camera; recipe provenance remains the original generation recipe. Sensor resolution remains recipe-defined when the canvas resizes. Recorded trajectory playback remains supported. The GPU-free `generate_scene(seed, sample_index)` WASM export also works without WebGPU. `examples/seed-42-sample-0/` is a real native capture with downloadable payloads and a static fallback page; update it alongside future generator/sensor changes. No CUDA physics runs in the browser.

Verification: Rust workspace **13 tests**, Python **4 tests**, CPU physics **2 CTests** and CUDA physics **5 CTests** passed; release WASM built. Three 640×480 samples passed independent per-pixel instance/class/depth checks (**921,600 pixels**); all four classes and uint32 IDs above int32 were visible. Native replay matched raw color/depth/IDs/classes and sRGB PNG exactly, and the existing **76,800-pixel** analytical projection/depth/occlusion check passed. Five native/WASM scene comparisons, including seed/index `u32::MAX`, were exactly equal as parsed scene data. Browser generation, comparison display, camera modification/reset, downloaded-scene native replay, trajectory seek/mounted-camera controls and a 390px-wide WebGPU-unavailable fallback were exercised; live WebGPU and fallback screenshots were inspected. Cross-GPU pixel equality and sim-to-real improvement are not claimed.

Two-axis review found one CLI regression: the new help check converted a Unix filename through `std::env::args()`. It was corrected to fallible conversion from `args_os()`; an actual non-UTF-8 output filename changed from a panic to a successful 401-instance PNG render, and focused review cleared the correction. The checkout also lacked the already-declared embedded window skybox; its original Poly Haven CC0 asset was restored and checksum-verified. The ignored Python environment was rebuilt from its lock after its Nix-store interpreter disappeared.

The remaining Phase 4 roadmap below still includes richer scene families, camera/image effects, asynchronous export, downstream dataset adapters and real-data evaluation. This slice does not satisfy the full Phase 4 exit gate.

#### Remaining Phase 4 work

- [ ] **Extend the versioned scene schema and procedural generator beyond the bounded box family**
  - Add towers, terrain, richer clutter/obstacles, asset references, lights, materials and weather parameters.
  - Retain generator version, seed and realized parameters across those additions.
- [ ] **Implement camera and image-domain variation**
  - Intrinsics/extrinsics, exposure, transfer function, distortion, noise, blur, rolling shutter, and compression as explicit, independently testable stages.
- [ ] **Implement the asynchronous exporter**
  - Export color, depth, instance IDs, categories, camera calibration, and optional 2D/3D boxes.
  - Provide format adapters based on downstream consumers and validate round trips.
- [ ] **Establish the real-data protocol**
  - Label ontology, privacy/provenance rules, leakage-resistant splits, baselines, model versions, and confidence intervals.
- [ ] **Implement evaluation adapters and reports**
  - Downstream segmentation/detection metrics are primary.
  - Representation metrics and synthetic-vs-real distribution diagnostics are secondary.
- [ ] **Tune without test leakage**
  - Tune generator distributions only on allowed splits; run the held-out real test at declared release gates.

**Exit gate:** Dataset samples are traceable and geometrically validated; at least one complete training/evaluation matrix is reported on an untouched real test split. No fixed mIoU target is set before the dataset, ontology, and baseline exist.

### Phase 5: Evidence-gated research

**Goal:** Add complexity only where a measured baseline exposes a valuable gap.

- [ ] **Adversarial scenario and failure search**
  - Search valid structured parameters, preserve exact replays, and verify curriculum generalization.
- [ ] **Learned force/torque residual**
  - Proceed only with suitable data and improvement on held-out open- and closed-loop metrics.
- [ ] **Latent predictive model**
  - Proceed only with a defined planning or sample-efficiency use case and a non-learned baseline.
- [ ] **Future vehicle-family assessment**
  - Specify a separate state/contact interface and choose an established contact engine or solver strategy before adding ground vehicles.

**Exit gate:** Each research module independently demonstrates net value under a predeclared evaluation and can be disabled without changing core simulator semantics.

---

## 11. Verification Matrix

| Area | Required evidence |
| :--- | :--- |
| Frame and quaternion conventions | Round-trip basis tests; known 90° rotations; body/world force direction fixtures |
| Rigid-body dynamics | Analytic special cases, convergence trend, invariant checks, and comparison to a high-accuracy reference |
| Rotor allocation | Per-rotor force/torque fixtures and symmetric hover equilibrium |
| IMU | Stationary, free-fall, constant-rate rotation, bias, and sample-rate scaling fixtures |
| Randomness | Same-seed repeatability; independence from reset order and unrelated environments; distribution checks |
| Episode semantics | Separate termination/truncation cases; all autoreset modes; preserved final observations and statistics |
| CUDA adapter | Shape/device failures, non-default stream ordering, absence of steady-state allocation, CPU/GPU numerical comparison |
| Performance | All four benchmark tiers with full metadata and qualified rate units |
| Render bridge | Backpressure behavior, dropped-frame accounting, timestamp freshness, staged-path fallback |
| Visual outputs | Known color, z-depth, ID, occlusion, camera projection, and background fixtures |
| Recorder | Schema-version rejection/migration behavior and native/web replay consistency |
| Dataset export | Provenance completeness, format round trip, mask/category integrity, calibration consistency |
| Sim-to-real claims | Leakage-resistant split, named regimes, per-class metrics, uncertainty, and untouched real test results |
| Learned modules | Held-out improvement, closed-loop effect, bounded failure behavior, and total compute cost |

Numerical tolerances belong to each versioned fixture and are derived from precision, timestep, horizon, and task sensitivity. A single global trajectory-error threshold is not meaningful.

---

## 12. Risks and Decision Gates

| Risk | Why it matters | Gate or fallback |
| :--- | :--- | :--- |
| CUDA/`wgpu` external-memory integration is unavailable or fragile | It is backend-specific and unsafe; WebGPU does not expose raw CUDA pointers | Stage selected snapshots through bounded buffers; keep native interop optional |
| Rendering saturates memory bandwidth or VRAM | Pixel cost scales with resolution, outputs, and visual cameras—not physics count | Cap `N_visual`, decimate sensor rate, measure megapixels/s, and separate export workloads |
| Simulator exploits unrealistic dynamics | High RL throughput can optimize model error faster | Validate regimes, randomize justified uncertainty, compare real logs, and retain safety margins |
| Real-data leakage inflates transfer results | Adjacent video frames and repeated locations are strongly correlated | Split by flight/site/session and hold back a final test set |
| Kernel fusion obscures correctness | Fused reset/reward/state code can erase terminal data or complicate tests | Preserve the typed `StepResult`; fuse only behind equivalence fixtures |
| Nondeterminism breaks reproduction | Parallel execution and reset ordering can alter random streams | Counter-based RNG and explicit backend determinism guarantees |
| Learned residual destabilizes control | Extrapolated force/torque can inject energy | Bound/gate outputs and test closed-loop behavior outside training regimes |
| Scope expands into a general robotics engine | Contact-rich vehicles require fundamentally different solvers and state | Finish multirotor gates first; design a new adapter only when a second family is funded |

### Decisions intentionally left open

These are engineering decisions, not missing theory:

- exact integrator and physics timestep;
- kernel fusion and launch strategy;
- internal tensor packing;
- PyTorch extension mechanism and build system;
- RL implementation/library;
- staged-buffer sizing and render batch strategy;
- whether native CUDA/Vulkan interop is worth maintaining;
- recording and dataset container formats;
- scene-search algorithm; and
- learned model architecture and inference runtime.

Changing a fixed physical or observable contract requires a versioned migration. Changing an open engineering choice requires benchmark evidence and must not leak new complexity through the module interface.

---

## 13. Design References

- [ROS REP 103: coordinate conventions and SI units](https://www.ros.org/reps/rep-0103.html)
- [Gymnasium vector environment interface and autoreset semantics](https://gymnasium.farama.org/api/vector/)
- [DLPack specification](https://dmlc.github.io/dlpack/latest/)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [`wgpu` 30 `Device` native HAL escape hatches](https://docs.rs/wgpu/30.0.1/wgpu/struct.Device.html#method.as_hal)
