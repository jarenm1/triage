# Triage: procedural visual control and sim-to-real transfer

Updated: 2026-09-12. Status: research direction and implementation plan, not a report of demonstrated visual transfer.

## 1. Direction and immediate decision

Build Triage around a testable question:

**Can a small recurrent RGB policy learn transferable collision avoidance from cheap, diverse procedural simulation, with little deployment-domain data and no manually labeled perception dataset?**

The main training path is `synthetic RGB history -> learned representation -> RL policy -> control`; deployment substitutes real camera observations. Simulator state can support rewards, diagnostics, auxiliary targets, and a critic experiment. It does not define the canonical visual latent or an obligatory privileged teacher.

Start with slow, local navigation through opaque obstacles, at fixed altitude and approximately fixed heading. Add dynamic obstacles after static transfer works. General navigation, open-world semantics, and aggressive six-degree-of-freedom flight are later claims requiring separate evidence.

### Decisions

- **First engineering slice: E000, a reproducible RGB closed-loop experiment harness.** Connect observations to actions, collisions, resets, saved runs, and replay before expanding the procedural grammar.
- **First research experiment: E001, procedural visual transfer.** Compare narrow appearance randomization, broad randomization, and broad randomization plus paired-render consistency under the same control architecture and budget.
- Use the existing Rust rendering and CUDA/Python infrastructure first. Benchmark 256 rendered environments with staged copies. Increase parallelism based on the measured full-loop profile.
- Use high-level planar velocity commands for the new navigation task. Do not simultaneously solve visual transfer and raw-motor control.
- Use a small CNN and recurrent policy for the research baseline; a four-frame CNN is sufficient for the E000 integration smoke test.
- Spend the first real-world effort on camera/latency checks and a small closed-loop course, not on collecting a large passive video dataset.
- Record every experiment, including failed runs, in [the results ledger](experiments/RESULTS.md). Commit specifications and compact reports; retain large artifacts outside Git with checksums and a backup.

All resource budgets, model sizes, performance targets, and data-hour schedules below are proposed starting points. Measure them on the actual GPU and deployment platform.

### Existing work and document authority

The previous plan is preserved in [the engineering archive](docs/archive/plan-2026-09-12.md), originating at Git commit `839125f`. Use it for detailed physical conventions, implemented interfaces, historical measurements, and replay/sensor contracts. This document supersedes its research priorities and milestone ordering. Treat historical performance numbers as historical, not as new measurements.

## 2. What is sound, and what needs correction

### Sound parts

1. Control rewards provide a reason to preserve information useful for action rather than reconstruct every pixel.
2. Procedural geometry can cover many collision-relevant structures without naming or modeling every semantic object class.
3. Rendering the same physical history with different appearances provides unusually strong, cheap counterfactual supervision.
4. Temporal observations are essential for ego-motion, moving obstacles, and reasoning about recently occluded space.
5. Holding out generator families is more informative than holding out seeds from a single grammar.
6. A fast physical pathway and slower semantic context are compatible with an externally RGB-to-action system.
7. A modest GPU can support useful visual-control experiments if image storage, batch sizes, and auxiliary models are bounded.

### Assumptions to reject or qualify

**Support inclusion is insufficient.** “Real geometry is contained in the training distribution” is not an operational guarantee. Relevant situations must occur with enough probability and at useful difficulty. A huge distribution of mostly irrelevant or impossible worlds can reduce sample efficiency. Measure coverage of control-relevant encounters, not generator variety alone.

**Geometry coverage does not imply observation coverage.** Projection, texture statistics, aliasing, exposure, motion blur, rolling shutter, transparency, and actuator delay can break transfer even when the obstacle shapes are covered. Cheap rendering should preserve useful visual evidence, not merely silhouettes.

**Invariance is conditional on the task, history, and available evidence.** A wall's hue may be irrelevant; a signal's hue may change the reward or permitted action. Lighting that hides an obstacle changes uncertainty and may require slowing down. Those observations need not map to identical states.

**RGB cannot reveal unobservable quantities.** A single monocular frame generally cannot establish metric distance or velocity without additional assumptions. Motion, known ego-action effects, and optional onboard inertial measurements help, but do not remove every ambiguity. Completely hidden actors require risk-aware behavior, not clairvoyance.

**Low resolution creates physical limits.** For a 64-pixel-wide, 90-degree horizontal-FOV camera, focal length is approximately 32 pixels. A 2 cm obstacle at 3 m spans about 0.21 pixels. No representation objective can reliably recover an obstacle absent from the sampled image. Increase resolution, reduce speed, change optics, or narrow the task envelope when necessary.

**Aggressive randomization can destroy useful cues.** Redrawing textures independently every frame corrupts motion evidence. Erasing all texture can remove parallax cues. Randomize persistent surface appearance over episodes; vary exposure and illumination with plausible temporal continuity. Treat sensor noise separately.

**Unlimited simulation is not unlimited learning.** Rendering, encoder training, rollout memory, and experiment iteration will probably dominate simple physics stepping.

**Direct RGB learning does not universally beat privileged learning.** A state teacher can fail because of partial observability or imitation mismatch, but privileged critics, auxiliary labels, and teacher/student policies can also be strong. Keep them as measured baselines rather than assuming either architecture wins.

**Passive video does not identify action consequences by itself.** Without actions or a valid causal model, observing a transition does not tell us what a different control command would have done. Log commands and timestamps whenever collection permits; that is cheap instrumentation, not manual action annotation.

### Research contribution to pursue

The components individually have substantial prior art. The defensible contribution is evidence that a particular combination of **procedural encounter coverage, task-valid appearance interventions, and temporal visual control** improves real closed-loop outcomes per GPU-hour and per hour of deployment data.

A useful result would include:

- Generalization across held-out geometry families, render styles, and motion families, followed by real closed-loop evaluation.
- An ablation showing whether paired interventions or temporal learning improve over ordinary domain randomization.
- A measured data/compute/latency frontier on an 8 GB GPU.
- A reproducible benchmark and failure corpus showing where the approach stops working.

A benchmark-only or negative result is credible if the controls isolate the failure. “A broad generator plus several known losses” is not, by itself, a novel method claim.

## 3. Closest research and what to borrow

| Area / reference | Relevance and limitation |
| :--- | :--- |
| [CAD2RL](https://fsadeghi.github.io/CAD2RL/) | Direct monocular RGB-to-velocity collision avoidance trained in randomized simulation and transferred to real flight. This is close prior art for the basic thesis; it does not establish our proposed dynamic-world and held-out-family claims. |
| [Domain randomization](https://arxiv.org/abs/1703.06907) | Supports varying synthetic appearance rather than requiring photorealism. Coverage and randomization design still matter. |
| [DrQ-v2](https://github.com/facebookresearch/drqv2) and [SVEA](https://arxiv.org/abs/2107.00644) | Strong visual-RL augmentation baselines. SVEA also motivates caution about destabilizing value learning with severe augmentation. |
| [Deep Bisimulation for Control](https://arxiv.org/abs/2006.10742) | A close formal connection to discarding behaviorally irrelevant information without pixel reconstruction. Reward/transition equivalence is more meaningful than arbitrary latent similarity; partial observability complicates direct application. |
| [Self-Predictive Representations](https://arxiv.org/abs/2007.05929) | A small action-conditioned latent predictor and target encoder are a practical starting point for temporal auxiliary learning. |
| [Learning High-Speed Flight in the Wild](https://arxiv.org/abs/2110.05113) | A strong simulation-trained privileged-learning reference for fast flight. Its sensing/training setup is not proof of raw synthetic-RGB transfer. |
| [SIGN](https://arxiv.org/abs/2508.12394) | Recent drone visual RL with image augmentation and future prediction. Its separate depth-based safety module is important when comparing real-world success claims. |
| [Procgen](https://arxiv.org/abs/1912.01588) | Useful procedural generalization methodology. Seed generalization alone is weaker than family and renderer holdouts. |
| [Prioritized Level Replay](https://arxiv.org/abs/2010.03934) and [ACCEL](https://accelagent.github.io/) | Start with replay of informative levels; consider mutation-based curricula after a fixed distribution works. |
| [DINOv2](https://arxiv.org/abs/2304.07193) | A frozen pretrained visual-feature baseline and potential semantic source. Imported pretraining is a large external data prior and must be disclosed. |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) | Relevant separation of action-free video learning and action-conditioned prediction. Its internet-scale pretraining is not a locally affordable training recipe. |
| [Invariant representation limits under domain shift](https://arxiv.org/abs/1901.09453) | Matching sim/real feature marginals is not sufficient for correct target behavior and can be harmful under conditional shift. |

Use these as methodological references, not as a claim that any cited method already solves Triage's complete task. Keep comparisons explicit about sensors, privileged information, external pretraining, safety intervention, and real-data access.

## 4. Current implementation and reusable contracts

Source inventory at the plan update:

- `cuda/src/hover_env.cu` and `cuda/src/hover_env.h`: hover/tracking task around the multirotor physics batch. The current policy observation is 22 float32 state values; actions are four raw rotor commands, transformed around hover thrust.
- `rl/environment.py`: ctypes bridge with borrowed native CUDA tensors, explicit ownership/lifetimes, and stream constraints. This is not an RGB bridge.
- `rl/policy.py`: a feed-forward MLP with Gaussian action and scalar value heads; no active recurrent state.
- `rl/train.py` and `rl/vendor/torch_pufferl.py`: PPO-style training with trajectory reuse/V-trace-related machinery, checkpoints, JSON output, and evaluation. Reuse deliberately rather than describing it as an unmodified textbook PPO implementation.
- `crates/sim-graphics/`: a `wgpu` renderer with box, plane, cylinder, sphere, and custom-mesh support. GPU textures and host readback exist; direct renderer-to-CUDA image sharing is additional work.
- `README.md`: seeded scene generation, realized-scene replay, RGB/depth/instance outputs, and sensor provenance are documented separately from state-policy training.

The critical missing connection is **rendered camera observations consumed by a closed-loop navigation learner**. A GPU renderer and a GPU physics batch do not imply this connection already exists.

Retain the existing SI/frame conventions, quaternion handling, fixed-step timing, action interpretation, final-observation/autoreset semantics, and versioned scene/sensor contracts. Read the relevant implementation and archived sections before changing them. Introduce a separately named visual-navigation task rather than silently changing hover/tracking observations or action meanings.

The physics conventions are SI units, ENU world, FLU body, and scalar-first Hamilton body-to-world quaternions. The existing inspection scenes use a Y-up convention, so implement and test an explicit scene-to-physics/camera transform. Raw linear RGBA8 captures and sRGB PNGs are different encodings: choose one versioned policy preprocessing path and apply the corresponding conversion to real images. These two integration seams belong in E000, not in later appearance tuning.

For the new task, define image layout, color space, camera frame/extrinsics, capture timestamp, observation age, control cadence, collision shape, and reset behavior as versioned contracts. End an episode on collision or task completion; distinguish a time-limit truncation and retain the final pre-reset image/history for bootstrapping.

## 5. Concrete v0 architecture

### Task and control boundary

Use a fixed-altitude, approximately fixed-heading vehicle navigating a short course. Its navigation action is desired body-frame forward/lateral velocity, with forward velocity allowed to reach zero. Start at a maximum forward speed around 0.5 m/s; choose final limits from measured platform response and visible obstacle size.

For the initial simulator task, use a bounded-acceleration velocity-response model, a finite vehicle footprint, and swept collision checks. Label this as a navigation abstraction, not validated full flight dynamics. Later evaluate with the existing multirotor model and an explicit tracking controller.

On hardware, use an established stabilized flight controller with a demonstrated planar velocity-command interface. Stabilization alone does not provide velocity tracking: declare the velocity/position estimator and its sensor or external-localization dependencies. Measure command lag, braking, tracking error, loss-of-command behavior, and operator intervention before navigation trials. The current privileged-state motor policy is not automatically a suitable real controller.

The initial instruction is “advance through the course without contact before the deadline.” The actor receives RGB history, previous commands, frame timing, and the requested motion direction/speed. Optional real-available inertial inputs are an explicitly named variant. Simulator position, obstacle state, depth, and perfect velocity are not actor inputs. Any low-level stabilization/localization sensors must be disclosed even when they are not navigation inputs.

### Model

| Component | Proposed starting point |
| :--- | :--- |
| Camera | 64x64 RGB, approximately 20 Hz; test 96x96 or 128x128 only if observability or measured performance warrants it |
| Spatial encoder | Four small stride-2 convolutions, channels 32/64/64/128; flatten the spatial map and project to 256 features |
| Temporal state | GRU with 256 hidden units, consuming visual features, previous action, timing, and declared onboard inputs |
| Actor / critic | Small MLP heads; Gaussian bounded velocity policy and visual-history value estimate |
| Prediction head | Training-only action-conditioned MLP or small recurrent predictor over the learned state |
| Consistency head | Training-only projection/prediction heads and a stop-gradient target encoder |
| Size target | Roughly 1-2 million trainable parameters for the physical policy, verified from the implemented network |
| Inference target | Measure batch-one p50/p95/p99 latency; initially aim for p95 below 10 ms on the declared target device, not just the training GPU |

Keep spatial location in the encoder output. Immediate global average pooling can discard the left/right information needed for avoidance. Start without a transformer or an image-generating world model.

### Where temporal modeling belongs

- The CNN handles spatial evidence from the current frame.
- The recurrent policy state integrates observations and actions into a compact, approximate belief state. It should retain recently occluded obstacles and motion evidence.
- The auxiliary predictor encourages that state to encode action consequences. Do not run multi-step latent planning in v0.
- A future semantic module can use a longer, slower video window, independently of the physical control update rate.

Use ordered sequences, episode masks, sequence-start state, and burn-in in training. A proposed starting point is 32 learning steps plus 8 burn-in steps. Recompute features during optimization; do not train an evolving encoder from permanently cached latent features. Check that memory actually helps using a frame-stack baseline and ambiguous-motion cases.

For the first recurrent learner, use within-rollout PPO epochs and disable cross-update trajectory reuse until recurrent off-policy correction is explicitly tested. Before E001 main runs, verify ordered minibatches, hidden-state resets, final-history truncation bootstrapping, and stored behavior log-probabilities on a small recurrent fixture. Document any stale sequence-start state and finite-burn-in approximation; eight burn-in frames do not guarantee reconstruction of the full history.

### 8 GB memory and throughput discipline

Start E000's full-loop benchmark at 256 environments. Keep the batch size configurable and use the profile to decide whether to test 512, 1,024, or larger batches. Measure rendering, transfers, collection, and learner updates together rather than extrapolating from state-only environment throughput.

At 256 environments, 64 rollout steps, and 64x64 RGB uint8, one stored image rollout is **192 MiB**. A complete second appearance view doubles that image storage; materializing it all as float32 multiplies each copy's image storage by four. Store uint8 observations and normalize only minibatches. Generate paired views for a subset of sequences if necessary.

Use mixed precision where validated, bounded sequence minibatches, and a rollout buffer rather than a huge GPU image replay buffer for the first on-policy implementation. Target measured total device use below 7 GiB to leave operating headroom; account for `wgpu`/CUDA allocations and context overhead as well as learner allocations.

Profile physics, rendering, host/device transfers, encoding, policy inference, optimization, and artifact writing separately. Report aggregate GPU-hours including failed runs and tuning. Optimize the dominant cost only after the E000 trace exists.

## 6. Procedural geometry and appearance

### Generate encounters, not just shapes

Represent a world as independently versioned factors: geometry family, layout/topology, scale, surface appearance, camera/sensor model, dynamics family, task, and random streams.

Sample control-relevant quantities explicitly:

- Clearance relative to vehicle radius and uncertainty margin.
- Opening width, corridor curvature, obstacle thickness, and surface orientation.
- Occlusion duration, sight distance, and visibility before a decision.
- Relative closing speed, time to contact, required braking, and lateral escape space.
- Clutter density and connectivity of reachable free space.
- Apparent feature size in pixels at the intended speed and distance.

Normalize geometry against vehicle size and dynamics. A geometric path through a gap is insufficient if the vehicle cannot brake or turn into it. An initial braking estimate is `v * total_delay + v^2 / (2 * braking_acceleration)`, plus footprint and uncertainty margins; dynamic encounters need relative-motion reasoning as well.

### Implementation sequence

1. Start with existing boxes, cylinders, planes, and simple openings, using the same transforms and geometric parameters for rendering and collision.
2. Add independent composition families: surface warps, branching graphs, convex/concave assemblies, and varied layouts.
3. Add curved/custom meshes, cellular/Voronoi structures, porous structures, height fields, and bounded CSG/SDF families only when they expand measured encounter coverage.
4. Keep renderer choice separate from generator family. Generate or mesh expensive implicit geometry at reset rather than assuming arbitrary SDF ray marching is cheap for every pixel and environment.

The first renderer is the existing staged `wgpu` path. If profiling identifies it as prohibitive, compare batched rasterization with a bounded CUDA primitive raycaster. A custom general-purpose renderer or external-memory bridge is not a prerequisite for the first transfer experiment.

### Avoid learning the generator

- Give geometry, appearance, dynamics, and camera parameters independent random streams and cross their combinations.
- Use both recognizable simple geometry and irregular composites. Do not equate novelty with utility.
- Hold out whole construction mechanisms, parameter ranges, and combinations, not merely family names that share the same implementation.
- Include a second rendering configuration or backend in evaluation when practical. A new geometry family rendered by the same shader still shares renderer artifacts.
- Maintain a sealed final family suite and a separate development family suite. Once a held-out family guides tuning, it is development data; add a new final holdout.
- Audit accidental shortcuts such as object color predicting motion, fixed texture scale revealing distance, or goal location correlating with a generator seed.

### Appearance interventions

Randomize material hue, texture frequency/orientation, contrast, background, illumination, and sensor effects within declared ranges. Preserve texture attachment and temporal coherence within a history. Include low-texture and low-contrast episodes, but record their observability rather than demanding identical representations under every degradation.

The same physical state is not enough to validate an invariance pair. The task rules, relevant signals, action history, and visibility conditions must also be compatible. Keep task-signaling colors and gestures out of nuisance randomization unless their meaning is transformed consistently.

## 7. Dynamic actors without semantic assets

Generate bodies from primitives and motion controllers independently. Useful physical actor families include:

- Rigid bodies following bounded-acceleration trajectories, with speed changes and pauses.
- Crossing and intercepting trajectories with sampled arrival times.
- Bodies entering from behind occluders, with controlled warning time.
- Pendulums, hinged surfaces, and chains of capsules with constrained joint motion.
- Falling and bouncing bodies under gravity with simple contact models.
- Reactive agents with goals, reaction delays, limited sensing, and varied avoidance/aggressiveness parameters.

Use latent intent variables and correlated stochastic accelerations, not independently sampled positions. A kinematically animated body with bounded velocity is not automatically a physically simulated articulated body; record which model produced it. Validate speed, acceleration, joint limits, and contact behavior appropriate to each family.

Start with one crossing opaque body, then an occluded crossing body. The same current image can require different actions depending on preceding motion: use this as a temporal diagnostic. Later hold out motion mechanisms, such as reactive movement after training on scripted trajectories.

Completely unpredictable or permanently hidden hazards can make an encounter unsolvable. Record unavoidable-collision cases separately and reject them from the learnable curriculum unless the intended behavior is earlier slowing or information gathering. Simulator state may verify feasibility; it must not leak into the actor.

## 8. Objectives and training sequence

Optimize a small set of objectives incrementally. The conceptual total is `L_control + lambda_pair * L_pair + lambda_pred * L_pred + lambda_aux * L_aux`; real-video adaptation is a separately reported stage. Log each loss, its gradient contribution, and its coefficient schedule.

### A. Control objective: always present

Begin with the existing PPO-style infrastructure after adapting the observation/action interface. Use task completion, forward progress, collision cost, a deadline, and modest control smoothness penalties. Specify all coefficients and time units in the versioned task configuration. Check rewards against stop-forever, wall-following, and simulator-exploit policies.

The primary endpoint is success without collision or intervention before the deadline, not training return. Rewards can use simulator state without making it a policy input.

### B. Paired appearance consistency: first research addition

Re-render an identical physical history and identical action history under two independently sampled, task-valid appearance settings. Align projected recurrent representations with a stop-gradient target and add policy-distribution consistency on these valid pairs.

Re-render both histories and reconstruct their recurrent state; comparing one changed frame against an unrelated hidden state is not the intended intervention. Preserve terminal state, camera timing, and actor trajectories. With reactive actors, replay the recorded physical history for appearance pairs rather than regenerating a different interaction.

Use episode-start paired sequences first, initializing each appearance branch's recurrent state independently to zero. For a later mid-episode paired segment, re-encode that appearance's prefix from the last reset, separately for online and target networks, or explicitly specify and test a bounded-history approximation. Do not initialize the alternate appearance with a hidden state computed from the original appearance.

Do not force the entire latent to be a single canonical simulator state. Keep task-sensitive information available to the actor. Use RL plus target-encoder/predictor design and, if needed, variance/covariance regularization to discourage collapse. Measure feature variance and action discrimination; low consistency loss alone proves nothing.

Add explicit counterexamples: recolor an irrelevant wall and expect similar behavior; change a task signal and expect different behavior. For v0, a tiny synthetic stop/go task can diagnose this distinction without claiming real traffic-light competence.

### C. Action-conditioned prediction: second addition

Predict future target-encoder features from recurrent state and recorded actions at short horizons, initially 1, 2, and 4 control steps. Add longer horizons only if they improve closed-loop outcomes. This is a training auxiliary, not an image reconstruction task.

A deterministic one-step predictor can learn persistence or average incompatible futures. Stochastic actors and occlusion require uncertainty or multiple hypotheses if prediction is expanded. Compare against action-free prediction and action-shuffled controls to test whether action consequences were learned.

### D. Privileged auxiliary supervision and diagnostics

First fit frozen-representation probes for collision risk, relative motion, traversability, and short-horizon action-conditioned outcomes using simulator labels. Probe results diagnose missing information, but do not establish real transfer.

If an auxiliary head materially improves control, train it jointly as a named ablation. Simulator depth/flow labels are cheap and acceptable; their accuracy is secondary to control. A privileged critic is another separate ablation. Do not impose privileged actor imitation as the default representation-learning pipeline.

For counterfactual diagnostics, branch selected simulator states over a small action set and estimate short-horizon risk/return under a declared continuation policy. These are measured continuation values, not oracle optimal Q-values. Test whether useful action rankings survive appearance changes.

### E. Small real-video adaptation: only after zero-shot evaluation

Use the deployment camera and representative motion for real clips. Compare zero adaptation with 15 minutes, 1 hour, and 4 hours of unique real footage; extend to 8 hours only if the curve justifies it. Use nested subsets and separate sites/sessions for development and final evaluation.

Start with conservative temporal/masked feature prediction or frozen-feature distillation. Replay synthetic RL/consistency data during adaptation and constrain policy drift on a fixed simulation anchor set. Update only a small adapter or selected encoder layers initially. Re-evaluate closed-loop behavior after every adaptation stage.

Action-free video losses cannot substitute for action-conditioned supervision. If real commands and timing are available, use them in a separately named interaction-data condition. Do not invent action labels from an inverse model and count them as ground truth.

### F. Domain invariance: deferred experiment

Do not begin with an adversarial sim-versus-real classifier. Marginal feature alignment can merge states requiring different actions. If attempted, align conditional on comparable task/motion contexts and test action discrimination and real control explicitly. Chance-level domain classification is not a success metric.

## 9. Real data and semantic edge cases

### What data is unavoidable?

There is no universal minimum number of real-video hours. Zero real training images can work for a bounded task, as CAD2RL illustrates; that does not imply zero real engineering knowledge or zero validation.

At minimum, obtain task-specific evidence about camera observations, command response, timing, and closed-loop behavior. Semantic conventions require an information source: pretrained data, explicit task rules, demonstrations, or a small targeted annotation set. Geometry alone cannot determine an arbitrary gesture's intended meaning.

Keep separate accounting for:

1. Deployment-domain unlabeled video, measured as unique duration and coverage, not repeated training exposure.
2. Robot interaction data with commands/telemetry, plus any labeled demonstrations.
3. Manual setup/calibration effort and task-specific labels.
4. External pretrained weights and their disclosed training-data provenance.
5. Real development trials and final evaluation trials, including interventions and failures.

A small amount of basic camera work is worth doing: identify image orientation, approximate FOV, camera-to-body mounting, exposure/frame-rate behavior, and timestamp/command delay. Use a full calibration procedure only when measurements show those approximations are inadequate. Camera mismatch cannot be wished away by domain randomization.

### Slower semantic conditioning

Do not add a semantic model to E000 or E001. Later, keep a separate access path to RGB or higher-resolution crops; a heavily compressed physical latent may already have discarded a sign's content.

Run a frozen small pretrained image/video encoder at a lower rate, provisionally 1-5 Hz, and learn a small context adapter. Condition the fast policy through a compact context vector or FiLM-style modulation. Supply context age, confidence, and expiry. Train with missing, stale, and incorrect context.

Begin with one explicit convention, such as a demonstrated stop or directional gesture. Define its meaning and provide a small source of task-specific supervision or verified pretrained interpretation. A generic visual embedding does not automatically encode the correct instruction.

Keep fast collision avoidance responsive between semantic updates. Slow processing is suitable only when the event permits the delay; urgent signals require earlier visibility, a fast trigger, or lower operating speed. A confidence estimate is not a safety guarantee. Report semantic success and intervention separately from physical avoidance.

## 10. Evaluation and strongest baselines

### Evaluation matrix

Vary one axis at a time, then evaluate combinations:

| Axis | Development / final tests |
| :--- | :--- |
| Geometry | Unseen seeds; unseen parameter ranges; held-out construction families; mixed-family layouts |
| Appearance | Held-out palettes/textures/lighting; low contrast; sensor corruption; different rendering configuration |
| Dynamics | Different speed/acceleration ranges; occluded crossing; held-out motion controllers; reactive actors |
| Camera / actuation | Measured FOV/timing variation; frame drops; blur; command delay and braking variation |
| Domain | Simulation; real development course; sealed real sites/layouts/sessions |

Match physical task difficulty where possible when comparing sim and real. A success difference across unrelated courses conflates visual transfer with geometry and dynamics shifts. Use simple measured course dimensions for a controlled subset; no scanned digital twin is required.

### Primary measurements

- Real task success without contact, deadline violation, or safety intervention, with numerator and denominator.
- Collision and intervention rates, completion time, distance traveled, and progress at termination. Stopping indefinitely is a timeout, not success.
- Sim-to-real success drop in percentage points on matched task strata, with per-stratum results.
- Environment transitions and GPU-hours to a preregistered success threshold; include tuning and failed-run totals separately.
- Unique real-video hours and number of interaction trajectories used at each stage.
- Batch-one inference p50/p95/p99, full capture-to-command age, jitter, and missed control deadlines on the declared deployment setup.
- Peak total device memory, learner-allocated/reserved memory, host RAM, and renderer/copy/learner timings.
- Performance per held-out geometry, appearance, and dynamics family, not just a favorable aggregate.

Log simulated time, control steps, physics substeps, rendered images, and optimization steps separately. “Steps per second” must state which step is counted.

### Evidence that the representation transfers

Closed-loop real improvement at controlled data/compute is the main evidence. Support it with:

- Appearance intervention tests on the same physical histories, measuring policy divergence and action-risk rankings.
- Relevant-change tests where collision geometry, motion, or a task signal changes and the policy must respond differently.
- Frozen-encoder comparisons with identical small readouts and equal adaptation data/budget, distinguishing representation quality from policy retraining.
- Temporal ambiguity tests and removal/shuffling of history or action inputs.
- A decomposition of visual, dynamics, and task-distribution gaps using the evaluation matrix.

Latent similarity, a domain classifier, image reconstruction quality, FID/FVD, depth accuracy, and segmentation accuracy are diagnostics only unless an experiment shows they predict control outcomes. A policy that fails all difficult cases consistently is invariant but useless.

### Baseline ladder

| Baseline | Purpose / fairness requirement |
| :--- | :--- |
| Stop, straight-line, and simple reactive controller | Detect reward/evaluation loopholes and establish trivial behavior |
| Same visual architecture with narrow randomization | Establish the gain from broad randomization |
| Same visual architecture with broad randomization and ordinary augmentations | Main baseline for paired-render and prediction objectives |
| Four-frame CNN versus CNN+GRU | Measure the value of memory at similar capacity and budget |
| DrQ-v2 visual control | Strong off-policy/data-efficiency comparison; account for replay memory and give it an explicit tuning budget |
| Frozen small DINOv2 features plus the same temporal/action head | Test whether existing real-world visual pretraining is a better use of local compute; disclose imported data and extra resolution/cost |
| Simulator depth plus small policy; real measured/estimated depth counterpart | Separate control/dynamics difficulty from RGB transfer. Disclose additional sensors/models and measure their latency |
| Privileged-state policy or critic; privileged teacher/student | Diagnostic upper/reference baselines, not guaranteed upper bounds for every training setup |

The full ladder is not a demand to implement all baselines before the first experiment. Run the controlled same-architecture comparisons first, then the strongest affordable external-feature and off-policy alternatives. Match either transitions or GPU-hours and report both; one matching criterion cannot equalize every resource.

### Statistical protocol

Use at least three independent training seeds for claims beyond a smoke test. Evaluate the same immutable scenario manifest across methods. Randomize/counterbalance real trial order to reduce battery, lighting, and operator effects.

Pilot real testing may use approximately 30 trials per candidate to expose gross failures, not to establish small gains. For finalists, plan approximately 100-200 trials per method across multiple layouts/sessions and training seeds, then choose sample counts from the precision required. Report per-seed results and uncertainty across layout/session clusters; repeated frames and correlated attempts are not independent samples.

Freeze checkpoint selection on development evaluation. Count safety interventions as policy failures, while recording infrastructure failures separately under predeclared rules. An external shield's success must not be attributed to the RGB policy. Record the actual executed commands as well as requested commands.

## 11. First slice: E000, reproducible RGB control loop

**Status: implementation and profiling complete; acceptance evidence partial. The remaining E000 gates are checkpoint reload and the visual-dependence diagnostic.**

The deliverable is a small working experiment, not a new general simulator or a library of shapes.

### Fixed scope

- A separately named `visual_nav_v0` task with planar motion, velocity-response dynamics, finite footprint, and swept collision checks.
- A short, straight course with randomized offset box/cylinder obstacles and openings, plus empty-course and unavoidable-collision fixtures for diagnostics.
- 64x64 RGB at a declared camera rate, with an initial benchmark batch of 256 environments and the existing renderer/readback path.
- Two continuous velocity commands. Version the speed/acceleration limits, task deadline, success predicate, collision margin, and reward in one resolved task configuration.
- A four-frame CNN policy and visual critic for integration; no privileged obstacle/depth inputs. Implement a scripted controller for fixed-action replay and comparison.
- One PPO-style collection/update cycle, checkpoint reload/evaluation, per-episode output, and end-to-end profile.
- One mandatory paired-render fixture: render the same short fixed-action trajectory with two appearances and verify unchanged physics/outcomes. Large-scale paired rendering and extra representation losses are outside E000.

Start with fixed geometry fixtures, then small seeded variations. The task geometry should be simple enough to inspect without a 3D asset pipeline. Use simulator truth for collision and progress measurements, not actor features.

### Work order and code seams

1. **Run protocol first.** Define resolved configuration, run manifest, named seed streams, terminal episode records, and immutable run-directory creation. Wrap the new task; reuse existing checkpoint/logging code where appropriate.
2. **Navigation task.** Add a new native task/adapter under `cuda/src/` and `rl/`, without changing the hover/tracking ABI. Specify reset, truncation, final observation, and command timing.
3. **RGB bridge.** Add a Rust renderer adapter around `crates/sim-graphics/`, with persistent render targets and bounded staging buffers. Document ownership, image conversion, and synchronization; do not assume a `wgpu` texture is a CUDA tensor.
4. **Closed-loop learner.** Connect image batches, previous actions, and episode masks to a small visual policy. Render the required terminal frame before autoreset; do not bootstrap from a new episode's image.
5. **Replay and measurement.** Save realized scenes, camera/task configuration, fixed action traces, selected frames, and episode outcomes. Produce a profile and a ledger entry from the resulting run.

For control observations, backpressure or an explicitly simulated stale-frame policy is required when rendering is late. Dropping visualization frames is acceptable; silently dropping or relabeling control frames changes the task.

### Acceptance evidence

- Two fresh-process fixed-action runs with the same realized scenes and stream identities reproduce resets, actor motion, termination causes, and task metrics within declared tolerances.
- Replaying a realized scene does not invoke the generator. Replaying a different appearance changes RGB without changing the physical trace or task outcome under fixed actions.
- Multiple environments can reset independently without cross-contaminating images, history, RNG streams, or terminal observations.
- A visual rollout completes, a learning update produces finite losses and nonzero encoder gradients, and a saved checkpoint can be reloaded for evaluation.
- A short fixed-fixture learning diagnostic shows the RGB path affects behavior; include an image-shuffle or blank-image diagnostic where geometry varies. A gradient alone is not evidence that vision is used.
- Inspect actual rendered images and a replay visually. Confirm obstacle/collider alignment and camera orientation on known fixtures.
- Record measured peak VRAM, stage timings, copying costs, and total artifact size at 256 environments. Demonstrate that the full harness fits the 8 GB device budget; record an explicit failed gate if it does not.
- Validate manifest/schema references and every required artifact checksum; preserve the failed-run manifest if any gate fails.

Use a pilot compute cap of 2 GPU-hours for E000 smoke/profiling runs. If full-loop cost makes even a tiny learning diagnostic impractical, identify the bottleneck and make one focused rendering/batching change before E001. Do not spend weeks optimizing an unmeasured interoperability design.

## 12. First falsification experiment: E001

**Question:** Does the procedural RGB approach transfer on an easy, observable physical navigation task, and does paired appearance training improve over ordinary randomization?

Keep static opaque obstacles, fixed altitude/heading, slow motion, and one declared camera/control setup. Use simple real box/panel/cylinder arrangements with measured coarse dimensions. A wheeled or camera-rig diagnostic may isolate visual errors, but is not real-flight evidence.

Before hardware trials, demonstrate the velocity-tracking and failsafe checks in section 5 and select a feasible depth/reactive reference, with its sensors, implementation effort, and trial budget included in the E001 specification. Without a functioning reference and measured command response, a poor real RGB result is diagnostically inconclusive, not an isolated visual-transfer failure. Complete the recurrent-learner fixture checks from section 5 before comparing the three arms.

### Arms

- A: CNN+GRU control with narrow appearance randomization and a fixed ordinary-augmentation recipe.
- B: the identical architecture, learner, and ordinary augmentations with broad appearance randomization.
- C: B plus valid paired-history representation/policy consistency.

The primary comparison is equal training GPU-hours, not simultaneously equal transitions. Hold geometry sampling, control interface, architecture capacity, ordinary augmentations, and checkpoint-selection protocol fixed. Three training seeds per arm. Choose loss weights only on development data, with an explicit small tuning budget. Count C's extra rendering and representation updates inside its budget.

Start with a fixed 8 GPU-hour training budget per run: nine main runs, at most 72 GPU-hours before separately recorded evaluation and tuning. Do not stop a main run early just because it reaches the success threshold; report time-to-threshold as a secondary endpoint. Save predetermined transition milestones for a secondary equal-transition comparison over the range all arms reach. Record exact completed transitions, budget overshoot at the last update, and failed/censored runs. If a run reaches the cap without learning the simulation task, it is a training/budget failure, not evidence about sim-to-real transfer.

Evaluate held-out simulated geometry and appearance, then zero-shot real development trials before any real-video adaptation. Use a separate sealed final real suite for the later claim. The first iteration should fit roughly within the first two weeks after E000 and hardware access; do not delay real evaluation until the grammar is elaborate.

### Decision gates

Suggested pilot criteria to freeze in the E001 specification before results are viewed:

- Require at least 90% held-out simulation success on the easy course before diagnosing a real gap.
- If all RGB variants achieve that simulation criterion but less than 50% real success, while an appropriate depth/reactive reference succeeds on at least 90% of comparable trials and timing/control checks pass, reject the current zero-shot visual recipe for this scope.
- If broad randomization transfers well but paired consistency does not improve results at equal resources, keep the simpler method and reject the extra objective, not the generator concept.
- If the best RGB variant achieves around 80% or better real pilot success with a modest matched sim-to-real gap, continue to a larger held-out evaluation and dynamic obstacles. This is a development gate, not a statistical proof or deployment threshold.
- If simulator and reference policies both fail on hardware, fix task observability, control mismatch, or evaluation setup before drawing representation conclusions.

No finite small experiment can falsify every possible procedural-learning method, especially given existing positive prior art. E001 can quickly falsify **this low-data, low-compute recipe on its declared easy task**. Failure there is a strong reason not to spend months on broader geometry and semantics yet.

## 13. Curriculum after a fixed-distribution result

Start E003 only after E001/E002 have stable evaluation. Use a simple mixture, provisionally 50% fresh uniform worlds, 30% replayed high-learning-potential worlds, and 20% local mutations. These weights are hyperparameters, not theory.

Store full realized scenarios and failure traces. Prioritize learnable failures, estimated regret, or improvement on revisit rather than maximum failure alone. Normalize priorities across families and deduplicate nearly identical encounters. Keep a minimum sampling floor for every training family.

Validate reachability, observability, and dynamic feasibility. Cap the contribution of unavoidable or numerically pathological scenarios. Use an independent development suite to decide whether mining helps; never mine the sealed final suite. Compare against uniform sampling at equal total simulation and learner compute, including the cost of searching for hard worlds.

## 14. Result recording

### Versioned specification before each experiment

Assign an experiment ID (`E000`, `E001`, ...), specification version, owner, and date. Commit the question, baseline/variant matrix, task/sensor/action contracts, split manifests, primary endpoint, seed list, transition/GPU-hour caps, checkpoint selection rule, tuning allowance, confidence procedure, and stop/continue criteria before the main run.

Use the selected E000 definition above as the initial specification; implement machine-readable validation in E000. A change after seeing outcomes is a new specification version, with the reason recorded. Keep pilot results distinct from confirmatory results.

### Artifact layout

Use an explicit artifact root, initially ignored `target/experiments/`, with a unique directory per run. Do not overwrite a run or use “latest” as its identity. A run ID should include experiment/spec version, seed, and a unique suffix; identity is not the same as reproducibility.

Each run contains:

| Artifact | Required contents |
| :--- | :--- |
| `manifest.json` | Schema version, run/experiment/spec IDs, source revision, start/end/status, hardware/software fingerprint, parent/pretraining references, split/config/checkpoint hashes, data provenance, and reproducibility tier |
| `config.json` | Fully resolved configuration including defaults, time units, all loss/reward coefficients, schedules, precision settings, task/camera/geometry/dynamics versions, and control interface |
| `seeds.json` | Root seed, named streams, RNG algorithm/version, stable identity derivation, and actual train/development/final manifest references |
| `metrics.jsonl` | Append-only training/resource records with control transitions, simulated seconds, images rendered, learner updates, elapsed time, losses, and measured memory |
| `episodes.jsonl` | One terminal record per evaluated episode: scenario/family/split IDs, success, collision, intervention, termination reason, duration, progress, requested/executed-command trace reference, and sensor mode |
| `evaluation.json` | Aggregates derived from episode records, denominators, per-family/per-seed values, uncertainty method, checkpoint choice, and evaluation code revision |
| `checkpoints/` | Policy and declared training state, format version, architecture/config references, and explicit restoration capability |
| `scenarios/` and `traces/` | Realized selected/failing worlds, actor trajectories/state, actions, timestamps, selected RGB frames/video, and replay metadata |
| `profile/` | Stage timings, warmup/sample protocol, device-memory samples, profiler traces where requested, and inference measurements |
| `artifacts.sha256` | Relative paths and checksums of finalized artifacts; exclude this checksum file itself |
| `report.md` | Short question/result/limitations/decision summary with references to the artifacts, not a replacement for raw records |

For every claim-bearing real trial, retain the exact policy-input stream or losslessly reconstructible inputs, not just selected video clips. Include capture/delivery timestamps, preprocessing version, dropped/stale-frame decisions, recurrent reset markers, requested and executed commands, and intervention events. Link this audit trace from the episode record and durable ledger entry. Auditability does not imply numerically identical policy-in-the-loop replay. Selected clips are sufficient only for illustrative training diagnostics.

During a run, mark status as running. On completion or failure, finalize metadata atomically and checksum the artifact set. If interrupted before finalization, retain the partial run and mark it interrupted during recovery. Do not silently treat missing episodes as successful or remove failed seeds.

Capture only a whitelist of environment metadata. Avoid credentials, complete environment dumps, and unnecessary identifying information in real video. Record collection permissions and access restrictions for any human-containing footage.

### Git and durable storage

Commit experiment specifications, small aggregate reports, schema versions, and [ledger entries](experiments/RESULTS.md). Keep videos, checkpoints, profiles, and bulk episode data out of Git. Before a result is accepted, copy its finalized artifact directory to a configured durable location and record both location and manifest digest in the ledger. An ignored `target/` directory alone is not durable evidence.

The ledger records status (`planned`, `running`, `completed`, `failed`, `inconclusive`, or `superseded`), source revision, run IDs, artifact digest/location, metrics with denominators, resource/data cost, and the next decision. Use `not measured` rather than zero for missing metrics. Include a separate decision history so a changed hypothesis is traceable.

## 15. Reproducibility contract

### Three distinct claims

1. **Scenario reproducibility:** reconstruct the same realized scene, camera/task configuration, motion/controller parameters, and random samples from archived data. Pin serialization and generator versions.
2. **Numerical replay:** under declared hardware/software and fixed actions, reproduce state/observation/outcome traces within specified tolerances. Exact pixels across GPUs or graphics backends are not promised.
3. **Statistical reproducibility:** across independent training seeds, reproduce the reported performance distribution and resource costs within uncertainty. A single deterministic seed is not this claim.

State which tier each run supports. Never equate a stored seed with a reproducible learning experiment.

### Seeds, splits, and scenarios

Derive independent streams for geometry, layout, appearance, camera noise, actor intent/motion, vehicle parameters, policy initialization, minibatches, and evaluation. Use a pinned algorithm and canonical identity encoding, with identities such as `(root_seed, split_id, scenario_id, episode_id, stream_id, sample_index)`. Do not use wall-clock scheduling or an implementation-dependent hash to determine samples.

Environment reset order must not perturb another environment's stream. If the implementation promises batching-independent scenario replay, use stable scenario identity rather than batch slot alone and test that promise.

Save realized scene data in addition to seeds, because generator code evolves. Save actor/controller state and random-stream position for dynamic replay; a seed and starting mesh are insufficient. Appearance-pair records must reference the same physical-history digest.

Version immutable split manifests. Keep separate training, development, held-out family, and real final-test manifests. Separate real sites/sessions as well as frames; adjacent clips from the same traversal must not straddle adaptation and final evaluation. Record every use of test data for debugging and reclassify contaminated sets as development.

### Environment and executable provenance

Record the Git revision and dirty status; claim-bearing runs use a clean committed source tree. Exploratory dirty runs require a saved patch and hashes of relevant untracked inputs and are labeled accordingly. Do not capture secrets while recording source provenance.

Record `flake.lock`, `Cargo.lock`, installed Python package versions/lock information, vendored learner commit, compiled native-library hashes, compiler/build flags, GPU model/VRAM, driver, CUDA, PyTorch, graphics backend/adapter, shader/config versions, and precision/determinism settings. Nix pins user-space dependencies, not the physical GPU or host driver.

Store exact launch arguments and resolved paths or content-addressed references. Include imported checkpoint/model identifiers, hashes, licenses, preprocessing, and external data assumptions. On the same environment, request deterministic kernels where supported and document exceptions and their cost.

### Checkpoints and replay

The current training checkpoint includes policy/optimizer/RNG/configuration metadata but does not restore a complete mid-episode simulator trajectory. E000 promises saved-policy evaluation and fixed-action replay, not exact mid-training continuation.

Exact training resume is a later, explicit capability requiring simulator state, episode counters, all RNG states/counters, actor state, recurrent state/history, observation timing/queues, rollout/replay contents, optimizer/scheduler/precision-scaler state, and curriculum state at a synchronized boundary. Test uninterrupted versus resumed execution before claiming it.

For the first slice, restart training from the committed specification when necessary and evaluate saved weights from fresh named scenarios. For replay, distinguish fixed-action physical replay from policy-in-the-loop replay, which can diverge after small numerical observation differences.

### Measurement and verification

- Use simulation timestamps for the environment and monotonic host timestamps for elapsed time; state how real camera/controller clocks are related.
- Warm up kernels, synchronize at measurement boundaries, and report batch size and capture/inference mode. Do not time asynchronous enqueue calls as completed GPU work.
- Record PyTorch peak allocated/reserved memory and total device usage covering renderer and CUDA contexts. Measure on an otherwise idle GPU or disclose competing load.
- Specify numeric tolerances and success margins before comparisons. Borderline collisions or threshold changes require inspection, not silently relaxed tolerances.
- Test behavioral contracts: seed isolation, geometry/collision agreement, image orientation/timing, independent resets, final frames, checksum validation, and replay. Avoid tests that merely freeze prose or incidental formatting.
- Run relevant checks from [contributing.md](contributing.md), plus a GPU visual smoke/replay inspection for rendered-observation changes. Publish commands, outcomes, and any unavailable checks in the experiment report.

## 16. Evidence-gated 90-day schedule

Assume access to a controllable camera-equipped platform and a safe test space early in the schedule. If that access is unavailable, label the result simulation-only and do not substitute a video benchmark for closed-loop transfer.

| Days | Work | Required evidence / gate |
| :--- | :--- | :--- |
| 1-5 | E000 harness, run records, scene/action replay, tiny visual learner, profile; identify real camera/controller | Reproducible RGB loop and measured bottlenecks; no claimed transfer |
| 6-20 | E001 three-arm static experiment; recurrent baseline; held-out geometry/appearance; first real development trials | A measured zero-shot gap with control/observability references; stop or simplify on the E001 failure gate |
| 21-35 | E002 frame-stack/recurrent comparison, action-conditioned prediction, one crossing and one occluded actor family | Dynamic closed-loop gain over the static/broad-randomization baseline, with memory/action ablations |
| 36-50 | E003 uniform versus prioritized/mutated worlds; extra independent geometry and motion families | Gains on untouched development families at equal total compute, not just mined worlds |
| 51-65 | E004 real-video dose curve and frozen-pretrained-feature baseline; selective encoder adaptation | Real success versus unique video hours; no catastrophic loss on synthetic anchor tasks |
| 66-75 | E005 one semantic convention only if physical transfer works; otherwise spend this interval on diagnosed physical failures | Benefit from explicit semantic information with stale/missing-context tests and measured latency |
| 76-90 | Freeze methods; strongest affordable baseline comparisons; sealed sim/real evaluation; artifact replay audit | Multi-seed results, failures, data/compute/latency table, and a continue/pivot decision |

Treat later rows as conditional, not commitments to add complexity regardless of evidence. Budget roughly 300-500 total local GPU-hours for the 90-day study, including failed runs and tuning, then revise after E000 measurements. Do not launch a full Cartesian product of losses, generators, seeds, and video amounts. Screen on development data, then replicate a small finalist set.

## 17. Expected first failures and responses

| Likely failure | Diagnostic and next action |
| :--- | :--- |
| Renderer/readback dominates | E000 full-loop profile; batch views/reuse targets before considering a bounded custom renderer or native interop |
| Policy ignores RGB | Vary obstacle layout, shuffle/blank images, inspect encoder gradients and action changes; remove position/seed shortcuts |
| High sim success, poor real control | Check camera/timing/action response, compare matched courses and a depth/reactive reference, then narrow appearance/sensor failures |
| Thin/dark obstacles invisible at 64x64 | Measure apparent size/contrast and stopping distance; change resolution, speed, or task envelope |
| Appearance loss erases useful information | Relevant-change counterexamples, action-risk discrimination, lower weighting or restrict valid intervention pairs |
| Prediction learns persistence or averages danger | Compare action-free/shuffled-action baselines; use longer informative encounters or a small uncertainty-aware head |
| Real-video adaptation hurts control | Freeze more of the encoder, replay synthetic anchors, reduce update budget, or keep the zero-shot policy |
| Curriculum concentrates on impossible cases | Feasibility checks, family floors, deduplication, and a uniform-sampling control |
| Semantic context arrives too late or is wrong | Shorter validity windows, context-drop training, lower speed, or a faster task-specific path; record intervention |

The immediate deliverable is E000 and its evidence bundle. A positive E001 result earns work on dynamics and broader generator families. A negative result should identify which assumption failed before Triage grows another subsystem.
