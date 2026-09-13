# Triage experiment ledger

Use [PLAN.md](../PLAN.md) for experiment definitions, acceptance gates, artifact schemas, and reproducibility contracts. This ledger records plans, evidence, and decisions; it does not replace per-run artifacts.

Updated: 2026-09-13. E000 implementation, fixed-action replay, artifact recording, visual inspection, full-loop profiling, checkpoint reload/resume, and the visual-dependence diagnostic are complete. Blank-image arm passes; the shuffle arm shows the policy is scene-invariant because the task's forward-progress reward does not require per-scene discrimination — a vision-load-bearing course is an E001 spec decision.

## Experiment register

| ID / spec | Status | Question / scope | Runs and evidence | Decision |
| :--- | :--- | :--- | :--- | :--- |
| E000 / 2 | partial | Can we collect, learn from, and replay a small closed-loop RGB navigation task with complete provenance? | Renderer benchmark, corrected 256-environment CNN smoke, paired fixed-action replay, artifact records, visual inspection, VRAM profile, checkpoint reload/resume, and visual-dependence diagnostic complete | Blank arm passes; shuffle needs a course where correct action depends on the scene — freeze in E001 spec |
| E001 / 1 | specified, not run | Does broad procedural RGB randomization transfer, and does paired-history consistency improve it? | Spec frozen below; metrics not measured | Run three arms at equal GPU-hours; evaluate held-out geometry/appearance before real trials |
| E002 / draft | planned, conditional | Do recurrence and action-conditioned prediction improve dynamic/occluded-obstacle control? | Not run | Proceed after static transfer is diagnosed |
| E003 / draft | planned, conditional | Does prioritized/mutated world sampling outperform uniform sampling at equal total compute? | Not run | Keep final generator families out of mining |
| E004 / draft | planned, conditional | How does real-video adaptation compare with zero-shot and frozen pretrained features? | Not run | Account separately for unique footage and imported pretraining |
| E005 / draft | planned, conditional | Can one semantic convention improve behavior within a bounded latency budget? | Not run | Attempt only after physical transfer works |

## E000 / 2 preregistration

- Source starting point: `839125f`; record the actual implementation commit in each run manifest.
- Specification: [PLAN.md, section 11](../PLAN.md#11-first-slice-e000-reproducible-rgb-control-loop).
- Owner: Triage maintainer; implementation next.
- Task: new `visual_nav_v0`, planar velocity response, finite footprint, swept collision, short opaque-obstacle course.
- Inputs: 64x64 RGB history, previous actions, declared timing/instruction fields. Actor has no simulator obstacle/depth/position features.
- Initial integration model: four-frame CNN with visual critic; scripted fixed-action controller for replay.
- Initial rendered benchmark batch: 256 environments, using the existing staged renderer path. Keep batch size configurable; report full-loop throughput, per-stage timings, and peak total VRAM.
- Root seeds for integration runs: 11 and 29. Repeat each fixed-action run in a fresh process. Root seeds are not substitutes for the named streams and saved realized scenes.
- Fixtures: empty course; offset box; cylinder; opening; terminal collision; timeout; simultaneous and independent environment resets; one mandatory two-appearance fixed-action replay. Include a separate deliberately unavoidable encounter for diagnostic labeling.
- Primary endpoints: correct RGB/action/reset timing, reproducible fixed-action outcomes, checkpoint evaluation, actual visual dependence, and a complete checksummed evidence bundle.
- Resource limit: 2 aggregate GPU-hours for smoke/profiling runs before review. No target training success or throughput is claimed before implementation.
- Configuration freeze: commit numeric dynamics, camera, reward, success, and tolerance definitions before recorded acceptance runs. During integration these may change, but bump the specification when its declared behavior changes.
- Acceptance: satisfy the section 11 evidence checklist, report stage timings and peak total device memory, and inspect rendered frames/replay.
- Interpretation: passing E000 establishes integration and measurement capability, not sim-to-real transfer.

## E001 / 1 preregistration

- Specification date: 2026-09-13. Source revision at freeze: `3994ef4f`. Changes after viewing outcomes require a new spec version with the reason recorded.
- Task: `visual_nav_v0` on `apps/rgb-env` — planar velocity control through a three-obstacle course, 64x64 RGB four-frame history plus previous action as the only actor inputs. No privileged obstacle/depth/position features.
- Geometry: per-(env, episode) seeded jitter of the base three-obstacle layout (position ±0.5/±0.4 m, scale 0.85-1.15x). Training uses root seeds 11, 29, 47. Held-out geometry evaluation uses seed 101 (development) and seed 202 (final, sealed until arms are compared).
- Appearance: seeded per-env floor and obstacle colors (current `mix64` scheme). Arm A (narrow) fixes appearance to the seed-11 distribution; arm B (broad) samples the full seeded range; arm C is B plus paired-history consistency (same trajectory rendered under two appearances).
- Architecture: `RecurrentRGBPolicy` (CNN+GRU, 556,741 parameters), identical across arms. Ordinary augmentations fixed to none beyond the seeded variation; the learner is the vendored PuffeRL PPO configuration in `rl/rgb_train.py`.
- Budget: 25M transitions per run (~3.5 GPU-hours at the measured ~2,000 env-frames/s), 3 arms x 3 seeds = 9 runs, ~32 GPU-hours total training cap. Evaluation and the small tuning allowance (one sweep over the arm-C consistency weight on development seed 101 only) are recorded separately.
- Checkpoint selection: final checkpoint at budget end; milestone saves at 25/50/75% of transitions for the secondary equal-transition comparison. No early stopping at the success threshold.
- Primary endpoint: held-out-geometry success rate (development seed 101) per arm at equal GPU-hours. Secondary: time-to-threshold, equal-transition comparison, collision rate.
- Reference: a privileged-state reactive controller (steers toward the corridor gap using simulator obstacle truth) as the simulation reference; a depth-camera reactive baseline is the designated real-trial reference, implementation deferred to hardware bring-up. A poor RGB result without a working reference is diagnostically inconclusive per PLAN.md section 12.
- Decision gates (from PLAN.md section 12, frozen): high held-out sim success required before diagnosing a real gap; if all arms pass sim but fail real while the reference succeeds, reject the zero-shot recipe; if B transfers and C does not improve at equal resources, keep B and reject the extra objective; strong real pilot success with a modest matched gap continues to larger held-out evaluation and dynamic obstacles.
- Known open item carried from E000: the shuffle diagnostic needs a course where the correct action depends on the observed scene. The E001 held-out-geometry evaluation provides this; the conclusive shuffle result is expected on the E001 fixture, not retrofitted to E000.

## E000 / 2 observed benchmark result

**Status: partial, integration, replay profile, checkpoint reload, and visual-dependence diagnostic complete; shuffle arm needs a vision-load-bearing course.**

- Implementation revision: `2dbb72a6d3e2ba044b66e1c7937270bf91c0d7a9`.
- Claim run: `target/experiments/e000-rgb-benchmark-256-claim/`; manifest SHA-256 `62e27f1caa2b594bfce1b46d2f9758b02158f8bf4b32807e9a2c1ee25c1c3630`; artifact-set checksum `424824499b926258c21c5ad1235609ede0540ce66de1715c3d62b64a5de74665`.
- Configuration: 256 independent RGB views, 64x64, 10 measured steps after 2 warmup steps, staged `wgpu` readback, Vulkan on an NVIDIA GeForce RTX 2070 SUPER.
- Mean measured step: **49.09 ms**, composed of **35.85 ms** submission/render work and **13.22 ms** readback. Throughput: **5,214 rendered environment-frames/s**, or **21.36 million pixels/s**.
- Each step read back 4,194,304 RGBA bytes. The first three replay checksums matched exactly between fresh processes: `8db6a2f833b04e53`, `2e3904cb83916842`, `a5039d508ff6ab70`.
- 100-step profile: `target/experiments/e000-rgb-benchmark-256-profile/`; mean step **50.56 ms**, **5,064 environment-frames/s**; observed peak GPU memory from 20 ms `nvidia-smi` polling: **956 MiB**. Profile artifact-set checksum: `f779c9c59f215bf9551570f216fb4ec82eb9f9e69120b15d2ce9ee6def306170`.
- Exploratory scale probes before the claim run measured approximately **2,982 frames/s at 512 views** and **1,642 frames/s at 1,024 views**. They are not claim-bearing because their manifests predate the implementation commit.
- Interpretation: the staged renderer path fits comfortably within 8 GB at 256 views and is fast enough to justify continuing. The renderer benchmark and policy smoke now exercise native actions, RGB observations, PufferLib rollout storage, and one CNN PPO update. This is integration evidence, not sim-to-real transfer.

## E000 / 2 RGB/PufferLib integration smoke

- Native bridge: `apps/rgb-env` renders 256 64x64 views, stages actions host-side, converts RGBA readbacks to four-frame RGB histories, and exposes rewards, termination flags, terminal observations, and episode metrics through `rl/rgb_environment.py`.
- Integration revision: `594b00c`.
- Smoke command: `nix develop --command cuda/build/rl-venv/bin/python rl/rgb_train.py --num-envs 256 --horizon 4 --steps 1024 --device 0 --library target/release/librgb_env.so`.
- Result: corrected frame-major history conversion, one 1,024-transition rollout and PPO update, finite losses, checksum `6115641783539149663`, 456,901 policy parameters, and PyTorch peak allocated memory of 742,236,160 bytes.
- The earlier `6c6634a` smoke is superseded because its Python view treated native frame-major history as interleaved channels. It is not used as evidence.
- The corrected reset, terminal, and autoreset samples were visually inspected; env slots 0 and 1 show distinct appearance colors.

## E000 / 2 checkpoint reload and training resume

- Verification revision: `c575479a` (`rl/rgb_checkpoint.py`, `rl/rgb_train.py --resume`).
- Commands: `rl/rgb_train.py --num-envs 256 --horizon 4 --steps 1024 --checkpoint target/experiments/e000/checkpoint.pt`, then `--steps 2048 --resume target/experiments/e000/checkpoint.pt` (same seed 11, device 0, `librgb_env.so`).
- Result: the resumed run restored policy, optimizer, `epoch`/`global_step` counters, and RNG state; it continued from step 1,024 to 2,048 with finite losses and re-saved the checkpoint. The saved checkpoint records `epoch=2`, `global_step=2048`, SHA-256 `b55ebc42120080e4b09de8dc607aebeac227ab35d2640972cc84b3711b3267c4` (3,671,989 bytes).
- A separate eval-only load (`load_checkpoint` without a learner) restored the policy, passed observation/action-shape checks, and produced finite action distributions and values on fresh observations.
- Limitation per PLAN.md section "Checkpoints and replay": resume restores learner state, not a mid-episode simulator trajectory; the environment restarts from its seed. Exact mid-training continuation remains a later capability.

## E000 / 2 visual-dependence diagnostic

- Verification revision: `c575479a` (`rl/rgb_diagnostic.py`); policy checkpoint at 16,384 transitions, seed 11, 256 environments, 256 deterministic (`distribution.mean`) steps per mode.
- Artifacts: `target/experiments/e000/diagnostic.json` (2,048-step policy) and `target/experiments/e000/diagnostic-16k.json` (16,384-step policy).
- Shuffle arm inconclusive at 16k: normal vs shuffle differed only ~5e-6 in mean action. First suspected cause — near-identical scenes (~4/255 mean pixel difference) — was fixed by seeded per-env appearance variation (floor + obstacle colors, `mix64`-hashed from seed and env index; ~27-62/255 pairwise difference, per-env mean std 17.9).
- Shuffle remains flat at 32k on varied appearance (delta ~5e-5) and at 48k on varied geometry (delta ~6e-5; blank still diverges ~3x). Two confounds, both measured: the reward is dominated by forward progress and jittered layouts stay traversable, so a scene-invariant policy is near-optimal; and the policy is still weak (0 completed episodes, ~0.11 mean forward action), so "scene-invariant" is ambiguous with "too weak to use the scene." The diagnostic is functioning; the conclusive shuffle result is now gated on policy competence on a vision-load-bearing course, not on fixture diversity.
- Artifacts: `target/experiments/e000/diagnostic-32k-varied.json` and `target/experiments/e000/diagnostic-48k-geometry.json`. Appearance and geometry variation are seeded per (env, episode) and episode-persistent; the physics contract is unchanged.
- Interpretation: E000's "RGB path affects behavior" gate is met at the coarse level (blank vs rendered). Per-scene dependence requires a course where the correct action depends on the observed scene; freeze that requirement in the E001 specification rather than retrofitting E000's fixed course.
- No episodes completed in any mode within 256 steps (max-steps 2000); termination-based endpoints were not exercised.

## E000 / 2 fixed-action replay and full-loop profile

- Clean source revision: `594b00c`; both fresh-process runs used seed 11, 256 environments, 64x64 RGB, 64 measured steps, two warmup steps, and `max_steps=64`.
- Runs: `target/experiments/e000-rgb-replay-256-seed11-run3/` artifact set `9dcc821edbf74b400fbfd99479fd48a3f2653bc92dbaca972ffd1d9f745a39bd`; run4 artifact set `988b4e5e86e8fe42b98c70aa99d4c4b2f850479c0bca4feab1bb2f3905ca4e83`.
- Both runs produced reset/final checksum `092121544e67af15`, trace SHA-256 `0e2b49e626379eb7e529295e2381658853881671581522428b7731be0a617d14`, and identical `actions.jsonl` and `episodes.jsonl` hashes. Each produced 256 truncated episodes at the declared deadline.
- Mean full-loop step time was **127.08 ms** in run3 and **129.83 ms** in run4, for **2,014** and **1,972 environment-frames/s**. Mean staged components were action host staging **0.05-0.13 ms**, native dynamics **0.003 ms**, render/readback **47.68-48.64 ms**, native history handling **2.29-2.35 ms**, and host-to-learner observation copy/conversion **76.91-78.72 ms**.
- Peak sampled total device memory was **1,131 MiB** and **1,125 MiB** against an 8,192 MiB device. The full staged harness fits the device budget.
- Interpretation: fixed actions, terminal observations, autoreset, episode metrics, and RGB delivery are numerically reproducible across fresh processes. The current bottleneck is host observation conversion/copying, not simulation dynamics. The full loop is about 2,000 environment-frames/s versus 5,214 renderer-only frames/s, which identifies zero-copy or fused history conversion as the next performance seam without blocking E000 integration.

## Per-experiment result entry

Copy this structure into a dated subsection when a run completes, fails, or becomes inconclusive. Use `not measured` for unavailable numbers; retain failed seeds and partial artifacts.

| Field | Record |
| :--- | :--- |
| Experiment / specification | ID, version, committed specification reference |
| Status / dates | planned, running, completed, failed, inconclusive, or superseded; start/end |
| Hypothesis / primary endpoint | Question and criterion fixed before the main run |
| Implementation | Clean Git revision, executable/native-library hashes, or explicitly labeled exploratory patch |
| Methods / selection | Baselines, variants, tuning allowance, development-based checkpoint selection |
| Run IDs / seeds | Every main, pilot, failed, and tuning run; label their roles |
| Splits / data | Manifest hashes; unique real-video hours, interaction trajectories, task labels, imported pretraining |
| Artifacts | Durable location, finalized manifest SHA-256, local working location, and complete policy-input/command audit traces for claim-bearing real trials |
| Outcome counts | Successes/episodes, collisions, interventions, timeouts, infrastructure failures, with denominators |
| Generalization | Per-family/per-session/per-seed results; matched sim-to-real percentage-point drop |
| Uncertainty | Interval/aggregation method and sampling unit; identify correlated trials |
| Efficiency | Transitions, rendered frames, simulated time, GPU-hours including tuning/failures, and wall time |
| Runtime | Device identity, batch-one latency percentiles, capture-to-command age, peak total VRAM and learner memory |
| Verification | Commands/checks, replay tolerance/results, checksum audit, visual inspection, unavailable checks |
| Limitations / deviations | Changes from the preregistration, contaminated holdouts, missing evidence |
| Decision | Continue, simplify, reject a component, fix a diagnosed failure, or stop; next experiment |

## Decision history

### D000: select the visual-transfer direction, 2026-09-12

- Decision: train the navigation actor from synthetic RGB rather than require a canonical privileged simulator latent.
- Rationale: test control-relevant representation learning directly while keeping privileged supervision and teachers as explicit baselines.
- First work: E000 reproducible RGB loop, then E001 static transfer; postpone large procedural grammars, curricula, and semantic modules.
- Evidence status: research/engineering judgment informed by the linked literature and repository inventory. No new policy training, transfer, or performance measurement was performed for this planning change.

### D001: benchmark E000 at 256 environments, 2026-09-12

- Decision: change the initial full-loop benchmark batch from 16 to 256 environments, at the user's request.
- Specification: E000 / 2 supersedes E000 / 1 before any experiment runs. Use measured rendering, transfer, and learner costs to decide further scaling.
- Evidence status: specification change only; no benchmark has been run.
