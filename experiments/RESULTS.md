# Triage experiment ledger

Use [PLAN.md](../PLAN.md) for experiment definitions, acceptance gates, artifact schemas, and reproducibility contracts. This ledger records plans, evidence, and decisions; it does not replace per-run artifacts.

Updated: 2026-09-12. No claim-bearing visual-transfer experiments have been run. E000 now has a native staged RGB/PufferLib integration smoke; full replay and provenance evidence remain unfinished.

## Experiment register

| ID / spec | Status | Question / scope | Runs and evidence | Decision |
| :--- | :--- | :--- | :--- | :--- |
| E000 / 2 | partial | Can we collect, learn from, and replay a small closed-loop RGB navigation task with complete provenance? | Renderer benchmark and a 256-environment staged RGB/PufferLib PPO smoke passed; full replay/provenance evidence not complete | Finish fixed-action replay, artifact records, and visual inspection before E001 |
| E001 / draft | planned, conditional on E000 | Does broad procedural RGB randomization transfer, and does paired-history consistency improve it? | Not run; metrics not measured | Freeze exact task/splits/seeds/budgets after E000 profiling, before main runs |
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

## E000 / 2 observed benchmark result

**Status: partial, renderer benchmark and RGB policy smoke complete; full E000 evidence not complete.**

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
- Smoke command: `nix develop --command cuda/build/rl-venv/bin/python rl/rgb_train.py --num-envs 256 --horizon 4 --steps 1024 --device 0 --library target/release/librgb_env.so`.
- Result: one 1,024-transition rollout and PPO update completed with finite losses, checksum `6143592202583394605`, 456,901 policy parameters, and PyTorch peak allocated memory of 742,236,160 bytes.
- Fresh-process seed-29 repeats produced the same checksum: `7993221089584785578` on both runs.
- This smoke does not claim fixed-action replay, saved checkpoints, full-loop stage timings, total device memory, or visual dependence.

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
