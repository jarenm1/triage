# Triage experiment ledger

Use [PLAN.md](../PLAN.md) for experiment definitions, acceptance gates, artifact schemas, and reproducibility contracts. This ledger records plans, evidence, and decisions; it does not replace per-run artifacts.

Updated: 2026-09-12. No experiments under the new visual-transfer protocol have been run. Historical state-policy and sensor results are preserved in [the archived engineering plan](../docs/archive/plan-2026-09-12.md).

## Experiment register

| ID / spec | Status | Question / scope | Runs and evidence | Decision |
| :--- | :--- | :--- | :--- | :--- |
| E000 / 2 | planned, selected next | Can we collect, learn from, and replay a small closed-loop RGB navigation task with complete provenance? | Not run; metrics not measured | Implement and benchmark the 256-environment RGB harness and result/replay contract before expanding geometry |
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
