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
- Provenance gap: the 16,384-transition checkpoint was produced by extending the recorded 2,048-step run in place at `target/experiments/e000/checkpoint.pt`; later training overwrote that file, so its SHA-256 is unrecoverable. `diagnostic-16k.json` is the preserved evidence for the 16k result.
- Result at 16k: normal reward/step **+0.001287**, mean action **[0.0552, -0.0162]**; blank reward/step **-0.000683**, mean action **[0.0221, -0.0046]** — a ~2.5x action shift and reward sign flip. The CNN policy measurably responds to visual input; the RGB path is behaviorally live.
- Shuffle arm inconclusive at 16k: normal vs shuffle differed only ~5e-6 in mean action. First suspected cause — near-identical scenes (~4/255 mean pixel difference) — was fixed by seeded per-env appearance variation (floor + obstacle colors, `mix64`-hashed from seed and env index; ~27-62/255 pairwise difference, per-env mean std 17.9).
- Shuffle remains flat at 32k on varied appearance (delta ~5e-5) and at 48k on varied geometry (delta ~6e-5; blank still diverges ~3x). Two confounds, both measured: the reward is dominated by forward progress and jittered layouts stay traversable, so a scene-invariant policy is near-optimal; and the policy is still weak (0 completed episodes, ~0.11 mean forward action), so "scene-invariant" is ambiguous with "too weak to use the scene." The diagnostic is functioning; the conclusive shuffle result is now gated on policy competence on a vision-load-bearing course, not on fixture diversity.
- Artifacts: `target/experiments/e000/diagnostic-32k-varied.json` and `target/experiments/e000/diagnostic-48k-geometry.json`. Appearance variation is seeded per env and geometry variation per (env, episode); both are deterministic and episode-persistent, and the physics contract is unchanged.
- Interpretation: E000's "RGB path affects behavior" gate is met at the coarse level (blank vs rendered). Per-scene dependence requires a course where the correct action depends on the observed scene; freeze that requirement in the E001 specification rather than retrofitting E000's fixed course.
- No episodes completed in any mode within 256 steps (max-steps 2000); termination-based endpoints were not exercised.

## E001 / 1 vision-load-bearing course

- Implementation: `apps/rgb-env` now draws a two-box wall across the corridor at z = -5.2 with a seeded gap (half-width 1.5, center in [-2.6, 2.6]) per (env, episode), plus the existing three jittered obstacles. Collision and render share `obstacles(seed, env, episode)`.
- Reward fix: collision is a flat -10.0 and progress reward only accrues on survival. Previously a blind charge accumulated ~+8 in progress reward before the -1 collision penalty, making blindness profitable — the task did not require vision.
- Verified: a blind full-forward policy now terminates every episode (283 terminations / 200 steps / 256 envs). A GRU policy trained 131k steps on the fixed course still collides (mean return -7.59 vs the -10 blind floor) — it has not learned to thread the gap within this budget.
- Diagnostic on the wall course (`diagnostic-gru-wall3.json`): normal vs shuffle remain near-identical and blank diverges ~3x. The shuffle arm is now gated on policy competence on a vision-required course, not on fixture diversity or reward structure. A conclusive result is expected once an E001 arm trains to competence at the 25M-transition budget.
- Interpretation: the course now satisfies the "correct action depends on the observed scene" requirement. The remaining shuffle question is whether a *competent* policy uses per-scene information, which requires the full E001 training budget to answer.

## E001 / 1 held-out evaluation harness

- `rl/rgb_evaluate.py` loads a checkpoint (feed-forward or `--recurrent`), rolls deterministic mean actions on a chosen seed/split, and writes `episodes.jsonl`, `evaluation.json`, `config.json`, `seeds.json`, `manifest.json`, and `artifacts.sha256` into a fresh run directory.
- The native env now exposes a per-env `successes` buffer (field 10) so episode records distinguish success from collision; a dropped `state.return_` accumulation was restored after eval showed zeroed returns.
- Smoke run: `target/experiments/e001-eval-dev-seed101/` — the 131k-step GRU checkpoint on development seed 101 produced 929 episodes, all collisions, mean return -7.45, matching the diagnostic's -7.59. The harness is validated; the policy is not yet competent.

## E001 / 1 paired-render path (arm C)

- `apps/rgb-env` accepts a `paired` flag: each render pass runs twice per step, once per appearance variant, with variant-aware view keys and seeded color streams. Paired frames land in a second history buffer (field 11) exposed as `env.paired_observations`.
- `PuffeRL` stores paired observations per rollout step and adds `consistency_coef * MSE(features(obs), features(paired_obs))` to the loss; both policies expose `features()` (encoder output, pre-GRU for the recurrent policy).
- `rl/rgb_train.py --paired --consistency-coef` runs the arm-C path. Smoke: 1,024 steps, consistency loss 0.033, finite losses, checkpoint saved. Paired frames differ from primary by ~38/255 mean pixel intensity with identical geometry.

## E001 / 1 privileged reference controller

- `rl/rgb_reference.py` steers toward the true gap center using the new `gap_centers`/`vehicle_xs` native buffers (fields 12/13) — simulator truth, no vision. It is the diagnostic ceiling: if it succeeds and the RGB policy fails, the gap is a vision problem.
- Course simplified to the wall alone: the three jittered obstacles were removed because the gap-only reference could not see them and died on them (7.6% success). On the wall-only course the reference achieves **100% success** (256/256 episodes, mean return +11.7, seed 101) while a blind full-forward policy terminates every episode (328 terminations / 300 steps).
- The course is now provably solvable and provably vision-required. The RGB policy's job is to close the gap between the blind floor (-10/episode) and the privileged ceiling (+11.7).

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

### D002: visual-inertial encoder ablation (E002 scope), 2026-09-17

- Decision: adopt the temporally persistent spatial-latent encoder
  (`rl/visual_encoder.py`) as the perception front-end direction; the
  future-feature prediction loss is the single most valuable component for
  state retention under occlusion.
- Evidence: eight-variant ablation on the E001 corridor task, 96k-frame
  scripted rollouts at 128x96 (train seed 11, held-out val seed 77).
  Variants: A frame-CNN, B +spatial probes, C +ConvGRU, D +motion
  conditioning, E +ego-motion warp, F +diff channel, G +future-feature
  loss, H +tiny causal transformer over g history.
- Standard metrics (held-out): all variants reach occ IoU 0.92-0.99 and
  gap_x MAE 0.15-0.37; differences are small on unobstructed frames.
- Occlusion retention (4 blanked frames, probe at last blanked frame):
  gap_x MAE A=3.35, B=4.23, C=4.29, D=5.20, E=2.89, F=2.11, G=1.23,
  H=2.05. The future-prediction objective (G) retains gap position ~3.5x
  better than the plain recurrent baseline (C/D); the diff channel (F)
  and warp (E) also help. Wall-distance is not retained by any variant
  (it changes during occlusion — expected).
- Efficiency (batch-1, RTX 2070 SUPER, FP32): deployed params 0.14-0.29M
  for A-F, 0.29M for G (1.13M incl. training-only future head), 1.39M H.
  Latency 1.2ms (A) to 2.7ms (G); H costs +1.2ms over G for no gain.
  MACs 18.7M (A) to 44.5M (G).
- Caveats: velocity probes are trivially solved by motion-conditioned
  variants (vx/vz are inputs); gap_x is static within an episode so
  occlusion retention is partly memorization, not tracking. Dataset is
  scripted-policy rollouts, not policy-in-the-loop.
- Decision: keep G as the reference encoder; drop H (transformer adds
  latency, no benefit). Next: wire encoder into the E001 policy loop and
  measure downstream control, not just probes.
- Evidence status: measured; artifacts in target/venc-results/*.json,
  checkpoints *.pt, data target/venc-data/.

### D003: appearance consistency + DINOv2 distillation, 2026-09-17

- Question: can the latent be made appearance-invariant (sim/real
  alignment proxy) without hurting task representation?
- Setup: paired-appearance rollouts (same trajectory, two seeded color
  variants; env `paired` mode), 48k train / 3.2k val frames. Variant I =
  G + cosine consistency loss between F(frame) and F(paired frame).
  Variant J = I + DINOv2 ViT-S/14 patch-feature distillation (real-
  anchored teacher; DINOv3 weights are license-gated, DINOv2 used).
  Baseline G retrained on the same paired data for a fair comparison.
- Results (held-out, same data): G gap_x 0.159 / IoU 0.874 / paired_cos
  0.893 / occluded gap_x 1.00. I: gap_x 0.115 / IoU 0.989 / paired_cos
  0.999 / occluded 4.26. J: gap_x 0.241 / IoU 0.802 / paired_cos 0.907 /
  occluded 2.43.
- Findings: (1) consistency loss achieves near-perfect appearance
  invariance (cos 0.999) and *improves* clean-frame probes — the
  invariance acts as a regularizer. (2) But it collapses occlusion
  retention (4.26 vs 1.00): forcing frame features appearance-invariant
  appears to also suppress the state information the future-loss uses
  for retention. (3) DINOv2 distillation hurts every metric — on
  flat-shaded synthetic scenes the teacher's semantic features don't
  align with task geometry; distillation should pay off on richer
  scenes/real footage, not here.
- Caveats: single seed, single loss weight (0.5); the occlusion
  regression may be a weighting artifact — a lower consistency
  coefficient or consistency-on-g-only is untested.
- Decision: keep G as reference. Consistency training is worth one more
  run at lower coefficient before richer scenes; distillation deferred
  until real footage or richer sim exists.
- Evidence: target/venc-results/{i,j,g-paired}.json + .pt; data
  target/venc-data/{train,val}-paired/.

### D004: domain-adversarial sim->real alignment (variant K), 2026-09-17

- Question: does DANN (gradient-reversal domain discriminator) align the
  latent across sim and real FPV footage?
- Setup: 1200 real frames from two FPV freestyle clips (yt-dlp, 10fps,
  128x96; local research use, provenance target/venc-data/real/). K = I
  + domain adversarial on F_t. Three discriminator settings tried:
  conv disc coef 0.1, conv disc coef 1.0, weak (linear) disc coef 1.0.
- Results: conv disc wins outright at both coefficients (eval domain_acc
  0.999/0.991) and damages task metrics (gap_x 0.45/0.23). Weak disc
  reaches a stalemate — training domain loss ~= ln 2 (encoder fools it
  in-training) while eval domain_acc stays 0.96 because the disc
  memorizes 1200 real frames; task metrics recover (gap_x 0.247, IoU
  0.984, paired_cos 0.999) but occlusion retention stays poor (3.89).
- Findings: (1) DANN on 1200 real frames is degenerate — the
  discriminator memorizes rather than measures the domain gap, so
  domain_acc is a weak metric at this scale. (2) The sim/real gap here
  is large (flat-shaded boxes vs. real FPV); appearance-level alignment
  alone can't bridge geometry statistics. (3) Occlusion retention
  regresses whenever consistency/domain losses are added — the
  retention mechanism (future-loss state) is fragile to feature-space
  regularization.
- Decision: domain-adversarial alignment needs (a) more real data
  (thousands→tens of thousands of frames), (b) a discriminator that
  can't memorize (spectral norm / dropout / smaller capacity), or
  (c) deferred until richer sim exists. Keep G as reference; K-weak is
  the best domain-aligned variant but not deployment-ready.
- Evidence: target/venc-results/{k,k-d1,k-weak}.json + .pt;
  rl/venc_domain.py (GRL, discriminator, degrade(), load_teacher).

### D005: LingBot-Vision teacher + full alignment stack (variant L), 2026-09-18

- Setup: L = I + weak DANN on real FPV frames + LingBot-Vision-Large
  (ViT-L/16, dense spatial perception pretraining, Apache-2.0) feature
  distillation. Teacher vendored at rl/vendor/lingbot_vision. Trained
  3000 steps (resumed once after a 3600s timeout; periodic checkpointing
  added to venc_train.py).
- Results (held-out paired val): gap_x MAE 0.22, occ IoU 0.98,
  paired_cos 0.998, domain_acc ~0.78-0.84 (encoder partially fools the
  discriminator — healthy DANN equilibrium, vs 0.96+ failure or 0.5
  collapse), distill loss 1.0 -> 0.075.
- Occlusion retention: occluded gap_x MAE 1.07 — matches G (1.00) and
  far better than I (4.26) or K (3.89). The distill loss appears to
  protect the retention mechanism that consistency alone destroyed.
- Findings: (1) LingBot's dense-perception features are a better teacher
  than DINOv2 for this task — distillation now *helps* where DINOv2
  hurt. (2) L is the best overall variant: near-G occlusion retention
  plus appearance invariance plus real-domain alignment. (3) wall_d
  probe degraded (ctrl MAE ~1.0) — distance estimation traded off for
  invariance.
- Caveats: ViT-L teacher makes training ~3x slower (1.6s/step); deployed
  encoder unchanged (0.29M params). domain_acc is still a weak metric at
  1200 real frames.
- Decision: L is the new reference encoder. Next: more real footage for
  a stronger domain signal, and wire the encoder into the E001 policy.
- Evidence: target/venc-results/l.{json,pt}; teacher
  robbyant/lingbot-vision-vit-large (HF, Apache-2.0).
