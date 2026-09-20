# Visual encoder — decision log

Every design decision for the visual-inertial encoder, including ones that
never became code. Measured results are in `experiments/RESULTS.md` (D002–D005);
this file records *why*, including rejected options.

## Architecture

| Decision | Choice | Why |
|---|---|---|
| Latent resolution | 16×12×64 at 128×96 input | 10×6 loses small obstacles (wires, gap edges) — one cell = 32×32px. 16×12 keeps sub-cell features encodable in channels. |
| Stem | stride-8 conv, DW-separable blocks | Compress early (DeepSeek-style); convs fuse well in TensorRT; batch-1 latency is memory-bound, not FLOP-bound. |
| Temporal state | ConvGRU on spatial latent | Persistent spatial state > pooled vector for navigation; GRU's update gate is the right inductive bias at small data scale. |
| Ego-motion | learned warp conditioned on Δpose + validity mask | Camera translates with drone → parallax is the obstacle-distance signal. Learned flow handles translation without depth. Mask distinguishes "no info" from "empty space". |
| Motion conditioning | FiLM bias on GRU gates + `dt` input | Δpose always conditions fusion even when warp is off (clean ablation). `dt` makes recurrence rate-independent for the async estimator. |
| Diff channel | `|norm(F_t) − norm(H')|` | Cheap explicit surprise signal; normalize first or it fires on lighting. |
| Future prediction | predict next-frame `F` (stop-grad), not `H` | Predicting the raw recurrent state collapses; predicting encoded features can't. Doubles as eval metric. |
| Global transformer | rejected after ablation (variant H) | Second temporal integrator over pooled `g` adds latency (+1.2ms), loses occlusion retention (2.05 vs 1.23). A second spatial GRU on a coarser map is the better slow-timescale design if ever needed. |
| RoPE | rejected | 60–192 tokens: learned 2D embeds equivalent, more TRT-friendly. |
| MoE / sparse attention / MLA | rejected | All solve scale problems (huge params, long context) this model doesn't have. At batch-1 on embedded, routing overhead is pure latency. |
| API | `propagate(state, motion)` / `update(frame, state, motion)` | The async estimator split: IMU-rate warp-only step vs camera-rate fuse step. First-class state, no hidden globals. |

## Training objectives

| Decision | Choice | Why |
|---|---|---|
| Probes | gap_x, wall_d, vx, vz + 32-col free-space | Privileged sim labels are free; probes force nav-relevant info into the latent. Occupancy is analytic ray-cast — no renderer changes. |
| Appearance consistency | cosine loss on paired renders (variant I) | Same scene, two seeded colors → same latent. Trains "sim/real alignment" proxy. Works (paired_cos 0.999) but alone collapses occlusion retention. |
| Teacher | LingBot-Vision-Large (ViT-L/16) | Dense-spatial-perception pretraining > DINOv2 semantics for geometry. DINOv3 preferred but license-gated; LingBot is Apache-2.0 and open. |
| Distillation | project F_t → teacher patch space, cosine | Variant J (DINOv2) hurt everything; variant L (LingBot) helped — teacher choice is the whole game. |
| Domain adversarial | weak linear disc, coef 1.0, on F_t | Conv disc memorizes 1200 real frames and wins (domain_acc 0.99). Weak disc reaches equilibrium (~0.8). DANN needs 10k+ real frames to be meaningful. |
| Domain disc as metric | preferred over as loss (planned variant M) | Distill on sim+real frames is a positive alignment signal ("become this") vs adversarial negative ("don't look like that"). Positive optimizes cleanly; disc becomes a measurement, not a loss. |

## Data

| Decision | Choice | Why |
|---|---|---|
| Camera | onboard FPV (eye at vehicle +0.4y, −z) | Was chase cam — wrong viewpoint for FPV domain alignment, rendered the drone's own body. Fixed; all pre-FPV results are chase-view. |
| Collection | scripted reference policy + noise + 20% random envs | Coverage without a trained policy; privileged labels free. |
| Real footage | yt-dlp FPV clips → ffmpeg 10fps → 128×96 | 1200 frames = 120s. Too few — disc memorizes. Need 10k+ and *diverse* locations/cameras, uncut for temporal structure. |
| Real data limits | no pose/action labels | Real footage can train alignment + temporal statistics but NOT action-conditioned future prediction — that stays sim-only, which is why latent alignment matters. |

## Deployment / RL integration

| Decision | Choice | Why |
|---|---|---|
| FP16 on Orin | yes, ~2-3× over FP32 | Ampere tensor cores + TRT fusion. INT8 later. |
| RL training cost | freeze encoder, batch inference | 10k envs × 300k steps = 3×10⁹ forwards; frozen encoder = no backward graph, half memory, big batches. Frozen-first is also the cleaner ablation ("does the representation transfer?"). |
| Frame stack | replaced, not added | Recurrent state carries history; encoder consumes latest frame only. |

## Known gaps / next steps

- Retrain G/L on FPV data (all current numbers are chase-view)
- Variant M: distill on sim+real, disc eval-only
- More diverse real footage (10k+ frames, varied locations)
- Policy integration: frozen L encoder + PPO vs RecurrentRGBPolicy baseline — the "is it useful" test
- TensorRT/Orin benchmark

## Submission-repo contents (if extracted)

Code: `rl/visual_encoder.py`, `rl/venc_train.py`, `rl/venc_domain.py`,
`rl/venc_eval.py`, `rl/venc_collect.py`, `rl/test_visual_encoder.py`,
`rl/vendor/lingbot_vision/` (Apache-2.0), `notebooks/train_venc_colab.ipynb`.
Docs: this file, `docs/visual-encoder.md`, RESULTS.md D002–D005.
Data: `target/venc-data/*` (or regenerate via `venc_collect` — needs the
Rust sim `librgb_env.so`, which is the one hard dependency to port).
