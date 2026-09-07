#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* All arrays are CUDA device storage on the configured device, owned by env.
 * Calls are single-host-caller and ordered across supplied CUDA streams. Inputs
 * must be ready on that stream and remain alive until its work completes.
 * Buffer consumers must use the producing stream or establish their own event
 * ordering. Reading a buffer does not synchronize. Normal step/reset do not
 * allocate or synchronize the host. Creation and destruction may wait.
 */
enum triage_hover_field {
  TRIAGE_HOVER_OBSERVATIONS = 0,       /* float [N,22] */
  TRIAGE_HOVER_REWARDS = 1,            /* float [N] */
  TRIAGE_HOVER_TERMINATED = 2,         /* float [N] */
  TRIAGE_HOVER_TRUNCATED = 3,          /* float [N] */
  TRIAGE_HOVER_FINAL_OBSERVATIONS = 4, /* float [N,22], pre-autoreset */
  TRIAGE_HOVER_COMPLETED_RETURNS = 5,  /* float [N], zero unless done */
  TRIAGE_HOVER_COMPLETED_LENGTHS = 6,  /* float [N], zero unless done */
  TRIAGE_HOVER_RESET_STATUS = 7,       /* uint8 [N]: 0 unselected, 1 applied,
                                         2 invalid state, 3 invalid params */
  TRIAGE_HOVER_EPISODE_COUNTS = 8,     /* uint64 [N], completed episodes */
  TRIAGE_HOVER_CURRENT_RETURNS = 9,    /* float [N], zero after autoreset */
  TRIAGE_HOVER_CURRENT_LENGTHS = 10,   /* float [N], zero after autoreset */
  /* Tracking-only fields; requesting them from a hover env is an error.
   * Final observations and final_targets describe the just-finished step,
   * before both command replacement and physical autoreset. Targets and
   * command_index describe the next observation, including after autoreset.
   * All tracking buffers are read-only to consumers.
   */
  TRIAGE_TRACKING_TARGETS = 11,          /* float [N,3], next target xyz */
  TRIAGE_TRACKING_FINAL_TARGETS = 12,    /* float [N,3], preceding target xyz */
  TRIAGE_TRACKING_COMMAND_DURATION = 13, /* float [N], preceding duration */
  TRIAGE_TRACKING_COMMAND_ELAPSED = 14,  /* float [N], preceding age, 1-based */
  TRIAGE_TRACKING_COMMAND_FINISHED =
      15, /* float [N], scheduled end, no failure */
  TRIAGE_TRACKING_COMMAND_SETTLED = 16, /* float [N], finished and last 50 steps
                                          distance <= .2 and speed <= .2 */
  TRIAGE_TRACKING_COMMAND_INDEX = 17, /* uint64 [N], next command, zero-based */
  TRIAGE_TRACKING_ACTION_SATURATION =
      18,                           /* float [N], fraction |tanh(raw)|>=.99 */
  TRIAGE_TRACKING_COMMAND_PLAN = 19 /* float [N,101,4], xyz and duration;
                                     replaced on each episode reset */
};

/* n and max_steps must be positive; max_steps <= 2^24 (exact float lengths).
 * stream is cudaStream_t cast to void*. A null stream means the default stream.
 * Explicit reset restarts the seed's episode sequence and clears task
 * statistics. Actions are contiguous float32 [N,4] raw policy outputs; the
 * caller guarantees their allocation extent. Commands are hover +
 * .15*tanh(action), clamped [0,1]. Nonfinite actions produce a failed
 * transition and a valid same-step reset.
 */
void *triage_hover_create(int device, size_t n, uint64_t seed, int max_steps,
                          void *stream);
/* Tracking uses the same reset/step/buffer/destroy/error functions below.
 * max_steps must be in [1,2000]; schedule 0 is 75% uniform integer [200,500],
 * 25% uniform integer [20,100], and schedule 1 is fixed 500. The complete plan
 * is generated on-device before initial observations, keyed by seed, row,
 * episode and command independently of physical state/reset sampling.
 * Targets are uniform in x/y [-1,1], z [.75,1.75], rejection sampled to lie
 * [.5,1.5] from the preceding command (initially (0,0,1)).
 * Observation positions are world position minus current target; all other
 * fields and motor mapping match hover. Safety is target-independent:
 * x/y [-3,3], z [.05,3], tilt <= .8, finite state/actions.
 * Reward and final buffers use the old command; replacement never resets
 * physical state. Failure overrides scheduled completion; timeout counts as
 * command completion only when its scheduled boundary coincides.
 * Explicit reset clears preceding-step metrics to zero, pairs final_targets
 * with initial final_observations, and restarts episode/command indices at 0.
 * Evaluators needing the terminated episode's future commands must clone its
 * initial plan before stepping; autoreset replaces it in-place.
 * Playback-only schedule 2 is long-flight-v1: max_steps in [1,6000], targets
 * (10,0,2),(10,10,2),(-10,10,2),(-10,-10,2),(10,-10,2),(0,0,2),
 * each held 1000 controls. Its arena is x/y [-20,20], z [.05,5];
 * tilt/nonfinite checks are unchanged.
 * This out-of-training-distribution scenario is not a tracking acceptance
 * suite.
 */
void *triage_tracking_create(int device, size_t n, uint64_t seed, int max_steps,
                             int schedule, void *stream);
int triage_hover_reset(void *env, uint64_t seed, void *stream);
int triage_hover_step(void *env, const float *actions, void *stream);
void *triage_hover_buffer(void *env, int field);
int triage_hover_destroy(void *env);

/* Optional visualization staging. Single host caller, env lifetime rules apply.
 * Configure once outside rollout (1..64 unique host IDs, 2..16 slots).
 * submit returns 1 if queued, 0 if full (drop), -1 on error. It never waits.
 * poll returns the oldest completed frame, 0 if not ready, -1 on error.
 * Output capacity must equal selected count; poll copies only ready pinned
 * memory and releases that slot. Timings are GPU milliseconds for selected
 * export+packing and D2H respectively, excluding transfer stream queue delay.
 * Disable/destruction may wait; never destroy env while a caller uses staging.
 */
typedef struct triage_snapshot_vehicle {
  uint32_t environment_id;
  uint64_t episode_id;
  float position_w[3];
  float attitude_wb[4];
  float target_w[3];
} triage_snapshot_vehicle;
int triage_snapshot_configure(void *env, const uint32_t *ids, size_t count,
                              size_t slots);
int triage_snapshot_submit(void *env, uint64_t step, void *stream);
int triage_snapshot_poll(void *env, triage_snapshot_vehicle *output,
                         size_t count, uint64_t *step, float *gather_ms,
                         float *copy_ms);
int triage_snapshot_disable(void *env);
/* Last error on this host thread, valid until the next ABI call on the thread.
 * Failures return NULL/-1; destroy(NULL) succeeds. No C++ exception crosses
 * ABI.
 */
const char *triage_hover_error(void);

#ifdef __cplusplus
}
#endif
