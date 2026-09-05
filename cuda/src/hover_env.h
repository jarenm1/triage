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
  TRIAGE_HOVER_CURRENT_LENGTHS = 10    /* float [N], zero after autoreset */
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
int triage_hover_reset(void *env, uint64_t seed, void *stream);
int triage_hover_step(void *env, const float *actions, void *stream);
void *triage_hover_buffer(void *env, int field);
int triage_hover_destroy(void *env);
/* Last error on this host thread, valid until the next ABI call on the thread.
 * Failures return NULL/-1; destroy(NULL) succeeds. No C++ exception crosses
 * ABI.
 */
const char *triage_hover_error(void);

#ifdef __cplusplus
}
#endif
