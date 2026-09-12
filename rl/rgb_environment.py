"""Staged host RGB adapter for the E000 native environment.

The renderer and the Python learner do not share device memory yet. Actions are
staged CUDA -> host, and RGB observations are staged host -> learner device.
This is deliberate E000 instrumentation rather than a zero-copy contract.
"""

import ctypes
import time
from pathlib import Path

import numpy as np
import torch


class RGBEnv:
    """Small planar visual-navigation environment with staged RGB transfers."""

    def __init__(
        self,
        n=256,
        seed=1,
        max_steps=2000,
        width=64,
        height=64,
        device=0,
        library=None,
    ):
        if n <= 0 or max_steps <= 0 or width <= 0 or height <= 0:
            raise ValueError("n, max_steps, width and height must be positive")
        self.n = n
        self.seed = seed
        self.max_steps = max_steps
        self.width = width
        self.height = height
        self.history_frames = 4
        self.observation_shape = (height, width, self.history_frames * 3)
        self.action_size = 2
        self.device = torch.device(
            "cuda", device
        ) if torch.cuda.is_available() else torch.device("cpu")
        path = (
            Path(library)
            if library
            else Path(__file__).resolve().parents[1] / "target/release/librgb_env.so"
        )
        self.lib = ctypes.CDLL(str(path))
        p = ctypes.c_void_p
        self.lib.triage_rgb_create.argtypes = [
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        self.lib.triage_rgb_create.restype = p
        self.lib.triage_rgb_reset.argtypes = [p, ctypes.c_uint64]
        self.lib.triage_rgb_reset.restype = ctypes.c_int
        self.lib.triage_rgb_step.argtypes = [p, ctypes.POINTER(ctypes.c_float)]
        self.lib.triage_rgb_step.restype = ctypes.c_int
        self.lib.triage_rgb_buffer.argtypes = [p, ctypes.c_int]
        self.lib.triage_rgb_buffer.restype = p
        self.lib.triage_rgb_checksum.argtypes = [p]
        self.lib.triage_rgb_checksum.restype = ctypes.c_uint64
        self.lib.triage_rgb_destroy.argtypes = [p]
        self.lib.triage_rgb_destroy.restype = ctypes.c_int
        self.lib.triage_rgb_error.argtypes = []
        self.lib.triage_rgb_error.restype = ctypes.c_char_p
        self.lib.triage_rgb_timings.argtypes = [p, ctypes.POINTER(ctypes.c_double)]
        self.lib.triage_rgb_timings.restype = ctypes.c_int

        self.handle = self.lib.triage_rgb_create(n, seed, max_steps, width, height)
        if not self.handle:
            self._raise()
        self.last_step_timings = {}
        self._host_observations = self._view(
            0, ctypes.c_uint8, (n, self.history_frames, height, width, 3)
        )
        self._host_rewards = self._view(1, ctypes.c_float, (n,))
        self._host_terminated = self._view(2, ctypes.c_float, (n,))
        self._host_truncated = self._view(3, ctypes.c_float, (n,))
        self._host_final_observations = self._view(
            4, ctypes.c_uint8, (n, self.history_frames, height, width, 3)
        )
        self._host_completed_returns = self._view(5, ctypes.c_float, (n,))
        self._host_completed_lengths = self._view(6, ctypes.c_float, (n,))
        self._host_episode_counts = self._view(7, ctypes.c_uint64, (n,))
        self._host_current_returns = self._view(8, ctypes.c_float, (n,))
        self._host_current_lengths = self._view(9, ctypes.c_float, (n,))

        self.observations = torch.empty(
            self.observation_shape, dtype=torch.float32, device=self.device
        ).expand(n, -1, -1, -1).clone()
        self.final_observations = torch.empty_like(self.observations)
        self.rewards = torch.empty(n, dtype=torch.float32, device=self.device)
        self.terminated = torch.empty_like(self.rewards)
        self.truncated = torch.empty_like(self.rewards)
        self.completed_returns = torch.empty_like(self.rewards)
        self.completed_lengths = torch.empty_like(self.rewards)
        self.episode_counts = torch.empty(n, dtype=torch.uint64, device=self.device)
        self.current_returns = torch.empty_like(self.rewards)
        self.current_lengths = torch.empty_like(self.rewards)
        self._refresh()

    def _view(self, field, ctype, shape):
        pointer = self.lib.triage_rgb_buffer(self.handle, field)
        if not pointer:
            self._raise()
        address = ctypes.cast(pointer, ctypes.POINTER(ctype))
        return np.ctypeslib.as_array(address, shape=shape)

    def _raise(self):
        message = self.lib.triage_rgb_error()
        text = ctypes.string_at(message).decode() if message else "unknown native error"
        raise RuntimeError(text)

    def _copy(self, target, source):
        target.copy_(torch.from_numpy(source), non_blocking=False)
    def _copy_history(self, target, source):
        channels_last = source.transpose(0, 2, 3, 1, 4).reshape(target.shape)
        target.copy_(torch.from_numpy(channels_last), non_blocking=False)

    def _refresh(self):
        self._copy_history(self.observations, self._host_observations)
        self._copy_history(self.final_observations, self._host_final_observations)
        self._copy(self.rewards, self._host_rewards)
        self._copy(self.terminated, self._host_terminated)
        self._copy(self.truncated, self._host_truncated)
        self._copy(self.completed_returns, self._host_completed_returns)
        self._copy(self.completed_lengths, self._host_completed_lengths)
        self._copy(self.episode_counts, self._host_episode_counts)
        self._copy(self.current_returns, self._host_current_returns)
        self._copy(self.current_lengths, self._host_current_lengths)

    def reset(self, seed=None):
        seed = self.seed if seed is None else seed
        if self.lib.triage_rgb_reset(self.handle, seed) != 0:
            self._raise()
        self.seed = seed
        self._refresh()
        self.last_step_timings = {}

    def step(self, actions):
        if actions.shape != (self.n, self.action_size):
            raise ValueError(f"actions must have shape [{self.n},2]")
        total_start = time.perf_counter_ns()
        stage_start = time.perf_counter_ns()
        staged = actions.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
        action_stage_ms = (time.perf_counter_ns() - stage_start) / 1_000_000.0
        pointer = staged.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        native_start = time.perf_counter_ns()
        if self.lib.triage_rgb_step(self.handle, pointer) != 0:
            self._raise()
        native_ms = (time.perf_counter_ns() - native_start) / 1_000_000.0
        copy_start = time.perf_counter_ns()
        self._refresh()
        observation_copy_ms = (time.perf_counter_ns() - copy_start) / 1_000_000.0
        timings = (ctypes.c_double * 3)()
        if self.lib.triage_rgb_timings(self.handle, timings) != 0:
            self._raise()
        self.last_step_timings = {
            "action_stage_ms": action_stage_ms,
            "native_step_ms": native_ms,
            "dynamics_ms": timings[0],
            "render_readback_ms": timings[1],
            "history_ms": timings[2],
            "observation_copy_ms": observation_copy_ms,
            "total_ms": (time.perf_counter_ns() - total_start) / 1_000_000.0,
        }

    def checksum(self):
        return int(self.lib.triage_rgb_checksum(self.handle))

    def close(self):
        if self.handle is not None:
            handle, self.handle = self.handle, None
            if self.lib.triage_rgb_destroy(handle) != 0:
                self._raise()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
