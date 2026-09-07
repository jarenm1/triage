"""Zero-copy, stream-ordered access to native CUDA hover and tracking tasks."""

import ctypes
from pathlib import Path

import torch

TASK_VERSION = 1
TASK_CONFIG = {
    "version": TASK_VERSION,
    "observation_size": 22,
    "action_size": 4,
    "target": [0.0, 0.0, 1.0],
    "physics_dt": 0.000625,
    "control_substeps": 16,
    "control_hz": 100,
    "action_scale": 0.15,
    "max_steps": 2000,
    "action_transform": "clamp(hover_command + 0.15*tanh(raw),0,1)",
    "hover_command": "derived from Base quad physical parameters",
    "initialization": {
        "position_half_range": 0.15,
        "roll_pitch_half_range": 0.05,
        "yaw": "uniform",
        "velocity_half_range": 0.05,
        "body_rate_half_range": 0.02,
        "rotor_speed": "hover equilibrium",
    },
    "failure": {
        "minimum_z": 0.05,
        "maximum_distance": 2.0,
        "maximum_tilt": 0.8,
        "nonfinite": True,
    },
    "reward": {
        "position_squared": 2.0,
        "velocity_squared": 0.1,
        "body_rate_squared": 0.05,
        "tilt_squared": 0.5,
        "mean_squared_tanh_action_penalty": 0.005,
        "failure": -1.0,
    },
    "evaluation": {"steps": 2000, "maximum_distance": 1.0, "maximum_tilt": 0.7},
}

TRACKING_TASK_CONFIG = {
    **TASK_CONFIG,
    "version": 1,
    "name": "randomized_target_tracking",
    "target": {
        "bounds": [[-1.0, 1.0], [-1.0, 1.0], [0.75, 1.75]],
        "displacement": [0.5, 1.5],
        "previous_initial": [0.0, 0.0, 1.0],
        "sampling": "uniform box rejection, independent of physical trajectory",
        "schedule": {"long_probability": 0.75, "long": [200, 500], "short": [20, 100]},
        "rng": "seed/env/episode/command; independent of physical reset RNG",
        "plan_commands": 101,
    },
    "observation_position": "world position minus active target; no future target or countdown",
    "failure": {
        "x_bounds": [-3.0, 3.0],
        "y_bounds": [-3.0, 3.0],
        "z_bounds": [0.05, 3.0],
        "maximum_tilt": 0.8,
        "nonfinite": True,
    },
    "target_transition": "reward/final observation use old target; next observation uses new target; no physical reset",
    "evaluation": {
        "steps": 2000,
        "settling_duration": 500,
        "settling_last_steps": 50,
        "settling_distance": 0.2,
        "settling_speed": 0.2,
        "settled_fraction": 0.9,
        "survival_fraction": 0.95,
        "mixed_rms_baseline_ratio": 0.75,
        "failed_step_squared_error": 37.0625,
    },
}


class _CudaPtr:
    def __init__(self, owner, ptr, shape, typestr):
        self.owner = owner
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": shape,
            "typestr": typestr,
            "version": 2,
        }


class HoverEnv:
    """Views are borrowed until close; all callers must use one CUDA stream.

    Native calls and torch consumers share the stream active at creation. Retain
    this object while using any view. close() invalidates every borrowed view.
    """

    def __init__(
        self, n=1024, seed=1, max_steps=2000, device=0, library=None, *, _schedule=None
    ):
        if n <= 0 or max_steps <= 0:
            raise ValueError("n and max_steps must be positive")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Flight environments require CUDA; no CPU fallback exists"
            )
        self.device = torch.device("cuda", device)
        self.n = n
        self.max_steps = max_steps
        self.seed = seed
        self.handle = None
        self._views = []
        path = (
            Path(library)
            if library
            else Path(__file__).resolve().parents[1] / "cuda/build/libtriage_hover.so"
        )
        self.lib = ctypes.CDLL(str(path))
        p, i, u, z = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64, ctypes.c_size_t
        signatures = {
            "create": ([i, z, u, i, p], p),
            "reset": ([p, u, p], i),
            "step": ([p, p, p], i),
            "buffer": ([p, i], p),
            "destroy": ([p], i),
            "error": ([], ctypes.c_char_p),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, "triage_hover_" + name)
            fn.argtypes, fn.restype = args, result
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.current_stream(self.device)
            if _schedule is None:
                self.handle = self.lib.triage_hover_create(
                    device, n, seed, max_steps, self.stream.cuda_stream
                )
            else:
                create = self.lib.triage_tracking_create
                create.argtypes, create.restype = [i, z, u, i, i, p], p
                self.handle = create(
                    device, n, seed, max_steps, _schedule, self.stream.cuda_stream
                )
            if not self.handle:
                self._raise()
            fields = [
                ("observations", (n, 22), "<f4"),
                ("rewards", (n,), "<f4"),
                ("terminated", (n,), "<f4"),
                ("truncated", (n,), "<f4"),
                ("final_observations", (n, 22), "<f4"),
                ("completed_returns", (n,), "<f4"),
                ("completed_lengths", (n,), "<f4"),
                ("reset_status", (n,), "|u1"),
                ("episode_counts", (n,), "<u8"),
                ("current_returns", (n,), "<f4"),
                ("current_lengths", (n,), "<f4"),
            ]
            if _schedule is not None:
                fields.extend(
                    [
                        ("targets", (n, 3), "<f4"),
                        ("final_targets", (n, 3), "<f4"),
                        ("command_duration", (n,), "<f4"),
                        ("command_elapsed", (n,), "<f4"),
                        ("command_finished", (n,), "<f4"),
                        ("command_settled", (n,), "<f4"),
                        ("command_index", (n,), "<u8"),
                        ("action_saturation", (n,), "<f4"),
                        ("command_plan", (n, 101, 4), "<f4"),
                    ]
                )
            try:
                for field, (name, shape, typestr) in enumerate(fields):
                    ptr = self.lib.triage_hover_buffer(self.handle, field)
                    if not ptr:
                        self._raise()
                    view = _CudaPtr(self, ptr, shape, typestr)
                    self._views.append(view)
                    setattr(self, name, torch.as_tensor(view, device=self.device))
            except BaseException:
                self.close()
                raise

    def _raise(self):
        error = self.lib.triage_hover_error()
        raise RuntimeError(
            error.decode() if error else "Native environment call failed"
        )

    def _stream(self):
        if self.handle is None:
            raise RuntimeError("Environment is closed")
        stream = torch.cuda.current_stream(self.device)
        if stream.cuda_stream != self.stream.cuda_stream:
            raise RuntimeError(
                "Use the CUDA stream on which the environment was created"
            )
        return stream.cuda_stream

    def reset(self, seed=None):
        seed = self.seed if seed is None else seed
        if self.lib.triage_hover_reset(self.handle, seed, self._stream()) != 0:
            self._raise()
        self.seed = seed

    def step(self, actions):
        if (
            actions.device != self.device
            or actions.dtype != torch.float32
            or actions.shape != (self.n, 4)
            or not actions.is_contiguous()
        ):
            raise ValueError(
                "actions must be contiguous CUDA float32 [N,4] on the environment device"
            )
        if (
            self.lib.triage_hover_step(self.handle, actions.data_ptr(), self._stream())
            != 0
        ):
            self._raise()
        actions.record_stream(self.stream)

    def close(self):
        if self.handle is not None:
            # Teardown is outside rollout; wait before freeing borrowed allocations.
            self.stream.synchronize()
            handle, self.handle = self.handle, None
            if self.lib.triage_hover_destroy(handle) != 0:
                self._raise()
            self._views.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TrackingEnv(HoverEnv):
    """Tracking views share HoverEnv lifetime/stream rules; command_plan is read-only."""

    def __init__(
        self, n=1024, seed=1, max_steps=2000, device=0, library=None, schedule="mixed"
    ):
        schedules = {
            "mixed": (0, 2000),
            "settling": (1, 2000),
            "long-flight-v1": (2, 6000),
        }
        if schedule not in schedules:
            raise ValueError("schedule must be mixed, settling, or long-flight-v1")
        native_schedule, limit = schedules[schedule]
        if not 0 < max_steps <= limit:
            raise ValueError(f"{schedule} max_steps must be in [1,{limit}]")
        self.schedule = schedule
        super().__init__(
            n,
            seed,
            max_steps,
            device,
            library,
            _schedule=native_schedule,
        )
