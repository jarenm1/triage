"""Optional bounded native state staging and worker-owned NDJSON transport.

The simulation thread alone calls CUDA/native APIs. The worker only sees owned
compact CPU bytes; file/socket IO and JSON encoding never run in a control call.
"""

import ctypes
import json
import queue
import socket
import threading
import time
from contextlib import ExitStack


class Vehicle(ctypes.Structure):
    _fields_ = [
        ("environment_id", ctypes.c_uint32),
        ("episode_id", ctypes.c_uint64),
        ("position_w", ctypes.c_float * 3),
        ("attitude_wb", ctypes.c_float * 4),
        ("target_w", ctypes.c_float * 3),
    ]


def encode_line(value):
    return (json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n").encode()


class SnapshotProducer:
    """Close before the environment. submit/poll never wait for GPU or IO.

    Ring slots remain owned by CUDA until the D2H completion event is ready.
    A successful poll copies that immutable result to bounded worker storage,
    freeing the native slot regardless of disk/network backpressure.
    """

    def __init__(self, env, header, *, record=None, listen=None, slots=4):
        self.env = env
        self.ids = tuple(header["environment_ids"])
        if not self.ids or len(self.ids) > 64 or len(set(self.ids)) != len(self.ids):
            raise ValueError("select 1..64 unique environment IDs")
        if any(type(i) is not int or i < 0 or i >= env.n for i in self.ids):
            raise ValueError("environment ID outside batch")
        if not 2 <= slots <= 16:
            raise ValueError("slots must be in [2,16]")
        self._rows = (Vehicle * len(self.ids))()
        self._step = ctypes.c_uint64()
        self._gather = ctypes.c_float()
        self._copy = ctypes.c_float()
        p, z, u, f = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_float
        signatures = {
            "configure": ([p, ctypes.POINTER(ctypes.c_uint32), z, z], ctypes.c_int),
            "submit": ([p, u, p], ctypes.c_int),
            "poll": (
                [
                    p,
                    ctypes.POINTER(Vehicle),
                    z,
                    ctypes.POINTER(u),
                    ctypes.POINTER(f),
                    ctypes.POINTER(f),
                ],
                ctypes.c_int,
            ),
            "disable": ([p], ctypes.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(env.lib, "triage_snapshot_" + name)
            fn.argtypes, fn.restype = args, result
        self.stats = {
            "attempted": 0,
            "staged": 0,
            "ring_dropped": 0,
            "polled": 0,
            "worker_dropped": 0,
            "encoded": 0,
            "recorded": 0,
            "network_replaced": 0,
            "network_sent": 0,
            "connections": 0,
            "gather_gpu_ms": 0.0,
            "d2h_gpu_ms": 0.0,
            "submit_host_ms": 0.0,
            "poll_host_ms": 0.0,
            "encode_worker_ms": 0.0,
            "record_worker_ms": 0.0,
        }
        self._queue = queue.Queue(maxsize=slots)
        self._stop = threading.Event()
        self._error = None
        self._closed = False
        self._header = encode_line(header)
        self._record = None
        self._listener = None
        with ExitStack() as resources:
            if record:
                self._record = resources.enter_context(open(record, "xb"))
            if listen:
                host, port = listen.rsplit(":", 1)
                if host not in ("localhost", "127.0.0.1"):
                    raise ValueError("live listener must use localhost or 127.0.0.1")
                self._listener = resources.enter_context(
                    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                )
                self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._listener.bind(("127.0.0.1", int(port)))
                self._listener.listen(1)
                self._listener.setblocking(False)
            ids = (ctypes.c_uint32 * len(self.ids))(*self.ids)
            if env.lib.triage_snapshot_configure(env.handle, ids, len(ids), slots) < 0:
                env._raise()
            self._resources = resources.pop_all()
        self._worker = threading.Thread(
            target=self._run, name="trajectory-io", daemon=True
        )
        self._worker.start()

    def _check(self):
        if self._closed:
            raise RuntimeError("Snapshot producer is closed")
        if self._error is not None:
            raise RuntimeError("Snapshot IO worker failed") from self._error

    def submit(self, step):
        self._check()
        start = time.perf_counter()
        result = self.env.lib.triage_snapshot_submit(
            self.env.handle, step, self.env._stream()
        )
        self.stats["submit_host_ms"] += (time.perf_counter() - start) * 1000
        if result < 0:
            self.env._raise()
        self.stats["attempted"] += 1
        self.stats["staged" if result else "ring_dropped"] += 1
        return bool(result)

    def poll(self):
        self._check()
        start = time.perf_counter()
        # At most the fixed ring capacity can be ready; never waits on an event.
        while True:
            result = self.env.lib.triage_snapshot_poll(
                self.env.handle,
                self._rows,
                len(self.ids),
                ctypes.byref(self._step),
                ctypes.byref(self._gather),
                ctypes.byref(self._copy),
            )
            if result < 0:
                self.env._raise()
            if not result:
                break
            self.stats["polled"] += 1
            self.stats["gather_gpu_ms"] += self._gather.value
            self.stats["d2h_gpu_ms"] += self._copy.value
            try:
                self._queue.put_nowait((self._step.value, bytes(self._rows)))
            except queue.Full:
                self.stats["worker_dropped"] += 1
        self.stats["poll_host_ms"] += (time.perf_counter() - start) * 1000

    def _run(self):
        client = None
        sending = b""
        offset = 0
        sending_frame = False
        pending = None
        latest = None
        try:
            if self._record:
                self._record.write(self._header)
                self._record.flush()
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    step, raw = self._queue.get(timeout=0.002)
                except queue.Empty:
                    pass
                else:
                    start = time.perf_counter()
                    rows = (Vehicle * len(self.ids)).from_buffer_copy(raw)
                    frame = {
                        "step": step,
                        "time_seconds": step * 0.01,
                        "vehicles": [
                            {
                                "environment_id": row.environment_id,
                                "episode_id": row.episode_id,
                                "position_w": list(row.position_w),
                                "attitude_wb": list(row.attitude_wb),
                                "target_w": list(row.target_w),
                            }
                            for row in rows
                        ],
                    }
                    latest = encode_line(frame)
                    self.stats["encode_worker_ms"] += (
                        time.perf_counter() - start
                    ) * 1000
                    self.stats["encoded"] += 1
                    if self._record:
                        start = time.perf_counter()
                        self._record.write(latest)
                        self.stats["record_worker_ms"] += (
                            time.perf_counter() - start
                        ) * 1000
                        self.stats["recorded"] += 1
                    if client:
                        if pending is not None:
                            self.stats["network_replaced"] += 1
                        pending = latest
                if self._listener and client is None:
                    try:
                        client, _ = self._listener.accept()
                    except BlockingIOError:
                        pass
                    else:
                        client.setblocking(False)
                        client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)
                        self.stats["connections"] += 1
                        sending, offset, sending_frame = self._header, 0, False
                        pending = latest
                if client:
                    if not sending and pending is not None:
                        sending, pending, offset, sending_frame = pending, None, 0, True
                    if sending:
                        # An unsent frame may be replaced; a partial line must finish.
                        if sending_frame and offset == 0 and pending is not None:
                            sending, pending = pending, None
                            self.stats["network_replaced"] += 1
                        try:
                            sent = client.send(memoryview(sending)[offset:])
                            if sent == 0:
                                raise ConnectionError("viewer disconnected")
                            offset += sent
                            if offset == len(sending):
                                if sending_frame:
                                    self.stats["network_sent"] += 1
                                sending, offset = b"", 0
                        except BlockingIOError:
                            pass
                        except (ConnectionError, OSError):
                            client.close()
                            client = None
                            sending, pending, offset = b"", None, 0
            # Never wait for a live client during shutdown; disk recording drains.
        except (OSError, ValueError, TypeError) as error:
            self._error = error
        finally:
            if client:
                client.close()
            try:
                self._resources.close()
            except OSError as error:
                self._error = self._error or error

    def close(self):
        if self._closed:
            return
        try:
            # Teardown may wait. Continue polling until every staged frame has
            # completed; the independent D2H stream need not finish with physics.
            while self.stats["polled"] < self.stats["staged"]:
                self.poll()
                if self.stats["polled"] < self.stats["staged"]:
                    time.sleep(0.001)
        finally:
            self._stop.set()
            self._worker.join()
            self._closed = True
            if self.env.lib.triage_snapshot_disable(self.env.handle) < 0:
                self.env._raise()
        if self._error is not None:
            raise RuntimeError("Snapshot IO worker failed") from self._error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
