"""Bounded, fail-closed recording helpers for independent PICO demonstrations.

Importing this module does not initialize SIMPLE, an SDK client, rendering, or
model inference. The asynchronous adapter preserves the existing capture
contract and never admits an episode into a training dataset by itself.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import queue
import threading
import time
from typing import Any
import uuid

import numpy as np


class CaptureRejected(RuntimeError):
    """A sticky capture failure; this episode cannot subsequently complete."""


def _snapshot(value: Any) -> Any:
    """Copy capture data without invoking arbitrary objects' deepcopy hooks."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot be queued as capture evidence")
        return value.copy(order="C")
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool, np.generic, Path)):
        return value
    raise TypeError(f"Unsupported capture evidence type: {type(value).__name__}")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON evidence keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite JSON evidence is not accepted")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported JSON evidence type: {type(value).__name__}")


def _copy_command(command: Any) -> Any:
    if command is None:
        return None
    parameters = getattr(command, "parameters", None)
    required = {"controller", "body_token", "hand_targets"}
    if (getattr(command, "type", None) != "sonic_v1_1"
            or not isinstance(parameters, dict) or set(parameters) != required
            or parameters["controller"] is None
            or not isinstance(parameters["hand_targets"], dict)):
        raise ValueError("Expected an actual sonic_v1_1 command with named hands")
    # In particular, do not use ActionCmd.copy(): its deepcopy also copies the
    # controller and would try to copy an ONNX Runtime session.
    return type(command)(command.type, controller=parameters["controller"],
                         body_token=_snapshot(parameters["body_token"]),
                         hand_targets=_snapshot(parameters["hand_targets"]))


class AsyncTrainingCapture:
    """Queue isolated event snapshots; one writer owns all capture appends.

    Queue saturation, an invalid snapshot, or a writer error permanently
    rejects the episode. ``close`` drains/skips outstanding work, joins the
    writer, then returns a manifest, including on failures. ``capture_factory``
    is a dependency-injection hook for CPU tests; deployment defaults to the
    unchanged ``replay_handover_teacher.TrainingCapture``.
    """

    def __init__(self, directory, *, metadata, joint_names, warmup_ticks,
                 start_time, max_queue=32, capture_factory=None):
        if type(max_queue) is not int or max_queue < 1:
            raise ValueError("max_queue must be a positive integer")
        if capture_factory is None:
            from replay_handover_teacher import TrainingCapture
            capture_factory = TrainingCapture
        self.directory = Path(directory)
        self._capture = capture_factory(self.directory, metadata=_snapshot(metadata),
            joint_names=_snapshot(joint_names), warmup_ticks=warmup_ticks, start_time=start_time)
        self._queue = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        self._close_done = threading.Event()
        self._stop = object()
        self._closing = False
        self._requested_completed = False
        self._error = None
        self._manifest = None
        self._enqueued = self._written = 0
        self._max_queue = max_queue
        self._writer = threading.Thread(target=self._write_loop,
            name="PicoTrainingCaptureWriter", daemon=False)
        self._writer.start()

    @property
    def failed(self):
        with self._lock:
            return self._error is not None

    @property
    def error(self):
        with self._lock:
            return self._error

    @property
    def worker_alive(self):
        return self._writer.is_alive()

    def _fail(self, error):
        with self._lock:
            if self._error is None:
                self._error = f"{type(error).__name__}: {error}"

    def enqueue(self, **append_kwargs):
        with self._lock:
            if self._error is not None:
                raise CaptureRejected(self._error)
            if self._closing:
                raise CaptureRejected("Capture is already closing or closed")
            try:
                snapshot = {key: _copy_command(value) if key == "command" else _snapshot(value)
                            for key, value in append_kwargs.items()}
                self._queue.put_nowait(snapshot)
            except Exception as error:
                self._error = f"{type(error).__name__}: {error}"
                if isinstance(error, queue.Full):
                    self._error = "Capture queue full: refusing an episode with missing evidence"
                raise CaptureRejected(self._error) from error
            self._enqueued += 1

    def _write_loop(self):
        while True:
            item = self._queue.get()
            try:
                if item is self._stop:
                    break
                if not self.failed:
                    try:
                        self._capture.append(**item)
                        with self._lock:
                            self._written += 1
                    except BaseException as error:
                        self._fail(error)
            finally:
                self._queue.task_done()
        # No append can run concurrently with the capture's close/flush.
        try:
            if self._requested_completed and self._written == 0 and not self.failed:
                self._fail(ValueError("A completed capture requires recorded evidence"))
            completed = self._requested_completed and not self.failed
            manifest = self._capture.close(completed=completed)
            if not isinstance(manifest, dict):
                raise TypeError("Capture close must return a manifest dictionary")
            if completed and manifest.get("status") != "raw_capture_completed_pending_independent_audit":
                raise ValueError("Underlying capture did not complete pending independent audit")
            self._manifest = _snapshot(manifest)
        except BaseException as error:
            self._fail(error)
            self._manifest = {"status": "partial_capture_rejected"}

    def close(self, *, completed):
        if type(completed) is not bool:
            raise TypeError("completed must be a boolean")
        with self._lock:
            first_close = not self._closing
            if first_close:
                self._closing = True
                self._requested_completed = completed
        if not first_close:
            self._close_done.wait()
            return _snapshot(self._manifest)
        try:
            # A second Ctrl+C during cleanup must not strand the non-daemon
            # writer. Record the interruption and finish draining/joining.
            while True:
                try:
                    self._queue.put(self._stop, timeout=.1)
                    break
                except queue.Full:
                    continue
                except BaseException as error:
                    self._fail(error)
            while True:
                try:
                    self._writer.join(timeout=.1)
                except BaseException as error:
                    self._fail(error)
                    continue
                if not self._writer.is_alive():
                    break
            manifest = self._manifest or {"status": "partial_capture_rejected"}
            manifest["accepted_new_dataset"] = False
            if self.failed or not completed:
                manifest["status"] = "partial_capture_rejected"
            manifest["async_capture"] = {
                "requested_completed": completed,
                "completed": bool(completed and not self.failed),
                "max_queue": self._max_queue,
                "enqueued_events": self._enqueued,
                "written_events": self._written,
                "failure": self.error,
                "writer_joined": not self._writer.is_alive(),
                "accepted_new_dataset": False,
            }
            self._manifest = manifest
            try:
                receipt = self.directory / "async_capture.json"
                receipt.write_text(json.dumps(_json_value(manifest), ensure_ascii=False,
                    indent=2, allow_nan=False) + "\n")
            except BaseException as error:
                self._fail(error)
                manifest["status"] = "partial_capture_rejected"
                manifest["async_capture"].update(completed=False, failure=self.error)
            return _snapshot(manifest)
        finally:
            self._close_done.set()


class RecordingState:
    """Pure per-demonstration state; completion only means pending audit.

    Call ``on_tick`` once per real 50 Hz control tick. A start request waits
    for a global tick divisible by five, then captures a real one-second
    warmup prefix. Create a fresh instance for the next independent episode.
    """

    def __init__(self, warmup_ticks=50, *, human_demo_uuid=None):
        if type(warmup_ticks) is not int or warmup_ticks < 50 or warmup_ticks % 5:
            raise ValueError("Warmup needs >=50 real ticks, aligned to both clocks")
        self.warmup_ticks = warmup_ticks
        self.human_demo_uuid = str(uuid.UUID(str(human_demo_uuid))) if human_demo_uuid else str(uuid.uuid4())
        self.status = "idle"
        self.start_tick50 = None
        self._last_tick50 = None
        self._reasons = []

    def request_start(self):
        if self.status != "idle":
            raise RuntimeError("Start needs a fresh idle demonstration")
        self.status = "start_requested"

    def _receipt(self, *, started=False):
        active = self.status in ("warmup", "recording")
        local = None if self.start_tick50 is None or self._last_tick50 is None else self._last_tick50 - self.start_tick50
        capture_due = active and local is not None and ((local * 30 // 50) * 50 + 29) // 30 == local
        return {"status": self.status, "human_demo_uuid": self.human_demo_uuid,
                "start_tick50": self.start_tick50, "local_tick50": local,
                "started": started, "recording": active, "warmup": self.status == "warmup",
                "capture_due": capture_due, "action_valid": self.status == "recording",
                "pending_audit": self.status == "pending_audit",
                "accepted_new_dataset": False, "reasons": list(self._reasons)}

    def on_tick(self, tick50):
        if type(tick50) is not int or tick50 < 0:
            return self.reject("invalid physical tick")
        if self._last_tick50 is not None:
            if tick50 <= self._last_tick50:
                return self.reject("physical ticks did not advance monotonically")
            if self.status in ("warmup", "recording") and tick50 != self._last_tick50 + 1:
                return self.reject("missing physical tick / control overrun")
        self._last_tick50 = tick50
        started = self.status == "start_requested" and tick50 % 5 == 0
        if started:
            self.start_tick50 = tick50
            self.status = "warmup"
        if self.status == "warmup" and tick50 - self.start_tick50 >= self.warmup_ticks:
            self.status = "recording"
        return self._receipt(started=started)

    def request_save(self):
        if self.status != "recording":
            raise RuntimeError("Save requires active recording after the real warmup")
        self.status = "pending_audit"
        return self._receipt()

    def discard(self, reason="operator_discard"):
        if self.status not in ("rejected", "discarded"):
            self.status = "discarded"
        self._reasons.append(str(reason))
        return self._receipt()

    def reject(self, reason):
        self.status = "rejected"
        self._reasons.append(str(reason))
        return self._receipt()

    def close(self, *, completed=False):
        if type(completed) is not bool:
            raise TypeError("completed must be a boolean")
        if completed and self.status != "pending_audit":
            return self.reject("completed capture was not requested for independent audit")
        if not completed and self.status not in ("rejected", "discarded"):
            return self.reject("capture closed without completion")
        return self._receipt()


class CommandLedger:
    """Append immutable JSONL evidence under a genuinely new human identity.

    Raw PICO ``va``/``wva`` may be retained as source evidence, but this ledger
    never declares those channels validated derivatives or usable labels.
    """

    _RESERVED = {"event_index", "event_type", "human_demo_uuid", "recorded_monotonic_ns",
                 "accepted_new_dataset", "raw_pico_va_wva_validated", "raw_derivative_policy"}

    def __init__(self, path, *, human_demo_uuid=None, metadata=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.human_demo_uuid = str(uuid.UUID(str(human_demo_uuid))) if human_demo_uuid else str(uuid.uuid4())
        self._lock = threading.Lock()
        self._stream = self.path.open("x", buffering=1)
        self._count = 0
        self._closed = False
        try:
            self.append("session", metadata={} if metadata is None else metadata)
        except BaseException:
            self._stream.close()
            self._closed = True
            raise

    def append(self, event_type, **fields):
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("A nonempty ledger event type is required")
        if set(fields) & self._RESERVED:
            raise ValueError("Ledger identity and unvalidated-derivative flags cannot be overridden")
        copied = _json_value(_snapshot(fields))
        with self._lock:
            if self._closed:
                raise RuntimeError("Command ledger is closed")
            event = {**copied, "event_index": self._count, "event_type": event_type,
                     "human_demo_uuid": self.human_demo_uuid,
                     "recorded_monotonic_ns": time.monotonic_ns(),
                     "accepted_new_dataset": False, "raw_pico_va_wva_validated": False,
                     "raw_derivative_policy": "archived_untrusted_not_control_or_labels"}
            self._stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            self._count += 1
            return event

    def close(self):
        with self._lock:
            if not self._closed:
                self._stream.flush()
                self._stream.close()
                self._closed = True
            digest = hashlib.sha256()
            with self.path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return {"path": str(self.path), "sha256": digest.hexdigest(),
                    "event_count": self._count, "human_demo_uuid": self.human_demo_uuid,
                    "accepted_new_dataset": False, "raw_pico_va_wva_validated": False}
