"""Read-only XRoboToolkit 1.1.1 tracking input, without its caching getters.

All poses are XYZ + quaternion XYZW, directly from this packet. In particular,
missing Body data is never replaced with the last valid skeleton. Unreliable
SDK ``va``/``wva`` values are retained in raw_payload and never used as velocity.

The SDK/service has one feedback consumer. Do not run the upstream demo,
``pico_manager`` or another diagnostic receiver at the same time. The advisory
lock prevents a second instance of *this* receiver, not unrelated SDK clients.
This module never starts/stops the service or sends commands to the headset.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass, field
import fcntl
import json
import math
import os
from pathlib import Path
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


DEFAULT_SDK_PATH = "/opt/apps/roboticsservice/SDK/x64/libPXREARobotSDK.so"
_TRACKING_EVENT = 1 << 25


def _readonly_array(value: Any) -> np.ndarray:
    # A bytes-backed array cannot be made writable with setflags(write=True).
    array = np.asarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _pose(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.split(",")
    try:
        return _readonly_array(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Tracking pose contains non-numeric values") from exc


@dataclass(frozen=True)
class PicoFrame:
    device_timestamp_ns: int | None
    host_received_s: float
    sequence: int
    head_pose7: np.ndarray | None
    left_pose7: np.ndarray | None
    right_pose7: np.ndarray | None
    body24x7: np.ndarray | None
    input_mode: int | None
    app_focused: bool
    head_status: int | None
    inputs: Mapping[str, float]
    buttons: Mapping[str, bool]
    _raw_json: str = field(repr=False)

    @property
    def raw_payload(self) -> dict:
        """Independent JSON copy, including original untrusted va/wva fields."""
        return json.loads(self._raw_json)


def parse_tracking(outer: Mapping | str | bytes, host_received_s: float,
                   sequence: int = 0) -> PicoFrame:
    """Parse one complete raw callback; never consult previous packets.

    Missing fields remain missing for validate_frame to reject. Malformed JSON
    or non-numeric pose text raises ValueError instead of reusing an old frame.
    """
    if isinstance(outer, (str, bytes)):
        outer = json.loads(outer)
    if not isinstance(outer, Mapping):
        raise ValueError("Tracking packet must be a JSON object")
    if outer.get("functionName", "Tracking") != "Tracking":
        raise ValueError("Packet is not a Tracking message")
    raw_json = json.dumps(dict(outer), allow_nan=False)
    value = outer.get("value", outer)
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("Tracking value must be an object")

    head = value.get("Head", {})
    controllers = value.get("Controller", {})
    body = value.get("Body", {})
    app = value.get("appState", {})
    if not all(isinstance(item, Mapping) for item in (head, controllers, body, app)):
        raise ValueError("Tracking sections must be objects")
    left, right = controllers.get("left", {}), controllers.get("right", {})
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ValueError("Controller sections must be objects")
    joints = body.get("joints")
    skeleton = None
    if joints is not None:
        if not isinstance(joints, list) or any(not isinstance(j, Mapping) for j in joints):
            raise ValueError("Body.joints must be a list of pose objects")
        poses = [_pose(j.get("p")) for j in joints]
        if any(p is None or p.shape != (7,) for p in poses):
            raise ValueError("Each Body joint must contain one seven-value p pose")
        skeleton = _readonly_array(poses) if poses else _readonly_array(np.empty((0, 7)))
    timestamp = value.get("timeStampNs")
    if timestamp is not None and (isinstance(timestamp, bool) or not isinstance(timestamp, int)):
        raise ValueError("timeStampNs must be an integer number of nanoseconds")
    mode = value.get("Input")
    if mode is not None and (isinstance(mode, bool) or not isinstance(mode, int)):
        raise ValueError("Input mode must be an integer")
    status = head.get("status")
    if status is not None and (isinstance(status, bool) or not isinstance(status, int)):
        raise ValueError("Head.status must be an integer")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("sequence must be a non-negative integer")
    received = float(host_received_s)
    if not math.isfinite(received):
        raise ValueError("host_received_s must be finite monotonic time")
    inputs, buttons = {}, {}
    for side, control in (("left", left), ("right", right)):
        for output, key in (("trigger", "trigger"), ("grip", "grip"),
                            ("axis_x", "axisX"), ("axis_y", "axisY")):
            sample = control.get(key, 0.0)
            if isinstance(sample, bool) or not isinstance(sample, (float, int)):
                raise ValueError(f"{side}.{key} must be numeric")
            inputs[f"{side}_{output}"] = float(sample)
        for output, key in (("menu", "menuButton"), ("axis_click", "axisClick")):
            sample = control.get(key, False)
            if not isinstance(sample, bool):
                raise ValueError(f"{side}.{key} must be boolean")
            buttons[f"{side}_{output}"] = sample
        for output, key in zip(("X", "Y") if side == "left" else ("A", "B"),
                               ("primaryButton", "secondaryButton")):
            sample = control.get(key, False)
            if not isinstance(sample, bool):
                raise ValueError(f"{side}.{key} must be boolean")
            buttons[output] = sample
    return PicoFrame(timestamp, received, sequence, _pose(head.get("pose")),
                     _pose(left.get("pose")), _pose(right.get("pose")), skeleton,
                     mode, app.get("focus") is True, status,
                     MappingProxyType(inputs), MappingProxyType(buttons), raw_json)


def validate_frame(frame: PicoFrame, now: float | None = None,
                   previous: PicoFrame | None = None, max_age_s: float = .20,
                   max_hand_speed_m_s: float = 6.,
                   max_body_speed_m_s: float = 6.) -> dict:
    """Fail closed on stale, incomplete, switched or implausible input.

    A repeated *same sequence* may be held for control until max_age_s, but is
    never a fresh/recordable observation. A new callback with a repeated device
    timestamp is invalid. previous should be the last inspected distinct frame,
    including rejected frames, so timestamp/mode discontinuities stay visible.
    Speeds use packet timestamps and pose differences, never SDK va/wva fields.
    An orientation discontinuity above 30 rad/s is also rejected, treating q
    and -q as the same rotation. This deliberately stops rather than smooths
    tracking-mode/relocalization jumps into plausible-looking demonstrations.
    This is input quality screening, not a claim of retarget or task success.
    """
    now = time.monotonic() if now is None else float(now)
    limits = (max_age_s, max_hand_speed_m_s, max_body_speed_m_s)
    if not math.isfinite(now) or not all(math.isfinite(v) and v > 0 for v in limits):
        raise ValueError("Time and validation limits must be finite, limits positive")
    reasons, warnings = [], ["sdk_va_wva_not_used"]
    age = now - frame.host_received_s
    if age < -1e-6:
        reasons.append("host_time_in_future")
    if age > max_age_s:
        reasons.append("stale")
    if frame.input_mode != 1:
        reasons.append("controller_input_required")
    if not frame.app_focused:
        reasons.append("app_not_focused")
    if frame.head_status != 3:
        reasons.append("head_tracking_invalid")
    if frame.device_timestamp_ns is None or frame.device_timestamp_ns <= 0:
        reasons.append("device_timestamp_missing_or_invalid")

    pose_ok = {}
    for name, pose, shape in (("head", frame.head_pose7, (7,)),
                              ("left_controller", frame.left_pose7, (7,)),
                              ("right_controller", frame.right_pose7, (7,)),
                              ("body", frame.body24x7, (24, 7))):
        okay = pose is not None and pose.shape == shape and bool(np.isfinite(pose).all())
        pose_ok[name] = okay
        if not okay:
            reasons.append(f"{name}_missing_shape_or_nonfinite")
            continue
        norm = np.linalg.norm(pose[..., 3:7], axis=-1)
        if not np.all(np.abs(norm - 1.) <= .01):
            pose_ok[name] = False
            reasons.append(f"{name}_quaternion_not_unit")
        if name.endswith("controller") and np.linalg.norm(pose[:3]) < 1e-8:
            reasons.append(f"{name}_zero_placeholder")
        if name == "body" and np.max(np.ptp(pose[:, :3], axis=0)) < .05:
            reasons.append("body_collapsed_placeholder")
    for key, value in frame.inputs.items():
        lower = -1. if "axis" in key else 0.
        if not math.isfinite(value) or not lower <= value <= 1.:
            reasons.append(f"{key}_outside_range")

    new_frame = previous is None or frame.sequence != previous.sequence
    if previous is not None:
        if frame.sequence < previous.sequence:
            reasons.append("sequence_regressed")
        if frame.input_mode != previous.input_mode:
            reasons.append("input_mode_changed")
        if frame.host_received_s < previous.host_received_s:
            reasons.append("host_timestamp_regressed")
        if frame.device_timestamp_ns is not None and previous.device_timestamp_ns is not None:
            delta_ns = frame.device_timestamp_ns - previous.device_timestamp_ns
            if delta_ns < 0:
                reasons.append("device_timestamp_regressed")
            elif delta_ns == 0 and new_frame:
                reasons.append("device_timestamp_repeated")
            elif delta_ns > 0 and not new_frame:
                reasons.append("same_sequence_changed_timestamp")
            if delta_ns > 0:
                dt = delta_ns / 1e9
                # TCP may deliver several sampled packets in one burst: receive
                # intervals are not physical sample intervals. Use device time
                # for differences, rejecting a long gap/clock leap separately.
                if dt > max_age_s:
                    reasons.append("device_timestamp_gap")
                for name, current, older, limit in (
                    ("head", frame.head_pose7, previous.head_pose7, max_body_speed_m_s),
                    ("left_controller", frame.left_pose7, previous.left_pose7, max_hand_speed_m_s),
                    ("right_controller", frame.right_pose7, previous.right_pose7, max_hand_speed_m_s),
                    ("body", frame.body24x7, previous.body24x7, max_body_speed_m_s)):
                    if (pose_ok[name] and older is not None and older.shape == current.shape
                            and np.isfinite(older).all()):
                        speed = np.linalg.norm(current[..., :3] - older[..., :3], axis=-1) / dt
                        if np.max(speed) > limit:
                            reasons.append(f"{name}_position_jump")
                        old_q = older[..., 3:7]
                        old_norm = np.linalg.norm(old_q, axis=-1)
                        if np.all(np.abs(old_norm - 1.) <= .01):
                            q = current[..., 3:7]
                            cosine = np.abs(np.sum(q * old_q, axis=-1)) / (np.linalg.norm(q, axis=-1) * old_norm)
                            angular_speed = 2. * np.arccos(np.clip(cosine, 0., 1.)) / dt
                            if np.max(angular_speed) > 30.:
                                reasons.append(f"{name}_orientation_jump")
    fresh = new_frame and 0 <= age <= max_age_s
    valid = not reasons
    return {"valid": valid, "fresh": fresh, "recordable": valid and fresh,
            "new_frame": new_frame, "age_s": age, "reasons": reasons,
            "warnings": warnings}


class _DeviceState(C.Structure):
    _fields_ = [("devID", C.c_char * 32), ("stateJson", C.c_char * 16352)]


_Callback = C.CFUNCTYPE(None, C.c_void_p, C.c_int, C.c_int, C.c_void_p)


class PicoReceiver:
    """One read-only native SDK client; start/close never manage other services."""

    def __init__(self, sdk_path: str | Path = DEFAULT_SDK_PATH,
                 lock_path: str | Path | None = None):
        self.sdk_path = str(sdk_path)
        self.lock_path = Path(lock_path or f"/tmp/cvpr-pico-sdk-{os.getuid()}.lock")
        self._mutex = threading.Lock()
        self._lifecycle = threading.RLock()
        self._lock_file = self._sdk = self._callback = None
        self._started = False
        self._sequence = 0
        self._latest: tuple[bytes, float, int] | None = None

    def start(self) -> PicoReceiver:
        with self._lifecycle:
            if self._started:
                return self
            handle = self.lock_path.open("a+")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise RuntimeError("Another PicoReceiver owns the single-consumer SDK lock") from exc
            self._lock_file = handle
            try:
                self._sdk = C.CDLL(self.sdk_path)
                self._callback = _Callback(self._receive)
                self._sdk.PXREAInit.argtypes = [C.c_void_p, _Callback, C.c_uint]
                self._sdk.PXREAInit.restype = C.c_int
                self._sdk.PXREADeinit.argtypes = []
                self._sdk.PXREADeinit.restype = C.c_int
                result = self._sdk.PXREAInit(None, self._callback, 0xffffffff)
                if result != 0:
                    raise RuntimeError(f"PXREAInit failed: {result}")
                self._started = True
            except BaseException:
                self._sdk = self._callback = None
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()
                self._lock_file = None
                raise
        return self

    def _receive(self, context, kind, status, pointer):
        if kind == _TRACKING_EVENT and pointer:
            # Copy SDK-owned memory before returning from this callback. The
            # packet and receive time are published under one lock; no getters
            # can accidentally combine pose/body fields from different packets.
            received = time.monotonic()
            raw = bytes(C.cast(pointer, C.POINTER(_DeviceState)).contents.stateJson)
            with self._mutex:
                self._sequence += 1
                self._latest = (raw, received, self._sequence)
        elif kind in (1 << 3, 1 << 5):
            with self._mutex:
                self._latest = None

    def snapshot(self) -> PicoFrame | None:
        with self._mutex:
            packet = self._latest
        if packet is None:
            return None
        raw, received, sequence = packet
        return parse_tracking(raw, received, sequence)

    def close(self):
        with self._lifecycle:
            if self._started:
                result = self._sdk.PXREADeinit()
                if result != 0:
                    # Keep callback and exclusivity alive: native code may still
                    # call it after a failed deinit. A caller may retry close.
                    raise RuntimeError(f"PXREADeinit failed: {result}")
            self._started = False
            # Deinit has completed while the native callback is kept alive.
            self._sdk = self._callback = None
            with self._mutex:
                self._latest = None
            if self._lock_file is not None:
                fcntl.flock(self._lock_file, fcntl.LOCK_UN)
                self._lock_file.close()
                self._lock_file = None

    def __enter__(self) -> PicoReceiver:
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
