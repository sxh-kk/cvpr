"""One frozen preparation boundary before the first real captured sample.

This module validates supplied evidence only: no SDK calls, files, model
inference or physics.  It excludes preparation wall time by *shifting* the old
deadline, preserving earlier lateness.  The caller must establish that shifted
deadline before encoding/appending the first frame, retain encoder history,
honour pending operator commands, and save the real subsequent warmup.

The receiver is latest-only.  Endpoint validation does not prove that every SDK
packet between the two endpoints was inspected, nor that a task succeeded.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
import numbers
import time
import uuid as uuidlib

import numpy as np

from pico_live_input import PicoFrame, validate_frame


CONTRACT = "cvpr-pico-frozen-capture-start-v1"
MAX_TRACKING_GAP_S = .20


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(name + " must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + " must be a finite real number")
    return result


def _count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return int(value)


def _state_bytes(value):
    array = np.asarray(value)
    if array.ndim != 1 or not array.size or array.dtype.kind != "f" or array.dtype.itemsize != 8:
        # Full MuJoCo integration snapshots are float64. Conversion of uint64
        # or extended precision could hide a real change in the original data.
        raise ValueError("Integration state must be a nonempty float64 vector")
    array = np.asarray(array, dtype="<f8")
    if not np.isfinite(array).all():
        raise ValueError("Integration state must be finite")
    return array.tobytes(order="C"), int(array.size)


def _source_evidence(frame):
    return {"sequence": frame.sequence, "device_timestamp_ns": frame.device_timestamp_ns,
        "host_received_s": frame.host_received_s,
        "raw_packet": frame.raw_payload,
        "raw_packet_sha256": hashlib.sha256(frame._raw_json.encode("utf-8")).hexdigest(),
        "input_mode": frame.input_mode, "app_focused": frame.app_focused,
        "head_status": frame.head_status}


class CaptureStartGate:
    """Consume exactly one attempt to finish a not-yet-captured preparation.

    ``integration_state`` must be the actual full mjSTATE_INTEGRATION snapshot,
    and counters must describe the actual writers, not estimated frame counts.
    The gate cannot ascertain the origin of fabricated arguments.  Its receipt
    belongs in independent immutable start evidence; do not mutate the metadata
    already copied into the capture and command ledger.
    """

    def __init__(self, uuid, global_tick50, sim_time_s, substeps, integration_state,
                 source_frame, clock=time.monotonic):
        if not callable(clock):
            raise ValueError("clock must be callable")
        self._clock = clock
        self._started_s = _number(clock(), "setup start time")
        self._uuid = str(uuidlib.UUID(str(uuid)))
        self._global_tick = _count(global_tick50, "global_tick50")
        if self._global_tick % 5:
            raise ValueError("Capture start must align with the shared 50/30Hz grid")
        self._sim_time = _number(sim_time_s, "sim_time_s")
        self._substeps = _count(substeps, "substeps")
        if self._sim_time < 0 or self._substeps % 4:
            raise ValueError("Capture start needs a nonnegative time and completed four-substep tick")
        self._state, self._state_width = _state_bytes(integration_state)
        if not isinstance(source_frame, PicoFrame):
            raise ValueError("source_frame must be one parsed PICO packet")
        initial_quality = validate_frame(source_frame, now=self._started_s)
        if not initial_quality["recordable"]:
            raise ValueError("Invalid source before setup: " + ",".join(initial_quality["reasons"]))
        # PicoFrame owns immutable byte-backed pose arrays and immutable input
        # mappings. Keep this endpoint, rather than replacing it with the new one.
        self._source = source_frame
        self._attempted = False

    def finish(self, old_deadline, *, after_global_tick, after_sim_time,
               after_substeps, after_state, frame, quality, event_count,
               ledger_event_count, quality_bytes, force_bytes, max_lateness=.1):
        if self._attempted:
            raise ValueError("Capture-start gate has already been consumed")
        # Failed evidence cannot be repaired by retrying the same frozen gate.
        self._attempted = True
        finished_s = _number(self._clock(), "setup finish time")
        elapsed = finished_s - self._started_s
        if elapsed < 0 or elapsed > MAX_TRACKING_GAP_S:
            raise ValueError("Setup wall time is reversed or exceeds the 200ms tracking gap")
        deadline = _number(old_deadline, "old_deadline")
        budget = _number(max_lateness, "max_lateness")
        if not 0 < budget <= .1:
            raise ValueError("max_lateness may not exceed the original 100ms limit")
        old_lag = self._started_s - deadline
        if old_lag > budget:
            raise ValueError("Existing lateness already exceeds the real-time budget")
        if (_count(after_global_tick, "after_global_tick") != self._global_tick
                or _number(after_sim_time, "after_sim_time") != self._sim_time
                or _count(after_substeps, "after_substeps") != self._substeps):
            raise ValueError("Physics clock or tick changed during frozen setup")
        state, width = _state_bytes(after_state)
        if state != self._state or width != self._state_width:
            raise ValueError("Integrated physical state changed during frozen setup")
        counts = {"event_count": _count(event_count, "event_count"),
            "ledger_event_count": _count(ledger_event_count, "ledger_event_count"),
            "quality_bytes": _count(quality_bytes, "quality_bytes"),
            "force_bytes": _count(force_bytes, "force_bytes")}
        if counts != {"event_count": 0, "ledger_event_count": 1,
                      "quality_bytes": 0, "force_bytes": 0}:
            raise ValueError("Setup exclusion is forbidden after capture/writer data starts")
        if not isinstance(frame, PicoFrame):
            raise ValueError("A fresh parsed PICO packet is required after setup")
        actual_quality = validate_frame(frame, now=finished_s, previous=self._source)
        if (not isinstance(quality, Mapping)
                or any(not isinstance(quality.get(key), (bool,np.bool_)) or not bool(quality[key])
                       for key in ("valid", "fresh", "recordable"))
                or quality.get("reasons") != [] or not actual_quality["recordable"]):
            raise ValueError("Fresh full-body continuity rejected: " + ",".join(actual_quality["reasons"]))
        shifted_deadline = deadline + elapsed
        remaining_lag = finished_s - shifted_deadline
        # Exclusion does not reset earlier lateness to zero. Tight tolerance
        # accounts only for rounding of large monotonic second values.
        tolerance = 8 * max(math.ulp(finished_s), math.ulp(deadline), math.ulp(shifted_deadline))
        if abs(remaining_lag - old_lag) > tolerance:
            raise ValueError("Deadline shift did not preserve previous lateness")
        receipt = {"contract": CONTRACT, "human_demo_uuid": self._uuid,
            "accepted_new_dataset": False, "eligible_for_packing": False,
            "frozen_before_first_capture": True, "single_use": True,
            "setup_started_monotonic_s": self._started_s,
            "setup_finished_monotonic_s": finished_s, "setup_wall_s": elapsed,
            "old_deadline_monotonic_s": deadline,
            "shifted_deadline_monotonic_s": shifted_deadline,
            "old_lateness_s": old_lag, "preserved_lateness_s": remaining_lag,
            "maximum_lateness_s": budget, "maximum_tracking_gap_s": MAX_TRACKING_GAP_S,
            "global_tick50_before": self._global_tick, "global_tick50_after": self._global_tick,
            "sim_time_s_before": self._sim_time, "sim_time_s_after": self._sim_time,
            "substeps_before": self._substeps, "substeps_after": self._substeps,
            "integration_state_sha256_before": hashlib.sha256(self._state).hexdigest(),
            "integration_state_sha256_after": hashlib.sha256(state).hexdigest(),
            "integration_state_width": width,
            "integration_state_encoding": "little-endian float64 contiguous bytes",
            "writer_counts_before_first_capture": counts,
            "source_before": _source_evidence(self._source), "source_after": _source_evidence(frame),
            "source_sequence_delta": frame.sequence - self._source.sequence,
            "source_device_delta_s": (frame.device_timestamp_ns - self._source.device_timestamp_ns)/1e9,
            "source_host_received_delta_s": frame.host_received_s - self._source.host_received_s,
            "source_endpoint_quality": actual_quality,
            "all_intermediate_sdk_packets_inspected": False,
            "captured_grid_samples_before_commit": 0,
            "limitation": "Endpoint-only tracking validation; encoder history, first-frame processing, real warmup and subsequent recording deadlines remain the caller's responsibility"}
        return shifted_deadline, receipt
