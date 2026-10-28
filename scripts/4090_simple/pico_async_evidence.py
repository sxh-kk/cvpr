"""FIFO CPU evidence adapter; live models/controllers never cross processes.

The parent freezes actual measurements and command values at submission.  The
worker alone owns JSON/files and the independent deferred-capture FK scratch.
Successful raw capture completion is a barrier over all preceding evidence,
including the session trace; it remains pending independent RTX/data admission.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
import uuid

from pico_deferred_capture import (DeferredStateCapture, PreparedDeferredModel,
                                   _action, _validate_header)
from pico_evidence_writer import EvidenceWriter, EvidenceWriterError
from pico_teleop_recording import (CaptureRejected, CommandLedger, _json_value,
                                  _snapshot)

CONTRACT = "cvpr-pico-async-deferred-evidence-v1"
_FLAGS = {"accepted_new_dataset": False, "raw_pico_va_wva_validated": False,
          "raw_derivative_policy": "archived_untrusted_not_control_or_labels"}


class _ArchivedCommand:
    def __init__(self, kind, **parameters):
        self.type, self.parameters = kind, parameters


def _restore_action(action):
    if action is None:
        return None
    # Sentinel satisfies capture's structural validation.  No controller method
    # or inference path exists in this worker.
    return _ArchivedCommand("sonic_v1_1", controller=object(),
        body_token=action["body_token64"],
        hand_targets=dict(zip(action["hand_joint_names"], action["hand14"])))


class _WorkerLedger:
    def __init__(self, path, human_demo_uuid):
        self.path, self.uuid, self.count = Path(path), human_demo_uuid, 0
        self.stream = self.path.open("x", buffering=1)
        self.receipt = None

    def append(self, event):
        event = _json_value(event)
        if (event.get("event_index") != self.count
                or event.get("human_demo_uuid") != self.uuid
                or type(event.get("recorded_monotonic_ns")) is not int
                or event["recorded_monotonic_ns"] < 0
                or not isinstance(event.get("event_type"), str) or not event["event_type"]
                or any(event.get(key) != value for key, value in _FLAGS.items())):
            raise ValueError("Frozen command ledger event identity/index/clock mismatch")
        self.stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self.count += 1

    def close(self):
        if self.receipt is None:
            try:
                self.stream.flush()
            finally:
                self.stream.close()
            self.receipt = {"path": str(self.path), "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "event_count": self.count, "human_demo_uuid": self.uuid,
                "accepted_new_dataset": False, "raw_pico_va_wva_validated": False}
        return self.receipt


class _EvidenceHandler:
    def __init__(self, *, model_path, model_sha256, joint_names, metadata,
                 file_identity, trace_path=None):
        self.prepared = PreparedDeferredModel.from_archive(model_path,
            model_sha256=model_sha256, joint_names=joint_names,
            metadata=metadata, file_identity=file_identity)
        self.trace = None
        if trace_path is not None:
            self.trace = Path(trace_path).open("a", buffering=1)
        self.capture = self.ledger = None
        self.sinks = {}
        self.capture_manifest = None
        self.ledger_receipt = None
        self.closed = False

    def _loaded_receipt(self):
        return {"contract": CONTRACT, "model_path": str(self.prepared.model_path),
            "model_sha256": self.prepared.model_sha256,
            "file_identity": list(self.prepared._stat),
            "state_spec": self.prepared.spec, "state_width": self.prepared.verifier.width,
            "joint_names": self.prepared.names,
            "object_body_names": {role: self.prepared.model.body(body).name
                for role, body in self.prepared.verifier.objects.items()},
            "model_loaded_from_authenticated_bytes": True,
            "live_model_received": False, "controller_inference_available": False,
            "accepted_new_dataset": False}

    def _stream(self, name):
        if name == "trace":
            if self.trace is None:
                raise ValueError("No session trace was configured")
            return self.trace
        if name not in self.sinks:
            raise ValueError("No current episode stream: " + name)
        return self.sinks[name]

    def _finish_episode(self, completed):
        if self.capture is None:
            raise ValueError("No current episode")
        if self.capture.closed:
            return {"capture": self.capture_manifest, "ledger": self.ledger_receipt}
        errors = []
        for stream in self.sinks.values():
            try:
                if not stream.closed:
                    try:
                        stream.flush()
                    finally:
                        stream.close()
            except Exception as error:
                errors.append(type(error).__name__ + ": " + str(error))
        try:
            if self.ledger is not None:
                self.ledger_receipt = self.ledger.close()
        except Exception as error:
            errors.append(type(error).__name__ + ": " + str(error))
        try:
            if self.trace is not None and not self.trace.closed:
                self.trace.flush()
        except Exception as error:
            errors.append(type(error).__name__ + ": " + str(error))
        if errors:
            self.capture._error = self.capture.error or "Evidence finalization failed: " + "; ".join(errors)
        self.capture_manifest = self.capture.close(completed=completed)
        if errors or (completed and self.capture.failed):
            raise CaptureRejected(self.capture.error or "; ".join(errors))
        return {"capture": self.capture_manifest, "ledger": self.ledger_receipt}

    def handle(self, op, payload):
        try:
            return self._handle(op, payload)
        except Exception as error:
            if self.capture is not None and not self.capture.closed:
                self.capture._error = self.capture.error or type(error).__name__ + ": " + str(error)
            raise

    def _handle(self, op, payload):
        if op == "identity":
            return self._loaded_receipt()
        if op == "begin":
            if self.capture is not None and not self.capture.closed:
                raise ValueError("Previous episode was not closed")
            self.capture_manifest = self.ledger_receipt = None
            self.sinks = {}
            episode_directory = Path(payload["directory"])
            episode_directory.mkdir(parents=True, exist_ok=True)
            self.capture = DeferredStateCapture(episode_directory / "deferred_capture",
                metadata=payload["metadata"], joint_names=payload["joint_names"],
                warmup_ticks=payload["warmup_ticks"], start_time=payload["start_time"],
                model=self.prepared.model, prepared_model=self.prepared)
            directory = self.capture.directory
            self.ledger = _WorkerLedger(directory.parent / "commands.jsonl", payload["human_demo_uuid"])
            self.ledger.append(payload["session_event"])
            self.sinks["quality"] = (directory.parent / "raw_quality_samples.jsonl").open("x", buffering=1)
            self.sinks["force"] = (directory.parent / "applied_force_samples.jsonl").open("x", buffering=1)
            return {"directory": str(directory), "ledger_count": self.ledger.count,
                    "quality_tell": self.sinks["quality"].tell(), "force_tell": self.sinks["force"].tell()}
        if op == "capture_append":
            fields = dict(payload)
            fields["command"] = _restore_action(fields.pop("actual_action"))
            self.capture.append(**fields)
            return None
        if op == "ledger_append":
            self.ledger.append(payload)
            return None
        if op in ("sink_write", "sink_json", "sink_tell", "sink_close"):
            stream = self._stream(payload["name"])
            if op == "sink_write":
                stream.write(payload["value"])
            elif op == "sink_json":
                stream.write(json.dumps(_json_value(payload["value"]), ensure_ascii=False,
                                        allow_nan=False) + "\n")
            elif op == "sink_tell":
                stream.flush()
                return stream.tell()
            elif not stream.closed:
                stream.flush()
                stream.close()
            return None
        if op == "capture_close":
            return self._finish_episode(payload["completed"])
        if op == "ledger_close":
            self.ledger_receipt = self.ledger.close()
            return self.ledger_receipt
        raise ValueError("Unknown evidence operation: " + str(op))

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                if self.capture is not None and not self.capture.closed:
                    self._finish_episode(False)
            finally:
                if self.trace is not None and not self.trace.closed:
                    try:
                        self.trace.flush()
                    finally:
                        self.trace.close()
        return {"contract": CONTRACT, "capture": self.capture_manifest,
                "ledger": self.ledger_receipt, "accepted_new_dataset": False}


def evidence_handler_factory(**kwargs):
    return _EvidenceHandler(**kwargs)


class _Sink:
    def __init__(self, session, name):
        self.session, self.name, self.closed = session, name, False

    def _payload(self, value=None):
        if self.closed:
            self.session._fail(ValueError("Evidence stream is closed"))
        return {"name": self.name, "value": value}

    def write_json(self, value):
        self.session._submit("sink_json", self._payload(value))

    def write(self, value):
        if not isinstance(value, str):
            self.session._fail(TypeError("Evidence stream.write requires a string"))
        self.session._submit("sink_write", self._payload(value))
        return len(value)

    def tell(self):
        return self.session._call("sink_tell", self._payload())

    def close(self):
        if not self.closed:
            self.session._call("sink_close", {"name": self.name})
            self.closed = True


class _Ledger:
    def __init__(self, session, human_demo_uuid):
        self.session, self.uuid = session, str(uuid.UUID(human_demo_uuid))
        self._count, self.closed, self._receipt = 0, False, None

    def _event(self, event_type, fields, recorded_monotonic_ns):
        if (not isinstance(event_type, str) or not event_type.strip()
                or set(fields) & CommandLedger._RESERVED):
            raise ValueError("Ledger type must be nonempty and reserved identity fields cannot be overridden")
        if type(recorded_monotonic_ns) is not int or recorded_monotonic_ns < 0:
            raise ValueError("Recorded monotonic time must be a nonnegative integer")
        return {**fields, "event_index": self._count, "event_type": event_type,
            "human_demo_uuid": self.uuid, "recorded_monotonic_ns": recorded_monotonic_ns, **_FLAGS}

    def append(self, event_type, **fields):
        return self.append_recorded(event_type, recorded_monotonic_ns=time.monotonic_ns(), **fields)

    def append_recorded(self, event_type, *, recorded_monotonic_ns, **fields):
        """Explicit fixed timestamp hook for CPU archive parity, never worker time."""
        try:
            if self.closed:
                raise ValueError("Ledger is closed")
            event = self._event(event_type, fields, recorded_monotonic_ns)
            self.session._submit("ledger_append", event)
            self._count += 1
            return event
        except Exception as error:
            self.session._fail(error)

    def close(self):
        if not self.closed:
            self._receipt = self.session._call("ledger_close")
            self.closed = True
        return _snapshot(self._receipt)


class _Capture:
    def __init__(self, session, directory, names, warmup_ticks, start_time):
        self.session, self.directory = session, Path(directory)
        self.names, self.warmup_ticks, self.start_time = list(names), warmup_ticks, float(start_time)
        self.count = self.warmup_count = self.command_count = 0
        self.closed, self.manifest = False, None

    @property
    def failed(self):
        return self.session.failed

    @property
    def error(self):
        return self.session.error

    def append(self, **fields):
        try:
            if self.closed:
                raise ValueError("Capture is closed")
            fields = dict(fields)
            fields["actual_action"] = _action(fields.pop("command"))
            self.session._submit("capture_append", fields)
            self.count += 1
            warmup = type(fields["tick50"]) is int and fields["tick50"] < self.warmup_ticks
            self.warmup_count += int(warmup)
            self.command_count += int(not warmup)
        except Exception as error:
            self.session._fail(error)

    enqueue = append

    def close(self, *, completed):
        if type(completed) is not bool:
            raise TypeError("completed must be a boolean")
        if self.closed:
            return _snapshot(self.manifest)
        try:
            result = self.session._call("capture_close", {"completed": completed},
                                        timeout_s=self.session.finalize_timeout_s)
            self.manifest, self.closed = result["capture"], True
            self.session._ledger._receipt = result["ledger"]
            self.session._ledger.closed = True
            for sink in self.session._episode_sinks:
                sink.closed = True
            return _snapshot(self.manifest)
        except Exception:
            # Closing the worker invokes handler cleanup with completed=False.
            # A failed RPC must never return a seemingly completed manifest.
            self.session.close()
            raise


class DeferredEvidenceSession:
    def __init__(self, prepared_model, *, cpus=None, trace_path=None,
                 max_pending=64, startup_timeout_s=15., finalize_timeout_s=30.):
        if not isinstance(prepared_model, PreparedDeferredModel) or prepared_model.in_use:
            raise ValueError("Need an idle prepared physical model")
        if prepared_model.model_path.is_symlink() or prepared_model._file_identity() != prepared_model._stat:
            raise ValueError("Prepared model archive changed")
        if (isinstance(finalize_timeout_s, bool) or not isinstance(finalize_timeout_s, (int, float))
                or not math.isfinite(finalize_timeout_s) or finalize_timeout_s <= 0):
            raise ValueError("Need a positive finalization timeout")
        self.finalize_timeout_s = float(finalize_timeout_s)
        self._error, self._closed, self._close_receipt = None, False, None
        self._prepared_ref = prepared_model
        self._capture = self._ledger = None
        self._episode_sinks = []
        bindings = {role: prepared_model.model.body(body).name
                    for role, body in prepared_model.verifier.objects.items()}
        self.writer = EvidenceWriter(evidence_handler_factory,
            init_kwargs={"model_path": str(prepared_model.model_path.absolute()),
                "model_sha256": prepared_model.model_sha256, "joint_names": prepared_model.names,
                "metadata": {"object_body_names": bindings}, "file_identity": prepared_model._stat,
                "trace_path": None if trace_path is None else str(Path(trace_path).absolute())},
            cpus=cpus, max_pending=max_pending, startup_timeout_s=startup_timeout_s)
        try:
            self._identity = self.writer.call("identity")
        except Exception:
            self.writer.close()
            raise
        prepared_model.in_use = True
        self.trace = None if trace_path is None else _Sink(self, "trace")

    @property
    def error(self):
        return self._error or self.writer.error

    @property
    def failed(self):
        return self.error is not None

    def _fail(self, error):
        self._error = self._error or type(error).__name__ + ": " + str(error)
        raise EvidenceWriterError(self._error) from error

    def check(self):
        if self._error:
            raise EvidenceWriterError(self._error)
        return self.writer.check()

    def receipt(self):
        return _snapshot({"contract": CONTRACT, "loaded_model": self._identity,
                          "writer": self.writer.check(), "accepted_new_dataset": False})

    def _submit(self, op, payload=None):
        try:
            self.check()
            return self.writer.submit(op, payload)
        except Exception as error:
            self._fail(error)

    def _call(self, op, payload=None, *, timeout_s=5.):
        try:
            self.check()
            return self.writer.call(op, payload, timeout_s=timeout_s)
        except Exception as error:
            self._fail(error)

    def begin(self, directory, *, metadata, joint_names, warmup_ticks, start_time,
              human_demo_uuid, recorded_monotonic_ns=None):
        try:
            if self._capture is not None and not self._capture.closed:
                raise ValueError("Previous episode is not closed")
            _validate_header(list(joint_names), warmup_ticks, start_time)
            ledger = _Ledger(self, human_demo_uuid)
            timestamp = time.monotonic_ns() if recorded_monotonic_ns is None else recorded_monotonic_ns
            event = ledger._event("session", {"metadata": metadata}, timestamp)
            self._call("begin", {"directory": str(Path(directory).absolute()), "metadata": metadata,
                "joint_names": list(joint_names), "warmup_ticks": warmup_ticks, "start_time": start_time,
                "human_demo_uuid": ledger.uuid, "session_event": event})
            ledger._count = 1
            self._ledger = ledger
            self._capture = _Capture(self, Path(directory) / "deferred_capture", joint_names, warmup_ticks, start_time)
            self._episode_sinks = [_Sink(self, "quality"), _Sink(self, "force")]
            return self._capture, ledger, *self._episode_sinks
        except Exception as error:
            self._fail(error)

    def close(self, *, timeout_s=None):
        if not self._closed:
            self._close_receipt = self.writer.close(timeout_s=self.finalize_timeout_s if timeout_s is None else timeout_s)
            if self._error:
                self._close_receipt = {**self._close_receipt, "status": "failed",
                    "completed": False, "adapter_error": self._error}
            self._prepared_ref.in_use = False
            self._closed = True
        return _snapshot(self._close_receipt)
