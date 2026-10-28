"""Real CPU spawn/FK parity tests; no SDK, rendering, policy or GPU."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_pico_deferred_capture as fixture
from pico_async_evidence import DeferredEvidenceSession, _EvidenceHandler
from pico_deferred_capture import DeferredStateCapture, PreparedDeferredModel, PENDING, load_deferred, prepare_deferred_model
from pico_evidence_writer import EvidenceWriterError
from pico_teleop_recording import CaptureRejected, CommandLedger
from replay_handover_teacher import capture_tick_index


UUID = "31e4d023-794d-4127-8ea2-0c30d91548a4"


class AsyncEvidenceTests(unittest.TestCase):
    def prepare(self, directory):
        model = fixture.make_model()
        metadata = {"source_kind": "live_pico_human", "human_demo_uuid": UUID,
            "object_body_names": {"target": "target", "container": "container"},
            "nested": {"actual": [1, 2]}}
        prepared = prepare_deferred_model(Path(directory) / "prepared", model=model,
            joint_names=fixture.NAMES, metadata=metadata)
        return model, mujoco.MjData(model), prepared, metadata

    def session(self, prepared, **kwargs):
        return DeferredEvidenceSession(prepared, cpus=[min(os.sched_getaffinity(0))],
            max_pending=1024, **kwargs)

    def state(self, model, data):
        result = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(model, data, result, mujoco.mjtState.mjSTATE_INTEGRATION)
        return result

    def test_real_spawn_matches_synchronous_all_values_and_state_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, data, prepared, metadata = self.prepare(root)
            original_model = prepared.model_path.read_bytes()
            trace = root / "control_trace.jsonl"
            trace.write_text('{"already_written_warmup": true}\n')
            session = self.session(prepared, trace_path=trace)
            try:
                identity = session.receipt()
                self.assertEqual(identity["loaded_model"]["model_sha256"], prepared.model_sha256)
                self.assertNotEqual(identity["writer"]["worker"]["pid"], os.getpid())
                self.assertEqual(identity["writer"]["worker"]["CUDA_VISIBLE_DEVICES"], "")
                self.assertEqual(identity["writer"]["worker"]["nice"], 19)
                capture, ledger, quality, force = session.begin(root / "async", metadata=metadata,
                    joint_names=fixture.NAMES, warmup_ticks=50, start_time=1.25,
                    human_demo_uuid=UUID, recorded_monotonic_ns=100)
                self.assertEqual((capture.count, ledger._count, quality.tell(), force.tell()), (0, 1, 0, 0))
                (root / "sync").mkdir()
                synchronous = DeferredStateCapture(root / "sync/deferred_capture", metadata=metadata,
                    joint_names=fixture.NAMES, warmup_ticks=50, start_time=1.25, model=model)
                with mock.patch("pico_teleop_recording.time.monotonic_ns", return_value=100):
                    sync_ledger = CommandLedger(root / "sync/commands.jsonl", human_demo_uuid=UUID, metadata=metadata)
                for tick in range(51):
                    if capture_tick_index(tick) is None:
                        continue
                    values = fixture.DeferredCaptureTests().frame(capture, model, data, tick)
                    before = self.state(model, data)
                    synchronous.append(**values)
                    capture.append(**values)
                    self.assertTrue(np.array_equal(before, self.state(model, data)))
                    fields = {"actual_token": np.arange(64, dtype=np.float32), "raw": {"nested": [tick]}}
                    with mock.patch("pico_teleop_recording.time.monotonic_ns", return_value=1000 + tick):
                        sync_ledger.append("command", **fields)
                    ledger.append_recorded("command", recorded_monotonic_ns=1000 + tick, **fields)
                    q, f, t = {"native": False, "tick": tick}, {"applied": [0.] * 6, "tick": tick}, {"tick": tick}
                    quality.write_json(q)
                    force.write_json(f)
                    session.trace.write_json(t)
                    # Mutations after submit must never reach the child.
                    values["integration_state"][:] = -999
                    values["physical_qpos"][:] = -999
                    values["source_fields"]["source_raw_packet"]["va"][0] = -999
                    if values["command"] is not None:
                        values["command"].parameters["body_token"][:] = -999
                    fields["raw"]["nested"][0] = -999
                    q["tick"] = f["tick"] = t["tick"] = -999
                # Reconstruct the simple tick sequence after caller mutation.
                ticks = [tick for tick in range(51) if capture_tick_index(tick) is not None]
                manifest = capture.close(completed=True)
                sync_manifest = synchronous.close(completed=True)
                self.assertEqual(manifest, sync_manifest)
                self.assertEqual(manifest["status"], PENDING)
                self.assertEqual((capture.count, ledger._count), (31, 32))
                self.assertEqual(capture.close(completed=False), manifest)
                quality.close(); force.close()
                async_ledger = ledger.close()
                sync_ledger_receipt = sync_ledger.close()
                self.assertEqual(async_ledger["sha256"], sync_ledger_receipt["sha256"])
                self.assertEqual(set(async_ledger), set(sync_ledger_receipt))
                for name in ("events.jsonl", "states.npy"):
                    self.assertEqual((root / "async/deferred_capture" / name).read_bytes(),
                                     (root / "sync/deferred_capture" / name).read_bytes())
                self.assertEqual([json.loads(x) for x in (root / "async/raw_quality_samples.jsonl").read_text().splitlines()],
                                 [{"native": False, "tick": tick} for tick in ticks])
                self.assertEqual([json.loads(x) for x in (root / "async/applied_force_samples.jsonl").read_text().splitlines()],
                                 [{"applied": [0.] * 6, "tick": tick} for tick in ticks])
                self.assertEqual([json.loads(x) for x in trace.read_text().splitlines()],
                                 [{"already_written_warmup": True}] + [{"tick": tick} for tick in ticks])
                self.assertEqual(prepared.model_path.read_bytes(), original_model)
                self.assertEqual(load_deferred(capture.directory)["manifest"]["status"], PENDING)
            finally:
                closed = session.close()
            self.assertTrue(closed["completed"])
            self.assertFalse(prepared.in_use)

    def test_model_archive_loader_authenticates_exact_independent_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            model, data, prepared, metadata = self.prepare(temp)
            before = prepared.model_path.read_bytes()
            loaded = PreparedDeferredModel.from_archive(prepared.model_path,
                model_sha256=prepared.model_sha256, joint_names=fixture.NAMES,
                metadata=metadata, file_identity=prepared._stat)
            self.assertIsNot(loaded.model, model)
            self.assertIsNot(loaded.verifier.data, data)
            self.assertEqual(loaded.verifier.width, prepared.verifier.width)
            self.assertEqual(before, prepared.model_path.read_bytes())
            for changes in ({"model_sha256": "0" * 64}, {"file_identity": [0, 0, 0, 0]},
                            {"metadata": {"object_body_names": {"target": "container", "container": "container"}}}):
                arguments = dict(model_sha256=prepared.model_sha256, joint_names=fixture.NAMES,
                                 metadata=metadata, file_identity=prepared._stat)
                arguments.update(changes)
                with self.assertRaises((ValueError, KeyError)):
                    PreparedDeferredModel.from_archive(prepared.model_path, **arguments)
            link = Path(temp) / "link.mjb"
            link.symlink_to(prepared.model_path)
            with self.assertRaises(OSError):
                PreparedDeferredModel.from_archive(link, model_sha256=prepared.model_sha256,
                    joint_names=fixture.NAMES, metadata=metadata)

    def test_worker_fk_failure_and_stream_error_cannot_complete_capture(self):
        for failure in ("FK", "stream_json"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                model, data, prepared, metadata = self.prepare(temp)
                session = self.session(prepared)
                capture, ledger, quality, force = session.begin(Path(temp) / "episode", metadata=metadata,
                    joint_names=fixture.NAMES, warmup_ticks=50, start_time=1.25, human_demo_uuid=UUID)
                values = fixture.DeferredCaptureTests().frame(capture, model, data, 0)
                if failure == "FK":
                    values["objects"]["target"]["pose7"][0] += .5
                    capture.append(**values)
                else:
                    capture.append(**values)
                    quality.write_json({"invalid": float("nan")})
                with self.assertRaises(EvidenceWriterError):
                    capture.close(completed=True)
                self.assertTrue(capture.failed)
                receipt = session.close()
                self.assertFalse(receipt["completed"])
                manifest = json.loads((capture.directory / "capture.json").read_text())
                self.assertEqual(manifest["status"], "partial_capture_rejected")
                self.assertIsNotNone(manifest["failure"])

    def test_parent_command_failure_is_sticky_and_closes_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            model, data, prepared, metadata = self.prepare(temp)
            session = self.session(prepared)
            capture, ledger, quality, force = session.begin(Path(temp) / "episode", metadata=metadata,
                joint_names=fixture.NAMES, warmup_ticks=50, start_time=1.25, human_demo_uuid=UUID)
            values = fixture.DeferredCaptureTests().frame(capture, model, data, 0)
            values["command"] = object()
            with self.assertRaises(EvidenceWriterError):
                capture.append(**values)
            with self.assertRaises(EvidenceWriterError):
                session.check()
            self.assertFalse(session.close()["completed"])
            self.assertEqual(json.loads((capture.directory / "capture.json").read_text())["status"], "partial_capture_rejected")

    def test_warmup_only_and_ledger_write_failure_cannot_complete(self):
        for failure in ("warmup_only", "ledger_json"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                model, data, prepared, metadata = self.prepare(temp)
                session = self.session(prepared)
                capture, ledger, quality, force = session.begin(Path(temp) / "episode", metadata=metadata,
                    joint_names=fixture.NAMES, warmup_ticks=50, start_time=1.25, human_demo_uuid=UUID)
                capture.append(**fixture.DeferredCaptureTests().frame(capture, model, data, 0))
                if failure == "ledger_json":
                    ledger.append("command", invalid=float("nan"))
                with self.assertRaises(EvidenceWriterError):
                    capture.close(completed=True)
                self.assertFalse(session.close()["completed"])
                manifest = json.loads((capture.directory / "capture.json").read_text())
                self.assertEqual(manifest["status"], "partial_capture_rejected")
                self.assertIsNotNone(manifest["failure"])

    def test_multiple_episodes_and_new_session_append_trace_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            model, data, prepared, metadata = self.prepare(temp)
            trace = Path(temp) / "trace.jsonl"
            trace.write_text("old\n")
            for reset in range(2):
                session = self.session(prepared, trace_path=trace)
                for episode in range(2):
                    session.trace.write_json({"reset": reset, "episode": episode})
                    capture, ledger, quality, force = session.begin(Path(temp) / f"episode_{reset}_{episode}",
                        metadata=metadata, joint_names=fixture.NAMES, warmup_ticks=50,
                        start_time=1.25, human_demo_uuid=UUID)
                    values = fixture.DeferredCaptureTests().frame(capture, model, data, 0)
                    capture.append(**values)
                    self.assertEqual(capture.close(completed=False)["status"], "partial_capture_rejected")
                    self.assertEqual(ledger.close()["event_count"], 1)
                    quality.close(); force.close()
                self.assertTrue(session.close()["completed"])
            self.assertEqual(trace.read_text().splitlines(), ["old"] +
                [json.dumps({"reset": r, "episode": e}) for r in range(2) for e in range(2)])

    def test_worker_death_cannot_be_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            model, data, prepared, metadata = self.prepare(temp)
            session = self.session(prepared)
            capture, *_ = session.begin(Path(temp) / "episode", metadata=metadata,
                joint_names=fixture.NAMES, warmup_ticks=50, start_time=1.25, human_demo_uuid=UUID)
            session.writer._process.terminate()
            session.writer._process.join(2.)
            with self.assertRaises(EvidenceWriterError):
                capture.close(completed=True)
            self.assertFalse(session.close()["completed"])

    def test_close_flush_failure_prevents_pending_rtx_and_releases_files(self):
        class FailingFlush:
            def __init__(self, stream):
                self.stream = stream

            @property
            def closed(self):
                return self.stream.closed

            def flush(self):
                raise OSError("fixture disk flush failed")

            def close(self):
                self.stream.close()

        with tempfile.TemporaryDirectory() as temp:
            model, data, prepared, metadata = self.prepare(temp)
            handler = _EvidenceHandler(model_path=prepared.model_path, model_sha256=prepared.model_sha256,
                joint_names=fixture.NAMES, metadata=metadata, file_identity=prepared._stat)
            session_event = {"metadata": metadata, "event_index": 0, "event_type": "session",
                "human_demo_uuid": UUID, "recorded_monotonic_ns": 123,
                "accepted_new_dataset": False, "raw_pico_va_wva_validated": False,
                "raw_derivative_policy": "archived_untrusted_not_control_or_labels"}
            handler.handle("begin", {"directory": str(Path(temp) / "episode"), "metadata": metadata,
                "joint_names": fixture.NAMES, "warmup_ticks": 50, "start_time": 1.25,
                "human_demo_uuid": UUID, "session_event": session_event})
            fixture.DeferredCaptureTests().fill(handler.capture, model, data)
            real_stream = handler.sinks["quality"]
            handler.sinks["quality"] = FailingFlush(real_stream)
            with self.assertRaisesRegex(CaptureRejected, "disk flush failed"):
                handler.handle("capture_close", {"completed": True})
            manifest = json.loads((handler.capture.directory / "capture.json").read_text())
            self.assertEqual(manifest["status"], "partial_capture_rejected")
            self.assertTrue(manifest["requested_completed"])
            self.assertTrue(real_stream.closed)
            handler.close()

    def test_old_closed_proxies_cannot_write_into_new_episode(self):
        with tempfile.TemporaryDirectory() as temp:
            model, data, prepared, metadata = self.prepare(temp)
            session = self.session(prepared)
            capture, ledger, quality, force = session.begin(Path(temp) / "first", metadata=metadata,
                joint_names=fixture.NAMES, warmup_ticks=50, start_time=1.25, human_demo_uuid=UUID)
            capture.close(completed=False)
            next_capture, *_ = session.begin(Path(temp) / "second", metadata=metadata,
                joint_names=fixture.NAMES, warmup_ticks=50, start_time=1.25, human_demo_uuid=UUID)
            with self.assertRaises(EvidenceWriterError):
                quality.write_json({"belongs_to": "old episode"})
            self.assertFalse(session.close()["completed"])
            self.assertEqual((Path(temp) / "second/raw_quality_samples.jsonl").read_bytes(), b"")
            self.assertEqual(json.loads((next_capture.directory / "capture.json").read_text())["status"], "partial_capture_rejected")


if __name__ == "__main__":
    unittest.main()
