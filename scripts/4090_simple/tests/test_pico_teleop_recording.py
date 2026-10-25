"""CPU recording isolation/failure tests; no SDK, Isaac, or GPU is loaded."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pico_teleop_recording import AsyncTrainingCapture, CaptureRejected, CommandLedger, RecordingState


class FakeCommand:
    def __init__(self, kind, **parameters):
        self.type, self.parameters = kind, parameters


class NeverCopyController:
    def __deepcopy__(self, memo):
        raise AssertionError("The live controller/ORT session must never be copied")


class FakeCapture:
    def __init__(self, directory, **metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = metadata
        self.frames = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.fail = False
        self.append_threads = []
        self.close_thread = None
        self.completed = None

    def append(self, **frame):
        self.append_threads.append(threading.get_ident())
        self.started.set()
        if self.block and not self.release.wait(timeout=5):
            raise RuntimeError("Test writer gate timed out")
        if self.fail:
            raise ValueError("Injected image write failure")
        self.frames.append(frame)

    def close(self, *, completed):
        self.close_thread = threading.get_ident()
        self.completed = completed
        return {"status": "raw_capture_completed_pending_independent_audit" if completed else "partial_capture_rejected",
                "event_count": len(self.frames), "accepted_new_dataset": False}


class AsyncCaptureTests(unittest.TestCase):
    def make_capture(self, directory, *, block=False, fail=False, max_queue=2):
        created = []
        def factory(path, **metadata):
            capture = FakeCapture(path, **metadata)
            capture.block, capture.fail = block, fail
            created.append(capture)
            return capture
        writer = AsyncTrainingCapture(Path(directory) / "capture", metadata={"label": [1]},
            joint_names=[f"joint{i}" for i in range(43)], warmup_ticks=50,
            start_time=0., max_queue=max_queue, capture_factory=factory)
        return writer, created[0]

    def test_queued_pixels_arrays_diagnostics_objects_and_commands_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            writer, target = self.make_capture(directory, block=True)
            controller = NeverCopyController()
            pixels = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
            token = np.arange(64, dtype=np.float32)
            hands = {"left_thumb_0": .2}
            diagnostic = {"links": {"head": {"position": np.array([1., 2., 3.])}}}
            objects = {"target": {"pose7": np.array([1., 2., 3., 1., 0., 0., 0.])}}
            names = ["joint0"]
            command = FakeCommand("sonic_v1_1", controller=controller, body_token=token, hand_targets=hands)
            writer.enqueue(rgb=pixels, command=command, renderer_pose_diagnostics=diagnostic,
                           objects=objects, observed_joint_names=names)
            self.assertTrue(target.started.wait(timeout=3))
            pixels[:] = 0; token[:] = -1; hands["left_thumb_0"] = 9
            diagnostic["links"]["head"]["position"][:] = -1
            objects["target"]["pose7"][:] = -1; names.append("changed")
            target.release.set()
            manifest = writer.close(completed=True)
            saved = target.frames[0]
            np.testing.assert_array_equal(saved["rgb"], np.arange(24, dtype=np.uint8).reshape(2, 4, 3))
            np.testing.assert_array_equal(saved["command"].parameters["body_token"], np.arange(64, dtype=np.float32))
            self.assertIs(saved["command"].parameters["controller"], controller)
            self.assertEqual(saved["command"].parameters["hand_targets"], {"left_thumb_0": .2})
            np.testing.assert_array_equal(saved["renderer_pose_diagnostics"]["links"]["head"]["position"], [1., 2., 3.])
            self.assertEqual(saved["objects"]["target"]["pose7"][0], 1.)
            self.assertEqual(saved["observed_joint_names"], ["joint0"])
            self.assertEqual(len(set(target.append_threads)), 1)
            self.assertNotEqual(target.append_threads[0], threading.get_ident())
            self.assertEqual(target.close_thread, target.append_threads[0])
            self.assertFalse(writer.worker_alive)
            self.assertTrue(manifest["async_capture"]["completed"])
            self.assertFalse(manifest["accepted_new_dataset"])
            with self.assertRaises(CaptureRejected):
                writer.enqueue(command=None)
            self.assertEqual(writer.close(completed=True), manifest)

    def test_queue_full_is_sticky_and_joined_before_rejected_result(self):
        with tempfile.TemporaryDirectory() as directory:
            writer, target = self.make_capture(directory, block=True, max_queue=1)
            writer.enqueue(command=None, frame=0)
            self.assertTrue(target.started.wait(timeout=3))
            writer.enqueue(command=None, frame=1)
            with self.assertRaisesRegex(CaptureRejected, "queue full"):
                writer.enqueue(command=None, frame=2)
            self.assertTrue(writer.failed)
            with self.assertRaises(CaptureRejected):
                writer.enqueue(command=None, frame=3)
            target.release.set()
            manifest = writer.close(completed=True)
            self.assertFalse(target.completed)
            self.assertFalse(manifest["async_capture"]["completed"])
            self.assertEqual(manifest["status"], "partial_capture_rejected")
            self.assertFalse(manifest["accepted_new_dataset"])
            self.assertFalse(writer.worker_alive)
            self.assertTrue(manifest["async_capture"]["writer_joined"])
            on_disk = json.loads((Path(directory) / "capture/async_capture.json").read_text())
            self.assertIn("queue full", on_disk["async_capture"]["failure"])

    def test_writer_failure_rejects_even_when_completion_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            writer, target = self.make_capture(directory, fail=True)
            writer.enqueue(command=None)
            manifest = writer.close(completed=True)
            self.assertFalse(target.completed)
            self.assertTrue(writer.failed)
            self.assertFalse(writer.worker_alive)
            self.assertIn("Injected image write failure", manifest["async_capture"]["failure"])
            self.assertFalse(manifest["async_capture"]["completed"])
            self.assertEqual(manifest["status"], "partial_capture_rejected")

    def test_invalid_snapshot_is_sticky_and_incomplete_close_never_admits(self):
        with tempfile.TemporaryDirectory() as directory:
            writer, target = self.make_capture(directory)
            with self.assertRaises(CaptureRejected):
                writer.enqueue(command=None, rgb=np.array([object()], dtype=object))
            manifest = writer.close(completed=True)
            self.assertFalse(target.completed)
            self.assertEqual(manifest["status"], "partial_capture_rejected")
        with tempfile.TemporaryDirectory() as directory:
            writer, target = self.make_capture(directory)
            writer.enqueue(command=None)
            manifest = writer.close(completed=False)
            self.assertFalse(target.completed)
            self.assertFalse(manifest["async_capture"]["completed"])
            self.assertFalse(manifest["accepted_new_dataset"])

    def test_capture_close_failure_still_joins_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            writer, target = self.make_capture(directory)
            def broken_close(**kwargs):
                raise OSError("Injected final flush failure")
            target.close = broken_close
            writer.enqueue(command=None)
            manifest = writer.close(completed=True)
            self.assertFalse(writer.worker_alive)
            self.assertEqual(manifest["status"], "partial_capture_rejected")
            self.assertIn("final flush", manifest["async_capture"]["failure"])

    def test_empty_capture_and_underlying_rejection_cannot_be_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            writer, target = self.make_capture(directory)
            manifest = writer.close(completed=True)
            self.assertFalse(target.completed)
            self.assertFalse(manifest["async_capture"]["completed"])
            self.assertIn("requires recorded evidence", manifest["async_capture"]["failure"])
        with tempfile.TemporaryDirectory() as directory:
            writer, target = self.make_capture(directory)
            target.close = lambda **kwargs: {"status": "partial_capture_rejected"}
            writer.enqueue(command=None)
            manifest = writer.close(completed=True)
            self.assertFalse(writer.worker_alive)
            self.assertFalse(manifest["async_capture"]["completed"])
            self.assertIn("did not complete", manifest["async_capture"]["failure"])

    def test_interrupted_join_still_cleans_up_and_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            writer, target = self.make_capture(directory)
            writer.enqueue(command=None)
            real_join = writer._writer.join
            join_calls = []
            def interrupted_once(*args, **kwargs):
                join_calls.append(1)
                if len(join_calls) == 1:
                    raise KeyboardInterrupt("Injected interruption during cleanup")
                return real_join(*args, **kwargs)
            writer._writer.join = interrupted_once
            manifest = writer.close(completed=True)
            self.assertGreaterEqual(len(join_calls), 2)
            self.assertFalse(writer.worker_alive)
            self.assertFalse(manifest["async_capture"]["completed"])
            self.assertEqual(manifest["status"], "partial_capture_rejected")
            self.assertIn("KeyboardInterrupt", manifest["async_capture"]["failure"])


class RecordingStateTests(unittest.TestCase):
    def test_start_alignment_real_warmup_and_pending_audit_are_distinct(self):
        state = RecordingState()
        state.on_tick(2); state.request_start()
        self.assertFalse(state.on_tick(3)["recording"])
        self.assertFalse(state.on_tick(4)["started"])
        first = state.on_tick(5)
        self.assertTrue(first["started"]); self.assertTrue(first["warmup"])
        self.assertEqual(first["local_tick50"], 0)
        warmup_frames = int(first["capture_due"])
        with self.assertRaises(RuntimeError): state.request_save()
        for tick in range(6, 55):
            value = state.on_tick(tick)
            self.assertTrue(value["warmup"])
            self.assertFalse(value["action_valid"])
            warmup_frames += int(value["capture_due"])
        self.assertEqual(warmup_frames, 30)
        value = state.on_tick(55)
        self.assertEqual(value["status"], "recording")
        self.assertTrue(value["action_valid"]); self.assertTrue(value["capture_due"])
        saved = state.request_save()
        self.assertTrue(saved["pending_audit"])
        self.assertFalse(saved["accepted_new_dataset"])
        self.assertFalse(state.on_tick(56)["recording"])
        self.assertFalse(state.close(completed=True)["accepted_new_dataset"])
        with self.assertRaises(RuntimeError): state.request_save()
        with self.assertRaises(RuntimeError): state.request_start()

    def test_missing_physical_tick_invalid_input_and_discard_never_succeed(self):
        state = RecordingState(); state.request_start(); state.on_tick(0)
        self.assertEqual(state.on_tick(2)["status"], "rejected")
        with self.assertRaises(RuntimeError): state.request_save()
        self.assertEqual(state.close(completed=True)["status"], "rejected")
        self.assertFalse(state.close(completed=True)["accepted_new_dataset"])
        other = RecordingState(); other.request_start(); other.on_tick(0)
        self.assertEqual(other.reject("tracking stale")["status"], "rejected")
        self.assertEqual(other.discard()["status"], "rejected")
        discarded = RecordingState()
        self.assertEqual(discarded.discard()["status"], "discarded")
        self.assertFalse(discarded.close(completed=False)["accepted_new_dataset"])
        with self.assertRaises(RuntimeError): discarded.request_save()
        self.assertNotEqual(state.human_demo_uuid, discarded.human_demo_uuid)
        self.assertEqual(RecordingState().on_tick(True)["status"], "rejected")


class CommandLedgerTests(unittest.TestCase):
    def test_identity_raw_derivative_warning_and_bytes_are_fixed_at_append(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "commands.jsonl"
            identity = str(uuid.uuid4())
            ledger = CommandLedger(path, human_demo_uuid=identity, metadata={"operator": "local"})
            raw = {"va": "1,2,3,4,5,6", "wva": "0,0,0,0,0,0"}
            token = np.zeros(64, dtype=np.float32)
            ledger.append("command", raw_pico=raw, body_token64=token)
            raw["va"] = "changed"; token[:] = 9
            with self.assertRaises(ValueError):
                ledger.append("command", raw_pico_va_wva_validated=True)
            with self.assertRaises(ValueError): ledger.append("command", bad=np.nan)
            receipt = ledger.close()
            values = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(receipt["event_count"], 2)
            self.assertEqual(values[1]["human_demo_uuid"], identity)
            self.assertEqual(values[1]["raw_pico"]["va"], "1,2,3,4,5,6")
            self.assertEqual(values[1]["body_token64"], [0.] * 64)
            self.assertFalse(values[1]["raw_pico_va_wva_validated"])
            self.assertFalse(values[1]["accepted_new_dataset"])
            self.assertEqual(ledger.close(), receipt)
            with self.assertRaises(RuntimeError): ledger.append("command")
            with self.assertRaises(FileExistsError): CommandLedger(path)


if __name__ == "__main__":
    unittest.main()
