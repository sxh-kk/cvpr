"""CPU MuJoCo archive tests; no SDK, model inference, renderer or GPU."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pico_deferred_capture import DeferredStateCapture, PENDING, load_deferred, prepare_deferred_model
from pico_teleop_recording import CaptureRejected
from replay_handover_teacher import MODEL_HAND_NAMES, capture_tick_index


NAMES = [f"body_joint_{i}" for i in range(29)] + list(MODEL_HAND_NAMES)


class FakeCommand:
    def __init__(self, kind, **parameters):
        self.type, self.parameters = kind, parameters


class NeverCopyController:
    def __deepcopy__(self, memo):
        raise AssertionError("Controller/ORT must never be deep-copied")


def make_model():
    children = "".join(f'<body name="link_{i}" pos="0 0 .01"><joint name="{name}" type="hinge"/>'
                       '<geom type="sphere" size=".01" mass=".01"/></body>' for i, name in enumerate(NAMES))
    return mujoco.MjModel.from_xml_string('<mujoco><option timestep=".005"/><worldbody>'
        '<body name="pelvis" pos="0 0 .8"><freejoint name="root"/><geom type="sphere" size=".05"/>'
        + children + '</body><body name="target" pos=".2 0 .6"><freejoint/>'
        '<geom type="sphere" size=".02"/></body><body name="container" pos=".4 0 .5"><freejoint/>'
        '<geom type="sphere" size=".02"/></body></worldbody></mujoco>')


class DeferredCaptureTests(unittest.TestCase):
    def make_capture(self, directory, *, start_time=1.25):
        model = make_model()
        metadata = {"source_kind": "live_pico_human", "human_demo_uuid": "original-uuid",
                    "object_body_names": {"target": "target", "container": "container"},
                    "nested": {"actual": [1, 2]}}
        capture = DeferredStateCapture(Path(directory) / "deferred_capture", metadata=metadata,
            joint_names=NAMES, warmup_ticks=50, start_time=start_time, model=model)
        return capture, model, mujoco.MjData(model), metadata

    def frame(self, capture, model, data, tick):
        data.time = capture.start_time + tick / 50
        # Include state channels mj_forward could overwrite, not only qpos/qvel.
        data.qacc_warmstart[:] = np.arange(model.nv) * .01 + tick
        data.qvel[:] = np.arange(model.nv) * .001
        for i, name in enumerate(NAMES):
            data.joint(name).qpos[0] = i * .001 + tick * .0001
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        token = np.arange(64, dtype=np.float32) + tick
        hands = {name: float(i) / 100 for i, name in enumerate(MODEL_HAND_NAMES)}
        command = None if tick < 50 else FakeCommand("sonic_v1_1", controller=NeverCopyController(),
            body_token=token, hand_targets=hands)
        return dict(tick50=tick, sim_time_s=float(data.time), observed_joint_names=list(NAMES),
            observed_qpos=np.array([data.joint(name).qpos[0] for name in NAMES], dtype=np.float32),
            physical_qpos=np.array([data.joint(name).qpos[0] for name in NAMES]),
            physical_qvel=np.array([data.joint(name).qvel[0] for name in NAMES]),
            root_pose7=np.r_[data.body("pelvis").xpos, data.body("pelvis").xquat],
            root_qvel6=data.qvel[:6],
            objects={role: {"pose7": np.r_[data.body(role).xpos, data.body(role).xquat]} for role in ("target", "container")},
            command=command, integration_state=state,
            preview_rgb=np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
            source_fields={"source_sequence": tick + 1, "source_raw_packet": {"va": [1., 2.], "wva": [3., 4.]},
                           "observation_sim_time_s": float(data.time)})

    def fill(self, capture, model, data, last=50):
        for tick in range(last + 1):
            if capture_tick_index(tick) is not None:
                capture.append(**self.frame(capture, model, data, tick))

    def test_completed_archive_roundtrip_exact_state_metadata_commands_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            capture, model, data, metadata = self.make_capture(directory)
            metadata["nested"]["actual"][0] = 99
            self.fill(capture, model, data)
            manifest = capture.close(completed=True)
            self.assertEqual(manifest["status"], PENDING)
            self.assertFalse(capture.failed)
            self.assertEqual(manifest["event_count"], 31)
            self.assertEqual(manifest["warmup_event_count"], 30)
            self.assertEqual(manifest["command_event_count"], 1)
            self.assertFalse(manifest["rtx_ready"])
            self.assertFalse(manifest["accepted_new_dataset"])
            self.assertFalse(manifest["eligible_for_packing"])
            self.assertEqual(manifest["metadata"]["nested"]["actual"], [1, 2])
            archive = load_deferred(capture.directory)
            self.assertFalse(archive["states"].flags.writeable)
            self.assertIsInstance(archive["states"], np.memmap)
            self.assertEqual(archive["joint_names"], NAMES)
            self.assertEqual(archive["state_spec"], int(mujoco.mjtState.mjSTATE_INTEGRATION))
            self.assertIsNone(archive["events"][0]["action"])
            last = archive["events"][-1]
            self.assertEqual(last["action"]["body_token64"], (np.arange(64) + 50).tolist())
            self.assertEqual(last["action"]["hand_joint_names"], list(MODEL_HAND_NAMES))
            self.assertEqual(last["source_fields"]["source_raw_packet"]["va"], [1., 2.])
            self.assertNotIn("rgb", last)
            self.assertFalse(last["preview"]["training_observation"])
            self.assertFalse(last["preview"]["pixels_saved"])
            self.assertEqual(capture.close(completed=False), manifest)

    def test_append_copies_arrays_commands_objects_source_and_never_deepcopies_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            capture, model, data, _ = self.make_capture(directory)
            self.fill(capture, model, data, last=49)
            values = self.frame(capture, model, data, 50)
            expected_state = values["integration_state"].copy()
            expected_preview = hashlib.sha256(values["preview_rgb"].tobytes()).hexdigest()
            capture.append(**values)
            values["integration_state"][:] = -999
            values["observed_qpos"][:] = -123
            values["physical_qpos"][:] = -456
            values["objects"]["target"]["pose7"][:] = -111
            values["command"].parameters["body_token"][:] = -222
            values["command"].parameters["hand_targets"][MODEL_HAND_NAMES[0]] = -333
            values["source_fields"]["source_raw_packet"]["va"][0] = -444
            values["preview_rgb"][:] = 0
            capture.close(completed=True)
            result = load_deferred(capture.directory)
            event = result["events"][-1]
            self.assertTrue(np.array_equal(result["states"][-1], expected_state))
            self.assertEqual(event["action"]["body_token64"][0], 50.)
            self.assertEqual(event["action"]["hand14"][0], 0.)
            self.assertEqual(event["source_fields"]["source_raw_packet"]["va"][0], 1.)
            self.assertEqual(event["preview"]["pixel_sha256"], expected_preview)

    def test_missing_grid_wrong_clock_and_wrong_real_state_fail_sticky(self):
        for failure in ("skipped", "clock", "q43", "root", "object", "state", "obs_clock", "hands", "token", "nonfinite"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                capture, model, data, _ = self.make_capture(directory)
                values = self.frame(capture, model, data, 0)
                if failure == "skipped":
                    values = self.frame(capture, model, data, 2)
                elif failure == "clock":
                    values["sim_time_s"] += .01
                elif failure == "q43":
                    values["physical_qpos"][0] += .5
                    values["observed_qpos"][0] += .5
                elif failure == "root":
                    values["root_pose7"][0] += .5
                elif failure == "object":
                    values["objects"]["target"]["pose7"][0] += .5
                elif failure == "state":
                    values["integration_state"][0] += .5
                elif failure == "obs_clock":
                    values["source_fields"]["observation_sim_time_s"] += .001
                elif failure in ("hands", "token"):
                    self.fill(capture, model, data, last=49)
                    values = self.frame(capture, model, data, 50)
                    if failure == "hands":
                        del values["command"].parameters["hand_targets"][MODEL_HAND_NAMES[0]]
                    else:
                        values["command"].parameters["body_token"] = np.zeros(63)
                elif failure == "nonfinite":
                    values["source_fields"]["source_raw_packet"]["va"][0] = float("nan")
                with self.assertRaises(CaptureRejected):
                    capture.append(**values)
                self.assertTrue(capture.failed)
                self.assertIsInstance(capture.error, str)
                with self.assertRaises(CaptureRejected):
                    capture.append(**self.frame(capture, model, data, 0))
                outcome = capture.close(completed=True)
                self.assertEqual(outcome["status"], "partial_capture_rejected")
                self.assertFalse(outcome["accepted_new_dataset"])
                self.assertTrue(capture.stream.closed)

    def test_warmup_only_and_discarded_capture_cannot_complete(self):
        for completed in (True, False):
            with self.subTest(completed=completed), tempfile.TemporaryDirectory() as directory:
                capture, model, data, _ = self.make_capture(directory)
                self.fill(capture, model, data, last=49 if completed else 50)
                outcome = capture.close(completed=completed)
                self.assertEqual(outcome["status"], "partial_capture_rejected")
                self.assertFalse(outcome["rtx_ready"])
                self.assertFalse(outcome["accepted_new_dataset"])

    def test_sha_and_event_state_binding_tampering_fail(self):
        for filename in ("model.mjb", "events.jsonl", "states.npy"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                capture, model, data, _ = self.make_capture(directory)
                self.fill(capture, model, data)
                capture.close(completed=True)
                path = capture.directory / filename
                path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    load_deferred(capture.directory)
        with tempfile.TemporaryDirectory() as directory:
            capture, model, data, _ = self.make_capture(directory)
            self.fill(capture, model, data)
            manifest = capture.close(completed=True)
            path = capture.directory / "events.jsonl"
            events = [json.loads(row) for row in path.read_text().splitlines()]
            events[-1]["root_pose7"][0] += .1
            path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
            manifest["events_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            (capture.directory / "capture.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "root pose"):
                load_deferred(capture.directory)

    def test_reader_rejects_symlink_and_false_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            capture, model, data, _ = self.make_capture(directory)
            self.fill(capture, model, data)
            manifest = capture.close(completed=True)
            state_path = capture.directory / "states.npy"
            outside = Path(directory) / "outside.npy"
            state_path.rename(outside)
            state_path.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "inside"):
                load_deferred(capture.directory)
            state_path.unlink()
            outside.rename(state_path)
            manifest["accepted_new_dataset"] = True
            (capture.directory / "capture.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "claim"):
                load_deferred(capture.directory)

    def test_empty_archive_close_rejects_and_remains_readable_for_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            capture, _, _, _ = self.make_capture(directory)
            outcome = capture.close(completed=True)
            self.assertEqual(outcome["status"], "partial_capture_rejected")
            loaded = load_deferred(capture.directory)
            self.assertEqual(loaded["states"].shape[0], 0)
            self.assertEqual(loaded["events"], [])

    def test_prepared_model_keeps_save_hash_and_mjdata_allocation_outside_record_constructor(self):
        with tempfile.TemporaryDirectory() as directory:
            model = make_model()
            metadata = {"object_body_names": {"target": "target", "container": "container"}}
            prepared = prepare_deferred_model(Path(directory) / "prepared", model=model,
                joint_names=NAMES, metadata=metadata)
            with mock.patch("pico_deferred_capture._sha", side_effect=AssertionError("No live model hash")), \
                    mock.patch("pico_deferred_capture._PhysicalVerifier", side_effect=AssertionError("No live scratch allocation")), \
                    mock.patch("mujoco.mj_saveModel", side_effect=AssertionError("No live model save")):
                capture = DeferredStateCapture(Path(directory) / "capture", metadata=metadata,
                    joint_names=NAMES, warmup_ticks=50, start_time=1.25, model=model, prepared_model=prepared)
            self.assertEqual((capture.directory / "model.mjb").stat().st_ino, prepared.model_path.stat().st_ino)
            self.assertIs(capture._verifier, prepared.verifier)
            with self.assertRaisesRegex(ValueError, "unclosed"):
                DeferredStateCapture(Path(directory) / "overlap", metadata=metadata,
                    joint_names=NAMES, warmup_ticks=50, start_time=1.25, model=model, prepared_model=prepared)
            self.assertTrue(prepared.in_use)
            self.fill(capture, model, mujoco.MjData(model))
            capture.close(completed=True)
            self.assertFalse(prepared.in_use)
            self.assertEqual(load_deferred(capture.directory)["manifest"]["status"], PENDING)
            next_capture = DeferredStateCapture(Path(directory) / "next", metadata=metadata,
                joint_names=NAMES, warmup_ticks=50, start_time=1.25, model=model, prepared_model=prepared)
            next_capture.close(completed=False)
            self.assertFalse(prepared.in_use)


if __name__ == "__main__":
    unittest.main()
