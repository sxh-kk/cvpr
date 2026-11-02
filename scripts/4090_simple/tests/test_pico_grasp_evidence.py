"""Target-only Grasp recording/selection; CPU FK and spawn, no physics steps."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handover_contact_quality import SOLVED_STATE_STAGE
from pico_async_evidence import DeferredEvidenceSession
from pico_deferred_capture import (CONTRACT, GRASP_CONTRACT, DeferredStateCapture,
    evidence_object_roles, load_deferred, prepare_deferred_model)
from pico_episode_history import _physical_precheck
from pico_episode_review import complete_geometry
from pico_grasp_quality import (GRASP_TASK_ID, SOURCE_OFFICIAL_TASK_ID, OFFICIAL_SOURCE_SHA256,
    ADAPTER_CONTRACT, FRAME_KEY, GATE_VERSION, evaluate_grasp_capture,
    sample_grasp_solved_quality, validate_grasp_task_identity)
from pico_teleop_recording import CaptureRejected
from replay_handover_teacher import capture_tick_index
from test_pico_deferred_capture import NAMES, FakeCommand


UUID = "25a2a7d1-91ad-4ae0-bc11-3b996b371500"


def metadata():
    identity = {"contract": ADAPTER_CONTRACT, "task_id": GRASP_TASK_ID,
        "source_official_task_id": SOURCE_OFFICIAL_TASK_ID,
        "official_task_source_sha256": OFFICIAL_SOURCE_SHA256, "adapter_sha256": "a" * 64}
    frame = {**identity, "world_translation_m": [0., 0., .785],
        "native_xml_root_m": [0., 0., 0.], "sonic_xml_root_m": [0., 0., .793],
        "robot_anchor_compensation_m": [0., 0., -.793], "source_physics_dt": .002,
        "physics_dt": .005, "render_hz": 50, "native_reward_unmodified": True,
        "robot_physics_equivalent": False, "source_config_sha256": "b" * 64,
        "adapted_payload_sha256": "c" * 64,
        "required_hooks": ["mujoco_world_wrist_camera", "isaac_room_translation"],
        "adapted_payload_scope": ["uid", "metadata", "robot_cfg", "sensor_cfgs", "dr_cfgs", "dr_state_dict"]}
    return {"task_id": GRASP_TASK_ID, "source_official_task_id": SOURCE_OFFICIAL_TASK_ID,
        "source_kind": "live_pico_human", "human_demo_uuid": UUID,
        "actual_task_source_sha256": OFFICIAL_SOURCE_SHA256, "actual_native_success_criteria": .9,
        "evidence_object_roles": ["target"], "object_body_names": {"target": "actual_target"},
        "source_code_sha256_at_session_start": {"pico_grasp_task.py": "a" * 64},
        "grasp_adapter_identity": identity, "grasp_frame_adaptation": frame,
        "environment_config": {FRAME_KEY: deepcopy(frame)}}


def make_model():
    children = "".join(f'<body pos="0 0 .01"><joint name="{name}"/><geom type="sphere" size=".01" mass=".01"/></body>'
                       for name in NAMES)
    return mujoco.MjModel.from_xml_string('<mujoco><option timestep=".005"/><worldbody>'
        '<body name="pelvis" pos="0 0 .8"><freejoint/><geom type="sphere" size=".05"/>' + children + '</body>'
        '<body name="actual_target" pos=".2 0 .6"><freejoint/><geom type="sphere" size=".02"/></body>'
        '<body name="distractor" pos=".4 0 .5"><freejoint/><geom type="sphere" size=".02"/></body>'
        '</worldbody></mujoco>')


def frame(capture, model, data, tick):
    data.time = capture.start_time + tick / 50
    data.qacc_warmstart[:] = np.arange(model.nv) + tick
    data.body("distractor").xfrc_applied[:] = 0.
    # Only set state then derive FK; no call to mj_step/forward or an environment.
    data.joint(model.njnt - 1).qpos[0] = .4 + tick * .001
    mujoco.mj_kinematics(model, data)
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    return {"tick50": tick, "sim_time_s": float(data.time), "observed_joint_names": NAMES,
        "observed_qpos": np.zeros(43), "physical_qpos": np.zeros(43), "physical_qvel": np.zeros(43),
        "root_pose7": np.r_[data.body("pelvis").xpos, data.body("pelvis").xquat], "root_qvel6": np.zeros(6),
        "objects": {"target": {"pose7": np.r_[data.body("actual_target").xpos, data.body("actual_target").xquat]}},
        "command": None if tick < 50 else FakeCommand("sonic_v1_1", controller=object(),
            body_token=np.arange(64, dtype=np.float32), hand_targets={name: .2 for name in NAMES[-14:]}),
        "integration_state": state, "source_fields": {"source_raw_packet": {"real": [tick]},
            "observation_sim_time_s": float(data.time)}}


def rows(count=51, *, successful=True, start=1.25):
    return [{"completed_steps": i + 1, "sim_time_s": start + (i + 1) / 50,
        "solver_stage": SOLVED_STATE_STAGE, "actual_target_xyz_m": [0., 0., .6],
        "native_task_reward": .1, "native_baseline_target_z_m": .6,
        "official_task_success": successful and i == 0} for i in range(count)]


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def write_rows(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def episode(directory, *, successful=True):
    meta = metadata()
    model, start = make_model(), 1.25
    data = mujoco.MjData(model)
    capture = DeferredStateCapture(directory / "deferred_capture", metadata=meta, joint_names=NAMES,
        warmup_ticks=50, start_time=start, model=model)
    for tick in range(51):
        if capture_tick_index(tick) is not None:
            capture.append(**frame(capture, model, data, tick))
    manifest = capture.close(completed=True)
    quality = rows(successful=successful, start=start)
    forces = [{"sim_time_s": start + (i + 1) / 200, "all_zero": True,
        "qfrc_applied_all_zero": True, "xfrc_applied_all_zero": True,
        "qfrc_applied_nonzero_count": 0, "xfrc_applied_nonzero_count": 0,
        "qfrc_applied_max_abs": 0., "xfrc_applied_max_abs": 0.,
        "max_abs_qfrc_applied": 0., "max_abs_xfrc_applied": 0.} for i in range(204)]
    outcome = {"capture": manifest, "state": {"status": "pending_audit", "pending_audit": True,
        "local_tick50": 50}, "geometry_pending": True}
    write_json(directory / "metadata.json", meta)
    write_json(directory / "episode.json", outcome)
    write_rows(directory / "raw_quality_samples.jsonl", quality)
    write_rows(directory / "applied_force_samples.jsonl", forces)
    return meta, outcome


class GraspQualityTests(unittest.TestCase):
    def test_native_only_not_recomputed_from_lift_reward_contacts_or_last_frame(self):
        values = rows(3); before = deepcopy(values)
        result = evaluate_grasp_capture(values, initial_target_z_m=float("nan"))
        self.assertEqual(result["selection_status"], "selected")
        self.assertTrue(result["eligible_physics"])
        self.assertEqual(result["acceptance_gate_version"], GATE_VERSION)
        self.assertNotIn("contact_evidence", result)
        self.assertFalse(result["dataset_ready"])
        self.assertEqual(values, before)
        self.assertFalse(values[-1]["official_task_success"])

    def test_native_false_is_official_failure_only_with_complete_evidence(self):
        result = evaluate_grasp_capture(rows(3, successful=False))
        self.assertEqual(result["selection_status"], "official_task_not_successful")
        self.assertFalse(result["official_task_success"])
        self.assertTrue(result["native_success_evidence_complete"])

    def test_missing_invalid_measurements_are_technical_not_task_failure(self):
        for key, value in (("official_task_success", 1), ("native_task_reward", None),
                           ("native_baseline_target_z_m", float("nan")),
                           ("actual_target_xyz_m", [0., True, .6]), ("solver_stage", "post_forward"),
                           ("sim_time_s", 99.), ("completed_steps", 8)):
            with self.subTest(key=key):
                values = rows(3); values[1][key] = value
                result = evaluate_grasp_capture(values)
                self.assertEqual(result["selection_status"], "technical_evidence_incomplete")
                self.assertIsNone(result["official_task_success"])
                self.assertFalse(result["eligible_physics"])
        values = rows(3); values[1]["native_baseline_target_z_m"] += .01
        self.assertIn("baseline changed", str(evaluate_grasp_capture(values)["capture_validation_errors"]))

    def test_explicit_official_adapter_source_and_frame_identity(self):
        self.assertTrue(validate_grasp_task_identity(metadata())["passed"])
        for key in ("source_official_task_id", "actual_task_source_sha256", "evidence_object_roles",
                    "grasp_adapter_identity", "source_code_sha256_at_session_start", "grasp_frame_adaptation"):
            altered = metadata(); altered.pop(key)
            self.assertFalse(validate_grasp_task_identity(altered)["passed"], key)
        altered = metadata(); altered["actual_native_success_criteria"] = True
        self.assertFalse(validate_grasp_task_identity(altered)["passed"])

    def test_frame_receipt_is_exact_and_bound_to_actual_saved_environment(self):
        for key, value in (("world_translation_m", [0., 0., 0.]), ("native_reward_unmodified", 1),
                           ("robot_physics_equivalent", True), ("source_physics_dt", .005),
                           ("adapter_sha256", "b" * 64), ("source_config_sha256", "invalid"),
                           ("robot_anchor_compensation_m", [0., 0., float("nan")])):
            with self.subTest(key=key):
                altered = metadata(); altered["grasp_frame_adaptation"][key] = value
                self.assertFalse(validate_grasp_task_identity(altered)["passed"])
        altered = metadata(); altered["environment_config"][FRAME_KEY]["physics_dt"] = .002
        self.assertFalse(validate_grasp_task_identity(altered)["passed"])

    def test_sampler_copies_real_target_without_container_or_state_mutation(self):
        model = make_model(); data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        result = sample_grasp_solved_quality(model, data, completed_steps=1,
            target_body="actual_target", solver_stage=SOLVED_STATE_STAGE)
        self.assertEqual(result["target_body_name"], "actual_target")
        self.assertNotIn("container_pose7", result)
        self.assertNotIn("official_task_success", result)
        after = state.copy(); mujoco.mj_getState(model, data, after, mujoco.mjtState.mjSTATE_INTEGRATION)
        np.testing.assert_array_equal(state, after)
        data.xpos[model.body("actual_target").id, 2] += 1
        self.assertEqual(result["actual_target_xyz_m"][2], .6)


class GraspCaptureTests(unittest.TestCase):
    def test_roles_are_explicit_and_legacy_default_keeps_original_contract(self):
        self.assertEqual(evidence_object_roles({}), ("target", "container"))
        self.assertEqual(evidence_object_roles(metadata()), ("target",))
        altered = metadata(); altered["task_id"] = SOURCE_OFFICIAL_TASK_ID
        with self.assertRaises(ValueError):
            evidence_object_roles(altered)
        altered = metadata(); altered["object_body_names"]["container"] = "distractor"
        with self.assertRaises(ValueError):
            evidence_object_roles(altered)
        self.assertNotEqual(CONTRACT, GRASP_CONTRACT)

    def test_target_only_roundtrip_retains_distractor_in_full_integration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); meta, outcome = episode(root)
            loaded = load_deferred(root / "deferred_capture")
            self.assertEqual(loaded["manifest"]["contract"], GRASP_CONTRACT)
            self.assertTrue(all(set(event["objects"]) == {"target"} for event in loaded["events"]))
            self.assertFalse(loaded["states"].flags.writeable)
            data = mujoco.MjData(loaded["model"])
            mujoco.mj_setState(loaded["model"], data, loaded["states"][-1].copy(), mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_kinematics(loaded["model"], data)
            self.assertAlmostEqual(data.body("distractor").xpos[0], .45)
            path = root / "deferred_capture/capture.json"
            manifest = loaded["manifest"]; manifest["contract"] = CONTRACT; write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "contract"):
                load_deferred(path.parent)

    def test_spawn_worker_uses_target_only_and_matches_synchronous_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, model = Path(temporary), make_model()
            data, meta = mujoco.MjData(model), metadata()
            prepared = prepare_deferred_model(root / "prepared", model=model, joint_names=NAMES, metadata=meta)
            session = DeferredEvidenceSession(prepared, cpus=[min(os.sched_getaffinity(0))], max_pending=128)
            try:
                self.assertEqual(set(session.receipt()["loaded_model"]["object_body_names"]), {"target"})
                asynchronous, ledger, quality, force = session.begin(root / "async", metadata=meta,
                    joint_names=NAMES, warmup_ticks=50, start_time=1.25, human_demo_uuid=UUID)
                synchronous = DeferredStateCapture(root / "sync", metadata=meta,
                    joint_names=NAMES, warmup_ticks=50, start_time=1.25, model=model)
                for tick in range(51):
                    if capture_tick_index(tick) is not None:
                        value = frame(synchronous, model, data, tick)
                        synchronous.append(**value); asynchronous.append(**value)
                self.assertEqual(synchronous.close(completed=True), asynchronous.close(completed=True))
                for name in ("events.jsonl", "states.npy"):
                    self.assertEqual((root / "sync" / name).read_bytes(),
                                     (root / "async/deferred_capture" / name).read_bytes())
                self.assertEqual(load_deferred(asynchronous.directory)["manifest"]["contract"], GRASP_CONTRACT)
            finally:
                self.assertTrue(session.close()["completed"])


class GraspHistoryReviewTests(unittest.TestCase):
    def test_native_grasp_is_selected_without_mobile_handover_or_geometry_rules(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); meta, outcome = episode(root)
            with patch("pico_mobile_quality.evaluate_mobile_capture", side_effect=AssertionError("wrong task")), \
                 patch("handover_contact_quality.evaluate_capture", side_effect=AssertionError("wrong task")):
                reasons, report = _physical_precheck(root, outcome, meta)
            self.assertEqual(reasons, [], report)
            self.assertEqual(report["selection_status"], "selected")
            self.assertEqual(report["acceptance_gate_version"], GATE_VERSION)
            self.assertEqual(report["excluded_unsupervised_warmup_ticks50"], 0)
            self.assertTrue(report["task_identity_validation"]["passed"])
            self.assertEqual(set(report["evaluator_source_sha256"]), {"pico_grasp_quality.py", "pico_episode_history.py"})
            job = {"episode_json_sha256": hashlib.sha256((root / "episode.json").read_bytes()).hexdigest(),
                   "metadata_sha256": hashlib.sha256((root / "metadata.json").read_bytes()).hexdigest()}
            self.assertEqual(complete_geometry(root, job)["status"], "not_required")
            self.assertFalse((root / "quality_geometry_completion.json").exists())

    def test_missing_flag_bad_sha_or_applied_force_are_technical(self):
        for failure in ("flag", "source", "force", "frame", "capture_metadata", "state_dtype", "row_sha"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); meta, outcome = episode(root)
                if failure == "flag":
                    values = rows(); values[0].pop("official_task_success")
                    write_rows(root / "raw_quality_samples.jsonl", values)
                elif failure == "source":
                    meta["actual_task_source_sha256"] = "b" * 64
                elif failure == "frame":
                    meta.pop("grasp_frame_adaptation")
                elif failure == "capture_metadata":
                    meta["human_demo_uuid"] = "different"
                elif failure == "state_dtype":
                    path = root / "deferred_capture/states.npy"
                    np.save(path, np.load(path).astype(np.float32))
                    outcome["capture"]["states_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    write_json(root / "deferred_capture/capture.json", outcome["capture"])
                elif failure == "row_sha":
                    path = root / "deferred_capture/events.jsonl"
                    values = [json.loads(line) for line in path.read_text().splitlines()]
                    values[0]["integration_state_sha256"] = "b" * 64; write_rows(path, values)
                    outcome["capture"]["events_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    write_json(root / "deferred_capture/capture.json", outcome["capture"])
                else:
                    path = root / "applied_force_samples.jsonl"
                    values = [json.loads(line) for line in path.read_text().splitlines()]
                    values[0]["all_zero"] = False; write_rows(path, values)
                reasons, report = _physical_precheck(root, outcome, meta)
                self.assertTrue(reasons)
                self.assertTrue(report["technical_blocked"], report)
                self.assertEqual(report["selection_status"], "technical_evidence_incomplete")
                self.assertIsNone(report["official_task_success"])
                self.assertFalse(report["eligible_physics"])

    def test_native_false_does_not_become_technical_and_review_seal_cannot_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); meta, outcome = episode(root, successful=False)
            reasons, report = _physical_precheck(root, outcome, meta)
            self.assertEqual(report["selection_status"], "official_task_not_successful")
            self.assertFalse(report["technical_blocked"])
            self.assertTrue(reasons)
            with self.assertRaisesRegex(ValueError, "SHA"):
                complete_geometry(root, {"episode_json_sha256": "0" * 64})

    def test_grasp_clock_header_warmup_version_and_pending_flags_are_technical(self):
        for key, value in (("physics_hz", 100), ("decoder_hz", 25), ("command_rate_hz", 15),
                           ("version", True), ("warmup_ticks", 0), ("warmup_seconds", 2.),
                           ("rtx_ready", True), ("training_rgb_acquired", True)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); meta, outcome = episode(root)
                outcome["capture"][key] = value
                write_json(root / "deferred_capture/capture.json", outcome["capture"])
                reasons, report = _physical_precheck(root, outcome, meta)
                self.assertTrue(reasons)
                self.assertTrue(report["technical_blocked"], report)
                self.assertEqual(report["selection_status"], "technical_evidence_incomplete")
                self.assertIsNone(report["official_task_success"])


if __name__ == "__main__":
    unittest.main()
