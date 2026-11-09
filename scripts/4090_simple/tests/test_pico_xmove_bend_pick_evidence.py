"""Independent XMoveBendPick evidence dispatch; real CPU FK/spawn, no simulation."""
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
import test_pico_grasp_evidence as fixture
from pico_async_evidence import DeferredEvidenceSession
from pico_deferred_capture import (CONTRACT, GRASP_CONTRACT, BEND_PICK_CONTRACT, XMOVE_PICK_CONTRACT, XMOVE_BEND_PICK_CONTRACT,
    DeferredStateCapture, evidence_profile, load_deferred, prepare_deferred_model)
from pico_episode_history import _physical_precheck
from pico_episode_review import complete_geometry
from pico_teleop_recording import CaptureRejected
from pico_xmove_bend_pick_quality import (TASK_ID, SOURCE_TASK_ID, OFFICIAL_SOURCE_SHA256,
    ADAPTER_CONTRACT, FRAME_KEY, GATE_VERSION, validate_xmove_bend_pick_task_identity)
from replay_handover_teacher import capture_tick_index


def metadata():
    identity = {"contract": ADAPTER_CONTRACT, "task_id": TASK_ID,
        "source_official_task_id": SOURCE_TASK_ID,
        "official_task_source_sha256": OFFICIAL_SOURCE_SHA256, "adapter_sha256": "a" * 64}
    adaptation = {**identity, "world_translation_m": [0., 0., 0.],
        "sonic_control_policy": "joystick_mode0_full_body_mode2", "sonic_encoder_modes": [0, 2],
        "native_ground_z_m": 0., "native_robot_spawn_z_m": 0., "native_table_top_z_m": .45,
        "source_robot_uid": "g1_sonic", "robot_uid": "g1_sonic",
        "source_physics_dt": .005, "physics_dt": .005, "render_hz": 50,
        "native_reward_unmodified": True, "robot_physics_equivalent": False,
        "coordinate_transform_applied": False, "source_config_sha256": "b" * 64,
        "adapted_payload_sha256": "c" * 64,
        "source_material_state_missing": True,
        "source_lighting_state_missing": True,
        "lighting_restoration": "saved layout light instances only; original empty DR lighting state preserved, no random light generation",
        "material_restoration": "saved table/object layout materials only; unsaved ground/robot use backend defaults",
        "adapted_payload_scope": ["uid", "metadata", "robot_cfg", "sensor_cfgs", "dr_cfgs", "dr_state_dict"]}
    return {"task_id": TASK_ID, "source_official_task_id": SOURCE_TASK_ID,
        "xmove_bend_pick_control_policy": "joystick_mode0_full_body_mode2",
        "xmove_bend_pick_encoder_modes": [0, 2],
        "mobile_control_policy": "locked_upper_mode0", "mobile_encoder_mode": "legacy_mode2_mode0",
        "source_kind": "live_pico_human", "human_demo_uuid": fixture.UUID,
        "actual_task_source_sha256": OFFICIAL_SOURCE_SHA256, "actual_native_success_criteria": .9,
        "evidence_object_roles": ["target"], "object_body_names": {"target": "actual_target"},
        "source_code_sha256_at_session_start": {"pico_xmove_bend_pick_task.py": "a" * 64},
        "xmove_bend_pick_adapter_identity": identity, "xmove_bend_pick_adaptation": adaptation,
        "environment_config": {FRAME_KEY: deepcopy(adaptation), "dr_state_dict": {"material": None, "lighting": {}}}}


def episode(directory, *, successful=True):
    with patch.object(fixture, "metadata", metadata):
        return fixture.episode(directory, successful=successful)


class XMoveBendPickEvidenceTests(unittest.TestCase):
    def test_independent_target_contract_and_legacy_default(self):
        self.assertEqual(evidence_profile({}), (CONTRACT, ("target", "container")))
        self.assertEqual(evidence_profile(metadata()), (XMOVE_BEND_PICK_CONTRACT, ("target",)))
        self.assertEqual(len({CONTRACT, GRASP_CONTRACT, BEND_PICK_CONTRACT, XMOVE_PICK_CONTRACT, XMOVE_BEND_PICK_CONTRACT}), 5)
        for change in ({"task_id": SOURCE_TASK_ID},
                       {"source_official_task_id": "simple/G1WholebodyTabletopGraspMP-v0"},
                       {"evidence_object_roles": ["target", "container"]},
                       {"object_body_names": {"target": "actual_target", "container": "distractor"}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                evidence_profile({**metadata(), **change})

    def test_real_target_only_archive_retains_all_actors_and_rejects_alias_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); episode(root)
            loaded = load_deferred(root / "deferred_capture")
            self.assertEqual(loaded["manifest"]["contract"], XMOVE_BEND_PICK_CONTRACT)
            self.assertTrue(all(set(row["objects"]) == {"target"} for row in loaded["events"]))
            self.assertEqual(loaded["states"].dtype, np.float64)
            data = mujoco.MjData(loaded["model"])
            mujoco.mj_setState(loaded["model"], data, loaded["states"][-1].copy(), mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_kinematics(loaded["model"], data)
            self.assertAlmostEqual(data.body("distractor").xpos[0], .45)
            path = root / "deferred_capture/capture.json"
            for wrong_contract in (CONTRACT, GRASP_CONTRACT, BEND_PICK_CONTRACT, XMOVE_PICK_CONTRACT):
                manifest = deepcopy(loaded["manifest"]); manifest["contract"] = wrong_contract
                fixture.write_json(path, manifest)
                with self.assertRaisesRegex(ValueError, "contract"):
                    load_deferred(path.parent)

    def test_target_capture_does_not_silently_drop_extra_measured_object(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, meta = fixture.make_model(), metadata()
            capture = DeferredStateCapture(Path(temporary) / "capture", metadata=meta,
                joint_names=fixture.NAMES, warmup_ticks=50, start_time=1.25, model=model)
            value = fixture.frame(capture, model, mujoco.MjData(model), 0)
            value["objects"]["container"] = deepcopy(value["objects"]["target"])
            with self.assertRaises(CaptureRejected):
                capture.append(**value)
            self.assertTrue(capture.failed)
            self.assertEqual(capture.close(completed=True)["status"], "partial_capture_rejected")

    def test_spawn_preserves_independent_contract_roles_and_exact_archive_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, model, meta = Path(temporary), fixture.make_model(), metadata()
            prepared = prepare_deferred_model(root / "prepared", model=model,
                joint_names=fixture.NAMES, metadata=meta)
            session = DeferredEvidenceSession(prepared, cpus=[min(os.sched_getaffinity(0))], max_pending=128)
            try:
                loaded = session.receipt()["loaded_model"]
                self.assertEqual(loaded["physical_capture_contract"], XMOVE_BEND_PICK_CONTRACT)
                self.assertEqual(loaded["evidence_object_roles"], ["target"])
                asynchronous, _, _, _ = session.begin(root / "async", metadata=meta,
                    joint_names=fixture.NAMES, warmup_ticks=50, start_time=1.25, human_demo_uuid=fixture.UUID)
                synchronous = DeferredStateCapture(root / "sync", metadata=meta,
                    joint_names=fixture.NAMES, warmup_ticks=50, start_time=1.25, model=model)
                data = mujoco.MjData(model)
                for tick in range(51):
                    if capture_tick_index(tick) is not None:
                        value = fixture.frame(synchronous, model, data, tick)
                        synchronous.append(**value); asynchronous.append(**value)
                self.assertEqual(synchronous.close(completed=True), asynchronous.close(completed=True))
                for name in ("events.jsonl", "states.npy"):
                    self.assertEqual((root / "sync" / name).read_bytes(),
                        (root / "async/deferred_capture" / name).read_bytes())
            finally:
                self.assertTrue(session.close()["completed"])


class XMoveBendPickHistoryTests(unittest.TestCase):
    def test_native_policy_is_separate_from_grasp_mobile_and_container_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); meta, outcome = episode(root)
            self.assertTrue(validate_xmove_bend_pick_task_identity(meta)["passed"])
            with patch("pico_mobile_quality.evaluate_mobile_capture", side_effect=AssertionError("wrong task")), \
                 patch("pico_grasp_quality.evaluate_grasp_capture", side_effect=AssertionError("wrong task")), \
                 patch("handover_contact_quality.evaluate_capture", side_effect=AssertionError("wrong task")):
                reasons, report = _physical_precheck(root, outcome, meta)
            self.assertEqual(reasons, [], report)
            self.assertEqual(report["task_id"], TASK_ID)
            self.assertEqual(report["acceptance_gate_version"], GATE_VERSION)
            self.assertEqual(report["selection_status"], "selected")
            self.assertEqual(set(report["evaluator_source_sha256"]),
                {"pico_xmove_bend_pick_quality.py", "pico_episode_history.py"})
            self.assertFalse(report["dataset_ready"])
            job = {"episode_json_sha256": hashlib.sha256((root / "episode.json").read_bytes()).hexdigest(),
                "metadata_sha256": hashlib.sha256((root / "metadata.json").read_bytes()).hexdigest()}
            self.assertEqual(complete_geometry(root, job)["status"], "not_required")
            self.assertFalse((root / "quality_geometry_completion.json").exists())

    def test_native_false_is_official_failure_without_new_hold_or_walk_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); meta, outcome = episode(root, successful=False)
            reasons, report = _physical_precheck(root, outcome, meta)
            self.assertTrue(reasons)
            self.assertFalse(report["technical_blocked"])
            self.assertEqual(report["selection_status"], "official_task_not_successful")

    def test_original_capture_corruption_and_identity_drift_block_as_technical(self):
        for failure in ("source", "identity", "flag", "force", "grasp_contract", "row_sha", "state_dtype", "clock", "modes", "mode_bool", "policy", "legacy_policy"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); meta, outcome = episode(root)
                if failure == "modes":
                    meta["xmove_bend_pick_encoder_modes"] = [1, 2]
                elif failure == "mode_bool":
                    meta["xmove_bend_pick_encoder_modes"] = [False, 2]
                elif failure == "policy":
                    meta["xmove_bend_pick_control_policy"] = "planner_mode1"
                elif failure == "legacy_policy":
                    meta["mobile_control_policy"] = "upper_lower_planner_mode1"
                elif failure == "source":
                    meta["actual_task_source_sha256"] = "b" * 64
                elif failure == "identity":
                    meta.pop("xmove_bend_pick_adaptation")
                elif failure == "flag":
                    rows = fixture.rows(); rows[0].pop("official_task_success")
                    fixture.write_rows(root / "raw_quality_samples.jsonl", rows)
                elif failure == "force":
                    path = root / "applied_force_samples.jsonl"
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[0]["all_zero"] = False; fixture.write_rows(path, rows)
                else:
                    capture = outcome["capture"]
                    if failure == "grasp_contract":
                        capture["contract"] = GRASP_CONTRACT
                    elif failure == "clock":
                        capture["physics_hz"] = 100
                    elif failure == "row_sha":
                        path = root / "deferred_capture/events.jsonl"
                        rows = [json.loads(line) for line in path.read_text().splitlines()]
                        rows[0]["integration_state_sha256"] = "b" * 64
                        fixture.write_rows(path, rows)
                        capture["events_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    else:
                        path = root / "deferred_capture/states.npy"
                        np.save(path, np.load(path).astype(np.float32))
                        capture["states_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    fixture.write_json(root / "deferred_capture/capture.json", capture)
                reasons, report = _physical_precheck(root, outcome, meta)
                self.assertTrue(reasons)
                self.assertTrue(report["technical_blocked"], report)
                self.assertIsNone(report["official_task_success"])
                self.assertFalse(report["native_success_evidence_complete"])
                self.assertEqual(report["selection_status"], "technical_evidence_incomplete")


if __name__ == "__main__":
    unittest.main()
