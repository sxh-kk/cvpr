"""CPU-only task isolation, real-carry and unchanged placement regressions."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handover_contact_quality import SOLVED_STATE_STAGE, evaluate_capture
from pico_mobile_quality import GATE_VERSION, MOBILE_TASK_ID, evaluate_mobile_capture
from pico_episode_history import _physical_precheck
from test_audit_pico_teleop_episode import make_episode


def mobile_rows():
    rows = []
    for index in range(155):
        held = 5 <= index < 105
        x = 0. if index < 5 else min(.8, (index - 5) * .8 / 99)
        rows.append({"completed_steps": index + 1, "sim_time_s": (index + 1) / 50,
            "actual_target_xyz_m": [x, 0., .7 if held else .55 if index < 5 else .61],
            "actual_root_xyz_m": [x, 0., .75], "pelvis_z_m": .75,
            "contacts": {"left_hand": held, "right_hand": False, "container": index >= 105},
            "container_geometry": {"inside_container_footprint": True, "signed_margin_m": .02,
                "collision_vertical_ranges_overlap": True, "container_upright_cosine": 1.,
                "container_support_weight_fraction": 1., "support_measurement": "live_solved_mujoco_contact_force"},
            "solver_stage": SOLVED_STATE_STAGE, "official_task_success": False})
    return rows


class MobileQualityTests(unittest.TestCase):
    def check(self, rows):
        return evaluate_mobile_capture(rows, initial_target_z_m=.55)

    def test_single_hand_truth_passes_without_handover_sequence_or_native_success(self):
        rows = mobile_rows(); before = deepcopy(rows)
        report = self.check(rows)
        self.assertTrue(report["eligible_physics"], report)
        self.assertEqual(report["acceptance_gate_version"], GATE_VERSION)
        self.assertEqual(report["final_placement_geometric_validation"]["gate_version"], "placement-necessary-geometry-v2")
        self.assertFalse(report["official_task_success_observed"])
        self.assertFalse(report["native_success_used_as_physical_gate"])
        self.assertFalse(report["dataset_ready"])
        self.assertNotIn("verified_handover_success", report)
        self.assertFalse(report["contact_evidence"]["handover_contact_order_required"])
        self.assertFalse(evaluate_capture(rows, initial_target_z_m=.55)["eligible_physics"])
        self.assertEqual(rows, before)

    def test_right_single_hand_both_hands_and_hand_switch_are_all_legal(self):
        for mode in ("right", "both", "switch"):
            rows = mobile_rows()
            for index, row in enumerate(rows):
                if row["contacts"]["left_hand"]:
                    row["contacts"].update(left_hand=mode == "both" or mode == "switch" and index < 50,
                                           right_hand=mode != "switch" or index >= 50)
            self.assertTrue(self.check(rows)["eligible_physics"], mode)

    def test_native_true_does_not_override_failed_physics_and_is_saved_separately(self):
        rows = mobile_rows()
        for row in rows:
            row["actual_root_xyz_m"][0] = 0.
            row["official_task_success"] = True
        report = self.check(rows)
        self.assertTrue(report["official_task_success_observed"])
        self.assertEqual(report["official_task_success_measured_frames"], len(rows))
        self.assertFalse(report["eligible_physics"])

    def test_target_and_root_each_must_move_in_same_lifted_held_interval(self):
        for mode in ("target", "root", "split", "short", "airborne_no_hands"):
            rows = mobile_rows()
            for index, row in enumerate(rows):
                if mode == "target": row["actual_target_xyz_m"][0] = 0.
                if mode == "root": row["actual_root_xyz_m"][0] = 0.
                if mode == "split" and index < 105: row["actual_root_xyz_m"][0] = 0.
                if mode == "short":
                    row["actual_target_xyz_m"][0] *= .749 / .8
                    row["actual_root_xyz_m"][0] *= .749 / .8
                if mode == "airborne_no_hands": row["contacts"]["left_hand"] = False
            self.assertFalse(self.check(rows)["eligible_physics"], mode)

    def test_touching_or_shuffling_on_table_does_not_establish_lift(self):
        rows = mobile_rows()
        for row in rows: row["actual_target_xyz_m"][2] = .55
        self.assertFalse(self.check(rows)["eligible_physics"])

    def test_returning_target_to_pickup_point_is_not_transport(self):
        rows = mobile_rows()
        for row in rows[-50:]: row["actual_target_xyz_m"][0] = 0.
        report = self.check(rows)
        self.assertTrue(report["contact_evidence"]["checks"]["target_and_root_carried_075m"])
        self.assertFalse(report["eligible_physics"])

    def test_final_release_and_pairwise_stability_cannot_be_hidden_by_endpoint(self):
        for mode in ("left", "right", "motion"):
            rows = mobile_rows()
            if mode == "motion": rows[-20]["actual_target_xyz_m"][1] = .006
            else: rows[-1]["contacts"][mode + "_hand"] = True
            self.assertFalse(self.check(rows)["eligible_physics"], mode)

    def test_contact_and_solved_support_allow_exactly_one_missing_final_frame(self):
        for field in ("contact", "support"):
            rows = mobile_rows()
            def drop(row):
                if field == "contact": row["contacts"]["container"] = False
                else: row["container_geometry"]["container_support_weight_fraction"] = .49
            drop(rows[-1])
            self.assertTrue(self.check(rows)["eligible_physics"], field)
            drop(rows[-2])
            self.assertFalse(self.check(rows)["eligible_physics"], field)

    def test_geometry_upright_inside_overlap_and_original_support_are_required(self):
        for key, value in (("container_upright_cosine", .8), ("inside_container_footprint", False),
                           ("collision_vertical_ranges_overlap", False), ("support_measurement", "pose_reconstruction")):
            rows = mobile_rows()
            for row in rows[-50:]: row["container_geometry"][key] = value
            self.assertFalse(self.check(rows)["eligible_physics"], key)

    def test_missing_invalid_root_and_other_evidence_fail_closed(self):
        for key, value in (("actual_root_xyz_m", None), ("actual_root_xyz_m", [0., float("nan"), .7]),
                           ("actual_root_xyz_m", [0., 0.]), ("actual_root_xyz_m", [False, 0., .7]),
                           ("solver_stage", "post_forward"), ("completed_steps", 900), ("sim_time_s", 9.),
                           ("container_geometry", None), ("pelvis_z_m", .39)):
            rows = mobile_rows(); rows[-1][key] = value
            self.assertFalse(self.check(rows)["eligible_physics"], (key, value))
        rows = mobile_rows(); rows[-1]["contacts"].pop("right_hand")
        self.assertFalse(self.check(rows)["eligible_physics"])
        self.assertFalse(self.check([])["eligible_physics"])
        self.assertFalse(self.check([None])["eligible_physics"])


class MobileHistoryDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        make_episode(self.directory)

    def load(self, filename):
        return json.loads((self.directory / filename).read_text())

    def install_mobile(self):
        path = self.directory / "quality_samples.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for index, row in enumerate(rows):
            phase = index - 50
            x = 0. if phase < 0 else min(.8, phase * .8 / 49)
            row["actual_root_xyz_m"] = [x, 0., .75]
            row["actual_target_xyz_m"] = [x, 0., .55 if phase < 0 else .7 if phase < 50 else .61]
            row["contacts"] = {"left_hand": 0 <= phase < 50, "right_hand": False, "container": phase >= 50}
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        metadata = self.load("metadata.json"); metadata["task_id"] = MOBILE_TASK_ID
        return metadata, self.load("episode.json"), rows

    def test_mobile_dispatch_excludes_real_warmup_and_retains_force_integrity(self):
        metadata, outcome, _ = self.install_mobile()
        with patch("handover_contact_quality.evaluate_capture", side_effect=AssertionError("must not use handover")):
            reasons, report = _physical_precheck(self.directory, outcome, metadata)
        self.assertEqual(reasons, [], report)
        self.assertTrue(report["eligible_physics"])
        self.assertEqual(report["excluded_unsupervised_warmup_ticks50"], 50)
        path = self.directory / "applied_force_samples.jsonl"
        forces = [json.loads(line) for line in path.read_text().splitlines()]
        forces[1]["all_zero"] = False
        path.write_text("".join(json.dumps(row) + "\n" for row in forces))
        reasons, _ = _physical_precheck(self.directory, outcome, metadata)
        self.assertTrue(any("200Hz外力" in reason for reason in reasons), reasons)

    def test_unknown_or_missing_task_never_defaults_to_handover(self):
        metadata = self.load("metadata.json"); outcome = self.load("episode.json")
        for task in ("unknown", None):
            metadata["task_id"] = task
            with patch("handover_contact_quality.evaluate_capture", side_effect=AssertionError("wrong dispatch")):
                reasons, report = _physical_precheck(self.directory, outcome, metadata)
            self.assertFalse(report)
            self.assertTrue(any("任务身份" in reason for reason in reasons))

    def test_handover_dispatch_is_unchanged(self):
        reasons, report = _physical_precheck(self.directory, self.load("episode.json"), self.load("metadata.json"))
        self.assertEqual(reasons, [])
        self.assertTrue(report["verified_handover_success"])
        self.assertEqual(report["acceptance_gate_version"], "handover-physics-v2")

    def test_mobile_failed_gate_has_chinese_transport_reason(self):
        metadata, outcome, rows = self.install_mobile()
        for row in rows: row["actual_root_xyz_m"][0] = 0.
        (self.directory / "quality_samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        reasons, report = _physical_precheck(self.directory, outcome, metadata)
        self.assertFalse(report["eligible_physics"])
        self.assertTrue(any("0.75米" in reason for reason in reasons), reasons)
        self.assertFalse(any("右手→双手→左手" in reason for reason in reasons))

    def test_missing_mobile_root_reports_a_specific_chinese_evidence_error(self):
        metadata, outcome, rows = self.install_mobile()
        rows[60].pop("actual_root_xyz_m")
        (self.directory / "quality_samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        reasons, report = _physical_precheck(self.directory, outcome, metadata)
        self.assertFalse(report["eligible_physics"])
        self.assertIn("缺少或无效的实际机器人根部三维位置证据", reasons)

    def test_warmup_grab_and_capture_sha_tamper_remain_rejected(self):
        metadata, outcome, rows = self.install_mobile()
        rows[10]["contacts"]["left_hand"] = True
        (self.directory / "quality_samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        reasons, _ = _physical_precheck(self.directory, outcome, metadata)
        self.assertTrue(any("起始无监督历史" in reason for reason in reasons), reasons)
        with (self.directory / "capture/events.jsonl").open("a") as stream: stream.write("\n")
        reasons, _ = _physical_precheck(self.directory, outcome, metadata)
        self.assertTrue(any("SHA" in reason for reason in reasons), reasons)


if __name__ == "__main__":
    unittest.main()
