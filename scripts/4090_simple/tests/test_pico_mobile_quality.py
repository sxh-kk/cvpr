"""CPU-only native Mobile task selection and task dispatch regressions."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handover_contact_quality import SOLVED_STATE_STAGE
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

    def native_rows(self, flags=(False, True, False)):
        return [{"completed_steps": index + 1, "sim_time_s": (index + 1) / 50,
                 "solver_stage": SOLVED_STATE_STAGE, "official_task_success": flag}
                for index, flag in enumerate(flags)]

    def assert_technical_failure(self, report, error_text):
        self.assertFalse(report["eligible_physics"], report)
        self.assertFalse(report["verified_mobile_pick_place_success"])
        self.assertFalse(report["native_success_evidence_complete"])
        self.assertIsNone(report["official_task_success"])
        self.assertEqual(report["selection_status"], "technical_evidence_incomplete")
        self.assertTrue(any(error_text in error for error in report["capture_validation_errors"]), report)

    def test_native_true_passes_without_lift_transport_contacts_or_geometry(self):
        rows = self.native_rows(); before = deepcopy(rows)
        report = self.check(rows)
        self.assertTrue(report["eligible_physics"], report)
        self.assertTrue(report["verified_mobile_pick_place_success"])
        self.assertEqual(report["acceptance_gate_version"], "mobile-native-task-success-v2")
        self.assertEqual(report["acceptance_gate_version"], GATE_VERSION)
        self.assertEqual(report["task_id"], MOBILE_TASK_ID)
        self.assertEqual(report["selection_policy"], "native_only")
        self.assertEqual(report["selection_status"], "selected")
        self.assertTrue(report["official_task_success"])
        self.assertTrue(report["official_task_success_observed"])
        self.assertTrue(report["native_success_evidence_complete"])
        self.assertTrue(report["native_success_used_as_physical_gate"])
        self.assertFalse(report["official_task_flag_modified"])
        self.assertFalse(report["dataset_ready"])
        self.assertEqual(report["capture_validation_errors"], [])
        self.assertEqual(report["official_task_success_measured_frames"], len(rows))
        self.assertEqual(report["official_task_success_expected_frames"], len(rows))
        self.assertNotIn("verified_handover_success", report)
        self.assertNotIn("contact_evidence", report)
        self.assertNotIn("final_placement_geometric_validation", report)
        self.assertEqual(rows, before)

    def test_any_original_native_true_is_selected_even_if_final_flag_false(self):
        for success_index in (0, 51, 154):
            rows = mobile_rows()
            rows[success_index]["official_task_success"] = True
            report = self.check(rows)
            self.assertTrue(report["eligible_physics"], success_index)
            self.assertEqual(report["official_task_success_measured_frames"], 155)
            self.assertEqual(report["capture_validation_errors"], [])

    def test_all_native_false_fails_even_when_old_additional_physics_gates_pass(self):
        rows = mobile_rows(); before = deepcopy(rows)
        report = self.check(rows)
        self.assertFalse(report["eligible_physics"])
        self.assertFalse(report["verified_mobile_pick_place_success"])
        self.assertFalse(report["official_task_success"])
        self.assertFalse(report["official_task_success_observed"])
        self.assertTrue(report["native_success_evidence_complete"])
        self.assertEqual(report["selection_status"], "official_task_not_successful")
        self.assertEqual(report["capture_validation_errors"], [])
        self.assertEqual(rows, before)

    def test_extra_physics_fields_do_not_override_native_true(self):
        rows = self.native_rows()
        for row in rows:
            row.update(actual_root_xyz_m=None, actual_target_xyz_m=[0., 0., 0.],
                       pelvis_z_m=0., contacts={"left_hand": True}, container_geometry=None)
        self.assertTrue(self.check(rows)["eligible_physics"])

    def test_initial_height_is_only_a_compatibility_argument(self):
        rows = self.native_rows()
        for initial in (None, float("nan"), "unused", -100):
            report = evaluate_mobile_capture(rows, initial_target_z_m=initial)
            self.assertTrue(report["eligible_physics"], initial)
        self.assertTrue(evaluate_mobile_capture(rows)["eligible_physics"])

    def test_missing_native_boolean_is_technical_even_with_other_native_true(self):
        rows = self.native_rows()
        rows[0].pop("official_task_success")
        report = self.check(rows)
        self.assert_technical_failure(report, "official_task_success")
        self.assertTrue(report["official_task_success_observed"])
        self.assertEqual(report["official_task_success_measured_frames"], 2)
        self.assertEqual(report["official_task_success_expected_frames"], 3)

    def test_invalid_native_values_are_not_coerced_to_success_or_task_failure(self):
        for value in (None, 0, 1, "true", "false", float("nan"), [], {}):
            with self.subTest(value=value):
                rows = self.native_rows((False, False, False))
                rows[1]["official_task_success"] = value
                self.assert_technical_failure(self.check(rows), "official_task_success")
        rows = self.native_rows((None,))
        report = self.check(rows)
        self.assert_technical_failure(report, "official_task_success")
        self.assertIsNone(report["official_task_success_observed"])

    def test_nonfinite_and_invalid_clocks_are_technical_failures(self):
        for value in (float("nan"), float("inf"), float("-inf"), True, None, ".04"):
            with self.subTest(value=value):
                rows = self.native_rows()
                rows[1]["sim_time_s"] = value
                self.assert_technical_failure(self.check(rows), "sim_time_s")

    def test_clock_must_cover_every_original_50hz_step(self):
        for clock in (.02, .041, .08):
            rows = self.native_rows()
            rows[1]["sim_time_s"] = clock
            self.assert_technical_failure(self.check(rows), "real50Hz")

    def test_steps_must_be_original_dense_integers_starting_at_one(self):
        for value in (None, True, 2., 1, 3, 900):
            with self.subTest(value=value):
                rows = self.native_rows()
                rows[1]["completed_steps"] = value
                self.assert_technical_failure(self.check(rows), "completed_steps")
        rows = self.native_rows()
        for row in rows:
            row["completed_steps"] += 50
        self.assert_technical_failure(self.check(rows), "completed_steps")

    def test_unsolved_or_missing_acquisition_stage_is_technical(self):
        rows = self.native_rows()
        rows[1]["solver_stage"] = "post_forward"
        self.assert_technical_failure(self.check(rows), "solved")
        rows[1].pop("solver_stage")
        self.assert_technical_failure(self.check(rows), "solved")

    def test_empty_and_nonmapping_samples_are_technical(self):
        self.assert_technical_failure(self.check([]), "no measured samples")
        self.assert_technical_failure(self.check([None]), "not a mapping")
        rows = self.native_rows(); rows[0] = None
        self.assert_technical_failure(self.check(rows), "not a mapping")


class MobileHistoryDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        make_episode(self.directory)

    def load(self, filename):
        return json.loads((self.directory / filename).read_text())

    def write_rows(self, filename, rows):
        (self.directory / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))

    def install_mobile(self, *, successful=True):
        rows = [json.loads(line) for line in (self.directory / "quality_samples.jsonl").read_text().splitlines()]
        for index, row in enumerate(rows):
            row["official_task_success"] = successful and index == 10
        self.write_rows("quality_samples.jsonl", rows)
        metadata = self.load("metadata.json")
        metadata.update(task_id=MOBILE_TASK_ID,
            actual_task_source_sha256="fee537d56e134b0399ff147013438168601498ab9b3ee068658dcad5fb8ca521",
            actual_native_success_criteria=.9)
        (self.directory / "metadata.json").write_text(json.dumps(metadata) + "\n")
        return metadata, self.load("episode.json"), rows

    def assert_technical(self, report):
        self.assertFalse(report["eligible_physics"], report)
        self.assertTrue(report["technical_blocked"])
        self.assertTrue(report["technical_errors"])
        self.assertEqual(report["selection_status"], "technical_evidence_incomplete")
        self.assertIsNone(report["official_task_success"])

    def test_mobile_dispatch_uses_all_original_native_steps_and_retains_force_integrity(self):
        metadata, outcome, rows = self.install_mobile()
        with patch("handover_contact_quality.evaluate_capture", side_effect=AssertionError("must not use handover")):
            reasons, report = _physical_precheck(self.directory, outcome, metadata)
        self.assertEqual(reasons, [], report)
        self.assertTrue(report["eligible_physics"])
        self.assertEqual(report["excluded_unsupervised_warmup_ticks50"], 0)
        self.assertEqual(report["official_task_success_measured_frames"], len(rows))
        self.assertEqual(report["selection_policy"], "native_only")
        self.assertFalse(report["technical_blocked"])
        forces = [json.loads(line) for line in (self.directory / "applied_force_samples.jsonl").read_text().splitlines()]
        forces[1]["all_zero"] = False
        self.write_rows("applied_force_samples.jsonl", forces)
        reasons, report = _physical_precheck(self.directory, outcome, metadata)
        self.assertTrue(any("200Hz外力" in reason for reason in reasons), reasons)
        self.assert_technical(report)

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

    def test_complete_all_false_is_task_failure_without_additional_success_reasons(self):
        metadata, outcome, _ = self.install_mobile(successful=False)
        reasons, report = _physical_precheck(self.directory, outcome, metadata)
        self.assertFalse(report["eligible_physics"])
        self.assertFalse(report["official_task_success"])
        self.assertFalse(report["technical_blocked"])
        self.assertEqual(report["selection_status"], "official_task_not_successful")
        self.assertEqual(reasons, ["官方任务成功标志未出现"])
        self.assertFalse(any("0.75米" in reason or "右手→双手→左手" in reason for reason in reasons))

    def test_extra_root_contact_geometry_lift_and_initial_height_are_not_required(self):
        metadata, outcome, rows = self.install_mobile()
        metadata.pop("initial_target_z_m")
        rows = [{key: row[key] for key in
                 ("completed_steps", "sim_time_s", "solver_stage", "official_task_success")} for row in rows]
        self.write_rows("quality_samples.jsonl", rows)
        reasons, report = _physical_precheck(self.directory, outcome, metadata)
        self.assertEqual(reasons, [], report)
        self.assertTrue(report["eligible_physics"])

    def test_raw_native_evidence_takes_priority_over_completed_geometry_quality(self):
        metadata, outcome, rows = self.install_mobile(successful=False)
        raw = deepcopy(rows); raw[10]["official_task_success"] = True
        self.write_rows("raw_quality_samples.jsonl", raw)
        reasons, report = _physical_precheck(self.directory, outcome, metadata)
        self.assertEqual(reasons, [], report)
        self.assertTrue(report["eligible_physics"])
        self.assertEqual(report["native_flag_source_file"], "raw_quality_samples.jsonl")
        raw[0].pop("official_task_success")
        self.write_rows("raw_quality_samples.jsonl", raw)
        rows[-1]["official_task_success"] = True
        self.write_rows("quality_samples.jsonl", rows)
        reasons, report = _physical_precheck(self.directory, outcome, metadata)
        self.assert_technical(report)
        self.assertTrue(any("official_task_success" in reason for reason in reasons), reasons)

    def test_missing_or_nonboolean_native_is_technical_even_with_other_true(self):
        for value in (None, 1, "true"):
            metadata, outcome, rows = self.install_mobile()
            rows[0]["official_task_success"] = value
            self.write_rows("quality_samples.jsonl", rows)
            reasons, report = _physical_precheck(self.directory, outcome, metadata)
            self.assert_technical(report)
            self.assertTrue(any("official_task_success" in reason for reason in reasons), reasons)
            self.assertNotIn("官方任务成功标志未出现", reasons)

    def test_initial_warmup_contact_or_motion_is_not_an_extra_mobile_success_gate(self):
        metadata, outcome, rows = self.install_mobile()
        rows[10]["contacts"]["left_hand"] = True
        rows[10]["actual_target_xyz_m"] = [10., 10., 10.]
        self.write_rows("quality_samples.jsonl", rows)
        reasons, report = _physical_precheck(self.directory, outcome, metadata)
        self.assertEqual(reasons, [], report)
        self.assertTrue(report["eligible_physics"])

    def test_capture_sha_tamper_remains_a_technical_failure(self):
        metadata, outcome, _ = self.install_mobile()
        with (self.directory / "capture/events.jsonl").open("a") as stream: stream.write("\n")
        reasons, report = _physical_precheck(self.directory, outcome, metadata)
        self.assertTrue(any("SHA" in reason for reason in reasons), reasons)
        self.assert_technical(report)

    def test_native_clock_nan_is_technical_not_official_task_failure(self):
        metadata, outcome, rows = self.install_mobile()
        rows[2]["sim_time_s"] = float("nan")
        self.write_rows("quality_samples.jsonl", rows)
        reasons, report = _physical_precheck(self.directory, outcome, metadata)
        self.assert_technical(report)
        self.assertNotIn("官方任务成功标志未出现", reasons)

    def test_missing_or_wrong_official_task_binding_is_technical(self):
        for key, value in (("actual_task_source_sha256", None), ("actual_task_source_sha256", "0" * 64),
                           ("actual_native_success_criteria", None), ("actual_native_success_criteria", .8),
                           ("actual_native_success_criteria", True)):
            with self.subTest(key=key, value=value):
                metadata, outcome, _ = self.install_mobile()
                if value is None:
                    metadata.pop(key)
                else:
                    metadata[key] = value
                reasons, report = _physical_precheck(self.directory, outcome, metadata)
                self.assert_technical(report)
                self.assertFalse(report["task_identity_validation"]["passed"])
                self.assertNotIn("官方任务成功标志未出现", reasons)


if __name__ == "__main__":
    unittest.main()
