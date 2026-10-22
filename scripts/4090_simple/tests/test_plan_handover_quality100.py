"""CPU-only planner tests: lineage, quotas and proposals never equal data."""
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "quality100_planner", Path(__file__).resolve().parents[1] / "plan_handover_quality100.py")
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


def fixture():
    source_key = "test-source-Handover-v1"
    manifest = {"status": "complete", "release_ready": True, "is_official": False,
                "physics_acceptance_gate": "handover-physics-v2", "source_dataset_key": source_key,
                "episodes": []}
    selection = {"schema_version": "handover-final50-release-selection-v1",
                 "source_dataset_key": source_key, "episodes": []}
    entries = [(i, "train", "original_or_repaired") for i in range(24)]
    entries += [(100 + i, "train", "task_scene_regeneration") for i in range(21)]
    entries += [(i, "val", "original_or_repaired") for i in planner.HELDOUT_VAL]
    for source_id, split, category in entries:
        video = hashlib.sha256(f"video{source_id}".encode()).hexdigest()
        family = [{"source_dataset_key": source_key, "source_episode_index": source_id}]
        lineage = {"data_category": category, "lineage_complete": True,
                   "split_restriction": "unrestricted", "lineage_group_sources": family}
        if category == "task_scene_regeneration":
            lineage.update(split_restriction="train_only", motion_template_source_episode_index=0)
            family.append({"source_dataset_key": source_key, "source_episode_index": 0})
        manifest["episodes"].append({"source_episode_index": source_id, "split": split,
            "lineage": lineage, "source_video_sha256": video,
            "source_parquet_sha256": hashlib.sha256(f"parquet{source_id}".encode()).hexdigest()})
        selection["episodes"].append({"source_scene_episode_index": source_id,
            "source_dataset_key": source_key, "split": split, "data_category": category,
            "reviewed_video_sha256": video, "numeric_status": "passed",
            "physics_gate": "handover-physics-v2", "visual_qa_passed": True})
    manifest["selection_sha256"] = "a" * 64
    return manifest, selection


def build(manifest, selection, **kwargs):
    return planner.build_plan(manifest, selection, manifest_sha256="b" * 64,
                              selection_sha256="a" * 64, **kwargs)


class Quality100PlanTests(unittest.TestCase):
    def test_round_robin_balances_all24_original_train_ancestors(self):
        manifest, selection = fixture()
        plan = build(manifest, selection)
        self.assertEqual(plan["target_accepted_count"], 100)
        self.assertEqual(len(plan["candidates"]), 120)
        counts = Counter(c["ancestor"]["source_episode_index"] for c in plan["candidates"])
        self.assertEqual(counts, {i: 5 for i in range(24)})
        self.assertEqual({c["ancestor"]["source_episode_index"] for c in plan["candidates"][:24]}, set(range(24)))
        self.assertEqual(plan["acceptance_limits"]["maximum_accepted_per_ancestor"], 5)

    def test_no_val_or_template_regeneration_is_eligible(self):
        manifest, selection = fixture()
        plan = build(manifest, selection)
        ids = {c["ancestor"]["source_episode_index"] for c in plan["candidates"]}
        self.assertFalse(ids & set(planner.HELDOUT_VAL))
        self.assertFalse(ids & set(range(100, 121)))
        self.assertEqual(len(plan["excluded_source_scenes"]), 26)
        manifest["episodes"][0]["lineage"]["split_restriction"] = "train_only"
        with self.assertRaisesRegex(ValueError, "Exactly24"):
            build(manifest, selection)

    def test_shared_train_val_ancestor_or_selection_disagreement_refused(self):
        manifest, selection = fixture()
        manifest["episodes"][-1]["lineage"]["lineage_group_sources"].append(
            {"source_dataset_key": manifest["source_dataset_key"], "source_episode_index": 0})
        with self.assertRaisesRegex(ValueError, "shared TRAIN/VAL"):
            build(manifest, selection)
        manifest, selection = fixture()
        selection["episodes"][0]["split"] = "val"
        with self.assertRaisesRegex(ValueError, "split, category"):
            build(manifest, selection)

    def test_accepted_ancestor_cannot_hide_a_template_parent(self):
        manifest, selection = fixture()
        manifest["episodes"][1]["lineage"]["motion_template_source_episode_index"] = 0
        with self.assertRaisesRegex(ValueError, "unambiguous original ancestor"):
            build(manifest, selection)

    def test_target_below_coverage_or_above_capacity_and_looser_cap_rejected(self):
        manifest, selection = fixture()
        for kwargs in ({"target_accepted": 23}, {"target_accepted": 121},
                       {"max_per_ancestor": 4}, {"max_per_ancestor": 6},
                       {"target_accepted": True}, {"seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build(manifest, selection, **kwargs)

    def test_seed_is_deterministic_with_actual_distinct_object_proposals(self):
        manifest, selection = fixture()
        original = copy.deepcopy((manifest, selection))
        plan = build(manifest, selection, seed=41)
        self.assertEqual(plan, build(manifest, selection, seed=41))
        self.assertEqual((manifest, selection), original)
        self.assertNotEqual(plan["candidates"], build(manifest, selection, seed=42)["candidates"])
        shuffled = copy.deepcopy(manifest)
        shuffled["episodes"].reverse()
        self.assertEqual(plan["candidates"], build(shuffled, selection, seed=41)["candidates"])
        proposals = {json.dumps(c["initial_object_deltas"], sort_keys=True) for c in plan["candidates"]}
        self.assertEqual(len(proposals), 120)
        self.assertEqual(len({c["candidate_id"] for c in plan["candidates"]}), 120)
        for c in plan["candidates"]:
            for values in c["initial_object_deltas"].values():
                self.assertTrue(all(0 < abs(v) <= .01 for v in values["xy_delta_m"]))
                self.assertTrue(0 < abs(values["world_z_yaw_delta_rad"]) <= .05)

    def test_plan_never_complete_or_accepted_and_has_strict_future_gate(self):
        manifest, selection = fixture()
        plan = build(manifest, selection)
        self.assertEqual(plan["status"], "planned_not_collected")
        self.assertEqual((plan["collected_count"], plan["accepted_count"]), (0, 0))
        self.assertFalse(plan["progress_and_finalization"]["dataset_complete"])
        self.assertFalse(plan["progress_and_finalization"]["final_ready"])
        self.assertFalse(plan["progress_and_finalization"]["collector_implemented"])
        for candidate in plan["candidates"]:
            self.assertFalse(candidate["accepted"])
            self.assertIsNone(candidate["independent_acceptance_sha256"])
            self.assertEqual(candidate["trajectory_files"], [])
        gate = plan["required_physical_acceptance"]
        self.assertEqual((gate["policy_action_hz"], gate["low_level_controller_hz"], gate["authoritative_physics_hz"]), (30, 50, 200))
        self.assertEqual(gate["acceptance_observation_hz"], 50)
        thresholds = gate["source_thresholds"]
        self.assertEqual(thresholds["final_window_frames"], 50)
        self.assertEqual(thresholds["minimum_final_container_contact_frames"], 49)
        self.assertTrue(thresholds["hands_released_on_every_final_frame"])
        self.assertEqual(thresholds["maximum_pairwise_displacement_m_exclusive"], .005)
        self.assertEqual(gate["source_geometry_v2_thresholds"]["minimum_support_frames"], 49)
        self.assertIn("before observation mj_forward", gate["solver_force_measurement_phase"])
        self.assertNotIn("both_hands_released_every_physics_tick", gate)
        self.assertTrue(gate["no_during_control_object_restore"])

    def test_test_space_is_frozen_disjoint_pending_and_not_old_val(self):
        manifest, selection = fixture()
        plan = build(manifest, selection)
        test = plan["frozen_test_reservation"]
        self.assertEqual(test["status"], "awaiting_independent_lineage_binding")
        self.assertFalse(test["existing_VAL_relabelled_as_TEST"])
        self.assertTrue(test["training_access_forbidden"])
        self.assertTrue(all(s >= planner.TEST_SEED_MIN for s in test["reset_seeds"]))
        for c in plan["candidates"]:
            self.assertTrue(all(c[key] <= planner.TRAIN_SEED_MAX for key in ("physics_seed", "material_seed", "generation_seed")))
        self.assertEqual(test["seed_manifest_sha256"], planner.digest(test["reset_seeds"]))

    def test_changed_selection_hash_or_unaccepted_source_refused(self):
        manifest, selection = fixture()
        manifest["selection_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "Selection bytes"):
            build(manifest, selection)
        manifest, selection = fixture()
        manifest["release_ready"] = False
        with self.assertRaisesRegex(ValueError, "accepted"):
            build(manifest, selection)

    def test_disk_output_is_fresh_plan_only_and_invalid_request_writes_nothing(self):
        manifest, selection = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            selection_path = source / "selection.json"
            selection_path.write_text(json.dumps(selection))
            manifest["selection_sha256"] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
            manifest_path = source / "conversion_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            output = root / "plan"
            with self.assertRaises(ValueError):
                planner.prepare(manifest_path, selection_path, output, target_accepted=200)
            self.assertFalse(output.exists())
            plan = planner.prepare(manifest_path, selection_path, output)
            self.assertEqual(plan["accepted_count"], 0)
            self.assertEqual({p.name for p in output.iterdir()}, {"candidate_plan.json", "PLAN_ONLY_NOT_COLLECTED"})
            self.assertIn("accepted=0", (output / "PLAN_ONLY_NOT_COLLECTED").read_text())
            self.assertEqual(plan["source_manifest_sha256"], hashlib.sha256(manifest_path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "fresh absolute output"):
                planner.prepare(manifest_path, selection_path, output)


if __name__ == "__main__":
    unittest.main()
