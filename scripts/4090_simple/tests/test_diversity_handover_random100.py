"""No new trajectory is collected: synthetic negative diversity contracts only."""
from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diversity_handover_random100 as diversity


def registry():
    sources = [i for i in range(100) if i not in diversity.HELDOUT][:50]
    return {"contract": diversity.REGISTRY_CONTRACT, "admissions": [{"candidate_id": f"new-{i}",
        "source_episode_index": source, "seed": i, "run": f"/not-a-real-run/new-{i}"}
        for i, source in enumerate(sources * 2)]}


def observations():
    result = []
    for i, t in enumerate(np.linspace(-1., 1., 100)):
        source = {role: {"source_pose7": [0., 0., .6, 1., 0., 0., 0.]} for role in ("target", "container")}
        angle = .25 * t
        actual = {role: {"pose7": [.05 * t, -.05 * t, .6, math.cos(angle / 2), 0., 0., math.sin(angle / 2)]}
                  for role in source}
        result.append({"source": source, "actual": actual, "actual_MDL_path": f"/fixture/material{i // 2}.mdl", "actual_MDL_name": f"material{i // 2}"})
    return result


def motion(rows=80):
    t = np.linspace(0., 1., rows)
    joints = np.stack([.1 * t + .01 * i for i in range(43)], axis=1)
    base = np.c_[t * .1, np.sin(t * 2) * .01, t * 0 + .75]
    points = np.stack([base, base + np.c_[t * .03, t * 0 + .2, t * .02],
                       base + np.c_[t * -.03, t * 0 - .2, t * .02]], axis=1)
    return joints, points


class DiversityContractTests(unittest.TestCase):
    def test_exact100_at_least50_original_sources_cap2(self):
        self.assertEqual(len(diversity.validate_registry_shape(registry())), 50)
        for change in (lambda r: r["admissions"].pop(),
                       lambda r: r["admissions"][0].update(source_episode_index=39),
                       lambda r: r["admissions"][0].update(source_episode_index=38),
                       lambda r: r["admissions"][1].update(source_episode_index=r["admissions"][0]["source_episode_index"]),
                       lambda r: r["admissions"][1].update(candidate_id=r["admissions"][0]["candidate_id"]),
                       lambda r: r["admissions"][1].update(run=r["admissions"][0]["run"])):
            bad = registry(); change(bad)
            with self.assertRaises(ValueError): diversity.validate_registry_shape(bad)

    def test_measured_scene_coverage_not_seed_cardinality(self):
        passed, metrics = diversity.scene_coverage(observations())
        self.assertTrue(passed); self.assertEqual(metrics["actual_table_MDL_identity_count"], 50)
        bad = observations()
        for row in bad:
            row["actual"]["target"]["pose7"] = [0., 0., .6, 1., 0., 0., 0.]
        self.assertFalse(diversity.scene_coverage(bad)[0])
        bad = observations()
        for row in bad: row["actual_MDL_path"] = "/one.mdl"; row["actual_MDL_name"] = "one"
        self.assertFalse(diversity.scene_coverage(bad)[0])

    def test_global_rigid_transform_does_not_create_new_motion(self):
        joints, points = motion(); angle = .8
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0.], [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
        transformed = points @ rotation.T + np.array([40., -10., 2.])
        first = diversity.normalize_measured_motion(joints, points, np.eye(3))
        second = diversity.normalize_measured_motion(joints, transformed, rotation)
        self.assertTrue(np.allclose(first["fk"], second["fk"], atol=1e-12))
        passed, metrics = diversity.near_duplicate_metrics([first, second], ["seed1", "seed999"])
        self.assertFalse(passed); self.assertEqual(len(metrics["near_duplicate_pairs"]), 1)

    def test_normalized_time_rejects_only_duration_change_as_diversity(self):
        q1, p1 = motion(80); q2, p2 = motion(160)
        first = diversity.normalize_measured_motion(q1, p1, np.eye(3))
        second = diversity.normalize_measured_motion(q2, p2, np.eye(3))
        self.assertFalse(diversity.near_duplicate_metrics([first, second], ["fast", "slow"])[0])
        second["joints"] += .06
        self.assertTrue(diversity.near_duplicate_metrics([first, second], ["first", "different"])[0])

    def test_missing_lifted_stage_or_zero_new_control_cannot_pass(self):
        stages = {name: {"start_step": 1 + i * 5, "end_step": 5 + i * 5, "frame_count": 5}
                  for i, name in enumerate(("right_only", "both_hands", "left_only"))}
        admission = {"physical_gate": {"contact_evidence": {"strict_handover_success": True, "ordered_lifted_stages": stages},
            "final_placement_geometric_validation": {"require_3d_evidence": True, "v2_passed": True}}}
        diagnostic = {"current_arm_joint_delta": [.01] * 14, "encoder_input_sha256": "a" * 64,
                      **{key: False for key in ("simulator_state_mutated", "object_state_mutated", "external_force_applied", "acceptance_gate_modified")}}
        self.assertEqual(diversity._stages_and_adaptation(admission, [{"diagnostics": diagnostic}])[:2], (True, True))
        diagnostic["current_arm_joint_delta"] = [0.] * 14
        self.assertFalse(diversity._stages_and_adaptation(admission, [{"diagnostics": diagnostic}])[1])
        stages["left_only"] = None
        self.assertFalse(diversity._stages_and_adaptation(admission, [{"diagnostics": diagnostic}])[0])

    def test_replacement_advice_only_existing_ids_and_no_axis_only_deletion(self):
        rows = observations(); ids = [f"actual-{i}" for i in range(100)]
        _, metric = diversity.scene_coverage(rows)
        pairs = {"near_duplicate_pairs": [{"first": ids[0], "second": ids[1]}]}
        stages = [True] * 100; stages[3] = False
        adapted = [True] * 100; adapted[4] = False
        advice = diversity._replacement_advice(ids, rows, metric, pairs, stages, adapted)
        self.assertEqual(advice["replacement_candidate_ids"], [ids[1], ids[3], ids[4]])
        self.assertFalse(advice["replacement_guarantees_pass"])
        report = dict(advice, registry_candidate_ids=ids, passed=False)
        report["replacement_candidate_ids"] += ["not-an-admitted-candidate", ids[1]]
        self.assertEqual(diversity.replacement_candidates(report), [ids[1], ids[3], ids[4]])
        report["passed"] = True
        self.assertEqual(diversity.replacement_candidates(report), [])
        for row in rows: row["actual"]["target"]["pose7"] = [0., 0., .6, 1., 0., 0., 0.]
        _, metric = diversity.scene_coverage(rows)
        advice = diversity._replacement_advice(ids, rows, metric, {"near_duplicate_pairs": []}, [True] * 100, [True] * 100)
        self.assertEqual(advice["replacement_candidate_ids"], [])
        self.assertEqual(len(advice["targeted_scene_sampling_gaps"]), 3)

    def test_duplicate_actual_scene_ab_suggests_later_candidate(self):
        rows = observations(); ids = [f"actual-{i}" for i in range(100)]
        _, metric = diversity.scene_coverage(rows)
        scenes = [f"real-scene-{i}" for i in range(100)]; scenes[1] = scenes[0]
        advice = diversity._replacement_advice(ids, rows, metric, {"near_duplicate_pairs": []}, [True] * 100, [True] * 100, scenes)
        self.assertEqual(advice["replacement_candidate_ids"], [ids[1]])
        self.assertEqual(advice["replacement_reasons_by_candidate"][ids[1]], ["duplicate_actual_scene_proposal_of:" + ids[0]])

    def test_formal_requirements_cannot_omit_or_weaken_final50_or_unique_scene(self):
        from audit_handover_random100 import ADDITIONAL_DATASET_QUALITY
        value = {"contract": "cvpr-handover-random100-rebuild-requirements-v1", "user_requested_count": 100,
            "minimum_distinct_accepted_original_sources": 50, "maximum_accepted_per_source": 2,
            "protected_VAL_sources": [39, 65, 80, 92, 94], "truncated_source_excluded": 38,
            "admission": {"unchanged_physical_gate": "handover-physics-v2"},
            "formal_scene_sampling": {"XY_max_m": .05, "yaw_max_rad": .25,
                "minimum_accepted_XY_range_each_role_axis_m": .06, "minimum_accepted_XY_std_each_role_axis_m": .012,
                "minimum_accepted_yaw_range_each_role_rad": .25, "minimum_distinct_actual_TRAIN_table_MDL": 50,
                "unique_actual_scene_proposal_sha256": True},
            "additional_dataset_quality": deepcopy(ADDITIONAL_DATASET_QUALITY)}
        diversity.validate_requirements(value)
        for mutate in (lambda r: r.pop("additional_dataset_quality"),
                       lambda r: r["additional_dataset_quality"]["actual_full_outer_footprint_final50"].update(window_states=49),
                       lambda r: r["additional_dataset_quality"]["actual_full_outer_footprint_final50"].update(minimum_outer_hull_margin_m=0.),
                       lambda r: r["formal_scene_sampling"].pop("unique_actual_scene_proposal_sha256"),
                       lambda r: r["formal_scene_sampling"].update(unique_actual_scene_proposal_sha256=False)):
            bad = deepcopy(value); mutate(bad)
            with self.assertRaises(ValueError): diversity.validate_requirements(bad)

    def test_material_deficit_suggests_only_redundant_real_material_rows(self):
        rows = observations(); ids = [f"actual-{i}" for i in range(100)]
        rows[-2]["actual_MDL_path"] = rows[-1]["actual_MDL_path"] = rows[0]["actual_MDL_path"]
        rows[-2]["actual_MDL_name"] = rows[-1]["actual_MDL_name"] = rows[0]["actual_MDL_name"]
        _, metric = diversity.scene_coverage(rows)
        self.assertEqual(metric["actual_table_MDL_identity_count"], 49)
        advice = diversity._replacement_advice(ids, rows, metric, {"near_duplicate_pairs": []}, [True] * 100, [True] * 100)
        self.assertEqual(advice["replacement_candidate_ids"], [ids[1]])
        self.assertIn("duplicate_actual_table_MDL", advice["replacement_reasons_by_candidate"][ids[1]][0])


if __name__ == "__main__": unittest.main()
