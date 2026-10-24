"""Synthetic unit fixtures only; no failed pilot becomes accepted integration."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pack_handover_random100 as packer
import verify_handover_random100 as verifier


def requirements():
    return {"contract": "cvpr-handover-random100-rebuild-requirements-v1", "user_requested_count": 100,
        "protected_VAL_sources": sorted(packer.HELDOUT), "truncated_source_excluded": 38,
        "minimum_distinct_accepted_original_sources": 50, "maximum_accepted_per_source": 2,
        "additional_dataset_quality": deepcopy(packer.ADDITIONAL_DATASET_QUALITY),
        "formal_TRAIN_seed_range_inclusive": [1,2**30-1],
        "formal_scene_sampling": {"unique_actual_scene_proposal_sha256":True}}


def entries():
    sources = [s for s in range(100) if s not in packer.EXCLUDED][:50]
    return [{"candidate_id": f"fixture-{i}", "source_episode_index": sources[i // 2], "seed": i+1,
             "hand_condition": "declared", "run": f"/synthetic/run-{i}",
             "admission_path": f"/synthetic/admission-{i}.json", "admission_sha256": f"{i:064x}",
             "scene_proposal_sha256": f"{i+100:064x}"} for i in range(100)]


def admission(entry):
    return {"contract": packer.ADMISSION_CONTRACT, "status": "independent_capture_audit_passed",
        "eligible_dataset": True, "require_rgb": True, "replay_expert": True, "source_split": "train",
        "source_episode_index": entry["source_episode_index"], "run": entry["run"],
        "scene_proposal_sha256": entry["scene_proposal_sha256"], "scene_readback_verified": True,
        "physical_gate": {"acceptance_gate_version": "handover-physics-v2", "eligible_physics": True,
            "verified_handover_success": True, "capture_validation_errors": [],
            "contact_evidence": {"strict_handover_success": True},
            "final_placement_geometric_validation": {"v2_passed": True, "require_3d_evidence": True}},
        "expert_reproduction": {"status": "all_generated_commands_reproduced_on_CPU"},
        "RGB_state_action_history_audit": {"status": "all_actual_RTX_events_pixels_actions_and_renderer_poses_passed",
            "capture_content_sha256": "b"*64, "eligible_current_rows": [30], "history_rows": {"30": [0,6,12,18,24,30]}},
        "quality_gate_redefined": False, "source_fixed_tokens_used_as_new_labels": False,
        "bindings": {"/synthetic": "a"*64}, "actual_full_outer_footprint_final50": footprint()}


def footprint():
    return {"contract":packer.FOOTPRINT_CONTRACT,"passed":True,"checked_states":50,"window_states":50,
        "required_minimum_outer_hull_margin_m":.005,"minimum_outer_hull_margin_m":.006,
        "all_states_all_vertices_inside_outer_hull":True,"AABB_or_source_bbox_used_as_vertices":False,
        "exact_inner_cavity_claimed":False,"original_handover_physics_v2_modified":False,"physics_steps_performed":0,
        "geometry_source":"actual_MJB_compiled_collision_vertices_at_each_saved_integration_state",
        "measurements":[{"physics_state_row":i+100,"sim_time_s":(i+100)/50,"minimum_outer_hull_margin_m":.006,
            "target_collision_vertex_count":8,"container_collision_vertex_count":8,"outer_hull_vertex_count":4,
            "all_target_collision_vertices_inside_outer_hull":True,"passed":True} for i in range(50)]}


class Random100PackTests(unittest.TestCase):
    def test_complete_exact100_min50_cap2_selection(self):
        before = deepcopy(entries()); counts = packer.validate_selection(before, requirements())
        self.assertEqual(len(counts), 50); self.assertEqual(set(counts.values()), {2})
        self.assertEqual(before, entries())

    def test_incomplete_and_overfull_packages_rejected(self):
        for value in (entries()[:99], entries()+[dict(entries()[0], candidate_id="extra")]):
            with self.assertRaisesRegex(ValueError, "exactly100"):
                packer.validate_selection(value, requirements())

    def test_all_VAL_and_truncated_source_rejected(self):
        for source in (*packer.HELDOUT, 38, -1, 100, True):
            value = entries(); value[0]["source_episode_index"] = source
            with self.subTest(source=source), self.assertRaises(ValueError):
                packer.validate_selection(value, requirements())

    def test_source_coverage_and_cap_are_both_hard_requirements(self):
        value = entries(); value[0]["source_episode_index"] = value[2]["source_episode_index"]
        with self.assertRaises(ValueError): packer.validate_selection(value, requirements())
        limited = requirements(); limited["minimum_distinct_accepted_original_sources"] = 24
        with self.assertRaises(ValueError): packer.validate_selection(entries(), limited)

    def test_duplicate_execution_or_admission_rejected(self):
        for field in ("candidate_id", "run", "admission_path", "admission_sha256", "scene_proposal_sha256"):
            value = entries(); value[0][field] = value[1][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                packer.validate_selection(value, requirements())

    def test_unsafe_candidate_and_unknown_hand_semantics_rejected(self):
        for field, value in (("candidate_id", "../../escape"), ("candidate_id", "/root"),
                             ("hand_condition", "unknown"), ("seed", 1.5), ("scene_proposal_sha256", "bad")):
            rows = entries(); rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                packer.validate_selection(rows, requirements())

    def test_zero_and_TEST_seed_ranges_rejected(self):
        for seed in (0,-1,2**30,2**31-1):
            value=entries();value[0]["seed"]=seed
            with self.subTest(seed=seed),self.assertRaises(ValueError):packer.validate_selection(value,requirements())

    def test_additional_actual_footprint_requires_every_final50_measurement(self):
        packer.validate_full_footprint(footprint())
        for key,value in (("contract","AABB"),("checked_states",49),("passed",False),
            ("AABB_or_source_bbox_used_as_vertices",True),("minimum_outer_hull_margin_m",.004),
            ("required_minimum_outer_hull_margin_m",0.),("exact_inner_cavity_claimed",True)):
            receipt=footprint();receipt[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):packer.validate_full_footprint(receipt)
        receipt=footprint();receipt["measurements"][17]["minimum_outer_hull_margin_m"]=-.001
        with self.assertRaises(ValueError):packer.validate_full_footprint(receipt)
        receipt=footprint();receipt["measurements"]=receipt["measurements"][:-1]
        with self.assertRaises(ValueError):packer.validate_full_footprint(receipt)
        receipt=footprint();receipt["measurements"][20]["physics_state_row"]+=1
        with self.assertRaises(ValueError):packer.validate_full_footprint(receipt)

    def diversity(self):
        return {"contract": packer.DIVERSITY_CONTRACT, "passed": True,
            "requirements_sha256": "a"*64, "registry_sha256": "b"*64,
            "admission_sha256_by_candidate": {e["candidate_id"]: e["admission_sha256"] for e in entries()},
            "checks": dict.fromkeys(packer.CHECKS, True), "no_seed_only_diversity_claim": True,
            "bindings": {"/synthetic": "c"*64}}

    def test_complete_diversity_must_bind_this_exact_registry(self):
        packer.validate_diversity(self.diversity(), "a"*64, "b"*64, entries())
        for field in ("passed", "requirements_sha256", "registry_sha256", "no_seed_only_diversity_claim"):
            value = self.diversity(); value[field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                packer.validate_diversity(value, "a"*64, "b"*64, entries())

    def test_no_missing_diversity_axis_can_pass(self):
        for key in packer.CHECKS:
            value = self.diversity(); del value["checks"][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                packer.validate_diversity(value, "a"*64, "b"*64, entries())
        value = self.diversity(); value["admission_sha256_by_candidate"]["fixture-0"] = "d"*64
        with self.assertRaises(ValueError): packer.validate_diversity(value, "a"*64, "b"*64, entries())

    def test_strict_new_online_RTX_admission_required(self):
        entry = entries()[0]; packer.validate_admission(admission(entry), entry)
        for field, value in (("contract", "cvpr-handover-quality100-capture-audit-v1"),
                ("eligible_dataset", False), ("require_rgb", False), ("replay_expert", False),
                ("source_split", "val"), ("source_episode_index", 94), ("scene_readback_verified", False),
                ("source_fixed_tokens_used_as_new_labels", True), ("quality_gate_redefined", True)):
            changed = admission(entry); changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): packer.validate_admission(changed, entry)

    def test_partial_physics_or_RGB_or_expert_proof_rejected(self):
        entry = entries()[0]
        for field, subfield in (("physical_gate", "eligible_physics"),
                ("expert_reproduction", "status"), ("RGB_state_action_history_audit", "history_rows"),
                ("RGB_state_action_history_audit", "capture_content_sha256")):
            changed = admission(entry); changed[field][subfield] = None
            with self.subTest(field=field, subfield=subfield), self.assertRaises(ValueError):
                packer.validate_admission(changed, entry)

    def test_source_and_registry_payload_changed_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bound"; path.write_bytes(b"test")
            bindings = {str(path): packer.legacy.sha(path)}; packer.verify_bindings(bindings)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError): packer.verify_bindings(bindings)

    def test_candidate_identity_uses_source_seed_and_actual_scene(self):
        first = packer.candidate_identity(1, 10, "a"*64)
        self.assertEqual(first, packer.candidate_identity(1, 10, "a"*64))
        for result in (packer.candidate_identity(2,10,"a"*64), packer.candidate_identity(1,11,"a"*64),
                       packer.candidate_identity(1,10,"b"*64)):
            self.assertNotEqual(first, result)

    def test_verifier_rejects_supplement_or_final_ready_without_touching_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"PACK_READY.json"
            path.write_text(json.dumps({"contract": "cvpr-handover-supplement-lerobot-pack-v1"}))
            original=path.read_bytes()
            with self.assertRaises(ValueError): verifier.inspect_pack(directory)
            self.assertEqual(original,path.read_bytes())

    def test_real_history_selector_truncates_incomplete_future_without_padding(self):
        n=90
        events=[{"nominal_elapsed_s": i/30, "physical_tick50": (i*50+29)//30, "action_valid": i>=30}
                for i in range(n)]
        rows,histories=packer.legacy.eligible_currents(events)
        self.assertEqual(rows[0],30); self.assertEqual(rows[-1],n-30)
        self.assertEqual(len(rows),31)
        self.assertTrue(all(len(histories[str(row)])==6 and histories[str(row)][-1]==row for row in rows))
        self.assertTrue(all(row+30<=n for row in rows))


if __name__ == "__main__":
    unittest.main()
