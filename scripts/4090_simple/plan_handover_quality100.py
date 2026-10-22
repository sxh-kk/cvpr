"""Plan bounded, lineage-balanced Handover recollection; collect nothing.

No candidate becomes an accepted demonstration in this module. Perturbations
are proposals for new physical executions and synchronized captures, not edits
to existing trajectories. The source release and all training runs are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

TASK = "simple/G1WholebodyHandoverTeleop-v0"
HELDOUT_VAL = (39, 65, 80, 92, 94)
EXPECTED_TRAIN_ANCESTORS = 24
TRAIN_SEED_MAX = 2**30 - 1
TEST_SEED_MIN = 2**30


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def seed_for(seed, ancestor, variant, domain):
    raw = int(digest([seed, ancestor, variant, domain])[:16], 16)
    return 1 + raw % TRAIN_SEED_MAX


def offset(seed, ancestor, variant, domain, radius):
    # Two half-open sides exclude exact zero, so proposals alter physical reset.
    raw = int(digest([seed, ancestor, variant, domain])[:16], 16)
    magnitude = radius * (1 + (raw >> 1) % 1_000_000) / 1_000_000
    return magnitude if raw & 1 else -magnitude


def eligible_ancestors(manifest, selection):
    if (manifest.get("status") != "complete" or manifest.get("release_ready") is not True
            or manifest.get("is_official") is not False
            or manifest.get("physics_acceptance_gate") != "handover-physics-v2"
            or selection.get("schema_version") != "handover-final50-release-selection-v1"
            or selection.get("source_dataset_key") != manifest.get("source_dataset_key")):
        raise ValueError("Require the accepted, identity-matched final50 Handover release and selection")
    source_key = manifest["source_dataset_key"]
    selected = {}
    for row in selection["episodes"]:
        key = row["source_scene_episode_index"]
        if type(key) is not int or key in selected:
            raise ValueError("Selection scene identities must be unique integers")
        selected[key] = row
    if len(selected) != 50 or len(manifest["episodes"]) != 50:
        raise ValueError("Require the existing50-episode selection")
    ancestors, exclusions, scene_ids, split_ancestors = [], [], set(), {"train": set(), "val": set()}
    for row in manifest["episodes"]:
        scene = row["source_episode_index"]
        if type(scene) is not int or scene in scene_ids or scene not in selected:
            raise ValueError("Release scenes must match the unique selection")
        scene_ids.add(scene)
        chosen, lineage = selected[scene], row["lineage"]
        if (chosen.get("split") != row.get("split")
                or chosen.get("source_dataset_key") != source_key
                or chosen.get("data_category") != lineage.get("data_category")
                or chosen.get("reviewed_video_sha256") != row.get("source_video_sha256")
                or chosen.get("numeric_status") != "passed"
                or chosen.get("physics_gate") != "handover-physics-v2"
                or chosen.get("visual_qa_passed") is not True):
            raise ValueError("Release/selection split, category or prior acceptance identity differs")
        split = row["split"]
        if split not in split_ancestors or lineage.get("lineage_complete") is not True:
            raise ValueError("Require complete TRAIN/VAL ancestry")
        family = {(a["source_dataset_key"], a["source_episode_index"])
                  for a in lineage["lineage_group_sources"]}
        if not family:
            raise ValueError("Empty ancestor set")
        split_ancestors[split].update(family)
        if split == "val":
            exclusions.append({"source_scene_episode_index": scene, "reason": "existing_VAL_never_collect_for_TRAIN"})
            continue
        if (lineage.get("data_category") != "original_or_repaired"
                or lineage.get("split_restriction") != "unrestricted"):
            exclusions.append({"source_scene_episode_index": scene, "reason": "generated_or_train_only_not_an_independent_motion_source"})
            continue
        if (scene in HELDOUT_VAL or family != {(source_key, scene)}
                or "motion_template_source_episode_index" in lineage):
            raise ValueError("Eligible original TRAIN source must have one unambiguous original ancestor")
        ancestors.append({"source_dataset_key": source_key, "source_episode_index": scene,
                          "source_original_category": "original_or_repaired",
                          "source_parquet_sha256": row["source_parquet_sha256"],
                          "source_video_sha256": row["source_video_sha256"]})
    if ({row["source_episode_index"] for row in manifest["episodes"] if row["split"] == "val"}
            != set(HELDOUT_VAL) or split_ancestors["train"] & split_ancestors["val"]):
        raise ValueError("Preserve the five VAL scenes and prohibit shared TRAIN/VAL ancestry")
    if len(ancestors) != EXPECTED_TRAIN_ANCESTORS:
        raise ValueError("Exactly24 accepted original TRAIN ancestors are required for coverage")
    return sorted(ancestors, key=lambda row: row["source_episode_index"]), exclusions


def build_plan(manifest, selection, *, manifest_sha256, selection_sha256,
               target_accepted=100, max_per_ancestor=5, seed=20261023):
    if manifest.get("selection_sha256") != selection_sha256:
        raise ValueError("Selection bytes differ from the accepted release binding")
    if (type(target_accepted) is not int or type(max_per_ancestor) is not int
            or type(seed) is not int or seed < 0 or not 1 <= max_per_ancestor <= 5):
        raise ValueError("Use integer seed/target and per-ancestor cap in1..5")
    ancestors, exclusions = eligible_ancestors(manifest, selection)
    if not len(ancestors) <= target_accepted <= len(ancestors) * max_per_ancestor:
        raise ValueError("Target must cover all24 ancestors and fit the fixed per-ancestor capacity")
    attempts_per_ancestor = math.ceil(target_accepted / len(ancestors))
    # Fixed order round-robin prevents the first template consuming the budget.
    ordered = sorted(ancestors, key=lambda row: digest([seed, row["source_episode_index"], "order"]))
    candidates = []
    for variant in range(attempts_per_ancestor):
        for ancestor in ordered:
            source_id = ancestor["source_episode_index"]
            objects = {role: {"xy_delta_m": [offset(seed, source_id, variant, f"{role}.{axis}", .01)
                                             for axis in ("x", "y")],
                              "world_z_yaw_delta_rad": offset(seed, source_id, variant, f"{role}.yaw", .05)}
                       for role in ("target", "container")}
            proposal = {"ancestor": ancestor, "variant": variant,
                        "physics_seed": seed_for(seed, source_id, variant, "physics"),
                        "material_seed": seed_for(seed, source_id, variant, "material"),
                        "generation_seed": seed_for(seed, source_id, variant, "generation"),
                        "initial_object_deltas": objects}
            identity = digest(proposal)
            candidates.append({"candidate_id": f"handover-source{source_id:03d}-variant{variant:02d}-{identity[:16]}",
                               "candidate_spec_sha256": identity, **proposal,
                               "split": "train", "status": "planned_not_collected", "accepted": False,
                               "collection_requirement": "new_physical_execution_and_real_synchronized_capture",
                               "initial_pose_reference": "independently verified source row0 world Z-up poses, not environment_config final poses",
                               "apply_object_delta": "once before control/history only; world-Z quaternion premultiplication",
                               "trajectory_files": [], "independent_acceptance_sha256": None})
    # Distinct seed space is frozen now, but it does not prove independent lineage.
    test_seeds = [TEST_SEED_MIN + int(digest([seed, "reserved_test", i])[:16], 16) % TEST_SEED_MIN
                  for i in range(20)]
    if (len({c["candidate_id"] for c in candidates}) != len(candidates)
            or len({c["candidate_spec_sha256"] for c in candidates}) != len(candidates)):
        raise ValueError("Duplicate candidate proposal")
    return {"contract": "cvpr-handover-quality100-candidate-plan-v1", "task": TASK,
            "status": "planned_not_collected", "accepted_count": 0, "collected_count": 0,
            "target_accepted_count": target_accepted, "candidate_count": len(candidates), "seed": seed,
            "source_manifest_sha256": manifest_sha256, "selection_sha256": selection_sha256,
            "original_ancestors": ancestors, "excluded_source_scenes": exclusions,
            "acceptance_limits": {"maximum_total_accepted": target_accepted,
                                  "maximum_accepted_per_ancestor": max_per_ancestor,
                                  "minimum_distinct_accepted_ancestors": len(ancestors),
                                  "all24_ancestors_required": True,
                                  "maximum_planned_attempts_per_ancestor": attempts_per_ancestor},
            "existing_validation_sources": list(HELDOUT_VAL),
            "frozen_test_reservation": {"status": "awaiting_independent_lineage_binding",
                "training_access_forbidden": True, "independent_test_certified": False,
                "existing_VAL_relabelled_as_TEST": False, "reset_seeds": test_seeds,
                "seed_manifest_sha256": digest(test_seeds),
                "required_before_final_ready": "Bind and freeze independent source/motion/scene ancestry; no TRAIN derivative, no existing model-selection VAL scene"},
            "required_physical_acceptance": {
                "controller": "SONICv1.1 fixed encoder/decoder/observation hashes plus complete numeric body/hand PD and effort profiles",
                "policy_action_hz": 30, "low_level_controller_hz": 50, "authoritative_physics_hz": 200,
                "acceptance_observation_hz": 50,
                "action_execution": "exact deployed30Hz sample-and-hold; no substitution with successful50Hz source replay",
                "observations": "new real RGB, measured states and actually executed actions with complete synchronized50Hz physical/contact observations; actual200Hz MuJoCo physics",
                "solver_force_measurement_phase": "read actual solved contact/support forces after the final real mj_step of each50Hz frame and before observation mj_forward; do not infer them from refreshed constraint forces",
                "no_file_copy_as_demonstration": True, "no_during_control_object_restore": True,
                "no_kinematic_object_attachment_or_external_force_success": True,
                "gate": "complete original handover-physics-v2 with source replay_container_geometry placement-necessary-geometry-v2; no additional200Hz acceptance threshold",
                "ordered_stages": ["right_hand_pickup", "two_hand_transfer", "left_hand_support", "container_placement", "both_hands_release"],
                "source_thresholds": {"fps": 50.0, "minimum_stage_frames": 5,
                    "minimum_stage_lift_m_inclusive": .05, "minimum_lift_m_inclusive": .1,
                    "minimum_pelvis_z_m_inclusive": .4, "final_window_frames": 50,
                    "minimum_final_container_contact_frames": 49,
                    "hands_released_on_every_final_frame": True,
                    "stability_window_s": 1.0, "stability_window_frames": 50,
                    "maximum_pairwise_displacement_m_exclusive": .005},
                "source_geometry_v2_thresholds": {"container_xy_com_inset_m": .005,
                    "maximum_container_tilt_deg": 30., "minimum_support_weight_fraction": .5,
                    "minimum_support_frames": 49, "window_frames": 50},
                "geometry_limit": "outer XY hull plus vertical overlap/upright pose/support are necessary placement evidence, not exact inner-cavity certification; full handover/release/stability and visual review still required",
                "evidence": "independent acceptance report SHA bound to real capture payload hashes; self-reported success or official contact flag is insufficient"},
            "progress_and_finalization": {"collector_implemented": False, "acceptance_aggregator_implemented": False,
                "dataset_complete": False, "final_ready": False,
                "final_ready_blockers": ["no_candidates_collected", "no_independent_acceptance_payloads", "TEST_lineage_not_bound"],
                "cannot_assert_complete_from_plan_or_success_claim": True},
            "limits": ["Small pose/material perturbations are proposed, not measured diversity or successful demonstrations.",
                       "24 source motion families remain24, even if100 or120 new executions succeed.",
                       "100 accepted training demonstrations do not establish100% learned-policy success.",
                       "Failed candidates must remain in the attempt denominator; a failed quota cannot be relabelled complete.",
                       "Lineage-balanced sampling and actual trajectory/scene near-duplicate checks remain necessary after capture."],
            "candidates": candidates}


def prepare(manifest_path, selection_path, output, **kwargs):
    manifest_path, selection_path = (Path(p).resolve(strict=True) for p in (manifest_path, selection_path))
    output = Path(output)
    repo = Path(__file__).resolve().parents[2]
    if (not output.is_absolute() or output.exists() or output.is_symlink()
            or output.resolve().is_relative_to(repo)
            or any(output.resolve().is_relative_to(p.parent) for p in (manifest_path, selection_path))):
        raise ValueError("Use a fresh absolute output outside the repository and source metadata directories")
    manifest_bytes, selection_bytes = manifest_path.read_bytes(), selection_path.read_bytes()
    plan = build_plan(json.loads(manifest_bytes), json.loads(selection_bytes),
                      manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                      selection_sha256=hashlib.sha256(selection_bytes).hexdigest(), **kwargs)
    plan["source_files"] = {"manifest": str(manifest_path), "selection": str(selection_path)}
    plan["generated_utc"] = datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True, exist_ok=False)
    text = json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    (output / "candidate_plan.json").write_text(text)
    (output / "PLAN_ONLY_NOT_COLLECTED").write_text(
        f"status=planned_not_collected\naccepted=0\ncollected=0\nplan_sha256={hashlib.sha256(text.encode()).hexdigest()}\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-accepted", type=int, default=100)
    parser.add_argument("--max-per-ancestor", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    plan = prepare(args.manifest, args.selection, args.output, target_accepted=args.target_accepted,
                   max_per_ancestor=args.max_per_ancestor, seed=args.seed)
    print(json.dumps({key: plan[key] for key in ("status", "candidate_count", "target_accepted_count", "accepted_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
