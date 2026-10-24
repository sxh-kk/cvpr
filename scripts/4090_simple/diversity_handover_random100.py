"""Measured scene/motion diversity audit for one independent strict100 registry.

Seeds and candidate names never count as motion diversity. FK comes from each
admitted integration stream, with one initial-pelvis SE3 removed, followed by
normalized-time resampling. This does not establish new human sessions or TEST.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CONTRACT = "cvpr-handover-random100-diversity-audit-v1"
REGISTRY_CONTRACT = "cvpr-handover-random100-admission-registry-v1"
CHECKS = ("source_coverage", "scene_coverage", "near_duplicate", "stage_coverage", "actual_command_adaptation")
HELDOUT = frozenset((39, 65, 80, 92, 94, 38))
THRESHOLDS = {"episodes": 100, "minimum_distinct_sources": 50, "maximum_per_source": 2,
    "unique_actual_scene_proposal_sha256": True,
    "minimum_XY_range_each_role_axis_m": .06, "minimum_XY_population_std_each_role_axis_m": .012,
    "minimum_yaw_span_each_role_rad": .25, "minimum_actual_table_MDL_identities": 50,
    "motion_resampled_points": 64, "near_duplicate_joint_RMS_rad": .015,
    "near_duplicate_FK_RMS_m": .008, "minimum_peak_actual_expert_arm_correction_rad": .001}


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""): value.update(block)
    return value.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def jsonl(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_registry_shape(registry):
    require(registry.get("contract") == REGISTRY_CONTRACT, "Unknown independent registry contract")
    entries = registry.get("admissions", [])
    require(len(entries) == 100, "Formal registry must contain exactly100 strict admissions")
    ids = [entry["candidate_id"] for entry in entries]
    sources = [entry["source_episode_index"] for entry in entries]
    require(all(type(source) is int and 0 <= source < 100 and source not in HELDOUT for source in sources),
            "Protected VAL or incomplete source entered TRAIN")
    counts = Counter(sources)
    require(len(set(ids)) == 100 and len(counts) >= 50 and max(counts.values()) <= 2,
            "Require unique candidates, at least50 original sources and cap2")
    require(len({str(Path(entry["run"]).resolve()) for entry in entries}) == 100,
            "Reusing the same execution under different seeds/IDs is forbidden")
    return counts


def validate_requirements(requirements):
    from audit_handover_random100 import ADDITIONAL_DATASET_QUALITY
    require(requirements.get("contract") == "cvpr-handover-random100-rebuild-requirements-v1"
            and requirements.get("user_requested_count") == 100
            and requirements.get("minimum_distinct_accepted_original_sources") == 50
            and requirements.get("maximum_accepted_per_source") == 2
            and set(requirements.get("protected_VAL_sources", [])) == HELDOUT - {38}
            and requirements.get("truncated_source_excluded") == 38, "Unknown or weakened independent100 requirements")
    expected = {"XY_max_m": .05, "yaw_max_rad": .25, "minimum_accepted_XY_range_each_role_axis_m": .06,
        "minimum_accepted_XY_std_each_role_axis_m": .012, "minimum_accepted_yaw_range_each_role_rad": .25,
        "minimum_distinct_actual_TRAIN_table_MDL": 50}
    require(all(requirements.get("formal_scene_sampling", {}).get(key) == value for key, value in expected.items()),
            "Unknown or weakened formal measured scene-diversity requirements")
    require(requirements.get("formal_scene_sampling", {}).get("unique_actual_scene_proposal_sha256") is True,
            "Every formal admission requires a different actual scene proposal file SHA")
    require(requirements.get("admission", {}).get("unchanged_physical_gate") == "handover-physics-v2",
            "Original physics acceptance gate cannot be replaced")
    require(requirements.get("additional_dataset_quality") == ADDITIONAL_DATASET_QUALITY,
            "Independent actual full outer-footprint final50 dataset quality requirement missing or weakened")


def _yaw_delta(actual, source):
    a, b = np.asarray(actual, float).copy(), np.asarray(source, float).copy()
    require(a.shape == b.shape == (4,) and np.isfinite(a).all() and np.isfinite(b).all(), "Invalid actual quaternion")
    require(np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0, "Actual quaternion norm must be positive")
    a /= np.linalg.norm(a); b /= np.linalg.norm(b)
    w, x, y, z = a; v, i, j, k = b * np.array([1., -1., -1., -1.])
    q = np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j, w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2]**2 + q[3]**2))


def scene_coverage(observations):
    """Use read-back poses/materials, never proposal seed cardinality."""
    metrics = {}; passed = True
    for role in ("target", "container"):
        xy = np.asarray([np.asarray(row["actual"][role]["pose7"][:2]) - row["source"][role]["source_pose7"][:2]
                         for row in observations], float)
        yaw = np.asarray([_yaw_delta(row["actual"][role]["pose7"][3:], row["source"][role]["source_pose7"][3:])
                          for row in observations])
        require(xy.shape == (len(observations), 2) and np.isfinite(xy).all() and np.isfinite(yaw).all(),
                "Actual scene coordinates are incomplete/nonfinite")
        span, std = np.ptp(xy, axis=0), np.std(xy, axis=0, ddof=0)
        yaw_span = float(np.ptp(yaw))
        metrics[role] = {"actual_XY_delta_range_m": span.tolist(), "actual_XY_delta_population_std_m": std.tolist(),
                         "actual_world_yaw_delta_span_rad": yaw_span}
        passed &= bool(np.all(span >= .06) and np.all(std >= .012) and yaw_span >= .25)
    identities = {(row["actual_MDL_path"], row["actual_MDL_name"]) for row in observations}
    metrics.update(actual_table_MDL_identity_count=len(identities),
        actual_table_MDL_file_count=len({path for path, name in identities}),
        actual_table_MDL_identity_definition="canonical actual bound MDL resource path plus actual MDL subidentifier",
        actual_table_MDL_identities=[list(identity) for identity in sorted(identities)])
    return bool(passed and len(identities) >= 50), metrics


def normalize_measured_motion(joints, world_fk, pelvis_rotation0, *, samples=64):
    """Remove one global SE3 and resample actual full controlled motion in time."""
    joints, points = np.asarray(joints, float), np.asarray(world_fk, float)
    rotation = np.asarray(pelvis_rotation0, float)
    require(joints.ndim == 2 and joints.shape[1] == 43 and points.shape == (len(joints), 3, 3)
            and len(joints) >= 50 and rotation.shape == (3, 3)
            and np.isfinite(joints).all() and np.isfinite(points).all()
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
            and math.isclose(np.linalg.det(rotation), 1., abs_tol=1e-8), "Invalid measured joint/FK motion")
    local = (points - points[0, 0]) @ rotation
    time = np.linspace(0., 1., len(joints)); target = np.linspace(0., 1., samples)
    def interpolate(value):
        flat = value.reshape(len(value), -1)
        return np.stack([np.interp(target, time, flat[:, column]) for column in range(flat.shape[1])], axis=1)
    return {"joints": interpolate(joints), "fk": interpolate(local), "native_controlled_rows": len(joints)}


def measured_descriptor(run):
    import mujoco
    from prepare_carrybox_v11 import TRAIN_STATE_NAMES
    model = mujoco.MjModel.from_binary_path(str(run / "physics_model.mjb")); data = mujoco.MjData(model)
    with np.load(run / "trajectory.npz", allow_pickle=False) as values:
        states = values["physics_state_integration"]; spec = int(values["physics_state_spec"])
        require(spec == int(mujoco.mjtState.mjSTATE_INTEGRATION), "Measured motion needs actual integration states")
        controlled = states[50:]
        require(len(controlled) >= 50, "No complete measured expert trajectory")
        qadr = [int(model.joint(name).qposadr[0]) for name in TRAIN_STATE_NAMES]
        bodies = [model.body(name).id for name in ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link")]
        joints, points, rotation0 = [], [], None
        for state in controlled:
            mujoco.mj_setState(model, data, state, spec); mujoco.mj_kinematics(model, data)
            joints.append(data.qpos[qadr].copy()); points.append(data.xpos[bodies].copy())
            if rotation0 is None: rotation0 = data.xmat[bodies[0]].reshape(3, 3).copy()
    return normalize_measured_motion(joints, points, rotation0)


def near_duplicate_metrics(descriptors, candidate_ids):
    require(len(descriptors) == len(candidate_ids) >= 2, "Need multiple measured motions")
    nearest = [None] * len(descriptors); duplicates = []
    for first in range(len(descriptors)):
        for second in range(first + 1, len(descriptors)):
            q = float(np.sqrt(np.mean((descriptors[first]["joints"] - descriptors[second]["joints"])**2)))
            fk = float(np.sqrt(np.mean((descriptors[first]["fk"] - descriptors[second]["fk"])**2)))
            distance = max(q / THRESHOLDS["near_duplicate_joint_RMS_rad"], fk / THRESHOLDS["near_duplicate_FK_RMS_m"])
            require(math.isfinite(distance), "Nonfinite measured-motion comparison")
            pair = {"first": candidate_ids[first], "second": candidate_ids[second],
                    "joint_RMS_rad": q, "FK_RMS_m": fk, "normalized_distance": distance}
            for index in (first, second):
                if nearest[index] is None or distance < nearest[index]["normalized_distance"]: nearest[index] = pair
            if distance <= 1: duplicates.append(pair)
    return not duplicates, {"near_duplicate_pairs": duplicates,
        "nearest_by_candidate": dict(zip(candidate_ids, nearest)),
        "minimum_normalized_pair_distance": min(row["normalized_distance"] for row in nearest),
        "distance_definition": "max(joint_RMS/0.015rad, FK_RMS/0.008m); <=1 is near duplicate",
        "time_normalization": "64 uniform normalized-time samples over actual post-warmup controlled trajectory",
        "rigid_transform_normalization": "one initial measured pelvis SE3 removed from pelvis/left/right-wrist FK",
        "seed_or_source_ID_used_as_motion_distance": False}


def _stages_and_adaptation(admission, commands):
    physical = admission["physical_gate"]; contact = physical["contact_evidence"]
    stages = contact.get("ordered_lifted_stages", {})
    ordered = all(isinstance(stages.get(name), dict) for name in ("right_only", "both_hands", "left_only"))
    previous = 0
    if ordered:
        for name in ("right_only", "both_hands", "left_only"):
            item = stages[name]
            ordered &= (type(item.get("start_step")) is int and type(item.get("end_step")) is int
                        and item["start_step"] > previous and item["end_step"] - item["start_step"] + 1 >= 5
                        and item.get("frame_count") == item["end_step"] - item["start_step"] + 1)
            previous = item["end_step"]
    geometry = physical.get("final_placement_geometric_validation", {})
    stage_pass = (ordered and contact.get("strict_handover_success") is True and geometry.get("require_3d_evidence") is True
                  and geometry.get("v2_passed") is True)
    peaks = [float(np.max(np.abs(np.asarray(row["diagnostics"].get("current_arm_joint_delta", []), float))))
             for row in commands if len(row["diagnostics"].get("current_arm_joint_delta", [])) == 14]
    require(all(math.isfinite(peak) for peak in peaks), "Nonfinite actual generated arm correction")
    new_control = bool(peaks) and max(peaks) >= THRESHOLDS["minimum_peak_actual_expert_arm_correction_rad"]
    for row in commands:
        diagnostic = row["diagnostics"]
        new_control &= all(diagnostic.get(key) is False for key in
            ("simulator_state_mutated", "object_state_mutated", "external_force_applied", "acceptance_gate_modified"))
        new_control &= isinstance(diagnostic.get("encoder_input_sha256"), str) and len(diagnostic["encoder_input_sha256"]) == 64
    return bool(stage_pass), bool(new_control), {"ordered_lifted_stages": stages,
        "peak_actual_expert_arm_correction_rad": max(peaks, default=0.), "actual_generated_commands": len(commands)}


def replacement_candidates(report):
    """Return existing registry IDs suggested for replacement, never new IDs.

    Suggestions do not qualify another execution or preserve other coverage by
    construction. A replacement requires a new strict admission and a complete
    re-audit of the resulting exact100 registry at unchanged thresholds.
    """
    if report.get("passed") is True:
        return []
    allowed = set(report.get("registry_candidate_ids", []))
    return list(dict.fromkeys(candidate for candidate in report.get("replacement_candidate_ids", [])
                              if isinstance(candidate, str) and candidate in allowed))


def _replacement_advice(candidate_ids, observations, scene_metric, motion_metric, stage_pass, adapted, scene_sha256=None):
    """Conservative deterministic suggestions; deficient axis spans need sampling."""
    allowed = set(candidate_ids); reasons = {}
    def add(candidate, reason):
        require(candidate in allowed, "Replacement suggestion escaped the audited registry")
        reasons.setdefault(candidate, [])
        if reason not in reasons[candidate]: reasons[candidate].append(reason)
    for pair in motion_metric["near_duplicate_pairs"]:
        add(pair["second"], "near_duplicate_of:" + pair["first"])
    scene_owner = {}
    for candidate, digest in zip(candidate_ids, scene_sha256 or []):
        if digest in scene_owner: add(candidate, "duplicate_actual_scene_proposal_of:" + scene_owner[digest])
        else: scene_owner[digest] = candidate
    for candidate, stage, adaptation in zip(candidate_ids, stage_pass, adapted):
        if not stage: add(candidate, "stage_coverage_failed")
        if not adaptation: add(candidate, "actual_command_adaptation_failed")
    deficit = max(0, THRESHOLDS["minimum_actual_table_MDL_identities"] - scene_metric["actual_table_MDL_identity_count"])
    seen = set(); redundant = []
    for candidate, row in zip(candidate_ids, observations):
        identity = (row["actual_MDL_path"], row["actual_MDL_name"])
        if identity in seen: redundant.append(candidate)
        else: seen.add(identity)
    # Prioritize already suggested redundant rows to avoid unnecessary removals.
    redundant.sort(key=lambda candidate: candidate not in reasons)
    for candidate in redundant[:deficit]: add(candidate, "duplicate_actual_table_MDL_when_material_coverage_deficient")
    gaps = []
    for role in ("target", "container"):
        metric = scene_metric[role]
        for axis, span, std in zip("XY", metric["actual_XY_delta_range_m"], metric["actual_XY_delta_population_std_m"]):
            if span < .06 or std < .012:
                gaps.append({"role": role, "axis": axis, "kind": "actual_XY_coverage",
                             "range_m": span, "population_std_m": std})
        if metric["actual_world_yaw_delta_span_rad"] < .25:
            gaps.append({"role": role, "kind": "actual_yaw_coverage",
                         "span_rad": metric["actual_world_yaw_delta_span_rad"]})
    return {"replacement_candidate_ids": list(reasons), "replacement_reasons_by_candidate": reasons,
        "targeted_scene_sampling_gaps": gaps,
        "replacement_policy": "Only existing registry IDs are suggested; XY/yaw coverage alone cannot identify a reliable deletion. "
            "Use additional strictly admitted executions, preserve every prior run, and re-audit the complete registry without lowering thresholds.",
        "replacement_guarantees_pass": False}


def audit(registry_path):
    """Read an immutable completed registry; never collect or promote data."""
    registry_path = Path(registry_path).resolve(strict=True)
    report = {"contract": CONTRACT, "passed": False, "checks": {key: False for key in CHECKS},
        "created_utc": datetime.now(timezone.utc).isoformat(), "thresholds": dict(THRESHOLDS),
        "no_seed_only_diversity_claim": True, "physics_steps_performed": 0, "GPU_used": False,
        "independent_human_sessions_claimed": False, "independent_blind_TEST_certified": False,
        "registry_sha256": sha(registry_path), "requirements_sha256": None,
        "admission_sha256_by_candidate": {}, "metrics": {}, "registry_candidate_ids": [],
        "replacement_candidate_ids": [], "replacement_reasons_by_candidate": {}}
    report["auditor_sha256"] = sha(__file__)
    try:
        from pack_handover_random100 import validate_admission
        from audit_handover_random100 import CONTRACT as ADMISSION_CONTRACT
        registry = read(registry_path); counts = validate_registry_shape(registry)
        report["registry_candidate_ids"] = [entry["candidate_id"] for entry in registry["admissions"]]
        requirements_path = Path(registry["requirements_path"]).resolve(strict=True); requirements = read(requirements_path)
        require(sha(requirements_path) == registry["requirements_sha256"], "Requirements SHA changed")
        validate_requirements(requirements)
        report["requirements_sha256"] = sha(requirements_path)
        bindings = {str(registry_path): report["registry_sha256"], str(requirements_path): report["requirements_sha256"]}
        def bind(path, expected=None):
            path = str(Path(path).resolve(strict=True)); expected = expected or sha(path)
            require(path not in bindings or bindings[path] == expected, "Conflicting bound payload identities")
            if path not in bindings: require(sha(path) == expected, "Bound accepted evidence changed")
            bindings[path] = expected
        observations, descriptors, ids, content_ids, stages, adaptations, scene_ids = [], [], [], set(), [], [], []
        for entry in registry["admissions"]:
            run = Path(entry["run"]).resolve(strict=True); path = Path(entry["admission_path"]).resolve(strict=True)
            bind(path, entry["admission_sha256"]); admission = read(path); validate_admission(admission, entry)
            require(admission["contract"] == ADMISSION_CONTRACT, "Wrong independent admission authority")
            for group in ("bindings", "source_sha256"):
                for source_path, expected in admission[group].items(): bind(source_path, expected)
            report["admission_sha256_by_candidate"][entry["candidate_id"]] = entry["admission_sha256"]
            args = read(run / "execution_args.json"); proposal = read(run / "scene_proposal.json")
            scene = read(run / "scene_readback.json")["actual_readback"]
            require(args["source_episode"] == entry["source_episode_index"] and args["seed"] == entry["seed"]
                    and args["hand_condition"] == entry["hand_condition"]
                    and args["xy_max_m"] == requirements["formal_scene_sampling"]["XY_max_m"]
                    and args["yaw_max_rad"] == requirements["formal_scene_sampling"]["yaw_max_rad"],
                    "Registry source/seed/hand condition or formal scene bounds differ")
            bind(run / "scene_proposal.json", entry["scene_proposal_sha256"])
            scene_ids.append(entry["scene_proposal_sha256"])
            material = scene["material"]; mdl = str(Path(material["table_mdl_path"]).resolve(strict=True))
            bind(mdl, proposal["material"]["table_mdl_sha256"])
            observations.append({"actual": scene["objects"], "source": proposal["objects"],
                "actual_MDL_path": mdl, "actual_MDL_name": material["table_mdl_name"]})
            content = admission["RGB_state_action_history_audit"]["capture_content_sha256"]
            require(content not in content_ids, "Duplicate actual capture under a different registry identity")
            content_ids.add(content)
            for name in ("physics_model.mjb", "trajectory.npz", "expert_commands.jsonl"):
                require(str((run / name).resolve()) in admission["bindings"], "Required measured motion is not admission-bound")
            descriptors.append(measured_descriptor(run)); ids.append(entry["candidate_id"])
            stage, adapted, detail = _stages_and_adaptation(admission, jsonl(run / "expert_commands.jsonl"))
            stages.append(stage); adaptations.append(adapted)
            report["metrics"].setdefault("stages_and_adaptation_by_candidate", {})[entry["candidate_id"]] = detail
        scene_pass, scene_metric = scene_coverage(observations)
        scene_counts = Counter(scene_ids)
        scene_metric.update(unique_actual_scene_proposal_sha256_count=len(scene_counts),
            duplicate_actual_scene_proposal_sha256={digest: [candidate for candidate, scene in zip(ids, scene_ids) if scene == digest]
                                                    for digest, count in scene_counts.items() if count > 1},
            scene_identity_definition="SHA256 of each actual pretty-JSON scene_proposal file; AB hand conditions do not create new scenes")
        scene_pass &= len(scene_counts) == 100
        duplicate_pass, motion_metric = near_duplicate_metrics(descriptors, ids)
        report["checks"] = dict(source_coverage=True, scene_coverage=scene_pass, near_duplicate=duplicate_pass,
            stage_coverage=all(stages), actual_command_adaptation=all(adaptations))
        report["metrics"].update(episodes=100, distinct_sources=len(counts), accepted_per_source=dict(counts),
            scene_coverage=scene_metric, measured_motion=motion_metric,
            stage_failed_candidate_ids=[candidate for candidate, passed in zip(ids, stages) if not passed],
            adaptation_failed_candidate_ids=[candidate for candidate, passed in zip(ids, adaptations) if not passed])
        report.update(_replacement_advice(ids, observations, scene_metric, motion_metric, stages, adaptations, scene_ids))
        require(all(sha(path) == expected for path, expected in bindings.items()), "Accepted source snapshot changed during diversity audit")
        report["bindings"] = bindings; report["passed"] = all(report["checks"].values())
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return report
