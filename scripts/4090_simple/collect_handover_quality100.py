"""Serial, resumable candidate execution and independent TRAIN capture auditing.

This module never changes training or talks to a remote host. Camera-free
screening never counts as data. Only a separately audited RTX recapture may
enter the accepted registry; packaging and independent TEST remain separate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import shutil

from plan_handover_quality100 import (digest, seed_for, offset, HELDOUT_VAL,
                                      EXPECTED_TRAIN_ANCESTORS)

REPO = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve().parent / "replay_handover_teacher.py"
FRAME_CALIBRATION = REPO / "VLMExpert/third_party/SIMPLE/src/simple/sensors/head_camera_mount.py"
SPEC_KEYS = ("ancestor", "variant", "physics_seed", "material_seed", "generation_seed", "initial_object_deltas")
CORE_FILES = tuple(Path(__file__).resolve().parent / name for name in (
    "collect_handover_quality100.py", "plan_handover_quality100.py", "replay_handover_teacher.py",
    "handover_eval_runtime.py", "prepare_handover_eval.py", "handover_contact_quality.py", "handover_geometry_quality.py",
    "prepare_handover_teacher_reference.py"))
AUDIT_CONTRACT = "cvpr-handover-independent-rtx-capture-audit-v3"
QUEUE_CONTRACT = "cvpr-handover-quality100-serial-queue-v3"
AUTHORITY_REVISION = "integration-source-row0-command-trace-renderer-pose-startup-reference-v4"
WARMUP_SCHEDULE_CONTRACT = "cvpr-handover-train-warmup-schedule-v1"
WARMUP_SCOPE = "TRAIN_teacher_collection_only"
WARMUP_KINDS = {"measured_pose_pd": "measured_pose_pd", "live_pose_sonic_v1_1": "live_initial_pose_sonic_v1_1"}
PADDED_WARMUP = "live_pose_sonic_v1_1_repeat_first_snapshot"
WARMUP_KINDS[PADDED_WARMUP] = "live_initial_pose_repeat_first_snapshot_sonic_v1_1"
LIVE_WARMUP_SOURCE = REPO / "VLMExpert/third_party/SIMPLE/src/simple/baselines/live_pose_sonic_warmup.py"
RENDERER_POSE_CONTRACT = "cvpr-handover-renderer-pose-evidence-v1"
RENDER_POSITION_TOLERANCE_M = .001
RENDER_ROTATION_TOLERANCE_RAD = math.radians(.1)


class CaptureRejected(ValueError):
    """Retain independently measured rejection evidence with the raw attempt."""
    def __init__(self, message, receipt):
        self.receipt = receipt
        super().__init__(message + ": " + json.dumps(receipt["renderer_pose_audit"], sort_keys=True))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def local_file(root, relative):
    root, relative = Path(root).resolve(strict=True), Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Capture paths must be relative and remain in the capture directory")
    path = root / relative
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing/nonportable capture file: {relative}")
    return path


def finite_vector(value, size, label):
    import numpy as np
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"Expected finite {label}[{size}]")
    return vector


def finite_pose(value, label):
    pose = finite_vector(value, 7, label)
    if not math.isclose(float(sum(pose[3:]**2)), 1., rel_tol=0, abs_tol=2e-4):
        raise ValueError(f"Nonunit physical quaternion: {label}")
    return pose


def validate_candidate(candidate, ancestors):
    ancestor = candidate["ancestor"]
    source_id = ancestor["source_episode_index"]
    if (source_id in HELDOUT_VAL or ancestors.get(source_id) != ancestor
            or candidate.get("split") != "train"
            or candidate.get("accepted") is not False
            or type(candidate.get("variant")) is not int or candidate["variant"] < 0):
        raise ValueError("Candidate must retain an exact eligible original TRAIN ancestor")
    spec = {key: candidate[key] for key in SPEC_KEYS}
    if candidate.get("candidate_spec_sha256") != digest(spec):
        raise ValueError("Candidate proposal differs from its preregistered SHA")
    expected_id = f"handover-source{source_id:03d}-variant{candidate['variant']:02d}-{digest(spec)[:16]}"
    if candidate.get("candidate_id") != expected_id:
        raise ValueError("Unsafe candidate identifier")
    if set(candidate["initial_object_deltas"]) != {"target", "container"}:
        raise ValueError("Both initial object perturbations are required")
    for value in candidate["initial_object_deltas"].values():
        xy = finite_vector(value["xy_delta_m"], 2, "object XY delta")
        yaw = value["world_z_yaw_delta_rad"]
        if any(not 0 < abs(x) <= .01 for x in xy) or not isinstance(yaw, (int, float)) or not 0 < abs(yaw) <= .05:
            raise ValueError("Only preregistered nonzero small XY/yaw perturbations are supported")


def load_plan(path):
    path = Path(path).resolve(strict=True)
    plan = read(path)
    if (plan.get("contract") != "cvpr-handover-quality100-candidate-plan-v1"
            or plan.get("status") != "planned_not_collected"
            or plan.get("accepted_count") != 0 or plan.get("collected_count") != 0):
        raise ValueError("Input must be an uncollected preregistered candidate plan")
    ancestors = {r["source_episode_index"]: r for r in plan["original_ancestors"]}
    limits = plan["acceptance_limits"]
    if (len(ancestors) != EXPECTED_TRAIN_ANCESTORS
            or limits["minimum_distinct_accepted_ancestors"] != EXPECTED_TRAIN_ANCESTORS
            or limits["maximum_accepted_per_ancestor"] > 5
            or plan["target_accepted_count"] != 100
            or set(ancestors) & set(HELDOUT_VAL)):
        raise ValueError("Require the100-target/24-original-ancestor/cap5 protocol")
    ids, specs = set(), set()
    for candidate in plan["candidates"]:
        validate_candidate(candidate, ancestors)
        if candidate["candidate_id"] in ids or candidate["candidate_spec_sha256"] in specs:
            raise ValueError("Duplicate candidate identity")
        ids.add(candidate["candidate_id"])
        specs.add(candidate["candidate_spec_sha256"])
    if {c["ancestor"]["source_episode_index"] for c in plan["candidates"]} != set(ancestors):
        raise ValueError("Candidate list does not cover every original TRAIN ancestor")
    return plan, sha256(path)


def load_warmup_schedule(plan, plan_sha256, path=None):
    """Preregister TRAIN teacher controls; never a VAL/TEST policy oracle."""
    expected_sources = {str(row["source_episode_index"]) for row in plan["original_ancestors"]}
    if path is None:
        document = {"contract": WARMUP_SCHEDULE_CONTRACT, "scope": WARMUP_SCOPE, "plan_sha256": plan_sha256,
                    "controls": {source: "measured_pose_pd" for source in sorted(expected_sources, key=int)}}
        source_path, identity = None, None
    else:
        source = Path(path).resolve(strict=True)
        document = read(source)
        source_path, identity = str(source), sha256(source)
    if (document.get("contract") != WARMUP_SCHEDULE_CONTRACT or document.get("scope") != WARMUP_SCOPE
            or document.get("plan_sha256") != plan_sha256
            or set(document.get("controls", {})) != expected_sources
            or any(value not in WARMUP_KINDS for value in document["controls"].values())):
        raise ValueError("TRAIN warmup schedule must bind the base plan and exactly its24 original TRAIN sources")
    references = document.get("teacher_references", {})
    if not isinstance(references, dict) or not set(references) <= expected_sources:
        raise ValueError("Modified teacher references must retain explicit original TRAIN ancestry")
    for source, reference in references.items():
        if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
                or not Path(reference["path"]).is_absolute() or sha256(reference["path"]) != reference["sha256"]):
            raise ValueError("Modified teacher reference path/SHA differs from preregistration")
        artifact = read(reference["path"])
        if (artifact.get("contract") != "cvpr-handover-modified-teacher-reference-v1"
                or artifact.get("source_episode_index") != int(source)
                or artifact.get("scope") != "TRAIN_teacher_only"
                or artifact.get("evaluation_oracle_allowed") is not False):
            raise ValueError("Modified teacher references are forbidden as a validation/evaluation oracle")
    return {**document, "path": source_path, "sha256": identity,
            "evaluation_oracle_allowed": False, "warmup_supervision": "none"}


def bind_result(run, candidate, plan_sha256, *, expected_warmup="measured_pose_pd", expected_reference=None):
    run = Path(run).resolve(strict=True)
    result = read(run / "result.json")
    if (result.get("status") != "teacher_diagnostic_completed"
            or result.get("candidate_id") != candidate["candidate_id"]
            or result.get("candidate_spec_sha256") != candidate["candidate_spec_sha256"]
            or result.get("plan_sha256") != plan_sha256
            or result.get("source_episode_index") != candidate["ancestor"]["source_episode_index"]
            or result.get("command_rate_hz") != 30 or result.get("physics_hz") != 200
            or result.get("decoder_hz") != 50 or result.get("source_split") != "train"
            or result.get("finite_physics") is not True or result.get("fell") is not False
            or result.get("accepted_new_dataset") is not False):
        raise ValueError("Completed30Hz TRAIN candidate/physics identity is missing or differs")
    controller = result["controller_identity"]
    if (controller.get("pd_profile") != "fixed_cpp_baseline"
            or controller.get("hand_torque_profile") != "recording_0p7_nm"
            or controller.get("body_token_version") != "body_token_v1_1"):
        raise ValueError("Capture must use the predeclared SONICv1.1/cap0.7 controller recipe")
    reference = result.get("teacher_reference")
    if expected_reference is None:
        if reference is not None:
            raise ValueError("Teacher command changes require an explicitly preregistered reference")
    elif (not isinstance(reference, dict)
            or {key: reference.get(key) for key in ("path", "sha256")} != expected_reference
            or sha256(expected_reference["path"]) != expected_reference["sha256"]):
        raise ValueError("Actual modified teacher reference differs from the frozen TRAIN schedule")
    warmup = result["warmup_control"]
    if (expected_warmup not in WARMUP_KINDS or warmup.get("kind") != WARMUP_KINDS[expected_warmup]
            or result.get("warmup_ticks") != 50
            or (expected_warmup in ("measured_pose_pd", PADDED_WARMUP) and warmup.get("seconds") != 1.)):
        raise ValueError("Actual one-second TRAIN warmup differs from the preregistered control schedule")
    if expected_warmup in ("live_pose_sonic_v1_1", PADDED_WARMUP):
        code = result["capture_code_sha256"]
        if (code.get(str(LIVE_WARMUP_SOURCE)) != sha256(LIVE_WARMUP_SOURCE)
                or warmup.get("helper_source_sha256") != code[str(LIVE_WARMUP_SOURCE)]
                or warmup.get("hand_policy") != "measured" or warmup.get("state_forcing") is not False
                or warmup.get("history_injection") is not (expected_warmup == PADDED_WARMUP) or warmup.get("initialization_count") != 1
                or warmup.get("encoder_mode") != 0):
            raise ValueError("Live TRAIN warmup must retain the bound no-forcing real-history source recipe")
    if result.get("initial_object_deltas") != candidate["initial_object_deltas"]:
        raise ValueError("Physical reset perturbation differs from the preregistered candidate")
    receipt = result["perturbation_receipt"]
    if receipt.get("apply_count") != 1 or receipt.get("phase") != "before_warmup":
        raise ValueError("Object perturbation must happen once before warmup")
    if result.get("object_restores_during_control") is not False:
        raise ValueError("Object restoration during control is prohibited")
    if result.get("code_unchanged_during_run") is not True:
        raise ValueError("Capture code was not held fixed throughout the physical run")
    source_hashes = result["source_sha256"]
    if not source_hashes:
        raise ValueError("Source identity bindings are absent")
    for path, expected in source_hashes.items():
        path = Path(path)
        if not path.is_absolute() or not path.is_file() or sha256(path) != expected:
            raise ValueError("Local source file identity changed")
    code = result["capture_code_sha256"]
    required = {"replay_handover_teacher.py", "handover_eval_runtime.py", "handover_contact_quality.py", "handover_geometry_quality.py"}
    if not required <= {Path(p).name for p in code}:
        raise ValueError("Capture runtime and original acceptance helpers must be identity-bound")
    for path, expected in code.items():
        path = Path(path)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("Capture code changed before independent audit")
    snapshots = result["code_snapshots"]
    if {entry["original_path"] for entry in snapshots} != set(code):
        raise ValueError("Capture executable snapshots do not cover every bound runtime source")
    for entry in snapshots:
        if sha256(local_file(run, entry["path"])) != entry["sha256"] or entry["sha256"] != code[entry["original_path"]]:
            raise ValueError("Historical capture source snapshot identity differs")
    return result


def audit_applied_forces(run, result):
    """Inspect actual serialized integration states with CPU MuJoCo, never step."""
    import mujoco
    import numpy as np
    run = Path(run)
    model_path, trajectory_path = run / "physics_model.mjb", run / "trajectory.npz"
    if (sha256(model_path) != result["physics_model"]["sha256"]
            or sha256(trajectory_path) != result["trajectory"]["sha256"]):
        raise ValueError("Physical model/trajectory identity changed")
    model = mujoco.MjModel.from_binary_path(str(model_path))
    if model.neq != 0 or not np.isclose(model.opt.timestep, .005, rtol=0, atol=1e-12):
        raise ValueError("Unregistered equality attachment or non200Hz model")
    data = mujoco.MjData(model)
    with np.load(trajectory_path, allow_pickle=False) as values:
        spec = int(values["physics_state_spec"])
        states, times = values["physics_state_integration"], values["sim_time_s"]
        expected_rows = 51 + result["source_commands"]
        if (spec != int(mujoco.mjtState.mjSTATE_INTEGRATION)
                or states.shape != (expected_rows, mujoco.mj_stateSize(model, spec))
                or times.shape != (expected_rows,) or not np.isfinite(states).all()
                or not np.allclose(np.diff(times), .02, rtol=0, atol=1e-6)):
            raise ValueError("Complete actual50Hz integration state/time history is required")
        for state in states:
            mujoco.mj_setState(model, data, state, spec)
            if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
                raise ValueError("External applied forces are forbidden in every serialized physical state")
    guard = result["force_guard"]
    log_path = local_file(run, guard["path"])
    if sha256(log_path) != guard["sha256"]:
        raise ValueError("Every200Hz applied-force guard log must retain its identity")
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    if (len(rows) != 4 * (50 + result["source_commands"]) or guard.get("steps") != len(rows)
            or guard.get("all_applied_forces_zero") is not True
            or guard.get("equality_count") != 0 or guard.get("target_robot_attachment") is not False):
        raise ValueError("Complete every200Hz applied-force and no-attachment evidence is required")
    for i, row in enumerate(rows):
        if (row.get("physics_step") != i + 1 or row.get("qfrc_applied_all_zero") is not True
                or row.get("xfrc_applied_all_zero") is not True
                or row.get("max_abs_qfrc_applied") != 0 or row.get("max_abs_xfrc_applied") != 0
                or (i and not math.isclose(row["sim_time_s"] - rows[i-1]["sim_time_s"], .005, rel_tol=0, abs_tol=1e-6))):
            raise ValueError("Applied forces/clock differ in an actual200Hz guard sample")
    return {"integration_rows": expected_rows, "guard_steps": len(rows), "equality_count": 0,
            "bindings": {str(p): sha256(p) for p in (model_path, trajectory_path, log_path)}}


def audit_physics(run, candidate, plan_sha256, *, expected_warmup="measured_pose_pd", expected_reference=None, _with_history=False):
    """Recompute the original50Hz gate; producer success flags alone cannot pass."""
    from handover_contact_quality import evaluate_capture
    run = Path(run).resolve(strict=True)
    result = bind_result(run, candidate, plan_sha256, expected_warmup=expected_warmup, expected_reference=expected_reference)
    forces = audit_applied_forces(run, result)
    history = authoritative_history(run, result, candidate, expected_warmup=expected_warmup)
    samples_path = run / "quality_samples.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line]
    if len(samples) != result["source_commands"]:
        raise ValueError("Every actual50Hz teacher tick needs an independent physical measurement")
    if any(row.get("completed_steps") != i+1 or not math.isclose(row["sim_time_s"], history["times"][51+i], rel_tol=0, abs_tol=1e-6)
           for i, row in enumerate(samples)):
        raise ValueError("Live solved-stage quality samples differ from their actual50Hz absolute clock")
    quality = evaluate_capture(samples, initial_target_z_m=result["initial_target_z_m"])
    if (quality.get("eligible_physics") is not True or quality.get("verified_handover_success") is not True
            or quality != read(run / "physical_quality.json")
            or quality != result.get("physical_quality")):
        raise ValueError("Independently recomputed full handover-physics-v2 gate did not pass")
    report = {"status": "independent_physics_audit_passed", "eligible_dataset": False,
            "candidate_id": candidate["candidate_id"], "candidate_spec_sha256": candidate["candidate_spec_sha256"],
            "plan_sha256": plan_sha256, "source_episode_index": result["source_episode_index"],
            "physical_gate": quality, "initial_target_z_m": result["initial_target_z_m"],
            "source_commands": result["source_commands"],
            "teacher_reference": result.get("teacher_reference"),
            "startup_history_padding_audit": history.get("startup_history_padding_audit"),
            "teacher_collection_warmup": {"control": expected_warmup, "actual_receipt_kind": result["warmup_control"]["kind"],
                "seconds": 1., "scope": WARMUP_SCOPE, "warmup_supervision": "none", "evaluation_oracle_allowed": False},
            "applied_force_audit": forces,
            "authority_revision": AUTHORITY_REVISION,
            "bindings": {**forces["bindings"], **history["bindings"], **{str(p): sha256(p) for p in (run / "result.json", samples_path, run / "physical_quality.json")}}}
    return (report, history) if _with_history else report


def modified_reference_commands(result, parquet, replay, hand_names, *, original_source=None,
                                candidate=None, expected_warmup=None):
    """Independently verify an appended teacher tail against the protected source.

    References never replace resets, observations or source ancestry. The whole
    original native50 command prefix must remain bitwise identical. Only the
    explicitly documented appended command rows become new teacher labels.
    """
    import numpy as np
    reference = result.get("teacher_reference")
    if reference is None:
        return None
    path = Path(reference["path"]).resolve(strict=True)
    if result["source_sha256"].get(str(path)) != reference["sha256"] or sha256(path) != reference["sha256"]:
        raise ValueError("Modified teacher JSON is not bound with original source inputs")
    artifact = read(path)
    if (artifact.get("contract") != "cvpr-handover-modified-teacher-reference-v1"
            or artifact.get("source_episode_index") != result["source_episode_index"]
            or artifact.get("parent_source_parquet_sha256") != sha256(parquet)
            or artifact.get("parent_source_commands") != len(replay)
            or artifact.get("original_prefix_rows") != len(replay)
            or artifact.get("scope") != "TRAIN_teacher_only"
            or artifact.get("evaluation_oracle_allowed") is not False
            or artifact.get("command_rate_hz") != 50
            or artifact.get("body_token_version") != "body_token_v1_1"
            or artifact.get("fixed_encoder") != {"sha256":"fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae",
                                                  "decoder_sha256":"34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509"}
            or not isinstance(artifact.get("method"), (str, dict)) or not artifact["method"]):
        raise ValueError("Modified teacher lineage, original prefix or TRAIN-only declaration differs")
    names = [f"{side}_hand_{finger}_{j}_joint" for side in ("left", "right")
             for finger, count in (("thumb", 3), ("index", 2), ("middle", 2)) for j in range(count)]
    if artifact.get("hand_joint_names") != names:
        raise ValueError("Modified teacher hand order must be the documented native replay order")
    repair_inputs = artifact.get("repair_inputs_sha256")
    if not isinstance(repair_inputs, dict) or not repair_inputs:
        raise ValueError("Modified teacher repair must retain its actual physical generation evidence")
    for evidence, identity in repair_inputs.items():
        evidence = Path(evidence)
        if (not evidence.is_absolute() or result["source_sha256"].get(str(evidence)) != identity
                or sha256(evidence) != identity):
            raise ValueError("Modified teacher generation evidence is missing or changed")
    entry = artifact["commands"]
    commands_path = local_file(path.parent, entry["path"])
    if (result["source_sha256"].get(str(commands_path)) != entry["sha256"]
            or sha256(commands_path) != entry["sha256"]):
        raise ValueError("Modified teacher command array is not bound with source inputs")
    commands = np.load(commands_path, allow_pickle=False)
    if (commands.dtype != np.dtype("float32") or commands.ndim != 2
            or commands.shape != (entry["rows"], 78) or entry.get("columns") != 78
            or entry.get("dtype") != "float32" or not len(replay) < len(commands) <= len(replay)+250
            or len(commands) != result["source_commands"] or not np.isfinite(commands).all()
            or not np.array_equal(commands[:len(replay)], replay)):
        raise ValueError("Modified teacher must preserve the full source prefix and append at most5s of finite commands")
    actual_entry = artifact["actual_executed_commands"]
    applied_path = local_file(path.parent, actual_entry["path"])
    if (result["source_sha256"].get(str(applied_path)) != actual_entry["sha256"]
            or sha256(applied_path) != actual_entry["sha256"]):
        raise ValueError("Modified teacher actual execution table is not bound")
    applied = np.load(applied_path, allow_pickle=False)
    rows = np.arange(len(commands))
    indices = (((rows*30//50)*50)+29)//30
    if (applied.dtype != np.dtype("float32") or applied.shape != commands.shape
            or actual_entry.get("rows") != len(commands) or actual_entry.get("columns") != 78
            or actual_entry.get("dtype") != "float32"
            or not np.array_equal(applied, commands[indices])):
        raise ValueError("Modified teacher native50 reference does not reproduce its actual30Hz held commands")
    if original_source is not None:
        # Production always supplies the independently read original parquet,
        # lineage row and frozen current candidate. The manual numeric checks
        # above remain independent of this shared reference-format validator.
        from prepare_handover_teacher_reference import validate_reference
        if candidate is None or expected_warmup is None:
            raise ValueError("Modified teacher provenance requires the frozen execution candidate and startup")
        execution = {"candidate_id": candidate["candidate_id"],
            "candidate_spec_sha256": candidate["candidate_spec_sha256"],
            "plan_sha256": result["plan_sha256"],
            "source_episode_index": candidate["ancestor"]["source_episode_index"]}
        independently_checked, receipt = validate_reference(path, original_source,
            expected_candidate=execution, expected_warmup=expected_warmup, allow_new_execution_candidate=True)
        if (not np.array_equal(independently_checked, commands) or receipt != reference
                or result.get("source_parent_rows") != len(replay)
                or result.get("teacher_reference_inputs_unchanged") is not True):
            raise ValueError("Modified teacher origin/execution/source receipt differs from independent reference validation")
    return commands[:, :64], commands[:, 64:][:, [names.index(name) for name in hand_names]]


def source_commands(result, *, with_initial=False, candidate=None, expected_warmup=None):
    """Read the identity-bound original source labels independently of the producer."""
    import numpy as np
    import pyarrow.parquet as pq
    paths = [Path(p) for p in result["source_sha256"]]
    parquet = [p for p in paths if p.suffix == ".parquet" and "Handover-sonic11-train" in p.parts]
    info = [p for p in paths if p.name == "info.json" and "Handover-sonic11-train" in p.parts]
    if len(parquet) != 1 or len(info) != 1:
        raise ValueError("One identity-bound TRAIN source parquet and schema are required")
    table = pq.read_table(parquet[0], columns=["action.body_token_v1_1", "action", "replay_action78",
        "replay.initial_state43", "observation.base_pose", "observation.base_vel", "observation.object_poses", "timestamp"])
    values = table.to_pydict()
    tokens, hands = np.asarray(values["action.body_token_v1_1"], dtype=np.float32), np.asarray(values["action"], dtype=np.float32)
    schema = read(info[0])
    names = schema["features"]["action"]["names"]
    selection = [p for p in paths if p.name == "selection.json"]
    episodes = [p for p in paths if p.name == "episodes.jsonl" and "Handover-sonic11-train" in p.parts]
    if len(selection) != 1 or len(episodes) != 1 or schema.get("fps") != 50 or str(schema.get("sonic_version")) != "1.1":
        raise ValueError("Independent source membership and SONICv1.1 clock/schema binding missing")
    source_id = result["source_episode_index"]
    chosen = [r for r in read(selection[0])["episodes"] if r["source_scene_episode_index"] == source_id]
    if len(chosen) != 1 or chosen[0].get("split") != "train" or chosen[0].get("data_category") != "original_or_repaired":
        raise ValueError("Captured source is not an independently confirmed original TRAIN ancestor")
    dense_index = int(parquet[0].stem.removeprefix("episode_"))
    rows = [json.loads(line) for line in episodes[0].read_text().splitlines() if line]
    rows = [r for r in rows if r["episode_index"] == dense_index]
    if (len(rows) != 1 or rows[0].get("source_scene_episode_index") != source_id
            or rows[0].get("data_category") != "original_or_repaired"
            or rows[0].get("split_restriction") != "unrestricted"
            or rows[0].get("lineage_complete") is not True
            or rows[0].get("lineage_group_sources") != [{"source_dataset_key": rows[0]["source_dataset_key"], "source_episode_index": source_id}]):
        raise ValueError("Bound source row ancestry/dense-index mapping is inconsistent")
    record = result["source_provenance"]["source_record"]
    parquet_identity = record["required_files"]["parquet"]
    if (record.get("source_dataset_key") != rows[0]["source_dataset_key"]
            or record.get("train_episode_index") != dense_index or record.get("source_rows") != len(tokens)
            or record.get("parquet") != parquet_identity
            or parquet_identity["sha256"] != result["source_sha256"][str(parquet[0])]
            or not parquet[0].as_posix().endswith("/" + parquet_identity["path"])):
        raise ValueError("Actual TRAIN parquet identity differs from layered source provenance")
    if (tokens.shape != (record["source_rows"], 64) or hands.shape != (len(tokens), 14)
            or len(set(names)) != 14 or not np.isfinite(tokens).all() or not np.isfinite(hands).all()
            or not np.allclose(values["timestamp"], np.arange(len(tokens))/50, rtol=0, atol=2e-5)):
        raise ValueError("Bound source action schema is incomplete")
    # Replay labels and training labels must describe the same named targets.
    replay_names = [f"{side}_hand_{finger}_{j}_joint" for side in ("left", "right")
                    for finger, count in (("thumb", 3), ("index", 2), ("middle", 2)) for j in range(count)]
    replay = np.asarray(values["replay_action78"], dtype=np.float32)
    if (set(names) != set(replay_names) or replay.shape != (len(tokens), 78)
            or not np.array_equal(replay[:, :64], tokens)
            or not np.array_equal(replay[:, 64:], hands[:, [names.index(n) for n in replay_names]])):
        raise ValueError("Source replay commands differ from named TRAIN action labels")
    if result.get("teacher_reference") is not None and (candidate is None or expected_warmup is None):
        raise ValueError("Modified teacher source audit requires the frozen candidate/startup context")
    original_source = {"source_split": "train", "commands": replay, "row": rows[0],
        "source_record": record, "sources": {str(parquet[0]): result["source_sha256"][str(parquet[0])]}}
    modified = modified_reference_commands(result, parquet[0], replay, names,
        original_source=original_source, candidate=candidate, expected_warmup=expected_warmup)
    if modified is None:
        if len(tokens) != result["source_commands"]:
            raise ValueError("Unmodified teacher count differs from the protected source")
    else:
        tokens, hands = modified
    if not with_initial:
        return tokens, hands, names
    physics_paths = [p for p in paths if p.name == f"source_episode_{source_id:06d}_physics.json"]
    if len(physics_paths) != 1:
        raise ValueError("Require a SHA-bound source physical reset record")
    physics = read(physics_paths[0])
    initial_joints = finite_vector(values["replay.initial_state43"][0], 43, "source row0 reset joints")
    reset_names = physics["reset_joint_names"]
    if (len(set(reset_names)) != 43 or not np.array_equal(initial_joints, physics["reset_joint_state43"])
            or physics.get("gravity") != [0., 0., -9.81]):
        raise ValueError("Source row0 reset/physics identity differs")
    config = json.loads(rows[0]["environment_config"])
    object_names = schema["features"]["observation.object_poses"]["names"]
    object_values = finite_vector(values["observation.object_poses"][0], 14, "source row0 object poses")
    objects = {}
    for role in ("target", "container"):
        terms = [f"{role}.{term}" for term in ("pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z")]
        objects[role] = {"pose7": finite_pose([object_values[object_names.index(n)] for n in terms], f"source {role}"),
            "body_name": config["dr_state_dict"][role]["label"], "mass": physics["objects"][role]["mass"]}
    initial = {"joint_names": reset_names, "joint_qpos": initial_joints,
        "root_pose7": finite_pose(values["observation.base_pose"][0], "source row0 root"),
        "root_qvel6": finite_vector(values["observation.base_vel"][0], 6, "source row0 root velocity"),
        "objects": objects}
    return tokens, hands, names, initial


def authoritative_history(run, result, candidate, *, expected_warmup="measured_pose_pd"):
    """Associate every sidecar with real serialized qpos/qvel/time, without stepping.

    mj_forward is used once only to validate rigid geometry against the MJB.
    It is never used to manufacture or replace the original solved-force log.
    """
    import mujoco
    import numpy as np
    from handover_geometry_quality import build_container_geometry_cache
    run = Path(run)
    tokens, hands, hand_names, source = source_commands(result, with_initial=True,
        candidate=candidate, expected_warmup=expected_warmup)
    record = result["source_provenance"]["source_record"]
    if (record.get("source_episode_index") != candidate["ancestor"]["source_episode_index"]
            or record.get("source_dataset_key") != candidate["ancestor"]["source_dataset_key"]
            or record.get("plan_declared_source_parquet_sha256") != candidate["ancestor"]["source_parquet_sha256"]):
        raise ValueError("Original-plan versus actual-TRAIN source identity is not bound")
    model = mujoco.MjModel.from_binary_path(str(run / "physics_model.mjb"))
    data = mujoco.MjData(model)
    if not np.array_equal(model.opt.gravity, [0., 0., -9.81]):
        raise ValueError("Physical MJB gravity differs from the bound source")
    names = source["joint_names"]
    joints = [model.joint(n).id for n in names]
    if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
        raise ValueError("All43 source named joints must be scalar hinges")
    qadr = [int(model.jnt_qposadr[j]) for j in joints]
    dadr = [int(model.jnt_dofadr[j]) for j in joints]

    def free_joint(name):
        body = model.body(name).id
        j = int(model.body_jntadr[body])
        if model.body_jntnum[body] != 1 or j < 0 or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Authority requires named floating root: {name}")
        return int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])

    root_q, root_dq = free_joint("pelvis")
    cache_path = local_file(run, "container_geometry_cache.json")
    cache = read(cache_path)
    object_adrs = {}
    for role, item in source["objects"].items():
        if cache.get(f"{role}_root_body_name") != item["body_name"]:
            raise ValueError("Geometry-cache object role differs from SHA-bound source scene")
        object_adrs[role] = free_joint(item["body_name"])
        if not np.isclose(model.body_mass[model.body(item["body_name"]).id], item["mass"], rtol=1e-5, atol=1e-7):
            raise ValueError("Physical MJB/source object mass differs")
    if len({root_q, *(q for q, _ in object_adrs.values())}) != 3:
        raise ValueError("Root, target and container must be distinct free bodies")
    with np.load(run / "trajectory.npz", allow_pickle=False) as values:
        spec = int(values["physics_state_spec"])
        states, times = values["physics_state_integration"], values["sim_time_s"]
        side_names = values["joint_names"].tolist()
        side_joints, side_roots, side_vel = values["actual_joint_q43"], values["root_pose7"], values["actual_qvel"]
    count = 51 + len(tokens)
    if (len(side_names) != 43 or set(side_names) != set(names) or side_joints.shape != (count, 43)
            or side_roots.shape != (count, 7) or side_vel.shape != (count, model.nv)):
        raise ValueError("Physical-history auxiliary schema differs from source/model authority")
    perm = [names.index(n) for n in side_names]
    physical_q, physical_dq, roots, root_velocities = [], [], [], []
    object_poses = {role: [] for role in object_adrs}
    # SIMPLE's reset performs exactly one real5ms step before this cold start.
    expected_times = model.opt.timestep + np.arange(count)/50
    if (not np.allclose(times, expected_times, rtol=0, atol=1e-6)
            or not math.isclose(result["initialization"]["sim_time_s"], expected_times[0], rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Absolute reset/integration clock differs from the real5ms SIMPLE reset")
    for index, state in enumerate(states):
        mujoco.mj_setState(model, data, state, spec)
        if not math.isclose(float(data.time), expected_times[index], rel_tol=0, abs_tol=1e-6):
            raise ValueError("Serialized integration absolute clock differs from its50Hz row")
        q, dq = data.qpos[qadr].copy(), data.qvel[dadr].copy()
        root, root_velocity = data.qpos[root_q:root_q+7].copy(), data.qvel[root_dq:root_dq+6].copy()
        if (not np.array_equal(side_joints[index], q[perm])
                or not np.allclose(side_roots[index], root, rtol=0, atol=1e-12)
                or not np.array_equal(side_vel[index], data.qvel)):
            raise ValueError("Auxiliary measured arrays differ from authoritative integration qpos/qvel/root")
        physical_q.append(q); physical_dq.append(dq); roots.append(root); root_velocities.append(root_velocity)
        for role, (adr, _) in object_adrs.items():
            object_poses[role].append(data.qpos[adr:adr+7].copy())
        if index == 0:
            expected_velocity = np.zeros(model.nv)
            expected_velocity[root_dq:root_dq+6] = source["root_qvel6"]
            if (not np.array_equal(q, source["joint_qpos"])
                    or not np.allclose(root, source["root_pose7"], rtol=0, atol=1e-12)
                    or not np.array_equal(data.qvel, expected_velocity)):
                raise ValueError("Actual initial joints/root/velocities differ from SHA-bound source row0")
            for role, item in source["objects"].items():
                delta = candidate["initial_object_deltas"][role]
                receipt = result["perturbation_receipt"]["objects"][role]
                original = finite_pose(receipt["source_pose7"], "source reset pose")
                applied = finite_pose(receipt["applied_pose7"], "applied reset pose")
                expected = item["pose7"].copy(); expected[:2] += delta["xy_delta_m"]
                c, s = math.cos(delta["world_z_yaw_delta_rad"]/2), math.sin(delta["world_z_yaw_delta_rad"]/2)
                w, x, y, z = expected[3:]
                expected[3:] = [c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w]
                if (not np.array_equal(original, item["pose7"])
                        or receipt.get("xy_delta_m") != delta["xy_delta_m"]
                        or receipt.get("world_z_yaw_delta_rad") != delta["world_z_yaw_delta_rad"]
                        or not np.allclose(applied, expected, rtol=0, atol=1e-12)
                        or not np.allclose(object_poses[role][0], expected, rtol=0, atol=1e-12)):
                    raise ValueError("Source-row0/object-reset authority or preregistered perturbation differs")
            if not math.isclose(result["initial_target_z_m"], object_poses["target"][0][2], rel_tol=0, abs_tol=1e-12):
                raise ValueError("Gate lift origin differs from authoritative initial target height")
            # Kinematics/rigid geometry validation only. All force evidence still
            # comes from the live solved-stage samples and serialized force fields.
            mujoco.mj_forward(model, data)
            rebuilt = build_container_geometry_cache(model, data,
                container_body=source["objects"]["container"]["body_name"],
                target_body=source["objects"]["target"]["body_name"])
            if set(cache) != set(rebuilt):
                raise ValueError("Geometry cache schema differs from the actual MJB")
            for key, expected in rebuilt.items():
                actual = cache[key]
                if isinstance(expected, list):
                    if np.shape(actual) != np.shape(expected) or not np.allclose(actual, expected, rtol=0, atol=1e-10):
                        raise ValueError("Geometry cache vertices/roles differ from the actual MJB")
                elif actual != expected:
                    raise ValueError("Geometry cache identity differs from the actual MJB")
    history = {"joint_names": names, "q": np.asarray(physical_q), "dq": np.asarray(physical_dq),
        "root": np.asarray(roots), "root_velocity": np.asarray(root_velocities),
        "objects": {k: np.asarray(v) for k, v in object_poses.items()}, "times": times,
        "tokens": tokens, "hands": hands, "hand_names": hand_names,
        "model": model, "integration_states": states, "state_spec": spec,
        "bindings": {str(cache_path): sha256(cache_path)}}
    trace_path = local_file(run, "trace.jsonl")
    trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    if len(trace) != count-1:
        raise ValueError("Every actual50Hz warmup/teacher control tick must have a complete trace")
    live_events = []
    if expected_warmup in ("live_pose_sonic_v1_1", PADDED_WARMUP):
        warmup = result["warmup_control"]
        live_events = warmup["events"]
        expected_calls = 50 if expected_warmup == PADDED_WARMUP else 40
        if (len(live_events) != expected_calls or warmup.get("encoder_calls") != expected_calls
                or not math.isclose(warmup["first_sim_time_s"], times[0], rel_tol=0, abs_tol=1e-6)
                or not np.array_equal(finite_vector(warmup["reference_body_q29"],29,"initial live reference body").astype(np.float32), history["q"][0,:29].astype(np.float32))
                or not np.allclose(warmup["reference_root_wxyz"], history["root"][0,3:], rtol=0, atol=1e-12)
                or set(warmup["reference_hands"]) != set(hand_names)):
            raise ValueError("Live TRAIN warmup reference/history differs from actual initial observed state")
        if not np.array_equal(np.asarray([warmup["reference_hands"][n] for n in hand_names], dtype=np.float32),
                              history["q"][0,[names.index(n) for n in hand_names]].astype(np.float32)):
            raise ValueError("Live TRAIN warmup hand reference is not the initial measured observation")
    decoder_calls = 0
    for tick, row in enumerate(trace):
        if (row.get("tick50") != tick or row.get("physics_state_row_before") != tick
                or row.get("physics_state_row_after") != tick+1
                or not math.isclose(row["sim_time_s"], times[tick], rel_tol=0, abs_tol=1e-6)
                or not math.isclose(row["next_sim_time_s"], times[tick+1], rel_tol=0, abs_tol=1e-6)
                or not math.isclose(row["elapsed_s"], tick/50, rel_tol=0, abs_tol=1e-6)):
            raise ValueError("Full50Hz trace absolute clock/integration row association differs")
        for side, index in (("before", tick), ("after", tick+1)):
            for role in ("target", "container"):
                if not np.allclose(finite_pose(row[f"measured_{side}"][f"{role}_pose7"], f"trace {role}"),
                        history["objects"][role][index], rtol=0, atol=1e-12):
                    raise ValueError("Full50Hz trace object pose differs from authoritative integration")
        actual_hands = row["actual_hand_targets"]
        if set(actual_hands) != set(hand_names):
            raise ValueError("Full50Hz command trace hand identities differ")
        if tick < 50:
            expected_hands = history["q"][tick, [names.index(n) for n in hand_names]]
            if row.get("source_command_index") is not None:
                raise ValueError("Warmup must never reference teacher source labels")
            if expected_warmup == "measured_pose_pd" or (expected_warmup != PADDED_WARMUP and tick < 10):
                if row.get("phase") != "warmup_measured_pose_pd" or row.get("actual_body_token64") is not None:
                    raise ValueError("Measured-pose warmup trace must have no fabricated body token")
                increment = 0
            else:
                token = finite_vector(row["actual_body_token64"], 64, "actual live warmup token")
                event = live_events[tick if expected_warmup == PADDED_WARMUP else tick-10]
                if (row.get("phase") != "warmup_" + expected_warmup
                        or not math.isclose(event["sim_time_s"], times[tick], rel_tol=0, abs_tol=1e-6)
                        or not np.array_equal(token.astype(np.float32), finite_vector(event["token64"],64,"live receipt token").astype(np.float32))
                        or event.get("state_forcing") is not False
                        or event.get("history_injection") is not (expected_warmup == PADDED_WARMUP and tick < 9)):
                    raise ValueError("Live warmup token/time differs from its bound real-history receipt")
                expected_hands = history["q"][0, [names.index(n) for n in hand_names]]
                increment = 1
        else:
            source_index = (((tick-50)*30//50)*50 + 29)//30
            expected_hands = hands[source_index]
            if (row.get("phase") != "teacher_command" or row.get("source_command_index") != source_index
                    or not np.array_equal(finite_vector(row["actual_body_token64"], 64, "trace body token").astype(np.float32), tokens[source_index])):
                raise ValueError("Actual50Hz command trace violates deployed floor30/ceil50 sample-and-hold")
            increment = 1
        if not np.array_equal(finite_vector([actual_hands[n] for n in hand_names], 14, "trace hand targets").astype(np.float32), expected_hands.astype(np.float32)):
            raise ValueError("Actual50Hz trace hand targets differ from the measured warmup/source command")
        if row.get("decoder_calls_before") != decoder_calls or row.get("decoder_calls_after") != decoder_calls+increment:
            raise ValueError("Actual50Hz SONIC decoder calls differ from scheduled warmup/teacher history")
        decoder_calls += increment
    if result.get("decoder_calls") != decoder_calls:
        raise ValueError("Final SONIC decoder count differs from the full scheduled control trace")
    if expected_warmup == PADDED_WARMUP:
        history["startup_history_padding_audit"] = audit_startup_history(history, trace, result)
    # The200Hz log also uses absolute times, so shifting the entire event clock
    # or changing only a trace/sidecar clock cannot preserve association.
    guard_path = local_file(run, result["force_guard"]["path"])
    guard = [json.loads(line) for line in guard_path.read_text().splitlines() if line]
    if not np.allclose([r["sim_time_s"] for r in guard], times[0] + np.arange(1, 4*(count-1)+1)/200, rtol=0, atol=1e-6):
        raise ValueError("Actual200Hz force-guard absolute clock differs from integration history")
    history["bindings"][str(trace_path)] = sha256(trace_path)
    return history


def audit_startup_history(history, trace, result):
    """Rebuild every padded/real decoder input from actual integration states.

    Only initial snapshot copies are permitted, with their real t0 retained.
    Previous actions are checked against the preceding executed controller
    output. Geometry/IMU computations below are not contact-force evidence.
    """
    import importlib.util
    import sys
    import mujoco
    import numpy as np
    path = REPO / "VLMExpert/third_party/SIMPLE/src/simple/baselines/sonic_v1_1.py"
    if result["capture_code_sha256"].get(str(path)) != sha256(path):
        raise ValueError("Startup history requires the frozen actual SONIC numeric contract")
    simple_src = str(REPO / "VLMExpert/third_party/SIMPLE/src")
    if simple_src not in sys.path:
        sys.path.insert(0, simple_src)
    spec = importlib.util.spec_from_file_location("handover_startup_sonic_numeric_contract", path)
    sonic = importlib.util.module_from_spec(spec); spec.loader.exec_module(sonic)
    receipt = result["warmup_control"]["startup_history_padding"]
    count = len(trace)
    if (receipt.get("contract") != "cvpr-sonic-v11-startup-repeat-first-snapshot-v1"
            or receipt.get("policy") != "repeat_first_snapshot" or receipt.get("seeded_history_slots") != 10
            or receipt.get("same_timestamp") is not True or receipt.get("initialization_count") != 1
            or receipt.get("state_forcing") is not False or receipt.get("source_future_used") is not False
            or receipt.get("history_injection") is not True or receipt.get("padded_decoder_calls") != 9
            or receipt.get("total_decoder_calls") != count or len(receipt.get("decoder_inputs", [])) != count
            or not math.isclose(receipt["first_all_real_decoder_sim_time_s"], history["times"][9], rel_tol=0, abs_tol=1e-6)):
        raise ValueError("Only a declared causal9-call initial-snapshot padding prefix is permitted")
    first = receipt["first_snapshot"]
    if digest(first) != receipt.get("first_snapshot_sha256"):
        raise ValueError("Startup first actual snapshot identity differs")
    names, model = history["joint_names"], history["model"]
    body = [names.index(name) for name in sonic.BODY_JOINT_NAMES]
    permutation = sonic.POLICY_TO_BODY
    default, scale = sonic.DEFAULT_ANGLES, sonic.ACTION_SCALE
    initial_previous = np.asarray(((history["q"][0, body]-default)/scale)[permutation], np.float32)
    if (first.get("body_joint_names") != list(sonic.BODY_JOINT_NAMES)
            or not np.array_equal(finite_vector(first["physical_body_qpos29"],29,"startup measured body"), history["q"][0,body])
            or not np.array_equal(finite_vector(first["physical_body_qvel29"],29,"startup measured velocity"), history["dq"][0,body])
            or not np.allclose(first["root_pose7"], history["root"][0], rtol=0, atol=1e-12)
            or not np.array_equal(first["root_qvel6"], history["root_velocity"][0])
            or not np.array_equal(first["body_default_angles29"],default)
            or not np.array_equal(first["body_action_scale29"],scale)
            or not np.array_equal(finite_vector(first["controller_reset_previous_raw_action29"],29,"reset previous action"), np.zeros(29))
            or not np.array_equal(np.asarray(first["effective_initial_previous_raw_action29"],np.float32),initial_previous)):
        raise ValueError("Startup reference/default/previous action differs from the first true integration state")
    observed_names = first["observed_joint_names"]
    if (set(observed_names) != set(names) or len(observed_names) != len(names)
            or not np.array_equal(np.asarray(first["observed_joint_qpos"],np.float32),
                                  history["q"][0,[names.index(n) for n in observed_names]].astype(np.float32))):
        raise ValueError("Startup observation is not the initial real measured joint snapshot")
    data = mujoco.MjData(model)
    pelvis = model.body("pelvis").id
    samples = []
    for tick, row in enumerate(trace):
        mujoco.mj_setState(model, data, history["integration_states"][tick], history["state_spec"])
        # Match actual pre-control IMU kinematics, without advancing simulation.
        mujoco.mj_forward(model, data)
        angular = np.zeros(6)
        mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,pelvis,angular,1)
        gravity = data.xmat[pelvis].reshape(3,3).T @ np.array([0.,0.,-1.])
        if tick == 0:
            previous = initial_previous
            if (not np.allclose(first["body_local_angular_velocity3"],angular[:3],rtol=0,atol=1e-12)
                    or not np.allclose(first["projected_gravity3"],gravity,rtol=0,atol=1e-12)
                    or not np.array_equal(first["actual_qvel"],data.qvel)):
                raise ValueError("Startup first IMU/full velocity differs from actual integration state")
        else:
            previous = finite_vector(trace[tick-1]["controller_last_raw_policy_action29"],29,"preceding actual raw action").astype(np.float32)
        output = finite_vector(row["controller_last_raw_policy_action29"],29,"actual raw output").astype(np.float32)
        target = finite_vector(row["body_pd_target29"],29,"actual body PD target")
        if not np.allclose(target, default + output[sonic.BODY_TO_POLICY]*scale, rtol=0,atol=1e-7):
            raise ValueError("Actual controller output differs from the applied body target")
        samples.append((angular[:3],(history["q"][tick,body]-default)[permutation],
                        history["dq"][tick,body][permutation],previous,gravity))
        event = receipt["decoder_inputs"][tick]
        pad = max(0,9-tick)
        indices = [0]*pad + list(range(max(0,tick-9),tick+1))
        origins = ([{"kind":"startup_repeat_first_snapshot","padded_from_initial":True,"padded_from_actual_timestep_index":0} for _ in range(pad)]
            + [{"kind":"actual_physical_sample","actual_timestep_index":index} for index in indices[pad:]])
        if (event.get("decoder_call_index") != tick or event.get("padding_count") != pad
                or event.get("history_origin") != origins or event.get("all_ten_samples_real") is not (pad == 0)
                or row.get("startup_decoder_input_index") != tick or row.get("startup_history_padding_slots") != pad
                or not math.isclose(event["sim_time_s"],history["times"][tick],rel_tol=0,abs_tol=1e-6)
                or not np.allclose(event["history_sample_times_s"],history["times"][indices],rtol=0,atol=1e-6)):
            raise ValueError("Startup history must use only t0 copies then the true causal50Hz integration indices")
        token = finite_vector(row["actual_body_token64"],64,"actual decoder token")
        actual = np.asarray(event["input_values_float32"],np.float32)
        expected = np.concatenate([token] + [np.stack([samples[index][term] for index in indices]).reshape(-1)
                                            for term in range(5)]).astype(np.float32)
        if (actual.shape != (994,) or not np.array_equal(actual,expected)
                or event.get("input_sha256") != hashlib.sha256(actual.astype('<f4',copy=False).tobytes()).hexdigest()):
            raise ValueError("Actual decoder input differs from independently reconstructed real/padded history")
    return {"contract":"cvpr-handover-independent-startup-history-audit-v1", "decoder_inputs_checked":count,
            "padded_calls":9,"all_real_from_call":9,"source_future_used":False,
            "previous_action_chain_checked":True,"physical_state_source":"mjSTATE_INTEGRATION",
            "history_payload_sha256":digest(receipt)}


def audit_renderer_pose(event, model, data, *, world_reference, frame_calibration):
    """Compare raw renderer diagnostics to integration-derived pure kinematics.

    New data-admission tolerances are a declared convention, not a measured
    physics/solver accuracy claim. No render, solver or dynamics step occurs.
    """
    import mujoco
    import numpy as np
    if event.get("renderer_camera") != "head_stereo_left":
        raise ValueError("RGB event must identify its real head_stereo_left renderer camera")
    diag = event["renderer_pose_diagnostics"]
    if (not math.isclose(diag["mujoco_sim_time_s"], float(data.time), rel_tol=0, abs_tol=1e-6)
            or not math.isclose(diag["isaac_world_time_s"], world_reference, rel_tol=0, abs_tol=1e-9)):
        raise ValueError("Renderer diagnostic absolute time or nonadvancing Isaac world clock differs")
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)

    def usd_pose(value):
        if value.get("valid") is not True or not isinstance(value.get("path"), str) or not value["path"]:
            raise ValueError("Actual renderer USD prim pose is absent/invalid")
        return finite_pose([*value["position_m"], *value["quaternion_wxyz"]], "actual USD pose")

    def errors(first, second):
        # Quaternion signs are equivalent. Normalize only for computing a
        # rotation residual; finite_pose already checked the observed norm.
        a, b = first[3:]/np.linalg.norm(first[3:]), second[3:]/np.linalg.norm(second[3:])
        return float(np.linalg.norm(first[:3]-second[:3])), float(2*math.acos(float(np.clip(abs(np.dot(a,b)), 0, 1))))

    # USD and SONIC MJCF deliberately use different waist joint origins.
    # Map their documented source frames at the measured pose, never fit a
    # translation from the first observation or infer it from a render residual.
    torso, yaw, roll = (model.body(n).id for n in ("torso_link", "waist_yaw_link", "waist_roll_link"))
    usd_torso_position = frame_calibration.usd_torso_frame_position(data.xpos[torso],
        data.xmat[yaw].reshape(3,3), data.xmat[roll].reshape(3,3), model.body_pos[roll], model.body_pos[torso])
    stats = {"max_link_position_error_m": 0., "max_link_rotation_error_rad": 0.}
    links = diag["links"]
    for name in ("pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"):
        entry = links[name]
        body = model.body(name).id
        expected = np.r_[data.xpos[body], data.xquat[body]]
        measured = finite_pose([*entry["mujoco_position_m"], *entry["mujoco_quaternion_wxyz"]], "diagnostic MuJoCo link")
        if not np.allclose(measured, expected, rtol=0, atol=1e-9):
            raise ValueError("Renderer MuJoCo link evidence differs from authoritative integration kinematics")
        actual_usd = usd_pose(entry["usd"])
        if name == "torso_link":
            stats["raw_torso_position_error_m"], stats["raw_torso_rotation_error_rad"] = errors(actual_usd, expected)
            expected = np.r_[usd_torso_position, expected[3:]]
        elif name in ("left_wrist_yaw_link", "right_wrist_yaw_link"):
            side = "left" if name.startswith("left") else "right"
            shoulder = model.body(f"{side}_shoulder_pitch_link").id
            stats[f"raw_{side}_wrist_position_error_m"], stats[f"raw_{side}_wrist_rotation_error_rad"] = errors(actual_usd, expected)
            expected = expected.copy()
            expected[:3] = frame_calibration.usd_wrist_frame_position(expected[:3], data.xpos[torso],
                usd_torso_position, data.xmat[torso].reshape(3,3), model.body_pos[shoulder], side)
        position, rotation = errors(actual_usd, expected)
        stats["max_link_position_error_m"] = max(stats["max_link_position_error_m"], position)
        stats["max_link_rotation_error_rad"] = max(stats["max_link_rotation_error_rad"], rotation)
    for entry in links.values():
        usd_pose(entry["usd"])
    camera = diag["cameras"]["head_stereo_left"]
    cid = model.camera("head_stereo_left").id
    rotation = np.asarray(camera["mujoco_rotation_usd_axes"], dtype=np.float64)
    camera_position = finite_vector(camera["mujoco_position_m"], 3, "diagnostic MuJoCo camera position")
    if (rotation.shape != (9,) or not np.isfinite(rotation).all()
            or not np.allclose(camera_position, data.cam_xpos[cid], rtol=0, atol=1e-9)
            or not np.allclose(rotation, data.cam_xmat[cid], rtol=0, atol=1e-9)):
        raise ValueError("Renderer MuJoCo camera evidence differs from authoritative integration kinematics")
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, data.cam_xmat[cid])
    expected = np.r_[data.cam_xpos[cid], quaternion]
    usd = usd_pose(camera["usd"])
    sensor = finite_pose([*camera["sensor_position_m"], *camera["sensor_quaternion_wxyz_usd_axes"]], "actual camera sensor")
    sign = 1 if np.dot(usd[3:],sensor[3:]) >= 0 else -1
    if (np.max(np.abs(usd[:3]-sensor[:3])) > 1e-5
            or not np.allclose(usd[3:],sign*sensor[3:],rtol=0,atol=1e-5)):
        raise ValueError("Actual camera sensor and its USD prim pose differ")
    stats["raw_camera_usd_position_error_m"], stats["raw_camera_usd_rotation_error_rad"] = errors(usd, expected)
    stats["raw_camera_sensor_position_error_m"], stats["raw_camera_sensor_rotation_error_rad"] = errors(sensor, expected)
    # Use the actual source USD chain to compare corresponding sensor frames.
    # Original MJ camera extrinsics and raw residuals remain above, unmodified.
    expected[:3] = frame_calibration.usd_head_camera_frame_position(usd_torso_position, data.xmat[torso].reshape(3,3))
    stats["camera_usd_position_error_m"], stats["camera_usd_rotation_error_rad"] = errors(usd, expected)
    stats["camera_sensor_position_error_m"], stats["camera_sensor_rotation_error_rad"] = errors(sensor, expected)
    return stats


def load_frame_calibration(model, result):
    """Load the separately reviewed source-asset mapping and validate its assumptions."""
    import importlib.util
    import numpy as np
    if result["capture_code_sha256"].get(str(FRAME_CALIBRATION)) != sha256(FRAME_CALIBRATION):
        raise ValueError("Renderer joint-chain calibration source must be bound to the actual capture")
    spec = importlib.util.spec_from_file_location("handover_source_frame_calibration", FRAME_CALIBRATION)
    calibration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibration)
    # These are the explicitly documented source model assumptions; altered
    # axes/body rotations need a newly reviewed mapping, not silent adaptation.
    bodies = [model.body(n).id for n in ("waist_yaw_link", "waist_roll_link", "torso_link")]
    if model.body_parentid[bodies[1]] != bodies[0] or model.body_parentid[bodies[2]] != bodies[1]:
        raise ValueError("Waist source-frame mapping requires its declared parent chain")
    for body, axis in zip(bodies, ([0.,0.,1.], [1.,0.,0.], [0.,1.,0.])):
        if (not np.allclose(np.abs(model.body_quat[body]), [1.,0.,0.,0.], rtol=0, atol=1e-12)
                or model.body_jntnum[body] != 1):
            raise ValueError("Waist frame calibration assumes identity body rotations and one source joint")
        joint = int(model.body_jntadr[body])
        if (not np.allclose(model.jnt_axis[joint], axis, rtol=0, atol=1e-12)
                or not np.allclose(model.jnt_pos[joint], 0, rtol=0, atol=1e-12)):
            raise ValueError("Waist frame calibration source joint axis/origin differs")
    for side in ("left", "right"):
        if model.body_parentid[model.body(f"{side}_shoulder_pitch_link").id] != bodies[2]:
            raise ValueError("Wrist source-frame mapping requires its declared shoulder/torso parent chain")
    return calibration


def calibration_asset_bindings(result, calibration):
    """Verify actual local source assets, including the installation's symlinks."""
    root = Path(result["resource_data_root"])
    if not root.is_absolute() or str(root.resolve(strict=True)) != str(root) or not root.is_dir():
        raise ValueError("Calibration data root must be an actual canonical local directory")
    expected = calibration.SOURCE_IDENTITIES
    if (result.get("calibration_source_asset_sha256") != expected
            or result.get("calibration_source_assets_unchanged") is not True):
        raise ValueError("Actual calibration source identities/unchanged-during-run proof differs")
    bindings = {}
    for relative, identity in expected.items():
        # The bundled robot directory is intentionally a symlink; bind the
        # actual resolved payload rather than rejecting the legitimate layout.
        path = root / relative
        if not path.is_file() or sha256(path) != identity:
            raise ValueError("Actual calibration source asset is missing or changed")
        bindings[str(path.resolve(strict=True))] = identity
    return bindings


def audit_capture(run, capture, candidate, plan_sha256, *, expected_warmup="measured_pose_pd", expected_reference=None):
    """Read-only CPU audit of actual RTX pixels, clocks, labels and physical gate."""
    import numpy as np
    from PIL import Image
    run, capture = Path(run).resolve(strict=True), Path(capture).resolve(strict=True)
    physics, history = audit_physics(run, candidate, plan_sha256, expected_warmup=expected_warmup,
                                    expected_reference=expected_reference, _with_history=True)
    result, manifest = read(run / "result.json"), read(capture / "capture.json")
    expected = {"plan_sha256": plan_sha256, "candidate_id": candidate["candidate_id"],
                "candidate_spec_sha256": candidate["candidate_spec_sha256"],
                "source_episode_index": candidate["ancestor"]["source_episode_index"],
                "initial_target_z_m": result["initial_target_z_m"], "command_rate_hz": 30,
                "physics_hz": 200, "decoder_hz": 50, "render_engine": "Isaac RTX",
                "warmup_seconds": 1., "object_restores_during_control": False,
                "warmup_control": {"kind": expected_warmup, "seconds": 1.},
                "renderer_pose_contract": RENDERER_POSE_CONTRACT, "render_mount_equivalence_verified": False,
                "resource_data_root": result["resource_data_root"],
                "calibration_source_asset_sha256": result["calibration_source_asset_sha256"],
                "calibration_source_assets_unchanged": True,
                "initial_object_deltas": candidate["initial_object_deltas"],
                "perturbation_receipt": result["perturbation_receipt"], "source_sha256": result["source_sha256"]}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("RTX capture identity, controller clock, reset or source provenance differs")
    if manifest.get("teacher_reference") != result.get("teacher_reference"):
        raise ValueError("RTX command-reference lineage differs from its actual physical replay")
    if expected_warmup == PADDED_WARMUP:
        if (manifest.get("startup_history_padding_receipt") != result["warmup_control"]["startup_history_padding"]
                or manifest.get("warmup_control_actual_kind") != WARMUP_KINDS[PADDED_WARMUP]):
            raise ValueError("RTX startup padding receipt differs from independently audited physical history")
    if (manifest.get("contract") != "cvpr-handover-teacher-rtx-capture-v1"
            or manifest.get("status") != "raw_capture_completed_pending_independent_audit"
            or manifest.get("accepted_new_dataset") is not False
            or manifest.get("code_unchanged_during_run") is not True):
        raise ValueError("Complete raw RTX capture with frozen code is required, not self-admission")
    if result.get("render_mode") != "isaac" or result.get("rgb_collected") is not True:
        raise ValueError("Camera-free screening is never an accepted dataset")
    if not {"isaacsim.py", "sonic_loco_manip.py", "action.py"} <= {Path(p).name for p in result["capture_code_sha256"]}:
        raise ValueError("Actual renderer, environment and typed-action executable source must be SHA-bound")
    events_path = local_file(capture, manifest["events_path"])
    if sha256(events_path) != manifest["events_sha256"]:
        raise ValueError("Event stream SHA mismatch")
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
    total50 = 50 + result["source_commands"]
    expected_count = (total50 - 1) * 30 // 50 + 1
    if len(events) != expected_count or manifest.get("event_count") != len(events):
        raise ValueError("Complete30Hz event stream including genuine warmup is required")
    tokens, hands, hand_names = history["tokens"], history["hands"], history["hand_names"]
    trajectory_names = history["joint_names"]
    bindings = dict(physics["bindings"])
    bindings.update({str(capture / "capture.json"): sha256(capture / "capture.json"), str(events_path): sha256(events_path)})
    joint_names, pixel_paths, fingerprints = None, set(), []
    world_reference = manifest["isaac_world_time_reference_s"]
    if not isinstance(world_reference, (int, float)) or not math.isfinite(world_reference):
        raise ValueError("Actual nonadvancing Isaac world time reference is required")
    import mujoco
    render_data = mujoco.MjData(history["model"])
    frame_calibration = load_frame_calibration(history["model"], result)
    bindings[str(FRAME_CALIBRATION)] = sha256(FRAME_CALIBRATION)
    bindings.update(calibration_asset_bindings(result, frame_calibration))
    render_maxima = {}
    for role, delta in candidate["initial_object_deltas"].items():
        receipt = result["perturbation_receipt"]["objects"][role]
        source_pose = finite_pose(receipt["source_pose7"], "source reset pose")
        applied_pose = finite_pose(receipt["applied_pose7"], "applied reset pose")
        expected_pose = source_pose.copy()
        expected_pose[:2] += delta["xy_delta_m"]
        c, s = math.cos(delta["world_z_yaw_delta_rad"] / 2), math.sin(delta["world_z_yaw_delta_rad"] / 2)
        w, x, y, z = source_pose[3:]
        expected_pose[3:] = [c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w]
        if (not np.allclose(expected_pose, applied_pose, rtol=0, atol=1e-12)
                or not np.allclose(events[0]["objects"][role]["pose7"], applied_pose, rtol=0, atol=1e-9)):
            raise ValueError("Actual first RGB/state reset is not the preregistered physical XY/yaw perturbation")
    for index, event in enumerate(events):
        tick = (index * 50 + 29) // 30
        phase = "warmup" if tick < 50 else "teacher_command"
        if (event.get("event_index") != index or event.get("capture_index30") != index
                or event.get("physical_tick50") != tick or event.get("phase") != phase
                or event.get("action_valid") is not (phase == "teacher_command")
                or not math.isclose(event["sim_time_s"], history["times"][tick], rel_tol=0, abs_tol=1e-6)
                or not math.isclose(event["actual_elapsed_s"], tick / 50, rel_tol=0, abs_tol=1e-6)
                or not math.isclose(event["nominal_elapsed_s"], index / 30, rel_tol=0, abs_tol=1e-9)):
            raise ValueError(f"Event{index}: deployed30Hz sample-and-hold clock or phase differs")
        names = event["joint_names"]
        if len(names) != 43 or len(set(names)) != 43 or (joint_names is not None and names != joint_names):
            raise ValueError("Every event needs the same43 unique named physical joints")
        joint_names = names
        state = finite_vector(event["state43"], 43, "measured state")
        physical = finite_vector(event["physical_joint_q43_float64"], 43, "physical joints")
        velocity = finite_vector(event["physical_joint_qvel43"], 43, "physical joint velocities")
        if not np.array_equal(state.astype(np.float32), physical.astype(np.float32)):
            raise ValueError("Observation state is not the corresponding measured physical state")
        if set(names) != set(trajectory_names):
            raise ValueError("Captured joint identities differ from serialized physical history")
        perm = [trajectory_names.index(name) for name in names]
        root_pose = finite_pose(event["root_pose7"], "root pose")
        root_velocity = finite_vector(event["root_qvel6"], 6, "root velocity")
        if (not np.array_equal(physical, history["q"][tick, perm])
                or not np.array_equal(velocity, history["dq"][tick, perm])
                or not np.allclose(root_pose, history["root"][tick], rtol=0, atol=1e-12)
                or not np.array_equal(root_velocity, history["root_velocity"][tick])):
            raise ValueError("Captured measured state does not match its actual integration-history tick")
        for role in ("target", "container"):
            if not np.allclose(finite_pose(event["objects"][role]["pose7"], f"{role} pose"),
                               history["objects"][role][tick], rtol=0, atol=1e-12):
                raise ValueError("Captured object pose differs from authoritative integration-history tick")
        mujoco.mj_setState(history["model"], render_data, history["integration_states"][tick], history["state_spec"])
        render_errors = audit_renderer_pose(event, history["model"], render_data,
            world_reference=world_reference, frame_calibration=frame_calibration)
        for key, value in render_errors.items():
            render_maxima[key] = max(render_maxima.get(key, 0.), value)
        action = event.get("action")
        if phase == "warmup":
            if action is not None or event.get("source_command_index") is not None:
                raise ValueError("Warmup has no fabricated teacher/body-token labels")
        else:
            source_index = ((index - 30) * 50 + 29) // 30
            if event.get("source_command_index") != source_index or source_index >= len(tokens):
                raise ValueError("Teacher event references an incorrect source50Hz row")
            names = action["hand_joint_names"]
            if len(names) != 14 or set(names) != set(hand_names):
                raise ValueError("Teacher hand targets must retain their named source semantics")
            perm = [hand_names.index(name) for name in names]
            if (not np.array_equal(finite_vector(action["body_token64"], 64, "body token").astype(np.float32), tokens[source_index])
                    or not np.array_equal(finite_vector(action["hand14"], 14, "hand targets").astype(np.float32), hands[source_index, perm])):
                raise ValueError("Captured action differs from the actual source teacher command")
        rgb = event["rgb"]
        image_path = local_file(capture, rgb["path"])
        if image_path in pixel_paths or image_path.suffix.lower() != ".png" or sha256(image_path) != rgb["file_sha256"]:
            raise ValueError("Every event needs its own unchanged lossless PNG")
        pixel_paths.add(image_path)
        with Image.open(image_path) as image:
            if image.mode != "RGB" or image.size != (rgb["width"], rgb["height"]):
                raise ValueError("RGB PNG dimensions or channels differ from the capture receipt")
            pixels = np.asarray(image)
            pixel_sha = hashlib.sha256(pixels.tobytes()).hexdigest()
        if rgb.get("channels") != 3 or rgb.get("dtype") != "uint8" or pixel_sha != rgb["pixel_sha256"]:
            raise ValueError("Raw RGB pixels differ from the immutable event binding")
        bindings[str(image_path)] = rgb["file_sha256"]
        fingerprints.append({"pixels": pixel_sha, "state": event["state43"], "objects": event["objects"], "action": action})
    render_pass = all(value <= (RENDER_POSITION_TOLERANCE_M if key.endswith("_m") else RENDER_ROTATION_TOLERANCE_RAD)
                      for key, value in render_maxima.items() if not key.startswith("raw_"))
    renderer_report = {"contract": RENDERER_POSE_CONTRACT, "events_checked": len(events),
        "camera": "head_stereo_left", "isaac_world_time_reference_s": world_reference,
        "maxima": render_maxima, "position_tolerance_m": RENDER_POSITION_TOLERANCE_M,
        "rotation_tolerance_degrees": math.degrees(RENDER_ROTATION_TOLERANCE_RAD),
        "within_predeclared_render_pose_tolerances": render_pass,
        "tolerance_scope": "New collection admission convention; not physical/solver accuracy, historical measurement, or proof of image content",
        "frame_mapping": {"contract": frame_calibration.CALIBRATION_CONTRACT,
            "kind": "source-asset waist-yaw/waist-roll joint-chain mapping at actual measured pose; never first-frame fitted",
            "helper_sha256": sha256(FRAME_CALIBRATION), "source_asset_identities": frame_calibration.SOURCE_IDENTITIES,
            "resource_data_root": result["resource_data_root"], "actual_source_assets_sha_verified": True,
            "usd_waist_roll_position_m": frame_calibration.USD_WAIST_ROLL_POSITION.tolist(),
            "usd_torso_position_in_roll_m": frame_calibration.USD_TORSO_POSITION_IN_ROLL.tolist(),
            "usd_torso_to_d435_position_m": frame_calibration.USD_TORSO_TO_D435_POSITION.tolist(),
            "usd_shoulder_pitch_positions_m": {k:v.tolist() for k,v in frame_calibration.USD_SHOULDER_PITCH_POSITIONS.items()},
            "gate": "corresponding frame residuals; raw torso/camera extrinsics remain reported independently",
            "scope": "actual USD joint-state/time synchronization with source frames; not geometrical identity of both assets or a new contact/force observation"},
        "mujoco_pose_source": "mj_setState integration followed only by mj_kinematics/mj_camlight; no synthetic force or dynamics replay"}
    if not render_pass:
        raise CaptureRejected("Actual renderer poses exceed preregistered data-admission tolerances", {
            "contract": AUDIT_CONTRACT, "authority_revision": AUTHORITY_REVISION,
            "status": "independent_capture_rejected", "eligible_dataset": False,
            "candidate_id": candidate["candidate_id"], "candidate_spec_sha256": candidate["candidate_spec_sha256"],
            "plan_sha256": plan_sha256, "source_episode_index": result["source_episode_index"],
            "teacher_collection_warmup": physics["teacher_collection_warmup"],
            "renderer_pose_audit": renderer_report, "bindings": bindings, "source_sha256": result["source_sha256"]})
    return {"contract": AUDIT_CONTRACT, "authority_revision": AUTHORITY_REVISION, "status": "independent_capture_audit_passed",
            "eligible_dataset": True, "candidate_id": candidate["candidate_id"],
            "candidate_spec_sha256": candidate["candidate_spec_sha256"], "plan_sha256": plan_sha256,
            "source_episode_index": result["source_episode_index"], "event_count": len(events),
            "teacher_events": len(events)-30, "warmup_events": 30,
            "teacher_collection_warmup": physics["teacher_collection_warmup"],
            "teacher_reference": physics["teacher_reference"],
            "startup_history_padding_audit": physics["startup_history_padding_audit"],
            "capture_content_sha256": digest(fingerprints), "physical_gate": physics["physical_gate"],
            "renderer_pose_audit": renderer_report,
            "bindings": bindings, "source_sha256": result["source_sha256"],
            "diversity_receipt": {key: result["perturbation_receipt"].get(key) for key in ("material_seed_applied", "generation_seed_applied", "seed_scope")},
            "training_payload_packed": False, "policy_success_rate_claim": None, "generated_utc": now()}


@contextmanager
def queue_lock(root):
    with (Path(root) / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def queue_summary(state):
    accepted = [a for a in state["attempts"] if a["status"] == "accepted"]
    counts = Counter(a["source_episode_index"] for a in accepted)
    valid = (len(accepted) == 100 and len(counts) == 24 and max(counts.values(), default=0) <= 5)
    if len(accepted) > 100 or max(counts.values(), default=0) > 5:
        raise ValueError("Accepted registry exceeds its fixed total/ancestor quotas")
    state.update(status="training_capture_target_accepted" if valid else "incomplete",
                 accepted_count=len(accepted), accepted_per_ancestor={str(k): v for k, v in sorted(counts.items())},
                 collected_count=sum(a.get("rtx_capture_completed", False) for a in state["attempts"]),
                 train_capture_target_satisfied=valid, dataset_pack_complete=False, final_ready=False,
                 final_ready_blockers=["LeRobot_pack_not_independently_accepted", "TEST_lineage_pending"], updated_utc=now())
    return state


def initialize_queue(plan_path, output):
    plan, identity = load_plan(plan_path)
    output = Path(output)
    if (not output.is_absolute() or output.exists() or output.is_symlink()
            or output.resolve().is_relative_to(REPO) or output.resolve().is_relative_to(Path(plan_path).resolve().parent)):
        raise ValueError("Use a fresh external queue directory outside the plan")
    output.mkdir(parents=True)
    state = {"contract": QUEUE_CONTRACT, "authority_revision": AUTHORITY_REVISION, "plan_path": str(Path(plan_path).resolve()),
             "plan_sha256": identity, "attempts": [], "registrations": [], "created_utc": now()}
    write(output / "queue.json", queue_summary(state))
    return state


def verified_queue(root):
    root = Path(root).resolve(strict=True)
    state = read(root / "queue.json")
    if state.get("contract") != QUEUE_CONTRACT or state.get("authority_revision") != AUTHORITY_REVISION:
        raise ValueError("Unknown/stale queue authority; retain old raw outputs and create a fresh queue")
    plan, identity = load_plan(state["plan_path"])
    if identity != state["plan_sha256"]:
        raise ValueError("Frozen collection plan changed")
    if any(a["status"] == "accepted" for a in state["attempts"]) and "recipe" not in state:
        raise ValueError("Accepted registry requires its pre-execution frozen audit recipe")
    if "recipe" in state:
        recipe = state["recipe"]
        required_code = {str(p.relative_to(REPO)) for p in CORE_FILES}
        if (recipe.get("authority_revision") != AUTHORITY_REVISION or recipe.get("audit_contract") != AUDIT_CONTRACT
                or not required_code <= set(recipe["code_sha256"])):
            raise ValueError("Frozen queue lacks the current collector/auditor authority")
        schedule = recipe["warmup_schedule"]
        if load_warmup_schedule(plan, identity, schedule["path"]) != schedule:
            raise ValueError("Frozen TRAIN warmup schedule changed")
        if schedule["path"] is not None:
            archived = local_file(root, "frozen_warmup_schedule.json")
            if sha256(archived) != schedule["sha256"]:
                raise ValueError("Frozen TRAIN warmup schedule snapshot changed")
        recipe_path = local_file(root, "frozen_recipe.json")
        if sha256(recipe_path) != state["frozen_recipe_sha256"] or read(recipe_path) != state["recipe"]:
            raise ValueError("Frozen collector/auditor recipe changed")
        for relative, expected in state["recipe"]["code_sha256"].items():
            if (sha256(local_file(root / "frozen-code", relative)) != expected
                    or sha256(local_file(REPO, relative)) != expected):
                raise ValueError("Frozen executable/acceptance source snapshot changed")
        for path, expected in state["recipe"]["input_receipt_sha256"].items():
            if not Path(path).is_file() or sha256(path) != expected:
                raise ValueError("Frozen source input receipt changed")
    candidates = [(c, state["plan_path"], identity) for c in plan["candidates"]]
    for registration in state["registrations"]:
        path = local_file(root, registration["path"])
        if sha256(path) != registration["sha256"]:
            raise ValueError("Preregistered extension changed")
        extra, extra_sha = load_plan(path)
        if extra["original_ancestors"] != plan["original_ancestors"]:
            raise ValueError("Extension changed the frozen original ancestors")
        candidates.extend((c, str(path), extra_sha) for c in extra["candidates"])
    seen, seen_specs = set(), set()
    for candidate, _, _ in candidates:
        if candidate["candidate_id"] in seen or candidate["candidate_spec_sha256"] in seen_specs:
            raise ValueError("Candidate repeated across immutable registrations")
        seen.add(candidate["candidate_id"])
        seen_specs.add(candidate["candidate_spec_sha256"])
    by_id = {c["candidate_id"]: (c, str(Path(path).resolve()), identity) for c, path, identity in candidates}
    attempted, accepted_content = set(), set()
    for index, attempt in enumerate(state["attempts"]):
        cid = attempt["candidate_id"]
        if cid not in by_id or cid in attempted:
            raise ValueError("Attempt registry repeats or invents a preregistered candidate")
        attempted.add(cid)
        candidate, plan_path, identity = by_id[cid]
        if (attempt.get("directory") != f"attempt-{index:05d}"
                or attempt.get("candidate_spec_sha256") != candidate["candidate_spec_sha256"]
                or attempt.get("plan_sha256") != identity or attempt.get("plan_path") != plan_path
                or attempt.get("source_episode_index") != candidate["ancestor"]["source_episode_index"]):
            raise ValueError("Attempt candidate/plan/source identity differs from frozen registration")
        scheduled = state["recipe"]["warmup_schedule"]["controls"][str(attempt["source_episode_index"])]
        reference = state["recipe"]["warmup_schedule"].get("teacher_references", {}).get(str(attempt["source_episode_index"]))
        if attempt.get("warmup_control") != scheduled:
            raise ValueError("Attempt TRAIN warmup differs from the frozen source schedule")
        if attempt.get("teacher_reference") != reference:
            raise ValueError("Attempt teacher reference differs from the frozen TRAIN schedule")
        if attempt["status"] == "accepted":
            expected_path = f"{attempt['directory']}/independent_capture_audit.json"
            if attempt.get("audit_path") != expected_path or attempt.get("rtx_capture_completed") is not True:
                raise ValueError("Accepted registry lacks its own completed RTX audit")
            audit_path = local_file(root, expected_path)
            audit = read(audit_path)
            if (sha256(audit_path) != attempt["audit_sha256"] or audit.get("contract") != AUDIT_CONTRACT
                    or audit.get("authority_revision") != AUTHORITY_REVISION
                    or audit.get("status") != "independent_capture_audit_passed" or audit.get("eligible_dataset") is not True):
                raise ValueError("Accepted registry audit receipt is missing or changed")
            expected = {"candidate_id": cid, "candidate_spec_sha256": candidate["candidate_spec_sha256"],
                        "plan_sha256": identity, "source_episode_index": attempt["source_episode_index"]}
            if any(audit.get(k) != v for k, v in expected.items()):
                raise ValueError("Accepted audit candidate/plan/source identity differs")
            for mapping in (audit.get("bindings"), audit.get("source_sha256")):
                if not mapping:
                    raise ValueError("Accepted audit payload/source bindings are missing")
                for path, expected_sha in mapping.items():
                    p = Path(path)
                    if not p.is_absolute() or not p.is_file() or sha256(p) != expected_sha:
                        raise ValueError("Accepted audit payload/source is missing or changed")
            # Recompute the authority relation too: a self-written eligible flag,
            # even with an updated receipt hash, cannot authorize queue recovery.
            run = root / attempt["directory"] / "rtx"
            actual = audit_capture(run, run / "capture", candidate, identity, expected_warmup=scheduled,
                                   **({"expected_reference": reference} if reference is not None else {}))
            stable = lambda value: {k: v for k, v in value.items() if k != "generated_utc"}
            if stable(actual) != stable(audit):
                raise ValueError("Accepted audit no longer matches independent physical/pixel/source revalidation")
            content = audit["capture_content_sha256"]
            if content in accepted_content:
                raise ValueError("Accepted registry repeats the same physical/RGB/action capture")
            accepted_content.add(content)
    queue_summary(state)  # Reapply total/family limits after validating identities.
    return state, plan, candidates


def register_extension(root, *, variants_per_ancestor=1, maximum_total_candidates=240):
    root = Path(root).resolve(strict=True)
    if type(variants_per_ancestor) is not int or not 1 <= variants_per_ancestor <= 5:
        raise ValueError("Register1..5 extra proposed variants at a time")
    with queue_lock(root):
        state, base, candidates = verified_queue(root)
        extra_count = 24 * variants_per_ancestor
        if len(candidates) + extra_count > maximum_total_candidates:
            raise ValueError("Preregistered total exceeds the explicit attempt budget")
        start = max(c["variant"] for c, _, _ in candidates) + 1
        extra = copy.deepcopy(base)
        extra["candidates"] = []
        for variant in range(start, start + variants_per_ancestor):
            for ancestor in base["original_ancestors"]:
                n = ancestor["source_episode_index"]
                item = copy.deepcopy(next(c for c in base["candidates"] if c["ancestor"] == ancestor))
                item.update(variant=variant, physics_seed=seed_for(base["seed"], n, variant, "physics"),
                            material_seed=seed_for(base["seed"], n, variant, "material"),
                            generation_seed=seed_for(base["seed"], n, variant, "generation"),
                            initial_object_deltas={role: {"xy_delta_m": [offset(base["seed"], n, variant, f"{role}.{axis}", .01) for axis in ("x", "y")],
                                "world_z_yaw_delta_rad": offset(base["seed"], n, variant, f"{role}.yaw", .05)} for role in ("target", "container")})
                h = digest({k: item[k] for k in SPEC_KEYS})
                item.update(candidate_spec_sha256=h, candidate_id=f"handover-source{n:03d}-variant{variant:02d}-{h[:16]}")
                extra["candidates"].append(item)
        extra.update(candidate_count=extra_count, generated_utc=now(), parent_plan_sha256=state["plan_sha256"],
                     registration_purpose="additional_preregistered_attempts_not_accepted_demonstrations")
        path = root / f"registration-{len(state['registrations']) + 1:04d}.json"
        if path.exists():
            raise FileExistsError(path)
        write(path, extra)
        state["registrations"].append({"path": path.name, "sha256": sha256(path)})
        write(root / "queue.json", queue_summary(state))
        return {"registered_candidates": extra_count, "path": str(path), "sha256": sha256(path), "accepted_count": state["accepted_count"]}


def _process_identity(pid):
    try:
        parts = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if parts[0] == "Z" else parts[19]
    except (OSError, IndexError):
        return None


def execute_phase(command, env, log_path, state, attempt, phase, root, timeout):
    with Path(log_path).open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        attempt.update(status=phase + "_running", process_pid=process.pid, process_birth=_process_identity(process.pid),
                       current_command=command, phase_started_utc=now())
        write(root / "queue.json", queue_summary(state))
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()  # Only this owned local child; no training/remote process.
            process.wait()
            raise RuntimeError(f"Owned {phase} child exceeded its bounded timeout")
        attempt.update(process_pid=None, process_birth=None, last_returncode=returncode)
        if returncode != 0:
            raise RuntimeError(f"{phase} child returned{ returncode}; all partial evidence retained")


def advisory_pool(state, plan, companion_queue):
    """Read a live companion registry for scheduling only, never admission.

    All selected captures must still be independently re-audited by the final
    packer using their own frozen authorities. A hot registry count is not a
    completed dataset or a learned-policy measurement.
    """
    allowed = {row["source_episode_index"] for row in plan["original_ancestors"]}
    companion = Path(companion_queue).resolve(strict=True)
    other = read(companion / "queue.json")
    other_plan, identity = load_plan(other["plan_path"])
    if (identity != other["plan_sha256"] or other_plan["original_ancestors"] != plan["original_ancestors"]
            or other_plan["source_manifest_sha256"] != plan["source_manifest_sha256"]
            or other_plan["selection_sha256"] != plan["selection_sha256"]):
        raise ValueError("Companion scheduling queue differs in original source identity")
    counts, identities = Counter(), set()
    for registry in (state, other):
        for attempt in registry["attempts"]:
            if attempt["status"] != "accepted":
                continue
            source = attempt["source_episode_index"]
            if source not in allowed or attempt["candidate_id"] in identities:
                raise ValueError("Advisory pool contains forbidden ancestry or a repeated candidate")
            identities.add(attempt["candidate_id"]); counts[source] += 1
    capped = Counter({source:min(count,5) for source,count in counts.items()})
    return capped, {"scope":"scheduling_hint_only_pending_final_independent_audit", "companion_queue":str(companion),
        "eligible_by_count_at_most5_per_ancestor":sum(capped.values()),"ancestors_by_count":len(capped),
        "selection_possible_by_count":sum(capped.values()) >= 100 and set(capped) == allowed,
        "dataset_ready":False,"policy_success_rate_claim":None,"observed_utc":now()}


def collect(root, *, snapshot_root, data_root, sonic_resources, python, max_candidates=1, timeout=900, candidate_id=None, warmup_schedule=None, companion_queue=None):
    root = Path(root).resolve(strict=True)
    inputs = {key: str(Path(value).resolve(strict=True)) for key, value in {
        "snapshot_root": snapshot_root, "data_root": data_root, "sonic_resources": sonic_resources, "python": python}.items()}
    if any(root.is_relative_to(Path(v)) for key, v in inputs.items() if key != "python"):
        raise ValueError("Queue outputs must not mutate source or construction resources")
    if type(max_candidates) is not int or max_candidates < 1 or timeout <= 0:
        raise ValueError("Use a positive bounded candidate count and timeout")
    if companion_queue is not None:
        companion_queue = str(Path(companion_queue).resolve(strict=True))
        if Path(companion_queue) == root:
            raise ValueError("A collection queue cannot be its own companion")
    with queue_lock(root):
        state, plan, candidates = verified_queue(root)
        if candidate_id is not None:
            if max_candidates != 1 or not any(c["candidate_id"] == candidate_id for c, _, _ in candidates):
                raise ValueError("Explicit pilot must select exactly one existing preregistered candidate")
            candidates = [item for item in candidates if item[0]["candidate_id"] == candidate_id]
        receipt_files = sorted(Path(inputs["snapshot_root"]).glob("*audit*.json"))
        code_files = (*CORE_FILES, LIVE_WARMUP_SOURCE, REPO / "VLMExpert/third_party/SIMPLE/src/simple/baselines/sonic_v1_1.py",
                      REPO / "VLMExpert/third_party/SIMPLE/src/simple/core/action.py",
                      FRAME_CALIBRATION,
                      REPO / "VLMExpert/third_party/SIMPLE/src/simple/envs/sonic_loco_manip.py",
                      REPO / "VLMExpert/third_party/SIMPLE/src/simple/engines/isaacsim.py",
                      REPO / "VLMExpert/third_party/SIMPLE/src/simple/engines/mujoco.py")
        schedule_path = warmup_schedule
        if schedule_path is None and "recipe" in state:
            schedule_path = state["recipe"]["warmup_schedule"]["path"]
        schedule = load_warmup_schedule(plan, state["plan_sha256"], schedule_path)
        recipe = {"authority_revision": AUTHORITY_REVISION, "audit_contract": AUDIT_CONTRACT,
                  "inputs": inputs, "command_rate": 30, "warmup_schedule": schedule, "warmup_seconds": 1.,
                  "simple_source_root":str(REPO / "VLMExpert/third_party/SIMPLE/src"),
                  "pd_profile": "fixed_cpp_baseline", "hand_torque_profile": "recording_0p7_nm",
                  "code_sha256": {str(p.relative_to(REPO)): sha256(p) for p in code_files},
                  "input_receipt_sha256": {str(p): sha256(p) for p in receipt_files},
                  "source_manifest_sha256": plan["source_manifest_sha256"], "selection_sha256": plan["selection_sha256"]}
        recipe["companion_queue_scheduling_only"] = companion_queue
        if "recipe" in state and state["recipe"] != recipe:
            raise ValueError("Frozen queue recipe/source locations changed; create a separately reviewed queue")
        if "recipe" not in state:
            frozen = root / "frozen-code"
            for path in code_files:
                target = frozen / path.relative_to(REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            write(root / "frozen_recipe.json", recipe)
            state["frozen_recipe_sha256"] = sha256(root / "frozen_recipe.json")
            if schedule["path"] is not None:
                shutil.copyfile(schedule["path"], root / "frozen_warmup_schedule.json")
                if sha256(root / "frozen_warmup_schedule.json") != schedule["sha256"]:
                    raise ValueError("TRAIN warmup schedule changed while freezing")
        state["recipe"] = recipe
        for attempt in state["attempts"]:
            if attempt["status"].endswith("_running"):
                if (attempt.get("process_pid") and attempt.get("process_birth")
                        and _process_identity(attempt["process_pid"]) == attempt["process_birth"]):
                    raise RuntimeError("Prior owned child is still active; do not launch a duplicate")
                attempt.update(status="interrupted_evidence_retained", error="Previous queue owner exited; no output reused or deleted")
        write(root / "queue.json", queue_summary(state))
        tried = {a["candidate_id"] for a in state["attempts"]}
        processed = 0
        for candidate, plan_path, plan_sha in candidates:
            if processed >= max_candidates or state["train_capture_target_satisfied"]:
                break
            n = candidate["ancestor"]["source_episode_index"]
            selected_warmup = schedule["controls"][str(n)]
            reference = schedule.get("teacher_references", {}).get(str(n))
            counts = Counter(a["source_episode_index"] for a in state["attempts"] if a["status"] == "accepted")
            if companion_queue is not None:
                counts, hint = advisory_pool(state, plan, companion_queue)
                state["companion_scheduling_hint"] = hint
                if hint["selection_possible_by_count"]:
                    state["scheduling_stop_reason"] = "combined_count_target_reached_pending_final_independent_audit"
                    write(root / "queue.json", queue_summary(state))
                    break
            if candidate["candidate_id"] in tried or counts[n] >= 5:
                continue
            remaining = 100 - sum(counts.values())
            missing = {a["source_episode_index"] for a in plan["original_ancestors"]} - set(counts)
            if remaining <= len(missing) and n not in missing:
                continue  # Reserve at least one acceptance slot for every ancestor.
            processed += 1
            attempt_dir = root / f"attempt-{len(state['attempts']):05d}"
            attempt_dir.mkdir(exist_ok=False)
            attempt = {"candidate_id": candidate["candidate_id"], "candidate_spec_sha256": candidate["candidate_spec_sha256"],
                       "plan_path": plan_path, "plan_sha256": plan_sha, "source_episode_index": n,
                       "warmup_control": selected_warmup,
                       "teacher_reference": reference,
                       "status": "registered", "directory": attempt_dir.name, "created_utc": now()}
            state["attempts"].append(attempt)
            write(root / "queue.json", queue_summary(state))
            common = [inputs["python"], str(SCRIPT), "--snapshot-root", inputs["snapshot_root"], "--data-root", inputs["data_root"],
                "--sonic-resources", inputs["sonic_resources"], "--candidate-plan", str(plan_path), "--candidate-id", candidate["candidate_id"],
                "--command-rate", "30", "--warmup-control", selected_warmup, "--warmup-seconds", "1", "--pd-profile", "fixed_cpp_baseline",
                "--hand-torque-profile", "recording_0p7_nm"]
            if reference is not None:
                common.extend(["--teacher-reference", reference["path"]])
            phase = "screen"
            try:
                if load_warmup_schedule(plan, state["plan_sha256"], schedule["path"]) != schedule:
                    raise ValueError("TRAIN warmup schedule changed before candidate execution")
                env = dict(os.environ)
                env["PYTHONPATH"] = recipe["simple_source_root"] + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
                env["CUDA_VISIBLE_DEVICES"] = ""
                screen = attempt_dir / "screen"
                execute_phase([*common, "--render-mode", "none", "--output", str(screen)], env,
                              attempt_dir / "screen.log", state, attempt, phase, root, timeout)
                preflight = audit_physics(screen, candidate, plan_sha, expected_warmup=selected_warmup,
                                          expected_reference=reference)
                write(attempt_dir / "preflight_audit.json", preflight)
                attempt["preflight_audit_sha256"] = sha256(attempt_dir / "preflight_audit.json")
                phase = "capture"
                rendered, capture = attempt_dir / "rtx", attempt_dir / "rtx/capture"
                capture_env = dict(env)
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    capture_env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
                else:
                    capture_env.pop("CUDA_VISIBLE_DEVICES", None)
                execute_phase([*common, "--render-mode", "isaac", "--capture-training", str(capture), "--output", str(rendered)],
                              capture_env, attempt_dir / "capture.log", state, attempt, phase, root, timeout)
                attempt["rtx_capture_completed"] = True
                audit = audit_capture(rendered, capture, candidate, plan_sha, expected_warmup=selected_warmup,
                                      expected_reference=reference)
                if load_warmup_schedule(plan, state["plan_sha256"], schedule["path"]) != schedule:
                    raise ValueError("TRAIN warmup schedule changed during candidate execution")
                previous = [read(root / a["audit_path"]) for a in state["attempts"] if a["status"] == "accepted"]
                if audit["capture_content_sha256"] in {a["capture_content_sha256"] for a in previous}:
                    raise ValueError("Duplicated complete physical/RGB/action capture is not a new demonstration")
                audit_path = attempt_dir / "independent_capture_audit.json"
                write(audit_path, audit)
                attempt.update(status="accepted", audit_path=str(audit_path.relative_to(root)), audit_sha256=sha256(audit_path))
            except Exception as error:
                attempt.update(status=phase + "_rejected", error=repr(error))
                if isinstance(error, CaptureRejected):
                    rejection = attempt_dir / "independent_rejection.json"
                    write(rejection, error.receipt)
                    attempt.update(rejection_path=str(rejection.relative_to(root)), rejection_sha256=sha256(rejection))
            attempt.update(ended_utc=now(), process_pid=None, process_birth=None)
            write(root / "queue.json", queue_summary(state))
        return queue_summary(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    init = commands.add_parser("init")
    init.add_argument("--plan", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    status = commands.add_parser("status")
    status.add_argument("--queue", type=Path, required=True)
    extend = commands.add_parser("extend")
    extend.add_argument("--queue", type=Path, required=True)
    extend.add_argument("--variants-per-ancestor", type=int, default=1)
    extend.add_argument("--maximum-total-candidates", type=int, default=240)
    audit = commands.add_parser("audit-capture")
    audit.add_argument("--plan", type=Path, required=True)
    audit.add_argument("--candidate-id", required=True)
    audit.add_argument("--run-dir", type=Path, required=True)
    audit.add_argument("--capture-dir", type=Path, required=True)
    audit.add_argument("--warmup-schedule", type=Path, help="Predeclared TRAIN-only source controls; absent means all measured")
    run = commands.add_parser("collect")
    run.add_argument("--queue", type=Path, required=True)
    for name in ("snapshot-root", "data-root", "sonic-resources", "python"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--max-candidates", type=int, default=1)
    run.add_argument("--timeout", type=float, default=900)
    run.add_argument("--candidate-id", help="Select one existing preregistered pilot; never alters its proposal or retries an old attempt")
    run.add_argument("--warmup-schedule", type=Path, help="Freeze the complete24-source TRAIN-only warmup schedule before any child executes")
    run.add_argument("--companion-queue", type=Path, help="Read another queue only to avoid excess collection; final independent admission is still required")
    args = parser.parse_args()
    if args.mode == "init":
        result = initialize_queue(args.plan, args.output)
    elif args.mode == "status":
        result = queue_summary(verified_queue(args.queue)[0])
    elif args.mode == "extend":
        result = register_extension(args.queue, variants_per_ancestor=args.variants_per_ancestor,
                                    maximum_total_candidates=args.maximum_total_candidates)
    elif args.mode == "audit-capture":
        plan, identity = load_plan(args.plan)
        candidate = next(c for c in plan["candidates"] if c["candidate_id"] == args.candidate_id)
        schedule = load_warmup_schedule(plan, identity, args.warmup_schedule)
        result = audit_capture(args.run_dir, args.capture_dir, candidate, identity,
            expected_warmup=schedule["controls"][str(candidate["ancestor"]["source_episode_index"])],
            expected_reference=schedule.get("teacher_references", {}).get(str(candidate["ancestor"]["source_episode_index"])))
    else:
        result = collect(args.queue, snapshot_root=args.snapshot_root, data_root=args.data_root,
            sonic_resources=args.sonic_resources, python=args.python, max_candidates=args.max_candidates,
            timeout=args.timeout, candidate_id=args.candidate_id, warmup_schedule=args.warmup_schedule,
            companion_queue=args.companion_queue)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
