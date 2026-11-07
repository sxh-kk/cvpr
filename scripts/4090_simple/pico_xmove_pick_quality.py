"""Target-only evidence for the explicit SONIC adaptation of SIMPLE XMovePick.

SIMPLE's original reward/check_success methods provide the per-step native
flag. This module never calls them again or introduces a second success rule.
Recording integrity is separate from native task success and dataset admission.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral, Real

from handover_contact_quality import SOLVED_STATE_STAGE


TASK_ID = "cvpr/G1SonicXMovePickPico-v0"
SOURCE_OFFICIAL_TASK_ID = "simple/G1WholebodyXMovePickTeleop-v0"
OFFICIAL_SOURCE_SHA256 = "aae4858775cdbcbddf31a3fe432e80d91519a316b0068e93b7024e1aafad6735"
GATE_VERSION = "xmove_pick-native-task-success-v1"
ADAPTER_CONTRACT = "cvpr-pico-xmove-pick-sonic-controller-adapter-v1"
FRAME_KEY = "pico_xmove_pick_adaptation"
SOURCE_TASK_ID = SOURCE_OFFICIAL_TASK_ID
XMOVE_PICK_CONTRACT = "cvpr-live-pico-xmove-pick-deferred-physical-capture-v1"


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    return float(value)


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_xmove_pick_task_identity(metadata):
    """Return explicit native-source/adapter bindings, never installed class identity.

    The archived official file SHA binds both original compute_reward and
    check_success. The independent adapter SHA is matched against the actual
    session source inventory. The unchanged world frame has an explicit receipt.
    """
    errors = []
    if not isinstance(metadata, Mapping):
        return {"passed": False, "errors": ["XMovePick metadata must be a mapping"]}
    for key, expected in (("task_id", TASK_ID),
                          ("source_official_task_id", SOURCE_OFFICIAL_TASK_ID),
                          ("actual_task_source_sha256", OFFICIAL_SOURCE_SHA256),
                          ("evidence_object_roles", ["target"])):
        if metadata.get(key) != expected:
            errors.append("XMovePick " + key + " differs from the explicit audited identity")
    criteria = metadata.get("actual_native_success_criteria")
    if type(criteria) not in (int, float) or criteria != .9:
        errors.append("XMovePick actual_native_success_criteria must be the original0.9")
    adapter = metadata.get("xmove_pick_adapter_identity")
    adapter = adapter if isinstance(adapter, Mapping) else {}
    for key, expected in (("contract", ADAPTER_CONTRACT), ("task_id", TASK_ID),
                          ("source_official_task_id", SOURCE_OFFICIAL_TASK_ID)):
        if adapter.get(key) != expected:
            errors.append("XMovePick adapter " + key + " identity differs")
    if adapter.get("official_task_source_sha256") != OFFICIAL_SOURCE_SHA256:
        errors.append("XMovePick adapter is not bound to original reward/check_success source SHA")
    source = metadata.get("source_code_sha256_at_session_start")
    source = source if isinstance(source, Mapping) else {}
    adapter_sha = adapter.get("adapter_sha256")
    if not _digest(adapter_sha) or source.get("pico_xmove_pick_task.py") != adapter_sha:
        errors.append("XMovePick adapter SHA differs from the captured session source inventory")
    frame = metadata.get("xmove_pick_adaptation")
    if not isinstance(frame, Mapping) or not frame:
        errors.append("XMovePick measured world-frame adaptation receipt is missing")
    else:
        for key in ("contract", "task_id", "source_official_task_id", "official_task_source_sha256", "adapter_sha256"):
            if frame.get(key) != adapter.get(key):
                errors.append("XMovePick frame receipt differs from adapter identity: " + key)
        expected_frame = {"world_translation_m": [0., 0., 0.],
            "source_robot_uid": "g1_sonic", "robot_uid": "g1_sonic",
            "source_physics_dt": .005, "physics_dt": .005, "render_hz": 50,
            "native_reward_unmodified": True, "robot_physics_equivalent": False,
            "coordinate_transform_applied": False,
            "adapted_payload_scope": ["uid", "metadata", "robot_cfg", "sensor_cfgs", "dr_cfgs", "dr_state_dict"]}
        for key, expected in expected_frame.items():
            value = frame.get(key)
            if value != expected or (type(expected) is bool and type(value) is not bool):
                errors.append("XMovePick adaptation receipt field differs: " + key)
        for key in ("source_config_sha256", "adapted_payload_sha256"):
            if not _digest(frame.get(key)):
                errors.append("XMovePick adaptation receipt lacks payload SHA: " + key)
        environment = metadata.get("environment_config")
        if not isinstance(environment, Mapping) or environment.get(FRAME_KEY) != frame:
            errors.append("XMovePick frame receipt differs from the actual saved environment state marker")
        elif (type(frame.get("source_material_state_missing")) is not bool
                or frame["source_material_state_missing"] != (environment.get("dr_state_dict", {}).get("material") is None)
                or frame.get("material_restoration") != "saved table/object layout materials only; unsaved ground/robot use backend defaults"):
            errors.append("XMovePick omitted material provenance differs from the explicit saved-layout restoration policy")
    return {"passed": not errors, "errors": errors,
        "expected_task_source_sha256": OFFICIAL_SOURCE_SHA256,
        "recorded_task_source_sha256": metadata.get("actual_task_source_sha256"),
        "source_official_task_id": SOURCE_OFFICIAL_TASK_ID,
        "adapter_sha256": adapter_sha,
        "expected_success_criteria": .9, "recorded_success_criteria": criteria,
        "evidence_object_roles": ["target"],
        "identity_scope": "original XMovePick reward/check_success source plus explicit SONIC controller identity; world coordinates unchanged; no archived controller equivalence claim"}


def sample_xmove_pick_solved_quality(model, data, *, completed_steps, target_body, solver_stage):
    """Copy the actual solved target state; do not forward, integrate or infer.

    Caller adds official_task_success, native_task_reward and
    native_baseline_target_z_m from the same env.step return/task state. The
    independent original 200 Hz applied-force stream remains mandatory.
    """
    import numpy as np
    if solver_stage != SOLVED_STATE_STAGE:
        raise ValueError("XMovePick solved evidence must precede observation refresh")
    if isinstance(completed_steps, bool) or not isinstance(completed_steps, Integral) or completed_steps < 1:
        raise ValueError("XMovePick completed_steps must be a positive integer")
    if not isinstance(target_body, (str, Integral)) or isinstance(target_body, bool):
        raise ValueError("XMovePick target body must be its actual name or integral body id")
    target = model.body(target_body).id if isinstance(target_body, str) else int(target_body)
    if not 0 < target < model.nbody:
        raise ValueError("XMovePick requires an actual target body")
    xyz = np.asarray(data.xpos[target], dtype=np.float64)
    quat = np.asarray(data.xquat[target], dtype=np.float64)
    if (xyz.shape != (3,) or quat.shape != (4,) or not np.isfinite(xyz).all()
            or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1., rtol=0, atol=1e-6)):
        raise ValueError("XMovePick actual solved target pose must be finite and unit-quaternion")
    return {"completed_steps": int(completed_steps), "sim_time_s": _finite(data.time, "sim_time_s"),
        "solver_stage": SOLVED_STATE_STAGE, "actual_target_xyz_m": xyz.tolist(),
        "target_pose7": np.r_[xyz, quat].tolist(), "target_body_name": model.body(target).name}


def evaluate_xmove_pick_capture(samples, *, initial_target_z_m=None):
    """Select any real native success with complete dense solved evidence.

    Height and the task's reward baseline are retained and validated as finite
    measurements. Neither is used to add lift/contact/hold/final-frame gates.
    ``initial_target_z_m`` is compatibility-only and never replaces the native
    first-reward baseline. Full200Hz force/SHA/state checks belong to History.
    """
    rows, errors, native = list(samples), [], []
    rewards, baselines, heights = [], [], []
    if not rows:
        errors.append("no measured samples")
    previous_time = None
    for index, row in enumerate(rows):
        prefix = f"sample{index}: "
        if not isinstance(row, Mapping):
            errors.append(prefix + "not a mapping")
            continue
        if row.get("solver_stage") != SOLVED_STATE_STAGE:
            errors.append(prefix + "original solved acquisition stage missing")
        if type(row.get("completed_steps")) is not int or row["completed_steps"] != index + 1:
            errors.append(prefix + "expected dense completed_steps starting1")
        try:
            clock = _finite(row.get("sim_time_s"), "sim_time_s")
            if previous_time is not None and not math.isclose(clock - previous_time, .02, rel_tol=0, abs_tol=1e-6):
                errors.append(prefix + "real50Hz acquisition clock differs")
            previous_time = clock
        except ValueError as error:
            errors.append(prefix + str(error))
        if type(row.get("official_task_success")) is not bool:
            errors.append(prefix + "original boolean official_task_success missing or invalid")
        else:
            native.append(row["official_task_success"])
        try:
            xyz = row.get("actual_target_xyz_m")
            if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
                raise ValueError("actual_target_xyz_m must contain three measured coordinates")
            xyz = [_finite(value, "actual_target_xyz_m") for value in xyz]
            heights.append(xyz[2])
            rewards.append(_finite(row.get("native_task_reward"), "native_task_reward"))
            baselines.append(_finite(row.get("native_baseline_target_z_m"), "native_baseline_target_z_m"))
            if baselines and not math.isclose(baselines[-1], baselines[0], rel_tol=0, abs_tol=1e-12):
                errors.append(prefix + "original native reward baseline changed within the episode")
        except (ValueError, TypeError) as error:
            errors.append(prefix + str(error))
    complete = not errors
    success = any(native) if complete else None
    selected = success is True
    return {"schema": "cvpr-xmove_pick-native-task-success-capture-quality-v1",
        "acceptance_gate_version": GATE_VERSION, "task_id": TASK_ID,
        "source_official_task_id": SOURCE_OFFICIAL_TASK_ID,
        "measurement_hz": 50, "selection_policy": "native_only",
        "selection_status": "technical_evidence_incomplete" if not complete else
                            "selected" if selected else "official_task_not_successful",
        "eligible_physics": selected, "verified_xmove_pick_success": selected,
        "native_success_evidence_complete": complete, "capture_validation_errors": errors,
        "official_task_success": success, "official_task_success_observed": any(native) if native else None,
        "official_task_success_measured_frames": len(native), "official_task_success_expected_frames": len(rows),
        "official_task_flag_modified": False, "native_success_used_as_physical_gate": True,
        "native_reward_baseline_target_z_m": baselines[0] if baselines else None,
        "measured_target_z_range_m": [min(heights), max(heights)] if heights else None,
        "measured_native_reward_range": [min(rewards), max(rewards)] if rewards else None,
        "dataset_ready": False,
        "scope": "Captured original XMovePick native success on the explicit SONIC adaptation; target-only evidence, no extra contact, hold, lift or final-state success gate; recording integrity and independent dataset admission remain separate"}
