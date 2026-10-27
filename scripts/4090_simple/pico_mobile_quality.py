"""Select Mobile P&P captures using the original SIMPLE task success flag.

The flag is recorded after each original 50 Hz solved step. Selection does not
recompute or add requirements to SIMPLE's task logic. Capture/SHA/zero-force
checks remain with EpisodeHistory; this report alone is not dataset admission.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

from handover_contact_quality import SOLVED_STATE_STAGE


MOBILE_TASK_ID = "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
GATE_VERSION = "mobile-native-task-success-v2"


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    return float(value)


def evaluate_mobile_capture(samples, *, initial_target_z_m=None):
    """Require complete original evidence, then select if any native flag is true.

    ``initial_target_z_m`` is accepted for compatibility and does not affect
    selection. SIMPLE's check_success calls its own compute_reward; the original
    recorded result is authoritative, including its contact and reward semantics.
    Missing or invalid acquisition evidence is a technical failure, so the
    verified ``official_task_success`` remains unknown in that case.
    """
    rows = list(samples)
    errors = []
    native = []
    if not rows:
        errors.append("no measured samples")
    previous_time = None
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"sample{index}: not a mapping")
            continue
        if row.get("solver_stage") != SOLVED_STATE_STAGE:
            errors.append(f"sample{index}: original solved force acquisition stage missing")
        if type(row.get("completed_steps")) is not int or row["completed_steps"] != index + 1:
            errors.append(f"sample{index}: expected dense completed_steps starting1")
        try:
            current = _finite(row.get("sim_time_s"), "sim_time_s")
        except ValueError as error:
            errors.append(f"sample{index}: {error}")
        else:
            if previous_time is not None and not math.isclose(current - previous_time, .02, rel_tol=0, abs_tol=1e-6):
                errors.append(f"sample{index}: real50Hz acquisition clock differs")
            previous_time = current
        if type(row.get("official_task_success")) is not bool:
            errors.append(f"sample{index}: original boolean official_task_success missing or invalid")
        else:
            native.append(row["official_task_success"])

    evidence_complete = not errors
    official_success = any(native) if evidence_complete else None
    passed = official_success is True
    status = ("technical_evidence_incomplete" if not evidence_complete else
              "selected" if passed else "official_task_not_successful")
    return {
        "schema": "cvpr-mobile-native-task-success-capture-quality-v2",
        "acceptance_gate_version": GATE_VERSION,
        "task_id": MOBILE_TASK_ID,
        "measurement_hz": 50,
        "selection_policy": "native_only",
        "selection_status": status,
        "verified_mobile_pick_place_success": passed,
        "eligible_physics": passed,
        "capture_validation_errors": errors,
        "native_success_evidence_complete": evidence_complete,
        "official_task_success": official_success,
        "official_task_success_observed": any(native) if native else None,
        "official_task_success_measured_frames": len(native),
        "official_task_success_expected_frames": len(rows),
        "official_task_flag_modified": False,
        "native_success_used_as_physical_gate": True,
        "dataset_ready": False,
        "scope": "Mobile P&P task selection uses any original per-step SIMPLE native success flag with complete 50 Hz solved evidence; no additional lift, transport, grasp, release, geometry, support, stability or fall success gates; full dataset admission remains separate",
    }
