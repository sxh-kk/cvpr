"""Additional physical precheck for human Mobile Pick & Place captures.

These collection gates are deliberately independent of SIMPLE's native task
success and of the right/both/left contact sequence used by Handover. Original
50 Hz solved samples are read only; capture/SHA/zero-force checks remain with
EpisodeHistory and full independent dataset admission is still required.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

from handover_contact_quality import SOLVED_STATE_STAGE
from handover_geometry_quality import final_placement_geometric_validation


MOBILE_TASK_ID = "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
GATE_VERSION = "mobile-pick-place-physics-v1"


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    return float(value)


def _xyz(row, key):
    value = row.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(key + " requires three measured world coordinates")
    return tuple(_finite(component, key) for component in value)


def _contacts(row):
    values = row.get("contacts")
    if not isinstance(values, Mapping):
        raise ValueError("measured contacts missing")
    result = []
    for key in ("left_hand", "right_hand", "container"):
        if type(values.get(key)) is not bool:
            raise ValueError("measured boolean contact missing: " + key)
        result.append(values[key])
    return tuple(result)


def _contact_evidence(rows, initial_target_z_m):
    initial_z = _finite(initial_target_z_m, "initial_target_z_m")
    target, root, pelvis, contacts = [], [], [], []
    for row in rows:
        target.append(_xyz(row, "actual_target_xyz_m"))
        root.append(_xyz(row, "actual_root_xyz_m"))
        pelvis.append(_finite(row.get("pelvis_z_m"), "pelvis_z_m"))
        contacts.append(_contacts(row))

    # A single hand, either hand, or both hands is legal. One continuous
    # lifted/held interval must move target AND root by 0.75 m in horizontal
    # world coordinates. Walking separately from an object cannot establish it.
    carrying = None
    start = None
    longest_lifted_hold = 0
    maximum_held_lift = None
    maximum_target_move = maximum_root_move = 0.0
    for index, (xyz, (left, right, container)) in enumerate(zip(target, contacts)):
        lift = xyz[2] - initial_z
        if left or right:
            maximum_held_lift = lift if maximum_held_lift is None else max(maximum_held_lift, lift)
        held_lifted = (left or right) and not container and lift >= .1
        if not held_lifted:
            start = None
            continue
        if start is None:
            start = index
        count = index - start + 1
        longest_lifted_hold = max(longest_lifted_hold, count)
        target_move = math.dist(target[start][:2], xyz[:2])
        root_move = math.dist(root[start][:2], root[index][:2])
        maximum_target_move = max(maximum_target_move, target_move)
        maximum_root_move = max(maximum_root_move, root_move)
        if carrying is None and count >= 5 and target_move >= .75 and root_move >= .75:
            carrying = {"start_step": start + 1, "end_step": index + 1,
                        "frame_count": count, "target_horizontal_displacement_m": target_move,
                        "root_horizontal_displacement_m": root_move}

    final_positions = target[-50:]
    final_contacts = contacts[-50:]
    full_final = len(final_positions) == 50
    release_count = sum(not left and not right for left, right, _ in final_contacts)
    container_count = sum(container for _, _, container in final_contacts)
    diameter = max((math.dist(a, b) for index, a in enumerate(final_positions)
                    for b in final_positions[index + 1:]), default=None)
    final_transport = math.dist(target[0][:2], target[-1][:2]) if target else None
    checks = {
        "contact_data_complete": bool(rows),
        "root_position_evidence_complete": bool(rows),
        "object_lifted": maximum_held_lift is not None and maximum_held_lift >= .1,
        "lifted_hand_hold_sustained": longest_lifted_hold >= 5,
        "target_and_root_carried_075m": carrying is not None,
        "target_final_transport_075m": final_transport is not None and final_transport >= .75,
        "final_window_complete": full_final,
        "both_hands_released_final_window": full_final and release_count == 50,
        "final_container_contact_sustained": full_final and container_count >= 49,
        "stability_window_complete": full_final,
        "final_object_stable": full_final and diameter is not None and diameter < .005,
        "pelvis_above_fall_threshold": bool(pelvis) and min(pelvis) >= .4,
    }
    return {
        "gate_version": "mobile-lift-carry-release-contact-v1",
        "strict_mobile_pick_place_success": all(checks.values()),
        "contact_placement_success": all(checks[key] for key in (
            "final_window_complete", "both_hands_released_final_window",
            "final_container_contact_sustained", "stability_window_complete", "final_object_stable")),
        "checks": checks, "sample_count": len(rows), "contact_measurement_frames": len(rows),
        "unknown_contact_measurement_frames": 0,
        "lift": {"initial_target_z_m": initial_z, "maximum_held_lift_m": maximum_held_lift,
                 "longest_continuous_lifted_hold_frames": longest_lifted_hold},
        "transport": {"qualifying_carried_interval": carrying,
                      "maximum_target_held_horizontal_displacement_m": maximum_target_move,
                      "maximum_root_held_horizontal_displacement_m": maximum_root_move,
                      "final_target_horizontal_displacement_m": final_transport,
                      "distance_definition": "horizontal net displacement, not accumulated path length"},
        "final_placement": {"hands_released_frames": release_count,
                            "container_contact_frames": container_count,
                            "maximum_pairwise_displacement_m": diameter},
        "minimum_pelvis_z_m": min(pelvis) if pelvis else None,
        "thresholds": {"fps": 50, "minimum_held_lift_m_inclusive": .1,
                       "minimum_lifted_hold_frames": 5, "minimum_carried_horizontal_displacement_m_inclusive": .75,
                       "minimum_final_target_horizontal_displacement_m_inclusive": .75,
                       "final_window_frames": 50, "minimum_final_container_contact_frames": 49,
                       "maximum_pairwise_displacement_m_exclusive": .005, "minimum_pelvis_z_m_inclusive": .4},
        "handover_contact_order_required": False,
        "official_task_flag_modified": False,
        "evidence_limits": ["The 0.75 m transport threshold is an additional collection gate, not an official benchmark rule.",
                            "Measured hand contacts and lift do not certify grasp forces or exact inner-cavity containment."],
    }


def evaluate_mobile_capture(samples, *, initial_target_z_m):
    """Recompute task-specific physical gates; never trust native success alone."""
    rows = list(samples)
    errors = []
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
    contact = geometry = None
    try:
        contact = _contact_evidence(rows, initial_target_z_m)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        errors.append("contact/mobile transport evidence: " + str(error))
    try:
        geometry = final_placement_geometric_validation(rows, window_frames=50, require_3d_evidence=True)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        errors.append("geometry evidence: " + str(error))
    native = [row["official_task_success"] for row in rows
              if isinstance(row, Mapping) and type(row.get("official_task_success")) is bool]
    passed = not errors and bool(contact and contact["strict_mobile_pick_place_success"]) and bool(geometry and geometry["passed"])
    return {"schema": "cvpr-mobile-pick-place-physics-capture-quality-v1",
            "acceptance_gate_version": GATE_VERSION, "task_id": MOBILE_TASK_ID, "measurement_hz": 50,
            "verified_mobile_pick_place_success": passed, "eligible_physics": passed,
            "contact_evidence": contact, "final_placement_geometric_validation": geometry,
            "capture_validation_errors": errors,
            "official_task_success_observed": any(native) if native else None,
            "official_task_success_measured_frames": len(native),
            "official_task_flag_modified": False, "native_success_used_as_physical_gate": False,
            "dataset_ready": False,
            "scope": "Additional human Mobile P&P collection gates: measured lift and same-interval target/root 0.75 m carry plus unchanged placement-necessary-geometry-v2; not official benchmark success or full dataset admission"}
