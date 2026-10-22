# Portions derived from SIMPLE, Copyright (c) 2025 Songlin Wei and Contributors.
# Source snapshots are retained without changing their numerical acceptance rules.
"""Read-only handover evidence from recorded physical replay measurements.

This diagnostic is separate from the simulator's official task success flag.
Contact history cannot establish all object-inside-container geometry or prove
that a contact supplied the forces supporting an object.
"""

from __future__ import annotations

SOURCE_SHA256 = '5af226bcff435531d5fd0596380a452ddf47adb58d4db4cf2405de0d146cc0dc'

from collections.abc import Iterable, Mapping
import math
from numbers import Integral
from typing import Any


SOLVED_STATE_STAGE = "post_mj_step_before_observation_refresh"


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"handover {name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"handover {name} must be finite")
    return result


def _contact(contacts: Mapping[str, Any], *aliases: str) -> tuple[bool, bool]:
    values = [contacts[key] for key in aliases if key in contacts]
    if not values:
        return False, False
    if any(not isinstance(value, bool) for value in values):
        raise ValueError(f"handover contact {aliases[0]} must be boolean")
    if any(value != values[0] for value in values):
        raise ValueError(f"handover contact aliases {aliases} disagree")
    return values[0], True


def summarize_handover_evidence(
    samples: Iterable[Mapping[str, Any]],
    *,
    fps: float = 50.0,
    initial_target_z_m: float | None = None,
    minimum_lift_m: float = 0.1,
    minimum_pelvis_z_m: float = 0.4,
    minimum_stage_frames: int = 5,
    minimum_stage_lift_m: float = 0.05,
    final_window_frames: int = 50,
    minimum_final_container_contact_frames: int = 49,
    stability_window_s: float = 1.0,
    maximum_final_displacement_m: float = 0.005,
) -> dict[str, Any]:
    """Check ordered right-only, both-hands, left-only, released placement.

    Positions are world XYZ in metres. Samples must have consecutive positive
    ``completed_steps``, ``actual_target_xyz_m`` and ``pelvis_z_m``. Contacts
    accept ``left_hand``/``right_hand`` or ``left``/``right``, plus ``container``.
    Missing contact measurements are reported as unknown and cannot pass.

    The final placement window requires both hands off on every frame and at
    least 49/50 container-contact frames by default. Stability is the maximum
    *pairwise* distance over the last ceil(fps * stability_window_s) positions;
    endpoints returning to the same place cannot hide intermediate movement.
    A one-second 50 Hz window contains 50 observations, spanning 0.98 seconds
    between its first and last samples. Stage duration is reported in frames
    and nominal sampled seconds. Each held-object stage must last at least
    five consecutive frames with target height at least 5 cm above its initial
    height. Mere table-level touches do not establish a lifted handover.

    No sample is mutated, and no official task flag is read or changed.
    """
    fps = _finite(fps, "fps")
    stability_window_s = _finite(stability_window_s, "stability_window_s")
    minimum_lift_m = _finite(minimum_lift_m, "minimum_lift_m")
    minimum_stage_lift_m = _finite(minimum_stage_lift_m, "minimum_stage_lift_m")
    minimum_pelvis_z_m = _finite(minimum_pelvis_z_m, "minimum_pelvis_z_m")
    maximum_final_displacement_m = _finite(maximum_final_displacement_m, "maximum_final_displacement_m")
    if fps <= 0 or stability_window_s <= 0 or maximum_final_displacement_m <= 0:
        raise ValueError("handover fps, stability window and displacement threshold must be positive")
    if minimum_lift_m < 0 or minimum_stage_lift_m < 0:
        raise ValueError("handover lift threshold must be nonnegative")
    for name, value in (("minimum_stage_frames", minimum_stage_frames), ("final_window_frames", final_window_frames)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"handover {name} must be a positive integer")
    if (isinstance(minimum_final_container_contact_frames, bool)
            or not isinstance(minimum_final_container_contact_frames, Integral)
            or not 0 <= minimum_final_container_contact_frames <= final_window_frames):
        raise ValueError("handover final contact count must be an integer within the final window")

    rows = list(samples)
    positions: list[tuple[float, ...]] = []
    pelvis: list[float] = []
    steps: list[int] = []
    contacts_by_frame: list[tuple[bool, bool, bool, bool]] = []
    sim_times: list[float | None] = []
    for row in rows:
        step = row.get("completed_steps")
        if isinstance(step, bool) or not isinstance(step, Integral) or step < 1:
            raise ValueError("handover completed_steps must be positive integers")
        if steps and step != steps[-1] + 1:
            raise ValueError("handover samples must have consecutive completed_steps")
        try:
            xyz = tuple(_finite(v, "target position") for v in row["actual_target_xyz_m"])
            z = _finite(row["pelvis_z_m"], "pelvis_z_m")
        except (KeyError, TypeError) as error:
            raise ValueError("handover samples require target XYZ and pelvis height") from error
        if len(xyz) != 3:
            raise ValueError("handover actual_target_xyz_m must contain three coordinates")
        contact_values = row.get("contacts", {})
        if contact_values is None:
            contact_values = {}
        if not isinstance(contact_values, Mapping):
            raise ValueError("handover contacts must be a mapping")
        left, known_left = _contact(contact_values, "left_hand", "left")
        right, known_right = _contact(contact_values, "right_hand", "right")
        container, known_container = _contact(contact_values, "container")
        contacts_by_frame.append((left, right, container, known_left and known_right and known_container))
        positions.append(xyz)
        pelvis.append(z)
        steps.append(int(step))
        sim_times.append(_finite(row["sim_time_s"], "sim_time_s") if "sim_time_s" in row else None)

    initial_z = (_finite(initial_target_z_m, "initial_target_z_m") if initial_target_z_m is not None
                 else positions[0][2] if positions else None)

    def interval(start: int, end: int) -> dict[str, Any]:
        result: dict[str, Any] = {
            "start_step": steps[start], "end_step": steps[end],
            "frame_count": end - start + 1,
            "start_time_s": steps[start] / fps, "end_time_s": steps[end] / fps,
            "nominal_duration_s": (end - start + 1) / fps,
            "sample_span_s": (end - start) / fps,
        }
        if initial_z is not None:
            result["maximum_lift_m"] = max(p[2] for p in positions[start:end + 1]) - initial_z
        if sim_times[start] is not None and sim_times[end] is not None:
            result.update(start_sim_time_s=sim_times[start], end_sim_time_s=sim_times[end])
        return result

    def runs(predicate, *, lifted=False) -> list[dict[str, Any]]:
        result = []
        start = None
        for index in range(len(rows) + 1):
            active = index < len(rows) and predicate(contacts_by_frame[index])
            if active and lifted:
                active = initial_z is not None and positions[index][2] - initial_z >= minimum_stage_lift_m
            if active and start is None:
                start = index
            elif not active and start is not None:
                if index - start >= minimum_stage_frames:
                    result.append(interval(start, index - 1))
                start = None
        return result

    stage_runs = {
        "right_only": runs(lambda c: c[3] and c[1] and not c[0]),
        "both_hands": runs(lambda c: c[3] and c[0] and c[1]),
        "left_only": runs(lambda c: c[3] and c[0] and not c[1]),
    }
    sequence: dict[str, dict[str, Any] | None] = {}
    after_step = 0
    for name in ("right_only", "both_hands", "left_only"):
        chosen = next((run for run in stage_runs[name] if run["start_step"] > after_step), None)
        sequence[name] = chosen
        if chosen is None:
            # Later contacts cannot repair a missing earlier phase.
            after_step = steps[-1] + 1 if steps else 1
        else:
            after_step = chosen["end_step"]
    ordered = all(sequence.values())
    lifted_runs = {
        "right_only": runs(lambda c: c[3] and c[1] and not c[0], lifted=True),
        "both_hands": runs(lambda c: c[3] and c[0] and c[1], lifted=True),
        "left_only": runs(lambda c: c[3] and c[0] and not c[1], lifted=True),
    }
    lifted_sequence: dict[str, dict[str, Any] | None] = {}
    after_step = 0
    for name in ("right_only", "both_hands", "left_only"):
        chosen = next((run for run in lifted_runs[name] if run["start_step"] > after_step), None)
        lifted_sequence[name] = chosen
        after_step = chosen["end_step"] if chosen is not None else steps[-1] + 1 if steps else 1

    final_start = max(0, len(rows) - final_window_frames)
    final_contacts = contacts_by_frame[final_start:]
    released_frames = sum(known and not left and not right for left, right, _, known in final_contacts)
    container_frames = sum(known and container for _, _, container, known in final_contacts)
    full_final_window = len(final_contacts) == final_window_frames
    stability_frames = max(1, math.ceil(fps * stability_window_s))
    final_positions = positions[-stability_frames:]
    displacement = (max((math.dist(a, b) for i, a in enumerate(final_positions)
                         for b in final_positions[i + 1:]), default=0.0) if final_positions else None)
    full_stability_window = len(final_positions) == stability_frames
    peak_index = max(range(len(positions)), key=lambda i: positions[i][2]) if positions else None
    maximum_lift = positions[peak_index][2] - initial_z if peak_index is not None else None
    minimum_pelvis = min(pelvis) if pelvis else None
    known_frames = sum(c[3] for c in contacts_by_frame)
    checks = {
        "contact_data_complete": bool(rows) and known_frames == len(rows),
        "ordered_right_both_left": ordered,
        "ordered_lifted_right_both_left": all(lifted_sequence.values()),
        "final_window_complete": full_final_window,
        "both_hands_released_final_window": full_final_window and released_frames == final_window_frames,
        "final_container_contact_sustained": full_final_window and container_frames >= minimum_final_container_contact_frames,
        "stability_window_complete": full_stability_window,
        "final_object_stable": full_stability_window and displacement is not None and displacement < maximum_final_displacement_m,
        "object_lifted": maximum_lift is not None and maximum_lift >= minimum_lift_m,
        "pelvis_above_fall_threshold": minimum_pelvis is not None and minimum_pelvis >= minimum_pelvis_z_m,
    }
    return {
        "gate_version": "sustained-lifted-handover-contact-v2",
        "diagnostic": "additional contact-sequence and stability evidence; separate from official task success",
        "strict_handover_success": all(checks.values()),
        "contact_handover_success": all(checks[key] for key in (
            "contact_data_complete", "ordered_lifted_right_both_left", "object_lifted", "pelvis_above_fall_threshold")),
        "contact_placement_success": all(checks[key] for key in (
            "final_window_complete", "both_hands_released_final_window", "final_container_contact_sustained",
            "stability_window_complete", "final_object_stable", "pelvis_above_fall_threshold")),
        "official_task_flag_modified": False,
        "sample_count": len(rows),
        "contact_measurement_frames": known_frames,
        "unknown_contact_measurement_frames": len(rows) - known_frames,
        "time_basis": "episode seconds = completed_steps / fps; optional simulator time reported separately",
        "thresholds": {
            "fps": fps, "minimum_stage_frames": int(minimum_stage_frames),
            "minimum_stage_lift_m_inclusive": minimum_stage_lift_m,
            "minimum_lift_m_inclusive": minimum_lift_m,
            "minimum_pelvis_z_m_inclusive": minimum_pelvis_z_m,
            "final_window_frames": int(final_window_frames),
            "minimum_final_container_contact_frames": int(minimum_final_container_contact_frames),
            "hands_released_on_every_final_frame": True,
            "stability_window_s": stability_window_s, "stability_window_frames": stability_frames,
            "maximum_pairwise_displacement_m_exclusive": maximum_final_displacement_m,
        },
        "checks": checks,
        "ordered_stages": sequence,
        "all_contact_stage_runs": stage_runs,
        "ordered_lifted_stages": lifted_sequence,
        "all_lifted_contact_stage_runs": lifted_runs,
        "final_placement": {
            "window": interval(final_start, len(rows) - 1) if rows else None,
            "hands_released_frames": released_frames,
            "container_contact_frames": container_frames,
            "maximum_pairwise_displacement_m": displacement,
            "stability_observed_sample_span_s": (len(final_positions) - 1) / fps if final_positions else 0.0,
        },
        "lift": {
            "initial_target_z_m": initial_z,
            "initial_height_source": "explicit initial_target_z_m" if initial_target_z_m is not None else "first observed replay target height",
            "maximum_lift_m": maximum_lift,
            "peak_step": steps[peak_index] if peak_index is not None else None,
        },
        "minimum_pelvis_z_m": minimum_pelvis,
        "evidence_limits": [
            "Contact booleans do not prove grasp forces, support direction, or every geometric object-inside-container condition.",
            "The pelvis height threshold is a fall proxy and does not measure every orientation condition.",
            "This diagnostic does not replace, alter, or certify the simulator's official task success flag.",
        ],
    }


def sample_live_state(model, data, *, cache, completed_steps, target_body,
                      container_body, solver_stage, pelvis_body="pelvis"):
    """Read original solved forces before observation refresh; never step/forward.

    Call in the physical-step hook immediately after ``mj_step`` and before the
    environment's ``_get_obs`` calls ``mj_forward``. In a 200 Hz simulation keep
    every fourth result for this source50Hz gate. RGB/state observation refresh
    occurs afterwards and is a separate clock/provenance record.

    A caller must bind the hook/source identity in its capture manifest. This
    marker documents a required call site; it cannot authenticate forged data.
    """
    if solver_stage != SOLVED_STATE_STAGE:
        raise ValueError("Original solved force requires post-mj_step/pre-observation-refresh sampling")
    if isinstance(completed_steps, bool) or not isinstance(completed_steps, Integral) or completed_steps < 1:
        raise ValueError("Completed50Hz steps must be positive integers")
    from handover_geometry_quality import _body_id, container_placement_status, target_contacts

    target = _body_id(model, target_body)
    container = _body_id(model, container_body)
    pelvis = _body_id(model, pelvis_body)
    return {
        "completed_steps": int(completed_steps), "sim_time_s": float(data.time),
        "actual_target_xyz_m": data.xpos[target].tolist(),
        "pelvis_z_m": float(data.xpos[pelvis, 2]),
        "contacts": target_contacts(model, data, model.body(target).name,
                                    container_label=model.body(container).name),
        "container_geometry": container_placement_status(model, data, cache=cache,
                                    container_body=container, target_body=target),
        "solver_stage": SOLVED_STATE_STAGE,
    }


def evaluate_capture(samples, *, initial_target_z_m):
    """Combine unchanged source contact and full geometry/support gates at50Hz.

    Missing evidence or the wrong acquisition phase cannot be eligible. This
    evaluates physical evidence only: RTX/pixel/action alignment, controller
    identity, lineage, warmup/history and independent splits remain mandatory
    dataset acceptance steps owned by the producer/auditor.
    """
    from handover_geometry_quality import (final_placement_geometric_validation,
                                          SOURCE_SHA256 as geometry_sha,
                                          OBJECT_SOURCE_SHA256 as object_sha)
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
        if row.get("completed_steps") != index + 1:
            errors.append(f"sample{index}: expected dense completed_steps starting1")
        try:
            current = _finite(row["sim_time_s"], "sample timestamp")
        except (KeyError, ValueError):
            errors.append(f"sample{index}: finite simulator timestamp missing")
        else:
            if previous_time is not None and not math.isclose(current - previous_time, .02, rel_tol=0, abs_tol=1e-6):
                errors.append(f"sample{index}: real50Hz acquisition clock differs")
            previous_time = current
    contact = geometry = None
    try:
        contact = summarize_handover_evidence(rows, fps=50., initial_target_z_m=initial_target_z_m)
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"contact evidence: {error}")
    try:
        geometry = final_placement_geometric_validation(rows, window_frames=50, require_3d_evidence=True)
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"geometry evidence: {error}")
    passed = (not errors and bool(contact and contact["strict_handover_success"])
              and bool(geometry and geometry["passed"]))
    return {
        "schema": "cvpr-handover-physics-capture-quality-v1",
        "acceptance_gate_version": "handover-physics-v2", "measurement_hz": 50,
        "verified_handover_success": passed, "eligible_physics": passed,
        "contact_evidence": contact, "final_placement_geometric_validation": geometry,
        "capture_validation_errors": errors,
        "source_sha256": {"replay_handover.py": SOURCE_SHA256,
                          "replay_container_geometry.py": geometry_sha,
                          "replay_objects.py": object_sha},
        "official_task_flag_modified": False, "dataset_ready": False,
        "scope": "Original50Hz necessary contact/geometry/upright/solved-support gates; not a200Hz standard, exact inner-cavity guarantee, official benchmark, or complete RGB/action/lineage dataset audit",
    }
