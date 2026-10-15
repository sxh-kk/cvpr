"""Conservative, auditable quality gates for released EgoExo4D-HM fits.

These gates assess supervision, not camera observations.  ``fitted_nose_world``
must come from released OpenPose67 ``joints3d[..., 0, :]``; the archive's
``observation.state.head_world`` is Aria CPF and cannot substitute for it.
All positions must share a registered world frame, in metres.  Missing camera
evidence is reported separately and does not turn otherwise finite fits invalid.

Apply the resulting frame mask to the complete fitted body, explicit hands and
feet *before* deriving ground/contact/windows.  No label value is clamped or
changed here.  Thresholds are conservative screening settings, not an accuracy
certificate for the retained fits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntFlag

import numpy as np


class HMQualityReason(IntFlag):
    NATIVE_INVALID = 1
    FIT_NONFINITE = 2
    CAMERA_UNAVAILABLE = 4
    HEAD_CAMERA_FAR = 8
    RELATIVE_JUMP = 16
    TIME_GAP = 32
    HEAD_CAMERA_WARNING = 64


@dataclass(frozen=True)
class HMQualityPolicy:
    """At native30Hz a relative jump must exceed 35 cm to trigger rejection."""

    max_head_camera_distance_m: float = 1.0
    warning_head_camera_distance_m: float = .5
    relative_jump_margin_m: float = .25
    relative_speed_allowance_m_s: float = 3.
    max_time_gap_s: float = .075

    def __post_init__(self):
        values = asdict(self)
        if any(not np.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("HM quality thresholds must be finite and positive")
        if self.warning_head_camera_distance_m >= self.max_head_camera_distance_m:
            raise ValueError("Warning distance must be below rejection distance")


@dataclass(frozen=True)
class HMQualityResult:
    quality_valid: np.ndarray
    native_valid: np.ndarray
    fit_finite: np.ndarray
    distance_checked: np.ndarray
    transition_checked: np.ndarray
    transition_valid: np.ndarray
    reasons: np.ndarray
    head_camera_distance_m: np.ndarray
    relative_jump_m: np.ndarray

    def arrays(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _mask(value, count, name):
    value = np.asarray(value)
    if value.shape != (count,) or not np.isin(value, (False, True)).all():
        raise ValueError(f"{name} must contain one boolean per frame")
    return value.astype(bool, copy=False)


def score_hm_fit(fitted_nose_world, camera_world, *, native_valid, camera_valid,
                 timestamps_s, fit_finite=None, policy=None):
    """Return independent frame, transition, evidence and reason masks.

    A jump uses the change in ``nose_world-camera_world``: common camera/body
    translation cancels.  Both endpoints of a discontinuity are rejected, since
    without extra evidence one cannot identify the bad side.  Invalid native
    fits do not poison a neighbouring valid frame through this jump test.

    ``transition_valid[t]`` describes edge ``t-1 -> t``.  The first edge is
    always unavailable.  Time gaps prevent a comparison rather than invalidating
    otherwise usable per-frame labels.  Extra finite checks should cover fitted
    SMPL-H parameters, not unrelated missing hand landmarks.
    """
    policy = HMQualityPolicy() if policy is None else policy
    nose = np.asarray(fitted_nose_world, dtype=np.float64)
    camera = np.asarray(camera_world, dtype=np.float64)
    if nose.ndim != 2 or nose.shape[1:] != (3,) or camera.shape != nose.shape:
        raise ValueError("Nose and camera positions must have matching T x 3 shapes")
    count = len(nose)
    native = _mask(native_valid, count, "native_valid")
    camera_mask = _mask(camera_valid, count, "camera_valid")
    finite = np.isfinite(nose).all(-1)
    if fit_finite is not None:
        finite &= _mask(fit_finite, count, "fit_finite")
    times = np.asarray(timestamps_s, dtype=np.float64)
    if times.shape != (count,):
        raise ValueError("timestamps_s must contain one timestamp per frame")

    reasons = np.zeros(count, dtype=np.uint16)
    reasons[~native] |= int(HMQualityReason.NATIVE_INVALID)
    reasons[~finite] |= int(HMQualityReason.FIT_NONFINITE)
    camera_evidence = camera_mask & np.isfinite(camera).all(-1)
    reasons[~camera_evidence] |= int(HMQualityReason.CAMERA_UNAVAILABLE)
    distance_checked = camera_evidence & np.isfinite(nose).all(-1)
    distance = np.full(count, np.nan, dtype=np.float64)
    relative = np.full_like(nose, np.nan)
    relative[distance_checked] = nose[distance_checked] - camera[distance_checked]
    distance[distance_checked] = np.linalg.norm(relative[distance_checked], axis=-1)
    far = distance_checked & (distance > policy.max_head_camera_distance_m)
    warning = distance_checked & (distance > policy.warning_head_camera_distance_m)
    reasons[far] |= int(HMQualityReason.HEAD_CAMERA_FAR)
    reasons[warning] |= int(HMQualityReason.HEAD_CAMERA_WARNING)

    base_valid = native & finite
    time_edge = np.zeros(count, dtype=bool)
    if count > 1:
        dt = np.diff(times)
        time_edge[1:] = (np.isfinite(dt) & (dt > 0) & (dt <= policy.max_time_gap_s))
    transition_checked = np.zeros(count, dtype=bool)
    jump = np.full(count, np.nan, dtype=np.float64)
    jump_edge = np.zeros(count, dtype=bool)
    if count > 1:
        transition_checked[1:] = (time_edge[1:] & distance_checked[1:]
                                  & distance_checked[:-1] & base_valid[1:] & base_valid[:-1])
        ids = np.flatnonzero(transition_checked)
        jump[ids] = np.linalg.norm(relative[ids] - relative[ids-1], axis=-1)
        threshold = policy.relative_jump_margin_m + policy.relative_speed_allowance_m_s * (times[ids] - times[ids-1])
        jump_edge[ids] = jump[ids] > threshold
        reasons[1:][~time_edge[1:]] |= int(HMQualityReason.TIME_GAP)
    jump_frame = jump_edge.copy()
    if count > 1:
        jump_frame[:-1] |= jump_edge[1:]
    reasons[jump_frame] |= int(HMQualityReason.RELATIVE_JUMP)
    quality_valid = base_valid & ~far & ~jump_frame
    transition_valid = transition_checked & ~jump_edge
    if count > 1:
        transition_valid[1:] &= quality_valid[1:] & quality_valid[:-1]
    return HMQualityResult(quality_valid, native.copy(), finite, distance_checked,
                           transition_checked, transition_valid, reasons, distance, jump)


def summarize_hm_quality(result):
    """JSON-safe counts and distance distributions, with independent reasons."""
    def distribution(value, mask):
        value = value[mask & np.isfinite(value)]
        return {"count": len(value), "p50_p90_p95_p99_max": (
            np.percentile(value, [50, 90, 95, 99, 100]).tolist() if len(value) else None)}

    baseline = result.native_valid & result.fit_finite
    return {
        "frames": len(result.quality_valid),
        "native_valid_frames": int(result.native_valid.sum()),
        "native_finite_frames": int(baseline.sum()),
        "quality_valid_frames": int(result.quality_valid.sum()),
        "newly_rejected_frames": int((baseline & ~result.quality_valid).sum()),
        "distance_unchecked_frames": int((~result.distance_checked).sum()),
        "transition_checked_edges": int(result.transition_checked.sum()),
        "reason_counts": {reason.name.lower(): int(((result.reasons & reason) != 0).sum())
                          for reason in HMQualityReason},
        "newly_rejected_by_reason": {
            reason.name.lower(): int((baseline & ~result.quality_valid & ((result.reasons & reason) != 0)).sum())
            for reason in HMQualityReason},
        "head_camera_distance_all_finite_m": distribution(result.head_camera_distance_m, result.distance_checked),
        "head_camera_distance_native_finite_m": distribution(result.head_camera_distance_m, baseline & result.distance_checked),
        "relative_jump_checked_m": distribution(result.relative_jump_m, result.transition_checked),
        "distance_bin_edges_m": [0, .1, .2, .3, .5, .75, 1., 2., 5., 10., "infinity"],
        "distance_native_finite_histogram": np.histogram(
            result.head_camera_distance_m[baseline & result.distance_checked],
            bins=[0, .1, .2, .3, .5, .75, 1., 2., 5., 10., np.inf])[0].tolist(),
    }


def audit_camera_motion(camera_world, camera_valid, timestamps_s, *, top_k=10, max_native_gap_s=.075):
    """Describe camera speed tails without changing any supervision mask.

    Native edges and phase-zero 5Hz samples are reported separately.  Large
    common world-frame corrections can produce large world velocities while the
    body/camera relationship remains correct; these numbers are audit evidence,
    not an automatic camera or body rejection rule.
    """
    camera = np.asarray(camera_world, dtype=np.float64)
    if camera.ndim != 2 or camera.shape[1:] != (3,):
        raise ValueError("camera_world must have shape T x 3")
    valid = _mask(camera_valid, len(camera), "camera_valid") & np.isfinite(camera).all(-1)
    times = np.asarray(timestamps_s, dtype=np.float64)
    if times.shape != (len(camera),) or top_k < 1 or not np.isfinite(max_native_gap_s) or max_native_gap_s <= 0:
        raise ValueError("Invalid camera-motion audit settings or timestamps")
    report = {"audit_only": True, "changes_masks": False, "coordinate_scope": "registered_world_meters",
              "caveat": "Large velocities may include world-frame corrections; physical cause is not established."}
    for step, name in ((1, "native30Hz"), (6, "training5Hz_phase0")):
        current = np.arange(step, len(camera), step)
        previous = current - step
        dt = times[current] - times[previous]
        keep = valid[current] & valid[previous] & np.isfinite(dt) & (dt > 0) & (dt <= step * max_native_gap_s)
        current, previous, dt = current[keep], previous[keep], dt[keep]
        displacement = np.linalg.norm(camera[current] - camera[previous], axis=-1)
        speed = displacement / dt
        top = np.argsort(speed)[-top_k:][::-1]
        report[name] = {
            "edges_checked": len(speed), "speed_p50_p90_p95_p99_max_m_s": (
                np.percentile(speed, [50, 90, 95, 99, 100]).tolist() if len(speed) else None),
            "top_edges": [{"previous_native_frame": int(previous[i]), "native_frame": int(current[i]),
                           "dt_s": float(dt[i]), "displacement_m": float(displacement[i]),
                           "speed_m_s": float(speed[i])} for i in top],
        }
    return report
