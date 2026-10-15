"""Physical screening of the *final* shape16 SMPL-H head, in world metres.

This is a supervision gate, not a fit correction or a ground-truth certificate.
The captured EgoSuite head and HM Aria CPF are independent observations with
source-specific anatomy; neither is silently substituted for FK joint 15.
Unknown observations cannot certify a fitted body. Existing per-channel masks
must still be ANDed with this result and must never be promoted to valid.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntFlag
import numpy as np


class FitQualityReason(IntFlag):
    NATIVE_INVALID = 1
    FIT_NONFINITE = 2
    CAMERA_UNAVAILABLE = 4
    HEAD_OBSERVATION_UNAVAILABLE = 8
    HEAD_CAMERA_FAR = 16
    HEAD_OBSERVATION_FAR = 32
    RELATIVE_JUMP = 64
    TIME_GAP = 128


class FitQualityWarning(IntFlag):
    HEAD_CAMERA_DISTANCE = 1
    HEAD_OBSERVATION_DISTANCE = 2


@dataclass(frozen=True)
class FitQualityPolicy:
    max_head_camera_distance_m: float = .5
    warning_head_camera_distance_m: float = .3
    max_head_observation_distance_m: float = .35
    warning_head_observation_distance_m: float = .2
    relative_jump_margin_m: float = .15
    relative_speed_allowance_m_s: float = 3.
    max_time_gap_s: float = .075

    def __post_init__(self):
        if any(not np.isfinite(x) or x <= 0 for x in asdict(self).values()):
            raise ValueError("Fit quality thresholds must be finite and positive")
        if (self.warning_head_camera_distance_m >= self.max_head_camera_distance_m
                or self.warning_head_observation_distance_m >= self.max_head_observation_distance_m):
            raise ValueError("Warning distances must be below rejection distances")


@dataclass(frozen=True)
class FitQualityResult:
    quality_valid: np.ndarray
    native_valid: np.ndarray
    fit_finite: np.ndarray
    newly_rejected: np.ndarray
    camera_checked: np.ndarray
    head_observation_checked: np.ndarray
    distance_checked: np.ndarray
    head_observation_distance_checked: np.ndarray
    transition_checked: np.ndarray
    transition_valid: np.ndarray
    reasons: np.ndarray
    warnings: np.ndarray
    head_camera_distance_m: np.ndarray
    head_observation_distance_m: np.ndarray
    relative_jump_m: np.ndarray

    def arrays(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _mask(value, count, name):
    value = np.asarray(value)
    if value.shape != (count,) or not np.isin(value, (False, True)).all():
        raise ValueError(f"{name} must contain one boolean per frame")
    return value.astype(bool, copy=False)


def native_fit_availability(position_valid, rotation_valid):
    """Any existing body22 label requires assessment, even if head is completed."""
    position, rotation = np.asarray(position_valid), np.asarray(rotation_valid)
    if (position.ndim != 2 or position.shape[1:] != (22,) or rotation.shape != position.shape
            or not np.isin(position, (False, True)).all() or not np.isin(rotation, (False, True)).all()):
        raise ValueError("Expected T x 22 boolean body position/rotation availability")
    return position.astype(bool).any(-1) | rotation.astype(bool).any(-1)


def fk_head_world(model, arrays):
    """Production-equivalent head15 FK, retaining all sixteen shape components.

    Only ancestors [0,3,6,9,12,15] are needed. This avoids computing unrelated
    joints/quaternions for every native30Hz frame. Nonfinite parameters produce
    an unavailable head, never a replacement label. The returned float32 head
    uses the same final cast as ``SMPLHBodyModel.decode``.
    """
    from .dataset.smplh_archive import _exp_so3

    count = len(arrays["timestamp"])
    values = {}
    finite = np.ones(count, dtype=bool)
    for key, width in (("smplh.root_orient", 3), ("smplh.pose_body", 63),
                       ("smplh.trans", 3), ("smplh.betas", 16)):
        value = np.asarray(arrays[key], dtype=np.float64)
        if value.shape != (count, width):
            raise ValueError(f"{key} must have shape T x {width}")
        finite &= np.isfinite(value).all(-1)
        values[key] = value
    result = np.full((count, 3), np.nan, dtype=np.float32)
    ids = np.flatnonzero(finite)
    if not len(ids):
        return result, finite
    chain = np.array([0, 3, 6, 9, 12, 15])
    rest = model.rest[chain][None] + np.einsum("tb,jcb->tjc", values["smplh.betas"][ids], model.shape_basis[chain])
    aa = np.concatenate((values["smplh.root_orient"][ids, None],
                         values["smplh.pose_body"][ids].reshape(len(ids), 21, 3)), axis=1)[:, chain]
    local = _exp_so3(aa)
    position = rest[:, 0] + values["smplh.trans"][ids]
    rotation = local[:, 0]
    for joint in range(1, len(chain)):
        position = position + np.einsum("tij,tj->ti", rotation, rest[:, joint] - rest[:, joint-1])
        rotation = rotation @ local[:, joint]
    result[ids] = position.astype(np.float32)
    finite &= np.isfinite(result).all(-1)
    return result, finite


def score_fit_quality(fitted_head_world, camera_world, head_observation_world, *, native_valid,
                      camera_valid, head_observation_valid, timestamps_s, fit_finite=None, policy=None):
    """Screen final head geometry and camera-relative discontinuities.

    An edge is checked only when both endpoints have native labels, finite FK,
    both independent observations and a positive dt <= max_time_gap_s. A jump
    exceeds margin + speed*dt and rejects BOTH endpoints. Shared world movement
    cancels in head-camera. Time gaps alone mark an unavailable edge; they do
    not reject a frame with good independent geometric evidence.
    """
    policy = FitQualityPolicy() if policy is None else policy
    head, camera, observed = [np.asarray(x, dtype=np.float64)
                              for x in (fitted_head_world, camera_world, head_observation_world)]
    if head.ndim != 2 or head.shape[1:] != (3,) or camera.shape != head.shape or observed.shape != head.shape:
        raise ValueError("Head, camera and source head positions must have matching T x 3 shapes")
    count = len(head)
    native = _mask(native_valid, count, "native_valid")
    finite = np.isfinite(head).all(-1)
    if fit_finite is not None:
        finite &= _mask(fit_finite, count, "fit_finite")
    times = np.asarray(timestamps_s, dtype=np.float64)
    if times.shape != (count,):
        raise ValueError("timestamps_s must contain one timestamp per frame")
    camera_checked = _mask(camera_valid, count, "camera_valid") & np.isfinite(camera).all(-1)
    head_checked = _mask(head_observation_valid, count, "head_observation_valid") & np.isfinite(observed).all(-1)
    distance_checked = camera_checked & np.isfinite(head).all(-1)
    observed_checked = head_checked & np.isfinite(head).all(-1)
    relative = np.full(head.shape, np.nan, dtype=np.float64)
    relative[distance_checked] = head[distance_checked] - camera[distance_checked]
    distance = np.full(count, np.nan, dtype=np.float64)
    head_distance = distance.copy()
    distance[distance_checked] = np.linalg.norm(relative[distance_checked], axis=-1)
    head_distance[observed_checked] = np.linalg.norm(head[observed_checked] - observed[observed_checked], axis=-1)
    far = distance_checked & (distance > policy.max_head_camera_distance_m)
    head_far = observed_checked & (head_distance > policy.max_head_observation_distance_m)
    reasons = np.zeros(count, dtype=np.uint16)
    for mask, reason in ((~native, FitQualityReason.NATIVE_INVALID), (~finite, FitQualityReason.FIT_NONFINITE),
                         (~camera_checked, FitQualityReason.CAMERA_UNAVAILABLE),
                         (~head_checked, FitQualityReason.HEAD_OBSERVATION_UNAVAILABLE),
                         (far, FitQualityReason.HEAD_CAMERA_FAR), (head_far, FitQualityReason.HEAD_OBSERVATION_FAR)):
        reasons[mask] |= int(reason)
    warnings = np.zeros(count, dtype=np.uint16)
    warnings[distance_checked & (distance > policy.warning_head_camera_distance_m)] |= int(FitQualityWarning.HEAD_CAMERA_DISTANCE)
    warnings[observed_checked & (head_distance > policy.warning_head_observation_distance_m)] |= int(FitQualityWarning.HEAD_OBSERVATION_DISTANCE)
    base = native & finite & camera_checked & head_checked
    transition_checked = np.zeros(count, dtype=bool)
    jump_edge = np.zeros(count, dtype=bool)
    jump = np.full(count, np.nan, dtype=np.float64)
    if count > 1:
        dt = np.diff(times)
        time_ok = np.isfinite(dt) & (dt > 0) & (dt <= policy.max_time_gap_s)
        reasons[1:][~time_ok] |= int(FitQualityReason.TIME_GAP)
        transition_checked[1:] = base[:-1] & base[1:] & time_ok
        ids = np.flatnonzero(transition_checked)
        jump[ids] = np.linalg.norm(relative[ids] - relative[ids-1], axis=-1)
        jump_edge[ids] = jump[ids] > policy.relative_jump_margin_m + policy.relative_speed_allowance_m_s * (times[ids] - times[ids-1])
    jump_frames = jump_edge.copy()
    if count > 1:
        jump_frames[:-1] |= jump_edge[1:]
    reasons[jump_frames] |= int(FitQualityReason.RELATIVE_JUMP)
    valid = base & ~far & ~head_far & ~jump_frames
    transition_valid = transition_checked & ~jump_edge
    if count > 1:
        transition_valid[1:] &= valid[:-1] & valid[1:]
    return FitQualityResult(valid, native.copy(), finite, native & ~valid, camera_checked, head_checked,
                            distance_checked, observed_checked, transition_checked, transition_valid,
                            reasons, warnings, distance, head_distance, jump)


def summarize_fit_quality(result):
    def distribution(value, mask):
        value = value[mask & np.isfinite(value)]
        return {"count": len(value), "p50_p90_p95_p99_max": (
            np.percentile(value, [50, 90, 95, 99, 100]).tolist() if len(value) else None)}

    return {
        "frames": len(result.native_valid), "native_valid_frames": int(result.native_valid.sum()),
        "native_finite_frames": int((result.native_valid & result.fit_finite).sum()),
        "quality_valid_frames": int(result.quality_valid.sum()),
        "newly_rejected_frames": int(result.newly_rejected.sum()),
        "camera_unknown_native_frames": int((result.native_valid & ~result.camera_checked).sum()),
        "head_observation_unknown_native_frames": int((result.native_valid & ~result.head_observation_checked).sum()),
        "transition_checked_edges": int(result.transition_checked.sum()),
        "reason_counts": {r.name.lower(): int(((result.reasons & int(r)) != 0).sum()) for r in FitQualityReason},
        "warning_counts": {r.name.lower(): int(((result.warnings & int(r)) != 0).sum()) for r in FitQualityWarning},
        "newly_rejected_by_reason": {r.name.lower(): int((result.newly_rejected & ((result.reasons & int(r)) != 0)).sum()) for r in FitQualityReason},
        "head_camera_distance_native_m": distribution(result.head_camera_distance_m, result.native_valid & result.distance_checked),
        "head_observation_distance_native_m": distribution(result.head_observation_distance_m, result.native_valid & result.head_observation_distance_checked),
        "relative_jump_checked_m": distribution(result.relative_jump_m, result.transition_checked),
    }
