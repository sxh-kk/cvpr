"""Source-specific native camera tracking and continuity quality, without FK.

Aria scores are confidence, not probabilities: zero is rejected; 0.5 is not
arbitrarily discarded. Closed-loop poses remain in the registered HM world;
open-loop dynamics/session IDs only qualify continuity, never replace poses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntFlag
import numpy as np


@dataclass(frozen=True)
class CameraQualityPolicy:
    max_pose_error_s: float = .02
    max_native_gap_s: float = .075
    max_mps_gap_s: float = .02
    velocity_margin_m: float = .10
    velocity_multiplier: float = 3.
    angular_margin_rad: float = .20
    angular_multiplier: float = 3.
    max_camera_speed_m_s: float = 15.
    max_camera_angular_speed_rad_s: float = 20.

    def __post_init__(self):
        if any(not np.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError("Camera quality limits must be finite and positive")


class CameraQualityReason(IntFlag):
    NATIVE_UNAVAILABLE = 1
    CLOSED_SCORE = 2
    CLOSED_TIME = 4
    OPEN_SCORE = 8
    OPEN_TIME = 16
    CLOSED_INTERVAL = 32
    OPEN_INTERVAL = 64
    VELOCITY_DISAGREEMENT = 128
    ANGULAR_DISAGREEMENT = 256
    PHYSICAL_SPEED = 512
    PHYSICAL_ANGULAR_SPEED = 1024
    NATIVE_TIME = 2048
    TRANSFORM_INVALID = 4096


SOURCE_POLICIES = {
    "egoexo4d_hm": {
        "confidence_observed": True,
        "confidence_semantics": "MPS quality_score finite in (0,1]; no 0.5 rejection threshold",
        "poses": "registered_closed_loop_RGB_camera_preserved",
        "continuity": "all_source_samples_closed_graph_open_session_quality_gap_and_dynamics",
        "open_loop_used_as_pose_or_world": False,
        "independent_world_observations": "SMPLH_parameters_hands_feet_RGB_DINO_preserved",
    },
    "egosuite": {
        "confidence_observed": False,
        "confidence_semantics": "not_provided_in_foxglove_FrameTransforms_or_pose_HeadCamFrame",
        "poses": "existing_source_mocap_camera_transform_and_native_masks_preserved",
        "continuity": "native_finite_timestamp_and_conservative_rigid_motion_limits",
        "camera_tracking_score_certified": False,
        "independent_world_observations": "SMPLH_parameters_hands_feet_head_RGB_DINO_preserved",
    },
}


def quaternion_angle(first, second):
    a, b = np.asarray(first, np.float64), np.asarray(second, np.float64)
    norm = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = np.divide(abs(np.sum(a*b, axis=-1)), norm, out=np.ones_like(norm), where=norm > 1e-12)
    return 2*np.arccos(np.clip(cosine, 0, 1))


def nearest_indices(times, query):
    """Closest source timestamp; exact ties select AFTER, matching MPS C++.

    Production passes integer nanoseconds, avoiding float midpoint rounding.
    """
    times, query = np.asarray(times), np.asarray(query)
    if times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("Native MPS timestamps must be nonempty strictly increasing")
    right = np.clip(np.searchsorted(times, query), 0, len(times)-1)
    left = np.maximum(right-1, 0)
    return np.where(abs(times[left]-query) < abs(times[right]-query), left, right)


def interval_valid(indices, point_good, timestamps, ids, max_gap_s):
    """All source points and edges between adjacent queries, including hidden gaps."""
    index = np.asarray(indices)
    good = np.asarray(point_good, bool)
    times = np.asarray(timestamps)
    edge = good.copy()
    edge[1:] &= good[:-1] & (np.diff(times) > 0) & (np.diff(times) <= max_gap_s)
    if ids is not None:
        edge[1:] &= np.asarray(ids)[1:] == np.asarray(ids)[:-1]
    bad_prefix = np.r_[0, np.cumsum(~edge, dtype=np.int64)]
    out = good[index].copy()
    if len(index)>1:
        # edge[k] belongs to k-1 -> k. First query edge is not consumed.
        out[1:] &= ((index[1:] >= index[:-1]) &
                    (bad_prefix[index[1:]+1]-bad_prefix[index[:-1]+1] == 0) & good[index[:-1]])
    return out


def camera_metrics(position, quaternion, times_s, native_valid):
    position, quaternion, times = np.asarray(position), np.asarray(quaternion), np.asarray(times_s, np.float64)
    count = len(times)
    valid = np.asarray(native_valid, bool).copy()
    finite = (np.isfinite(position).all(-1) & np.isfinite(quaternion).all(-1)
              & (np.linalg.norm(quaternion, axis=-1) > 1e-8) & np.isfinite(times))
    dt, distance, angle = np.zeros(count), np.zeros(count), np.zeros(count)
    dt[1:], distance[1:] = np.diff(times), np.linalg.norm(np.diff(position, axis=0), axis=-1)
    angle[1:] = quaternion_angle(quaternion[:-1], quaternion[1:])
    return {"native_valid": valid, "transform_finite": finite, "dt_s": dt,
            "camera_distance_m": distance, "camera_angle_rad": angle}


def evaluate_metrics(metrics, *, source, policy=CameraQualityPolicy()):
    """Return point validity and native incoming-edge validity with reason bits."""
    if isinstance(policy, dict): policy = CameraQualityPolicy(**policy)
    if source not in SOURCE_POLICIES: raise ValueError("Unknown camera quality source")
    native = np.asarray(metrics["native_valid"], bool)
    count, reason = len(native), np.zeros(len(native), np.uint16)
    reason[~native] |= np.uint16(CameraQualityReason.NATIVE_UNAVAILABLE)
    finite = np.asarray(metrics["transform_finite"], bool)
    reason[~finite] |= np.uint16(CameraQualityReason.TRANSFORM_INVALID)
    point = native & finite
    edge = point.copy()
    dt = np.asarray(metrics["dt_s"])
    time_ok = np.isfinite(dt) & (dt > 0) & (dt <= policy.max_native_gap_s)
    time_ok[0] = True
    reason[~time_ok] |= np.uint16(CameraQualityReason.NATIVE_TIME)
    edge &= time_ok
    if count > 1: edge[1:] &= point[:-1]
    if source == "egoexo4d_hm":
        for name, bit in (("closed_score_good", CameraQualityReason.CLOSED_SCORE),
                          ("closed_time_good", CameraQualityReason.CLOSED_TIME),
                          ("open_score_good", CameraQualityReason.OPEN_SCORE),
                          ("open_time_good", CameraQualityReason.OPEN_TIME)):
            good = np.asarray(metrics[name], bool)
            reason[~good] |= np.uint16(bit)
            point &= good
        for name, bit in (("closed_interval_good", CameraQualityReason.CLOSED_INTERVAL),
                          ("open_interval_good", CameraQualityReason.OPEN_INTERVAL)):
            good = np.asarray(metrics[name], bool)
            edge &= good
            reason[~good] |= np.uint16(bit)
        # Bounds integrate the larger endpoint speed with a generous multiplier;
        # they flag resets/corrections much larger than reported physical motion.
        velocity_bad = metrics["camera_distance_m"] > (policy.velocity_margin_m
            + policy.velocity_multiplier * metrics["source_linear_speed_bound_m_s"] * np.maximum(dt, 0))
        angular_bad = metrics["camera_angle_rad"] > (policy.angular_margin_rad
            + policy.angular_multiplier * metrics["source_angular_speed_bound_rad_s"] * np.maximum(dt, 0))
        velocity_bad[0] = angular_bad[0] = False
        reason[velocity_bad] |= np.uint16(CameraQualityReason.VELOCITY_DISAGREEMENT)
        reason[angular_bad] |= np.uint16(CameraQualityReason.ANGULAR_DISAGREEMENT)
        edge &= ~(velocity_bad | angular_bad)
    linear_bad = metrics["camera_distance_m"] > policy.max_camera_speed_m_s * np.maximum(dt, 0)
    angular_bad = metrics["camera_angle_rad"] > policy.max_camera_angular_speed_rad_s * np.maximum(dt, 0)
    linear_bad[0] = angular_bad[0] = False
    reason[linear_bad] |= np.uint16(CameraQualityReason.PHYSICAL_SPEED)
    reason[angular_bad] |= np.uint16(CameraQualityReason.PHYSICAL_ANGULAR_SPEED)
    edge &= ~(linear_bad | angular_bad)
    edge &= point
    if count > 1: edge[1:] &= point[:-1]
    # Every bad edge is a continuity boundary even if both absolute poses exist.
    return {"quality_valid": point, "transition_valid": edge, "reasons": reason}


def selected_transition_valid(native_point, native_edge, selected_relative):
    """Downsample validity only after checking every intervening native sample."""
    point, edge, selected = np.asarray(native_point, bool), np.asarray(native_edge, bool), np.asarray(selected_relative)
    if point.shape != edge.shape or selected.ndim != 1 or not len(selected) or np.any(np.diff(selected) <= 0):
        raise ValueError("Camera transition input shapes/order differ")
    bad = (~point | ~edge).astype(np.int64)
    prefix = np.r_[0, np.cumsum(bad)]
    out = point[selected].copy()
    if len(selected)>1:
        out[1:] &= point[selected[:-1]] & (prefix[selected[1:]+1]-prefix[selected[:-1]+1] == 0)
    return out
