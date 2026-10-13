"""EE4D-style stationary toe-height DBSCAN and training-window fallback.

Use physical speeds (m/s), mask invalid endpoints, and never interpret a zero
height as a missing value. Estimates are heuristic labels, not measured planes.
"""

import numpy as np


def attach_ground(episode, config=None):
    config = config or {}
    mode = config.get("mode", "estimate")
    result = {"height_m": 0.0, "valid": False, "verified": False,
              "normal": [0.0, 0.0, 1.0], "frame": "output_world", "mode": mode,
              "foot_joint_indices": [10, 11],
              "window_fallback": {"mode": "mean_valid_feet" if mode == "estimate" else "disabled",
                                  "offset_m": 0.02}}
    if mode not in {"estimate", "manual", "disabled"}:
        raise ValueError("ground.mode must be estimate, manual, or disabled")
    if mode == "disabled":
        result["reason"] = "disabled"
    elif not episode.metadata.get("coordinate_verified", False):
        result["reason"] = "output_world_vertical_axis_unverified"
    elif mode == "manual":
        height = float(config["height_m"])
        if not np.isfinite(height):
            raise ValueError("ground.height_m must be finite in the output world frame")
        result.update(height_m=height, valid=True, verified=bool(config.get("verified", False)),
                      source=config.get("source", "manual_output_world_height"))
    else:
        result.update(estimate_from_feet(episode, config))
    episode.metadata["ground"] = result
    return episode


def estimate_from_feet(episode, config):
    speed_limit = float(config.get("static_speed_m_s", .15))
    eps = float(config.get("cluster_eps_m", .005))
    offset = float(config.get("toe_offset_m", .01))
    max_gap = float(config.get("max_gap_s", .075))
    min_samples = config.get("cluster_min_samples", 3)
    if isinstance(min_samples, bool) or not isinstance(min_samples, (int, np.integer)) or min_samples < 1:
        raise ValueError("ground.cluster_min_samples must be a positive integer")
    for removed in ["min_support_s", "max_cluster_spread_m"]:
        if removed in config:
            raise ValueError(f"ground.{removed} is no longer used; remove it to use EE4D-style DBSCAN")
    params = [speed_limit, eps, offset, max_gap]
    if not np.isfinite(params).all() or min(speed_limit, eps, max_gap) <= 0 or offset < 0:
        raise ValueError("Invalid ground estimation thresholds")
    result = {"method": "stationary_foot_height_dbscan", "parameters": {
        "static_speed_m_s": speed_limit, "cluster_eps_m": eps, "cluster_min_samples": int(min_samples),
        "toe_offset_m": offset, "max_gap_s": max_gap, "noise_policy": "exclude_label_minus_one"},
        "valid": False, "verified": False, "clusters": []}
    ts = episode.timestamps
    if len(ts) < 2:
        result["reason"] = "too_few_frames"
        return result
    feet = episode.positions["body"][:, [10, 11]]
    mask = episode.position_valid["body"][:, [10, 11]]
    dt = np.diff(ts)
    edges = mask[:-1] & mask[1:] & (dt <= max_gap)[:, None]
    speed = np.linalg.norm(np.diff(feet, axis=0), axis=-1)/dt[:, None]
    # Both neighboring observable edges must be slow. No differencing across a gap.
    peak = np.zeros(mask.shape)
    observed = np.zeros(mask.shape, bool)
    for target in [slice(None, -1), slice(1, None)]:
        peak[target] = np.maximum(peak[target], np.where(edges, speed, 0))
        observed[target] |= edges
    stationary = mask & observed & (peak <= speed_limit)
    frame_ids, foot_ids = np.nonzero(stationary)
    heights = feet[frame_ids, foot_ids, 2]
    result["candidate_count"] = len(heights)
    if not len(heights):
        result["reason"] = "no_stationary_feet"
        return result
    from sklearn.cluster import DBSCAN
    # Preserve duplicate-point density while avoiding a quadratic neighborhood
    # allocation for long perfectly stationary sequences.
    unique, inverse, counts = np.unique(heights, return_inverse=True, return_counts=True)
    clustering = DBSCAN(eps=eps, min_samples=int(min_samples)).fit(unique[:, None], sample_weight=counts)
    labels = clustering.labels_[inverse]
    result["noise_count"] = int(np.count_nonzero(labels == -1))
    # Estimate supported time with actual timestamps; invalid gaps add no duration.
    root = episode.positions["body"][:, 0, 2]
    root_valid = episode.position_valid["body"][:, 0]
    for label in np.unique(labels[labels >= 0]):
        group = np.flatnonzero(labels == label)
        frame = np.unique(frame_ids[group])
        gaps = np.diff(ts[frame])
        duration = float(gaps[gaps <= max_gap].sum())
        spread = float(np.ptp(heights[group]))
        roots = root[frame[root_valid[frame]]]
        result["clusters"].append({"median_height_m": float(np.median(heights[group])),
                                   "root_median_height_m": float(np.median(roots)) if len(roots) else None,
                                   "sample_count": len(group), "support_s": duration, "spread_m": spread})
    if not result["clusters"]:
        result["reason"] = "no_density_height_cluster"
        return result
    chosen = min(result["clusters"], key=lambda c: c["median_height_m"])
    result.update(height_m=chosen["median_height_m"]-offset, valid=True,
                  source="estimated_stationary_feet", reason="estimated")
    result["multiple_height_levels"] = any(c["median_height_m"] > chosen["median_height_m"]+.04
                                          for c in result["clusters"])
    return result


def resolve_window_ground(body_position, body_valid, ground, coordinate_verified):
    """Resolve one already selected training window, without mutating the archive.

    A valid zero-height floor is retained. Only failed estimation may fall back;
    disabled/manual/unverified-world modes cannot be made valid by this helper.
    """
    result = dict(ground)
    if not coordinate_verified:
        result.update(valid=False, verified=False, reason="output_world_vertical_axis_unverified")
        return result
    if result.get("valid", False):
        if not np.isfinite(result["height_m"]):
            raise ValueError("A valid floor must have a finite height")
        result["window_fallback_used"] = False
        return result
    policy = result.get("window_fallback", {"mode": "mean_valid_feet", "offset_m": .02})
    if result.get("mode") != "estimate" or policy["mode"] != "mean_valid_feet":
        result["window_fallback_used"] = False
        return result
    positions = np.asarray(body_position, float)
    mask = np.asarray(body_valid, bool)
    if positions.ndim != 3 or positions.shape[1:] != (22, 3) or mask.shape != positions.shape[:2]:
        raise ValueError("Window floor fallback requires dense22 positions and per-joint masks")
    feet = positions[:, [10, 11]]
    valid = mask[:, [10, 11]] & np.isfinite(feet).all(-1)
    offset = float(policy["offset_m"])
    if not np.isfinite(offset) or offset < 0:
        raise ValueError("Window floor fallback offset must be finite and nonnegative")
    result["window_fallback_valid_point_count"] = int(valid.sum())
    result["window_fallback_used"] = False
    if not valid.any():
        result["reason"] = "window_has_no_valid_feet"
        return result
    result.update(height_m=float(feet[..., 2][valid].mean()-offset), valid=True, verified=False,
                  source="training_window_mean_valid_feet", reason="window_fallback",
                  estimation_failure_reason=ground.get("reason", "unspecified"), window_fallback_used=True)
    return result
