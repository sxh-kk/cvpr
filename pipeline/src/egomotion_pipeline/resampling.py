import numpy as np
from .geometry import slerp, rigid_transform, matrix_to_quat, quat_multiply
from .schema import DINO_FLAT_DIM


def transform_episode(episode, config):
    episode.check()
    transform = rigid_transform(config.get("world_from_source", np.eye(4)))
    scale = float(config.get("position_scale", 1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("position_scale must be positive")
    for key, p in episode.positions.items():
        episode.positions[key] = (p*scale)@transform[:3, :3].T + transform[:3, 3]
    world_q = matrix_to_quat(transform[:3, :3])
    for key, q in episode.rotations.items():
        episode.rotations[key] = quat_multiply(world_q, q)
    correction = config.get("body_basis_correction")
    if correction is not None:
        correction = np.asarray(correction, dtype=float)
        if correction.shape != (22, 3, 3):
            raise ValueError("body_basis_correction must contain 22 rotation matrices")
        for r in correction:
            rigid_transform(np.block([[r, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))
        episode.rotations["body"] = quat_multiply(episode.rotations["body"], matrix_to_quat(correction))
    episode.metadata.update({"world_from_source": transform.tolist(), "position_scale": scale,
                             "coordinate_verified": bool(config.get("coordinate_verified", False)),
                             "rotation_basis_verified": bool(config.get("rotation_basis_verified", False)),
                             "body_basis_correction": None if correction is None else correction.tolist()})
    from .ground import attach_ground
    from .head_reference import attach_head_reference
    episode.check()
    attach_ground(episode, config.get("ground"))
    attach_head_reference(episode, config.get("head_reference"))
    return episode.check()


def resample(episode, fps=30, max_gap_s=.075, snap_tolerance_s=1e-5):
    """No extrapolation, no bridging invalid endpoints/large gaps; SLERP world rotations."""
    episode.check()
    ts = episode.timestamps
    if fps != 30:
        raise ValueError("This archive is fixed at 30 Hz; training-rate conversion belongs downstream")
    if max_gap_s <= 0 or not 0 <= snap_tolerance_s < .5/fps:
        raise ValueError("Invalid gap/timestamp tolerance")
    count = int(np.floor((ts[-1]-ts[0])*fps + snap_tolerance_s*fps)) + 1
    target = ts[0] + np.arange(count, dtype=np.float64)/fps
    right = np.searchsorted(ts, target, side="left").clip(0, len(ts)-1)
    left = np.maximum(right-1, 0)
    nearest = np.where(np.abs(ts[left]-target) <= np.abs(ts[right]-target), left, right)
    exact = np.abs(ts[nearest]-target) <= snap_tolerance_s
    left = np.where(exact, nearest, left)
    right = np.where(exact, nearest, right)
    span = ts[right]-ts[left]
    weight = np.divide(target-ts[left], span, out=np.zeros_like(target), where=span > 0)
    in_range = ((target >= ts[0]-snap_tolerance_s) & (target <= ts[-1]+snap_tolerance_s)
                & (span <= max_gap_s) & (weight >= -1e-6) & (weight <= 1+1e-6))
    arrays = {"timestamp": (np.arange(count)/fps).astype(np.float32),
              "observation.source_timestamp": target,
              "observation.source_frame_left": left.astype(np.int64),
              "observation.source_frame_right": right.astype(np.int64),
              "observation.source_interpolation_weight": weight.astype(np.float32)}
    for kind, values, masks in [("position", episode.positions, episode.position_valid),
                                ("rotation", episode.rotations, episode.rotation_valid)]:
        for key, value in values.items():
            mask = masks[key][left] & masks[key][right] & in_range[:, None]
            if kind == "position":
                sampled = (1-weight[:, None, None])*value[left]+weight[:, None, None]*value[right]
                sampled = np.where(mask[..., None], sampled, 0)
            else:
                sampled = slerp(value[left], value[right], weight)
                sampled[~mask] = [1, 0, 0, 0]
            suffix = "_rotation" if kind == "rotation" else ""
            arrays[f"observation.state.{key}_world{suffix}"] = sampled.reshape(count, -1).astype(np.float32)
            arrays[f"observation.valid.{key}_{kind}"] = mask
            arrays[f"observation.interpolated.{key}_{kind}"] = mask & ~exact[:, None]
    arrays["observation.features.dino"] = np.zeros((count, DINO_FLAT_DIM), np.float32)
    arrays["observation.valid.dino"] = np.zeros(count, bool)
    arrays["observation.video_timestamp"] = np.zeros(count, np.float64)
    arrays["observation.video_time_error"] = np.zeros(count, np.float32)
    ground = episode.metadata.get("ground", {"height_m": 0., "valid": False, "verified": False})
    verified = episode.metadata.get("head_reference", {}).get("basis_verified", False)
    # Derive projected heading from the resampled full rotation; interpolating
    # an already projected heading is different when pitch/roll change.
    from .head_reference import derive_ground_features
    derive_ground_features(arrays, ground, verified)
    from .unified_dense22 import encode_unified_dense22
    arrays.update(encode_unified_dense22(arrays, {"source": episode.source, "metadata": episode.metadata}))
    return arrays
