"""Read dense22 training windows from absolute archive data; no model imports."""

from dataclasses import dataclass
from copy import deepcopy
import numpy as np

from .geometry import quat_to_matrix, matrix_to_quat, quat_multiply
from .ground import resolve_window_ground
from .head_reference import derive_ground_features
from .schema import POSITION_SHAPES, ROTATION_SHAPES


@dataclass
class TrainingWindow:
    arrays: dict
    metadata: dict
    # EE4D conditioning: local full head pose (9) + floor trajectory delta (9).
    trajectory: np.ndarray
    trajectory_valid: np.ndarray


def prepare_window(arrays, metadata, *, start_frame, num_frames, fps=15, canonicalize=True):
    """Select training FPS/window first, resolve floor, then rebuild local data.

    start_frame addresses the 30Hz archive; num_frames counts selected frames.
    Returned arrays are copies. Fallback is window-specific and never written
    back to the archive. No padding, extrapolation or intermediate invalid
    values are fabricated.
    """
    for name, value in [("start_frame", start_frame), ("num_frames", num_frames), ("fps", fps)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if start_frame < 0 or num_frames < 2 or fps < 1 or 30 % fps:
        raise ValueError("Select at least two frames, a nonnegative start and an integer divisor of 30Hz")
    selected = start_frame + np.arange(num_frames)*(30//fps)
    if selected[-1] >= len(arrays["timestamp"]):
        raise ValueError("Training window exceeds episode; no extrapolation or padding is performed")
    result = {key: np.asarray(value)[selected].copy() for key, value in arrays.items()}
    result["timestamp"] = (np.arange(num_frames)/fps).astype(np.float32)
    meta = deepcopy(metadata)
    ground = resolve_window_ground(
        result["observation.state.body_world"].reshape(num_frames, 22, 3),
        result["observation.valid.body_position"], meta["ground"], meta.get("coordinate_verified", False))
    transform = np.eye(4)
    if canonicalize:
        p_valid = result["observation.valid.head_reference_position"].reshape(num_frames)
        r_valid = result["observation.valid.head_reference_rotation"].reshape(num_frames)
        if not (p_valid[0] and r_valid[0] and meta.get("head_reference", {}).get("basis_verified", False)):
            raise ValueError("Window canonicalization requires a valid first camera reference with verified axes")
        head_rotation = quat_to_matrix(result["observation.state.head_reference_world_rotation"][0])
        horizontal = head_rotation[:2, 0]
        length = np.linalg.norm(horizontal)
        if length <= 1e-6:
            raise ValueError("First camera reference axis has a degenerate horizontal projection")
        x, y = horizontal/length
        rotation = np.array([[x, y, 0.], [-y, x, 0.], [0., 0., 1.]])
        origin = result["observation.state.head_reference_world"][0].copy()
        origin[2] = 0
        translation = -rotation @ origin
        transform[:3, :3], transform[:3, 3] = rotation, translation
        world_q = matrix_to_quat(rotation)
        for key, shape in POSITION_SHAPES.items():
            values = result[f"observation.state.{key}_world"].reshape(num_frames, *shape)
            valid = result[f"observation.valid.{key}_position"].reshape(num_frames, shape[0])
            result[f"observation.state.{key}_world"] = np.where(
                valid[..., None], values @ rotation.T + translation, 0).reshape(num_frames, -1).astype(np.float32)
        for key, shape in ROTATION_SHAPES.items():
            values = result[f"observation.state.{key}_world_rotation"].reshape(num_frames, *shape)
            valid = result[f"observation.valid.{key}_rotation"].reshape(num_frames, shape[0])
            values = quat_multiply(world_q, values)
            values[~valid] = [1, 0, 0, 0]
            result[f"observation.state.{key}_world_rotation"] = values.reshape(num_frames, -1).astype(np.float32)
    verified = bool(meta.get("head_reference", {}).get("basis_verified", False))
    derive_ground_features(result, ground, verified)
    # Physical v5 caches must follow a changed ground/world reference. Keep
    # historical v3/v4 windows at their original physical column contract.
    if "observation.state.body_world_rot6d_xyz" in result:
        from .unified_dense22 import encode_unified_dense22
        result.update(encode_unified_dense22(result, meta))
    meta.update(ground=ground, training_window={
        "archive_fps": 30, "fps": int(fps), "source_frame_indices": selected.tolist(),
        "canonicalized": bool(canonicalize), "window_from_archive_world": transform.tolist(),
        "floor_resolution_stage": "after_training_fps_and_window_selection"})
    trajectory, valid = head_trajectory(result)
    return TrainingWindow(result, meta, trajectory, valid)


def head_trajectory(arrays):
    """18D head condition with deltas at the selected training rate (row 6D)."""
    count = len(arrays["timestamp"])
    floor_rotation = quat_to_matrix(arrays["observation.state.head_reference_floor_world_rotation"])
    floor_position = arrays["observation.state.head_reference_floor_world"].copy()
    head_rotation = quat_to_matrix(arrays["observation.state.head_reference_world_rotation"])
    head_position = arrays["observation.state.head_reference_world"]
    local_rotation = np.swapaxes(floor_rotation, -1, -2) @ head_rotation
    local_position = np.einsum("tji,tj->ti", floor_rotation, head_position-floor_position)
    # EE4D translates the world floor to Z=0 before encoding floor deltas.
    floor_position[:, 2] -= arrays["observation.environment.floor_height"]
    delta_rotation = floor_rotation.copy()
    delta_position = floor_position.copy()
    delta_rotation[1:] = np.swapaxes(floor_rotation[:-1], -1, -2) @ floor_rotation[1:]
    delta_position[1:] = np.einsum("tji,tj->ti", floor_rotation[:-1], np.diff(floor_position, axis=0))
    floor_valid = (arrays["observation.valid.head_reference_floor_position"].reshape(count)
                   & arrays["observation.valid.head_reference_floor_rotation"].reshape(count))
    delta_valid = floor_valid.copy()
    delta_valid[1:] &= floor_valid[:-1]
    local_valid = floor_valid & arrays["observation.valid.head_reference_position"].reshape(count)
    local_valid &= arrays["observation.valid.head_reference_rotation"].reshape(count)
    values = np.concatenate([
        local_rotation[:, :2].reshape(count, 6), local_position,
        delta_rotation[:, :2].reshape(count, 6), delta_position], axis=-1).astype(np.float32)
    valid = np.concatenate([np.repeat(local_valid[:, None], 9, axis=1),
                            np.repeat(delta_valid[:, None], 9, axis=1)], axis=-1)
    return np.where(valid, values, 0), valid
