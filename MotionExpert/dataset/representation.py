"""Camera-floor K12 motion geometry from the shared SMPL-H world FK.

Bodies are twelve ordered (row-6D rotation, XYZ) blocks. Hands are explicit
wrist + twenty finger points. Rotations are joint-to-world, never parent-local.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import torch

from ..config import K12_JOINTS
from ..utils.rotation_conversions import rotation_6d_to_matrix
from .motion_layout import motion_layout

REPRE_TYPE = "smplh12_rot6d_hands21_contacts_v1"
MOTION_DIM = 245
TRAJECTORY_DIM = 18
BODY_SLICE = slice(0, 108)
GLOBAL_SLICE = slice(108, 117)
LEFT_HAND_SLICE = slice(117, 180)
RIGHT_HAND_SLICE = slice(180, 243)
CONTACT_SLICE = slice(243, 245)


def quaternion_matrix(value):
    """wxyz quaternions; invalid values must be excluded by the caller's mask."""
    q = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    valid = np.isfinite(q).all(-1) & (norm[..., 0] > 1e-8)
    q = np.divide(q, norm, out=np.zeros_like(q), where=norm > 1e-8)
    q[~valid] = [1, 0, 0, 0]
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1).reshape(*q.shape[:-1], 3, 3)


def resolve_ground(body, valid, metadata, *, task, forecast_prefix):
    """Do not consume future feet when an estimated floor fails in forecasting."""
    ground = deepcopy(metadata.get("ground", {}))
    if not metadata.get("coordinate_verified", False):
        ground.update(valid=False, reason="world_vertical_axis_unverified")
        return ground
    if ground.get("valid", False):
        if not np.isfinite(ground.get("height_m", np.nan)):
            raise ValueError("Valid ground requires finite height_m")
        ground.update(window_fallback_used=False, estimation_scope="archive")
        return ground
    policy = ground.get("window_fallback", {"mode": "mean_valid_feet", "offset_m": .02})
    if ground.get("mode") != "estimate" or policy.get("mode") != "mean_valid_feet":
        ground.update(window_fallback_used=False)
        return ground
    extent = min(int(forecast_prefix), len(body)) if task == "fore" else len(body)
    feet = body[:extent, [10, 11]]
    mask = valid[:extent, [10, 11]] & np.isfinite(feet).all(-1)
    offset = float(policy.get("offset_m", .02))
    if not np.isfinite(offset) or offset < 0:
        raise ValueError("Floor fallback offset must be finite and nonnegative")
    ground.update(window_fallback_valid_point_count=int(mask.sum()),
                  window_fallback_used=False,
                  estimation_scope="observed_prefix" if task == "fore" else "full_window",
                  estimation_frame_count=extent)
    if mask.any():
        ground.update(height_m=float(feet[..., 2][mask].mean()-offset), valid=True,
                      verified=False, source="training_window_mean_valid_feet",
                      reason="window_fallback", window_fallback_used=True)
    else:
        ground.update(reason="window_has_no_valid_feet", valid=False)
    return ground


def _array(arrays, key, shape):
    value = np.asarray(arrays[key])
    return value.reshape(len(value), *shape)


def _valid(arrays, modality, kind, count):
    values = _array(arrays, f"observation.state.{modality}_world" + ("_rotation" if kind == "rotation" else ""), (count, 4 if kind == "rotation" else 3))
    mask = _array(arrays, f"observation.valid.{modality}_{kind}", (count,)).astype(bool)
    mask &= np.isfinite(values).all(-1)
    if kind == "rotation":
        mask &= np.linalg.norm(values, axis=-1) > 1e-8
    return values, mask


def body_world_components(arrays):
    """Read the full22 world XYZ/wxyz emitted by SMPL-H FK with its masks."""
    body, body_valid = _valid(arrays, "body", "position", 22)
    quaternion, rotation_valid = _valid(arrays, "body", "rotation", 22)
    return body, body_valid, quaternion_matrix(quaternion), rotation_valid


def detect_contacts(body_world, valid, timestamps, ground, *, speed_limit=.2, height_limit=.1, fps=5):
    """Original 2cm at 10Hz threshold expressed as 0.2m/s; no gap bridging."""
    count = len(body_world)
    contacts = np.zeros((count, 2), dtype=np.float32)
    contact_valid = np.zeros((count, 2), dtype=bool)
    if count < 2 or not ground.get("valid", False):
        return contacts, contact_valid
    feet, mask = body_world[:, [10, 11]], valid[:, [10, 11]]
    dt = np.diff(np.asarray(timestamps, dtype=np.float64))
    usable_dt = np.isfinite(dt) & (dt > 0) & (dt <= 1.5/fps)
    edges = mask[1:] & mask[:-1] & usable_dt[:, None]
    speed = np.linalg.norm(np.diff(feet, axis=0), axis=-1) / np.where(usable_dt, dt, 1)[:, None]
    contact_valid[1:] = edges
    contacts[1:] = ((speed < speed_limit) & (feet[1:, :, 2]-ground["height_m"] < height_limit)).astype(np.float32)
    # The first frame has no previous sample, matching the original first-edge convention.
    contact_valid[0], contacts[0] = edges[0], contacts[1]
    return np.where(contact_valid, contacts, 0), contact_valid


@dataclass
class MotionWindow:
    motion: np.ndarray
    motion_valid: np.ndarray
    trajectory: np.ndarray
    trajectory_valid: np.ndarray
    metadata: dict


def build_motion_window(arrays, metadata, *, task="fore", forecast_prefix=6, fps=5,
                       body_basis_correction=None, rotation_basis_verified=None,
                       contact_speed_m_s=.2, contact_height_m=.1,
                       joint_indices=None, source_feet_world=None, source_feet_valid=None):
    """Encode already selected frames. Neither archive nor input arrays are mutated."""
    if task != "fore" or forecast_prefix != 6 or fps != 5:
        raise ValueError(f"Unknown task {task!r}")
    count = len(arrays["timestamp"])
    if count < 2:
        raise ValueError("Motion windows require at least two selected frames")
    meta = deepcopy(metadata)
    body, body_valid, body_world_rotation, rotation_valid = body_world_components(arrays)
    selected = np.asarray(K12_JOINTS if joint_indices is None else joint_indices)
    if (selected.ndim != 1 or selected.dtype.kind not in "iu" or not len(selected)
            or np.any((selected < 0) | (selected >= 22)) or len(np.unique(selected)) != len(selected)):
        raise ValueError("joint_indices must be unique integer body joints in [0,22)")
    joint_count = len(selected)
    feet_body, feet_valid = body, body_valid
    if (source_feet_world is None) != (source_feet_valid is None):
        raise ValueError("Source feet require both positions and validity")
    if source_feet_world is not None:
        feet = np.asarray(source_feet_world).reshape(count, 2, 3)
        feet_mask = np.asarray(source_feet_valid)
        if feet_mask.dtype != np.bool_ or feet_mask.shape != (count, 2):
            raise ValueError("source_feet_valid must be [T,2] booleans")
        feet_body = np.zeros((count, 22, 3), dtype=np.float64)
        feet_valid = np.zeros((count, 22), dtype=bool)
        feet_body[:, [10, 11]] = feet
        feet_valid[:, [10, 11]] = feet_mask & np.isfinite(feet).all(-1)
    head, head_valid = _valid(arrays, "head_reference", "position", 1)
    head_q, head_rotation_valid = _valid(arrays, "head_reference", "rotation", 1)
    head, head_q = head[:, 0], head_q[:, 0]
    head_valid, head_rotation_valid = head_valid[:, 0], head_rotation_valid[:, 0]
    basis_verified = bool(meta.get("head_reference", {}).get("basis_verified", False))
    if not (basis_verified and meta.get("coordinate_verified", False) and head_valid[0] and head_rotation_valid[0]):
        raise ValueError("Window requires verified world/camera axes and a valid first camera pose")
    head_rotation = quaternion_matrix(head_q)
    horizontal = head_rotation[:, :2, 0]
    norm = np.linalg.norm(horizontal, axis=-1)
    if norm[0] <= 1e-6:
        raise ValueError("First camera reference axis has degenerate horizontal projection")
    horizontal = np.divide(horizontal, norm[:, None], out=np.zeros_like(horizontal), where=norm[:, None] > 1e-6)
    # Canonical world keeps Z-up and places the first camera XY at the origin.
    x, y = horizontal[0]
    canonical_rotation = np.array([[x, y, 0], [-y, x, 0], [0, 0, 1]], dtype=np.float64)
    origin = head[0].copy(); origin[2] = 0
    transform = np.eye(4); transform[:3, :3] = canonical_rotation
    transform[:3, 3] = -canonical_rotation @ origin
    ground = resolve_ground(feet_body, feet_valid, meta, task=task, forecast_prefix=forecast_prefix)
    floor_height = float(ground.get("height_m", 0)) if ground.get("valid", False) else 0.
    canonical_body = body @ canonical_rotation.T + transform[:3, 3]
    canonical_head = head @ canonical_rotation.T + transform[:3, 3]
    canonical_head_rotation = canonical_rotation @ head_rotation
    canonical_heading = np.einsum("ij,tj->ti", canonical_rotation[:2, :2], horizontal)
    floor_rotation = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    floor_rotation[:, 0, 0], floor_rotation[:, 1, 0] = canonical_heading[:, 0], canonical_heading[:, 1]
    floor_rotation[:, 0, 1], floor_rotation[:, 1, 1] = -canonical_heading[:, 1], canonical_heading[:, 0]
    floor_position = canonical_head.copy(); floor_position[:, 2] = floor_height
    floor_valid = head_valid & head_rotation_valid & (norm > 1e-6) & bool(ground.get("valid", False))
    # Neutral transforms at unavailable references are masked, never valid labels.
    floor_rotation[~floor_valid] = np.eye(3)
    floor_position[~floor_valid] = 0
    local_body = np.einsum("tji,tkj->tki", floor_rotation, canonical_body-floor_position[:, None])
    body_rotation = canonical_rotation @ body_world_rotation
    if body_basis_correction is not None:
        correction = np.asarray(body_basis_correction, dtype=np.float64)
        if correction.shape != (22, 3, 3):
            raise ValueError("Body basis mapping must contain 22 proper 3x3 rotations")
        if not np.allclose(np.swapaxes(correction, -1, -2) @ correction, np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(correction), 1, atol=1e-5):
            raise ValueError("Body basis mapping contains improper rotations")
        body_rotation = body_rotation @ correction
    verified = bool(meta.get("rotation_basis_verified", False)) if rotation_basis_verified is None else bool(rotation_basis_verified)
    if not verified:
        rotation_valid[:] = False
    local_body_rotation = np.swapaxes(floor_rotation, -1, -2)[:, None] @ body_rotation
    body_features = np.concatenate([local_body_rotation[:, selected, :2].reshape(count, joint_count, 6), local_body[:, selected]], -1).reshape(count, joint_count*9)
    body_mask = np.concatenate([np.repeat((rotation_valid[:, selected] & floor_valid[:, None])[..., None], 6, -1),
                                np.repeat((body_valid[:, selected] & floor_valid[:, None])[..., None], 3, -1)], -1).reshape(count, joint_count*9)
    delta_rotation = floor_rotation.copy()
    flat_floor = floor_position.copy(); flat_floor[:, 2] = 0
    delta_position = flat_floor.copy()
    delta_rotation[1:] = np.swapaxes(floor_rotation[:-1], -1, -2) @ floor_rotation[1:]
    delta_position[1:] = np.einsum("tji,tj->ti", floor_rotation[:-1], np.diff(flat_floor, axis=0))
    delta_valid = floor_valid.copy(); delta_valid[1:] &= floor_valid[:-1]
    delta = np.concatenate([delta_rotation[:, :2].reshape(count, 6), delta_position], -1)
    local_head_rotation = np.swapaxes(floor_rotation, -1, -2) @ canonical_head_rotation
    local_head_position = np.einsum("tji,tj->ti", floor_rotation, canonical_head-floor_position)
    trajectory = np.concatenate([local_head_rotation[:, :2].reshape(count, 6), local_head_position, delta], -1)
    trajectory_valid = np.repeat(np.stack([floor_valid, delta_valid], -1), 9, -1)
    hands, hand_masks = [], []
    for name in ("hand_left", "hand_right"):
        positions, valid = _valid(arrays, name, "position", 21)
        positions = positions @ canonical_rotation.T + transform[:3, 3]
        local = np.einsum("tji,tkj->tki", floor_rotation, positions-floor_position[:, None])
        hands.append(local.reshape(count, 63)); hand_masks.append(np.repeat(valid & floor_valid[:, None], 3, -1))
    contact, contact_valid = detect_contacts(feet_body, feet_valid, arrays["timestamp"], ground,
                                           speed_limit=contact_speed_m_s, height_limit=contact_height_m, fps=fps)
    motion = np.concatenate([body_features, delta, *hands, contact], -1)
    motion_valid = np.concatenate([body_mask, np.repeat(delta_valid[:, None], 9, -1), *hand_masks, contact_valid], -1)
    motion_valid &= np.isfinite(motion); trajectory_valid &= np.isfinite(trajectory)
    meta.update(ground=ground, training_window={
        "fps": int(fps), "archive_fps": 30, "canonicalized": True,
        "window_from_archive_world": transform.tolist(),
        "archive_world_from_window": np.linalg.inv(transform).tolist(),
        "floor_height_m": floor_height, "task_floor_policy": task,
        "rotation_basis_verified": bool(meta.get("rotation_basis_verified", False)),
        "rotation_supervision_enabled": verified,
        "target_joint_basis": meta.get("rotation_target_basis", "unspecified"),
        "rotation_basis_mode": meta.get("rotation_supervision_policy", "verified"),
        "archive_body_encoding": "smplh_fk_world_xyz_wxyz",
        "floor_rotation": floor_rotation.tolist(), "floor_position": floor_position.tolist(),
    })
    return MotionWindow(np.where(motion_valid, motion, 0).astype(np.float32), motion_valid,
                       np.where(trajectory_valid, trajectory, 0).astype(np.float32), trajectory_valid, meta)


def _checked_transform(value, reference, name):
    transform = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if transform.shape != (4, 4) or not bool(torch.isfinite(transform).all()):
        raise ValueError(f"{name} must be a finite 4x4 rigid transform")
    check = transform.double()
    identity = torch.eye(3, device=check.device, dtype=check.dtype)
    bottom = torch.tensor([0.,0.,0.,1.], device=check.device,dtype=check.dtype)
    if not torch.allclose(check[3],bottom,atol=1e-5,rtol=0) or not torch.allclose(check[:3,:3].T@check[:3,:3],identity,atol=1e-5,rtol=0) or not torch.isclose(torch.linalg.det(check[:3,:3]),check.new_tensor(1.),atol=1e-5,rtol=0):
        raise ValueError(f"{name} must contain a proper rigid rotation")
    return transform


def decode_motion(motion, *, window_to_world=None, initial_floor_transform=None,
                        floor_height=0., start_frame=0, delta_valid=None):
    """Recover body/hand world poses from an unnormalized known motion layout.

    ``initial_floor_transform`` is the previous observed camera-floor transform
    in WINDOW coordinates for ``start_frame > 0``. Without it the first delta
    is the initial pose. Invalid chains raise; callers must explicitly restart
    from a known anchor rather than integrate across missing reference frames.
    ``window_to_world`` maps the resulting window poses to archive world.
    """
    value = torch.as_tensor(motion)
    if not value.is_floating_point(): value = value.float()
    if value.ndim != 2:
        raise ValueError(f"Expected T x D motion, got {tuple(value.shape)}")
    layout = motion_layout(value.shape[-1])
    if not 0 <= start_frame < len(value): raise ValueError("start_frame outside motion")
    if start_frame > 0 and initial_floor_transform is None:
        raise ValueError("Future decoding requires a last-observed floor transform")
    if delta_valid is not None and not bool(torch.as_tensor(delta_valid)[start_frame:].all()):
        raise ValueError("Cannot integrate an invalid floor-delta chain; supply a new anchor after each gap")
    delta = value[start_frame:, layout.global_slice]
    matrix = rotation_6d_to_matrix(delta[:, :6])
    # The reference keeps yaw only; project decoded 6D to that convention.
    axis = matrix[:, :2, 0]
    norm = torch.linalg.vector_norm(axis, dim=-1)
    if bool((norm <= 1e-7).any()): raise ValueError("Degenerate decoded floor heading")
    axis = axis/norm[:, None]
    rotations = torch.eye(3, device=value.device, dtype=value.dtype).repeat(len(delta), 1, 1)
    rotations[:, 0, 0], rotations[:, 1, 0] = axis[:, 0], axis[:, 1]
    rotations[:, 0, 1], rotations[:, 1, 1] = -axis[:, 1], axis[:, 0]
    translations = delta[:, 6:].clone(); translations[:, 2] = 0
    if initial_floor_transform is not None:
        anchor = _checked_transform(initial_floor_transform,value,"initial_floor_transform")
        prev_r, prev_p = anchor[:3, :3], anchor[:3, 3].clone()
        prev_p[2] = 0
    else:
        prev_r = torch.eye(3, device=value.device, dtype=value.dtype)
        prev_p = torch.zeros(3, device=value.device, dtype=value.dtype)
    all_r, all_p = [], []
    for r, p in zip(rotations, translations):
        next_p = prev_p + prev_r @ p
        next_r = prev_r @ r
        all_r.append(next_r); all_p.append(next_p)
        prev_r, prev_p = next_r, next_p
    floor_r, floor_p = torch.stack(all_r), torch.stack(all_p)
    floor_p[:, 2] = float(floor_height)
    local = value[start_frame:]
    body = local[:, layout.body_slice].reshape(-1, len(layout.joint_indices), 9)
    body_r = floor_r[:, None] @ rotation_6d_to_matrix(body[..., :6])
    body_p = torch.einsum("tij,tkj->tki", floor_r, body[..., 6:]) + floor_p[:, None]
    left = torch.einsum("tij,tkj->tki", floor_r, local[:, layout.left_hand_slice].reshape(-1, 21, 3)) + floor_p[:, None]
    right = torch.einsum("tij,tkj->tki", floor_r, local[:, layout.right_hand_slice].reshape(-1, 21, 3)) + floor_p[:, None]
    if window_to_world is not None:
        transform = _checked_transform(window_to_world,value,"window_to_world")
        r, p = transform[:3, :3], transform[:3, 3]
        body_p, left, right = body_p @ r.T + p, left @ r.T + p, right @ r.T + p
        body_r, floor_r, floor_p = r @ body_r, r @ floor_r, floor_p @ r.T + p
    return {"body_positions": body_p, "body_rotations": body_r, "left_hand": left,
            "right_hand": right, "floor_positions": floor_p, "floor_rotations": floor_r,
            "contacts": local[:, layout.contact_slice]}


# Public compatibility alias for the dense decoder.
decode_motion = decode_motion
