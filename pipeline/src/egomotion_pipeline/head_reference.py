"""EE4D/EgoEgo reference at the camera origin, and its floor projection.

Exactly reproduce EE4D's post-multiplication, without approximating its legacy
SLAM clock or substituting CPF/anatomical head positions for camera positions.
"""

import numpy as np
from .geometry import matrix_to_quat, quat_to_matrix, quat_multiply, rigid_transform

EE4D_CAMERA_FROM_REFERENCE = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])

# EE4D consumes the unrotated Aria RGB calibration. EgoSuite instead publishes
# an upright OpenCV camera frame. Both yield reference axes left/up/forward;
# the extra B for OpenCV makes the total right-multiplication B @ B.
CAMERA_BASIS_POLICIES = {
    "aria_rgb_native": ("native_aria_rgb", np.eye(3)),
    "egosuite_opencv_camera_to_world": ("egosuite_opencv", EE4D_CAMERA_FROM_REFERENCE),
}


def resolve_camera_basis(source_basis, config):
    if "camera_from_ee4d_camera" in config:
        policy, correction = "manual", config["camera_from_ee4d_camera"]
        default_verified = False
    elif source_basis in CAMERA_BASIS_POLICIES:
        policy, correction = CAMERA_BASIS_POLICIES[source_basis]
        default_verified = True
    else:
        policy, correction, default_verified = "unknown", np.eye(3), False
    correction = np.asarray(correction, dtype=float)
    reference_rotation(correction)  # Validate even when the output mask is false.
    return policy, correction, bool(config.get("basis_verified", default_verified))


def reference_rotation(camera_from_ee4d_camera):
    correction = np.asarray(camera_from_ee4d_camera, dtype=float)
    if correction.shape != (3, 3):
        raise ValueError("head_reference.camera_from_ee4d_camera must be a 3x3 proper rotation")
    transform = np.eye(4)
    transform[:3, :3] = correction
    rigid_transform(transform)
    return correction @ EE4D_CAMERA_FROM_REFERENCE


def floor_reference(position, quat, position_valid, rotation_valid, ground, basis_verified):
    forward = quat_to_matrix(quat)[:, 0, :2, 0]
    length = np.linalg.norm(forward, axis=-1)
    horizontal = np.divide(forward, length[:, None], out=np.zeros_like(forward), where=length[:, None] > 1e-6)
    heading = np.broadcast_to(np.eye(3), (len(quat), 3, 3)).copy()
    heading[:, 0, 0], heading[:, 1, 0] = horizontal[:, 0], horizontal[:, 1]
    heading[:, 0, 1], heading[:, 1, 1] = -horizontal[:, 1], horizontal[:, 0]
    degenerate = length <= 1e-6
    heading[degenerate] = np.eye(3)
    projected = position.copy()
    projected[:, :, 2] = ground["height_m"]
    valid = position_valid & rotation_valid & (~degenerate)[:, None] & bool(ground["valid"]) & bool(basis_verified)
    projected = np.where(valid[..., None], projected, 0)
    q = matrix_to_quat(heading)[:, None]
    q[~valid] = [1, 0, 0, 0]
    return projected, q, valid


def positions_in_floor_reference(position, position_valid, floor_position, floor_quat, floor_valid):
    """Dense positions in the camera floor frame; no joint rotation basis needed.

    The origin is the camera's XY projected onto the world floor. Its local Z
    stays world-up, so joint Z is height above the floor, not camera-relative Z.
    """
    rotation = quat_to_matrix(floor_quat)[:, 0]
    local = np.einsum("tji,tkj->tki", rotation, position-floor_position)
    valid = position_valid & floor_valid
    return np.where(valid[..., None], local, 0), valid


def derive_ground_features(arrays, ground, basis_verified):
    """Rebuild every floor-dependent feature from full world poses in-place.

    Shared by conversion and training-window reads. Never reuse archived local
    zeros after resolving a missing floor in a selected training window.
    """
    count = len(arrays["timestamp"])
    arrays["observation.environment.floor_height"] = np.full(count, ground["height_m"], np.float32)
    arrays["observation.environment.floor_normal"] = np.broadcast_to([0., 0., 1.], (count, 3)).astype(np.float32)
    arrays["observation.valid.floor"] = np.full(count, ground["valid"], bool)
    arrays["observation.valid.floor_verified"] = np.full(count, ground.get("verified", False) and ground["valid"], bool)
    arrays["observation.valid.head_reference_basis"] = np.full(count, basis_verified, bool)
    head_position_valid = arrays["observation.valid.head_reference_position"].reshape(count, 1)
    head_rotation_valid = arrays["observation.valid.head_reference_rotation"].reshape(count, 1)
    floor_pos, floor_rot, floor_mask = floor_reference(
        arrays["observation.state.head_reference_world"].reshape(count, 1, 3),
        arrays["observation.state.head_reference_world_rotation"].reshape(count, 1, 4),
        head_position_valid, head_rotation_valid, ground, basis_verified)
    arrays["observation.state.head_reference_floor_world"] = floor_pos.reshape(count, 3).astype(np.float32)
    arrays["observation.state.head_reference_floor_world_rotation"] = floor_rot.reshape(count, 4).astype(np.float32)
    interpolated = (arrays["observation.interpolated.head_reference_position"].reshape(count, 1)
                    | arrays["observation.interpolated.head_reference_rotation"].reshape(count, 1)) & floor_mask
    for kind in ["position", "rotation"]:
        arrays[f"observation.valid.head_reference_floor_{kind}"] = floor_mask.copy()
        arrays[f"observation.interpolated.head_reference_floor_{kind}"] = interpolated.copy()
    body_local, body_mask = positions_in_floor_reference(
        arrays["observation.state.body_world"].reshape(count, 22, 3),
        arrays["observation.valid.body_position"], floor_pos, floor_rot, floor_mask)
    arrays["observation.state.body_head_floor"] = body_local.reshape(count, 66).astype(np.float32)
    arrays["observation.valid.body_head_floor_position"] = body_mask
    arrays["observation.interpolated.body_head_floor_position"] = (
        arrays["observation.interpolated.body_position"] | interpolated) & body_mask
    height_mask = head_position_valid[:, 0] & bool(ground["valid"])
    height = arrays["observation.state.head_reference_world"][:, 2]-ground["height_m"]
    arrays["observation.state.head_reference_height"] = np.where(height_mask, height, 0).astype(np.float32)
    arrays["observation.valid.head_reference_height"] = height_mask
    return arrays


def attach_head_reference(episode, config=None):
    config = config or {}
    mode = config.get("mode", "ee4d")
    if mode not in {"ee4d", "disabled"}:
        raise ValueError("head_reference.mode must be ee4d or disabled")
    if mode == "disabled":
        for key in ["head_reference", "head_reference_floor"]:
            episode.position_valid[key][:] = False
            episode.rotation_valid[key][:] = False
        episode.metadata["head_reference"] = {"mode": mode, "basis_verified": False}
        return episode.check()
    source_basis = episode.metadata.get("camera_pose_basis", "unspecified")
    policy, correction, verified = resolve_camera_basis(source_basis, config)
    rotation = reference_rotation(correction)
    position = episode.positions["camera"].copy()
    quat = quat_multiply(episode.rotations["camera"], matrix_to_quat(rotation))
    episode.positions["head_reference"] = position
    episode.rotations["head_reference"] = quat
    episode.position_valid["head_reference"] = episode.position_valid["camera"].copy()
    episode.rotation_valid["head_reference"] = episode.rotation_valid["camera"].copy()
    episode.metadata["head_reference"] = {
        "mode": mode, "definition": "EE4D_egoego_reference_at_camera_origin", "basis_verified": verified,
        "basis_policy": policy, "source_camera_basis": source_basis,
        "basis_verification_semantics": "documented_axis_convention_or_explicit_manual_override; not_physical_mount_calibration",
        "camera_from_ee4d_camera": correction.tolist(),
        "ee4d_camera_from_reference": EE4D_CAMERA_FROM_REFERENCE.tolist(),
        "native_camera_from_reference": rotation.tolist(),
        "translation_offset_m": [0., 0., 0.],
        "position_semantics": "selected_ego_camera_optical_center",
        "floor_projection": "project_reference_local_X_to_world_XY; z=floor_height",
        "body_head_floor": "R_world_floor.T @ (body_world - camera_floor_world); 22 positions; meters",
        "native_head_preserved": episode.metadata.get("head_semantics", "unspecified")}
    ground = episode.metadata["ground"]
    floor_position, floor_rotation, floor_valid = floor_reference(
        position, quat, episode.position_valid["head_reference"], episode.rotation_valid["head_reference"], ground, verified)
    episode.positions["head_reference_floor"] = floor_position
    episode.rotations["head_reference_floor"] = floor_rotation
    episode.position_valid["head_reference_floor"] = floor_valid.copy()
    episode.rotation_valid["head_reference_floor"] = floor_valid.copy()
    return episode.check()
