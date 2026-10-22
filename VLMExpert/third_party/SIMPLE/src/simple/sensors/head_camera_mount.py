"""G1 head-mesh/D435 calibration and explicit USD/MuJoCo frame conversion.

SIMPLE: SIMulation-based Policy Learning and Evaluation
Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

The actual USD keeps its original torso joint frames, while SONIC's MJCF moves
the waist-roll/pitch origins. A torso-origin difference is therefore a function
of the measured waist pose, not an arbitrary fitted constant translation.
These constants come from the local source assets named and hashed below.
This module does not move the Isaac camera, edit assets or step either engine.
"""
from __future__ import annotations

import numpy as np


CALIBRATION_CONTRACT = "g1-head-mesh-d435-source-asset-calibration-v1"
CALIBRATED_HEAD_MESH_SHA256 = "005fb67fbd3eff94aa8bf4a6e83238174e9f91b6721f7111594322f223724411"
SOURCE_IDENTITIES = {
    "robots/g1/g1_29dof_wholebody_dex3.urdf": "19f8054183088bfd95a754322e5adf72fa400a16ea38b5c121a9ab596453f54a",
    "robots/g1/g1_29dof_wholebody_dex3.usd": "ca45b13a503b0f74234c68e8ac7accfb0c982a69b2f0891a3b4f0145480e0adf",
    "robots/g1/g1_29dof_with_dex3_base_fix.usd": "d23b9bbc8b59cf20fa1983f55874f855a76e038b9e22eae39542e62be5b4ba3d",
    "robots/g1/meshes/head_link.STL": CALIBRATED_HEAD_MESH_SHA256,
}
# CPU Usd.Stage inspection of the composed source asset, relative to head_link.
HEAD_MESH_TO_D435_POSITION = np.array([0.05365999788045883, 0.01752999983727932, 0.47387000918388367])
_pitch = 0.8307767239493009  # source URDF d435_joint rpy
HEAD_MESH_TO_D435_QUATERNION = np.array([np.cos(_pitch / 2), 0., np.sin(_pitch / 2), 0.])
USD_WAIST_ROLL_POSITION = np.array([-0.003963499795645475, 0., 0.04399999976158142])
USD_TORSO_POSITION_IN_ROLL = np.zeros(3)
USD_TORSO_TO_D435_POSITION = HEAD_MESH_TO_D435_POSITION - USD_WAIST_ROLL_POSITION
USD_SHOULDER_PITCH_POSITIONS = {
    "left": np.array([0.0039563, 0.10022, 0.24778], dtype=np.float32).astype(np.float64),
    "right": np.array([0.0039563, -0.10021, 0.24778], dtype=np.float32).astype(np.float64),
}


def _vector(value, size, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain {size} finite values")
    return array


def _quaternion(value):
    q = _vector(value, 4, "quaternion wxyz")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Quaternion must have nonzero norm")
    return q / norm


def rotation(value):
    w, x, y, z = _quaternion(value)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _matrix(value, label):
    r = np.asarray(value, dtype=np.float64)
    if (r.shape != (3, 3) or not np.isfinite(r).all()
            or not np.allclose(r.T @ r, np.eye(3), rtol=0, atol=1e-6)
            or not np.isclose(np.linalg.det(r), 1., rtol=0, atol=1e-6)):
        raise ValueError(f"{label} must be a proper rotation")
    return r


def _multiply(a, b):
    w, x, y, z = _quaternion(a)
    v, i, j, k = _quaternion(b)
    return _quaternion([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                        w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def head_camera_mount(geometries, camera_position=(0., 0., 0.)):
    """Return torso-local D435 sensor pose from supported actual head geometry.

    Each mapping has mesh_name, mesh_sha256, position, quaternion and scale.
    Duplicate collision/visual geometries must describe the same rigid pose.
    Unsupported/changed head meshes fail closed rather than guessing a model.
    Camera position is the source CameraEntity offset in the D435 frame; the
    stereo right offset is [0,-baseline,0], shared with Isaac's unchanged mount.
    """
    heads = [g for g in geometries if g.get("mesh_name") == "head_link"]
    if not heads:
        raise ValueError("Supported named head_link mesh calibration is required")
    poses = []
    for geom in heads:
        if geom.get("mesh_sha256") != CALIBRATED_HEAD_MESH_SHA256:
            raise ValueError("Changed head_link mesh has no source-bound camera calibration")
        if not np.array_equal(_vector(geom.get("scale", (1., 1., 1.)), 3, "head mesh scale"), np.ones(3)):
            raise ValueError("Scaled head mesh has no source-bound camera calibration")
        p = _vector(geom.get("position", (0., 0., 0.)), 3, "head geometry position")
        q = _quaternion(geom.get("quaternion", (1., 0., 0., 0.)))
        poses.append((p, q))
    p, q = poses[0]
    if any(not np.allclose(other_p, p, rtol=0, atol=1e-10)
           or not np.allclose(rotation(other_q), rotation(q), rtol=0, atol=1e-10)
           for other_p, other_q in poses[1:]):
        raise ValueError("Conflicting head_link visual/collision geometry poses")
    mount_q = _multiply(q, HEAD_MESH_TO_D435_QUATERNION)
    mount_p = p + rotation(q) @ HEAD_MESH_TO_D435_POSITION
    return mount_p + rotation(mount_q) @ _vector(camera_position, 3, "camera offset"), mount_q


def usd_torso_frame_position(mj_torso_position, waist_yaw_rotation, waist_roll_rotation,
                             mj_waist_roll_position, mj_torso_position_in_roll):
    """Predict USD torso origin from actual MJ FK and fixed source joint frames.

    Arguments use body names torso_link, waist_yaw_link, waist_roll_link.
    Read rotations from data.xmat and relative positions from model.body_pos.
    Both source chains have identical joint axes and identity body rotations;
    callers must verify those model assumptions. USD torso orientation is the
    measured MJ torso orientation. No first-frame fitting or state forcing.
    """
    return (_vector(mj_torso_position, 3, "MJ torso position")
            + _matrix(waist_yaw_rotation, "waist yaw rotation") @
            (USD_WAIST_ROLL_POSITION - _vector(mj_waist_roll_position, 3, "MJ waist roll position"))
            + _matrix(waist_roll_rotation, "waist roll rotation") @
            (USD_TORSO_POSITION_IN_ROLL - _vector(mj_torso_position_in_roll, 3, "MJ torso position in roll")))


def usd_head_camera_frame_position(usd_torso_position, mj_torso_rotation, camera_position=(0., 0., 0.)):
    """Predict actual unchanged USD sensor origin after the source chain mapping.

    First obtain usd_torso_position with usd_torso_frame_position. This keeps
    geometric frame conversion separate from raw MJ camera-vs-USD residuals.
    """
    local = USD_TORSO_TO_D435_POSITION + rotation(HEAD_MESH_TO_D435_QUATERNION) @ _vector(camera_position, 3, "camera offset")
    return _vector(usd_torso_position, 3, "USD torso position") + _matrix(mj_torso_rotation, "MJ torso rotation") @ local


def usd_wrist_frame_position(mj_wrist_position, mj_torso_position, usd_torso_position,
                            mj_torso_rotation, mj_shoulder_pitch_position, side):
    """Map unchanged arm-chain FK after its source shoulder-origin difference.

    Source USD shoulder pitch position is10mm above the SONIC MJCF position.
    Inspected downstream arm origins agree within5e-9m; source quaternion
    rounding differs by<5e-7. This is a known-asset translation mapping, not a
    general retargeter. The caller must bind the listed source assets/model.
    """
    if side not in USD_SHOULDER_PITCH_POSITIONS:
        raise ValueError("Wrist side must be left or right")
    return (_vector(mj_wrist_position, 3, "MJ wrist position")
            + _vector(usd_torso_position, 3, "USD torso position")
            - _vector(mj_torso_position, 3, "MJ torso position")
            + _matrix(mj_torso_rotation, "MJ torso rotation") @
            (USD_SHOULDER_PITCH_POSITIONS[side] - _vector(mj_shoulder_pitch_position, 3, "MJ shoulder pitch position")))
