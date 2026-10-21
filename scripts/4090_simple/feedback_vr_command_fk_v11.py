"""Mode1 expert with recorded controller-command wrist FK world targets.

The audited source writer records ``action.eef`` as fixed-base, pelvis-relative
FK of the controller q_target, at wrist_yaw_link origins, in xyz+wxyz order.
It is NOT the missing original VR input. This new expert substitutes those
command wrist poses for measured wrist FK, retaining the previous expert's
measured torso-derived head, future legs, heading and actual-pelvis feedback.
The pinned C++ IK wrist tip offset is applied exactly once, before world mapping.
"""
from __future__ import annotations

import numpy as np

from feedback_vr_fk_v11 import (FeedbackVrFkEncoder, TIP_OFFSETS, conjugate,
                               matrix_wxyz, quaternion_multiply, source_world_fk)


def wrist_poses(values, frames, label):
    value = np.asarray(values, dtype=np.float64)
    if value.shape != (frames, 14) or not np.isfinite(value).all():
        raise ValueError(f"{label} must contain aligned finite real xyz+wxyz wrist poses [N,14]")
    poses = value.reshape(frames, 2, 7).copy()
    matrix_wxyz(poses[:, :, 3:])
    poses[:, :, 3:] /= np.linalg.norm(poses[:, :, 3:], axis=-1, keepdims=True)
    return poses


def command_world_fk(model, states43, root_xyz, root_wxyz, action_eef, observation_eef):
    """Verify the recorded pelvis frame against independent FK; change wrists only."""
    positions = np.asarray(root_xyz, dtype=np.float64)
    roots = np.asarray(root_wxyz, dtype=np.float64)
    n = len(states43)
    command = wrist_poses(action_eef, n, "action.eef")
    observation = wrist_poses(observation_eef, n, "observation.eef_state")
    points, orientations, metadata = source_world_fk(model, states43, positions, roots)
    root_rotation = matrix_wxyz(roots)
    wrist_rotation = matrix_wxyz(orientations[:, :2])
    wrist_origins = points[:, :2] - np.einsum("nkij,kj->nki", wrist_rotation, TIP_OFFSETS[:2])
    local_origins = np.einsum("nji,nkj->nki", root_rotation, wrist_origins - positions[:, None])
    local_quats = quaternion_multiply(conjugate(roots[:, None]), orientations[:, :2])
    xyz_error = local_origins - observation[:, :, :3]
    rotation_error = matrix_wxyz(local_quats) - matrix_wxyz(observation[:, :, 3:])
    # Source xyz are float32 recordings of fixed-base Pinocchio FK; the exact
    # robot's independent MuJoCo FK agrees to <4e-7m RMSE on TRAIN 0/1/2.
    if np.max(np.abs(xyz_error)) > 1e-5 or np.max(np.abs(rotation_error)) > 1e-5:
        raise ValueError("Recorded observation.eef_state does not match pelvis-relative wrist_yaw_link FK")
    command_local_tips = command[:, :, :3] + np.einsum(
        "nkij,kj->nki", matrix_wxyz(command[:, :, 3:]), TIP_OFFSETS[:2])
    new_points, new_orientations = points.copy(), orientations.copy()
    new_points[:, :2] = positions[:, None] + np.einsum("nij,nkj->nki", root_rotation, command_local_tips)
    normalized_roots = roots / np.linalg.norm(roots, axis=-1, keepdims=True)
    new_orientations[:, :2] = quaternion_multiply(normalized_roots[:, None], command[:, :, 3:])
    matrix_wxyz(new_orientations)
    metadata = dict(metadata, wrist_pose_source="action.eef: controller q_target fixed-base wrist_yaw_link FK",
                    recorded_pose_frame="source pelvis relative; xyz+wxyz; no tip offset in recorded pose",
                    head_pose_source="unchanged measured source torso FK plus pinned .35m tip offset",
                    observation_fk_xyz_max_abs_m=float(np.max(np.abs(xyz_error))),
                    observation_fk_rotation_matrix_max_abs=float(np.max(np.abs(rotation_error))),
                    new_reference="Command-FK wrist world targets with measured torso-derived head; not recovered original VR commands")
    return new_points, new_orientations, metadata


class FeedbackVrCommandFkEncoder(FeedbackVrFkEncoder):
    """Same official mode1 encoder/layout; a separately named wrist reference."""
    def __init__(self, values, sonic):
        super().__init__(values, sonic)
        n = len(self.states)
        self.command = wrist_poses(values["reference_full_action_eef"], n, "action.eef").reshape(n, 14)
        self.observation = wrist_poses(values["reference_full_observation_eef"], n, "observation.eef_state").reshape(n, 14)

    def __call__(self, frame, raw):
        if self.points is None:
            self.points, self.orientations, self.fk_metadata = command_world_fk(
                raw.mujoco.mjModel, self.states, self.root_xyz, self.roots, self.command, self.observation)
        token, metadata = super().__call__(frame, raw)
        metadata.update(mode="feedback-vr-command-fk-v11", changed_from_measured_fk="Only the two wrist reference world poses",
                        source_command_eef14=self.command[frame].tolist(),
                        source_observation_eef14=self.observation[frame].tolist(),
                        semantics="New command-FK world wrist expert; measured torso head retained; original VR/planner commands remain unavailable")
        return token, metadata
