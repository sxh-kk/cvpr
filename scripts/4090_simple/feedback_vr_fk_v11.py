"""New mode1 world-FK reference expert; not recovery of missing teleop commands.

The original measured trajectory defines fixed world wrist/head reference
points through the same robot's FK. Actual pelvis translation and full
orientation express these targets in the current robot frame. Source future
legs remain the locomotion reference. Only CPU MuJoCo FK and official ONNX run.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from prepare_carrybox_v11 import (ENCODER_SHA, FUTURE_OFFSETS, POLICY_TO_BODY, TRAIN_STATE_NAMES,
                                  matrix_wxyz, sha256)

LOWER_POLICY_INDICES = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18])
TIP_OFFSETS = np.array([[.18, -.025, 0.], [.18, .025, 0.], [0., 0., .35]])
MODE1_SLICES = ((0, 4), (644, 650), (650, 770), (770, 890), (890, 899), (899, 911))


def conjugate(quaternion):
    return np.asarray(quaternion) * np.array([1., -1., -1., -1.])


def quaternion_multiply(left, right):
    a, b = np.asarray(left), np.asarray(right)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), axis=-1)


def mode1_input(body_q, body_dq, root_wxyz, world_points, world_quaternions,
                frame, actual_position, actual_wxyz):
    """Pinned C++ layout/mask, with explicitly NEW actual-root world feedback."""
    q, dq, roots, points, quaternions = (np.asarray(value, dtype=np.float64) for value in
                                        (body_q, body_dq, root_wxyz, world_points, world_quaternions))
    n = len(q)
    if (q.shape != (n, 29) or dq.shape != q.shape or roots.shape != (n, 4)
            or points.shape != (n, 3, 3) or quaternions.shape != (n, 3, 4)):
        raise ValueError("Mode1 needs aligned real source q/dq/root and three world FK poses")
    future = int(frame) + FUTURE_OFFSETS
    if int(frame) != frame or frame < 0 or future[-1] >= n:
        raise ValueError("Mode1 requires complete genuine step5 future legs, without padding")
    position, quaternion = np.asarray(actual_position), np.asarray(actual_wxyz)
    if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all():
        raise ValueError("Actual pelvis must provide its real world position and wxyz")
    actual_rotation = matrix_wxyz(quaternion)
    matrix_wxyz(quaternions[frame])  # Also enforces finite unit reference quaternions.
    local_points = np.einsum("ij,kj->ki", actual_rotation.T, points[frame] - position)
    local_quaternions = quaternion_multiply(conjugate(quaternion), quaternions[frame])
    yaw = np.arctan2(actual_rotation[1, 0], actual_rotation[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    heading_inverse = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
    heading = heading_inverse @ matrix_wxyz(roots[frame])
    value = np.zeros((1, 1751), dtype=np.float32)
    value[0, 0] = 1.  # Scalar mode ID followed by three zero fillers.
    value[0, 644:650] = heading[:, :2].reshape(-1)
    value[0, 650:770] = q[future][:, POLICY_TO_BODY][:, LOWER_POLICY_INDICES].reshape(-1)
    value[0, 770:890] = dq[future][:, POLICY_TO_BODY][:, LOWER_POLICY_INDICES].reshape(-1)
    value[0, 890:899] = local_points.reshape(-1)
    value[0, 899:911] = local_quaternions.reshape(-1)  # Three wxyz, not 6D rotations.
    if not np.isfinite(value).all():
        raise ValueError("World-FK expert produced a nonfinite encoder input")
    return value


def source_world_fk(model, states43, root_xyz, root_wxyz):
    """Use independent MjData; never overwrite live simulation or invent head input."""
    import mujoco
    states, positions, quaternions = (np.asarray(value, dtype=np.float64) for value in
                                     (states43, root_xyz, root_wxyz))
    n = len(states)
    if states.shape != (n, 43) or positions.shape != (n, 3) or quaternions.shape != (n, 4):
        raise ValueError("Source FK requires complete measured state43 and real world root poses")
    if not np.isfinite(states).all() or not np.isfinite(positions).all():
        raise ValueError("Source FK input is nonfinite")
    matrix_wxyz(quaternions)
    data = mujoco.MjData(model)
    # Canonical C++ bodies: left wrist, right wrist, torso. Resolve by joint
    # attachment so visual mesh aliases cannot change the coordinate frame.
    body_ids = [int(model.joint(name).bodyid[0]) for name in
                ("left_wrist_yaw_joint", "right_wrist_yaw_joint", "waist_pitch_joint")]
    expected_names = ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link")
    names = tuple(model.body(index).name for index in body_ids)
    if names != expected_names:
        raise ValueError(f"Source robot FK bodies differ from the pinned G1 protocol: {names}")
    points, orientations = np.empty((n, 3, 3)), np.empty((n, 3, 4))
    for frame in range(n):
        data.qpos[:] = model.qpos0
        data.qpos[:7] = np.r_[positions[frame], quaternions[frame]]
        for name, value in zip(TRAIN_STATE_NAMES, states[frame]):
            data.joint(name).qpos[0] = value
        mujoco.mj_forward(model, data)
        for tip, body in enumerate(body_ids):
            points[frame, tip] = data.xpos[body] + data.xmat[body].reshape(3, 3) @ TIP_OFFSETS[tip]
            orientations[frame, tip] = data.xquat[body]
    if not np.isfinite(points).all() or not np.isfinite(orientations).all():
        raise ValueError("Source world FK is nonfinite")
    matrix_wxyz(orientations)
    return points, orientations, {"bodies": list(names), "tip_offsets_local_m": TIP_OFFSETS.tolist(),
                                   "source_frames": n, "robot_model_nq": model.nq,
                                   "new_reference": "Source measured q/root FK world wrist points and torso-derived head; not original teleop commands"}


class FeedbackVrFkEncoder:
    def __init__(self, values, sonic):
        import onnxruntime as ort
        encoder = Path(sonic) / "model_encoder.onnx"
        if sha256(encoder) != ENCODER_SHA:
            raise ValueError("World-FK expert requires the fixed official SONIC1.1 encoder")
        self.states = np.asarray(values["reference_full_state43"], dtype=np.float64)
        self.q = self.states[:, :29]
        self.dq = np.asarray(values["reference_full_body_dq"], dtype=np.float64)
        self.root_xyz = np.asarray(values["reference_full_root_xyz"], dtype=np.float64)
        self.roots = np.asarray(values["reference_full_root_wxyz"], dtype=np.float64)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(encoder), sess_options=options, providers=["CPUExecutionProvider"])
        if list(self.session.get_inputs()[0].shape) != [1, 1751] or list(self.session.get_outputs()[0].shape) != [1, 64]:
            raise ValueError("Unexpected mode1 encoder tensor contract")
        self.points = self.orientations = self.fk_metadata = None

    def __call__(self, frame, raw):
        model, data = raw.mujoco.mjModel, raw.mujoco.mjData
        if self.points is None:
            self.points, self.orientations, self.fk_metadata = source_world_fk(model, self.states, self.root_xyz, self.roots)
        pelvis = model.body("pelvis").id
        position, quaternion = data.xpos[pelvis].copy(), data.xquat[pelvis].copy()
        value = mode1_input(self.q, self.dq, self.roots, self.points, self.orientations, frame, position, quaternion)
        token = self.session.run(None, {self.session.get_inputs()[0].name: value})[0][0]
        if token.shape != (64,) or not np.isfinite(token).all():
            raise ValueError("Official mode1 encoder produced an invalid token")
        metadata = {"mode": "feedback-vr-fk-v11", "encoder_mode": 1, "source_frame_index": int(frame),
                    "reference_frame_indices": (int(frame) + FUTURE_OFFSETS).tolist(),
                    "encoder_input_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                    "active_slices": [list(pair) for pair in MODE1_SLICES],
                    "actual_pelvis_xyz": position.tolist(), "actual_pelvis_wxyz": quaternion.tolist(),
                    "reference_world_points": self.points[frame].tolist(),
                    "reference_world_wxyz": self.orientations[frame].tolist(),
                    "reference_fk": self.fk_metadata, "reference_anchor_A": "identity; fixed source world reference",
                    "semantics": "New explicit world-FK reference expert, not recovery of missing original head/wrist commands"}
        return token, metadata
