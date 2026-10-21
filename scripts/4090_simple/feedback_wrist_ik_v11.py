"""Bounded mode0 world-wrist IK reference expert, without object feedback.

Source measured q/root define fixed world wrist poses. Only arm references
change to compensate the observed root drift; legs, waist, hand commands,
controller gains and all live simulator state remain untouched.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np

from prepare_carrybox_v11 import ENCODER_SHA, FUTURE_OFFSETS, TRAIN_STATE_NAMES, encoder_input, matrix_wxyz, sha256

ARM_BODY_INDICES = np.arange(15, 29)
TIP_OFFSETS = np.array([[.18, -.025, 0.], [.18, .025, 0.]])
JOINT_DELTA_LIMIT_RAD = .20
POSITION_DELTA_LIMIT_M = .06
ORIENTATION_DELTA_LIMIT_RAD = np.deg2rad(10.)


def rotation_error(target, current):
    """Shortest world-axis rotation vector, compatible with a world Jacobian."""
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(target @ current.T).reshape(-1))
    if quaternion[0] < 0:
        quaternion *= -1
    length = np.linalg.norm(quaternion[1:])
    return np.zeros(3) if length < 1e-14 else quaternion[1:] * (2*np.arctan2(length, quaternion[0])/length)


def rotation_from_vector(vector):
    angle = np.linalg.norm(vector)
    if angle < 1e-14:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross @ cross)


def clipped_vector(vector, maximum):
    length = np.linalg.norm(vector)
    return np.asarray(vector) * min(1., maximum/max(length, 1e-15))


class WorldWristIK:
    """FK/IK work uses its own MjData; it never changes the live simulation."""
    def __init__(self, model, states43, root_xyz, root_wxyz):
        self.model = model
        self.states = np.asarray(states43, dtype=np.float64)
        self.positions = np.asarray(root_xyz, dtype=np.float64)
        self.rotations = matrix_wxyz(root_wxyz)
        self.quaternions = np.asarray(root_wxyz, dtype=np.float64)
        n = len(self.states)
        if self.states.shape != (n, 43) or self.positions.shape != (n, 3):
            raise ValueError("World-wrist IK requires complete named state43/root observations")
        if not np.isfinite(self.states).all() or not np.isfinite(self.positions).all():
            raise ValueError("Nonfinite source world-wrist reference")
        self.data = mujoco.MjData(model)
        self.qadr = np.array([model.joint(name).qposadr[0] for name in TRAIN_STATE_NAMES])
        self.root_qadr = int(model.joint("floating_base_joint").qposadr[0])
        self.wrists = [model.body(side+"_wrist_yaw_link").id for side in ("left", "right")]
        self.arm_dofs = [[int(model.joint(name).dofadr[0]) for name in TRAIN_STATE_NAMES[start:start+7]]
                         for start in (15, 22)]
        self.points, self.orientations = np.empty((n, 2, 3)), np.empty((n, 2, 3, 3))
        for frame in range(n):
            self._set_reference(self.states[frame], self.positions[frame], self.rotations[frame])
            for side in range(2):
                self.points[frame, side], self.orientations[frame, side] = self._pose(side)

    def _set_reference(self, state, position, rotation):
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.qadr] = state
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(rotation).reshape(-1))
        self.data.qpos[self.root_qadr:self.root_qadr+7] = np.r_[position, quaternion]
        mujoco.mj_forward(self.model, self.data)

    def _pose(self, side):
        body = self.wrists[side]
        rotation = self.data.xmat[body].reshape(3, 3).copy()
        return self.data.xpos[body] + rotation @ TIP_OFFSETS[side], rotation

    def _solve_arm(self, side, source_state, target_point, target_rotation):
        indices = np.arange(15+side*7, 22+side*7)
        qadr, dofs = self.qadr[indices], self.arm_dofs[side]
        original = source_state[indices]
        lower, upper = original-JOINT_DELTA_LIMIT_RAD, original+JOINT_DELTA_LIMIT_RAD
        for index, name in enumerate(np.asarray(TRAIN_STATE_NAMES)[indices]):
            joint = self.model.joint(name)
            if bool(joint.limited[0]):
                lower[index] = max(lower[index], joint.range[0])
                upper[index] = min(upper[index], joint.range[1])
        if np.any(lower > upper):
            raise ValueError("Source arm is outside the joint limits plus the bounded correction radius")
        before_point, before_rotation = self._pose(side)
        position_delta = target_point-before_point
        orientation_delta = rotation_error(target_rotation, before_rotation)
        goal_point = before_point+clipped_vector(position_delta, POSITION_DELTA_LIMIT_M)
        goal_rotation = rotation_from_vector(clipped_vector(orientation_delta, ORIENTATION_DELTA_LIMIT_RAD)) @ before_rotation
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        iterations = 0
        for iteration in range(25):
            point, rotation = self._pose(side)
            position_error = goal_point-point
            angle_error = rotation_error(goal_rotation, rotation)
            if np.linalg.norm(position_error) < 1e-4 and np.linalg.norm(angle_error) < 1e-3:
                break
            mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.wrists[side])
            jacobian = np.vstack((jacp[:, dofs], .15*jacr[:, dofs]))
            error = np.r_[position_error, .15*angle_error]
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .01**2*np.eye(6), error)
            candidate = np.clip(self.data.qpos[qadr]+np.clip(delta, -.03, .03), lower, upper)
            change = np.max(abs(candidate-self.data.qpos[qadr]))
            self.data.qpos[qadr] = candidate
            mujoco.mj_forward(self.model, self.data)
            iterations = iteration+1
            if change < 1e-7:
                break
        after_point, after_rotation = self._pose(side)
        correction = self.data.qpos[qadr]-original
        return correction.copy(), {"iterations":iterations,
            "fixed_source_world_point":target_point.tolist(), "bounded_world_goal_point":goal_point.tolist(),
            "source_goal_position_error_before_m":float(np.linalg.norm(position_delta)),
            "source_goal_position_error_after_m":float(np.linalg.norm(target_point-after_point)),
            "bounded_goal_position_residual_m":float(np.linalg.norm(goal_point-after_point)),
            "source_goal_orientation_error_before_rad":float(np.linalg.norm(orientation_delta)),
            "source_goal_orientation_error_after_rad":float(np.linalg.norm(rotation_error(target_rotation, after_rotation))),
            "joint_bound_active":bool(np.any(abs(correction) >= JOINT_DELTA_LIMIT_RAD-1e-6)),
            "position_goal_clipped":bool(np.linalg.norm(position_delta)>POSITION_DELTA_LIMIT_M),
            "orientation_goal_clipped":bool(np.linalg.norm(orientation_delta)>ORIENTATION_DELTA_LIMIT_RAD)}

    def correct(self, frame, actual_position, actual_rotation):
        future = int(frame)+FUTURE_OFFSETS
        if frame < 0 or future[-1] >= len(self.states):
            raise ValueError("Need complete real future q/root; never pad an IK reference")
        # Explicit forecast: the observed current source-to-actual rigid drift
        # stays constant across this command's 0.9 s reference horizon.
        drift_rotation = actual_rotation @ self.rotations[frame].T
        predicted_positions = np.asarray(actual_position) + np.einsum(
            "ij,nj->ni", drift_rotation, self.positions[future]-self.positions[frame])
        predicted_rotations = drift_rotation[None] @ self.rotations[future]
        corrected = self.states[future, :29].copy()
        diagnostics = []
        for ordinal, source_frame in enumerate(future):
            self._set_reference(self.states[source_frame], predicted_positions[ordinal], predicted_rotations[ordinal])
            per_side = []
            for side in range(2):
                delta, diagnostic = self._solve_arm(side, self.states[source_frame],
                    self.points[source_frame, side], self.orientations[source_frame, side])
                corrected[ordinal, 15+side*7:22+side*7] += delta
                per_side.append(diagnostic)
            diagnostics.append(per_side)
        return corrected, {"future_source_frames":future.tolist(),
            "actual_root_xyz":np.asarray(actual_position).tolist(), "actual_root_rotation":actual_rotation.tolist(),
            "predicted_future_root_xyz":predicted_positions.tolist(),
            "root_forecast":"constant currently observed source-to-actual SE3 drift; not future observation",
            "wrist_diagnostics_left_right":diagnostics}


class FeedbackWristIkEncoder:
    def __init__(self, values, sonic):
        import onnxruntime as ort
        path = Path(sonic)/"model_encoder.onnx"
        if sha256(path) != ENCODER_SHA:
            raise ValueError("World-wrist IK requires the fixed official SONIC1.1 encoder")
        self.states = np.asarray(values["reference_full_state43"], dtype=np.float64)
        self.q = self.states[:, :29]
        self.dq = np.asarray(values["reference_full_body_dq"], dtype=np.float64)
        self.xyz = np.asarray(values["reference_full_root_xyz"], dtype=np.float64)
        self.roots = np.asarray(values["reference_full_root_wxyz"], dtype=np.float64)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        if list(self.session.get_inputs()[0].shape) != [1,1751] or list(self.session.get_outputs()[0].shape) != [1,64]:
            raise ValueError("Official IK reference encoder contract mismatch")
        self.ik = None

    def __call__(self, frame, raw):
        model, data = raw.mujoco.mjModel, raw.mujoco.mjData
        if self.ik is None:
            self.ik = WorldWristIK(model, self.states, self.xyz, self.roots)
        pelvis = model.body("pelvis").id
        position = data.xpos[pelvis].copy()
        rotation = data.xmat[pelvis].reshape(3,3).copy()
        future = int(frame)+FUTURE_OFFSETS
        corrected, diagnostics = self.ik.correct(frame, position, rotation)
        q, dq = self.q.copy(), self.dq.copy()
        q[future] = corrected
        delta = corrected-self.q[future]
        # Only arm derivative references change. The finite-difference of the
        # bounded correction curve augments the original measured native dq.
        dq[future,15:29] += np.gradient(delta[:,15:29], .1, axis=0, edge_order=1)
        value = encoder_input(q,dq,self.roots,frame)
        yaw = np.arctan2(rotation[1,0],rotation[0,0])
        c,s = np.cos(yaw),np.sin(yaw)
        value[0,584:644] = (np.array([[c,s,0],[-s,c,0],[0,0,1]])[None] @ matrix_wxyz(self.roots[future]))[...,:2].reshape(-1)
        token = self.session.run(None,{self.session.get_inputs()[0].name:value})[0][0]
        if token.shape != (64,) or not np.isfinite(token).all():
            raise ValueError("Bounded wrist IK produced a nonfinite official token")
        return token,{"mode":"feedback-wrist-ik-v11", "encoder_mode":0,"source_frame_index":int(frame),
            "reference_frame_indices":future.tolist(), "encoder_input_sha256":hashlib.sha256(value.tobytes()).hexdigest(),
            "reference_anchor_A":"identity; original source world wrist targets remain fixed",
            "joint_delta_limit_rad":JOINT_DELTA_LIMIT_RAD,"position_delta_limit_m":POSITION_DELTA_LIMIT_M,
            "orientation_delta_limit_rad":float(ORIENTATION_DELTA_LIMIT_RAD),
            "arm_joint_names":list(TRAIN_STATE_NAMES[15:29]),"arm_q_correction":delta[:,15:29].tolist(),
            "arm_dq_correction":(dq[future,15:29]-self.dq[future,15:29]).tolist(),
            "legs_waist_reference":"original q and dq unchanged", "hands":"original native action14 unchanged",
            "semantics":"new bounded world-wrist IK mode0 reference; actual box feedback and phase gating disabled",
            **diagnostics}
