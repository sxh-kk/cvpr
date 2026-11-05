"""Optional causal, robot-only Cartesian wrist reference for startup probes.

This module performs FK/IK in a private ``MjData``. It does not apply commands,
step physics, inspect task objects/contacts, or change a live simulator state.
Its fixed retreat is a diagnostic candidate, not a collision-free guarantee.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from simple.baselines.dual_source_psi0 import BODY_JOINT_NAMES, SIMPLE_HAND_JOINT_NAMES


ARM_JOINT_NAMES = BODY_JOINT_NAMES[15:29]
ROBOT_JOINT_NAMES = BODY_JOINT_NAMES + SIMPLE_HAND_JOINT_NAMES
SIDES = ("left", "right")
TRANSITION_SECONDS = .2
BACKWARD_METERS = .05
OUTWARD_METERS = .03
UPWARD_METERS = .05


def _finite(value, shape, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result.copy()


def _positive(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be finite and positive")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def _rotation_error(target, current):
    """Shortest world-axis rotation vector for a world-frame Jacobian."""
    quaternion = np.empty(4, dtype=np.float64)
    relative = np.ascontiguousarray(target @ current.T).reshape(9)
    mujoco.mju_mat2Quat(quaternion, relative)
    if quaternion[0] < 0:
        quaternion *= -1
    length = float(np.linalg.norm(quaternion[1:]))
    if length < 1e-14:
        return np.zeros(3, dtype=np.float64)
    return quaternion[1:] * (2 * np.arctan2(length, quaternion[0]) / length)


class WarmupWristReference:
    """Fix a first-frame wrist retreat, then solve only the current two arms.

    ``initial_data`` must be the first real post-reset robot state. At each
    reference update, ``reference(current_data, body_reference29)`` returns
    ``(desired_body29, diagnostic)``. The first 15 supplied reference values
    (legs/waist) are copied exactly; only arm14 is replaced. IK uses the current
    actual floating base and other robot joints, not desired non-arm joints.

    The caller chooses/records its update frequency and applies/limits the
    resulting targets. This helper does not implement a policy handoff, PD,
    actuator limits, or previous-action history; those remain caller duties.
    """

    def __init__(self, model, initial_data, *, max_iterations=40, damping=.01,
                 orientation_weight_m=.15, max_joint_step_rad=.04,
                 position_tolerance_m=1e-4, orientation_tolerance_rad=1e-3):
        if (isinstance(max_iterations, (bool, np.bool_)) or
                not isinstance(max_iterations, (int, np.integer)) or not 1 <= max_iterations <= 100):
            raise ValueError("max_iterations must be an integer in [1, 100]")
        self.max_iterations = int(max_iterations)
        self.damping = _positive(damping, "damping")
        self.orientation_weight_m = _positive(orientation_weight_m, "orientation_weight_m")
        self.max_joint_step_rad = _positive(max_joint_step_rad, "max_joint_step_rad")
        if self.max_joint_step_rad > .2:
            raise ValueError("max_joint_step_rad may not exceed .2 radians")
        self.position_tolerance_m = _positive(position_tolerance_m, "position_tolerance_m")
        self.orientation_tolerance_rad = _positive(orientation_tolerance_rad, "orientation_tolerance_rad")
        self.model = model
        self._data = mujoco.MjData(model)
        self._root_qadr = int(model.joint("floating_base_joint").qposadr[0])
        if int(model.joint("floating_base_joint").type[0]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError("Wrist reference requires the named free floating_base_joint")
        joints = [model.joint(name) for name in ROBOT_JOINT_NAMES]
        if any(int(joint.type[0]) != int(mujoco.mjtJoint.mjJNT_HINGE) for joint in joints):
            raise ValueError("The named robot43 joints must be scalar hinge joints")
        self._qadr = np.array([int(joint.qposadr[0]) for joint in joints], dtype=np.intp)
        self._dofs = np.array([int(joint.dofadr[0]) for joint in joints], dtype=np.intp)
        if len(set(self._qadr)) != 43 or len(set(self._dofs)) != 43:
            raise ValueError("Named robot joints must have distinct coordinates")
        self._wrists = [int(model.body(side + "_wrist_yaw_link").id) for side in SIDES]
        self._pelvis = int(model.body("pelvis").id)
        arm_joints = joints[15:29]
        self._lower = np.array([float(joint.range[0]) if joint.limited[0] else -np.inf
                                for joint in arm_joints], dtype=np.float64)
        self._upper = np.array([float(joint.range[1]) if joint.limited[0] else np.inf
                                for joint in arm_joints], dtype=np.float64)
        if (np.isnan(self._lower).any() or np.isnan(self._upper).any() or
                np.any(self._lower >= self._upper)):
            raise ValueError("Invalid arm joint limits")
        self._jacp = np.zeros((3, model.nv), dtype=np.float64)
        self._jacr = np.zeros((3, model.nv), dtype=np.float64)
        self._copy_robot(initial_data)
        self._fk()
        self.start_time_s = float(initial_data.time)
        if not np.isfinite(self.start_time_s) or self.start_time_s < 0:
            raise ValueError("Initial simulator time must be finite and nonnegative")
        pelvis_rotation = self._data.xmat[self._pelvis].reshape(3, 3)
        heading = pelvis_rotation[:2, 0]
        heading_length = float(np.linalg.norm(heading))
        if heading_length < 1e-8:
            raise ValueError("Initial pelvis heading has no horizontal projection")
        forward = np.r_[heading / heading_length, 0.]
        left = np.array([-forward[1], forward[0], 0.])
        self._initial_points = np.stack([self._pose(side)[0] for side in range(2)])
        self._initial_rotations = np.stack([self._pose(side)[1] for side in range(2)])
        self._displacements = np.stack([
            -BACKWARD_METERS * forward + sign * OUTWARD_METERS * left + np.array([0., 0., UPWARD_METERS])
            for sign in (1., -1.)])
        self._last_time_s = self.start_time_s
        self._last_output_arm = None
        initial_robot = {"root_xyz_wxyz7": self._data.qpos[self._root_qadr:self._root_qadr + 7].tolist(),
                         "joint_names43": list(ROBOT_JOINT_NAMES),
                         "joint_q43": self._data.qpos[self._qadr].tolist(),
                         "sim_time_s": self.start_time_s}
        identity = hashlib.sha256(json.dumps(initial_robot, sort_keys=True, separators=(",", ":"),
                                            allow_nan=False).encode()).hexdigest()
        self._receipt = {
            "kind": "first_live_robot_world_wrist_retreat_v1", "diagnostic_candidate_only": True,
            "helper_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "initial_measured_robot": initial_robot, "initial_measured_robot_sha256": identity,
            "arm_joint_names14": list(ARM_JOINT_NAMES), "wrist_body_names": [model.body(i).name for i in self._wrists],
            "world_displacements_left_right_m": self._displacements.tolist(),
            "initial_world_positions_left_right_m": self._initial_points.tolist(),
            "fixed_world_rotations_left_right": self._initial_rotations.tolist(),
            "fixed_heading_world_forward": forward.tolist(),
            "profile": {"backward_m": BACKWARD_METERS, "outward_each_m": OUTWARD_METERS,
                        "upward_m": UPWARD_METERS, "transition_s": TRANSITION_SECONDS,
                        "interpolation": "u=clip((current_time-first_real_time)/.2,0,1); 3*u*u-2*u*u*u",
                        "after_transition": "hold fixed world positions and initial wrist world orientations"},
            "solver": {"method": "world-frame 6x7 damped least squares per arm", "max_iterations": self.max_iterations,
                       "damping": self.damping, "orientation_weight_m": self.orientation_weight_m,
                       "max_joint_step_rad": self.max_joint_step_rad, "position_tolerance_m": self.position_tolerance_m,
                       "orientation_tolerance_rad": self.orientation_tolerance_rad,
                       "joint_limits": "named model hinge limits, never applied to live state"},
            "kinematics": "private MjData; mj_kinematics + mj_comPos for Jacobians; no mj_forward or physics stepping",
            "non_arm_ik_state": "current actual robot base and joints; supplied legs/waist reference returned unchanged",
            "object_feedback": False, "contact_feedback": False, "future_teacher": False,
            "live_state_writes": False, "history_injection": False,
            "actuator_control_and_handoff": "not implemented here; caller must apply bounded targets and actual-action history",
            "calls": 0, "events": []}

    @property
    def receipt(self):
        return deepcopy(self._receipt)

    def _copy_robot(self, source):
        if source.model is not self.model:
            raise ValueError("Current robot MjData must belong to the bound model")
        root = _finite(source.qpos[self._root_qadr:self._root_qadr + 7], (7,), "Robot root xyz/wxyz")
        if not np.isclose(np.linalg.norm(root[3:]), 1., rtol=0, atol=1e-5):
            raise ValueError("Robot root quaternion must have unit length")
        joints = _finite(source.qpos[self._qadr], (43,), "Robot named joint positions")
        # Read only the robot coordinates. Task/object coordinates are neither
        # copied from the live state nor queried by this helper.
        self._data.qpos[self._root_qadr:self._root_qadr + 7] = root
        self._data.qpos[self._qadr] = joints

    def _fk(self):
        mujoco.mj_kinematics(self.model, self._data)
        # mj_jac uses global motion dofs produced by mj_comPos. This performs
        # neither contact detection nor a dynamics/physics step.
        mujoco.mj_comPos(self.model, self._data)

    def _pose(self, side):
        wrist = self._wrists[side]
        return (self._data.xpos[wrist].copy(), self._data.xmat[wrist].reshape(3, 3).copy())

    def _solve_arm(self, side, goal_point, goal_rotation):
        part = slice(7 * side, 7 * side + 7)
        coordinates = self._qadr[15:29][part]
        dofs = self._dofs[15:29][part]
        lower, upper = self._lower[part], self._upper[part]
        initial = self._data.qpos[coordinates].copy()
        before_point, before_rotation = self._pose(side)
        bounded = np.clip(initial, lower, upper)
        initial_clipped = int(np.count_nonzero(bounded != initial))
        self._data.qpos[coordinates] = bounded
        self._fk()
        clipped_elements = initial_clipped
        largest_iteration_step = 0.
        iterations, stop_reason = 0, "iteration_limit"
        for iteration in range(self.max_iterations):
            point, rotation = self._pose(side)
            position_error = goal_point - point
            orientation_error = _rotation_error(goal_rotation, rotation)
            if (np.linalg.norm(position_error) <= self.position_tolerance_m and
                    np.linalg.norm(orientation_error) <= self.orientation_tolerance_rad):
                stop_reason = "within_tolerance"
                break
            mujoco.mj_jac(self.model, self._data, self._jacp, self._jacr, point, self._wrists[side])
            jacobian = np.vstack((self._jacp[:, dofs], self.orientation_weight_m * self._jacr[:, dofs]))
            error = np.r_[position_error, self.orientation_weight_m * orientation_error]
            if not np.isfinite(jacobian).all() or not np.isfinite(error).all():
                raise ValueError("Nonfinite robot-only wrist Jacobian or residual")
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + self.damping ** 2 * np.eye(6), error)
            if not np.isfinite(delta).all():
                raise ValueError("Nonfinite robot-only wrist IK update")
            limited_delta = np.clip(delta, -self.max_joint_step_rad, self.max_joint_step_rad)
            candidate_unbounded = self._data.qpos[coordinates] + limited_delta
            candidate = np.clip(candidate_unbounded, lower, upper)
            clipped_elements += int(np.count_nonzero(candidate != candidate_unbounded))
            change = float(np.max(np.abs(candidate - self._data.qpos[coordinates])))
            largest_iteration_step = max(largest_iteration_step, change)
            self._data.qpos[coordinates] = candidate
            self._fk()
            iterations = iteration + 1
            if change < 1e-10:
                stop_reason = "joint_limit_or_stationary"
                break
        after_point, after_rotation = self._pose(side)
        position_residual = float(np.linalg.norm(goal_point - after_point))
        orientation_residual = float(np.linalg.norm(_rotation_error(goal_rotation, after_rotation)))
        converged = (position_residual <= self.position_tolerance_m and
                     orientation_residual <= self.orientation_tolerance_rad)
        if converged:
            stop_reason = "within_tolerance"
        desired = self._data.qpos[coordinates].copy()
        diagnostic = {
            "side": SIDES[side], "iterations": iterations, "max_iterations": self.max_iterations,
            "converged": bool(converged), "stop_reason": stop_reason,
            "goal_world_position_m": goal_point.tolist(), "goal_world_rotation": goal_rotation.tolist(),
            "actual_world_position_before_m": before_point.tolist(), "ik_world_position_after_m": after_point.tolist(),
            "position_error_before_m": float(np.linalg.norm(goal_point - before_point)),
            "orientation_error_before_rad": float(np.linalg.norm(_rotation_error(goal_rotation, before_rotation))),
            "position_residual_m": position_residual, "orientation_residual_rad": orientation_residual,
            "initial_joint_limit_clipped_elements": initial_clipped,
            "joint_limit_clipped_elements": clipped_elements, "joint_limit_active": bool(np.any(
                (np.abs(desired - lower) < 1e-8) | (np.abs(desired - upper) < 1e-8))),
            "maximum_iteration_joint_step_rad": largest_iteration_step,
            "maximum_joint_change_from_current_actual_rad": float(np.max(np.abs(desired - initial))),
            "desired_joint_q7": desired.tolist()}
        return desired, diagnostic

    def reference(self, current_data, body_reference29):
        """Return arm-only desired body29 plus bounded-solver diagnostics.

        Input observations and the provided reference are never modified.
        Unreachable goals are returned with ``converged=False`` and residuals;
        they are not hidden by moving the fixed world target.
        """
        output = _finite(body_reference29, (29,), "Supplied body reference29")
        time_s = float(current_data.time)
        if not np.isfinite(time_s) or time_s < self.start_time_s - 1e-9 or time_s < self._last_time_s - 1e-9:
            raise ValueError("Robot reference time must be finite, chronological and after initialization")
        self._copy_robot(current_data)
        self._fk()
        elapsed = max(0., time_s - self.start_time_s)
        u = float(np.clip(elapsed / TRANSITION_SECONDS, 0., 1.))
        blend = u * u * (3. - 2. * u)
        goals = self._initial_points + blend * self._displacements
        diagnostics = []
        for side in range(2):
            desired, diagnostic = self._solve_arm(side, goals[side], self._initial_rotations[side])
            output[15 + side * 7:22 + side * 7] = desired
            diagnostics.append(diagnostic)
        if not np.isfinite(output).all():
            raise ValueError("Robot-only wrist reference produced nonfinite desired joints")
        event = {"sim_time_s": time_s, "elapsed_s": elapsed, "transition_fraction": u,
                 "smoothstep": blend, "world_goal_held": bool(u == 1.),
                 "arms_left_right": diagnostics, "legs_waist_reference_unchanged": True,
                 "all_arms_converged": all(item["converged"] for item in diagnostics),
                 "maximum_arm_target_change_since_previous_call_rad": (None if self._last_output_arm is None else
                     float(np.max(np.abs(output[15:29] - self._last_output_arm))))}
        self._last_output_arm = output[15:29].copy()
        self._last_time_s = time_s
        self._receipt["calls"] += 1
        self._receipt["events"].append(deepcopy(event))
        return output, event
