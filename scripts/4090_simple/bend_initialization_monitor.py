"""Passive per-substep verification of the revised BendPick startup window."""
from copy import deepcopy

import numpy as np


class BendInitializationMonitor:
    """Inspect true integration contacts and private FK; never mutate live data."""

    def __init__(self, model, data, target_body_id, *, warmup_ticks=50):
        import mujoco
        self.model = model
        self.target = int(target_body_id)
        dt = float(model.opt.timestep)
        if not np.isclose(dt, .005, rtol=0, atol=1e-12) or type(warmup_ticks) is not int or warmup_ticks != 50:
            raise ValueError('Bend initialization monitor requires 200Hz physics and 50 warmup ticks')
        if float(data.time) != 0.:
            raise ValueError('Startup monitor must bind before the reset physics step')
        self.maximum_substeps = 1 + 4 * warmup_ticks  # Native reset step plus genuine 50Hz history.
        self._shadow = mujoco.MjData(model)
        self._target_bodies = self._descendants(self.target)
        self._robot_bodies = self._descendants(int(model.body('pelvis').id))
        self.initial_time = float(data.time)
        self.steps = 0
        self.samples = []
        self.contacts = []
        self.initial_xyz, self.initial_quaternion = self._pose(data)
        self.maximum_displacement_m = 0.
        self.maximum_rotation_degrees = 0.
        self.finite = True

    def _descendants(self, parent):
        result = {parent}
        for body in range(self.model.nbody):
            current = body
            while current:
                if current == parent:
                    result.add(body)
                    break
                current = int(self.model.body_parentid[current])
        return result

    def _pose(self, data):
        import mujoco
        # Contacts belong to the engine's solved interval; these poses describe
        # the integrated endpoint. FK in private MjData preserves live dynamics,
        # warm starts, qacc and sensors exactly.
        self._shadow.qpos[:] = data.qpos
        mujoco.mj_kinematics(self.model, self._shadow)
        mujoco.mj_comPos(self.model, self._shadow)
        return (self._shadow.body(self.target).xpos.copy(), self._shadow.body(self.target).xquat.copy())

    def after_step(self, model, data):
        import mujoco
        if model is not self.model:
            raise ValueError('Initialization monitor was reused with a different physical model')
        if self.steps >= self.maximum_substeps:
            return
        self.steps += 1
        expected = self.initial_time + self.steps * .005
        if not np.isclose(float(data.time), expected, rtol=0, atol=1e-9):
            raise ValueError('Startup physics advanced an unexpected number of substeps')
        xyz, quaternion = self._pose(data)
        finite = np.isfinite(np.r_[xyz, quaternion, data.qpos, data.qvel]).all()
        self.finite = bool(self.finite and finite)
        if not finite:
            raise ValueError('Nonfinite startup physical state')
        displacement = float(np.linalg.norm(xyz - self.initial_xyz))
        angle = float(np.degrees(2 * np.arccos(np.clip(abs(float(quaternion @ self.initial_quaternion)), 0., 1.))))
        self.maximum_displacement_m = max(self.maximum_displacement_m, displacement)
        self.maximum_rotation_degrees = max(self.maximum_rotation_degrees, angle)
        interval_contacts = []
        for index in range(data.ncon):
            contact = data.contact[index]
            g1, g2 = map(int, contact.geom)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            if not ((b1 in self._target_bodies and b2 in self._robot_bodies) or
                    (b2 in self._target_bodies and b1 in self._robot_bodies)):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            interval_contacts.append({'substep': self.steps, 'interval_end_sim_time_s': float(data.time),
                'body1': model.body(b1).name, 'body2': model.body(b2).name,
                'geom1': model.geom(g1).name, 'geom2': model.geom(g2).name,
                'distance_m': float(contact.dist), 'normal_force_N': float(force[0])})
        self.contacts.extend(interval_contacts)
        self.samples.append({'substep': self.steps, 'sim_time_s': float(data.time),
            'target_xyz_m': xyz.tolist(), 'target_quaternion_wxyz': quaternion.tolist(),
            'target_displacement_m': displacement, 'target_rotation_degrees': angle,
            'robot_target_contacts': len(interval_contacts)})

    def receipt(self):
        passed = (self.steps == self.maximum_substeps and self.finite and not self.contacts and
            self.maximum_displacement_m <= .002 and self.maximum_rotation_degrees <= 2.)
        return {'contract': 'cvpr-bend-pick-revised-startup-passive-verification-v1',
            'status': 'passed' if passed else 'failed_or_incomplete',
            'observed_physics_substeps': self.steps, 'required_physics_substeps': self.maximum_substeps,
            'warmup_control_ticks': 50, 'native_reset_physics_substeps': 1,
            'maximum_target_displacement_m': self.maximum_displacement_m,
            'maximum_target_rotation_degrees': self.maximum_rotation_degrees,
            'engineering_displacement_limit_m': .002, 'engineering_rotation_limit_degrees': 2.,
            'threshold_scope': 'initialization validity only; original native task success/reward/time budget unchanged',
            'finite_states': self.finite, 'robot_target_contact_count': len(self.contacts),
            'contacts': deepcopy(self.contacts), 'samples': deepcopy(self.samples),
            'live_physical_state_writes': False, 'private_FK_only': True,
            'controller_commands_modified': False, 'physics_steps_added': 0,
            'policy_phase_interventions': False}
