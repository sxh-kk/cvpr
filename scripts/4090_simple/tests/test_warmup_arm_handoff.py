"""Causal controller/history and finite handoff checks, without rendering/GPU."""
from pathlib import Path
import sys
import tempfile
from contextlib import redirect_stderr
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from test_sonic_pd_cache import make_controller, scalar_pd_reference, NAMES
from simple.baselines.sonic_v1_1 import BODY_JOINT_NAMES, SIMPLE_HAND_JOINT_NAMES, DEFAULT_ANGLES, ACTION_SCALE, POLICY_TO_BODY
import run_closed_loop
import evaluate_mobile_pp30 as evaluate


def model_fixture():
    bodies = ''.join(f'<body name="part{i}" pos="{i*.01} 0 0"><joint name="{n}" axis="0 1 0"/>'
                     '<geom type="sphere" pos=".1 0 -.05" size=".01" mass=".1" contype="0" conaffinity="0"/></body>'
                     for i, n in enumerate(reversed(NAMES)))
    motors = ''.join(f'<motor joint="{n}" ctrlrange="-20 20"/>' for n in NAMES)
    return mujoco.MjModel.from_xml_string('<mujoco><option timestep=".005"/><worldbody><body name="pelvis">'
        '<freejoint name="floating_base_joint"/><geom size=".05" mass="5" contype="0" conaffinity="0"/>'
        + bodies + '</body></worldbody><actuator>' + motors + '</actuator></mujoco>')


class WarmupArmHandoffTests(unittest.TestCase):
    def setUp(self):
        self.model = model_fixture()
        self.data = mujoco.MjData(self.model)
        self.data.time = .005
        mujoco.mj_forward(self.model, self.data)
        self.robot = SimpleNamespace(mjModel=self.model, mjData=self.data)
        self.controller = make_controller()
        self.controller._body_target = None
        self.hands = dict.fromkeys(SIMPLE_HAND_JOINT_NAMES, 0.)

    def raw(self, target):
        return np.asarray(((target-DEFAULT_ANGLES)/ACTION_SCALE)[POLICY_TO_BODY], np.float32)

    def test_first_physics_step_gravity_is_arm_only_uses_zero_velocity_and_original_limits(self):
        c, d, m = self.controller, self.data, self.model
        c.set_warmup_arm_target(np.zeros(14), {"sim_time_s":d.time}, gravity_compensation=True)
        d.qvel[:] = .3
        mujoco.mj_forward(m, d)
        before = [d.qpos.copy(), d.qvel.copy(), d.time]
        c.apply_physics(self.robot, None, self.hands)
        scratch = mujoco.MjData(m)
        scratch.qpos[:] = d.qpos
        scratch.qvel[:] = 0.
        mujoco.mj_forward(m, scratch)
        pd = make_controller()
        pd._body_target = np.zeros(29)
        oracle = mujoco.MjData(m)
        oracle.qpos[:] = d.qpos
        oracle.qvel[:] = d.qvel
        scalar_pd_reference(pd, m, oracle, self.hands)
        for aid in range(m.nu):
            joint = m.joint(int(m.actuator_trnid[aid, 0]))
            extra = scratch.qfrc_bias[int(joint.dofadr[0])] if joint.name in BODY_JOINT_NAMES[15:] else 0.
            self.assertAlmostEqual(d.ctrl[aid], np.clip(oracle.ctrl[aid]+extra, -20., 20.), places=12)
        for value, expected in zip([d.qpos, d.qvel, d.time], before):
            np.testing.assert_array_equal(value, expected)
        self.assertEqual(c.warmup_arm_receipt['gravity_physics_steps'], 1)
        self.assertEqual(c.warmup_arm_receipt['first_gravity_sim_time_s'], .005)
        m.actuator_ctrlrange[:] = [-.001, .001]
        c.apply_physics(self.robot, None, self.hands)
        self.assertLessEqual(np.max(np.abs(d.ctrl)), .001)

    def test_actual_previous_target_history_and_continuous_finite_handoff(self):
        c, d = self.controller, self.data
        previous = None
        last_warmup_target = None
        for tick in range(62):
            d.time = .005 + .02*tick
            if tick < 50:
                desired = np.full(14, .04*min(tick/10, 1.))
                c.set_warmup_arm_target(desired, {"sim_time_s":d.time}, gravity_compensation=True)
            elif tick == 50:
                last_warmup_target = c._body_target.copy()
                c.finish_warmup_arms({"sim_time_s":d.time})
            token = None if tick < 10 else np.zeros(64, np.float32)
            c.apply_physics(self.robot, token, self.hands)
            if previous is not None:
                np.testing.assert_array_equal(c._history[-1][4], self.raw(previous))
            if tick == 50:
                np.testing.assert_array_equal(c._body_target[15:], last_warmup_target[15:])
            previous = c._body_target.copy()
        self.assertIsNone(c._warmup_arm_target)
        self.assertEqual(c._warmup_arm_force_weight, 0.)
        self.assertAlmostEqual(c.warmup_arm_receipt['handoff_completed_sim_time_s'], 1.205)
        self.assertEqual(c.warmup_arm_receipt['events'][50]['learned_target_weight'], 0.)
        self.assertEqual(c.warmup_arm_receipt['events'][-1]['learned_target_weight'], 1.)
        self.assertEqual(len(c._history), 60)

    def test_invalid_reference_and_ranges_do_not_write_ctrl_and_reset_is_fresh(self):
        c, d = self.controller, self.data
        for bad in [np.zeros(13), np.full(14, np.nan)]:
            before = d.ctrl.copy()
            with self.assertRaises(ValueError):
                c.set_warmup_arm_target(bad, {"sim_time_s":d.time})
            np.testing.assert_array_equal(d.ctrl, before)
        c.set_warmup_arm_target(np.ones(14), {"sim_time_s":d.time})
        joint = self.model.joint(BODY_JOINT_NAMES[15])
        joint.limited[0] = True
        joint.range[:] = [-.1, .1]
        before = d.ctrl.copy()
        with self.assertRaisesRegex(ValueError, 'joint range'):
            c.apply_physics(self.robot, None, self.hands)
        np.testing.assert_array_equal(d.ctrl, before)
        c.reset()
        self.assertIsNone(c._warmup_arm_target)
        self.assertIsNone(c._warmup_gravity_binding)
        self.assertFalse(c.warmup_arm_receipt['enabled'])
        self.assertEqual(c.warmup_arm_receipt['events'], [])

    def test_cli_cannot_enable_new_arms_without_explicit_live_warmup(self):
        with tempfile.TemporaryDirectory() as root:
            argv = ['run_closed_loop.py', '--data-root', root, '--sonic-resources', root,
                    '--output', root+'/out', '--seconds', '1.2']
            with patch.object(sys, 'argv', argv):
                self.assertEqual(run_closed_loop.parse_args().warmup_arm_profile, 'legacy')
            with patch.object(sys, 'argv', argv+['--warmup-arm-profile', 'wrist_retreat_v1']), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run_closed_loop.parse_args()
            with patch.object(sys, 'argv', argv+['--warmup-arm-profile', 'wrist_retreat_v1',
                                               '--warmup-control', 'live_pose_sonic_v1_1', '--sync-observations',
                                               '--tasks', run_closed_loop.MOBILE_PP_TASK]):
                self.assertEqual(run_closed_loop.parse_args().warmup_arm_profile, 'wrist_retreat_v1')

    def test_campaign_rejects_missing_real_arm_commands_or_changed_variant(self):
        q = [0.]*14
        profile = {'backward_m':.05, 'outward_each_m':.03, 'upward_m':.05, 'transition_s':.2}
        event = {'sim_time_s':.005, 'legs_waist_reference_unchanged':True,
                 'arms_left_right':[{'desired_joint_q7':q[:7]}, {'desired_joint_q7':q[7:]}]}
        reference = {'kind':'first_live_robot_world_wrist_retreat_v1', 'object_feedback':False,
            'contact_feedback':False, 'future_teacher':False, 'live_state_writes':False,
            'history_injection':False, 'profile':profile, 'calls':1, 'events':[event]}
        control = {'enabled':True, 'arm_gravity_compensation':True, 'gravity_physics_steps':4,
            'first_gravity_sim_time_s':.005, 'events':[{'sim_time_s':.005, 'applied_arm14':q,
                'learned_target_weight':0., 'gravity_weight':1.}]}
        episode = {'warmup_frames':1, 'warmup_control':{'arm_profile':'wrist_retreat_v1',
                   'arm_reference':reference, 'arm_controller':control}}
        ticks = [{'sim_time_s':.005}]
        evaluate.validate_warmup_arms(episode, ticks, 'wrist_retreat_v1')
        with self.assertRaisesRegex(ValueError, 'frozen source'):
            evaluate.validate_warmup_arms(episode, ticks, 'wrist_retreat_v1', 'wrong-source')
        with self.assertRaisesRegex(ValueError, 'profile differs'):
            evaluate.validate_warmup_arms(episode, ticks, 'legacy')
        control['gravity_physics_steps'] = 0
        with self.assertRaisesRegex(ValueError, 'physical steps'):
            evaluate.validate_warmup_arms(episode, ticks, 'wrist_retreat_v1')
        control['gravity_physics_steps'] = 4
        control['events'][0]['applied_arm14'] = [1.]*14
        with self.assertRaisesRegex(ValueError, 'encoded preparation'):
            evaluate.validate_warmup_arms(episode, ticks, 'wrist_retreat_v1')


if __name__ == '__main__':
    unittest.main()
