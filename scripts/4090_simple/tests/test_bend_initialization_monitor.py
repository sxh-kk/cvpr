"""Actual MuJoCo integration verifies the observer does not modify dynamics."""
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bend_initialization_monitor import BendInitializationMonitor


def fixture(target_x=.5):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".005" gravity="0 0 0"/>
      <worldbody>
        <body name="pelvis" pos="0 0 1"><geom name="robot" type="sphere" size=".1" mass="1"/></body>
        <body name="target" pos="{target_x} 0 1"><freejoint/>
          <geom name="box" type="box" size=".05 .05 .05" mass=".1"/>
        </body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class StartupMonitorTests(unittest.TestCase):
    def assert_passive(self, model, data, observer):
        names = ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'qfrc_constraint', 'qfrc_bias',
                 'ctrl', 'xpos', 'xquat', 'xipos', 'xmat', 'sensordata')
        before = {name: getattr(data, name).copy() for name in names}
        observer.after_step(model, data)
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(data, name), value, err_msg=name)

    def test_complete_window_and_policy_steps_are_passive(self):
        model, data = fixture()
        reference = mujoco.MjData(model)
        mujoco.mj_forward(model, reference)
        monitor = BendInitializationMonitor(model, data, model.body('target').id)
        for _ in range(221):
            mujoco.mj_step(model, data)
            mujoco.mj_step(model, reference)
            self.assert_passive(model, data, monitor)
            np.testing.assert_array_equal(data.qpos, reference.qpos)
            np.testing.assert_array_equal(data.qvel, reference.qvel)
        receipt = monitor.receipt()
        self.assertEqual(receipt['observed_physics_substeps'], 201)
        self.assertEqual(receipt['status'], 'passed')
        self.assertEqual(receipt['robot_target_contact_count'], 0)
        self.assertFalse(receipt['live_physical_state_writes'])
        self.assertEqual(receipt['physics_steps_added'], 0)

    def test_real_robot_contact_is_reported(self):
        model, data = fixture(.11)
        monitor = BendInitializationMonitor(model, data, model.body('target').id)
        for _ in range(201):
            mujoco.mj_step(model, data)
            self.assert_passive(model, data, monitor)
        self.assertGreater(monitor.receipt()['robot_target_contact_count'], 0)
        self.assertEqual(monitor.receipt()['status'], 'failed_or_incomplete')

    def test_drift_without_robot_contact_is_not_accepted(self):
        model, data = fixture()
        data.qvel[0] = .01
        monitor = BendInitializationMonitor(model, data, model.body('target').id)
        for _ in range(201):
            mujoco.mj_step(model, data)
            self.assert_passive(model, data, monitor)
        receipt = monitor.receipt()
        self.assertEqual(receipt['robot_target_contact_count'], 0)
        self.assertGreater(receipt['maximum_target_displacement_m'], .002)
        self.assertEqual(receipt['status'], 'failed_or_incomplete')

    def test_rotation_and_incomplete_window_do_not_pass(self):
        model, data = fixture()
        data.qvel[5] = .1
        monitor = BendInitializationMonitor(model, data, model.body('target').id)
        self.assertEqual(monitor.receipt()['status'], 'failed_or_incomplete')
        for _ in range(201):
            mujoco.mj_step(model, data)
            monitor.after_step(model, data)
        self.assertGreater(monitor.receipt()['maximum_target_rotation_degrees'], 2.)
        self.assertEqual(monitor.receipt()['status'], 'failed_or_incomplete')

    def test_time_and_model_binding_reject_invalid_observation(self):
        model, data = fixture()
        for invalid in (True, 50., 50.5, '50'):
            with self.subTest(warmup_ticks=invalid), self.assertRaisesRegex(ValueError, '50 warmup ticks'):
                BendInitializationMonitor(model, data, model.body('target').id, warmup_ticks=invalid)
        monitor = BendInitializationMonitor(model, data, model.body('target').id)
        mujoco.mj_step(model, data)
        with self.assertRaisesRegex(ValueError, 'before the reset'):
            BendInitializationMonitor(model, data, model.body('target').id)
        other_model, other_data = fixture()
        with self.assertRaisesRegex(ValueError, 'different physical model'):
            monitor.after_step(other_model, other_data)
        mujoco.mj_step(model, data)
        with self.assertRaisesRegex(ValueError, 'unexpected number'):
            monitor.after_step(model, data)


if __name__ == '__main__':
    unittest.main()
