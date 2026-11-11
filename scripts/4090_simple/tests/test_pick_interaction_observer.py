import unittest
import numpy as np
import mujoco

from pick_interaction_observer import PickInteractionObserver


XML = '''<mujoco><option timestep=".005"/><worldbody>
<geom name="table_geom" type="box" size="1 1 .1" pos="0 0 .1"/>
<body name="pelvis" pos="0 0 .4"><freejoint/>
<geom name="robot" type="box" size=".1 .1 .1" mass="1"/></body>
<body name="target" pos=".15 0 .4"><freejoint/>
<geom name="target_geom" type="box" size=".1 .1 .1" mass=".1"/></body>
<body name="toy" pos=".30 0 .4"><freejoint/>
<geom name="toy_geom" type="box" size=".1 .1 .1" mass=".1"/></body>
</worldbody></mujoco>'''


class PassiveInteractionTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model,self.data)
        self.observer = PickInteractionObserver(self.model,self.data,
            self.model.body('target').id,[self.model.body('toy').id])

    def snapshot(self,data):
        return {key:getattr(data,key).copy() for key in ('qpos','qvel','qacc','ctrl','xpos','xquat','xmat',
            'qfrc_constraint','qfrc_smooth','efc_force','qacc_warmstart','sensordata')}

    def test_real_solved_contacts_are_classified_and_live_state_is_unchanged(self):
        mujoco.mj_step(self.model,self.data)
        before = self.snapshot(self.data)
        self.observer.phase='policy_latent'
        self.observer.after_step(self.model,self.data)
        for key,value in before.items():
            np.testing.assert_array_equal(getattr(self.data,key),value,err_msg=key)
        receipt=self.observer.receipt()
        self.assertGreater(receipt['counts']['policy_latent:robot_target'],0)
        self.assertGreater(receipt['counts']['policy_latent:target_distractor'],0)
        self.assertEqual(receipt['observed_physics_substeps'],1)
        self.assertFalse(receipt['live_state_writes'])
        contacts=self.observer.drain()
        self.assertTrue(contacts)
        self.assertEqual(contacts[0]['solved_interval_start_s'],0)
        self.assertEqual(contacts[0]['solved_interval_end_s'],.005)
        self.assertEqual(self.observer.drain(),[])

    def test_observed_and_unobserved_trajectories_are_bitwise_equal(self):
        plain=mujoco.MjData(self.model)
        mujoco.mj_forward(self.model,plain)
        for _ in range(60):
            mujoco.mj_step(self.model,plain)
            mujoco.mj_step(self.model,self.data)
            self.observer.after_step(self.model,self.data)
            for key,value in self.snapshot(plain).items():
                np.testing.assert_array_equal(getattr(self.data,key),value,err_msg=key)
        self.assertEqual(self.data.time,plain.time)

    def test_missed_physics_or_different_model_is_rejected(self):
        mujoco.mj_step(self.model,self.data)
        mujoco.mj_step(self.model,self.data)
        with self.assertRaises(ValueError):self.observer.after_step(self.model,self.data)
        other=mujoco.MjModel.from_xml_string(XML)
        with self.assertRaises(ValueError):self.observer.after_step(other,mujoco.MjData(other))

    def test_completed_ledger_requires_native_reset_and_exact_control_substeps(self):
        for _ in range(5):
            mujoco.mj_step(self.model,self.data)
            self.observer.after_step(self.model,self.data)
        self.assertEqual(self.observer.completed_receipt(1)['status'],'complete')
        with self.assertRaises(ValueError):self.observer.completed_receipt(2)
        with self.assertRaises(ValueError):self.observer.completed_receipt(True)
        mujoco.mj_step(self.model,self.data)
        with self.assertRaises(ValueError):self.observer.completed_receipt(1)


if __name__ == '__main__':
    unittest.main()
