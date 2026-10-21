"""CPU physical force, oriented clearance and continuous-time task contracts."""
from types import SimpleNamespace
import os
from pathlib import Path
from unittest.mock import patch
import unittest

import mujoco
import numpy as np

from simple.assets.carry_box import CarryBoxAsset
from simple.core.types import Pose
from simple.tasks.g1_wholebody_lift_box_hold_sonic import G1WholebodyLiftBoxHoldTaskSonic, LIFT_BOX_INSTRUCTION, TASK_CONTRACT


class LiftBoxHoldTests(unittest.TestCase):
    def setUp(self):
        spec=mujoco.MjSpec()
        world=spec.worldbody
        world.add_geom(name="ground",type=mujoco.mjtGeom.mjGEOM_PLANE,size=[0,0,1])
        CarryBoxAsset().build_mujoco(world,Pose([0,0,.3369],[1,0,0,0]))
        world.add_body(name="right_wrist_yaw_link",pos=[0,0,.09]).add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.03,0,0])
        world.add_body(name="table",pos=[2,0,.0699]).add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,size=[.2,.2,.05])
        self.model=spec.compile()
        self.data=mujoco.MjData(self.model)
        self.sim=SimpleNamespace(mjModel=self.model,mjData=self.data)
        self.task=G1WholebodyLiftBoxHoldTaskSonic.__new__(G1WholebodyLiftBoxHoldTaskSonic)
        self.task._target=SimpleNamespace(asset=SimpleNamespace(label="cube"))
        self.task.success_criteria=1.0
        self.task._reset_progress()

    def sample(self,time):
        self.data.time=time
        mujoco.mj_forward(self.model,self.data)
        return self.task.compute_reward({},mujoco_env=self.sim)

    def test_real_loaded_palm_and_exact_continuous_second_no_duplicate_credit(self):
        for frame in range(50):
            self.sample(frame/50)
        self.assertTrue(self.task.progress["loaded_hand_contact"])
        self.assertGreater(self.task.progress["hand_normal_force_n"],.01)
        self.assertFalse(self.task.check_success({}))
        self.sample(1.0)
        self.assertTrue(self.task.check_success({}))
        previous=self.task.reward
        self.sample(1.0)
        self.assertEqual(self.task.reward,previous)
        self.assertEqual(self.task.instruction,LIFT_BOX_INSTRUCTION)
        self.assertEqual(TASK_CONTRACT["metric"],"lift_box_airborne_hand_contact_v1")

    def test_contact_without_real_force_does_not_count_and_break_restarts_interval(self):
        for frame in range(40):self.sample(frame/50)
        self.data.qvel[2]=10
        self.sample(.8)
        self.assertGreater(self.data.ncon,0)
        self.assertFalse(self.task.progress["loaded_hand_contact"])
        self.assertEqual(self.task.reward,0)
        self.data.qvel[2]=0
        self.sample(.82)
        self.assertEqual(self.task.reward,0)
        self.sample(.84)
        self.assertAlmostEqual(self.task.reward,.02)
        self.sample(1.5)
        self.assertTrue(self.task.progress["continuity_gap"])
        self.assertEqual(self.task.reward,0)

    def test_oriented_lowest_corner_is_used_instead_of_center_height(self):
        self.data.qpos[2]=.33
        self.data.qpos[3:7]=[np.cos(np.pi/8),np.sin(np.pi/8),0,0]
        self.sample(0)
        self.assertGreater(self.task.progress["target_z"],.217+.10)
        self.assertAlmostEqual(self.task.progress["box_lowest_z_m"],.33-(.19+.217)/np.sqrt(2))
        self.assertFalse(self.task.progress["clearance_ge_10cm"])
        self.assertFalse(self.task.check_success({}))

    def test_floor_and_table_support_cannot_pass(self):
        self.model.body_pos[self.model.body("table").id]=[0,0,.0699]
        self.sample(0)
        self.assertTrue(self.task.progress["table_contact"])
        self.assertFalse(self.task.progress["condition"])
        self.model.body_pos[self.model.body("table").id]=[2,0,.0699]
        self.data.qpos[2]=.2169
        self.sample(.02)
        self.assertTrue(self.task.progress["floor_contact"])
        self.assertFalse(self.task.check_success({}))

    def test_invalid_time_force_and_reset_never_preserve_a_success(self):
        for frame in range(51):self.sample(frame/50)
        self.assertTrue(self.task.check_success({}))
        with self.assertRaises(ValueError):self.sample(.5)
        self.assertFalse(self.task.check_success({}))
        with self.assertRaises(ValueError):self.sample(float("nan"))
        self.sample(2)
        def invalid_force(model,data,index,force):force[:]=np.nan
        with patch("mujoco.mj_contactForce",side_effect=invalid_force):
            with self.assertRaises(ValueError):self.task.compute_reward({},mujoco_env=self.sim)
        self.assertFalse(self.task.check_success({}))
        self.task._reset_progress()
        self.assertEqual(self.task.reward,0)
        self.assertIsNone(self.task._last_reward_time)

    def test_knee_assistance_is_allowed_with_a_real_loaded_hand(self):
        spec=mujoco.MjSpec()
        CarryBoxAsset().build_mujoco(spec.worldbody,Pose([0,0,.3369],[1,0,0,0]))
        for name,y in (("left_hand_middle_1_link",.08),("left_knee_link",-.08)):
            spec.worldbody.add_body(name=name,pos=[0,y,.09]).add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.03,0,0])
        model=spec.compile()
        data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        self.task.compute_reward({},mujoco_env=SimpleNamespace(mjModel=model,mjData=data))
        self.assertTrue(self.task.progress["condition"])
        self.assertGreater(self.task.progress["hand_normal_force_n"],.01)
        self.assertIn("left_knee_link",self.task.progress["other_robot_contact_bodies"])

    def test_full_reset_preserves_independent_task_instruction_uid_and_metric(self):
        from simple.tasks.g1_wholebody_xmove_bend_carry_box_sonic import G1WholebodyXMoveBendCarryBoxTaskSonic
        robot=SimpleNamespace(uid="g1_sonic",head_camera_orientation=[1,0,0,0],head_cam_link="d435_link",reset=lambda **kw:None)
        data_root=Path(os.environ.get("CVPR_SIMPLE_RUNTIME","/home/zxc/CVPR_Evaluation_runtime"))/"resources/simple"
        if not data_root.is_dir():self.skipTest("Local task reset requires the external SIMPLE assets")
        with patch.dict(os.environ,{"SIMPLE_DATA_DIR":str(data_root)}):
            with patch("simple.robots.registry.RobotRegistry.make",return_value=robot):
                task=G1WholebodyLiftBoxHoldTaskSonic()
            task.reset(seed=10001)
            self.assertEqual(task.uid,"g1_wholebody_lift_box_hold_sonic")
            self.assertEqual(task.instruction,LIFT_BOX_INSTRUCTION)
            task.reward=1.0
            task.progress["success"]=True
            task.reset(seed=10002)
        self.assertEqual(task.reward,0)
        self.assertFalse(task.check_success({}))
        self.assertEqual(G1WholebodyXMoveBendCarryBoxTaskSonic.metadata["success_metric"],"true_tabletop_continuous_simtime_v1")

    def test_nonfinite_geometry_clears_progress_and_fails(self):
        self.sample(0)
        geom=next(i for i in range(self.model.ngeom) if self.model.geom_bodyid[i]==self.model.body("cube").id)
        self.data.geom_xmat[geom,0]=np.nan
        with self.assertRaises(ValueError):self.task.compute_reward({},mujoco_env=self.sim)
        self.assertFalse(self.task.check_success({}))


if __name__=="__main__":unittest.main()
