"""CPU-only physical and reset contracts for the isolated CarryBox backport."""
from copy import deepcopy
import os
from pathlib import Path
import random
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from simple.assets.carry_box import CarryBoxAsset
from simple.assets.primitive import Box as TableBox
from simple.core.actor import ObjectActor
from simple.core.layout import Layout
from simple.core.types import Pose
from simple.tasks.g1_wholebody_xmove_bend_carry_box_sonic import G1WholebodyXMoveBendCarryBoxTaskSonic


class CarryBoxLocalContracts(unittest.TestCase):
    def test_analytic_cube_has_official_geometry_dynamics_and_external_render_source(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CVPR_SIMPLE_RUNTIME": directory}):
            asset = CarryBoxAsset()
            spec = mujoco.MjSpec()
            asset.build_mujoco(spec.worldbody, Pose([0, 0, .217], [1, 0, 0, 0]))
            model = spec.compile()
            self.assertEqual(model.geom_type[0], mujoco.mjtGeom.mjGEOM_BOX)
            np.testing.assert_allclose(model.geom_size[0], [.115, .19, .217])
            np.testing.assert_allclose(model.geom_friction[0], [.8, .05, .005])
            np.testing.assert_allclose(model.geom_solref[0], [.005, 2])
            np.testing.assert_allclose(model.geom_rgba[0], [.76, .6, .42, 1])
            self.assertAlmostEqual(model.body_mass[1], .1)
            expected_inertia = .1 / 12 * np.array([.38**2+.434**2, .23**2+.434**2, .23**2+.38**2])
            np.testing.assert_allclose(model.body_inertia[1], expected_inertia)
            self.assertTrue(Path(asset.usd_path).is_relative_to(directory))
            self.assertIn('def Xform "Meshes"', Path(asset.usd_path).read_text())
            self.assertEqual(CarryBoxAsset().usd_path, asset.usd_path)

    def test_full_layout_seed_repeat_and_state_roundtrip_use_floor_initials(self):
        robot = SimpleNamespace(uid="g1_sonic", head_camera_orientation=[1, 0, 0, 0],
                                head_cam_link="d435_link", reset=lambda **kwargs: None)
        rng_before = np.random.get_state()
        python_before = random.getstate()
        with patch("simple.robots.registry.RobotRegistry.make", return_value=robot):
            task = G1WholebodyXMoveBendCarryBoxTaskSonic()
        task.reset(seed=10001)
        initial = deepcopy(task.state_dict())
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts/4090_simple"))
        from run_closed_loop import write_json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json(path, {"status": "finished", "environment": initial, "progress": task.progress})
            self.assertEqual(json.loads(path.read_text())["environment"]["dr_state_dict"]["scene"]["table"]["size"][2], .1)
        position = initial["dr_state_dict"]["spatial"]["cube"]["position"]
        self.assertTrue(-.6 <= position[0] <= -.5)
        self.assertTrue(-.05 <= position[1] <= .05)
        self.assertEqual(position[2], .217)
        self.assertEqual(task.layout.robot.pose.position, [-1.2, 0, 0])
        table = task.layout.actors["table"]
        self.assertAlmostEqual(table.pose.position[2] + .5*table.size[2], .45)
        task.reset(seed=10001)
        np.testing.assert_array_equal(task.target.pose.position, position)
        task.reset(seed=10002)
        self.assertNotEqual(task.target.pose.position, position)
        task.reset(seed=99999, options={"state_dict": initial})
        np.testing.assert_array_equal(task.target.pose.position, position)
        # A terminal layout pose must never override serialized initial DR state.
        initial["layout"]["actors"]["target"]["pose"]["position"] = [0, 0, 2]
        task.reset(options={"state_dict": initial})
        np.testing.assert_array_equal(task.target.pose.position, position)
        with self.assertRaises(AssertionError):
            task.reset(options={"state_dict": {**initial, "uid": "stale_task_name"}})
        np.testing.assert_equal(np.random.get_state(), rng_before)
        self.assertEqual(random.getstate(), python_before)

    def test_contact_success_uses_true_top_continuous_simtime_and_no_hand(self):
        asset = CarryBoxAsset()
        layout = Layout()
        layout.add_object("target", asset)
        layout.add_primitive("table", TableBox([1.25, .79, .1], [.3, 0, .4], [1, 0, 0, 0]))
        spec = mujoco.MjSpec()
        world = spec.worldbody
        world.add_body(name="table", pos=[.3, 0, .4]).add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[.625, .395, .05])
        asset.build_mujoco(world, Pose([0, 0, .6669], [1, 0, 0, 0]))
        hand = world.add_body(name="right_wrist_yaw_link", pos=[2, 0, .6])
        hand.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.03, 0, 0])
        model = spec.compile()
        data = mujoco.MjData(model)
        simulator = SimpleNamespace(mjModel=model, mjData=data)
        task = G1WholebodyXMoveBendCarryBoxTaskSonic.__new__(G1WholebodyXMoveBendCarryBoxTaskSonic)
        task._layout, task._target, task.success_criteria = layout, layout.actors["target"], .9
        task._reset_progress()

        def reward_at(time):
            data.time = time
            mujoco.mj_forward(model, data)
            return task.compute_reward({}, mujoco_env=simulator)

        reward_at(0)
        self.assertTrue(task.progress["on_table_top"])
        self.assertEqual(task.progress["table_top_z"], .45)
        self.assertAlmostEqual(reward_at(.5), .5)
        self.assertAlmostEqual(reward_at(.5), .5)
        self.assertFalse(task.check_success({}))
        self.assertTrue(reward_at(.91) >= .9)
        # Lowered to floor: no table-top contact and reset hold duration.
        data.qpos[2] = .217
        self.assertEqual(reward_at(1), 0)
        self.assertFalse(task.check_success({}))
        # Restarting contact begins a fresh interval, not the preceding gap.
        data.qpos[2] = .6669
        self.assertEqual(reward_at(1.5), 0)
        # A real named-hand contact prevents released-placement success.
        model.body_pos[model.body("right_wrist_yaw_link").id] = [0, 0, .9]
        reward_at(1.7)
        self.assertFalse(task.progress["hands_off"])
        self.assertFalse(task.check_success({}))
        with self.assertRaises(ValueError):
            reward_at(.1)


if __name__ == "__main__":
    unittest.main()
