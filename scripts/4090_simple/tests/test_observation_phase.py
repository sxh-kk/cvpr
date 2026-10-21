"""CPU proof of MuJoCo's post-integration derived-state phase and refresh."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

import mujoco
import numpy as np

from simple.envs.observation_contract import refresh_mujoco_observation
from simple.envs.sonic_loco_manip import SonicLocoManipEnv


class ObservationPhaseTests(unittest.TestCase):
    def test_observation_only_mode_skips_all_four_isaac_physics_substeps(self):
        calls=[]
        def wrong_step(*args,**kwargs):raise AssertionError("Isaac must not step during MuJoCo integration")
        fake=SimpleNamespace(control_decimal=4,step_count=0,isaac_observation_only=True,
            mujoco=SimpleNamespace(apply_action=lambda action:calls.append("pd"),step=lambda **kw:calls.append("mj")),
            isaac=SimpleNamespace(step=wrong_step),_get_obs=lambda:{},_get_info=lambda:{},
            task=SimpleNamespace(compute_reward=lambda *a,**kw:0,check_success=lambda *a,**kw:False))
        SonicLocoManipEnv.step(fake,None)
        self.assertEqual(calls,["pd","mj"]*4)

    def test_actual_isaac_observation_only_step_does_not_call_world_step(self):
        path=Path(__file__).resolve().parents[3]/"VLMExpert/third_party/SIMPLE/src/simple/engines/isaacsim.py"
        original=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef)and n.name=="IsaacSimSimulator")
        method=next(n for n in original.body if isinstance(n,ast.FunctionDef)and n.name=="step")
        module=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name="Bridge",bases=[],keywords=[],body=[method],decorator_list=[])],type_ignores=[]))
        namespace={}
        exec(compile(module,str(path),"exec"),namespace)
        bridge=namespace["Bridge"]()
        bridge.is_isaac_reset=True
        calls=[]
        bridge.refresh_from_mujoco=lambda physics,**kw:calls.append(kw)
        bridge.step(object(),observation_only=True)
        self.assertEqual(calls,[{"update_kinematics":True}])
    def make_physics(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <option timestep=".005" gravity="0 0 0"/>
          <worldbody><body name="pelvis"><joint name="turn" axis="0 0 1"/>
          <geom type="sphere" size=".1" pos=".3 0 0" mass="1"/></body></worldbody>
        </mujoco>''')
        data = mujoco.MjData(model)
        data.qpos[0] = .4
        data.qvel[0] = .7
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        return model, data

    def test_refresh_fixes_real_phase_mismatch_without_moving_state_or_time(self):
        model, data = self.make_physics()
        old_rotation = data.xmat[model.body("pelvis").id].copy()
        q, v, now = data.qpos.copy(), data.qvel.copy(), data.time
        expected = np.array([[np.cos(q[0]), -np.sin(q[0]), 0],
                             [np.sin(q[0]), np.cos(q[0]), 0], [0, 0, 1]])
        self.assertGreater(np.max(np.abs(old_rotation.reshape(3,3) - expected)), .001)
        refresh_mujoco_observation(model, data)
        np.testing.assert_allclose(data.xmat[model.body("pelvis").id].reshape(3,3), expected, atol=1e-14)
        np.testing.assert_array_equal(data.qpos, q)
        np.testing.assert_array_equal(data.qvel, v)
        self.assertEqual(data.time, now)

    def test_environment_refreshes_before_bridge_render_and_uses_mujoco_joint_authority(self):
        model, data = self.make_physics()
        events = []
        class Isaac:
            def refresh_from_mujoco(self, physics):
                events.append("bridge_render")
                self.yaw = np.arctan2(physics.mjData.xmat[1,3], physics.mjData.xmat[1,0])
        physics = SimpleNamespace(mjModel=model, mjData=data,
                                  get_robot_qpos=lambda: {"turn":float(data.qpos[0])})
        def render():
            events.append("get_rgb")
            return {"head_stereo_left":np.zeros((2,3,3),np.uint8)}
        fake = SimpleNamespace(sync_mujoco_observations=True, mujoco=physics, isaac=Isaac(), _render_frame=render)
        before = data.time
        observation = SonicLocoManipEnv._get_obs(fake)
        self.assertEqual(events, ["bridge_render", "get_rgb"])
        self.assertAlmostEqual(fake.isaac.yaw, data.qpos[0])
        self.assertAlmostEqual(float(observation["joint_qpos"][0]), data.qpos[0], places=7)
        self.assertEqual(observation["sim_time_s"], before)
        self.assertEqual(data.time, before)

    def test_actual_isaac_bridge_optin_removes_rounding_and_renders_without_time_advance(self):
        # Load the real two bridge methods without importing GPU-only Isaac Kit.
        path = Path(__file__).resolve().parents[3] / "VLMExpert/third_party/SIMPLE/src/simple/engines/isaacsim.py"
        tree = ast.parse(path.read_text())
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "IsaacSimSimulator")
        methods = [node for node in original.body if isinstance(node, ast.FunctionDef)
                   and node.name in ("sync_states", "refresh_from_mujoco")]
        module = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name="Bridge", bases=[], keywords=[],
                                                          body=methods, decorator_list=[])], type_ignores=[]))
        renders = []
        namespace = {"np":np, "rep":SimpleNamespace(orchestrator=SimpleNamespace(step=lambda **kw:renders.append(kw)))}
        exec(compile(module, str(path), "exec"), namespace)
        bridge = namespace["Bridge"]()
        values = {}
        root = np.array([.123456789, .000123456, .789123456, 1, 0, 0, 0])
        q = .123456789
        bridge.articulated_objects, bridge.need_gravity = {}, True
        bridge.update_visuals = lambda: None
        bridge.robot = SimpleNamespace(get_dof_index=lambda name:0,
            set_joint_positions=lambda positions, **kw:values.update(q=positions.copy()),
            set_world_pose=lambda **kw:values.update(root=np.r_[kw["position"],kw["orientation"]]),
            _articulation_view=SimpleNamespace(num_dof=1, set_joint_velocities=lambda value:None))
        bridge.task = SimpleNamespace(robot=SimpleNamespace(jname_mujoco_to_isaac=lambda name:name))
        physics = SimpleNamespace(get_states=lambda:[0,{"turn":q},[],[],[],np.round(root,4)],
                                  get_robot_qpos=lambda:{"turn":q}, mjData=SimpleNamespace(qpos=root.copy()))
        bridge.sync_states(physics)
        self.assertEqual(values["q"], [round(q,6)])
        np.testing.assert_array_equal(values["root"], np.round(root,4))
        bridge.refresh_from_mujoco(physics)
        self.assertEqual(values["q"], [q])
        np.testing.assert_array_equal(values["root"], root)
        self.assertEqual(renders, [{"rt_subframes":1,"pause_timeline":True,"delta_time":0.0}])


if __name__ == "__main__":
    unittest.main()
