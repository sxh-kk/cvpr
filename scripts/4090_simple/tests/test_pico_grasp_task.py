import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pico_grasp_task as grasp


def fixture():
    pose = lambda xyz: {"position": xyz, "quaternion": [1., 0., 0., 0.]}
    table = {"uid": "box", "size": [2., 1., .1], "pose": pose([.13, 0., -.05]), "material": {"name": "wood"}}
    target = {"uid": "10", "asset": {"res_id": "graspnet1b", "uid": "10", "label": "chips_can"}, "pose": pose([-.65, 0., .04]), "material": {"roughness": .5}}
    distractor = {"uid": "6", "asset": {"res_id": "graspnet1b", "uid": "6", "label": "bowl"}, "pose": pose([.2, .1, .02]), "material": {"roughness": .3}}
    scene = {"uid": "hssd:scene2", "name": "room", "center_offset": [.15, 0., 0.], "table": copy.deepcopy(table)}
    light = {"uid": "Light_0", "center_light_postion": [.1, 0., 1.4], "pose": {"position": [-1., 0., 0.]}, "intensity": 100.}
    camera = {"cam_id": "side", "mount": "eye_on_base", "pose": pose([.3, .9, .675]), "cam_cfg": {}}
    return {"uid": grasp.TASK_UID, "metadata": {"physics_dt": .002, "render_hz": 50}, "robot_cfg": {"uid": "g1_wholebody"},
        "dr_cfgs": {}, "sensor_cfgs": {"side": {"mount": "eye_on_base", "pose": pose([.3, .9, .675])}},
        "dr_state_dict": {"spatial": {"g1_wholebody": pose([-1., 0., 0.]), "10": copy.deepcopy(target["pose"]), "6": copy.deepcopy(distractor["pose"])},
            "target": copy.deepcopy(target["asset"]),
            "scene": copy.deepcopy(scene), "lighting": {"Light_0": copy.deepcopy(light)}, "material": {"target": {"roughness": .5}}},
        "layout": {"actors": {"table": table, "target": target, "distractor_6": distractor}, "scene": scene,
            "lights": [light], "cameras": {"side": camera}, "scene_info": {}}}


class FrameTests(unittest.TestCase):
    def test_full_world_shift_preserves_relative_geometry_and_source(self):
        original = fixture()
        before = copy.deepcopy(original)
        adapted, receipt = grasp.adapt_official_config(original)
        self.assertEqual(original, before)
        self.assertEqual(receipt, adapted[grasp.FRAME_KEY])
        self.assertFalse(receipt["robot_physics_equivalent"])
        np.testing.assert_allclose(adapted["dr_state_dict"]["spatial"]["g1_sonic"]["position"], [-1., 0., -.008], atol=1e-15)
        for key in ("target", "distractor_6", "table"):
            np.testing.assert_allclose(np.array(adapted["layout"]["actors"][key]["pose"]["position"])
                - original["layout"]["actors"][key]["pose"]["position"], grasp.WORLD_TRANSLATION_M)
        self.assertEqual(adapted["layout"]["lights"][0]["pose"], original["layout"]["lights"][0]["pose"])
        self.assertAlmostEqual(adapted["layout"]["lights"][0]["center_light_postion"][2], 2.185)
        self.assertAlmostEqual(adapted["sensor_cfgs"]["side"]["pose"]["position"][2], 1.46)

    def test_double_translation_and_wrong_task_rejected(self):
        state, _ = grasp.adapt_official_config(fixture())
        with self.assertRaises(ValueError):
            grasp.adapt_official_config(state)
        for change in ({"uid": "g1_handover"}, {"robot_cfg": {"uid": "g1_sonic"}}):
            source = fixture()
            source.update(change)
            with self.assertRaises(ValueError):
                grasp.adapt_official_config(source)

    def test_changed_native_clock_or_floor_frame_rejected(self):
        for mutate in (lambda s: s['metadata'].update(physics_dt=.005),
                lambda s: s['metadata'].update(render_hz=30),
                lambda s: s['dr_state_dict']['spatial']['g1_wholebody']['position'].__setitem__(2, .1)):
            state = fixture()
            mutate(state)
            with self.assertRaises(ValueError):
                grasp.adapt_official_config(state)

    def test_reset_payload_tampering_rejected(self):
        state, _ = grasp.adapt_official_config(fixture())
        state["dr_state_dict"]["spatial"]["10"]["position"][0] += .1
        with self.assertRaisesRegex(ValueError, "payload changed"):
            grasp.validate_adapted_config(state)

    def test_source_and_adapter_identity_tampering_rejected(self):
        for field in ("official_task_source_sha256", "adapter_sha256", "task_id"):
            state, _ = grasp.adapt_official_config(fixture())
            state[grasp.FRAME_KEY][field] = "wrong"
            with self.assertRaisesRegex(ValueError, "identity"):
                grasp.validate_adapted_config(state)

    def test_missing_or_duplicate_object_identity_rejected(self):
        for mutate in (lambda s: s["layout"]["actors"].pop("target"),
                lambda s: s["layout"]["actors"].update(other=copy.deepcopy(s["layout"]["actors"]["target"]))):
            source = fixture()
            mutate(source)
            with self.assertRaises(ValueError):
                grasp.adapt_official_config(source)

    def test_layout_never_uses_other_than_original_spatial(self):
        source = fixture()
        source["layout"]["actors"]["target"]["pose"]["position"][0] += .02
        with self.assertRaisesRegex(ValueError, "initial spatial"):
            grasp.adapt_official_config(source)

    def test_binding_accepts_renumbering_and_only_float_roundoff(self):
        state, _ = grasp.adapt_official_config(fixture())
        actual = copy.deepcopy(state)
        actual["layout"]["actors"]["distractor_0"] = actual["layout"]["actors"].pop("distractor_6")
        actual["layout"]["cameras"]["side"]["pose"]["position"][0] += 1e-7
        self.assertTrue(grasp.grasp_environment_binding_receipt(state, actual)["matched"])
        actual["layout"]["cameras"]["side"]["pose"]["position"][0] += .001
        self.assertFalse(grasp.grasp_environment_binding_receipt(state, actual)["matched"])

    def test_binding_checks_lights_material_scene_and_table_size(self):
        for mutate in (lambda s: s["layout"]["lights"][0].update(intensity=200.),
                lambda s: s["layout"]["actors"]["target"]["material"].update(roughness=.9),
                lambda s: s["layout"]["scene"].update(name="other"),
                lambda s: s["layout"]["actors"]["table"]["size"].__setitem__(0, 3.)):
            state, _ = grasp.adapt_official_config(fixture())
            actual = copy.deepcopy(state)
            mutate(actual)
            self.assertFalse(grasp.grasp_environment_binding_receipt(state, actual)["matched"])

    def test_binding_rejects_target_distractor_role_swap(self):
        state, _ = grasp.adapt_official_config(fixture())
        actual = copy.deepcopy(state)
        actors = actual['layout']['actors']
        actors['target'], actors['distractor_6'] = actors['distractor_6'], actors['target']
        with self.assertRaisesRegex(ValueError, 'target role'):
            grasp.grasp_environment_binding_receipt(state, actual)

    def test_target_role_requires_saved_asset_namespace_and_actor_uid(self):
        for mutate in (lambda t: t['asset'].update(res_id='other'),
                lambda t: t.update(uid='6'), lambda t: t['asset'].update(uid='6')):
            state = fixture()
            mutate(state['layout']['actors']['target'])
            with self.assertRaisesRegex(ValueError, 'target role'):
                grasp.adapt_official_config(state)

    def test_robot_local_camera_mounts_not_translated(self):
        source = fixture()
        source["sensor_cfgs"]["head"] = {"mount": "eye_in_head", "pose": {"position": [0., 0., .1]}}
        state, _ = grasp.adapt_official_config(source)
        self.assertEqual(state["sensor_cfgs"]["head"], source["sensor_cfgs"]["head"])


SOURCE = Path('/home/zxc/CVPR_Evaluation_runtime/resources/simple_finetune_sources/G1WholebodyTabletopGraspMP-v0/meta/episodes.jsonl')
DATA = Path('/home/zxc/CVPR_Evaluation_runtime/pico_grasp_official_20261102/preparation_001/resources/data_root')
HAS_SIMPLE = importlib.util.find_spec('simple') is not None


@unittest.skipUnless(HAS_SIMPLE and SOURCE.is_file() and DATA.is_dir(), 'Requires installed SIMPLE and frozen runtime fixtures')
class ActualSimpleTests(unittest.TestCase):
    def source(self):
        source = json.loads(json.loads(SOURCE.read_text().splitlines()[0])["environment_config"])
        def relocate(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, str) and v.startswith('/home/wenke/SIMPLE/data/'):
                        value[k] = str(DATA / v.removeprefix('/home/wenke/SIMPLE/data/'))
                    else:
                        relocate(v)
            elif isinstance(value, list):
                for v in value:
                    relocate(v)
        relocate(source)
        return source

    def fake_robot(self):
        from simple.robots.protocols import WristCamMountable, HeadCamMountable
        class Robot(WristCamMountable, HeadCamMountable):
            uid = 'g1_sonic'
            wrist_camera_orientation = head_camera_orientation = [1., 0., 0., 0.]
            def reset(self, **kwargs):
                self.spawn_pose = kwargs['spawn_pose']
        return Robot()

    def test_actual_task_reset_full_dr_baseline_and_predicate_unchanged(self):
        from simple.tasks.g1_wholebody_tabletop_grasp_mp import G1WholebodyTabletopGraspMP as Official
        from simple.tasks.registry import TaskRegistry
        state, _ = grasp.adapt_official_config(self.source())
        before = copy.deepcopy(Official.metadata)
        with patch('simple.robots.registry.RobotRegistry.make', return_value=self.fake_robot()) as make:
            task = grasp.make_grasp_task({'SIMULATE_DT': .005}, adapted_config=state)
            task.reset()
            self.assertEqual(make.call_args.kwargs['uid'], 'g1_sonic')
            self.assertIs(type(task).compute_reward, Official.compute_reward)
            self.assertIs(type(task).check_success, Official.check_success)
            self.assertIsNone(task._init_target_height)
            self.assertEqual(task.compute_reward({'target': [0, 0, .824]}), 0.)
            self.assertFalse(task.check_success({'target': [0, 0, .8419]}))
            self.assertTrue(task.check_success({'target': [0, 0, .8421]}))
            self.assertTrue(grasp.grasp_environment_binding_receipt(state, task.state_dict())['matched'])
            task.reset()
            self.assertIsNone(task._init_target_height)
            with self.assertRaises(ValueError):
                task.reset(options={'state_dict': state, 'dr_level': 0})
        self.assertEqual(Official.metadata, before)
        self.assertEqual(grasp.TASK_UID, task.uid)

    def test_separate_tasks_do_not_share_dr_or_metadata(self):
        with patch('simple.robots.registry.RobotRegistry.make', side_effect=lambda **kw: self.fake_robot()):
            a = grasp.make_grasp_task({'SIMULATE_DT': .005}, source_config=self.source())
            b = grasp.make_grasp_task({'SIMULATE_DT': .005}, source_config=self.source())
            self.assertIsNot(a.dr.randomizers, b.dr.randomizers)
            self.assertIsNot(a.metadata, b.metadata)
            a.dr_cfgs['target'].asset_id = 'wrong'
            self.assertEqual(b.dr_cfgs['target'].asset_id, 'graspnet1b:10')

    def test_actual_xml_world_anchor_geometry_proof(self):
        proof = grasp.cpu_frame_geometry_proof(self.source(), DATA)
        self.assertTrue(proof['passed'], proof)
        self.assertEqual(proof['physics_steps'], 0)
        self.assertEqual(proof['renderers_created'], 0)
        self.assertFalse(proof['robot_physics_equivalent'])
        self.assertLessEqual(proof['maximum_translation_error_m'], 1e-12)

    def test_per_instance_frame_hooks_run_before_any_step_without_double_shift(self):
        import mujoco
        def update():
            raw.mujoco.mjModel = mujoco.MjModel.from_xml_string('<mujoco><worldbody><camera name="wrist" pos="1.5 0 .8"/></worldbody></mujoco>')
            raw.mujoco.mjData = mujoco.MjData(raw.mujoco.mjModel)
        cameras = {'wrist': SimpleNamespace(mount='eye_in_hand'), 'side': SimpleNamespace(mount='eye_on_base', pose=SimpleNamespace(position=[.3, .9, 1.46], quaternion=[1., 0, 0, 0]))}
        calls = []
        prim = SimpleNamespace(GetAttribute=lambda name: SimpleNamespace(Set=lambda xyz: calls.append(tuple(xyz))))
        side = SimpleNamespace(set_local_pose=lambda **kw: calls.append(kw))
        raw = SimpleNamespace(task=SimpleNamespace(adapter_task_id=grasp.TASK_ID, layout=SimpleNamespace(cameras=cameras,
            scene=SimpleNamespace(uid='room', center_offset=[.15, 0, .785]))),
            mujoco=SimpleNamespace(update_layout=update), isaac=SimpleNamespace(update_layout=lambda: None,
                scenes={'room': (prim, {'position': [0, 0, .9]})}, cameras={'side': side}))
        grasp.install_frame_adapter(raw)
        for _ in range(2):
            raw.mujoco.update_layout()
            np.testing.assert_allclose(raw.mujoco.mjModel.cam_pos[0], [1.5, 0, 1.585])
        raw.isaac.update_layout()
        np.testing.assert_allclose(calls[0], [.15, 0, -.115])
        np.testing.assert_allclose(calls[1]['translation'], [.3, .9, .675])
        with self.assertRaises(ValueError):
            grasp.install_frame_adapter(raw)


if __name__ == '__main__':
    unittest.main()
