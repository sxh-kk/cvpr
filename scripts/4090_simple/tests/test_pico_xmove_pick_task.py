"""Native XMovePick reset isolation, reward preservation and plan provenance."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pico_xmove_pick_collection_plan as plan
from pico_xmove_pick_collection_progress import CollectionProgress
import pico_xmove_pick_task as task_module
from test_pico_grasp_task import fixture as grasp_fixture


SOURCE = Path('/home/zxc/CVPR_Evaluation_runtime/core6_v11_100/sources/G1WholebodyXMovePickTeleop-v0/meta/episodes.jsonl')
DATA = Path('/home/zxc/CVPR_Evaluation_runtime/pico_grasp_official_20261102/preparation_001/resources/data_root')


def fixture():
    state = grasp_fixture()
    state['uid'] = task_module.TASK_UID
    state['robot_cfg']['uid'] = 'g1_sonic'
    state['metadata']['physics_dt'] = .005
    state['dr_state_dict']['spatial']['g1_sonic'] = state['dr_state_dict']['spatial'].pop('g1_wholebody')
    return state


class NativeBindingTests(unittest.TestCase):
    def test_identity_receipt_changes_no_coordinate_or_config_value(self):
        source = fixture()
        original = copy.deepcopy(source)
        result, receipt = task_module.adapt_official_config(source)
        self.assertEqual(source, original)
        self.assertEqual(result.pop(task_module.FRAME_KEY), receipt)
        self.assertEqual(result, original)
        self.assertFalse(receipt['coordinate_transform_applied'])
        self.assertFalse(receipt['robot_physics_equivalent'])

    def test_rejects_wrong_task_robot_clock_and_post_motion_layout(self):
        for mutate in (lambda s: s.update(uid='wrong'),
                lambda s: s['robot_cfg'].update(uid='g1_wholebody'),
                lambda s: s['metadata'].update(physics_dt=.002),
                lambda s: s['layout']['actors']['target']['pose']['position'].__setitem__(2, 1.)):
            value = fixture()
            mutate(value)
            with self.assertRaises(ValueError):
                task_module.adapt_official_config(value)

    def test_binding_target_and_scene_tamper_fail_closed(self):
        expected, _ = task_module.adapt_official_config(fixture())
        actual = copy.deepcopy(expected)
        actual['layout']['lights'][0]['intensity'] += 1
        self.assertFalse(task_module.xmove_pick_environment_binding_receipt(expected, actual)['matched'])
        actual = copy.deepcopy(expected)
        actual['layout']['actors']['target']['asset']['uid'] = 'wrong'
        with self.assertRaises(ValueError):
            task_module.xmove_pick_environment_binding_receipt(expected, actual)

    def test_source_and_progress_do_not_accept_other_task(self):
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            plan.source_rows(b'{"episode_index":0}\n')
        rows = [{'official_episode_index': i, 'task_id': plan.TASK_ID, 'plan_sha256': 'a'*64} for i in range(99)]
        with tempfile.TemporaryDirectory() as directory:
            progress = CollectionProgress(rows, Path(directory)/'progress.json')
            try:
                self.assertEqual(progress.receipt()['total'], 99)
                self.assertEqual(progress.acceptance_gate_version, 'xmove_pick-native-task-success-v1')
            finally:
                progress.close()
            rows[0]['task_id'] = 'cvpr/G1SonicTabletopGraspPico-v0'
            with self.assertRaises(ValueError):
                CollectionProgress(rows, Path(directory)/'other.json')


@unittest.skipUnless(SOURCE.is_file(), 'requires frozen local official metadata')
class FrozenSourceTests(unittest.TestCase):
    def test_all99_derive_only_four_poses_and_authoritative_dr_lights(self):
        rows = plan.source_rows(SOURCE.read_bytes())
        self.assertEqual(len(rows), 99)
        for item in rows:
            original = json.loads(item['original'])
            reset = json.loads(item['reset'])
            self.assertEqual(original['dr_state_dict'], reset['dr_state_dict'])
            self.assertEqual(reset['layout']['lights'], list(original['dr_state_dict']['lighting'].values()))
            for mapping in item['row']['pose_mapping']:
                role = mapping['layout_role']
                self.assertEqual(reset['layout']['actors'][role]['pose'], mapping['initial_dr_spatial_pose'])
                reset['layout']['actors'][role]['pose'] = mapping['original_layout_pose']
            reset['layout']['lights'] = original['layout']['lights']
            self.assertEqual(original, reset)

    def test_frozen_plan_integrity_after_source_and_reset_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = plan.build_plan(SOURCE.resolve(), Path(directory)/'plan')
            rows = plan.load_plan(manifest)
            self.assertEqual([r['official_episode_index'] for r in rows], list(range(99)))
            config = Path(rows[0]['reset_config_path'])
            config.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'config changed'):
                plan.load_plan(manifest)


@unittest.skipUnless(importlib.util.find_spec('simple') and SOURCE.is_file() and DATA.is_dir(), 'requires installed SIMPLE/runtime resources')
class ActualTaskTests(unittest.TestCase):
    def source(self):
        value = json.loads(plan.source_rows(SOURCE.read_bytes())[0]['reset'])
        def relocate(node):
            if isinstance(node, dict):
                for key, child in node.items():
                    if isinstance(child, str) and '/SIMPLE/data/' in child:
                        node[key] = str(DATA / child.split('/SIMPLE/data/', 1)[1])
                    else:
                        relocate(child)
            elif isinstance(node, list):
                for child in node:
                    relocate(child)
        relocate(value)
        return value

    def fake_robot(self):
        from simple.robots.protocols import HeadCamMountable
        class Robot(HeadCamMountable):
            uid = 'g1_sonic'
            head_camera_orientation = [1., 0., 0., 0.]
            def reset(self, **kwargs):
                self.spawn_pose = kwargs['spawn_pose']
        return Robot()

    def test_actual_reset_and_original_reward_methods_unchanged(self):
        from simple.tasks.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop as Official
        before = copy.deepcopy(Official.metadata)
        # SIMPLE resolves asset metadata from its configured runtime root.
        from simple.utils import resolve_data_path
        source = self.source()
        for actor in source['layout']['actors'].values():
            if 'asset' in actor:
                uid = str(actor['asset']['uid'])
                path = resolve_data_path(f'assets/graspnet/ruled_models/{int(uid):03d}_ruled.usd')
                actor['asset']['usd_path'] = path
                selected = source['dr_state_dict']['target'] if actor is source['layout']['actors']['target'] else source['dr_state_dict']['distractors'][uid]
                selected['usd_path'] = path
        state, _ = task_module.adapt_official_config(source)
        with patch('simple.robots.registry.RobotRegistry.make', return_value=self.fake_robot()):
            task = task_module.make_xmove_pick_task({'SIMULATE_DT': .005}, adapted_config=state)
            task.reset()
            self.assertIs(type(task).compute_reward, Official.compute_reward)
            self.assertIs(type(task).check_success, Official.check_success)
            self.assertEqual(task.compute_reward({'target': [0, 0, .7]}), 0.)
            self.assertFalse(task.check_success({'target': [0, 0, .7899]}))
            self.assertTrue(task.check_success({'target': [0, 0, .7901]}))
            self.assertTrue(task_module.xmove_pick_environment_binding_receipt(state, task.state_dict())['matched'])
            self.assertIsNone(task.state_dict()['dr_state_dict']['material'])
            self.assertTrue(task.frame_receipt['source_material_state_missing'])
            task.reset()
            self.assertIsNone(task._init_target_height)
        self.assertEqual(Official.metadata, before)


if __name__ == '__main__':
    unittest.main()
