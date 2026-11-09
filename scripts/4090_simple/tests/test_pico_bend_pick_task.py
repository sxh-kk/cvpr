"""BendPick100 provenance, native reward and the task-specific floor shift."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pico_bend_pick_collection_plan as plan
from pico_bend_pick_collection_progress import CollectionProgress
import pico_bend_pick_task as bend
import pico_bend_pick_quality as quality
from test_pico_grasp_task import fixture as grasp_fixture

SOURCE = Path('/home/zxc/CVPR_Evaluation_runtime/core6_v11_100/sources/G1WholebodyBendPickMP-v0/meta/episodes.jsonl')
DATA = Path('/home/zxc/CVPR_Evaluation_runtime/pico_grasp_official_20261102/preparation_001/resources/data_root')


def fixture():
    state = grasp_fixture()
    state['uid'] = bend.TASK_UID
    state['dr_state_dict']['spatial']['g1_wholebody']['position'][2] = .38
    return state


class BendFrameTests(unittest.TestCase):
    def test_native_floor_shift_is405_not785_and_source_is_unchanged(self):
        source = fixture()
        before = copy.deepcopy(source)
        state, receipt = bend.adapt_official_config(source)
        self.assertEqual(source, before)
        self.assertEqual(receipt['world_translation_m'], [0., 0., .405])
        self.assertEqual(receipt['native_ground_z_m'], -.405)
        self.assertEqual(receipt['native_robot_spawn_z_m'], .38)
        self.assertEqual(receipt['sonic_encoder_mode'], 2)
        self.assertEqual(receipt['sonic_control_policy'], 'full_body_smpl_mode2')
        self.assertFalse(receipt['robot_physics_equivalent'])
        self.assertAlmostEqual(state['dr_state_dict']['spatial']['g1_sonic']['position'][2], -.008)
        self.assertAlmostEqual(state['layout']['actors']['table']['pose']['position'][2], .355)

    def test_rejects_grasp_spawn_or_changed_table_baseline(self):
        for mutate in (lambda s: s['dr_state_dict']['spatial']['g1_wholebody']['position'].__setitem__(2, 0.),
                lambda s: s['dr_state_dict']['scene']['table']['size'].__setitem__(2, .2)):
            state = fixture()
            mutate(state)
            with self.assertRaises(ValueError):
                bend.adapt_official_config(state)

    def test_controller_mode_and_spawn_declaration_cannot_be_changed(self):
        for field, value in [('sonic_encoder_mode', 1), ('native_ground_z_m', -.785),
                ('native_robot_spawn_z_m', 0.), ('sonic_control_policy', 'continuous_vr_mode1')]:
            state, _ = bend.adapt_official_config(fixture())
            state[bend.FRAME_KEY][field] = value
            with self.assertRaises(ValueError):
                bend.validate_adapted_config(state)

    def test_quality_identity_rejects_standing_mode1(self):
        state, receipt = bend.adapt_official_config(fixture())
        metadata = {'task_id': bend.TASK_ID, 'source_official_task_id': bend.SOURCE_TASK_ID,
            'actual_task_source_sha256': bend.OFFICIAL_TASK_SOURCE_SHA256,
            'actual_native_success_criteria': .9, 'evidence_object_roles': ['target'],
            'bend_pick_encoder_mode': 2, 'bend_pick_control_policy': 'full_body_smpl_mode2',
            'bend_pick_adapter_identity': bend.adapter_identity(),
            'source_code_sha256_at_session_start': {'pico_bend_pick_task.py': receipt['adapter_sha256']},
            'bend_pick_frame_adaptation': receipt, 'environment_config': state}
        self.assertTrue(quality.validate_bend_pick_task_identity(metadata)['passed'])
        metadata['bend_pick_encoder_mode'] = 1
        self.assertFalse(quality.validate_bend_pick_task_identity(metadata)['passed'])

    def test_progress_rejects_other_profile(self):
        rows = [{'official_episode_index': i, 'task_id': plan.TASK_ID,
            'plan_sha256': 'a'*64} for i in range(100)]
        with tempfile.TemporaryDirectory() as directory:
            progress = CollectionProgress(rows, Path(directory)/'progress.json')
            try:
                self.assertEqual(progress.receipt()['total'], 100)
                self.assertEqual(progress.acceptance_gate_version, quality.GATE_VERSION)
            finally:
                progress.close()
            rows[0]['task_id'] = 'cvpr/G1SonicTabletopGraspPico-v0'
            with self.assertRaises(ValueError):
                CollectionProgress(rows, Path(directory)/'wrong.json')


@unittest.skipUnless(SOURCE.is_file(), 'requires frozen official metadata')
class FrozenSourceTests(unittest.TestCase):
    def test_all100_configs_preserved_and_each_adapts_to_low_table(self):
        rows = plan.source_rows(SOURCE.read_bytes())
        self.assertEqual(len(rows), 100)
        for row in rows:
            original = json.loads(row['original'])
            state, _ = bend.adapt_official_config(original)
            top = state['dr_state_dict']['scene']['table']
            self.assertAlmostEqual(top['pose']['position'][2] + top['size'][2]/2, .405)
            self.assertEqual(original['sensor_cfgs']['head_stereo'], state['sensor_cfgs']['head_stereo'])

    def test_build_load_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = plan.build_plan(SOURCE.resolve(), Path(directory)/'plan')
            rows = plan.load_plan(manifest)
            self.assertEqual([r['official_episode_index'] for r in rows], list(range(100)))
            Path(rows[0]['reset_config_path']).write_text('{}')
            with self.assertRaises(ValueError):
                plan.load_plan(manifest)


@unittest.skipUnless(importlib.util.find_spec('simple') and SOURCE.is_file() and DATA.is_dir(), 'requires SIMPLE/runtime resources')
class ActualTaskTests(unittest.TestCase):
    def source(self):
        from simple.utils import resolve_data_path
        state = json.loads(plan.source_rows(SOURCE.read_bytes())[0]['original'])
        for actor in state['layout']['actors'].values():
            if 'asset' in actor:
                uid = str(actor['asset']['uid'])
                path = resolve_data_path(f'assets/graspnet/ruled_models/{int(uid):03d}_ruled.usd')
                actor['asset']['usd_path'] = path
                selected = state['dr_state_dict']['target'] if actor is state['layout']['actors']['target'] else state['dr_state_dict']['distractors'][uid]
                selected['usd_path'] = path
        return state

    def fake_robot(self):
        from simple.robots.protocols import WristCamMountable, HeadCamMountable
        class Robot(WristCamMountable, HeadCamMountable):
            uid = 'g1_sonic'
            wrist_camera_orientation = head_camera_orientation = [1., 0., 0., 0.]
            def reset(self, **kwargs):
                self.spawn_pose = kwargs['spawn_pose']
        return Robot()

    def test_reset_full_binding_and_native_nine_cm_success(self):
        from simple.tasks.g1_wholebody_bend_pick_mp import G1WholebodyBendPickMP as Official
        state, _ = bend.adapt_official_config(self.source())
        original_metadata = copy.deepcopy(Official.metadata)
        with patch('simple.robots.registry.RobotRegistry.make', return_value=self.fake_robot()):
            task = bend.make_bend_pick_task({'SIMULATE_DT': .005}, adapted_config=state)
            task.reset()
            self.assertIs(type(task).compute_reward, Official.compute_reward)
            self.assertIs(type(task).check_success, Official.check_success)
            self.assertEqual(task.compute_reward({'target': [0, 0, .5184]}), 0.)
            self.assertFalse(task.check_success({'target': [0, 0, .6083]}))
            self.assertTrue(task.check_success({'target': [0, 0, .6085]}))
            binding = bend.bend_pick_environment_binding_receipt(state, task.state_dict())
            self.assertTrue(binding['matched'], binding['mismatches'])
            task.reset()
            self.assertIsNone(task._init_target_height)
        self.assertEqual(Official.metadata, original_metadata)

    def test_actual_robot_xml_zero_step_geometry_proof(self):
        proof = bend.cpu_frame_geometry_proof(self.source(), DATA)
        self.assertTrue(proof['passed'], proof)
        self.assertEqual(proof['physics_steps'], 0)
        self.assertEqual(proof['renderers_created'], 0)
        self.assertFalse(proof['robot_physics_equivalent'])
        np.testing.assert_allclose(proof['models'][0]['points']['floor'], [0., 0., -.405])
        np.testing.assert_allclose(proof['models'][1]['points']['floor'], [0., 0., 0.])


if __name__ == '__main__':
    unittest.main()
