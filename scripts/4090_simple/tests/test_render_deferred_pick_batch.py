"""CPU-only private dispatch, full-state restore, frozen cohorts and chunk proof."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import render_deferred_pick_batch as batch
from test_pico_deferred_capture import make_model


def environment():
    return {'uid': 'private_pick', 'metadata': {}, 'robot_cfg': {'uid': 'g1_sonic'},
        'sensor_cfgs': {'head': {'width': 640, 'height': 360}},
        'dr_state_dict': {'scene': {'uid': 'scene0', 'conf': {'scale': 1}, 'name': 'room', 'data_dir': 'room'}},
        'layout': {'lights': [{'uid': 'light0', 'type': 'CylinderLight'}], 'actors': {
            'table': {'uid': 'box', 'size': [1, 1, .1], 'pose': [0, 0, .4]},
            'target': {'asset': {'label': 'box', 'usd_path': 'box.usd'}, 'pose': [0, 0, .5], 'material': {'r': .5}}}}}


class PickBatchTests(unittest.TestCase):
    def test_all_three_private_factories_and_bend_only_frame_hook(self):
        for profile, task_id in batch.PROFILES.items():
            with self.subTest(profile=profile):
                module, env = mock.Mock(), mock.Mock()
                module.adapter_identity.return_value = {'task_id': task_id}
                constructor = getattr(module, 'make_' + profile + '_task')
                factory = mock.Mock(return_value=env)
                with mock.patch.object(batch, 'task_module', return_value=module), \
                     mock.patch.dict(sys.modules, {'simple.envs.sonic_loco_manip': SimpleNamespace(SonicLocoManipEnv=factory)}):
                    actual, binding = batch.make_recorded_renderer(task_id, {'dt': .005}, environment())
                self.assertIs(actual, env)
                constructor.assert_called_once_with({'dt': .005}, adapted_config=environment())
                factory.assert_called_once()
                if profile == 'bend_pick':
                    module.install_frame_adapter.assert_called_once_with(env)
                else:
                    module.install_frame_adapter.assert_not_called()
                    self.assertEqual(binding['world_translation_m'], [0., 0., 0.])
        with self.assertRaises(ValueError):
            batch.profile_for('cvpr/G1SonicTabletopGraspPico-v0')

    def test_frame_hook_failure_closes_new_environment(self):
        module, env = mock.Mock(), mock.Mock()
        module.install_frame_adapter.side_effect = RuntimeError('hook failed')
        with mock.patch.object(batch, 'task_module', return_value=module), \
             mock.patch.dict(sys.modules, {'simple.envs.sonic_loco_manip': SimpleNamespace(SonicLocoManipEnv=mock.Mock(return_value=env))}):
            with self.assertRaisesRegex(RuntimeError, 'hook failed'):
                batch.make_recorded_renderer(batch.PROFILES['bend_pick'], {}, environment())
        env.close.assert_called_once()

    def test_already_adapted_environment_is_never_adapted_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'room').mkdir(); (root/'room/room.usd').write_text('scene')
            (root/'box.usd').write_text('object')
            state = environment(); (root/'environment.json').write_text(json.dumps(state))
            module = mock.Mock(); module.validate_adapted_config.return_value = {'shift': .405}
            with mock.patch.object(batch, 'task_module', return_value=module):
                actual, proof = batch.recorded_environment(root/'environment.json', root, batch.PROFILES['bend_pick'])
            self.assertEqual(actual, state)
            self.assertTrue(proof['already_adapted_payload_unchanged'])
            module.adapt_official_config.assert_not_called()

    def test_real_full_state_restore_does_not_step_or_modify_any_integration_channel(self):
        model = make_model(); data = mujoco.MjData(model)
        data.qacc_warmstart[:] = np.arange(model.nv) * .123
        data.qvel[:] = np.arange(model.nv) * .01
        data.time = 3.25
        state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        with mock.patch('mujoco.mj_step', side_effect=AssertionError('No physics replay')):
            before = batch.restore_recorded_state(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
            batch.assert_recorded_state_unchanged(model, data, before, mujoco.mjtState.mjSTATE_INTEGRATION)
        np.testing.assert_array_equal(state, before)
        self.assertEqual(batch.file_sha256(batch.original.__file__), batch.ORIGINAL_RENDERER_SHA256)

    def test_pool_reuses_one_private_environment_but_rejects_geometry_and_task_aliases(self):
        env = mock.Mock(); pool = batch.RendererPool(mock.Mock(return_value=(env, {})))
        task_id = batch.PROFILES['xmove_pick']
        for _ in range(2):
            lease, _ = pool.acquire(task_id, {}, environment()); lease.reset(); lease.close()
        self.assertEqual(env.reset.call_count, 2)
        env.close.assert_not_called()
        with self.assertRaises(ValueError):
            pool.acquire(batch.PROFILES['bend_pick'], {}, environment())
        changed = environment(); changed['layout']['actors']['table']['size'][0] = 2
        with self.assertRaises(ValueError):
            pool.acquire(task_id, {}, changed)
        pool.close(); pool.close(); env.close.assert_called_once()

    def make_frozen(self, root):
        rows = []
        for profile, task_id in batch.PROFILES.items():
            for slot in range(100):
                directory = root/profile/str(slot); directory.mkdir(parents=True)
                (directory/'deferred_capture').mkdir()
                metadata = {'human_demo_uuid': profile + str(slot), 'task_id': task_id,
                    'official_initialization': {'official_episode_index': slot}, 'environment_config': environment()}
                source = {'metadata.json': metadata, 'episode.json': {},
                    'deferred_capture/capture.json': {'metadata': metadata, 'event_count': 1},
                    'deferred_capture/events.jsonl': {}, 'deferred_capture/model.mjb': {},
                    'deferred_capture/states.npy': {}, 'raw_quality_samples.jsonl': {},
                    'applied_force_samples.jsonl': {}, 'commands.jsonl': {}}
                inventory = {}
                for name, content in source.items():
                    path = directory/name; path.write_text(json.dumps(content))
                    inventory[name] = {'size_bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns,
                        'sha256': batch.file_sha256(path)}
                rows.append({'uuid': metadata['human_demo_uuid'], 'task_id': task_id, 'profile': profile,
                    'number': slot+1, 'cohort_slot': slot, 'official_episode_index': slot,
                    'directory': str(directory), 'source_files': inventory})
        candidate = {'contract': batch.FROZEN_CONTRACT, 'count': 300, 'episodes': rows,
            'expected_task_counts': {profile: 100 for profile in batch.PROFILES}}
        candidate_path = root/'candidate.json'; candidate_path.write_text(json.dumps(candidate))
        excluded = {key: rows[1][key] for key in ('uuid', 'task_id', 'directory', 'number')}
        processing = {**candidate, 'count': 299, 'episodes': rows[:1]+rows[2:], 'candidate_count': 300,
            'candidate_manifest_path': str(candidate_path), 'candidate_manifest_sha256': batch.file_sha256(candidate_path),
            'expected_task_counts': {'xmove_pick': 99, 'bend_pick': 100, 'xmove_bend_pick': 100},
            'excluded_episodes': [{**excluded, 'reason': 'no supervised success', 'excluded_from_processing': True}]}
        path = root/'processing.json'; path.write_text(json.dumps(processing))
        return path, processing

    def test_frozen299_exact_candidate_difference_source_hashes_and_explicit_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); path, manifest = self.make_frozen(root)
            with mock.patch.object(batch, 'recorded_task', side_effect=lambda m: m['task_id']):
                self.assertEqual(batch.frozen_inputs(path)['count'], 299)
                changed = copy.deepcopy(manifest); changed['expected_task_counts']['xmove_pick'] = 100
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'counts'):
                    batch.frozen_inputs(path)
                changed = copy.deepcopy(manifest); changed['episodes'][0]['cohort_slot'] = 99
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'changed from candidate'):
                    batch.frozen_inputs(path)
                path.write_text(json.dumps(manifest))
                source = Path(manifest['episodes'][0]['directory'])/'commands.jsonl'
                source.write_text('corruption')
                with self.assertRaisesRegex(ValueError, 'source stat changed'):
                    batch.frozen_inputs(path)

    def test_partial_outputs_and_capacity_failure_never_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertIsNone(batch.completed_output(root))
            (root/'rtx_frame_bindings.jsonl').write_text('original')
            with self.assertRaisesRegex(ValueError, 'Partial'):
                batch.completed_output(root)
            self.assertEqual((root/'rtx_frame_bindings.jsonl').read_text(), 'original')
            (root/'rtx_frame_bindings.jsonl').unlink()
            (root/'deferred_capture').mkdir(); (root/'deferred_capture/capture.json').write_text('{"event_count":100}')
            with mock.patch.object(batch.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)):
                with self.assertRaisesRegex(ValueError, 'Insufficient'):
                    batch.capacity_preflight([{'directory': str(root)}])
            self.assertFalse((root/'capture').exists())

    def args_rows(self, root):
        (root/'manifest.json').write_text('{}')
        rows = []
        for index in range(3):
            directory = root/str(index); directory.mkdir()
            rows.append({'uuid': 'uuid'+str(index), 'task_id': batch.PROFILES['xmove_pick'],
                'profile': 'xmove_pick', 'number': index+1, 'directory': str(directory)})
        return SimpleNamespace(manifest=root/'manifest.json', profile='xmove_pick', data_root=root,
            report=root/'report.json', offset=1, limit=1), rows

    def test_chunk_offset_producer_binding_and_environment_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args, rows = self.args_rows(root)
            env = mock.Mock(); factory = mock.Mock(return_value=(env, {}))
            def render(args, pool):
                lease, _ = pool.acquire(batch.PROFILES['xmove_pick'], {}, environment())
                lease.reset(options={'state_dict': str(args.episode)}); lease.close()
                return {'completed_frames': 2, 'environment_binding': {'matched': True}}
            with mock.patch.object(batch, 'frozen_inputs', return_value={'count': 299, 'episodes': rows}), \
                 mock.patch.object(batch, 'capacity_preflight', return_value=[]), \
                 mock.patch.object(batch, 'completed_output', return_value=None), \
                 mock.patch.object(batch, 'make_recorded_renderer', factory), \
                 mock.patch.object(batch, 'render_episode', side_effect=render):
                report = batch.run(args)
            self.assertEqual(report['selected_uuids'], ['uuid1'])
            self.assertEqual(report['offset'], 1)
            self.assertEqual(report['status'], 'completed_pending_independent_audit')
            self.assertTrue(report['cleanup_return_verified'])
            self.assertFalse((root/'0/rtx_completion.json').exists())
            execution = batch.load_json(root/'1/rtx_completion.json')['batch_execution']
            self.assertEqual(execution['frozen_manifest_sha256'], batch.file_sha256(args.manifest))
            self.assertEqual(execution['batch_code_sha256'], batch.file_sha256(batch.__file__))
            env.close.assert_called_once()

    def test_close_hard_exit_flushes_report_before_exit_for_supervisor_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args, rows = self.args_rows(root)
            program = '''
import os, sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
sys.path.insert(0,sys.argv[1]); import render_deferred_pick_batch as b
p=Path(sys.argv[2]); row={'uuid':'u','task_id':b.PROFILES['xmove_pick'],'profile':'xmove_pick','number':1,'directory':str(p/'0')}
args=SimpleNamespace(manifest=p/'manifest.json',profile='xmove_pick',data_root=p,report=p/'report.json',offset=0,limit=1)
def close(pool):
    report=b.load_json(args.report)
    assert report['status']=='render_completed_pending_process_exit'
    assert report['selected_uuids']==['u'] and report['cleanup_return_verified'] is False
    os._exit(0)
with mock.patch.object(b,'frozen_inputs',return_value={'count':299,'episodes':[row]}), mock.patch.object(b,'capacity_preflight',return_value=[]), mock.patch.object(b,'completed_output',return_value=None), mock.patch.object(b,'render_episode',return_value={'completed_frames':1,'environment_binding':{'matched':True}}), mock.patch.object(b.RendererPool,'close',close):
    b.run(args)
raise AssertionError('Must hard exit')
'''
            result = subprocess.run([sys.executable, '-c', program, str(Path(batch.__file__).parent), str(root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = batch.load_json(args.report)
            self.assertEqual(report['status'], 'render_completed_pending_process_exit')
            self.assertNotIn('finished_utc', report)


if __name__ == '__main__':
    unittest.main()
