"""CPU-only checks of batch lifetime, cache isolation and non-overwrite rules."""
import copy
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import render_deferred_pico_batch as batch
from pico_deferred_capture import GRASP_TASK_ID


def environment():
    return {'uid': 'grasp', 'metadata': {}, 'robot_cfg': {'uid': 'g1_sonic'},
        'sensor_cfgs': {'head': {'width': 640, 'height': 360, 'mount': 'head'}},
        'dr_state_dict': {'scene': {'uid': 'scene2', 'conf': {'scale': 1}, 'name': 'room', 'data_dir': 'room'}},
        'layout': {'lights': [], 'actors': {'target': {'asset': {'label': 'can', 'usd_path': 'can.usd'},
            'pose': [0, 0, 1], 'material': {'roughness': .5}}}}}


class BatchRendererTests(unittest.TestCase):
    def test_cache_allows_new_poses_materials_and_new_asset_labels(self):
        compatible = batch.CompatibleGraspResources()
        compatible.add(environment())
        changed = environment()
        changed['layout']['actors']['target'].update(pose=[1, 0, 1], material={'roughness': .8})
        compatible.add(changed)
        changed['layout']['actors']['distractor'] = {'asset': {'label': 'bowl', 'usd_path': 'bowl.usd'}}
        compatible.add(changed)

    def test_cache_rejects_aliases_and_camera_changes(self):
        changes = [lambda e: e['sensor_cfgs']['head'].update(width=480),
            lambda e: e['robot_cfg'].update(uid='other'),
            lambda e: e['dr_state_dict']['scene']['conf'].update(scale=2),
            lambda e: e['layout']['actors']['target']['asset'].update(usd_path='other.usd')]
        for change in changes:
            with self.subTest(change=change):
                compatible = batch.CompatibleGraspResources()
                compatible.add(environment())
                changed = environment()
                change(changed)
                with self.assertRaises(ValueError):
                    compatible.add(changed)

    def test_two_leases_keep_one_real_environment_and_reset_each(self):
        env = mock.Mock()
        env.unwrapped = SimpleNamespace(task='same private task')
        binding = {'installed': True}
        factory = mock.Mock(return_value=(env, binding))
        pool = batch.RendererPool(factory)
        for _ in range(2):
            lease, actual_binding = pool.acquire(GRASP_TASK_ID, {'dt': .005}, environment())
            self.assertIs(lease.unwrapped, env.unwrapped)
            self.assertIs(actual_binding, binding)
            lease.reset(seed=4, options={'state_dict': 'new reset payload'})
            lease.close()
        factory.assert_called_once()
        self.assertEqual(env.reset.call_count, 2)
        env.close.assert_not_called()
        pool.close()
        pool.close()
        env.close.assert_called_once()

    def test_wrong_task_or_changed_sonic_cannot_reuse_environment(self):
        factory = mock.Mock(return_value=(mock.Mock(), {}))
        pool = batch.RendererPool(factory)
        with self.assertRaises(ValueError):
            pool.acquire('handover', {}, environment())
        factory.assert_not_called()
        pool.acquire(GRASP_TASK_ID, {'dt': .005}, environment())
        with self.assertRaises(ValueError):
            pool.acquire(GRASP_TASK_ID, {'dt': .01}, environment())
        pool.close()

    def test_no_output_is_fresh_but_partial_output_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertIsNone(batch.completed_output(path))
            (path/'deferred_render_environment.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Partial'):
                batch.completed_output(path)
            self.assertEqual((path/'deferred_render_environment.json').read_text(), '{}')

    def finished_output(self, path):
        (path/'capture').mkdir()
        (path/'deferred_capture').mkdir()
        source = {'manifest': 'capture.json', 'events': 'events.jsonl', 'model': 'model.mjb', 'states': 'states.npy'}
        receipt = {'status': 'deferred_rtx_completed_pending_independent_audit',
            'archived_physics_steps': 0, 'sonic_inferences': 0, 'full_integration_state_unchanged': True,
            'state_event_numeric_checks_passed': True, 'environment_binding': {'matched': True},
            'completed_frames': 1, 'frame_bindings_count': 1}
        for kind, filename in source.items():
            (path/'deferred_capture'/filename).write_bytes(b'original source')
            receipt['source_'+kind+'_sha256'] = batch.renderer.file_sha256(path/'deferred_capture'/filename)
        (path/'deferred_render_environment.json').write_text('{}')
        capture = {'completed': True}
        (path/'capture/capture.json').write_text(json.dumps(capture))
        receipt['capture'] = capture
        receipt['capture_manifest_sha256'] = batch.renderer.file_sha256(path/'capture/capture.json')
        receipt['render_code_sha256'] = batch.renderer.file_sha256(batch.renderer.__file__)
        (path/'capture/frame.png').write_bytes(b'PNG fixture; only file hash is used in resume check')
        png_sha = batch.renderer.file_sha256(path/'capture/frame.png')
        event = {'event_index': 0, 'rgb': {'path': 'frame.png', 'file_sha256': png_sha}}
        (path/'capture/events.jsonl').write_text(json.dumps(event)+'\n')
        binding = {'rtx_event_sha256': batch.renderer.canonical_event_sha256(event), 'rgb_file_sha256': png_sha}
        bridge = path/'rtx_frame_bindings.jsonl'
        bridge.write_text(json.dumps(binding)+'\n')
        receipt.update(frame_bindings_path=bridge.name, frame_bindings_sha256=batch.renderer.file_sha256(bridge))
        (path/'rtx_completion.json').write_text(json.dumps(receipt))
        return receipt

    def test_reuse_hashes_every_original_source_bridge_event_and_png(self):
        for target in ['deferred_capture/states.npy', 'rtx_frame_bindings.jsonl',
                       'capture/events.jsonl', 'capture/frame.png', 'capture/capture.json']:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                expected = self.finished_output(path)
                self.assertEqual(batch.completed_output(path), expected)
                with (path/target).open('ab') as stream:
                    stream.write(b' ')
                if target == 'capture/events.jsonl':
                    # Whitespace is semantically identical JSON and permitted;
                    # changing the actual decoded event must still fail.
                    (path/target).write_text((path/target).read_text().replace('"event_index": 0', '"event_index": 1'))
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    batch.completed_output(path)

    def test_rejected_output_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.finished_output(path)
            receipt = path/'rtx_completion.json'
            receipt.write_text(json.dumps({'status': 'rejected', 'error': 'original failure'}))
            before = receipt.read_bytes()
            with self.assertRaisesRegex(ValueError, 'Rejected'):
                batch.completed_output(path)
            self.assertEqual(receipt.read_bytes(), before)

    def test_failed_render_restores_factory_and_closes_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = path/'frozen.json'
            manifest.write_text('{}')
            args = SimpleNamespace(manifest=manifest, report=path/'report.json', limit=1, data_root=path)
            row = {'directory': str(path), 'uuid': 'u', 'number': 1}
            env = mock.Mock()
            factory = mock.Mock(return_value=(env, {}))
            def fail(_):
                lease, _ = batch.renderer.make_recorded_renderer(GRASP_TASK_ID, {}, environment())
                lease.close()
                raise RuntimeError('render failed')
            with mock.patch.object(batch, 'frozen_inputs', return_value={'episodes': [row]}), \
                 mock.patch.object(batch, 'completed_output', return_value=None), \
                 mock.patch.object(batch.renderer, 'make_recorded_renderer', factory), \
                 mock.patch.object(batch.renderer, 'render', side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, 'render failed'):
                    batch.run(args)
                self.assertIs(batch.renderer.make_recorded_renderer, factory)
            env.close.assert_called_once()
            self.assertEqual(json.loads(args.report.read_text())['status'], 'rejected')

    def test_successful_batch_resets_each_episode_and_records_wrapper_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = path/'frozen.json'
            manifest.write_text('{}')
            rows = []
            for index in range(2):
                episode = path/str(index)
                episode.mkdir()
                rows.append({'directory': str(episode), 'uuid': str(index), 'number': index})
            args = SimpleNamespace(manifest=manifest, report=path/'report.json', limit=2, data_root=path)
            env = mock.Mock()
            factory = mock.Mock(return_value=(env, {'installed': True}))
            def render(args):
                lease, _ = batch.renderer.make_recorded_renderer(GRASP_TASK_ID, {}, environment())
                lease.reset(options={'state_dict': str(args.episode)})
                lease.close()
                return {'completed_frames': 3, 'environment_binding': {'matched': True}}
            with mock.patch.object(batch, 'frozen_inputs', return_value={'episodes': rows}), \
                 mock.patch.object(batch, 'completed_output', return_value=None), \
                 mock.patch.object(batch.renderer, 'make_recorded_renderer', factory), \
                 mock.patch.object(batch.renderer, 'render', side_effect=render):
                result = batch.run(args)
            factory.assert_called_once()
            self.assertEqual(env.reset.call_count, 2)
            env.close.assert_called_once()
            self.assertEqual(result['status'], 'completed_pending_independent_audit')
            self.assertEqual(result['new_environment_leases'], 2)
            self.assertTrue(result['cleanup_requested'])
            self.assertTrue(result['cleanup_return_verified'])
            for index, row in enumerate(rows):
                receipt_path = Path(row['directory'])/'rtx_completion.json'
                receipt = json.loads(receipt_path.read_text())
                execution = receipt['batch_execution']
                self.assertEqual(execution['batch_code_sha256'], batch.renderer.file_sha256(batch.__file__))
                self.assertEqual(execution['environment_lease_number'], index+1)
                self.assertEqual(execution['real_environment_count'], 1)
                self.assertEqual(result['episodes'][index]['rtx_completion_sha256'], batch.renderer.file_sha256(receipt_path))

    def test_cleanup_failure_never_reports_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = path/'frozen.json'
            manifest.write_text('{}')
            args = SimpleNamespace(manifest=manifest, report=path/'report.json', limit=1, data_root=path)
            row = {'directory': str(path), 'uuid': 'u', 'number': 1}
            # Reused output means no renderer is initialized; isolate cleanup's
            # exception to prove it cannot leave a success status behind.
            receipt = {'completed_frames': 1, 'environment_binding': {'matched': True}}
            (path/'rtx_completion.json').write_text('{}')
            with mock.patch.object(batch, 'frozen_inputs', return_value={'episodes': [row]}), \
                 mock.patch.object(batch, 'completed_output', return_value=receipt), \
                 mock.patch.object(batch.RendererPool, 'close', side_effect=RuntimeError('close failed')):
                with self.assertRaisesRegex(RuntimeError, 'close failed'):
                    batch.run(args)
            report = json.loads(args.report.read_text())
            self.assertEqual(report['status'], 'rejected')
            self.assertIn('close failed', report['close_error'])
            self.assertFalse(report['cleanup_return_verified'])

    def test_hard_process_exit_in_close_preserves_completed_render_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'frozen.json').write_text('{}')
            (path/'rtx_completion.json').write_text('{}')
            program = '''
import os, sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
sys.path.insert(0, sys.argv[1])
import render_deferred_pico_batch as batch
path = Path(sys.argv[2])
args = SimpleNamespace(manifest=path/'frozen.json',report=path/'report.json',limit=1,data_root=path)
row = {'directory':str(path),'uuid':'u','number':1}
receipt = {'completed_frames':1,'environment_binding':{'matched':True}}
def close_and_exit(self):
    saved=batch.load_json(args.report)
    assert saved['status']=='render_completed_pending_process_exit'
    assert len(saved['episodes'])==1
    assert saved['new_environment_leases']==0
    assert saved['cleanup_requested'] is True
    assert saved['cleanup_return_verified'] is False
    assert 'render_finished_utc' in saved
    os._exit(0)
with mock.patch.object(batch,'frozen_inputs',return_value={'episodes':[row]}), \\
     mock.patch.object(batch,'completed_output',return_value=receipt), \\
     mock.patch.object(batch.RendererPool,'close',close_and_exit):
    batch.run(args)
raise AssertionError('Close must hard-exit')
'''
            result = subprocess.run([sys.executable, '-c', program,
                str(Path(batch.__file__).parent), str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = json.loads((path/'report.json').read_text())
            self.assertEqual(saved['status'], 'render_completed_pending_process_exit')
            self.assertFalse(saved['cleanup_return_verified'])
            self.assertNotIn('finished_utc', saved)


if __name__ == '__main__':
    unittest.main()
