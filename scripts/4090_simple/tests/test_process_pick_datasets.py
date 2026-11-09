"""CPU-only orchestration checks, including real compacted-PNG resume."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import process_pick_datasets as process
import pico_pick_audit as auditor
import pico_pick_rgb_storage as storage
import test_pico_pick_rgb_storage as rgb_fixture


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


class PickProcessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def rows(self, count=25):
        rows = []
        for index in range(count):
            root = self.root/str(index)
            root.mkdir()
            rows.append({'directory': str(root), 'raw30_frames': 10})
        return rows

    def test_chunk_capacity_keeps_master_workspace_and_maximum20(self):
        rows = self.rows()
        reserve = process.RESERVE_BYTES
        per_episode = 10 * process.FRAME_BUDGET_BYTES
        self.assertEqual(len(process.choose_chunk(rows, 0, free_bytes=reserve+21*per_episode)), 20)
        self.assertEqual(len(process.choose_chunk(rows, 0, free_bytes=reserve+4*per_episode)), 3)
        with self.assertRaises(process.ProcessingDeferred):
            process.choose_chunk(rows, 0, free_bytes=reserve+2*per_episode-1)
        self.assertFalse(any((Path(row['directory'])/'capture').exists() for row in rows))

    def test_chunk_stops_before_existing_outputs_never_rerenders(self):
        rows = self.rows(5)
        (Path(rows[2]['directory'])/'capture').mkdir()
        self.assertEqual(process.choose_chunk(rows, 0, free_bytes=1 << 40), rows[:2])
        with self.assertRaises(process.ProcessingDeferred):
            process.choose_chunk(rows, 2, free_bytes=1 << 40)

    def test_original_byte_tamper_detected_even_with_restored_size_mtime(self):
        root = self.root/'episode'
        root.mkdir()
        # Supplemental117 deliberately has no official_initialization field.
        metadata = root/'metadata.json'
        write_json(metadata, {'task_id': process.TASKS['xmove_pick'], 'human_demo_uuid': 'supplement117'})
        raw = root/'raw.bin'
        raw.write_bytes(b'abcde')
        row = {'directory': str(root), 'task_id': process.TASKS['xmove_pick'], 'uuid': 'supplement117',
            'source_files': {path.name: {'size_bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns,
                'sha256': process.sha(path)} for path in (metadata, raw)}}
        self.assertEqual(process.verify_originals([row]), 2)
        stat = raw.stat()
        raw.write_bytes(b'ABCDE')
        os.utime(raw, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'Frozen original bytes'):
            process.verify_originals([row])

    def test_pack_only_environment_and_capacity_preflight(self):
        (self.root/'logs').mkdir()
        plan = {'processing_root': str(self.root), 'reserve_free_bytes': process.RESERVE_BYTES,
            'pack_pythonpath': '/private/pillow:/scripts', 'project_log': str(self.root/'log.md')}
        fake_child = SimpleNamespace(wait=lambda: 0)
        with patch.object(process, 'require_idle'), patch.object(process.shutil, 'disk_usage',
                return_value=SimpleNamespace(free=1 << 40)), patch.object(process.subprocess, 'Popen',
                return_value=fake_child) as launch, patch.dict(os.environ, {'PYTHONPATH': 'base-path'}):
            process.run_child(plan, ['model-python', 'pack.py'], 'pack', packing=True)
            self.assertEqual(launch.call_args.kwargs['env']['PYTHONPATH'], plan['pack_pythonpath'])
            self.assertEqual(launch.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            process.run_child(plan, ['model-python', 'dino.py'], 'dino', gpu=True)
            self.assertEqual(launch.call_args.kwargs['env']['PYTHONPATH'], 'base-path')
        with patch.object(process, 'require_idle'), patch.object(process.shutil, 'disk_usage',
                return_value=SimpleNamespace(free=process.RESERVE_BYTES+99)), patch.object(process.subprocess, 'Popen') as launch:
            with self.assertRaises(process.ProcessingDeferred):
                process.run_child(plan, ['model-python'], 'does-not-launch', required_extra_bytes=100)
            launch.assert_not_called()
        self.assertFalse((self.root/'logs/does-not-launch.log').exists())

    def test_child_failure_preserves_log_and_never_marks_ready(self):
        (self.root/'logs').mkdir()
        plan = {'processing_root': str(self.root), 'reserve_free_bytes': process.RESERVE_BYTES,
            'project_log': str(self.root/'log.md')}
        with patch.object(process, 'require_idle'), patch.object(process.shutil, 'disk_usage',
                return_value=SimpleNamespace(free=1 << 40)), patch.object(process.subprocess, 'Popen',
                return_value=SimpleNamespace(wait=lambda: 9)):
            with self.assertRaisesRegex(ValueError, 'exited9'):
                process.run_child(plan, ['python', 'child'], 'failure')
        self.assertTrue((self.root/'logs/failure.log').is_file())
        self.assertFalse((self.root/'PICO_DATASET_READY.json').exists())

    def test_pack_capacity_counts_copies_but_not_same_device_sealed_ledgers(self):
        import pack_pico_handover as packer
        root, template = self.root/'episode', self.root/'template'
        (root/'capture').mkdir(parents=True)
        (root/'deferred_capture').mkdir()
        template.mkdir()
        for name in packer.SOURCE_FILES:
            (template/name).write_bytes(b'{}')
        shared = self.root/'source.py'
        shared.write_bytes(b'sealed source')
        trace = self.root/'trace.jsonl'
        trace.write_bytes(b'x'*(1 << 20))
        commands = root/'commands.jsonl'
        commands.write_bytes(b'a'*(1 << 20))
        metadata = root/'metadata.json'
        metadata.write_bytes(b'{}')
        (root/'capture/rgb_master.mp4').write_bytes(b'rgb'*(1 << 16))
        for relative in ('collection_record.json', 'capture/rgb_storage.json', 'capture/rgb_storage_complete.json'):
            write_json(root/relative, {})
        write_json(root/'rtx_completion.json', {'producer_source_sha256': {str(shared): process.sha(shared)}})
        protocol = packer.task_protocol(process.TASKS['bend_pick'])
        report = {'bindings': {str(path): process.sha(path) for path in (commands, metadata)},
            'auditor_source_sha256': {str(shared): process.sha(shared)},
            protocol['reference_key']: {'control_trace': {'path': str(trace), 'sha256': process.sha(trace)},
                'historical_sources': {'manifest_path': str(shared), 'manifest_sha256': process.sha(shared),
                    'files': [{'path': str(shared), 'sha256': process.sha(shared), 'filename': shared.name}]}}}
        write_json(root/'admission.json', report)
        rows = [{'directory': str(root), 'uuid': 'uuid', 'task_id': process.TASKS['bend_pick'], 'raw30_frames': 400}]
        with patch.object(packer, 'pick_batch_sources', return_value={}):
            base = process.pack_capacity_bytes(rows, self.root/'packed', template)
            commands.write_bytes(commands.read_bytes()*2)
            self.assertEqual(process.pack_capacity_bytes(rows, self.root/'packed', template), base)
            metadata.write_bytes(b'm'*(1 << 20))
            self.assertEqual(process.pack_capacity_bytes(rows, self.root/'packed', template), base+(1 << 20)-4096)
            if Path('/dev/shm').is_dir() and Path('/dev/shm').stat().st_dev != root.stat().st_dev:
                other_device = process.pack_capacity_bytes(rows, '/dev/shm/pick-budget-not-created', template)
                self.assertGreater(other_device, base+3*(1 << 20))
        self.assertGreater(base, 400 * (16 << 10) + 2 * (1 << 20))

    def test_ready_cache_is_bound_to_current_task_and_every_payload(self):
        release = self.root/'release'
        payload = release/'packed/train.bin'
        write_json(payload, {'value': 1})
        reference = self.root/'reference.json'
        write_json(reference, {'reference': True})
        reference_dino = self.root/'reference-dino'
        write_json(reference_dino/'manifest.json', {'reference-dino': True})
        write_json(release/'packed/PACK_READY.json', {'pack': True})
        write_json(release/'dino/manifest.json', {'cache': True})
        plan = {'run_config': str(reference), 'reference_dino': str(reference_dino)}
        def identity(path):
            return {'sha256': process.sha(path), 'size_bytes': path.stat().st_size,
                'mtime_ns': path.stat().st_mtime_ns}
        ready_path = release/'PICO_DATASET_READY.json'
        ready = {'contract': 'cvpr-pico-pick-production-verification-v1', 'status': 'passed',
            'task_id': process.TASKS['bend_pick'], 'train_episodes': 100,
            'production_ready_for_bound_pack_and_cache': True, 'training_started': False,
            'verifier_sha256': process.sha(process.SCRIPT/'verify_pico_handover.py'),
            'release_root': str(release), 'pack_root': str(release/'packed'), 'dino_root': str(release/'dino'),
            'pack_ready_sha256': process.sha(release/'packed/PACK_READY.json'),
            'cache_manifest_sha256': process.sha(release/'dino/manifest.json'),
            'payload_files': {str(path.relative_to(release)): identity(path)
                for directory in (release/'packed', release/'dino') for path in directory.rglob('*') if path.is_file()},
            'external_reference_files': {str(path): identity(path) for path in (reference, reference_dino/'manifest.json')}}
        write_json(ready_path, ready)
        self.assertEqual(process.verify_ready(ready_path, release, plan, 'bend_pick', 100), ready)
        with self.assertRaisesRegex(ValueError, 'readiness'):
            process.verify_ready(ready_path, release, plan, 'xmove_pick', 100)
        other = self.root/'different-config.json'
        write_json(other, {'different': True})
        with self.assertRaisesRegex(ValueError, 'configuration'):
            process.verify_ready(ready_path, release, dict(plan, run_config=str(other)), 'bend_pick', 100)
        stat = payload.stat()
        payload.write_bytes(payload.read_bytes().replace(b'1', b'2'))
        os.utime(payload, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'Bound production input changed'):
            process.verify_ready(ready_path, release, plan, 'bend_pick', 100)


class CompactedResumeTests(unittest.TestCase):
    def setUp(self):
        # Reuse an actual RGB/physics fixture; no simulation, GPU or subprocess.
        self.fixture = rgb_fixture.ExactRGBStorageTests()
        self.fixture.setUp()
        self.root = self.fixture.root
        self.row = {'directory': str(self.root), 'uuid': 'human-uuid', 'profile': 'bend_pick',
            'task_id': process.TASKS['bend_pick'], 'number': 7, 'raw30_frames': 5}
        self.report = {'contract': auditor.CONTRACT, 'task_id': self.row['task_id'],
            'human_demo_uuid': self.row['uuid'], 'episode_directory': str(self.root),
            'eligible_for_packing': True, 'status': 'independent_live_human_episode_audit_passed',
            'selection_policy': 'native_only', 'technical_blocked': False, 'errors': [],
            'accepted_new_dataset': False, 'checks': {name: {'passed': True} for name in auditor.BASE_CHECKS},
            'pick_supervised_coverage': {'supervised_native_success_observed': True},
            'auditor_source_sha256': {str(process.SCRIPT/name): process.sha(process.SCRIPT/name)
                for name in auditor.AUDITOR_FILES},
            'bindings': {str(path): process.sha(path) for path in self.fixture.original}}
        self.report['bindings'][str(self.fixture.raw)] = process.sha(self.fixture.raw)
        write_json(self.root/'admission.json', self.report)

    def tearDown(self):
        self.fixture.tearDown()

    def test_removed_png_resume_uses_one_full_proof_not_one_per_png(self):
        storage.compact_episode(self.root)
        with (patch.object(storage, 'verify_compacted_episode', wraps=storage.verify_compacted_episode) as proof,
                patch.object(auditor, 'audit_episode', side_effect=AssertionError('must never re-audit removed PNGs'))):
            self.assertEqual(process.reusable_admission(self.row), self.report)
            self.assertEqual(proof.call_count, 1)
        self.assertFalse(any(path.exists() for path in self.fixture.original))

    def test_missing_png_without_master_proof_is_never_accepted(self):
        next(iter(self.fixture.original)).unlink()
        with self.assertRaisesRegex(ValueError, 'no exact current PNG proof'):
            process.reusable_admission(self.row)

    def test_compacted_admission_or_authority_tamper_is_not_reaudited(self):
        storage.compact_episode(self.root)
        for field, value in [('human_demo_uuid', 'different'), ('task_id', process.TASKS['xmove_pick'])]:
            changed = dict(self.report, **{field: value})
            write_json(self.root/'admission.json', changed)
            with self.assertRaisesRegex(ValueError, 'rejected/stale'):
                process.reusable_admission(self.row)
        changed = copy.deepcopy(self.report)
        changed['auditor_source_sha256'][next(iter(changed['auditor_source_sha256']))] = '0'*64
        write_json(self.root/'admission.json', changed)
        with self.assertRaisesRegex(ValueError, 'auditor changed'):
            process.reusable_admission(self.row)

    def test_admit_and_compact_reuses_pilot_without_render_or_audit(self):
        storage.compact_episode(self.root)
        plan = {'historical_sources': 'unused', 'project_log': str(self.root/'log.md')}
        with (patch.object(process, 'verify_originals', return_value=1),
                patch.object(auditor, 'audit_episode', side_effect=AssertionError('pilot must be reused'))):
            result = process.admit_and_compact(plan, self.row)
        self.assertTrue(result['reused_admission'])
        self.assertEqual(result['stage'], 'admitted_and_compacted')
        self.assertEqual(result['number'], 7)
        self.assertTrue(storage.verify_compacted_episode(self.root)['passed'])


if __name__ == '__main__':
    unittest.main()
