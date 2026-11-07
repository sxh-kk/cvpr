"""Fail-closed cohort and raw-preservation checks without GPU/data mutation."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import process_grasp_dataset as process


class FrozenCohortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [{'official_episode_index': i, 'uuid': f'human-{i}',
                      'directory': str(self.root/str(i)), 'number': 126 if i == 3 else 200+i}
                     for i in range(100)]

    def plan(self, rows=None):
        path = self.root/'frozen.json'
        path.write_text(json.dumps({'contract': 'cvpr-grasp100-rtx-admission-frozen-inputs-v1',
                                   'count': 100, 'episodes': self.rows if rows is None else rows}))
        return {'contract': process.CONTRACT, 'task_id': process.TASK,
                'training_started': False, 'upload_authorized': False,
                'frozen_manifest': str(path), 'frozen_manifest_sha256': process.sha(path)}

    def test_frozen_replacement_and_manifest_tampering(self):
        plan = self.plan()
        self.assertEqual(len(process.frozen_cohort(plan)['episodes']), 100)
        Path(plan['frozen_manifest']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Frozen cohort changed'):
            process.frozen_cohort(plan)

    def test_duplicate_uuid_and_missing_scene_rejected(self):
        self.rows[1]['uuid'] = self.rows[0]['uuid']
        with self.assertRaisesRegex(ValueError, 'distinct human UUID'):
            process.frozen_cohort(self.plan())
        self.rows[1]['uuid'] = 'distinct'
        self.rows[1]['official_episode_index'] = 0
        with self.assertRaisesRegex(ValueError, 'distinct human UUID'):
            process.frozen_cohort(self.plan())

    def test_old007_cannot_reenter_or_replace126(self):
        self.rows[3]['number'] = 7
        with self.assertRaisesRegex(ValueError, 'replacement126'):
            process.frozen_cohort(self.plan())

    def test_local_scope_cannot_authorize_upload(self):
        plan = self.plan()
        plan['upload_authorized'] = True
        with self.assertRaisesRegex(ValueError, 'no training or upload'):
            process.frozen_cohort(plan)

    def test_raw_mutation_fails_even_with_preserved_size_and_mtime(self):
        directory = self.root/'episode'
        directory.mkdir()
        path = directory/'metadata.json'
        path.write_text(json.dumps({'task_id': process.TASK, 'human_demo_uuid': 'human',
                                   'official_initialization': {'official_episode_index': 3}}))
        stat = path.stat()
        row = {'directory': str(directory), 'uuid': 'human', 'official_episode_index': 3,
               'source_files': {'metadata.json': {'size_bytes': stat.st_size,
                    'mtime_ns': stat.st_mtime_ns, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}}
        frozen = {'episodes': [row]}
        self.assertEqual(process.verify_originals(frozen), 1)
        path.write_bytes(path.read_bytes().replace(b'human', b'other'))
        import os
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'Original evidence bytes changed'):
            process.verify_originals(frozen)


class ProcessExitTests(unittest.TestCase):
    def test_hard_exit_completion_requires_observed_zero_and_binds_report(self):
        self.run_case(0)

    def test_nonzero_exit_is_retained_and_never_promoted(self):
        self.run_case(9)

    def run_case(self, code):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'logs').mkdir()
            frozen = root/'frozen.json'
            frozen.write_text(json.dumps({'episodes': [{'directory': str(root/'episode')}]}))
            report = root/'batch.json'
            report.write_text(json.dumps({'status': 'render_completed_pending_process_exit'}))
            plan = {'release': str(root), 'project_log': str(root/'project.log'),
                    'frozen_manifest': str(frozen)}

            class FinishedChild:
                pid = 123

                def __init__(self, *args, **kwargs):
                    pass

                def wait(self):
                    return code

            with patch.object(process, 'require_idle'), patch.object(process.subprocess, 'Popen', FinishedChild), \
                    patch.object(process.shutil, 'disk_usage', return_value=SimpleNamespace(free=10*1024**3)):
                argv = ['python', 'render_deferred_pico_batch.py', '--report', str(report)]
                if code:
                    with self.assertRaisesRegex(ValueError, 'failed with exit9'):
                        process.run_child(plan, argv, 'rtx_test')
                else:
                    process.run_child(plan, argv, 'rtx_test')
            exit_receipt = json.loads((root/'batch_EXIT.json').read_text())
            self.assertEqual(exit_receipt['exit_code'], code)
            self.assertEqual(exit_receipt['report_sha256'], process.sha(report))
            self.assertEqual(exit_receipt['parentdriver_sha256'], process.sha(process.__file__))
            self.assertEqual(json.loads(report.read_text())['status'], 'render_completed_pending_process_exit')


if __name__ == '__main__':
    unittest.main()
