"""Pick-specific transfer identities and real local remote-script byte checks."""
import copy
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import upload_pico_handover as upload
import verify_pico_handover as verify
import test_upload_pico_handover as fixtures


class UploadPickTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    def fixture(self, task_id):
        root = self.base/(upload.PICK_TASK_PREFIXES[task_id]+'local')
        root.mkdir()
        fixtures.fixture(root, task_id=task_id)
        remote = upload.REMOTE_BASE/(upload.PICK_TASK_PREFIXES[task_id]+'test')
        return root, remote, upload.inventory(root, remote)

    def execute(self, code, value, *, check=True):
        # Only substitute the remote BASE constant, keeping all its guards.
        code = code.replace(str(upload.REMOTE_BASE), str(self.base))
        return subprocess.run([sys.executable, '-B', '-c', code], input=json.dumps(value),
            text=True, capture_output=True, check=check)

    def test_all_three_native_tasks_share_contracts_with_distinct_namespaces(self):
        self.assertEqual(len(upload.PICK_TASK_IDS), 3)
        self.assertEqual(len(set(upload.PICK_TASK_PREFIXES.values())), 3)
        for task_id in sorted(upload.PICK_TASK_IDS):
            with self.subTest(task_id=task_id):
                root, remote, value = self.fixture(task_id)
                self.assertEqual(value['task_id'], task_id)
                self.assertEqual(value['contract'], upload.PICK_CONTRACT)
                self.assertEqual(upload.remote_target(remote, task_id=task_id), remote)
                self.assertFalse(value['remote_production_ready'])
                self.assertTrue(value['DINO_relocation_pending'])
                other = next(task for task in upload.PICK_TASK_IDS if task != task_id)
                with self.assertRaisesRegex(ValueError, 'task-matched'):
                    upload.inventory(root, upload.REMOTE_BASE/(upload.PICK_TASK_PREFIXES[other]+'test'))

    def test_pick_never_infers_missing_or_cross_task_ready_and_pack_identity(self):
        for task_id in sorted(upload.PICK_TASK_IDS):
            with self.subTest(task_id=task_id):
                root, remote, _ = self.fixture(task_id)
                path = root/verify.READY_NAME
                ready = verify.read(path)
                original = copy.deepcopy(ready)
                for bad in (None, upload.HANDOVER_TASK_ID, 'cvpr/UnknownPick-v0'):
                    ready['task_id'] = bad
                    fixtures.write(path, ready)
                    with self.assertRaises(ValueError):
                        upload.inventory(root, remote)
                fixtures.write(path, original)
                pack_path = root/'packed/PACK_READY.json'
                pack = verify.read(pack_path)
                for bad in (None, next(task for task in upload.PICK_TASK_IDS if task != task_id)):
                    pack['task_id'] = bad
                    fixtures.write(pack_path, pack)
                    original['payload_files']['packed/PACK_READY.json'] = verify.identity(pack_path)
                    original['pack_ready_sha256'] = verify.sha(pack_path)
                    fixtures.write(path, original)
                    with self.assertRaisesRegex(ValueError, 'pack contract'):
                        upload.inventory(root, remote)

    def test_all_three_actual_remote_scripts_check_bytes_mtime_and_explicit_task(self):
        for task_id in sorted(upload.PICK_TASK_IDS):
            with self.subTest(task_id=task_id):
                root, remote, value = self.fixture(task_id)
                target = self.base/remote.name
                value['remote_root'] = str(target)
                self.execute(upload.REMOTE_PREPARE, value)
                for relative in value['files']:
                    path = target/relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(root/relative, path)
                    os.utime(path, ns=(1, 1))
                report = json.loads(self.execute(upload.REMOTE_VERIFY, value).stdout)
                self.assertEqual(report['contract'], upload.PICK_BYTE_CONTRACT)
                self.assertEqual(report['task_id'], task_id)
                self.assertFalse(report['remote_production_ready'])
                self.assertTrue(report['DINO_relocation_pending'])
                for relative, expected in value['files'].items():
                    self.assertEqual(verify.identity(target/relative), expected)
                before = (target/upload.BYTE_NAME).read_bytes()
                invalid = dict(value, task_id=next(task for task in upload.PICK_TASK_IDS if task != task_id))
                self.assertNotEqual(self.execute(upload.REMOTE_PREPARE, invalid, check=False).returncode, 0)
                self.assertEqual((target/upload.BYTE_NAME).read_bytes(), before)
                self.assertFalse((target/'A100_PICO_DATASET_READY.json').exists())

    def test_remote_rejects_outside_base_nested_namespace_scope_and_escape_before_writing(self):
        task_id = sorted(upload.PICK_TASK_IDS)[0]
        _, remote, original = self.fixture(task_id)
        target = self.base/remote.name
        cases = [dict(original, remote_root=str(self.base/'nested'/remote.name)),
            dict(original, remote_root=str(self.base/'pico_handover_wrong')),
            dict(original, remote_root=str(target), task_id=None),
            dict(original, remote_root=str(target), contract=upload.CONTRACT),
            dict(original, remote_root=str(target), data_directories=['packed', '../../escape']),
            dict(original, remote_root=str(target), root_files=['A100_PICO_DATASET_READY.json'])]
        escaped = copy.deepcopy(original)
        escaped['remote_root'] = str(target)
        escaped['files']['packed/../../escape'] = next(iter(escaped['files'].values()))
        cases.append(escaped)
        for value in cases:
            with self.subTest(value=value):
                result = self.execute(upload.REMOTE_PREPARE, value, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(target.exists())
        # The production script itself rejects this temporary path without
        # replacing BASE; no network or real server directory is touched.
        original['remote_root'] = str(target)
        result = subprocess.run([sys.executable, '-B', '-c', upload.REMOTE_PREPARE],
            input=json.dumps(original), text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(target.exists())

    def test_remote_byte_tamper_cannot_replace_previous_success_receipt(self):
        task_id = sorted(upload.PICK_TASK_IDS)[0]
        root, remote, value = self.fixture(task_id)
        target = self.base/remote.name
        value['remote_root'] = str(target)
        self.execute(upload.REMOTE_PREPARE, value)
        for relative in value['files']:
            destination = target/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/relative, destination)
        self.execute(upload.REMOTE_VERIFY, value)
        before = (target/upload.BYTE_NAME).read_bytes()
        feature = target/'dino/features.npy'
        stat = feature.stat()
        feature.write_bytes(b'x'*stat.st_size)
        os.utime(feature, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(self.execute(upload.REMOTE_VERIFY, value, check=False).returncode, 0)
        self.assertEqual((target/upload.BYTE_NAME).read_bytes(), before)

    def test_transfer_uses_hardlink_preserving_rsync_without_real_network(self):
        for task_id in sorted(upload.PICK_TASK_IDS):
            with self.subTest(task_id=task_id):
                root, remote, _ = self.fixture(task_id)
                remote_rsync = '/remote/private tools/bin/rsync'
                def fake_run(argv, **kwargs):
                    self.assertEqual(argv[0], 'rsync')
                    self.assertIn('-a', argv)
                    self.assertIn('-H', argv)
                    self.assertIn('-z', argv)
                    self.assertIn('--compress-level=1', argv)
                    option = next(item for item in argv if item.startswith('--rsync-path='))
                    self.assertEqual(shlex.split(option.partition('=')[2]), [remote_rsync])
                    if argv[-2].endswith('/'+upload.BYTE_NAME):
                        value = verify.read(root/upload.MANIFEST_NAME)
                        fixtures.write(root/upload.BYTE_NAME, {'contract': upload.PICK_BYTE_CONTRACT,
                            'task_id': task_id, 'inventory_sha256': verify.sha(root/upload.MANIFEST_NAME),
                            'status': 'all_copied_bytes_and_identities_verified', 'remote_root': str(remote),
                            'remote_production_ready': False,
                            'cpu_runtime': {'CUDA_VISIBLE_DEVICES': '', 'nice': 19, 'cpus': [0]}})
                    return SimpleNamespace(returncode=0)
                with patch.object(upload, 'remote_python') as execute, patch.object(upload.subprocess, 'run',
                        side_effect=fake_run) as run:
                    result = upload.transfer(root, remote, host='unit-test.invalid', remote_rsync_path=remote_rsync)
                self.assertEqual(run.call_count, 4)
                self.assertEqual(execute.call_count, 2)
                self.assertEqual(result['task_id'], task_id)
                self.assertFalse(result['remote_production_ready'])
                self.assertEqual(verify.read(root/upload.MANIFEST_NAME)['remote_rsync_path'], remote_rsync)

    def test_remote_rsync_is_literal_and_pending_plan_binds_exact_path(self):
        task_id = sorted(upload.PICK_TASK_IDS)[0]
        root, remote, default = self.fixture(task_id)
        self.assertEqual(default['remote_rsync_path'], 'rsync')
        wrapper = '/gaozt-test1/private tools/bin/rsync'
        proposed = upload.inventory(root, remote, remote_rsync_path=wrapper)
        fixtures.write(root/upload.MANIFEST_NAME, proposed)
        with patch.object(upload, 'remote_python') as execute, patch.object(upload.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                upload.transfer(root, remote, remote_rsync_path='/another/rsync')
            execute.assert_not_called(); run.assert_not_called()
            for invalid in ('', '/', './rsync', 'rsync --server', '/tmp/rsync; touch /tmp/pwn',
                    '/tmp/rsync $(id)', '/tmp/rsync\nexit', '/tmp/../rsync', '/tmp/rsync '):
                with self.subTest(path=invalid), self.assertRaisesRegex(ValueError, 'literal absolute executable path'):
                    upload.inventory(root, remote, remote_rsync_path=invalid)

    def test_actual_local_rsync_preserves_hardlinks_and_source_identities(self):
        if shutil.which('rsync') is None:
            self.skipTest('rsync not available')
        source, target = self.base/'source', self.base/'target'
        source.mkdir(); target.mkdir()
        first, second = source/'first.bin', source/'second.bin'
        first.write_bytes(b'immutable-ledger'*4096)
        os.link(first, second)
        identities = {path: verify.identity(path) for path in (first, second)}
        links = first.stat().st_nlink
        subprocess.run(['rsync', '-a', '-H', '-z', '--compress-level=1', '--checksum', str(source)+'/', str(target)+'/'],
            check=True, capture_output=True)
        self.assertEqual((target/first.name).stat().st_ino, (target/second.name).stat().st_ino)
        self.assertNotEqual(first.stat().st_ino, (target/first.name).stat().st_ino)
        self.assertEqual(first.stat().st_nlink, links)
        self.assertEqual({path: verify.identity(path) for path in identities}, identities)
        self.assertEqual(verify.sha(first), verify.sha(target/first.name))


if __name__ == '__main__':
    unittest.main()
