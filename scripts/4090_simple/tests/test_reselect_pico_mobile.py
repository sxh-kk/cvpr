"""Actual CLI smoke checks against temporary fixtures, never a real collection."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

import test_pico_episode_history as fixtures


class ReselectMobileCLITests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MobileNativeReselectionTests('runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.source, _ = self.fixture.mobile(legacy_discard=True)
        self.script = Path(__file__).resolve().parents[1] / 'reselect_pico_mobile.py'
        self.cpu = min(os.sched_getaffinity(0))

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(self.script), '--cpu', str(self.cpu), *map(str, args)],
            text=True, capture_output=True, check=False, timeout=30)

    def test_snapshot_is_nonmutating_and_explicit_apply_preserves_original_number(self):
        root = self.fixture.root
        snapshot = root / 'frozen.json'
        index_before = self.fixture.index.read_bytes()
        completed = self.run_cli('snapshot', '--index', self.fixture.index, '--uuid', self.source.name,
                                 '--output', snapshot)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.fixture.index.read_bytes(), index_before)
        self.assertEqual(self.source.resolve(), root / 'discarded/001')
        receipt = root / 'applied.json'
        completed = self.run_cli('apply', '--index', self.fixture.index, '--snapshot', snapshot,
                                 '--operation-uuid', str(uuid.uuid4()), '--output', receipt)
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        result = json.loads(receipt.read_text())
        self.assertEqual(result['status'], 'committed')
        self.assertEqual(self.source.resolve(), root / 'kept/001')
        self.assertEqual(json.loads(self.fixture.index.read_text())['next_number'], 2)
        duplicate = self.run_cli('snapshot', '--index', self.fixture.index, '--uuid', self.source.name,
                                 '--output', snapshot)
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn('already exists', duplicate.stderr)

    def test_later_record_requires_new_explicit_snapshot_and_cannot_be_auto_added(self):
        root = self.fixture.root
        snapshot = root / 'frozen.json'
        self.assertEqual(self.run_cli('snapshot', '--index', self.fixture.index, '--uuid', self.source.name,
                                     '--output', snapshot).returncode, 0)
        self.fixture.mobile()
        before = self.fixture.index.read_bytes()
        receipt = root / 'stale.json'
        completed = self.run_cli('apply', '--index', self.fixture.index, '--snapshot', snapshot,
                                 '--operation-uuid', str(uuid.uuid4()), '--output', receipt)
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertEqual(json.loads(receipt.read_text())['status'], 'error')
        self.assertEqual(before, self.fixture.index.read_bytes())
        self.assertEqual(self.source.resolve(), root / 'discarded/001')


if __name__ == '__main__':
    unittest.main()
