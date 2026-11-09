"""Pick protocol, exact compacted RGB and immutable-hardlink boundaries."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pack_pico_handover as pack
import verify_pico_handover as verify
import pico_pick_audit as audit
import pico_pick_rgb_storage as storage
import test_pico_pick_rgb_storage as rgb_fixture


class PickProtocolTests(unittest.TestCase):
    def test_three_pick_tasks_have_explicit_target_only_native_protocols(self):
        for task_id in pack.PICK_TASK_IDS:
            with self.subTest(task_id=task_id):
                protocol = pack.task_protocol(task_id)
                self.assertTrue(protocol['native'])
                self.assertTrue(protocol['pick'])
                self.assertTrue(protocol['target_only'])
                self.assertEqual(protocol['task_id'], task_id)
                self.assertEqual(protocol['object_roles'], ('target',))
                self.assertEqual(protocol['reference_key'], 'pick_reference_audit')
                self.assertEqual(protocol['base_checks'], audit.BASE_CHECKS)
                self.assertEqual(protocol['auditor_files'], audit.AUDITOR_FILES)
                self.assertEqual(verify.production_protocol(task_id)['production_contract'],
                    'cvpr-pico-pick-production-verification-v1')
        with self.assertRaises(ValueError):
            pack.task_protocol('simple/G1WholebodyXMovePickTeleop-v0')
        self.assertFalse(pack.task_protocol(pack.HANDOVER_TASK_ID)['target_only'])
        self.assertTrue(pack.task_protocol(pack.GRASP_TASK_ID)['target_only'])

    def test_task_contract_alias_cannot_impersonate_another_native_task(self):
        for task_id in pack.PICK_TASK_IDS:
            value = {'task_id': task_id, 'contract': 'cvpr-pico-pick-lerobot-pack-v1'}
            self.assertEqual(pack.artifact_protocol(value, 'pack')['task_id'], task_id)
            value['contract'] = pack.GRASP_CONTRACT
            with self.assertRaises(ValueError):
                pack.artifact_protocol(value, 'pack')

    def test_batch_parent_driver_needs_its_observed_exit_pair(self):
        batch = {'contract': 'cvpr-pico-pick-deferred-rtx-batch-v1', 'uuid': 'unique-human',
            'completed_report_binds_this_receipt': True}
        bindings = {}
        for role in ('batch_code', 'original_renderer_code', 'report', 'frozen_manifest', 'parentdriver'):
            batch[role+'_path'] = '/retained/'+role
            batch[role+'_sha256'] = 'a'*64
            bindings[batch[role+'_path']] = 'a'*64
        admission = {'human_demo_uuid': 'unique-human', 'bindings': bindings,
            'deferred_rtx_bridge': {'batch_execution': batch}}
        with self.assertRaises(ValueError):
            pack.pick_batch_sources(admission)


class ImmutableRawLinkTests(unittest.TestCase):
    def test_sealed_ledger_allowlist_never_links_mutable_records_or_old_tasks(self):
        pick = pack.task_protocol(next(iter(pack.PICK_TASK_IDS)))
        old = pack.task_protocol(pack.GRASP_TASK_ID)
        for relative in pack.PICK_IMMUTABLE_LEDGERS:
            self.assertTrue(pack.immutable_evidence_link(pick, relative, True))
            self.assertFalse(pack.immutable_evidence_link(old, relative, True))
            self.assertFalse(pack.immutable_evidence_link(pick, relative, False))
        for relative in ('metadata.json', 'collection_record.json', 'admission.json',
                         'capture/rgb_storage.json', 'external/commands.jsonl'):
            self.assertFalse(pack.immutable_evidence_link(pick, relative, True))
        self.assertTrue(pack.immutable_evidence_link(old, 'deferred_capture/states.npy', True))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root/'model.mjb'
        self.raw.write_bytes(b'exact immutable raw model bytes')
        self.digest = pack.sha(self.raw)
        self.bindings = pack.Bindings()
        self.bindings.verify(self.raw, self.digest)

    def test_only_observed_single_link_can_acknowledge_changed_ctime(self):
        before, links = self.bindings.stat(self.raw), self.raw.stat().st_nlink
        target = self.root/'packed/model.mjb'
        pack._copy(self.raw, target, self.digest, immutable_hardlink=True)
        self.assertEqual(target.stat().st_ino, self.raw.stat().st_ino)
        # Some filesystems expose one timestamp for multiple quick operations;
        # require rejection when ctime actually changed, not a sleep for timing.
        if self.bindings.stat(self.raw) != before:
            with self.assertRaises(ValueError):
                self.bindings.unchanged()
        self.bindings.acknowledge_immutable_link(self.raw, self.digest, before, links)
        self.bindings.unchanged()

    def test_content_change_cannot_be_laundered_by_link_ctime_acknowledgement(self):
        before, links = self.bindings.stat(self.raw), self.raw.stat().st_nlink
        os.link(self.raw, self.root/'alias.mjb')
        self.raw.write_bytes(bytes(value ^ 1 for value in self.raw.read_bytes()))
        os.utime(self.raw, ns=(self.raw.stat().st_atime_ns, before[3]))
        with self.assertRaises(ValueError):
            self.bindings.acknowledge_immutable_link(self.raw, self.digest, before, links)

    def test_unobserved_two_links_are_not_a_single_acknowledged_operation(self):
        before, links = self.bindings.stat(self.raw), self.raw.stat().st_nlink
        os.link(self.raw, self.root/'first.mjb')
        os.link(self.raw, self.root/'second.mjb')
        with self.assertRaises(ValueError):
            self.bindings.acknowledge_immutable_link(self.raw, self.digest, before, links)

    def test_shared_trace_links_once_and_preserves_same_hash_distinct_source(self):
        staging = self.root/'release'
        relative = Path('traces')/(self.digest+'.jsonl')
        changes = []
        for _ in range(2):
            pack._shared_copy(self.raw, staging, relative, self.digest,
                immutable_hardlink=True, link_changes=changes)
        self.assertEqual(len(changes), 1)
        for source, expected, before, links in changes:
            self.bindings.acknowledge_immutable_link(source, expected, before, links)
        self.bindings.unchanged()
        other = self.root/'distinct.jsonl'
        other.write_bytes(self.raw.read_bytes())
        links_before = other.stat().st_nlink
        pack._shared_copy(other, staging, relative, self.digest,
            immutable_hardlink=True, link_changes=changes)
        self.assertEqual(len(changes), 1)
        self.assertEqual(other.stat().st_nlink, links_before)


class CompactedPickPackTests(unittest.TestCase):
    def setUp(self):
        self.fixture = rgb_fixture.ExactRGBStorageTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def test_missing_png_requires_complete_exact_reproduction_authority(self):
        image = next(iter(self.fixture.original))
        digest = hashlib.sha256(self.fixture.original[image]).hexdigest()
        image.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            pack.Bindings().verify(image, digest)
        image.write_bytes(self.fixture.original[image])
        storage.compact_episode(self.fixture.root)
        bindings = pack.Bindings()
        self.assertEqual(bindings.verify(image, digest), image)
        bindings.unchanged()
        with self.assertRaises(ValueError):
            bindings.verify(image, '0'*64)

    def test_missing_png_does_not_rehash_the_whole_master_per_frame(self):
        storage.compact_episode(self.fixture.root)
        bindings = pack.Bindings()
        master = self.fixture.capture/'rgb_master.mp4'
        with patch.object(pack, 'sha', wraps=pack.sha) as digest:
            for image, original in self.fixture.original.items():
                bindings.verify(image, hashlib.sha256(original).hexdigest())
            calls = [call for call in digest.call_args_list if Path(call.args[0]) == master]
        self.assertEqual(len(calls), 2)  # Establish expected SHA, then bind once.
        master.write_bytes(master.read_bytes()+b'changed')
        with self.assertRaises(ValueError):
            bindings.verify(next(iter(self.fixture.original)),
                hashlib.sha256(next(iter(self.fixture.original.values()))).hexdigest())

    def portable(self):
        storage.compact_episode(self.fixture.root)
        portable = Path(self.fixture.temporary.name)/'portable'
        portable.mkdir()
        master = portable/'master.mp4'
        item = {'directory': self.fixture.root, 'capture': self.fixture.capture, 'events': self.fixture.frames}
        video = pack.encode_pick_master(item, master)
        video['path'] = 'master.mp4'
        archive, payload = {}, {'master.mp4': pack.sha(master)}
        for name in ('rgb_storage.json', 'rgb_storage_complete.json'):
            target = portable/name
            shutil.copyfile(self.fixture.capture/name, target)
            archive[name] = {'path': name, 'sha256': pack.sha(target)}
            payload[name] = pack.sha(target)
        events = portable/'events.jsonl'
        shutil.copyfile(self.fixture.capture/'events.jsonl', events)
        payload['events.jsonl'] = pack.sha(events)
        return portable, payload, {'rgb_storage_archive': archive}, {'capture/events.jsonl': 'events.jsonl'}, video

    def test_portable_verifier_recreates_every_original_png_byte(self):
        portable, payload, entry, evidence, video = self.portable()
        result = verify.verify_pick_storage_video(portable, payload, entry, evidence, video)
        self.assertTrue(result['all_original_png_bytes_reproducible'])
        self.assertEqual(result['frames'], len(self.fixture.frames))
        # Corrupt only the final PNG hash: checking a first-frame sample or only
        # RGB pixels would incorrectly accept this portable payload.
        path = portable/'events.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]['rgb']['file_sha256'] = '0'*64
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        payload['events.jsonl'] = pack.sha(path)
        with self.assertRaisesRegex(ValueError, 'exact archived PNG'):
            verify.verify_pick_storage_video(portable, payload, entry, evidence, video)

    def test_portable_storage_proof_cannot_be_omitted(self):
        portable, payload, entry, evidence, video = self.portable()
        del entry['rgb_storage_archive']['rgb_storage_complete.json']
        with self.assertRaisesRegex(ValueError, 'receipts missing'):
            verify.verify_pick_storage_video(portable, payload, entry, evidence, video)

    def test_video_link_ctime_is_acknowledged_by_pack_workflow(self):
        storage.compact_episode(self.fixture.root)
        bindings = pack.Bindings()
        for image, original in self.fixture.original.items():
            bindings.verify(image, hashlib.sha256(original).hexdigest())
        item = {'directory': self.fixture.root, 'capture': self.fixture.capture, 'events': self.fixture.frames}
        output = Path(self.fixture.temporary.name)/'linked-master.mp4'
        pack.encode_pick_master(item, output)
        # Match the pack() acknowledgement stage. The video master is already
        # a bound missing-PNG authority before its low-space hardlink is made.
        for linked, expected, before, links in item.get('immutable_link_changes', []):
            bindings.acknowledge_immutable_link(linked, expected, before, links)
        bindings.unchanged()


if __name__ == '__main__':
    unittest.main()
