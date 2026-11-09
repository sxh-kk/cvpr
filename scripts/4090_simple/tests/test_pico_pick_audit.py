"""Independent pick admission dispatch, supervised coverage and CPU references."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pico_pick_audit as audit
from handover_contact_quality import SOLVED_STATE_STAGE


def rows(success=True):
    result = []
    for index in range(60):
        lift = .1 if success and index >= 50 else 0.
        result.append({'completed_steps': index + 1, 'sim_time_s': 1. + (index + 1)/50,
            'solver_stage': SOLVED_STATE_STAGE, 'actual_target_xyz_m': [0., 0., .5 + lift],
            'native_baseline_target_z_m': .5, 'native_task_reward': lift/.1,
            'official_task_success': bool(lift)})
    return result


class NativeAdmissionTests(unittest.TestCase):
    def test_unknown_or_aliased_task_never_falls_back_to_handover(self):
        for identity in ('unknown', 'simple/G1WholebodyXMovePickTeleop-v0', 'cvpr/G1SonicTabletopGraspPico-v0'):
            with self.assertRaises(ValueError):
                audit.task_profile({'task_id': identity})
        self.assertEqual(len(audit.PROFILES), 3)

    def test_native_gate_dispatches_all_three_and_rejects_corrupted_flag(self):
        for identity in audit.PROFILES:
            with self.subTest(identity=identity):
                result = audit.native_gate(rows(), 1., {'task_id': identity})
                self.assertEqual(result['task_id'], identity)
                self.assertTrue(result['official_task_success'])
                damaged = rows()
                damaged[0]['official_task_success'] = True
                with self.assertRaisesRegex(ValueError, 'boolean differs'):
                    audit.native_gate(damaged, 1., {'task_id': identity})

    def test_native_reward_recomputation_uses_refreshed_same_time_archive(self):
        quality = rows()
        # The solved-before-refresh pose is deliberately different. Native
        # reward belongs to the separately refreshed state at this same time.
        quality[50]['actual_target_xyz_m'][2] = .58
        event = {'physical_tick50': 51, 'sim_time_s': quality[50]['sim_time_s'],
            'objects': {'target': {'pose7': [0., 0., .6, 1., 0., 0., 0.]}}}
        metadata = {'task_id': next(iter(audit.PROFILES))}
        result = audit.native_gate(quality, 1., metadata, [event])
        self.assertEqual(result['native_reward_recomputed_refreshed_archive_frames'], 1)
        event['objects']['target']['pose7'][2] = .57
        with self.assertRaisesRegex(ValueError, 'refreshed archived target'):
            audit.native_gate(quality, 1., metadata, [event])

    def test_warmup_only_success_is_not_supervised_training_coverage(self):
        quality = rows(False)
        quality[0]['official_task_success'] = True
        for identity in audit.PROFILES:
            self.assertTrue(audit.native_gate(quality[:0] + [dict(row, native_task_reward=1. if index == 0 else 0.)
                for index, row in enumerate(quality)], 1., {'task_id': identity})['official_task_success'])
        with self.assertRaisesRegex(ValueError, 'supervised suffix'):
            audit.supervised_coverage(quality, 50)
        self.assertTrue(audit.supervised_coverage(rows(), 50)['supervised_native_success_observed'])

    def test_missing_raw_or_rtx_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'metadata.json').write_text(json.dumps({'task_id': next(iter(audit.PROFILES))}))
            result = audit.audit_episode(directory, write_report=False)
            self.assertFalse(result['eligible_for_packing'])
            self.assertEqual(result['contract'], audit.CONTRACT)
            self.assertTrue(result['errors'])
            self.assertFalse(Path(directory, 'admission.json').exists())


class BatchProducerTests(unittest.TestCase):
    def fixture(self, root):
        episode = root/'episode'
        episode.mkdir()
        sources = {}
        for name in ('metadata.json', 'episode.json', 'commands.jsonl', 'raw_quality_samples.jsonl',
                'applied_force_samples.jsonl', 'deferred_capture/capture.json', 'deferred_capture/events.jsonl',
                'deferred_capture/model.mjb', 'deferred_capture/states.npy'):
            path = episode/name
            path.parent.mkdir(exist_ok=True)
            path.write_text('immutable input ' + name)
            stat = path.stat()
            sources[name] = {'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': audit.sha(path)}
        rows = []
        for task_id, spec in audit.PROFILES.items():
            for slot in range(100):
                selected = spec[0] == 'xmove_pick' and slot == 21
                rows.append({'uuid': str(uuid.uuid5(uuid.NAMESPACE_DNS, task_id+str(slot))),
                    'task_id': task_id, 'profile': spec[0], 'cohort_slot': slot, 'number': slot + 1,
                    'directory': str(episode if selected else root/(spec[0]+str(slot))), 'source_files': sources})
        candidate_path = root/'candidate.json'
        candidate_path.write_text(json.dumps({'contract': audit.FROZEN_CONTRACT, 'count': 300, 'episodes': rows}))
        exclusion = {key: rows[0][key] for key in ('uuid', 'task_id', 'directory', 'number')}
        exclusion.update(reason='warmup-only success; user chose later recapture', excluded_from_processing=True)
        manifest = {'contract': audit.FROZEN_CONTRACT, 'count': 299, 'candidate_count': 300,
            'episodes': rows[1:], 'expected_task_counts': {'xmove_pick': 99, 'bend_pick': 100, 'xmove_bend_pick': 100},
            'candidate_manifest_path': str(candidate_path), 'candidate_manifest_sha256': audit.sha(candidate_path),
            'excluded_episodes': [exclusion]}
        frozen = root/'processing.json'
        frozen.write_text(json.dumps(manifest))
        report_path = root/'report.json'
        wrapper = Path(audit.__file__).with_name('render_deferred_pick_batch.py').resolve()
        original = Path(audit.__file__).with_name('render_deferred_pico.py').resolve()
        execution = {'contract': audit.BATCH_CONTRACT, 'batch_code_path': str(wrapper), 'batch_code_sha256': audit.sha(wrapper),
            'original_renderer_code_path': str(original), 'original_renderer_code_sha256': audit.sha(original),
            'real_environment_count': 1, 'original_reset_and_environment_binding_per_episode': True,
            'environment_lease_number': 1, 'batch_report_path': str(report_path),
            'frozen_manifest_path': str(frozen), 'frozen_manifest_sha256': audit.sha(frozen)}
        completion = {'batch_execution': execution, 'completed_frames': 3, 'environment_binding': {'matched': True}}
        receipt = episode/'rtx_completion.json'
        receipt.write_text(json.dumps(completion))
        selected = rows[21]
        report = {'contract': audit.BATCH_CONTRACT, 'status': 'completed_pending_independent_audit',
            'accepted_new_dataset': False, 'batch_code_path': str(wrapper), 'batch_code_sha256': audit.sha(wrapper),
            'original_renderer_code_path': str(original), 'original_renderer_code_sha256': audit.sha(original),
            'cleanup_requested': True, 'cleanup_return_verified': True, 'new_environment_leases': 1,
            'manifest_path': str(frozen), 'manifest_sha256': audit.sha(frozen),
            'profile': 'xmove_pick', 'task_id': selected['task_id'], 'offset': 20,
            'selected_count': 1, 'frozen_count': 299, 'selected_uuids': [selected['uuid']],
            'episodes': [{'uuid': selected['uuid'], 'task_id': selected['task_id'], 'number': selected['number'],
                'directory': str(episode), 'completed_frames': 3, 'environment_binding': {'matched': True},
                'rtx_completion_sha256': audit.sha(receipt), 'reused_completed_output': False}]}
        report_path.write_text(json.dumps(report))
        return episode, completion, {'task_id': selected['task_id'], 'human_demo_uuid': selected['uuid']}, report_path, report

    def test_nonprefix_299_chunk_has_complete_frozen_producer_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode, completion, metadata, _, _ = self.fixture(Path(temporary))
            result = audit._check_batch_execution(episode, completion, metadata, {})
            self.assertEqual(result['contract'], audit.BATCH_CONTRACT)
            self.assertTrue(result['completed_report_binds_this_receipt'])
            self.assertEqual(result['uuid'], metadata['human_demo_uuid'])

    def test_batch_rejects_wrong_offset_producer_task_receipt_or_incomplete_status(self):
        for failure in ('offset', 'producer', 'task', 'receipt', 'running'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                episode, completion, metadata, path, report = self.fixture(Path(temporary))
                if failure == 'offset':
                    report['offset'] = 0
                elif failure == 'producer':
                    report['batch_code_sha256'] = '0' * 64
                elif failure == 'task':
                    report['episodes'][0]['task_id'] = audit.bend.TASK_ID
                elif failure == 'receipt':
                    report['episodes'][0]['rtx_completion_sha256'] = '0' * 64
                else:
                    report['status'] = 'running'
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    audit._check_batch_execution(episode, completion, metadata, {})


RUNTIME = Path('/home/zxc/CVPR_Evaluation_runtime')
EXAMPLES = (('pico_xmove_pick_official_20261107', '091'),
    ('pico_bend_pick_official_20261109', '091'), ('pico_xmove_bend_pick_official_20261109', '010'))
HAS_RAW = all((RUNTIME/name/'round_001/collection/kept'/number/'metadata.json').is_file() for name, number in EXAMPLES)


@unittest.skipUnless(HAS_RAW, 'requires retained raw CPU fixtures; never starts simulator or ONNX')
class ActualRawReferenceTests(unittest.TestCase):
    def values(self, name, number):
        directory = RUNTIME/name/'round_001/collection/kept'/number
        metadata = json.loads((directory/'metadata.json').read_text())
        commands = [json.loads(line) for line in (directory/'commands.jsonl').read_text().splitlines()][1:]
        events = [json.loads(line) for line in (directory/'deferred_capture/events.jsonl').read_text().splitlines()]
        qualities = [json.loads(line) for line in (directory/'raw_quality_samples.jsonl').read_text().splitlines()]
        return metadata, commands, events, qualities

    def test_actual_three_task_rewards_match_refreshed_archive_not_solved_pose(self):
        for name, number in EXAMPLES:
            with self.subTest(name=name):
                metadata, commands, events, qualities = self.values(name, number)
                result = audit.native_gate(qualities, events[0]['sim_time_s'], metadata, events)
                self.assertTrue(result['official_task_success'])
                self.assertEqual(result['native_reward_recomputed_refreshed_archive_frames'], len(events) - 1)

    def test_all_executed_reference_branches_and_tamper(self):
        branches = set()
        for name, number in EXAMPLES:
            metadata, commands, events, _ = self.values(name, number)
            seen, helper = set(), None
            for command, event in zip(commands, events):
                evidence = command['reference_evidence']
                branch = (evidence.get('encoder_mode'), evidence.get('contract'), evidence.get('producer'))
                if branch in seen:
                    continue
                seen.add(branch); branches.add(branch)
                frame = audit.parse_tracking(command['source_raw_packet'], command['source_host_received_s'], command['source_sequence'])
                helper = audit.check_reference(frame, event, evidence, metadata, {}, helper=helper,
                    checked_identity=metadata['encoder_identity'])
                if evidence.get('encoder_mode') == 2:
                    damaged = deepcopy(evidence)
                    damaged['reference_input_sha256'] = '0' * 64
                    with self.assertRaisesRegex(ValueError, 'SMPL input differs'):
                        audit.check_reference(frame, event, damaged, metadata, {}, checked_identity=metadata['encoder_identity'])
        self.assertEqual({value[0] for value in branches}, {0, 1, 2})
        self.assertTrue(any(value[2] == 'measured_mobile_entry_pose_static_preparation_hold' for value in branches))

    def test_historical_adapter_is_not_replaced_with_current_checkout(self):
        metadata, _, _, _ = self.values(*EXAMPLES[1])
        adapter = metadata['bend_pick_adapter_identity']['adapter_sha256']
        # The historical-source helper has its own SHA/immutability tests.
        # Here isolate the admission boundary: even a different current task
        # file must not replace the original snapshot already authenticated.
        history = {'files': [{'filename': 'pico_bend_pick_task.py', 'path': '/retained/original/pico_bend_pick_task.py', 'sha256': adapter}]}
        original_sha = audit.sha
        def independent_sha(path):
            if Path(path).name == 'pico_bend_pick_task.py':
                raise AssertionError('current adapter cannot substitute historical capture bytes')
            return original_sha(path)
        with patch.object(audit.mobile_audit, 'check_historical_sources', return_value=history) as checked, \
                patch.object(audit, 'sha', side_effect=independent_sha):
            result = audit.validate_metadata(metadata, {}, '/retained/source_manifest.json')
        checked.assert_called_once()
        self.assertFalse(result['current_adapter_substituted_for_capture'])


if __name__ == '__main__':
    unittest.main()
