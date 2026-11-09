"""BendPick full-body mode2 target-only evidence, with real CPU FK/spawn."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_pico_grasp_evidence as fixture
from pico_async_evidence import DeferredEvidenceSession
from pico_bend_pick_quality import (TASK_ID, SOURCE_TASK_ID, OFFICIAL_SOURCE_SHA256,
    ADAPTER_CONTRACT, FRAME_KEY, GATE_VERSION, validate_bend_pick_task_identity)
from pico_deferred_capture import (CONTRACT, GRASP_CONTRACT, XMOVE_PICK_CONTRACT,
    BEND_PICK_CONTRACT, DeferredStateCapture, evidence_profile, load_deferred, prepare_deferred_model)
from pico_episode_history import _physical_precheck
from pico_episode_review import complete_geometry
from replay_handover_teacher import capture_tick_index

_grasp_metadata = fixture.metadata


def metadata():
    value = _grasp_metadata()
    old_frame = value.pop('grasp_frame_adaptation')
    value.pop('grasp_adapter_identity')
    identity = {'contract': ADAPTER_CONTRACT, 'task_id': TASK_ID,
        'source_official_task_id': SOURCE_TASK_ID, 'official_task_source_sha256': OFFICIAL_SOURCE_SHA256,
        'adapter_sha256': 'a' * 64}
    frame = {**old_frame, **identity, 'world_translation_m': [0., 0., .405],
        'native_ground_z_m': -.405, 'native_robot_spawn_z_m': .38,
        'robot_anchor_compensation_m': [0., 0., -.793],
        'sonic_encoder_mode': 2, 'sonic_control_policy': 'full_body_smpl_mode2'}
    value.update(task_id=TASK_ID, source_official_task_id=SOURCE_TASK_ID,
        actual_task_source_sha256=OFFICIAL_SOURCE_SHA256,
        source_code_sha256_at_session_start={'pico_bend_pick_task.py': 'a' * 64},
        bend_pick_adapter_identity=identity, bend_pick_frame_adaptation=frame,
        environment_config={FRAME_KEY: deepcopy(frame)}, bend_pick_encoder_mode=2,
        bend_pick_control_policy='full_body_smpl_mode2')
    return value


def episode(root, *, successful=True):
    with patch.object(fixture, 'metadata', metadata):
        return fixture.episode(root, successful=successful)


class BendPickEvidenceTests(unittest.TestCase):
    def test_independent_contract_does_not_impersonate_existing_tasks(self):
        self.assertEqual(evidence_profile({}), (CONTRACT, ('target', 'container')))
        self.assertEqual(evidence_profile(metadata()), (BEND_PICK_CONTRACT, ('target',)))
        self.assertEqual(len({CONTRACT, GRASP_CONTRACT, XMOVE_PICK_CONTRACT, BEND_PICK_CONTRACT}), 4)
        for field, value in [('task_id', SOURCE_TASK_ID),
                             ('source_official_task_id', 'simple/G1WholebodyTabletopGraspMP-v0'),
                             ('object_body_names', {'target': 'actual_target', 'container': 'distractor'})]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                evidence_profile({**metadata(), field: value})

    def test_mode2_native_selection_needs_no_mode1_planner_or_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); meta, outcome = episode(root)
            self.assertTrue(validate_bend_pick_task_identity(meta)['passed'])
            with patch('pico_grasp_quality.evaluate_grasp_capture', side_effect=AssertionError('wrong task')), \
                 patch('pico_xmove_pick_quality.evaluate_xmove_pick_capture', side_effect=AssertionError('wrong task')), \
                 patch('pico_mobile_quality.evaluate_mobile_capture', side_effect=AssertionError('wrong task')), \
                 patch('handover_contact_quality.evaluate_capture', side_effect=AssertionError('wrong task')):
                reasons, report = _physical_precheck(root, outcome, meta)
            self.assertEqual(reasons, [], report)
            self.assertEqual(report['task_id'], TASK_ID)
            self.assertEqual(report['acceptance_gate_version'], GATE_VERSION)
            self.assertEqual(report['selection_status'], 'selected')
            self.assertFalse(report['dataset_ready'])
            self.assertEqual(set(report['evaluator_source_sha256']),
                {'pico_bend_pick_quality.py', 'pico_episode_history.py'})
            loaded = load_deferred(root / 'deferred_capture')
            self.assertEqual(loaded['manifest']['contract'], BEND_PICK_CONTRACT)
            for row in loaded['events']:
                self.assertEqual(set(row['objects']), {'target'})
                self.assertNotIn('planner_reference', row['source_fields'])
                if row['action_valid']:
                    self.assertEqual(len(row['action']['body_token64']), 64)
                    self.assertEqual(len(row['action']['hand14']), 14)
            job = {'episode_json_sha256': hashlib.sha256((root / 'episode.json').read_bytes()).hexdigest(),
                'metadata_sha256': hashlib.sha256((root / 'metadata.json').read_bytes()).hexdigest()}
            result = complete_geometry(root, job)
            self.assertEqual(result['status'], 'not_required')
            self.assertTrue(result['task_identity_validation']['passed'])
            self.assertFalse((root / 'quality_geometry_completion.json').exists())

    def test_native_false_is_failure_without_new_bend_hold_or_final_state_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); meta, outcome = episode(root, successful=False)
            reasons, report = _physical_precheck(root, outcome, meta)
            self.assertTrue(reasons)
            self.assertFalse(report['technical_blocked'])
            self.assertEqual(report['selection_status'], 'official_task_not_successful')

    def test_spawn_keeps_mode2_identity_and_exact_state_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, model, meta = Path(temporary), fixture.make_model(), metadata()
            prepared = prepare_deferred_model(root / 'prepared', model=model,
                joint_names=fixture.NAMES, metadata=meta)
            session = DeferredEvidenceSession(prepared, cpus=[min(os.sched_getaffinity(0))], max_pending=128)
            try:
                self.assertEqual(session.receipt()['loaded_model']['physical_capture_contract'], BEND_PICK_CONTRACT)
                asynchronous, _, _, _ = session.begin(root / 'async', metadata=meta,
                    joint_names=fixture.NAMES, warmup_ticks=50, start_time=1.25, human_demo_uuid=fixture.UUID)
                synchronous = DeferredStateCapture(root / 'sync', metadata=meta,
                    joint_names=fixture.NAMES, warmup_ticks=50, start_time=1.25, model=model)
                data = mujoco.MjData(model)
                for tick in range(51):
                    if capture_tick_index(tick) is not None:
                        value = fixture.frame(synchronous, model, data, tick)
                        synchronous.append(**value); asynchronous.append(**value)
                self.assertEqual(synchronous.close(completed=True), asynchronous.close(completed=True))
                for name in ('states.npy', 'events.jsonl'):
                    self.assertEqual((root / 'sync' / name).read_bytes(),
                        (root / 'async/deferred_capture' / name).read_bytes())
            finally:
                self.assertTrue(session.close()['completed'])

    def test_wrong_mode_identity_contract_and_corrupt_native_evidence_are_technical(self):
        for failure in ('mode1', 'mode_bool', 'policy', 'source', 'frame', 'contract', 'row_sha', 'force', 'flag'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); meta, outcome = episode(root)
                if failure in ('mode1', 'mode_bool'):
                    meta['bend_pick_encoder_mode'] = 1 if failure == 'mode1' else True
                elif failure == 'policy':
                    meta['bend_pick_control_policy'] = 'planner_legs'
                elif failure == 'source':
                    meta['actual_task_source_sha256'] = 'b' * 64
                elif failure == 'frame':
                    meta.pop('bend_pick_frame_adaptation')
                elif failure == 'flag':
                    values = fixture.rows(); values[0].pop('official_task_success')
                    fixture.write_rows(root / 'raw_quality_samples.jsonl', values)
                elif failure == 'force':
                    path = root / 'applied_force_samples.jsonl'
                    values = [json.loads(line) for line in path.read_text().splitlines()]
                    values[0]['all_zero'] = False; fixture.write_rows(path, values)
                else:
                    capture = outcome['capture']
                    if failure == 'contract':
                        capture['contract'] = GRASP_CONTRACT
                    else:
                        path = root / 'deferred_capture/events.jsonl'
                        values = [json.loads(line) for line in path.read_text().splitlines()]
                        values[0]['integration_state_sha256'] = 'b' * 64
                        fixture.write_rows(path, values)
                        capture['events_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                    fixture.write_json(root / 'deferred_capture/capture.json', capture)
                reasons, report = _physical_precheck(root, outcome, meta)
                self.assertTrue(reasons)
                self.assertTrue(report['technical_blocked'], report)
                self.assertEqual(report['selection_status'], 'technical_evidence_incomplete')
                self.assertFalse(report['native_success_evidence_complete'])
                self.assertIsNone(report['official_task_success'])


if __name__ == '__main__':
    unittest.main()
