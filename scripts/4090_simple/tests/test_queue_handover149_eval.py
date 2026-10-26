"""Serial continuation gates: no partial, wrong-model or technical-error admission."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('serial_queue', Path(__file__).resolve().parents[1] / 'queue_handover149_eval.py')
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


class CompletionGate(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.digest = 'a' * 64
        self.cases = [dict(status='passed', not_scored=False, step=4000,
            model_sha256=self.digest, native_success=False, strict_success=False,
            case={'case_id': f'level_{level}_episode_{episode:02d}'})
            for level in range(3) for episode in range(10)]
        self.results = dict(status='complete', expected_episodes=30, accepted_episodes=30,
            step=4000, model_sha256=self.digest, cases=self.cases,
            native=dict(successes=0, episodes=30), strict=dict(successes=0, episodes=30))
        q.write(self.root / 'final_raw_acceptance.json', dict(status='passed', actual_cases=30,
            step=4000, model_sha256=self.digest))
        self.state = dict(status='complete', step=4000, model_sha256=self.digest,
            completed=[row['case']['case_id'] for row in self.cases], failures=[],
            final_raw_acceptance=q.binding(self.root / 'final_raw_acceptance.json'))
        self.save()

    def save(self):
        q.write(self.root / 'state.json', self.state)
        q.write(self.root / 'results.json', self.results)

    def test_task_failures_are_scientific_results_and_do_not_block(self):
        report = q.complete_result(self.root, 4000, self.digest)
        self.assertEqual(report['native'], dict(successes=0, episodes=30))

    def test_running_process_waits_even_with_thirty_rows(self):
        self.state.update(status='running', pid=os.getpid())
        self.save()
        self.assertIsNone(q.complete_result(self.root, 4000, self.digest))

    def test_dead_evaluator_blocks(self):
        self.state.update(status='running', pid=99999999)
        self.save()
        with self.assertRaisesRegex(RuntimeError, 'no live evaluator'):
            q.complete_result(self.root, 4000)

    def test_technical_error_is_never_a_policy_failure(self):
        self.state['failures'] = [dict(error='DDS timeout', counted_as_policy_failure=False)]
        self.save()
        with self.assertRaisesRegex(RuntimeError, 'technical error'):
            q.complete_result(self.root, 4000)

    def test_partial_final_missing_case_duplicate_and_wrong_identity_block(self):
        mutations = [
            lambda r: r.update(status='partial'),
            lambda r: r.update(accepted_episodes=29),
            lambda r: r['cases'].pop(),
            lambda r: r['cases'][1].update(case=copy.deepcopy(r['cases'][0]['case'])),
            lambda r: r['cases'][0].update(model_sha256='b' * 64),
            lambda r: r['cases'][0].update(step=5000),
            lambda r: r['cases'][0].update(not_scored=True),
            lambda r: r['cases'][0].update(status='failed'),
            lambda r: r['native'].update(successes=1),
        ]
        original = copy.deepcopy(self.results)
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.results = copy.deepcopy(original)
                mutate(self.results)
                self.save()
                with self.assertRaises(RuntimeError):
                    q.complete_result(self.root, 4000, self.digest)

    def test_changed_final_raw_receipt_blocks(self):
        raw = q.read(self.root / 'final_raw_acceptance.json')
        raw['changed'] = True
        q.write(self.root / 'final_raw_acceptance.json', raw)
        with self.assertRaisesRegex(RuntimeError, 'does not bind'):
            q.complete_result(self.root, 4000)

    def test_expected_checkpoint_and_model_are_enforced(self):
        for step, digest in ((6000, self.digest), (4000, 'b' * 64)):
            with self.subTest(step=step, digest=digest), self.assertRaises(RuntimeError):
                q.complete_result(self.root, step, digest)

    def test_new_directory_waits(self):
        self.assertIsNone(q.complete_result(self.root / 'new', 4000))

    def test_child_process_identity_rejects_reused_pid(self):
        identity = q.process_identity(os.getpid())
        self.assertTrue(q.same_live_process(identity))
        identity['start_ticks'] += 1
        self.assertFalse(q.same_live_process(identity))

    def test_last_json_and_invalid_tool_output(self):
        self.assertEqual(q.parse_last_json('diagnostic\n{"source":"ok"}\n'), {'source': 'ok'})
        with self.assertRaises(RuntimeError):
            q.parse_last_json('no receipt')


if __name__ == '__main__':
    unittest.main()
