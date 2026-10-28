import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pico_mobile_collection_progress import CollectionProgress, OPERATOR_ABORT_REASONS
from pico_teleop_recording import RecordingState


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'progress.json'
        self.rows = [{'plan_sha256':'a'*64,'official_episode_index':i} for i in range(2)]
        self.progress = CollectionProgress(self.rows,self.path)
        self.addCleanup(lambda:self.progress.close() if not self.progress.lock.closed else None)

    def submit(self, **kwargs):
        identity = str(uuid.uuid4())
        self.progress.submit(identity,Path(self.temp.name)/identity,**kwargs)
        return identity

    def result(self, identity, **updates):
        return {'latest_completed': {'uuid':identity,'pending':False,'status':'ready',
            'available':True,'collection_enabled':True,
            'acceptance_gate_version':'mobile-native-task-success-v2',
            'selection_policy':'native_only','selection_status':'selected','disposition':'kept',
            'official_task_success_observed':True,**updates}}

    def test_exact_independent_native_result_required(self):
        identity = self.submit()
        self.assertTrue(self.progress.blocked)
        self.assertFalse(self.progress.observe(self.result(str(uuid.uuid4()))))
        self.assertFalse(self.progress.observe(self.result(identity,pending=True)))
        self.assertEqual(self.progress.current['official_episode_index'],0)
        self.assertTrue(self.progress.observe(self.result(identity)))
        self.assertEqual(self.progress.current['official_episode_index'],1)
        self.assertFalse(self.progress.observe(self.result(identity)))
        self.assertEqual(self.progress.receipt()['accepted_count'],1)

    def test_failed_attempt_retries_same_initialization(self):
        identity = self.submit()
        self.assertTrue(self.progress.observe(self.result(identity,disposition='discarded',
            selection_status='official_task_not_successful',official_task_success_observed=False)))
        self.assertFalse(self.progress.blocked)
        self.assertEqual(self.progress.current['official_episode_index'],0)
        identity = self.submit()
        self.assertTrue(self.progress.observe(self.result(identity,disposition='discarded',
            selection_status='operator_discarded')))
        self.assertEqual(self.progress.receipt()['accepted_count'],0)

    def test_technical_failure_blocks_without_quality_rejection(self):
        identity = self.submit()
        self.assertFalse(self.progress.observe(self.result(identity,disposition='discarded',
            selection_status='technical_evidence_incomplete')))
        self.assertEqual(self.progress.receipt()['status'],'error')
        self.assertEqual(self.progress.state['pending']['uuid'],identity)
        self.assertEqual(self.progress.receipt()['accepted_count'],0)

    def test_incomplete_success_attribution_is_not_selected(self):
        identity = self.submit()
        self.progress.observe(self.result(identity,official_task_success_observed=None))
        self.assertEqual(self.progress.receipt()['status'],'error')

    def test_resume_pending_and_lock_against_duplicate_collectors(self):
        identity = self.submit()
        with self.assertRaisesRegex(ValueError,'另一个'):
            CollectionProgress(self.rows,self.path)
        self.progress.close()
        self.progress = CollectionProgress(self.rows,self.path)
        self.assertTrue(self.progress.blocked)
        self.assertEqual(self.progress.receipt()['pending_uuid'],identity)
        self.assertTrue(self.progress.observe(self.result(identity)))

    def test_previous_manual_discard_revokes_only_its_initialization(self):
        identity = self.submit()
        self.progress.observe(self.result(identity))
        self.assertTrue(self.progress.observe(self.result(identity,disposition='discarded',
            selection_status='operator_discarded')))
        self.assertEqual(self.progress.current['official_episode_index'],0)

    def test_completed_plan_cannot_wrap_to_zero(self):
        for _ in range(2):
            self.progress.observe(self.result(self.submit()))
        self.assertEqual(self.progress.receipt()['status'],'complete')
        self.assertIsNone(self.progress.current)
        self.assertTrue(self.progress.blocked)
        with self.assertRaises(ValueError):
            self.submit()

    def test_different_manifest_cannot_reuse_progress(self):
        self.progress.close()
        rows = [{**row,'plan_sha256':'b'*64} for row in self.rows]
        with self.assertRaisesRegex(ValueError,'冻结'):
            CollectionProgress(rows,self.path)

    def test_recovery_rejects_wrong_initialization_and_duplicate_counting(self):
        identity = self.submit()
        self.progress.close()
        original=json.loads(self.path.read_text())
        for updates in ({'accepted':{'0':identity,'1':identity},'pending':None},
                {'accepted':{'1':identity},'pending':None},
                {'pending':{**original['pending'],'official_episode_index':1}}):
            self.path.write_text(json.dumps({**original,**updates}))
            with self.assertRaises(ValueError):
                CollectionProgress(self.rows,self.path)

    def test_pending_worker_error_is_visible_and_repaired_exact_review_recovers(self):
        identity=self.submit()
        self.progress.observe(self.result(identity,pending=True,status='error',error='disk I/O failed'))
        self.assertEqual(self.progress.receipt()['status'],'error')
        self.assertEqual(self.progress.state['pending']['uuid'],identity)
        self.assertTrue(self.progress.observe(self.result(identity)))
        self.assertEqual(self.progress.receipt()['status'],'ready')

    def test_pending_later_case_survives_previous_discard_and_restart(self):
        first=self.submit()
        self.progress.observe(self.result(first))
        second=self.submit()
        self.progress.observe(self.result(first,disposition='discarded',selection_status='operator_discarded'))
        self.progress.close()
        self.progress=CollectionProgress(self.rows,self.path)
        self.assertEqual(self.progress.receipt()['current_official_episode_index'],1)
        self.progress.observe(self.result(second))
        self.assertEqual(self.progress.current['official_episode_index'],0)

    def test_late_review_cannot_release_permanent_technical_failure(self):
        identity=self.submit()
        self.progress.fail('writer failure requires diagnosis')
        self.progress.observe(self.result(identity,pending=True,status='error',error='temporary review read error'))
        self.assertFalse(self.progress.observe(self.result(identity)))
        self.assertTrue(self.progress.blocked)
        self.assertEqual(self.progress.receipt()['accepted_count'],0)

    def partial_abort_review(self, identity):
        # Actual close(false) semantics: preserve a rejected RecordingState and
        # partial capture; independent History reports technical_blocked rather
        # than inventing native-task failure for this operator-ended attempt.
        state = RecordingState()
        state.request_start(); state.on_tick(0); state.on_tick(1)
        state.reject(self.progress.state['pending']['operator_abort_reason'])
        sealed = {'state':state.close(completed=False),
            'capture':{'status':'partial_capture_rejected','requested_completed':False,'failure':None}}
        self.assertFalse(sealed['state']['pending_audit'])
        self.assertEqual(sealed['capture']['status'],'partial_capture_rejected')
        return self.result(identity,status='technical_blocked',available=False,disposition='discarded',
            discarded=True,selection_status='technical_evidence_incomplete',
            official_task_success_observed=None)

    def test_active_intentional_partial_abort_archives_then_retries_same_case(self):
        for reason in OPERATOR_ABORT_REASONS:
            with self.subTest(reason=reason):
                identity=self.submit(operator_abort_reason=reason)
                archived=self.partial_abort_review(identity)
                self.assertFalse(self.progress.observe(self.result(identity,pending=True)))
                self.assertFalse(self.progress.observe(self.result(str(uuid.uuid4()),**{
                    k:v for k,v in archived['latest_completed'].items() if k!='uuid'})))
                self.assertTrue(self.progress.blocked)
                self.assertTrue(self.progress.observe(archived))
                self.assertFalse(self.progress.blocked)
                self.assertEqual(self.progress.current['official_episode_index'],0)
                self.assertEqual(self.progress.receipt()['accepted_count'],0)
                self.assertEqual(self.progress.state['attempts'][-1]['operator_abort_reason'],reason)
                self.assertEqual(archived['latest_completed']['selection_status'],'technical_evidence_incomplete')

    def test_abort_writer_failure_stays_permanent_and_cannot_be_relabelled(self):
        identity=self.submit(operator_abort_reason='operator_discard')
        self.progress.fail('capture writer failure')
        self.assertFalse(self.progress.observe(self.partial_abort_review(identity)))
        self.assertTrue(self.progress.blocked)
        self.assertEqual(self.progress.state['pending']['uuid'],identity)
        self.assertEqual(self.progress.state['error'],'capture writer failure')
        self.assertEqual(self.progress.receipt()['accepted_count'],0)

    def test_abort_transient_review_retries_and_resume_preserve_intent(self):
        identity=self.submit(operator_abort_reason='operator_pause')
        self.progress.observe(self.result(identity,pending=True,status='error',error='temporary I/O'))
        self.progress.close();self.progress=CollectionProgress(self.rows,self.path)
        self.assertEqual(self.progress.state['pending']['operator_abort_reason'],'operator_pause')
        self.assertTrue(self.progress.observe(self.partial_abort_review(identity)))
        self.assertEqual(self.progress.receipt()['status'],'ready')
        self.assertEqual(self.progress.receipt()['accepted_count'],0)

    def test_abort_intent_never_counts_even_if_a_success_result_arrives(self):
        identity=self.submit(operator_abort_reason='operator_discard')
        self.assertFalse(self.progress.observe(self.result(identity)))
        self.assertTrue(self.progress.blocked)
        self.assertEqual(self.progress.receipt()['accepted_count'],0)

    def test_invalid_abort_cannot_disguise_nonoperator_failure_or_tamper_recovery(self):
        with self.assertRaises(ValueError):self.submit(operator_abort_reason='capture_failed')
        identity=self.submit(operator_abort_reason='operator_reset')
        self.progress.close();original=json.loads(self.path.read_text())
        for change in ('unknown_reason','accepted_abort'):
            value=json.loads(json.dumps(original))
            if change=='unknown_reason':
                value['attempts'][0]['operator_abort_reason']='capture_failed'
                value['pending']['operator_abort_reason']='capture_failed'
            else:
                value.update(accepted={'0':identity},pending=None)
            self.path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):CollectionProgress(self.rows,self.path)


if __name__ == '__main__':
    unittest.main()
