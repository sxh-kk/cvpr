"""Finalization isolates episode failures; all heavy work is replaced by fakes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from finish_pico_session import finish


class FinishPicoTests(unittest.TestCase):
    def episode(self, session, name, outcome):
        path=Path(session)/'episodes'/name
        path.mkdir(parents=True)
        (path/'episode.json').write_text(json.dumps(outcome))
        return path

    def test_corrupt_episode_does_not_hide_later_audit_and_task_failure_is_not_technical(self):
        with tempfile.TemporaryDirectory() as directory:
            bad=self.episode(directory,'01_corrupt',{})
            (bad/'episode.json').write_text('{broken json')
            good=self.episode(directory,'02_saved',{'state':{'status':'pending_audit'}})
            audit=mock.Mock(return_value={'status':'task_failed','eligible_for_packing':False,'errors':['no placement']})
            with mock.patch.dict(sys.modules,{'audit_pico_teleop_episode':types.SimpleNamespace(audit_episode=audit)}):
                report=finish(directory)
            audit.assert_called_once_with(good)
            self.assertEqual(report['technical_errors'],1)
            self.assertEqual(report['eligible_for_packing'],0)
            self.assertEqual(report['episodes'][0]['status'],'technical_error')
            self.assertEqual(report['episodes'][1]['status'],'task_failed')
            self.assertEqual(json.loads((Path(directory)/'session_finalization.json').read_text()),report)

    def test_renderer_failure_cannot_be_promoted_by_a_stale_success_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.episode(directory,'saved',{'status':'pending_rtx','state':{'status':'pending_audit'}})
            def failed_renderer(command,**kwargs):
                (path/'rtx_completion.json').write_text(json.dumps({'status':'deferred_rtx_completed_pending_independent_audit'}))
                return subprocess.CompletedProcess(command,1)
            audit=mock.Mock()
            with mock.patch('finish_pico_session.subprocess.run',side_effect=failed_renderer), \
                    mock.patch.dict(sys.modules,{'audit_pico_teleop_episode':types.SimpleNamespace(audit_episode=audit)}):
                report=finish(directory)
            audit.assert_not_called()
            self.assertEqual(report['technical_errors'],1)
            self.assertEqual(report['episodes'][0]['renderer_returncode'],1)

    def test_existing_complete_rtx_is_reaudited_without_rerender(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.episode(directory,'saved',{'status':'pending_rtx','state':{'status':'pending_audit'}})
            (path/'rtx_completion.json').write_text(json.dumps({'status':'deferred_rtx_completed_pending_independent_audit'}))
            audit=mock.Mock(return_value={'status':'eligible','eligible_for_packing':True})
            with mock.patch('finish_pico_session.subprocess.run') as renderer, \
                    mock.patch.dict(sys.modules,{'audit_pico_teleop_episode':types.SimpleNamespace(audit_episode=audit)}):
                report=finish(directory)
            renderer.assert_not_called()
            audit.assert_called_once_with(path)
            self.assertEqual(report['technical_errors'],0)
            self.assertEqual(report['eligible_for_packing'],1)
            self.assertFalse(report['accepted_new_dataset'])

    def test_operator_and_automatic_discards_never_render_or_audit(self):
        for marker,value in [('operator_discard.json',{'reason':'operator requested'}),
                             ('collection_record.json',{'disposition':'discarded','reasons':['no lifted handover']})]:
            with self.subTest(marker=marker),tempfile.TemporaryDirectory() as directory:
                path=self.episode(directory,'saved',{'status':'pending_rtx','state':{'status':'pending_audit'}})
                (path/marker).write_text(json.dumps(value))
                audit=mock.Mock()
                with mock.patch('finish_pico_session.subprocess.run') as renderer, \
                     mock.patch.dict(sys.modules,{'audit_pico_teleop_episode':types.SimpleNamespace(audit_episode=audit)}):
                    report=finish(directory)
                renderer.assert_not_called();audit.assert_not_called()
                self.assertEqual(report['episodes'][0]['status'],'skipped_discarded')
                self.assertEqual(report['eligible_for_packing'],0)

    def test_original_episode_alias_to_numbered_kept_directory_still_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);actual=root/'kept'/'001';actual.mkdir(parents=True)
            import hashlib
            identity='f35e2c4b-d407-4422-a67d-533fd740ff2b'
            (actual/'episode.json').write_text(json.dumps({'state':{'status':'pending_audit','human_demo_uuid':identity}}))
            (actual/'collection_record.json').write_text(json.dumps({'disposition':'kept','status':'committed',
                'human_demo_uuid':identity,'source_episode_directory':str(root/'episodes'/'original_uuid'),
                'episode_json_sha256':hashlib.sha256((actual/'episode.json').read_bytes()).hexdigest()}))
            (root/'episodes').mkdir();alias=root/'episodes'/'original_uuid';alias.symlink_to(actual,target_is_directory=True)
            audit=mock.Mock(return_value={'status':'eligible','eligible_for_packing':True})
            with mock.patch.dict(sys.modules,{'audit_pico_teleop_episode':types.SimpleNamespace(audit_episode=audit)}):
                report=finish(root)
            audit.assert_called_once_with(alias)
            self.assertEqual(report['eligible_for_packing'],1)
            # A legacy record has no submission marker. Its durable queue
            # intent must still block admission before the worker moves it.
            requests=root/'.pico_episode_review'/'discard_requests'
            requests.mkdir(parents=True)
            (requests/(identity+'.json')).write_text(json.dumps({
                'uuid':identity,'operator_discard_requested':True}))
            audit.reset_mock()
            with mock.patch.dict(sys.modules,{'audit_pico_teleop_episode':types.SimpleNamespace(audit_episode=audit)}):
                report=finish(root)
            audit.assert_not_called()
            self.assertEqual(report['episodes'][0]['status'],'skipped_discarded')

    def test_pending_background_review_never_starts_rtx_or_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            self.episode(directory,'pending',{'status':'pending_rtx','geometry_pending':True,
                'state':{'status':'pending_audit'}})
            with mock.patch('finish_pico_session.subprocess.run') as renderer:
                report=finish(directory)
            renderer.assert_not_called()
            self.assertEqual(report['episodes'][0]['status'],'technical_error')
            self.assertEqual(report['eligible_for_packing'],0)


if __name__=='__main__':
    unittest.main()
