"""Queue boundaries: existing work waits, actual errors stay errors, no retries."""
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import run_pick_eval_queue as queue
import fetch_pick_checkpoint as fetch


class QueueTests(unittest.TestCase):
    def final(self, successes=0):
        rows = [dict(level=l,case_index=i,status='complete',native_success=(l*10+i<successes),
            initialization_profile=queue.INITIALIZATION_PROFILE,rtc_execution={'status':'passed'})
            for l in range(3)for i in range(10)]
        return dict(contract='cvpr-pick-fixed-official30-sonic-rtc-diagnostic-v1',status='complete',
            profile='xmove_pick',task_id=queue.PROFILES['xmove_pick']['task_id'],checkpoint_step=5000,
            model_sha256='1'*64,planned_cases=30,technically_valid_cases=30,technical_errors=0,
            initialization_profile=queue.INITIALIZATION_PROFILE,original_official_initialization=True,
            cases=rows,native_successes=successes,evaluation_policy=dict(rtc_mode='test_time',rtc_enabled=True,
                execute_steps=15,control_ticks_per_window=25))

    def test_zero_success_is_complete_and_cross_task_step_or_incomplete_is_rejected(self):
        queue.validate_final(self.final(0),'xmove_pick','1'*64)
        queue.validate_final(self.final(30),'xmove_pick','1'*64)
        for key,value in [('checkpoint_step',5000.),('task_id','other'),('technically_valid_cases',29),
                ('model_sha256','2'*64),('native_successes',1),('initialization_profile','stable_support_v1')]:
            broken=deepcopy(self.final());broken[key]=value
            with self.assertRaises(ValueError,msg=key):queue.validate_final(broken,'xmove_pick','1'*64)
        broken=self.final();broken['cases'][-1]=deepcopy(broken['cases'][0])
        with self.assertRaises(ValueError):queue.validate_final(broken,'xmove_pick','1'*64)

    def test_waits_for_existing_training_without_launching_or_restarting_it(self):
        pending=dict(state=None,queue={'active_profile':'xmove_pick'},exit=None)
        complete=dict(state={'state':'complete','returncode':0,'task_id':queue.PROFILES['xmove_bend_pick']['task_id']},queue={},exit=None)
        update=Mock()
        with patch.object(queue,'training_state',side_effect=[pending,complete]),patch.object(queue.time,'sleep') as sleep:
            self.assertEqual(queue.wait_training('xmove_bend_pick',update),complete)
        self.assertEqual(sleep.call_args.args,(30,))
        update.assert_called_once_with(stage='waiting_existing_training',server_active_profile='xmove_pick',training_state='not_started')
        with patch.object(queue,'training_state',return_value=dict(pending,exit={'exit_code':1})):
            with self.assertRaises(ValueError):queue.wait_training('xmove_bend_pick',update)

    def test_foreign_gpu_is_waited_for_and_unresolved_previous_cleanup_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            previous=Path(temp)/'exit.json';previous.write_text(json.dumps({'cleanup':{'status':'verified_stopped'}}))
            with patch.object(queue,'PRIOR_EXIT',previous),patch.object(queue.subprocess,'check_output',side_effect=['999\n','']),patch.object(queue.time,'sleep') as sleep:
                queue.wait_gpu(Mock())
                sleep.assert_called_once_with(15)
            previous.write_text(json.dumps({'cleanup':{'status':'failed'}}))
            with patch.object(queue,'PRIOR_EXIT',previous):
                with self.assertRaises(ValueError):queue.wait_gpu(Mock())

    def test_signal_during_child_launch_defers_until_owned_cleanup(self):
        process=SimpleNamespace(pid=123456,returncode=None)
        identity=dict(pid=process.pid,parent=os.getpid(),uid=os.getuid(),group=process.pid,session=process.pid)
        cleanup=Mock(return_value={'status':'verified_stopped'})
        helpers=SimpleNamespace(process_identity=lambda pid:identity,collect_owned=Mock(),cleanup_campaign=cleanup)
        def popen(*args,**kwargs):
            queue.interrupted(signal.SIGTERM,None)
            return process
        with tempfile.TemporaryDirectory() as temp,patch.object(queue.subprocess,'Popen',side_effect=popen):
            with self.assertRaises(KeyboardInterrupt):queue.run_owned(['fixture'],Path(temp)/'log',True,helpers,Mock())
        self.assertEqual(cleanup.call_count,1)
        self.assertIs(cleanup.call_args.args[0],process)
        self.assertIn(process.pid,cleanup.call_args.args[1])
        self.assertFalse(queue.LAUNCHING)
        self.assertIsNone(queue.PENDING_SIGNAL)

    def test_nonzero_real_stage_exit_is_not_hidden(self):
        class FakeProcess:
            pid=654321
            returncode=7
            def poll(self):return self.returncode
            def wait(self):return self.returncode
        process=FakeProcess()
        identity=dict(pid=process.pid,parent=os.getpid(),uid=os.getuid(),group=process.pid,session=process.pid)
        cleanup=Mock(return_value={'status':'verified_stopped'})
        helpers=SimpleNamespace(process_identity=lambda pid:identity,collect_owned=Mock(),cleanup_campaign=cleanup)
        with tempfile.TemporaryDirectory() as temp,patch.object(queue.subprocess,'Popen',return_value=process):
            with self.assertRaisesRegex(ValueError,'actual exit code 7'):
                queue.run_owned(['fixture'],Path(temp)/'log',True,helpers,Mock())
        cleanup.assert_called_once()

    def test_fetch_interrupt_reaps_only_current_ssh_before_returning(self):
        class FakeSSH:
            def __init__(self):self.stopped=False;self.waits=0
            def wait(self,timeout=None):
                self.waits+=1
                if self.waits==1:raise fetch.StopRequested('fixture')
                self.stopped=True;return -15
            def poll(self):return -15 if self.stopped else None
            def terminate(self):self.stopped=True
        ssh=FakeSSH()
        with tempfile.TemporaryDirectory() as temp,patch.object(fetch.subprocess,'Popen',return_value=ssh):
            with self.assertRaises(fetch.StopRequested):fetch.stream_transfer(['fixture'],Path(temp)/'partial',Path(temp)/'log')
        self.assertTrue(ssh.stopped)
        self.assertEqual(ssh.waits,2)


if __name__ == '__main__':
    unittest.main()
