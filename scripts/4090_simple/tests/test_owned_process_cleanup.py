"""Owned process cleanup tests; only small disposable CPU subprocesses are used."""
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCRIPTS))
import owned_process_cleanup as c


def stat_text(pid,group,*,state='S',starttime=100):
    fields=[state,'0',str(group)]+['0']*16+[str(starttime)]+['0']*8
    return f'{pid} (process with ) in name) '+ ' '.join(fields)+'\n'


class MockProcTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.group=987654;self.pid=987655;self.token='/attempt/exact-002'
        self.entry=self.add_member(self.pid)
        self.proc_patch=patch.object(c,'PROC_ROOT',self.root);self.proc_patch.start();self.addCleanup(self.proc_patch.stop)
        self.group_patch=patch.object(c.os,'getpgid',side_effect=lambda pid:self.groups[pid])
        self.group_patch.start();self.addCleanup(self.group_patch.stop)
        self.kill_patch=patch.object(c.os,'killpg');self.kill=self.kill_patch.start();self.addCleanup(self.kill_patch.stop)

    def add_member(self,pid,*,token=None,group=None):
        if not hasattr(self,'groups'):self.groups={}
        group=self.group if group is None else group
        entry=self.root/str(pid);entry.mkdir()
        (entry/'stat').write_text(stat_text(pid,group))
        (entry/'environ').write_bytes(b'A=1\0'+(c.TOKEN_KEY+(self.token if token is None else token)).encode()+b'\0')
        self.groups[pid]=group
        return entry

    def run_cleanup(self,timeout=0):
        return c.stop_owned_group(self.group,self.token,timeout)

    def test_zombie_or_dead_reread_after_environ_permission_error_is_safe(self):
        for state in ('Z','X'):
            with self.subTest(state=state),patch.object(c,'_read_stat',side_effect=[
                c.ProcessStat(self.pid,'S',self.group,100),c.ProcessStat(self.pid,state,self.group,100)]),\
                patch.object(c,'_read_environment',side_effect=PermissionError(13,'exit race')):
                self.assertEqual(self.run_cleanup()['live_members_before'],[])
        self.kill.assert_not_called()

    def test_disappearance_after_environ_permission_error_is_safe(self):
        with patch.object(c,'_read_stat',side_effect=[c.ProcessStat(self.pid,'S',self.group,100),FileNotFoundError()]),\
             patch.object(c,'_read_environment',side_effect=PermissionError(13,'exit race')):
            self.assertEqual(self.run_cleanup()['live_members_after'],[])
        self.kill.assert_not_called()

    def test_live_permission_error_remains_fatal_and_never_signals(self):
        with patch.object(c,'_read_environment',side_effect=PermissionError(13,'live process')):
            with self.assertRaises(PermissionError):self.run_cleanup()
        self.kill.assert_not_called()

    def test_unreadable_followup_stat_does_not_exempt_permission_failure(self):
        with patch.object(c,'_read_stat',side_effect=[c.ProcessStat(self.pid,'S',self.group,100),PermissionError(13,'stat')]),\
             patch.object(c,'_read_environment',side_effect=PermissionError(13,'environ')):
            with self.assertRaises(PermissionError):self.run_cleanup()
        self.kill.assert_not_called()

    def test_permission_error_cannot_hide_pid_reuse_even_when_replacement_is_zombie(self):
        with patch.object(c,'_read_stat',side_effect=[
            c.ProcessStat(self.pid,'S',self.group,100),c.ProcessStat(self.pid,'Z',self.group,101)]),\
            patch.object(c,'_read_environment',side_effect=PermissionError(13,'environ')):
            with self.assertRaisesRegex(RuntimeError,'PID .* reused'):self.run_cleanup()
        self.kill.assert_not_called()

    def test_token_must_match_whole_environment_entry(self):
        for token in (self.token+'suffix','prefix'+self.token,'foreign'):
            (self.entry/'environ').write_bytes((c.TOKEN_KEY+token).encode()+b'\0')
            with self.subTest(token=token),self.assertRaisesRegex(RuntimeError,'unowned live member'):
                self.run_cleanup()
        self.kill.assert_not_called()

    def test_entire_group_must_be_owned_before_any_signal(self):
        self.add_member(self.pid+1,token='foreign')
        with self.assertRaisesRegex(RuntimeError,'unowned live member'):self.run_cleanup()
        self.kill.assert_not_called()

    def test_foreign_group_is_neither_read_for_token_nor_signaled(self):
        self.groups[self.pid]=self.group+10
        with patch.object(c,'_read_environment')as environment:
            self.assertEqual(self.run_cleanup()['live_members_before'],[])
        environment.assert_not_called();self.kill.assert_not_called()

    def test_new_foreign_member_before_term_revokes_signal_authority(self):
        with patch.object(c,'_read_environment',side_effect=[[(c.TOKEN_KEY+self.token).encode()],
            [(c.TOKEN_KEY+'foreign').encode()]]):
            with self.assertRaisesRegex(RuntimeError,'unowned live member'):self.run_cleanup()
        self.kill.assert_not_called()

    def test_pid_starttime_change_between_scans_revokes_signal_authority(self):
        old=c.ProcessStat(self.pid,'S',self.group,100);new=c.ProcessStat(self.pid,'S',self.group,101)
        with patch.object(c,'_read_stat',side_effect=[old,old,new]):
            with self.assertRaisesRegex(RuntimeError,'PID .* reused'):self.run_cleanup()
        self.kill.assert_not_called()

    def test_new_foreign_member_before_kill_blocks_escalation(self):
        own=[(c.TOKEN_KEY+self.token).encode()];foreign=[(c.TOKEN_KEY+'foreign').encode()]
        with patch.object(c,'_read_environment',side_effect=[own,own,own,foreign]):
            with self.assertRaisesRegex(RuntimeError,'unowned live member'):self.run_cleanup()
        self.assertEqual(self.kill.call_args_list,[unittest.mock.call(self.group,signal.SIGTERM)])

    def test_term_escalates_to_kill_and_requires_empty_terminal_state(self):
        def signal_group(group,signum):
            if signum == signal.SIGKILL:
                (self.entry/'stat').write_text(stat_text(self.pid,self.group,state='Z'))
        self.kill.side_effect=signal_group
        report=self.run_cleanup()
        self.assertEqual(report,{'process_group':self.group,'live_members_before':[self.pid],'live_members_after':[]})
        self.assertEqual(self.kill.call_args_list,[unittest.mock.call(self.group,signal.SIGTERM),
            unittest.mock.call(self.group,signal.SIGKILL)])

    def test_persistent_live_member_fails_final_state_acceptance(self):
        clock=SimpleNamespace(value=0.)
        def advance(delay):clock.value+=delay
        with patch.object(c.time,'monotonic',side_effect=lambda:clock.value),patch.object(c.time,'sleep',side_effect=advance):
            with self.assertRaisesRegex(RuntimeError,'did not stop'):self.run_cleanup()
        self.assertEqual(self.kill.call_count,2)

    def test_getpgid_permission_error_is_only_skipped_for_proven_foreign_group(self):
        self.group_patch.stop()
        with patch.object(c.os,'getpgid',side_effect=PermissionError(13,'namespace')):
            (self.entry/'stat').write_text(stat_text(self.pid,self.group+1))
            self.assertEqual(self.run_cleanup()['live_members_before'],[])
            (self.entry/'stat').write_text(stat_text(self.pid,self.group))
            with self.assertRaises(PermissionError):self.run_cleanup()
        self.kill.assert_not_called()

    def test_no_evaluator_group_or_invalid_token_can_be_signaled(self):
        for group in (0,1,True,os.getpgrp()):
            with self.subTest(group=group),self.assertRaises(ValueError):c.stop_owned_group(group,self.token)
        for token in ('',None,'bad\0token'):
            with self.subTest(token=token),self.assertRaises(ValueError):c.stop_owned_group(self.group,token)
        self.kill.assert_not_called()


class RealProcessTests(unittest.TestCase):
    def child(self,*,ignore_term=False,token='owned-cpu-test'):
        source=('import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);' if ignore_term
            else 'import time;')+"print('ready',flush=True);time.sleep(60)"
        process=subprocess.Popen([sys.executable,'-c',source],start_new_session=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','QUALITY100_ATTEMPT_TOKEN':token},
            stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.addCleanup(self.reap,process)
        self.assertEqual(process.stdout.readline().strip(),'ready')
        return process

    def reap(self,process):
        if process.poll() is None:process.kill()
        process.wait(timeout=3)
        process.stdout.close();process.stderr.close()

    def test_real_cpu_process_graceful_shutdown(self):
        child=self.child()
        report=c.stop_owned_group(child.pid,'owned-cpu-test',timeout=1)
        child.wait(timeout=3)
        self.assertIn(child.pid,report['live_members_before'])
        self.assertEqual(report['live_members_after'],[])

    def test_real_cpu_process_that_ignores_term_is_killed(self):
        child=self.child(ignore_term=True)
        report=c.stop_owned_group(child.pid,'owned-cpu-test',timeout=.02)
        child.wait(timeout=3)
        self.assertEqual(child.returncode,-signal.SIGKILL)
        self.assertEqual(report['live_members_after'],[])

    def test_actual_exit_to_zombie_during_environ_check_is_not_a_cleanup_failure(self):
        child=self.child()
        original=c._read_environment
        def exiting_environment(entry):
            if int(entry.name) != child.pid:return original(entry)
            child.terminate()
            deadline=time.monotonic()+3
            while c._read_stat(entry).state not in c.DEAD_STATES:
                if time.monotonic()>deadline:raise AssertionError('CPU child did not become zombie')
                time.sleep(.01)
            raise PermissionError(13,'simulated environ failure across actual zombie transition')
        with patch.object(c,'_read_environment',side_effect=exiting_environment),patch.object(c.os,'killpg')as signal_group:
            report=c.stop_owned_group(child.pid,'owned-cpu-test',timeout=1)
        signal_group.assert_not_called()
        self.assertEqual(report['live_members_after'],[])
        child.wait(timeout=3)

    def test_real_foreign_token_is_not_signaled(self):
        child=self.child(token='another-attempt')
        with patch.object(c.os,'killpg')as signal_group:
            with self.assertRaisesRegex(RuntimeError,'unowned live member'):
                c.stop_owned_group(child.pid,'owned-cpu-test',timeout=.02)
        signal_group.assert_not_called()
        self.assertIsNone(child.poll())


class InstallTests(unittest.TestCase):
    def test_helper_supports_runtime_style_file_import_without_sys_modules_registration(self):
        spec=importlib.util.spec_from_file_location('independent_cleanup_fixture',SCRIPTS/'owned_process_cleanup.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.install))

    def test_install_changes_only_cleanup_helper_and_existing_stop_owned_uses_it(self):
        runtime=ModuleType('cpu_runtime_fixture')
        runtime.model='unchanged';runtime.controller='unchanged';runtime.protocol='unchanged'
        exec('def stop_owned_group(group, token, timeout=8): return "old"\n'
            'def stop_owned(group, token): return stop_owned_group(group, token)\n',runtime.__dict__)
        prior={key:value for key,value in runtime.__dict__.items() if key != 'stop_owned_group'}
        record=c.install(runtime)
        self.assertFalse(record['already_installed'])
        self.assertEqual(prior,{key:value for key,value in runtime.__dict__.items() if key != 'stop_owned_group'})
        with patch.object(c,'_members',return_value=[]):
            self.assertEqual(runtime.stop_owned(987654,'exact')['live_members_after'],[])
        self.assertTrue(c.install(runtime)['already_installed'])

    def test_install_requires_existing_runtime_cleanup_contract(self):
        with self.assertRaises(ValueError):c.install(SimpleNamespace())


if __name__ == '__main__':
    unittest.main()
