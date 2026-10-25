"""CPU library policy tests; no SDK, CUDA context, model, or simulator startup."""
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pico_cpu_runtime as runtime


def fake_libraries(*, interop=8, fail=None, ignore_native=False):
    state={"intra":4,"inter":interop,"cv":7,"blas":8,"openmp":4,"numexpr":6}
    calls=[]
    def setter(name,value):
        calls.append((name,value))
        if fail==name:
            raise RuntimeError("cannot set "+name+" after work started")
        state[name]=value
    torch=SimpleNamespace(__version__="2.test",get_num_threads=lambda:state['intra'],
        get_num_interop_threads=lambda:state['inter'],
        set_num_threads=lambda value:setter('intra',value),
        set_num_interop_threads=lambda value:setter('inter',value))
    cv=SimpleNamespace(__version__="4.test",getNumThreads=lambda:state['cv'],
        setNumThreads=lambda value:setter('cv',value))
    def pool_info():
        return [{"user_api":"blas","internal_api":"openblas","num_threads":state['blas'],
                 "version":"fake","filepath":"/fake/blas.so"},
                {"user_api":"openmp","internal_api":"openmp","num_threads":state['openmp'],
                 "version":"fake","filepath":"/fake/omp.so"}]
    def limits(*,limits):
        calls.append(('native_limit',limits))
        previous=state['blas'],state['openmp']
        if not ignore_native:
            state['blas']=state['openmp']=limits
        def restore():
            calls.append(('restore',None))
            state['blas'],state['openmp']=previous
        return SimpleNamespace(restore_original_limits=restore)
    ctl=SimpleNamespace(__version__="3.test",threadpool_info=pool_info,threadpool_limits=limits)
    numexpr=SimpleNamespace(__version__="2.test",get_num_threads=lambda:state['numexpr'],
        set_num_threads=lambda value:setter('numexpr',value))
    return {'numpy':SimpleNamespace(__version__="2.test"),'torch':torch,'cv2':cv,'threadpoolctl':ctl},numexpr,state,calls


class RuntimePolicyTests(unittest.TestCase):
    def fake_context(self, libraries, *, numexpr=None):
        stack=ExitStack()
        stack.enter_context(patch.object(runtime,'_active_guard',None))
        stack.enter_context(patch.object(runtime,'_configured_once',False))
        stack.enter_context(patch.object(runtime.importlib,'import_module',side_effect=lambda name:libraries[name]))
        stack.enter_context(patch.dict(runtime.sys.modules,{'numexpr':numexpr}))
        stack.enter_context(patch.dict(os.environ,{'OMP_NUM_THREADS':'12','OPENBLAS_NUM_THREADS':'19'}))
        return stack

    def test_import_does_not_apply_thread_policy_or_import_native_libraries(self):
        environment=dict(os.environ)
        with patch.object(runtime.importlib,'import_module',side_effect=AssertionError('no native import')):
            importlib.reload(runtime)
        self.assertEqual(dict(os.environ),environment)
        self.assertIsNone(runtime._active_guard)
        self.assertFalse(runtime._configured_once)

    def test_runtime_limits_loaded_pools_and_reports_actual_before_after(self):
        libraries,numexpr,state,calls=fake_libraries()
        with self.fake_context(libraries,numexpr=numexpr):
            guard=runtime.configure_cpu_runtime()
            receipt=guard.receipt
            self.assertEqual(receipt['status'],'applied')
            self.assertTrue(receipt['verified_one_thread'])
            self.assertEqual(receipt['environment_before']['OMP_NUM_THREADS'],'12')
            self.assertTrue(all(value=='1' for value in receipt['environment_after'].values()))
            self.assertEqual(receipt['before']['torch_intraop_threads'],4)
            self.assertEqual(receipt['before']['native_pools'][0]['num_threads'],8)
            for name in ('torch_intraop_threads','torch_interop_threads','opencv_threads','numexpr_threads'):
                self.assertEqual(receipt['after'][name],1)
            self.assertTrue(all(pool['num_threads']==1 for pool in receipt['after']['native_pools']))
            json.dumps(receipt,allow_nan=False)
            receipt['after']['torch_intraop_threads']=999
            self.assertEqual(guard.receipt['after']['torch_intraop_threads'],1)
            guard.close()

    def test_active_policy_is_idempotent_and_close_does_not_claim_global_rollback(self):
        libraries,_,state,calls=fake_libraries()
        with self.fake_context(libraries):
            guard=runtime.configure_cpu_runtime()
            self.assertIs(runtime.configure_cpu_runtime(),guard)
            self.assertEqual(calls.count(('inter',1)),1)
            self.assertEqual(calls.count(('intra',1)),1)
            closed=guard.close()
            self.assertEqual(closed['status'],'closed')
            self.assertTrue(closed['native_limit_scope_released'])
            self.assertFalse(closed['global_process_policy_restored'])
            self.assertEqual(state['blas'],8)
            self.assertEqual(state['intra'],1)
            self.assertEqual(state['inter'],1)
            guard.close()
            self.assertEqual(calls.count(('restore',None)),1)
            with self.assertRaisesRegex(RuntimeError,'already configured and closed'):
                runtime.configure_cpu_runtime()

    def test_existing_interop_one_is_not_reset_after_parallel_initialization(self):
        libraries,_,_,calls=fake_libraries(interop=1,fail='inter')
        with self.fake_context(libraries):
            guard=runtime.configure_cpu_runtime()
            self.assertNotIn(('inter',1),calls)
            guard.close()

    def test_setter_failure_is_explicit_and_contains_partial_state_not_success(self):
        for failure in ('inter','cv'):
            with self.subTest(failure=failure):
                libraries,_,_,calls=fake_libraries(fail=failure)
                with self.fake_context(libraries):
                    with self.assertRaises(runtime.CpuRuntimeConfigurationError) as raised:
                        runtime.configure_cpu_runtime()
                    receipt=raised.exception.receipt
                    self.assertEqual(receipt['status'],'failed')
                    self.assertFalse(receipt['verified_one_thread'])
                    self.assertIsNone(runtime._active_guard)
                    self.assertIn('partially applied',str(raised.exception))
                    self.assertIn('observed_after_failure',receipt)
                    json.dumps(receipt,allow_nan=False)

    def test_native_pool_ignoring_limit_and_no_detected_blas_are_not_success(self):
        for no_blas in (False,True):
            libraries,_,_,calls=fake_libraries(ignore_native=not no_blas)
            if no_blas:
                libraries['threadpoolctl'].threadpool_info=lambda:[]
            with self.fake_context(libraries):
                with self.assertRaises(runtime.CpuRuntimeConfigurationError) as raised:
                    runtime.configure_cpu_runtime()
                self.assertFalse(raised.exception.receipt['verified_one_thread'])
                self.assertEqual(raised.exception.receipt['failed_action'],'measure_and_verify_after')
                self.assertEqual(calls.count(('restore',None)),1)

    def test_failed_guard_close_is_reported_and_not_reused_as_verified_active(self):
        libraries,_,_,_=fake_libraries()
        with self.fake_context(libraries):
            guard=runtime.configure_cpu_runtime()
            def failed_restore():
                raise RuntimeError('native release failed')
            guard._limiter.restore_original_limits=failed_restore
            with self.assertRaises(runtime.CpuRuntimeConfigurationError) as raised:
                guard.close()
            self.assertEqual(raised.exception.receipt['status'],'close_failed')
            with self.assertRaisesRegex(runtime.CpuRuntimeConfigurationError,'not in a verified active state'):
                runtime.configure_cpu_runtime()

    def test_launcher_pins_environment_before_both_python_calls(self):
        launcher=Path(runtime.__file__).with_name('start_pico_teleop.sh')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'local.env.sh').write_text('export OPENBLAS_NUM_THREADS=19\n')
            log=root/'environment.log'
            fake=root/'fake-python'
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" '
                '"$OPENBLAS_NUM_THREADS" "$NUMEXPR_NUM_THREADS" "$BLIS_NUM_THREADS" '
                '"$VECLIB_MAXIMUM_THREADS" >> "$PICO_TEST_ENV_LOG"\n')
            fake.chmod(0o700)
            environment={**os.environ,'CVPR_SIMPLE_RUNTIME':str(root),'PICO_SIM_PYTHON':str(fake),
                'PICO_TEST_ENV_LOG':str(log),'OPENBLAS_NUM_THREADS':'30'}
            result=subprocess.run(['bash',str(launcher),'--output',str(root/'unused')],
                env=environment,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(log.read_text().splitlines(),['1']*12)

    def test_actual_cpu_subprocess_reports_one_for_torch_numpy_opencv_and_loaded_pools(self):
        directory=Path(runtime.__file__).parent
        program='''
import json,sys
sys.path.insert(0,sys.argv[1])
import numpy as np
import torch
import cv2
from threadpoolctl import threadpool_info
from pico_cpu_runtime import configure_cpu_runtime
guard=configure_cpu_runtime()
np.ones((16,16)) @ np.ones((16,16))
torch.ones((8,8),device='cpu') @ torch.ones((8,8),device='cpu')
cv2.resize(np.zeros((8,8,3),dtype=np.uint8),(4,4))
receipt=guard.receipt
receipt['post_cpu_work']={'torch_intraop':torch.get_num_threads(),
 'torch_interop':torch.get_num_interop_threads(),'opencv':cv2.getNumThreads(),
 'native_pools':threadpool_info(),'cuda_initialized':torch.cuda.is_initialized()}
receipt['close_status']=guard.close()['status']
print(json.dumps(receipt,allow_nan=False))
'''
        environment={**os.environ,**{name:'1' for name in runtime.THREAD_ENVIRONMENT},
            'CUDA_VISIBLE_DEVICES':'','PYTHONNOUSERSITE':'1'}
        result=subprocess.run([sys.executable,'-c',program,str(directory)],env=environment,
            capture_output=True,text=True,timeout=45)
        self.assertEqual(result.returncode,0,result.stderr)
        receipt=json.loads(result.stdout)
        self.assertTrue(receipt['verified_one_thread'])
        self.assertTrue(all(receipt['versions'][name] for name in ('numpy','torch','opencv','threadpoolctl')))
        for name in ('torch_intraop','torch_interop','opencv'):
            self.assertEqual(receipt['post_cpu_work'][name],1)
        self.assertTrue(receipt['post_cpu_work']['native_pools'])
        self.assertTrue(all(pool['num_threads']==1 for pool in receipt['post_cpu_work']['native_pools']))
        self.assertFalse(receipt['post_cpu_work']['cuda_initialized'])
        self.assertEqual(receipt['close_status'],'closed')
        evidence_path=os.environ.get('PICO_CPU_TEST_RECEIPT')
        if evidence_path:
            Path(evidence_path).write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    unittest.main()
