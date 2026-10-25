"""Temporary fake-backend tests. No real model, MuJoCo, EGL, SDK or physics."""
import ast
import hashlib
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pico_async_preview as preview


class FakeBackend:
    def __init__(self,configuration):
        self.configuration=configuration
        self.root=Path(configuration['model_path']).parent
        if Path(configuration['model_path']).name=='slow.mjb':
            time.sleep(.15)
        self.closed=False
        self.render_count=0

    def render(self,state,*,sim_time_s):
        self.render_count+=1
        marker=self.root/f'entered_{self.configuration["epoch"]}_{int(state[1])}'
        marker.write_text('fake render started')
        if state[1]==101:
            time.sleep(.08)
        elif state[1]==102:
            time.sleep(.16)
        elif state[1]==103:
            time.sleep(3.)
        elif state[1]==104:
            raise RuntimeError('synthetic render failure')
        elif state[1]==105:
            state[0]+=1  # Deliberately invalid backend for the mutation guard.
        elif state[1]==106 and self.render_count==1:
            time.sleep(.16)  # One-time shader-like warmup, without physics.
        images={name:np.full((height,width,3),int(state[1])%256,np.uint8)
                for name,(width,height) in self.configuration['cameras'].items()}
        return images,{'synthetic_backend':True,'full_integration_state_unchanged':True,
                      'physics_steps':0,'forward_dynamics_calls':0,'SDK_started':False}

    def close(self):
        self.closed=True
        (self.root/f'closed_epoch_{self.configuration["epoch"]}').write_text('closed fake backend')


def wait(predicate,timeout=5.):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        result=predicate()
        if result:
            return result
        time.sleep(.003)
    raise AssertionError('Fake preview worker timed out')


class AsyncPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.model=self.root/'fake.mjb'
        self.model.write_bytes(b'explicit fake MJB for CPU IPC tests only')
        self.sha=hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.cameras={'head_stereo_left':(8,6)}

    def worker(self,**kwargs):
        worker=preview.AsyncMujocoPreview(backend_factory=FakeBackend,
            max_width=16,max_height=12,max_state_width=32,**kwargs).start()
        self.addCleanup(worker.close)
        return worker

    def configure(self,worker,*,epoch=0,path=None,**kwargs):
        return worker.configure(path or self.model,self.sha,state_spec=123,state_width=8,
            cameras=self.cameras,epoch=epoch,**kwargs)

    def submit(self,worker,value=5,*,stamp=1.,tick=0,epoch=0):
        state=np.array([stamp,value,0,0,0,0,0,0],dtype=np.float64)
        return state,worker.submit(state,stamp,epoch,tick)

    def test_real_spawn_fake_frame_is_copied_bound_and_consumed_once(self):
        worker=self.worker()
        status=self.configure(worker)
        self.assertTrue(worker.is_ready)
        self.assertEqual(status['model_sha256'],self.sha)
        state,accepted=self.submit(worker,7)
        self.assertTrue(accepted)
        original=state.copy()
        state[:]=999
        frame=wait(worker.poll)
        self.assertEqual(frame.source_tick50,0)
        self.assertEqual(frame.sim_time_s,1.)
        self.assertEqual(frame.state_sha256,hashlib.sha256(original.tobytes()).hexdigest())
        self.assertEqual(frame.images['head_stereo_left'].shape,(6,8,3))
        np.testing.assert_array_equal(frame.images['head_stereo_left'],7)
        self.assertFalse(frame.images['head_stereo_left'].flags.writeable)
        self.assertLess(frame.receive_age_s,.1)
        self.assertGreaterEqual(frame.received_monotonic_s,frame.render_completed_monotonic_s)
        self.assertFalse(frame.evidence['training_observation'])
        self.assertTrue(frame.evidence['preview_only'])
        self.assertTrue(frame.evidence['synthetic_backend'])
        self.assertIsNone(worker.poll())

    def test_stereo_and_explicit_camera_dimensions(self):
        worker=self.worker()
        worker.configure(self.model,self.sha,123,8,
            {'head_stereo_left':{'width':8,'height':6},'head_stereo_right':(8,6)},epoch=0)
        self.submit(worker,8)
        frame=wait(worker.poll)
        self.assertEqual(set(frame.images),{'head_stereo_left','head_stereo_right'})
        self.assertEqual(frame.evidence['model_sha256'],self.sha)

    def test_busy_mailboxes_return_immediately_without_claiming_new_frame(self):
        worker=self.worker()
        self.configure(worker)
        with worker._requests.lock:
            started=time.monotonic()
            _,accepted=self.submit(worker)
            self.assertFalse(accepted)
            self.assertLess(time.monotonic()-started,.05)
        self.assertEqual(worker.status()['dropped_submits'],1)
        self.assertEqual(worker._sequence,0)
        with worker._results.lock:
            started=time.monotonic()
            self.assertIsNone(worker.poll())
            self.assertLess(time.monotonic()-started,.05)
        self.assertTrue(self.submit(worker)[1])
        self.assertIsNotNone(wait(worker.poll))

    def test_latest_only_pending_state_can_drop_without_dropping_physical_source(self):
        worker=self.worker(max_frame_age_s=.5)
        self.configure(worker)
        self.submit(worker,101)
        wait(lambda:(self.root/'entered_0_101').exists())
        first=np.array([1.02,6,0,0,0,0,0,0],float)
        second=np.array([1.04,9,0,0,0,0,0,0],float)
        self.assertTrue(worker.submit(first,1.02,0,1))
        self.assertTrue(worker.submit(second,1.04,0,2))
        original=second.copy()
        frame=wait(lambda:next((frame for frame in [worker.poll()] if frame is not None and frame.source_tick50==2),None))
        np.testing.assert_array_equal(frame.images['head_stereo_left'],9)
        np.testing.assert_array_equal(second,original)
        self.assertGreaterEqual(worker.status()['dropped_pending_requests'],1)
        self.assertFalse((self.root/'entered_0_6').exists())

    def test_duplicate_or_regressing_source_never_becomes_fake_new_preview(self):
        worker=self.worker()
        self.configure(worker)
        self.submit(worker,5,stamp=1.,tick=3)
        frame=wait(worker.poll)
        self.assertEqual(frame.source_tick50,3)
        for stamp,tick in ((1.,3),(1.02,2),(.99,4)):
            self.assertFalse(self.submit(worker,6,stamp=stamp,tick=tick)[1])
        self.assertIsNone(worker.poll())
        self.assertEqual(worker.status()['rendered_frames'],1)

    def test_paused_identical_redraw_recovers_slow_first_frame_without_relaxing_age(self):
        worker=self.worker()
        self.configure(worker)
        state=np.array([1.,106,0,0,0,0,0,0],float)
        self.assertTrue(worker.submit_paused(state,1.,0,0))
        wait(lambda:worker.status().get('rendered_frames')==1)
        self.assertIsNone(worker.poll())
        self.assertEqual(worker.status()['expired_results'],1)
        self.assertFalse(worker.submit(state,1.,0,0))
        changed=state.copy();changed[-1]=1
        self.assertFalse(worker.submit_paused(changed,1.,0,0))
        self.assertFalse(worker.submit_paused(state,.99,0,0))
        self.assertTrue(worker.submit_paused(state,1.,0,0))
        frame=wait(worker.poll)
        self.assertEqual(frame.source_tick50,0)
        self.assertEqual(frame.sim_time_s,1.)
        self.assertLess(frame.receive_age_s,.1)
        self.assertEqual(frame.state_sha256,hashlib.sha256(state.tobytes()).hexdigest())
        self.assertTrue(frame.evidence['paused_render_request'])
        self.assertTrue(frame.evidence['repeated_physical_source'])
        self.assertFalse(frame.evidence['training_observation'])
        self.assertIsNone(worker.poll())
        self.configure(worker,epoch=1)
        changed[1]=7
        self.assertTrue(worker.submit_paused(changed,1.,1,0))
        frame=wait(worker.poll)
        self.assertFalse(frame.evidence['repeated_physical_source'])
        # A new epoch has its own source identity; active duplicates remain invalid.
        self.assertFalse(worker.submit(changed,1.,1,0))

    def test_reset_epoch_never_returns_old_images_even_when_old_render_finishes_late(self):
        worker=self.worker(max_frame_age_s=.5)
        self.configure(worker)
        self.submit(worker,101)
        wait(lambda:(self.root/'entered_0_101').exists())
        ready=self.configure(worker,epoch=1)
        self.assertIsNone(ready['last_frame_id'])
        self.assertEqual(ready['rendered_frames'],0)
        self.assertTrue((self.root/'closed_epoch_0').exists())
        self.assertIsNone(worker.poll())
        self.submit(worker,17,epoch=1)
        frame=wait(worker.poll)
        self.assertEqual(frame.epoch,1)
        np.testing.assert_array_equal(frame.images['head_stereo_left'],17)
        with self.assertRaises(ValueError):
            self.submit(worker,8,stamp=1.02,tick=1,epoch=0)
        with self.assertRaises(ValueError):
            self.configure(worker,epoch=1)

    def test_expired_frame_is_dropped_and_failure_is_visible_without_black_fallback(self):
        worker=self.worker()
        self.configure(worker)
        self.submit(worker,102)
        wait(lambda:worker.status().get('rendered_frames')==1)
        self.assertIsNone(worker.poll())
        self.assertEqual(worker.status()['expired_results'],1)
        self.submit(worker,104,stamp=1.02,tick=1)
        state=wait(lambda:worker.status() if worker.status().get('error') else None)
        self.assertIn('synthetic render failure',state['error'])
        self.assertFalse(worker.is_ready)
        self.assertIsNone(worker.poll())
        self.assertFalse(self.submit(worker,8,stamp=1.04,tick=2)[1])
        self.configure(worker,epoch=1)
        self.assertTrue(self.submit(worker,8,epoch=1)[1])
        self.assertIsNotNone(wait(worker.poll))

    def test_mutating_backend_is_rejected_before_image_publication(self):
        worker=self.worker()
        self.configure(worker)
        self.submit(worker,105)
        state=wait(lambda:worker.status() if worker.status().get('error') else None)
        self.assertIn('mutated submitted state',state['error'])
        self.assertIsNone(worker.poll())
        self.assertFalse(worker.is_ready)

    def test_mjb_sha_and_late_configuration_acknowledgement_fail_closed(self):
        worker=self.worker()
        with self.assertRaisesRegex(RuntimeError,'SHA mismatch'):
            worker.configure(self.model,'0'*64,123,8,self.cameras,epoch=0)
        self.assertFalse(worker.is_ready)
        slow=self.root/'slow.mjb';slow.write_bytes(self.model.read_bytes())
        with self.assertRaises(TimeoutError):
            self.configure(worker,epoch=1,path=slow,timeout_s=.01)
        time.sleep(.2)
        self.assertFalse(worker.is_ready)
        self.assertFalse(self.submit(worker,1,epoch=1)[1])
        self.configure(worker,epoch=2)
        self.assertTrue(worker.is_ready)

    def test_graceful_and_hung_close_leave_no_fake_worker_alive(self):
        worker=self.worker()
        self.configure(worker)
        pid=worker._process.pid
        worker.close()
        self.assertTrue((self.root/'closed_epoch_0').exists())
        with self.assertRaises(ProcessLookupError):os.kill(pid,0)
        self.assertEqual(worker.status()['status'],'closed')
        self.assertIsNone(worker.poll())
        worker.close()
        with self.assertRaises(ValueError):worker.start()
        hung=self.worker()
        self.configure(hung)
        self.submit(hung,103)
        wait(lambda:(self.root/'entered_0_103').exists())
        pid=hung._process.pid
        started=time.monotonic()
        hung.close(timeout_s=.01)
        self.assertLess(time.monotonic()-started,1.2)
        with self.assertRaises(ProcessLookupError):os.kill(pid,0)

    def test_cpu_policy_and_invalid_input_capacity(self):
        worker=self.worker()
        status=self.configure(worker)
        self.assertEqual(status['worker']['nice'],19)
        self.assertEqual(status['worker']['thread_limit'],1)
        self.assertEqual(len(status['worker']['cpus']),1)
        self.assertFalse(status['worker']['SDK_started'])
        for value in (np.zeros(7),np.full(8,np.nan)):
            with self.assertRaises(ValueError):worker.submit(value,1.,0,0)
        with self.assertRaises(ValueError):worker.submit(np.zeros(8),1.,False,0)
        with self.assertRaises(ValueError):
            worker.configure(self.model,self.sha,123,8,{'head_stereo_left':(100,100)},epoch=1)
        with self.assertRaises(ValueError):preview.AsyncMujocoPreview(max_frame_age_s=0)

    def test_actual_backend_has_no_sdk_inference_or_physics_calls(self):
        tree=ast.parse(Path(preview.__file__).read_text())
        calls=[node.func for node in ast.walk(tree) if isinstance(node,ast.Call)]
        attributes={node.attr for node in calls if isinstance(node,ast.Attribute)}
        for forbidden in ('mj_step','mj_forward','InferenceSession','CDLL'):
            self.assertNotIn(forbidden,attributes)
        self.assertIn('mj_kinematics',attributes)
        self.assertIn('mj_camlight',attributes)
        self.assertNotIn('Event',attributes)  # Control cannot wait on a worker event lock.


if __name__=='__main__':
    unittest.main()
