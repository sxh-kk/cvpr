from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import sonic_locomotion_planner as adapter


def sequence(n=36):
    q=np.zeros((n,36));q[:,2]=adapter.DEFAULT_HEIGHT
    for i in range(n):
        q[i,0]=i/30;q[i,3:7]=[math.cos(i*.01/2),0,0,math.sin(i*.01/2)];q[i,7:]=i/30
    return q


class AdapterTests(unittest.TestCase):
    def test_bootstrap_is_not_claimed_as_measured_history_and_preserves_BODY_order(self):
        joints=np.arange(29)/100;context,evidence=adapter.bootstrap_context(joints)
        self.assertFalse(evidence['four_measured_history_frames_claimed'])
        np.testing.assert_array_equal(context[:,7:],np.tile(joints,(4,1)))
        np.testing.assert_array_equal(context[:,:7],np.tile([0,0,adapter.DEFAULT_HEIGHT,1,0,0,0],(4,1)))
        yaw=.7;align=adapter.WorldAlignment.from_actual([2,-1,.73,math.cos(yaw/2),0,0,math.sin(yaw/2)])
        world=align.to_world(context)
        np.testing.assert_allclose(world[:,:3],np.tile([2,-1,adapter.DEFAULT_HEIGHT],(4,1)),atol=1e-12)
        np.testing.assert_allclose(align.to_canonical(world),context,atol=1e-12)
        np.testing.assert_allclose(align.direction_to_canonical([math.cos(yaw),math.sin(yaw),0]),[1,0,0],atol=1e-12)
        self.assertEqual(align.evidence()['world_Z_translation_m'],0)

    def test_official_resampling_slerp_velocity_and_marked_endpoint_hold(self):
        forecast=adapter.resample_30_to_50(sequence())
        self.assertEqual(len(forecast.qpos50),60);self.assertEqual(forecast.valid_forecast_mask.sum(),59)
        np.testing.assert_allclose(forecast.qpos50[:59,0],np.arange(59)/50,atol=1e-12)
        np.testing.assert_allclose(forecast.dq29[:58],1.,atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(forecast.qpos50[:,3:7],axis=1),1.,atol=1e-12)
        np.testing.assert_allclose(forecast.qpos50[10,3:7],[math.cos(.06/2),0,0,math.sin(.06/2)],atol=1e-12)
        self.assertEqual(forecast.metadata['official_endpoint_hold_rows50'],[59])

    def test_complete46_window_rejects_endpoint_or_implicit_padding(self):
        forecast=adapter.resample_30_to_50(sequence())
        self.assertEqual(adapter.future_window(forecast,13)[0].shape,(46,36))
        for bad in (14,60,-1):
            with self.assertRaises(ValueError):adapter.future_window(forecast,bad)

    def test_context_is_reference_future_at_current_plus2_then30Hz(self):
        forecast=adapter.resample_30_to_50(sequence())
        context,evidence=adapter.context_from_forecast(forecast,3)
        np.testing.assert_allclose(context[:,0],(5+np.arange(4)*50/30)/50,atol=1e-12)
        self.assertEqual(evidence['generated_start_frame50'],5)
        self.assertIn('not actual robot history',evidence['context_source'])
        with self.assertRaises(ValueError):adapter.context_from_forecast(forecast,55)

    def test_crossfade_matches_eight_frame_timeline_and_never_mutates_inputs(self):
        a=sequence();b=sequence();a[:,7:]=1;b[:,7:]=3
        old=adapter.resample_30_to_50(a);new=adapter.resample_30_to_50(b)
        before=old.qpos50.copy();merged=adapter.crossfade8(old,new,10,12)
        self.assertEqual(len(merged.qpos50),62)
        np.testing.assert_array_equal(merged.qpos50[:3,7:],1.)
        np.testing.assert_array_equal(merged.qpos50[6,7:],2.)
        np.testing.assert_array_equal(merged.qpos50[10,7:],3.)
        np.testing.assert_array_equal(old.qpos50,before)
        self.assertEqual(adapter.future_window(merged,0)[0].shape,(46,36))

    def test_replan_is_not_forced_every10Hz_and_low_watermark_is_explicit(self):
        forecast=adapter.resample_30_to_50(sequence())
        command={'mode':1,'speed':.4,'height':-1.,'movement_direction':[0,1,0],'facing_direction':[1,0,0]}
        self.assertIsNone(adapter.replan_reason(command,deepcopy(command),.1,forecast,0))
        self.assertEqual(adapter.replan_reason(command,command,.2,forecast,10),'strict_future_low_watermark_explicit_adapter_extension')
        self.assertEqual(adapter.replan_reason(command,command,1.,forecast,0),'official_periodic_interval')
        change={**command,'mode':0}
        self.assertEqual(adapter.replan_reason(change,command,.1,forecast,0),'official_mode_changed')

    def test_limit_report_does_not_clip_or_hide_model_output(self):
        q=sequence();q[0,7]=3.;before=q.copy()
        report=adapter.joint_range_report(q,np.full(29,-2.),np.full(29,2.))
        self.assertFalse(report['encoded_lower12_within_physical_ranges'])
        self.assertFalse(report['values_clipped_or_repaired']);np.testing.assert_array_equal(q,before)





class FakeCPUPlanner:
    """Spawn-pickleable synthetic planner; never opens an ONNX/model file."""
    def __init__(self, path):
        import time
        if path.name == 'slow_loading':
            time.sleep(.4)

    def infer(self, context, alignment, **command):
        import time
        if command['speed'] == 123:
            time.sleep(.15)
        if command['speed'] == 124:
            time.sleep(3.)
        if command['seed'] == 999:
            raise ValueError('synthetic test inference error')
        qpos = np.tile(context[0], (60,1))
        qpos[:,0] = command['seed']
        forecast = adapter.Forecast(qpos, np.zeros((60,29)), np.ones(60,bool), {'synthetic_test':True})
        return forecast, forecast, {'synthetic_test':True, 'command':command, 'model_loaded':False}, {}, None


def wait_until(predicate, timeout=8.):
    import time
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        value=predicate()
        if value:
            return value
        time.sleep(.005)
    raise AssertionError('Isolated fake CPU worker did not reach expected state')


class AsyncPlannerTests(unittest.TestCase):
    def planner(self, path='unused_test_model', **kwargs):
        planner=adapter.AsyncPlanner(path, planner_factory=FakeCPUPlanner, **kwargs).start()
        self.addCleanup(planner.close)
        wait_until(lambda:planner.status()['ready'])
        return planner

    def request(self, planner, *, speed=.3, seed=1, mode=1, timeout_s=1., context=None):
        if context is None:
            context=sequence()[:4]
        alignment=adapter.WorldAlignment.from_actual([0,0,.75,1,0,0,0])
        kwargs={'mode':mode,'movement_world':[1,0,0] if mode else [0,0,0],
            'facing_world':[1,0,0],'speed':speed,'seed':seed,'timeout_s':timeout_s}
        return planner.submit(context, alignment, **kwargs)

    def test_load_runs_in_isolated_cpu_worker_and_idle_is_explicit_once_only(self):
        import time
        started=time.monotonic()
        planner=adapter.AsyncPlanner('slow_loading',planner_factory=FakeCPUPlanner).start()
        self.addCleanup(planner.close)
        self.assertLess(time.monotonic()-started,.3)
        wait_until(lambda:planner.status()['ready'])
        request_id=self.request(planner,mode=0,speed=-1.,seed=4)
        result=wait_until(planner.poll)
        self.assertEqual(result.request_id,request_id)
        self.assertEqual(result.evidence['command']['mode'],0)
        self.assertEqual(result.evidence['command']['movement_world'].tolist(),[0,0,0])
        self.assertFalse(result.evidence['model_loaded'])
        self.assertTrue(result.evidence['reference_only'])
        self.assertTrue(result.evidence['async_cpu_worker']['cpu_only'])
        self.assertEqual(result.evidence['async_cpu_worker']['ORT_intra_op_threads'],1)
        self.assertEqual(result.evidence['async_cpu_worker']['ORT_inter_op_threads'],1)
        self.assertEqual(len(result.evidence['async_cpu_worker']['cpus']),1)
        self.assertIsNone(planner.poll())

    def test_latest_only_drops_superseded_inference_and_copies_caller_arrays(self):
        planner=self.planner()
        first=self.request(planner,speed=123,seed=1)
        wait_until(lambda:planner.status().get('status')=='working')
        context=sequence()[:4]
        original=context[0,7]
        second=self.request(planner,mode=0,speed=-1,seed=2,context=context)
        self.assertGreater(second,first)
        context[:]=500
        result=wait_until(planner.poll)
        self.assertEqual(result.request_id,second)
        self.assertEqual(result.canonical.qpos50[0,7],original)
        self.assertEqual(result.world.qpos50[0,0],2)
        self.assertGreaterEqual(planner.status()['superseded_results'],1)
        self.assertIsNone(planner.poll())

    def test_expired_result_is_never_reused_and_following_request_can_recover(self):
        planner=self.planner()
        self.request(planner,speed=123,seed=1,timeout_s=.02)
        wait_until(lambda:planner.status().get('expired_results',0)>=1)
        self.assertTrue(planner.status()['latest_request_expired'])
        self.assertIsNone(planner.poll())
        second=self.request(planner,mode=0,speed=-1,seed=3)
        result=wait_until(planner.poll)
        self.assertEqual(result.request_id,second)
        self.assertEqual(result.world.qpos50[0,0],3)
        self.assertIsNone(planner.poll())

    def test_expiration_after_completion_before_poll_is_also_rejected(self):
        import time
        planner=self.planner()
        self.request(planner,seed=4,timeout_s=.2)
        wait_until(lambda:planner.status()['inference_calls']==1)
        deadline=planner._latest['expires_monotonic']
        while time.monotonic()<deadline+.01:
            time.sleep(.005)
        self.assertIsNone(planner.poll())
        self.assertEqual(planner.status()['dropped_polled_results'],1)

    def test_main_mailbox_busy_paths_never_wait_and_do_not_claim_a_request(self):
        import time
        planner=self.planner()
        with planner._requests.lock:
            started=time.monotonic()
            self.assertIsNone(self.request(planner,seed=5))
            self.assertLess(time.monotonic()-started,.05)
        self.assertEqual(planner.status()['latest_request_id'],0)
        self.assertEqual(planner.status()['busy_submits'],1)
        with planner._results.lock:
            started=time.monotonic()
            self.assertIsNone(planner.poll())
            self.assertLess(time.monotonic()-started,.05)

    def test_error_is_visible_and_close_terminates_hung_cpu_worker(self):
        import time
        import os
        planner=self.planner()
        self.request(planner,seed=999)
        state=wait_until(lambda:planner.status() if planner.status().get('error') else None)
        self.assertIn('synthetic test inference error',state['error'])
        self.assertIsNone(planner.poll())
        self.request(planner,speed=124,seed=2,timeout_s=.1)
        wait_until(lambda:planner.status().get('status')=='working')
        pid=planner._process.pid
        started=time.monotonic()
        planner.close(timeout_s=.01)
        self.assertLess(time.monotonic()-started,1.2)
        self.assertEqual(planner.status()['status'],'closed')
        self.assertIsNone(planner.poll())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid,0)
        with self.assertRaises(ValueError):
            planner.start()
        planner.close()

    def test_explicit_mode_and_input_validation_and_dynamic_source_paths(self):
        planner=self.planner()
        with self.assertRaises(ValueError):
            self.request(planner,mode=27)
        with self.assertRaises(ValueError):
            self.request(planner,timeout_s=0)
        with self.assertRaises(ValueError):
            self.request(planner,speed=float('nan'))
        context=sequence()[:4];context[0,3:7]=0
        with self.assertRaises(ValueError):
            self.request(planner,context=context)
        self.assertTrue(all(path.is_file() for path in adapter.SOURCE_FILES))
        self.assertTrue(all(path.is_relative_to(adapter.SCRIPT_DIR.parents[1]) for path in adapter.SOURCE_FILES))
        self.assertNotIn('/home/zxc/CVPR_Evaluation',Path(adapter.__file__).read_text())
        self.assertEqual(adapter.MODEL_SHA256,'39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea')


if __name__=='__main__':
    unittest.main()
