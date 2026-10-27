"""Mobile bridge timeline/failure tests: no real models, SDK, GPU or physics."""
import json
import math
import os
from pathlib import Path
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pico_mobile_control import (
    CONTINUOUS_CONTRACT, MobileControlConfig, MobileControlError, MobilePlannerControl,
    SynchronousDiagnosticWorker, neutral_controls, transition_ready,
)
from prepare_carrybox_v11 import POLICY_TO_BODY
from sonic_locomotion_planner import AsyncPlanner, Forecast, PlannerResult


ROOT = np.array([0., 0., .78, 1., 0., 0., 0.])
BODY = np.arange(29, dtype=float)/100.


class Clock:
    value = 100.

    def __call__(self):
        return self.value


def frame(now, tick=0, *, lx=0., ly=0., rx=0., ry=0.):
    return {'inputs': {'left_axis_x': lx, 'left_axis_y': ly,
        'right_axis_x': rx, 'right_axis_y': ry}, 'host_received_s': now,
        'device_timestamp_ns': 1000000000+tick*20000000, 'sequence': tick+1}


def forecast(rows=72, *, marker=0.):
    q = np.zeros((rows, 36))
    q[:, 2], q[:, 3] = .78, 1.
    # Distinct future indices make delayed rebasing visible.
    q[:, 7] = marker+np.arange(rows)/100.
    q[:, 19:] = .8
    return Forecast(q, np.ones((rows, 29))*.5, np.ones(rows, dtype=bool), {})


class FakeWorker:
    """Manually deliver worker results to test arbitrary load/inference delays."""
    def __init__(self, clock):
        self.clock, self.requests, self.result = clock, [], None
        self.ready, self.closed, self.busy = True, False, False

    def start(self):
        return self

    def status(self):
        return {'status': 'closed' if self.closed else 'idle' if self.ready else 'loading',
                'ready': self.ready}

    def submit(self, context, alignment, **kwargs):
        if self.busy:
            return None
        request_id = len(self.requests)+1
        self.requests.append({'id': request_id, 'context': context.copy(),
            'alignment': alignment, 'submitted': self.clock(), **kwargs})
        return request_id

    def deliver(self, value=None, *, request=None, result_id=None, expires=None):
        request = request or self.requests[-1]
        value = value or forecast()
        self.result = PlannerResult(request['id'] if result_id is None else result_id,
            value, value, {'fake_worker_only': True}, request['submitted'], self.clock(),
            request['submitted']+request['timeout_s'] if expires is None else expires)

    def poll(self):
        result, self.result = self.result, None
        return result

    def close(self):
        self.closed = True


class ImmediateFakePlanner:
    def infer(self, context, alignment, **command):
        value = forecast()
        return value, value, {'fake_offline': True}, {}, None


class SpawnFakePlanner:
    """Pickleable stand-in, executed inside the real AsyncPlanner subprocess."""
    def __init__(self, path):
        self.path = path

    def infer(self, context, alignment, **command):
        value = forecast()
        return value, value, {'fake_model_only': True, 'inference_pid': os.getpid()}, {}, None


class MobileBridgeTests(unittest.TestCase):
    def bridge(self, *, ready=True, config=None, diagnostic=False, reference_policy='locked_upper_mode0'):
        self.clock = Clock()
        self.worker = FakeWorker(self.clock)
        self.worker.ready = ready
        bridge = MobilePlannerControl('unused_fake_path', config=config,
            planner_worker=self.worker, clock=self.clock, diagnostic_only=diagnostic,
            reference_policy=reference_policy)
        self.addCleanup(bridge.close)
        bridge.boot(ROOT, BODY)
        return bridge

    def prepared(self, **kwargs):
        bridge = self.bridge(**kwargs)
        self.worker.deliver()
        self.assertTrue(bridge.poll()['ready'])
        return bridge

    def step(self, bridge, tick, *, vr_world_points=None, vr_world_quaternions=None, **axes):
        self.clock.value = 100.+tick*.02
        return bridge.step(tick, ROOT, BODY, frame(self.clock(), tick, **axes),
            vr_world_points=vr_world_points, vr_world_quaternions=vr_world_quaternions)

    def live_vr(self):
        return np.array([[.4, .3, .9], [.4, -.3, .9], [0., 0., 1.3]]), np.tile([1., 0., 0., 0.], (3, 1))

    def test_continuous_mode1_changes_both_live_arms_without_changing_lower_reference(self):
        points, orientations = self.live_vr()
        base = self.prepared(reference_policy='continuous_vr_mode1')
        before_q, before_dq = base.active.qpos50.copy(), base.active.dq29.copy()
        value, evidence = self.step(base, 0, vr_world_points=points, vr_world_quaternions=orientations)
        np.testing.assert_array_equal(base.active.qpos50, before_q)
        np.testing.assert_array_equal(base.active.dq29, before_dq)
        self.assertIsNone(base.frozen_upper)
        self.assertEqual(value[0, 0], 1.)
        self.assertEqual(evidence['encoder_mode'], 1)
        self.assertEqual(evidence['contract'], CONTINUOUS_CONTRACT)
        self.assertEqual(evidence['reference_policy'], 'continuous_vr_mode1')
        self.assertFalse(evidence['upper_body_frozen'])
        self.assertFalse(evidence['planner_upper_body_used_as_reference'])
        self.assertFalse(any(key.startswith('frozen_upper') for key in evidence))
        self.assertEqual(evidence['live_upper_reference']['world_points_m'], points.tolist())
        self.assertEqual(evidence['live_upper_reference']['source_sequence'], 1)
        self.assertEqual(base.status()['contract'], CONTINUOUS_CONTRACT)
        self.assertEqual(base.status()['reference_policy'], 'continuous_vr_mode1')
        np.testing.assert_array_equal(value[0, 4:644], 0.)  # inactive full-body mode0 channels
        for side in (0, 1):
            changed = points.copy(); changed[side] += [.07, .03, -.02]
            changed_orientations = orientations.copy(); changed_orientations[side] = [0., 0., 0., 1.]
            other = self.prepared(reference_policy='continuous_vr_mode1')
            changed_value, changed_evidence = self.step(other, 0,
                vr_world_points=changed, vr_world_quaternions=changed_orientations)
            np.testing.assert_array_equal(changed_value[0, 644:890], value[0, 644:890])
            self.assertFalse(np.array_equal(changed_value[0, 890 + side * 3:893 + side * 3],
                                           value[0, 890 + side * 3:893 + side * 3]))
            self.assertFalse(np.array_equal(changed_value[0, 899 + side * 4:903 + side * 4],
                                           value[0, 899 + side * 4:903 + side * 4]))
            self.assertEqual(changed_evidence['encoder_mode'], 1)

    def test_continuous_missing_or_invalid_live_vr_never_falls_back_to_mode0(self):
        points, orientations = self.live_vr()
        for kwargs in ({}, {'vr_world_points': points}, {'vr_world_quaternions': orientations},
                {'vr_world_points': np.zeros((2, 3)), 'vr_world_quaternions': orientations},
                {'vr_world_points': np.full((3, 3), np.nan), 'vr_world_quaternions': orientations},
                {'vr_world_points': points, 'vr_world_quaternions': np.zeros((3, 4))}):
            bridge = self.prepared(reference_policy='continuous_vr_mode1')
            with self.assertRaises(MobileControlError):
                self.step(bridge, 0, **kwargs)
            self.assertEqual(bridge.status()['status'], 'paused')
            self.assertFalse(bridge.status()['ready'])

    def test_reference_policy_is_explicit_and_locked_cannot_silently_ignore_live_vr(self):
        with self.assertRaises(ValueError):
            self.bridge(reference_policy='guess')
        bridge = self.prepared()
        points, orientations = self.live_vr()
        with self.assertRaisesRegex(MobileControlError, 'locked_reference_policy'):
            self.step(bridge, 0, vr_world_points=points, vr_world_quaternions=orientations)

    def test_continuous_keeps_real_tick_progression_with_fresh_vr_each_step(self):
        bridge = self.prepared(reference_policy='continuous_vr_mode1')
        points, orientations = self.live_vr()
        outputs = []
        for tick in range(4):
            current = points.copy(); current[0, 0] += tick * .01
            value, evidence = self.step(bridge, tick, vr_world_points=current,
                vr_world_quaternions=orientations)
            outputs.append(value[0, 890])
            self.assertEqual(evidence['tick50'], tick)
            self.assertEqual(evidence['live_upper_reference']['device_timestamp_ns'],
                             frame(self.clock(), tick)['device_timestamp_ns'])
        self.assertTrue(all(b > a for a, b in zip(outputs, outputs[1:])))
        with self.assertRaisesRegex(MobileControlError, 'nonconsecutive'):
            self.step(bridge, 5, vr_world_points=points, vr_world_quaternions=orientations)

    def test_boot_waits_for_model_load_before_short_inference_deadline(self):
        bridge = self.bridge(ready=False)
        self.assertFalse(bridge.status()['ready'])
        self.assertEqual(self.worker.requests, [])
        self.clock.value += 10.
        self.worker.ready = True
        self.assertFalse(bridge.poll()['ready'])
        self.assertEqual(len(self.worker.requests), 1)
        self.assertEqual(self.worker.requests[0]['submitted'], 110.)
        self.worker.deliver()
        self.assertTrue(bridge.poll()['ready'])
        self.assertEqual(self.worker.requests[0]['mode'], 0)
        self.assertEqual(self.worker.requests[0]['speed'], -1.)

    def test_frozen_actual_upper17_velocities_and_no_actual_state_mutation(self):
        bridge = self.prepared()
        root, body = ROOT.copy(), BODY.copy()
        original = bridge.active
        before_q, before_dq = original.qpos50.copy(), original.dq29.copy()
        value, evidence = bridge.step(0, root, body, frame(self.clock()))
        policy_q = value[0, 4:294].reshape(10, 29)
        policy_dq = value[0, 294:584].reshape(10, 29)
        upper_policy = np.flatnonzero(POLICY_TO_BODY >= 12)
        np.testing.assert_allclose(policy_q[:, upper_policy],
            np.tile(BODY[POLICY_TO_BODY[upper_policy]], (10, 1)), atol=1e-7)
        np.testing.assert_array_equal(policy_dq[:, upper_policy], 0.)
        np.testing.assert_array_equal(root, ROOT)
        np.testing.assert_array_equal(body, BODY)
        np.testing.assert_array_equal(original.qpos50, before_q)
        np.testing.assert_array_equal(original.dq29, before_dq)
        self.assertEqual(len(evidence['frozen_upper_body_names']), 17)
        self.assertFalse(evidence['actual_state_written'])
        json.dumps(evidence, allow_nan=False)

    def test_official_forward_right_turn_basis_and_speed_bound(self):
        bridge = self.prepared()
        self.step(bridge, 0)
        _, evidence = self.step(bridge, 1, ly=1., rx=1.)
        command = evidence['desired_command']
        self.assertEqual(command['mode'], 1)
        self.assertEqual(command['speed'], .3)
        np.testing.assert_allclose(command['movement_direction'], [math.cos(.03), -math.sin(.03), 0.])
        self.assertLess(command['facing_direction'][1], 0.)
        self.assertEqual(self.worker.requests[-1]['mode'], 1)

    def test_sub_deadzone_lateral_noise_cannot_become_diagonal_or_walk(self):
        bridge = self.prepared()
        self.step(bridge, 0)
        _, evidence = self.step(bridge, 1, lx=.14, ly=.14)
        self.assertEqual(evidence['desired_command']['mode'], 0)
        _, evidence = self.step(bridge, 2, lx=.14, ly=.16)
        np.testing.assert_array_equal(evidence['desired_command']['movement_direction'], [1., 0., 0.])

    def test_delayed_result_rebases_at_actual_acceptance_tick_not_submission(self):
        bridge = self.prepared()
        self.step(bridge, 0)
        self.step(bridge, 1, ly=1.)  # new forecast starts at absolute tick3 (+2)
        request = self.worker.requests[-1]
        for tick in range(2, 5):
            self.step(bridge, tick, ly=1.)
        self.clock.value = 100.1
        self.worker.deliver(forecast(marker=1.), request=request)
        value, evidence = self.step(bridge, 5, ly=1.)
        self.assertEqual(evidence['timeline_origin_tick50'], 5)
        self.assertEqual(evidence['last_inference']['generated_tick50'], 3)
        self.assertEqual(evidence['last_inference']['accepted_tick50'], 5)
        # Arrival preserves the old current pose. Five frames into the blend,
        # old absolute tick10 mixes with new row7 (forecast starts at tick3).
        policy_index = int(np.flatnonzero(POLICY_TO_BODY == 0)[0])
        self.assertAlmostEqual(float(value[0, 4+policy_index]), .05, places=6)
        self.assertAlmostEqual(float(value[0, 4+29+policy_index]), .375*.10+.625*1.07, places=6)

    def test_expired_pending_plan_sticky_pauses_instead_of_old_walking(self):
        bridge = self.prepared()
        self.step(bridge, 0)
        self.step(bridge, 1, ly=1.)
        self.clock.value = 100.23
        with self.assertRaisesRegex(MobileControlError, 'expired'):
            bridge.step(2, ROOT, BODY, frame(self.clock(), 2, ly=1.))
        self.assertEqual(bridge.status()['status'], 'paused')
        self.assertIsNone(bridge.active)
        with self.assertRaises(MobileControlError):
            self.step(bridge, 2)

    def test_old_worker_id_and_expired_return_are_rejected(self):
        for result_id, expires in ((99, None), (None, 99.)):
            with self.subTest(result_id=result_id):
                bridge = self.bridge()
                self.worker.deliver(result_id=result_id, expires=expires)
                with self.assertRaises(MobileControlError):
                    bridge.poll()
                self.assertEqual(bridge.status()['status'], 'paused')

    def test_low_watermark_requests_early_and_never_pads_missing46_future(self):
        bridge = self.bridge()
        self.worker.deliver(forecast(rows=59))
        bridge.poll()
        self.step(bridge, 0)
        self.step(bridge, 1)
        self.step(bridge, 2)
        self.assertEqual(len(self.worker.requests), 2)
        self.assertEqual(bridge.pending['reason'], 'async_complete46_low_watermark')
        # Pending worker is still within deadline, but window contains a hole.
        bridge.active.valid_forecast_mask[47] = False
        with self.assertRaisesRegex(MobileControlError, '46-frame'):
            self.step(bridge, 3)
        self.assertEqual(bridge.status()['status'], 'paused')

    def test_stop_supersedes_pending_walk_and_busy_stop_rejects(self):
        bridge = self.prepared()
        self.step(bridge, 0)
        self.step(bridge, 1, ly=1.)
        walk_id = bridge.pending['id']
        self.step(bridge, 2)
        self.assertGreater(bridge.pending['id'], walk_id)
        self.assertEqual(bridge.pending['command']['mode'], 0)
        self.step(bridge, 3, ly=1.)
        self.worker.busy = True
        with self.assertRaisesRegex(MobileControlError, 'mailbox_busy'):
            self.step(bridge, 4)

    def test_tick_jump_tracking_loss_stale_and_reversed_input_fail_closed(self):
        for failure in ('tick', 'tracking', 'stale', 'backwards_time'):
            with self.subTest(failure=failure):
                bridge = self.prepared()
                self.step(bridge, 0)
                self.clock.value = 100.02
                sample = frame(self.clock(), 1)
                kwargs, tick = {}, 1
                if failure == 'tick':
                    tick = 2
                elif failure == 'tracking':
                    kwargs['tracking_valid'] = False
                elif failure == 'stale':
                    sample['host_received_s'] -= .2
                else:
                    sample['device_timestamp_ns'] = 1
                with self.assertRaises(MobileControlError):
                    bridge.step(tick, ROOT, BODY, sample, **kwargs)
                self.assertFalse(bridge.status()['ready'])

    def test_neutral_entry_and_recalibrated_boot_capture_new_actual_upper(self):
        bridge = self.prepared()
        with self.assertRaisesRegex(MobileControlError, 'neutral'):
            self.step(bridge, 0, ly=.8)
        new_body = BODY+.1
        bridge.boot(ROOT, new_body)
        self.worker.deliver()
        bridge.poll()
        _, evidence = self.step(bridge, 0)
        np.testing.assert_array_equal(evidence['frozen_upper_body_q17'], new_body[12:])
        self.assertEqual(evidence['epoch'], 2)

    def test_unsupported_axes_become_explicit_idle_without_rejecting_or_enabling_direction(self):
        for axes, raw_axes, ignored in (({'lx': .7}, [.7, 0., 0., 0.], 'unsupported_lateral_axis'),
                                       ({'ly': -.8}, [0., -.8, 0., 0.], 'unsupported_backward_axis')):
            bridge = self.prepared()
            self.step(bridge, 0)
            _, evidence = self.step(bridge, 1, **axes)
            self.assertEqual(evidence['desired_command']['mode'], 0)
            self.assertEqual(evidence['desired_command']['movement_direction'], [0., 0., 0.])
            self.assertEqual(evidence['source']['axes_left_x_y_right_x_y'], raw_axes)
            self.assertEqual(evidence['source']['applied_axes_left_x_y_right_x_y'], [0.] * 4)
            self.assertIn(ignored, evidence['source']['ignored_reason'])
            self.assertEqual(bridge.status()['status'], 'active')

    def test_malformed_axes_and_live_diagnostic_directions_still_fail_closed(self):
        for axes in ({'lx': float('nan')}, {'ly': -1.1}, {'rx': float('nan')}, {'ry': float('inf')}):
            bridge = self.prepared()
            self.step(bridge, 0)
            with self.assertRaises(MobileControlError):
                self.step(bridge, 1, **axes)
        with self.assertRaises(ValueError):
            self.bridge(config=MobileControlConfig(allow_strafe_backward=True))

    def test_supported_forward_and_turn_remain_legal_with_ignored_axis_components(self):
        bridge = self.prepared()
        self.step(bridge, 0)
        _, evidence = self.step(bridge, 1, lx=.8, ly=.9, rx=.5, ry=-.7)
        command, source = evidence['desired_command'], evidence['source']
        self.assertEqual(command['mode'], 1)
        np.testing.assert_allclose(command['movement_direction'], command['facing_direction'])
        self.assertLess(command['facing_direction'][1], 0.)
        self.assertEqual(source['axes_left_x_y_right_x_y'], [.8, .9, .5, -.7])
        self.assertEqual(source['applied_axes_left_x_y_right_x_y'], [0., .9, .5, 0.])
        self.assertEqual(source['ignored_axes'], ['left_axis_x', 'right_axis_y'])
        self.assertEqual(source['ignored_reason'], ['unsupported_lateral_axis', 'unbound_right_vertical_axis'])

    def test_backward_with_right_turn_is_idle_translation_but_keeps_turning(self):
        bridge = self.prepared()
        self.step(bridge, 0)
        _, evidence = self.step(bridge, 1, lx=-.5, ly=-.8, rx=.5)
        command, source = evidence['desired_command'], evidence['source']
        self.assertEqual(command['mode'], 0)
        self.assertEqual(command['movement_direction'], [0., 0., 0.])
        self.assertLess(command['facing_direction'][1], 0.)
        self.assertEqual(source['applied_axes_left_x_y_right_x_y'], [0., 0., .5, 0.])
        self.assertEqual(set(source['ignored_reason']), {'unsupported_lateral_axis', 'unsupported_backward_axis'})

    def test_masked_backward_submits_fresh_stop_over_pending_walk_then_forward_recovers(self):
        bridge = self.prepared()
        self.step(bridge, 0)
        self.step(bridge, 1, ly=.9)
        walk_request = self.worker.requests[-1]['id']
        _, stopped = self.step(bridge, 2, lx=.8, ly=-.9)
        self.assertGreater(self.worker.requests[-1]['id'], walk_request)
        self.assertEqual(self.worker.requests[-1]['mode'], 0)
        self.assertEqual(stopped['desired_command']['mode'], 0)
        self.assertEqual(stopped['last_inference']['request_id'], 1)  # original bootstrap, not invented stop output
        self.assertEqual(bridge.pending['reason'], 'operator_stop_start_supersedes_pending')
        self.worker.deliver()
        _, applied = self.step(bridge, 3, ly=-.9)
        self.assertEqual(applied['desired_command']['mode'], 0)
        self.assertEqual(applied['applied_plan_command']['mode'], 0)
        self.assertFalse(applied['source']['source_packet_held_for_control_interval'])
        _, resumed = self.step(bridge, 4, ly=.9)
        self.assertEqual(resumed['desired_command']['mode'], 1)
        self.assertEqual(self.worker.requests[-1]['mode'], 1)
        self.assertEqual(resumed['source']['ignored_reason'], [])
        self.assertEqual(bridge.last_tick, 4)
        self.assertTrue(bridge.status()['ready'])

    def test_masked_sideways_requests_idle_over_an_already_applied_walk(self):
        bridge = self.prepared()
        self.step(bridge, 0)
        self.step(bridge, 1, ly=.8)
        self.worker.deliver()
        _, walking = self.step(bridge, 2, ly=.8)
        self.assertEqual(walking['applied_plan_command']['mode'], 1)
        walking_id = walking['last_inference']['request_id']
        _, stopped = self.step(bridge, 3, lx=.8)
        self.assertEqual(stopped['desired_command']['mode'], 0)
        self.assertEqual(self.worker.requests[-1]['mode'], 0)
        self.assertGreater(self.worker.requests[-1]['id'], walking_id)
        self.assertEqual(bridge.pending['reason'], 'official_mode_changed')
        self.assertEqual(stopped['source']['device_timestamp_ns'], frame(100.06, 3)['device_timestamp_ns'])
        self.worker.deliver()
        _, settled = self.step(bridge, 4, lx=.8)
        self.assertEqual(settled['applied_plan_command']['mode'], 0)
        self.assertEqual(settled['source']['ignored_reason'], ['unsupported_lateral_axis'])
        self.assertEqual(bridge.status()['status'], 'active')

    def test_input_contract_v2_preserves_original_packet_without_modifying_it(self):
        bridge = self.prepared()
        self.step(bridge, 0)
        self.clock.value = 100.02
        sample = frame(self.clock(), 1, lx=.1, ly=-.2, rx=.2, ry=.3)
        original = json.loads(json.dumps(sample))
        _, evidence = bridge.step(1, ROOT, BODY, sample)
        self.assertEqual(sample, original)
        self.assertEqual(evidence['contract'], 'cvpr-pico-mobile-frozen-upperbody-async-reference-v2')
        self.assertEqual(bridge.status()['contract'], evidence['contract'])
        self.assertEqual(evidence['source']['axes_left_x_y_right_x_y'], [.1, -.2, .2, .3])

    def test_explicit_offline_sync_uses_same_interface_and_is_forbidden_live(self):
        clock = Clock()
        worker = SynchronousDiagnosticWorker(ImmediateFakePlanner(), clock=clock)
        with self.assertRaises(ValueError):
            MobilePlannerControl(None, planner_worker=worker)
        bridge = MobilePlannerControl(None, planner_worker=worker, clock=clock,
            diagnostic_only=True, config=MobileControlConfig(allow_strafe_backward=True))
        self.addCleanup(bridge.close)
        self.assertTrue(bridge.boot(ROOT, BODY)['ready'])
        bridge.step(0, ROOT, BODY, frame(clock()))
        clock.value += .02
        _, evidence = bridge.step(1, ROOT, BODY, frame(clock(), 1, lx=1.))
        np.testing.assert_allclose(evidence['desired_command']['movement_direction'], [0., -1., 0.])
        self.assertTrue(evidence['synchronous_diagnostic_only'])

    def test_transition_gate_uses_actual_velocity_tilt_and_all_four_axes(self):
        sample = frame(100.)
        self.assertTrue(neutral_controls(sample))
        self.assertTrue(transition_ready(sample, ROOT, np.zeros(6))['passed'])
        self.assertFalse(transition_ready(sample, ROOT, [0., 0., 0., 0., 0., .4])['passed'])
        self.assertFalse(transition_ready(frame(100., ry=.4), ROOT, np.zeros(6))['passed'])
        tilted = ROOT.copy()
        tilted[3:] = [math.cos(.4), math.sin(.4), 0., 0.]
        self.assertFalse(transition_ready(sample, tilted, np.zeros(6))['passed'])

    def test_real_async_worker_bridge_keeps_inference_out_of_control_process(self):
        worker = AsyncPlanner('unused_fake_model_path', planner_factory=SpawnFakePlanner,
                              timeout_s=.25)
        bridge = MobilePlannerControl(None, planner_worker=worker,
            config=MobileControlConfig(planning_timeout_s=.25))
        self.addCleanup(bridge.close)
        bridge.boot(ROOT, BODY)
        deadline = time.monotonic()+10.
        while not bridge.poll()['ready']:
            if time.monotonic() >= deadline:
                self.fail('Real isolated fake worker did not become ready')
            time.sleep(.005)
        now = time.monotonic()
        value, evidence = bridge.step(0, ROOT, BODY, frame(now), now=now)
        self.assertEqual(value.shape, (1, 1751))
        inference = evidence['last_inference']['evidence']
        self.assertNotEqual(inference['inference_pid'], os.getpid())
        self.assertTrue(inference['async_cpu_worker']['cpu_only'])
        self.assertTrue(inference['fake_model_only'])


if __name__ == '__main__':
    unittest.main()
