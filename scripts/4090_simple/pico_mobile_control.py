"""Opt-in PICO joystick / asynchronous SONIC planner reference bridge.

The live path only submits and polls an existing isolated CPU worker. It never
loads a model or runs inference in the 50 Hz caller. Generated joint/root poses
are references, never writes to an actual robot/simulator state or data labels.
The collector owns B, mode2 manipulation, finger commands, and transition gates.
CPU tests certify bookkeeping/rejection, not walking, load carrying or latency.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping

import numpy as np

from handover_sonic_teleop_reference import rotation
from pico_locomotion_reference import encoder_reference
from prepare_carrybox_v11 import BODY_NAMES
from sonic_locomotion_planner import (
    AsyncPlanner, Forecast, PlannerResult, WorldAlignment, bootstrap_context,
    context_from_forecast, crossfade8, future_window, replan_reason,
)


CONTRACT = 'cvpr-pico-mobile-frozen-upperbody-async-reference-v2'
CONTINUOUS_CONTRACT = 'cvpr-pico-mobile-continuous-vr-mode1-async-reference-v1'
UPPER_BODY_NAMES = tuple(BODY_NAMES[12:])
CONTROL_DT = .02


class MobileControlError(RuntimeError):
    """No reference may be executed after this error; explicitly boot again."""


def _get(frame, key, default=None):
    return frame.get(key, default) if isinstance(frame, Mapping) else getattr(frame, key, default)


def _axes(frame):
    inputs = _get(frame, 'inputs')
    if not isinstance(inputs, Mapping):
        raise ValueError('A complete same-packet PICO inputs mapping is required')
    result = []
    for name in ('left_axis_x', 'left_axis_y', 'right_axis_x', 'right_axis_y'):
        value = inputs.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1.:
            raise ValueError('Missing, nonfinite or out-of-range controller axis: '+name)
        result.append(float(value))
    return tuple(result)


def neutral_controls(frame, deadzone=.15):
    """All four physical axes must be neutral before changing control modes."""
    return all(abs(value) < deadzone for value in _axes(frame))


def transition_ready(frame, actual_root_pose7, actual_root_qvel6, *, deadzone=.15):
    """Measured standing/neutral transition gate; no state edits or inference.

    The caller must also require valid/fresh tracking and a calibrated mode2
    producer. A single passing sample is not a long-duration stability claim.
    """
    root = np.asarray(actual_root_pose7, dtype=float)
    velocity = np.asarray(actual_root_qvel6, dtype=float)
    if (root.shape != (7,) or velocity.shape != (6,)
            or not np.isfinite(np.r_[root, velocity]).all()
            or abs(np.linalg.norm(root[3:])-1.) > 1e-6):
        raise ValueError('Finite measured root pose/unit quaternion and root velocity required')
    tilt = math.degrees(math.acos(float(np.clip(rotation(root[3:])[2, 2], -1., 1.))))
    checks = {'sticks_neutral': neutral_controls(frame, deadzone), 'root_height': bool(root[2] > .6),
        'upright': tilt < 25., 'linear_speed': bool(np.linalg.norm(velocity[:3]) < .1),
        'angular_speed': bool(np.linalg.norm(velocity[3:]) < .2)}
    return {'passed': all(checks.values()), 'checks': checks, 'tilt_deg': tilt,
            'measured_state_changed': False}


@dataclass(frozen=True)
class MobileControlConfig:
    deadzone: float = .15
    maximum_speed_m_s: float = .3
    yaw_gain_rad_s: float = 1.5
    maximum_input_age_s: float = .10
    planning_timeout_s: float = .20
    loading_timeout_s: float = 60.
    # Unsupported directions can only be opened in an explicit diagnostic.
    allow_strafe_backward: bool = False

    def __post_init__(self):
        values = (self.deadzone, self.maximum_speed_m_s, self.yaw_gain_rad_s,
                  self.maximum_input_age_s, self.planning_timeout_s, self.loading_timeout_s)
        if (not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values)
                or self.deadzone >= 1 or not .1 <= self.maximum_speed_m_s <= .3
                or self.maximum_input_age_s > .2 or self.planning_timeout_s > .25
                or type(self.allow_strafe_backward) is not bool):
            raise ValueError('Invalid bounded mobile control configuration')


class SynchronousDiagnosticWorker:
    """Explicit offline adapter using the SAME bridge/timeline, never live.

    The caller constructs/loads the supplied planner before control starts.
    This adapter intentionally blocks in submit and cannot certify real time.
    """
    diagnostic_only = True

    def __init__(self, planner, *, clock=time.monotonic):
        self.planner, self.clock = planner, clock
        self.sequence, self.result, self.closed = 0, None, False

    def start(self):
        return self

    def status(self):
        return {'ready': not self.closed, 'status': 'closed' if self.closed else 'idle',
                'synchronous_diagnostic_only': True}

    def submit(self, context, alignment, *, timeout_s=.2, **command):
        submitted = self.clock()
        world, canonical, evidence, _, _ = self.planner.infer(context, alignment, **command)
        self.sequence += 1
        self.result = PlannerResult(self.sequence, world, canonical, evidence,
            submitted, self.clock(), submitted+timeout_s)
        return self.sequence

    def poll(self):
        result, self.result = self.result, None
        return result

    def close(self):
        self.closed, self.result = True, None


class MobilePlannerControl:
    """50 Hz asynchronous planner with explicit locked or live-VR references.

    boot(root, BODY29) starts loading, then poll() until status()['ready'].
    step(tick50, root, BODY29, frame) returns (float32[1,1751], JSON evidence).
    tick50 starts at zero and advances exactly once per real physics interval.
    The first frame must have neutral sticks. B/fingers are never consumed here.
    A fault is sticky: no output until explicit boot after caller recalibration.
    """
    def __init__(self, model_path, *, config=None, planner_worker=None,
                 clock=time.monotonic, diagnostic_only=False, seed=20261027,
                 reference_policy='locked_upper_mode0'):
        self.config = config or MobileControlConfig()
        if reference_policy not in ('locked_upper_mode0', 'continuous_vr_mode1'):
            raise ValueError('Explicit supported Mobile reference_policy required')
        self.reference_policy = reference_policy
        self.contract = CONTINUOUS_CONTRACT if reference_policy == 'continuous_vr_mode1' else CONTRACT
        self.clock, self.diagnostic_only, self.seed = clock, bool(diagnostic_only), seed
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('Nonnegative int64 planner seed required')
        if self.config.allow_strafe_backward and not self.diagnostic_only:
            raise ValueError('Backward/lateral control is restricted to isolated diagnostics')
        self.worker = planner_worker or AsyncPlanner(model_path,
            timeout_s=self.config.planning_timeout_s)
        if getattr(self.worker, 'diagnostic_only', False) and not self.diagnostic_only:
            raise ValueError('Synchronous worker forbidden in live control')
        self.phase, self.reason = 'not_booted', None
        self.active = self.pending = None
        self.last_tick = None
        self._started, self._closed = False, False
        self.epoch = 0

    @staticmethod
    def _state(root, body):
        root, body = np.asarray(root, dtype=float), np.asarray(body, dtype=float)
        if (root.shape != (7,) or body.shape != (29,)
                or not np.isfinite(np.r_[root, body]).all()
                or abs(np.linalg.norm(root[3:])-1.) > 1e-6):
            raise ValueError('Finite actual root/unit wxyz and named BODY29 required')
        return root.copy(), body.copy()

    def _now(self, now):
        now = self.clock() if now is None else now
        if type(now) not in (int, float) or not math.isfinite(now):
            raise ValueError('Finite monotonic host time required')
        return float(now)

    def _fail(self, reason):
        self.pause(reason)
        raise MobileControlError(reason)

    def boot(self, actual_root_pose7, actual_body29, now=None):
        if self._closed:
            raise MobileControlError('closed_mobile_controller')
        root, body = self._state(actual_root_pose7, actual_body29)
        now = self._now(now)
        if root[2] <= .6 or rotation(root[3:])[2, 2] <= math.cos(math.radians(25)):
            raise MobileControlError('boot_requires_upright_actual_robot')
        if not self._started:
            self.worker.start()
            self._started = True
        self.epoch += 1
        self.phase, self.reason = 'loading', None
        self.active, self.pending, self.last_tick = None, None, None
        self.origin_tick, self.boot_wall = 0, now
        self.alignment = WorldAlignment.from_actual(root)
        self.context, self.bootstrap = bootstrap_context(body)
        self.frozen_upper = body[12:].copy() if self.reference_policy == 'locked_upper_mode0' else None
        self.yaw = math.atan2(rotation(root[3:])[1, 0], rotation(root[3:])[0, 0])
        self.last_source_timestamp = self.last_source_sequence = None
        self.last_host_time = now
        self.last_command = self._idle()
        self.last_plan_tick = 0
        self.last_inference = None
        self.last_replan_reason = None
        self.mailbox_busy = 0
        return self.poll(now)

    def _idle(self):
        return {'mode': 0, 'movement_direction': [0., 0., 0.],
                'facing_direction': [math.cos(self.yaw), math.sin(self.yaw), 0.],
                'speed': -1., 'height': -1.}

    @staticmethod
    def _valid_forecast(forecast):
        q, dq, valid = forecast.qpos50, forecast.dq29, forecast.valid_forecast_mask
        if (q.ndim != 2 or q.shape[1] != 36 or len(q) > 110
                or dq.shape != (len(q), 29) or valid.shape != (len(q),)
                or valid.dtype != np.bool_ or not np.isfinite(q).all()
                or not np.isfinite(dq).all()
                or np.max(np.abs(np.linalg.norm(q[:, 3:7], axis=1)-1.)) > 1e-6):
            raise ValueError('Invalid planner reference array contract')
        # Copy worker arrays: neither caller nor frozen override alters evidence.
        return Forecast(q.copy(), dq.copy(), valid.copy(), dict(forecast.metadata))

    def _submit(self, command, tick, now, reason, *, boot=False):
        if boot:
            context, generated, context_evidence = self.context.copy(), 0, self.bootstrap
        else:
            context, context_evidence = context_from_forecast(self.active, tick-self.origin_tick)
            generated = self.origin_tick+context_evidence['generated_start_frame50']
            context = self.alignment.to_canonical(context)
        request = self.worker.submit(context, self.alignment, mode=command['mode'],
            movement_world=command['movement_direction'], facing_world=command['facing_direction'],
            speed=command['speed'], height=command['height'], seed=self.seed,
            timeout_s=self.config.planning_timeout_s)
        if request is None:
            self.mailbox_busy += 1
            return False
        self.pending = {'id': request, 'expires': now+self.config.planning_timeout_s,
            'submitted_tick50': tick, 'generated_tick50': generated, 'command': command.copy(),
            'reason': reason, 'boot': boot, 'context': context_evidence}
        return True

    def _consume(self, tick, now):
        result = self.worker.poll()
        if result is not None:
            if self.pending is None or result.request_id != self.pending['id']:
                self._fail('planner_result_sequence_mismatch')
            pending = self.pending
            if now >= pending['expires'] or now >= result.expires_monotonic:
                self._fail('planner_result_expired')
            fresh = self._valid_forecast(result.world)
            if pending['boot']:
                future_window(fresh, 0)
                self.active, self.origin_tick = fresh, 0
                self.phase = 'ready'
            else:
                current = tick-self.origin_tick
                merged = crossfade8(self.active, fresh, current,
                                    pending['generated_tick50']-self.origin_tick)
                future_window(merged, 0)
                self.active, self.origin_tick = merged, tick
            self.last_command = pending['command']
            self.last_plan_tick = pending['submitted_tick50']
            self.last_replan_reason = pending['reason']
            self.last_inference = {'request_id': result.request_id, 'evidence': result.evidence,
                'context': pending['context'], 'submitted_tick50': pending['submitted_tick50'],
                'generated_tick50': pending['generated_tick50'], 'accepted_tick50': tick}
            self.pending = None
        elif self.pending is not None and now >= self.pending['expires']:
            self._fail('planner_request_expired_no_reference')

    def poll(self, now=None):
        now = self._now(now)
        if self.phase in ('paused', 'not_booted', 'closed'):
            return self.status()
        try:
            worker_status = self.worker.status()
            if worker_status.get('status') in ('error', 'closed'):
                self._fail('planner_worker_unavailable: '+str(worker_status.get('error')))
            if self.phase != 'loading' and worker_status.get('ready') is not True:
                self._fail('planner_worker_lost_readiness')
            if self.phase == 'loading':
                if now-self.boot_wall >= self.config.loading_timeout_s:
                    self._fail('planner_model_loading_timeout')
                if worker_status.get('ready') and self.pending is None:
                    self._submit(self._idle(), 0, now, 'official_static_bootstrap', boot=True)
            self._consume(0 if self.last_tick is None else self.last_tick, now)
            return self.status()
        except MobileControlError:
            raise
        except Exception as exc:
            self._fail('planner_poll_rejected: '+str(exc))

    def _input_command(self, frame, now, tracking_valid):
        if tracking_valid is not True:
            self._fail('tracking_invalid_pause_required')
        original_axes = _axes(frame)
        lx, ly, rx, ry = original_axes
        received, timestamp, sequence = (_get(frame, key) for key in
            ('host_received_s', 'device_timestamp_ns', 'sequence'))
        if (type(received) not in (int, float) or not math.isfinite(received)
                or not -.02 <= now-received <= self.config.maximum_input_age_s
                or type(timestamp) is not int or timestamp <= 0
                or type(sequence) is not int or sequence < 0):
            self._fail('pico_input_stale_or_identity_invalid')
        if self.last_source_timestamp is not None:
            if timestamp < self.last_source_timestamp or sequence < self.last_source_sequence:
                self._fail('pico_input_time_or_sequence_reversed')
            if ((timestamp == self.last_source_timestamp) != (sequence == self.last_source_sequence)):
                self._fail('pico_input_timestamp_sequence_disagree')
        reused = timestamp == self.last_source_timestamp
        self.last_source_timestamp, self.last_source_sequence = timestamp, sequence
        if self.last_tick is None and not neutral_controls(frame, self.config.deadzone):
            self._fail('first_mobile_tick_requires_neutral_sticks')
        ignored_axes, ignored_reason = [], []
        if not self.config.allow_strafe_backward:
            # Unsupported intent is a fresh bounded input, not a tracking or
            # timing failure. Mask it explicitly instead of rejecting a whole
            # human episode. Pure sideways/backward intent becomes a new IDLE
            # command, which follows the same stop/replan path as released sticks.
            if lx != 0.:
                ignored_axes.append('left_axis_x')
                ignored_reason.append('unsupported_lateral_axis')
            if ly < 0.:
                ignored_axes.append('left_axis_y')
                ignored_reason.append('unsupported_backward_axis')
            lx = 0.
            ly = max(0., ly)
        if ry != 0.:
            ignored_axes.append('right_axis_y')
            ignored_reason.append('unbound_right_vertical_axis')
        if abs(rx) >= self.config.deadzone:
            self.yaw -= self.config.yaw_gain_rad_s*rx*CONTROL_DT
        facing = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.])
        magnitude = min(float(np.hypot(lx, ly)), 1.)
        if magnitude < self.config.deadzone:
            command = self._idle()
        else:
            # Exact official axis basis: forward=ly*facing, right=-lx*perp.
            direction = ly*facing-lx*np.array([-facing[1], facing[0], 0.])
            direction /= np.linalg.norm(direction)
            amount = (magnitude-self.config.deadzone)/(1.-self.config.deadzone)
            speed = min(self.config.maximum_speed_m_s, .1+.5*amount)
            command = {'mode': 1, 'movement_direction': direction.tolist(),
                'facing_direction': facing.tolist(), 'speed': speed, 'height': -1.}
        return command, {'source_sequence': sequence, 'device_timestamp_ns': timestamp,
            'source_age_s': now-received, 'source_packet_reused_as_new_sample': False,
            'source_packet_held_for_control_interval': reused,
            'axes_left_x_y_right_x_y': list(original_axes),
            'applied_axes_left_x_y_right_x_y': [lx, ly, rx, 0.],
            'ignored_axes': ignored_axes, 'ignored_reason': ignored_reason,
            'axis_policy': 'diagnostic_planar_all_directions' if self.config.allow_strafe_backward
                else 'production_forward_only_with_explicit_lateral_backward_mask'}

    def step(self, tick50, actual_root_pose7, actual_body29, frame, *, now=None, tracking_valid=True,
             vr_world_points=None, vr_world_quaternions=None):
        if self.phase not in ('ready', 'active'):
            raise MobileControlError('mobile_reference_not_ready: '+str(self.reason or self.phase))
        try:
            continuous = self.reference_policy == 'continuous_vr_mode1'
            if continuous and (vr_world_points is None or vr_world_quaternions is None):
                self._fail('continuous_mode1_requires_live_vr_points_and_quaternions')
            if not continuous and (vr_world_points is not None or vr_world_quaternions is not None):
                self._fail('locked_reference_policy_does_not_accept_live_vr_targets')
            now = self._now(now)
            root, _ = self._state(actual_root_pose7, actual_body29)
            expected = 0 if self.last_tick is None else self.last_tick+1
            if type(tick50) is not int or tick50 != expected:
                self._fail('nonconsecutive_actual_50hz_tick')
            if now < self.last_host_time:
                self._fail('host_monotonic_time_reversed')
            command, source = self._input_command(frame, now, tracking_valid)
            worker_status = self.worker.status()
            if (worker_status.get('status') in ('error', 'closed')
                    or worker_status.get('ready') is not True):
                self._fail('planner_worker_unavailable')
            self._consume(tick50, now)
            cursor = tick50-self.origin_tick
            # Stop/start supersedes a pending opposite mode; changing facing
            # waits for bounded current inference to avoid latest-only starvation.
            reason = None
            if self.pending is not None and command['mode'] != self.pending['command']['mode']:
                reason = 'operator_stop_start_supersedes_pending'
            elif self.pending is None:
                if command['mode'] != self.last_command['mode']:
                    reason = 'official_mode_changed'
                elif tick50 % 5 == 0:
                    reason = replan_reason(command, self.last_command,
                        max(0., (tick50-self.last_plan_tick)*CONTROL_DT), self.active, cursor)
                # Reserve inference time BEFORE all46 samples run out.
                reserve = math.ceil(self.config.planning_timeout_s/CONTROL_DT)+2
                lookahead = cursor+reserve
                try:
                    future_window(self.active, lookahead)
                except ValueError:
                    reason = reason or 'async_complete46_low_watermark'
            if reason is not None:
                submitted = self._submit(command, tick50, now, reason)
                if not submitted and reason == 'operator_stop_start_supersedes_pending':
                    self._fail('stop_start_request_mailbox_busy')
                # Diagnostic worker returns immediately; live just polls once.
                self._consume(tick50, now)
                cursor = tick50-self.origin_tick
            qpos, dq = future_window(self.active, cursor)
            if continuous:
                # Only planner lower-body channels enter mode1. Current upper
                # goals come from the caller's live, explicitly world-bound VR
                # points; no planner or frozen upper-body pose overwrites them.
                value, evidence = encoder_reference(qpos, dq, root,
                    vr_world_points=vr_world_points, vr_world_quaternions=vr_world_quaternions)
                evidence.update(live_upper_reference={
                    'source': 'caller_supplied_current_VR_three_point_world_targets',
                    'source_sequence': source['source_sequence'],
                    'device_timestamp_ns': source['device_timestamp_ns'],
                    'point_order': list(evidence['VR_point_order']),
                    'world_points_already_offset': True,
                    'world_points_m': np.asarray(vr_world_points, dtype=float).tolist(),
                    'world_quaternions_wxyz': np.asarray(vr_world_quaternions, dtype=float).tolist(),
                    'binding_scope': 'collector must bind these targets to the recorded current source packet'},
                    planner_upper_body_used_as_reference=False,
                    upper_body_frozen=False)
            else:
                qpos[:, 19:36] = self.frozen_upper
                dq[:, 12:29] = 0.
                value, evidence = encoder_reference(qpos, dq, root)
                evidence.update(frozen_upper_body_names=list(UPPER_BODY_NAMES),
                    frozen_upper_body_q17=self.frozen_upper.tolist(), frozen_upper_body_velocity17=[0.]*17,
                    frozen_upper_source='actual measured BODY29 at explicit mobile-mode entry',
                    upper_body_frozen=True)
            self.last_tick, self.last_host_time, self.phase = tick50, now, 'active'
            evidence.update(contract=self.contract, reference_policy=self.reference_policy,
                tick50=tick50, cursor50=cursor,
                timeline_origin_tick50=self.origin_tick, epoch=self.epoch,
                source=source, desired_command=command, applied_plan_command=self.last_command,
                movement_scope='diagnostic_all_planar_directions' if self.config.allow_strafe_backward
                    else 'forward_and_facing_only_lateral_and_backward_axes_ignored',
                pending_request_id=self.pending['id'] if self.pending else None,
                pending_deadline_monotonic=self.pending['expires'] if self.pending else None,
                last_replan_reason=self.last_replan_reason, last_inference=self.last_inference,
                finger_targets_source='collector passes current operator finger targets separately',
                actual_state_written=False, model_forecasts_are_training_labels=False,
                synchronous_diagnostic_only=self.diagnostic_only,
                live_timing_and_loaded_mobile_task_validated=False)
            return value, evidence
        except MobileControlError:
            raise
        except Exception as exc:
            self._fail('mobile_reference_rejected: '+str(exc))

    def status(self):
        return {'contract': self.contract, 'reference_policy': self.reference_policy, 'status': self.phase,
            'ready': self.phase in ('ready', 'active') and self.active is not None,
            'pause_reason': self.reason, 'last_tick50': self.last_tick,
            'pending_request_id': self.pending['id'] if self.pending else None,
            'epoch': self.epoch, 'mailbox_busy_submits': getattr(self, 'mailbox_busy', 0),
            'diagnostic_only': self.diagnostic_only, 'physical_task_readiness_certified': False}

    def pause(self, reason='operator_pause'):
        self.phase, self.reason = 'paused', str(reason)
        self.active, self.pending = None, None
        return self.status()

    def close(self):
        if not self._closed:
            self.pause('closed')
            self.worker.close()
            self._closed, self.phase = True, 'closed'
