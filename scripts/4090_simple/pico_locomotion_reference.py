"""Opt-in SONIC v1.1 planner references; never starts the live PICO receiver.

The synchronous timeline is for deterministic offline physical diagnostics.
Use AsyncPlanner for a future real-time caller: a planner inference must never
run inside the current collector's 50 Hz control loop. No production entry
imports this module. Forecasts are desired references, not observed actions.
"""
from __future__ import annotations

import numpy as np

from handover_sonic_teleop_reference import build_teleop_reference, rotation
from prepare_carrybox_v11 import encoder_input, FUTURE_OFFSETS
from sonic_locomotion_planner import (
    OfficialPlannerCPU, WorldAlignment, bootstrap_context, context_from_forecast,
    crossfade8, future_window, replan_reason,
)


def encoder_reference(qpos46, dq46, actual_root_pose7, *, vr_world_points=None,
                      vr_world_quaternions=None):
    """Assemble official mode0 or explicit mode1, with the actual robot heading.

    Mode1 requires both already-offset world VR targets and their orientations.
    It must not silently replace the live collector's SMPL mode2 producer.
    """
    q = np.asarray(qpos46, dtype=np.float64)
    dq = np.asarray(dq46, dtype=np.float64)
    root = np.asarray(actual_root_pose7, dtype=np.float64)
    if (q.shape != (46, 36) or dq.shape != (46, 29) or root.shape != (7,)
            or not all(np.isfinite(v).all() for v in (q, dq, root))):
        raise ValueError('Finite complete46 qpos/dq and actual root required')
    if abs(np.linalg.norm(root[3:]) - 1.) > 1e-6:
        raise ValueError('Actual root must use unit wxyz')
    if (vr_world_points is None) != (vr_world_quaternions is None):
        raise ValueError('Mode1 requires both VR positions and orientations')
    if vr_world_points is not None:
        value, evidence = build_teleop_reference(
            q[:, 7:19], dq[:, :12], 0, root, vr_world_points,
            vr_world_quaternions, q[0, 3:7])
        return value, dict(evidence, producer='planner_lower_body_plus_explicit_VR_mode1',
                           model_forecasts_are_training_labels=False)
    value = encoder_input(q[:, 7:], dq, q[:, 3:7], 0)
    actual_r = rotation(root[3:])
    yaw = np.arctan2(actual_r[1, 0], actual_r[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    inverse = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
    value[0, 584:644] = (inverse[None] @ np.stack(
        [rotation(v) for v in q[FUTURE_OFFSETS, 3:7]]))[..., :2].reshape(-1)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite mode0 reference')
    return value, {'encoder_mode': 0, 'producer': 'official_full_body_planner_mode0',
                   'actual_heading_removed': True, 'complete46_future': True,
                   'model_forecasts_are_training_labels': False}


class OfflinePlannerReference:
    """Official V2 timeline plus explicit strict-future low-watermark replan.

    Each call advances exactly one real simulated 50 Hz interval. Initial
    context uses one measured BODY29 pose repeated four times, explicitly a
    bootstrap. Later contexts use forecasts, matching the official planner.
    """
    def __init__(self, planner_model, *, seed=20261025, planner=None):
        self.planner = planner or OfficialPlannerCPU(planner_model)
        self.seed = int(seed)
        self.active = None
        self.cursor = 0
        self.last_tick = None
        self.last_command = None
        self.last_plan_tick = None
        self.next_check = 0

    @staticmethod
    def _command(command):
        if hasattr(command, 'planner_fields'):
            command = command.planner_fields()
        command = dict(command)
        if set(command) != {'mode', 'movement_direction', 'facing_direction', 'speed', 'height'}:
            raise ValueError('Complete explicit planner command required')
        if type(command['mode']) is not int or command['mode'] not in (0, 1):
            raise ValueError('This bridge permits IDLE or SLOW_WALK only')
        direction = np.asarray(command['movement_direction'], dtype=float)
        facing = np.asarray(command['facing_direction'], dtype=float)
        if (direction.shape != (3,) or facing.shape != (3,)
                or not np.isfinite(np.r_[direction, facing, command['speed'], command['height']]).all()
                or abs(direction[2]) > 1e-8 or abs(facing[2]) > 1e-8
                or abs(np.linalg.norm(facing) - 1.) > 1e-6):
            raise ValueError('Finite planar unit facing and direction required')
        if command['mode'] == 0:
            if np.linalg.norm(direction) > 1e-8 or command['speed'] != -1.:
                raise ValueError('IDLE must have zero movement and explicit speed=-1')
        elif not (.1 <= command['speed'] <= .4 and abs(np.linalg.norm(direction)-1) <= 1e-6):
            raise ValueError('SLOW_WALK must use unit direction and speed .1.. .4 m/s')
        return command

    def step(self, tick50, actual_root_pose7, actual_body29, command, **vr_targets):
        if type(tick50) is not int or tick50 < 0:
            raise ValueError('Nonnegative integer 50 Hz tick required')
        command = self._command(command)
        if self.active is None:
            if tick50 != 0:
                raise ValueError('Reference must start at tick zero')
            self.alignment = WorldAlignment.from_actual(actual_root_pose7)
            context, self.bootstrap = bootstrap_context(actual_body29)
            self.active, _, self.last_evidence, _, _ = self.planner.infer(
                context, self.alignment, seed=self.seed,
                mode=command['mode'], movement_world=command['movement_direction'],
                facing_world=command['facing_direction'], speed=command['speed'], height=command['height'])
            self.last_tick = self.last_plan_tick = tick50
            self.next_check = tick50 + 5
            self.last_command = command
            reason = 'official_static_bootstrap'
        else:
            if tick50 != self.last_tick + 1:
                raise ValueError('Reference must advance with every actual 50 Hz interval')
            self.cursor += 1
            self.last_tick = tick50
            reason = None
            if command['mode'] != self.last_command['mode']:
                reason = 'official_mode_changed'
            elif tick50 >= self.next_check:
                reason = replan_reason(command, self.last_command,
                                       (tick50-self.last_plan_tick)/50., self.active, self.cursor)
                self.next_check = tick50 + 5
            try:
                future_window(self.active, self.cursor)
            except ValueError:
                reason = 'strict_current_window_invalid_no_padding'
            if reason is not None:
                context, context_evidence = context_from_forecast(self.active, self.cursor)
                fresh, _, self.last_evidence, _, _ = self.planner.infer(
                    self.alignment.to_canonical(context), self.alignment, seed=self.seed,
                    mode=command['mode'], movement_world=command['movement_direction'],
                    facing_world=command['facing_direction'], speed=command['speed'], height=command['height'])
                self.last_evidence['context_evidence'] = context_evidence
                self.active = crossfade8(self.active, fresh, self.cursor,
                                         context_evidence['generated_start_frame50'])
                self.cursor = 0
                self.last_plan_tick = tick50
                self.last_command = command
                self.next_check = tick50 + 5
        qpos, dq = future_window(self.active, self.cursor)
        value, evidence = encoder_reference(qpos, dq, actual_root_pose7, **vr_targets)
        evidence.update(intent=command, tick50=tick50, cursor50=self.cursor,
                        replan_reason=reason, last_inference=self.last_evidence,
                        forecast_root_pose7=qpos[0, :7].tolist(), bootstrap=self.bootstrap,
                        synchronous_offline_only=True)
        return value, evidence
