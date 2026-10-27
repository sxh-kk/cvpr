"""Task identity and operator transitions for a separate Mobile P&P collection.

These checks inspect supplied observations only. Importing never starts an SDK,
planner, simulator or renderer, and does not change the Handover default.
"""
from __future__ import annotations

from collections import deque
import hashlib
import numpy as np

HANDOVER_TASK = 'simple/G1WholebodyHandoverTeleop-v0'
MOBILE_TASK = 'simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0'
PROFILES = {
    'handover': (HANDOVER_TASK, 'g1_wholebody_handover_teleop'),
    'mobile_pp': (MOBILE_TASK, 'g1_wholebody_locomotion_pick_between_tables_teleop'),
}


def task_identity(profile):
    if profile not in PROFILES:
        raise ValueError('Unknown explicit collection task: ' + str(profile))
    return PROFILES[profile]


def require_mobile_transition(frame, root_velocity6, *, recording_warmup=False):
    """A mode switch requires neutral sticks and an actually stopped robot."""
    if recording_warmup:
        raise ValueError('请先完成一秒起始录制，再切换移动模式')
    names = ('left_axis_x', 'left_axis_y', 'right_axis_x', 'right_axis_y')
    if any(name not in frame.inputs or type(frame.inputs[name]) not in (int,float) for name in names):
        raise ValueError('缺少实际摇杆输入，不能切换模式')
    axes = np.asarray([frame.inputs[name] for name in names], dtype=float)
    velocity = np.asarray(root_velocity6, dtype=float)
    if axes.shape != (4,) or not np.isfinite(axes).all() or (np.abs(axes) > .15).any():
        raise ValueError('切换模式前请松开两个摇杆')
    if velocity.shape != (6,) or not np.isfinite(velocity).all():
        raise ValueError('缺少实际机器人速度，不能切换模式')
    if np.linalg.norm(velocity[:2]) > .05 or np.linalg.norm(velocity[3:]) > .15:
        raise ValueError('机器人仍在移动或转动，请松杆站稳后再切换')
    return {'sticks_neutral': True, 'actual_xy_speed_m_s': float(np.linalg.norm(velocity[:2])),
        'actual_angular_speed_rad_s': float(np.linalg.norm(velocity[3:])),
        'scope': 'mode transition only; does not certify navigation or task success'}


def require_manipulation_pose_match(entry_smpl24, current_smpl24, *, maximum_distance_m=.03):
    """Require the original root-local human pose before restoring mode2.

    This conservative switch guard does not certify decoder continuity or a
    physical grasp. Neither reference is used as a replacement observation.
    """
    first, current = np.asarray(entry_smpl24, dtype=float), np.asarray(current_smpl24, dtype=float)
    if (first.shape != (24, 3) or current.shape != (24, 3)
            or not np.isfinite(np.r_[first.ravel(), current.ravel()]).all()):
        raise ValueError('缺少进入移动时的实际人体姿态，不能恢复手臂操作')
    distances = np.linalg.norm(current-first, axis=1)
    maximum = float(distances.max())
    if maximum > maximum_distance_m:
        raise ValueError('请恢复切入移动时的身体和手臂姿态后再按B（最大偏差 %.1f 厘米）' % (maximum*100))
    return {'maximum_root_local_smpl_joint_distance_m': maximum,
        'maximum_allowed_distance_m': maximum_distance_m,
        'maximum_distance_smpl_joint_index': int(distances.argmax()),
        'joint_distances_m': distances.tolist(),
        'entry_smpl24_sha256': hashlib.sha256(first.astype('<f8').tobytes()).hexdigest(),
        'return_smpl24_sha256': hashlib.sha256(current.astype('<f8').tobytes()).hexdigest(),
        'scope': 'human pose return guard; not decoder or grasp certification'}


def collection_buttons(buttons, previous, *, mobile=False):
    mapping = [('X', 'calibrate'), ('Y', 'toggle_follow'), ('A', 'toggle_record')]
    if mobile:
        mapping.append(('B', 'toggle_navigation'))
    return [action for key, action in mapping
        if buttons.get(key, False) and not previous.get(key, False)]


class MobilePreparationReference:
    """Hold the measured entry pose while the asynchronous planner prepares.

    This is an explicit static desired reference, never a human observation or
    a planner forecast. Live finger targets remain independent in the collector.
    """
    def __init__(self, root_pose7, body_q29, *, source_sequence, device_timestamp_ns):
        root = np.asarray(root_pose7, dtype=float)
        body = np.asarray(body_q29, dtype=float)
        if (root.shape != (7,) or body.shape != (29,)
                or not np.isfinite(np.r_[root, body]).all()
                or abs(np.linalg.norm(root[3:])-1.) > 1e-6):
            raise ValueError('Preparation hold requires the actual root and BODY29')
        self.qpos = np.repeat(np.r_[root, body][None], 46, axis=0)
        self.dq = np.zeros((46, 29), dtype=float)
        self.source_sequence = int(source_sequence)
        self.device_timestamp_ns = int(device_timestamp_ns)

    def encoder_reference(self, actual_root_pose7):
        from pico_locomotion_reference import encoder_reference
        value, evidence = encoder_reference(self.qpos, self.dq, actual_root_pose7)
        evidence.update(producer='measured_mobile_entry_pose_static_preparation_hold',
            complete46_future=False, static_desired_reference=True,
            planner_inference_performed=False,
            held_body_q29=self.qpos[0, 7:].tolist(),
            pose_source_sequence=self.source_sequence,
            pose_source_device_timestamp_ns=self.device_timestamp_ns,
            current_human_pose_used=False)
        return value, evidence


class MobileStoppedWindow:
    """Require 26 consecutive 50Hz endpoints spanning a full 0.500 seconds."""
    def __init__(self):
        self.rows = deque(maxlen=26)
        self.last_tick = None

    def clear(self):
        self.rows.clear()
        self.last_tick = None

    def observe(self, tick50, frame, velocity6, *, tilt_deg, tracking_valid):
        if self.last_tick is not None and tick50 != self.last_tick+1:
            self.clear()
        self.last_tick = tick50
        try:
            require_mobile_transition(frame,velocity6)
            passed = (tracking_valid is True and np.isfinite(tilt_deg) and tilt_deg <= 15.)
        except (ValueError,TypeError):
            passed = False
        self.rows.append(bool(passed))

    @property
    def ready(self):
        return len(self.rows) == 26 and all(self.rows)
