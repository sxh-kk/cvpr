"""Opt-in CPU PICO pelvis motion -> SONIC v1.1 planner intent.

This is a local control mapping, not the upstream joystick implementation. It
never opens an SDK subscriber, runs a model, changes a simulator, or creates a
SONIC reference/token. A caller must pass its existing subscriber's same-packet
PicoFrame and its current actual robot root pose. Production collection does not
import this module. Physical planner/decoder walking still needs separate tests.

Body[0] is the SDK's estimated pelvis pose, not an independently measured waist
tracker. Its room displacement is differentiated using device time; SDK va/wva
and host receive intervals are never velocity labels. Valid stopping obeys the
acceleration limit. Invalid/repeated input and pause stop immediately instead.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
from numbers import Real
import time

import numpy as np
from scipy.spatial.transform import Rotation

from pico_live_input import PicoFrame, validate_frame


CONTRACT = "cvpr-pico-pelvis-planner-intent-experimental-v1"
IDLE, SLOW_WALK = 0, 1
MIN_SLOW_WALK_SPEED_M_S = .1
_Y_UP_TO_Z_UP = Rotation.from_euler("x", math.pi / 2)
_PICO_BODY_BASIS = Rotation.from_euler("y", math.pi)
_SMPL_BASE_INVERSE = Rotation.from_quat([.5, .5, .5, .5]).inv()


def _finite_scalar(value, name, *, positive=True):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    if positive and value <= 0:
        raise ValueError(name + " must be positive")
    return float(value)


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _device_identity(frame):
    if type(frame.sequence) is not int or frame.sequence < 0:
        raise ValueError("Invalid source sequence")
    if (type(frame.device_timestamp_ns) is not int or
            not 0 < frame.device_timestamp_ns < 2**63):
        raise ValueError("Device timestamp must be positive signed-int64 nanoseconds")


def _heading(rotation):
    forward = rotation.as_matrix()[:2, 0]
    if np.linalg.norm(forward) < .2:
        raise ValueError("Root heading is undefined near a vertical forward axis")
    return math.atan2(float(forward[1]), float(forward[0]))


def _robot_heading(actual_root_pose7):
    root = np.asarray(actual_root_pose7, dtype=np.float64)
    if root.shape != (7,) or not np.isfinite(root).all():
        raise ValueError("Actual robot root must be finite xyz+wxyz")
    if abs(np.linalg.norm(root[3:]) - 1.) > 1e-4:
        raise ValueError("Actual robot root quaternion must be unit wxyz")
    return _heading(Rotation.from_quat(root[[4, 5, 6, 3]]))


def pico_pelvis_zup(body24x7):
    """Return room position, heading and tilt using the existing SMPL root basis.

    Rotation is Rx(+90) * Q_pico * Ry(180) * inverse(SMPL-base). Translation
    uses only the coordinate basis Rx(+90): [x,y,z] -> [x,-z,y]. The local
    skeleton basis correction is not a rotation of the room displacement.
    """
    body = np.asarray(body24x7, dtype=np.float64)
    if body.shape != (24, 7) or not np.isfinite(body).all():
        raise ValueError("Require one finite PICO Body[24,xyz+xyzw] packet")
    if np.max(np.linalg.norm(body[:, :3], axis=1)) > 10.:
        raise ValueError("PICO position exceeds the existing room-scale metre contract")
    norms = np.linalg.norm(body[:, 3:], axis=1)
    if np.any(np.abs(norms - 1.) > .01):
        raise ValueError("PICO body quaternions must be unit xyzw")
    orientation = (_Y_UP_TO_Z_UP * Rotation.from_quat(body[0, 3:]) *
                   _PICO_BODY_BASIS * _SMPL_BASE_INVERSE)
    tilt = math.acos(float(np.clip(orientation.as_matrix()[2, 2], -1., 1.)))
    return _Y_UP_TO_Z_UP.apply(body[0, :3]), _heading(orientation), tilt


@dataclass(frozen=True)
class LocomotionIntentConfig:
    max_age_s: float = .20
    minimum_device_dt_s: float = .001
    speed_deadzone_m_s: float = .05
    max_human_root_speed_m_s: float = 2.5
    max_root_step_m: float = .20
    max_robot_speed_m_s: float = .4
    max_robot_acceleration_m_s2: float = .5
    velocity_filter_tau_s: float = .06
    max_facing_rate_rad_s: float = .8
    calibration_max_tilt_deg: float = 15.
    tracking_max_tilt_deg: float = 75.

    def __post_init__(self):
        for name, value in asdict(self).items():
            scalar = _finite_scalar(value, name, positive=name != "velocity_filter_tau_s")
            object.__setattr__(self, name, scalar)
        if self.velocity_filter_tau_s < 0:
            raise ValueError("velocity_filter_tau_s cannot be negative")
        if self.minimum_device_dt_s >= self.max_age_s:
            raise ValueError("Device time limits are inconsistent")
        if self.speed_deadzone_m_s >= self.max_human_root_speed_m_s:
            raise ValueError("Deadzone must be below the accepted human speed")
        if self.max_robot_speed_m_s < MIN_SLOW_WALK_SPEED_M_S:
            raise ValueError("Robot speed limit must cover the declared SLOW_WALK range")
        if not 0 < self.calibration_max_tilt_deg <= self.tracking_max_tilt_deg < 90:
            raise ValueError("Require 0 < calibration tilt <= tracking tilt < 90 degrees")


@dataclass(frozen=True)
class PlannerIntent:
    mode: int
    movement_world: tuple[float, float, float]
    facing_world: tuple[float, float, float]
    speed_m_s: float
    height_m: float
    valid: bool
    fresh: bool
    calibration_required: bool
    source_sequence: int | None
    device_timestamp_ns: int | None
    evidence: dict

    def planner_fields(self):
        """Canonical planner inputs, without transport or generating a reference."""
        return {"mode": self.mode, "movement_direction": list(self.movement_world),
            "facing_direction": list(self.facing_world),
            "speed": self.speed_m_s if self.mode == SLOW_WALK else -1.,
            "height": self.height_m}

    def as_dict(self):
        return asdict(self)


class PicoRootPlannerIntent:
    """Stateful causal estimator; use it only in a separate opt-in caller.

    Movement is velocity following, not an absolute position servo. Speed and
    acceleration limits mean robot displacement need not equal human room
    displacement. The internal continuous velocity obeys its acceleration bound.
    Speeds below the declared 0.1m/s SLOW_WALK range emit IDLE. Crossing that
    boundary is a discrete mode change, not a strictly continuous planner speed.
    CPU tests cannot establish the physical behavior of these transitions.
    """
    def __init__(self, config=None):
        self.config = config if config is not None else LocomotionIntentConfig()
        if not isinstance(self.config, LocomotionIntentConfig):
            raise TypeError("config must be LocomotionIntentConfig")
        self._clear()

    def _clear(self):
        self._calibration = None
        self._previous_frame = None
        self._previous_position = None
        self._filtered_velocity = np.zeros(2)
        self._velocity = np.zeros(2)
        self._facing_yaw = None

    @property
    def calibrated(self):
        return self._calibration is not None

    @property
    def calibration(self):
        return copy.deepcopy(self._calibration)

    def _idle(self, reason, actual_root_pose7=None, *, frame=None,
              invalidate=True, fresh=False, quality=None):
        if invalidate:
            self._clear()
        else:
            # A repeated packet cannot maintain the previous walking velocity.
            self._velocity[:] = 0.
            self._filtered_velocity[:] = 0.
        try:
            yaw = _robot_heading(actual_root_pose7)
            facing_source = "current_actual_robot_root"
        except (ValueError, TypeError, OverflowError):
            yaw = 0.
            facing_source = "deterministic_idle_fallback_no_reference"
        self._facing_yaw = yaw if self.calibrated else None
        sequence, timestamp = getattr(frame, "sequence", None), getattr(frame, "device_timestamp_ns", None)
        sequence = sequence if type(sequence) is int and sequence >= 0 else None
        timestamp = timestamp if type(timestamp) is int and timestamp > 0 else None
        return PlannerIntent(IDLE, (0., 0., 0.), (math.cos(yaw), math.sin(yaw), 0.),
            0., -1., False, fresh, not self.calibrated,
            sequence, timestamp,
            {"contract": CONTRACT, "reason": str(reason), "tracking": quality,
             "facing_source": facing_source, "planner_reference_reuse_allowed": False,
             "sdk_va_wva_used": False, "safety_stop_bypasses_acceleration_limit": True})

    def pause(self, actual_root_pose7=None, *, reason="operator_pause"):
        """Stop immediately and require a new calibration before any movement."""
        return self._idle(reason, actual_root_pose7)

    def calibrate(self, frame, actual_root_pose7, *, now_s=None):
        """Bind human heading to actual robot heading without changing either pose."""
        self._clear()  # A rejected recalibration cannot leave an old mapping active.
        now = time.monotonic() if now_s is None else _finite_scalar(now_s, "now_s", positive=False)
        if not isinstance(frame, PicoFrame):
            raise ValueError("Calibration needs one PicoFrame from the existing subscriber")
        _device_identity(frame)
        quality = validate_frame(frame, now=now, max_age_s=self.config.max_age_s)
        if not quality["recordable"]:
            raise ValueError("Calibration tracking rejected: " + ",".join(quality["reasons"]))
        position, human_yaw, tilt = pico_pelvis_zup(frame.body24x7)
        if math.degrees(tilt) > self.config.calibration_max_tilt_deg:
            raise ValueError("Calibration requires an upright pelvis reference")
        robot_yaw = _robot_heading(actual_root_pose7)
        alignment = _wrap(robot_yaw - human_yaw)
        self._calibration = {"contract": CONTRACT, "control_policy": "local_pelvis_velocity_to_planner",
            "human_root_source": "same-packet SDK Body[0] estimated pelvis, not an asserted waist tracker",
            "human_initial_zup_position_m": position.tolist(), "human_initial_heading_rad": human_yaw,
            "robot_initial_heading_rad": robot_yaw, "planar_alignment_yaw_rad": alignment,
            "source_sequence": frame.sequence, "device_timestamp_ns": frame.device_timestamp_ns,
            "position_basis": "Rx(+pi/2): [x,y,z] -> [x,-z,y]",
            "orientation_basis": "Rx(pi/2)*Q_pico*Ry(pi)*inverse([.5,.5,.5,.5] SMPL base)",
            "config": asdict(self.config), "absolute_position_servo": False,
            "independent_ankle_tracker_health_verified": False,
            "sdk_va_wva_used": False, "planner_reference_generated": False,
            "physical_walking_validated": False}
        self._previous_frame, self._previous_position = frame, position.copy()
        self._facing_yaw = robot_yaw
        return self.calibration

    def update(self, frame, actual_root_pose7, *, now_s=None):
        """Consume a new device sample or return a fresh, explicit zero IDLE intent."""
        if not self.calibrated:
            return self._idle("calibration_required", actual_root_pose7, frame=frame)
        try:
            now = time.monotonic() if now_s is None else _finite_scalar(now_s, "now_s", positive=False)
            _robot_heading(actual_root_pose7)
            if not isinstance(frame, PicoFrame):
                raise ValueError("missing_or_invalid_pico_frame")
            _device_identity(frame)
            quality = validate_frame(frame, now=now, previous=self._previous_frame,
                max_age_s=self.config.max_age_s)
            if not quality["valid"]:
                return self._idle("tracking_rejected", actual_root_pose7, frame=frame, quality=quality)
            if not quality["recordable"]:
                return self._idle("no_new_device_sample", actual_root_pose7,
                    frame=frame, invalidate=False, quality=quality)
            position, human_yaw, tilt = pico_pelvis_zup(frame.body24x7)
            if math.degrees(tilt) > self.config.tracking_max_tilt_deg:
                raise ValueError("pelvis_tilt_outside_tracking_limit")
            dt = (frame.device_timestamp_ns - self._previous_frame.device_timestamp_ns) / 1e9
            if not self.config.minimum_device_dt_s <= dt <= self.config.max_age_s:
                raise ValueError("device_interval_outside_limits")
            displacement = position - self._previous_position
            if np.linalg.norm(displacement) > self.config.max_root_step_m:
                raise ValueError("pelvis_position_jump")
            raw_velocity = displacement / dt
            if np.linalg.norm(raw_velocity) > self.config.max_human_root_speed_m_s:
                raise ValueError("pelvis_velocity_jump")
            angle = self._calibration["planar_alignment_yaw_rad"]
            rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                 [math.sin(angle), math.cos(angle)]])
            world_velocity = rotation @ raw_velocity[:2]
            target = world_velocity if np.linalg.norm(world_velocity) > self.config.speed_deadzone_m_s else np.zeros(2)
            tau = self.config.velocity_filter_tau_s
            alpha = 1. if tau == 0 else -math.expm1(-dt / tau)
            filtered = self._filtered_velocity + alpha * (target - self._filtered_velocity)
            norm = np.linalg.norm(filtered)
            desired = filtered * min(1., self.config.max_robot_speed_m_s / max(norm, 1e-12))
            change = desired - self._velocity
            max_change = self.config.max_robot_acceleration_m_s2 * dt
            limited = self._velocity + change * min(1., max_change / max(np.linalg.norm(change), 1e-12))
            speed = float(np.linalg.norm(limited))
            moving = speed >= MIN_SLOW_WALK_SPEED_M_S
            movement = limited / speed if moving else np.zeros(2)
            target_yaw = _wrap(human_yaw + angle)
            turn = float(np.clip(_wrap(target_yaw - self._facing_yaw),
                -self.config.max_facing_rate_rad_s * dt, self.config.max_facing_rate_rad_s * dt))
            facing_yaw = _wrap(self._facing_yaw + turn)
            evidence = {"contract": CONTRACT, "control_policy": "local_pelvis_velocity_to_planner",
                "device_dt_s": dt, "host_age_s": quality["age_s"],
                "pelvis_position_zup_m": position.tolist(), "pelvis_step_zup_m": displacement.tolist(),
                "raw_pelvis_velocity_zup_m_s": raw_velocity.tolist(),
                "raw_planar_velocity_world_m_s": world_velocity.tolist(),
                "filtered_planar_velocity_world_m_s": filtered.tolist(),
                "limited_planar_velocity_world_m_s": limited.tolist(),
                "command_delta_velocity_m_s": (limited - self._velocity).tolist(),
                "maximum_delta_velocity_m_s": max_change, "facing_rate_rad_s": turn / dt,
                "planner_declared_slow_walk_minimum_m_s": MIN_SLOW_WALK_SPEED_M_S,
                "low_speed_ramp_idle": bool(0 < speed < MIN_SLOW_WALK_SPEED_M_S),
                "acceleration_limit_scope": "internal continuous velocity; planner has a discrete IDLE/SLOW_WALK minimum-speed transition",
                "sdk_va_wva_used": False, "host_interval_used_for_velocity": False,
                "planner_reference_reuse_allowed": False, "planner_reference_generated": False,
                "physical_walking_validated": False}
            self._previous_frame, self._previous_position = frame, position.copy()
            self._filtered_velocity, self._velocity, self._facing_yaw = filtered, limited, facing_yaw
            return PlannerIntent(SLOW_WALK if moving else IDLE,
                (float(movement[0]), float(movement[1]), 0.),
                (math.cos(facing_yaw), math.sin(facing_yaw), 0.), speed if moving else 0.,
                -1., True, True, False, frame.sequence, frame.device_timestamp_ns, evidence)
        except (ValueError, TypeError, OverflowError) as error:
            return self._idle(str(error), actual_root_pose7, frame=frame)
