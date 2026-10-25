"""CPU-only human-room-motion intent tests; no SDK, model, or simulation."""
from dataclasses import replace
import importlib
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pico_locomotion_intent as intent
from pico_live_input import parse_tracking


def robot_root(yaw=0.):
    q = Rotation.from_euler("z", yaw).as_quat()
    return np.r_[[0., 0., .78], q[[3, 0, 1, 2]]]


def packet(sequence, timestamp_s, *, raw_translation=(0., 0., 0.), yaw=0., host_time=None):
    position = np.asarray(raw_translation, dtype=float)
    root = np.array([0., 1., 0.]) + position
    q = Rotation.from_euler("y", yaw).as_quat()
    # Keep a non-collapsed complete source skeleton, with both controller poses.
    body = np.tile(np.r_[root, q], (24, 1))
    body[:, :3] += np.array([[.03 * (i % 3), -.04 * (i // 3), .015 * (i % 2)] for i in range(24)])
    controller = lambda offset: {"pose": np.r_[root + offset, q].tolist(),
        "trigger": 0., "grip": 0., "axisX": 0., "axisY": 0.,
        "menuButton": False, "axisClick": False,
        "primaryButton": False, "secondaryButton": False}
    value = {"timeStampNs": round(timestamp_s * 1e9), "Input": 1,
        "appState": {"focus": True}, "Head": {"status": 3, "pose": np.r_[root + [0., .7, 0.], q].tolist()},
        "Controller": {"left": controller([-.3, .3, -.2]), "right": controller([.3, .3, -.2])},
        "Body": {"joints": [{"p": pose.tolist(), "va": [3e27] * 6} for pose in body]}}
    return parse_tracking({"functionName": "Tracking", "value": json.dumps(value)},
        timestamp_s if host_time is None else host_time, sequence)


class PelvisIntentTests(unittest.TestCase):
    def controller(self, **kwargs):
        return intent.PicoRootPlannerIntent(intent.LocomotionIntentConfig(**kwargs))

    def assert_idle(self, command, *, recalibrate=True):
        self.assertEqual(command.mode, intent.IDLE)
        self.assertEqual(command.movement_world, (0., 0., 0.))
        self.assertEqual(command.speed_m_s, 0.)
        self.assertFalse(command.valid)
        self.assertFalse(command.evidence["planner_reference_reuse_allowed"])
        self.assertEqual(command.calibration_required, recalibrate)
        json.dumps(command.as_dict(), allow_nan=False)

    def test_import_and_calls_cannot_initialize_sdk_or_models(self):
        with patch("ctypes.CDLL", side_effect=AssertionError("No SDK")), \
             patch("subprocess.Popen", side_effect=AssertionError("No services")):
            importlib.reload(intent)
            adapter = self.controller()
            adapter.calibrate(packet(1, 1.), robot_root(), now_s=1.)
            adapter.update(packet(2, 1.02, raw_translation=[0., 0., -.002]), robot_root(), now_s=1.02)

    def test_default_basis_maps_raw_forward_to_robot_heading_and_right_to_robot_right(self):
        for robot_yaw, expected in ((0., [1., 0.]), (math.pi / 2, [0., 1.])):
            adapter = self.controller(velocity_filter_tau_s=0., max_robot_acceleration_m_s2=10.)
            adapter.calibrate(packet(1, 1.), robot_root(robot_yaw), now_s=1.)
            command = adapter.update(packet(2, 1.1, raw_translation=[0., 0., -.02]),
                robot_root(robot_yaw), now_s=1.1)
            np.testing.assert_allclose(command.movement_world[:2], expected, atol=1e-12)
            np.testing.assert_allclose(command.facing_world[:2], expected, atol=1e-12)
            self.assertAlmostEqual(command.speed_m_s, .2)
        adapter = self.controller(velocity_filter_tau_s=0., max_robot_acceleration_m_s2=10.)
        adapter.calibrate(packet(1, 1.), robot_root(), now_s=1.)
        command = adapter.update(packet(2, 1.1, raw_translation=[.02, 0., 0.]), robot_root(), now_s=1.1)
        np.testing.assert_allclose(command.movement_world, [0., -1., 0.], atol=1e-12)

    def test_room_origin_cancels_and_vertical_motion_is_not_walking(self):
        origin = np.array([2., .2, -1.])
        adapter = self.controller()
        receipt = adapter.calibrate(packet(1, 1., raw_translation=origin), robot_root(), now_s=1.)
        command = adapter.update(packet(2, 1.1, raw_translation=origin + [0., .02, 0.]),
            robot_root(), now_s=1.1)
        self.assertTrue(command.valid)
        self.assertEqual(command.mode, intent.IDLE)
        self.assertEqual(command.speed_m_s, 0.)
        self.assertFalse(receipt["absolute_position_servo"])
        self.assertFalse(receipt["independent_ankle_tracker_health_verified"])

    def test_device_time_not_tcp_receive_time_sets_velocity(self):
        adapter = self.controller(velocity_filter_tau_s=0., max_robot_acceleration_m_s2=10.)
        adapter.calibrate(packet(1, 10., host_time=20.), robot_root(), now_s=20.)
        # Two sampled poses arrived together: physical delta is 100ms, not 1ms.
        command = adapter.update(packet(2, 10.1, host_time=20.001, raw_translation=[0., 0., -.02]),
            robot_root(), now_s=20.001)
        self.assertTrue(command.valid)
        self.assertAlmostEqual(command.speed_m_s, .2)
        self.assertAlmostEqual(command.evidence["device_dt_s"], .1)
        self.assertFalse(command.evidence["sdk_va_wva_used"])
        self.assertFalse(command.evidence["host_interval_used_for_velocity"])

    def test_vector_speed_and_acceleration_limits_cover_direction_reversal(self):
        adapter = self.controller(velocity_filter_tau_s=0.)
        adapter.calibrate(packet(1, 1.), robot_root(), now_s=1.)
        position = np.zeros(3)
        old_velocity = np.zeros(2)
        for i in range(1, 101):
            dt = (.01, .025, .02)[i % 3]
            current_time = 1. + sum((.01, .025, .02)[j % 3] for j in range(1, i + 1))
            position += np.array([0., 0., -1. if i < 55 else 1.]) * dt
            command = adapter.update(packet(i + 1, current_time, raw_translation=position),
                robot_root(), now_s=current_time)
            self.assertTrue(command.valid, command.evidence)
            current_velocity = np.asarray(command.evidence['limited_planar_velocity_world_m_s'])
            self.assertLessEqual(command.speed_m_s, .4 + 1e-12)
            self.assertLessEqual(np.linalg.norm(current_velocity - old_velocity), .5 * dt + 1e-10)
            old_velocity = current_velocity
        self.assertLess(old_velocity[0], 0.)

    def test_deadzone_suppresses_position_jitter_without_model_references(self):
        adapter = self.controller()
        adapter.calibrate(packet(1, 1.), robot_root(), now_s=1.)
        command = adapter.update(packet(2, 1.02, raw_translation=[.0001, 0., -.0002]),
            robot_root(), now_s=1.02)
        self.assertEqual(command.mode, intent.IDLE)
        self.assertEqual(command.speed_m_s, 0.)
        self.assertFalse(command.evidence["planner_reference_generated"])

    def test_repeated_sample_zeroes_command_and_cannot_hold_old_movement(self):
        adapter = self.controller(velocity_filter_tau_s=0.)
        adapter.calibrate(packet(1, 1.), robot_root(), now_s=1.)
        frame = packet(2, 1.02, raw_translation=[0., 0., -.01])
        first = adapter.update(frame, robot_root(), now_s=1.02)
        self.assertGreater(np.linalg.norm(first.evidence['limited_planar_velocity_world_m_s']), 0.)
        self.assert_idle(adapter.update(frame, robot_root(), now_s=1.03), recalibrate=False)
        resumed = adapter.update(packet(3, 1.04, raw_translation=[0., 0., -.02]), robot_root(), now_s=1.04)
        self.assertLessEqual(np.linalg.norm(resumed.evidence['limited_planar_velocity_world_m_s']),
            .5 * .02 + 1e-12)

    def test_stale_pause_and_missing_tracking_stop_immediately_and_require_calibration(self):
        for failure in ("stale", "pause", "missing"):
            with self.subTest(failure=failure):
                adapter = self.controller()
                adapter.calibrate(packet(1, 1.), robot_root(), now_s=1.)
                frame = packet(2, 1.02, raw_translation=[0., 0., -.01])
                adapter.update(frame, robot_root(), now_s=1.02)
                if failure == "pause":
                    command = adapter.pause(robot_root(.5))
                else:
                    command = adapter.update(None if failure == "missing" else frame,
                        robot_root(.5), now_s=1.3)
                self.assert_idle(command)
                np.testing.assert_allclose(command.facing_world[:2], [math.cos(.5), math.sin(.5)])
                self.assert_idle(adapter.update(packet(3, 1.32), robot_root(), now_s=1.32))

    def test_clock_regressions_gaps_input_modes_and_relocalization_jumps_fail_closed(self):
        baseline = packet(1, 1.)
        variants = [packet(2, .99), packet(2, 1.), packet(2, 1.3),
            packet(2, 1.02, raw_translation=[.3, 0., 0.]),
            packet(2, 1.1, raw_translation=[.21, 0., 0.]),
            replace(packet(2, 1.02), input_mode=2),
            replace(packet(2, 1.02), sequence=0),
            replace(packet(2, 1.02), body24x7=None),
            replace(packet(2, 1.02), app_focused=False)]
        for invalid in variants:
            with self.subTest(frame=invalid.sequence, timestamp=invalid.device_timestamp_ns):
                adapter = self.controller()
                adapter.calibrate(baseline, robot_root(), now_s=1.)
                self.assert_idle(adapter.update(invalid, robot_root(), now_s=invalid.host_received_s))

    def test_nonfinite_body_and_bad_actual_robot_pose_cannot_emit_movement(self):
        for bad in (float("nan"), float("inf")):
            for component in ("human", "robot", "now"):
                adapter = self.controller()
                adapter.calibrate(packet(1, 1.), robot_root(), now_s=1.)
                frame, actual, now = packet(2, 1.02), robot_root(), 1.02
                if component == "human":
                    body = frame.body24x7.copy()
                    body[0, 0] = bad
                    frame = replace(frame, body24x7=body)
                elif component == "robot":
                    actual[0] = bad
                else:
                    now = bad
                self.assert_idle(adapter.update(frame, actual, now_s=now))

    def test_turning_is_heading_calibrated_wrapped_and_rate_limited(self):
        adapter = self.controller(velocity_filter_tau_s=0.)
        initial = packet(1, 1., yaw=math.pi - .01)
        adapter.calibrate(initial, robot_root(math.pi - .02), now_s=1.)
        command = adapter.update(packet(2, 1.1, yaw=-math.pi + .02),
            robot_root(math.pi - .02), now_s=1.1)
        self.assertTrue(command.valid)
        self.assertAlmostEqual(command.evidence["facing_rate_rad_s"], .3)
        limited = adapter.update(packet(3, 1.2, yaw=-math.pi + .4), robot_root(), now_s=1.2)
        self.assertTrue(limited.valid)
        self.assertAlmostEqual(limited.evidence["facing_rate_rad_s"], .8)

    def test_quaternion_sign_and_invalid_device_identity_cannot_change_or_reuse_heading(self):
        adapter = self.controller()
        adapter.calibrate(packet(1, 1.), robot_root(), now_s=1.)
        frame = packet(2, 1.02)
        body = frame.body24x7.copy()
        body[:, 3:] *= -1
        command = adapter.update(replace(frame, body24x7=body), robot_root(), now_s=1.02)
        self.assertTrue(command.valid)
        np.testing.assert_allclose(command.facing_world, [1., 0., 0.], atol=1e-12)
        self.assert_idle(adapter.update(replace(packet(3, 1.04), device_timestamp_ns=float('nan')),
            robot_root(), now_s=1.04))

    def test_input_and_calibration_receipts_do_not_share_mutable_state(self):
        adapter = self.controller()
        frame = packet(1, 1.)
        original = frame.body24x7.copy()
        receipt = adapter.calibrate(frame, robot_root(), now_s=1.)
        receipt["planar_alignment_yaw_rad"] = 100.
        receipt["config"]["max_robot_speed_m_s"] = 100.
        self.assertNotEqual(adapter.calibration["planar_alignment_yaw_rad"], 100.)
        self.assertEqual(adapter.calibration["config"]["max_robot_speed_m_s"], .4)
        np.testing.assert_array_equal(frame.body24x7, original)

    def test_failed_recalibration_clears_old_mapping_and_config_is_checked(self):
        adapter = self.controller()
        adapter.calibrate(packet(1, 1.), robot_root(), now_s=1.)
        with self.assertRaises(ValueError):
            adapter.calibrate(replace(packet(2, 1.02), head_status=0), robot_root(), now_s=1.02)
        self.assertFalse(adapter.calibrated)
        for kwargs in ({"max_robot_speed_m_s": float("nan")},
                       {"max_robot_acceleration_m_s2": 0.},
                       {"max_age_s": True}, {"velocity_filter_tau_s": -1.}):
            with self.assertRaises(ValueError):
                intent.LocomotionIntentConfig(**kwargs)

    def test_planner_fields_have_exact_named_contract_and_no_token_or_transport(self):
        adapter = self.controller()
        adapter.calibrate(packet(1, 1.), robot_root(), now_s=1.)
        for i in range(1, 16):
            command = adapter.update(packet(i + 1, 1. + .1 * i, raw_translation=[0., 0., -.02 * i]),
                robot_root(), now_s=1. + .1 * i)
        fields = command.planner_fields()
        self.assertEqual(set(fields), {"mode", "movement_direction", "facing_direction", "speed", "height"})
        self.assertEqual(fields["mode"], intent.SLOW_WALK)
        self.assertEqual(fields["height"], -1.)
        self.assertAlmostEqual(np.linalg.norm(fields["movement_direction"]), 1.)
        self.assertAlmostEqual(np.linalg.norm(fields["facing_direction"]), 1.)
        json.dumps(command.as_dict(), allow_nan=False)

    def test_below_official_minimum_speed_is_idle_while_internal_velocity_ramps(self):
        adapter = self.controller(velocity_filter_tau_s=0.)
        adapter.calibrate(packet(1, 1.), robot_root(), now_s=1.)
        for i in range(1, 9):
            command = adapter.update(packet(i + 1, 1. + .02 * i, raw_translation=[0., 0., -.01 * i]),
                robot_root(), now_s=1. + .02 * i)
            self.assertEqual(command.mode, intent.IDLE)
            self.assertEqual(command.speed_m_s, 0.)
            self.assertEqual(command.planner_fields()['speed'], -1.)
            self.assertTrue(command.evidence['low_speed_ramp_idle'])
            self.assertGreater(np.linalg.norm(command.evidence['limited_planar_velocity_world_m_s']), 0.)


if __name__ == "__main__":
    unittest.main()
