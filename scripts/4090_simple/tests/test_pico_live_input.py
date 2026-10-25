"""Pure packet and fake-SDK tests; no headset/service/simulation is touched."""
from copy import deepcopy
import ctypes as C
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pico_live_input as pico


def payload():
    """XRoboToolkit 1.1.1 shape, including stringified outer value and poses."""
    value = {
        "appState": {"focus": True}, "Input": 1, "timeStampNs": 1792909969183632128,
        "Head": {"pose": "0.1,1.6,0.2,0,0,0,1", "status": 3},
        "Controller": {
            "left": {"pose": "-0.3,1.2,0.4,0,0,0,1", "trigger": .7, "grip": .3,
                     "axisX": 0., "axisY": -.2, "primaryButton": True,
                     "secondaryButton": False, "menuButton": False, "axisClick": False},
            "right": {"pose": "0.3,1.2,0.4,0,0,0,1", "trigger": .6, "grip": .2,
                      "axisX": .1, "axisY": 0., "primaryButton": False,
                      "secondaryButton": True, "menuButton": False, "axisClick": False}},
        "Body": {"joints": [{"p": f"{i*.02},1.0,.1,0,0,0,1", "t": 0,
                              "va": "0,0,0,-3.29E+27,0,0", "wva": "bad-sdk-evidence"}
                             for i in range(24)]}}
    return {"functionName": "Tracking", "value": json.dumps(value)}


def edit(outer, mutate):
    new = deepcopy(outer)
    value = json.loads(new["value"])
    mutate(value)
    new["value"] = json.dumps(value)
    return new


class PicoTrackingTests(unittest.TestCase):
    def test_real_structure_current_analog_buttons_and_immutable_evidence(self):
        outer = payload()
        frame = pico.parse_tracking(outer, 10., 1)
        self.assertEqual(frame.body24x7.shape, (24, 7))
        self.assertEqual(frame.inputs["left_trigger"], .7)
        self.assertEqual(frame.inputs["right_grip"], .2)
        self.assertTrue(frame.buttons["X"])
        self.assertTrue(frame.buttons["B"])
        self.assertFalse(frame.buttons["A"])
        for array in (frame.head_pose7, frame.left_pose7, frame.right_pose7, frame.body24x7):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        with self.assertRaises(TypeError):
            frame.inputs["left_trigger"] = 0.
        copied = frame.raw_payload
        copied["value"] = "changed"
        self.assertEqual(frame.raw_payload, outer)
        result = pico.validate_frame(frame, now=10.1)
        self.assertTrue(result["valid"])
        self.assertTrue(result["recordable"])
        self.assertIn("sdk_va_wva_not_used", result["warnings"])

    def test_absent_sections_do_not_reuse_previous_tracking(self):
        first = pico.parse_tracking(payload(), 10., 1)
        for key, expected in (("Body", "body"), ("Head", "head"),
                              ("Controller", "left_controller")):
            outer = edit(payload(), lambda v: (v.pop(key), v.__setitem__("timeStampNs", v["timeStampNs"]+10000000)))
            frame = pico.parse_tracking(outer, 10.01, 2)
            result = pico.validate_frame(frame, now=10.02, previous=first)
            self.assertFalse(result["valid"])
            self.assertIn(f"{expected}_missing_shape_or_nonfinite", result["reasons"])
        self.assertIsNone(pico.parse_tracking(edit(payload(), lambda v: v.pop("Body")), 10.).body24x7)

    def test_stale_focus_mode_and_invalid_head_status(self):
        frame = pico.parse_tracking(payload(), 10., 1)
        result = pico.validate_frame(frame, now=10.21)
        self.assertIn("stale", result["reasons"])
        self.assertFalse(result["recordable"])
        for mutate, reason in ((lambda v: v["appState"].update(focus=False), "app_not_focused"),
                               (lambda v: v.update(Input=2), "controller_input_required"),
                               (lambda v: v["Head"].update(status=0), "head_tracking_invalid")):
            second = pico.parse_tracking(edit(payload(), mutate), 10.01, 2)
            self.assertIn(reason, pico.validate_frame(second, now=10.02, previous=frame)["reasons"])
        switched = pico.parse_tracking(edit(payload(), lambda v: v.update(Input=2)), 10.01, 2)
        returned = pico.parse_tracking(payload(), 10.02, 3)
        self.assertIn("input_mode_changed", pico.validate_frame(returned, now=10.03, previous=switched)["reasons"])

    def test_quaternion_placeholder_and_body_shape_are_rejected(self):
        for mutate, reason in (
            (lambda v: v["Controller"]["left"].update(pose="-.3,1,.2,0,0,0,2"), "left_controller_quaternion_not_unit"),
            (lambda v: v["Controller"]["left"].update(pose="0,0,0,0,0,0,1"), "left_controller_zero_placeholder"),
            (lambda v: v["Body"]["joints"].pop(), "body_missing_shape_or_nonfinite"),
            (lambda v: v["Body"]["joints"][0].update(p=".1,1,.2,0,0,0,0"), "body_quaternion_not_unit"),
            (lambda v: v["Controller"]["right"].update(trigger=1.1), "right_trigger_outside_range")):
            frame = pico.parse_tracking(edit(payload(), mutate), 10.)
            self.assertIn(reason, pico.validate_frame(frame, now=10.01)["reasons"])

    def test_timestamp_holds_repeats_regression_and_new_callback(self):
        first = pico.parse_tracking(payload(), 10., 1)
        hold = pico.validate_frame(first, now=10.05, previous=first)
        self.assertTrue(hold["valid"])
        self.assertFalse(hold["fresh"])
        self.assertFalse(hold["recordable"])
        repeated = pico.parse_tracking(payload(), 10.01, 2)
        self.assertIn("device_timestamp_repeated", pico.validate_frame(repeated, now=10.02, previous=first)["reasons"])
        regressed = pico.parse_tracking(edit(payload(), lambda v: v.update(timeStampNs=1)), 10.01, 2)
        self.assertIn("device_timestamp_regressed", pico.validate_frame(regressed, now=10.02, previous=first)["reasons"])
        new = pico.parse_tracking(edit(payload(), lambda v: v.update(timeStampNs=v["timeStampNs"]+10000000)), 10.01, 2)
        self.assertTrue(pico.validate_frame(new, now=10.02, previous=first)["recordable"])

    def test_pose_jumps_and_clock_leap_cannot_use_broken_velocity(self):
        first = pico.parse_tracking(payload(), 10., 1)
        for target, reason in (("hand", "left_controller_position_jump"), ("body", "body_position_jump")):
            def mutate(v):
                v["timeStampNs"] += 10000000
                if target == "hand":
                    v["Controller"]["left"]["pose"] = "-1.3,1.2,.4,0,0,0,1"
                else:
                    v["Body"]["joints"][0]["p"] = "1,1,.1,0,0,0,1"
            second = pico.parse_tracking(edit(payload(), mutate), 10.01, 2)
            self.assertIn(reason, pico.validate_frame(second, now=10.02, previous=first)["reasons"])
        leap = pico.parse_tracking(edit(payload(), lambda v: v.update(timeStampNs=v["timeStampNs"]+10000000000)), 10.01, 2)
        self.assertIn("device_timestamp_gap", pico.validate_frame(leap, now=10.02, previous=first)["reasons"])

    def test_network_callback_bursts_use_device_sample_interval(self):
        first = pico.parse_tracking(payload(), 10., 1)
        def mutate(v):
            v["timeStampNs"] += 10000000
            v["Controller"]["left"]["pose"] = "-.29,1.2,.4,0,0,0,1"
        second = pico.parse_tracking(edit(payload(), mutate), 10.00005, 2)
        self.assertTrue(pico.validate_frame(second, now=10.00006, previous=first)["valid"])

    def test_raw_dictionary_input_is_copied_and_bad_payloads_raise(self):
        value = json.loads(payload()["value"])
        frame = pico.parse_tracking(value, 10.)
        value["Head"]["pose"] = "mutated"
        self.assertEqual(frame.head_pose7.shape, (7,))
        for outer in ("not-json", {"functionName": "Other"},
                      {"value": []}, {"Head": {"pose": "not-numeric"}},
                      {"timeStampNs": 1.1}):
            with self.assertRaises(ValueError):
                pico.parse_tracking(outer, 10.)

    def test_quaternion_sign_does_not_jump_but_true_rotation_discontinuity_does(self):
        first = pico.parse_tracking(payload(), 10., 1)
        def changed_pose(pose):
            return edit(payload(), lambda v: (v.update(timeStampNs=v["timeStampNs"]+10000000),
                                              v["Controller"]["left"].update(pose=pose)))
        equivalent = pico.parse_tracking(changed_pose("-.3,1.2,.4,0,0,0,-1"), 10.01, 2)
        self.assertTrue(pico.validate_frame(equivalent, now=10.02, previous=first)["valid"])
        jumped = pico.parse_tracking(changed_pose("-.3,1.2,.4,1,0,0,0"), 10.01, 2)
        self.assertIn("left_controller_orientation_jump", pico.validate_frame(jumped, now=10.02, previous=first)["reasons"])


class _FakeFunction:
    def __init__(self, implementation):
        self.implementation = implementation

    def __call__(self, *args):
        return self.implementation(*args)


class ReceiverLifecycleTests(unittest.TestCase):
    def test_init_deinit_only_atomic_copy_disconnect_and_lock(self):
        calls = []
        class FakeSDK:
            PXREAInit = _FakeFunction(lambda *args: calls.append("init") or 0)
            PXREADeinit = _FakeFunction(lambda: calls.append("deinit") or 0)
        with tempfile.TemporaryDirectory() as directory, patch.object(pico.C, "CDLL", return_value=FakeSDK()):
            path = Path(directory) / "receiver.lock"
            receiver = pico.PicoReceiver(lock_path=path).start()
            second = pico.PicoReceiver(lock_path=path)
            with self.assertRaises(RuntimeError):
                second.start()
            state = pico._DeviceState()
            state.stateJson = json.dumps(payload()).encode()
            receiver._receive(None, 1 << 25, 0, C.addressof(state))
            frame = receiver.snapshot()
            state.stateJson = b'{}'
            self.assertEqual(frame.body24x7.shape, (24, 7))
            self.assertEqual(receiver.snapshot().body24x7.shape, (24, 7))
            receiver._receive(None, 1 << 5, 0, None)
            self.assertIsNone(receiver.snapshot())
            receiver.close()
            receiver.close()
            with second:
                self.assertIsNone(second.snapshot())
        self.assertEqual(calls, ["init", "deinit", "init", "deinit"])

    def test_failed_deinit_keeps_callback_and_lock_until_successful_retry(self):
        outcomes = iter((-1, 0))
        class FakeSDK:
            PXREAInit = _FakeFunction(lambda *args: 0)
            PXREADeinit = _FakeFunction(lambda: next(outcomes))
        with tempfile.TemporaryDirectory() as directory, patch.object(pico.C, "CDLL", return_value=FakeSDK()):
            path = Path(directory) / "receiver.lock"
            receiver = pico.PicoReceiver(lock_path=path).start()
            with self.assertRaises(RuntimeError):
                receiver.close()
            self.assertIsNotNone(receiver._callback)
            with self.assertRaises(RuntimeError):
                pico.PicoReceiver(lock_path=path).start()
            receiver.close()
            self.assertIsNone(receiver._callback)


if __name__ == "__main__":
    unittest.main()
