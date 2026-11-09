"""Execute the actual dual-mode collector branches without SDK or simulation."""
import ast
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pico_simple_teleop as entry
import test_pico_mobile_integration as legacy


class LockedNavigation(legacy.FakeMobile):
    """CPU dispatch oracle: navigation is mode0, never live three-point mode1."""
    def step(self, tick, root, body, frame, *, tracking_valid,
             vr_world_points=None, vr_world_quaternions=None):
        if vr_world_points is not None or vr_world_quaternions is not None:
            raise AssertionError("Dual-mode navigation must not request mode1 VR")
        self.ticks.append((tick, frame.device_timestamp_ns, tracking_valid))
        return np.zeros((1, 1751), dtype=np.float32), {
            "encoder_mode": 0, "reference_policy": "locked_upper_mode0",
            "upper_body_frozen": True, "tick50": tick}


class XMoveBendCollectorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.driver = legacy.CollectorMobileIntegrationTests()

    def routing(self, profile, policy):
        statements = next(node for node in legacy.run_tree().body
                          if isinstance(node, ast.FunctionDef) and node.name == "run").body
        start = next(i for i, node in enumerate(statements) if legacy.is_assign(node, "mobile"))
        end = next(i for i, node in enumerate(statements) if legacy.is_assign(node, "collection_prefix"))
        namespace = {"profile": profile, "args": SimpleNamespace(mobile_control_policy=policy)}
        legacy.execute(statements[start:end + 1], namespace)
        return namespace

    def namespace(self, mode="pose"):
        ns = self.driver.namespace()
        physical_root = np.array([1.4, -.3, .76, .7071067811865476, 0., 0., .7071067811865476])
        episode = SimpleNamespace(status="recording", human_demo_uuid="same-recording")
        def forbidden_reset(*args, **kwargs):
            raise AssertionError("B transitions must not reset physical state or the recording")
        ns.update(mobile=False, joystick_task=True, xmove_bend_pick=True,
            bend_pick=False, xmove_pick=False, grasp=False, target_only=True,
            mode1_task=False, mobile_continuous=False,
            mobile_policy="locked_upper_mode0", mobile_control=LockedNavigation(),
            mobile_mode=mode, following=True, episode=episode,
            frame=self.driver.frame(1), mobile_stopped=SimpleNamespace(ready=True),
            actual_root_velocity=lambda *_: np.zeros(6),
            root_pose=lambda *_: physical_root.copy(),
            control_trace=io.StringIO(), reset_scene=forbidden_reset,
            env=SimpleNamespace(reset=forbidden_reset),
            _physical_root=physical_root, _episode=episode)
        ns["data"].qpos = np.arange(12, dtype=float)
        ns["data"].qvel = np.arange(12, dtype=float) / 10
        ns["controller"].reset = forbidden_reset
        return ns

    def assert_recording_and_physics_unchanged(self, ns, before):
        self.assertIs(ns["episode"], ns["_episode"])
        self.assertTrue(ns["following"])
        np.testing.assert_array_equal(ns["_physical_root"], before[0])
        np.testing.assert_array_equal(ns["data"].qpos, before[1])
        np.testing.assert_array_equal(ns["data"].qvel, before[2])

    def state_copy(self, ns):
        return ns["_physical_root"].copy(), ns["data"].qpos.copy(), ns["data"].qvel.copy()

    def test_actual_routing_is_target_only_switchable_and_fail_closed(self):
        ns = self.routing("xmove_bend_pick", "locked_upper_mode0")
        self.assertTrue(ns["joystick_task"])
        self.assertTrue(ns["target_only"])
        self.assertFalse(ns["mobile"])
        self.assertFalse(ns["mobile_continuous"])
        self.assertFalse(ns["mode1_task"])
        self.assertEqual(ns["object_roles"], ("target",))
        self.assertEqual(ns["collection_prefix"], "xmove_bend_pick")
        with self.assertRaisesRegex(ValueError, "requires joystick mode0"):
            self.routing("xmove_bend_pick", "continuous_vr_mode1")
        for profile in ("handover", "mobile_pp", "grasp", "xmove_pick", "bend_pick"):
            with self.subTest(profile=profile):
                previous = self.routing(profile, "continuous_vr_mode1")
                self.assertEqual(previous["joystick_task"], profile == "mobile_pp")
                self.assertEqual(previous["mobile_continuous"], profile in ("mobile_pp", "grasp", "xmove_pick"))

    def test_real_button_mapping_enables_b_only_for_switchable_joystick_tasks(self):
        buttons = next(node for node in ast.walk(legacy.run_tree())
                       if isinstance(node, ast.FunctionDef) and node.name == "operator_buttons")
        for profile, policy, expected in (("xmove_bend_pick", "locked_upper_mode0", True),
                ("mobile_pp", "locked_upper_mode0", True),
                ("mobile_pp", "continuous_vr_mode1", False),
                ("xmove_pick", "continuous_vr_mode1", False),
                ("bend_pick", "continuous_vr_mode1", False)):
            ns = dict(vars(entry), **self.routing(profile, policy))
            legacy.execute([buttons], ns)
            self.assertEqual(ns["operator_buttons"]({"B": True}, {}), ["toggle_navigation"] if expected else [])
            self.assertEqual(ns["operator_buttons"]({"B": True}, {"B": True}), [])

    def test_b_entry_freezes_measured_pose_without_resetting_recording(self):
        ns = self.namespace()
        before = self.state_copy(ns)
        ns["frame"].body24x7[18, 0] = .015
        self.driver.toggle_navigation(ns)
        self.assertEqual(ns["mobile_mode"], "preparing_navigation")
        self.assertEqual(ns["mobile_tick"], 0)
        self.assertEqual(len(ns["mobile_control"].boots), 1)
        np.testing.assert_array_equal(ns["mobile_control"].boots[0][0], before[0])
        frozen = ns["mobile_entry_smpl"].copy()
        ns["frame"].body24x7[18, 0] = .2
        np.testing.assert_array_equal(ns["mobile_entry_smpl"], frozen)
        self.assert_recording_and_physics_unchanged(ns, before)
        self.assertEqual(json.loads(ns["control_trace"].getvalue())["mode"], "preparing_navigation")

    def test_warmup_unsettled_sticks_and_speed_cannot_switch_or_end_recording(self):
        for reason in ("warmup", "not_settled", "stick", "moving"):
            with self.subTest(reason=reason):
                ns = self.namespace()
                if reason == "warmup": ns["episode"].status = "warmup"
                if reason == "not_settled": ns["mobile_stopped"].ready = False
                if reason == "stick": ns["frame"].inputs["left_axis_y"] = .4
                if reason == "moving": ns["actual_root_velocity"] = lambda *_: [.1, 0, 0, 0, 0, 0]
                before = self.state_copy(ns)
                self.driver.toggle_navigation(ns)
                self.assertEqual(ns["mobile_mode"], "pose")
                self.assertEqual(ns["mobile_control"].boots, [])
                self.assert_recording_and_physics_unchanged(ns, before)

    def test_navigation_preparation_and_mode0_commands_keep_same_episode_and_ledger(self):
        ns = self.namespace()
        self.driver.toggle_navigation(ns)
        ns["mobile_control"].ready = False
        self.driver.tick(ns, 0)
        self.assertEqual(ns["mobile_mode"], "preparing_navigation")
        self.assertEqual(ns["reference_evidence"]["encoder_mode"], 0)
        self.assertTrue(ns["reference_evidence"]["static_desired_reference"])
        self.assertEqual(ns["mobile_control"].ticks, [])
        ns["mobile_control"].ready = True
        for tick in range(1, 5): self.driver.tick(ns, tick)
        self.assertEqual(ns["mobile_mode"], "navigation")
        self.assertTrue(all(row[1]["reference_evidence"]["encoder_mode"] == 0 for row in ns["_ledger_rows"]))
        self.assertTrue(all(call[0] == "planner_encode" for call in ns["reference"].calls))
        self.assertEqual([row[1]["physical_tick50"] for row in ns["_ledger_rows"]], [0, 2, 4])
        self.assertIs(ns["episode"], ns["_episode"])

    def test_bent_or_changed_pose_return_is_rejected_without_interrupting_navigation(self):
        for reason in ("arm_pose", "human_tilt"):
            with self.subTest(reason=reason):
                ns = self.namespace("navigation")
                if reason == "arm_pose": ns["frame"].body24x7[18, 0] = .08
                else:
                    def reject_tilt(*args): raise ValueError("humanTilt=25 exceeds15")
                    ns["require_neutral_calibration"] = reject_tilt
                before = self.state_copy(ns)
                self.driver.toggle_navigation(ns)
                self.assertEqual(ns["mobile_mode"], "navigation")
                self.assertEqual(ns["reference"].calibrations, [])
                self.assertEqual(ns["mobile_control"].pauses, [])
                self.assertIn("当前录制继续", ns["operator_message"])
                self.assert_recording_and_physics_unchanged(ns, before)

    def test_return_binds_current_heading_then_new_mode2_command_without_physical_reset(self):
        ns = self.namespace("navigation")
        self.driver.tick(ns, 0)
        # Navigation changed the actual world position/heading; this is the
        # measured return pose, never an instruction to move physical state.
        ns["_physical_root"][:] = [2.2, -.8, .77, 0., 0., 0., 1.]
        before = self.state_copy(ns)
        self.driver.toggle_navigation(ns)
        self.assertEqual(ns["mobile_mode"], "pose")
        self.assertEqual(ns["mobile_control"].pauses, ["return_to_manipulation"])
        np.testing.assert_array_equal(ns["reference"].calibrations[-1][1], before[0])
        self.assertIsNone(ns["reference"].last_timestamp_ns)
        self.assertIsNone(ns["mobile_preparation"])
        self.assertIsNone(ns["mobile_entry_smpl"])
        self.assertIn("全身弯腰", ns["operator_message"])
        self.assert_recording_and_physics_unchanged(ns, before)
        transition = json.loads(ns["control_trace"].getvalue())
        self.assertTrue(transition["return_pose_check"]["human_upright"]["passed"])
        self.assertEqual(transition["calibration"]["actual_root_pose7"], before[0].tolist())
        navigation_calls = len(ns["mobile_control"].ticks)
        self.driver.tick(ns, 2)
        self.assertEqual(ns["reference"].calls[-1][0:2], ("pose_encode", self.driver.frame(2).device_timestamp_ns))
        self.assertEqual(len(ns["mobile_control"].ticks), navigation_calls)
        self.assertEqual(ns["_ledger_rows"][-1][1]["reference_evidence"]["encoder_mode"], 2)

    def test_production_stopped_window_is_observed_for_new_joystick_task(self):
        ns = self.namespace()
        ns.update(mobile_stopped=entry.MobileStoppedWindow(), tilt=0.)
        branch = next(node for node in ast.walk(legacy.run_tree()) if isinstance(node, ast.If)
            and ast.unparse(node.test) == "joystick_task"
            and any(isinstance(child, ast.Call) and ast.unparse(child.func) == "mobile_stopped.observe"
                    for child in ast.walk(node)))
        for tick in range(26):
            ns.update(global_tick=tick, frame=self.driver.frame(tick))
            legacy.execute([branch], ns)
            self.assertEqual(ns["mobile_stopped"].ready, tick == 25)

    def test_native_sample_uses_same_step_in_both_control_modes(self):
        for mode in ("navigation", "pose"):
            ns = self.namespace(mode)
            buffer = io.StringIO()
            ns.update(pending_mobile_quality_row={"completed_steps": 51, "sim_time_s": 1.02},
                quality_stream=buffer, raw=SimpleNamespace(task=SimpleNamespace(_init_target_height=.51)),
                env=SimpleNamespace(step=lambda command: ({}, .91, True, False, {})))
            legacy.execute(legacy.step_and_native_block(), ns)
            row = json.loads(buffer.getvalue())
            self.assertTrue(row["official_task_success"])
            self.assertEqual(row["native_task_reward"], .91)
            self.assertEqual(row["native_baseline_target_z_m"], .51)
            self.assertIsNone(ns["pending_mobile_quality_row"])
            self.assertIs(ns["episode"], ns["_episode"])


if __name__ == "__main__":
    unittest.main()
