"""Real-model CPU contracts for the optional robot-only startup wrist IK."""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "VLMExpert/third_party/SIMPLE/src"))
try:
    import mujoco
except ImportError:
    mujoco = None

if mujoco is not None:
    from simple.baselines.dual_source_psi0 import BODY_JOINT_NAMES, SIMPLE_HAND_JOINT_NAMES
    from simple.baselines.warmup_wrist_reference import WarmupWristReference


RUNTIME = Path(os.environ.get("CVPR_SIMPLE_RUNTIME", "/home/zxc/CVPR_Evaluation_runtime"))
MODEL_XML = RUNTIME / "resources/simple/robots/g1_sonic/g1_29dof_with_hand.xml"
DIAGNOSTIC = RUNTIME / "mobile_warmup_safe_20261105/level0_case04_baseline"


def integration_state(model, data):
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    return state


@unittest.skipUnless(mujoco is not None and MODEL_XML.is_file(),
                     "Real MuJoCo G1 resources are unavailable")
class WarmupWristReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Missing resources permit a skip; present but invalid resources fail.
        cls.model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        cls.root_qadr = int(cls.model.joint("floating_base_joint").qposadr[0])
        cls.robot_qadr = np.array([int(cls.model.joint(name).qposadr[0])
                                  for name in BODY_JOINT_NAMES + SIMPLE_HAND_JOINT_NAMES])

    def data(self, yaw=0., model=None):
        model = self.model if model is None else model
        data = mujoco.MjData(model)
        root = int(model.joint("floating_base_joint").qposadr[0])
        data.qpos[root + 3:root + 7] = [np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]
        data.time = .005
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        return data

    def body_reference(self, data):
        return np.array([float(data.joint(name).qpos[0]) for name in BODY_JOINT_NAMES])

    def assert_arm_limits(self, model, body):
        for index, name in enumerate(BODY_JOINT_NAMES[15:29], start=15):
            joint = model.joint(name)
            if joint.limited[0]:
                self.assertGreaterEqual(body[index], float(joint.range[0]) - 1e-12)
                self.assertLessEqual(body[index], float(joint.range[1]) + 1e-12)

    def test_first_frame_identity_and_supplied_legs_waist_are_preserved(self):
        data = self.data()
        helper = WarmupWristReference(self.model, data)
        measured = self.body_reference(data)
        reference = measured.copy()
        reference[:15] = np.linspace(-.2, .3, 15)
        unchanged = reference.copy()
        actual_arm = measured[15:29].copy()
        output, event = helper.reference(data, reference)
        np.testing.assert_array_equal(output[:15], reference[:15])
        np.testing.assert_array_equal(output[15:29], actual_arm)
        np.testing.assert_array_equal(reference, unchanged)
        self.assertTrue(event["all_arms_converged"])
        self.assertEqual(event["smoothstep"], 0.)
        self.assertTrue(all(item["iterations"] == 0 for item in event["arms_left_right"]))

    def test_no_live_state_writes_or_dynamics_contact_calls(self):
        data = self.data()
        data.qvel[:] = np.linspace(-.1, .1, self.model.nv)
        data.ctrl[:] = .125
        reference = self.body_reference(data)
        before = integration_state(self.model, data)
        derived = {name: getattr(data, name).copy() for name in ("qacc", "xpos", "xmat", "geom_xpos")}
        with ExitStack() as stack:
            for name in ("mj_forward", "mj_step", "mj_collision", "mj_contactForce"):
                stack.enter_context(patch.object(mujoco, name, side_effect=AssertionError("Dynamics/contact access forbidden")))
            helper = WarmupWristReference(self.model, data)
            helper.reference(data, reference)
        np.testing.assert_array_equal(before, integration_state(self.model, data))
        for name, value in derived.items():
            np.testing.assert_array_equal(value, getattr(data, name))
        receipt = helper.receipt
        self.assertFalse(receipt["object_feedback"] or receipt["contact_feedback"] or
                         receipt["future_teacher"] or receipt["live_state_writes"] or receipt["history_injection"])

    def test_offsets_rotate_with_initial_horizontal_pelvis_heading(self):
        for yaw in (0., np.pi / 2, -.7):
            with self.subTest(yaw=yaw):
                helper = WarmupWristReference(self.model, self.data(yaw))
                rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                                     [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
                expected = np.array([[-.05, .03, .05], [-.05, -.03, .05]]) @ rotation.T
                np.testing.assert_allclose(helper.receipt["world_displacements_left_right_m"], expected, atol=1e-12)
                np.testing.assert_allclose(helper.receipt["fixed_heading_world_forward"], rotation[:, 0], atol=1e-12)

    def test_smoothstep_and_world_goals_stay_fixed_after_transition(self):
        data = self.data(.4)
        helper = WarmupWristReference(self.model, data)
        reference = self.body_reference(data)
        first = helper.receipt
        initial_points = np.array(first["initial_world_positions_left_right_m"])
        displacement = np.array(first["world_displacements_left_right_m"])
        rotations = np.array(first["fixed_world_rotations_left_right"])
        for elapsed, expected_blend in ((0., 0.), (.05, .15625), (.1, .5), (.2, 1.), (.4, 1.)):
            data.time = helper.start_time_s + elapsed
            if elapsed == .4:
                # Current robot root moves/turns; the first world goals do not.
                data.qpos[self.root_qadr:self.root_qadr + 3] += [.015, -.01, .005]
                data.qpos[self.root_qadr + 3:self.root_qadr + 7] = [np.cos(.15), 0., 0., np.sin(.15)]
            output, event = helper.reference(data, reference)
            self.assertAlmostEqual(event["smoothstep"], expected_blend)
            for side, arm in enumerate(event["arms_left_right"]):
                np.testing.assert_allclose(arm["goal_world_position_m"], initial_points[side] + expected_blend * displacement[side], atol=1e-12)
                np.testing.assert_allclose(arm["goal_world_rotation"], rotations[side], atol=1e-12)
            np.testing.assert_array_equal(output[:15], reference[:15])
            self.assertEqual(event["world_goal_held"], elapsed >= .2)

    def test_real_model_final_goal_has_bounded_iterations_and_small_residuals(self):
        data = self.data()
        helper = WarmupWristReference(self.model, data)
        reference = self.body_reference(data)
        data.time += .2
        output, event = helper.reference(data, reference)
        self.assertTrue(event["all_arms_converged"])
        for arm in event["arms_left_right"]:
            self.assertLessEqual(arm["iterations"], 40)
            self.assertLessEqual(arm["position_residual_m"], 1e-4)
            self.assertLessEqual(arm["orientation_residual_rad"], 1e-3)
            self.assertLessEqual(arm["maximum_iteration_joint_step_rad"], .04 + 1e-12)
        self.assert_arm_limits(self.model, output)
        self.assertTrue(np.isfinite(output).all())

    def test_unreachable_goal_reports_residuals_and_respects_joint_limits(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        for name in BODY_JOINT_NAMES[15:29]:
            joint = model.joint(name)
            joint.limited[0] = 1
            joint.range[:] = [-.001, .001]
        data = self.data(model=model)
        helper = WarmupWristReference(model, data, max_iterations=4)
        data.time += .2
        output, event = helper.reference(data, self.body_reference(data))
        self.assertFalse(event["all_arms_converged"])
        self.assert_arm_limits(model, output)
        self.assertTrue(any(arm["joint_limit_clipped_elements"] > 0 for arm in event["arms_left_right"]))
        self.assertTrue(all(arm["iterations"] <= 4 for arm in event["arms_left_right"]))
        self.assertTrue(all(arm["position_residual_m"] > 1e-4 for arm in event["arms_left_right"]))
        self.assertTrue(np.isfinite(output).all())

    def test_initial_measurement_identity_and_receipt_cannot_be_mutated_by_caller(self):
        data = self.data()
        helper = WarmupWristReference(self.model, data)
        receipt = helper.receipt
        canonical = json.dumps(receipt["initial_measured_robot"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.assertEqual(receipt["initial_measured_robot_sha256"], hashlib.sha256(canonical.encode()).hexdigest())
        self.assertEqual(receipt["calls"], 0)
        receipt["profile"]["backward_m"] = 100
        self.assertEqual(helper.receipt["profile"]["backward_m"], .05)
        _, event = helper.reference(data, self.body_reference(data))
        event["arms_left_right"][0]["goal_world_position_m"][0] += 100
        self.assertNotEqual(event, helper.receipt["events"][0])
        json.dumps(helper.receipt, allow_nan=False)
        changed = self.data()
        changed.qpos[self.robot_qadr[-1]] += .01
        other = WarmupWristReference(self.model, changed)
        self.assertNotEqual(helper.receipt["initial_measured_robot_sha256"], other.receipt["initial_measured_robot_sha256"])

    def test_invalid_reference_robot_and_solver_values_are_rejected(self):
        for bad in (np.zeros(28), np.r_[np.nan, np.zeros(28)], np.r_[np.inf, np.zeros(28)]):
            with self.subTest(reference=bad.shape):
                data = self.data()
                helper = WarmupWristReference(self.model, data)
                with self.assertRaises(ValueError):
                    helper.reference(data, bad)
        for coordinate in (self.root_qadr, self.robot_qadr[20]):
            data = self.data()
            data.qpos[coordinate] = np.nan
            with self.assertRaises(ValueError):
                WarmupWristReference(self.model, data)
        data = self.data()
        data.qpos[self.root_qadr + 3:self.root_qadr + 7] = 0
        with self.assertRaises(ValueError):
            WarmupWristReference(self.model, data)
        for options in ({"max_iterations": 0}, {"max_iterations": 101}, {"max_iterations": True},
                        {"damping": np.nan}, {"orientation_weight_m": 0}, {"max_joint_step_rad": .3}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    WarmupWristReference(self.model, self.data(), **options)

    def test_invalid_initial_current_and_backwards_times_are_rejected(self):
        for time_s in (-.01, np.nan, np.inf):
            data = self.data()
            data.time = time_s
            with self.assertRaises(ValueError):
                WarmupWristReference(self.model, data)
        for time_s in (-.01, np.nan, np.inf):
            data = self.data()
            helper = WarmupWristReference(self.model, data)
            data.time = time_s
            with self.assertRaises(ValueError):
                helper.reference(data, self.body_reference(data))
        data = self.data()
        helper = WarmupWristReference(self.model, data)
        data.time += .2
        helper.reference(data, self.body_reference(data))
        data.time -= .1
        with self.assertRaisesRegex(ValueError, "chronological"):
            helper.reference(data, self.body_reference(data))

    def test_different_model_data_is_rejected_without_mutation(self):
        other_model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        other_data = self.data(model=other_model)
        before = integration_state(other_model, other_data)
        with self.assertRaisesRegex(ValueError, "bound model"):
            WarmupWristReference(self.model, other_data)
        data = self.data()
        helper = WarmupWristReference(self.model, data)
        with self.assertRaisesRegex(ValueError, "bound model"):
            helper.reference(other_data, self.body_reference(other_data))
        np.testing.assert_array_equal(before, integration_state(other_model, other_data))

    @unittest.skipUnless((DIAGNOSTIC / "physics_model.mjb").is_file() and
                         (DIAGNOSTIC / "post_reset_integration.npy").is_file(),
                         "Optional original c04 diagnostic model/state is unavailable")
    def test_task_object_pose_does_not_change_goal_identity_or_robot_ik(self):
        model = mujoco.MjModel.from_binary_path(str(DIAGNOSTIC / "physics_model.mjb"))
        original = np.load(DIAGNOSTIC / "post_reset_integration.npy", allow_pickle=False)
        first, second = mujoco.MjData(model), mujoco.MjData(model)
        for data in (first, second):
            mujoco.mj_setState(model, data, original, mujoco.mjtState.mjSTATE_INTEGRATION)
        object_joint = model.joint("cracker_box_joint")
        address = int(object_joint.qposadr[0])
        second.qpos[address:address + 3] += [10., -7., 3.]
        left, right = WarmupWristReference(model, first), WarmupWristReference(model, second)
        self.assertEqual(left.receipt["initial_measured_robot_sha256"], right.receipt["initial_measured_robot_sha256"])
        np.testing.assert_array_equal(left.receipt["world_displacements_left_right_m"], right.receipt["world_displacements_left_right_m"])
        for data in (first, second):
            data.time += .2
        before = integration_state(model, second)
        q1, event1 = left.reference(first, self.body_reference(first))
        q2, event2 = right.reference(second, self.body_reference(second))
        np.testing.assert_array_equal(q1, q2)
        self.assertEqual(event1, event2)
        np.testing.assert_array_equal(before, integration_state(model, second))


if __name__ == "__main__":
    unittest.main()
