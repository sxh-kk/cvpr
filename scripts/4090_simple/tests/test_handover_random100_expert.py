"""CPU task-controller contracts; synthetic contacts never certify a rollout."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handover_random100_expert import (ExpertConfig, HandoverObjectFeedbackExpert,
    ObjectRelativeReference, ObjectWristIK, PHASES, physical_feedback, solved_contact_durations,
    ContainerPlacementReference, container_goal_geometry, support_margin, eroded_COM_region,
    geometry_upright_rotation, project_to_convex_region, BoundedPositionIntegral)
from prepare_carrybox_v11 import SIMPLE_HAND_NAMES, STATE_TO_TRAIN, TRAIN_STATE_NAMES


def source_fixture(n=100):
    target = np.tile([.5, -.2, .8, 1., 0, 0, 0], (n, 1))
    target[:, 2] += np.minimum(np.arange(n) / 200., .15)
    container = np.tile([.6, .2, .7, 1., 0, 0, 0], (n, 1))
    points = target[:, None, :3] + np.array([[0., .05, .02], [0., -.05, .02]])
    rotations = np.tile(np.eye(3), (n, 2, 1, 1))
    hands = np.tile(np.linspace(0, .8, n)[:, None], (1, 14))
    stages = dict(zip(PHASES, [0, 10, 30, 50, 70, n - 1, n - 1]))
    return points, rotations, {"target": target, "container": container}, hands, stages


def observed(target=None, container=None):
    return {"target": np.array(target if target is not None else [.5, -.2, .8, 1., 0, 0, 0]),
            "container": np.array(container if container is not None else [.6, .2, .7, 1., 0, 0, 0])}


def feedback(left=False, right=False, contact=False, support=False, inside=False, frames=12):
    value = {"contacts": {"left_hand": left, "right_hand": right, "container": contact},
            "container_geometry": {"inside_container_footprint": inside,
                "support_measurement": "live_solved_mujoco_contact_force",
                "container_support_weight_fraction": .8 if support else 0.,
                "target_vertices_minimum_outer_xy_margin_m": .02,
                "collision_vertical_ranges_overlap": True,
                "container_tilt_from_initial_upright_deg": 0.,
                "world_bounds_m": {"min": [.5, .05, .6], "max": [.7, .35, .75]},
                "world_hull_xy_m": [[.5, .05], [.7, .05], [.7, .35], [.5, .35]],
                "target_com_world_m": [.5, -.2, .9],
                "target_collision_world_bounds_m": {"min": [.47, -.23, .8], "max": [.53, -.17, 1.]}},
            "actual_wrist_points_left_right": [[.5, -.15, .92], [.5, -.25, .92]],
            "actual_wrist_rotations_left_right": np.tile(np.eye(3), (2, 1, 1)).tolist(),
            "target_geometry_cache": {"target_root_local_collision_vertices_m":
                [[x, y, z] for x in (-.03, .03) for y in (-.07, .07) for z in (-.1, .1)],
                "target_com_offset_local_m": [0., 0., 0.]}}
    value["recent_contact_samples"] = [{**value, "completed_steps": i + 1,
        "solver_stage": "post_mj_step_before_observation_refresh"} for i in range(frames)]
    return value


class BoundedPositionIntegralTests(unittest.TestCase):
    def test_reference_bias_is_removed_with_bounded_rate_and_total_offset(self):
        config = ExpertConfig(); integral = BoundedPositionIntegral(config)
        bias = np.array([.06, -.025, 0.])
        previous = integral.value.copy()
        for _ in range(600):
            value, _ = integral.step(bias - integral.value, 1 / 30., np.array([True, True, False]))
            self.assertLessEqual(np.linalg.norm(value), config.placement_position_integral_limit_m + 1e-12)
            self.assertLessEqual(np.linalg.norm(value - previous), config.placement_position_integral_speed_m_s / 30 + 1e-12)
            previous = value.copy()
        self.assertLess(np.linalg.norm(bias - value), .001)
        self.assertEqual(value[2], 0.)

    def test_inactive_axes_cannot_wind_up_and_reset_removes_stale_offset(self):
        integral = BoundedPositionIntegral(ExpertConfig())
        for _ in range(300):
            value, _ = integral.step([1., 1., -1.], .1, np.array([True, False, False]))
        self.assertAlmostEqual(value[0], .08)
        np.testing.assert_array_equal(value[1:], [0., 0.])
        value, receipt = integral.step([1., 1., -1.], .1, np.ones(3, dtype=bool), reset=True)
        np.testing.assert_array_equal(value, np.zeros(3))
        self.assertTrue(receipt["reset"])


class ObjectPlannerTests(unittest.TestCase):
    def make(self):
        points, rotations, objects, hands, stages = source_fixture()
        return ObjectRelativeReference(points, rotations, objects, hands, stage_frames=stages)

    def reach_placement(self, planner):
        low, raised = observed(), observed(target=[.5, -.2, .9, 1., 0, 0, 0])
        planner.step(0, low, feedback())
        planner.step(5, low, feedback(right=True))
        planner.step(10, raised, feedback(right=True))
        planner.step(15, raised, feedback(left=True, right=True))
        planner.step(20, raised, feedback(left=True))
        planner.step(25, raised, feedback(left=True))
        result = planner.step(30, raised, feedback(left=True))
        self.assertEqual(result[-1]["phase"], "placement")
        return raised

    def test_elapsed_time_alone_cannot_pass_a_contact_gate_or_exhaust_source(self):
        planner = self.make();planner.step(0, observed())
        result = planner.step(5000, observed())
        self.assertEqual(result[-1]["phase"], "approach")
        self.assertEqual(result[-1]["actual_reference_frame"], 10)
        self.assertTrue(result[4].any())
        self.assertIn("synthesized", result[-1]["future_reference_semantics"])
        self.assertFalse(result[-1]["task_success_certified"])

    def test_phase_order_requires_lifted_right_then_both_then_left(self):
        planner = self.make();planner.step(0, observed())
        result = planner.step(5, observed(), feedback(left=True, right=True))
        self.assertEqual(result[-1]["phase"], "approach")
        planner.step(10, observed(), feedback(right=True))
        self.assertEqual(planner.phase_index, 1)
        planner.step(15, observed(), feedback(right=True))
        self.assertEqual(planner.phase_index, 1)
        lifted = observed(target=[.5, -.2, .9, 1., 0, 0, 0])
        planner.step(20, lifted, feedback(left=True))
        self.assertEqual(planner.phase_index, 1)
        planner.step(25, lifted, feedback(right=True))
        self.assertEqual(planner.phase_index, 2)
        planner.step(30, lifted, feedback(left=True))
        self.assertEqual(planner.phase_index, 2)

    def test_random_target_translation_changes_world_goals(self):
        first, second = self.make(), self.make()
        a = first.step(0, observed())[0]
        pose = observed();pose["target"][:3] += [.04, -.03, .01]
        b = second.step(0, pose)[0]
        np.testing.assert_allclose(b - a, np.tile([.04, -.03, .01], (2, 1)), atol=1e-12)

    def test_actual_target_rotation_retargets_wrist_offset_and_rotation(self):
        planner = self.make();pose = observed();pose["target"][3:] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        goals, rotations, *rest = planner.step(0, pose)
        np.testing.assert_allclose(goals[0] - pose["target"][:3], [-.05, 0, .02], atol=1e-12)
        np.testing.assert_allclose(rotations[0], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)

    def test_lift_tracking_is_not_cancelled_by_object_relative_coordinates(self):
        planner = self.make();planner.step(0, observed())
        result = planner.step(5, observed(), feedback(right=True))
        self.assertGreater(result[-1]["planned_target_xyz_m"][2], .8)
        self.assertGreater(result[-1]["task_feedback_delta_xyz_m"][2], 0)
        self.assertGreater(result[0][1, 2], .82)

    def test_placement_uses_actual_container_and_alignment_feedback(self):
        planner = self.make();pose = self.reach_placement(planner)
        pose["container"][:3] += [.07, .02, 0]
        f = feedback(left=True)
        for edge in ("min", "max"):
            f["container_geometry"]["world_bounds_m"][edge][0] += .07
            f["container_geometry"]["world_bounds_m"][edge][1] += .02
        for vertex in f["container_geometry"]["world_hull_xy_m"]:
            vertex[0] += .07;vertex[1] += .02
        result = planner.step(40, pose, f)
        self.assertEqual(result[-1]["wrist_anchor_left_right"][0], "latched_actual_left_grasp_to_independent_container_COM")
        self.assertGreater(result[-1]["independent_container_placement"]["terminal_container_goal_xy_m"][0], .6)
        self.assertEqual(result[-1]["actual_objects_pose7"]["container"], pose["container"].tolist())
        self.assertTrue(result[-1]["alignment_holds_additional_descent"])
        self.assertGreaterEqual(result[-1]["task_feedback_delta_xyz_m"][2], 0)

    def test_xy_error_gates_only_descent_and_does_not_deadlock_style_or_task_XY(self):
        fixture = source_fixture()
        fixture[0][:, :, 2] -= np.arange(100)[:, None] * .003
        planner = ObjectRelativeReference(*fixture[:4], stage_frames=fixture[4])
        pose = self.reach_placement(planner)
        pose["container"][:3] += [.15, .05, 0]
        first = planner.step(40, pose, feedback(left=True))
        second = planner.step(45, pose, feedback(left=True))
        self.assertTrue(second[-1]["alignment_holds_additional_descent"])
        self.assertGreater(second[-1]["actual_reference_frame"], first[-1]["actual_reference_frame"])
        self.assertTrue(np.all(second[0][:, 2] >= first[0][:, 2] - 1e-12))
        a = first[-1]["independent_container_placement"]
        b = second[-1]["independent_container_placement"]
        center = np.array(b["terminal_container_goal_xy_m"])
        self.assertLess(np.linalg.norm(np.array(b["command_target_COM_world_m"])[:2] - center),
                        np.linalg.norm(np.array(a["command_target_COM_world_m"])[:2] - center))
        self.assertFalse(b["source_target_waypoint_used_as_terminal_goal"])

    def test_release_requires_solved_support_and_recloses_when_it_is_lost(self):
        planner = self.make();pose = self.reach_placement(planner)
        result = planner.step(40, pose, feedback(left=True, contact=True, inside=True))
        self.assertEqual(result[-1]["phase"], "placement")
        np.testing.assert_array_equal(result[2][:7], planner.grip_hands[:7])
        result = planner.step(45, pose, feedback(left=True, contact=True, inside=True, support=True))
        self.assertEqual(result[-1]["phase"], "release")
        self.assertGreater(result[-1]["release_progress"], 0)
        result = planner.step(50, pose, feedback(left=True, contact=True, inside=True))
        self.assertEqual(result[-1]["phase"], "placement")
        self.assertEqual(result[-1]["release_progress"], 0)
        np.testing.assert_array_equal(result[2][:7], planner.grip_hands[:7])

    def test_short_safe_load_transfer_gap_cannot_certify_hold_or_restart_finger_closing(self):
        planner = self.make();pose = self.reach_placement(planner)
        planner.step(40, pose, feedback(left=True, contact=True, inside=True, support=True))
        self.assertEqual(planner.phase_index, 5)
        opening_before = planner.release_progress
        f = feedback(left=True, contact=True, inside=True)
        f["container_geometry"].update(collision_vertical_ranges_overlap=True,
            container_tilt_from_initial_upright_deg=0.)
        f["recent_contact_samples"] = [deepcopy(row) for row in f["recent_contact_samples"]]
        for row in f["recent_contact_samples"][:-1]:
            row["container_geometry"]["container_support_weight_fraction"] = .8
        result = planner.step(45, pose, f)
        self.assertEqual(result[-1]["phase"], "release")
        self.assertTrue(result[-1]["release_support_grace_active"])
        self.assertGreater(planner.release_progress, opening_before)
        # Sustained support loss still recloses, even with safe geometry.
        f["recent_contact_samples"] = feedback(left=True, contact=True, inside=True)["recent_contact_samples"]
        result = planner.step(50, pose, f)
        self.assertEqual(result[-1]["phase"], "placement")
        self.assertEqual(planner.release_progress, 0.)

    def test_release_rejects_missing_dense_history_and_unsafe_supported_geometry(self):
        for invalid in ("missing_history", "tilted", "no_vertical_overlap"):
            with self.subTest(invalid=invalid):
                planner = self.make();pose = self.reach_placement(planner)
                planner.step(40, pose, feedback(left=True, contact=True, inside=True, support=True))
                f = feedback(left=True, contact=True, inside=True, support=invalid != "missing_history")
                if invalid == "missing_history":f.pop("recent_contact_samples")
                elif invalid == "tilted":f["container_geometry"]["container_tilt_from_initial_upright_deg"] = 45.
                else:f["container_geometry"]["collision_vertical_ranges_overlap"] = False
                result = planner.step(45, pose, f)
                self.assertEqual(result[-1]["phase"], "placement")
                self.assertFalse(result[-1]["release_support_grace_active"])

    def test_unreleased_hands_cannot_reach_hold_even_with_support(self):
        planner = self.make();pose = self.reach_placement(planner)
        for tick in (40, 45, 50, 55):
            result = planner.step(tick, pose, feedback(left=True, contact=True, inside=True, support=True))
        self.assertEqual(result[-1]["phase"], "release")
        result = planner.step(60, pose, feedback(contact=True, inside=True, support=True))
        self.assertEqual(result[-1]["phase"], "hold")
        result = planner.step(10000, pose, feedback(contact=True, inside=True, support=True))
        self.assertEqual(result[-1]["actual_reference_frame"], 99)
        self.assertTrue(result[4][1:].all())
        np.testing.assert_array_equal(result[2], planner.open_hands)

    def test_source_arrays_and_actual_feedback_are_not_mutated(self):
        fixture = source_fixture();before = [fixture[0].copy(), fixture[1].copy(), fixture[2]["target"].copy(), fixture[3].copy()]
        planner = ObjectRelativeReference(*fixture[:4], stage_frames=fixture[4])
        pose = observed();original = {k: v.copy() for k, v in pose.items()};planner.step(0, pose)
        for a, b in zip(before, [fixture[0], fixture[1], fixture[2]["target"], fixture[3]]):np.testing.assert_array_equal(a, b)
        for k in pose:np.testing.assert_array_equal(pose[k], original[k])

    def test_invalid_feedback_tick_and_pose_rejected(self):
        with self.assertRaisesRegex(ValueError, "boolean"):physical_feedback({"contacts": {"right_hand": 1}})
        planner = self.make();planner.step(0, observed())
        with self.assertRaisesRegex(ValueError, "increasing"):planner.step(0, observed())
        pose = observed();pose["target"][3:] = [2, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, "unit length"):planner.step(1, pose)
        for value in (0, -1, float("nan")):
            with self.assertRaises(ValueError):ExpertConfig(contact_dwell_s=value)

    def test_unstamped_force_value_does_not_authorize_support(self):
        f = feedback(contact=True, support=True, inside=True)
        f["container_geometry"].pop("support_measurement")
        self.assertFalse(physical_feedback(f)["supported"])

    def test_grasp_reference_anchor_uses_closing_frame_before_lift(self):
        points, rotations, objects, hands, _ = source_fixture()
        objects["target"][:, 2] = .8;objects["target"][80:, 2] = 1.
        hands[:] = 0.;hands[25:, 7:] = .8
        planner = ObjectRelativeReference(points, rotations, objects, hands)
        self.assertEqual(planner.stage_frames[1], 25)
        self.assertIn("closing", planner.anchor_method)

    def test_contact_history_gap_and_isolated_touches_cannot_form_dwell(self):
        f = feedback(right=True, frames=10)
        f["recent_contact_samples"][-2]["contacts"] = {"left_hand": False, "right_hand": False, "container": False}
        self.assertEqual(solved_contact_durations(f)["right_only"], .02)
        planner = self.make();planner.step(0, observed());result = planner.step(100, observed(), f)
        self.assertEqual(result[-1]["phase"], "approach")
        f["recent_contact_samples"][-1]["completed_steps"] += 1
        with self.assertRaisesRegex(ValueError, "consecutive"):solved_contact_durations(f)

    def test_phase_transition_never_jumps_source_clock_to_a_later_anchor(self):
        planner = self.make();planner.step(0, observed())
        result = planner.step(5, observed(), feedback(right=True))
        self.assertEqual(result[-1]["phase"], "right_lift")
        self.assertEqual(result[-1]["actual_reference_frame"], 5)
        self.assertLess(result[-1]["actual_reference_frame"], planner.stage_frames[1])

    def test_pregrasp_ignores_fallen_target_roll_and_pitch(self):
        planner = self.make();pose = observed();pose["target"][3:] = [np.sqrt(.5), 0, np.sqrt(.5), 0]
        result = planner.step(0, pose)
        np.testing.assert_allclose(result[0], planner.points[0], atol=1e-12)
        np.testing.assert_allclose(result[1], planner.rotations[0], atol=1e-12)
        self.assertEqual(result[-1]["target_orientation_feedback"], "yaw_only_until_stable_grasp_and_lift")

    def test_left_transfer_needs_stable_left_and_opens_right_gradually(self):
        planner = self.make();planner.step(0, observed())
        planner.step(5, observed(), feedback(right=True))
        lifted = observed(target=[.5, -.2, .9, 1., 0, 0, 0])
        planner.step(10, lifted, feedback(right=True))
        result = planner.step(15, lifted, feedback(left=True, right=True, frames=7))
        self.assertEqual(result[-1]["phase"], "handover")
        result = planner.step(20, lifted, feedback(left=True, right=True, frames=12))
        self.assertEqual(result[-1]["phase"], "left_transfer")
        self.assertGreater(result[-1]["right_release_progress"], 0.)
        self.assertLess(result[-1]["right_release_progress"], 1.)
        result = planner.step(25, lifted, feedback(right=True))
        self.assertEqual(result[-1]["right_release_progress"], 0.)
        np.testing.assert_array_equal(result[2][7:], planner.grip_hands[7:])

    def test_grasp_closes_gradually_only_after_actual_wrist_alignment(self):
        planner = self.make();planner.step(0, observed())
        far = feedback();far["actual_wrist_points_left_right"] = np.full((2, 3), 5.)
        result = planner.step(20, observed(), far)
        self.assertEqual(result[-1]["actual_reference_frame"], 10)
        self.assertEqual(result[-1]["grasp_close_progress"], 0.)
        near = feedback();near["actual_wrist_points_left_right"] = result[0].copy()
        result = planner.step(25, observed(), near)
        self.assertGreater(result[-1]["grasp_close_progress"], 0.)
        self.assertLess(result[-1]["grasp_close_progress"], 1.)


class ContainerPlacementTests(unittest.TestCase):
    def make(self):
        f = feedback(left=True)
        points = np.asarray(f["actual_wrist_points_left_right"])
        rotations = np.asarray(f["actual_wrist_rotations_left_right"])
        controller = ContainerPlacementReference(np.array([.5, -.2, .9, 1., 0, 0, 0]), points, rotations, f, ExpertConfig())
        return controller, f

    def test_fingers_open_at_actual_wrist_pose_before_bounded_withdrawal(self):
        controller, f = self.make();pose = np.asarray(f["actual_wrist_points_left_right"])[0]
        for progress in (.1, .5, .9):
            points, _, d = controller.step(f, physical_feedback(f), .1, releasing=True, release_progress=progress)
            np.testing.assert_array_equal(points[0], pose)
            self.assertEqual(d["release_wrist_retreat_m"], 0.)
        points, _, d = controller.step(f, physical_feedback(f), .1, releasing=True, release_progress=1.)
        self.assertAlmostEqual(points[0, 2] - pose[2], .008)
        self.assertTrue(d["release_wrist_pose_latched"])
        controller.step(f, physical_feedback(f), .1, releasing=False)
        self.assertIsNone(controller.release_point)

    def test_actual_geometry_centroid_is_independent_of_source_mid_trajectory(self):
        controller, f = self.make()
        _, _, d = controller.step(f, physical_feedback(f), .1)
        self.assertGreaterEqual(d["terminal_container_goal_xy_m"][0], .54)
        self.assertGreaterEqual(d["terminal_container_goal_xy_m"][1], .13)
        self.assertEqual(d["goal_source"], "real_collision_vertices_halfspace_eroded_COM_region")
        self.assertFalse(d["source_target_waypoint_used_as_terminal_goal"])
        self.assertGreater(d["command_target_COM_world_m"][1], -.2)

    def test_large_actual_XY_error_still_advances_toward_basket_while_holding_height(self):
        controller, f = self.make()
        for _ in range(8):_, _, d = controller.step(f, physical_feedback(f), 1.)
        np.testing.assert_allclose(np.array(d["command_target_COM_world_m"])[:2], controller.initial_goal_xy, atol=1e-12)
        self.assertTrue(d["descent_gated_by_actual_container_XY"])
        self.assertGreaterEqual(d["command_target_COM_world_m"][2], .9)

    def test_only_actual_final_container_alignment_permits_descent(self):
        controller, f = self.make()
        for _ in range(8):controller.step(f, physical_feedback(f), 1.)
        old = controller.command_com[2]
        f["container_geometry"]["target_com_world_m"] = [*controller.initial_goal_xy, .9]
        f["container_geometry"]["target_collision_world_bounds_m"]["min"] = [.57, .17, .8]
        _, _, d = controller.step(f, physical_feedback(f), .1)
        self.assertTrue(d["actual_terminal_XY_aligned"])
        self.assertLess(d["command_target_COM_world_m"][2], old)
        self.assertAlmostEqual(old - d["command_target_COM_world_m"][2], .0025)

    def test_actual_collision_bottom_must_clear_rim_before_lateral_transport(self):
        _, f = self.make()
        f["container_geometry"]["target_com_world_m"][2] = .7
        f["container_geometry"]["target_collision_world_bounds_m"]["min"][2] = .6
        controller = ContainerPlacementReference(np.array([.5, -.2, .7, 1., 0, 0, 0]),
            np.asarray(f["actual_wrist_points_left_right"]), np.asarray(f["actual_wrist_rotations_left_right"]), f, ExpertConfig())
        for _ in range(20):_, _, d = controller.step(f, physical_feedback(f), 1.)
        np.testing.assert_allclose(np.array(d["command_target_COM_world_m"])[:2], [.5, -.2], atol=1e-12)
        self.assertTrue(d["XY_transport_gated_only_by_actual_rim_clearance"])
        f["container_geometry"]["target_com_world_m"][2] = .9
        f["container_geometry"]["target_collision_world_bounds_m"]["min"][2] = .8
        _, _, d = controller.step(f, physical_feedback(f), .1)
        self.assertFalse(d["XY_transport_gated_only_by_actual_rim_clearance"])
        self.assertGreater(d["command_target_COM_world_m"][1], -.2)

    def test_nonzero_vertical_tracking_error_can_clear_unchanged_physical_rim_gate(self):
        _, f = self.make()
        f["container_geometry"]["target_com_world_m"][2] = .7
        f["container_geometry"]["target_collision_world_bounds_m"]["min"][2] = .6
        config = ExpertConfig()
        controller = ContainerPlacementReference(np.array([.5, -.2, .7, 1., 0, 0, 0]),
            np.asarray(f["actual_wrist_points_left_right"]), np.asarray(f["actual_wrist_rotations_left_right"]), f, config)
        actual_z = .7;transported = False
        # Reference-only servo model with2cm static height error, no physics
        # success claim. The actual gate retains the original2.5cm clearance.
        for _ in range(120):
            _, _, d = controller.step(f, physical_feedback(f), .1)
            actual_z = max(actual_z, controller.command_com[2] - .02)
            f["container_geometry"]["target_com_world_m"][2] = actual_z
            f["container_geometry"]["target_collision_world_bounds_m"]["min"][2] = actual_z - .1
            if controller.command_com[1] > -.2:
                transported = True;break
        self.assertTrue(transported)
        self.assertGreaterEqual(actual_z - .1, .75 + config.placement_clearance_m)
        self.assertEqual(d["physical_rim_clearance_m"], .025)

    def test_support_search_reference_can_overcome_vertical_servo_bias_then_stop_on_force(self):
        _, f = self.make()
        f["container_geometry"]["target_com_world_m"] = [.6, .2, .9]
        f["container_geometry"]["target_collision_world_bounds_m"]["min"] = [.57, .17, .8]
        controller = ContainerPlacementReference(np.array([.6, .2, .9, 1., 0, 0, 0]),
            np.asarray(f["actual_wrist_points_left_right"]), np.asarray(f["actual_wrist_rotations_left_right"]), f, ExpertConfig())
        found = False
        for _ in range(150):
            controller.step(f, physical_feedback(f), .1)
            actual_z = controller.command_com[2] + .02
            f["container_geometry"]["target_com_world_m"][2] = actual_z
            f["container_geometry"]["target_collision_world_bounds_m"]["min"][2] = actual_z - .1
            if actual_z - .1 <= .615:
                found = True;break
        self.assertTrue(found)
        f["contacts"]["container"] = True
        f["container_geometry"].update(inside_container_footprint=True, container_support_weight_fraction=.8)
        before = controller.command_com.copy()
        _, _, d = controller.step(f, physical_feedback(f), .1)
        np.testing.assert_array_equal(controller.command_com, before)
        self.assertLess(d["terminal_target_COM_floor_z_m"], .6 + .1)

    def test_latched_actual_grasp_survives_source_end_or_fallen_object_pose(self):
        controller, f = self.make()
        points, _, first = controller.step(f, physical_feedback(f), .1)
        before = controller.command_com.copy()
        f["contacts"]["left_hand"] = False
        f["container_geometry"]["target_com_world_m"] = [2., -3., .3]
        f["container_geometry"]["target_collision_world_bounds_m"]["min"] = [1.97, -3.03, .2]
        new, _, second = controller.step(f, physical_feedback(f), 10.)
        self.assertTrue(second["left_grasp_lost_before_support"])
        np.testing.assert_array_equal(controller.command_com, before)
        np.testing.assert_array_equal(new[0], before + controller.target_rotation @ controller.grasp_offset)
        self.assertEqual(first["latched_actual_left_grasp"], second["latched_actual_left_grasp"])

    def test_missing_real_container_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "container lower"):
            container_goal_geometry({}, ExpertConfig())
        f = feedback(left=True);f["container_geometry"]["world_hull_xy_m"] = [[0., 0.], [1., 0.], [2., 0.]]
        with self.assertRaisesRegex(ValueError, "degenerate"):
            container_goal_geometry(f, ExpertConfig())

    def test_complete_projection_erosion_uses_every_vertex_and_COM_offset(self):
        hull = np.array([[0., 0.], [1., 0.], [1., .5], [0., .5]])
        offsets = np.array([[x, y, z] for x in (-.1, .1) for y in (-.05, .05) for z in (-.2, .2)])
        region, normals, boundaries = eroded_COM_region(hull, offsets, .01)
        np.testing.assert_allclose(region.min(axis=0), [.11, .06], atol=1e-12)
        np.testing.assert_allclose(region.max(axis=0), [.89, .44], atol=1e-12)
        for point in region:
            self.assertGreaterEqual(min(support_margin(hull, point + p[:2]) for p in offsets), .01 - 1e-10)
        self.assertLess(np.linalg.norm(project_to_convex_region([-.5, -.5], region) - [.11, .06]), 1e-10)

    def test_upright_yaw_alignment_chooses_nearest_pi_symmetric_orientation(self):
        from feedback_wrist_ik_v11 import rotation_from_vector, rotation_error
        offsets = np.array([[x, y, z] for x in (-.03, .03) for y in (-.08, .08) for z in (-.1, .1)])
        hull = np.array([[-.1, -.2], [.1, -.2], [.1, .2], [-.1, .2]])
        tilted = rotation_from_vector(np.array([.3, .2, .2]))
        goal = geometry_upright_rotation(np.eye(3), offsets, hull, tilted)
        np.testing.assert_allclose(goal[:, 2], [0, 0, 1], atol=1e-12)
        self.assertLess(np.linalg.norm(rotation_error(goal, np.eye(3))), 1e-10)

    def test_actual_full_projection_is_required_before_release(self):
        planner = ObjectPlannerTests().make()
        pose = ObjectPlannerTests().reach_placement(planner)
        f = feedback(left=True, contact=True, inside=True, support=True)
        f["container_geometry"]["target_vertices_minimum_outer_xy_margin_m"] = -.01
        result = planner.step(40, pose, f)
        self.assertEqual(result[-1]["phase"], "placement")
        self.assertEqual(result[-1]["release_progress"], 0.)

    def test_oversized_footprint_has_no_fabricated_safe_region(self):
        with self.assertRaisesRegex(ValueError, "No COM region"):
            eroded_COM_region(np.array([[0, 0], [.1, 0], [.1, .1], [0, .1]]),
                np.array([[-.2, -.2, 0], [.2, -.2, 0], [.2, .2, 0], [-.2, .2, 0]]), .01)

    def test_actual_object_angle_error_continues_to_correct_after_base_rotation_finishes(self):
        from feedback_wrist_ik_v11 import rotation_from_vector, rotation_error
        controller, f = self.make()
        controller.command_rotation = controller.goal_rotation.copy()
        f["actual_target_rotation_world"] = rotation_from_vector(np.array([.30, 0., 0.])) @ controller.goal_rotation
        _, commanded, d = controller.step(f, physical_feedback(f), .1)
        correction = np.asarray(d["bounded_object_orientation_feedback_rotvec_world"])
        self.assertLess(correction[0], 0.)
        self.assertLessEqual(np.linalg.norm(correction), .060000001)
        for _ in range(10):_, commanded, d = controller.step(f, physical_feedback(f), .1)
        self.assertLessEqual(np.linalg.norm(d["bounded_object_orientation_feedback_rotvec_world"]), .250000001)
        self.assertGreater(np.linalg.norm(rotation_error(np.array(d["command_target_rotation_world"]),
                                                        controller.goal_rotation)), .1)
        self.assertAlmostEqual(d["actual_target_orientation_error_rad"], .30, places=8)
        self.assertTrue(d["descent_gated_by_actual_container_XY"])

    def test_goal_is_deeper_in_complete_footprint_safe_region(self):
        controller, _ = self.make()
        nearest = project_to_convex_region(controller.entry_com[:2], controller.safe_region)
        self.assertGreater(support_margin(controller.safe_region, controller.initial_goal_xy),
                           support_margin(controller.safe_region, nearest) + .015)

    def test_safe_actual_projection_can_descend_despite_orientation_tracking_error(self):
        from feedback_wrist_ik_v11 import rotation_from_vector
        controller, f = self.make()
        f["container_geometry"]["target_com_world_m"] = [*controller.initial_goal_xy, .9]
        f["actual_target_rotation_world"] = rotation_from_vector(np.array([.30, 0., 0.]))
        before = controller.command_com[2]
        _, _, d = controller.step(f, physical_feedback(f), .1)
        self.assertGreater(d["actual_target_orientation_error_rad"], .15)
        self.assertGreaterEqual(d["actual_complete_footprint_margin_m"], .005)
        self.assertTrue(d["actual_rim_clearance_observed_before_descent"])
        self.assertFalse(d["descent_gated_by_actual_container_XY"])
        self.assertLess(controller.command_com[2], before)

    def test_actual_projection_loss_stops_descent_without_lowering_quality_threshold(self):
        controller, f = self.make()
        f["container_geometry"]["target_com_world_m"] = [*controller.initial_goal_xy, .9]
        controller.step(f, physical_feedback(f), .1)
        before = controller.command_com[2]
        f["container_geometry"]["target_com_world_m"][0] += .12
        _, _, d = controller.step(f, physical_feedback(f), .1)
        self.assertLess(d["actual_complete_footprint_margin_m"], .005)
        self.assertTrue(d["descent_gated_by_actual_container_XY"])
        self.assertGreaterEqual(controller.command_com[2], before)

    def test_transient_rim_clear_far_from_goal_does_not_disable_raise_recovery(self):
        _, f = self.make()
        f["container_geometry"]["target_com_world_m"][2] = .7
        f["container_geometry"]["target_collision_world_bounds_m"]["min"][2] = .6
        controller = ContainerPlacementReference(np.array([.5, -.2, .7, 1., 0, 0, 0]),
            np.asarray(f["actual_wrist_points_left_right"]), np.asarray(f["actual_wrist_rotations_left_right"]), f, ExpertConfig())
        f["container_geometry"]["target_com_world_m"][2] = .9
        f["container_geometry"]["target_collision_world_bounds_m"]["min"][2] = .8
        controller.step(f, physical_feedback(f), .1)
        self.assertTrue(controller.rim_clearance_observed)
        self.assertFalse(controller.descent_started)
        f["container_geometry"]["target_com_world_m"][2] = .7
        f["container_geometry"]["target_collision_world_bounds_m"]["min"][2] = .6
        before = controller.command_com[2]
        _, _, d = controller.step(f, physical_feedback(f), .1)
        self.assertFalse(d["actual_target_bottom_clears_container_rim"])
        self.assertFalse(d["descent_started_after_actual_XY_projection_and_current_rim_clear"])
        self.assertGreater(controller.command_com[2], before)

    def test_position_integral_cannot_bypass_rim_and_resets_on_lost_grip(self):
        controller, f = self.make()
        f["container_geometry"]["target_com_world_m"][2] = .7
        f["container_geometry"]["target_collision_world_bounds_m"]["min"][2] = .6
        for _ in range(30):
            _, _, d = controller.step(f, physical_feedback(f), .1)
        np.testing.assert_array_equal(d["applied_position_integral_world_m"], np.zeros(3))
        f["container_geometry"]["target_com_world_m"][2] = .9
        f["container_geometry"]["target_collision_world_bounds_m"]["min"][2] = .8
        for _ in range(20):
            _, _, d = controller.step(f, physical_feedback(f), .1)
        self.assertGreater(np.linalg.norm(d["applied_position_integral_world_m"]), 0.)
        f["contacts"]["left_hand"] = False
        _, _, d = controller.step(f, physical_feedback(f), .1)
        self.assertTrue(d["bounded_position_integral"]["reset"])
        np.testing.assert_array_equal(d["applied_position_integral_world_m"], np.zeros(3))


RUNTIME = Path(os.environ.get("CVPR_SIMPLE_RUNTIME", "/home/zxc/CVPR_Evaluation_runtime"))
ROBOT = RUNTIME / "resources/simple/robots/g1_sonic/g1_29dof_with_hand.xml"
SONIC = RUNTIME / "resources/sonic_v1_1"
SOURCE = RUNTIME / "resources/simple_finetune_sources/G1WholebodyXMoveBendCarryBoxSonic-v0/data/chunk-000/episode_000001.parquet"


@unittest.skipUnless(ROBOT.is_file() and SOURCE.is_file() and (SONIC / "model_encoder.onnx").is_file(),
                     "Requires local actual robot/source and pinned official CPU encoder")
class ActualRobotAndEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco
        import pyarrow.parquet as pq
        table = pq.read_table(SOURCE).slice(550, 60)
        cls.states = np.asarray(table["observation.state"].to_pylist())[:, STATE_TO_TRAIN]
        cls.roots = np.asarray(table["observation.base_pose"].to_pylist())
        cls.model = mujoco.MjModel.from_xml_path(str(ROBOT))
        cls.ik = ObjectWristIK(cls.model, cls.states, cls.roots[:, :3], cls.roots[:, 3:])

    def values(self):
        n = len(self.states)
        target = np.tile(np.r_[self.ik.points[0, 1], [1, 0, 0, 0]], (n, 1))
        container = target.copy();container[:, 0] += .3
        return {"reference_full_state43": self.states, "reference_full_body_dq": np.zeros((n, 29)),
            "reference_full_root_xyz": self.roots[:, :3], "reference_full_root_wxyz": self.roots[:, 3:],
            "reference_full_object_poses": {"target": target, "container": container},
            "reference_full_hand_commands14": np.tile(np.linspace(0, .5, n)[:, None], (1, 14)),
            "reference_full_declared_hand_commands14": np.tile(np.linspace(0, .7, n)[:, None], (1, 14)),
            "reference_full_state_joint_names": TRAIN_STATE_NAMES,
            "reference_full_hand_command_names": SIMPLE_HAND_NAMES}

    def test_current_ik_reduces_real_fk_error_and_keeps_live_data_unchanged(self):
        import mujoco
        live = mujoco.MjData(self.model)
        before = [live.qpos.copy(), live.qvel.copy(), live.ctrl.copy(), live.time]
        desired = self.ik.points[0] + [.03, .02, .015]
        delta, reports = self.ik.solve_current(0, self.roots[0, :3], self.ik.rotations[0],
            desired, self.ik.orientations[0], ExpertConfig())
        self.assertLessEqual(np.max(np.abs(delta)), .450000001)
        self.assertLess(np.mean([r["position_error_after_m"] for r in reports]),
                        .6 * np.mean([r["position_error_before_m"] for r in reports]))
        for a, b in zip(before, [live.qpos, live.qvel, live.ctrl, live.time]):np.testing.assert_array_equal(a, b)

    def test_official_cpu_encoder_and_actual_private_ik_emit_feedback_bound_commands(self):
        import mujoco
        values = self.values();expert = HandoverObjectFeedbackExpert(values, SONIC)
        live = mujoco.MjData(self.model);live.qpos[self.ik.qadr] = self.states[0]
        live.qpos[self.ik.root_qadr:self.ik.root_qadr + 7] = self.roots[0]
        mujoco.mj_forward(self.model, live)
        objects = {k: SimpleNamespace(xpos=v[0, :3].copy(), xquat=v[0, 3:].copy())
                   for k, v in values["reference_full_object_poses"].items()}
        raw = SimpleNamespace(mujoco=SimpleNamespace(mjModel=self.model, mjData=live, mj_objects=objects))
        before = [live.qpos.copy(), live.qvel.copy(), live.ctrl.copy(), live.time]
        token, hands, first = expert(0, raw, feedback())
        objects["target"].xpos += [.05, -.02, 0]
        token2, hands2, second = expert(2, raw, feedback())
        for value, shape in ((token, (64,)), (token2, (64,)), (hands, (14,)), (hands2, (14,))):
            self.assertEqual(value.shape, shape);self.assertTrue(np.isfinite(value).all())
        self.assertEqual(expert.session.get_providers(), ["CPUExecutionProvider"])
        self.assertNotEqual(first["encoder_input_sha256"], second["encoder_input_sha256"])
        self.assertNotEqual(first["desired_wrist_points_left_right"], second["desired_wrist_points_left_right"])
        self.assertEqual(second["body_token_sha256"], __import__("hashlib").sha256(token2.tobytes()).hexdigest())
        self.assertEqual(second["actual_objects_pose7"]["target"][:3], objects["target"].xpos.tolist())
        self.assertEqual(len(second["current_two_wrist_ik"]), 2)
        self.assertFalse(second["physical_success_claim"])
        for a, b in zip(before, [live.qpos, live.qvel, live.ctrl, live.time]):np.testing.assert_array_equal(a, b)

    def test_named_hand_order_is_checked_before_encoding(self):
        values = self.values();values["reference_full_hand_command_names"] = tuple(reversed(SIMPLE_HAND_NAMES))
        with self.assertRaisesRegex(ValueError, "thumb/index/middle"):
            HandoverObjectFeedbackExpert(values, SONIC)

    def test_placement_uses_actual_posture_full_ranges_and_slew_limits(self):
        actual = self.states[0].copy()
        position, rotation = self.roots[0, :3], self.ik.rotations[0]
        self.ik._set_reference(actual, position, rotation)
        original, _ = self.ik._pose(0)
        com = self.ik.data.subtree_com[self.model.body("pelvis").id, :2].copy()
        polygon = com + np.array([[-.25, -.25], [.25, -.25], [.25, .25], [-.25, .25]])
        reachable = actual.copy();reachable[15] += .6;reachable[18] += .3
        for i, name in enumerate(TRAIN_STATE_NAMES[:29]):
            joint = self.model.joint(name)
            if bool(joint.limited[0]):reachable[i] = np.clip(reachable[i], *joint.range)
        self.ik._set_reference(reachable, position, rotation);goal, orientation = self.ik._pose(0)
        self.assertGreater(np.linalg.norm(goal - original), .12)
        command, reports, meta, _, _ = self.ik.solve_placement(0, actual, position, rotation,
            goal, orientation, actual[:29], .04, ExpertConfig(), support_polygon=polygon)
        self.assertFalse(reports[0]["position_goal_clipped"])
        self.assertFalse(reports[0]["source_reach_radius_applied"])
        self.assertLess(reports[0]["full_physical_range_solution_residual_m"], .01)
        self.assertLessEqual(np.max(np.abs(command - actual[:29])), .060000001)
        self.assertLessEqual(np.max(np.abs(command[12:15] - actual[12:15])) / .04, .350000001)
        self.assertTrue(meta["torso_IK_enabled"])
        self.assertGreaterEqual(meta["full_solution_robot_COM_margin_m"], meta["required_robot_COM_margin_m"] - 1e-8)
        self.assertFalse(meta["right_wrist_task_constraint_enabled"])
        for i, name in enumerate(TRAIN_STATE_NAMES[:29]):
            joint = self.model.joint(name)
            if bool(joint.limited[0]):
                self.assertGreaterEqual(command[i], joint.range[0] - 1e-12)
                self.assertLessEqual(command[i], joint.range[1] + 1e-12)

    def test_active_torso_limits_do_not_stall_free_arm_joints(self):
        actual = self.states[0].copy()
        actual[13] = self.model.joint("waist_roll_joint").range[0]
        actual[14] = self.model.joint("waist_pitch_joint").range[1]
        position, rotation = self.roots[0, :3], self.ik.rotations[0]
        self.ik._set_reference(actual, position, rotation)
        com = self.ik.data.subtree_com[self.model.body("pelvis").id, :2].copy()
        polygon = com + np.array([[-.25, -.25], [.25, -.25], [.25, .25], [-.25, .25]])
        reachable = actual.copy();reachable[15:22] += [.3, .1, -.4, -.5, .2, .2, .1]
        for i, name in enumerate(TRAIN_STATE_NAMES[:29]):
            joint = self.model.joint(name)
            if bool(joint.limited[0]):reachable[i] = np.clip(reachable[i], *joint.range)
        self.ik._set_reference(reachable, position, rotation);goal, orientation = self.ik._pose(0)
        _, report, meta, _, _ = self.ik.solve_placement(0, actual, position, rotation,
            goal, orientation, actual[:29], .04, ExpertConfig(), support_polygon=polygon)
        self.assertLess(report[0]["full_physical_range_solution_residual_m"], .005)
        self.assertGreaterEqual(meta["full_solution_robot_COM_margin_m"], meta["required_robot_COM_margin_m"] - 1e-8)

    def test_full_range_placement_encoder_branch_uses_live_posture_without_writing_it(self):
        import mujoco
        values = self.values();expert = HandoverObjectFeedbackExpert(values, SONIC)
        live = mujoco.MjData(self.model);live.qpos[self.ik.qadr] = self.states[0]
        live.qpos[self.ik.root_qadr:self.ik.root_qadr + 7] = self.roots[0]
        mujoco.mj_forward(self.model, live)
        objects = {k: SimpleNamespace(xpos=v[0, :3].copy(), xquat=v[0, 3:].copy())
                   for k, v in values["reference_full_object_poses"].items()}
        raw = SimpleNamespace(mujoco=SimpleNamespace(mjModel=self.model, mjData=live, mj_objects=objects))
        expert(0, raw, feedback())
        expert.planner.phase_index = 4
        before = [live.qpos.copy(), live.qvel.copy(), live.ctrl.copy(), live.time]
        token, hands, meta = expert(2, raw, feedback(left=True))
        self.assertEqual(token.shape, (64,));self.assertTrue(np.isfinite(token).all())
        self.assertFalse(meta["current_two_wrist_ik"][0]["source_reach_radius_applied"])
        self.assertEqual(len(meta["desired_body_q29"]), 29)
        self.assertFalse(meta["placement_actual_posture_IK"]["right_wrist_task_constraint_enabled"])
        self.assertLessEqual(meta["current_two_wrist_ik"][0]["actual_maximum_command_velocity_rad_s"], 1.5000001)
        locomotion = meta["locomotion_reference"]
        self.assertEqual(locomotion["contract"], "cvpr-source-lower15-native-future-v6")
        indices = np.asarray(locomotion["source_future_frames"])
        np.testing.assert_allclose(locomotion["lower15_future_q"], self.states[indices, :15])
        np.testing.assert_allclose(locomotion["lower15_future_dq"], values["reference_full_body_dq"][indices, :15])
        self.assertFalse(locomotion["task_waist_residual_applied"])
        self.assertFalse(meta["placement_actual_posture_IK"]["task_torso_IK_enabled"])
        np.testing.assert_array_equal(meta["current_arm_delta_velocity"], np.zeros(14))
        for a, b in zip(before, [live.qpos, live.qvel, live.ctrl, live.time]):np.testing.assert_array_equal(a, b)
        # Native clock continues independently of the arm/style phase and
        # becomes an explicit generated stand hold after the actual source end.
        _, _, tail = expert(1000, raw, feedback(left=True))
        self.assertFalse(any(tail["locomotion_reference"]["real_native_future_mask"]))
        np.testing.assert_array_equal(tail["locomotion_reference"]["lower15_future_dq"], np.zeros((10, 15)))

    def test_actual_hold_remains_an_explicit_locomotion_control_condition(self):
        self.assertEqual(ExpertConfig(placement_locomotion="actual_hold").placement_locomotion, "actual_hold")
        with self.assertRaisesRegex(ValueError, "placement locomotion"):
            ExpertConfig(placement_locomotion="unknown")

    def test_actual_hold_static_horizon_has_zero_reference_velocity(self):
        import mujoco
        values = self.values()
        expert = HandoverObjectFeedbackExpert(values, SONIC, config=ExpertConfig(placement_locomotion="actual_hold"))
        live = mujoco.MjData(self.model);live.qpos[self.ik.qadr] = self.states[0]
        live.qpos[self.ik.root_qadr:self.ik.root_qadr + 7] = self.roots[0]
        mujoco.mj_forward(self.model, live)
        objects = {k: SimpleNamespace(xpos=v[0, :3].copy(), xquat=v[0, 3:].copy())
                   for k, v in values["reference_full_object_poses"].items()}
        raw = SimpleNamespace(mujoco=SimpleNamespace(mjModel=self.model, mjData=live, mj_objects=objects))
        expert(0, raw, feedback());expert.planner.phase_index = 4
        original = expert.session;seen = []
        def record(outputs, inputs):
            seen.append(next(iter(inputs.values())).copy())
            return original.run(outputs, inputs)
        expert.session = SimpleNamespace(run=record, get_inputs=original.get_inputs)
        expert(2, raw, feedback(left=True))
        q = seen[-1][0, 4:294].reshape(10, 29)
        np.testing.assert_array_equal(q, np.tile(q[0], (10, 1)))
        np.testing.assert_array_equal(seen[-1][0, 294:584], np.zeros(290))

    def test_placement_encoder_mode_accepts_only_explicit_g1_or_teleop(self):
        self.assertEqual(ExpertConfig().placement_encoder_mode, "g1")
        self.assertEqual(ExpertConfig(placement_encoder_mode="teleop").placement_encoder_mode, "teleop")
        with self.assertRaisesRegex(ValueError, "placement encoder mode"):
            ExpertConfig(placement_encoder_mode="unknown")

    def test_official_mode1_is_placement_only_and_preserves_source_head_style(self):
        import mujoco
        from feedback_wrist_ik_v11 import rotation_from_vector
        values = self.values()
        g1 = HandoverObjectFeedbackExpert(values, SONIC)
        teleop = HandoverObjectFeedbackExpert(values, SONIC, config=ExpertConfig(placement_encoder_mode="teleop"))
        live = mujoco.MjData(self.model);live.qpos[self.ik.qadr] = self.states[0]
        root = self.roots[0].copy()
        actual_R = rotation_from_vector(np.array([.08, -.04, .2])) @ self.ik.rotations[0]
        mujoco.mju_mat2Quat(root[3:], np.ascontiguousarray(actual_R).reshape(-1))
        live.qpos[self.ik.root_qadr:self.ik.root_qadr + 7] = root
        mujoco.mj_forward(self.model, live)
        objects = {k: SimpleNamespace(xpos=v[0, :3].copy(), xquat=v[0, 3:].copy())
                   for k, v in values["reference_full_object_poses"].items()}
        raw = SimpleNamespace(mujoco=SimpleNamespace(mjModel=self.model, mjData=live, mj_objects=objects))
        first_g1, _, mode0 = g1(0, raw, feedback())
        first_teleop, _, prefix = teleop(0, raw, feedback())
        np.testing.assert_array_equal(first_g1, first_teleop)
        self.assertEqual(mode0["encoder_input_sha256"], prefix["encoder_input_sha256"])
        self.assertEqual(prefix["encoder_mode"], 0)
        self.assertIsNone(prefix["placement_teleop_reference"])
        class RecordingSession:
            def __init__(self, delegate):self.delegate = delegate;self.value = None
            def get_inputs(self):return self.delegate.get_inputs()
            def run(self, outputs, inputs):
                self.value = next(iter(inputs.values())).copy()
                return self.delegate.run(outputs, inputs)
        recording = RecordingSession(teleop.session);teleop.session = recording
        teleop.planner.phase_index = 4
        before = [live.qpos.copy(), live.qvel.copy(), live.ctrl.copy(), live.time]
        token, hands, meta = teleop(2, raw, feedback(left=True))
        reference = recording.value
        self.assertEqual(reference.shape, (1, 1751));self.assertEqual(reference.dtype, np.float32)
        self.assertEqual(meta["encoder_mode"], 1)
        self.assertEqual(meta["mode"], "online_object_task_teleop_sonic11_mode1")
        np.testing.assert_array_equal(reference[0, :4], [1, 0, 0, 0])
        np.testing.assert_array_equal(reference[0, 4:644], 0.)
        td = meta["placement_teleop_reference"]
        points = np.asarray(td["world_points_left_right_head_m"])
        np.testing.assert_allclose(reference[0, 890:899].reshape(3, 3), (points - root[:3]) @ actual_R, atol=1e-6)
        np.testing.assert_array_equal(td["apply_delta_heading_wxyz"], [1, 0, 0, 0])
        self.assertTrue(td["official_source_verification"]["official_sources_rehashed"])
        self.assertFalse(td["G1_arm_joint_IK_encoded"])
        self.assertFalse(meta["G1_arm_IK_solution_used_for_token"])
        self.assertEqual(meta["desired_body_q29_direct_encoder_mask"], [True] * 12 + [False] * 17)
        native = meta["locomotion_reference"]
        np.testing.assert_allclose(reference[0, 650:770].reshape(10, 12), np.asarray(native["lower15_future_q"])[:, :12], atol=1e-6)
        private = mujoco.MjData(self.model)
        private.qpos[self.ik.qadr] = self.states[meta["actual_reference_frame"]]
        private.qpos[self.ik.root_qadr:self.ik.root_qadr + 7] = root
        mujoco.mj_forward(self.model, private)
        torso = self.model.body("torso_link").id
        expected_head = private.xpos[torso] + private.xmat[torso].reshape(3, 3) @ [0., 0., .35]
        np.testing.assert_allclose(points[2], expected_head, atol=1e-12)
        self.assertTrue(np.isfinite(token).all());self.assertEqual(token.shape, (64,))
        for a, b in zip(before, [live.qpos, live.qvel, live.ctrl, live.time]):np.testing.assert_array_equal(a, b)

    def test_declared_and_inferred_hand_conditions_are_explicit(self):
        values = self.values()
        first = HandoverObjectFeedbackExpert(values, SONIC, config=ExpertConfig(hand_condition="declared"))
        np.testing.assert_array_equal(first.hands, values["reference_full_declared_hand_commands14"])
        values["hand_condition"] = "inferred_effective"
        second = HandoverObjectFeedbackExpert(values, SONIC, config=ExpertConfig(hand_condition="declared"))
        np.testing.assert_array_equal(second.hands, values["reference_full_hand_commands14"])
        self.assertFalse(np.array_equal(first.hands, second.hands))


if __name__ == "__main__":
    unittest.main()
