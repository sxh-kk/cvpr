"""Pure recorded-evidence fixtures; no policy, SDK or physics model is run."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sonic_locomotion_physics as physics


def evidence():
    samples, forces = [], []
    for index in range(550):
        tick = index-50
        progress = max(0., min(tick-99, 300))*.003
        feet = {}
        for side, lift_range in (("left", (125, 140)), ("right", (160, 176))):
            z = .025 if lift_range[0] <= tick < lift_range[1] else -.0002
            feet[side] = {"minimum_collision_z_m": z, "ground_clearance_m": z,
                "ankle_world_xyz": [progress, 0., z+.035],
                "world_collision_bounds_min_m": [progress-.05, -.03, z],
                "world_collision_bounds_max_m": [progress+.12, .03, z+.01]}
        samples.append({"global_tick50": index, "post_warmup_tick50": tick,
            "physics_state_row": 4*(index+1), "sim_time_s": (index+1)*.02,
            "actual_root_pose7": [progress, 0., .77, 1., 0., 0., 0.],
            "actual_root_qvel6": [.15 if 100 <= tick < 400 else 0., 0., 0., 0., 0., 0.],
            "pelvis_tilt_rad": 0., "feet": feet})
        for substep in range(4):
            step = index*4+substep
            def state(at):
                return {"sim_time_s": at/200, "maximum_xfrc_applied": 0.,
                    "maximum_qfrc_applied": 0., "neq": 0, "nmocap": 0}
            forces.append({"physics_step200": step+1, "before": state(step), "after": state(step+1)})
    return samples, forces


class WalkingEvidenceTests(unittest.TestCase):
    def audit(self, samples, forces, **kwargs):
        return physics.audit_locomotion(samples, forces, movement_world=[1., 0., 0.], **kwargs)

    def test_real_measured_progress_stop_and_two_foot_swings_pass_without_admission_claim(self):
        samples, forces = evidence()
        original = copy.deepcopy(samples)
        report = self.audit(samples, forces)
        self.assertTrue(report["passed"], report)
        self.assertAlmostEqual(report["metrics"]["signed_direction_progress_m"], .9)
        self.assertFalse(report["accepted_new_dataset"])
        self.assertFalse(report["human_following_validated"])
        self.assertEqual(samples, original)

    def test_direction_is_signed_and_forecast_cannot_rescue_actual_stationarity(self):
        samples, forces = evidence()
        for row in samples:
            row["forecast_root_pose7_at_control_start"] = [10., 0., .77, 1., 0., 0., 0.]
            row["actual_root_pose7"][0] *= -1
        report = self.audit(samples, forces)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["actual_direction_progress"])
        for row in samples:
            row["actual_root_pose7"][0] = 0.
        self.assertFalse(self.audit(samples, forces)["passed"])

    def test_last_half_second_uses_maximum_actual_speed_not_average(self):
        samples, forces = evidence()
        samples[-2]["actual_root_qvel6"][1] = .101
        report = self.audit(samples, forces)
        self.assertFalse(report["checks"]["stopped_last_0p5s"])
        samples[-2]["actual_root_qvel6"][1] = .1
        self.assertFalse(self.audit(samples, forces)["checks"]["stopped_last_0p5s"])

    def test_ankle_lift_does_not_substitute_for_actual_collision_sole_lift(self):
        samples, forces = evidence()
        for row in samples:
            foot = row["feet"]["left"]
            foot["ankle_world_xyz"][2] = 10.
            foot["minimum_collision_z_m"] = foot["ground_clearance_m"] = -.001
            foot["world_collision_bounds_min_m"][2] = -.001
        self.assertFalse(self.audit(samples, forces)["checks"]["left_foot_swing_and_landing"])

    def test_lift_without_later_landing_is_rejected(self):
        samples, forces = evidence()
        for row in samples:
            if row["post_warmup_tick50"] >= 125:
                foot = row["feet"]["left"]
                foot["minimum_collision_z_m"] = foot["ground_clearance_m"] = .025
                foot["world_collision_bounds_min_m"][2] = .025
                foot["world_collision_bounds_max_m"][2] = .035
        self.assertFalse(self.audit(samples, forces)["checks"]["left_foot_swing_and_landing"])

    def test_missing_or_gapped_real_clocks_fail_closed(self):
        for change in ("50hz", "200hz", "force_gap", "substep_binding", "missing_warmup"):
            with self.subTest(change=change):
                samples, forces = evidence()
                if change == "50hz": samples[90]["sim_time_s"] += .01
                elif change == "200hz": forces[90]["after"]["sim_time_s"] += .001
                elif change == "force_gap": forces.pop()
                elif change == "substep_binding": samples[90]["physics_state_row"] += 1
                else: samples = samples[50:]
                self.assertFalse(self.audit(samples, forces)["passed"])

    def test_nonzero_assistance_and_nonfinite_values_rejected(self):
        for change in ("force", "mocap", "equality", "nan", "bool"):
            with self.subTest(change=change):
                samples, forces = evidence()
                state = forces[42]["before"]
                if change == "force": state["maximum_xfrc_applied"] = 1e-12
                elif change == "mocap": state["nmocap"] = 1
                elif change == "equality": state["neq"] = 1
                elif change == "bool": state["maximum_xfrc_applied"] = False
                else: samples[42]["actual_root_pose7"][0] = float("nan")
                self.assertFalse(self.audit(samples, forces)["passed"])

    def test_height_tilt_and_quaternion_must_describe_actual_state(self):
        samples, forces = evidence()
        samples[70]["actual_root_pose7"][2] = .6
        self.assertFalse(self.audit(samples, forces)["checks"]["root_height"])
        samples, forces = evidence()
        angle = np.deg2rad(26)
        samples[70]["actual_root_pose7"][3:] = [np.cos(angle/2), np.sin(angle/2), 0., 0.]
        samples[70]["pelvis_tilt_rad"] = float(angle)
        self.assertFalse(self.audit(samples, forces)["checks"]["upright"])
        samples[70]["pelvis_tilt_rad"] = 0.
        self.assertFalse(self.audit(samples, forces)["passed"])

    def test_ground_height_and_direction_configuration_are_explicit(self):
        samples, forces = evidence()
        for row in samples:
            for foot in row["feet"].values():
                for key in ("minimum_collision_z_m",): foot[key] += .4
                foot["world_collision_bounds_min_m"][2] += .4
                foot["world_collision_bounds_max_m"][2] += .4
                foot["ankle_world_xyz"][2] += .4
        self.assertTrue(self.audit(samples, forces, ground_z_m=.4)["passed"])
        self.assertFalse(self.audit(samples, forces)["passed"])
        self.assertFalse(physics.audit_locomotion(samples, forces, movement_world=[0., 0., 1.])["passed"])


class CollisionGeometryTests(unittest.TestCase):
    def test_rotated_box_sole_bounds_are_exact_support_extents(self):
        r = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
        low, high = physics.collision_bounds("box", [.085, .03, .005], [1., 2., .1], r)
        np.testing.assert_allclose(low, [.97, 1.995, .015], rtol=0, atol=1e-15)
        np.testing.assert_allclose(high, [1.03, 2.005, .185], rtol=0, atol=1e-15)

    def test_sphere_radius_is_world_invariant_and_unsupported_shape_refused(self):
        low, high = physics.collision_bounds("sphere", [.005, 0., 0.], [0., 0., .03], np.eye(3))
        self.assertAlmostEqual(low[2], .025)
        self.assertAlmostEqual(high[2], .035)
        with self.assertRaises(ValueError): physics.collision_bounds("mesh", [1., 1., 1.], [0., 0., 0.], np.eye(3))

    def test_foot_fk_reads_enabled_collisions_without_any_physics_loop(self):
        fake_mujoco = SimpleNamespace(mjtGeom=SimpleNamespace(mjGEOM_BOX=6, mjGEOM_SPHERE=2))
        model = SimpleNamespace(body=lambda name: SimpleNamespace(id=0), geom_bodyid=np.array([0, 0]),
            geom_contype=np.array([0, 1]), geom_conaffinity=np.array([0, 1]), geom_type=np.array([6, 2]),
            geom_size=np.array([[10., 10., 10.], [.005, 0., 0.]]))
        data = SimpleNamespace(geom_xpos=np.array([[0., 0., 10.], [0., 0., .035]]),
            geom_xmat=np.tile(np.eye(3).reshape(1, 9), (2, 1)), xpos=np.array([[0., 0., .07]]))
        with patch.dict(sys.modules, {"mujoco": fake_mujoco}):
            foot = physics.foot_geometry(model, data, "left")
            self.assertEqual(foot["geom_indices"], [1])
            self.assertAlmostEqual(foot["ground_clearance_m"], .03)
            model.geom_contype[:] = model.geom_conaffinity[:] = 0
            with self.assertRaises(ValueError): physics.foot_geometry(model, data, "left")


if __name__ == "__main__":
    unittest.main()
