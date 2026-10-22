"""CPU checks of source strict gates, solved support and rejection boundaries."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import handover_contact_quality as contact
import handover_geometry_quality as geometry


def accepted_rows():
    rows = []
    for index in range(65):
        left, right = (False, True) if index < 5 else (True, True) if index < 10 else (True, False) if index < 15 else (False, False)
        rows.append({"completed_steps": index + 1, "sim_time_s": (index + 1) / 50,
            "actual_target_xyz_m": [0., 0., .7 if index < 15 else .61], "pelvis_z_m": .75,
            "contacts": {"left_hand": left, "right_hand": right, "container": index >= 15},
            "container_geometry": {"inside_container_footprint": True, "signed_margin_m": .02,
                "collision_vertical_ranges_overlap": True, "container_upright_cosine": 1.,
                "container_support_weight_fraction": 1., "support_measurement": "live_solved_mujoco_contact_force"},
            "solver_stage": contact.SOLVED_STATE_STAGE})
    return rows


class StrictCaptureTests(unittest.TestCase):
    def check(self, rows):
        return contact.evaluate_capture(rows, initial_target_z_m=.55)

    def test_full_gate_passes_but_does_not_authorize_dataset_or_modify_rows(self):
        rows = accepted_rows(); before = deepcopy(rows)
        report = self.check(rows)
        self.assertTrue(report["eligible_physics"])
        self.assertTrue(report["verified_handover_success"])
        self.assertEqual(report["measurement_hz"], 50)
        self.assertEqual(report["acceptance_gate_version"], "handover-physics-v2")
        self.assertFalse(report["dataset_ready"])
        self.assertEqual(rows, before)

    def test_touching_basket_without_lifted_transfer_fails(self):
        rows = accepted_rows()
        for row in rows:
            row["contacts"] = {"left_hand": False, "right_hand": False, "container": True}
        self.assertFalse(self.check(rows)["eligible_physics"])

    def test_palm_contact_in_final_window_rejects_release(self):
        rows = accepted_rows(); rows[-1]["contacts"]["left_hand"] = True
        self.assertFalse(self.check(rows)["eligible_physics"])

    def test_object_returns_to_same_endpoint_but_moves_in_between_fails(self):
        rows = accepted_rows(); rows[-20]["actual_target_xyz_m"][0] = .006
        self.assertFalse(self.check(rows)["eligible_physics"])

    def test_insufficient_or_fabricated_support_fails_full_gate(self):
        for key, value in (("container_support_weight_fraction", .49),
                           ("support_measurement", "offline_reconstructed_pose")):
            rows = accepted_rows()
            for row in rows[-50:]: row["container_geometry"][key] = value
            report = self.check(rows)
            self.assertTrue(report["contact_evidence"]["strict_handover_success"])
            self.assertFalse(report["eligible_physics"])

    def test_non_upright_basket_fails(self):
        rows = accepted_rows(); rows[-1]["container_geometry"]["container_upright_cosine"] = .8
        self.assertFalse(self.check(rows)["eligible_physics"])

    def test_missing_geometry_or_clock_or_pre_refresh_marker_fails(self):
        for key in ("container_geometry", "sim_time_s", "solver_stage"):
            rows = accepted_rows(); rows[-1].pop(key)
            self.assertFalse(self.check(rows)["eligible_physics"])
        rows = accepted_rows(); rows[-1]["sim_time_s"] += .005
        self.assertFalse(self.check(rows)["eligible_physics"])

    def test_unknown_contacts_and_low_pelvis_fail(self):
        for mutation in (lambda r: r[-1]["contacts"].pop("right_hand"),
                         lambda r: r[0].update(pelvis_z_m=.39)):
            rows = accepted_rows(); mutation(rows)
            self.assertFalse(self.check(rows)["eligible_physics"])

    def test_geometry_only_legacy_pass_cannot_override_full_v2_missing_force(self):
        rows = accepted_rows()
        for row in rows: row["container_geometry"].pop("container_support_weight_fraction")
        legacy = geometry.final_placement_geometric_validation(rows)
        self.assertTrue(legacy["passed"])
        self.assertFalse(self.check(rows)["eligible_physics"])


class LiveSolvedSupportTests(unittest.TestCase):
    def test_actual_solved_force_is_upward_and_reading_does_not_forward_or_step(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".005"/>
          <worldbody><body name="container" pos="0 0 .02"><geom type="box" size=".2 .2 .02"/></body>
          <body name="target" pos="0 0 .09"><freejoint/><geom type="box" size=".03 .03 .05" mass=".1"/></body>
          <body name="pelvis" pos="0 0 .75"><geom type="sphere" size=".01" contype="0" conaffinity="0"/></body>
          <body name="left_wrist_yaw_link" pos="1 0 1"><geom type="sphere" size=".01"/></body>
          <body name="right_wrist_yaw_link" pos="-1 0 1"><geom type="sphere" size=".01"/></body>
          </worldbody></mujoco>''')
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        cache = geometry.build_container_geometry_cache(model, data, container_body="container", target_body="target")
        for _ in range(300): mujoco.mj_step(model, data)
        before = (data.qpos.copy(), data.qvel.copy(), float(data.time))
        with patch.object(mujoco, "mj_forward", side_effect=AssertionError("must not refresh")), patch.object(mujoco, "mj_step", side_effect=AssertionError("must not step")):
            row = contact.sample_live_state(model, data, cache=cache, completed_steps=1,
                    target_body="target", container_body="container", solver_stage=contact.SOLVED_STATE_STAGE)
        self.assertTrue(row["contacts"]["container"])
        self.assertGreater(row["container_geometry"]["container_support_weight_fraction"], .5)
        np.testing.assert_array_equal(data.qpos, before[0]); np.testing.assert_array_equal(data.qvel, before[1])
        self.assertEqual(float(data.time), before[2])
        with self.assertRaisesRegex(ValueError, "pre-observation-refresh"):
            contact.sample_live_state(model, data, cache=cache, completed_steps=1,
                    target_body="target", container_body="container", solver_stage="post_mj_forward")

    def test_wrist_palm_subtree_counts_as_hand_contact(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="target"><freejoint/><geom type="sphere" size=".1"/></body>
          <body name="container" pos="2 0 0"><geom type="box" size=".1 .1 .1"/></body>
          <body name="left_wrist_yaw_link" pos=".15 0 0"><geom type="sphere" size=".1"/></body>
          <body name="right_wrist_yaw_link" pos="-2 0 0"><geom type="sphere" size=".1"/></body>
          </worldbody></mujoco>''')
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        state = geometry.target_contacts(model, data, "target", container_label="container")
        self.assertTrue(state["left_hand"])
        self.assertFalse(state["right_hand"])


if __name__ == "__main__": unittest.main()
