"""CPU checks of the bounded new world-wrist reference, never policy success."""
from pathlib import Path
import os
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feedback_wrist_ik_v11 import WorldWristIK, rotation_error, rotation_from_vector
from prepare_carrybox_v11 import STATE_TO_TRAIN

RUNTIME = Path(os.environ.get("CVPR_SIMPLE_RUNTIME", "/home/zxc/CVPR_Evaluation_runtime"))
ROBOT = RUNTIME / "resources/simple/robots/g1_sonic/g1_29dof_with_hand.xml"
SOURCE = RUNTIME / "resources/simple_finetune_sources/G1WholebodyXMoveBendCarryBoxSonic-v0/data/chunk-000/episode_000001.parquet"


class RotationTests(unittest.TestCase):
    def test_world_rotation_error_inverts_world_axis_increment(self):
        current = rotation_from_vector(np.array([.2, -.1, .3]))
        increment = np.array([.03, -.04, .05])
        np.testing.assert_allclose(rotation_error(rotation_from_vector(increment) @ current, current), increment, atol=1e-12)


@unittest.skipUnless(ROBOT.is_file() and SOURCE.is_file(), "Requires the local fixed native CarryBox source and robot")
class ActualRobotIkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pyarrow.parquet as pq
        table = pq.read_table(SOURCE).slice(550, 60)
        cls.states = np.asarray(table["observation.state"].to_pylist())[:, STATE_TO_TRAIN]
        cls.roots = np.asarray(table["observation.base_pose"].to_pylist())
        cls.model = mujoco.MjModel.from_xml_path(str(ROBOT))

    def test_zero_root_error_is_exact_identity_for_all_body_references(self):
        ik = WorldWristIK(self.model, self.states, self.roots[:,:3], self.roots[:,3:])
        corrected, _ = ik.correct(0, self.roots[0,:3], ik.rotations[0])
        np.testing.assert_array_equal(corrected, self.states[np.arange(10)*5, :29])

    def test_real_grasp_pose_offset_reduces_fk_error_with_bounds_and_without_live_state_changes(self):
        live = mujoco.MjData(self.model)
        live.time = 2.75
        before = [live.qpos.copy(), live.qvel.copy(), live.ctrl.copy(), live.time]
        ik = WorldWristIK(self.model, self.states, self.roots[:,:3], self.roots[:,3:])
        corrected, metadata = ik.correct(0, self.roots[0,:3]+[.04,.025,.015], ik.rotations[0])
        original = self.states[np.arange(10)*5, :29]
        np.testing.assert_array_equal(corrected[:,:15], original[:,:15])
        self.assertLessEqual(np.max(abs(corrected-original)), .200000001)
        errors_before = [side["source_goal_position_error_before_m"] for node in metadata["wrist_diagnostics_left_right"] for side in node]
        errors_after = [side["source_goal_position_error_after_m"] for node in metadata["wrist_diagnostics_left_right"] for side in node]
        self.assertLess(np.mean(errors_after), .6*np.mean(errors_before))
        for first, last in zip(before, [live.qpos, live.qvel, live.ctrl, live.time]):
            np.testing.assert_array_equal(first, last)


if __name__ == "__main__":
    unittest.main()
