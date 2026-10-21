"""Pure-CPU contracts for the new explicit mode1 world-FK reference expert."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("feedback_vr_fk_v11", SCRIPTS / "feedback_vr_fk_v11.py")
expert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expert)


class WorldFkContracts(unittest.TestCase):
    def fixture(self):
        n = 61
        q = np.arange(n * 29, dtype=np.float64).reshape(n, 29) / 100
        dq = -q / 2
        root = np.array([.9, .1, -.2, .3])
        root /= np.linalg.norm(root)
        local_quats = np.array([[1., 0, 0, 0], [.8, .6, 0, 0], [.8, 0, .6, 0]])
        body_quats = expert.quaternion_multiply(root, local_quats)
        position = np.array([2., -1., .8])
        local_points = np.array([[.4, .3, .2], [.4, -.3, .2], [0, 0, .7]])
        points = position + (expert.matrix_wxyz(root) @ local_points.T).T
        return q, dq, np.tile(root, (n, 1)), np.tile(points, (n, 1, 1)), np.tile(body_quats, (n, 1, 1)), position, root, local_points, local_quats

    def test_mode1_mask_lower12_and_scalar_header(self):
        q, dq, roots, points, orientations, position, root, _, _ = self.fixture()
        value = expert.mode1_input(q, dq, roots, points, orientations, 0, position, root)
        mask = np.zeros(1751, dtype=bool)
        for start, stop in expert.MODE1_SLICES:
            mask[start:stop] = True
        self.assertEqual(value.shape, (1, 1751))
        self.assertEqual(value.dtype, np.float32)
        self.assertEqual(value[0, :4].tolist(), [1., 0, 0, 0])
        self.assertTrue(np.all(value[0, ~mask] == 0))
        self.assertTrue(np.array_equal(value[0, 650:770].reshape(10, 12), q[expert.FUTURE_OFFSETS, :12].astype(np.float32)))
        self.assertTrue(np.array_equal(value[0, 770:890].reshape(10, 12), dq[expert.FUTURE_OFFSETS, :12].astype(np.float32)))

    def test_actual_equals_source_matches_cpp_local_pose_math(self):
        q, dq, roots, points, orientations, position, root, local_points, local_quats = self.fixture()
        value = expert.mode1_input(q, dq, roots, points, orientations, 0, position, root)
        self.assertTrue(np.allclose(value[0, 890:899].reshape(3, 3), local_points, atol=1e-7))
        self.assertTrue(np.allclose(value[0, 899:911].reshape(3, 4), local_quats, atol=1e-7))
        # Only yaw is removed in the heading6D branch, while VR poses use full root orientation.
        rotation = expert.matrix_wxyz(root)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        expected = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ rotation
        self.assertTrue(np.allclose(value[0, 644:650], expected[:, :2].reshape(-1), atol=1e-7))

    def test_world_targets_stay_fixed_and_actual_translation_changes_local_error(self):
        q, dq, roots, points, orientations, position, root, _, _ = self.fixture()
        first = expert.mode1_input(q, dq, roots, points, orientations, 0, position, root)
        delta = np.array([.1, -.03, .02])
        moved = expert.mode1_input(q, dq, roots, points, orientations, 0, position + delta, root)
        expected_change = -expert.matrix_wxyz(root).T @ delta
        self.assertTrue(np.allclose((moved-first)[0, 890:899].reshape(3, 3), expected_change, atol=1e-7))
        self.assertTrue(np.array_equal(first[0, 650:890], moved[0, 650:890]))
        self.assertTrue(np.array_equal(first[0, 899:911], moved[0, 899:911]))

    def test_no_tail_padding_and_invalid_actual_pose_rejected(self):
        q, dq, roots, points, orientations, position, root, _, _ = self.fixture()
        with self.assertRaises(ValueError):
            expert.mode1_input(q, dq, roots, points, orientations, 16, position, root)
        with self.assertRaises(ValueError):
            expert.mode1_input(q, dq, roots, points, orientations, 0, position, root * 2)


if __name__ == "__main__":
    unittest.main()
