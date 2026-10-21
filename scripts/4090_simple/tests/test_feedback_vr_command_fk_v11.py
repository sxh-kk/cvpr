"""CPU contracts for the independently named recorded command-FK VR variant."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import feedback_vr_command_fk_v11 as expert
from feedback_vr_fk_v11 import mode1_input


class CommandFkContracts(unittest.TestCase):
    def fixture(self):
        n = 61
        root = np.array([.9, .1, -.2, .3]); root /= np.linalg.norm(root)
        roots = np.tile(root, (n, 1))
        xyz = np.tile([2., -1., .8], (n, 1))
        observed = np.tile([[.2, .3, -.1, 1., 0., 0., 0.],
                            [.2, -.3, -.1, .8, .6, 0., 0.]], (n, 1, 1))
        command = observed.copy()
        command[:, 0, :3] += [.03, .01, .04]
        command[:, 1, 3:] = [.8, 0., .6, 0.]
        local_quats = np.concatenate((observed[:, :, 3:], np.tile([[[1., 0, 0, 0]]], (n, 1, 1))), axis=1)
        quats = expert.quaternion_multiply(roots[:, None], local_quats)
        points = np.concatenate((observed[:, :, :3], np.tile([[[0., 0., .5]]], (n, 1, 1))), axis=1)
        points += np.einsum("nkij,kj->nki", expert.matrix_wxyz(local_quats), expert.TIP_OFFSETS)
        points = xyz[:, None] + np.einsum("nij,nkj->nki", expert.matrix_wxyz(roots), points)
        return np.zeros((n, 43)), xyz, roots, command.reshape(n, 14), observed.reshape(n, 14), points, quats

    def build(self, fixture):
        states, xyz, roots, command, observed, points, quats = fixture
        with patch.object(expert, "source_world_fk", return_value=(points, quats, {"fixture": True})):
            return expert.command_world_fk(None, states, xyz, roots, command, observed)

    def test_source_pose_recovers_command_local_pose_and_applies_offset_once(self):
        fixture = self.fixture()
        states, xyz, roots, command, _, points, quats = fixture
        new_points, new_quats, metadata = self.build(fixture)
        self.assertTrue(np.array_equal(new_points[:, 2], points[:, 2]))
        self.assertTrue(np.array_equal(new_quats[:, 2], quats[:, 2]))
        value = mode1_input(states[:, :29], np.zeros((61, 29)), roots, new_points, new_quats, 0, xyz[0], roots[0])
        poses = command.reshape(61, 2, 7)
        expected = poses[0, :, :3] + np.einsum("kij,kj->ki", expert.matrix_wxyz(poses[0, :, 3:]), expert.TIP_OFFSETS[:2])
        self.assertTrue(np.allclose(value[0, 890:896].reshape(2, 3), expected, atol=1e-7))
        self.assertTrue(np.allclose(value[0, 899:907].reshape(2, 4), poses[0, :, 3:], atol=1e-7))
        self.assertLess(metadata["observation_fk_xyz_max_abs_m"], 1e-14)
        self.assertLess(metadata["observation_fk_rotation_matrix_max_abs"], 1e-14)
        self.assertFalse(np.array_equal(new_points[:, :2], points[:, :2]))

    def test_only_wrists_change_and_inputs_are_not_mutated(self):
        fixture = self.fixture(); original = [value.copy() for value in fixture]
        states, xyz, roots, _, _, points, quats = fixture
        new_points, new_quats, _ = self.build(fixture)
        old = mode1_input(states[:, :29], np.zeros((61, 29)), roots, points, quats, 0, xyz[0], roots[0])
        new = mode1_input(states[:, :29], np.zeros((61, 29)), roots, new_points, new_quats, 0, xyz[0], roots[0])
        changed = np.zeros(1751, dtype=bool); changed[890:896] = True; changed[899:907] = True
        self.assertTrue(np.array_equal(old[0, ~changed], new[0, ~changed]))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(fixture, original)))
        self.assertTrue(np.allclose(np.linalg.norm(new_quats, axis=-1), 1))

    def test_wrong_recorded_frame_or_orientation_is_rejected(self):
        fixture = list(self.fixture())
        fixture[4][0, 0] += .01
        with self.assertRaisesRegex(ValueError, "pelvis-relative"):
            self.build(fixture)
        fixture = list(self.fixture())
        fixture[4][0, 3:7] = [.8, .6, 0., 0.]
        with self.assertRaisesRegex(ValueError, "pelvis-relative"):
            self.build(fixture)

    def test_missing_nonfinite_or_invalid_quaternion_is_rejected(self):
        fixture = list(self.fixture()); fixture[3] = fixture[3][:-1]
        with self.assertRaisesRegex(ValueError, "aligned finite"):
            self.build(fixture)
        fixture = list(self.fixture()); fixture[3][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "aligned finite"):
            self.build(fixture)
        fixture = list(self.fixture()); fixture[3][0, 3:7] *= 2
        with self.assertRaisesRegex(ValueError, "unit length"):
            self.build(fixture)


if __name__ == "__main__":
    unittest.main()
