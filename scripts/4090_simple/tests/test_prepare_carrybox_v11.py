"""CPU reference checks for the native-motion encoder and time/name mapping."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "prepare_carrybox_v11", Path(__file__).resolve().parents[1] / "prepare_carrybox_v11.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rotation_xyz(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def quaternion_xyz(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (np.cos(roll/2), np.sin(roll/2), np.cos(pitch/2),
                             np.sin(pitch/2), np.cos(yaw/2), np.sin(yaw/2))
    return np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])


class CarryBoxConversionContracts(unittest.TestCase):
    def test_names_preserve_body_and_swap_both_hand_finger_groups(self):
        np.testing.assert_array_equal(module.STATE_TO_TRAIN[:29], np.arange(29))
        np.testing.assert_array_equal(module.STATE_TO_TRAIN[29:],
                                      [29, 30, 31, 34, 35, 32, 33, 36, 37, 38, 41, 42, 39, 40])
        np.testing.assert_array_equal(module.HAND_TO_TRAIN, [0, 1, 2, 5, 6, 3, 4, 7, 8, 9, 12, 13, 10, 11])

    def test_encoder_matches_scalar_reference_and_zero_mode_mask(self):
        frames = 80
        q = np.arange(frames * 29, dtype=np.float64).reshape(frames, 29) / 100
        dq = q + 31
        roots = np.array([quaternion_xyz(.1 + i*.001, -.2, .7 + i*.01) for i in range(frames)])
        current = 7
        observed = module.encoder_input(q, dq, roots, current)[0]
        # Independent scalar reference using explicit names and Rz Ry Rx,
        # rather than the implementation's quaternion-to-matrix conversion.
        expected_q, expected_dq, expected_orientation = [], [], []
        yaw = .7 + current*.01
        heading_inverse = rotation_xyz(0, 0, -yaw)
        for offset in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45):
            i = current + offset
            for j in module.POLICY_TO_BODY:
                expected_q.append(q[i, j]); expected_dq.append(dq[i, j])
            r = heading_inverse @ rotation_xyz(.1 + i*.001, -.2, .7 + i*.01)
            expected_orientation.extend([r[0, 0], r[0, 1], r[1, 0], r[1, 1], r[2, 0], r[2, 1]])
        np.testing.assert_allclose(observed[4:294], expected_q, rtol=0, atol=1e-6)
        np.testing.assert_allclose(observed[294:584], expected_dq, rtol=0, atol=4e-6)
        np.testing.assert_allclose(observed[584:644], expected_orientation, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(observed[:4], 0)
        np.testing.assert_array_equal(observed[644:], 0)
        self.assertEqual(observed.shape, (1751,))
        self.assertFalse(np.allclose(observed[584:590], [1, 0, 0, 1, 0, 0]))

    def test_reference_is_absolute_and_no_tail_is_repeated(self):
        q = np.full((46, 29), .37)
        roots = np.tile([1., 0, 0, 0], (46, 1))
        value = module.encoder_input(q, np.zeros_like(q), roots, 0)
        np.testing.assert_allclose(value[0, 4:294], .37)
        with self.assertRaisesRegex(ValueError, "full real"):
            module.encoder_input(q, np.zeros_like(q), roots, 1)
        roots[0] *= 2
        with self.assertRaisesRegex(ValueError, "unit"):
            module.encoder_input(q, np.zeros_like(q), roots, 0)

    def test_derivative_uses_measured_time_with_real_endpoints(self):
        t = (np.arange(60) / 50).astype(np.float32).astype(np.float64)
        slopes = np.arange(1, 30)
        q = t[:, None] * slopes + .8
        np.testing.assert_allclose(module.derived_velocity(q, t), np.broadcast_to(slopes, q.shape), atol=1e-11)
        with self.assertRaises(ValueError):
            module.derived_velocity(q, np.arange(60) / 30)

    def test_causal_resample_has_exact_native_5hz_history_and_real_future(self):
        for length in (46, 47, 100, 1272, 1515):
            selected = module.selected_source_indices(length)
            np.testing.assert_array_equal(selected, np.arange(len(selected)) * 50 // 30)
            self.assertTrue(np.all(selected / 50 <= np.arange(len(selected)) / 30 + 1e-12))
            self.assertLess(selected[-1] + 45, length)
            for n in range(30, len(selected)):
                rows = n - np.array([30, 24, 18, 12, 6, 0])
                np.testing.assert_array_equal(np.diff(selected[rows]), 10)
        with self.assertRaises(ValueError):
            module.selected_source_indices(45)

    def test_split_is_disjoint_whole_episode_and_layout_contiguous(self):
        split = module.fixed_split()
        self.assertEqual([len(split[k]) for k in ("train", "val", "test")], [40, 5, 5])
        self.assertEqual(sorted(sum(split.values(), [])), list(range(50)))
        self.assertEqual(module.fixed_split(), split)
        self.assertEqual(module.ENCODER_LAYOUT[0][1], 0)
        self.assertEqual(module.ENCODER_LAYOUT[-1][2], 1751)
        for previous, following in zip(module.ENCODER_LAYOUT, module.ENCODER_LAYOUT[1:]):
            self.assertEqual(previous[2], following[1])

    def test_policy_permutation_matches_fixed_cpp_not_mjcf_tree_order(self):
        path = module.CPP_DIR / "include/policy_parameters.hpp"
        self.assertEqual(module.sha256(path), module.PARAMETERS_SHA)
        match = re.search(r"mujoco_to_isaaclab\s*=\s*\{([^}]+)\}", path.read_text())
        self.assertIsNotNone(match)
        expected = [int(x) for x in re.findall(r"\d+", match.group(1))]
        np.testing.assert_array_equal(module.POLICY_TO_BODY, expected)


if __name__ == "__main__":
    unittest.main()
