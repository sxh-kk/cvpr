"""CPU regressions for source-bound camera/robot frame calibration; no renderer."""
from copy import deepcopy
import unittest

import numpy as np

from simple.sensors.head_camera_mount import (
    CALIBRATED_HEAD_MESH_SHA256, HEAD_MESH_TO_D435_POSITION,
    HEAD_MESH_TO_D435_QUATERNION, USD_SHOULDER_PITCH_POSITIONS,
    USD_TORSO_TO_D435_POSITION, USD_WAIST_ROLL_POSITION,
    head_camera_mount, rotation, usd_torso_frame_position,
    usd_head_camera_frame_position, usd_wrist_frame_position,
)


def geometry(z=-.054, quaternion=(1., 0., 0., 0.)):
    return {"mesh_name": "head_link", "mesh_sha256": CALIBRATED_HEAD_MESH_SHA256,
            "position": [.0039635, 0., z], "quaternion": quaternion, "scale": [1., 1., 1.]}


class HeadCameraMountTests(unittest.TestCase):
    def test_current_and_legacy_head_origins_keep_their_actual_mounts(self):
        current, q = head_camera_mount([geometry()])
        legacy, _ = head_camera_mount([geometry(-.044)])
        np.testing.assert_allclose(current, [.05762349788045883, .01752999983727932, .41987000918388367], atol=1e-14)
        np.testing.assert_allclose(legacy - current, [0., 0., .01], atol=1e-14)
        np.testing.assert_allclose(rotation(q), rotation(HEAD_MESH_TO_D435_QUATERNION), atol=1e-14)

    def test_head_geometry_rotation_and_stereo_baseline_are_composed(self):
        headq = [np.cos(.3), 0., 0., np.sin(.3)]
        g = geometry(quaternion=headq)
        left, q = head_camera_mount([g])
        right, _ = head_camera_mount([g], [0., -.05, 0.])
        expected_rotation = rotation(headq) @ rotation(HEAD_MESH_TO_D435_QUATERNION)
        np.testing.assert_allclose(left, np.array(g['position']) + rotation(headq) @ HEAD_MESH_TO_D435_POSITION)
        np.testing.assert_allclose(rotation(q), expected_rotation, atol=1e-14)
        np.testing.assert_allclose(right - left, expected_rotation @ [0., -.05, 0.], atol=1e-14)
        self.assertAlmostEqual(np.linalg.norm(right-left), .05)

    def test_identical_visual_collision_transforms_allow_quaternion_sign(self):
        a = geometry(); b = deepcopy(a); b['quaternion'] = [-1., 0., 0., 0.]
        np.testing.assert_allclose(head_camera_mount([a, b])[0], head_camera_mount([a])[0])

    def test_missing_changed_scaled_and_conflicting_head_geometry_fail_closed(self):
        for geometries in ([], [dict(geometry(), mesh_sha256='changed')],
                           [dict(geometry(), scale=[1., 1., 2.])],
                           [geometry(), geometry(-.044)]):
            with self.subTest(geometries=geometries), self.assertRaises(ValueError):
                head_camera_mount(geometries)

    def test_torso_mapping_uses_fixed_joint_offsets_at_nonzero_waist_pose(self):
        yaw = rotation([np.cos(.2), 0., 0., np.sin(.2)])
        roll = yaw @ rotation([np.cos(.3), np.sin(.3), 0., 0.])
        mj_roll = np.array([-.0039635, 0., .035]); mj_torso = np.array([0., 0., .019])
        root = np.array([.1, -.2, .7]); actual_mj = root + yaw @ mj_roll + roll @ mj_torso
        expected_usd = root + yaw @ USD_WAIST_ROLL_POSITION
        mapped = usd_torso_frame_position(actual_mj, yaw, roll, mj_roll, mj_torso)
        np.testing.assert_allclose(mapped, expected_usd, atol=1e-14)
        self.assertGreater(np.linalg.norm(mapped - (actual_mj + roll @ [0., 0., -.01])), .001)

    def test_source_camera_and_wrist_mapping_share_the_actual_joint_chain(self):
        torso = np.array([.2, .1, .9]); usd_torso = torso + [.001, -.002, -.01]
        R = rotation([np.cos(.25), 0., np.sin(.25), 0.])
        left = usd_head_camera_frame_position(usd_torso, R)
        right = usd_head_camera_frame_position(usd_torso, R, [0., -.05, 0.])
        np.testing.assert_allclose(left, usd_torso + R @ USD_TORSO_TO_D435_POSITION)
        np.testing.assert_allclose(right-left, R @ [0., -.05, 0.], atol=1e-14)
        for side in ('left', 'right'):
            shoulder = USD_SHOULDER_PITCH_POSITIONS[side] - [0., 0., .01]
            wrist = np.array([.3, .4, .5])
            mapped = usd_wrist_frame_position(wrist, torso, usd_torso, R, shoulder, side)
            np.testing.assert_allclose(mapped, wrist + usd_torso - torso + R @ [0., 0., .01])

    def test_invalid_rotation_and_unknown_arm_mapping_are_rejected(self):
        with self.assertRaises(ValueError):
            usd_torso_frame_position([0., 0., 0.], np.zeros((3, 3)), np.eye(3), [0., 0., 0.], [0., 0., 0.])
        with self.assertRaises(ValueError):
            usd_wrist_frame_position([0., 0., 0.], [0., 0., 0.], [0., 0., 0.], np.eye(3), [0., 0., 0.], 'other')


if __name__ == '__main__':
    unittest.main()
