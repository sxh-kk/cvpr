"""Pinned PICO VR3pt math and measured-FK/root attachment, with no simulation."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pico_mobile_vr_reference as vr
from handover_sonic_teleop_reference import build_teleop_reference


def fixture():
    rng = np.random.default_rng(20261027)
    body = np.c_[rng.normal(size=(24, 3)) * .3 + [0., 1., 0.],
                 Rotation.random(24, random_state=rng).as_quat()]
    root = np.r_[[.8, -.7, .77], Rotation.from_euler('xyz', [.07, -.05, .6]).as_quat(scalar_first=True)]
    root_rotation = Rotation.from_quat(root[3:], scalar_first=True)
    local_positions = np.array([[.2, .23, .17], [.24, -.25, .19], [.01, -.02, .065]])
    local_rotations = Rotation.from_euler('xyz', [[.1, -.3, .2], [-.2, .15, -.25], [.08, -.12, .18]])
    return body, root, root_rotation.apply(local_positions) + root[:3], (
        root_rotation * local_rotations).as_quat(scalar_first=True)


def official_math():
    """Execute only pinned pure functions/constant AST; never import manager."""
    path = vr.SOURCE_PATHS['pico_manager']
    parsed = ast.parse(path.read_text())
    selected = []
    for node in parsed.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'OFFSETS'
                                               for target in node.targets):
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in ('_compute_rel_transform', '_process_3pt_pose'):
            selected.append(node)
    if len(selected) != 3:
        raise AssertionError('Pinned pure PICO AST extraction must contain exactly three definitions')
    namespace = {'np': np, 'sRot': Rotation}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_process_3pt_pose']


class MobileVRReferenceTests(unittest.TestCase):
    def assert_rotations_equal(self, actual_wxyz, expected_wxyz, atol=1e-7):
        delta = Rotation.from_quat(actual_wxyz, scalar_first=True).inv() * Rotation.from_quat(
            expected_wxyz, scalar_first=True)
        self.assertLessEqual(float(np.max(delta.magnitude())), atol)

    def test_raw_three_points_match_pinned_official_AST_without_input_mutation(self):
        reference = official_math()
        body, *_ = fixture()
        original = body.copy()
        for count in range(6):
            current = body.copy()
            current[:, :3] += [.13 * count, -.1 * count, .2 * count]
            rigid = Rotation.from_euler('xyz', [.07 * count, -.12 * count, .09 * count])
            current[:, 3:] = (rigid * Rotation.from_quat(current[:, 3:])).as_quat()
            before = current.copy()
            np.testing.assert_array_equal(vr.body24_to_root_local_vr(current), reference(current))
            np.testing.assert_array_equal(current, before)
        np.testing.assert_array_equal(body, original)

    def test_actual_offsets_once_and_all_three_initial_targets_match_measured_FK(self):
        args = fixture()
        before = deepcopy(args)
        helper = vr.MobileVRReference()
        receipt = helper.calibrate(*args)
        points, quats, evidence = helper.prepare(*args[:2])
        expected = args[2] + Rotation.from_quat(args[3], scalar_first=True).apply(vr.VR_OFFSETS)
        np.testing.assert_allclose(points, expected, atol=2e-15, rtol=0)
        self.assert_rotations_equal(quats, args[3], atol=1e-14)
        self.assertFalse(receipt['official_identity_neck_and_fixed_005m_torso_assumed'])
        self.assertTrue(receipt['tip_offsets_applied_once_to_actual_body_origins'])
        self.assertEqual(evidence['neck_calibration'],
                         'measured_actual_torso_FK_origin_and_orientation_adaptation')
        for old, new in zip(before, args):
            np.testing.assert_array_equal(old, new)

    def test_measured_neck_is_not_silently_replaced_with_identity_and_fixed_height(self):
        args = fixture()
        helper = vr.MobileVRReference()
        helper.calibrate(*args)
        _, _, evidence = helper.prepare(*args[:2])
        neck_local = np.array(evidence['desired_quaternions_root_local_wxyz'])[2]
        self.assertGreater(Rotation.from_quat(neck_local, scalar_first=True).magnitude(), .1)
        self.assertGreater(np.linalg.norm(np.array(evidence['desired_points_root_local'])[2] - [0, 0, .4]), .02)
        changed = args[0].copy()
        changed[12, 3:] = (Rotation.from_rotvec([.05, 0, 0]) * Rotation.from_quat(changed[12, 3:])).as_quat()
        _, _, second = helper.prepare(changed, args[1])
        self.assertGreater(np.linalg.norm(np.array(second['desired_points_root_local'])[2]
                                         - evidence['desired_points_root_local'][2]), .005)

    def test_robot_translation_yaw_roll_pitch_carry_targets_without_world_anchor(self):
        args = fixture()
        helper = vr.MobileVRReference()
        helper.calibrate(*args)
        first_points, _, first = helper.prepare(*args[:2])
        moved_root = np.r_[[-1.3, 1.2, .82], Rotation.from_euler('xyz', [-.12, .1, 2.8]).as_quat(scalar_first=True)]
        points, quats, evidence = helper.prepare(args[0], moved_root)
        np.testing.assert_array_equal(evidence['desired_points_root_local'], first['desired_points_root_local'])
        np.testing.assert_array_equal(evidence['desired_quaternions_root_local_wxyz'],
                                      first['desired_quaternions_root_local_wxyz'])
        self.assertGreater(float(np.max(np.linalg.norm(points - first_points, axis=1))), 1.)
        packed, _ = build_teleop_reference(np.zeros((46, 12)), np.zeros((46, 12)), 0,
            moved_root, points, quats, moved_root[3:])
        np.testing.assert_allclose(packed[0, 890:899].reshape(3, 3),
                                   evidence['desired_points_root_local'], atol=5e-8, rtol=0)
        self.assert_rotations_equal(packed[0, 899:911].reshape(3, 4),
                                    evidence['desired_quaternions_root_local_wxyz'])
        self.assertFalse(packed[0, 4:644].any())
        self.assertFalse(packed[0, 911:].any())

    def test_common_human_room_translation_and_rotation_do_not_pull_robot_targets(self):
        args = fixture()
        helper = vr.MobileVRReference()
        helper.calibrate(*args)
        points, quats, _ = helper.prepare(*args[:2])
        changed = args[0].copy()
        rigid = Rotation.from_euler('xyz', [.2, -.7, .1])
        changed[:, :3] = rigid.apply(changed[:, :3]) + [.5, .2, -.3]
        changed[:, 3:] = (rigid * Rotation.from_quat(changed[:, 3:])).as_quat()
        points2, quats2, _ = helper.prepare(changed, args[1])
        np.testing.assert_allclose(points2, points, atol=2e-7, rtol=0)
        self.assert_rotations_equal(quats2, quats, atol=3e-7)

    def test_live_left_wrist_changes_without_freezing_or_changing_other_points(self):
        args = fixture()
        helper = vr.MobileVRReference()
        helper.calibrate(*args)
        points, quats, _ = helper.prepare(*args[:2])
        changed = args[0].copy()
        changed[22, :3] += [.08, -.04, .06]
        changed[22, 3:] = (Rotation.from_rotvec([.1, -.2, .05]) * Rotation.from_quat(changed[22, 3:])).as_quat()
        points2, quats2, evidence = helper.prepare(changed, args[1])
        self.assertGreater(np.linalg.norm(points2[0] - points[0]), .05)
        self.assertGreater((Rotation.from_quat(quats2[0], scalar_first=True).inv()
                            * Rotation.from_quat(quats[0], scalar_first=True)).magnitude(), .1)
        np.testing.assert_array_equal(points2[1:], points[1:])
        self.assert_rotations_equal(quats2[1:], quats[1:], atol=1e-14)
        self.assertFalse(evidence['raw_Head_or_Controller_poses_substituted'])

    def test_equivalent_quaternion_signs_preserve_targets(self):
        args = fixture()
        helper = vr.MobileVRReference()
        helper.calibrate(*args)
        points, quats, _ = helper.prepare(*args[:2])
        changed = args[0].copy()
        changed[:, 3:] *= -1
        root = args[1].copy()
        root[3:] *= -1
        points2, quats2, _ = helper.prepare(changed, root)
        np.testing.assert_allclose(points2, points, atol=1e-14, rtol=0)
        self.assert_rotations_equal(quats2, quats, atol=1e-14)

    def test_receipts_are_JSON_and_returned_values_do_not_mutate_next_preparation(self):
        args = fixture()
        helper = vr.MobileVRReference()
        receipt = helper.calibrate(*args)
        self.assertEqual(receipt['calibration_sha256'], helper.calibration['calibration_sha256'])
        receipt['actual_tip_points_world'][0][0] = 1000.
        points, quats, evidence = helper.prepare(*args[:2])
        expected_points, expected_quats = points.copy(), quats.copy()
        json.dumps(evidence, allow_nan=False)
        evidence['calibration']['calibration_sha256'] = 'changed'
        points[:] = 0.
        quats[:] = 0.
        p2, q2, second = helper.prepare(*args[:2])
        np.testing.assert_array_equal(p2, expected_points)
        np.testing.assert_array_equal(q2, expected_quats)
        self.assertEqual(second['calibration_sha256'], helper.calibration['calibration_sha256'])
        self.assertFalse(second['tracking_and_freshness_checked_by_helper'])
        self.assertFalse(second['model_inference_performed'])
        self.assertEqual(second['physics_steps_performed'], 0)
        self.assertFalse(second['physical_or_task_success_certified'])

    def test_explicit_recalibration_uses_new_actual_FK_and_invalid_calibration_is_atomic(self):
        args = fixture()
        helper = vr.MobileVRReference()
        initial = helper.calibrate(*args)
        invalid = deepcopy(args)
        invalid[3][0, 0] = np.nan
        with self.assertRaises(ValueError):
            helper.calibrate(*invalid)
        self.assertEqual(helper.calibration['calibration_sha256'], initial['calibration_sha256'])
        updated = deepcopy(args)
        updated[2][0] += [.07, .02, -.01]
        second = helper.calibrate(*updated)
        points, _, _ = helper.prepare(*updated[:2])
        np.testing.assert_allclose(points, updated[2] + Rotation.from_quat(
            updated[3], scalar_first=True).apply(vr.VR_OFFSETS), atol=2e-15, rtol=0)
        self.assertNotEqual(second['calibration_sha256'], initial['calibration_sha256'])

    def test_missing_calibration_and_invalid_body_root_or_actual_FK_fail_closed(self):
        args = fixture()
        helper = vr.MobileVRReference()
        with self.assertRaisesRegex(ValueError, 'calibration'):
            helper.prepare(*args[:2])
        mutations = [lambda a: a.__setitem__(0, a[0][:23]),
                     lambda a: a[0].__setitem__((22, 0), np.nan),
                     lambda a: a[0].__setitem__((12, slice(3, 7)), 0.),
                     lambda a: a[1].__setitem__(0, np.inf),
                     lambda a: a[1].__setitem__(slice(3, 7), 0.),
                     lambda a: a.__setitem__(2, np.zeros((2, 3))),
                     lambda a: a[3].__setitem__((0, 0), 2.)]
        for mutate in mutations:
            invalid = list(deepcopy(args))
            mutate(invalid)
            with self.assertRaises(ValueError):
                helper.calibrate(*invalid)
            self.assertIsNone(helper.calibration)
        helper.calibrate(*args)
        invalid = args[0].copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            helper.prepare(invalid, args[1])
        helper.calibration = None
        with self.assertRaisesRegex(ValueError, 'calibration'):
            helper.prepare(*args[:2])

    def test_source_pin_mismatch_is_not_silently_accepted(self):
        identities = vr.verify_official_sources()
        self.assertTrue(identities['official_sources_rehashed'])
        self.assertEqual(len(identities['source_sha256']), 2)
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / 'changed.py'
            changed.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                vr.verify_official_sources(dict(vr.SOURCE_PATHS, pico_manager=changed))


if __name__ == '__main__':
    unittest.main()
