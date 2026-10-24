"""Official-source mapping tests; no model, simulation or GPU is constructed."""
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import handover_sonic_teleop_reference as teleop


def euler(roll, pitch, yaw):
    x = [math.cos(roll / 2), math.sin(roll / 2), 0., 0.]
    y = [math.cos(pitch / 2), 0., math.sin(pitch / 2), 0.]
    z = [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]
    return teleop.multiply_wxyz(z, teleop.multiply_wxyz(y, x))


def fixture():
    q = np.arange(48 * 12, dtype=float).reshape(48, 12) / 100
    return [q, -q / 10, 2, np.r_[[1., -2., .7], euler(.3, -.2, .6)],
        np.array([[1.4, -1.8, 1.], [.9, -2.3, .9], [1.1, -2., 1.5]]),
        np.stack([euler(.1, .3, -.2), euler(-.4, .2, .7), euler(.1, -.1, .2)]), euler(.15, -.25, .4)]


class TeleopReferenceTests(unittest.TestCase):
    def test_layout_leg_order_future_samples_and_inactive_branches(self):
        values = fixture(); packed, evidence = teleop.build_teleop_reference(*values)
        self.assertEqual(packed.shape, (1, 1751)); self.assertEqual(packed.dtype, np.float32)
        self.assertTrue(np.array_equal(packed[0, :4], [1, 0, 0, 0]))
        idx = 2 + np.arange(10) * 5
        self.assertTrue(np.array_equal(packed[0, 650:770], values[0][idx].reshape(-1).astype(np.float32)))
        self.assertTrue(np.array_equal(packed[0, 770:890], values[1][idx].reshape(-1).astype(np.float32)))
        self.assertFalse(packed[0, 4:644].any()); self.assertFalse(packed[0, 911:].any())
        self.assertEqual(evidence['future_reference_indices'], idx.tolist())
        self.assertFalse(evidence['physics_steps_performed']); self.assertFalse(evidence['model_inference_performed'])

    def test_encoder_mode_is_scalar_plus_fillers(self):
        self.assertTrue(np.array_equal(teleop.encoder_mode4(2), [2, 0, 0, 0]))
        for invalid in (True, 3, 1., -1):
            with self.assertRaises(ValueError): teleop.encoder_mode4(invalid)

    def test_VR_full_roll_pitch_yaw_inverse_and_quaternion_order(self):
        args = fixture(); root = args[3]; r = teleop.rotation(root[3:])
        local_points = np.array([[.3, .2, .1], [.4, -.2, .15], [.0, .0, .8]])
        local_q = np.stack([euler(.1, -.2, .3), euler(-.3, .2, -.4), euler(.2, .1, .0)])
        args[4] = local_points @ r.T + root[:3]
        args[5] = np.stack([teleop.multiply_wxyz(root[3:], q) for q in local_q])
        value, _ = teleop.build_teleop_reference(*args)
        self.assertTrue(np.allclose(value[0, 890:899], local_points.reshape(-1), atol=1e-7))
        self.assertTrue(np.allclose(value[0, 899:911], local_q.reshape(-1), atol=1e-7))
        yaw_only_wrong = (args[4] - root[:3]) @ teleop.rotation(teleop.heading_wxyz(root[3:]))
        self.assertFalse(np.allclose(local_points, yaw_only_wrong, atol=.01))

    def test_explicit_offsets_left_right_mirror_torso_and_no_double_offset(self):
        origins = np.array([[1., 0., .8], [2., 0., .8], [0., 0., 1.]])
        quats = np.stack([euler(0, 0, math.pi/2), euler(0, 0, -math.pi/2), euler(math.pi/2, 0, 0)])
        points, point_q = teleop.vr_points_from_body_poses(origins, quats)
        self.assertTrue(np.allclose(points - origins, [[.025, .18, 0], [.025, -.18, 0], [0, -.35, 0]], atol=1e-12))
        args = fixture(); args[3] = np.array([0., 0., 0., 1., 0., 0., 0.]); args[4:6] = [points, point_q]
        value, evidence = teleop.build_teleop_reference(*args)
        self.assertTrue(np.allclose(value[0, 890:899], points.reshape(-1), atol=1e-7))
        self.assertFalse(evidence['offsets_applied_by_builder'])

    def test_anchor_heading_alignment_is_separate_first_two_columns_rowwise(self):
        args = fixture(); delta = euler(0., 0., -.3)
        value, _ = teleop.build_teleop_reference(*args, apply_delta_heading_wxyz=delta)
        expected = teleop.rotation(teleop.heading_wxyz(args[3][3:])).T @ teleop.rotation(delta) @ teleop.rotation(args[6])
        self.assertTrue(np.allclose(value[0, 644:650], expected[:, :2].reshape(-1), atol=1e-7))
        self.assertFalse(np.allclose(value[0, 644:650], expected[:2].reshape(-1), atol=.01))
        aligned = teleop.compute_apply_delta_heading(euler(.1, .2, .7), euler(-.2, .1, -.4), .2)
        self.assertTrue(np.allclose(teleop.rotation(aligned), teleop.rotation(euler(0., 0., 1.3)), atol=1e-12))

    def test_pure_deterministic_and_rejects_invalid_or_missing_future(self):
        args = fixture(); before = deepcopy(args)
        first, evidence = teleop.build_teleop_reference(*args); second, same = teleop.build_teleop_reference(*args)
        self.assertTrue(np.array_equal(first, second)); self.assertEqual(evidence, same)
        for original, value in zip(before, args): self.assertTrue(np.array_equal(original, value))
        for mutate in (lambda a: a.__setitem__(2, 3), lambda a: a.__setitem__(2, True),
                       lambda a: a[0].__setitem__((0, 0), np.nan),
                       lambda a: a[3].__setitem__(3, 2.), lambda a: a.__setitem__(4, np.zeros((2, 3)))):
            bad = deepcopy(args); mutate(bad)
            with self.assertRaises(ValueError): teleop.build_teleop_reference(*bad)
        with self.assertRaises(ValueError): teleop.build_teleop_reference(*args, lower_joint_names=list(reversed(teleop.LOWER_BODY_NAMES)))
        with self.assertRaises(ValueError): teleop.build_teleop_reference(*args, apply_delta_heading_wxyz=euler(.1, 0, 0))

    def test_actual_source_SHA_verification_fails_on_changed_artifact(self):
        receipt = teleop.verify_official_sources()
        self.assertTrue(receipt['official_sources_rehashed']); self.assertEqual(len(receipt['source_sha256']), 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.hpp'; path.write_text('changed')
            with self.assertRaises(ValueError): teleop.verify_official_sources({**teleop.SOURCE_PATHS, 'math_header': path})

    @unittest.skipUnless(shutil.which('g++'), 'Official C++ math comparison needs a CPU C++ compiler')
    def test_full1751_against_compiled_official_math_and_joint_mapping(self):
        # Gather loops below follow the pinned C++ source; quaternion/heading
        # algebra and joint index arrays execute the actual official headers.
        program = r'''
#include <array>
#include <tuple>
#include <vector>
#include <iostream>
#include <iomanip>
#include "math_utils.hpp"
#include "policy_parameters.hpp"
int main() {
  double q[48][29], dq[48][29];
  for(auto &frame:q)for(double &v:frame)std::cin>>v;
  for(auto &frame:dq)for(double &v:frame)std::cin>>v;
  std::array<double,3> root_pos;for(auto &v:root_pos)std::cin>>v;
  std::array<double,4> root, ref, delta;
  for(auto &v:root)std::cin>>v;
  std::array<std::array<double,3>,3> points;for(auto &p:points)for(auto &v:p)std::cin>>v;
  std::array<std::array<double,4>,3> orn;for(auto &p:orn)for(auto &v:p)std::cin>>v;
  for(auto &v:ref)std::cin>>v;for(auto &v:delta)std::cin>>v;
  std::vector<double> out(1751,0.);out[0]=1;
  auto aq=quat_mul_d(quat_conjugate_d(calc_heading_quat_d(root)),quat_mul_d(delta,ref));
  auto mat=quat_to_rotation_matrix_d(aq);
  for(int r=0;r<3;r++)for(int c=0;c<2;c++)out[644+r*2+c]=mat[r][c];
  for(int frame=0;frame<10;frame++) {
    std::array<double,29> policy_q{},policy_dq{};
    for(int j=0;j<29;j++) {policy_q[isaaclab_to_mujoco[j]]=q[2+frame*5][j];policy_dq[isaaclab_to_mujoco[j]]=dq[2+frame*5][j];}
    for(int j=0;j<12;j++) {int k=lower_body_joint_mujoco_order_in_isaaclab_index[j];out[650+frame*12+j]=policy_q[k];out[770+frame*12+j]=policy_dq[k];}
  }
  auto inv=quat_conjugate_d(root);
  for(int p=0;p<3;p++) {
    std::array<double,3> rel;for(int j=0;j<3;j++)rel[j]=points[p][j]-root_pos[j];
    auto local=quat_rotate_d(inv,rel);auto localq=quat_mul_d(inv,orn[p]);
    for(int j=0;j<3;j++)out[890+p*3+j]=local[j];for(int j=0;j<4;j++)out[899+p*4+j]=localq[j];
  }
  for(auto v:out)std::cout<<std::setprecision(17)<<v<<" ";
}
'''
        args = fixture(); q29 = np.sin(np.arange(48*29).reshape(48, 29) / 23)
        dq29 = np.cos(np.arange(48*29).reshape(48, 29) / 29)
        args[:2] = [q29[:, :12], dq29[:, :12]]; delta = euler(0., 0., -.3)
        packed, _ = teleop.build_teleop_reference(*args, apply_delta_heading_wxyz=delta)
        payload = np.r_[q29.reshape(-1), dq29.reshape(-1), args[3], args[4].reshape(-1), args[5].reshape(-1), args[6], delta]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'official_math.cpp'; executable = Path(directory) / 'official_math'
            source.write_text(program)
            subprocess.run(['g++', '-std=c++17', '-O0', '-I', str(teleop.DEPLOY / 'include'), str(source), '-o', str(executable)], check=True, capture_output=True, timeout=30)
            completed = subprocess.run([str(executable)], input=' '.join(map(repr, payload.tolist())), text=True,
                                       capture_output=True, check=True, timeout=10)
        expected = np.fromstring(completed.stdout, sep=' ').astype(np.float32)
        self.assertEqual(expected.shape, (1751,))
        self.assertTrue(np.allclose(packed[0], expected, rtol=0, atol=1e-7), np.max(np.abs(packed[0]-expected)))


if __name__ == '__main__': unittest.main()
