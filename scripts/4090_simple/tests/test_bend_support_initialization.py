"""Stable support is a reversible saved-scene revision, never live forcing."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bend_support_initialization as support

DATA = Path('/var/tmp/cvpr-bend-pick-ckpt5000-20261111/official_eval/data_root')


def geometry(local_com=(0., 0., 0.)):
    payload = {'contract': support.GEOMETRY_CONTRACT, 'status': 'passed', 'profile': support.PROFILE,
        'asset': {'res_id': 'graspnet1b', 'uid': '0', 'label': 'cracker_box'},
        'geometry': {'local_com_m': list(local_com), 'mass_kg': .1, 'geom_count': 2, 'body_name': 'cracker_box'}}
    return {**payload, 'manifest_identity': {'path': '/fixture/geometry.json', 'sha256': 'a'*64, 'size_bytes': 1},
        'validated_payload_sha256': support.digest(payload)}


def source(position=(-.326, -.04, .113), quaternion=(1., 0., 0., 0.), table_quaternion=(1., 0., 0., 0.)):
    pose = {'position': list(position), 'quaternion': list(quaternion)}
    table = {'size': [1.25, .79, .1], 'pose': {'position': [.3, 0., -.05], 'quaternion': list(table_quaternion)},
        'material': {'name': 'unchanged'}}
    return {'uid': 'g1_wholebody_bend_pick_mp', 'robot_cfg': {'uid': 'g1_wholebody'},
        'metadata': {'physics_dt': .002, 'max_episode_steps': 400, 'split': 'train'},
        'dr_state_dict': {'spatial': {'0': deepcopy(pose), 'g1_wholebody': {'position': [-.6, 0., .38]},
            '6': {'position': [1., 2., 3.]}}, 'material': {'robot': 'unchanged'}, 'lighting': {'all': 'unchanged'}},
        'layout': {'actors': {'target': {'pose': deepcopy(pose), 'asset': {'res_id': 'graspnet1b',
            'uid': '0', 'label': 'cracker_box'}, 'material': {'roughness': .25}}, 'table': table,
            'distractor_0': {'pose': {'position': [1., 2., 3.], 'quaternion': [1., 0., 0., 0.]}}}}}


def reverse(corrected, receipt):
    restored = deepcopy(corrected)
    for item in receipt['changed_fields']:
        cursor = restored
        for key in item['path'][:-1]: cursor = cursor[key]
        assert cursor[item['path'][-1]] == item['after']
        cursor[item['path'][-1]] = item['before']
    return restored


class SupportProjectionTests(unittest.TestCase):
    def test_minimal_xy_only_move_is_reversible_and_keeps_all_other_source_values(self):
        original = source(); before = deepcopy(original)
        corrected, receipt = support.derive_stable_support_reset(original, geometry())
        self.assertTrue(receipt['applied'])
        np.testing.assert_allclose(receipt['delta_world_xyz_m'], [.006, 0., 0.], atol=1e-16)
        self.assertEqual(original, before)
        self.assertEqual(reverse(corrected, receipt), original)
        self.assertEqual(receipt['source_config_sha256'], support.digest(original))
        self.assertEqual(receipt['corrected_config_sha256'], support.digest(corrected))
        self.assertEqual(len(receipt['changed_fields']), 2)
        self.assertEqual(corrected['layout']['actors']['target']['pose']['position'][2], .113)
        self.assertEqual(corrected['layout']['actors']['target']['pose']['quaternion'], [1., 0., 0., 0.])
        self.assertGreaterEqual(min(receipt['after_edge_clearances_m'].values()), .005-1e-15)
        self.assertFalse(receipt['warmup_modified'] or receipt['reward_modified'] or receipt['collision_modified'] or receipt['live_state_writes'])

    def test_interior_noop_is_exact_and_does_not_round_original_pose(self):
        original = source(position=(-.30, -.012345678901, .113))
        corrected, receipt = support.derive_stable_support_reset(original, geometry())
        self.assertEqual(corrected, original)
        self.assertFalse(receipt['applied'])
        self.assertEqual(receipt['changed_fields'], [])
        self.assertEqual(receipt['delta_world_xyz_m'], [0., 0., 0.])
        self.assertEqual(receipt['source_config_sha256'], receipt['corrected_config_sha256'])

    def test_uses_rotated_true_com_and_table_frame_not_target_origin(self):
        # yaw90 maps local COM+x to world+y. Correct only the violated local table side.
        q = [math.sqrt(.5), 0., 0., math.sqrt(.5)]
        original = source(position=(.3, -.63, .113), quaternion=q, table_quaternion=q)
        corrected, receipt = support.derive_stable_support_reset(original, geometry((.003, 0., .009)))
        self.assertAlmostEqual(receipt['before_com_xyz_m'][1], -.627)
        self.assertAlmostEqual(receipt['after_com_xyz_m'][1], -.620)
        np.testing.assert_allclose(receipt['delta_world_xyz_m'], [0., .007, 0.], atol=1e-15)
        self.assertEqual(reverse(corrected, receipt), original)

    def test_all_four_edges_and_corner_get_unique_nearest_rectangle_projection(self):
        for position, expected in [((-.4, 0., .1), (-.32, 0.)), ((1., 0., .1), (.92, 0.)),
            ((.3, -.5, .1), (.3, -.39)), ((.3, .5, .1), (.3, .39)), ((-.4, -.5, .1), (-.32, -.39))]:
            corrected, receipt = support.derive_stable_support_reset(source(position), geometry())
            np.testing.assert_allclose(corrected['layout']['actors']['target']['pose']['position'][:2], expected, atol=1e-15)
            self.assertEqual(receipt['delta_world_xyz_m'][2], 0.)

    def test_reject_bad_geometry_mutation_pose_disagreement_tilted_table_and_wrong_frame(self):
        g = geometry(); g['geometry']['local_com_m'][0] = .01
        with self.assertRaisesRegex(ValueError, 'unchanged validated'): support.derive_stable_support_reset(source(), g)
        mutations = [lambda s: s['layout']['actors']['target']['asset'].update(uid='1'),
            lambda s: s['dr_state_dict']['spatial']['0']['position'].__setitem__(0, 0.),
            lambda s: s['robot_cfg'].update(uid='g1_sonic'),
            lambda s: s['layout']['actors']['table']['pose'].update(quaternion=[math.cos(.1), math.sin(.1), 0., 0.]),
            lambda s: s['layout']['actors']['table'].update(size=[.001, 1., .1]),
            lambda s: s['layout']['actors']['target']['pose'].update(position=[float('nan'), 0., .1])]
        for mutate in mutations:
            original = source(); mutate(original)
            with self.assertRaises(ValueError): support.derive_stable_support_reset(original, geometry())


@unittest.skipUnless(DATA.is_dir(), 'Optional actual official asset fixture absent')
class ActualGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temporary.name)/'GEOMETRY.json'
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            cls.geometry = support.prepare_geometry_manifest(cls.path, DATA)

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def test_actual_builder_compiles_same_inertial_com_as_independent_full_model(self):
        expected = [-.0024414519334245064, -.0016307824790363738, .009015970338031197]
        np.testing.assert_array_equal(self.geometry['geometry']['local_com_m'], expected)
        self.assertEqual((self.geometry['geometry']['mass_kg'], self.geometry['geometry']['geom_count']), (.1, 2))
        self.assertEqual(support.load_geometry_manifest(self.path, DATA), self.geometry)
        self.assertEqual(len(self.geometry['collision_meshes']), 2)

    def test_tampered_COM_mesh_order_source_hash_and_existing_output_rejected(self):
        original = json.loads(self.path.read_text())
        mutations = [lambda m: m['geometry']['local_com_m'].__setitem__(0, .1),
            lambda m: m['collision_meshes'].reverse(), lambda m: m['source_identities']['engines/mujoco.py'].update(sha256='0'*64)]
        for index, mutate in enumerate(mutations):
            modified = deepcopy(original); mutate(modified)
            path = Path(self.temporary.name)/f'tampered{index}.json'; path.write_text(json.dumps(modified))
            with self.assertRaises(ValueError): support.load_geometry_manifest(path, DATA)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), self.assertRaisesRegex(ValueError, 'fresh'):
            support.prepare_geometry_manifest(self.path, DATA)

    def test_uniform_rule_corrects_exact_two_of_actual30_and_is_fully_reversible(self):
        from run_closed_loop import derive_bend_eval_reset
        affected = []
        for level in range(3):
            path = DATA/'eval/G1WholebodyBendPickMP-v0'/f'dr-level-{level}/meta/episodes.jsonl'
            for line in path.read_text().splitlines():
                row = json.loads(line); native = json.loads(row['environment_config'])
                compat, _ = derive_bend_eval_reset(native)
                corrected, receipt = support.derive_stable_support_reset(compat, self.geometry)
                self.assertEqual(reverse(corrected, receipt), compat)
                if receipt['applied']: affected.append((level, row['episode_index'], receipt['delta_world_xyz_m']))
        self.assertEqual([(l,i) for l,i,_ in affected], [(2,3), (2,8)])
        np.testing.assert_allclose([r[2][0] for r in affected], [.00941892197387384, .007974049581563114], atol=1e-15)

    def compile_actual(self, case_index=3):
        import mujoco
        from simple.core.actor import ObjectActor
        from simple.core.types import Pose
        from simple.engines.mujoco import MujocoSimulator
        from run_closed_loop import derive_bend_eval_reset
        row = json.loads((DATA/'eval/G1WholebodyBendPickMP-v0/dr-level-2/meta/episodes.jsonl').read_text().splitlines()[case_index])
        compatible, _ = derive_bend_eval_reset(json.loads(row['environment_config']))
        corrected, receipt = support.derive_stable_support_reset(compatible, self.geometry)
        target = corrected['layout']['actors']['target']; table = corrected['layout']['actors']['table']
        asset = support._asset('graspnet1b', '0', DATA); actor = ObjectActor(asset)
        actor.pose = Pose(position=(np.array(target['pose']['position'])+[0.,0.,.405]).tolist(), quaternion=target['pose']['quaternion'])
        spec = mujoco.MjSpec(); spec.option.timestep = .005
        MujocoSimulator._build_object(None, spec, spec.worldbody, actor)
        spec.worldbody.add_geom(name='table_geom', type=mujoco.mjtGeom.mjGEOM_BOX,
            size=np.array(table['size'])/2, pos=np.array(table['pose']['position'])+[0.,0.,.405], quat=table['pose']['quaternion'])
        model = spec.compile(); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        return model, data, receipt

    def test_before_physics_gate_binds_real_translated_geometry_receipt_and_is_read_only(self):
        import mujoco
        for index in (0, 3, 8):
            model, data, receipt = self.compile_actual(index)
            snapshot = {name: np.array(getattr(data, name)).copy() for name in ('qpos','qvel','qacc','ctrl','xpos','xipos','xmat')}
            report = support.verify_actual_support_model(model, data, receipt)
            self.assertEqual(report['initialization_receipt_sha256'], support.digest(receipt))
            self.assertEqual((report['status'], report['read_only'], report['physics_steps_taken']), ('passed', True, 0))
            self.assertEqual(float(data.time), 0.)
            for name, values in snapshot.items(): np.testing.assert_array_equal(getattr(data, name), values)
            # The gate must not silently fix or forward altered state.
            data.time = .005
            with self.assertRaisesRegex(ValueError, 'precede all physics'): support.verify_actual_support_model(model, data, receipt)
            data.time = 0.
            model.body_ipos[model.body('cracker_box').id, 0] += .001
            with self.assertRaisesRegex(ValueError, 'local COM'): support.verify_actual_support_model(model, data, receipt)

    def test_before_physics_gate_rejects_wrong_target_table_translation_or_mass(self):
        for kind in ('target', 'table', 'mass', 'translation'):
            model, data, receipt = self.compile_actual()
            if kind == 'target': data.xpos[model.body('cracker_box').id, 0] += .001
            if kind == 'table': data.geom_xpos[model.geom('table_geom').id, 0] += .001
            if kind == 'mass': model.body_mass[model.body('cracker_box').id] += .01
            if kind == 'translation': receipt['actual_world_translation_m'][2] = .785
            with self.subTest(kind=kind), self.assertRaises(ValueError): support.verify_actual_support_model(model, data, receipt)


if __name__ == '__main__': unittest.main()
