import json
from pathlib import Path
import numpy as np
import pytest
from egomotion_pipeline.body_model import BodyModel
from egomotion_pipeline.adapters import discover_egosuite, read_egosuite, read_hm
from egomotion_pipeline.geometry import quat_to_matrix, axis_angle_to_quat
from egomotion_pipeline.schema import PARENTS
from egomotion_pipeline.aria import capture_timestamps
from conftest import make_egosuite


def test_fk_shape_conditioning_root_pivot_and_ancestor_validity(model_path):
    model = BodyModel(model_path)
    trans = np.array([[1.,0,0], [1.,0,0]])
    root = np.array([[0,0,np.pi/2], [0,0,0]])
    pose = np.zeros((2,63)); pose[1, (16-1)*3] = np.nan
    beta = np.array([[2.,3.], [1.,0.]])
    pos, rot, mask = model.decode(trans, root, pose, beta, [True,True])
    rest = model.rest+np.einsum('b,jcb->jc', beta[0], model.shape_basis)
    assert np.allclose(pos[0,0], rest[0]+trans[0])
    expected = rest[0]+trans[0]+quat_to_matrix(axis_angle_to_quat(root[0]))@(rest[1]-rest[0])
    assert np.allclose(pos[0,1], expected)
    assert not mask[1,16] and not mask[1,18] and not mask[1,20]
    assert mask[1,17] and mask[1,19] and mask[1,21]
    assert np.allclose(quat_to_matrix(rot[0,21]), quat_to_matrix(axis_angle_to_quat(root[0])))
    with pytest.raises(ValueError, match="shape bases"):
        model.decode(trans, root, pose, np.zeros((2,16)), [True,True])
    padded = BodyModel(model_path, pad_shape_coefficients=True)
    p2, _, _ = padded.decode(trans, root, np.zeros((2,63)), np.pad(beta, ((0,0),(0,14))), [True,True])
    assert np.allclose(pos[0], p2[0])


def write_hm(path, count=4):
    path.parent.mkdir(parents=True)
    joints = np.broadcast_to(np.arange(67)[None,:,None], (count,67,3)).copy().astype(float)
    np.savez(path, trans=np.zeros((1,count,3)), root_orient=np.zeros((1,count,3)),
             pose_body=np.zeros((1,count,63)), betas_per_frame=np.zeros((1,count,2)),
             valid=np.array([[1,0,1,1]]), joints3d=joints[None],
             cam_R=np.broadcast_to(np.eye(3), (5,count,3,3)), cam_t=np.ones((5,count,3)),
             chunk_ranges=np.array([[0,2],[2,4]]))


def test_hm_topology_and_no_fake_ego_pose(model_path, tmp_path):
    path = tmp_path/'take1/points_triangulated_world_results_merged.npz'; write_hm(path)
    model = BodyModel(model_path)
    with pytest.raises(ValueError, match="not assumed to be ego"):
        read_hm(path, "take1", {}, model)
    ep = read_hm(path, "take1", {"require_camera": False}, model)
    assert np.allclose(ep.positions['hand_left'][0,0], 25)
    assert np.allclose(ep.positions['hand_right'][0,0], 46)
    assert not ep.position_valid['hand_left'][1].any()
    assert not ep.position_valid['body'][1].any()
    assert not ep.position_valid['camera'].any()
    assert ep.positions['body'].shape == (4,22,3)
    side = tmp_path/'side'; side.mkdir()
    camera = np.broadcast_to(np.eye(4), (4,4,4)).copy(); camera[:,:3,3] = [1,2,3]
    np.savez(side/'take1.npz', timestamp=np.arange(4)/30, camera_world=camera, camera_valid=[1,1,0,1])
    (side/'take1.json').write_text(json.dumps({"take_name":"take1", "clock":"aligned_take_seconds"}))
    ep = read_hm(path, 'take1', {'trajectory_root': str(side)}, model)
    assert np.allclose(ep.positions['camera'][0,0], [1,2,3])
    assert not ep.position_valid['camera'][2].any()
    np.savez(side/'take1.npz', timestamp=np.arange(4)/15, camera_world=camera, camera_valid=[1,1,1,1])
    with pytest.raises(ValueError, match="30 Hz timestamps"):
        read_hm(path, 'take1', {'trajectory_root': str(side)}, model)


@pytest.mark.parametrize("require_camera", [None, True])
@pytest.mark.parametrize("missing_camera_fields", [
    ("camera_world",), ("camera_valid",), ("camera_world", "camera_valid"),
])
def test_hm_requires_camera_fields_in_existing_sidecar(model_path, tmp_path, require_camera, missing_camera_fields):
    path = tmp_path/'take1/points_triangulated_world_results_merged.npz'; write_hm(path)
    model = BodyModel(model_path)
    side = tmp_path/'side'; side.mkdir()
    transforms = np.broadcast_to(np.eye(4), (4,4,4)).copy()
    fields = {"timestamp": np.arange(4)/30,
              "head_world": transforms, "head_valid": np.ones(4, bool),
              "camera_world": transforms, "camera_valid": np.ones(4, bool)}
    for key in missing_camera_fields:
        del fields[key]
    np.savez(side/'take1.npz', **fields)
    (side/'take1.json').write_text(json.dumps({"take_name":"take1", "clock":"aligned_take_seconds"}))
    config = {"trajectory_root": str(side)}
    if require_camera is not None:
        config["require_camera"] = require_camera
    with pytest.raises(ValueError, match="missing required camera fields") as exc:
        read_hm(path, 'take1', config, model)
    for key in missing_camera_fields:
        assert key in str(exc.value)


def test_hm_optional_camera_allows_head_only_sidecar(model_path, tmp_path):
    path = tmp_path/'take1/points_triangulated_world_results_merged.npz'; write_hm(path)
    model = BodyModel(model_path)
    side = tmp_path/'side'; side.mkdir()
    np.savez(side/'take1.npz', timestamp=np.arange(4)/30,
             head_world=np.broadcast_to(np.eye(4), (4,4,4)), head_valid=np.ones(4, bool))
    (side/'take1.json').write_text(json.dumps({"take_name":"take1", "clock":"aligned_take_seconds"}))
    ep = read_hm(path, 'take1', {"trajectory_root": str(side), "require_camera": False}, model)
    assert ep.position_valid['head'].all() and ep.rotation_valid['head'].all()
    assert not ep.position_valid['camera'].any() and not ep.rotation_valid['camera'].any()


def test_egosuite_shared_shard_episode_filter_and_video_offsets(tmp_path):
    root = make_egosuite(tmp_path/'egosuite')
    config = {'root': str(root)}
    items = list(discover_egosuite(config))
    assert len(items) == 2
    ep = read_egosuite(*items[1], config)
    assert len(ep.timestamps) == 5
    assert np.allclose(ep.positions['body'][0,:,0], 1)
    assert ep.video['from_timestamp'] == 5/30
    assert ep.metadata['operator_id'] == 'operator-1'
    assert 'task' not in ep.metadata and 'tasks' not in ep.metadata
    assert np.allclose(ep.rotations['body'][0,:,0], 1)
    config['episode_ids'] = ['uuid-1']
    assert len(list(discover_egosuite(config))) == 1


def test_reject_14_joint_data_in_full_body_reader(tmp_path):
    root = make_egosuite(tmp_path/'upper', body_joints=14)
    config = {'root': str(root)}
    with pytest.raises(ValueError, match="22-joint"):
        read_egosuite(*next(discover_egosuite(config)), config)


def test_sync_nan_and_integer_precision(tmp_path):
    path = tmp_path/'timesync.csv'
    path.write_text('rgb\n1234567890123456789\nNaN\n1234567890123456790\n')
    assert capture_timestamps(path, 'rgb', 0, 3).tolist() == [1234567890123456789,-1,1234567890123456790]
    with pytest.raises(ValueError, match='length mismatch'):
        capture_timestamps(path, 'rgb', 0, 4)
