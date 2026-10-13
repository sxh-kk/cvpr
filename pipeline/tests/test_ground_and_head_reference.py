import numpy as np
import pytest
from egomotion_pipeline.geometry import axis_angle_to_quat, quat_to_matrix, matrix_to_quat
from egomotion_pipeline.schema import Episode
from egomotion_pipeline.resampling import transform_episode, resample
from egomotion_pipeline.head_reference import EE4D_CAMERA_FROM_REFERENCE
from egomotion_pipeline.storage import Archive, read_arrays, write_arrays
from egomotion_pipeline.validation import validate
from egomotion_pipeline.cli import convert
from conftest import make_egosuite


def standing_episode(fps=30, floor=2., duration=1.):
    ts = np.arange(int(duration*fps)+1)/fps
    count = len(ts)
    body = np.zeros((count, 22, 3))
    body[:, :, 2] = floor+1
    body[:, [10, 11], 2] = floor+.01
    camera = np.zeros((count, 1, 3)); camera[:, :, 2] = floor+1.6
    # With EE4D's B, this camera pose produces an identity reference pose.
    rotation = np.broadcast_to(matrix_to_quat(EE4D_CAMERA_FROM_REFERENCE.T), (count, 1, 4)).copy()
    native_head = camera.copy(); native_head[:, :, 0] = .05
    native_head[:, :, 2] -= .08
    return Episode("standing", "fixture", ts, {"body": body, "camera": camera, "head": native_head},
                   {"camera": rotation}, metadata={"camera_pose_basis": "aria_rgb_native", "head_semantics": "aria_cpf"})


def prepare(episode, **config):
    return transform_episode(episode, {"coordinate_verified": True, **config})


def test_floor_zero_is_valid_and_world_positions_are_preserved():
    episode = prepare(standing_episode(floor=0.))
    ground = episode.metadata['ground']
    assert ground['valid'] and not ground['verified']
    assert ground['height_m'] == pytest.approx(0.)
    arrays = resample(episode)
    assert arrays['observation.valid.floor'].all()
    assert (arrays['observation.environment.floor_height'] == 0).all()
    assert np.allclose(arrays['observation.state.camera_world'][:,2], 1.6)
    assert np.allclose(arrays['observation.state.head_reference_height'], 1.6)


@pytest.mark.parametrize('fps', [15, 30, 60])
def test_floor_speed_threshold_is_in_seconds_and_ignores_jumping_outlier(fps):
    episode = standing_episode(fps)
    # An isolated erroneous low foot point must not become the floor.
    episode.positions['body'][len(episode.timestamps)//2, 10, 2] = -10
    episode = prepare(episode)
    assert episode.metadata['ground']['valid']
    assert episode.metadata['ground']['height_m'] == pytest.approx(2.)
    moving = standing_episode(fps)
    moving.positions['body'][:, [10,11], 0] = moving.timestamps[:,None]*.3
    moving = prepare(moving)
    assert not moving.metadata['ground']['valid']
    assert moving.metadata['ground']['reason'] == 'no_stationary_feet'


def test_floor_unknown_axis_missing_feet_and_disconnected_single_frames():
    episode = transform_episode(standing_episode(), {'coordinate_verified': False})
    assert not episode.metadata['ground']['valid']
    assert episode.metadata['ground']['reason'] == 'output_world_vertical_axis_unverified'
    episode = standing_episode(); episode.position_valid['body'] = np.ones((len(episode.timestamps),22),bool)
    episode.position_valid['body'][:, [10,11]] = False
    assert not prepare(episode).metadata['ground']['valid']
    episode = standing_episode(); episode.timestamps = np.arange(len(episode.timestamps))*.2
    assert not prepare(episode).metadata['ground']['valid']


def test_multiple_supported_levels_keep_the_low_floor_without_deleting_motion():
    episode = standing_episode(duration=2.)
    episode.positions['body'][35:, [10,11], 2] += .3
    original = episode.positions['body'].copy()
    episode = prepare(episode)
    assert episode.metadata['ground']['multiple_height_levels']
    assert episode.metadata['ground']['height_m'] == pytest.approx(2.)
    assert np.array_equal(episode.positions['body'], original)


def test_manual_height_in_output_frame_and_native_head_are_preserved():
    episode = standing_episode()
    transform = np.eye(4); transform[2,3] = 10
    episode = prepare(episode, world_from_source=transform,
                      ground={'mode':'manual','height_m':12.,'verified':True,'source':'scene_floor'})
    arrays = resample(episode)
    assert arrays['observation.valid.floor_verified'].all()
    assert np.allclose(arrays['observation.environment.floor_height'], 12)
    assert np.allclose(arrays['observation.state.head_reference_world'], arrays['observation.state.camera_world'])
    assert np.allclose(arrays['observation.state.head_world'][:,0], .05)
    assert np.allclose(arrays['observation.state.head_world'][:,2], 13.52)
    assert np.allclose(arrays['observation.state.head_reference_height'], 1.6)


def test_ee4d_exact_post_rotation_and_vendor_basis_correction():
    episode = standing_episode()
    camera_rotation = quat_to_matrix(episode.rotations['camera'])
    episode = prepare(episode)
    assert np.allclose(quat_to_matrix(episode.rotations['head_reference']), camera_rotation@EE4D_CAMERA_FROM_REFERENCE)
    correction = quat_to_matrix(axis_angle_to_quat([0,.4,0]))
    episode = prepare(standing_episode(), head_reference={
        'camera_from_ee4d_camera':correction.tolist(), 'basis_verified':True})
    assert np.allclose(quat_to_matrix(episode.rotations['head_reference']), camera_rotation@correction@EE4D_CAMERA_FROM_REFERENCE)
    reflection = np.eye(3); reflection[0,0] = -1
    with pytest.raises(ValueError, match='proper rotation'):
        prepare(standing_episode(), head_reference={'camera_from_ee4d_camera':reflection})


def test_projected_head_heading_uses_resampled_full_orientation_and_keeps_absolute_xy():
    episode = standing_episode(fps=25)
    ts = episode.timestamps
    reference = quat_to_matrix(axis_angle_to_quat(np.stack([ts*.6,ts*1.1,ts*.7],-1)))
    episode.rotations['camera'] = matrix_to_quat(reference@EE4D_CAMERA_FROM_REFERENCE.T)[:,None]
    episode.positions['camera'][:,0,:2] = [4,5]
    arrays = resample(prepare(episode))
    mask = arrays['observation.valid.head_reference_floor_position'][:,0]
    full_rotation = quat_to_matrix(arrays['observation.state.head_reference_world_rotation'])
    heading = quat_to_matrix(arrays['observation.state.head_reference_floor_world_rotation'])
    expected = full_rotation[:,:2,0]; expected /= np.linalg.norm(expected,axis=-1,keepdims=True)
    assert np.allclose(heading[mask,:2,0], expected[mask], atol=1e-6)
    assert np.allclose(heading[mask,:,2], [0,0,1], atol=1e-6)
    assert np.allclose(arrays['observation.state.head_reference_floor_world'][mask], [4,5,2])
    assert arrays['observation.interpolated.head_reference_floor_rotation'][1].all()


def test_degenerate_vertical_heading_missing_camera_and_unverified_vendor_basis():
    episode = standing_episode()
    reference = quat_to_matrix(axis_angle_to_quat([0,np.pi/2,0]))
    episode.rotations['camera'][:] = matrix_to_quat(reference@EE4D_CAMERA_FROM_REFERENCE.T)
    arrays = resample(prepare(episode))
    assert arrays['observation.valid.head_reference_rotation'].all()
    assert not arrays['observation.valid.head_reference_floor_rotation'].any()
    episode = standing_episode(); episode.metadata['camera_pose_basis'] = 'unknown_vendor'
    arrays = resample(prepare(episode))
    assert not arrays['observation.valid.head_reference_basis'].any()
    assert not arrays['observation.valid.head_reference_floor_rotation'].any()
    episode = standing_episode(); episode.position_valid['camera'] = np.zeros((len(episode.timestamps),1),bool)
    arrays = resample(prepare(episode))
    assert not arrays['observation.valid.head_reference_height'].any()


def test_archive_floor_head_validation_and_corruption_detection(tmp_path):
    root = make_egosuite(tmp_path/'source', episodes=1)
    config = {'sources':[{'type':'egosuite_lerobot','name':'egosuite','root':str(root),
                         'coordinate_verified':True,'rotation_basis_verified':True,
                         'ground':{'mode':'manual','height_m':0,'verified':True},
                         'head_reference':{'basis_verified':True}}]}
    convert(config,tmp_path/'archive')
    archive = Archive(tmp_path/'archive')
    assert validate(archive,strict=True,require_floor=True,require_head_reference=True)['frames'] == 5
    arrays = read_arrays(archive.data_path(0),archive.specs)
    arrays['observation.state.head_reference_world_rotation'][0] = [1,0,0,0]
    write_arrays(archive.data_path(0),arrays,archive.specs)
    with pytest.raises(ValueError,match='EE4D axis conversion'):
        validate(archive)


def test_require_flags_reject_unavailable_floor_and_unverified_head(tmp_path):
    root = make_egosuite(tmp_path/'source', episodes=1)
    config = {'sources':[{'type':'egosuite_lerobot','root':str(root),'coordinate_verified':True}]}
    convert(config,tmp_path/'archive')
    archive = Archive(tmp_path/'archive')
    with pytest.raises(ValueError,match='floor height unavailable'):
        validate(archive,require_floor=True)
    with pytest.raises(ValueError,match='camera basis unverified'):
        validate(archive,require_head_reference=True)
