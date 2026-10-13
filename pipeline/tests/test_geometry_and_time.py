import numpy as np
import pytest
from egomotion_pipeline.geometry import axis_angle_to_quat, matrix_to_quat, quat_to_matrix, slerp, rigid_transform
from egomotion_pipeline.schema import Episode
from egomotion_pipeline.resampling import resample, transform_episode


def test_rotation_roundtrip_and_antipodal_slerp():
    rng = np.random.default_rng(91)
    aa = rng.normal(size=(300, 3))*2
    aa[:4] = [[0,0,0], [np.pi,0,0], [0,np.pi,0], [0,0,np.pi]]
    q = axis_angle_to_quat(aa)
    assert np.allclose(quat_to_matrix(matrix_to_quat(quat_to_matrix(q))), quat_to_matrix(q), atol=1e-12)
    assert np.allclose(quat_to_matrix(slerp(q, -q, np.full(300, .5))), quat_to_matrix(q))
    middle = slerp(np.array([1.,0,0,0]), axis_angle_to_quat([0,0,np.pi]), .5)
    assert np.allclose(quat_to_matrix(middle)@[1,0,0], [0,1,0], atol=1e-12)


def test_resampling_velocity_gap_invalid_endpoint_and_absolute_15hz():
    ts = np.array([0., 1/60, 2/60, 3/60, 4/60, .3])
    body = np.broadcast_to((ts[:, None]*np.array([1,2,3]))[:, None, :], (6,22,3)).copy()
    rot = np.broadcast_to(axis_angle_to_quat(np.stack([ts*0, ts*0, ts*np.pi], -1))[:, None], (6,22,4)).copy()
    mask = np.ones((6,22), bool); mask[1, 0] = False
    ep = Episode("gap", "fixture", ts, {"body": body}, {"body": rot}, {"body": mask})
    arrays = resample(ep)
    assert np.allclose(arrays["timestamp"], np.arange(10)/30)
    assert not arrays["observation.valid.body_position"][3:9].any()
    assert arrays["observation.valid.body_position"][1].all()
    assert np.allclose(arrays["observation.state.body_world"][2].reshape(22,3)[0], [2/30,4/30,6/30])
    # Absolute representations can be decimated. The difference then covers 1/15 s.
    assert np.allclose(np.diff(arrays["observation.state.body_world"][:3:2, :3], axis=0)[0], [1/15,2/15,3/15])
    assert not arrays["observation.valid.camera_position"].any()


def test_slerp_interpolation_and_invalid_source():
    ts = np.array([0., .04, .08])
    body = np.zeros((3,22,3)); body[1, 0] = np.nan
    rot = np.broadcast_to(axis_angle_to_quat(np.stack([ts*0, ts*0, ts*2], -1))[:, None], (3,22,4)).copy()
    arrays = resample(Episode("24hz", "fixture", ts, {"body": body}, {"body": rot}))
    assert not arrays["observation.valid.body_position"][1, 0]
    assert arrays["observation.interpolated.body_rotation"][1].all()
    q = arrays["observation.state.body_world_rotation"][1].reshape(22,4)[0]
    assert np.allclose(quat_to_matrix(q), quat_to_matrix(axis_angle_to_quat([0,0,2/30])), atol=1e-7)
    assert (arrays["observation.state.body_world"][1,:3] == 0).all()


def test_world_and_local_basis_transforms_are_distinct():
    world = np.eye(4); world[:3,:3] = quat_to_matrix(axis_angle_to_quat([0,0,np.pi/2])); world[:3,3] = [10,0,0]
    correction = np.broadcast_to(quat_to_matrix(axis_angle_to_quat([np.pi/2,0,0])), (22,3,3))
    pos = np.broadcast_to([1.,0,0], (1,22,3)).copy()
    rot = np.broadcast_to([1.,0,0,0], (1,22,4)).copy()
    ep = transform_episode(Episode("frame", "fixture", np.array([0.]), {"body": pos}, {"body": rot}),
                           {"world_from_source": world, "body_basis_correction": correction})
    assert np.allclose(ep.positions["body"][0,0], [10,1,0])
    assert np.allclose(quat_to_matrix(ep.rotations["body"][0,0]), world[:3,:3]@correction[0])
    reflection = np.eye(4); reflection[0,0] = -1
    with pytest.raises(ValueError, match="proper rotation"):
        rigid_transform(reflection)


def test_reject_nonmonotonic_clock_and_wrong_archive_rate():
    with pytest.raises(ValueError, match="strictly increasing"):
        Episode("bad", "fixture", np.array([0.,0.]), {}, {}).check()
    with pytest.raises(ValueError, match="fixed at 30"):
        resample(Episode("good", "fixture", np.array([0.]), {}, {}), fps=15)
