"""Camera-origin EE4D references and dense22 positions share one convention."""

import json
import numpy as np
import pytest

from egomotion_pipeline.adapters import read_hm
from egomotion_pipeline.body_model import BodyModel
from egomotion_pipeline.cli import convert
from egomotion_pipeline.geometry import axis_angle_to_quat, matrix_to_quat, quat_to_matrix
from egomotion_pipeline.head_reference import EE4D_CAMERA_FROM_REFERENCE
from egomotion_pipeline.resampling import resample, transform_episode
from egomotion_pipeline.schema import BODY_NAMES, Episode, feature_specs
from egomotion_pipeline.storage import Archive, read_arrays, write_arrays
from egomotion_pipeline.validation import validate
from mcap_fixtures import make_mcap
from test_sources import write_hm


LOCAL = "observation.state.body_head_floor"
VALID = "observation.valid.body_head_floor_position"
INTERPOLATED = "observation.interpolated.body_head_floor_position"
OPENCV_FROM_REFERENCE = np.diag([-1., -1., 1.])
# In a forward/left/up world, the level reference axes are left/up/forward.
LEVEL_REFERENCE = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def camera_episode(basis="egosuite_opencv_camera_to_world", fps=30, floor=2., moving=False):
    timestamps = np.arange(fps + 1, dtype=float) / fps
    count = len(timestamps)
    camera = np.broadcast_to([4., -2., floor + 1.65], (count, 1, 3)).copy()
    body = np.broadcast_to([4.3, -2.2, floor + .8], (count, 22, 3)).copy()
    body[:, :, 0] += np.arange(22) * .015
    body[:, :, 1] += np.arange(22) * .012
    body[:, :, 2] += np.arange(22) * .025
    body[:, 15] = [4.2, -2.1, floor + 1.45]
    source_head = np.broadcast_to([4.1, -2.05, floor + 1.52], (count, 1, 3)).copy()
    reference = np.broadcast_to(LEVEL_REFERENCE, (count, 3, 3)).copy()
    if moving:
        camera[:, 0] += timestamps[:, None] * np.array([.7, -.3, .05])
        body += timestamps[:, None, None] * np.array([.4, .2, -.03])
        angles = np.stack([timestamps * .7, timestamps * 1.1, timestamps * 3.0], axis=-1)
        reference = quat_to_matrix(axis_angle_to_quat(angles)) @ LEVEL_REFERENCE
    camera_from_reference = (
        EE4D_CAMERA_FROM_REFERENCE if basis == "aria_rgb_native" else OPENCV_FROM_REFERENCE
    )
    camera_rotation = reference @ camera_from_reference.T
    return Episode(
        "camera-dense", "fixture", timestamps,
        {"body": body, "camera": camera, "head": source_head},
        {"camera": matrix_to_quat(camera_rotation)[:, None]},
        metadata={"camera_pose_basis": basis, "head_semantics": "source_head_preserved"},
    )


def prepare(episode, floor=2., **config):
    return transform_episode(episode, {
        "coordinate_verified": True,
        "ground": {"mode": "manual", "height_m": floor, "verified": True},
        **config,
    })


def local_positions(arrays):
    return arrays[LOCAL].reshape(-1, 22, 3)


def expected_local(arrays):
    world = arrays["observation.state.body_world"].reshape(-1, 22, 3)
    origin = arrays["observation.state.head_reference_floor_world"]
    rotation = quat_to_matrix(arrays["observation.state.head_reference_floor_world_rotation"])
    return np.einsum("tji,tkj->tki", rotation, world - origin[:, None])


def test_aria_and_opencv_camera_axes_produce_the_same_physical_reference():
    aria = prepare(camera_episode("aria_rgb_native", moving=True))
    egosuite = prepare(camera_episode(moving=True))
    assert aria.metadata["head_reference"]["basis_verified"]
    assert egosuite.metadata["head_reference"]["basis_verified"]
    assert np.allclose(egosuite.metadata["head_reference"]["native_camera_from_reference"],
                       OPENCV_FROM_REFERENCE)
    assert np.allclose(quat_to_matrix(aria.rotations["head_reference"]),
                       quat_to_matrix(egosuite.rotations["head_reference"]), atol=1e-12)
    arrays_aria, arrays_ego = resample(aria), resample(egosuite)
    assert arrays_aria[VALID].all() and arrays_ego[VALID].all()
    assert np.allclose(arrays_aria[LOCAL], arrays_ego[LOCAL], atol=1e-6)
    assert np.allclose(quat_to_matrix(arrays_aria["observation.state.head_reference_floor_world_rotation"]),
                       quat_to_matrix(arrays_ego["observation.state.head_reference_floor_world_rotation"]),
                       atol=1e-6)


def test_camera_center_is_reference_origin_and_both_anatomical_heads_are_preserved():
    episode = camera_episode()
    source_body, source_head, source_camera = (
        episode.positions[key].copy() for key in ["body", "head", "camera"]
    )
    episode = prepare(episode)
    arrays = resample(episode)
    assert np.array_equal(episode.positions["body"], source_body)
    assert np.array_equal(episode.positions["head"], source_head)
    assert np.array_equal(episode.positions["camera"], source_camera)
    assert np.array_equal(episode.positions["head_reference"], source_camera)
    assert np.allclose(arrays["observation.state.head_reference_floor_world"], [4., -2., 2.])
    # Reference floor axes are +world Y, -world X, +world Z.
    assert np.allclose(local_positions(arrays)[:, 15], [-.1, -.2, 1.45])
    assert not np.allclose(local_positions(arrays)[:, 15], 0)
    assert arrays[VALID].all()
    # Position supervision does not require SMPL-compatible joint rotations.
    assert not arrays["observation.valid.body_rotation"].any()
    assert not episode.metadata["rotation_basis_verified"]


def test_nonzero_floor_and_world_transform_reconstruct_all_dense_joints():
    transform = np.eye(4)
    transform[:3, :3] = quat_to_matrix(axis_angle_to_quat([0., 0., .63]))
    transform[:3, 3] = [11., -7., 3.]
    arrays = resample(prepare(camera_episode(moving=True), floor=5., world_from_source=transform))
    assert arrays[VALID].all()
    local = local_positions(arrays)
    reference_rotation = quat_to_matrix(arrays["observation.state.head_reference_floor_world_rotation"])
    reconstructed = np.einsum("tij,tkj->tki", reference_rotation, local)
    reconstructed += arrays["observation.state.head_reference_floor_world"][:, None]
    assert np.allclose(reconstructed, arrays["observation.state.body_world"].reshape(-1, 22, 3),
                       atol=3e-6)
    assert np.allclose(local[:, :, 2],
                       arrays["observation.state.body_world"].reshape(-1, 22, 3)[:, :, 2] - 5.,
                       atol=1e-6)


def test_tilted_camera_preserves_full_head_orientation_but_body_reference_is_z_up():
    episode = camera_episode()
    tilt = quat_to_matrix(axis_angle_to_quat([.6, -.35, .8]))
    reference = tilt @ LEVEL_REFERENCE
    episode.rotations["camera"][:] = matrix_to_quat(reference @ OPENCV_FROM_REFERENCE)
    arrays = resample(prepare(episode))
    assert arrays[VALID].all()
    full = quat_to_matrix(arrays["observation.state.head_reference_world_rotation"])
    projected = quat_to_matrix(arrays["observation.state.head_reference_floor_world_rotation"])
    assert np.allclose(full, reference, atol=1e-6)
    assert not np.allclose(full[:, :, 2], [0., 0., 1.])
    assert np.allclose(projected[:, :, 2], [0., 0., 1.], atol=1e-6)
    assert np.allclose(local_positions(arrays), expected_local(arrays), atol=1e-6)
    assert np.allclose(local_positions(arrays)[:, :, 2],
                       arrays["observation.state.body_world"].reshape(-1, 22, 3)[:, :, 2] - 2.)


@pytest.mark.parametrize("unavailable", [
    "camera_position", "camera_rotation", "basis", "ground", "reference", "vertical_axis",
])
def test_unavailable_reference_masks_and_zeros_all_local_positions(unavailable):
    episode = camera_episode()
    kwargs = {}
    if unavailable == "camera_position":
        episode.position_valid["camera"] = np.zeros((len(episode.timestamps), 1), bool)
    elif unavailable == "camera_rotation":
        episode.rotation_valid["camera"] = np.zeros((len(episode.timestamps), 1), bool)
    elif unavailable == "basis":
        kwargs["head_reference"] = {"basis_verified": False}
    elif unavailable == "ground":
        kwargs["ground"] = {"mode": "disabled"}
    elif unavailable == "reference":
        kwargs["head_reference"] = {"mode": "disabled"}
    else:
        reference = quat_to_matrix(axis_angle_to_quat([np.pi / 2, 0., 0.])) @ LEVEL_REFERENCE
        episode.rotations["camera"][:] = matrix_to_quat(reference @ OPENCV_FROM_REFERENCE)
    arrays = resample(prepare(episode, **kwargs))
    assert arrays["observation.valid.body_position"].all()
    assert not arrays[VALID].any()
    assert not arrays[INTERPOLATED].any()
    assert np.isfinite(arrays[LOCAL]).all()
    assert (arrays[LOCAL] == 0).all()


def test_unknown_or_manually_overridden_basis_requires_explicit_verification():
    unknown = resample(prepare(camera_episode("unknown_vendor")))
    assert not unknown[VALID].any()
    overridden = resample(prepare(camera_episode("aria_rgb_native"), head_reference={
        "camera_from_ee4d_camera": np.eye(3).tolist(),
    }))
    assert not overridden["observation.valid.head_reference_basis"].any()
    assert not overridden[VALID].any()
    verified = resample(prepare(camera_episode("aria_rgb_native"), head_reference={
        "camera_from_ee4d_camera": np.eye(3).tolist(), "basis_verified": True,
    }))
    assert verified[VALID].all()


def test_a_bad_joint_masks_only_that_joint_and_camera_gaps_mask_the_reference():
    episode = camera_episode()
    count = len(episode.timestamps)
    episode.position_valid["body"] = np.ones((count, 22), bool)
    episode.position_valid["body"][3, 18] = False
    episode.position_valid["camera"] = np.ones((count, 1), bool)
    episode.position_valid["camera"][8, 0] = False
    arrays = resample(prepare(episode))
    assert arrays[VALID][3].sum() == 21
    assert not arrays[VALID][3, 18]
    assert (local_positions(arrays)[3, 18] == 0).all()
    assert not arrays[VALID][8].any()
    assert (local_positions(arrays)[8] == 0).all()
    assert arrays[VALID][[2, 4, 7, 9]].all()


def test_25hz_input_is_resampled_before_projecting_body_into_head_floor_frame():
    episode = prepare(camera_episode(fps=25, moving=True))
    arrays = resample(episode)
    assert len(arrays[LOCAL]) == 31
    assert arrays[VALID].all()
    assert np.allclose(local_positions(arrays), expected_local(arrays), atol=1e-6)
    assert arrays[INTERPOLATED][1].all()
    assert not arrays[INTERPOLATED][0].any()
    assert not arrays[INTERPOLATED][6].any()  # .2 s is exact at both 25 and 30 Hz.
    assert np.array_equal(arrays[INTERPOLATED],
                          arrays[VALID] & (
                              arrays["observation.interpolated.body_position"]
                              | arrays["observation.interpolated.head_reference_floor_position"]
                              | arrays["observation.interpolated.head_reference_floor_rotation"]))
    # A local-first interpolation is numerically different under a moving camera.
    source_rotation = quat_to_matrix(episode.rotations["head_reference_floor"][:, 0])
    source_origin = episode.positions["head_reference_floor"][:, 0]
    source_local = np.einsum("tji,tkj->tki", source_rotation,
                             episode.positions["body"] - source_origin[:, None])
    weight = (1 / 30) / (1 / 25)
    local_first = (1 - weight) * source_local[0] + weight * source_local[1]
    assert np.max(np.abs(local_first - local_positions(arrays)[1])) > 1e-4


def test_dense_reference_fields_have_joint_names_and_survive_mcap_archive_roundtrip(tmp_path):
    specs = feature_specs()
    assert specs[LOCAL]["shape"] == [66]
    assert specs[LOCAL]["names"] == [f"{joint}.{axis}" for joint in BODY_NAMES for axis in "xyz"]
    for key in [VALID, INTERPOLATED]:
        assert specs[key]["dtype"] == "bool"
        assert specs[key]["shape"] == [22]
        assert specs[key]["names"] == BODY_NAMES
    source = make_mcap(tmp_path / "source" / "fixture.mcap")
    config = {"sources": [{
        "type": "egosuite_mcap", "name": "egosuite", "root": str(source),
        "coordinate_verified": True,
        "ground": {"mode": "manual", "height_m": .4, "verified": True},
    }]}
    convert(config, tmp_path / "archive")
    archive = Archive(tmp_path / "archive")
    assert validate(archive, require_floor=True, require_head_reference=True,
                    require_body_head_floor=True)["frames"] == 5
    arrays = read_arrays(archive.data_path(0), archive.specs)
    assert arrays[LOCAL].shape == (5, 66)
    assert arrays[VALID].shape == (5, 22)
    assert arrays[VALID].all()
    assert np.allclose(local_positions(arrays), expected_local(arrays), atol=1e-6)
    assert np.allclose(local_positions(arrays)[:, :, 2], .6)
    assert np.allclose(arrays["observation.state.head_reference_world"][:, 2], 1.5)
    assert np.allclose(arrays["observation.state.body_world"].reshape(-1, 22, 3)[:, 15, 2], 1.)
    arrays[LOCAL][0, 0] += .5
    write_arrays(archive.data_path(0), arrays, archive.specs)
    with pytest.raises(ValueError, match="body_head_floor"):
        validate(archive)


def test_require_dense_reference_rejects_valid_archive_with_missing_camera(tmp_path):
    source = make_mcap(tmp_path / "source" / "missing.mcap", missing_camera=[2])
    config = {"sources": [{
        "type": "egosuite_mcap", "name": "egosuite", "root": str(source),
        "coordinate_verified": True,
        "ground": {"mode": "manual", "height_m": .4, "verified": True},
    }]}
    convert(config, tmp_path / "archive")
    archive = Archive(tmp_path / "archive")
    assert validate(archive)["frames"] == 5
    arrays = read_arrays(archive.data_path(0), archive.specs)
    assert arrays[VALID][[0, 1, 3, 4]].all()
    assert not arrays[VALID][2].any()
    with pytest.raises(ValueError, match="body_head_floor has unavailable"):
        validate(archive, require_body_head_floor=True)


def test_hm_smplh_decode_and_aria_sidecar_use_the_same_dense_reference(model_path, tmp_path):
    source = tmp_path / "hm" / "take1" / "points_triangulated_world_results_merged.npz"
    write_hm(source)
    sidecar = tmp_path / "aria"
    sidecar.mkdir()
    camera = np.broadcast_to(np.eye(4), (4, 4, 4)).copy()
    camera[:, :3, :3] = LEVEL_REFERENCE @ EE4D_CAMERA_FROM_REFERENCE.T
    camera[:, :3, 3] = [1., 2., 6.]
    np.savez(sidecar / "take1.npz", timestamp=np.arange(4) / 30,
             camera_world=camera, camera_valid=[True, True, False, True])
    (sidecar / "take1.json").write_text(json.dumps({
        "take_name": "take1", "clock": "aligned_take_seconds",
        "camera_pose_basis": "aria_rgb_native",
    }))
    episode = read_hm(source, "take1", {"trajectory_root": str(sidecar)}, BodyModel(model_path))
    decoded_body = episode.positions["body"].copy()
    arrays = resample(prepare(episode, floor=3.))
    assert arrays[VALID].shape == (4, 22)
    assert arrays[VALID][[0, 3]].all()
    assert not arrays[VALID][1:3].any()  # HM-invalid body and then missing Aria pose.
    assert np.allclose(local_positions(arrays)[[0, 3]], expected_local(arrays)[[0, 3]], atol=1e-6)
    assert (local_positions(arrays)[1:3] == 0).all()
    assert np.allclose(arrays["observation.state.body_world"].reshape(-1, 22, 3), decoded_body)
    assert np.allclose(arrays["observation.state.head_reference_world"][[0, 3]], [1., 2., 6.])
