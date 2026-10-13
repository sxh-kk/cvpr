"""Independent geometry and provenance checks for native EgoSuite sidecars."""

from copy import deepcopy

import numpy as np
import pytest

from egomotion_pipeline.native_dense22 import (
    encode_native_dense22,
    native_dense22_specs,
    validate_native_dense22,
)


WORLD = "observation.state.body_world_rot6d_xyz"
FLOOR = "observation.state.body_floor_rot6d_xyz"


def _mask(reference, kind, prefix="valid"):
    return f"observation.{prefix}.body_{reference}_rot6d_xyz_{kind}"


def _axis_quaternion(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    return np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]


def _rodrigues(axis, angle):
    """Independent analytic SO(3), without production quaternion helpers."""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def _decode_row6d(value):
    first = value[..., :3]
    first = first / np.linalg.norm(first, axis=-1, keepdims=True)
    second = value[..., 3:6]
    second = second - (second * first).sum(-1, keepdims=True) * first
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    return np.stack([first, second, np.cross(first, second)], -2)


def make_native_input(count=4):
    rng = np.random.default_rng(834)
    body = rng.normal(size=(count, 22, 3)) + [2.5, -1.3, 1.8]
    matrices = np.empty((count, 22, 3, 3))
    quaternions = np.empty((count, 22, 4))
    for frame in range(count):
        for joint in range(22):
            angle = 0.2 + frame * 0.11 + joint * 0.023
            quaternions[frame, joint] = _axis_quaternion([1, 2, 3], angle)
            matrices[frame, joint] = _rodrigues([1, 2, 3], angle)
    floor_position = np.stack([
        np.array([4 + frame * 0.12, -3 - frame * 0.05, 0.2])
        for frame in range(count)
    ])
    floor_rotation = np.stack([_rodrigues([0, 0, 1], 0.4 + frame * 0.07)
                               for frame in range(count)])
    floor_quaternion = np.stack([_axis_quaternion([0, 0, 1], 0.4 + frame * 0.07)
                                 for frame in range(count)])
    arrays = {
        "timestamp": (np.arange(count) / 30).astype(np.float32),
        "index": np.arange(count, dtype=np.int64),
        "frame_index": np.arange(count, dtype=np.int64),
        "episode_index": np.zeros(count, dtype=np.int64),
        "task_index": np.zeros(count, dtype=np.int64),
        "observation.source_timestamp": np.arange(count) / 30,
        "observation.source_frame_left": np.arange(count, dtype=np.int64),
        "observation.source_frame_right": np.arange(count, dtype=np.int64),
        "observation.source_interpolation_weight": np.zeros(count, np.float32),
        "observation.state.body_world": body.reshape(count, 66).astype(np.float32),
        "observation.state.body_world_rotation": quaternions.reshape(count, 88).astype(np.float32),
        "observation.valid.body_position": np.ones((count, 22), bool),
        "observation.valid.body_rotation": np.ones((count, 22), bool),
        "observation.interpolated.body_position": np.zeros((count, 22), bool),
        "observation.interpolated.body_rotation": np.zeros((count, 22), bool),
        "observation.state.head_reference_floor_world": floor_position.astype(np.float32),
        "observation.state.head_reference_floor_world_rotation": floor_quaternion.astype(np.float32),
        "observation.valid.head_reference_floor_position": np.ones((count, 1), bool),
        "observation.valid.head_reference_floor_rotation": np.ones((count, 1), bool),
        "observation.interpolated.head_reference_floor_position": np.zeros((count, 1), bool),
        "observation.interpolated.head_reference_floor_rotation": np.zeros((count, 1), bool),
    }
    record = {
        "source": "egosuite", "source_uid": "synthetic-native",
        "source_key": "egosuite:synthetic-native", "episode_index": 0,
        "length": count, "input_fingerprint": "synthetic-native-input",
        "dataset_from_index": 0, "dataset_to_index": count,
        "metadata": {
            "source_format": "egosuite_mcap", "rotation_semantics": "source_global_wxyz",
            "body_position_source": "egosuite_mocap", "coordinate_verified": True,
            "rotation_basis_verified": False, "body_basis_correction": None,
            "head_reference": {"basis_verified": True},
            "ground": {"valid": True, "height_m": 0.2},
        },
    }
    return arrays, record, matrices, floor_rotation, floor_position


def test_row6d_world_and_floor_use_independent_analytic_rotations():
    arrays, record, expected_world, expected_floor, floor_position = make_native_input()
    frozen = {key: value.copy() for key, value in arrays.items()}
    record_before = deepcopy(record)
    output, metadata = encode_native_dense22(arrays, record)
    world = output[WORLD].reshape(-1, 22, 9)
    floor = output[FLOOR].reshape(-1, 22, 9)
    np.testing.assert_allclose(_decode_row6d(world[..., :6]), expected_world, atol=2e-7)
    expected_local = np.swapaxes(expected_floor, -1, -2)[:, None] @ expected_world
    np.testing.assert_allclose(_decode_row6d(floor[..., :6]), expected_local, atol=3e-7)
    np.testing.assert_array_equal(world[..., 6:], arrays["observation.state.body_world"].reshape(-1, 22, 3))
    expected_xyz = np.einsum("tji,tkj->tki", expected_floor,
                             world[..., 6:] - floor_position[:, None])
    np.testing.assert_allclose(floor[..., 6:], expected_xyz, atol=5e-7)
    np.testing.assert_allclose(np.einsum("tij,tkj->tki", expected_floor, floor[..., 6:])
                               + floor_position[:, None], world[..., 6:], atol=5e-7)
    assert output[WORLD].dtype == output[FLOOR].dtype == np.float32
    assert output[WORLD].shape == output[FLOOR].shape == (4, 198)
    for key, value in frozen.items():
        np.testing.assert_array_equal(arrays[key], value)
    assert record == record_before
    validate_native_dense22(arrays, output, record, metadata)


def test_invalid_quaternion_cannot_become_valid_after_normalization():
    arrays, record, _, _, _ = make_native_input()
    q = arrays["observation.state.body_world_rotation"].reshape(-1, 22, 4)
    q[0, 1] = 0
    q[0, 2] = np.nan
    q[0, 3] = np.inf
    q[0, 4] *= 3  # Finite non-unit source quaternions remain usable after normalization.
    output, metadata = encode_native_dense22(arrays, record)
    assert not output[_mask("world", "rotation")][0, 1:4].any()
    assert output[_mask("world", "rotation")][0, 4]
    assert output[_mask("world", "position")][0, 1:5].all()
    for key in (WORLD, FLOOR):
        assert np.isfinite(output[key]).all()
        np.testing.assert_array_equal(output[key].reshape(-1, 22, 9)[0, 1:4, :6], 0)
    validate_native_dense22(arrays, output, record, metadata)


def test_position_rotation_and_floor_masks_are_independent():
    arrays, record, _, _, _ = make_native_input()
    arrays["observation.valid.body_position"][0, 3] = False
    arrays["observation.valid.body_rotation"][0, 4] = False
    arrays["observation.valid.head_reference_floor_position"][1] = False
    arrays["observation.valid.head_reference_floor_rotation"][2] = False
    output, metadata = encode_native_dense22(arrays, record)
    assert not output[_mask("world", "position")][0, 3]
    assert output[_mask("world", "rotation")][0, 3]
    assert output[_mask("world", "position")][0, 4]
    assert not output[_mask("world", "rotation")][0, 4]
    assert output[_mask("floor", "rotation")][1].all()
    assert not output[_mask("floor", "position")][1].any()
    assert not output[_mask("floor", "rotation")][2].any()
    assert not output[_mask("floor", "position")][2].any()
    assert output[_mask("world", "position")][1:].all()
    assert output[_mask("world", "rotation")][1:].all()
    validate_native_dense22(arrays, output, record, metadata)


def test_missing_ground_keeps_world_labels_and_invalidates_floor_only():
    arrays, record, _, _, _ = make_native_input()
    record["metadata"]["ground"] = {"valid": False, "reason": "no_stationary_feet"}
    arrays["observation.valid.head_reference_floor_position"][:] = False
    arrays["observation.valid.head_reference_floor_rotation"][:] = False
    output, metadata = encode_native_dense22(arrays, record)
    for kind in ("rotation", "position"):
        assert output[_mask("world", kind)].all()
        assert not output[_mask("floor", kind)].any()
    np.testing.assert_array_equal(output[FLOOR], 0)
    validate_native_dense22(arrays, output, record, metadata)


def test_interpolation_dependency_masks_do_not_mix_xyz_and_rotation():
    arrays, record, _, _, _ = make_native_input()
    arrays["observation.interpolated.body_position"][0, 3] = True
    arrays["observation.interpolated.body_rotation"][0, 4] = True
    arrays["observation.interpolated.head_reference_floor_position"][1] = True
    arrays["observation.interpolated.head_reference_floor_rotation"][2] = True
    arrays["observation.valid.body_rotation"][0, 4] = False
    output, metadata = encode_native_dense22(arrays, record)
    assert output[_mask("world", "position", "interpolated")][0, 3]
    assert not output[_mask("world", "rotation", "interpolated")].any()
    assert not output[_mask("floor", "rotation", "interpolated")][1].any()
    assert output[_mask("floor", "position", "interpolated")][1].all()
    assert output[_mask("floor", "rotation", "interpolated")][2].all()
    assert output[_mask("floor", "position", "interpolated")][2].all()
    for reference in ("world", "floor"):
        for kind in ("position", "rotation"):
            interpolated = output[_mask(reference, kind, "interpolated")]
            assert not (interpolated & ~output[_mask(reference, kind)]).any()
    validate_native_dense22(arrays, output, record, metadata)


def test_native_converter_never_accesses_dino_and_has_no_derived_input_aliases():
    arrays, record, _, _, _ = make_native_input()

    class PoisonDino:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Native motion export accessed DINO")

    arrays["observation.features.dino"] = PoisonDino()
    output, _ = encode_native_dense22(arrays, record)
    assert "observation.features.dino" not in output
    output["observation.state.body_world"][0, 0] = 999
    assert arrays["observation.state.body_world"][0, 0] != 999
    specs = native_dense22_specs()
    assert specs[WORLD]["shape"] == specs[FLOOR]["shape"] == [198]
    assert not any("dino" in key for key in specs)


@pytest.mark.parametrize("field,value", [
    ("body_basis_correction", np.broadcast_to(np.eye(3), (22, 3, 3)).tolist()),
    ("rotation_basis_verified", True),
    ("rotation_semantics", "parent_local_wxyz"),
    ("coordinate_verified", "false"),
])
def test_wrong_provenance_is_rejected(field, value):
    arrays, record, _, _, _ = make_native_input()
    record["metadata"][field] = value
    with pytest.raises(ValueError):
        encode_native_dense22(arrays, record)


def test_foreign_source_and_nonboolean_masks_are_rejected():
    arrays, record, _, _, _ = make_native_input()
    record["source"] = "egoexo_hm"
    with pytest.raises(ValueError):
        encode_native_dense22(arrays, record)
    arrays, record, _, _, _ = make_native_input()
    arrays["observation.valid.body_rotation"] = np.ones((4, 22), dtype=np.int64)
    with pytest.raises(ValueError):
        encode_native_dense22(arrays, record)


@pytest.mark.parametrize("key", ["index", "frame_index", "episode_index", "task_index"])
def test_corrupted_source_frame_identity_is_rejected(key):
    arrays, record, _, _, _ = make_native_input()
    arrays[key][1] += 7
    with pytest.raises(ValueError):
        encode_native_dense22(arrays, record)


@pytest.mark.parametrize("field,value", [
    ("position_semantics", "smplx_joint_centers"),
    ("rotation_semantics", "certified_target_rotation"),
    ("source_inputs_modified", True),
    ("rotation_basis_verified", 0),
    ("source_identity", {"source_uid": "foreign-episode"}),
    ("source_coordinate_provenance", {}),
    ("valid_counts", {}),
])
def test_validator_rejects_tampered_native_semantic_provenance(field, value):
    arrays, record, _, _, _ = make_native_input()
    output, metadata = encode_native_dense22(arrays, record)
    metadata[field] = value
    with pytest.raises(ValueError):
        validate_native_dense22(arrays, output, record, metadata)


def test_invalid_floor_quaternion_does_not_affect_world_availability():
    arrays, record, _, _, _ = make_native_input()
    arrays["observation.state.head_reference_floor_world_rotation"][0] = 0
    arrays["observation.state.head_reference_floor_world_rotation"][1] = np.nan
    output, metadata = encode_native_dense22(arrays, record)
    for kind in ("position", "rotation"):
        assert output[_mask("world", kind)].all()
        assert not output[_mask("floor", kind)][:2].any()
    np.testing.assert_array_equal(output[FLOOR][:2], 0)
    validate_native_dense22(arrays, output, record, metadata)


@pytest.mark.parametrize("basis", [False, "true", 1, None])
def test_floor_labels_require_explicitly_verified_camera_reference_basis(basis):
    arrays, record, _, _, _ = make_native_input()
    record["metadata"]["head_reference"]["basis_verified"] = basis
    with pytest.raises(ValueError):
        encode_native_dense22(arrays, record)


def test_unavailable_camera_basis_does_not_prevent_valid_world_export():
    arrays, record, _, _, _ = make_native_input()
    record["metadata"]["head_reference"]["basis_verified"] = False
    arrays["observation.valid.head_reference_floor_position"][:] = False
    arrays["observation.valid.head_reference_floor_rotation"][:] = False
    output, metadata = encode_native_dense22(arrays, record)
    assert output[_mask("world", "position")].all()
    assert output[_mask("world", "rotation")].all()
    assert not output[_mask("floor", "position")].any()
    assert not output[_mask("floor", "rotation")].any()
    validate_native_dense22(arrays, output, record, metadata)


@pytest.mark.parametrize("key,component", [(WORLD, 0), (WORLD, 6), (FLOOR, 0), (FLOOR, 6)])
def test_independent_validator_rejects_corrupted_labels(key, component):
    arrays, record, _, _, _ = make_native_input()
    output, metadata = encode_native_dense22(arrays, record)
    output[key][0, component] += 0.1
    with pytest.raises(ValueError):
        validate_native_dense22(arrays, output, record, metadata)
