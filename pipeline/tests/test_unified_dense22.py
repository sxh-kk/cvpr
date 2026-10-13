"""Numerical and lifecycle checks for the physical dense22 + hands21 archive."""

from copy import deepcopy

import numpy as np
import pytest

from egomotion_pipeline.geometry import quat_to_matrix
from egomotion_pipeline.resampling import resample
from egomotion_pipeline.schema import (LEGACY_SCHEMA_VERSION, V4_SCHEMA_VERSION,
                                       SCHEMA_VERSION, feature_specs_for_schema)
from egomotion_pipeline.storage import Archive, atomic_json, read_arrays
from egomotion_pipeline.unified_dense22 import (BODY_STATE_COLUMNS, encode_unified_dense22,
                                              unified_metadata, unified_specs,
                                              representation_provenance, validate_unified_dense22)
from egomotion_pipeline.windows import prepare_window
from test_native_dense22 import make_native_input, _decode_row6d
from test_ground_and_head_reference import prepare
from test_dbscan_and_windows import moving_episode


def inputs(count=4):
    arrays, record, body_rotation, floor_rotation, origin = make_native_input(count)
    rng = np.random.default_rng(736)
    for side in ("left", "right"):
        # Deliberately do not snap hand wrists to body wrists: the archive
        # must preserve the measured supervision and report any offset.
        arrays[f"observation.state.hand_{side}_world"] = (
            rng.normal(size=(count, 21, 3))+[3, -2, 1.1]).astype(np.float32).reshape(count, 63)
        arrays[f"observation.valid.hand_{side}_position"] = np.ones((count, 21), bool)
        arrays[f"observation.interpolated.hand_{side}_position"] = np.zeros((count, 21), bool)
    return arrays, record, body_rotation, floor_rotation, origin


def test_unified_world_and_both_hand_round_trips_preserve_native_axes():
    arrays, record, body_rotation, floor_rotation, origin = inputs()
    before = {key: value.copy() for key, value in arrays.items()}
    metadata_before = deepcopy(record)
    out = encode_unified_dense22(arrays, record)
    assert set(out) == set(unified_specs()) and len(out) == 16
    world, local = (out[key].reshape(-1, 22, 9) for key in BODY_STATE_COLUMNS)
    np.testing.assert_allclose(_decode_row6d(world[..., :6]), body_rotation, atol=3e-7)
    np.testing.assert_allclose(_decode_row6d(local[..., :6]),
                               np.swapaxes(floor_rotation, -1, -2)[:, None] @ body_rotation, atol=4e-7)
    np.testing.assert_array_equal(world[..., 6:], arrays["observation.state.body_world"].reshape(-1, 22, 3))
    for side in ("left", "right"):
        hand = out[f"observation.state.hand_{side}_floor"].reshape(-1, 21, 3)
        recovered = np.einsum("tij,tkj->tki", floor_rotation, hand)+origin[:, None]
        np.testing.assert_allclose(recovered, arrays[f"observation.state.hand_{side}_world"].reshape(-1, 21, 3), atol=5e-7)
    declaration = unified_metadata(record)
    assert declaration["rotation_basis_verified"] is False
    assert declaration["axis_calibration_performed"] is False
    assert declaration["target_joint_basis"] == "egosuite_native_v1"
    for key in before:
        np.testing.assert_array_equal(arrays[key], before[key])
    assert record == metadata_before
    assert validate_unified_dense22(arrays, out, record)["passed"]


def test_independent_masks_and_interpolated_hand_dependencies():
    arrays, record, *_ = inputs()
    arrays["observation.valid.body_rotation"][0, 4] = False
    arrays["observation.valid.body_position"][0, 5] = False
    arrays["observation.valid.hand_left_position"][0, 7] = False
    arrays["observation.interpolated.hand_right_position"][0, 9] = True
    arrays["observation.interpolated.head_reference_floor_position"][1] = True
    arrays["observation.valid.head_reference_floor_position"][2] = False
    arrays["observation.valid.head_reference_floor_rotation"][3] = False
    out = encode_unified_dense22(arrays, record)
    assert out["observation.valid.body_world_rot6d_xyz_position"][0, 4]
    assert not out["observation.valid.body_world_rot6d_xyz_rotation"][0, 4]
    assert out["observation.valid.body_world_rot6d_xyz_rotation"][0, 5]
    assert not out["observation.valid.body_world_rot6d_xyz_position"][0, 5]
    assert out["observation.valid.body_floor_rot6d_xyz_rotation"][2].all()
    assert not out["observation.valid.body_floor_rot6d_xyz_position"][2:].any()
    for side in ("left", "right"):
        valid = out[f"observation.valid.hand_{side}_floor_position"]
        interp = out[f"observation.interpolated.hand_{side}_floor_position"]
        assert interp[1].all() and not valid[2:].any() and not (interp & ~valid).any()
        assert (out[f"observation.state.hand_{side}_floor"].reshape(-1, 21, 3)[~valid] == 0).all()
    assert out["observation.interpolated.hand_right_floor_position"][0, 9]
    assert not out["observation.valid.hand_left_floor_position"][0, 7]
    validate_unified_dense22(arrays, out, record)


def test_batch_and_sparse_frame_indices_encode_identically_without_reading_dino():
    arrays, record, *_ = inputs(7)

    class PoisonDino:
        def __array__(self, *args, **kwargs):
            raise AssertionError("motion encoding read DINO")

    arrays["observation.features.dino"] = PoisonDino()
    out = encode_unified_dense22(arrays, record)
    pieces = [encode_unified_dense22({key: value[start:end] for key, value in arrays.items()
                                    if key != "observation.features.dino"}, record)
              for start, end in ((0, 2), (2, 5), (5, 7))]
    for key in out:
        np.testing.assert_array_equal(np.concatenate([piece[key] for piece in pieces]), out[key])
    selected = [1, 3, 5]
    sparse = {key: value[selected] for key, value in arrays.items() if key != "observation.features.dino"}
    encoded = encode_unified_dense22(sparse, record)
    for key in out:
        np.testing.assert_array_equal(encoded[key], out[key][selected])


@pytest.mark.parametrize("field,index", [
    ("observation.state.body_world_rot6d_xyz", 0),
    ("observation.state.body_floor_rot6d_xyz", 6),
    ("observation.state.hand_left_floor", 7),
    ("observation.state.hand_right_floor", 9),
])
def test_validator_rejects_corrupted_body_or_explicit_hand_label(field, index):
    arrays, record, *_ = inputs()
    out = encode_unified_dense22(arrays, record)
    out[field][0, index] += .05
    with pytest.raises(ValueError):
        validate_unified_dense22(arrays, out, record)


def test_v5_window_rebuilds_world_and_floor_caches_after_canonicalization_and_fallback(tmp_path):
    episode = moving_episode()
    count = len(episode.timestamps)
    episode.positions["camera"][:, 0, :2] = [4, -3]
    for side in ("left", "right"):
        episode.positions[f"hand_{side}"] = np.broadcast_to([3., 2., 2.7], (count, 21, 3)).copy()
        episode.position_valid[f"hand_{side}"] = np.ones((count, 21), bool)
    episode = prepare(episode)
    arrays = resample(episode)
    assert not arrays["observation.valid.body_floor_rot6d_xyz_position"].any()
    assert not arrays["observation.valid.hand_left_floor_position"].any()
    window = prepare_window(arrays, episode.metadata, start_frame=6, num_frames=4, fps=15)
    assert window.arrays["observation.valid.body_floor_rot6d_xyz_position"].all()
    assert window.arrays["observation.valid.hand_left_floor_position"].all()
    assert not np.array_equal(window.arrays[BODY_STATE_COLUMNS[0]], arrays[BODY_STATE_COLUMNS[0]][[6, 8, 10, 12]])
    validate_unified_dense22(window.arrays, window.arrays, window.metadata)
    np.testing.assert_allclose(window.arrays[BODY_STATE_COLUMNS[1]].reshape(4, 22, 9)[..., 6:],
                               window.arrays["observation.state.body_head_floor"].reshape(4, 22, 3), atol=2e-6)
    floor_rotation = quat_to_matrix(window.arrays["observation.state.head_reference_floor_world_rotation"])
    for side in ("left", "right"):
        local = window.arrays[f"observation.state.hand_{side}_floor"].reshape(4, 21, 3)
        rebuilt = np.einsum("tij,tkj->tki", floor_rotation, local)
        rebuilt += window.arrays["observation.state.head_reference_floor_world"][:, None]
        np.testing.assert_allclose(rebuilt, window.arrays[f"observation.state.hand_{side}_world"].reshape(4, 21, 3), atol=2e-6)


def test_schema_registry_and_dense_upgrade_gate(tmp_path):
    v3 = feature_specs_for_schema(LEGACY_SCHEMA_VERSION)
    v4 = feature_specs_for_schema(V4_SCHEMA_VERSION)
    v5 = feature_specs_for_schema(SCHEMA_VERSION)
    assert set(v3) == set(v4) and set(v5)-set(v4) == set(unified_specs())
    assert v3["observation.features.dino"]["shape"] == [1024]
    assert v4["observation.features.dino"]["shape"] == [5120]
    archive = Archive(tmp_path/"archive", {"test": "gate"})
    atomic_json(archive.root/".pipeline/dense_v5_upgrade.json", {"state": "committing"})
    with pytest.raises(ValueError, match="Dense v5 integration is unfinished"):
        Archive(archive.root)
    assert Archive(archive.root, _allow_upgrade=True).schema_version == SCHEMA_VERSION
    atomic_json(archive.root/".pipeline/dense_v5_upgrade.json", {"state": "complete"})
    assert Archive(archive.root).specs == v5


def test_new_archive_has_physical_layout_digest_and_dino_independent_provenance(tmp_path):
    episode = prepare(moving_episode())
    archive = Archive(tmp_path/"archive", {"test": "digest"})
    archive.add(episode, resample(episode), "source-input")
    archive.finalize()
    arrays = read_arrays(archive.data_path(0), archive.specs)
    provenance = archive.records()[0]["motion_representation"]
    assert provenance == representation_provenance(arrays)
    assert len(provenance["producer_signature"]) == len(provenance["motion_digest"]) == 64
    arrays["observation.features.dino"][:] = 3.14
    arrays["observation.valid.dino"][:] = True
    arrays["observation.video_timestamp"][:] = 26.
    assert representation_provenance(arrays) == provenance
    arrays["observation.state.hand_left_world"][0, 3] += .1
    assert representation_provenance(arrays)["motion_digest"] != provenance["motion_digest"]


@pytest.mark.parametrize("entrypoint", ["records", "data_path", "record_path", "read_window", "add", "finalize", "dino", "parallel_dino"])
def test_preopened_archive_rechecks_new_upgrade_gate(tmp_path, entrypoint):
    from egomotion_pipeline.features import extract_features
    from egomotion_pipeline.parallel_features import extract_features_parallel
    archive = Archive(tmp_path/"archive", {"test": "hotgate"})
    archive.records()  # Populate the cache before an upgrade starts.
    atomic_json(archive.root/".pipeline/dense_v5_upgrade.json", {"state": "in_progress"})
    calls = {"records": archive.records, "data_path": lambda: archive.data_path(0),
             "record_path": lambda: archive.record_path(0),
             "read_window": lambda: archive.read_window(0, start_frame=0, num_frames=2),
             "add": lambda: archive.add(None, {}, "unused"), "finalize": archive.finalize,
             "dino": lambda: extract_features(archive, {}),
             "parallel_dino": lambda: extract_features_parallel(archive, {}, [0])}
    with pytest.raises(ValueError, match="Dense v5 integration is unfinished"):
        calls[entrypoint]()


def test_preopened_archive_rejects_changed_schema_even_after_completed_upgrade(tmp_path):
    from egomotion_pipeline.storage import read_json
    archive = Archive(tmp_path/"archive", {"test": "stale_schema"})
    path = archive.root/".pipeline/config.json"
    config = read_json(path)
    config["schema_version"] = V4_SCHEMA_VERSION
    atomic_json(path, config)
    with pytest.raises(ValueError, match="schema changed.*reopen"):
        archive.records()


def test_new_corrected_source_does_not_claim_native_joint_axes(tmp_path):
    episode = prepare(moving_episode())
    episode.source = "egosuite"
    episode.metadata["rotation_basis_verified"] = False
    episode.metadata["body_basis_correction"] = np.broadcast_to(np.eye(3), (22, 3, 3)).tolist()
    archive = Archive(tmp_path/"archive", {"test": "corrected_source_label"})
    archive.add(episode, resample(episode), "source-input")
    record = archive.records()[0]
    assert record["metadata"]["target_joint_basis"] == "source_corrected_unspecified"
    assert record["metadata"]["unified_dense22"]["target_joint_basis"] == "source_corrected_unspecified"
    assert record["metadata"]["rotation_basis_verified"] is False
    assert record["metadata"]["unified_dense22"]["rotation_basis_verified"] is False


def test_explicit_corrected_target_axes_are_retained_in_source_metadata():
    from egomotion_pipeline.storage import joint_basis_metadata
    metadata = {"body_basis_correction": [[[1]]], "target_joint_basis": "documented_target_v1"}
    result = joint_basis_metadata(metadata)
    assert result == metadata and result is not metadata
    with pytest.raises(ValueError, match="nonempty string"):
        joint_basis_metadata({**metadata, "target_joint_basis": ""})
