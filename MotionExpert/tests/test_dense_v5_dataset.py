"""V5 authority, causal window reconstruction and stable motion provenance."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from MotionExpert.config import K12_JOINTS
from MotionExpert.compute_joint_motion_stats import compute_stats
from MotionExpert.dataset.dense_motion_dataset import DenseMotionDataset, IMAGE_COLUMNS
from MotionExpert.dataset.dense_representation import (
    BODY_WORLD_6D_COLUMN, BODY_WORLD_6D_MASKS, build_dense_window, quaternion_matrix,
)
from MotionExpert.dataset.split_manifest import (
    V5_SCHEMA, archive_fingerprint, read_records, validate_manifest,
)
from MotionExpert.tests.test_dense_dataset import synthetic_arrays, synthetic_metadata, write_archive


def v5_arrays(original):
    arrays = deepcopy(original)
    count = len(arrays["timestamp"])
    rotation = quaternion_matrix(arrays["observation.state.body_world_rotation"].reshape(count, 22, 4))
    positions = arrays["observation.state.body_world"].reshape(count, 22, 3)
    arrays[BODY_WORLD_6D_COLUMN] = np.concatenate([rotation[..., :2, :].reshape(count, 22, 6), positions], -1).reshape(count, 198).astype(np.float32)
    for kind, key in BODY_WORLD_6D_MASKS.items():
        arrays[key] = arrays[f"observation.valid.body_{kind}"].copy()
        arrays[key.replace(".valid.", ".interpolated.")] = np.zeros((count, 22), bool)
    # These caches must never replace task-specific ground reconstruction.
    arrays["observation.state.body_floor_rot6d_xyz"] = np.full((count, 198), 999., np.float32)
    for kind in ("rotation", "position"):
        for flag in ("valid", "interpolated"):
            arrays[f"observation.{flag}.body_floor_rot6d_xyz_{kind}"] = np.zeros((count, 22), bool)
    for side in ("left", "right"):
        arrays[f"observation.state.hand_{side}_floor"] = np.full((count, 63), 999., np.float32)
        for flag in ("valid", "interpolated"):
            arrays[f"observation.{flag}.hand_{side}_floor_position"] = np.zeros((count, 21), bool)
    return arrays


def rewrite_v5(root, arrays, records, manifest_path):
    info = {"schema_version": V5_SCHEMA, "fps": 30,
            "features": {key: {"shape": [1 if value.ndim == 1 else value.shape[1]]}
                         for key, value in arrays.items()}}
    (root/"meta/info.json").write_text(json.dumps(info))
    (root/".pipeline/config.json").write_text(json.dumps({"schema_version": V5_SCHEMA}))
    (root/".pipeline/dense_v5_upgrade.json").write_text(json.dumps({"state": "complete"}))
    motion_digest = hashlib.sha256(arrays[BODY_WORLD_6D_COLUMN].tobytes()).hexdigest()
    for record in records:
        columns = {key: (pa.array(values) if values.ndim == 1 else
                        pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1]))
                   for key, values in arrays.items()}
        pq.write_table(pa.table(columns), root/f"data/chunk-000/file-{record['episode_index']:03d}.parquet", row_group_size=5)
        record["motion_representation"] = {"schema_version": V5_SCHEMA,
            "producer_signature": "a"*64, "motion_digest": motion_digest}
        record["metadata"]["unified_dense22"] = {"joint_basis": "egosuite_native_v1"}
        (root/f".pipeline/episodes/{record['episode_index']:08d}.json").write_text(json.dumps(record))
    manifest = json.loads(manifest_path.read_text())
    manifest["archive_fingerprint"] = archive_fingerprint(records)
    manifest_path.write_text(json.dumps(manifest))


def dataset_for(root, manifest_path, **kwargs):
    return DenseMotionDataset(archive_root=root, split_manifest=manifest_path,
        window=4, forecast_prefix=2, segment_stride=3, do_normalization=False,
        rotation_basis_mode="source_native", target_joint_basis="egosuite_native_v1", **kwargs)


def test_v5_authoritative_body_matches_v4_and_projects_only_required_columns(tmp_path):
    root = tmp_path/"archive"; root.mkdir()
    original = synthetic_arrays(11)
    manifest, records = write_archive(root, original)
    expected = dataset_for(root, manifest)[0]
    arrays = v5_arrays(original)
    arrays["observation.state.body_world"][:] = -500
    arrays["observation.state.body_world_rotation"][:] = 0
    arrays["observation.valid.body_position"][:] = False
    arrays["observation.valid.body_rotation"][:] = False
    rewrite_v5(root, arrays, records, manifest)
    dataset = dataset_for(root, manifest)
    sample = dataset[0]
    torch.testing.assert_close(sample["misc"]["motion"], expected["misc"]["motion"], atol=2e-6, rtol=0)
    torch.testing.assert_close(sample["y"]["trajectory_valid"], expected["y"]["trajectory_valid"])
    assert sample["misc"]["motion"].shape == (4, 245)
    assert sample["y"]["traj"].shape == (4, 18)
    assert sample["y"]["img_embs"].shape == (4, 5, 1024)
    columns = [requested for _, _, requested in dataset.reader.read_log]
    assert any(BODY_WORLD_6D_COLUMN in requested for requested in columns)
    for requested in columns:
        assert "observation.state.body_world" not in requested
        assert "observation.state.body_world_rotation" not in requested
        assert "observation.state.body_floor_rot6d_xyz" not in requested
        if "observation.features.dino" in requested:
            assert set(requested) == set(IMAGE_COLUMNS)


def test_v5_independent_masks_never_promote_zero_6d_or_discard_valid_interpolation():
    arrays = v5_arrays(synthetic_arrays(5))
    rotation_key, position_key = BODY_WORLD_6D_MASKS["rotation"], BODY_WORLD_6D_MASKS["position"]
    arrays[rotation_key][2, 5] = False
    blocks = arrays[BODY_WORLD_6D_COLUMN].reshape(5, 22, 9)
    blocks[2, 5, :6] = 0
    arrays[position_key][3, 4] = False
    blocks[3, 4, 6:] = np.nan
    arrays[rotation_key.replace(".valid.", ".interpolated.")][1, 5] = True
    arrays["observation.valid.hand_left_position"][1, 7] = False
    encoded = build_dense_window(arrays, synthetic_metadata(), rotation_basis_verified=True)
    body = encoded.motion_valid[:, :108].reshape(5, 12, 9)
    assert not body[2, K12_JOINTS.index(5), :6].any()
    assert body[2, K12_JOINTS.index(5), 6:].all()
    assert body[3, K12_JOINTS.index(4), :6].all()
    assert not body[3, K12_JOINTS.index(4), 6:].any()
    assert body[1, K12_JOINTS.index(5), :6].all()
    assert not encoded.motion_valid[1, 117+7*3:117+8*3].any()
    assert np.isfinite(encoded.motion).all()
    blocks[2, 5, :6] = 0
    arrays[rotation_key][2, 5] = True
    with pytest.raises(ValueError, match="orthonormal"):
        build_dense_window(arrays, synthetic_metadata(), rotation_basis_verified=True)


def test_v5_forecast_rebuilds_floor_and_hands_from_observed_world_only(tmp_path):
    root = tmp_path/"archive"; root.mkdir()
    original = synthetic_arrays(11)
    body = original["observation.state.body_world"].reshape(-1, 22, 3)
    body[4:, [10, 11], 2] = 10
    metadata = synthetic_metadata(); metadata["ground"]["valid"] = False
    manifest, records = write_archive(root, original, metadata)
    rewrite_v5(root, v5_arrays(original), records, manifest)
    dataset = dataset_for(root, manifest, cond_img_feat=False)
    recon = dataset[0]
    fore = dataset.process_sample_for_task(recon, "fore")
    assert fore["misc"]["floor_height"] == pytest.approx(-.70, abs=1e-6)
    assert recon["misc"]["floor_height"] > 1
    selected = {key: value[[0, 2, 4, 6]] for key, value in original.items()}
    expected = build_dense_window(selected, metadata, task="fore", forecast_prefix=2,
                                  rotation_basis_verified=True)
    np.testing.assert_allclose(fore["misc"]["motion"], expected.motion, atol=2e-6)
    assert fore["y"]["traj_mask"].tolist() == [0, 0, 1, 1]
    assert fore["y"]["img_mask"].tolist() == [0, 0, 1, 1]
    assert fore["y"]["loss_mask"].tolist() == [0, 0, 1, 1]
    assert all("observation.features.dino" not in requested for _, _, requested in dataset.reader.read_log)


@pytest.mark.parametrize("state", ["planned", "staging", "validated", "committing", "failed"])
def test_pending_v5_upgrade_blocks_reader_and_split_generation(tmp_path, state):
    root = tmp_path/"archive"; root.mkdir()
    manifest, _ = write_archive(root, synthetic_arrays(5))
    (root/".pipeline/dense_v5_upgrade.json").write_text(json.dumps({"state": state}))
    with pytest.raises(ValueError, match="upgrade unfinished"):
        read_records(root)
    with pytest.raises(ValueError, match="upgrade unfinished"):
        dataset_for(root, manifest, cond_img_feat=False)


def test_v5_fingerprint_binds_motion_and_producer_but_ignores_dino_status(tmp_path):
    root = tmp_path/"archive"; root.mkdir()
    manifest, records = write_archive(root, synthetic_arrays(5))
    old_manifest = json.loads(manifest.read_text())
    rewrite_v5(root, v5_arrays(synthetic_arrays(5)), records, manifest)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_manifest(old_manifest, records)
    identity = archive_fingerprint(records)
    changed = deepcopy(records)
    changed[0]["dino"] = {"state": "pending", "signature": "updated encoder", "valid_count": 0}
    changed[0]["quality"] = {"dino": 0}
    assert archive_fingerprint(changed) == identity
    for field in ("motion_digest", "producer_signature"):
        changed = deepcopy(records)
        changed[0]["motion_representation"][field] = "b"*64
        assert archive_fingerprint(changed) != identity
    changed[0]["motion_representation"]["motion_digest"] = "invalid"
    with pytest.raises(ValueError, match="SHA256"):
        archive_fingerprint(changed)


def test_v5_stats_require_new_motion_identity_and_read_no_dino(tmp_path):
    root = tmp_path/"archive"; root.mkdir()
    original = synthetic_arrays(11)
    manifest, records = write_archive(root, original)
    legacy = dataset_for(root, manifest, cond_img_feat=False)
    old_stats = compute_stats(legacy, limit_windows=1)
    stats_path = tmp_path/"old_stats.pt"; torch.save(old_stats, stats_path)
    rewrite_v5(root, v5_arrays(original), records, manifest)
    with pytest.raises(ValueError, match="statistics split manifest mismatch"):
        DenseMotionDataset(archive_root=root, split_manifest=manifest, stats_path=stats_path,
            window=4, forecast_prefix=2, segment_stride=3, cond_img_feat=False,
            rotation_basis_mode="source_native", target_joint_basis="egosuite_native_v1")
    dataset = dataset_for(root, manifest, cond_img_feat=False)
    current = compute_stats(dataset, limit_windows=1)
    assert current["archive_fingerprint"] != old_stats["archive_fingerprint"]
    assert current["motion_mean"].shape == (245,)
    assert current["traj_mean"].shape == (18,)
    for key in ("motion_mean", "motion_std", "traj_mean", "traj_std"):
        torch.testing.assert_close(current[key], old_stats[key], atol=2e-6, rtol=0)
    assert all("observation.features.dino" not in requested for _, _, requested in dataset.reader.read_log)


def test_v5_requires_complete_motion_identity(tmp_path):
    root = tmp_path/"archive"; root.mkdir()
    manifest, records = write_archive(root, synthetic_arrays(5))
    rewrite_v5(root, v5_arrays(synthetic_arrays(5)), records, manifest)
    path = root/".pipeline/episodes/00000000.json"
    record = json.loads(path.read_text()); del record["motion_representation"]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="missing or unsupported"):
        read_records(root)


def test_existing_dataset_rejects_live_upgrade_and_stale_schema_after_cutover(tmp_path):
    root = tmp_path/"archive"; root.mkdir()
    arrays = synthetic_arrays(5)
    manifest, records = write_archive(root, arrays)
    dataset = dataset_for(root, manifest, cond_img_feat=False)
    dataset[0]  # Prime the legacy row-group cache.
    (root/".pipeline/dense_v5_upgrade.json").write_text(json.dumps({"state": "staging"}))
    with pytest.raises(ValueError, match="upgrade unfinished"):
        dataset[0]
    rewrite_v5(root, v5_arrays(arrays), records, manifest)
    with pytest.raises(ValueError, match="schema changed"):
        dataset[0]


def test_existing_v5_dataset_allows_dino_updates_but_rejects_changed_motion_identity(tmp_path):
    root = tmp_path/"archive"; root.mkdir()
    arrays = synthetic_arrays(5)
    manifest, records = write_archive(root, arrays)
    rewrite_v5(root, v5_arrays(arrays), records, manifest)
    dataset = dataset_for(root, manifest, cond_img_feat=False)
    path = root/".pipeline/episodes/00000000.json"
    record = json.loads(path.read_text()); record["dino"] = {"state": "pending"}
    path.write_text(json.dumps(record))
    dataset[0]
    record["motion_representation"]["motion_digest"] = "b"*64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="motion identity changed"):
        dataset[0]


def test_record_inventory_rechecks_gate_when_migration_starts_during_read(tmp_path, monkeypatch):
    root = tmp_path/"archive"; root.mkdir()
    write_archive(root, synthetic_arrays(5))
    first_record = root/".pipeline/episodes/00000000.json"
    original_read_text = type(root).read_text

    def start_upgrade_after_first_record(path, *args, **kwargs):
        value = original_read_text(path, *args, **kwargs)
        if path == first_record:
            (root/".pipeline/dense_v5_upgrade.json").write_text(json.dumps({"state": "staging"}))
        return value

    monkeypatch.setattr(type(root), "read_text", start_upgrade_after_first_record)
    with pytest.raises(ValueError, match="upgrade unfinished"):
        read_records(root)
