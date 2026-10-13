"""Owned parameter reads, real shape16 FK and the image6->joint11 contract."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from MotionExpert.dataset import smplh_archive as archive_module
from MotionExpert.dataset.smplh_archive import (SMPLHParameterArchive, MOTION_SPECS, PARAMETER_SCHEMA_VERSION,
                                               SMPLHBodyModel, DINO_COLUMN, PARENTS, canonical_json,
                                               file_sha256, assert_smplh_archive_ready)
from MotionExpert.dataset.smplh_motion_dataset import SMPLHMotionDataset
from MotionExpert.dataset.motion_layout import K12_JOINT_INDICES


def _json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True))


def _table(path, arrays, *, parameter=False):
    columns = {key: (pa.array(value) if value.ndim == 1 else
                    pa.FixedSizeListArray.from_arrays(pa.array(value.reshape(-1)), value.shape[1]))
               for key, value in arrays.items()}
    table = pa.table(columns)
    if parameter:
        table = table.replace_schema_metadata({b"parameter_schema_version": PARAMETER_SCHEMA_VERSION.encode()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=13, compression="zstd")


def parameter_arrays(count=91, *, episode=0):
    arrays = {key: np.zeros((count,) if width == 1 else (count, width), dtype=dtype)
              for key, (dtype, width) in MOTION_SPECS.items()}
    frames = np.arange(count)
    arrays.update(frame_index=frames.astype(np.int64), episode_index=np.full(count, episode, np.int64),
                  index=frames.astype(np.int64) + episode * count, timestamp=(frames / 30).astype(np.float32))
    arrays["observation.source_timestamp"] = frames / 30
    arrays["observation.source_frame_left"] = frames.astype(np.int64)
    arrays["observation.source_frame_right"] = frames.astype(np.int64)
    arrays["observation.video_timestamp"] = frames / 30 + .001
    arrays["observation.video_time_error"][:] = .001
    arrays["smplh.trans"][:] = [2, 3, .5]
    arrays["smplh.trans"][:, 0] += frames * .002
    arrays["smplh.root_orient"][:, 2] = .2 + frames * .01
    arrays["smplh.pose_body"][:, 8] = frames * .003
    arrays["smplh.betas"][:, 0] = .2
    arrays["smplh.betas"][:, 15] = frames * .004
    arrays["source.feet_world"][:, [2, 5]] = .52
    arrays["source.feet_world"][:, [0, 3]] = 2 + frames[:, None] * .002
    arrays["source.feet_valid"][:] = True
    for component in ("position", "rotation"):
        arrays[f"estimated.valid.body_{component}"][:] = True
        arrays[f"source.observed.body_{component}"][:] = True
        arrays[f"quality.{component}_confidence"][:] = .6
    for key in arrays:
        if key.startswith("observation.valid."):
            arrays[key][:] = True
    arrays["observation.valid.floor_verified"][:] = False
    for name in ("head", "camera", "head_reference"):
        arrays[f"observation.state.{name}_world"][:] = [5, 4, 1.8]
        arrays[f"observation.state.{name}_world"][:, 0] += frames * .002
        yaw = .3 + frames * .01
        arrays[f"observation.state.{name}_world_rotation"] = np.stack(
            [np.cos(yaw / 2), np.zeros(count), np.zeros(count), np.sin(yaw / 2)], -1).astype(np.float32)
    for side in ("left", "right"):
        arrays[f"observation.state.hand_{side}_world"] = np.arange(count * 63, dtype=np.float32).reshape(count, 63) * .001
    arrays["observation.environment.floor_height"][:] = .5
    arrays["observation.environment.floor_normal"][:, 2] = 1
    return arrays


@pytest.fixture
def owned_archive(tmp_path, monkeypatch):
    root = tmp_path / "parameters"
    model_path = root / "weights/smplh/model.npz"
    model_path.parent.mkdir(parents=True)
    parents = PARENTS.tolist()
    for wrist in (20, 21):
        for _ in range(5):
            first = len(parents)
            parents.extend([wrist, first, first + 1])
    template = np.zeros((6890, 3), np.float32)
    template[0] = [.3, -.2, .4]
    for joint in range(1, 52):
        template[joint] = template[parents[joint]] + [.01, .02, .03]
    regressor = np.zeros((52, 6890), np.float32)
    regressor[np.arange(52), np.arange(52)] = 1
    shapes = np.zeros((6890, 3, 16), np.float32)
    for beta in range(16):
        shapes[:52, beta % 3, beta] = (.001 + np.arange(52) * .0001) * (beta + 1)
    np.savez_compressed(model_path, v_template=template, shapedirs=shapes,
                        J_regressor=regressor, kintree_table=np.array([parents, np.arange(52)]))
    model_sha = file_sha256(model_path)
    monkeypatch.setattr(archive_module, "MODEL_SHA256", model_sha)
    info = {"schema_version": PARAMETER_SCHEMA_VERSION, "fps": 30, "requires_old_dense_archive": False,
            "model": {"path": "weights/smplh/model.npz", "model_sha256": model_sha,
                      "body_joint_count": 22, "joint_count": 52, "shape_basis_count": 16},
            "dino": {"shape": [5, 1024], "storage_dtype": "float32", "flatten_order": "token_major"}}
    _json(root / "meta/info.json", info)
    records = []
    for idx in range(4):
        source = "egoexo4d_hm" if idx in (1, 2) else "egosuite"
        split = "val" if idx == 3 else "train"
        arrays = parameter_arrays(episode=idx)
        # Estimated completion stays separate from source observations.
        if idx == 0:
            arrays["source.observed.body_rotation"][:, 13] = False
            arrays["estimated.completed.body_rotation"][:, 13] = True
            arrays["observation.valid.hand_left_position"][:, 7] = False
        mp, vp = root / f"episodes/{idx:08d}/motion.parquet", root / f"episodes/{idx:08d}/vision.parquet"
        _table(mp, arrays, parameter=True)
        vision = {key: arrays[key].copy() for key in ("frame_index", "episode_index", "timestamp", "observation.source_timestamp",
                                                           "observation.valid.dino", "observation.video_timestamp", "observation.video_time_error")}
        # Published Suite promotes its source float32 grid; HM keeps float64.
        vision["timestamp"] = arrays["timestamp"].astype(np.float64) if source == "egosuite" else np.arange(91) / 30
        vision[DINO_COLUMN] = np.repeat(np.arange(91, dtype=np.float32)[:, None], 5120, axis=1)
        vision[DINO_COLUMN][36:] = np.nan  # Future visuals must never enter conditions.
        _table(vp, vision)
        metadata = {"coordinate_verified": idx != 2, "head_reference": {"basis_verified": True},
                    "rotation_basis_verified": source != "egosuite", "target_joint_basis": "smplh_body22_axes_v1",
                    "label_kind": "model_fitted_estimate", "body_basis_correction": None,
                    "ground": {"height_m": .5, "valid": True, "verified": False, "mode": "estimate",
                               "window_fallback": {"mode": "mean_valid_feet", "offset_m": .02}},
                    "training_eligible": idx != 2, "quarantine_reasons": ["camera_world_mismatch"] if idx == 2 else [],
                    "source_data": "/deleted-dense-source/motion.parquet"}
        if source == "egosuite":
            metadata["target_model_axes_defined"] = True
        record = {"episode_index": idx, "source": source, "source_key": f"{source}:{idx}", "source_uid": str(idx),
                  "length": 91, "dataset_from_index": idx * 91, "dataset_to_index": (idx + 1) * 91,
                  "split": split, "data_path": str(mp.relative_to(root)), "vision_path": str(vp.relative_to(root)),
                  "motion_sha256": file_sha256(mp), "vision_sha256": file_sha256(vp), "metadata": metadata,
                  "split_group_ids": [f"participant:{idx}", f"capture:{idx}"], "dino": {"state": "complete"}}
        _json(root / f".pipeline/episodes/{idx:08d}.json", record)
        receipt = {"schema_version": PARAMETER_SCHEMA_VERSION, "record": record, "completed_at": "2026-10-12",
                   "conversion_binding": "0" * 64, "motion": {"sha256": record["motion_sha256"]},
                   "vision": {"sha256": record["vision_sha256"]}, "output_identities": {
                       role: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                       for role, p in (("motion", mp), ("vision", vp))}}
        _json(root / f"episodes/{idx:08d}/receipt.json", receipt)
        records.append(record)
    splits = {name: [r["source_key"] for r in records if r["split"] == name] for name in ("train", "val", "test")}
    _json(root / "meta/splits.json", {"splits": splits, "split_group_ids": {g: r["split"] for r in records for g in r["split_group_ids"]}})
    fields = ("episode_index", "source", "source_key", "source_uid", "length", "dataset_from_index", "dataset_to_index", "split", "data_path", "vision_path")
    pq.write_table(pa.table({k: [r[k] for r in records] for k in fields}), root / "meta/episodes.parquet")
    _json(root / ".pipeline/dataset-state.json", {"state": "accepted", "independent_acceptance_pending": False,
                                                 "schema_version": PARAMETER_SCHEMA_VERSION, "episodes": 4, "frames": 364})
    return root


def dataset(root, **kwargs):
    return SMPLHMotionDataset(archive_root=root, do_normalization=False, **kwargs)


def rewrite_episode(root, index, *, motion_mutation=None, vision_mutation=None, metadata_mutation=None):
    record_path = root / f".pipeline/episodes/{index:08d}.json"
    receipt_path = root / f"episodes/{index:08d}/receipt.json"
    record, receipt = json.loads(record_path.read_text()), json.loads(receipt_path.read_text())
    if metadata_mutation:
        metadata_mutation(record["metadata"])
        original_motion_mutation = motion_mutation
        def sync_motion_metadata(arrays):
            if original_motion_mutation:
                original_motion_mutation(arrays)
            ground = record["metadata"]["ground"]
            arrays["observation.environment.floor_height"][:] = np.float32(ground["height_m"])
            arrays["observation.valid.floor"][:] = ground["valid"]
            arrays["observation.valid.floor_verified"][:] = ground["valid"] and ground.get("verified", False)
            arrays["observation.valid.head_reference_basis"][:] = record["metadata"]["head_reference"]["basis_verified"]
        motion_mutation = sync_motion_metadata
    for role, mutation in (("motion", motion_mutation), ("vision", vision_mutation)):
        if mutation is None:
            continue
        path = root / record["data_path" if role == "motion" else "vision_path"]
        table = pq.read_table(path)
        arrays = {}
        for key in table.column_names:
            column = table[key].combine_chunks()
            arrays[key] = (column.values.to_numpy(zero_copy_only=False).reshape(len(table), column.type.list_size).copy()
                           if pa.types.is_fixed_size_list(column.type) else column.to_numpy(zero_copy_only=False).copy())
        mutation(arrays)
        _table(path, arrays, parameter=role == "motion")
        record[f"{role}_sha256"] = file_sha256(path)
        receipt[role]["sha256"] = record[f"{role}_sha256"]
        receipt["output_identities"][role] = {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    receipt["record"] = record
    _json(record_path, record); _json(receipt_path, receipt)


def test_parameters_fk_shape16_and_prefix_only_visuals(owned_archive):
    d = dataset(owned_archive, min_valid_image_fraction=.5)
    assert d.motion_feature_dim == 245 and not d.diagnostic
    assert d.repre_type == "smplh12_rot6d_hands21_contacts_v1"
    assert d.sparse_joint_indices == list(K12_JOINT_INDICES)
    assert len(d.records) == 2 and len(d.rejected_episodes) == 1
    assert set(r["source"] for r in d.records) == {"egosuite", "egoexo4d_hm"}
    sample = d[0]
    assert sample["misc"]["motion"].shape == (11, 245)
    assert sample["y"]["loss_mask"].bool().all()
    assert sample["y"]["valid_img_embs"].tolist() == [True] * 6 + [False] * 5
    assert sample["y"]["img_mask"].tolist() == [0] * 6 + [1] * 5
    assert "motion_generation_mask" not in sample["y"]
    np.testing.assert_array_equal(sample["y"]["img_embs"][:, 0, 0], [0, 6, 12, 18, 24, 30] + [0] * 5)
    assert torch.isfinite(sample["y"]["img_embs"]).all()
    assert sample["misc"]["source.observed.body_rotation"][:, 13].eq(False).all()
    assert sample["misc"]["estimated.completed.body_rotation"][:, 13].all()
    assert sample["misc"]["feature_valid_mask"][:, :108].reshape(11, 12, 9)[:, 5, :6].all()
    assert not sample["misc"]["feature_valid_mask"][:, 117 + 7 * 3:117 + 8 * 3].any()
    assert not sample["misc"]["observation.valid.hand_left_position"][:, 7].any()
    assert sample["misc"]["observation.interpolated.hand_left_position"].shape == (11, 21)
    assert sample["misc"]["source.feet_valid"].shape == (11, 2)
    assert sample["misc"]["source.feet_world"].shape == (11, 6)
    assert "source.feet_world" not in sample["y"]
    assert "smplh.betas" not in sample["y"] and "betas" not in sample["y"]
    assert "observation.state.body_world" not in sample["y"]
    meta = json.loads(sample["misc"]["metadata_json"])
    assert meta["rotation_basis_verified"] is False and meta["target_model_axes_defined"] is True
    assert meta["training_window"]["source_frame_indices"] == list(range(0, 61, 6))
    decoded = d.ret_to_full_sequence(sample)
    arrays, _ = d.archive.read_window(0, load_dino=False)
    full_positions = arrays["observation.state.body_world"].reshape(11, 22, 3)
    np.testing.assert_allclose(decoded["body_positions"], full_positions[:, K12_JOINT_INDICES], atol=2e-6)
    assert decoded["body_positions"].shape == (11, 12, 3)
    assert not {"pose_body", "root_orient", "trans", "betas", "local_axis_angle"} & set(decoded)
    np.testing.assert_array_equal(sample["misc"]["smplh.betas"], arrays["smplh.betas"])
    np.testing.assert_array_equal(sample["misc"]["observation.state.body_world"], arrays["observation.state.body_world"])
    np.testing.assert_array_equal(sample["misc"]["observation.state.body_world_rotation"], arrays["observation.state.body_world_rotation"])
    assert not Path(meta["source_data"]).exists()
    assert len(d.reader.read_log) <= 64 and d.reader.cached_bytes <= d.reader.max_bytes


def test_motion_only_statistics_inputs_read_no_embeddings_and_filter_quarantine(owned_archive):
    d = dataset(owned_archive, cond_img_feat=False, episode_indices=[0, 1, 2])
    assert d.diagnostic
    d.reader.read_log.clear()
    for i in range(len(d)):
        d[i]
    assert all(DINO_COLUMN not in columns for _, _, columns in d.reader.read_log)
    with pytest.raises(KeyError):
        d.get_from_seq_and_st("egoexo4d_hm:2", 0)
    with pytest.raises(ValueError, match="phase"):
        d.get_from_seq_and_st(d.seq_names[0], 2)
    with pytest.raises(ValueError, match="complete"):
        d.get_from_seq_and_st(d.seq_names[0], 36)


def test_publication_gate_and_diagnostic_override(owned_archive):
    state_path = owned_archive / ".pipeline/dataset-state.json"
    state = json.loads(state_path.read_text())
    state.update(state="converted_pending_acceptance", independent_acceptance_pending=True)
    _json(state_path, state)
    with pytest.raises(ValueError, match="acceptance"):
        assert_smplh_archive_ready(owned_archive)
    with pytest.raises(ValueError, match="acceptance"):
        dataset(owned_archive)
    d = dataset(owned_archive, allow_unpublished=True, episode_indices=[0])
    assert d.allow_unpublished and d.diagnostic and d[0]["misc"]["motion"].shape == (11, 245)


def test_optional_real_report_binding_and_owned_archive_relocation(owned_archive, tmp_path):
    a = SMPLHParameterArchive(owned_archive)
    report = {"format": "independent-standalone-smplh-parameter-full-audit-v1", "scope": "full", "failures": [],
              **{key: True for key in ("passed", "parameter_and_source_integrity_passed", "original_egosuite_full_fit_acceptance_passed",
                                       "old_dense_dependency_absent", "cleanup_allowed")},
              "population": {"episodes": 4, "frames": 364}, "binding": a.binding}
    path = tmp_path / "external-job-acceptance.json"
    _json(path, report)
    state_path = owned_archive / ".pipeline/dataset-state.json"
    state = json.loads(state_path.read_text()); state["acceptance_report"] = str(path); _json(state_path, state)
    assert SMPLHParameterArchive(owned_archive).archive_fingerprint == a.archive_fingerprint
    report["binding"]["split_metadata_sha256"]["meta/splits.json"] = "0" * 64; _json(path, report)
    with pytest.raises(ValueError, match="split_metadata_sha256"):
        SMPLHParameterArchive(owned_archive)
    path.unlink()  # External provenance may be absent when an owned archive moves.
    assert SMPLHParameterArchive(owned_archive).archive_fingerprint == a.archive_fingerprint


def test_commit_changed_after_reader_init_is_rejected(owned_archive):
    a = SMPLHParameterArchive(owned_archive)
    path = owned_archive / "episodes/00000000/motion.parquet"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        a.read_window(0)
    with pytest.raises(ValueError, match="identity changed"):
        SMPLHParameterArchive(owned_archive)


def test_borrowed_model_and_split_leak_are_rejected(owned_archive):
    info_path = owned_archive / "meta/info.json"
    info = json.loads(info_path.read_text()); original = deepcopy(info)
    info["model"]["path"] = str(owned_archive / info["model"]["path"]); _json(info_path, info)
    with pytest.raises(ValueError, match="owned relative"):
        SMPLHParameterArchive(owned_archive)
    _json(info_path, original)
    path = owned_archive / "meta/splits.json"
    manifest = json.loads(path.read_text()); manifest["splits"]["val"].append(manifest["splits"]["train"][0]); _json(path, manifest)
    with pytest.raises(ValueError, match="overlapping"):
        SMPLHParameterArchive(owned_archive)


def test_model_uses_shape_component15_and_shaped_root_offset(owned_archive):
    a = SMPLHParameterArchive(owned_archive)
    arrays, _ = a.read_window(0, load_dino=False)
    p, _ = a.model.decode(arrays)
    root = a.model.rest[0] + np.einsum("tb,cb->tc", arrays["smplh.betas"], a.model.shape_basis[0])
    np.testing.assert_allclose(p[:, 0], root + arrays["smplh.trans"], atol=2e-7)
    changed = {k: v.copy() for k, v in arrays.items()}
    changed["smplh.betas"][:, 15] += 1
    p2, _ = a.model.decode(changed)
    np.testing.assert_allclose((p2 - p)[:, 0], np.broadcast_to(a.model.shape_basis[0, :, 15], (11, 3)), atol=3e-7)
    # Independent per-frame scipy SO(3), with the full sixteen-beta shaped rest.
    from scipy.spatial.transform import Rotation
    for frame in (0, 10):
        rest = a.model.rest + np.einsum("b,jcb->jc", changed["smplh.betas"][frame], a.model.shape_basis)
        aa = np.r_[changed["smplh.root_orient"][frame][None], changed["smplh.pose_body"][frame].reshape(21, 3)]
        local = Rotation.from_rotvec(aa).as_matrix()
        reference, rotations = np.zeros((22, 3)), np.zeros((22, 3, 3))
        reference[0], rotations[0] = rest[0] + changed["smplh.trans"][frame], local[0]
        for joint in range(1, 22):
            parent = PARENTS[joint]
            rotations[joint] = rotations[parent] @ local[joint]
            reference[joint] = reference[parent] + rotations[parent] @ (rest[joint] - rest[parent])
        np.testing.assert_allclose(p2[frame], reference, atol=2e-7)
    assert np.max(np.abs(p2 - p)) > .015


def test_betas_stay_audit_metadata_with_no_shape_target_or_condition(owned_archive):
    def mutation(arrays):
        arrays["estimated.valid.body_position"][:, 0] = False
        arrays["quality.position_confidence"][:, 0] = 0
    rewrite_episode(owned_archive, 0, motion_mutation=mutation)
    d = dataset(owned_archive)
    sample = d.get_from_seq_and_st("egosuite:0", 0)
    assert sample["misc"]["feature_valid_mask"].shape == (11, 245)
    assert not sample["misc"]["feature_valid_mask"][:, 6:9].any()
    assert sample["misc"]["motion"][:, 6:9].eq(0).all()
    assert sample["misc"]["smplh.betas"].shape == (11, 16)
    assert "betas" not in sample["y"] and "smplh.betas" not in sample["y"]
    assert sample["misc"]["source.observed.body_position"][:, 0].all()


def test_only_original_k12_projection_is_supported_and_betas_cannot_condition(owned_archive):
    d = dataset(owned_archive, sparse_joint_indices=list(K12_JOINT_INDICES))
    assert d[0]["misc"]["motion"].shape == (11, 245)
    with pytest.raises(ValueError, match="twelve sparse joints"):
        dataset(owned_archive, sparse_joint_indices=list(range(22)))
    with pytest.raises(ValueError, match="source FK and audit"):
        dataset(owned_archive, cond_betas=True)


def _invalidate_sparse_body(arrays, rows):
    for component in ("position", "rotation"):
        for prefix in ("estimated.valid", "source.observed", "source.interpolated", "estimated.completed"):
            arrays[f"{prefix}.body_{component}"][rows, list(K12_JOINT_INDICES)] = False
        arrays[f"quality.{component}_confidence"][rows, list(K12_JOINT_INDICES)] = 0
    arrays["source.feet_valid"][rows] = False
    arrays["source.feet_interpolated"][rows] = False


def test_index_rejects_future_labels_available_only_outside_k12(owned_archive):
    rewrite_episode(owned_archive, 0, motion_mutation=lambda arrays: _invalidate_sparse_body(arrays, slice(36, None)))
    # Non-selected body joints, hands and camera deltas remain valid.
    with pytest.raises(ValueError, match="no_valid_future_body_labels"):
        dataset(owned_archive, episode_indices=[0], segment_stride=100)


def test_direct_window_guard_checks_only_108_body_channels(owned_archive):
    rewrite_episode(owned_archive, 0, motion_mutation=lambda arrays: _invalidate_sparse_body(arrays, slice(36, 61)))
    d = dataset(owned_archive, episode_indices=[0])
    assert len(d) > 0 and (0, 0) not in d.idx_to_sidx_fidx
    with pytest.raises(ValueError, match="no valid future body labels"):
        d.get_from_seq_and_st("egosuite:0", 0)


def test_sparse_statistics_bind_k12_and_normalize_only_245_targets(owned_archive, monkeypatch):
    from MotionExpert import config
    from MotionExpert.compute_joint_motion_stats import _accumulate, _finalize_accumulators
    d = dataset(owned_archive, cond_img_feat=False)
    monkeypatch.setattr(config, "SMPLH_MODEL_SHA256", d.model_sha256)
    stats = _finalize_accumulators(_accumulate(d, range(len(d)), progress=False))
    assert stats["representation"] == "smplh12_rot6d_hands21_contacts_v1"
    assert stats["joint_indices"] == list(K12_JOINT_INDICES)
    assert stats["motion_mean"].shape == stats["motion_std"].shape == (245,)
    assert stats["motion_count"].shape == (245,)
    assert stats["statistics_scope"] == "all_valid_frames"
    normalized = SMPLHMotionDataset(archive_root=owned_archive, stats_path=stats, cond_img_feat=False)
    assert normalized[0]["misc"]["motion"].shape == (11, 245)
    np.testing.assert_array_equal(normalized[0]["misc"]["smplh.betas"], d[0]["misc"]["smplh.betas"])
    broken = dict(stats, joint_indices=list(range(22)))
    with pytest.raises(ValueError, match="joint_indices"):
        d._validate_statistics(broken)
    with pytest.raises(ValueError, match="matching model, schema"):
        SMPLHMotionDataset(archive_root=owned_archive, stats_path=broken)


@pytest.mark.parametrize("key", ["frame_index", "episode_index", "observation.video_time_error"])
def test_vision_join_uses_exact_owned_identity_and_time(owned_archive, key):
    def mutation(arrays):
        arrays[key][6] += 1
    rewrite_episode(owned_archive, 0, vision_mutation=mutation)
    a = SMPLHParameterArchive(owned_archive)
    with pytest.raises(ValueError, match="vision/motion"):
        a.read_window(0)


def test_estimated_completion_boundary_cannot_be_promoted_from_numeric_values(owned_archive):
    def mutation(arrays):
        arrays["estimated.completed.body_rotation"][:, 13] = False
    rewrite_episode(owned_archive, 0, motion_mutation=mutation)
    a = SMPLHParameterArchive(owned_archive)
    with pytest.raises(ValueError, match="completion"):
        a.read_window(0, load_dino=False)


def test_prefix_ground_uses_source_feet_and_predicted_history_is_not_world_anchor(owned_archive):
    rewrite_episode(owned_archive, 0, metadata_mutation=lambda meta: meta["ground"].update(valid=False))
    d = dataset(owned_archive, episode_indices=[0])
    baseline = d[0]
    assert baseline["misc"]["floor_height"] == pytest.approx(.5, abs=1e-7)
    def mutation(arrays):
        arrays["source.feet_world"].reshape(-1, 2, 3)[36:, :, 2] += 9
    rewrite_episode(owned_archive, 0, motion_mutation=mutation)
    changed = dataset(owned_archive, episode_indices=[0])[0]
    assert changed["misc"]["floor_height"] == baseline["misc"]["floor_height"]
    torch.testing.assert_close(changed["misc"]["motion"][:6], baseline["misc"]["motion"][:6])
    current = dataset(owned_archive, episode_indices=[0])
    pred = deepcopy(changed); pred["pred"]["motion"] = changed["misc"]["motion"].clone()
    future = current.ret_to_full_sequence(pred)
    pred["pred"]["motion"][:6] = torch.randn(6, 245) * 100
    future2 = current.ret_to_full_sequence(pred)
    assert future["body_positions"].shape == (5, 12, 3)
    torch.testing.assert_close(future["body_positions"], future2["body_positions"], atol=0, rtol=0)


def test_prefix_without_original_feet_cannot_use_fitted_or_future_feet(owned_archive):
    def mutation(arrays):
        arrays["source.feet_valid"][:36] = False
        arrays["source.observed.body_position"][:36, [10, 11]] = False
        arrays["estimated.completed.body_position"][:36, [10, 11]] = True
    rewrite_episode(owned_archive, 0, motion_mutation=mutation,
                    metadata_mutation=lambda meta: meta["ground"].update(valid=False))
    with pytest.raises(ValueError, match="No usable"):
        dataset(owned_archive, episode_indices=[0], segment_stride=100)


@pytest.mark.parametrize("key", ["observation.valid.floor", "observation.valid.head_reference_basis",
                                  "observation.environment.floor_height", "observation.environment.floor_normal"])
def test_stored_reference_fields_must_match_declared_coordinate_policy(owned_archive, key):
    def mutation(arrays):
        arrays[key][0] = False if arrays[key].dtype == bool else 7
    rewrite_episode(owned_archive, 0, motion_mutation=mutation)
    a = SMPLHParameterArchive(owned_archive)
    with pytest.raises(ValueError, match="reference|floor"):
        a.read_window(0, load_dino=False)
