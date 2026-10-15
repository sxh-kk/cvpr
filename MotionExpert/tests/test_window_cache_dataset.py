"""Tiny producer-to-Dataset checks; no training or production data access."""
from copy import deepcopy
from functools import wraps
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from MotionExpert import config as config_module
from MotionExpert.config import smplh_motion_expert_config
from MotionExpert.dataset.data_module import SMPLHMotionDataModule, cache_training_collate
from MotionExpert.dataset.smplh_archive import SMPLHParameterArchive
from MotionExpert.dataset.smplh_motion_dataset import SMPLHMotionDataset
from MotionExpert.tests.fixtures import owned_archive, smplh_stats, rewrite_episode
from MotionExpert.tests.test_vision_profile_cache import write_cache


@pytest.fixture
def cached_archive(owned_archive, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    vision = write_cache(owned_archive, tmp_path / "vision")
    archive = SMPLHParameterArchive(owned_archive, vision_profile_root=vision)
    monkeypatch.setattr(config_module, "SMPLH_MODEL_SHA256", archive.model_sha256)
    cfg = smplh_motion_expert_config()
    cfg.DATA.ARCHIVE_ROOT = cfg.DATA.DATA_DIR = str(owned_archive)
    cfg.DATA.SPLIT_MANIFEST = str(owned_archive / "meta/splits.json")
    cfg.DATA.VISION_PROFILE_ROOT = str(vision)
    cfg.DATA.MODEL_SHA256 = archive.model_sha256
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.BATCH_SIZE = 2
    cfg.DATA.PIN_MEMORY = False
    cfg.DATA.PERSISTENT_WORKERS = False
    cfg.DATA.STATS_PATH = str(tmp_path / "stats.pt")
    stats = smplh_stats(cfg)
    stats.update(archive_fingerprint=archive.archive_fingerprint,
                 split_fingerprint=archive.split_fingerprint, basis_mapping_fingerprint="none")
    stats["motion_mean"] = torch.linspace(-.3, .4, 245)
    stats["motion_std"] = torch.linspace(.7, 1.5, 245)
    stats["traj_mean"] = torch.linspace(-.2, .3, 18)
    stats["traj_std"] = torch.linspace(.8, 1.4, 18)
    stats["motion_mean"][108:117] = stats["traj_mean"][9:18]
    stats["motion_std"][108:117] = stats["traj_std"][9:18]
    stats["motion_mean"][243:245] = 0
    stats["motion_std"][243:245] = 1
    torch.save(stats, cfg.DATA.STATS_PATH)
    from MotionExpert.build_window_cache import build_window_cache
    output = tmp_path / "windows"
    build_window_cache(cfg, output, workers=0)
    cfg.DATA.WINDOW_CACHE_ROOT = str(output)
    return cfg


def _dataset(cfg, *, cached, split="train", normalized=True):
    return SMPLHMotionDataset(
        archive_root=cfg.DATA.ARCHIVE_ROOT, split=split, stats_path=cfg.DATA.STATS_PATH,
        do_normalization=normalized, vision_profile_root=cfg.DATA.VISION_PROFILE_ROOT,
        image_profile=cfg.DATA.IMAGE_PROFILE, min_valid_image_fraction=cfg.DATA.MIN_VALID_IMAGE_FRACTION,
        window_cache_root=cfg.DATA.WINDOW_CACHE_ROOT if cached else None)


def _same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _same(left[key], right[key])
    else:
        assert left == right


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize("normalized", [False, True])
def test_cached_windows_keep_order_conditions_normalization_and_decode(cached_archive, split, normalized):
    original = _dataset(cached_archive, cached=False, split=split, normalized=normalized)
    cached = _dataset(cached_archive, cached=True, split=split, normalized=normalized)
    assert cached.records == original.records
    assert cached.idx_to_sidx_fidx == original.idx_to_sidx_fidx
    assert cached.archive.archive_fingerprint == original.archive.archive_fingerprint
    for index in range(len(cached)):
        baseline, fast = original[index], cached[index]
        _same(baseline["y"], fast["y"])
        for key, value in fast["misc"].items():
            if key == "metadata_json":
                assert json.loads(value) == json.loads(baseline["misc"][key])
            else:
                _same(baseline["misc"][key], value)
        assert not fast["y"]["img_embs"][6:].any()
        assert not fast["y"]["valid_img_embs"][6:].any()
        fast["pred"]["motion"] = fast["misc"]["motion"]
        baseline["pred"]["motion"] = baseline["misc"]["motion"]
        _same(cached.ret_to_full_sequence(fast), original.ret_to_full_sequence(baseline))


def test_cache_worker_hot_path_skips_source_index_parquet_fk_and_join(cached_archive, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Cached sample touched the source read path")
    # Archive activation still checks identity, but cached activation imports its
    # published window index rather than issuing per-record Parquet index reads.
    @wraps(SMPLHMotionDataset._index_record)
    def forbidden_index(*args, **kwargs):
        return forbidden(*args, **kwargs)
    monkeypatch.setattr(SMPLHMotionDataset, "_index_record", forbidden_index)
    cached = _dataset(cached_archive, cached=True)
    monkeypatch.setattr(cached.archive, "assert_current", forbidden)
    monkeypatch.setattr(cached.archive.reader, "read", forbidden)
    monkeypatch.setattr(cached.archive.model, "decode", forbidden)
    monkeypatch.setattr(cached.archive.vision_cache, "_episode", forbidden)
    monkeypatch.setattr(cached.archive.vision_cache, "assert_current", forbidden)
    first, last = cached[0], cached[len(cached) - 1]
    assert first["y"]["img_embs"].shape == (11, 5, 1024)
    assert first["misc"]["motion"].shape == last["misc"]["motion"].shape == (11, 245)
    assert "smplh.betas" not in first["misc"]


def test_full_sample_and_sequence_interfaces_keep_original_audit(cached_archive):
    original = _dataset(cached_archive, cached=False)
    cached = _dataset(cached_archive, cached=True)
    _same(original.get_sample(0), cached.get_sample(0))
    assert cached.get_sample(0)["misc"]["smplh.betas"].shape == (11, 16)
    _same(original.get_from_seq_and_st(original.seq_names[0], 6, task="fore"),
          cached.get_from_seq_and_st(cached.seq_names[0], 6, task="fore"))
    _same(cached.process_sample_for_task(cached[0], "fore"), cached.get_sample(0, task="fore"))


def test_cache_loaders_keep_sampler_order_and_drop_only_unused_batch_fields(cached_archive):
    original_cfg = deepcopy(cached_archive)
    original_cfg.DATA.WINDOW_CACHE_ROOT = None
    original = SMPLHMotionDataModule(original_cfg)
    cached = SMPLHMotionDataModule(cached_archive)
    original.setup("fit")
    cached.setup("fit")
    original_train, cached_train = original.train_dataloader(), cached.train_dataloader()
    torch.manual_seed(73)
    before = list(original_train.sampler)
    torch.manual_seed(73)
    assert list(cached_train.sampler) == before
    assert original_train.drop_last == cached_train.drop_last
    assert cached_train.collate_fn is cache_training_collate
    before, after = next(iter(original.val_dataloader())), next(iter(cached.val_dataloader()))
    _same(before["y"], after["y"])
    assert set(after["misc"]) == {"motion", "source", "seq_name"}
    for key in after["misc"]:
        _same(before["misc"][key], after["misc"][key])
    assert "metadata_json" in before["misc"]


def test_cache_checks_run_once_per_split_epoch_in_main_process(cached_archive, monkeypatch):
    data = SMPLHMotionDataModule(cached_archive)
    data.setup("fit")
    checks = []
    monkeypatch.setattr(data.train_dataset, "assert_cache_current", lambda: checks.append("train"))
    monkeypatch.setattr(data.val_dataset, "assert_cache_current", lambda: checks.append("val"))
    data.trainer = SimpleNamespace(current_epoch=0, training=True, validating=False,
                                   testing=False, predicting=False)
    batch = {"sentinel": object()}
    assert data.on_before_batch_transfer(batch, 0) is batch
    assert data.on_before_batch_transfer(batch, 0) is batch
    assert checks == ["train"]
    data.trainer.current_epoch = 1
    data.on_before_batch_transfer(batch, 0)
    data.trainer.training, data.trainer.validating = False, True
    data.on_before_batch_transfer(batch, 0)
    data.on_before_batch_transfer(batch, 0)
    assert checks == ["train", "train", "val"]


def test_cache_activation_rejects_protocol_and_statistics_mismatch(cached_archive, tmp_path):
    with pytest.raises(ValueError, match="binding"):
        SMPLHMotionDataset(archive_root=cached_archive.DATA.ARCHIVE_ROOT, do_normalization=False,
                          vision_profile_root=cached_archive.DATA.VISION_PROFILE_ROOT,
                          window_cache_root=cached_archive.DATA.WINDOW_CACHE_ROOT,
                          min_valid_image_fraction=.25)
    stats = torch.load(cached_archive.DATA.STATS_PATH, weights_only=True)
    stats["split_fingerprint"] = "wrong-train-split"
    path = tmp_path / "wrong-stats.pt"
    torch.save(stats, path)
    cfg = deepcopy(cached_archive)
    cfg.DATA.STATS_PATH = str(path)
    with pytest.raises(ValueError, match="split_fingerprint"):
        _dataset(cfg, cached=True)


def test_completed_resume_keeps_all_committed_mtimes(cached_archive):
    from MotionExpert.build_window_cache import build_window_cache
    root = Path(cached_archive.DATA.WINDOW_CACHE_ROOT)
    paths = [root / "manifest.json", root / "progress.json", *root.rglob("*.npy"), *root.rglob("receipt.json")]
    before = {path: path.stat().st_mtime_ns for path in paths}
    manifest = json.loads((root / "manifest.json").read_text())
    assert build_window_cache(cached_archive, root, workers=0, block_frames=11) == manifest
    assert {path: path.stat().st_mtime_ns for path in paths} == before


def test_partial_episode_failure_resumes_without_rewriting_committed_shard(cached_archive, tmp_path, monkeypatch):
    from MotionExpert import build_window_cache as builder
    root = tmp_path / "interrupted-episodes"
    original = builder._write_episode
    calls = []
    def interrupted(task):
        if calls:
            raise RuntimeError("simulated episode interruption")
        calls.append(True)
        return original(task)
    monkeypatch.setattr(builder, "_write_episode", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        builder.build_window_cache(cached_archive, root, workers=0, block_frames=11)
    assert json.loads((root / "manifest.json").read_text())["state"] == "building"
    first = root / "episodes/00000000/motion.npy"
    modified = first.stat().st_mtime_ns
    monkeypatch.setattr(builder, "_write_episode", original)
    manifest = builder.build_window_cache(cached_archive, root, workers=0, block_frames=11)
    assert manifest["state"] == "complete"
    assert first.stat().st_mtime_ns == modified
    assert json.loads((root / "progress.json").read_text())["resumed_episodes"] == 1


def test_diagnostic_subset_is_rejected_for_formal_training(cached_archive, tmp_path):
    from MotionExpert import build_window_cache as builder
    from MotionExpert.dataset.window_cache import WindowCache, protocol_from_config
    root = tmp_path / "diagnostic-windows"
    with pytest.raises(ValueError, match="diagnostic"):
        builder.build_window_cache(cached_archive, root, workers=0, selected_episodes=[0])
    manifest = builder.build_window_cache(cached_archive, root, workers=0,
                                         selected_episodes=[0], diagnostic=True)
    assert manifest["scope"] == "selected_episodes" and manifest["diagnostic"]
    archive = _dataset(cached_archive, cached=False).archive
    with pytest.raises(ValueError, match="Diagnostic"):
        WindowCache(root, archive=archive, split="train", protocol=protocol_from_config(cached_archive))
    reader = WindowCache(root, archive=archive, split="train", protocol=protocol_from_config(cached_archive),
                         allow_diagnostic=True)
    assert len(reader.records) == 1


def test_completed_resume_rejects_corrupt_episode_digest(cached_archive):
    from MotionExpert.build_window_cache import build_window_cache
    root = Path(cached_archive.DATA.WINDOW_CACHE_ROOT)
    with (root / "episodes/00000000/motion.npy").open("ab") as stream:
        stream.write(b"broken")
    with pytest.raises(ValueError, match="digest/identity"):
        build_window_cache(cached_archive, root, workers=0)


def test_cross_block_ground_fallback_feet_and_camera_masks_match_original(cached_archive, tmp_path):
    from MotionExpert.build_window_cache import build_window_cache
    def geometry(arrays):
        frames = arrays["frame_index"]
        arrays["source.feet_world"][:, [2, 5]] = .52 + frames[:, None] * .001
        arrays["source.observed.body_position"][::12, 10] = False
        arrays["estimated.completed.body_position"][:] = (
            arrays["estimated.valid.body_position"] & ~arrays["source.observed.body_position"])
        arrays["source.feet_valid"][:] = arrays["source.observed.body_position"][:, [10, 11]]
        arrays["observation.valid.head_reference_rotation"][6] = False
    rewrite_episode(Path(cached_archive.DATA.ARCHIVE_ROOT), 0, motion_mutation=geometry,
                    metadata_mutation=lambda metadata: metadata["ground"].update(valid=False))
    cfg = deepcopy(cached_archive)
    cfg.DATA.VISION_PROFILE_ROOT = str(write_cache(Path(cfg.DATA.ARCHIVE_ROOT), tmp_path / "fallback-vision"))
    cfg.DATA.WINDOW_CACHE_ROOT = str(tmp_path / "fallback-windows")
    build_window_cache(cfg, cfg.DATA.WINDOW_CACHE_ROOT, workers=0, block_frames=11)
    original, cached = _dataset(cfg, cached=False, normalized=False), _dataset(cfg, cached=True, normalized=False)
    heights = []
    for index, (sequence, _) in enumerate(cached.idx_to_sidx_fidx):
        before, after = original[index], cached[index]
        _same(before["y"], after["y"])
        for key in ("motion", "traj", "feature_valid_mask", "trajectory_valid", "floor_transform"):
            _same(before["misc"][key], after["misc"][key])
        assert json.loads(before["misc"]["metadata_json"])["ground"] == json.loads(after["misc"]["metadata_json"])["ground"]
        if sequence == 0:
            assert after["misc"]["ground_fallback_used"]
            heights.append(after["misc"]["floor_height"])
    assert len(set(heights)) > 1


def test_packed_hot_path_never_opens_episode_arrays_or_closes_global_mmaps(cached_archive, monkeypatch):
    cached = _dataset(cached_archive, cached=True)
    reader = cached.window_cache
    reader.file_cache = 1
    owners = dict(reader._packed)
    episode_root = Path(cached_archive.DATA.WINDOW_CACHE_ROOT) / "episodes"
    original_load = np.load
    def guarded_load(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).is_relative_to(episode_root) and str(path).endswith(".npy"):
            raise AssertionError("Packed sample opened an episode array")
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", guarded_load)
    for index in (0, 2, 1, 3, 0, 3, 2, 1):
        sample = cached[index]
        native = sample["misc"]["start_idx"] + torch.arange(6) * 6
        torch.testing.assert_close(sample["y"]["img_embs"][:6, 0, 0], native.float() + 1000)
        assert all(reader._packed[key] is value and not value._mmap.closed for key, value in owners.items())
    assert not reader._cache
    assert len(reader._image_cache) == 1


def _refresh_array_identity(root, identity, values):
    from MotionExpert.dataset.window_cache import file_sha256
    path = root / identity["path"]
    np.save(path, values, allow_pickle=False)
    identity.update(size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, sha256=file_sha256(path))


@pytest.mark.parametrize("change", ["duplicate_start", "outside_start", "split_key", "packed_identity", "start_digest", "extra_descriptor", "descriptor_source"])
def test_packed_activation_rejects_wrong_window_keys_or_identity(cached_archive, change):
    from MotionExpert.dataset.window_cache import atomic_json, digest, file_sha256
    root = Path(cached_archive.DATA.WINDOW_CACHE_ROOT)
    manifest = json.loads((root / "manifest.json").read_text())
    packed = manifest["packed"]["train"]
    if change in ("duplicate_start", "outside_start"):
        identity = packed["arrays"]["start_frames"]
        values = np.load(root / identity["path"], allow_pickle=False)
        values[1 if change == "duplicate_start" else 0] = 0 if change == "duplicate_start" else 120
        _refresh_array_identity(root, identity, values)
    elif change == "split_key":
        index = manifest["indices"]["train"]
        path = root / index["path"]
        rows = np.load(path, allow_pickle=False)
        rows[0, 1] = 30
        np.save(path, rows, allow_pickle=False)
        index["sha256"] = packed["index_sha256"] = file_sha256(path)
    elif change == "packed_identity":
        packed["arrays"]["motion"]["mtime_ns"] += 1
    elif change == "start_digest":
        descriptor = next(item for item in manifest["episodes"] if item["episode_index"] == 0)
        receipt_path = root / descriptor["receipt_path"]
        receipt = json.loads(receipt_path.read_text())
        receipt["start_frames_sha256"] = "0" * 64
        atomic_json(receipt_path, receipt)
        descriptor["receipt_sha256"] = file_sha256(receipt_path)
        sources = [{"episode_index": item["episode_index"], "receipt_sha256": item["receipt_sha256"]}
                   for item in manifest["episodes"] if item["split"] == "train"]
        packed["source_receipts_sha256"] = digest(sources)
    elif change == "extra_descriptor":
        extra = deepcopy(manifest["episodes"][0])
        extra["episode_index"] = 2  # Quarantined, absent from cached records.
        manifest["episodes"].append(extra)
        sources = [{"episode_index": item["episode_index"], "receipt_sha256": item["receipt_sha256"]}
                   for item in manifest["episodes"] if item["split"] == "train"]
        packed["source_receipts_sha256"] = digest(sources)
    else:
        manifest["episodes"][0]["source_record_sha256"] = "0" * 64
    atomic_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="start-frame|index keys|identity|descriptor|population|order"):
        _dataset(cached_archive, cached=True)


def test_packing_interruption_is_unpublished_and_reuses_committed_split(cached_archive, tmp_path, monkeypatch):
    from MotionExpert import build_window_cache as builder
    root = tmp_path / "interrupted-packing"
    original = builder._pack_split
    def interrupted(root, split, *args, **kwargs):
        if split == "val":
            raise RuntimeError("simulated packing interruption")
        return original(root, split, *args, **kwargs)
    monkeypatch.setattr(builder, "_pack_split", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        builder.build_window_cache(cached_archive, root, workers=0)
    assert json.loads((root / "manifest.json").read_text())["state"] == "building"
    first = root / "packed/train/motion.npy"
    modified = first.stat().st_mtime_ns
    cfg = deepcopy(cached_archive)
    cfg.DATA.WINDOW_CACHE_ROOT = str(root)
    with pytest.raises(ValueError, match="complete"):
        _dataset(cfg, cached=True)
    monkeypatch.setattr(builder, "_pack_split", original)
    result = builder.build_window_cache(cached_archive, root, workers=0)
    assert result["state"] == "complete"
    assert first.stat().st_mtime_ns == modified


def test_episode_commit_rejects_start_keys_corrupted_during_save(cached_archive, tmp_path, monkeypatch):
    from MotionExpert import build_window_cache as builder
    root = tmp_path / "corrupt-written-starts"
    original_save = np.save
    original_memmap = np.lib.format.open_memmap
    corrupted = []

    def guarded_memmap(path, *args, **kwargs):
        assert Path(path).name != "start_frames.npy", "Start keys must use an ordinary small NPY write"
        return original_memmap(path, *args, **kwargs)

    def corrupt_save(path, values, *args, **kwargs):
        if Path(path).name == "start_frames.npy" and Path(path).parent.name == ".00000000.building":
            assert len(values) > 1 and np.any(values)
            values = np.zeros_like(values)
            corrupted.append(Path(path))
        return original_save(path, values, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, "open_memmap", guarded_memmap)
    monkeypatch.setattr(np, "save", corrupt_save)
    with pytest.raises(ValueError, match="Written episode start-frame keys differ"):
        builder.build_window_cache(cached_archive, root, workers=0)
    assert len(corrupted) == 1
    assert json.loads((root / "manifest.json").read_text())["state"] == "building"
    assert not (root / "episodes/00000000").exists()
    assert not (root / "episodes/.00000000.building/receipt.json").exists()


@pytest.mark.parametrize("forge_semantic_digest", [False, True])
def test_packing_rejects_start_keys_even_when_file_checksum_matches(
        cached_archive, tmp_path, monkeypatch, forge_semantic_digest):
    from MotionExpert import build_window_cache as builder
    from MotionExpert.dataset.window_cache import atomic_json, digest, file_sha256
    root = tmp_path / "corrupt-packing-starts"
    original_pack = builder._pack_split
    corrupted = []

    def corrupt_before_pack(output, split, binding, index, descriptors):
        if split == "train":
            descriptor = next(item for item in descriptors if item["windows"] > 1)
            receipt_path = output / descriptor["receipt_path"]
            receipt = json.loads(receipt_path.read_text())
            identity = receipt["arrays"]["start_frames"]
            values = np.zeros(identity["shape"], dtype=np.int64)
            _refresh_array_identity(output, identity, values)
            assert file_sha256(output / identity["path"]) == identity["sha256"]
            if forge_semantic_digest:
                receipt["start_frames_sha256"] = digest(values.tolist())
            atomic_json(receipt_path, receipt)
            descriptor["receipt_sha256"] = file_sha256(receipt_path)
            corrupted.append(descriptor["episode_index"])
        return original_pack(output, split, binding, index, descriptors)

    monkeypatch.setattr(builder, "_pack_split", corrupt_before_pack)
    with pytest.raises(ValueError, match="start-frame keys differ from the frozen split index"):
        builder.build_window_cache(cached_archive, root, workers=0)
    assert len(corrupted) == 1
    assert json.loads((root / "manifest.json").read_text())["state"] == "building"
    assert not (root / "packed/train").exists()
    assert not (root / "packed/.train.building/receipt.json").exists()


def test_packed_commit_rejects_start_keys_corrupted_during_flush(cached_archive, tmp_path, monkeypatch):
    from MotionExpert import build_window_cache as builder
    root = tmp_path / "corrupt-packed-starts"
    original_flush = np.memmap.flush
    corrupted = []

    def corrupt_flush(value, *args, **kwargs):
        path = Path(value.filename)
        if path.name == "start_frames.npy" and path.parent.name == ".train.building":
            assert np.any(value)
            value[:] = 0
            corrupted.append(path)
        return original_flush(value, *args, **kwargs)

    monkeypatch.setattr(np.memmap, "flush", corrupt_flush)
    with pytest.raises(ValueError, match="Packed start-frame keys differ.*after writing"):
        builder.build_window_cache(cached_archive, root, workers=0)
    assert len(corrupted) == 1
    assert json.loads((root / "manifest.json").read_text())["state"] == "building"
    assert not (root / "packed/train").exists()
    assert not (root / "packed/.train.building/receipt.json").exists()


@pytest.mark.parametrize("key", ["trajectory_valid", "motion"])
def test_episode_commit_rejects_nonstart_array_corrupted_during_save(
        cached_archive, tmp_path, monkeypatch, key):
    from MotionExpert import build_window_cache as builder
    root = tmp_path / "corrupt-episode-array"
    original_save = np.save
    original_memmap = np.lib.format.open_memmap
    corrupted = []

    def guarded_memmap(path, *args, **kwargs):
        assert "episodes" not in Path(path).parts, "Episode arrays must be generated in RAM"
        return original_memmap(path, *args, **kwargs)

    def corrupt_save(path, values, *args, **kwargs):
        if Path(path).name == f"{key}.npy" and Path(path).parent.name == ".00000000.building":
            values = values.copy()
            prefix = values.reshape(-1)[:max(1, 3912 // values.dtype.itemsize)]
            assert np.any(prefix)
            prefix[:] = 0
            corrupted.append(Path(path))
        return original_save(path, values, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, "open_memmap", guarded_memmap)
    monkeypatch.setattr(np, "save", corrupt_save)
    with pytest.raises(ValueError, match=f"Written episode array differs.*{key}"):
        builder.build_window_cache(cached_archive, root, workers=0)
    assert len(corrupted) == 1
    assert json.loads((root / "manifest.json").read_text())["state"] == "building"
    assert not (root / "episodes/00000000").exists()
    assert not (root / "episodes/.00000000.building/receipt.json").exists()


@pytest.mark.parametrize("key", ["trajectory_valid", "motion"])
def test_packed_commit_rejects_nonstart_array_corrupted_during_flush(
        cached_archive, tmp_path, monkeypatch, key):
    from MotionExpert import build_window_cache as builder
    root = tmp_path / "corrupt-packed-array"
    original_flush = np.memmap.flush
    corrupted = []

    def corrupt_flush(value, *args, **kwargs):
        path = Path(value.filename)
        if path.name == f"{key}.npy" and path.parent.name == ".train.building":
            prefix = value.reshape(-1)[:max(1, 3912 // value.dtype.itemsize)]
            assert np.any(prefix)
            prefix[:] = 0
            corrupted.append(path)
        return original_flush(value, *args, **kwargs)

    monkeypatch.setattr(np.memmap, "flush", corrupt_flush)
    with pytest.raises(ValueError, match=f"Packed array payload differs.*{key}"):
        builder.build_window_cache(cached_archive, root, workers=0)
    assert len(corrupted) == 1
    assert json.loads((root / "manifest.json").read_text())["state"] == "building"
    assert not (root / "packed/train").exists()
    assert not (root / "packed/.train.building/receipt.json").exists()


@pytest.mark.parametrize("shape,dtype", [((0, 11, 18), "float32"), ((9, 11, 18), "bool"),
                                       ((9, 11, 245), "float32")])
def test_payload_digest_uses_bounded_c_order_views(shape, dtype):
    import hashlib
    from MotionExpert.build_window_cache import _update_payload_digest
    value = np.arange(np.prod(shape), dtype=np.int64).reshape(shape).astype(dtype)
    blocks = []
    result = hashlib.sha256()

    class BoundedDigest:
        def update(self, block):
            assert isinstance(block, memoryview)
            assert block.nbytes <= 257
            blocks.append(block.nbytes)
            result.update(block)

    _update_payload_digest(BoundedDigest(), value, block_bytes=257)
    assert result.digest() == hashlib.sha256(value.tobytes(order="C")).digest()
    assert sum(blocks) == value.nbytes


@pytest.mark.parametrize("first_stage", ["fit", "validate"])
def test_cached_data_module_reuses_one_verified_archive_across_setup_stages(
        cached_archive, monkeypatch, first_stage):
    expected = {split: _dataset(cached_archive, cached=True, split=split)[0]
                for split in ("train", "val")}
    original_init = SMPLHParameterArchive.__init__
    constructors = []

    def counted_init(archive, *args, **kwargs):
        constructors.append(archive)
        return original_init(archive, *args, **kwargs)

    monkeypatch.setattr(SMPLHParameterArchive, "__init__", counted_init)
    data = SMPLHMotionDataModule(cached_archive)
    data.setup(first_stage)
    data.setup("fit")
    assert len(constructors) == 1
    assert data.train_dataset.archive is data.val_dataset.archive is constructors[0]
    assert data.train_dataset.window_cache is not data.val_dataset.window_cache
    for split in ("train", "val"):
        _same(getattr(data, f"{split}_dataset")[0], expected[split])
    train, validation = data.train_dataset, data.val_dataset
    data.setup("fit")
    data.setup("validate")
    assert len(constructors) == 1
    assert data.train_dataset is train and data.val_dataset is validation
