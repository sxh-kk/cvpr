"""Native30Hz cache binding, 5Hz motion selection and historical-only conditions."""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from ImageProcess import SharedImageProcessor
from MotionExpert.config import as_dict, config_from_dict, load_config, smplh_motion_expert_config
from MotionExpert.dataset.smplh_archive import SMPLHParameterArchive, DINO_COLUMN
from MotionExpert.dataset.smplh_motion_dataset import SMPLHMotionDataset
from MotionExpert.dataset.vision_cache import CONTRACT, VisionProfileCache, storage_digest
from MotionExpert.tests.fixtures import owned_archive, rewrite_episode, _json


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def write_cache(root, output, *, fps=30):
    archive = SMPLHParameterArchive(root)
    records = archive.records()
    process = SharedImageProcessor("256x480").metadata
    encoder = {"checkpoint": "unit-test-frozen-dino", "shape": [5, 1024], "image_process": process}
    step = 30 // fps
    total_frames = valid_frames = 0
    for record in records:
        directory = output / "episodes" / f"{record['episode_index']:08d}"
        directory.mkdir(parents=True)
        selected = np.arange(0, record["length"], step, dtype=np.int64)
        motion = pq.read_table(archive.data_path(record["episode_index"]))
        keys = ("frame_index", "episode_index", "index", "timestamp", "observation.source_timestamp",
                "observation.valid.dino", "observation.video_timestamp", "observation.video_time_error")
        join = {key: motion[key].combine_chunks().to_numpy(zero_copy_only=False)[selected].copy() for key in keys}
        join["source.video_frame_index"] = selected.copy()
        join["source.valid.dino"] = join["observation.valid.dino"].copy()
        value = np.broadcast_to(selected[:, None, None].astype(np.float32) + 1000,
                                (len(selected), 5, 1024)).copy()
        value[~join["observation.valid.dino"]] = 0
        np.save(directory / "features.npy", value)
        np.savez(directory / "join.npz", **join)
        count = int(join["observation.valid.dino"].sum())
        total_frames += len(selected)
        valid_frames += count
        receipt = {"contract": CONTRACT, "state": "complete", "frames": len(selected),
                   "episode_index": record["episode_index"], "source_key": record["source_key"],
                   "source_record_sha256": storage_digest(record), "valid_count": count,
                   "validity_equal_source": True, "encoder": encoder,
                   "output_identities": {name: {"size_bytes": (directory / name).stat().st_size,
                                                 "mtime_ns": (directory / name).stat().st_mtime_ns}
                                         for name in ("features.npy", "join.npz")}}
        _json(directory / "receipt.json", receipt)
    binding = {key: archive.binding[key] for key in ("info_sha256", "episodes_metadata_sha256", "split_metadata_sha256")}
    binding["records_sha256"] = storage_digest(records)
    _json(output / "manifest.json", {
        "contract": CONTRACT, "state": "complete", "scope": "all", "source_root": str(root),
        "source_binding": binding, "image_process": process,
        "fps": fps, "native_fps": 30, "frame_phase": 0, "sampling_step": step,
        "shape": [5, 1024], "storage_dtype": "float32", "episode_indices": list(range(len(records))),
        "source_episode_count": len(records), "completed_episodes": len(records),
        "total_frames": total_frames, "completed_frames": total_frames, "valid_frames": valid_frames,
        "validity_equal_source": True, "encoder": encoder,
    })
    return output


@pytest.fixture
def cache(owned_archive, tmp_path):
    return write_cache(owned_archive, tmp_path / "vision-256x480")


def update_cache_episode(cache, episode, *, mutate_join=None, mutate_features=None):
    directory = cache / "episodes" / f"{episode:08d}"
    receipt = json.loads((directory / "receipt.json").read_text())
    manifest = json.loads((cache / "manifest.json").read_text())
    if mutate_join is not None:
        with np.load(directory / "join.npz") as saved:
            join = {key: saved[key].copy() for key in saved.files}
        mutate_join(join)
        np.savez(directory / "join.npz", **join)
        previous = receipt["valid_count"]
        receipt["valid_count"] = int(join["observation.valid.dino"].sum())
        manifest["valid_frames"] += receipt["valid_count"] - previous
    if mutate_features is not None:
        value = np.load(directory / "features.npy")
        result = mutate_features(value)
        np.save(directory / "features.npy", value if result is None else result)
    receipt["output_identities"] = {
        name: {"size_bytes": (directory / name).stat().st_size, "mtime_ns": (directory / name).stat().st_mtime_ns}
        for name in ("features.npy", "join.npz")}
    _json(directory / "receipt.json", receipt)
    _json(cache / "manifest.json", manifest)


def test_new_features_are_mmap_prefix_only_and_master_motion_is_unchanged(owned_archive, cache):
    baseline = SMPLHMotionDataset(archive_root=owned_archive, do_normalization=False, cond_img_feat=False)
    data = SMPLHMotionDataset(archive_root=owned_archive, do_normalization=False,
                            vision_profile_root=cache, image_profile="256x480")
    assert data.archive.archive_fingerprint == baseline.archive.archive_fingerprint
    assert data.image_profile == "psi0_256x480_v1"
    assert data.idx_to_sidx_fidx == baseline.idx_to_sidx_fidx
    data.reader.read_log.clear()
    sample = data[0]
    assert sample["y"]["img_embs"].shape == (11, 5, 1024)
    assert sample["y"]["valid_img_embs"].tolist() == [True] * 6 + [False] * 5
    torch.testing.assert_close(sample["y"]["img_embs"][:6, 0, 0], torch.arange(0, 36, 6).float() + 1000)
    assert not sample["y"]["img_embs"][6:].any()
    assert sample["y"]["img_mask"][6:].bool().all()
    for key in ("motion", "traj", "feature_valid_mask", "trajectory_valid"):
        torch.testing.assert_close(sample["misc"][key], baseline[0]["misc"][key], rtol=0, atol=0)
    assert not any(DINO_COLUMN in columns for _, _, columns in data.reader.read_log)
    mapped = data.archive.vision_cache._cache[0]["features"]
    assert isinstance(mapped, np.memmap) and not mapped.flags.writeable
    feature_cache = data.archive.vision_cache
    assert feature_cache.fps == 30
    assert mapped.shape == (91, 5, 1024)
    assert feature_cache.metadata["sampling_step"] == 30 // feature_cache.fps
    assert data.fps == 5 and data.sampling_step == 6
    assert json.loads(sample["misc"]["metadata_json"])["vision_profile"]["image_process"]["width"] == 480
    copied = pickle.loads(pickle.dumps(data.archive.vision_cache))
    assert not copied._cache
    assert len(copied.read(data.records[0], np.arange(0, 36, 6), load_features=True)[DINO_COLUMN]) == 6


def test_future_features_are_not_materialized_or_validated_as_conditions(owned_archive, cache):
    manifest = json.loads((cache / "manifest.json").read_text())
    first_future_row = 36 // manifest["sampling_step"]
    update_cache_episode(cache, 0, mutate_features=lambda value: value.__setitem__(slice(first_future_row, None), np.nan))
    data = SMPLHMotionDataset(archive_root=owned_archive, do_normalization=False, vision_profile_root=cache)
    sample = data[0]
    assert torch.isfinite(sample["y"]["img_embs"]).all()
    assert not sample["y"]["valid_img_embs"][6:].any()
    assert not sample["y"]["img_embs"][6:].any()


@pytest.mark.parametrize("change", ["source", "validity", "profile_hash", "fps_step"])
def test_incompatible_manifest_is_rejected_before_training(owned_archive, cache, change):
    path = cache / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "pending":
        manifest["state"] = "extracting"
    elif change == "subset":
        manifest["scope"] = "subset"
    elif change == "phase":
        manifest["frame_phase"] = 1
    elif change == "source":
        manifest["source_binding"]["records_sha256"] = "0" * 64
    elif change == "validity":
        manifest["validity_equal_source"] = False
    elif change == "profile_hash":
        manifest["image_process"]["profile_sha256"] = "0" * 64
    elif change == "encoder_profile":
        manifest["encoder"]["image_process"] = SharedImageProcessor("256x320").metadata
    else:
        manifest["sampling_step"] = 6 if manifest["fps"] == 30 else 1
    _json(path, manifest)
    with pytest.raises(ValueError):
        SMPLHMotionDataset(archive_root=owned_archive, do_normalization=False, vision_profile_root=cache)


def test_explicit_image_profile_and_feature_shapes_are_checked(owned_archive, cache):
    with pytest.raises(ValueError, match="IMAGE_PROFILE"):
        SMPLHMotionDataset(archive_root=owned_archive, do_normalization=False,
                          vision_profile_root=cache, image_profile="256x320")
    update_cache_episode(cache, 0, mutate_features=lambda value: value[:, :4])
    data = SMPLHMotionDataset(archive_root=owned_archive, do_normalization=False, vision_profile_root=cache)
    with pytest.raises(ValueError, match="features must be float32"):
        data[0]




def test_lying_validity_equivalence_cannot_reuse_original_population(owned_archive, cache):
    def mutation(join):
        join["observation.valid.dino"][0] = False
        join["source.valid.dino"][0] = False
    update_cache_episode(cache, 0, mutate_join=mutation)
    with pytest.raises(ValueError, match="validity join differs"):
        SMPLHMotionDataset(archive_root=owned_archive, do_normalization=False,
                          vision_profile_root=cache, min_valid_image_fraction=.5)


def test_window_index_uses_new_masks_without_loading_any_embeddings(owned_archive, tmp_path):
    def invalidate(arrays):
        arrays["observation.valid.dino"][:36] = False
    rewrite_episode(owned_archive, 0, motion_mutation=invalidate, vision_mutation=invalidate)
    profile = write_cache(owned_archive, tmp_path / "vision-invalid-prefix")
    data = SMPLHMotionDataset(archive_root=owned_archive, do_normalization=False,
                            vision_profile_root=profile, cond_img_feat=False, min_valid_image_fraction=.5)
    assert data.rejected_windows["insufficient_real_DINO"] == 1
    assert (0, 0) not in data.idx_to_sidx_fidx and (0, 30) in data.idx_to_sidx_fidx
    assert all(entry["features"] is None for entry in data.archive.vision_cache._cache.values())
    assert all(DINO_COLUMN not in columns for _, _, columns in data.reader.read_log)
    assert not data[0]["y"]["valid_img_embs"].any()
