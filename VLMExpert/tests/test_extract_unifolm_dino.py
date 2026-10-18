"""Cache commit/resume/provenance tests without models, GPUs or LeRobot imports."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("_extract_unifolm_dino_test", ROOT / "VLMExpert/scripts/data/extract_unifolm_dino.py")
extract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extract)
cache = extract.cache

from ImageProcess import SharedImageProcessor


class Encoder:
    def __init__(self):
        self.calls = 0
        self.metadata = {"revision": "local", "local_file_sha256": {"model.safetensors": "a" * 64},
                         "image_process": SharedImageProcessor("256x320").metadata,
                         "preprocessing": {"do_resize": False, "do_center_crop": False,
                                           "do_rescale": True, "rescale_factor": 1 / 255,
                                           "do_normalize": True, "image_mean": [0.485, 0.456, 0.406],
                                           "image_std": [0.229, 0.224, 0.225]}}

    def prepare(self, images, *, processed, image_processor):
        assert processed
        assert image_processor.metadata == self.metadata["image_process"]
        assert all(image.shape == (256, 320, 3) and image.dtype == np.uint8 for image in images)
        return images

    def encode_prepared(self, images):
        self.calls += 1
        return np.stack([np.full((5, 1024), image[0, 0, 0], dtype=np.float32) for image in images])


@pytest.fixture
def planned(tmp_path):
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    info = {"fps": 30, "total_episodes": 2, "total_frames": 8, "chunks_size": 1000,
            "data_path": "data/episode_{episode_index:06d}.parquet",
            "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {"observation.images.egocentric": {"dtype": "video", "video_info": {"video.fps": 30}}}}
    (source / "meta/info.json").write_text(json.dumps(info))
    (source / "meta/episodes.jsonl").write_text("\n".join(json.dumps({"episode_index": i, "length": 4}) for i in range(2)))
    for i in range(2):
        for relative in (f"data/episode_{i:06d}.parquet", f"videos/observation.images.egocentric/episode_{i:06d}.mp4"):
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake source bytes" + bytes([i]))
    binding, items = extract.plan_source(source)
    encoder = Encoder()
    manifest = extract.build_manifest(binding, items, encoder.metadata)
    output = tmp_path / "cache"
    output.mkdir()
    cache.atomic_json(output / "manifest.json", manifest)
    return source, output, items, manifest, encoder


def rows(source, item):
    return np.arange(4, dtype=np.int64), np.arange(4, dtype=np.float64) / 30


def frames(count=4, *, shift=0):
    for i in range(count):
        yield i, i / 30 + shift, np.full((48, 64, 3), i + 1, dtype=np.uint8)


def commit(planned, *, index=0, iterator=None):
    source, output, items, manifest, encoder = planned
    receipt = extract.extract_episode(items[index], manifest, output, encoder, batch_size=3, workers=2,
                                      row_reader=rows, frame_iterator=frames() if iterator is None else iterator)
    return receipt


def finish_manifest(planned):
    _, output, _, manifest, _ = planned
    completed = {**manifest, "state": "complete"}
    cache.atomic_json(output / "manifest.json", completed)
    return completed


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_native_features_mmap_sidecar_and_canonical_rgb(planned, dtype):
    source, output, items, manifest, encoder = planned
    manifest["storage_dtype"] = dtype
    manifest["signature"] = cache.manifest_signature(manifest)
    receipt = commit(planned)
    finish_manifest(planned)
    loaded = cache.load_cache_manifest(output, source)
    episode = cache.load_episode_cache(output, 0, manifest=loaded, verify_checksums=True)
    assert isinstance(episode.features, np.memmap)
    assert not episode.features.flags.writeable
    assert episode.features.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(episode.features[:, 0, 0], [1, 2, 3, 4])
    np.testing.assert_array_equal(episode.frame_index, np.arange(4))
    np.testing.assert_array_equal(episode.source_frame_index, np.arange(4))
    np.testing.assert_array_equal(episode.timestamp, episode.video_timestamp)
    assert episode.valid.all()
    assert receipt["frames"] == 4 and encoder.calls == 2


def test_partial_pack_and_interrupted_episode_are_not_readable_but_resume(planned):
    _, output, _, manifest, encoder = planned
    with pytest.raises(ValueError, match="incomplete"):
        cache.load_cache_manifest(output)
    with pytest.raises(ValueError, match="decoded 3 frames"):
        commit(planned, iterator=frames(3))
    directory = cache.episode_directory(output, 0)
    assert not (directory / "receipt.json").exists()
    assert not (directory / "features.npy.partial").exists()
    with pytest.raises(ValueError, match="incomplete"):
        cache.load_episode_cache(output, 0, manifest=manifest)
    commit(planned)
    with pytest.raises(ValueError, match="incomplete"):
        cache.load_episode_cache(output, 0, manifest=manifest)
    calls = encoder.calls
    commit(planned, iterator=iter(()))
    assert encoder.calls == calls  # committed episode needs no decoder/model work


@pytest.mark.parametrize("count,shift", [(5, 0), (4, 0.2)])
def test_unverified_video_row_clock_mapping_never_commits(planned, count, shift):
    with pytest.raises(ValueError, match="one-frame-per-row|PTS do not align"):
        commit(planned, iterator=frames(count, shift=shift))
    assert not (cache.episode_directory(planned[1], 0) / "receipt.json").exists()


def test_stale_source_profile_and_modified_outputs_are_rejected(planned):
    source, output, items, _, _ = planned
    commit(planned)
    finish_manifest(planned)
    with pytest.raises(ValueError, match="profile"):
        cache.load_cache_manifest(output, source, expected_profile="256x480")
    directory = cache.episode_directory(output, 0)
    feature = directory / "features.npy"
    with feature.open("r+b") as stream:
        stream.seek(-2, 2)
        stream.write(b"xx")
    with pytest.raises(ValueError, match="Stale or modified"):
        cache.load_episode_cache(output, 0)
    meta = source / "meta/info.json"
    meta.write_text(meta.read_text() + "\n")
    with pytest.raises(ValueError, match="Stale or modified"):
        cache.load_cache_manifest(output, source)


def test_shards_share_one_manifest_and_partition_episode_writers(planned):
    source, _, _, _, encoder = planned
    a, first = extract.plan_source(source, num_shards=2, shard_index=0)
    b, second = extract.plan_source(source, num_shards=2, shard_index=1)
    assert [item["episode_index"] for item in first] == [0]
    assert [item["episode_index"] for item in second] == [1]
    one = extract.build_manifest(a, first, encoder.metadata)
    two = extract.build_manifest(b, second, encoder.metadata)
    assert one["signature"] == two["signature"]
    assert one["episode_indices"] == [0, 1]


def test_only_final_shard_publishes_complete_pack_and_resume_skips_encoder(planned, monkeypatch):
    source, output, _, _, encoder = planned
    monkeypatch.setattr(extract, "read_episode_rows", rows)
    monkeypatch.setattr(extract, "native_video_frames", lambda *args, **kwargs: frames())
    first = extract.extract_dataset(source, output, encoder, num_shards=2, shard_index=0)
    assert first["state"] == "extracting"
    with pytest.raises(ValueError, match="incomplete"):
        cache.load_cache_manifest(output, source)
    second = extract.extract_dataset(source, output, encoder, num_shards=2, shard_index=1)
    assert second["state"] == "complete"
    completed = cache.load_cache_manifest(output, source)
    for index in (0, 1):
        assert cache.load_episode_cache(output, index, manifest=completed).valid.all()
    calls = encoder.calls
    extract.extract_dataset(source, output, encoder, num_shards=2, shard_index=0)
    assert encoder.calls == calls


def test_overlapping_episode_writer_is_locked(planned):
    directory = cache.episode_directory(planned[1], 0)
    with extract.output_lock(directory):
        with pytest.raises(BlockingIOError):
            commit(planned)


def test_encoder_identity_and_independent_geometry_are_rejected(planned):
    _, output, _, manifest, _ = planned
    manifest["encoder"]["preprocessing"]["do_resize"] = True
    manifest["signature"] = cache.manifest_signature(manifest)
    finish_manifest(planned)
    with pytest.raises(ValueError, match="second resize"):
        cache.load_cache_manifest(output)
