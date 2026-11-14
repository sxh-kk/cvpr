"""CPU-only provenance and native-frame checks for the optional packed reader."""
from copy import deepcopy
import builtins
import io
import json
import multiprocessing
import os
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

from MotionExpert.dataset import packed_vision_cache as packed_module
from MotionExpert.dataset.packed_vision_cache import (
    ENV_VAR, PackedVisionCache, build_packed_vision_cache,
)
from MotionExpert.dataset.smplh_archive import SMPLHParameterArchive
from MotionExpert.tests.fixtures import owned_archive, _json
from MotionExpert.tests.test_vision_profile_cache import write_cache, update_cache_episode
from MotionExpert.tests.test_window_cache_dataset import cached_archive, _dataset


@pytest.fixture(autouse=True)
def tiny_fixture_does_not_require_production_memory_reserve(monkeypatch):
    # These fixtures pack less than 2 MiB. The production builder's 128 GiB
    # reserve is deliberately independent of the CI host's available RAM.
    monkeypatch.setattr(packed_module, "_check_memory_headroom", lambda: None)


@pytest.fixture
def packed_source(owned_archive, tmp_path):
    torch.set_num_threads(1)
    vision_root = write_cache(owned_archive, tmp_path / "native-vision")
    # Make every episode, native frame, token and feature channel distinguishable.
    # All values stay exactly representable in float32.
    for episode in range(4):
        update_cache_episode(
            vision_root, episode,
            mutate_features=lambda value, episode=episode: (
                np.arange(value.size, dtype=np.float32).reshape(value.shape)
                + np.float32(episode * 1_000_000)),
        )
    archive = SMPLHParameterArchive(owned_archive, vision_profile_root=vision_root)
    records = [r for r in archive.records()
               if r["metadata"].get("training_eligible") is not False
               and r["metadata"].get("canonical_training_eligible") is not False
               and not r["metadata"].get("quarantine_reasons", [])]
    assert [r["episode_index"] for r in records] == [0, 1, 3]
    assert {r["split"] for r in records} == {"train", "val"}  # Empty test split.
    records_path = tmp_path / "accepted-window-records.json"
    _json(records_path, records)
    return archive.vision_cache, records, records_path


@pytest.fixture
def packed(packed_source, tmp_path):
    vision, records, records_path = packed_source
    root = tmp_path / "packed"
    manifest = build_packed_vision_cache(
        vision.root, records_path, root, workers=0, block_rows=2,
    )
    return root, manifest, vision, records


def _reader(packed):
    root, _, vision, records = packed
    return PackedVisionCache(root, vision_cache=vision, records=records)


def _manifest_change(packed, mutation):
    root = packed[0]
    manifest = json.loads((root / "manifest.json").read_text())
    mutation(manifest)
    _json(root / "manifest.json", manifest)


def _packed_identity(root, manifest):
    stat = (root / manifest["packed"]["path"]).stat()
    manifest["packed"].update(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                              device=stat.st_dev, inode=stat.st_ino)
    _json(root / "manifest.json", manifest)


def test_builder_reads_native_zero_phase_for_noncontiguous_eligible_population(packed):
    root, manifest, vision, records = packed
    assert manifest["selected_episode_indices"] == [0, 1, 3]
    assert manifest["source_vision_episode_count"] == 4
    assert manifest["total_rows"] == sum((r["length"] + 5) // 6 for r in records)
    assert manifest["verification"]["source_equals_packed"] is True
    mapped = np.load(root / manifest["packed"]["path"], mmap_mode="r", allow_pickle=False)
    offset = 0
    for descriptor, record in zip(manifest["episodes"], records):
        native = np.load(vision._episodes[record["episode_index"]]["artifacts"]["features.npy"][0],
                         mmap_mode="r", allow_pickle=False)
        rows = (record["length"] + 5) // 6
        assert descriptor["row_offset"] == offset
        assert descriptor["rows"] == rows
        np.testing.assert_array_equal(mapped[offset:offset + rows], native[::6])
        offset += rows
        native._mmap.close()
    mapped._mmap.close()


@pytest.mark.parametrize("start", [0, 30, 60])
def test_reader_matches_native_historical_windows_and_returns_independent_copies(packed, start):
    reader = _reader(packed)
    selected = start + np.arange(6, dtype=np.int64) * 6
    for record in packed[3]:
        source_path = packed[2]._episodes[record["episode_index"]]["artifacts"]["features.npy"][0]
        source = np.load(source_path, mmap_mode="r", allow_pickle=False)
        expected = source[selected].copy()
        actual = reader.read(record, selected)
        assert actual.shape == (6, 5, 1024) and actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected)
        actual[:] = -999
        np.testing.assert_array_equal(reader.read(record, selected), expected)
        source._mmap.close()
    assert isinstance(reader._mapped, np.memmap) and not reader._mapped.flags.writeable


@pytest.mark.parametrize("selected", [
    np.arange(1, 37, 6, dtype=np.int64),
    np.arange(-6, 30, 6, dtype=np.int64),
    np.arange(66, 102, 6, dtype=np.int64),
    np.arange(0, 36, 6, dtype=np.float64),
    np.arange(0, 36, 6, dtype=np.int64)[None],
])
def test_reader_rejects_phase_dtype_rank_and_episode_boundary_errors(packed, selected):
    with pytest.raises(ValueError):
        _reader(packed).read(packed[3][0], selected)


def test_reader_rejects_unaccepted_episode_without_cross_episode_reads(packed):
    record = deepcopy(packed[3][0])
    record["episode_index"] = 2  # Present in native vision, excluded by accepted records.
    with pytest.raises(ValueError):
        _reader(packed).read(record, np.arange(0, 36, 6))


@pytest.mark.parametrize("field,value", [
    ("contract", "unrelated-cache"), ("state", "building"),
    ("fps", 30), ("native_fps", 25), ("sampling_step", 1), ("frame_phase", 1),
    ("shape", [4, 1024]), ("storage_dtype", "float16"),
    ("source_vision_manifest_sha256", "0" * 64),
    ("source_vision_episode_count", 3), ("selected_episode_indices", [0, 1]),
    ("total_rows", 1),
])
def test_reader_rejects_incompatible_or_partial_manifest(packed, field, value):
    _manifest_change(packed, lambda m: m.__setitem__(field, value))
    with pytest.raises(ValueError):
        _reader(packed)


def test_reader_binds_the_absolute_original_vision_root(packed, tmp_path):
    other_root = tmp_path / "different-vision"
    other_root.mkdir()
    _manifest_change(packed, lambda m: m.__setitem__("source_vision_root", str(other_root)))
    with pytest.raises(ValueError):
        _reader(packed)


@pytest.mark.parametrize("change", ["length", "rows", "offset", "record_hash", "duplicate"])
def test_reader_rejects_episode_length_offset_population_and_source_hash_errors(packed, change):
    def mutation(manifest):
        episode = manifest["episodes"][0]
        if change == "length":
            episode["length"] += 6
        elif change == "rows":
            episode["rows"] -= 1
        elif change == "offset":
            manifest["episodes"][1]["row_offset"] = 0
        elif change == "record_hash":
            episode["source_record_sha256"] = "0" * 64
        else:
            manifest["episodes"][1]["episode_index"] = episode["episode_index"]
    _manifest_change(packed, mutation)
    with pytest.raises(ValueError):
        _reader(packed)


def test_reader_rejects_an_extra_unverified_source_record(packed):
    records = deepcopy(packed[3])
    records[0]["source_key"] = "different-source"
    with pytest.raises(ValueError):
        PackedVisionCache(packed[0], vision_cache=packed[2], records=records)


@pytest.mark.parametrize("change", ["verification", "canonical_hash", "missing_episode"])
def test_reader_rejects_missing_full_population_verification_or_record_binding(packed, change):
    def mutation(manifest):
        if change == "verification":
            manifest["verification"]["source_equals_packed"] = False
        elif change == "canonical_hash":
            manifest["source_records"]["canonical_sha256"] = "0" * 64
        else:
            manifest["episodes"].pop()
    _manifest_change(packed, mutation)
    with pytest.raises(ValueError):
        _reader(packed)


@pytest.mark.parametrize("contents", [b"not an NPY", b"\x93NUMPY\x01\x00\x20\x00broken"])
def test_bad_packed_headers_are_rejected_even_when_file_identity_matches(packed, contents):
    root, manifest = packed[:2]
    path = root / manifest["packed"]["path"]
    path.chmod(0o644)
    path.write_bytes(contents)
    _packed_identity(root, manifest)
    with pytest.raises(ValueError):
        _reader(packed).read(packed[3][0], np.arange(0, 36, 6))


@pytest.mark.parametrize("change", ["shape", "dtype", "truncated"])
def test_packed_array_header_shape_dtype_and_payload_length_are_verified(packed, change):
    root, manifest = packed[:2]
    path = root / manifest["packed"]["path"]
    path.chmod(0o644)
    if change == "truncated":
        with path.open("r+b") as stream:
            stream.truncate(path.stat().st_size - 4)
    else:
        shape = (manifest["total_rows"], 4 if change == "shape" else 5, 1024)
        np.save(path, np.zeros(shape, dtype=np.float64 if change == "dtype" else np.float32))
    _packed_identity(root, manifest)
    with pytest.raises(ValueError):
        _reader(packed).read(packed[3][0], np.arange(0, 36, 6))


def test_epoch_identity_check_rejects_replaced_packed_file_even_with_same_size_and_mtime(packed):
    reader = _reader(packed)
    root, manifest = packed[:2]
    path = root / manifest["packed"]["path"]
    original = path.stat()
    replacement = path.with_name("replacement.npy")
    replacement.write_bytes(path.read_bytes())
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    os.replace(replacement, path)
    assert path.stat().st_size == original.st_size
    assert path.stat().st_mtime_ns == original.st_mtime_ns
    with pytest.raises(ValueError):
        reader.assert_current()


def test_epoch_identity_check_rejects_manifest_republication(packed):
    reader = _reader(packed)
    with (packed[0] / "manifest.json").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError):
        reader.assert_current()


@pytest.mark.parametrize("field", ["source_feature_identity", "source_receipt_identity"])
def test_reader_rejects_source_identities_that_differ_from_the_verified_native_cache(packed, field):
    _manifest_change(packed, lambda m: m["episodes"][0][field].__setitem__("mtime_ns", 0))
    with pytest.raises(ValueError):
        _reader(packed)


def _forbid_source_io(monkeypatch, source_root):
    """Catch both FileIO/mmap headers and stat-only fallbacks to the GPFS source."""
    def guard(original):
        def checked(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                location = Path(os.fsdecode(path))
                if location.is_relative_to(source_root):
                    raise AssertionError(f"Packed worker touched original vision files: {location}")
            return original(path, *args, **kwargs)
        return checked
    monkeypatch.setattr(builtins, "open", guard(builtins.open))
    monkeypatch.setattr(io, "open", guard(io.open))
    monkeypatch.setattr(os, "stat", guard(os.stat))


def test_worker_reads_only_the_local_mapping_after_source_startup_validation(packed, monkeypatch):
    reader = _reader(packed)
    assert reader._mapped is None
    # The source was already validated during startup. A worker must not pay
    # per-episode GPFS open/stat latency when opening or reusing its local map.
    _forbid_source_io(monkeypatch, packed[2].root)
    for record in packed[3] + packed[3][:1]:
        for start in (0, 30, 60):
            selected = start + np.arange(6, dtype=np.int64) * 6
            actual = reader.read(record, selected)
            expected_first = selected.astype(np.float32) * np.float32(5120)
            expected_first += np.float32(record["episode_index"] * 1_000_000)
            np.testing.assert_array_equal(actual[:, 0, 0], expected_first)


def test_pickling_drops_mappings_and_reopens_a_read_only_worker_owner(packed):
    reader = _reader(packed)
    expected = reader.read(packed[3][0], np.arange(0, 36, 6))
    copied = pickle.loads(pickle.dumps(reader))
    assert copied._mapped is None and not copied._verified
    np.testing.assert_array_equal(copied.read(packed[3][0], np.arange(0, 36, 6)), expected)
    assert copied._mapped is not reader._mapped
    assert not copied._mapped.flags.writeable and not reader._mapped._mmap.closed


def _fork_read(reader, record, pipe):
    try:
        inherited = reader._mapped
        output = reader.read(record, np.arange(0, 36, 6))
        pipe.send((os.getpid(), reader._pid, reader._mapped is not inherited,
                   reader._mapped.flags.writeable, output[0, 0, 0]))
    finally:
        pipe.close()


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="fork unavailable")
def test_forked_reader_discards_inherited_mapping_without_closing_parent(packed):
    reader = _reader(packed)
    record = packed[3][0]
    expected = reader.read(record, np.arange(0, 36, 6))
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_fork_read, args=(reader, record, child))
    process.start()
    child.close()
    try:
        assert parent.poll(15), "Forked reader did not finish its tiny local read"
        child_pid, owner_pid, reset, writable, value = parent.recv()
        process.join(timeout=15)
        assert process.exitcode == 0
        assert child_pid == owner_pid != os.getpid() and reset and not writable
        assert value == expected[0, 0, 0]
        assert reader._pid == os.getpid() and not reader._mapped._mmap.closed
        np.testing.assert_array_equal(reader.read(record, np.arange(0, 36, 6)), expected)
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def test_builder_refuses_to_overwrite_a_committed_output(packed, packed_source):
    with pytest.raises(FileExistsError):
        build_packed_vision_cache(packed[2].root, packed_source[2], packed[0], workers=0)


def test_reader_raw_record_hash_fast_path_binds_the_accepted_records_file(packed):
    root, manifest, vision, records = packed
    reader = PackedVisionCache(root, vision_cache=vision, records=records,
                               records_sha256=manifest["source_records"]["sha256"])
    reader.assert_current()
    with pytest.raises(ValueError):
        PackedVisionCache(root, vision_cache=vision, records=records, records_sha256="0" * 64)


def test_parallel_builder_keeps_disjoint_episode_ranges_bitwise_equal(packed_source, tmp_path):
    vision, records, records_path = packed_source
    root = tmp_path / "parallel-packed"
    build_packed_vision_cache(vision.root, records_path, root, workers=2, block_rows=1)
    reader = PackedVisionCache(root, vision_cache=vision, records=records)
    for record in records:
        selected = np.arange(0, record["length"], 6)
        source = np.load(vision._episodes[record["episode_index"]]["artifacts"]["features.npy"][0])
        np.testing.assert_array_equal(reader.read(record, selected), source[selected])


def test_failed_builder_never_publishes_a_usable_manifest(packed_source, tmp_path, monkeypatch):
    vision, records, records_path = packed_source
    root = tmp_path / "interrupted-packed"
    source_blocks = packed_module._iter_native5hz_blocks

    def interrupted(path, expected_frames, block_rows, *, cancelled=None):
        if Path(path).parent.name == "00000001":
            raise OSError("simulated source I/O failure")
        yield from source_blocks(path, expected_frames, block_rows, cancelled=cancelled)

    monkeypatch.setattr(packed_module, "_iter_native5hz_blocks", interrupted)
    with pytest.raises(OSError, match="simulated source I/O failure"):
        build_packed_vision_cache(vision.root, records_path, root, workers=2, block_rows=2)
    assert not (root / "manifest.json").exists()
    with pytest.raises((ValueError, FileNotFoundError)):
        PackedVisionCache(root, vision_cache=vision, records=records)


def test_memory_pressure_during_copy_does_not_publish_a_complete_cache(packed_source, tmp_path, monkeypatch):
    vision, _, records_path = packed_source
    root = tmp_path / "memory-pressure-packed"

    def pressure_after_allocation():
        if (root / "packed.npy").is_file():
            raise MemoryError("simulated memory pressure during copy")

    monkeypatch.setattr(packed_module, "_check_memory_headroom", pressure_after_allocation)
    with pytest.raises(MemoryError, match="simulated memory pressure during copy"):
        build_packed_vision_cache(vision.root, records_path, root, workers=2, block_rows=2)
    assert not (root / "manifest.json").exists()


@pytest.mark.parametrize("field,value", [("state", "extracting"), ("scope", "partial")])
def test_builder_rejects_a_partial_native_publication_before_creating_output(packed_source, tmp_path, field, value):
    vision, _, records_path = packed_source
    manifest = json.loads(vision.manifest_path.read_text())
    manifest[field] = value
    _json(vision.manifest_path, manifest)
    root = tmp_path / "rejected-packed"
    with pytest.raises(ValueError):
        build_packed_vision_cache(vision.root, records_path, root, workers=0)
    assert not root.exists()


def test_window_cache_env_preserves_all_samples_and_keeps_old_path_opt_in_only(cached_archive, tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    baseline = _dataset(cached_archive, cached=True)
    root = Path(cached_archive.DATA.WINDOW_CACHE_ROOT)
    records_path = root / baseline.window_cache.manifest["records_path"]
    packed_root = tmp_path / "packed-window-images"
    build_packed_vision_cache(cached_archive.DATA.VISION_PROFILE_ROOT, records_path,
                              packed_root, workers=0, block_rows=2)
    assert baseline.window_cache._packed_vision is None
    monkeypatch.setenv(ENV_VAR, str(packed_root))
    fast = _dataset(cached_archive, cached=True)
    assert fast.window_cache._packed_vision is not None
    assert fast.idx_to_sidx_fidx == baseline.idx_to_sidx_fidx
    originals = [baseline[index] for index in range(len(baseline))]
    with monkeypatch.context() as worker_patch:
        _forbid_source_io(worker_patch, Path(cached_archive.DATA.VISION_PROFILE_ROOT))
        for index, old in enumerate(originals):
            new = fast[index]
            for group in ("y", "misc"):
                for key, value in old[group].items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(new[group][key], value, rtol=0, atol=0)
                    else:
                        assert new[group][key] == value
    from MotionExpert.dataset.window_cache import WindowCache, protocol_from_config
    empty = WindowCache(root, archive=fast.archive, split="test",
                        protocol=protocol_from_config(cached_archive))
    assert empty.idx_to_sidx_fidx == [] and empty._packed_vision is not None


def test_empty_packed_population_can_be_published_without_mapping_invalid_rows(packed_source, tmp_path):
    vision, _, _ = packed_source
    records_path = tmp_path / "empty-records.json"
    _json(records_path, [])
    root = tmp_path / "empty-packed"
    manifest = build_packed_vision_cache(vision.root, records_path, root, workers=0)
    assert manifest["total_rows"] == 0 and manifest["episodes"] == []
    reader = PackedVisionCache(root, vision_cache=vision, records=[])
    reader.assert_current()
    assert reader._mapped is None
