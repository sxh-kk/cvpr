"""Bounded sequential source reads; no production data or GPU access."""
import io
import threading

import numpy as np
import pytest

from MotionExpert.dataset import packed_vision_cache as packed


@pytest.fixture(autouse=True)
def ignore_production_memory_reserve_for_tiny_fixtures(monkeypatch):
    monkeypatch.setattr(packed, "_check_memory_headroom", lambda: None)


def _source(path, frames, *, order="C", version=(1, 0)):
    values = np.arange(frames * 5 * 1024, dtype=np.float32).reshape(frames, 5, 1024).copy(order=order)
    if frames:
        # Include signed zero, infinities and a NaN payload: comparisons below
        # are bytewise, so floating-point equality cannot hide changed bits.
        values[0, 0, :6] = np.array([0, 0x80000000, 0x3F800000, 0x7FC01234,
                                     0xFF800000, 0x7F800000], dtype=np.uint32).view(np.float32)
    with path.open("wb") as stream:
        np.lib.format.write_array(stream, values, version=version, allow_pickle=False)
    return values


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("frames", [0, 1, 5, 6, 7, 12, 13, 31])
@pytest.mark.parametrize("block_rows", [1, 4])
def test_blocks_match_native_zero_phase_bitwise_and_own_their_storage(tmp_path, version, order, frames, block_rows):
    path = tmp_path / "native.npy"
    source = _source(path, frames, order=order, version=version)
    blocks = list(packed._iter_native5hz_blocks(path, frames, block_rows))
    assert [start for start, _ in blocks] == list(range(0, (frames + 5) // 6, block_rows))
    for start, values in blocks:
        assert values.dtype == np.float32 and values.flags.c_contiguous
        assert values.flags.owndata and len(values) <= block_rows
        expected = np.ascontiguousarray(source[start * 6:(start + block_rows) * 6:6])
        np.testing.assert_array_equal(values.view(np.uint8), expected.view(np.uint8))
    for (_, left), (_, right) in zip(blocks, blocks[1:]):
        assert not np.shares_memory(left, right)


def test_c_order_reads_once_unbuffered_without_any_mmap_and_bounds_native_buffer(tmp_path, monkeypatch):
    path = tmp_path / "native.npy"
    source = _source(path, 49)
    opened, calls, reads, allocations = [], [], [], []
    original_empty = np.empty

    class RecordingRaw(io.FileIO):
        def readinto(self, buffer):
            reads.append((self.tell(), len(buffer)))
            return super().readinto(buffer)

    def recording_open(filename, mode, buffering=-1):
        calls.append((filename, mode, buffering))
        stream = RecordingRaw(filename, mode)
        opened.append(stream)
        return stream

    def recording_empty(shape, *args, **kwargs):
        allocations.append(shape)
        return original_empty(shape, *args, **kwargs)

    def forbidden_mapping(*args, **kwargs):
        raise AssertionError("C-order source entered a load/mmap path")

    with monkeypatch.context() as patch:
        patch.setattr(packed, "open", recording_open, raising=False)
        patch.setattr(np, "empty", recording_empty)
        patch.setattr(np, "load", forbidden_mapping)
        patch.setattr(np, "memmap", forbidden_mapping)
        patch.setattr(packed, "load_npy_mmap", forbidden_mapping)
        blocks = list(packed._iter_native5hz_blocks(path, 49, 2))
    assert calls == [(str(path), "rb", 0)]
    assert len(opened) == 1 and opened[0].closed
    assert allocations == [(12, 5, 1024)]
    assert max(size for _, size in reads) == 12 * 5 * 1024 * 4
    assert sum(size for _, size in reads) == source.nbytes
    assert all(next_offset == offset + size for (offset, size), (next_offset, _) in zip(reads, reads[1:]))
    actual = np.concatenate([values for _, values in blocks])
    np.testing.assert_array_equal(actual.view(np.uint8), source[::6].copy().view(np.uint8))


@pytest.mark.parametrize("premature_eof", [False, True])
def test_short_reads_interruptions_and_midstream_eof_close_source(tmp_path, monkeypatch, premature_eof):
    path = tmp_path / "native.npy"
    source = _source(path, 19)
    streams = []

    class ShortReads(io.FileIO):
        attempts = 0

        def readinto(self, buffer):
            self.attempts += 1
            if self.attempts == 1:
                raise InterruptedError("interrupted read")
            if premature_eof and self.attempts > 3:
                return 0
            return super().readinto(buffer[:8191])

    def short_open(filename, mode, buffering=0):
        assert buffering == 0
        stream = ShortReads(filename, mode)
        streams.append(stream)
        return stream

    monkeypatch.setattr(packed, "open", short_open, raising=False)
    if premature_eof:
        with pytest.raises(ValueError, match="truncated while reading"):
            list(packed._iter_native5hz_blocks(path, 19, 2))
    else:
        blocks = list(packed._iter_native5hz_blocks(path, 19, 2))
        actual = np.concatenate([values for _, values in blocks])
        np.testing.assert_array_equal(actual.view(np.uint8), source[::6].copy().view(np.uint8))
        assert streams[0].attempts > 4
    assert len(streams) == 1 and streams[0].closed


@pytest.mark.parametrize("change", ["shape", "float64", "big_endian", "object", "header", "payload", "version"])
def test_invalid_source_header_or_payload_is_rejected_before_allocation(tmp_path, monkeypatch, change):
    path = tmp_path / "invalid.npy"
    values = np.zeros((2, 5, 1023) if change == "shape" else (2, 5, 1024),
                      dtype={"float64": "f8", "big_endian": ">f4", "object": object}.get(change, "f4"))
    np.save(path, values)
    if change == "header":
        path.write_bytes(path.read_bytes()[:12])
    elif change == "payload":
        path.write_bytes(path.read_bytes()[:-1])
    elif change == "version":
        path.write_bytes(b"\x93NUMPY\x04\x00")

    def no_allocation(*args, **kwargs):
        raise AssertionError("Invalid source reached allocation/mapping")

    monkeypatch.setattr(np, "empty", no_allocation)
    monkeypatch.setattr(np, "memmap", no_allocation)
    with pytest.raises(ValueError):
        list(packed._iter_native5hz_blocks(path, 2, 4096))


def test_fortran_fallback_maps_once_and_closes_mapping_and_source(tmp_path, monkeypatch):
    path = tmp_path / "fortran.npy"
    source = _source(path, 19, order="F")
    original_memmap, mappings, streams = np.memmap, [], []

    def tracked_memmap(stream, **kwargs):
        assert kwargs["order"] == "F" and kwargs["mode"] == "r"
        streams.append(stream)
        result = original_memmap(stream, **kwargs)
        mappings.append(result)
        return result

    monkeypatch.setattr(np, "memmap", tracked_memmap)
    blocks = list(packed._iter_native5hz_blocks(path, 19, 2))
    assert len(mappings) == 1 and mappings[0]._mmap.closed
    assert len(streams) == 1 and streams[0].closed
    actual = np.concatenate([values for _, values in blocks])
    np.testing.assert_array_equal(actual.view(np.uint8), source[::6].copy().view(np.uint8))


def test_cancellation_between_blocks_closes_file_without_another_read(tmp_path, monkeypatch):
    path = tmp_path / "native.npy"
    _source(path, 19)
    streams, attempts = [], []

    class TrackingRaw(io.FileIO):
        def readinto(self, buffer):
            attempts.append(len(buffer))
            return super().readinto(buffer)

    def tracking_open(filename, mode, buffering=0):
        stream = TrackingRaw(filename, mode)
        streams.append(stream)
        return stream

    monkeypatch.setattr(packed, "open", tracking_open, raising=False)
    cancelled = threading.Event()
    blocks = packed._iter_native5hz_blocks(path, 19, 1, cancelled=cancelled)
    next(blocks)
    assert len(attempts) == 1
    cancelled.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        next(blocks)
    assert len(attempts) == 1 and streams[0].closed


def test_explicit_generator_close_releases_source_handle(tmp_path, monkeypatch):
    path = tmp_path / "native.npy"
    _source(path, 19)
    original_open, streams = open, []

    def tracked_open(*args, **kwargs):
        stream = original_open(*args, **kwargs)
        streams.append(stream)
        return stream

    monkeypatch.setattr(packed, "open", tracked_open, raising=False)
    blocks = packed._iter_native5hz_blocks(path, 19, 1)
    next(blocks)
    assert not streams[0].closed
    blocks.close()
    assert streams[0].closed
