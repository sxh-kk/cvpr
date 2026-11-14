"""Small CPU-only checks for unbuffered NPY header reads and mmap ownership."""
import io
import struct

import numpy as np
import pytest

from MotionExpert.dataset import npy_io


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", ["<f4", ">f8", "complex128", "bool",
                                   np.dtype([("index", ">i4"), ("value", "<f4", (2,))])])
def test_mapping_matches_numpy_for_versions_orders_and_dtypes(tmp_path, version, order, dtype):
    expected = np.zeros((3, 4), dtype=dtype, order=order)
    if expected.dtype.fields:
        expected["index"] = np.arange(12).reshape(3, 4)
        expected["value"] = np.arange(24).reshape(3, 4, 2)
    else:
        expected[:] = np.arange(12).reshape(3, 4)
    path = tmp_path / "array.npy"
    with path.open("wb") as stream:
        np.lib.format.write_array(stream, expected, version=version, allow_pickle=False)
    baseline = np.load(path, mmap_mode="r", allow_pickle=False)
    mapped = npy_io.load_npy_mmap(path)
    try:
        assert isinstance(mapped, np.memmap)
        assert mapped.dtype == baseline.dtype == expected.dtype
        assert mapped.shape == baseline.shape == expected.shape
        assert mapped.strides == baseline.strides == expected.strides
        assert mapped.offset == baseline.offset
        assert mapped.filename == baseline.filename
        assert mapped.mode == "r" and not mapped.flags.writeable
        np.testing.assert_array_equal(mapped, expected)
        with pytest.raises(ValueError, match="read-only"):
            mapped[0] = expected[0]
    finally:
        mapped._mmap.close()
        baseline._mmap.close()


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize("shape,dtype", [((0,), "f4"), ((2, 0, 4), "i8"),
                                          ((), "f8"), ((2, 3), "V0")])
def test_empty_scalar_and_zero_itemsize_arrays(tmp_path, version, shape, dtype):
    expected = np.zeros(shape, dtype=dtype)
    path = tmp_path / "empty.npy"
    with path.open("wb") as stream:
        np.lib.format.write_array(stream, expected, version=version, allow_pickle=False)
    mapped = npy_io.load_npy_mmap(path)
    try:
        assert isinstance(mapped, np.memmap)
        assert mapped.shape == shape and mapped.dtype == expected.dtype
        np.testing.assert_array_equal(mapped, expected)
    finally:
        mapped._mmap.close()


def test_v3_utf8_structured_field_names_are_preserved(tmp_path):
    expected = np.zeros(2, dtype=[("动作", "f4"), ("位置🌍", "i8", (3,))])
    expected["动作"] = [1.5, 2.5]
    expected["位置🌍"] = np.arange(6).reshape(2, 3)
    path = tmp_path / "utf8.npy"
    with path.open("wb") as stream:
        np.lib.format.write_array(stream, expected, version=(3, 0), allow_pickle=False)
    mapped = npy_io.load_npy_mmap(path)
    try:
        assert mapped.dtype == expected.dtype
        np.testing.assert_array_equal(mapped, expected)
    finally:
        mapped._mmap.close()


def test_header_open_is_unbuffered_once_and_mapping_outlives_handle(tmp_path, monkeypatch):
    path = tmp_path / "large.npy"
    np.save(path, np.arange(2 * 1024 * 1024, dtype=np.float32))
    streams, calls, reads = [], [], []

    class RecordingRaw(io.FileIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

        def readinto(self, buffer):
            reads.append(len(buffer))
            return super().readinto(buffer)

    def gpfs_open(filename, mode="r", buffering=-1):
        calls.append((filename, mode, buffering))
        raw = RecordingRaw(filename, mode)
        streams.append(raw)
        # Emulate GPFS's advertised 4 MiB block size if buffering is omitted.
        return raw if buffering == 0 else io.BufferedReader(raw, buffer_size=4 * 1024 * 1024)

    def unexpected_numpy_load(*args, **kwargs):
        raise AssertionError("NPY mapping re-entered NumPy's buffered path")

    monkeypatch.setattr(npy_io, "open", gpfs_open, raising=False)
    monkeypatch.setattr(np, "load", unexpected_numpy_load)
    monkeypatch.setattr(np.lib.format, "open_memmap", unexpected_numpy_load)
    mapped = npy_io.load_npy_mmap(path)
    try:
        assert calls == [(str(path), "rb", 0)]
        assert len(streams) == 1 and streams[0].closed
        assert sum(reads) == mapped.offset < 1024
        assert all(0 < size < 1024 for size in reads)
        assert not mapped._mmap.closed
        np.testing.assert_array_equal(mapped[[0, 17, -1]], [0, 17, len(mapped) - 1])
    finally:
        mapped._mmap.close()
    assert mapped._mmap.closed


@pytest.mark.parametrize("dtype", [object, np.dtype([("value", object)])])
def test_object_arrays_are_rejected_without_unpickling(tmp_path, dtype, monkeypatch):
    path = tmp_path / "object.npy"
    np.save(path, np.zeros(2, dtype=dtype))

    def unexpected_memmap(*args, **kwargs):
        raise AssertionError("Object array reached mmap creation")

    monkeypatch.setattr(np, "memmap", unexpected_memmap)
    with pytest.raises(ValueError, match="Python objects"):
        npy_io.load_npy_mmap(path)


def _write_header(path, header, *, version=(1, 0), payload=b""):
    encoding = "utf8" if version == (3, 0) else "latin1"
    header = header.encode(encoding)
    length_format = "<H" if version == (1, 0) else "<I"
    path.write_bytes(np.lib.format.magic(*version) + struct.pack(length_format, len(header))
                     + header + payload)


@pytest.mark.parametrize("header", [
    "not a dictionary", "[]", "{'descr': '<f4', 'shape': (1,)}",
    "{'descr': '<f4', 'shape': [1], 'fortran_order': False}",
    "{'descr': '<f4', 'shape': (-1,), 'fortran_order': False}",
    "{'descr': '<f4', 'shape': (1,), 'fortran_order': 0}",
    "{'descr': 'invalid_dtype', 'shape': (1,), 'fortran_order': False}",
    "{'descr': '<f4', 'shape': (18446744073709551616,), 'fortran_order': False}",
    "{'descr': '<f4', 'shape': (0,), 'fortran_order': False}" + " " * 10000,
])
def test_invalid_headers_fail_and_close_file(tmp_path, monkeypatch, header):
    path = tmp_path / "bad.npy"
    _write_header(path, header)
    original_open, streams = open, []

    def tracked_open(*args, **kwargs):
        stream = original_open(*args, **kwargs)
        streams.append(stream)
        return stream

    monkeypatch.setattr(npy_io, "open", tracked_open, raising=False)
    with pytest.raises(ValueError):
        npy_io.load_npy_mmap(path)
    assert len(streams) == 1 and streams[0].closed


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize("cut", [1, 7, 9, 15, 127, 130])
def test_truncated_magic_header_and_payload_are_rejected(tmp_path, version, cut):
    path = tmp_path / "truncated.npy"
    with path.open("wb") as stream:
        np.lib.format.write_array(stream, np.arange(8, dtype=np.float64), version=version)
    path.write_bytes(path.read_bytes()[:cut])
    with pytest.raises(ValueError):
        npy_io.load_npy_mmap(path)


@pytest.mark.parametrize("contents", [b"not a numpy file", b"\x93NUMPY\x04\x00",
                                      b"PK\x03\x04pickle or zip data"])
def test_non_npy_and_unknown_versions_are_rejected(tmp_path, contents):
    path = tmp_path / "not.npy"
    path.write_bytes(contents)
    with pytest.raises(ValueError):
        npy_io.load_npy_mmap(path)


def test_memmap_creation_failure_also_closes_header_handle(tmp_path, monkeypatch):
    path = tmp_path / "array.npy"
    np.save(path, np.arange(3))
    streams = []

    def fail_mapping(stream, **kwargs):
        streams.append(stream)
        raise OSError("mapping failed")

    monkeypatch.setattr(np, "memmap", fail_mapping)
    with pytest.raises(OSError, match="mapping failed"):
        npy_io.load_npy_mmap(path)
    assert len(streams) == 1 and streams[0].closed
