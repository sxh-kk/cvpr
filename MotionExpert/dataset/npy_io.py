"""Read-only NPY mappings without filesystem-block-sized buffered header reads."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np


def load_npy_mmap(path):
    """Map an unpickled NPY file, retaining NumPy's header and dtype validation.

    GPFS can advertise a multi-MiB block size. Python's default buffered open
    then reads that much even for NumPy's six-byte magic check. Parse the header
    once through an unbuffered handle, and give the same handle to ``memmap``.
    The returned mapping owns its mmap independently of this closed handle.
    """
    with open(os.fspath(path), "rb", buffering=0) as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        elif version == (3, 0):
            # NumPy exposes no public v3 reader; its shared reader handles the
            # UTF-8 header correctly in both NumPy 1.x and 2.x.
            shape, fortran_order, dtype = np.lib.format._read_array_header(stream, version)
        else:
            raise ValueError(f"Unsupported NPY format version: {version}")
        if dtype.hasobject:
            raise ValueError("Array can't be memory-mapped: Python objects in dtype")
        if any(dimension < 0 for dimension in shape):
            raise ValueError("Negative dimensions are not allowed in an NPY array")
        offset = stream.tell()
        # Use Python integers so malformed large shapes cannot overflow before
        # checking the payload length. No array data is read for this check.
        if offset + math.prod(shape) * dtype.itemsize > os.fstat(stream.fileno()).st_size:
            raise ValueError("NPY array data is truncated")
        mapped = np.memmap(stream, dtype=dtype, mode="r", offset=offset, shape=shape,
                           order="F" if fortran_order else "C")
        if isinstance(path, Path):
            # Match memmap(path)'s filename type when its input is a Path.
            mapped.filename = path.resolve()
        return mapped
