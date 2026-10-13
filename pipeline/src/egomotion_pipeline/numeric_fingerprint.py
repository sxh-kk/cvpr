"""Chunk-independent hashes of typed numeric columns, excluding Parquet encoding."""

import hashlib
import json
import numpy as np

from .storage import digest


NON_MOTION_COLUMNS = frozenset({
    "observation.features.dino", "observation.valid.dino",
    "observation.video_timestamp", "observation.video_time_error",
})


class NumericColumnHasher:
    """Hash whole-column shape and C-order bytes across arbitrary row batches."""

    def __init__(self, specs, length):
        self.specs = specs
        self.length = int(length)
        self.rows = 0
        self.hashers = {}
        for name, spec in sorted(specs.items()):
            shape = [self.length] if spec["shape"] == [1] else [self.length, *spec["shape"]]
            header = json.dumps([name, np.dtype(spec["dtype"]).str, shape],
                                separators=(",", ":")).encode()
            self.hashers[name] = hashlib.sha256(header)

    def update(self, arrays):
        sizes = {len(np.asarray(arrays[name])) for name in self.specs}
        if len(sizes) != 1:
            raise ValueError("Numeric hash columns have different row counts")
        count = sizes.pop()
        if self.rows + count > self.length:
            raise ValueError("Numeric hash received too many rows")
        for name, spec in self.specs.items():
            value = np.asarray(arrays[name])
            shape = (count,) if spec["shape"] == [1] else (count, *spec["shape"])
            if value.dtype != np.dtype(spec["dtype"]) or value.shape != shape:
                raise ValueError(f"Numeric hash dtype/shape mismatch: {name}")
            self.hashers[name].update(memoryview(np.ascontiguousarray(value)).cast("B"))
        self.rows += count

    def finish(self):
        if self.rows != self.length:
            raise ValueError("Numeric hash received an incomplete column")
        columns = {name: hasher.hexdigest() for name, hasher in self.hashers.items()}
        return {"sha256": digest(columns), "columns": columns}


def numeric_hashes(arrays):
    if not arrays:
        raise ValueError("Cannot fingerprint empty columns")
    specs = {}
    for name, raw in arrays.items():
        value = np.asarray(raw)
        if value.ndim not in (1, 2):
            raise ValueError(f"Expected scalar or flat numeric feature: {name}")
        specs[name] = {"dtype": value.dtype.name,
                       "shape": [1] if value.ndim == 1 else list(value.shape[1:])}
    hasher = NumericColumnHasher(specs, len(next(iter(arrays.values()))))
    hasher.update(arrays)
    return hasher.finish()


def motion_digest_from_hashes(hashes):
    return digest({name: value for name, value in hashes["columns"].items()
                   if name not in NON_MOTION_COLUMNS})
