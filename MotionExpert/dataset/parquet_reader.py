"""Bounded row-group and file caches for immutable SMPL-H Parquet columns."""
from collections import OrderedDict
import os
from pathlib import Path
import numpy as np


class ProjectedParquetReader:
    """Cache only requested row groups/columns; each DataLoader worker owns a cache."""
    def __init__(self, *, cache_mb=128, file_cache=8):
        if cache_mb < 0 or file_cache < 1: raise ValueError("Invalid bounded Parquet cache settings")
        self.max_bytes = int(cache_mb*1024**2)
        self.max_files = int(file_cache)
        self._pid = os.getpid()
        self.files, self.groups = OrderedDict(), OrderedDict()
        self.cached_bytes = 0
        self.read_log = []

    def __getstate__(self):
        value = dict(self.__dict__)
        value.update(files=OrderedDict(), groups=OrderedDict(), cached_bytes=0, read_log=[])
        return value

    def _file(self, path):
        import pyarrow.parquet as pq
        if self._pid != os.getpid():
            self.files.clear(); self.groups.clear(); self.cached_bytes = 0; self.read_log.clear()
            self._pid = os.getpid()
        path = Path(path)
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key in self.files:
            self.files.move_to_end(key)
            return key, self.files[key]
        value = pq.ParquetFile(path)
        offsets = np.cumsum([0]+[value.metadata.row_group(i).num_rows for i in range(value.num_row_groups)])
        self.files[key] = (value, offsets)
        while len(self.files) > self.max_files: self.files.popitem(last=False)
        return key, self.files[key]

    def read(self, path, frame_indices, columns):
        file_key, (parquet, offsets) = self._file(path)
        indices = np.asarray(frame_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0 or indices.min() < 0 or indices.max() >= offsets[-1]:
            raise ValueError("Selected archive frame indices outside episode")
        columns = tuple(columns)
        missing = set(columns)-set(parquet.schema_arrow.names)
        if missing: raise ValueError(f"Missing archive columns: {sorted(missing)}")
        selected_groups = np.searchsorted(offsets, indices, side="right")-1
        result = {}
        for group in np.unique(selected_groups):
            key = (file_key, int(group), columns)
            if key in self.groups:
                arrays = self.groups[key]; self.groups.move_to_end(key)
            else:
                table = parquet.read_row_group(int(group), columns=list(columns))
                arrays = {}
                for name in columns:
                    array = table[name].combine_chunks()
                    if array.null_count: raise ValueError(f"Null values in {name}")
                    if hasattr(array, "values"):
                        if array.values.null_count: raise ValueError(f"Null list elements in {name}")
                        width = array.type.list_size
                        value = array.values.to_numpy(zero_copy_only=False).reshape(len(array),width)
                    else: value = array.to_numpy(zero_copy_only=False)
                    arrays[name] = value
                size = sum(a.nbytes for a in arrays.values())
                self.read_log.append((str(path), int(group), columns))
                # Diagnostic log is bounded as well.
                del self.read_log[:-64]
                if size <= self.max_bytes:
                    while self.groups and self.cached_bytes+size > self.max_bytes:
                        _, old = self.groups.popitem(last=False)
                        self.cached_bytes -= sum(a.nbytes for a in old.values())
                    self.groups[key] = arrays; self.cached_bytes += size
            locations = np.flatnonzero(selected_groups == group)
            local = indices[locations]-offsets[group]
            for name, values in arrays.items():
                if name not in result: result[name] = np.empty((len(indices), *values.shape[1:]), dtype=values.dtype)
                result[name][locations] = values[local]
        return result
