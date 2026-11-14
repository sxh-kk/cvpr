"""Optional local native5Hz DINO read cache, copied bitwise from accepted features."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

import numpy as np

from .npy_io import load_npy_mmap
from .vision_cache import CONTRACT as SOURCE_CONTRACT, storage_digest


CONTRACT = "motionexpert-packed-vision-5hz-v1"
ENV_VAR = "MOTIONEXPERT_VISION_READ_CACHE_ROOT"
_PROTOCOL = {"native_fps": 30, "fps": 5, "sampling_step": 6, "frame_phase": 0,
             "shape": [5, 1024], "storage_dtype": "float32"}


def _identity(path, *, inode=False):
    stat = os.stat(path)
    result = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if inode:
        result.update(device=stat.st_dev, inode=stat.st_ino)
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with open(os.fspath(path), "rb", buffering=0) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _owned_file(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Packed vision artifact must be an owned relative file")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Packed vision artifact escapes its root")
    return path


def _record_keys(records):
    if not isinstance(records, list):
        raise ValueError("Packed vision records must be a list")
    ids = []
    for record in records:
        if (not isinstance(record, dict) or type(record.get("episode_index")) is not int
                or record["episode_index"] < 0 or type(record.get("length")) is not int
                or record["length"] < 0):
            raise ValueError("Packed vision record episode/length is invalid")
        ids.append(record["episode_index"])
    if len(ids) != len(set(ids)):
        raise ValueError("Packed vision records contain duplicate episodes")
    return ids


def _check_memory_headroom():
    """An early pressure check, not a guarantee about an unreadable cgroup quota."""
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    if int(line.split()[1]) * 1024 < 128 * 1024 ** 3:
                        raise MemoryError("Less than 128 GiB available while packing vision features")
                    break
    except FileNotFoundError:
        pass


def _iter_native5hz_blocks(path, expected_frames, block_rows, *, cancelled=None):
    """Yield independent C-order 5Hz blocks using bounded sequential source I/O.

    Production features are C-order [N,5,1024]. Read consecutive native frames
    into RAM before selecting every sixth frame, avoiding sparse GPFS mmap
    faults. Fortran inputs retain the previous mmap behavior for compatibility.
    """
    if type(expected_frames) is not int or expected_frames < 0:
        raise ValueError("Packed vision expected_frames must be non-negative")
    if type(block_rows) is not int or block_rows < 1:
        raise ValueError("Packed vision block_rows must be positive")

    def check_read():
        if cancelled is not None and cancelled.is_set():
            raise RuntimeError("Packed vision build cancelled after another failure")
        _check_memory_headroom()

    check_read()
    with open(os.fspath(path), "rb", buffering=0) as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        elif version == (3, 0):
            # NumPy has no public UTF-8 v3 reader (also used by load_npy_mmap).
            shape, fortran_order, dtype = np.lib.format._read_array_header(stream, version)
        else:
            raise ValueError(f"Unsupported NPY format version: {version}")
        if dtype.hasobject or dtype != np.float32 or shape != (expected_frames, 5, 1024):
            raise ValueError("Packed vision source feature shape/dtype differs")
        offset = stream.tell()
        if offset + expected_frames * 5 * 1024 * dtype.itemsize > os.fstat(stream.fileno()).st_size:
            raise ValueError("Packed vision source array data is truncated")
        if not expected_frames:
            return
        rows = (expected_frames + 5) // 6
        if fortran_order:
            native = np.memmap(stream, dtype=dtype, mode="r", offset=offset, shape=shape, order="F")
            try:
                for start in range(0, rows, block_rows):
                    check_read()
                    stop = min(start + block_rows, rows)
                    yield start, native[start * 6:stop * 6:6].copy(order="C")
            finally:
                native._mmap.close()
            return
        native_buffer = np.empty((min(expected_frames, 6 * block_rows), 5, 1024), dtype=dtype)
        buffer_bytes = memoryview(native_buffer).cast("B")
        for start in range(0, rows, block_rows):
            check_read()
            native_count = min(6 * block_rows, expected_frames - start * 6)
            target_bytes = buffer_bytes[:native_count * 5 * 1024 * dtype.itemsize]
            received = 0
            while received < len(target_bytes):
                if cancelled is not None and cancelled.is_set():
                    raise RuntimeError("Packed vision build cancelled after another failure")
                try:
                    count = stream.readinto(target_bytes[received:])
                except InterruptedError:
                    continue
                if not count:
                    raise ValueError("Packed vision source array data is truncated while reading")
                received += count
            # A copy also owns single-row blocks, which NumPy can otherwise
            # consider contiguous views of the reusable native buffer.
            yield start, native_buffer[:native_count:6].copy(order="C")


def build_packed_vision_cache(vision_root, records_path, output_root, *, report_path=None,
                              workers=8, block_rows=256):
    """Copy only accepted record episodes, without FK, statistics or source QA.

    Each worker writes a disjoint row range. A complete manifest is published
    only after every native[::6] byte has been compared with its packed copy.
    Existing output directories are never reused or overwritten.
    """
    if type(workers) is not int or not 0 <= workers <= 64:
        raise ValueError("Packed vision workers must be between 0 and 64")
    if type(block_rows) is not int or block_rows < 1:
        raise ValueError("Packed vision block_rows must be positive")
    vision_root = Path(vision_root).resolve(strict=True)
    records_path = Path(records_path).resolve(strict=True)
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Packed vision output already exists: {output_root}")
    if report_path is not None and Path(report_path).exists():
        raise FileExistsError(f"Packed vision report already exists: {report_path}")
    manifest_path = _owned_file(vision_root, "manifest.json")
    source_manifest_identity = _identity(manifest_path)
    source_manifest_sha256 = _sha256(manifest_path)
    source = json.loads(manifest_path.read_text())
    if (source.get("contract") != SOURCE_CONTRACT or source.get("state") != "complete"
            or source.get("scope") != "all" or source.get("validity_equal_source") is not True
            or any(source.get(key) != value for key, value in {
                "native_fps": 30, "fps": 30, "sampling_step": 1, "frame_phase": 0,
                "shape": [5, 1024], "storage_dtype": "float32"}.items())):
        raise ValueError("Packed vision requires a complete accepted native30Hz source")
    records_identity = _identity(records_path)
    records_sha256 = _sha256(records_path)
    records = json.loads(records_path.read_text())
    ids = _record_keys(records)
    source_ids = source.get("episode_indices")
    if (not isinstance(source_ids, list) or any(type(item) is not int or item < 0 for item in source_ids)
            or len(set(source_ids)) != len(source_ids)
            or not set(ids).issubset(source_ids)
            or source.get("source_episode_count") != len(source_ids)
            or source.get("completed_episodes") != len(source_ids)):
        raise ValueError("Packed vision records are absent from the completed source")
    plans, offset = [], 0
    for record in records:
        rows = (record["length"] + 5) // 6
        plans.append((record, offset, rows))
        offset += rows
    cancelled = threading.Event()

    def collect(function, items, phase):
        pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            results = pool.map(function, items) if pool is not None else map(function, items)
            collected = []
            for descriptor in results:
                collected.append(descriptor)
                if len(collected) % 64 == 0 or len(collected) == len(items):
                    _check_memory_headroom()
                    print(f"{phase} {len(collected)}/{len(items)} episodes", file=sys.stderr, flush=True)
        except BaseException:
            cancelled.set()
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            if pool is not None:
                pool.shutdown(wait=True)
        return collected

    def prepare_episode(plan):
        if cancelled.is_set():
            raise RuntimeError("Packed vision preparation cancelled after another failure")
        record, row_offset, rows = plan
        episode, length = record["episode_index"], record["length"]
        prefix = f"episodes/{episode:08d}"
        receipt_path = _owned_file(vision_root, f"{prefix}/receipt.json")
        receipt_identity = _identity(receipt_path)
        receipt = json.loads(receipt_path.read_text())
        record_sha256 = storage_digest(record)
        if (receipt.get("contract") != SOURCE_CONTRACT or receipt.get("state") != "complete"
                or receipt.get("episode_index") != episode or receipt.get("frames") != length
                or receipt.get("source_key") != record.get("source_key")
                or receipt.get("source_record_sha256") != record_sha256
                or receipt.get("validity_equal_source") is not True
                or receipt.get("encoder") != source.get("encoder")):
            raise ValueError("Packed vision source receipt/record identity differs")
        path = _owned_file(vision_root, f"{prefix}/features.npy")
        identity = _identity(path)
        declared = receipt.get("output_identities", {}).get("features.npy", {})
        if identity != {key: declared.get(key) for key in ("size_bytes", "mtime_ns")}:
            raise ValueError("Packed vision source feature stamp changed")
        return {"episode_index": episode, "length": length, "rows": rows,
                "row_offset": row_offset, "source_record_sha256": record_sha256,
                "source_feature_identity": identity, "source_receipt_identity": receipt_identity}

    _check_memory_headroom()
    print(f"Preparing accepted source identities for {len(plans)} episodes with {workers} workers", file=sys.stderr, flush=True)
    episodes = collect(prepare_episode, plans, "Prepared accepted source identities")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    capacity = os.statvfs(output_root.parent)
    if offset * 5 * 1024 * 4 + 1024 * 1024 > capacity.f_bavail * capacity.f_frsize:
        raise OSError("Insufficient filesystem capacity for packed vision output")
    _check_memory_headroom()
    output_root.mkdir()
    packed_path = output_root / "packed.npy"
    packed = np.lib.format.open_memmap(packed_path, mode="w+", dtype=np.float32,
                                       shape=(offset, 5, 1024))

    def copy_episode(descriptor):
        if cancelled.is_set():
            raise RuntimeError("Packed vision build cancelled after another failure")
        episode = descriptor["episode_index"]
        prefix = vision_root / f"episodes/{episode:08d}"
        path = prefix / "features.npy"
        if _identity(path) != descriptor["source_feature_identity"]:
            raise ValueError("Packed vision source feature changed before copying")
        digest = hashlib.sha256()
        with closing(_iter_native5hz_blocks(path, descriptor["length"], block_rows,
                                            cancelled=cancelled)) as source_blocks:
            for start, values in source_blocks:
                if cancelled.is_set():
                    raise RuntimeError("Packed vision build cancelled after another failure")
                _check_memory_headroom()
                stop = min(start + block_rows, descriptor["rows"])
                target = packed[descriptor["row_offset"] + start:descriptor["row_offset"] + stop]
                target[:] = values
                if not np.array_equal(values.view(np.uint8), np.asarray(target).view(np.uint8)):
                    raise ValueError("Packed vision bitwise copy verification failed")
                digest.update(values.view(np.uint8))
        if (_identity(path) != descriptor["source_feature_identity"]
                or _identity(prefix / "receipt.json") != descriptor["source_receipt_identity"]):
            raise ValueError("Packed vision source changed while copying")
        return dict(descriptor, sha256=digest.hexdigest())

    try:
        verified = collect(copy_episode, episodes, "Packed and bitwise-verified")
        packed.flush()
    finally:
        packed._mmap.close()
    if (_identity(manifest_path) != source_manifest_identity
            or _sha256(manifest_path) != source_manifest_sha256
            or _identity(records_path) != records_identity or _sha256(records_path) != records_sha256):
        raise ValueError("Packed vision source publication/records changed while copying")
    packed_path.chmod(0o444)
    result = {"contract": CONTRACT, "state": "complete", **_PROTOCOL,
              "source_vision_root": str(vision_root),
              "source_vision_manifest_sha256": source_manifest_sha256,
              "source_vision_episode_count": len(source_ids),
              "source_records": {"path": str(records_path), "sha256": records_sha256,
                                 "canonical_sha256": storage_digest(records), **records_identity},
              "selected_episode_indices": ids, "total_rows": offset, "episodes": verified,
              "packed": {"path": "packed.npy", "shape": [offset, 5, 1024], "dtype": "float32",
                         **_identity(packed_path, inode=True)},
              "verification": {"method": "bitwise-native5hz-v1", "episodes": len(verified),
                               "rows": offset, "bytes": offset * 5 * 1024 * 4,
                               "source_equals_packed": True,
                               "payload_sha256": storage_digest([item["sha256"] for item in verified])}}
    if report_path is not None:
        _atomic_json(report_path, result)
    _atomic_json(output_root / "manifest.json", result)
    return result


class PackedVisionCache:
    """One local read-only mapping per process, bound to an accepted source.

    VisionProfileCache has already checked every source receipt/artifact stamp
    at startup. Construction compares against those validated descriptors in
    memory. Worker reads never reopen or stat the original GPFS episodes; only
    the local packed publication needs mapping/epoch identity checks.
    """

    def __init__(self, root, *, vision_cache, records, records_sha256=None):
        self.root = Path(root).resolve(strict=True)
        self.manifest_path = _owned_file(self.root, "manifest.json")
        self._manifest_identity = _identity(self.manifest_path, inode=True)
        manifest = json.loads(self.manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise ValueError("Packed vision manifest must be an object")
        self.manifest = manifest
        ids = _record_keys(records)
        if (manifest.get("contract") != CONTRACT or manifest.get("state") != "complete"
                or any(manifest.get(key) != value for key, value in _PROTOCOL.items())
                or manifest.get("source_vision_root") != str(vision_cache.root)
                or manifest.get("source_vision_manifest_sha256") != vision_cache.metadata["manifest_sha256"]
                or manifest.get("source_vision_episode_count") != len(vision_cache._episodes)
                or manifest.get("selected_episode_indices") != ids):
            raise ValueError("Packed vision source root/manifest/population/protocol differs")
        source_records = manifest.get("source_records", {})
        if (not isinstance(source_records, dict)
                or (records_sha256 is not None and source_records.get("sha256") != records_sha256)
                or (records_sha256 is None and source_records.get("canonical_sha256") != storage_digest(records))):
            raise ValueError("Packed vision accepted-record identity differs")
        descriptors = manifest.get("episodes")
        if not isinstance(descriptors, list) or len(descriptors) != len(records):
            raise ValueError("Packed vision episode descriptors are incomplete")
        self._episodes, offset = {}, 0
        for record, descriptor in zip(records, descriptors):
            episode, length = record["episode_index"], record["length"]
            rows = (length + 5) // 6
            original = vision_cache._episodes.get(episode)
            if (not isinstance(descriptor, dict) or original is None
                    or descriptor.get("episode_index") != episode or descriptor.get("length") != length
                    or descriptor.get("row_offset") != offset or descriptor.get("rows") != rows
                    or descriptor.get("source_record_sha256") != storage_digest(record)
                    or descriptor.get("source_feature_identity") != dict(zip(
                        ("size_bytes", "mtime_ns"), original["artifacts"]["features.npy"][1]))
                    or descriptor.get("source_receipt_identity") != dict(zip(
                        ("size_bytes", "mtime_ns"), original["receipt_stamp"]))
                    or not isinstance(descriptor.get("sha256"), str) or len(descriptor["sha256"]) != 64):
                raise ValueError("Packed vision episode length/offset/source identity differs")
            self._episodes[episode] = descriptor
            offset += rows
        verification = manifest.get("verification", {})
        if (not isinstance(verification, dict) or manifest.get("total_rows") != offset
                or verification.get("method") != "bitwise-native5hz-v1"
                or verification.get("source_equals_packed") is not True
                or verification.get("episodes") != len(records) or verification.get("rows") != offset
                or verification.get("bytes") != offset * 5 * 1024 * 4
                or verification.get("payload_sha256") != storage_digest([item["sha256"] for item in descriptors])):
            raise ValueError("Packed vision full bitwise verification is missing")
        self._packed_identity = manifest.get("packed", {})
        if (not isinstance(self._packed_identity, dict) or self._packed_identity.get("path") != "packed.npy"
                or self._packed_identity.get("shape") != [offset, 5, 1024]
                or self._packed_identity.get("dtype") != "float32"):
            raise ValueError("Packed vision packed shape/dtype differs")
        self.packed_path = _owned_file(self.root, "packed.npy")
        self._mapped, self._pid, self._verified = None, os.getpid(), set()
        self.assert_current()
        # Validate the payload header at startup; worker mappings are lazy.
        mapped = self._open()
        mapped._mmap.close()
        self._mapped = None

    def assert_current(self):
        if (_identity(self.manifest_path, inode=True) != self._manifest_identity
                or _identity(self.packed_path, inode=True) != {key: self._packed_identity.get(key)
                    for key in ("size_bytes", "mtime_ns", "device", "inode")}):
            raise ValueError("Packed vision publication/source identity changed")

    def _reset_for_process(self):
        if self._pid != os.getpid():
            if self._mapped is not None:
                self._mapped._mmap.close()
            self._mapped, self._pid, self._verified = None, os.getpid(), set()

    def _open(self):
        self._reset_for_process()
        if self._mapped is None:
            self.assert_current()
            mapped = load_npy_mmap(os.fspath(self.packed_path))
            if mapped.dtype != np.float32 or list(mapped.shape) != self._packed_identity["shape"]:
                mapped._mmap.close()
                raise ValueError("Packed vision mmap header shape/dtype differs")
            self._mapped = mapped
        return self._mapped

    def read(self, record, selected):
        self._reset_for_process()
        if not isinstance(record, dict) or type(record.get("episode_index")) is not int:
            raise ValueError("Packed vision requires an accepted native integer episode record")
        selected = np.asarray(selected)
        episode = record.get("episode_index")
        descriptor = self._episodes.get(episode)
        if (descriptor is None or record.get("length") != descriptor["length"]
                or selected.ndim != 1 or selected.dtype.kind not in "iu"
                or np.any(selected < 0) or np.any(selected % 6)
                or np.any(selected >= descriptor["length"]) or np.any(np.diff(selected.astype(np.int64)) <= 0)):
            raise ValueError("Packed vision selection/episode must use native zero-phase 5Hz frames")
        mapped = self._open()
        self._verified.add(episode)
        return np.asarray(mapped[descriptor["row_offset"] + selected // 6]).copy()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_mapped"], state["_verified"] = None, set()
        return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-root", required=True)
    parser.add_argument("--records-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", dest="report_path")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--block-rows", type=int, default=256)
    arguments = parser.parse_args(argv)
    result = build_packed_vision_cache(**vars(arguments))
    print(json.dumps({"state": result["state"], "episodes": len(result["episodes"]),
                      "rows": result["total_rows"], "bytes": result["verification"]["bytes"]}), flush=True)


if __name__ == "__main__":
    main()
