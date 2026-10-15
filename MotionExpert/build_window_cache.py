"""Build resumable, episode-sharded raw 5Hz/11-frame training windows."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .dataset.window_cache import (
    ARRAY_SPECS, CONTRACT, atomic_json, cache_binding, digest, file_sha256,
    protocol_from_config, stamp,
)

_WORKER_ARCHIVE = None
_WORKER_ROOT = None
_WORKER_BINDING = None
_WORKER_PROTOCOL = None
_BLOCK_FRAMES = None


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def _archive(cfg, allow_unpublished):
    from .dataset.smplh_archive import SMPLHParameterArchive
    data = cfg.DATA
    return SMPLHParameterArchive(
        getattr(data, "ARCHIVE_ROOT", None) or data.DATA_DIR,
        cache_mb=getattr(data, "ROW_GROUP_CACHE_MB", 128),
        file_cache=getattr(data, "PARQUET_FILE_CACHE", 8),
        allow_unpublished=allow_unpublished,
        vision_profile_root=data.VISION_PROFILE_ROOT, image_profile=data.IMAGE_PROFILE)


def _init_worker(archive, root, binding, protocol, block_frames):
    global _WORKER_ARCHIVE, _WORKER_ROOT, _WORKER_BINDING, _WORKER_PROTOCOL, _BLOCK_FRAMES
    import torch
    import pyarrow as pa
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    # Spawn serializes the already validated archive. Its Parquet/vision
    # __getstate__ methods discard open files and cached arrays, so each worker
    # opens only the episode artifacts it uses instead of repeating a complete
    # source-publication scan. Per-episode and final parent checks remain below.
    _WORKER_ARCHIVE = archive
    if cache_binding(_WORKER_ARCHIVE, protocol) != binding:
        raise ValueError("Window-cache worker source binding differs from the planning archive")
    _WORKER_ROOT, _WORKER_BINDING, _WORKER_PROTOCOL = Path(root), binding, protocol
    _BLOCK_FRAMES = block_frames


def _active_workers(control, maximum):
    """Read an explicitly supplied scheduling limit; never silently ignore it."""
    if control is None:
        return maximum
    try:
        value = json.loads(Path(control).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read window-cache worker control: {control}") from error
    if (not isinstance(value, dict) or set(value) != {"active_workers"}
            or type(value["active_workers"]) is not int
            or not 1 <= value["active_workers"] <= maximum):
        raise ValueError(f"Worker control requires integer active_workers in [1, {maximum}]")
    return value["active_workers"]


def _execute_episode_tasks(pool, tasks, *, max_workers, worker_control, progress):
    """Use public Future APIs to adjust task admission without restarting workers."""
    iterator, pending, exhausted = iter(tasks), set(), False
    previous_active = None
    while pending or not exhausted:
        active = _active_workers(worker_control, max_workers)
        if worker_control is not None and active != previous_active:
            log(f"Worker control active_workers={active}, max_workers={max_workers}")
        previous_active = active
        # The original no-control mode retains its small prefetch queue. With
        # control, pending Futures directly bound active episode tasks; a lower
        # limit drains existing work naturally before any further submission.
        limit = active if worker_control is not None else max_workers * 2
        while not exhausted and len(pending) < limit:
            task = next(iterator, None)
            if task is None:
                exhausted = True
                break
            pending.add(pool.submit(_write_episode, task))
        if not pending:
            break
        done, pending = wait(pending, return_when=FIRST_COMPLETED,
                             timeout=1 if worker_control is not None else None)
        for future in done:
            progress(future.result(), active)


def _index_episode(archive, record, protocol):
    """Reuse the old index method without initializing a full dataset per task."""
    from .dataset.smplh_motion_dataset import SMPLHMotionDataset
    dataset = SMPLHMotionDataset.__new__(SMPLHMotionDataset)
    dataset.archive, dataset.reader = archive, archive.reader
    dataset.sampling_step, dataset.segment_stride = 6, protocol["segment_stride"]
    dataset.window, dataset.forecast_prefix = 11, 6
    dataset.min_valid_image_fraction = protocol["min_valid_image_fraction"]
    dataset.sparse_joint_indices = list(protocol["joint_indices"])
    dataset.idx_to_sidx_fidx, dataset.rejected_windows = [], {}
    dataset._index_record(0, record)
    return [start for _, start in dataset.idx_to_sidx_fidx], dataset.rejected_windows


def _resume_receipt(root, record, starts, binding):
    path = root / f"episodes/{record['episode_index']:08d}/receipt.json"
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text())
    if (receipt.get("contract") != CONTRACT or receipt.get("state") != "complete"
            or receipt.get("binding_sha256") != digest(binding)
            or receipt.get("source_record_sha256") != digest(record)
            or receipt.get("windows") != len(starts)
            or receipt.get("start_frames_sha256") != digest(starts)
            or set(receipt.get("arrays", {})) != set(ARRAY_SPECS)):
        raise ValueError("Existing window-cache shard belongs to a different source/protocol/index")
    for key, (dtype, tail) in ARRAY_SPECS.items():
        identity = receipt["arrays"][key]
        relative = Path(identity["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Window-cache receipt artifact path is not owned")
        artifact = (root / relative).resolve(strict=True)
        if not artifact.is_relative_to(root.resolve()):
            raise ValueError("Window-cache artifact escapes its root")
        if (identity["dtype"] != dtype or identity["shape"] != [len(starts), *tail]
                or stamp(artifact) != (identity["size_bytes"], identity["mtime_ns"])
                or file_sha256(artifact) != identity["sha256"]):
            raise ValueError("Existing committed window-cache artifact failed its digest/identity check")
    return receipt


def _block(archive, record, selected):
    """Decode each selected world frame once, keeping true validity for every row."""
    arrays = archive._motion(record, selected)
    position, rotation = archive.model.decode(arrays)
    count = len(selected)
    for component, value in (("position", position), ("rotation", rotation)):
        valid = arrays[f"estimated.valid.body_{component}"]
        key = "observation.state.body_world" + ("_rotation" if component == "rotation" else "")
        arrays[key] = np.where(valid[..., None], value, 0 if component == "position" else [1, 0, 0, 0]).reshape(count, -1).astype(np.float32)
        arrays[f"observation.valid.body_{component}"] = valid.copy()
        arrays[f"observation.source_observed.body_{component}"] = arrays[f"source.observed.body_{component}"].copy()
        arrays[f"observation.estimated.completed.body_{component}"] = arrays[f"estimated.completed.body_{component}"].copy()
    image = archive.vision_cache.read(record, selected, expected=arrays, load_features=False)
    arrays["observation.valid.dino"] = image["observation.valid.dino"]
    metadata = deepcopy(record["metadata"])
    metadata.update(rotation_supervision_policy="smplh_model_defined", rotation_target_basis="smplh_body22_axes_v1",
                    target_model_axes_defined=True, model_sha256=archive.model_sha256,
                    archive_schema_version=archive.info["schema_version"],
                    vision_profile=deepcopy(archive.vision_cache.metadata))
    return arrays, metadata


def _write_episode(task):
    record, starts = task
    archive, root, protocol, binding = _WORKER_ARCHIVE, _WORKER_ROOT, _WORKER_PROTOCOL, _WORKER_BINDING
    receipt = _resume_receipt(root, record, starts, binding)
    if receipt is not None:
        return {"episode_index": record["episode_index"], "windows": len(starts), "resumed": True}
    from .dataset.smplh_representation import build_smplh_window
    episode = record["episode_index"]
    destination = root / f"episodes/{episode:08d}"
    temporary = root / f"episodes/.{episode:08d}.building"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    count = len(starts)
    arrays_out = {key: np.empty((count, *tail), dtype=dtype)
                  for key, (dtype, tail) in ARRAY_SPECS.items()}
    arrays_out["start_frames"][:] = starts
    template = None
    cursor = 0
    while cursor < count:
        first = starts[cursor]
        end_cursor = cursor + 1
        while end_cursor < count and (starts[end_cursor] - first) // 6 + 11 <= _BLOCK_FRAMES:
            end_cursor += 1
        selected = np.arange(first, starts[end_cursor - 1] + 61, 6, dtype=np.int64)
        block_arrays, metadata = _block(archive, record, selected)
        for row in range(cursor, end_cursor):
            offset = (starts[row] - first) // 6
            window_arrays = {key: value[offset:offset + 11] for key, value in block_arrays.items()}
            window = build_smplh_window(window_arrays, metadata, fps=5, forecast_prefix=6,
                                       contact_speed_m_s=protocol["contact_speed_m_s"],
                                       contact_height_m=protocol["contact_height_m"])
            details = window.metadata["training_window"]
            if not window.motion_valid[6:, :108].any():
                raise ValueError("Planned window lacks a valid future motion label")
            if template is None:
                template = deepcopy(window.metadata)
                for name in ("window_from_archive_world", "archive_world_from_window", "floor_height_m", "floor_rotation", "floor_position"):
                    template["training_window"].pop(name, None)
            for name, value in (("motion", window.motion), ("motion_valid", window.motion_valid),
                                ("trajectory", window.trajectory), ("trajectory_valid", window.trajectory_valid)):
                arrays_out[name][row] = value
            arrays_out["world_from_window"][row] = details["archive_world_from_window"]
            arrays_out["window_from_world"][row] = details["window_from_archive_world"]
            floor = np.tile(np.eye(4), (11, 1, 1))
            floor[:, :3, :3], floor[:, :3, 3] = details["floor_rotation"], details["floor_position"]
            arrays_out["floor_transform"][row] = floor
            arrays_out["floor_height"][row] = details["floor_height_m"]
            arrays_out["ground_fallback_used"][row] = window.metadata["ground"].get("window_fallback_used", False)
            arrays_out["ground_point_count"][row] = window.metadata["ground"].get("window_fallback_valid_point_count", 0)
            arrays_out["historical_image_valid"][row] = window_arrays["observation.valid.dino"][:6]
        cursor = end_cursor
    archive.assert_current(record)
    # Episodes are small enough to retain their generated values in RAM. A
    # self-consistent file checksum cannot prove a successful write: compare
    # every saved array with those values before publishing its receipt.
    for key, value in arrays_out.items():
        path = temporary / f"{key}.npy"
        np.save(path, value, allow_pickle=False)
        saved = np.load(path, allow_pickle=False)
        if (saved.dtype != value.dtype or saved.shape != value.shape
                or not np.array_equal(saved, value)):
            if key == "start_frames":
                raise ValueError("Written episode start-frame keys differ from the planned window index")
            raise ValueError(f"Written episode array differs from the generated values: {key}")
    identities = {}
    for key, (dtype, tail) in ARRAY_SPECS.items():
        path = temporary / f"{key}.npy"
        size, modified = stamp(path)
        identities[key] = {"path": str((destination / path.name).relative_to(root)), "dtype": dtype,
                           "shape": [count, *tail], "size_bytes": size, "mtime_ns": modified,
                           "sha256": file_sha256(path)}
    receipt = {"contract": CONTRACT, "state": "complete", "episode_index": episode,
               "split": record["split"], "source_record_sha256": digest(record),
               "binding_sha256": digest(binding), "windows": count,
               "start_frames_sha256": digest(starts), "arrays": identities,
               "template_metadata": template, "completed_at_unix": time.time()}
    # Commit data and receipt together. A process killed before this rename
    # leaves only a disposable .building directory; it cannot strand a shard.
    atomic_json(temporary / "receipt.json", receipt)
    if destination.exists():
        raise ValueError("An uncommitted destination exists; refusing to replace unknown files")
    os.replace(temporary, destination)
    return {"episode_index": episode, "windows": count, "resumed": False}


def _packed_receipt(root, split, binding, index, sources, windows):
    path = root / f"packed/{split}/receipt.json"
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text())
    if (receipt.get("contract") != CONTRACT or receipt.get("state") != "complete"
            or receipt.get("split") != split or receipt.get("binding_sha256") != digest(binding)
            or receipt.get("index_sha256") != index["sha256"]
            or receipt.get("source_receipts_sha256") != digest(sources)
            or receipt.get("windows") != windows
            or set(receipt.get("arrays", {})) != set(ARRAY_SPECS)):
        raise ValueError("Existing packed cache identity/index/source differs")
    for key, (dtype, tail) in ARRAY_SPECS.items():
        identity = receipt["arrays"][key]
        relative = Path(identity["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Packed cache artifact path must be owned")
        artifact = (root / relative).resolve(strict=True)
        if not artifact.is_relative_to(root.resolve()):
            raise ValueError("Packed cache artifact escapes its directory")
        if (identity["dtype"] != dtype or identity["shape"] != [windows, *tail]
                or stamp(artifact) != (identity["size_bytes"], identity["mtime_ns"])
                or file_sha256(artifact) != identity["sha256"]):
            raise ValueError("Committed packed cache failed its digest/identity check")
    return receipt


def _update_payload_digest(result, value, *, block_bytes=16 * 1024 * 1024):
    """Hash C-order array bytes in bounded views, including empty arrays."""
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        raise ValueError("Packed cache payload hashing requires C-order source arrays")
    if not array.nbytes:
        return
    raw = memoryview(array).cast("B")
    for first in range(0, raw.nbytes, block_bytes):
        result.update(raw[first:first + block_bytes])


def _pack_split(root, split, binding, index, descriptors):
    """One sequential copy per shard, then publish twelve split-wide mmap files."""
    windows = sum(item["windows"] for item in descriptors)
    sources = [{"episode_index": item["episode_index"], "receipt_sha256": item["receipt_sha256"]}
               for item in descriptors]
    existing = _packed_receipt(root, split, binding, index, sources, windows)
    if existing is not None:
        return existing
    index_path = root / index["path"]
    if file_sha256(index_path) != index["sha256"]:
        raise ValueError("Frozen split index digest differs before packing")
    frozen_index = np.load(index_path, allow_pickle=False)
    if (frozen_index.dtype != np.int64 or frozen_index.shape != (windows, 2)
            or index["windows"] != windows):
        raise ValueError("Frozen split index population/shape differs before packing")
    destination = root / f"packed/{split}"
    temporary = root / f"packed/.{split}.building"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    packed = {}
    source_hashes = {key: hashlib.sha256() for key in ARRAY_SPECS}
    for key, (dtype, tail) in ARRAY_SPECS.items():
        shape = (windows, *tail)
        if windows:
            packed[key] = np.lib.format.open_memmap(temporary / f"{key}.npy", mode="w+", dtype=dtype, shape=shape)
        else:
            packed[key] = np.empty(shape, dtype=dtype)
            np.save(temporary / f"{key}.npy", packed[key], allow_pickle=False)
    offset = 0
    for sequence, descriptor in enumerate(descriptors):
        receipt = json.loads((root / descriptor["receipt_path"]).read_text())
        count = descriptor["windows"]
        if descriptor.get("row_offset") != offset:
            raise ValueError("Episode row offsets differ from the frozen split index before packing")
        rows = frozen_index[offset:offset + count]
        source_starts = np.load(root / receipt["arrays"]["start_frames"]["path"], allow_pickle=False)
        if (source_starts.dtype != np.int64 or source_starts.shape != (count,)
                or not np.all(rows[:, 0] == sequence)
                or not np.array_equal(source_starts, rows[:, 1])
                or digest(source_starts.tolist()) != receipt["start_frames_sha256"]):
            raise ValueError("Episode start-frame keys differ from the frozen split index before packing")
        for key, spec in receipt["arrays"].items():
            source = np.load(root / spec["path"], mmap_mode="r" if count else None, allow_pickle=False)
            if list(source.shape) != spec["shape"] or str(source.dtype) != spec["dtype"]:
                raise ValueError("Episode array header differs before packing")
            _update_payload_digest(source_hashes[key], source)
            packed[key][offset:offset + count] = source
            if isinstance(source, np.memmap):
                source._mmap.close()
        offset += count
    for value in packed.values():
        if isinstance(value, np.memmap):
            value.flush()
            value._mmap.close()
    packed_starts = np.load(temporary / "start_frames.npy", allow_pickle=False)
    if (packed_starts.dtype != np.int64 or packed_starts.shape != (windows,)
            or not np.array_equal(packed_starts, frozen_index[:, 1])):
        raise ValueError("Packed start-frame keys differ from the frozen split index after writing")
    identities = {}
    for key, (dtype, tail) in ARRAY_SPECS.items():
        path = temporary / f"{key}.npy"
        saved = np.load(path, mmap_mode="r" if windows else None, allow_pickle=False)
        try:
            if saved.shape != (windows, *tail) or str(saved.dtype) != dtype:
                raise ValueError(f"Packed array header differs after writing: {key}")
            saved_hash = hashlib.sha256()
            _update_payload_digest(saved_hash, saved)
            if saved_hash.digest() != source_hashes[key].digest():
                raise ValueError(f"Packed array payload differs from its source arrays after writing: {key}")
        finally:
            if isinstance(saved, np.memmap):
                saved._mmap.close()
        size, modified = stamp(path)
        identities[key] = {"path": str((destination / path.name).relative_to(root)),
                           "dtype": dtype, "shape": [windows, *tail], "size_bytes": size,
                           "mtime_ns": modified, "sha256": file_sha256(path)}
    receipt = {"contract": CONTRACT, "state": "complete", "split": split,
               "binding_sha256": digest(binding), "windows": windows,
               "index_sha256": index["sha256"], "source_receipts_sha256": digest(sources),
               "arrays": identities, "completed_at_unix": time.time()}
    atomic_json(temporary / "receipt.json", receipt)
    if destination.exists():
        raise ValueError("Uncommitted packed destination exists; refusing to replace unknown files")
    os.replace(temporary, destination)
    return receipt


def build_window_cache(cfg, output_root, *, workers=2, selected_episodes=None,
                       diagnostic=False, block_frames=4096, allow_unpublished=False,
                       worker_control=None):
    """Build all splits, or an explicitly diagnostic subset; never publish QA acceptance."""
    if type(workers) is not int or workers < 0 or type(block_frames) is not int or block_frames < 11:
        raise ValueError("Invalid window-cache worker/block limits")
    if selected_episodes is not None and not diagnostic:
        raise ValueError("Selected-episode builds must be explicitly diagnostic")
    if worker_control is not None:
        if workers == 0:
            raise ValueError("Worker control requires a multiprocessing builder")
        worker_control = Path(worker_control).resolve()
        _active_workers(worker_control, workers)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".build.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        log("Validating source archive, model and native30Hz DINO publication")
        archive = _archive(cfg, allow_unpublished)
        log("Source publication validated; planning original window indices")
        protocol = protocol_from_config(cfg)
        binding = cache_binding(archive, protocol)
        source_records = archive.records()
        selected = None
        if selected_episodes is not None:
            if (not selected_episodes or any(type(index) is not int or not 0 <= index < len(source_records) for index in selected_episodes)
                    or len(set(selected_episodes)) != len(selected_episodes)):
                raise ValueError("Diagnostic episode identities must be unique valid source indices")
            selected = set(selected_episodes)
        tasks, records, rejected, rejected_windows = [], [], {}, {}
        last_index_log = time.monotonic()
        for record in source_records:
            if selected is not None and record["episode_index"] not in selected:
                continue
            meta = record["metadata"]
            if (meta.get("training_eligible") is False or meta.get("canonical_training_eligible") is False or meta.get("quarantine_reasons", [])):
                rejected[record["source_key"]] = list(meta.get("quarantine_reasons") or ["training_ineligible"])
                continue
            starts, reasons = _index_episode(archive, record, protocol)
            records.append(record)
            tasks.append((record, starts))
            for key, value in reasons.items():
                rejected_windows[key] = rejected_windows.get(key, 0) + value
            if time.monotonic() - last_index_log >= 30:
                log(f"Indexed {len(tasks)} eligible episodes; latest source episode {record['episode_index']}")
                last_index_log = time.monotonic()
        manifest_path = root / "manifest.json"
        base = {"contract": CONTRACT, "state": "building", "scope": "selected_episodes" if diagnostic else "all",
                "diagnostic": bool(diagnostic), "selected_episode_indices": sorted(selected) if selected is not None else None,
                "splits": ["train", "val", "test"], "binding": binding, "binding_sha256": digest(binding),
                "archive_root": str(archive.root), "vision_profile_root": str(archive.vision_cache.root),
                "records_path": "records.json",
                "rejected_episodes": rejected, "rejected_windows": rejected_windows}
        previous = None
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if (previous.get("binding") != binding or previous.get("diagnostic") != diagnostic
                    or previous.get("selected_episode_indices") != base["selected_episode_indices"]):
                raise ValueError("Existing window-cache manifest belongs to another run/source/profile")
        records_path = root / "records.json"
        if records_path.exists():
            if json.loads(records_path.read_text()) != records:
                raise ValueError("Existing window-cache records differ from the frozen source")
        else:
            atomic_json(records_path, records)
        base["records_sha256"] = file_sha256(records_path)
        if previous is not None and previous.get("state") == "complete" and previous.get("packed"):
            # A no-op resume must not temporarily revoke a complete cache used
            # by live DataLoaders. Verify committed digests without rewriting it.
            if previous.get("records_sha256") != base["records_sha256"]:
                raise ValueError("Completed window-cache records digest differs")
            if (set(previous.get("indices", {})) != set(base["splits"])
                    or set(previous["packed"]) != set(base["splits"])):
                raise ValueError("Completed packed cache does not cover all original splits")
            for record, starts in tasks:
                if _resume_receipt(root, record, starts, binding) is None:
                    raise ValueError("Completed window-cache shard is missing")
            for split, identity in previous.get("indices", {}).items():
                if file_sha256(root / identity["path"]) != identity["sha256"]:
                    raise ValueError("Completed window-cache split index digest differs")
                record_ids = [number for number, record in enumerate(records) if record["split"] == split]
                expected_rows = np.asarray([(sequence, start) for sequence, number in enumerate(record_ids)
                                            for start in tasks[number][1]], dtype=np.int64).reshape(-1, 2)
                saved_rows = np.load(root / identity["path"], allow_pickle=False)
                if identity["windows"] != len(expected_rows) or not np.array_equal(saved_rows, expected_rows):
                    raise ValueError("Completed split index differs from the original window keys")
                source_descriptors = [item for item in previous["episodes"] if item["split"] == split]
                sources = [{"episode_index": item["episode_index"], "receipt_sha256": item["receipt_sha256"]}
                           for item in source_descriptors]
                packed = _packed_receipt(root, split, binding, identity, sources, identity["windows"])
                if packed is None or packed != previous["packed"][split]:
                    raise ValueError("Completed packed cache is absent or differs from its manifest")
            archive.assert_current()
            return previous
        atomic_json(manifest_path, base)
        log(f"Building {len(tasks)} episodes with {workers or 1} CPU processes")
        completed, windows, resumed = 0, 0, 0
        last_build_log = time.monotonic()
        def progress(result, active_workers):
            nonlocal completed, windows, resumed, last_build_log
            completed += 1
            windows += result["windows"]
            resumed += int(result["resumed"])
            atomic_json(root / "progress.json", {"state": "building", "completed_episodes": completed,
                        "total_episodes": len(tasks), "completed_windows": windows, "resumed_episodes": resumed,
                        "active_workers": active_workers, "max_workers": workers or 1,
                        "updated_at_unix": time.time()})
            if time.monotonic() - last_build_log >= 30 or completed == len(tasks):
                log(f"Built {completed}/{len(tasks)} episodes, {windows} windows ({resumed} resumed)")
                last_build_log = time.monotonic()
        if workers == 0:
            global _WORKER_ARCHIVE, _WORKER_ROOT, _WORKER_BINDING, _WORKER_PROTOCOL, _BLOCK_FRAMES
            import torch
            import pyarrow as pa
            torch.set_num_threads(1)
            pa.set_cpu_count(1)
            pa.set_io_thread_count(1)
            _WORKER_ARCHIVE, _WORKER_ROOT, _WORKER_BINDING = archive, root, binding
            _WORKER_PROTOCOL, _BLOCK_FRAMES = protocol, block_frames
            for task in tasks:
                progress(_write_episode(task), 1)
        else:
            # Spawn drops live Arrow/vision caches through archive serialization;
            # it does not rerun the source constructor in every worker. At most
            # 2*workers episode tasks are in flight.
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_worker, initargs=(archive, str(root), binding, protocol, block_frames)) as pool:
                _execute_episode_tasks(pool, tasks, max_workers=workers,
                                       worker_control=worker_control, progress=progress)
        archive.assert_current()
        descriptors, indices, records_by_split = [], {}, {}
        for record, starts in tasks:
            receipt = root / f"episodes/{record['episode_index']:08d}/receipt.json"
            descriptors.append({"episode_index": record["episode_index"], "split": record["split"],
                                "windows": len(starts), "receipt_path": str(receipt.relative_to(root)),
                                "receipt_sha256": file_sha256(receipt), "source_record_sha256": digest(record)})
        packed = {}
        for split in base["splits"]:
            log(f"Packing split-wide {split} mmap arrays")
            record_ids = [index for index, record in enumerate(records) if record["split"] == split]
            rows = [(sequence, start) for sequence, index in enumerate(record_ids) for start in tasks[index][1]]
            path = root / f"index_{split}.npy"
            np.save(path, np.asarray(rows, dtype=np.int64).reshape(-1, 2), allow_pickle=False)
            records_by_split[split] = record_ids
            indices[split] = {"path": path.name, "windows": len(rows), "sha256": file_sha256(path)}
            offset = 0
            source_descriptors = [item for item in descriptors if item["split"] == split]
            for descriptor in source_descriptors:
                descriptor["row_offset"] = offset
                offset += descriptor["windows"]
            atomic_json(root / "progress.json", {"state": "packing", "split": split,
                        "completed_episodes": len(tasks), "completed_windows": windows,
                        "updated_at_unix": time.time()})
            packed[split] = _pack_split(root, split, binding, indices[split], source_descriptors)
        archive.assert_current()
        manifest = dict(base, state="complete", episodes=descriptors, indices=indices,
                        packed=packed, records_by_split=records_by_split, completed_episodes=len(tasks),
                        completed_windows=windows, completed_at_unix=time.time())
        atomic_json(manifest_path, manifest)
        atomic_json(root / "progress.json", {"state": "complete", "completed_episodes": len(tasks),
                    "completed_windows": windows, "resumed_episodes": resumed, "updated_at_unix": time.time()})
        log(f"Complete: {len(tasks)} episodes, {windows} windows")
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("/gaozt-test1/datasets/MixEgoMotion/training_cache/smplh245_5hz_w11_v1"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worker-control", type=Path,
                        help='Optional JSON {"active_workers": N}; change N without restarting (1..--workers)')
    parser.add_argument("--block-frames", type=int, default=4096)
    parser.add_argument("--diagnostic-episodes", type=int, nargs="+")
    args = parser.parse_args()
    # These process/thread settings apply only to this builder, never live ranks.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from .config import load_config
    cfg = load_config(args.config)
    manifest = build_window_cache(cfg, args.output_root, workers=args.workers,
                selected_episodes=args.diagnostic_episodes, diagnostic=args.diagnostic_episodes is not None,
                block_frames=args.block_frames, worker_control=args.worker_control)
    print(json.dumps({key: manifest[key] for key in ("state", "scope", "completed_episodes", "completed_windows")}))


if __name__ == "__main__":
    main()
