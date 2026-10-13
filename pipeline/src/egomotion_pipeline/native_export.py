"""Read-only source archive to resumable EgoSuite native dense22 sidecars.

Only motion columns are projected from the source Parquet files.  This exporter
never opens a source archive lock for writing and never copies or recomputes
DINO features.  Episode Parquet files and their provenance are committed in
that order; an episode JSON is the commit marker.
"""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import contextlib
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .storage import Archive, atomic_json, digest, read_arrays, read_json, write_arrays


EXPORT_FORMAT = "egomotion-native-dense22-sidecar-export-v1"


def _core():
    from . import native_dense22
    return native_dense22


def _file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _array_hashes(arrays):
    """Hash numeric columns including their names, dtypes and exact shapes."""
    columns = {}
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        header = json.dumps([name, value.dtype.str, list(value.shape)],
                            separators=(",", ":")).encode()
        checksum = hashlib.sha256(header)
        checksum.update(memoryview(value).cast("B"))
        columns[name] = checksum.hexdigest()
    return {"sha256": digest(columns), "columns": columns}


def _file_stat(path):
    value = Path(path).stat()
    return {"size": value.st_size, "mtime_ns": value.st_mtime_ns,
            "inode": value.st_ino}


def _paths(output_root, index):
    root = Path(output_root)
    return (root / f"data/chunk-{index // 1000:03d}/file-{index % 1000:03d}.parquet",
            root / f"episodes/{index:08d}.json")


def _source_path(archive_root, index):
    return Path(archive_root) / f"data/chunk-{index // 1000:03d}/file-{index % 1000:03d}.parquet"


def _read_projected(path, specs):
    core = _core()
    columns = list(core.MOTION_COLUMNS)
    if any("dino" in name.lower() for name in columns):
        raise ValueError("Native export source projection must not contain DINO")
    selected_specs = {name: specs[name] for name in columns}
    # Never call storage.read_arrays on a source file: it reads every column.
    table = pq.read_table(path, columns=columns, use_threads=False)
    if set(table.column_names) != set(columns):
        raise ValueError("Source motion projection differs from the declared columns")
    arrays = {}
    for name, spec in selected_specs.items():
        column = table[name].combine_chunks()
        dtype = np.dtype(spec["dtype"])
        scalar = pa.from_numpy_dtype(dtype)
        expected = scalar if spec["shape"] == [1] else pa.list_(scalar, spec["shape"][0])
        if column.type != expected or column.null_count:
            raise ValueError(f"Source motion dtype/shape/null mismatch: {name}")
        if spec["shape"] == [1]:
            arrays[name] = column.to_numpy(zero_copy_only=False).copy()
        else:
            if column.values.null_count:
                raise ValueError(f"Source motion values contain nulls: {name}")
            arrays[name] = column.values.to_numpy(zero_copy_only=False).reshape(
                len(table), *spec["shape"]).copy()
    return arrays


def _implementation():
    directory = Path(__file__).parent
    files = {name: _file_sha256(directory / name) for name in
             ("native_export.py", "native_dense22.py", "geometry.py", "schema.py", "storage.py")}
    return {"sha256": digest(files), "files": files}


def _inventory(archive_root):
    root = Path(archive_root)
    if not (root / ".pipeline/config.json").is_file() or not (root / "meta/info.json").is_file():
        raise ValueError("Source must be an existing pipeline archive")
    archive = Archive(root)
    all_records = archive.records()
    info = read_json(root / "meta/info.json")
    if info.get("fps") != 30 or info.get("total_episodes") != len(all_records):
        raise ValueError("Source archive inventory/info mismatch")
    if info.get("total_frames") != sum(int(record["length"]) for record in all_records):
        raise ValueError("Source archive frame inventory/info mismatch")
    if len({record["source_key"] for record in all_records}) != len(all_records):
        raise ValueError("Source archive has duplicate source keys")
    if len({int(record["episode_index"]) for record in all_records}) != len(all_records):
        raise ValueError("Source archive has duplicate episode indices")
    records = sorted((record for record in all_records if record["source"] == "egosuite"),
                     key=lambda record: int(record["episode_index"]))
    if not records:
        raise ValueError("Source archive has no EgoSuite records")
    for record in records:
        if not _source_path(root, int(record["episode_index"])).is_file():
            raise ValueError(f"Source Parquet is missing: {record['source_key']}")
        meta = record["metadata"]
        if meta.get("source_format") == "egosuite_mcap":
            # This is an inventory diagnostic, not an inference from filenames.
            if not record.get("input_fingerprint"):
                raise ValueError("MCAP record must preserve its input fingerprint")
    fingerprint = digest({"archive_schema": archive.schema_version,
                          "records": records})
    return archive, records, fingerprint


def _config(archive, source_root):
    core = _core()
    return {"format": EXPORT_FORMAT, "schema_version": core.SCHEMA_VERSION,
            "archive_root": str(Path(source_root).resolve()),
            "archive_schema_version": archive.schema_version, "fps": 30,
            "source": "egosuite", "joint_center_policy": "preserve_source_native",
            "joint_axis_policy": "preserve_source_native_no_right_correction",
            "floor_policy": "preserve_archive_reference_no_window_fallback",
            "rotation_encoding": "matrix_first_two_rows",
            "source_columns": list(core.MOTION_COLUMNS),
            "output_features": core.native_dense22_specs(),
            "dino_policy": "reuse_source_by_episode_and_frame_no_dino_io",
            "numpy_version": np.__version__, "pyarrow_version": pa.__version__}


def _summary(arrays):
    count = len(arrays["timestamp"])
    valid_counts = {name: int(np.asarray(value, dtype=bool).sum())
                    for name, value in arrays.items() if name.startswith("observation.valid.")}
    position = np.asarray(arrays["observation.valid.head_reference_floor_position"]).reshape(count).astype(bool)
    rotation = np.asarray(arrays["observation.valid.head_reference_floor_rotation"]).reshape(count).astype(bool)
    return {"frames": count, "valid_counts": valid_counts,
            "floor_invalid_frames": int((~(position & rotation)).sum()),
            "floor_rotation_invalid_frames": int((~rotation).sum()),
            "floor_position_invalid_frames": int((~position).sum())}


def _check_source_join(arrays, record):
    """Bind each projected row to the original archive/DINO frame identity."""
    count = int(record["length"])
    start, stop = int(record["dataset_from_index"]), int(record["dataset_to_index"])
    if stop - start != count:
        raise ValueError("Source record dataset index range differs from its length")
    expected = {"frame_index": np.arange(count, dtype=np.int64),
                "index": start + np.arange(count, dtype=np.int64),
                "episode_index": np.full(count, int(record["episode_index"]), dtype=np.int64),
                "task_index": np.zeros(count, dtype=np.int64)}
    for name, values in expected.items():
        if not np.array_equal(np.asarray(arrays[name]).reshape(-1), values):
            raise ValueError(f"Source {name} does not match its episode/frame join")
    timestamp = np.asarray(arrays["timestamp"]).reshape(-1)
    if not np.array_equal(timestamp, (np.arange(count) / 30).astype(np.float32)):
        raise ValueError("Source timestamp does not match its native 30Hz frame grid")


def _check_commit(record, committed, signature, source_hash, output_path):
    if committed.get("source_record_sha256") != digest(record):
        raise ValueError("Committed source record fingerprint changed; use a new derived output")
    if committed.get("source_input_fingerprint") != record.get("input_fingerprint"):
        raise ValueError("Committed source input fingerprint changed")
    if committed.get("signature_sha256") != signature:
        raise ValueError("Committed derived config/code signature differs")
    if committed.get("source_motion_sha256") != source_hash["sha256"]:
        raise ValueError("Source motion changed after the derived episode was committed")
    if not output_path.is_file():
        raise ValueError("Committed derived Parquet is missing")
    if committed.get("parquet_file_sha256") != _file_sha256(output_path):
        raise ValueError("Committed derived Parquet was changed or corrupted")


def _process_episode(task):
    """Commit or fully validate one episode; workers write distinct paths only."""
    source_root, output_root, record, source_specs, signature, resume, validate_only = task
    core = _core()
    index = int(record["episode_index"])
    source_path = _source_path(source_root, index)
    output_path, record_path = _paths(output_root, index)
    started = time.monotonic()
    before_stat = _file_stat(source_path)
    source_arrays = _read_projected(source_path, source_specs)
    source_hash = _array_hashes(source_arrays)
    if len(source_arrays["timestamp"]) != int(record["length"]):
        raise ValueError("Source projected frame count differs from its episode record")
    _check_source_join(source_arrays, record)
    committed = read_json(record_path) if record_path.is_file() else None
    if committed is not None:
        if not resume and not validate_only:
            raise ValueError("Episode already committed; resume=False forbids overwrite")
        _check_commit(record, committed, signature, source_hash, output_path)
        output = read_arrays(output_path, core.native_dense22_specs())
        report = core.validate_native_dense22(source_arrays, output, record,
                                             episode_metadata=committed["derived_metadata"])
        if _array_hashes(output)["sha256"] != committed.get("output_motion_sha256"):
            raise ValueError("Committed output numeric fingerprint mismatch")
        result_kind = "validated" if validate_only else "resumed"
    else:
        if validate_only:
            raise ValueError("Derived episode is not committed")
        output, derived_metadata = core.encode_native_dense22(source_arrays, record)
        if _array_hashes(source_arrays) != source_hash:
            raise ValueError("Native encoder mutated source input arrays")
        report = core.validate_native_dense22(source_arrays, output, record,
                                             episode_metadata=derived_metadata)
        output_hash = _array_hashes(output)
        write_arrays(output_path, output, core.native_dense22_specs())
        # Decode the on-disk file and check every field, rather than trusting the
        # in-memory validation before Arrow serialization.
        disk_output = read_arrays(output_path, core.native_dense22_specs())
        report = core.validate_native_dense22(source_arrays, disk_output, record,
                                             episode_metadata=derived_metadata)
        if _array_hashes(disk_output) != output_hash:
            raise ValueError("Derived Parquet serialization changed numeric values")
        committed = {"format": EXPORT_FORMAT, "schema_version": core.SCHEMA_VERSION,
                     "episode_index": index, "source": record["source"],
                     "source_key": record["source_key"], "source_uid": record["source_uid"],
                     "length": int(record["length"]),
                     "source_record_sha256": digest(record),
                     "source_input_fingerprint": record.get("input_fingerprint"),
                     "source_motion_sha256": source_hash["sha256"],
                     "source_motion_column_sha256": source_hash["columns"],
                     "output_motion_sha256": output_hash["sha256"],
                     "output_motion_column_sha256": output_hash["columns"],
                     "parquet_file_sha256": _file_sha256(output_path),
                     "signature_sha256": signature,
                     "source_data_path": str(source_path.resolve()),
                     "source_file_stat": before_stat,
                     "data_path": str(output_path.relative_to(output_root)),
                     "derived_metadata": derived_metadata,
                     "validation": report, "summary": _summary(disk_output)}
        result_kind = "exported"
        output = disk_output
    # A changing source file invalidates this result even if a DINO-only rewrite
    # caused the stat change.  Motion is compared again without projecting DINO.
    if _file_stat(source_path) != before_stat:
        raise ValueError("Source Parquet changed while this episode was processed")
    if _array_hashes(_read_projected(source_path, source_specs)) != source_hash:
        raise ValueError("Source motion changed while this episode was processed")
    if digest(read_json(Path(source_root) / f".pipeline/episodes/{index:08d}.json")) != digest(record):
        raise ValueError("Source episode metadata changed while processing")
    if result_kind == "exported":
        # Commit metadata only after all source/output checks have succeeded.
        atomic_json(record_path, committed)
    return {"kind": result_kind, "episode_index": index,
            "source_key": record["source_key"], "summary": _summary(output),
            "source_motion_sha256": source_hash["sha256"],
            "output_motion_sha256": committed["output_motion_sha256"],
            "validation": report, "elapsed_s": time.monotonic() - started}


def _worker_init():
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass


def _execute(tasks, workers):
    if workers == 1:
        for task in tasks:
            try:
                yield _process_episode(task)
            except Exception as error:
                yield {"kind": "error", "episode_index": int(task[2]["episode_index"]),
                       "source_key": task[2]["source_key"],
                       "error": f"{type(error).__name__}: {error}",
                       "traceback": traceback.format_exc()}
        return
    # Keep the pending queue bounded.  Workers return only compact metadata;
    # large episode arrays never travel through parent-process IPC.
    iterator = iter(tasks)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=_worker_init) as executor:
        pending = {}
        def submit_one():
            try:
                task = next(iterator)
            except StopIteration:
                return False
            pending[executor.submit(_process_episode, task)] = task
            return True
        for _ in range(workers * 2):
            if not submit_one():
                break
        while pending:
            done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            if not done:
                yield {"kind": "heartbeat"}
                continue
            for future in done:
                task = pending.pop(future)
                try:
                    yield future.result()
                except Exception as error:
                    yield {"kind": "error", "episode_index": int(task[2]["episode_index"]),
                           "source_key": task[2]["source_key"],
                           "error": f"{type(error).__name__}: {error}",
                           "traceback": traceback.format_exc()}
                submit_one()


@contextlib.contextmanager
def _output_lock(output_root):
    with (Path(output_root) / ".export.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another native export/validation process holds this output lock") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextlib.contextmanager
def _source_read_lock(archive_root):
    """Cooperate with pipeline writers using the existing lock, read-only."""
    lock_path = Path(archive_root) / ".pipeline/lock"
    if not lock_path.is_file():
        raise ValueError("Source archive must have its existing .pipeline/lock for read-only export")
    with lock_path.open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("An active source archive writer prevents native export/validation") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _safe_roots(archive_root, output_root):
    source = Path(archive_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if source == output or output in source.parents:
        raise ValueError("Derived output must not replace or contain the source archive")
    if output.is_relative_to(source) and output.relative_to(source).parts[0] != "derived":
        raise ValueError("Inside the source archive, sidecars must live under derived/")
    return source, output


def _manifest(source, archive, records, inventory_sha, implementation):
    config = _config(archive, source)
    config_sha = digest(config)
    signature = digest({"config_sha256": config_sha,
                        "implementation_sha256": implementation["sha256"]})
    return {"format": EXPORT_FORMAT, "schema_version": _core().SCHEMA_VERSION,
            "config": config, "config_sha256": config_sha,
            "implementation": implementation, "signature_sha256": signature,
            "source_inventory_sha256": inventory_sha,
            "source_inventory": [{"episode_index": int(record["episode_index"]),
                                  "source_key": record["source_key"],
                                  "length": int(record["length"]),
                                  "source_record_sha256": digest(record),
                                  "source_input_fingerprint": record.get("input_fingerprint")}
                                 for record in records],
            "total_episodes": len(records),
            "total_frames": sum(int(record["length"]) for record in records)}


def _aggregate(results):
    success = [value for value in results if value["kind"] != "error"]
    counts = {}
    for value in success:
        for name, count in value["summary"]["valid_counts"].items():
            counts[name] = counts.get(name, 0) + count
    maxima = {}
    for frame in ("world", "floor"):
        maxima[frame] = {}
        for value in success:
            for key, number in value["validation"].get(frame, {}).items():
                if "max" in key:
                    maxima[frame][key] = max(maxima[frame].get(key, 0.), number)
    return {"validated_episodes": len(success),
            "validated_frames": sum(value["summary"]["frames"] for value in success),
            "valid_counts": counts,
            "floor_invalid_frames": sum(value["summary"]["floor_invalid_frames"] for value in success),
            "floor_rotation_invalid_frames": sum(value["summary"]["floor_rotation_invalid_frames"] for value in success),
            "floor_position_invalid_frames": sum(value["summary"]["floor_position_invalid_frames"] for value in success),
            "floor_invalid_episodes": sum(value["summary"]["floor_invalid_frames"] > 0 for value in success),
            "maximum_encoding_and_roundtrip_errors": maxima,
            "episode_fingerprints": {str(value["episode_index"]): {
                "source_motion_sha256": value["source_motion_sha256"],
                "output_motion_sha256": value["output_motion_sha256"]} for value in success}}


def _run_export(archive_root, output_root, *, workers=4, limit=None, resume=True):
    """Export native dense22 sidecars without mutating source motion or DINO.

    A limit is a smoke-run prefix of the full manifest inventory and produces
    stage=partial; remove it to resume into the same output.  Source inventory,
    config and implementation must match exactly.  Committed data corruption
    is reported as failure and is never silently rebuilt.  The return value
    includes failed_episodes; a CLI must return nonzero if it is nonzero.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    if type(resume) is not bool:
        raise ValueError("resume must be a boolean")
    source, output = _safe_roots(archive_root, output_root)
    archive, records, inventory_sha = _inventory(source)
    implementation = _implementation()
    expected_manifest = _manifest(source, archive, records, inventory_sha, implementation)
    output.mkdir(parents=True, exist_ok=True)
    with _output_lock(output):
        manifest_path = output / "manifest.json"
        selected = records if limit is None else records[:limit]
        if limit is not None:
            selected_indices = {int(record["episode_index"]) for record in selected}
            for committed_path in sorted((output / "episodes").glob("*.json")):
                committed = read_json(committed_path)
                index = committed.get("episode_index")
                if type(index) is not int or committed_path.name != f"{index:08d}.json":
                    raise ValueError("Invalid committed episode identity in derived output")
                if index not in selected_indices:
                    # Check before touching any run/manifest/summary metadata.
                    # A smaller prefix must not hide already committed data.
                    raise ValueError(
                        f"limit excludes already committed episode {index}; "
                        "resume the complete scope or choose a separate pilot output")
        if manifest_path.exists():
            if not resume:
                raise ValueError("Output already initialized; resume=False forbids overwrite")
            if read_json(manifest_path) != expected_manifest:
                raise ValueError("Derived manifest source inventory/config/code mismatch; use a new output")
        else:
            if any(path.name != ".export.lock" for path in output.iterdir()):
                raise ValueError("Refusing a non-empty unrecognized derived output")
            atomic_json(manifest_path, expected_manifest)
        started = time.monotonic()
        results = []
        state = {"format": EXPORT_FORMAT, "schema_version": _core().SCHEMA_VERSION,
                 "stage": "running", "pid": os.getpid(), "workers": workers,
                 "total_episodes": len(records), "total_frames": expected_manifest["total_frames"],
                 "selected_episodes": len(selected), "processed_episodes": 0,
                 "skipped_episodes": 0, "failed_episodes": 0,
                 "committed_episodes": 0, "committed_frames": 0,
                 "floor_invalid_frames": 0, "failures": [], "elapsed_s": 0.}
        def emit(value):
            event = {"time_unix_s": time.time(), **value}
            with (output / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            if value["kind"] not in {"heartbeat", "start", "finish"}:
                if value["kind"] == "error":
                    print(f"NATIVE ERROR episode={value['episode_index']} {value['error']}", flush=True)
                elif value["kind"] in {"exported", "resumed", "validated"}:
                    summary = value["summary"]
                    print(f"NATIVE {value['kind']} episode={value['episode_index']} "
                          f"frames={summary['frames']} floor_invalid={summary['floor_invalid_frames']} "
                          f"mask_counts={json.dumps(summary['valid_counts'], sort_keys=True)}", flush=True)
        atomic_json(output / "status.json", state)
        emit({"kind": "start", "selected_episodes": len(selected),
              "total_episodes": len(records), "signature_sha256": expected_manifest["signature_sha256"]})
        tasks = [(str(source), str(output), record, archive.specs,
                  expected_manifest["signature_sha256"], resume, False) for record in selected]
        try:
            for value in _execute(tasks, workers):
                emit(value)
                state["elapsed_s"] = time.monotonic() - started
                if value["kind"] == "heartbeat":
                    atomic_json(output / "status.json", state)
                    continue
                results.append(value)
                if value["kind"] == "error":
                    state["failed_episodes"] += 1
                    state["failures"].append({key: value[key] for key in
                                              ("episode_index", "source_key", "error")})
                else:
                    state["processed_episodes" if value["kind"] == "exported" else "skipped_episodes"] += 1
                    state["committed_episodes"] += 1
                    state["committed_frames"] += value["summary"]["frames"]
                    state["floor_invalid_frames"] += value["summary"]["floor_invalid_frames"]
                atomic_json(output / "status.json", state)
            _, current_records, current_inventory = _inventory(source)
            source_unchanged = current_inventory == inventory_sha and current_records == records
            code_unchanged = _implementation() == implementation
            if not source_unchanged or not code_unchanged:
                raise ValueError("Source inventory or implementation changed during export")
            state["stage"] = "failed" if state["failed_episodes"] else (
                "completed" if len(selected) == len(records) else "partial")
            state["elapsed_s"] = time.monotonic() - started
            aggregate = _aggregate(results)
            validation = {"format": EXPORT_FORMAT, "stage": state["stage"],
                          **aggregate, "failures": state["failures"],
                          "source_inventory_unchanged": source_unchanged,
                          "code_unchanged": code_unchanged,
                          "source_inventory_sha256": inventory_sha,
                          "signature_sha256": expected_manifest["signature_sha256"],
                          "original_archive_written": False, "dino_columns_read": False}
            atomic_json(output / "validation.json", validation)
            info = {"format": EXPORT_FORMAT, "schema_version": _core().SCHEMA_VERSION,
                    "fps": 30, "total_episodes": state["committed_episodes"],
                    "total_frames": state["committed_frames"], "stage": state["stage"],
                    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                    "episode_path": "episodes/{episode_index:08d}.json",
                    "features": _core().native_dense22_specs(),
                    "source_archive": str(source), "dino_storage": "reference_source_archive",
                    "signature_sha256": expected_manifest["signature_sha256"]}
            atomic_json(output / "meta/info.json", info)
            atomic_json(output / "status.json", state)
            emit({"kind": "finish", "stage": state["stage"],
                  "committed_episodes": state["committed_episodes"],
                  "committed_frames": state["committed_frames"], "elapsed_s": state["elapsed_s"]})
            print(f"NATIVE {state['stage']} episodes={state['committed_episodes']}/{len(records)} "
                  f"frames={state['committed_frames']} failures={state['failed_episodes']} "
                  f"elapsed_s={state['elapsed_s']:.3f}", flush=True)
            return state
        except BaseException as error:
            state["stage"] = "failed"
            state["fatal_error"] = f"{type(error).__name__}: {error}"
            state["elapsed_s"] = time.monotonic() - started
            atomic_json(output / "status.json", state)
            emit({"kind": "fatal", "error": state["fatal_error"]})
            raise


def _validate_export(archive_root, output_root, *, workers=4):
    """Independently reread and revalidate every sidecar and source motion column."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    source, output = _safe_roots(archive_root, output_root)
    archive, records, inventory_sha = _inventory(source)
    implementation = _implementation()
    manifest = _manifest(source, archive, records, inventory_sha, implementation)
    if not output.is_dir() or not (output / "manifest.json").is_file():
        raise ValueError("No initialized native export to validate")
    with _output_lock(output):
        if read_json(output / "manifest.json") != manifest:
            raise ValueError("Derived manifest source inventory/config/code mismatch")
        tasks = [(str(source), str(output), record, archive.specs,
                  manifest["signature_sha256"], True, True) for record in records]
        started = time.monotonic()
        progress = {"format": EXPORT_FORMAT, "stage": "running", "pid": os.getpid(),
                    "workers": workers, "total_episodes": len(records),
                    "total_frames": manifest["total_frames"], "checked_episodes": 0,
                    "validated_episodes": 0, "validated_frames": 0,
                    "failed_episodes": 0, "elapsed_s": 0.}
        atomic_json(output / "validation_status.json", progress)
        results = []
        try:
            for value in _execute(tasks, workers):
                progress["elapsed_s"] = time.monotonic() - started
                if value["kind"] != "heartbeat":
                    results.append(value)
                    progress["checked_episodes"] += 1
                    if value["kind"] == "error":
                        progress["failed_episodes"] += 1
                        print(f"NATIVE VERIFY ERROR episode={value['episode_index']} {value['error']}", flush=True)
                    else:
                        progress["validated_episodes"] += 1
                        progress["validated_frames"] += value["summary"]["frames"]
                if value["kind"] == "heartbeat" or progress["checked_episodes"] % 25 == 0:
                    atomic_json(output / "validation_status.json", progress)
                    print(f"NATIVE VERIFY episodes={progress['checked_episodes']}/{len(records)} "
                          f"frames={progress['validated_frames']} failures={progress['failed_episodes']} "
                          f"elapsed_s={progress['elapsed_s']:.3f}", flush=True)
        except BaseException as error:
            progress.update(stage="failed", fatal_error=f"{type(error).__name__}: {error}",
                            elapsed_s=time.monotonic() - started)
            atomic_json(output / "validation_status.json", progress)
            raise
        failures = [{key: value[key] for key in ("episode_index", "source_key", "error")}
                    for value in results if value["kind"] == "error"]
        _, current_records, current_inventory = _inventory(source)
        source_unchanged = current_inventory == inventory_sha and current_records == records
        code_unchanged = _implementation() == implementation
        if not source_unchanged or not code_unchanged:
            failures.append({"episode_index": None, "source_key": None,
                             "error": "Source inventory or implementation changed during validation"})
        result = {"format": EXPORT_FORMAT, "stage": "failed" if failures else "completed",
                  **_aggregate(results), "failures": failures,
                  "source_inventory_unchanged": source_unchanged, "code_unchanged": code_unchanged,
                  "source_inventory_sha256": inventory_sha,
                  "signature_sha256": manifest["signature_sha256"],
                  "original_archive_written": False, "dino_columns_read": False}
        atomic_json(output / "validation.json", result)
        progress.update(stage=result["stage"], elapsed_s=time.monotonic() - started,
                        failures=failures)
        atomic_json(output / "validation_status.json", progress)
        print(f"NATIVE VERIFY {result['stage']} episodes={progress['checked_episodes']}/{len(records)} "
              f"frames={progress['validated_frames']} failures={len(failures)} "
              f"elapsed_s={progress['elapsed_s']:.3f}", flush=True)
        return result


def run_export(archive_root, output_root, *, workers=4, limit=None, resume=True):
    """Export with a source shared read lock and a derived exclusive write lock.

    See ``_run_export`` for resume rules and the returned status.  The existing
    source lock file is opened read-only; no source file is created or changed.
    """
    source, output = _safe_roots(archive_root, output_root)
    with _source_read_lock(source):
        return _run_export(source, output, workers=workers, limit=limit, resume=resume)


def validate_export(archive_root, output_root, *, workers=4):
    """Fully revalidate sidecars while excluding source archive writers."""
    source, output = _safe_roots(archive_root, output_root)
    with _source_read_lock(source):
        return _validate_export(source, output, workers=workers)
