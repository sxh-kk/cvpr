"""Resumable v4→v5 migration with validated staging and a retained v4 snapshot.

No encoder or source MCAP is opened. Every old numeric column is copied and
checked independently after serialization, including the completed DINO tokens.
"""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
import traceback
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .numeric_fingerprint import NumericColumnHasher, motion_digest_from_hashes
from .storage import Archive, archive_provenance, atomic_json, digest, read_json


FORMAT = "egomotion-unified-dense22-migration-v1"
JOURNAL = ".pipeline/dense_v5_upgrade.json"
BATCH_SIZE = 3000


def _now():
    return datetime.now(timezone.utc).isoformat()


def _stat(path):
    value = Path(path).stat()
    return {"size": value.st_size, "mtime_ns": value.st_mtime_ns, "inode": value.st_ino}


def _data_path(root, index):
    return Path(root) / f"data/chunk-{index // 1000:03d}/file-{index % 1000:03d}.parquet"


def _receipt_path(job, index):
    return Path(job) / f"receipts/{index:08d}.json"


def _batch_arrays(batch, specs):
    result = {}
    if set(batch.schema.names) != set(specs):
        raise ValueError("Migration Parquet/schema columns differ")
    for name, spec in specs.items():
        value = batch.column(batch.schema.get_field_index(name))
        scalar = pa.from_numpy_dtype(np.dtype(spec["dtype"]))
        expected = scalar if spec["shape"] == [1] else pa.list_(scalar, spec["shape"][0])
        if value.type != expected or value.null_count:
            raise ValueError(f"Migration dtype/shape/null mismatch: {name}")
        if spec["shape"] == [1]:
            result[name] = value.to_numpy(zero_copy_only=False)
        else:
            if value.values.null_count:
                raise ValueError(f"Migration nested null values: {name}")
            result[name] = value.values.to_numpy(zero_copy_only=False).reshape(len(batch), *spec["shape"])
    return result


def _new_columns_table(arrays, specs):
    columns = {}
    for name, spec in specs.items():
        value = np.asarray(arrays[name])
        if value.dtype != np.dtype(spec["dtype"]):
            raise ValueError(f"New feature dtype mismatch: {name}")
        if spec["shape"] == [1]:
            columns[name] = pa.array(value.reshape(-1))
        else:
            columns[name] = pa.FixedSizeListArray.from_arrays(pa.array(value.reshape(-1)), spec["shape"][0])
    return pa.table(columns)


def _check_batch_join(arrays, record, offset):
    count = len(arrays["timestamp"])
    frame = np.arange(offset, offset + count, dtype=np.int64)
    expected = {"frame_index": frame, "index": int(record["dataset_from_index"]) + frame,
                "episode_index": np.full(count, int(record["episode_index"]), np.int64),
                "task_index": np.zeros(count, np.int64)}
    for name, value in expected.items():
        if not np.array_equal(arrays[name], value):
            raise ValueError(f"Migration source frame identity mismatch: {name}")
    if not np.array_equal(arrays["timestamp"], (frame / 30).astype(np.float32)):
        raise ValueError("Migration source timestamp is not the archived 30Hz grid")


def _native_columns(native_root, index, additions):
    if native_root is None:
        return None
    path = _data_path(native_root, index)
    if not path.is_file():
        raise ValueError(f"Native dense22 episode is missing: {index}")
    specs = {name: spec for name, spec in additions.items() if "body_" in name}
    table = pq.read_table(path, columns=list(specs), use_threads=False)
    table = table.combine_chunks()
    return _batch_arrays(table.to_batches(max_chunksize=max(len(table), 1))[0], specs)


def _stage_episode(task):
    """Write and independently validate one file; workers own disjoint paths."""
    from .schema import SCHEMA_VERSION, feature_specs_for_schema
    from .unified_dense22 import encode_unified_dense22, validate_unified_dense22, unified_specs

    source_root, job, record, source_version, signature, expected_stat, native_root = task
    index, length = int(record["episode_index"]), int(record["length"])
    source_path, target_path = _data_path(source_root, index), _data_path(job, index)
    receipt_path = _receipt_path(job, index)
    if _stat(source_path) != expected_stat:
        raise ValueError(f"Source Parquet changed since inventory: {index}")
    record_sha = digest(record)
    if digest(read_json(Path(source_root) / f".pipeline/episodes/{index:08d}.json")) != record_sha:
        raise ValueError(f"Source episode provenance changed since inventory: {index}")
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if (receipt.get("source_record_sha256") != record_sha
                or receipt.get("producer_signature") != signature
                or receipt.get("source_file_stat") != expected_stat
                or not target_path.is_file()
                or receipt.get("staged_file_stat") != _stat(target_path)):
            raise ValueError(f"Committed migration staging receipt mismatch: {index}")
        return {"kind": "resumed", "receipt": receipt}

    source_specs, target_specs = feature_specs_for_schema(source_version), feature_specs_for_schema(SCHEMA_VERSION)
    additions = unified_specs()
    native = _native_columns(native_root, index, additions)
    if native is not None and len(next(iter(native.values()))) != length:
        raise ValueError(f"Native dense22 length mismatch: {index}")
    source = pq.ParquetFile(source_path)
    if set(source.schema_arrow.names) != set(source_specs) or source.metadata.num_rows != length:
        raise ValueError(f"Source Parquet inventory/schema mismatch: {index}")
    source_hasher = NumericColumnHasher(source_specs, length)
    expected_hasher = NumericColumnHasher(target_specs, length)
    valid_counts = {name: 0 for name in additions if name.startswith("observation.valid.")}
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = target_path.with_suffix(".parquet.tmp")
    writer = None
    offset = 0
    parity_error = 0.
    started = time.monotonic()
    try:
        for batch in source.iter_batches(batch_size=BATCH_SIZE, use_threads=False):
            arrays = _batch_arrays(batch, source_specs)
            _check_batch_join(arrays, record, offset)
            new = encode_unified_dense22(arrays, record)
            validate_unified_dense22(arrays, new, record)
            if native is not None:
                for name, original in native.items():
                    reference = original[offset:offset + len(batch)]
                    if reference.dtype == np.bool_:
                        if not np.array_equal(reference, new[name]):
                            raise ValueError(f"Native dense22 mask mismatch: {index}: {name}")
                    else:
                        error = float(np.max(np.abs(reference - new[name]))) if reference.size else 0.
                        parity_error = max(parity_error, error)
                        if error > 2e-5:
                            raise ValueError(f"Native dense22 value mismatch: {index}: {name}: {error}")
            for name in valid_counts:
                valid_counts[name] += int(new[name].sum())
            source_hasher.update(arrays)
            combined = {**arrays, **new}
            expected_hasher.update(combined)
            table = pa.Table.from_batches([batch])
            extra = _new_columns_table(new, additions)
            for name in additions:
                table = table.append_column(name, extra[name])
            table = table.select(list(target_specs))
            if writer is None:
                feature = "observation.features.dino"
                leaf = feature + ".list.element"
                dictionary = [name + (".list.element" if spec["shape"] != [1] else "")
                              for name, spec in target_specs.items() if name != feature]
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd", use_dictionary=dictionary,
                                         use_byte_stream_split=[leaf], compression_level={leaf: 1})
            writer.write_table(table, row_group_size=BATCH_SIZE)
            offset += len(batch)
        if writer is None:
            raise ValueError("Cannot migrate an empty episode")
        writer.close()
        writer = None
        if offset != length:
            raise ValueError("Source migration length changed")
        original_hashes, expected_hashes = source_hasher.finish(), expected_hasher.finish()
        # A separate decoder checks every old/new numeric value on disk. This
        # also verifies DINO without loading an encoder or modifying its status.
        disk = pq.ParquetFile(tmp)
        if disk.metadata.num_rows != length:
            raise ValueError("Staged Parquet row count differs")
        actual_hasher = NumericColumnHasher(target_specs, length)
        batches_validated = 0
        for batch in disk.iter_batches(batch_size=BATCH_SIZE, use_threads=False):
            actual = _batch_arrays(batch, target_specs)
            actual_hasher.update(actual)
            validate_unified_dense22({name: actual[name] for name in source_specs},
                                     {name: actual[name] for name in additions}, record)
            batches_validated += 1
        actual_hashes = actual_hasher.finish()
        if actual_hashes != expected_hashes:
            raise ValueError(f"Parquet serialization changed numeric columns: {index}")
        for name, original in original_hashes["columns"].items():
            if actual_hashes["columns"].get(name) != original:
                raise ValueError(f"Old numeric column changed during migration: {index}: {name}")
        if _stat(source_path) != expected_stat:
            raise ValueError(f"Source Parquet changed during migration: {index}")
        os.replace(tmp, target_path)
        receipt = {"format": FORMAT, "episode_index": index, "source_key": record["source_key"],
                   "length": length, "source_record_sha256": record_sha,
                   "source_input_fingerprint": record.get("input_fingerprint"),
                   "source_file_stat": expected_stat, "staged_file_stat": _stat(target_path),
                   "source_numeric_sha256": original_hashes["sha256"],
                   "source_column_sha256": original_hashes["columns"],
                   "output_numeric_sha256": actual_hashes["sha256"],
                   "output_column_sha256": actual_hashes["columns"],
                   "motion_digest": motion_digest_from_hashes(actual_hashes),
                   "producer_signature": signature, "source_schema": source_version,
                   "target_schema": SCHEMA_VERSION, "valid_counts": valid_counts,
                   "dino_policy": "all_numeric_values_and_masks_preserved_no_encoder",
                   "native_parity_max_abs": parity_error if native is not None else None,
                   "validation": {"passed": True, "independent_disk_batches": batches_validated,
                                  "all_old_columns_exact": True, "new_coordinates_and_masks": True},
                   "elapsed_seconds": time.monotonic() - started, "completed_at": _now()}
        atomic_json(receipt_path, receipt)
        return {"kind": "staged", "receipt": receipt}
    finally:
        if writer is not None:
            writer.close()
        if tmp.exists():
            tmp.unlink()


def _load_source_records(job):
    return [read_json(path) for path in sorted((Path(job) / "backup/episodes").glob("*.json"))]


def _receipt_inventory(job, records, signature):
    hashes, receipts = {}, []
    for record in records:
        index = int(record["episode_index"])
        receipt = read_json(_receipt_path(job, index))
        if (receipt.get("source_record_sha256") != digest(record)
                or receipt.get("producer_signature") != signature
                or receipt.get("length") != int(record["length"])
                or not receipt.get("validation", {}).get("passed")):
            raise ValueError(f"Invalid staged receipt: {index}")
        hashes[str(index)] = digest(receipt)
        receipts.append(receipt)
    return receipts, digest(hashes)


def _commit_v5(archive, journal, journal_path):
    """Resume safely across either data-directory rename and metadata publication."""
    from .schema import SCHEMA_VERSION, feature_specs_for_schema
    from .unified_dense22 import unified_metadata

    root = archive.root
    job = root / journal["job_relative_path"]
    records = _load_source_records(job)
    receipts, receipt_sha = _receipt_inventory(job, records, journal["producer_signature"])
    if receipt_sha != journal.get("receipts_sha256"):
        raise ValueError("Validated migration receipts changed before publication")
    backup_data, staged_data, live_data = job / "backup/data", job / "data", root / "data"
    journal.update(state="committing", phase="committing", last_error=None)
    atomic_json(journal_path, journal)
    if not backup_data.exists():
        if not live_data.is_dir() or not staged_data.is_dir():
            raise ValueError("Migration data directories cannot be committed safely")
        for receipt in receipts:
            idx = receipt["episode_index"]
            if _stat(_data_path(root, idx)) != receipt["source_file_stat"]:
                raise ValueError(f"Live v4 file changed before publication: {idx}")
            if _stat(_data_path(job, idx)) != receipt["staged_file_stat"]:
                raise ValueError(f"Staged v5 file changed before publication: {idx}")
        # Kept until the user deliberately removes a verified rollback version.
        os.replace(live_data, backup_data)
    if staged_data.exists():
        if live_data.exists():
            raise ValueError("Unexpected live data alongside staged v5 and v4 backup")
        os.replace(staged_data, live_data)
    elif not live_data.is_dir():
        raise ValueError("Both live and staged v5 data are missing")
    # Validate physical directory identities on every resume, including a crash
    # immediately after os.replace but before the journal could be updated.
    for receipt in receipts:
        idx = receipt["episode_index"]
        if (_stat(_data_path(root, idx)) != receipt["staged_file_stat"]
                or _stat(_data_path(job / "backup", idx)) != receipt["source_file_stat"]):
            raise ValueError(f"Published/backup file does not match its verified receipt: {idx}")
    for record, receipt in zip(records, receipts):
        updated = deepcopy(record)
        updated["motion_representation"] = {"schema_version": SCHEMA_VERSION,
                                              "producer_signature": journal["producer_signature"],
                                              "motion_digest": receipt["motion_digest"]}
        updated["metadata"]["unified_dense22"] = unified_metadata(record)
        for name, count in receipt["valid_counts"].items():
            width = feature_specs_for_schema(SCHEMA_VERSION)[name]["shape"][0]
            updated["quality"][name.removeprefix("observation.valid.")] = count / (record["length"] * width)
        updated["unified_migration"] = {"format": FORMAT,
                                          "job_relative_path": journal["job_relative_path"],
                                          "source_record_sha256": receipt["source_record_sha256"],
                                          "source_numeric_sha256": receipt["source_numeric_sha256"],
                                          "output_numeric_sha256": receipt["output_numeric_sha256"]}
        atomic_json(root / f".pipeline/episodes/{record['episode_index']:08d}.json", updated)
    original = read_json(job / "backup/config.json")
    updated_config = archive_provenance(original["config"])
    updated_config["previous_archive_provenance"] = original
    if "motion_conversion_provenance" in original:
        updated_config["motion_conversion_provenance"] = original["motion_conversion_provenance"]
    updated_config["dense_v5_migration"] = {"format": FORMAT,
                                             "job_relative_path": journal["job_relative_path"],
                                             "source_schema": journal["source_schema"],
                                             "producer_signature": journal["producer_signature"],
                                             "receipts_sha256": receipt_sha,
                                             "native_sidecar_policy": "historical_v4_artifact_no_training_dependency",
                                             "motion_policy": "preserve_world_joint_centers_and_native_axes",
                                             "dino_policy": "preserve_values_masks_signature_and_complete_status"}
    atomic_json(root / ".pipeline/config.json", updated_config)
    archive.schema_version = SCHEMA_VERSION
    archive.specs = feature_specs_for_schema(SCHEMA_VERSION)
    archive._records = None
    archive._by_key = {}
    archive.finalize()
    info = read_json(root / "meta/info.json")
    if (info["total_episodes"] != len(records)
            or info["total_frames"] != sum(record["length"] for record in records)
            or info["features"] != archive.specs):
        raise ValueError("Published unified metadata failed inventory validation")
    journal.update(state="complete", phase="complete", completed_at=_now(),
                   completed_episodes=len(records), completed_frames=info["total_frames"],
                   rollback_relative_path=str((job / "backup").relative_to(root)))
    atomic_json(journal_path, journal)
    atomic_json(job / "status.json", journal)
    return dict(journal)


def upgrade_dense_v5(root, *, workers=4, pilot_indices=None, native_root=None,
                     progress=None, commit=True, job_dir=None):
    """Stage/verify all v4 files before publication; pilots never alter live data."""
    from .schema import SCHEMA_VERSION, V4_SCHEMA_VERSION
    from .unified_dense22 import producer_signature

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    root = Path(root).resolve()
    archive = Archive(root, _allow_upgrade=True)
    journal_path = root / JOURNAL
    journal = read_json(journal_path) if journal_path.exists() else None
    if archive.schema_version == SCHEMA_VERSION and (not journal or journal.get("state") == "complete"):
        return {"state": "complete", "already_current": True, "schema_version": SCHEMA_VERSION}
    if pilot_indices is not None and commit:
        raise ValueError("A subset pilot cannot publish a dataset; use commit=False")
    if native_root is not None:
        native_root = str(Path(native_root).resolve())
        native_info = read_json(Path(native_root) / "meta/info.json")
        if native_info.get("fps") != 30:
            raise ValueError("Native sidecar must retain the 30Hz source grid")
    signature = producer_signature()
    with archive.lock():
        # Resume publication without treating already renamed v5 files as v4
        # input. The source inventory always lives in the retained snapshot.
        if journal and journal.get("phase") == "committing":
            if journal.get("producer_signature") != signature:
                raise ValueError("Migration producer changed; cannot resume publication")
            result = _commit_v5(archive, journal, journal_path)
            if progress:
                progress(result)
            return result
        if archive.schema_version != V4_SCHEMA_VERSION:
            raise ValueError("Unified migration requires a v4 archive; upgrade pending v3 DINO first")
        if journal and journal.get("state") != "complete" and pilot_indices is not None:
            raise ValueError("Resume the active full migration before starting another pilot")
        if journal and journal.get("state") != "complete":
            if (journal.get("format") != FORMAT or journal.get("producer_signature") != signature
                    or journal.get("target_schema") != SCHEMA_VERSION
                    or journal.get("native_root") != native_root):
                raise ValueError("Migration config/producer changed; resume using the original settings")
            job = root / journal["job_relative_path"]
            records = _load_source_records(job)
            inventory = read_json(job / "inventory.json")
            if digest(inventory) != journal.get("source_inventory_sha256"):
                raise ValueError("Migration source inventory changed")
            if read_json(root / ".pipeline/config.json") != read_json(job / "backup/config.json"):
                raise ValueError("Source archive provenance changed during staging")
        else:
            all_records = archive.records()
            info = read_json(root / "meta/info.json")
            if (not all_records or info.get("fps") != 30
                    or info.get("total_episodes") != len(all_records)
                    or info.get("total_frames") != sum(r["length"] for r in all_records)):
                raise ValueError("Source archive metadata/inventory mismatch")
            if [int(r["episode_index"]) for r in all_records] != list(range(len(all_records))):
                raise ValueError("Source episode indices must be contiguous")
            if len({r["source_key"] for r in all_records}) != len(all_records):
                raise ValueError("Source has duplicate episode identities")
            if pilot_indices is None:
                records = all_records
            else:
                selected = sorted(set(int(i) for i in pilot_indices))
                if not selected or selected[0] < 0 or selected[-1] >= len(all_records):
                    raise ValueError("Pilot episode indices outside the archive")
                records = [all_records[i] for i in selected]
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            job = Path(job_dir).resolve() if job_dir else root / f".pipeline/migrations/dense-v5-{stamp}-{uuid.uuid4().hex[:8]}"
            try:
                relative_job = str(job.relative_to(root))
            except ValueError as exc:
                raise ValueError("Migration staging must be on the same archive filesystem") from exc
            if job.exists() and any(job.iterdir()):
                raise ValueError("New migration job directory must be empty")
            job.mkdir(parents=True, exist_ok=True)
            backup = job / "backup"
            backup.mkdir()
            shutil.copy2(root / ".pipeline/config.json", backup / "config.json")
            shutil.copytree(root / "meta", backup / "meta")
            (backup / "episodes").mkdir()
            inventory = []
            expected_start = 0
            for record in records:
                idx = int(record["episode_index"])
                if int(record["dataset_to_index"]) - int(record["dataset_from_index"]) != int(record["length"]):
                    raise ValueError("Source frame range/length mismatch")
                if pilot_indices is None and int(record["dataset_from_index"]) != expected_start:
                    raise ValueError("Source dataset ranges are not contiguous")
                expected_start = int(record["dataset_to_index"])
                atomic_json(backup / f"episodes/{idx:08d}.json", record)
                inventory.append({"episode_index": idx, "source_key": record["source_key"],
                                  "length": int(record["length"]), "source_record_sha256": digest(record),
                                  "source_file_stat": _stat(archive.data_path(idx))})
            atomic_json(job / "inventory.json", inventory)
            journal = {"format": FORMAT, "state": "staging", "phase": "staging",
                       "source_schema": V4_SCHEMA_VERSION, "target_schema": SCHEMA_VERSION,
                       "started_at": _now(), "job_relative_path": relative_job,
                       "producer_signature": signature, "native_root": native_root,
                       "source_inventory_sha256": digest(inventory),
                       "total_episodes": len(records), "total_frames": sum(r["length"] for r in records),
                       "completed_episodes": 0, "completed_frames": 0,
                       "pilot": pilot_indices is not None, "workers": workers,
                       "original_provenance_sha256": digest(read_json(backup / "config.json"))}
        # A pilot has its own status only; regular archive users remain on v4.
        live_journal = journal_path if pilot_indices is None else job / "status.json"
        journal.update(state="staging", phase="staging", last_error=None,
                       completed_episodes=0, completed_frames=0, workers=workers)
        atomic_json(live_journal, journal)
        atomic_json(job / "status.json", journal)
        by_index = {item["episode_index"]: item for item in inventory}
        tasks = [(str(root), str(job), record, V4_SCHEMA_VERSION, signature,
                  by_index[record["episode_index"]]["source_file_stat"], native_root) for record in records]
        started = time.monotonic()

        def completed(result):
            receipt = result["receipt"]
            journal["completed_episodes"] += 1
            journal["completed_frames"] += receipt["length"]
            journal.update(last_episode=receipt["episode_index"], last_result=result["kind"],
                           updated_at=_now(), elapsed_seconds=time.monotonic() - started)
            atomic_json(live_journal, journal)
            atomic_json(job / "status.json", journal)
            if progress:
                progress(dict(journal))

        try:
            if workers == 1:
                for task in tasks:
                    completed(_stage_episode(task))
            else:
                with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
                    iterator = iter(tasks)
                    pending = set()
                    for _ in range(min(len(tasks), 2 * workers)):
                        pending.add(pool.submit(_stage_episode, next(iterator)))
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            completed(future.result())
                            task = next(iterator, None)
                            if task is not None:
                                pending.add(pool.submit(_stage_episode, task))
            receipts, receipt_sha = _receipt_inventory(job, records, signature)
            for record, receipt in zip(records, receipts):
                idx = record["episode_index"]
                if (_stat(archive.data_path(idx)) != receipt["source_file_stat"]
                        or digest(read_json(archive.record_path(idx))) != receipt["source_record_sha256"]
                        or _stat(_data_path(job, idx)) != receipt["staged_file_stat"]):
                    raise ValueError(f"Source/staged content changed after validation: {idx}")
            journal.update(state="validated", phase="validated", receipts_sha256=receipt_sha,
                           validated_at=_now(), last_error=None)
            atomic_json(live_journal, journal)
            atomic_json(job / "status.json", journal)
            if commit:
                result = _commit_v5(archive, journal, journal_path)
            else:
                result = dict(journal)
            if progress:
                progress(result)
            return result
        except BaseException as exc:
            journal.update(state="failed", last_error=f"{type(exc).__name__}: {exc}",
                           traceback=traceback.format_exc()[-6000:], updated_at=_now())
            atomic_json(live_journal, journal)
            atomic_json(job / "status.json", journal)
            if progress:
                progress(dict(journal))
            raise


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--native-root")
    parser.add_argument("--pilot-episodes", nargs="+", type=int)
    parser.add_argument("--no-commit", action="store_true")
    parser.add_argument("--job-dir")
    args = parser.parse_args(argv)

    def progress(state):
        print(json.dumps({key: state.get(key) for key in (
            "state", "phase", "completed_episodes", "total_episodes", "completed_frames",
            "total_frames", "last_episode", "elapsed_seconds", "last_error")}), flush=True)

    result = upgrade_dense_v5(args.dataset, workers=args.workers, native_root=args.native_root,
                              pilot_indices=args.pilot_episodes, commit=not args.no_commit,
                              job_dir=args.job_dir, progress=progress)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
