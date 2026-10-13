"""Fresh, streaming verification of live v5 data without a v4 dependency.

Every numeric column is decoded and hashed again, including DINO5. Migration
receipts are a comparison target, never a substitute for reading live data.
Only reports are written. A shared flock blocks normal archive writers.
"""

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import uuid

import numpy as np
import pyarrow.parquet as pq

from .geometry import quat_to_matrix
from .head_reference import (CAMERA_BASIS_POLICIES, floor_reference,
                             positions_in_floor_reference, reference_rotation)
from .numeric_fingerprint import NumericColumnHasher, motion_digest_from_hashes
from .schema import (SCHEMA_VERSION, V4_SCHEMA_VERSION, BODY_NAMES, HAND_NAMES, POSITION_SHAPES,
                     ROTATION_SHAPES, dino_format, feature_specs_for_schema)
from .storage import Archive, atomic_json, digest, read_json
from .unified_dense22 import ENCODING_POLICY, unified_metadata, validate_unified_dense22
from .unified_migration import _batch_arrays, _check_batch_join, _data_path, _stat

FORMAT = "egomotion-live-v5-full-revalidation-v1"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _require(ok, message):
    if not ok:
        raise ValueError(message)


@contextmanager
def _shared_archive_lock(archive):
    """Use precisely the same existing lock file as Archive.lock's writer."""
    archive.assert_ready()
    with (archive.root/".pipeline/lock").open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Archive writer is active; cannot start live v5 revalidation") from exc
        try:
            archive.assert_ready()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _contained(root, relative):
    target = (root/relative).resolve()
    _require(target.is_relative_to(root.resolve()), "Migration receipt directory escapes archive root")
    return target


def _inventory(archive):
    """Check current metadata and receipts; no backup directory is accessed."""
    root = archive.root
    _require(archive.schema_version == SCHEMA_VERSION, "Full revalidation requires the physical v5 schema")
    specs = feature_specs_for_schema(SCHEMA_VERSION)
    config = read_json(root/".pipeline/config.json")
    info = read_json(root/"meta/info.json")
    ontology = read_json(root/"meta/ontology.json")
    journal = read_json(root/".pipeline/dense_v5_upgrade.json")
    _require(journal.get("state") == "complete" and journal.get("target_schema") == SCHEMA_VERSION,
             "Dense v5 migration is not complete")
    _require(info.get("schema_version") == SCHEMA_VERSION and info.get("features") == specs
             and info.get("fps") == 30 and info.get("codebase_version") == "v3.0", "Live archive info/schema mismatch")
    _require(ontology.get("schema_version") == SCHEMA_VERSION and ontology.get("body_joints") == BODY_NAMES
             and ontology.get("hand_joints") == HAND_NAMES and ontology.get("unified_dense22") == ENCODING_POLICY,
             "Live archive ontology/encoding policy mismatch")
    signature = journal.get("producer_signature")
    _require(isinstance(signature, str) and len(signature) == 64 and all(c in "0123456789abcdef" for c in signature),
             "Invalid migration producer signature")
    _require(ontology.get("motion_representation_producer_signature") == signature, "Ontology producer signature differs")
    policy = config.get("dense_v5_migration", {})
    for key in ("producer_signature", "receipts_sha256", "job_relative_path", "source_schema"):
        _require(policy.get(key) == journal.get(key), f"Live config migration provenance differs: {key}")
    _require(digest(config.get("previous_archive_provenance")) == journal.get("original_provenance_sha256"),
             "Original provenance chain differs from completed migration")
    migration_job = _contained(root, journal["job_relative_path"])
    source_inventory = read_json(migration_job/"inventory.json")
    _require(digest(source_inventory) == journal.get("source_inventory_sha256"), "Migration source inventory digest differs")
    records = archive.records()
    _require(records and len(records) == info.get("total_episodes") == journal.get("total_episodes"),
             "Live archive/journal episode counts differ")
    _require(len(source_inventory) == len(records), "Migration inventory episode count differs")
    expected_record_names = {f"{i:08d}.json" for i in range(len(records))}
    _require({p.name for p in (root/".pipeline/episodes").glob("*.json")} == expected_record_names,
             "Live episode metadata file inventory differs")
    _require({p.name for p in (migration_job/"receipts").glob("*.json")} == expected_record_names,
             "Migration receipt file inventory differs")
    _require({str(p.relative_to(root)) for p in (root/"data").rglob("*.parquet")}
             == {str(_data_path(root, i).relative_to(root)) for i in range(len(records))},
             "Live Parquet file inventory differs from episode metadata")
    receipts, receipt_hashes, files, offset = {}, {}, {}, 0
    encoder_signatures = set()
    for index, record in enumerate(records):
        count = record.get("length")
        _require(type(count) is int and count > 0 and record.get("episode_index") == index
                 and record.get("dataset_from_index") == offset and record.get("dataset_to_index") == offset+count,
                 f"Episode/global metadata indices differ: {index}")
        original = source_inventory[index]
        _require(original.get("episode_index") == index and original.get("length") == count
                 and original.get("source_key") == record.get("source_key"), f"Source inventory episode identity differs: {index}")
        receipt = read_json(migration_job/f"receipts/{index:08d}.json")
        receipts[index], receipt_hashes[str(index)] = receipt, digest(receipt)
        _require(receipt.get("episode_index") == index and receipt.get("length") == count
                 and receipt.get("source_key") == record.get("source_key") and receipt.get("target_schema") == SCHEMA_VERSION
                 and receipt.get("producer_signature") == signature and receipt.get("validation", {}).get("passed") is True,
                 f"Migration receipt identity/validation differs: {index}")
        _require(receipt.get("source_record_sha256") == original.get("source_record_sha256"),
                 f"Migration receipt/source inventory provenance differs: {index}")
        # Recover the original metadata shape from the live record only. This
        # verifies unchanged ground, camera, source and DINO provenance without
        # accessing a retained v4 file or backup metadata.
        source_record = copy.deepcopy(record)
        for key in ("motion_representation", "unified_migration"):
            source_record.pop(key, None)
        source_record["metadata"].pop("unified_dense22", None)
        for key in set(specs)-set(feature_specs_for_schema(V4_SCHEMA_VERSION)):
            if key.startswith("observation.valid."):
                source_record["quality"].pop(key.removeprefix("observation.valid."), None)
        _require(digest(source_record) == receipt["source_record_sha256"],
                 f"Live source metadata provenance differs from migration receipt: {index}")
        columns = receipt.get("output_column_sha256")
        _require(isinstance(columns, dict) and set(columns) == set(specs) and digest(columns) == receipt.get("output_numeric_sha256"),
                 f"Migration receipt column hash inventory differs: {index}")
        _require(motion_digest_from_hashes({"columns": columns}) == receipt.get("motion_digest"),
                 f"Migration receipt motion digest differs: {index}")
        rep = record.get("motion_representation", {})
        _require(rep == {"schema_version": SCHEMA_VERSION, "producer_signature": signature, "motion_digest": receipt["motion_digest"]},
                 f"Live episode motion representation provenance differs: {index}")
        origin = record.get("unified_migration", {})
        for key in ("source_record_sha256", "source_numeric_sha256", "output_numeric_sha256"):
            _require(origin.get(key) == receipt.get(key), f"Live episode migration provenance differs: {index}: {key}")
        _require(origin.get("job_relative_path") == journal["job_relative_path"], f"Live episode migration job differs: {index}")
        declared, expected = record["metadata"].get("unified_dense22"), unified_metadata(record)
        _require(isinstance(declared, dict) and all(declared.get(k) == v for k, v in expected.items()),
                 f"Live episode encoding/axis/cache declaration differs: {index}")
        if record.get("dino", {}).get("state") == "complete":
            encoder = record["dino"].get("encoder", {})
            _require(all(encoder.get(k) == v for k, v in dino_format().items() if k in (
                "shape", "flat_shape", "token_order", "flatten_order", "pooling", "storage_dtype", "normalization")),
                f"Live DINO token format differs: {index}")
            encoder_signatures.add(digest(encoder))
        files[index] = _stat(_data_path(root, index))
        offset += count
    _require(offset == info.get("total_frames") == journal.get("total_frames"), "Live archive/journal total frame counts differ")
    _require(digest(receipt_hashes) == journal.get("receipts_sha256"), "Completed migration receipt inventory digest differs")
    _require(len(encoder_signatures) <= 1, "Live archive mixes DINO encoder signatures")
    # Verify the compact shared episode index as well as per-episode JSON.
    table = pq.read_table(root/"meta/episodes/chunk-000/file-000.parquet").to_pylist()
    _require(len(table) == len(records), "Shared episode table count differs")
    for row, record in zip(table, records):
        for key in ("episode_index", "length", "dataset_from_index", "dataset_to_index", "source", "source_uid"):
            _require(row.get(key) == record.get(key), f"Shared episode table identity differs: {record['episode_index']}: {key}")
    metadata_paths = [root/".pipeline/config.json", root/"meta/info.json", root/"meta/ontology.json",
                      root/".pipeline/dense_v5_upgrade.json", migration_job/"inventory.json"]
    metadata_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in metadata_paths}
    return records, receipts, files, {"schema_version": SCHEMA_VERSION, "columns": len(specs), "episodes": len(records),
        "frames": offset, "physical_bytes": sum(s["size"] for s in files.values()),
        "producer_signature": signature, "receipts_sha256": journal["receipts_sha256"],
        "migration_job": str(migration_job), "metadata_sha256": metadata_hashes,
        "no_backup_inputs": True, "full_column_read_required": True}


def _validate_batch(arrays, record):
    """Base archive invariants plus independent v5 geometric round trips."""
    count = len(arrays["timestamp"])
    for key, value in arrays.items():
        _require(np.isfinite(value).all(), f"Non-finite numeric column: {key}")
        if key.startswith("observation.interpolated."):
            _require(not np.any(value & ~arrays[key.replace("interpolated", "valid")]),
                     f"Invalid sample is marked interpolated: {key}")
    for modality, shape in POSITION_SHAPES.items():
        value = arrays[f"observation.state.{modality}_world"].reshape(count, *shape)
        valid = arrays[f"observation.valid.{modality}_position"].reshape(count, shape[0])
        _require(not np.any(value[~valid] != 0), f"Invalid original world XYZ must be zero: {modality}")
    for modality, shape in ROTATION_SHAPES.items():
        q = arrays[f"observation.state.{modality}_world_rotation"].reshape(count, *shape)
        valid = arrays[f"observation.valid.{modality}_rotation"].reshape(count, shape[0])
        _require(np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-5, rtol=0), f"Non-unit world quaternion: {modality}")
        _require(not np.any(q[~valid] != [1, 0, 0, 0]), f"Invalid original quaternion must be identity: {modality}")
    weight = arrays["observation.source_interpolation_weight"]
    _require(np.all((weight >= 0) & (weight <= 1)), "Source interpolation weights are outside [0,1]")
    ground = record["metadata"].get("ground", {})
    height = arrays["observation.environment.floor_height"]
    floor_valid = arrays["observation.valid.floor"].reshape(count)
    _require(np.allclose(height, ground.get("height_m", 0), atol=1e-5, rtol=1e-6)
             and np.all(floor_valid == bool(ground.get("valid", False))), "Floor fields differ from episode ground provenance")
    _require(np.all(arrays["observation.valid.floor_verified"] == bool(ground.get("valid") and ground.get("verified"))),
             "Floor verification differs from provenance")
    _require(np.allclose(arrays["observation.environment.floor_normal"], [0, 0, 1], atol=1e-6, rtol=0),
             "Stored floor is not horizontal in Z-up world")
    hp = arrays["observation.valid.head_reference_position"].reshape(count)
    hr = arrays["observation.valid.head_reference_rotation"].reshape(count)
    basis = arrays["observation.valid.head_reference_basis"].reshape(count)
    reference = record["metadata"].get("head_reference", {})
    _require(np.all(basis == bool(reference.get("basis_verified", False))), "Camera reference basis differs from provenance")
    hvalid = arrays["observation.valid.head_reference_height"].reshape(count)
    _require(np.array_equal(hvalid, hp & floor_valid), "Camera height mask differs")
    expected_height = arrays["observation.state.head_reference_world"][:, 2]-height
    _require(np.allclose(arrays["observation.state.head_reference_height"][hvalid], expected_height[hvalid], atol=1e-5),
             "Camera height disagrees with world floor")
    _require(not np.any(arrays["observation.state.head_reference_height"][~hvalid] != 0), "Invalid camera height must be zero")
    if reference.get("mode") == "ee4d":
        source_basis = record["metadata"].get("camera_pose_basis", "unspecified")
        _require(reference.get("source_camera_basis") == source_basis, "Camera axis provenance differs")
        if reference.get("basis_policy") != "manual":
            policy, correction = CAMERA_BASIS_POLICIES.get(source_basis, ("unknown", np.eye(3)))
            _require(reference.get("basis_policy") == policy
                     and np.allclose(reference["camera_from_ee4d_camera"], correction), "Camera axis policy differs")
        _require(np.array_equal(hp, arrays["observation.valid.camera_position"].reshape(count))
                 and np.array_equal(hr, arrays["observation.valid.camera_rotation"].reshape(count)), "Camera reference masks differ from camera")
        expected_r = quat_to_matrix(arrays["observation.state.camera_world_rotation"]) @ reference_rotation(reference["camera_from_ee4d_camera"])
        _require(np.allclose(quat_to_matrix(arrays["observation.state.head_reference_world_rotation"])[hr], expected_r[hr], atol=1e-5),
                 "Camera reference rotation disagrees with documented source conversion")
        _require(np.allclose(arrays["observation.state.head_reference_world"][hp], arrays["observation.state.camera_world"][hp], atol=1e-5),
                 "Camera reference and optical center differ")
    elif reference.get("mode") == "disabled":
        _require(not (hp.any() or hr.any() or basis.any()), "Disabled camera reference has valid labels")
    else:
        raise ValueError("Unknown camera reference mode")
    p, q, valid = floor_reference(arrays["observation.state.head_reference_world"].reshape(count, 1, 3),
        arrays["observation.state.head_reference_world_rotation"].reshape(count, 1, 4), hp[:, None], hr[:, None], ground, bool(basis.all()))
    valid = valid[:, 0]
    for kind in ("position", "rotation"):
        _require(np.array_equal(arrays[f"observation.valid.head_reference_floor_{kind}"].reshape(count), valid), "Camera floor reference mask differs")
    _require(np.allclose(arrays["observation.state.head_reference_floor_world"], p.reshape(count, 3), atol=1e-5), "Camera floor origin differs")
    _require(np.allclose(quat_to_matrix(arrays["observation.state.head_reference_floor_world_rotation"]), quat_to_matrix(q[:, 0]), atol=1e-5),
             "Camera floor yaw differs")
    local, mask = positions_in_floor_reference(arrays["observation.state.body_world"].reshape(count, 22, 3),
        arrays["observation.valid.body_position"], p, q, valid[:, None])
    _require(np.array_equal(arrays["observation.valid.body_head_floor_position"], mask)
             and np.allclose(arrays["observation.state.body_head_floor"].reshape(count, 22, 3), local, atol=1e-5, rtol=1e-6),
             "Original body floor cache differs from world/camera/ground")
    _require(not np.any(arrays["observation.state.body_head_floor"].reshape(count, 22, 3)[~mask] != 0), "Invalid original body floor XYZ must be zero")
    interp = (arrays["observation.interpolated.head_reference_position"].reshape(count, 1)
              | arrays["observation.interpolated.head_reference_rotation"].reshape(count, 1))
    for kind in ("position", "rotation"):
        _require(np.array_equal(arrays[f"observation.interpolated.head_reference_floor_{kind}"].reshape(count, 1), interp & valid[:, None]),
                 "Camera floor interpolation dependency differs")
    _require(np.array_equal(arrays["observation.interpolated.body_head_floor_position"],
             (arrays["observation.interpolated.body_position"] | interp) & mask), "Body floor interpolation dependency differs")
    dino = record.get("dino", {})
    dv = arrays["observation.valid.dino"].reshape(count)
    _require(dino.get("state") in ("pending", "complete"), "Unknown DINO extraction state")
    _require(dino.get("state") == "complete" or not dv.any(), "Pending DINO has valid labels")
    _require(not np.any(arrays["observation.features.dino"][~dv] != 0), "Invalid DINO token rows must be zero")
    if dino.get("state") == "complete":
        _require(np.all(np.abs(arrays["observation.video_time_error"][dv]) <= float(dino["max_video_error_s"])+1e-6),
                 "DINO video alignment error exceeds recorded tolerance")
    return validate_unified_dense22(arrays, arrays, record)


def _validate_stream(task):
    root, record, receipt, expected_stat, batch_size, run_id = task
    root = Path(root)
    index, length = record["episode_index"], record["length"]
    source = _data_path(root, index)
    started, offset, decoded_bytes, batches = time.monotonic(), 0, 0, 0
    result = {"format": FORMAT, "run_id": run_id, "episode_index": index, "source_key": record["source_key"],
              "expected_frames": length, "physical_bytes": expected_stat["size"], "path": str(source),
              "fresh_disk_numeric_read": True, "includes_all_dino_values": True, "backup_data_read": False}
    valid_counts, maxima = {}, {}
    try:
        _require(_stat(source) == expected_stat, "Live Parquet changed before validation")
        record_path = root/f".pipeline/episodes/{index:08d}.json"
        _require(digest(read_json(record_path)) == digest(record), "Live episode metadata changed before validation")
        specs = feature_specs_for_schema(SCHEMA_VERSION)
        parquet = pq.ParquetFile(source)
        _require(set(parquet.schema_arrow.names) == set(specs) and parquet.metadata.num_rows == length,
                 "Live Parquet physical schema/row count differs")
        hasher = NumericColumnHasher(specs, length)
        previous_capture = None
        for batch in parquet.iter_batches(batch_size=batch_size, use_threads=False):
            arrays = _batch_arrays(batch, specs)
            decoded_bytes += sum(v.nbytes for v in arrays.values())
            _check_batch_join(arrays, record, offset)
            capture = arrays["observation.source_timestamp"]
            _require(np.all(np.diff(capture) > 0) and (previous_capture is None or capture[0] > previous_capture),
                     "Source capture timestamps are not strictly increasing across batches")
            previous_capture = float(capture[-1])
            geometry = _validate_batch(arrays, record)
            hasher.update(arrays)
            for key, value in arrays.items():
                if key.startswith("observation.valid."):
                    valid_counts[key] = valid_counts.get(key, 0)+int(value.sum())
            for frame in ("world", "floor", "hand_left", "hand_right"):
                for name, value in geometry[frame].items():
                    if "max_abs" in name:
                        label = frame+"."+name
                        maxima[label] = max(maxima.get(label, 0.), float(value))
            offset += len(batch)
            batches += 1
        hashes = hasher.finish()
        result.update(actual_numeric_sha256=hashes["sha256"], actual_column_sha256=hashes["columns"],
                      actual_motion_digest=motion_digest_from_hashes(hashes))
        mismatched = [key for key in specs if hashes["columns"][key] != receipt["output_column_sha256"][key]]
        _require(not mismatched and hashes["sha256"] == receipt["output_numeric_sha256"],
                 "Live numeric column hash differs from independently staged receipt: "+", ".join(mismatched))
        _require(result["actual_motion_digest"] == record["motion_representation"]["motion_digest"] == receipt["motion_digest"],
                 "Live motion digest differs from episode and receipt provenance")
        for key, value in receipt.get("valid_counts", {}).items():
            _require(valid_counts.get(key) == value, "Live validity count differs from migration receipt: "+key)
        for key, value in valid_counts.items():
            quality_key = key.removeprefix("observation.valid.")
            quality = record.get("quality", {}).get(quality_key)
            expected = value/(length*specs[key]["shape"][0])
            _require(quality is not None and abs(float(quality)-expected) < 1e-12, "Live validity ratio differs from quality provenance: "+key)
        if record["dino"]["state"] == "complete":
            _require(record["dino"].get("valid_count") == valid_counts["observation.valid.dino"], "DINO valid_count differs from fresh masks")
        _require(offset == length and _stat(source) == expected_stat, "Live Parquet length/identity changed during validation")
        _require(digest(read_json(record_path)) == digest(record), "Live episode metadata changed during validation")
        result.update(passed=True, validation={"all_numeric_columns_redecoded": True, "all_column_hashes_match_receipt": True,
            "motion_digest_matches": True, "geometry_and_masks": True, "source_indices_and_time": True,
            "finite_dtype_shape_invalid_padding": True, "quality_and_dino_counts": True},
            valid_counts=valid_counts, geometry_maximum_errors=maxima,
            expected_rotation_basis_verified=bool(record["metadata"].get("rotation_basis_verified", False)),
            expected_floor_valid=bool(record["metadata"].get("ground", {}).get("valid", False)))
    except Exception as exc:
        result.update(passed=False, error=f"{type(exc).__name__}: {exc}", failed_batch_offset=offset,
                      traceback=traceback.format_exc())
    result.update(decoded_frames=offset, decoded_numeric_bytes=decoded_bytes, batches=batches,
                  elapsed_seconds=time.monotonic()-started, completed_at=_now())
    return result


def revalidate_unified_dataset(dataset, job_dir, *, workers=16, batch_size=3000,
                               episode_indices=None, progress=None):
    """Write fresh per-episode and aggregate reports; return passed=False on failure.

    A selection is explicitly a pilot. Success permits full-dataset decisions
    only when full_dataset_validation and all_live_episodes_validated are true.
    Existing reports are never reused as verification results.
    """
    root, job = Path(dataset).resolve(), Path(job_dir).resolve()
    job.mkdir(parents=True, exist_ok=True)
    started, run_id = time.monotonic(), uuid.uuid4().hex
    state = {"format": FORMAT, "run_id": run_id, "stage": "starting", "pid": os.getpid(),
             "dataset": str(root), "job_dir": str(job), "workers": workers, "batch_size": batch_size,
             "started_at": _now(), "completed_episodes": 0, "completed_frames": 0, "failed_episodes": 0,
             "decoded_numeric_bytes": 0, "validated_physical_bytes": 0,
             "full_dataset_validation": episode_indices is None, "backup_data_read": False,
             "lock": {"path": str(root/".pipeline/lock"), "mode": "shared_flock_same_as_archive_writer"}}
    results, errors = [], []
    def update(**fields):
        state.update(fields, updated_at=_now(), elapsed_seconds=time.monotonic()-started)
        atomic_json(job/"status.json", state)
        if progress:
            progress(dict(state))
    def accept(result):
        atomic_json(job/f"episodes/{result['episode_index']:08d}.json", result)
        results.append(result)
        if not result["passed"]:
            errors.append({"episode_index": result["episode_index"], "source_key": result["source_key"], "error": result["error"]})
        state["completed_episodes"] += 1
        state["completed_frames"] += result["decoded_frames"]
        state["failed_episodes"] += int(not result["passed"])
        state["decoded_numeric_bytes"] += result["decoded_numeric_bytes"]
        if result["passed"]:
            state["validated_physical_bytes"] += result["physical_bytes"]
        update(last_episode=result["episode_index"], last_passed=result["passed"])
        if state["completed_episodes"] == 1 or state["completed_episodes"] % 25 == 0 or state["completed_episodes"] == state["total_episodes"]:
            print(f"[{state['completed_episodes']}/{state['total_episodes']}] live v5 freshly decoded; "
                  f"frames={state['completed_frames']} failures={state['failed_episodes']}", flush=True)
    try:
        _require(type(workers) is int and workers >= 1 and type(batch_size) is int and batch_size >= 1,
                 "workers and batch_size must be positive integers")
        _require(root.is_dir(), "Archive root does not exist")
        update(stage="inventory")
        archive = Archive(root)
        with _shared_archive_lock(archive):
            records, receipts, files, inventory = _inventory(archive)
            atomic_json(job/"inventory-validation.json", inventory)
            if episode_indices is None:
                selected = records
            else:
                indices = list(episode_indices)
                _require(indices and all(type(i) is int and 0 <= i < len(records) for i in indices)
                         and len(set(indices)) == len(indices), "Invalid pilot episode selection")
                selected = [records[i] for i in sorted(indices)]
            update(stage="validating", total_episodes=len(selected), total_frames=sum(r["length"] for r in selected),
                   live_total_episodes=len(records), live_total_frames=inventory["frames"],
                   total_physical_bytes=sum(files[r["episode_index"]]["size"] for r in selected))
            tasks = [(str(root), r, receipts[r["episode_index"]], files[r["episode_index"]], batch_size, run_id) for r in selected]
            if workers == 1:
                for task in tasks:
                    accept(_validate_stream(task))
            else:
                iterator = iter(tasks)
                with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
                    active = {}
                    for _ in range(min(workers*2, len(tasks))):
                        task = next(iterator)
                        active[pool.submit(_validate_stream, task)] = task[1]["episode_index"]
                    while active:
                        ready, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
                        if not ready:
                            update(active_workers=len(active))
                        for future in ready:
                            active.pop(future)
                            accept(future.result())
                            task = next(iterator, None)
                            if task is not None:
                                active[pool.submit(_validate_stream, task)] = task[1]["episode_index"]
            for filename, sha in inventory["metadata_sha256"].items():
                _require(hashlib.sha256(Path(filename).read_bytes()).hexdigest() == sha, "Shared archive metadata changed during validation")
            receipt_hashes = {str(r["episode_index"]): digest(read_json(Path(inventory["migration_job"])/f"receipts/{r['episode_index']:08d}.json")) for r in records}
            _require(digest(receipt_hashes) == inventory["receipts_sha256"], "Migration receipts changed during live validation")
            archive.assert_ready()
            passed = not errors and len(results) == len(selected) and state["completed_frames"] == state["total_frames"]
            all_live = passed and episode_indices is None and len(results) == len(records)
            report = {**state, "passed": passed, "all_live_episodes_validated": all_live,
                      "full_numeric_redecode": True, "includes_all_dino_values": True,
                      "inventory": inventory, "errors": errors,
                      "episode_results": [{"episode_index": r["episode_index"], "passed": r["passed"],
                                           "report": f"episodes/{r['episode_index']:08d}.json"} for r in sorted(results, key=lambda x: x["episode_index"])],
                      "allowed_missingness": {"unverified_native_rotation_axes": sum(not r["metadata"].get("rotation_basis_verified", False) for r in selected),
                          "unknown_archive_ground": sum(not r["metadata"].get("ground", {}).get("valid", False) for r in selected)},
                      "v4_backup_dependency": False, "source_data_modified": False}
            report["stage"] = "completed" if passed else "failed"
            report["completed_at"] = _now()
            report["elapsed_seconds"] = time.monotonic()-started
            atomic_json(job/"validation.json", report)
            update(stage=report["stage"], passed=passed, all_live_episodes_validated=all_live,
                   errors=errors, completed_at=report["completed_at"])
            return report
    except Exception as exc:
        errors.append({"phase": state["stage"], "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        update(stage="failed", passed=False, all_live_episodes_validated=False, errors=errors)
        report = {**state, "passed": False, "all_live_episodes_validated": False, "errors": errors,
                  "full_numeric_redecode": False, "source_data_modified": False, "v4_backup_dependency": False,
                  "completed_at": _now()}
        atomic_json(job/"validation.json", report)
        return report
