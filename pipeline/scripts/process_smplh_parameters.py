"""Build an owned SMPL-H parameter archive without rewriting frozen fit code.

An episode commits motion, vision and provenance together. Source data remains
untouched; cleanup is a separate operation gated by independent acceptance.
"""
from __future__ import annotations

import os
for _thread_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_name, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import traceback
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from egomotion_pipeline.smplh_parameters import (
    MODEL_SHA256, PARAMETER_SCHEMA_VERSION, OPTIONAL_SPECS,
    parameter_specs, parameters_from_fitted, write_parameter_table,
)
from egomotion_pipeline.storage import Archive
from egomotion_pipeline.smplh_jobs import read_source_motion, arrays_sha256

FEATURE = "observation.features.dino"
VISION_COLUMNS = ("frame_index", "timestamp", "observation.source_timestamp",
                  "observation.valid.dino", "observation.video_timestamp",
                  "observation.video_time_error", FEATURE)
FIT_COLUMNS = ("fitted.local_axis_angle", "fitted.smpl_trans", "fitted.betas",
               "estimated.valid.body_world_position", "estimated.valid.body_world_rotation",
               "quality.position_confidence", "quality.rotation_confidence",
               "quality.observation_residual_m", "quality.observation_residual_valid")


def producer_code():
    """Bind every conversion and resume to the actual decoder/producer code."""
    package = Path(__file__).resolve().parents[1] / "src/egomotion_pipeline"
    names = ("smplh_parameters.py", "smplh_hm_parameters.py", "body_model.py",
             "geometry.py", "schema.py", "ground.py", "head_reference.py",
             "windows.py", "resampling.py", "smplh_jobs.py", "storage.py")
    return {str(path): file_sha(path) for path in (Path(__file__).resolve(),
            *(package / name for name in names))}


def conversion_binding(row, plan):
    return digest_json({"row": row, "plan": plan, "code": producer_code(),
                        "parameter_schema": PARAMETER_SCHEMA_VERSION})


def validate_commit(final, receipt, row, plan, record=None):
    """Resume only an unchanged atomic artifact from this exact source/job."""
    if receipt.get("conversion_binding") != conversion_binding(row, plan):
        raise ValueError("Existing conversion belongs to different source, inventory or code")
    saved = receipt["record"]
    if record is not None and record != saved:
        raise ValueError("Episode index differs from its committed receipt")
    for key in ("episode_index", "source", "source_key", "length", "split",
                "dataset_from_index", "dataset_to_index"):
        if saved[key] != row[key]: raise ValueError("Existing episode differs from frozen inventory: " + key)
    for role in ("motion", "vision"):
        path = final / f"{role}.parquet"
        if identity(path) != receipt["output_identities"][role]:
            raise ValueError("Committed output identity changed: " + role)
        if file_sha(path) != receipt[role]["sha256"]:
            raise ValueError("Committed output checksum differs: " + role)
    p = receipt["source_provenance"]
    if identity(row["source_path"]) != p["source_identity"]:
        raise ValueError("Committed source identity changed")
    if identity(receipt["vision"]["source_path"]) != receipt["vision"]["source_identity"]:
        raise ValueError("Committed vision source identity changed")
    if row["source"] == "egosuite":
        if identity(row["fitted_path"]) != p["fitted_identity"]:
            raise ValueError("Committed fitted parameter source changed")
        if file_sha(row["fitted_receipt_path"]) != p["fitted_receipt_sha256"]:
            raise ValueError("Committed fitting receipt changed")
    return saved


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode()


def digest_json(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def projection_digest(arrays, names):
    """Reproduce the frozen fitter's dtype/shape-qualified source fingerprint."""
    h = hashlib.sha256()
    for name in sorted(names):
        value = np.ascontiguousarray(arrays[name])
        h.update(json.dumps([name, value.dtype.str, list(value.shape)],
                            separators=(",", ":"), allow_nan=False).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("wb") as handle:
        handle.write(json_bytes(value)); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)
    sync_directory(path.parent)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def identity(path):
    stat = Path(path).stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns,
                device=stat.st_dev, inode=stat.st_ino)


def check_inventory_identity(path, expected):
    if not {"bytes", "mtime_ns", "device", "inode"} <= set(expected):
        raise ValueError("Frozen source identity is incomplete")
    current = {"path": str(Path(path).resolve()), **identity(path)}
    if any(current[key] != value for key, value in expected.items() if key in current):
        raise ValueError("Source differs from its frozen inventory identity: " + str(path))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


def table_arrays(table):
    arrays = {}
    for name in table.column_names:
        col = table[name].combine_chunks()
        if col.null_count:
            raise ValueError("Null source column: " + name)
        if pa.types.is_fixed_size_list(col.type):
            if col.values.null_count: raise ValueError("Null source list: " + name)
            arrays[name] = col.values.to_numpy(zero_copy_only=False).reshape(len(table), col.type.list_size).copy()
        else:
            arrays[name] = col.to_numpy(zero_copy_only=False).copy()
    return arrays


class HashingSink(io.RawIOBase):
    """Hash sequential Parquet bytes as they are written, without rereading DINO."""
    def __init__(self, path):
        super().__init__(); self.handle = Path(path).open("wb"); self.digest = hashlib.sha256()
    def writable(self): return True
    def seekable(self): return False
    def tell(self): return self.handle.tell()
    def write(self, value):
        written = self.handle.write(value); self.digest.update(memoryview(value)[:written]); return written
    def flush(self):
        if not self.handle.closed: self.handle.flush()
    def close(self):
        if not self.closed:
            self.flush(); os.fsync(self.handle.fileno()); super().close(); self.handle.close()


def copy_vision(source, target, arrays, source_frames):
    """Copy selected DINO values exactly; bind owned rows to motion identities."""
    source, target = Path(source), Path(target)
    before = identity(source)
    selected = np.asarray(source_frames, dtype=np.int64)
    count = len(arrays["timestamp"])
    if selected.shape != (count,) or np.any(np.diff(selected) <= 0) or selected[0] < 0:
        raise ValueError("Vision mapping must contain ordered unique source frames")
    parquet = pq.ParquetFile(source)
    if selected[-1] >= parquet.metadata.num_rows:
        raise ValueError("Motion maps outside the source vision sequence")
    schema = pa.schema([
        pa.field("frame_index", pa.int64()), pa.field("episode_index", pa.int64()),
        pa.field("timestamp", pa.float64()), pa.field("observation.source_timestamp", pa.float64()),
        pa.field("source.video_frame_index", pa.int64()), pa.field("observation.valid.dino", pa.bool_()),
        pa.field("observation.video_timestamp", pa.float64()), pa.field("observation.video_time_error", pa.float64()),
        pa.field(FEATURE, pa.list_(pa.float32(), 5120)),
    ], metadata={b"parameter_schema_version": PARAMETER_SCHEMA_VERSION.encode()})
    target.parent.mkdir(parents=True, exist_ok=True)
    column_hashes = {name: hashlib.sha256() for name in schema.names}
    sink = HashingSink(target)
    start = 0; copied = 0
    try:
        with pq.ParquetWriter(sink, schema, compression="zstd", compression_level=1,
                             use_dictionary=False, use_byte_stream_split=[FEATURE + ".list.element"]) as writer:
            for batch in parquet.iter_batches(batch_size=512, columns=list(VISION_COLUMNS), use_threads=False):
                end = start + len(batch)
                output_rows = np.flatnonzero((selected >= start) & (selected < end))
                if len(output_rows):
                    raw = table_arrays(pa.Table.from_batches([batch]))
                    rows = selected[output_rows] - start
                    if not np.array_equal(raw["frame_index"][rows], selected[output_rows]):
                        raise ValueError("Source DINO frame indices do not match explicit mapping")
                    for key in ("observation.valid.dino", "observation.video_timestamp", "observation.video_time_error"):
                        if not np.array_equal(raw[key][rows], arrays[key][output_rows]):
                            raise ValueError("DINO provenance differs from mapped motion: " + key)
                    values = {"frame_index": arrays["frame_index"][output_rows],
                              "episode_index": arrays["episode_index"][output_rows],
                              "timestamp": arrays["timestamp"][output_rows].astype(np.float64),
                              "observation.source_timestamp": arrays["observation.source_timestamp"][output_rows],
                              "source.video_frame_index": selected[output_rows],
                              "observation.valid.dino": raw["observation.valid.dino"][rows],
                              "observation.video_timestamp": raw["observation.video_timestamp"][rows],
                              "observation.video_time_error": raw["observation.video_time_error"][rows],
                              FEATURE: raw[FEATURE][rows]}
                    columns = {}
                    for key, value in values.items():
                        kind = schema.field(key).type
                        dtype = (np.float32 if key == FEATURE else np.bool_ if pa.types.is_boolean(kind)
                                 else np.int64 if pa.types.is_int64(kind) else np.float64)
                        value = np.ascontiguousarray(value, dtype=dtype)
                        column_hashes[key].update(value.tobytes())
                        columns[key] = (pa.FixedSizeListArray.from_arrays(pa.array(value.reshape(-1)), 5120)
                                        if key == FEATURE else pa.array(value))
                    writer.write_table(pa.table(columns, schema=schema), row_group_size=512)
                    copied += len(output_rows)
                start = end
                if start > selected[-1]: break
        if copied != count: raise ValueError("Incomplete mapped DINO copy")
    finally:
        sink.close()
    if identity(source) != before: raise ValueError("Source vision changed during copying")
    return {"source_path": str(source.resolve()), "source_identity": before,
            "rows": count, "sha256": sink.digest.hexdigest(),
            "column_byte_sha256": {key: h.hexdigest() for key, h in column_hashes.items()},
            "source_mapping": "explicit_source.video_frame_index", "dense_body_columns_copied": False}


def make_ego_inventory(job):
    plan = read_json(Path(job) / "plan.json")
    archive = Archive(plan["egosuite_archive"])
    splits = read_json("/root/workspace/wbc/guanzerong/cvpr/MotionExpert/exp/dense_k12/splits.json")
    membership = {key: split for split, keys in splits["splits"].items() for key in keys}
    rows = []
    for record in archive.records():
        index = int(record["episode_index"])
        row = {"episode_index": index, "source_episode_index": index, "source": "egosuite",
               "source_key": record["source_key"], "length": record["length"],
               "split": membership[record["source_key"]], "source_record": record,
               "source_path": str(archive.data_path(index)),
               "fitted_path": str(Path(plan["egosuite_fitted_root"]) / "episodes" / f"{index:08d}" / "motion.parquet"),
               "fitted_receipt_path": str(Path(plan["egosuite_fitted_root"]) / "episodes" / f"{index:08d}" / "receipt.json"),
               "dataset_from_index": record["dataset_from_index"], "dataset_to_index": record["dataset_to_index"]}
        rows.append(row)
    value = {"source": "egosuite", "episodes": len(rows), "frames": sum(r["length"] for r in rows), "rows": rows}
    value["inventory_sha256"] = digest_json(value)
    atomic_json(Path(job) / "inventory/egosuite.json", value)
    return value


def initialize_archive(job, output=None):
    job = Path(job); plan = read_json(job / "plan.json")
    root = Path(output or plan["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    model = root / "weights/smplh/male/model.npz"
    model.parent.mkdir(parents=True, exist_ok=True)
    if not model.exists(): shutil.copy2(plan["model_path"], model)
    if file_sha(model) != MODEL_SHA256: raise ValueError("Owned SMPL-H model does not match")
    source_hand_asset = Path("/gaozt-test1/guanzerong/UniEgoMotion/body_models/smplx/SMPLX_NEUTRAL.npz")
    hand_path = root / "weights/smplh/hand_basis.npz"
    if not hand_path.exists():
        with np.load(source_hand_asset, allow_pickle=False) as raw:
            np.savez_compressed(hand_path, **{k: raw[k] for k in ("hands_componentsl", "hands_componentsr", "hands_meanl", "hands_meanr")})
    info = {"schema_version": PARAMETER_SCHEMA_VERSION, "fps": 30,
            "body_storage": "SMPL-H root axis-angle3 + body local axis-angle63 + trans3 + real betas16",
            "body_xyz_and_6d": "transient FK after training frame selection; not physically stored",
            "hands": "source explicit21 points; SMPL-H hand PCA optional and independently masked",
            "model": {"path": str(model.relative_to(root)), "model_sha256": MODEL_SHA256,
                      "shape_basis_count": 16, "joint_count": 52, "body_joint_count": 22},
            "hand_basis": {"path": str(hand_path.relative_to(root)), "sha256": file_sha(hand_path),
                           "encoding": "optional left45/right45 PCA coefficients; add source hand means",
                           "source_asset": str(source_hand_asset), "body_asset_is_smplh": True},
            "motion_features": parameter_specs(), "optional_features": OPTIONAL_SPECS,
            "dino": {"checkpoint": "facebook/dinov2-with-registers-large", "shape": [5,1024],
                     "storage_dtype": "float32", "flatten_order": "token_major", "owned_separate_vision_files": True},
            "migration_job": str(job), "requires_old_dense_archive": False}
    path = root / "meta/info.json"
    if path.exists() and read_json(path) != info: raise ValueError("Archive already uses a different parameter contract")
    atomic_json(path, info)
    original_ontology = read_json(Path(plan["egosuite_archive"]) / "meta/ontology.json")
    original_ontology.update(schema_version=PARAMETER_SCHEMA_VERSION, target_joint_basis="smplh_body22_axes_v1",
                             body_xyz_storage=False, body_rotation_storage="local axis-angle; global row6D derived with FK")
    atomic_json(root / "meta/ontology.json", original_ontology)
    atomic_json(root / ".pipeline/config.json", {"schema_version": PARAMETER_SCHEMA_VERSION,
                "model_sha256": MODEL_SHA256, "job": str(job), "standalone": True})
    return root


def assert_frozen_job(job, *, code_only=False):
    job = Path(job)
    path = job / "code/producer-contract.json"
    if not path.exists(): raise ValueError("Full conversion requires a reviewed frozen producer contract")
    contract = read_json(path)
    if file_sha(job / "plan.json") != contract["plan_sha256"]:
        raise ValueError("Conversion plan changed after freeze")
    for name, sha in contract["files"].items():
        if code_only and not name.endswith(".py"): continue
        if file_sha(name) != sha: raise ValueError("Frozen conversion code/inventory changed: " + name)
    return contract


def finalize_archive(job):
    job = Path(job); plan = read_json(job / "plan.json"); root = Path(plan["output_root"])
    assert_frozen_job(job)
    inventory = read_json(job / "inventory/combined.json")
    expected = inventory["rows"]
    paths = sorted((root / ".pipeline/episodes").glob("*.json"))
    if len(paths) != len(expected): raise ValueError("Cannot finalize an incomplete population")
    records = [read_json(path) for path in paths]
    fields = ("episode_index", "source", "source_key", "length", "split", "dataset_from_index", "dataset_to_index")
    for row, record in zip(expected, records):
        if any(row[k] != record[k] for k in fields): raise ValueError("Final population differs from frozen inventory")
        for role in ("data_path", "vision_path"):
            path = (root / record[role]).resolve(strict=True)
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError("Final population contains an external/unowned artifact")
    splits = {name: [r["source_key"] for r in records if r["split"] == name] for name in ("train", "val", "test")}
    groups = {}
    for record in records:
        for group in record["split_group_ids"]:
            if group in groups and groups[group] != record["split"]:
                raise ValueError("A participant or capture leaks across splits")
            groups[group] = record["split"]
    summary = {"episodes": len(records), "frames": sum(r["length"] for r in records),
               "schema_version": PARAMETER_SCHEMA_VERSION, "sources": {}, "splits": {},
               "training_eligible_episodes": sum(bool(r["metadata"].get("training_eligible")) for r in records),
               "updated_at": now()}
    for source in ("egosuite", "egoexo4d_hm"):
        subset = [r for r in records if r["source"] == source]
        summary["sources"][source] = {"episodes": len(subset), "frames": sum(r["length"] for r in subset)}
    for name in splits:
        subset = [r for r in records if r["split"] == name]
        summary["splits"][name] = {"episodes": len(subset), "frames": sum(r["length"] for r in subset)}
    atomic_json(root / "meta/summary.json", summary)
    atomic_json(root / "meta/splits.json", {"splits": splits, "split_group_ids": groups,
                "policy": "EgoSuite original grouped splits; HM participant/capture connected components"})
    fields = ("episode_index", "source", "source_key", "source_uid", "length", "dataset_from_index", "dataset_to_index", "split", "data_path", "vision_path")
    pq.write_table(pa.table({k: [r[k] for r in records] for k in fields}), root / "meta/episodes.parquet", compression="zstd")
    atomic_json(root / ".pipeline/dataset-state.json", {"state": "converted_pending_acceptance",
                "independent_acceptance_pending": True, **summary})
    atomic_json(job / "reports/full-conversion.json", summary)
    return summary


def convert_ego(row, plan, output):
    source = Path(row["source_path"]); fitted_path = Path(row["fitted_path"])
    source_before = identity(source)
    source_arrays, _ = read_source_motion(source)
    fitted_before = identity(fitted_path)
    fitted_receipt_before = identity(row["fitted_receipt_path"])
    fitted_receipt = read_json(row["fitted_receipt_path"])
    fitted_receipt_sha = file_sha(row["fitted_receipt_path"])
    contract = fitted_receipt["declaration"]["contract"]
    check_inventory_identity(source, contract["source_identity"])
    if (fitted_receipt["state"] != "complete" or contract["model_sha256"] != MODEL_SHA256
            or contract["source_motion_sha256"] != projection_digest(source_arrays, contract["source_projection_columns"])
            or fitted_receipt["parquet_sha256"] != file_sha(fitted_path)):
        raise ValueError("Fitted SMPL-H artifact is not bound to this source/model")
    fit = table_arrays(pq.read_table(fitted_path, columns=list(FIT_COLUMNS)))
    arrays = parameters_from_fitted(source_arrays, fit)
    metadata = deepcopy(row["source_record"]["metadata"])
    metadata.update(target_joint_basis="smplh_body22_axes_v1", target_model_axes_defined=True,
                    label_kind="model_fitted_estimate", body_position_source="egosuite_dense22_fitted_smplh",
                    rotation_semantics="model_fitted_global_rotation_from_smplh_local_params",
                    source_native_axes_certified=False, absolute_rotation_accuracy_established=False)
    metadata.pop("unified_dense22", None)
    metadata["parameter_provenance"] = {"source": "egosuite", "fitting_job_signature": read_json(Path(plan["egosuite_fitting_job"]) / "status.json")["job_signature"],
                                      "fitted_receipt_sha256": fitted_receipt_sha,
                                      "shape_policy": "frozen_per_validated_calibration_group", "finger_pose_fitted": False}
    arrays["source.video_frame_index"] = np.arange(row["length"], dtype=np.int64)
    metadata["training_eligible"] = True
    provenance = {"source_type": "egosuite", "source_path": str(source), "source_identity": source_before,
                  "source_motion_sha256": arrays_sha256(source_arrays), "fitted_path": str(fitted_path),
                  "fitted_identity": fitted_before, "fitted_receipt_path": row["fitted_receipt_path"],
                  "fitted_receipt_sha256": fitted_receipt_sha, "fitted_receipt_identity": fitted_receipt_before}
    return arrays, metadata, provenance, source, np.arange(row["length"], dtype=np.int64)


def convert_hm(row, plan, output):
    from egomotion_pipeline.smplh_hm_parameters import export_hm_parameters
    check_inventory_identity(row["source_path"], row["source_identity"])
    check_inventory_identity(row["vision_source_path"], row["vision_identity"])
    check_inventory_identity(row["video_identity"]["path"], row["video_identity"])
    take = row["take_metadata"]
    result = export_hm_parameters(plan["hm_root"], take, plan["model_path"],
                                 episode_index=row["episode_index"], index_offset=row["dataset_from_index"])
    arrays, metadata = result.arrays, result.metadata
    metadata["parameter_provenance"] = {"source": "egoexo4d_hm", "shape_policy": "released_per_frame16_preserved",
                                       "finger_pose_encoding": "released_PCA_left45_right45", "body_refitted": False}
    metadata["frame_alignment"] = result.alignment
    metadata["training_eligible"] = bool(result.alignment.get("verified", False)
            and metadata.get("coordinate_verified", False)
            and metadata.get("canonical_training_eligible", False)
            and not metadata.get("quarantine_reasons"))
    provenance = {"source_type": "egoexo4d_hm", "source_path": row["source_path"],
                  "source_identity": identity(row["source_path"]), "source_npz_sha256": file_sha(row["source_path"]),
                  "take_metadata": take, "frame_alignment": result.alignment}
    return arrays, metadata, provenance, Path(row["vision_source_path"]), arrays["source.video_frame_index"]


def convert_one(row, plan, output):
    started = time.monotonic(); root = Path(output); index = row["episode_index"]
    production = root.resolve() == Path(plan["output_root"]).resolve()
    if production: assert_frozen_job(plan["job_dir"], code_only=True)
    record_path = root / ".pipeline/episodes" / f"{index:08d}.json"
    final = root / "episodes" / f"{index:08d}"
    if record_path.exists():
        record = read_json(record_path); receipt = read_json(final / "receipt.json")
        validate_commit(final, receipt, row, plan, record)
        if production: assert_frozen_job(plan["job_dir"], code_only=True)
        return {"episode_index": index, "source": row["source"], "frames": row["length"], "resumed": True,
                "elapsed_s": time.monotonic()-started}
    if final.exists():
        # A complete atomic episode may outlive the final record write after a crash.
        receipt = read_json(final / "receipt.json")
        validate_commit(final, receipt, row, plan)
        if production: assert_frozen_job(plan["job_dir"], code_only=True)
        atomic_json(record_path, receipt["record"])
        return {"episode_index": index, "source": row["source"], "frames": row["length"], "resumed": True,
                "elapsed_s": time.monotonic()-started}
    stage = root / ".staging" / f"{index:08d}-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    try:
        arrays, metadata, provenance, vision_source, source_frames = (
            convert_ego(row, plan, output) if row["source"] == "egosuite" else convert_hm(row, plan, output))
        if len(arrays["timestamp"]) != row["length"]: raise ValueError("Output length differs from frozen inventory")
        motion = write_parameter_table(stage / "motion.parquet", arrays)
        with (stage / "motion.parquet").open("rb") as handle: os.fsync(handle.fileno())
        vision = copy_vision(vision_source, stage / "vision.parquet", arrays, source_frames)
        if row["source"] == "egosuite":
            extraction = deepcopy(row["source_record"]["dino"])
            extraction_sha = digest_json(extraction)
        else:
            extraction_path = vision_source.with_suffix(".json")
            extraction_before = identity(extraction_path)
            extraction = read_json(extraction_path); extraction_sha = file_sha(extraction_path)
            if identity(extraction_path) != extraction_before:
                raise ValueError("DINO extraction metadata changed during copying")
        if extraction.get("state") != "complete" or extraction.get("encoder", {}).get("shape") != [5, 1024]:
            raise ValueError("Source DINO extraction metadata is incomplete or incompatible")
        record = {"episode_index": index, "source": row["source"], "source_key": row["source_key"],
                  "source_uid": row.get("source_uid", row["source_key"].split(":",1)[-1]),
                  "source_episode_index": row.get("source_episode_index"), "length": row["length"],
                  "dataset_from_index": row["dataset_from_index"], "dataset_to_index": row["dataset_to_index"],
                  "split": row["split"], "split_group_ids": row.get("split_group_ids", []),
                  "data_path": str((final / "motion.parquet").relative_to(root)),
                  "vision_path": str((final / "vision.parquet").relative_to(root)),
                  "metadata": metadata, "source_provenance": provenance,
                  "dino": {"state": "complete", "shape": [5,1024], "preserved_source_features": True,
                           "source_extraction": extraction, "source_extraction_metadata_sha256": extraction_sha,
                           "source_signature": extraction["signature"], "encoder": extraction["encoder"],
                           "token_order": ["cls", "register_0", "register_1", "register_2", "register_3"],
                           "flatten_order": "token_major"},
                  "motion_sha256": motion["sha256"], "vision_sha256": vision["sha256"]}
        if row["source"] == "egosuite":
            if identity(row["source_path"]) != provenance["source_identity"] or identity(row["fitted_path"]) != provenance["fitted_identity"]:
                raise ValueError("Fitting/source changed during parameter migration")
            if (identity(row["fitted_receipt_path"]) != provenance["fitted_receipt_identity"]
                    or file_sha(row["fitted_receipt_path"]) != provenance["fitted_receipt_sha256"]):
                raise ValueError("Fitted receipt changed during parameter migration")
        elif identity(row["source_path"]) != provenance["source_identity"]:
            raise ValueError("HM source changed during parameter migration")
        receipt = {"schema_version": PARAMETER_SCHEMA_VERSION, "record": record, "motion": motion,
                   "vision": vision, "source_provenance": provenance, "completed_at": now(),
                   "conversion_binding": conversion_binding(row, plan),
                   "output_identities": {role: identity(stage / f"{role}.parquet") for role in ("motion", "vision")}}
        atomic_json(stage / "receipt.json", receipt)
        if production: assert_frozen_job(plan["job_dir"], code_only=True)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.rename(stage, final)
        sync_directory(final.parent)
        atomic_json(record_path, record)
        return {"episode_index": index, "source": row["source"], "frames": row["length"], "resumed": False,
                "elapsed_s": time.monotonic()-started, "motion_sha256": motion["sha256"], "vision_sha256": vision["sha256"]}
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def run(job, source, workers, limit=None, output=None):
    job=Path(job); plan=read_json(job / "plan.json")
    if output is None: assert_frozen_job(job)
    root=initialize_archive(job, output)
    inventory_path=job / "inventory" / f"{source}.json"
    if not inventory_path.exists() and source != "egosuite":
        raise FileNotFoundError("HM conversion requires its frozen full inventory")
    inventory = read_json(inventory_path) if inventory_path.exists() else make_ego_inventory(job)
    rows=inventory["rows"]
    missing=[]
    if source == "egosuite":
        missing=[row["episode_index"] for row in rows if not Path(row["fitted_path"]).exists()]
        rows=[row for row in rows if Path(row["fitted_path"]).exists()]
    if limit is not None: rows=rows[:limit]
    failures=[]; completed=[]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(convert_one,row,plan,str(root)):row for row in rows}
        for future in as_completed(futures):
            row=futures[future]
            try:
                result=future.result(); completed.append(result)
                event={"event":"parameter_episode","at":now(),**result,"completed":len(completed),"selected":len(rows)}
            except Exception as exc:
                event={"event":"worker_failed","at":now(),"episode_index":row["episode_index"],
                       "source":row["source"],"error":repr(exc),"traceback":traceback.format_exc()}; failures.append(event)
            print(json.dumps(event,ensure_ascii=False),flush=True)
            with (job / "logs/migration-events.jsonl").open("a") as handle: handle.write(json.dumps(event,ensure_ascii=False)+"\n")
            atomic_json(job / f"status-{source}.json", {"state":"migrating","source":source,"completed":len(completed),
                        "selected":len(rows),"expected":len(inventory["rows"]),"completed_frames":sum(r["frames"] for r in completed),
                        "failures":len(failures),"updated_at":now(),"output_root":str(root)})
    report={"source":source,"selected":len(rows),"completed":len(completed),"frames":sum(r["frames"] for r in completed),
            "expected":len(inventory["rows"]),"full_population_complete":len(completed)==len(inventory["rows"]) and not failures,
            "missing_fitted_episode_indices":missing,"failures":failures,"output_root":str(root),"completed_at":now()}
    atomic_json(job / f"reports/migration-{source}.json",report)
    atomic_json(job / f"status-{source}.json", {**report,"state":"failed" if failures else
                ("converted" if report["full_population_complete"] else "waiting_for_source"),"updated_at":now()})
    if output is None: assert_frozen_job(job)
    if failures: raise RuntimeError(f"{len(failures)} parameter migration episodes failed")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("inventory-egosuite");p.add_argument("--job",required=True)
    p=sub.add_parser("initialize");p.add_argument("--job",required=True);p.add_argument("--output")
    p=sub.add_parser("finalize");p.add_argument("--job",required=True)
    p=sub.add_parser("run");p.add_argument("--job",required=True);p.add_argument("--source",choices=["egosuite","egoexo4d_hm"],required=True)
    p.add_argument("--workers",type=int,default=12);p.add_argument("--limit",type=int);p.add_argument("--output")
    args=parser.parse_args()
    if args.command=="inventory-egosuite":
        value=make_ego_inventory(args.job);print(json.dumps({k:v for k,v in value.items() if k!="rows"},ensure_ascii=False))
    elif args.command=="initialize": print(initialize_archive(args.job,args.output))
    elif args.command=="finalize": print(json.dumps(finalize_archive(args.job),ensure_ascii=False))
    else: print(json.dumps(run(args.job,args.source,args.workers,args.limit,args.output),ensure_ascii=False))


if __name__=="__main__": main()
