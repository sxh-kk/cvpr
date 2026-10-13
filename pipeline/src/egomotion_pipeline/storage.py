"""LeRobot v3 Parquet export, one episode per file, resumable episode commits."""

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .schema import (BODY_NAMES, HAND_NAMES, SCHEMA_VERSION, LEGACY_SCHEMA_VERSION,
                     DINO_TOKEN_DIM, feature_specs, dino_format)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_bytes(json_bytes(value))
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_arrays(path, arrays, specs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {}
    for key, spec in specs.items():
        value = np.asarray(arrays[key], dtype=spec["dtype"])
        shape = spec["shape"]
        if shape == [1]:
            columns[key] = pa.array(value.reshape(-1), type=pa.from_numpy_dtype(value.dtype))
        else:
            value = value.reshape(len(value), *shape)
            # Flat numeric sequences match LeRobot's datasets.Sequence representation.
            columns[key] = pa.FixedSizeListArray.from_arrays(pa.array(value.reshape(-1)), shape[0])
    table = pa.table(columns)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd", row_group_size=3000)
    os.replace(tmp, path)


def read_arrays(path, specs):
    table = pq.read_table(path)
    if set(table.column_names) != set(specs):
        raise ValueError(f"Parquet/schema columns differ: {path}")
    arrays = {}
    for key, spec in specs.items():
        array = table[key].combine_chunks()
        scalar_type = pa.from_numpy_dtype(np.dtype(spec["dtype"]))
        expected_type = scalar_type if spec["shape"] == [1] else pa.list_(scalar_type, spec["shape"][0])
        if array.type != expected_type or array.null_count or (spec["shape"] != [1] and array.values.null_count):
            raise ValueError(f"Parquet feature dtype/shape/null mismatch: {key}: {path}")
        if spec["shape"] == [1]:
            arrays[key] = array.to_numpy(zero_copy_only=False).copy()
        else:
            arrays[key] = array.values.to_numpy(zero_copy_only=False).reshape(len(table), *spec["shape"]).copy()
    return arrays


def archive_provenance(config):
    from . import __version__
    from .body_model import file_sha256
    implementation = {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {"schema_version": SCHEMA_VERSION, "config": config,
            "pipeline_version": __version__, "implementation_sha256": digest(implementation)}


def quality_report(arrays):
    result = {key.removeprefix("observation.valid."): float(np.mean(value))
              for key, value in arrays.items() if key.startswith("observation.valid.")}
    body = arrays["observation.state.body_world"].reshape(-1, 22, 3)
    for hand, wrist in [("hand_left", 20), ("hand_right", 21)]:
        hand_pos = arrays[f"observation.state.{hand}_world"].reshape(-1, 21, 3)
        mask = arrays["observation.valid.body_position"][:, wrist] & arrays[f"observation.valid.{hand}_position"][:, 0]
        error = np.linalg.norm(body[mask, wrist]-hand_pos[mask, 0], axis=-1)
        result[f"{hand}_wrist_offset_median_m"] = float(np.median(error)) if len(error) else None
        result[f"{hand}_wrist_offset_p95_m"] = float(np.quantile(error, .95)) if len(error) else None
    return result


class Archive:
    def __init__(self, root, config=None, *, _allow_upgrade=False):
        self.root = Path(root)
        self._records = None
        self._by_key = {}
        self.root.mkdir(parents=True, exist_ok=True)
        upgrade = self.root / ".pipeline/dino_upgrade.json"
        if not _allow_upgrade and upgrade.exists() and read_json(upgrade).get("state") != "complete":
            raise ValueError("DINO format upgrade is unfinished; resume upgrade-dino before reading/writing this archive")
        provenance = self.root / ".pipeline/config.json"
        self.schema_version = read_json(provenance).get("schema_version") if provenance.exists() else SCHEMA_VERSION
        if self.schema_version not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
            raise ValueError("Archive schema is incompatible with camera-floor dense22; reconvert to a new output directory")
        self.specs = feature_specs(DINO_TOKEN_DIM) if self.schema_version == LEGACY_SCHEMA_VERSION else feature_specs()
        if config is not None:
            requested = archive_provenance(config)
            if provenance.exists() and read_json(provenance) != requested:
                raise ValueError("Output contains a different config/schema; choose a new output directory")
            if not provenance.exists():
                if any(self.root.iterdir()):
                    raise ValueError("Refusing to initialize a non-empty, unrecognized output directory")
                atomic_json(provenance, requested)
        elif not provenance.exists():
            raise ValueError("Not an egomotion pipeline archive")

    @contextlib.contextmanager
    def lock(self):
        with (self.root/".pipeline/lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another process is writing this archive") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def records(self):
        if self._records is None:
            self._records = [read_json(p) for p in sorted((self.root/".pipeline/episodes").glob("*.json"))]
            self._by_key = {r["source_key"]: r for r in self._records}
        return self._records

    def data_path(self, index):
        return self.root / f"data/chunk-{index//1000:03d}/file-{index%1000:03d}.parquet"

    def record_path(self, index):
        return self.root / f".pipeline/episodes/{index:08d}.json"

    def read_window(self, episode_index, *, start_frame, num_frames, fps=15, canonicalize=True):
        """Read a selected dense22 window and rebuild its floor-dependent fields."""
        from .windows import prepare_window
        if isinstance(episode_index, bool) or not isinstance(episode_index, (int, np.integer)) or not 0 <= episode_index < len(self.records()):
            raise ValueError("episode_index is outside this archive")
        record = self.records()[episode_index]
        arrays = read_arrays(self.data_path(episode_index), self.specs)
        return prepare_window(arrays, record["metadata"], start_frame=start_frame,
                              num_frames=num_frames, fps=fps, canonicalize=canonicalize)

    def add(self, episode, arrays, input_fingerprint):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Legacy archive cannot accept new-schema frames; run upgrade-dino first")
        records = self.records()
        key = episode.source+":"+episode.uid
        old = self._by_key.get(key)
        if old:
            if old["input_fingerprint"] != input_fingerprint:
                raise ValueError(f"{key}: input changed since previous conversion")
            if not self.data_path(old["episode_index"]).exists():
                raise ValueError(f"{key}: committed Parquet is missing")
            return False
        index = len(records)
        count = len(arrays["timestamp"])
        start = records[-1]["dataset_to_index"] if records else 0
        arrays.update({"index": np.arange(start, start+count, dtype=np.int64),
                       "frame_index": np.arange(count, dtype=np.int64),
                       "episode_index": np.full(count, index, np.int64), "task_index": np.zeros(count, np.int64)})
        write_arrays(self.data_path(index), arrays, self.specs)
        record = {"episode_index": index, "source_key": key, "source": episode.source, "source_uid": episode.uid,
                  "length": count, "dataset_from_index": start, "dataset_to_index": start+count,
                  "input_fingerprint": input_fingerprint, "metadata": episode.metadata, "video": episode.video,
                  "quality": quality_report(arrays), "dino": {"state": "pending"}}
        atomic_json(self.record_path(index), record)
        records.append(record)
        self._by_key[key] = record
        return True

    def finalize(self):
        records = self.records()
        if not records:
            return
        rows = []
        for record in records:
            idx = record["episode_index"]
            rows.append({"episode_index": idx, "tasks": ["motion"], "length": record["length"],
                         "dataset_from_index": record["dataset_from_index"], "dataset_to_index": record["dataset_to_index"],
                         "data/chunk_index": idx//1000, "data/file_index": idx%1000,
                         "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
                         "source": record["source"], "source_uid": record["source_uid"],
                         "coordinate_verified": record["metadata"].get("coordinate_verified", False),
                         "rotation_basis_verified": record["metadata"].get("rotation_basis_verified", False),
                         "floor_height": record["metadata"].get("ground", {}).get("height_m", 0.),
                         "floor_valid": record["metadata"].get("ground", {}).get("valid", False),
                         "floor_verified": record["metadata"].get("ground", {}).get("verified", False),
                         "head_reference_basis_verified": record["metadata"].get("head_reference", {}).get("basis_verified", False),
                         "dino_ready": record["dino"]["state"] == "complete"})
        meta = self.root/"meta"
        ep_path = meta/"episodes/chunk-000/file-000.parquet"
        ep_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), ep_path.with_suffix(".tmp"), compression="zstd")
        os.replace(ep_path.with_suffix(".tmp"), ep_path)
        # LeRobot requires a task table. A constant structural label carries no source annotations.
        schema_meta = {"index_columns": ["task"], "column_indexes": [], "columns": [
            {"name": "task_index", "field_name": "task_index", "pandas_type": "int64", "numpy_type": "int64", "metadata": None},
            {"name": "task", "field_name": "task", "pandas_type": "unicode", "numpy_type": "object", "metadata": None}],
            "creator": {"library": "pyarrow", "version": pa.__version__}, "pandas_version": "2.0.0"}
        tasks = pa.table({"task_index": pa.array([0], pa.int64()), "task": ["motion"]})
        tasks = tasks.replace_schema_metadata({b"pandas": json.dumps(schema_meta).encode()})
        pq.write_table(tasks, meta/"tasks.parquet.tmp")
        os.replace(meta/"tasks.parquet.tmp", meta/"tasks.parquet")
        info = {"codebase_version": "v3.0", "robot_type": "human", "fps": 30,
                "total_episodes": len(records), "total_frames": sum(r["length"] for r in records), "total_tasks": 1,
                "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 200,
                "splits": {"all": f"0:{len(records)}"}, "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": None, "features": self.specs}
        atomic_json(meta/"info.json", info)
        atomic_json(meta/"ontology.json", {"body_joints": BODY_NAMES, "hand_joints": HAND_NAMES,
                   "schema_version": self.schema_version,
                   "dino": dino_format() if self.schema_version == SCHEMA_VERSION else {
                       "shape": [1, DINO_TOKEN_DIM], "flat_shape": [DINO_TOKEN_DIM], "token_order": ["cls"],
                       "storage_dtype": "float32", "status": "legacy_cls_only"},
                   "position_unit": "meter", "target_world_axes": "X forward, Y left, Z up; require coordinate_verified",
                   "rotation": "wxyz scalar-first, global, maps joint-local column vectors to world",
                   "body_shape": [22, 3], "body_rotation_shape": [22, 4], "hand_shape": [21, 3],
                   "invalid_position": "zero + false mask", "invalid_rotation": "identity + false mask",
                   "joint_local_axes": "source native unless a verified body_basis_correction is configured",
                   "video_storage": "original paths/offsets in .pipeline episode provenance; no video feature is declared",
                   "head_semantics": "head_world preserves source head/CPF; head_reference_world is EE4D reference at camera origin",
                   "head_reference_floor": "absolute world XY, z=floor_height, yaw from reference local +X projected to XY",
                   "head_reference_axes": "camera-optical-center reference: X left, Y up, Z optical-forward; source-specific axis conversion",
                   "body_head_floor": "22 positions: R_world_floor.T @ (body_world - camera_floor_world), meters; Z height above floor",
                   "body_head_floor_valid": "body_position AND projected-camera position/rotation validity; independent of body rotation basis",
                   "training_trajectory": "canonicalize each selected training window and compute relative trajectory only after training-FPS selection",
                   "training_floor_fallback": "Archive.read_window resolves failed floor estimates from selected valid feet mean Z minus 0.02m, then rebuilds local fields",
                   "floor": "horizontal plane in output world; estimated or annotated; validity separate from verification"})
