"""Audit new native camera quality only; inherited FK/hand QA is not repeated."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from .camera_quality import (CameraQualityPolicy, SOURCE_POLICIES, camera_metrics,
                            evaluate_metrics, nearest_indices, interval_valid)
from .dataset.smplh_archive import file_sha256
from .dataset.window_cache import atomic_json, digest

AUDIT_FORMAT = "smplh-camera-quality-audit-v1"
NATIVE_POINT = "observation.valid.camera_tracking"
NATIVE_EDGE = "observation.valid.camera_transition"
COLUMNS = ("timestamp", "frame_index", "observation.state.camera_world", "observation.state.camera_world_rotation",
           "observation.valid.camera_position", "observation.valid.camera_rotation",
           "observation.valid.head_reference_position", "observation.valid.head_reference_rotation",
           "observation.valid.head_reference_basis")


def _stamp(path):
    s = Path(path).stat()
    return {"bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def _columns(table):
    out = {}
    for name in table.column_names:
        arr = table[name].combine_chunks()
        if arr.null_count: raise ValueError("Null camera audit input")
        if hasattr(arr, "values"):
            out[name] = arr.values.to_numpy(zero_copy_only=False).reshape(len(arr), arr.type.list_size)
        else: out[name] = arr.to_numpy(zero_copy_only=False)
    return out


def _trajectory(path, *, open_loop=False):
    """Arrow CSV column projection avoids Python objects per 1kHz source sample."""
    import pyarrow as pa
    import pyarrow.csv as csv
    axes = "odometry" if open_loop else "world"
    identity = "session_uid" if open_loop else "graph_uid"
    velocity = "odometry" if open_loop else "device"
    names = ["tracking_timestamp_us", identity, "quality_score",
             *[f"device_linear_velocity_{x}_{velocity}" for x in "xyz"],
             *[f"angular_velocity_{x}_device" for x in "xyz"]]
    if not open_loop:
        names += [f"t{x}_world_device" for x in "xyz"] + [f"q{x}_world_device" for x in "wxyz"]
    types = {name: pa.float64() for name in names if name != identity}
    types[identity] = pa.string()
    types["tracking_timestamp_us"] = pa.int64()
    table = csv.read_csv(path, read_options=csv.ReadOptions(use_threads=False),
        convert_options=csv.ConvertOptions(include_columns=names, column_types=types))
    data = {name: table[name].combine_chunks().to_numpy(zero_copy_only=False) for name in names}
    times = data["tracking_timestamp_us"] * 1e-6
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("MPS timestamps missing or not strictly increasing")
    quality = data["quality_score"]
    linear = np.linalg.norm(np.stack([data[f"device_linear_velocity_{x}_{velocity}"] for x in "xyz"], -1), axis=-1)
    angular = np.linalg.norm(np.stack([data[f"angular_velocity_{x}_device"] for x in "xyz"], -1), axis=-1)
    good = (np.isfinite(quality) & (quality > 0) & (quality <= 1)
            & np.isfinite(linear) & np.isfinite(angular))
    result = {"time": times, "time_ns": data["tracking_timestamp_us"] * 1000, "id": data[identity], "good": good,
              "linear": linear, "angular": angular, "quality": quality}
    if not open_loop:
        result["position"] = np.stack([data[f"t{x}_world_device"] for x in "xyz"], -1)
        result["quaternion"] = np.stack([data[f"q{x}_world_device"] for x in "wxyz"], -1)
        good &= (np.isfinite(result["position"]).all(-1) & np.isfinite(result["quaternion"]).all(-1)
                 & (np.linalg.norm(result["quaternion"], axis=-1) > 1e-8))
    return result


def _schema_evidence(path):
    """Inspect actual source descriptors; no confidence value is invented."""
    from google.protobuf.descriptor_pb2 import FileDescriptorSet
    from mcap.reader import make_reader
    with Path(path).open("rb") as stream:
        summary = make_reader(stream).get_summary()
    if summary is None: raise ValueError("MCAP needs source schema summary")
    topics = {c.topic: c for c in summary.channels.values()}
    topic = "/sensor/camera/head_left/extrinsic"
    if topic not in topics: raise ValueError("MCAP camera extrinsic channel missing")
    schema = summary.schemas[topics[topic].schema_id]
    if schema.name != "foxglove.FrameTransforms":
        raise ValueError("Unreviewed camera confidence schema; refuse assumed confidence policy")
    parsed = FileDescriptorSet(); parsed.ParseFromString(schema.data)
    fields = {}
    for file in parsed.file:
        if file.package == "foxglove":
            for message in file.message_type:
                if message.name in {"FrameTransforms", "FrameTransform"}:
                    fields[message.name] = [f.name for f in message.field]
    if set(fields.get("FrameTransforms", [])) != {"transforms"} or set(fields.get("FrameTransform", [])) != {
            "timestamp", "parent_frame_id", "child_frame_id", "translation", "rotation"}:
        raise ValueError("Changed camera source schema needs an explicit confidence policy")
    return {"camera_topic": topic, "camera_schema": schema.name,
            "camera_schema_sha256": hashlib.sha256(schema.data).hexdigest(),
            "camera_fields": fields, "confidence_observed": False,
            "source_stat": _stamp(path), "source_path": str(path)}


def _source_metrics(record, arrays, policy):
    source, metadata = record["source"], record["metadata"]
    native = np.logical_and.reduce([arrays[name] for name in (
        "observation.valid.camera_position", "observation.valid.camera_rotation",
        "observation.valid.head_reference_position", "observation.valid.head_reference_rotation",
           "observation.valid.head_reference_basis")])
    times = arrays.get("source.capture_timestamp_ns", arrays["timestamp"].astype(np.float64)*1e9) * 1e-9
    metrics = camera_metrics(arrays["observation.state.camera_world"],
                             arrays["observation.state.camera_world_rotation"], times, native)
    if source == "egosuite":
        evidence = _schema_evidence(metadata["source_data"])
        return metrics, evidence
    if source != "egoexo4d_hm": raise ValueError("Unsupported camera source")
    provenance = metadata.get("aria_provenance", {})
    closed_path = Path(provenance["trajectory"]["path"])
    open_path = closed_path.with_name("open_loop_trajectory.csv")
    before = {str(p): _stamp(p) for p in (closed_path, open_path)}
    # The accepted source CSV digest is rechecked, not assumed from a filename.
    closed_sha = file_sha256(closed_path)
    if closed_sha != provenance["trajectory"].get("sha256"):
        raise ValueError("Original registered MPS trajectory changed since archive conversion")
    open_sha = file_sha256(open_path)
    closed, opened = _trajectory(closed_path), _trajectory(open_path, open_loop=True)
    if "source.capture_timestamp_ns" not in arrays:
        raise ValueError("Aria camera quality requires original integer capture_timestamp_ns")
    capture_ns = arrays["source.capture_timestamp_ns"]
    ci, oi = nearest_indices(closed["time_ns"], capture_ns), nearest_indices(opened["time_ns"], capture_ns)
    metrics.update(closed_score_good=closed["good"][ci], open_score_good=opened["good"][oi],
        closed_time_good=abs(closed["time_ns"][ci]-capture_ns) <= round(policy.max_pose_error_s*1e9),
        open_time_good=abs(opened["time_ns"][oi]-capture_ns) <= round(policy.max_pose_error_s*1e9),
        closed_interval_good=interval_valid(ci, closed["good"], closed["time_ns"], closed["id"], round(policy.max_mps_gap_s*1e9)),
        open_interval_good=interval_valid(oi, opened["good"], opened["time_ns"], opened["id"], round(policy.max_mps_gap_s*1e9)))
    # The RGB optical center has a lever arm relative to the device origin.
    arm = np.linalg.norm(np.asarray(provenance["device_camera_transform"], np.float64)[:3, 3])
    angular = np.maximum(closed["angular"][ci], opened["angular"][oi])
    linear = np.maximum(closed["linear"][ci], opened["linear"][oi]) + arm * angular
    metrics["source_linear_speed_bound_m_s"] = np.maximum(linear, np.r_[linear[:1], linear[:-1]])
    metrics["source_angular_speed_bound_rad_s"] = np.maximum(angular, np.r_[angular[:1], angular[:-1]])
    metrics.update(closed_quality=closed["quality"][ci], open_quality=opened["quality"][oi],
                   closed_time_error_s=(closed["time_ns"][ci]-capture_ns)*1e-9, open_time_error_s=(opened["time_ns"][oi]-capture_ns)*1e-9)
    from .dataset.representation import quaternion_matrix
    rotation = quaternion_matrix(closed["quaternion"][ci])
    source_camera = closed["position"][ci] + np.einsum("tij,j->ti", rotation,
        np.asarray(provenance["device_camera_transform"], np.float64)[:3, 3])
    good_identity = native & metrics["closed_time_good"]
    max_error = float(np.linalg.norm(source_camera[good_identity]-arrays["observation.state.camera_world"][good_identity], axis=-1).max()) if good_identity.any() else 0.
    if max_error > .02: raise ValueError(f"MPS source/archived camera identity differs: {max_error}")
    if any(_stamp(p) != before[str(p)] for p in (closed_path, open_path)):
        raise ValueError("MPS source changed during camera audit")
    evidence = {"confidence_observed": True, "source_paths": {
        "closed_loop": {"path": str(closed_path), "sha256": closed_sha, **before[str(closed_path)]},
        "open_loop": {"path": str(open_path), "sha256": open_sha, **before[str(open_path)]}},
        "closed_samples": len(closed["time"]), "open_samples": len(opened["time"]),
        "closed_quality_zero_samples": int((closed["quality"] == 0).sum()),
        "open_quality_zero_samples": int((opened["quality"] == 0).sum()),
        "closed_graph_count": len(set(closed["id"])), "open_session_count": len(set(opened["id"])),
        "source_camera_position_max_identity_error_m": max_error,
        "source_camera_position_identity_tolerance_m": .02,
        "open_loop_registered_as_world": False,
        "nearest_sampling": "integer_ns_closest_exact_tie_after_matches_MPS_1.5.5_Cpp_QueryMapByTimestamp"}
    return metrics, evidence


def audit_episode(task):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    root, output = Path(task["root"]), Path(task["output"])
    record, policy = task["record"], CameraQualityPolicy(**task["policy"])
    index = record["episode_index"]
    record_path = root / f".pipeline/episodes/{index:08d}.json"
    record_bytes = record_path.read_bytes()
    if json.loads(record_bytes) != record: raise ValueError("Camera audit parent record changed")
    motion = root / record["data_path"]
    before = _stamp(motion)
    receipt = json.loads((root / f"episodes/{index:08d}/receipt.json").read_text())
    if before != {k: receipt["output_identities"]["motion"][k] for k in before}:
        raise ValueError("Camera audit parent motion changed")
    file = pq.ParquetFile(motion)
    names = [*COLUMNS]
    if "source.capture_timestamp_ns" in file.schema_arrow.names: names.append("source.capture_timestamp_ns")
    arrays = _columns(file.read(columns=names, use_threads=False))
    if not np.array_equal(arrays["frame_index"], np.arange(record["length"])):
        raise ValueError("Camera audit native frame identity differs")
    metrics, evidence = _source_metrics(record, arrays, policy)
    result = evaluate_metrics(metrics, source=record["source"], policy=policy)
    point, edge = result["quality_valid"], result["transition_valid"]
    # Mark changes to old selected-endpoint semantics even without new raw masks.
    native = metrics["native_valid"]
    old_edge = native.copy(); old_edge[1:] &= native[:-1]
    new_rejected = native & ~point
    edge_rejected = old_edge & ~edge
    gaps = np.flatnonzero(~native)
    representation_changed = bool(len(gaps))
    changed = bool(new_rejected.any() or edge_rejected.any() or representation_changed)
    path = output / f"episode_{index:08d}.npz"
    if path.exists() or path.with_suffix(".json").exists(): raise FileExistsError("Camera audit sidecar immutable")
    saved = {**metrics, **result}
    with path.open("xb") as stream:
        np.savez_compressed(stream, **saved)
        stream.flush(); os.fsync(stream.fileno())
    summary = {"native_valid_frames": int(native.sum()), "quality_valid_frames": int(point.sum()),
        "newly_rejected_frames": int(new_rejected.sum()), "newly_rejected_native_edges": int(edge_rejected.sum()),
        "invalid_native_frames": int((~native).sum()), "motion_revision_required": changed,
        "representation_changed_due_native_continuity": representation_changed,
        "reason_counts": {str(bit): int(((result["reasons"] & bit) != 0).sum()) for bit in (1,2,4,8,16,32,64,128,256,512,1024,2048,4096)}}
    value = {"format": AUDIT_FORMAT, "episode_index": index, "source": record["source"],
        "source_key": record["source_key"], "split": record["split"], "length": record["length"],
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(), "motion_sha256": record["motion_sha256"],
        "motion_stat": before, "policy": asdict(policy), "source_policy": SOURCE_POLICIES[record["source"]],
        "source_evidence": evidence, "mask_path": path.name, "mask_sha256": file_sha256(path),
        "summary": summary, "completed_at": datetime.now(timezone.utc).isoformat()}
    if record_bytes != record_path.read_bytes() or _stamp(motion) != before: raise ValueError("Camera audit parent modified")
    atomic_json(path.with_suffix(".json"), value)
    return value


def run_audit(root, output, *, workers=8, episode_indices=None):
    from .dataset.smplh_archive import SMPLHParameterArchive
    if not 1 <= workers <= 8: raise ValueError("Use1..8 camera workers")
    archive = SMPLHParameterArchive(root)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError("Use a fresh camera audit directory")
    output.mkdir(parents=True, exist_ok=True)
    records = archive.records()
    selected = records if episode_indices is None else [records[i] for i in episode_indices]
    policy = asdict(CameraQualityPolicy())
    binding = {"camera_quality.py": file_sha256(Path(__file__).with_name("camera_quality.py")),
               "audit_camera_quality.py": file_sha256(__file__)}
    plan = {"format": AUDIT_FORMAT, "archive_root": str(archive.root), "binding": archive.binding,
            "policy": policy, "code_binding": binding, "episodes_planned": len(selected),
            "diagnostic": episode_indices is not None, "production_modified": False,
            "camera_quality_scope": "all_archive_episodes_including_parent_quarantines",
            "source_population": {"scope": "all_archive", "total": len(records),
                "episodes": len(records), "sources": {source: sum(r["source"]==source for r in records) for source in SOURCE_POLICIES}}}
    atomic_json(output / "plan.json", plan)
    entries, began = [], time.monotonic()
    tasks = [{"root": str(archive.root), "output": str(output), "record": record, "policy": policy} for record in selected]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        iterator, pending = iter(tasks), set()
        while pending or iterator is not None:
            while iterator is not None and len(pending)<workers:
                task = next(iterator, None)
                if task is None: iterator = None; break
                pending.add(pool.submit(audit_episode, task))
            if not pending: break
            done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                pending.remove(future); entry = future.result(); entries.append(entry)
                state = {"state": "auditing", "completed": len(entries), "total": len(selected),
                         "latest_episode": entry["episode_index"], "elapsed_s": time.monotonic()-began,
                         "updated_at": datetime.now(timezone.utc).isoformat()}
                atomic_json(output / "progress.json", state)
                print(json.dumps(state), flush=True)
    archive.assert_current()
    report = {**plan, "state": "complete", "episodes": sorted(entries, key=lambda e:e["episode_index"]),
              "episodes_completed": len(entries), "finished_at": datetime.now(timezone.utc).isoformat(),
              "inherited_FK_numerical_QA_repeated": False, "DINO_reextracted": False,
              "changed_episode_indices": sorted(e["episode_index"] for e in entries if e["summary"]["motion_revision_required"])}
    atomic_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-root", required=True); p.add_argument("--output-root", required=True)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--episodes", type=int, nargs="+")
    a = p.parse_args(); run_audit(a.archive_root, a.output_root, workers=a.workers, episode_indices=a.episodes)


if __name__ == "__main__": main()
