"""Read-only HM source quality audit; writes an independent, bound mask sidecar.

This command neither publishes a dataset nor mutates archive/cache/statistics.
Use an empty output directory and a small worker count beside active training.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from .hm_quality import HMQualityPolicy, HMQualityReason, audit_camera_motion, score_hm_fit, summarize_hm_quality


FORMAT = "egoexo4d-hm-fit-quality-audit-v1"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _numpy_column(table, name, width=1):
    column = table[name].combine_chunks()
    if column.null_count:
        raise ValueError(f"Nulls in required archive column {name}")
    if width == 1:
        return np.asarray(column.to_numpy(zero_copy_only=False))
    return np.asarray(column.values.to_numpy(zero_copy_only=False)).reshape(len(table), width)


def _assert_source_stat(path, identity):
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (identity.get("bytes"), identity.get("mtime_ns")):
        raise ValueError(f"Source identity changed: {path}")
    return stat.st_size, stat.st_mtime_ns


def audit_episode(archive_root, record_path, output_root, policy):
    """Audit one original fitted source joined to its committed camera rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    root, record_path, output = Path(archive_root), Path(record_path), Path(output_root)
    record_bytes = record_path.read_bytes()
    record = json.loads(record_bytes)
    if record["source"] != "egoexo4d_hm":
        raise ValueError("HM quality policy must not screen EgoSuite")
    metadata = record["metadata"]
    if not metadata.get("coordinate_verified"):
        raise ValueError("Nose/camera comparison requires established world registration")
    identity = metadata["source_identity"]
    source = Path(metadata["source_data"])
    if str(source) != identity.get("path"):
        raise ValueError("Source path differs from committed identity")
    source_stamp = _assert_source_stat(source, identity)
    motion_path = (root / record["data_path"]).resolve(strict=True)
    if not motion_path.is_relative_to(root.resolve()):
        raise ValueError("Committed motion path escapes archive")
    motion_stat = motion_path.stat()
    columns = ["timestamp", "observation.state.camera_world", "observation.valid.camera_position",
               "observation.source_frame_left", "observation.source_frame_right",
               "observation.source_interpolation_weight"]
    table = pq.read_table(motion_path, columns=columns, use_threads=False)
    if len(table) != record["length"]:
        raise ValueError("Archive camera row count differs from committed length")
    left = _numpy_column(table, "observation.source_frame_left")
    right = _numpy_column(table, "observation.source_frame_right")
    weight = _numpy_column(table, "observation.source_interpolation_weight")
    if not np.array_equal(left, right) or np.any(weight != 0):
        raise ValueError("Quality audit requires exact native source frame joins; interpolation is unsupported")
    if left.dtype.kind not in "iu" or not np.array_equal(left, np.arange(len(table))):
        raise ValueError("Expected the established native30Hz identity source row join")
    camera = _numpy_column(table, "observation.state.camera_world", 3)
    camera_valid = _numpy_column(table, "observation.valid.camera_position")
    timestamp = _numpy_column(table, "timestamp")
    count = len(table)
    with np.load(source, allow_pickle=False) as npz:
        joints = npz["joints3d"]
        if joints.shape != (1, count, 67, 3):
            raise ValueError("Released joints3d must use the established single-person OpenPose67 layout")
        nose = joints[0, :, 0, :].copy()
        del joints
        native = npz["valid"]
        if native.shape != (1, count):
            raise ValueError("Native valid shape differs from released frame count")
        finite = np.ones(count, dtype=bool)
        for name, width in (("trans", 3), ("root_orient", 3), ("pose_body", 63), ("betas_per_frame", 16)):
            value = npz[name]
            if value.shape != (1, count, width):
                raise ValueError(f"Unexpected fitted parameter shape: {name}")
            finite &= np.isfinite(value[0]).all(-1)
    result = score_hm_fit(nose, camera, native_valid=native[0], camera_valid=camera_valid,
                          timestamps_s=timestamp, fit_finite=finite, policy=HMQualityPolicy(**policy))
    if _assert_source_stat(source, identity) != source_stamp:
        raise ValueError("Source changed during audit")
    after = motion_path.stat()
    if (after.st_size, after.st_mtime_ns) != (motion_stat.st_size, motion_stat.st_mtime_ns):
        raise ValueError("Archive motion changed during audit")
    if record_path.read_bytes() != record_bytes:
        raise ValueError("Archive episode record changed during audit")

    index = record["episode_index"]
    array_path = output / f"episode_{index:08d}.npz"
    temporary = output / f"episode_{index:08d}.tmp"
    arrays = result.arrays()
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    with np.load(temporary, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(
                saved[name].dtype != value.dtype or saved[name].shape != value.shape
                or not np.array_equal(saved[name], value, equal_nan=True)
                for name, value in arrays.items()):
            raise ValueError("Quality sidecar failed exact write/read verification")
    os.replace(temporary, array_path)
    receipt = {
        "format": FORMAT, "episode_index": index, "source_key": record["source_key"],
        "split": record["split"], "policy": policy, "summary": summarize_hm_quality(result),
        "source_identity": identity, "source_stat_rechecked": True,
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "motion_sha256": record["motion_sha256"],
        "motion_stat": {"bytes": motion_stat.st_size, "mtime_ns": motion_stat.st_mtime_ns},
        "mask_path": array_path.name, "mask_sha256": _sha(array_path),
        "native_valid_sha256": hashlib.sha256(np.ascontiguousarray(result.native_valid).tobytes()).hexdigest(),
        "native_valid_sha256_encoding": "bool_C_order_bytes",
        "nose_joint": "released_OpenPose67_joint0_not_Aria_CPF",
        "frame_join": "native30Hz_identity_checked",
        "finite_parameter_fields": ["trans", "root_orient", "pose_body", "betas_per_frame"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "camera_motion_audit": audit_camera_motion(camera, camera_valid, timestamp),
    }
    _write_json(output / f"episode_{index:08d}.json", receipt)
    return receipt


def _aggregate(receipts):
    groups = {}
    for receipt in receipts:
        summary = receipt["summary"]
        # Split and acquisition/activity prefixes make calibration bias visible.
        strata = ["all", "split:" + receipt["split"],
                  "activity:" + "_".join(receipt["source_key"].split(":", 1)[1].split("_")[:2])]
        for name in strata:
            group = groups.setdefault(name, {"episodes": 0, "frames": 0, "native_finite_frames": 0,
                                           "quality_valid_frames": 0, "newly_rejected_frames": 0,
                                           "reason_counts": {}, "distance_native_finite_histogram": [0] * 10})
            group["episodes"] += 1
            for key in ("frames", "native_finite_frames", "quality_valid_frames", "newly_rejected_frames"):
                group[key] += summary[key]
            for key, value in summary["reason_counts"].items():
                group["reason_counts"][key] = group["reason_counts"].get(key, 0) + value
            group["distance_native_finite_histogram"] = [
                a+b for a, b in zip(group["distance_native_finite_histogram"], summary["distance_native_finite_histogram"])]
    return groups


def _audit_tasks(executor, records, root, output, policy, workers):
    """Bound admission so result and wakeup pipes cannot deadlock on submit.

    Python 3.10's ProcessPoolExecutor may block if thousands of wakeups are
    submitted before consuming large results.  At most two tasks per worker
    are queued here, and every completion is consumed before admission resumes.
    """
    remaining, pending, exhausted = iter(records), set(), False
    try:
        while pending or not exhausted:
            while not exhausted and len(pending) < workers * 2:
                path = next(remaining, None)
                if path is None:
                    exhausted = True
                    break
                pending.add(executor.submit(audit_episode, str(root), str(path), str(output), policy))
            if not pending:
                break
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
    except BaseException:
        for future in pending:
            future.cancel()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode-indices", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--max-distance-m", type=float, default=1.)
    parser.add_argument("--warning-distance-m", type=float, default=.5)
    parser.add_argument("--calibration-report", help="Completed small-sample audit, bound as independent calibration evidence")
    args = parser.parse_args()
    policy = HMQualityPolicy(max_head_camera_distance_m=args.max_distance_m,
                             warning_head_camera_distance_m=args.warning_distance_m)
    calibration = None
    if args.calibration_report:
        calibration_path = Path(args.calibration_report).resolve(strict=True)
        calibration_source = json.loads(calibration_path.read_text())
        if (calibration_source.get("format") != FORMAT or calibration_source.get("state") != "complete"
                or not calibration_source.get("diagnostic") or calibration_source.get("policy") != asdict(policy)
                or calibration_source.get("code_binding", {}).get("hm_quality.py") != _sha(Path(__file__).with_name("hm_quality.py"))):
            raise ValueError("Calibration must be a completed diagnostic audit using the identical policy")
        calibration = {"path": str(calibration_path), "sha256": _sha(calibration_path),
                       "code_binding": calibration_source.get("code_binding"),
                       "scope_note": "Small selected controls and known outliers; not a population accuracy certificate",
                       "roles": calibration_source.get("calibration_roles", {}),
                       "episodes": [{"episode_index": entry["episode_index"], "source_key": entry["source_key"],
                                     "summary": entry["summary"]} for entry in calibration_source["episodes"]]}
    root = Path(args.archive_root).resolve(strict=True)
    selected = None if args.episode_indices is None else set(args.episode_indices)
    if selected is None:
        # The published index identifies HM without opening thousands of unrelated
        # EgoSuite records on shared storage.  Each selected record is still read
        # and independently bound below.
        import pyarrow.parquet as pq

        index_path = root / "meta/episodes.parquet"
        index = pq.read_table(index_path, columns=["episode_index", "source"], use_threads=False).to_pydict()
        indices = index["episode_index"]
        if sorted(indices) != list(range(len(indices))):
            raise ValueError("Published episode index is incomplete or duplicated")
        summary = json.loads((root / "meta/summary.json").read_text())
        hm_indices = [idx for idx, source in zip(indices, index["source"]) if source == "egoexo4d_hm"]
        if len(indices) != summary["episodes"] or len(hm_indices) != summary["sources"]["egoexo4d_hm"]["episodes"]:
            raise ValueError("Published episode index differs from archive population")
        paths = [root / f".pipeline/episodes/{idx:08d}.json" for idx in hm_indices]
    else:
        paths = [root / f".pipeline/episodes/{idx:08d}.json" for idx in sorted(selected)]
    records, excluded = [], []
    hm_count = 0
    for path in paths:
        record = json.loads(path.read_text())
        if record["episode_index"] != int(path.stem) or record["source"] != "egoexo4d_hm":
            raise ValueError("Published HM episode identity differs from its committed record")
        if record["source"] == "egoexo4d_hm":
            hm_count += 1
            metadata = record["metadata"]
            reasons = list(metadata.get("quarantine_reasons") or [])
            if metadata.get("training_eligible") is False or metadata.get("canonical_training_eligible") is False:
                reasons.append("training_ineligible")
            if not metadata.get("coordinate_verified"):
                reasons.append("world_registration_unverified")
            if reasons:
                excluded.append({"episode_index": record["episode_index"], "source_key": record["source_key"],
                                 "reasons": sorted(set(reasons))})
            else:
                records.append(path)
    if not records or (selected is not None and {int(path.stem) for path in records} != selected):
        raise ValueError("Requested HM episodes are missing or include a different source")
    output = Path(args.output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Audit output must be outside the immutable archive")
    output.mkdir(parents=True, exist_ok=False)
    report = {"format": FORMAT, "state": "running", "archive_root": str(root),
              "code_binding": {name: _sha(Path(__file__).with_name(name))
                               for name in ("hm_quality.py", "audit_hm_quality.py")},
              "calibration_evidence": calibration,
              "diagnostic": selected is not None,
              "source_hm_population": {"scope": "selected" if selected is not None else "all_archive_HM",
                                       "episodes": hm_count, "eligible": len(records),
                                       "excluded": excluded},
              "production_modified": False, "source_content_rehashed": False,
              "source_identity_semantics": "committed SHA retained; size/mtime verified before and after reads",
              "policy": asdict(policy), "workers": args.workers, "episodes_planned": len(records),
              "episodes_completed": 0, "reason_bits": {reason.name.lower(): reason.value for reason in HMQualityReason},
              "started_at": datetime.now(timezone.utc).isoformat(), "episodes": [],
              "distance_histogram_bin_edges_m": [0, .1, .2, .3, .5, .75, 1., 2., 5., 10., "infinity"],
              "limitations": ["Retained fits are not certified ground truth.",
                              "Camera-unavailable evidence remains unchecked rather than automatically rejected.",
                              "Quality sidecars must be applied before ground/contact and normalization are rebuilt.",
                              "Independent masks are not automatically activated in any training reader."]}
    report_path = output / "report.json"
    _write_json(report_path, report)
    last_report = time.monotonic()
    try:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            for receipt in _audit_tasks(executor, records, root, output, asdict(policy), args.workers):
                report["episodes"].append(receipt)
                report["episodes_completed"] = len(report["episodes"])
                report["strata"] = _aggregate(report["episodes"])
                report["updated_at"] = datetime.now(timezone.utc).isoformat()
                # Avoid repeatedly writing a growing population report per clip.
                if report["episodes_completed"] == 1 or time.monotonic() - last_report >= 30:
                    _write_json(report_path, report)
                    last_report = time.monotonic()
                print(json.dumps({"completed": report["episodes_completed"], "total": len(records),
                                  "episode_index": receipt["episode_index"], "summary": receipt["summary"]}), flush=True)
        report["state"] = "complete"
        report["camera_motion_audit"] = {name: sorted(
            [{"episode_index": receipt["episode_index"], "source_key": receipt["source_key"], **edge}
             for receipt in report["episodes"] for edge in receipt["camera_motion_audit"][name]["top_edges"]],
            key=lambda edge: edge["speed_m_s"], reverse=True)[:20]
            for name in ("native30Hz", "training5Hz_phase0")}
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["episodes"].sort(key=lambda item: item["episode_index"])
        _write_json(report_path, report)
    except BaseException as error:
        report["state"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        _write_json(report_path, report)
        raise


if __name__ == "__main__":
    main()
