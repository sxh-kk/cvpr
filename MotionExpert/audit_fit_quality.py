"""Read-only final-FK body quality audit for both accepted SMPL-H sources.

Only native parameter/observation columns are read. No source asset, DINO,
checkpoint, label value or running training process is modified. A new report
is evidence for a separately built and independently accepted archive revision;
it does not publish or switch a training dataset.
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

from .fit_quality import (FitQualityPolicy, FitQualityReason, FitQualityWarning,
                          fk_head_world, native_fit_availability, score_fit_quality,
                          summarize_fit_quality)

AUDIT_FORMAT = "smplh-final-fk-quality-audit-v1"
FORMAT = AUDIT_FORMAT
SOURCE_POLICIES = {
    "egosuite": {"head_observation": "captured_egosuite_head_world_distinct_from_FK_head15",
                 "gate_scope": "estimated_body_only_preserve_independent_source_observations_hands_feet"},
    "egoexo4d_hm": {"head_observation": "Aria_CPF_distinct_from_RGB_optical_center_and_FK_head15",
                     "gate_scope": "complete_fit_linked_body_hands_feet"},
}
COLUMNS = {
    "timestamp": 1, "frame_index": 1, "episode_index": 1,
    "smplh.root_orient": 3, "smplh.pose_body": 63, "smplh.trans": 3, "smplh.betas": 16,
    "estimated.valid.body_position": 22, "estimated.valid.body_rotation": 22,
    "estimated.completed.body_position": 22, "source.observed.body_position": 22,
    "observation.state.camera_world": 3, "observation.valid.camera_position": 1,
    "observation.state.head_world": 3, "observation.valid.head_position": 1,
    "source.feet_valid": 2, "observation.valid.hand_left_position": 21,
    "observation.valid.hand_right_position": 21,
}
_MODEL = None
_BATCH_SIZE = 4096


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _stamp(path):
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def code_binding():
    root = Path(__file__).parent
    return {name: _sha(root / name) for name in (
        "fit_quality.py", "audit_fit_quality.py", "dataset/smplh_archive.py")}


def _init_worker(model, batch_size):
    import pyarrow as pa
    import torch
    global _MODEL, _BATCH_SIZE
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    torch.set_num_threads(1)
    _MODEL, _BATCH_SIZE = model, batch_size


def _numpy_column(table, name, width):
    column = table[name]
    if hasattr(column, "combine_chunks"):
        column = column.combine_chunks()
    if column.null_count:
        raise ValueError(f"Null required archive column {name}")
    if hasattr(column, "values"):
        if column.values.null_count:
            raise ValueError(f"Null list element in {name}")
        value = np.asarray(column.values.to_numpy(zero_copy_only=False)).reshape(len(table), -1)
        if value.shape[1] != width:
            raise ValueError(f"Unexpected width for {name}")
        return value[:, 0] if width == 1 else value
    if width != 1:
        raise ValueError(f"Required fixed-list column {name}")
    return np.asarray(column.to_numpy(zero_copy_only=False))


def _owned(root, relative):
    relative = Path(relative)
    path = (root / relative).resolve(strict=True)
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
        raise ValueError("Archive path escapes owned root")
    return path


def audit_episode(archive_root, record_path, output_root, policy, *, model=None, batch_size=None):
    """Compute fresh physical evidence, preserving the committed parent record."""
    import pyarrow.parquet as pq

    root, path, output = Path(archive_root).resolve(), Path(record_path), Path(output_root)
    model = _MODEL if model is None else model
    block_size = _BATCH_SIZE if batch_size is None else batch_size
    if model is None or block_size < 1:
        raise ValueError("An initialized shape16 SMPL-H model and positive batch size are required")
    policy = asdict(policy) if isinstance(policy, FitQualityPolicy) else dict(policy)
    frozen_policy = FitQualityPolicy(**policy)
    record_bytes = path.read_bytes()
    record = json.loads(record_bytes)
    source = record["source"]
    metadata = record["metadata"]
    if (source not in SOURCE_POLICIES or metadata.get("training_eligible") is False
            or metadata.get("canonical_training_eligible") is False or metadata.get("quarantine_reasons")
            or metadata.get("coordinate_verified") is not True):
        raise ValueError("Final-FK screening requires an eligible registered source episode")
    motion = _owned(root, record["data_path"])
    before = _stamp(motion)
    committed_receipt_path = _owned(root, f"episodes/{record['episode_index']:08d}/receipt.json")
    committed_receipt_bytes = committed_receipt_path.read_bytes()
    committed_receipt = json.loads(committed_receipt_bytes)
    if (committed_receipt.get("record") != record
            or committed_receipt.get("motion", {}).get("sha256") != record["motion_sha256"]
            or {key: committed_receipt.get("output_identities", {}).get("motion", {}).get(key)
                for key in before} != before):
        raise ValueError("Motion identity differs from accepted committed episode receipt")
    parquet = pq.ParquetFile(motion)
    if parquet.metadata.num_rows != record["length"]:
        raise ValueError("Native frame population differs from committed episode")
    pieces = {name: [] for name in ("timestamp", "head", "finite", "native", "camera", "camera_valid", "observed", "observed_valid", "head_observed", "head_completed")}
    hm_orphan_labels = 0
    cursor = 0
    for batch in parquet.iter_batches(batch_size=block_size, columns=list(COLUMNS), use_threads=False):
        arrays = {name: _numpy_column(batch, name, width) for name, width in COLUMNS.items()}
        count = len(batch)
        if (not np.array_equal(arrays["frame_index"], np.arange(cursor, cursor+count))
                or np.any(arrays["episode_index"] != record["episode_index"])):
            raise ValueError("Native rows are not the committed contiguous episode identity")
        head, finite = fk_head_world(model, arrays)
        native = native_fit_availability(arrays["estimated.valid.body_position"], arrays["estimated.valid.body_rotation"])
        if source == "egoexo4d_hm":
            fitlinked = arrays["source.feet_valid"].any(-1) | arrays["observation.valid.hand_left_position"].any(-1) | arrays["observation.valid.hand_right_position"].any(-1)
            hm_orphan_labels += int((~native & fitlinked).sum())
        values = (arrays["timestamp"].astype(np.float64), head, finite, native,
                  arrays["observation.state.camera_world"], arrays["observation.valid.camera_position"],
                  arrays["observation.state.head_world"], arrays["observation.valid.head_position"],
                  arrays["source.observed.body_position"][:, 15], arrays["estimated.completed.body_position"][:, 15])
        for name, value in zip(pieces, values):
            pieces[name].append(value)
        cursor += count
    if cursor != record["length"] or hm_orphan_labels:
        raise ValueError(f"Unexpected native population or HM fit-linked labels without body baseline: {hm_orphan_labels}")
    arrays = {name: np.concatenate(chunks, axis=0) for name, chunks in pieces.items()}
    result = score_fit_quality(arrays["head"], arrays["camera"], arrays["observed"], native_valid=arrays["native"],
                               camera_valid=arrays["camera_valid"], head_observation_valid=arrays["observed_valid"],
                               timestamps_s=arrays["timestamp"], fit_finite=arrays["finite"], policy=frozen_policy)
    if (_stamp(motion) != before or path.read_bytes() != record_bytes
            or committed_receipt_path.read_bytes() != committed_receipt_bytes):
        raise ValueError("Parent motion or committed episode identity changed during audit")
    index = record["episode_index"]
    mask_path = output / f"episode_{index:08d}.npz"
    temporary = output / f"episode_{index:08d}.npz.tmp"
    saved_arrays = {**result.arrays(), "head_observed": arrays["head_observed"], "head_completed": arrays["head_completed"]}
    if mask_path.exists() or temporary.exists() or (output / f"episode_{index:08d}.json").exists():
        raise FileExistsError("Audit sidecars are immutable; use a fresh output directory")
    with temporary.open("xb") as stream:
        np.savez(stream, **saved_arrays)
        stream.flush()
        os.fsync(stream.fileno())
    with np.load(temporary, allow_pickle=False) as saved:
        if set(saved.files) != set(saved_arrays) or any(saved[key].dtype != value.dtype or saved[key].shape != value.shape
                or not np.array_equal(saved[key], value, equal_nan=True) for key, value in saved_arrays.items()):
            raise ValueError("Quality NPZ failed exact write/read verification")
    os.replace(temporary, mask_path)
    summary = summarize_fit_quality(result)
    summary["head_observation_strata"] = {name: {
        "native_frames": int((arrays[name] & result.native_valid).sum()),
        "newly_rejected_frames": int((arrays[name] & result.newly_rejected).sum()),
        "max_head_camera_distance_m": (float(np.nanmax(result.head_camera_distance_m[arrays[name] & result.native_valid & result.distance_checked]))
            if (arrays[name] & result.native_valid & result.distance_checked).any() else None),
    } for name in ("head_observed", "head_completed")}
    receipt = {
        "format": AUDIT_FORMAT, "episode_index": index, "source": source,
        "source_key": record["source_key"], "split": record["split"], "length": record["length"],
        "policy": policy, "source_policy": SOURCE_POLICIES[source],
        "source_policy_sha256": hashlib.sha256(_canonical(SOURCE_POLICIES[source])).hexdigest(),
        "model_sha256": model.model_sha256, "head_joint_index": 15, "shape_components": 16,
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(), "motion_sha256": record["motion_sha256"],
        "parent_episode_receipt_sha256": hashlib.sha256(committed_receipt_bytes).hexdigest(),
        "motion_stat": before, "motion_content_rehashed": False,
        "source_identity": metadata.get("source_identity"),
        "source_identity_semantics": "accepted_parent_record_provenance_no_original_source_payload_read",
        "mask_path": mask_path.name, "mask_sha256": _sha(mask_path),
        "native_valid_sha256": hashlib.sha256(np.ascontiguousarray(result.native_valid).tobytes()).hexdigest(),
        "native_valid_sha256_encoding": "bool_C_order_bytes",
        "native_valid_semantics": "any_of_estimated.valid.body_position_or_rotation_22",
        "timestamp_column": "timestamp_float32_cast_float64", "frame_join": "native30Hz_contiguous_frame_index_checked",
        "code_binding": code_binding(), "summary": summary,
        "hm_fitlinked_without_body_baseline_frames": hm_orphan_labels,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / f"episode_{index:08d}.json", receipt)
    return receipt


def _audit_tasks(executor, paths, root, output, policy, workers):
    """Consume results while admitting at most two tasks per worker."""
    remaining, pending, exhausted = iter(paths), set(), False
    try:
        while pending or not exhausted:
            while not exhausted and len(pending) < 2 * workers:
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


def _aggregate(receipts):
    groups = {}
    for row in receipts:
        for name in ("all", "source:" + row["source"], "split:" + row["split"], "source_split:" + row["source"] + ":" + row["split"]):
            group = groups.setdefault(name, {"episodes": 0, "frames": 0, "native_valid_frames": 0,
                                            "quality_valid_frames": 0, "newly_rejected_frames": 0,
                                            "camera_unknown_native_frames": 0, "head_observation_unknown_native_frames": 0,
                                            "reason_counts": {}, "warning_counts": {}})
            group["episodes"] += 1
            for key in ("frames", "native_valid_frames", "quality_valid_frames", "newly_rejected_frames",
                        "camera_unknown_native_frames", "head_observation_unknown_native_frames"):
                group[key] += row["summary"][key]
            for kind in ("reason_counts", "warning_counts"):
                for key, value in row["summary"][kind].items():
                    group[kind][key] = group[kind].get(key, 0) + value
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode-indices", nargs="+", type=int)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--fk-batch-size", type=int, default=4096)
    parser.add_argument("--calibration-report", help="Train-only calibration evidence JSON; bound without using val/test to tune policy")
    args = parser.parse_args()
    if args.fk_batch_size < 1:
        parser.error("FK batch size must be positive")
    # The normal accepted reader performs the existing metadata/publication gate
    # once in the parent. Workers receive only the already owned body model.
    from .dataset.smplh_archive import SMPLHParameterArchive
    archive = SMPLHParameterArchive(args.archive_root, cache_mb=0, file_cache=1)
    root = archive.root
    all_records = archive.records()
    selected = None if args.episode_indices is None else set(args.episode_indices)
    if selected is not None and (not selected or not selected.issubset(range(len(all_records)))):
        raise ValueError("Requested episode does not exist in accepted parent")
    records, excluded = [], []
    for record in all_records:
        metadata = record["metadata"]
        reasons = list(metadata.get("quarantine_reasons") or [])
        if metadata.get("training_eligible") is False or metadata.get("canonical_training_eligible") is False:
            reasons.append("training_ineligible")
        if metadata.get("coordinate_verified") is not True:
            reasons.append("world_registration_unverified")
        if record["source"] not in SOURCE_POLICIES:
            raise ValueError("Unknown source has no declared head-observation semantics")
        if reasons:
            excluded.append({"episode_index": record["episode_index"], "source": record["source"],
                             "source_key": record["source_key"], "reasons": sorted(set(reasons))})
        elif selected is None or record["episode_index"] in selected:
            records.append(record)
    if not records or (selected is not None and {r["episode_index"] for r in records} != selected):
        raise ValueError("Requested episodes include excluded source episodes")
    output = Path(args.output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Audit output must be outside the immutable parent archive")
    policy = asdict(FitQualityPolicy())
    calibration = None
    if args.calibration_report:
        calibration_path = Path(args.calibration_report).resolve(strict=True)
        evidence = json.loads(calibration_path.read_text())
        if (evidence.get("scope") != "train_only_new_physical_metric" or evidence.get("policy") != policy
                or evidence.get("state") != "complete" or evidence.get("uses_sigma_cutoff") is not False
                or evidence.get("code_binding") != code_binding()):
            raise ValueError("Calibration must be completed train-only evidence for this exact code and policy")
        calibration = {"path": str(calibration_path), "sha256": _sha(calibration_path),
                       "scope": evidence["scope"], "threshold_selection": "physical_distances_with_train_only_controls_not_sigma_clipping"}
    before_binding = code_binding()
    identity_paths = [root / relative for relative in ("meta/info.json", "meta/splits.json", "meta/episodes.parquet", ".pipeline/dataset-state.json")]
    parent_binding = {str(path.relative_to(root)): _sha(path) for path in identity_paths}
    source_population = {"scope": "all_eligible_archive" if selected is None else "selected_diagnostic",
                         "total": len(all_records), "episodes": len(all_records), "eligible": len(all_records)-len(excluded), "excluded": excluded,
                         "sources": {source: {"episodes": sum(r["source"] == source for r in all_records),
                                              "eligible": sum(r["source"] == source and r["episode_index"] not in {x["episode_index"] for x in excluded} for r in all_records)}
                                     for source in SOURCE_POLICIES}}
    output.mkdir(parents=True, exist_ok=False)
    report = {"format": AUDIT_FORMAT, "state": "running", "archive_root": str(root),
              "diagnostic": selected is not None, "source_population": source_population,
              "policy": policy, "source_policies": SOURCE_POLICIES, "code_binding": before_binding,
              "model_sha256": archive.model.model_sha256, "model_path": str(archive.model_path),
              "parent_binding": parent_binding, "parent_records_manifest_sha256": archive.records_manifest_sha256,
              "parent_split_fingerprint": archive.split_fingerprint,
              "calibration_evidence": calibration, "workers": args.workers, "fk_batch_size": args.fk_batch_size,
              "reason_bits": {r.name.lower(): int(r) for r in FitQualityReason},
              "warning_bits": {r.name.lower(): int(r) for r in FitQualityWarning},
              "episodes_planned": len(records), "episodes_completed": 0, "episodes": [],
              "production_modified": False, "checkpoint_read": False, "motion_content_rehashed": False,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "limitations": ["Screening preserves parameter values; retained fits are not certified ground truth.",
                              "Unknown camera or source-head evidence cannot certify body supervision.",
                              "This audit does not publish an archive or change the training data binding.",
                              "Rebuild source-aware masks, ground, contact, windows and train statistics before use."]}
    report_path = output / "report.json"
    _write_json(report_path, report)
    last_report = time.monotonic()
    try:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker, initargs=(archive.model, args.fk_batch_size)) as executor:
            paths = [root / f".pipeline/episodes/{r['episode_index']:08d}.json" for r in records]
            for receipt in _audit_tasks(executor, paths, root, output, policy, args.workers):
                if receipt["code_binding"] != before_binding or receipt["model_sha256"] != report["model_sha256"]:
                    raise ValueError("Audit worker did not use frozen code/model identity")
                report["episodes"].append(receipt)
                report["episodes_completed"] = len(report["episodes"])
                if report["episodes_completed"] == 1 or time.monotonic() - last_report >= 30:
                    report["strata"] = _aggregate(report["episodes"])
                    report["updated_at"] = datetime.now(timezone.utc).isoformat()
                    _write_json(report_path, report)
                    last_report = time.monotonic()
                print(json.dumps({"completed": report["episodes_completed"], "total": len(records),
                                  "episode_index": receipt["episode_index"], "summary": receipt["summary"]}), flush=True)
        if code_binding() != before_binding or any(_sha(path) != parent_binding[str(path.relative_to(root))] for path in identity_paths):
            raise ValueError("Parent metadata or source code changed during audit")
        archive.assert_current()
        report["episodes"].sort(key=lambda row: row["episode_index"])
        if report["episodes_completed"] != report["episodes_planned"]:
            raise ValueError("Incomplete final-FK quality population")
        report["strata"] = _aggregate(report["episodes"])
        report["state"] = "complete"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(report_path, report)
    except BaseException as error:
        report["state"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        _write_json(report_path, report)
        raise


if __name__ == "__main__":
    main()
