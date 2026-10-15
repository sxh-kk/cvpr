"""Create a candidate HM quality revision without changing accepted source files.

Only quality-dependent masks and episode ground are revised.  Parameters, hands
XYZ, camera observations and visual joins retain their original values.  Frozen
artifacts are hard-linked; changed files and every publication document are new
files.  This command never accepts a revision or starts/resumes training.
"""
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

import numpy as np

from .dataset.smplh_archive import (
    PARAMETER_SCHEMA_VERSION, SMPLHParameterArchive, canonical_json, file_sha256,
)
from .dataset.vision_cache import JOIN_DTYPES, SOURCE_JOIN, storage_digest
from .dataset.window_cache import atomic_json, digest


AUDIT_FORMAT = "egoexo4d-hm-fit-quality-audit-v1"
REVISION_FORMAT = "smplh-hm-quality-revision-v1"
ALGORITHM_VERSION = 1
JOIN_COLUMNS = (*SOURCE_JOIN, "observation.valid.dino")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _owned(root, relative, *, exists=True):
    root, name = Path(root).resolve(), Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("Revision artifacts require owned relative paths")
    path = (root / name).resolve(strict=exists)
    if not path.is_relative_to(root) or (exists and not path.is_file()):
        raise ValueError("Revision artifact escapes its owned directory")
    return path


def _stamp(path):
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _matches_stamp(path, identity):
    return _stamp(path) == {name: identity.get(name) for name in ("bytes", "mtime_ns")}


def _hardlink(source, destination):
    """Never overwrite a hard link, including on a resumed build."""
    source, destination = Path(source), Path(destination)
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError(f"Unsafe existing hard-link destination: {destination}")
        if not os.path.samefile(source, destination):
            raise ValueError(f"Existing revision artifact is not the frozen source: {destination}")
    else:
        os.link(source, destination)
    after, saved = source.stat(), destination.stat()
    identity = lambda value: (value.st_size, value.st_mtime_ns, value.st_ino, value.st_dev)
    if identity(before) != identity(after) or identity(after) != identity(saved):
        raise ValueError("Frozen source changed while creating its hard link")


def _columns(table):
    import pyarrow as pa
    arrays = {}
    for name in table.column_names:
        column = table[name].combine_chunks()
        if column.null_count:
            raise ValueError(f"Null archive cells are unsupported: {name}")
        if pa.types.is_fixed_size_list(column.type):
            value = column.values.to_numpy(zero_copy_only=False).reshape(len(table), column.type.list_size)
        else:
            value = column.to_numpy(zero_copy_only=False)
        arrays[name] = np.asarray(value).copy()
    return arrays


def _table_from_arrays(template, arrays):
    import pyarrow as pa
    columns = {}
    for field in template.schema:
        values = arrays[field.name]
        columns[field.name] = (pa.FixedSizeListArray.from_arrays(
            pa.array(values.reshape(-1), type=field.type.value_type), field.type.list_size)
            if pa.types.is_fixed_size_list(field.type) else pa.array(values, type=field.type))
    return pa.table(columns, schema=template.schema)


def apply_quality_masks(arrays, quality_valid):
    """Gate existing availability; never create a source observation or edit XYZ."""
    count = len(arrays["timestamp"])
    good = np.asarray(quality_valid)
    if good.shape != (count,) or good.dtype != np.bool_:
        raise ValueError("Quality frame mask must be boolean T")
    for foot_key, body_key in (("source.feet_valid", "source.observed.body_position"),
                               ("source.feet_interpolated", "source.interpolated.body_position")):
        if not np.array_equal(arrays[foot_key], arrays[body_key][:, [10, 11]]):
            raise ValueError("Original source foot availability differs from the archive body-observation contract")
    result = {name: value.copy() for name, value in arrays.items()}
    for component in ("position", "rotation"):
        valid = result[f"estimated.valid.body_{component}"]
        observed = result[f"source.observed.body_{component}"]
        valid &= good[:, None]
        observed &= good[:, None]
        result[f"source.interpolated.body_{component}"] &= observed
        result[f"estimated.completed.body_{component}"] = valid & ~observed
        result[f"quality.{component}_confidence"][~valid] = 0
    result["quality.observation_residual_valid"] &= (
        result["estimated.valid.body_position"] & result["source.observed.body_position"])
    result["source.feet_valid"] &= result["source.observed.body_position"][:, [10, 11]]
    result["source.feet_interpolated"] &= (
        result["source.interpolated.body_position"][:, [10, 11]] & result["source.feet_valid"])
    for side in ("left", "right"):
        valid = result[f"observation.valid.hand_{side}_position"]
        valid &= good[:, None]
        result[f"observation.interpolated.hand_{side}_position"] &= valid
    if "smplh.valid.hand_pose" in result:
        result["smplh.valid.hand_pose"] &= good[:, None]
    return result


def estimate_revision_ground(arrays, metadata, *, root_rest, root_shape_basis):
    """Original native30Hz stationary-foot DBSCAN after quality gating.

    Source feet are already in the shared output world.  The root is needed
    only for descriptive cluster medians, so no complete-body FK is repeated.
    A failed episode estimate leaves the existing observed-prefix fallback.
    """
    previous = metadata.get("ground", {})
    mode = previous.get("mode", "estimate")
    if mode != "estimate":
        return deepcopy(previous)
    result = {"height_m": 0., "valid": False, "verified": False,
              "normal": [0., 0., 1.], "frame": "output_world", "mode": mode,
              "foot_joint_indices": [10, 11],
              "window_fallback": deepcopy(previous.get("window_fallback", {"mode": "mean_valid_feet", "offset_m": .02}))}
    if not metadata.get("coordinate_verified", False):
        result["reason"] = "output_world_vertical_axis_unverified"
        return result
    raw = previous.get("parameters", {})
    params = {"static_speed_m_s": float(raw.get("static_speed_m_s", .15)),
              "cluster_eps_m": float(raw.get("cluster_eps_m", .005)),
              "cluster_min_samples": raw.get("cluster_min_samples", 3),
              "toe_offset_m": float(raw.get("toe_offset_m", .01)),
              "max_gap_s": float(raw.get("max_gap_s", .075)),
              "noise_policy": "exclude_label_minus_one"}
    minimum = params["cluster_min_samples"]
    if (type(minimum) is not int or minimum < 1
            or any(not np.isfinite(params[key]) or params[key] <= 0 for key in (
                "static_speed_m_s", "cluster_eps_m", "max_gap_s"))
            or not np.isfinite(params["toe_offset_m"]) or params["toe_offset_m"] < 0):
        raise ValueError("Original episode ground thresholds are invalid")
    result.update(method="stationary_foot_height_dbscan", parameters=params, clusters=[])
    count = len(arrays["timestamp"])
    if count < 2:
        result["reason"] = "too_few_frames"
        return result
    # Episode timestamps used by the original converter are the float64 native
    # grid; the stored float32 timestamp is only the presentation clock.
    times = np.arange(count, dtype=np.float64) / 30
    feet = arrays["source.feet_world"].reshape(count, 2, 3)
    mask = arrays["source.feet_valid"].astype(bool) & np.isfinite(feet).all(-1)
    dt = np.diff(times)
    edges = mask[:-1] & mask[1:] & (dt <= params["max_gap_s"])[:, None]
    speed = np.linalg.norm(np.diff(feet, axis=0), axis=-1) / dt[:, None]
    peak, observed = np.zeros(mask.shape), np.zeros(mask.shape, dtype=bool)
    for target in (slice(None, -1), slice(1, None)):
        peak[target] = np.maximum(peak[target], np.where(edges, speed, 0))
        observed[target] |= edges
    frame_ids, foot_ids = np.nonzero(mask & observed & (peak <= params["static_speed_m_s"]))
    heights = feet[frame_ids, foot_ids, 2]
    result["candidate_count"] = len(heights)
    if not len(heights):
        result["reason"] = "no_stationary_feet"
        return result
    from sklearn.cluster import DBSCAN
    unique, inverse, counts = np.unique(heights, return_inverse=True, return_counts=True)
    labels = DBSCAN(eps=params["cluster_eps_m"], min_samples=minimum).fit(
        unique[:, None], sample_weight=counts).labels_[inverse]
    result["noise_count"] = int(np.count_nonzero(labels == -1))
    root = (np.asarray(root_rest)[None] + np.einsum(
        "tb,cb->tc", arrays["smplh.betas"], np.asarray(root_shape_basis)) + arrays["smplh.trans"])[:, 2]
    root_valid = arrays["estimated.valid.body_position"][:, 0]
    for label in np.unique(labels[labels >= 0]):
        group = np.flatnonzero(labels == label)
        frame = np.unique(frame_ids[group])
        gaps = np.diff(times[frame])
        roots = root[frame[root_valid[frame]]]
        result["clusters"].append({
            "median_height_m": float(np.median(heights[group])),
            "root_median_height_m": float(np.median(roots)) if len(roots) else None,
            "sample_count": len(group), "support_s": float(gaps[gaps <= params["max_gap_s"]].sum()),
            "spread_m": float(np.ptp(heights[group]))})
    if not result["clusters"]:
        result["reason"] = "no_density_height_cluster"
        return result
    chosen = min(result["clusters"], key=lambda c: c["median_height_m"])
    result.update(height_m=chosen["median_height_m"] - params["toe_offset_m"],
                  valid=True, source="estimated_stationary_feet", reason="estimated")
    result["multiple_height_levels"] = any(
        c["median_height_m"] > chosen["median_height_m"] + .04 for c in result["clusters"])
    return result


def _load_quality(root, receipt, record, record_bytes):
    if (receipt.get("format") != AUDIT_FORMAT or receipt.get("episode_index") != record["episode_index"]
            or receipt.get("source_key") != record["source_key"] or receipt.get("split") != record["split"]
            or receipt.get("record_sha256") != hashlib.sha256(record_bytes).hexdigest()
            or receipt.get("motion_sha256") != record["motion_sha256"]):
        raise ValueError("Quality receipt is not bound to this original episode")
    mask_path = _owned(root, receipt["mask_path"])
    if file_sha256(mask_path) != receipt.get("mask_sha256"):
        raise ValueError("Quality sidecar SHA differs")
    with np.load(mask_path, allow_pickle=False) as data:
        names = ("quality_valid", "native_valid", "fit_finite", "transition_valid", "reasons")
        arrays = {name: data[name].copy() for name in names}
    count = record["length"]
    if any(v.shape != (count,) for v in arrays.values()) or any(
            arrays[name].dtype != np.bool_ for name in names[:-1]):
        raise ValueError("Quality sidecar mask shape/dtype differs")
    if arrays["reasons"].dtype != np.uint16:
        raise ValueError("Quality reason bits must be uint16")
    native_sha = hashlib.sha256(arrays["native_valid"].tobytes(order="C")).hexdigest()
    if receipt.get("native_valid_sha256") != native_sha:
        raise ValueError("Quality native-valid upstream SHA differs")
    if np.any(arrays["quality_valid"] & ~(arrays["native_valid"] & arrays["fit_finite"])):
        raise ValueError("Quality policy attempts to upgrade unavailable fits")
    identity = record.get("metadata", {}).get("source_identity")
    if identity is not None and receipt.get("source_identity") != identity:
        raise ValueError("Quality native source identity differs from the committed episode")
    return arrays


def _verify_visual_join(source_root, record, source_vision_root, *, arrays=None, compare_rows=False):
    """Inherit frozen joins; compare native rows only after motion is rewritten."""
    prefix = f"episodes/{record['episode_index']:08d}"
    receipt_path = _owned(source_vision_root, f"{prefix}/receipt.json")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("source_record_sha256") != storage_digest(record):
        raise ValueError("Original visual receipt differs from its source episode")
    join_path = _owned(source_vision_root, f"{prefix}/join.npz")
    feature_path = _owned(source_vision_root, f"{prefix}/features.npy")
    for name, path in (("join.npz", join_path), ("features.npy", feature_path)):
        stat = _stamp(path)
        identity = receipt["output_identities"][name]
        if (stat["bytes"], stat["mtime_ns"]) != (identity["size_bytes"], identity["mtime_ns"]):
            raise ValueError("Original visual payload identity changed")
    if not compare_rows:
        return receipt, receipt_path
    if arrays is None:
        raise ValueError("Rewritten motion join verification requires its generated arrays")
    with np.load(join_path, allow_pickle=False) as saved:
        if set(saved.files) != set(JOIN_DTYPES):
            raise ValueError("Original visual join columns differ")
        join = {name: saved[name].copy() for name in saved.files}
    count = record["length"]
    for name, dtype in JOIN_DTYPES.items():
        if join[name].shape != (count,) or join[name].dtype != np.dtype(dtype):
            raise ValueError(f"Native visual join shape/dtype differs: {name}")
    for name in JOIN_COLUMNS:
        if name in arrays and not np.array_equal(join[name], arrays[name]):
            raise ValueError(f"Native visual/source row join differs: {name}")
    native = np.arange(count, dtype=np.int64)
    if (not np.array_equal(join["frame_index"], native)
            or not np.all(join["episode_index"] == record["episode_index"])
            or not np.array_equal(join["index"], native + record["dataset_from_index"])
            or not np.array_equal(join["timestamp"], (native / 30).astype(np.float32))
            or not np.array_equal(join["source.valid.dino"], join["observation.valid.dino"])):
        raise ValueError("Visual native grid/validity identity differs")
    with feature_path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
    if shape != (count, 5, 1024) or fortran or dtype != np.dtype("float32"):
        raise ValueError("Frozen DINO feature header differs")
    return receipt, receipt_path


def _resume_episode(target, original, expected_binding, quality_receipt, *, source_root, source_vision_root):
    record_path = target / f".pipeline/episodes/{original['episode_index']:08d}.json"
    receipt_path = target / f"episodes/{original['episode_index']:08d}/receipt.json"
    if not record_path.exists() or not receipt_path.exists():
        return None
    record, receipt = json.loads(record_path.read_text()), json.loads(receipt_path.read_text())
    revision = receipt.get("quality_revision", {})
    expected_mask = quality_receipt.get("mask_sha256") if quality_receipt else None
    if (receipt.get("record") != record or revision.get("build_binding_sha256") != expected_binding
            or revision.get("parent_record_sha256") != digest(original)
            or revision.get("quality_mask_sha256") != expected_mask):
        raise ValueError("Existing episode belongs to a different quality revision")
    for role, key in (("motion", "data_path"), ("vision", "vision_path")):
        identity = receipt["output_identities"][role]
        path = target / record[key]
        if role == "vision" and revision.get("legacy_vision_retired"):
            continue
        if not _matches_stamp(path, identity):
            raise ValueError("Completed revision episode artifact changed")
        if role == "motion" and revision["motion_rewritten"] and file_sha256(path) != record["motion_sha256"]:
            raise ValueError("Rewritten revision motion checksum changed")
        if not (role == "motion" and revision["motion_rewritten"]):
            if not os.path.samefile(path, Path(source_root) / original[key]):
                raise ValueError("Reused revision artifact is no longer the original hard link")
    profile_receipt = target / f"vision_profiles/psi0_256x480_v1_30hz/episodes/{original['episode_index']:08d}/receipt.json"
    if not profile_receipt.is_file():
        raise ValueError("Completed revision episode lacks rebound visual receipt")
    visual = json.loads(profile_receipt.read_text())
    if (visual.get("source_record_sha256") != storage_digest(record)
            or visual.get("quality_revision_reuse", {}).get("build_binding_sha256") != expected_binding):
        raise ValueError("Rebound visual receipt differs from the completed revision")
    for name in ("features.npy", "join.npz"):
        current = profile_receipt.parent / name
        old = Path(source_vision_root) / f"episodes/{original['episode_index']:08d}/{name}"
        if not os.path.samefile(current, old):
            raise ValueError("Rebound visual payload is no longer the original hard link")
        identity = visual["output_identities"][name]
        stamp = _stamp(current)
        if (stamp["bytes"], stamp["mtime_ns"]) != (identity["size_bytes"], identity["mtime_ns"]):
            raise ValueError("Rebound visual payload identity changed")
    return record, revision


def _prepare_episode_output(task):
    """Remove only our known incomplete artifacts; successful commits stay frozen."""
    target, index = Path(task["output_root"]), task["record"]["episode_index"]
    prefix = f"episodes/{index:08d}"
    if (task["record"]["data_path"] != f"{prefix}/motion.parquet"
            or task["record"]["vision_path"] != f"{prefix}/vision.parquet"):
        raise ValueError("Revision requires the published canonical episode file layout")
    marker_path = target / f".pipeline/episode-build-state/{index:08d}.json"
    expected = {"build_binding_sha256": task["build_binding_sha256"], "episode_index": index}
    names = [f"{prefix}/motion.parquet", f"{prefix}/vision.parquet", f"{prefix}/receipt.json",
             f"{prefix}/.motion.revision.tmp", f".pipeline/episodes/{index:08d}.json",
             *[f"vision_profiles/psi0_256x480_v1_30hz/{prefix}/{name}"
               for name in ("features.npy", "join.npz", "receipt.json")]]
    if marker_path.exists():
        if json.loads(marker_path.read_text()) != expected:
            raise ValueError("Incomplete revision episode marker differs from the frozen build")
        for relative in names:
            path = _owned(target, relative, exists=False)
            if path.exists():
                if not path.is_file():
                    raise ValueError("Unexpected non-file incomplete episode artifact")
                path.unlink()
    elif any((target / name).exists() for name in names):
        raise ValueError("Unknown uncommitted episode artifacts cannot be overwritten")
    atomic_json(marker_path, expected)


def _revise_episode(task):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    source, target = Path(task["source_root"]), Path(task["output_root"])
    original = task["record"]
    index = original["episode_index"]
    resumed = _resume_episode(target, original, task["build_binding_sha256"], task["quality_receipt"],
                              source_root=source, source_vision_root=task["source_vision_root"])
    if resumed:
        return {"episode_index": index, "record": resumed[0], "revision": resumed[1], "resumed": True}
    _prepare_episode_output(task)
    record_path = _owned(source, f".pipeline/episodes/{index:08d}.json")
    original_bytes = record_path.read_bytes()
    if json.loads(original_bytes) != original:
        raise ValueError("Original episode record changed after revision planning")
    original_receipt_path = _owned(source, f"episodes/{index:08d}/receipt.json")
    original_receipt = json.loads(original_receipt_path.read_text())
    motion = _owned(source, original["data_path"])
    if not _matches_stamp(motion, original_receipt["output_identities"]["motion"]):
        raise ValueError("Original motion identity changed after planning")
    record, arrays, rewritten, revised_cells = deepcopy(original), None, False, 0
    quality_receipt = task["quality_receipt"]
    quality = None
    destination = target / record["data_path"]
    if quality_receipt:
        quality = _load_quality(task["quality_dir"], quality_receipt, original, original_bytes)
        # Released native-invalid chunks already have unavailable fitted labels.
        # If the policy rejects no additional native-valid frames, all masks and
        # ground are inherited without decoding an unchanged Parquet again.
        changed_names, ground_recomputed = [], False
        if (quality["native_valid"] & ~quality["quality_valid"]).any():
            before_table = pq.read_table(motion, use_threads=False)
            before = _columns(before_table)
            arrays = apply_quality_masks(before, quality["quality_valid"])
            changed_names = [name for name in arrays if not np.array_equal(arrays[name], before[name])]
            revised_cells = sum(int(np.count_nonzero(arrays[name] != before[name])) for name in changed_names)
        if changed_names:
            if not np.array_equal(arrays["source.feet_valid"], before["source.feet_valid"]):
                ground_recomputed = True
                ground = estimate_revision_ground(arrays, record["metadata"], root_rest=task["root_rest"],
                                                  root_shape_basis=task["root_shape_basis"])
                record["metadata"]["ground"] = ground
                arrays["observation.environment.floor_height"][:] = np.float32(ground["height_m"])
                arrays["observation.valid.floor"][:] = bool(ground["valid"])
                arrays["observation.valid.floor_verified"][:] = bool(ground["valid"] and ground.get("verified", False))
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(".motion.revision.tmp")
            if destination.exists():
                raise ValueError("Uncommitted revision motion exists; do not overwrite frozen data")
            pq.write_table(_table_from_arrays(before_table, arrays), temporary, row_group_size=512, compression="zstd")
            saved = _columns(pq.read_table(temporary, use_threads=False))
            if set(saved) != set(arrays) or any(
                    saved[name].dtype != value.dtype or saved[name].shape != value.shape
                    or not np.array_equal(saved[name], value) for name, value in arrays.items()):
                raise ValueError("Revised Parquet failed exact readback verification")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            rewritten = True
            record["motion_sha256"] = file_sha256(destination)
        else:
            _hardlink(motion, destination)
        descriptor = {"format": REVISION_FORMAT, "quality_mask_sha256": quality_receipt["mask_sha256"],
                      "native_valid_sha256": quality_receipt["native_valid_sha256"],
                      "quality_receipt_sha256": digest(quality_receipt),
                      "quality_report_sha256": task["quality_report_sha256"],
                      "policy": quality_receipt["policy"],
                      "source_observed_semantics": "quality_gated_native_numeric_availability; original masks remain in parent revision",
                      "supervision_gating": "body22_rotations_positions_hands21_feet; original_values_preserved",
                      "transition_semantics": "native edge audit only; first unavailable edge does not suppress frame labels",
                      "ground_recomputed": ground_recomputed,
                      "parent_ground": deepcopy(original["metadata"].get("ground"))}
        record["metadata"]["hm_quality_revision"] = descriptor
    else:
        _hardlink(motion, destination)
    retired = bool(task["legacy_vision_retired"])
    if not retired:
        _hardlink(_owned(source, original["vision_path"]), target / record["vision_path"])
    visual, visual_receipt_path = _verify_visual_join(source, original, task["source_vision_root"],
                                                       arrays=arrays, compare_rows=rewritten)
    prefix = f"episodes/{index:08d}"
    profile_root = target / "vision_profiles/psi0_256x480_v1_30hz"
    for name in ("features.npy", "join.npz"):
        _hardlink(_owned(task["source_vision_root"], f"{prefix}/{name}"), profile_root / prefix / name)
    visual = deepcopy(visual)
    visual["source_record_sha256"] = storage_digest(record)
    visual["quality_revision_reuse"] = {
        "parent_receipt_sha256": file_sha256(visual_receipt_path),
        "parent_manifest_sha256": task["source_vision_manifest_sha256"],
        "build_binding_sha256": task["build_binding_sha256"], "join_all_native_rows_checked": rewritten,
        "join_evidence": ("changed_motion_all_native_rows_compared" if rewritten
                          else "inherited_immutable_motion_and_visual"),
        "features_content_rehashed": False, "payload_reuse": "same_device_inode_hardlink"}
    atomic_json(profile_root / prefix / "receipt.json", visual)
    revision = {"build_binding_sha256": task["build_binding_sha256"],
                "parent_record_sha256": digest(original),
                "parent_receipt_sha256": file_sha256(original_receipt_path),
                "quality_mask_sha256": quality_receipt["mask_sha256"] if quality_receipt else None,
                "motion_rewritten": rewritten, "revised_mask_cells": revised_cells,
                "legacy_vision_retired": retired, "image_join_values_preserved": True}
    receipt = deepcopy(original_receipt)
    receipt.update(record=record, completed_at=_now(), quality_revision=revision,
                   conversion_binding=digest({"parent": original_receipt["conversion_binding"],
                                              "revision": task["build_binding_sha256"]}))
    receipt["motion"]["sha256"] = record["motion_sha256"]
    receipt["output_identities"]["motion"] = _stamp(destination)
    # A retired legacy vision receipt keeps its historical identity unchanged.
    if not retired:
        receipt["output_identities"]["vision"] = _stamp(target / record["vision_path"])
    if (not _matches_stamp(motion, original_receipt["output_identities"]["motion"])
            or record_path.read_bytes() != original_bytes):
        raise ValueError("Original source changed while constructing its revision")
    atomic_json(target / f"episodes/{index:08d}/receipt.json", receipt)
    atomic_json(target / f".pipeline/episodes/{index:08d}.json", record)
    return {"episode_index": index, "record": record, "revision": revision, "resumed": False}


def _active_workers(path, maximum):
    if path is None:
        return maximum
    value = json.loads(Path(path).read_text())
    if (not isinstance(value, dict) or set(value) != {"active_workers"}
            or type(value["active_workers"]) is not int or not 1 <= value["active_workers"] <= maximum):
        raise ValueError("Worker control requires one active_workers integer within the configured maximum")
    return value["active_workers"]


def _tasks(pool, tasks, *, maximum, worker_control):
    pending, cursor = {}, 0
    while pending or cursor < len(tasks):
        active = _active_workers(worker_control, maximum)
        while cursor < len(tasks) and len(pending) < active:
            future = pool.submit(_revise_episode, tasks[cursor])
            pending[future] = tasks[cursor]["record"]["episode_index"]
            cursor += 1
        finished, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
        for future in finished:
            pending.pop(future)
            yield future.result()


def _source_eligible(record):
    metadata = record["metadata"]
    return (metadata.get("training_eligible") is not False
            and metadata.get("canonical_training_eligible") is not False
            and not metadata.get("quarantine_reasons", []))


def build_revision(source_root, quality_report, output_root, *, source_vision_root=None,
                   workers=2, worker_control=None, resume=False, diagnostic=False):
    """Build a recoverable candidate; independent acceptance is a separate action."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    source, output = Path(source_root).resolve(strict=True), Path(output_root).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Revision output must be separate from and outside the immutable source archive")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Revision workers must be an integer from1 to8")
    report_path = Path(quality_report).resolve(strict=True)
    report = json.loads(report_path.read_text())
    if (report.get("format") != AUDIT_FORMAT or report.get("state") != "complete"
            or Path(report.get("archive_root", "")).resolve() != source
            or report.get("production_modified") is not False):
        raise ValueError("A complete read-only quality audit of this source archive is required")
    audit_scope = report.get("source_hm_population", {}).get("scope", report.get("scope"))
    if not diagnostic and (report.get("diagnostic") is not False or audit_scope != "all_archive_HM"):
        raise ValueError("Formal revisions require the complete nondiagnostic all-HM quality audit")
    if output.is_relative_to(report_path.parent) or report_path.parent.is_relative_to(output):
        raise ValueError("Revision and quality audit directories must be separate")
    source_vision = Path(source_vision_root or source / "vision_profiles/psi0_256x480_v1_30hz").resolve(strict=True)
    if source_vision.is_relative_to(output) or output.is_relative_to(source_vision):
        raise ValueError("Revision output overlaps the frozen DINO profile")
    if output.exists() and not resume and any(output.iterdir()):
        raise ValueError("Use an empty revision directory or --resume its exact frozen plan")
    output.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != output.stat().st_dev or source_vision.stat().st_dev != output.stat().st_dev:
        raise ValueError("Revision requires the same filesystem for zero-copy hard-link reuse")
    with (output / ".revision-build.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        archive = SMPLHParameterArchive(source, vision_profile_root=source_vision)
        originals = archive.records()
        receipts = report.get("episodes")
        if not isinstance(receipts, list) or report.get("episodes_completed") != len(receipts):
            raise ValueError("Quality audit receipt population differs")
        quality = {receipt["episode_index"]: receipt for receipt in receipts}
        if len(quality) != len(receipts):
            raise ValueError("Duplicate quality episode receipts")
        hm_ids = {r["episode_index"] for r in originals if r["source"] == "egoexo4d_hm"}
        required = {r["episode_index"] for r in originals if r["source"] == "egoexo4d_hm" and _source_eligible(r)}
        if not set(quality) <= hm_ids or (not diagnostic and not required <= set(quality)):
            raise ValueError("Quality audit must cover every eligible HM episode; partial audits require --diagnostic")
        for index, receipt in quality.items():
            if receipt.get("policy") != report.get("policy"):
                raise ValueError("Quality episode policy differs from its complete report")
            _load_quality(report_path.parent, receipt, originals[index],
                          (source / f".pipeline/episodes/{index:08d}.json").read_bytes())
        binding = {"format": REVISION_FORMAT, "algorithm_version": ALGORITHM_VERSION,
                   "builder_sha256": file_sha256(__file__), "parent_archive_binding": archive.binding,
                   "parent_archive_fingerprint": archive.archive_fingerprint,
                   "quality_report_sha256": file_sha256(report_path), "policy": report["policy"],
                   "source_vision_manifest_sha256": archive.vision_cache.metadata["manifest_sha256"],
                   "parent_publication_sha256": file_sha256(archive.state_path),
                   "preserved_metadata_sha256": {
                       relative: file_sha256(source / relative)
                       for relative in ("meta/ontology.json", "meta/summary.json", ".pipeline/config.json")
                       if (source / relative).is_file()},
                   "scope": "selected_episodes" if diagnostic else "full"}
        parent_report = archive.state.get("acceptance_report")
        if parent_report:
            parent_report = Path(parent_report)
            if not parent_report.is_absolute():
                parent_report = source / parent_report
            if parent_report.is_file():
                binding["parent_acceptance_sha256"] = file_sha256(parent_report)
        plan = {"format": REVISION_FORMAT, "binding": binding, "binding_sha256": digest(binding),
                "source_root": str(source), "output_root": str(output),
                "source_vision_root": str(source_vision), "quality_report": str(report_path)}
        plan_path = output / ".pipeline/revision-plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("Revision resume plan differs from its frozen source, policy or builder")
        else:
            atomic_json(plan_path, plan)
        state_path = output / ".pipeline/dataset-state.json"
        if state_path.exists() and json.loads(state_path.read_text()).get("state") == "accepted":
            raise ValueError("Accepted revision is immutable; never rebuild it")
        atomic_json(state_path, {"schema_version": PARAMETER_SCHEMA_VERSION, "state": "building",
                                "independent_acceptance_pending": True,
                                "episodes": len(originals), "frames": archive.frames})
        for descriptor in (archive.info["model"], archive.info.get("hand_basis")):
            if descriptor:
                _hardlink(archive.owned_path(descriptor["path"]), output / descriptor["path"])
        for path in sorted((source / "meta").glob("splits*.json")):
            _hardlink(path, output / "meta" / path.name)
        # Ontology and summary describe unchanged axes and original acquisition
        # populations, not the new window/quality acceptance state. Their old
        # timestamps remain honest inherited inventory timestamps.
        for relative in ("meta/ontology.json", "meta/summary.json"):
            if relative in binding["preserved_metadata_sha256"]:
                _hardlink(source / relative, output / relative)
        if ".pipeline/config.json" in binding["preserved_metadata_sha256"]:
            _hardlink(source / ".pipeline/config.json", output / ".pipeline/parent-publication/config.json")
        atomic_json(output / ".pipeline/config.json", {
            "format": REVISION_FORMAT, "schema_version": PARAMETER_SCHEMA_VERSION,
            "model_sha256": archive.model_sha256, "standalone": True,
            "requires_old_dense_archive": False, "requires_parent_archive_at_training": False,
            "revision_plan": ".pipeline/revision-plan.json", "quality_report": ".pipeline/quality/report.json",
            "parent_configuration": (".pipeline/parent-publication/config.json"
                                     if ".pipeline/config.json" in binding["preserved_metadata_sha256"] else None)})
        _hardlink(archive.state_path, output / ".pipeline/parent-publication/dataset-state.json")
        if parent_report and parent_report.is_file():
            _hardlink(parent_report, output / ".pipeline/parent-publication/acceptance-report.json")
        audit_target = output / ".pipeline/quality"
        _hardlink(report_path, audit_target / "report.json")
        for index, receipt in quality.items():
            _hardlink(_owned(report_path.parent, receipt["mask_path"]), audit_target / receipt["mask_path"])
            audit_receipt = report_path.parent / f"episode_{index:08d}.json"
            if not audit_receipt.is_file() or json.loads(audit_receipt.read_text()) != receipt:
                raise ValueError("Quality individual receipt differs from its report")
            _hardlink(audit_receipt, audit_target / audit_receipt.name)
        info = deepcopy(archive.info)
        info["quality_revision"] = {"format": REVISION_FORMAT, "parent_archive_fingerprint": archive.archive_fingerprint,
                                    "quality_report_sha256": binding["quality_report_sha256"],
                                    "build_binding_sha256": plan["binding_sha256"], "policy": report["policy"],
                                    "parameter_values_preserved": True, "image_join_values_preserved": True,
                                    "ground_policy": "original_native30Hz_DBSCAN_after_quality_gating_then_observed_prefix_fallback"}
        atomic_json(output / "meta/info.json", info)
        tasks = [{"source_root": str(source), "output_root": str(output), "record": record,
                  "quality_receipt": quality.get(record["episode_index"]), "quality_dir": str(report_path.parent),
                  "quality_report_sha256": binding["quality_report_sha256"],
                  "build_binding_sha256": plan["binding_sha256"], "source_vision_root": str(source_vision),
                  "source_vision_manifest_sha256": binding["source_vision_manifest_sha256"],
                  "legacy_vision_retired": record["episode_index"] in archive._retired_vision,
                  "root_rest": archive.model.rest[0], "root_shape_basis": archive.model.shape_basis[0]}
                 for record in originals]
        results = {}
        progress_path = output / ".pipeline/revision-progress.json"
        def completed(result):
            results[result["episode_index"]] = result
            progress = {"state": "building", "completed": len(results), "total": len(tasks),
                        "latest_episode": result["episode_index"], "updated_at": _now(),
                        "rewritten_episodes": sum(x["revision"]["motion_rewritten"] for x in results.values())}
            atomic_json(progress_path, progress)
            print(json.dumps(progress), flush=True)
        if workers == 1:
            for task in tasks:
                completed(_revise_episode(task))
        else:
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ.setdefault(name, "1")
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                for result in _tasks(pool, tasks, maximum=workers, worker_control=worker_control):
                    completed(result)
        archive.assert_current()
        for relative, expected_sha in binding["preserved_metadata_sha256"].items():
            destination = (output / ".pipeline/parent-publication/config.json"
                           if relative == ".pipeline/config.json" else output / relative)
            if file_sha256(source / relative) != expected_sha or file_sha256(destination) != expected_sha:
                raise ValueError("Inherited ontology, inventory or producer provenance changed during revision")
        records = [results[i]["record"] for i in range(len(originals))]
        old_table = pq.read_table(source / "meta/episodes.parquet", use_threads=False)
        rows = [{name: record[name] for name in old_table.column_names} for record in records]
        episode_table = pa.Table.from_pylist(rows, schema=old_table.schema)
        metadata_path = output / "meta/episodes.parquet"
        temporary = metadata_path.with_name(".episodes.revision.tmp")
        pq.write_table(episode_table, temporary, compression="zstd")
        if pq.read_table(temporary).to_pylist() != rows:
            raise ValueError("Revision episode metadata readback differs")
        os.replace(temporary, metadata_path)
        if archive._retirement_path.exists():
            retirement = json.loads(archive._retirement_path.read_text())
            retirement["info_sha256"] = file_sha256(output / "meta/info.json")
            retirement["records_manifest_sha256"] = hashlib.sha256(canonical_json(records)).hexdigest()
            atomic_json(output / ".pipeline/retired-vision.json", retirement)
        visual_manifest = deepcopy(archive.vision_cache.manifest)
        profile_root = output / "vision_profiles/psi0_256x480_v1_30hz"
        visual_manifest["source_root"] = str(output)
        visual_manifest["output_root"] = str(profile_root)
        visual_manifest["source_binding"] = {
            "info_sha256": file_sha256(output / "meta/info.json"),
            "episodes_metadata_sha256": file_sha256(metadata_path),
            "split_metadata_sha256": archive.binding["split_metadata_sha256"],
            "records_sha256": storage_digest(records)}
        visual_manifest["quality_revision_reuse"] = {
            "format": REVISION_FORMAT, "parent_manifest_sha256": binding["source_vision_manifest_sha256"],
            "parent_profile_root": str(source_vision),
            "quality_report_sha256": binding["quality_report_sha256"], "payload_reuse": "same_device_inode_hardlink",
            "join_all_native_rows_checked": False,
            "changed_motion_native_join_checked": True,
            "unchanged_join_evidence": "inherited_immutable_motion_and_visual",
            "features_content_rehashed": False}
        atomic_json(profile_root / "manifest.json", visual_manifest)
        sources = {}
        for record in records:
            counts = sources.setdefault(record["source"], {"episodes": 0, "frames": 0})
            counts["episodes"] += 1
            counts["frames"] += record["length"]
        state = {"schema_version": PARAMETER_SCHEMA_VERSION, "state": "candidate",
                 "independent_acceptance_pending": True, "episodes": len(records), "frames": archive.frames,
                 "sources": sources, "quality_revision": ".pipeline/hm-quality-revision.json",
                 "formal_training_ready": False, "updated_at": _now()}
        atomic_json(state_path, state)
        candidate = SMPLHParameterArchive(output, allow_unpublished=True, vision_profile_root=profile_root)
        candidate.assert_current()
        result = {"format": REVISION_FORMAT, "state": "candidate", "scope": binding["scope"],
                  "independent_acceptance_pending": True, "build_binding": binding,
                  "build_binding_sha256": plan["binding_sha256"], "binding": candidate.binding,
                  "archive_fingerprint": candidate.archive_fingerprint, "source_root": str(source),
                  "output_root": str(output), "vision_profile_root": str(profile_root),
                  "population": {"episodes": len(records), "frames": archive.frames, "sources": sources},
                  "quality_episodes": sorted(quality), "required_quality_episodes": sorted(required),
                  "motion_rewritten_episodes": [i for i in sorted(results) if results[i]["revision"]["motion_rewritten"]],
                  "changed_episode_indices": [i for i in sorted(results) if results[i]["revision"]["motion_rewritten"]],
                  "unchanged_motion_sha256": {str(i): records[i]["motion_sha256"] for i in sorted(results)
                                              if not results[i]["revision"]["motion_rewritten"]},
                  "model_sha256": candidate.model_sha256, "parent_model_sha256": archive.model_sha256,
                  "split_fingerprint": candidate.split_fingerprint,
                  "parent_split_fingerprint": archive.split_fingerprint,
                  "coordinate_values_preserved": True,
                  "parameter_values_preserved": True, "camera_values_preserved": True,
                  "image_join_values_preserved": True, "features_reextracted": False,
                  "completed_at": _now()}
        atomic_json(output / ".pipeline/hm-quality-revision.json", result)
        atomic_json(progress_path, {"state": "candidate", "completed": len(records), "total": len(records),
                                    "independent_acceptance_pending": True, "updated_at": _now()})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--quality-dir", type=Path)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-vision-root")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worker-control", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--diagnostic", action="store_true", help="Candidate only; allow an explicitly partial HM audit")
    args = parser.parse_args()
    if (args.quality_report is None) == (args.quality_dir is None):
        parser.error("Provide exactly one of --quality-report or --quality-dir")
    quality_report = args.quality_report or args.quality_dir / "report.json"
    result = build_revision(args.source_root, quality_report, args.output_root,
                            source_vision_root=args.source_vision_root, workers=args.workers,
                            worker_control=args.worker_control, resume=args.resume, diagnostic=args.diagnostic)
    print(json.dumps({"state": result["state"], "output_root": result["output_root"],
                      "vision_profile_root": result["vision_profile_root"],
                      "rewritten_episodes": len(result["motion_rewritten_episodes"]),
                      "independent_acceptance_pending": True}), flush=True)


if __name__ == "__main__":
    main()
