"""Independently audit only HM quality changes; optionally publish the revision.

Unchanged accepted motion and all DINO payloads are inherited by hard-link
identity. Rewritten HM motion is compared column-by-column, including the
native visual join. Original physical qualification is never repeated.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .dataset.revision_acceptance import (
    ACCEPTANCE_FORMAT, AUDIT_FORMAT, REVISION_FORMAT, check_revision_acceptance,
)
from .dataset.smplh_archive import SMPLHParameterArchive, file_sha256
from .dataset.smplh_motion_dataset import SMPLHMotionDataset
from .dataset.vision_cache import JOIN_DTYPES, storage_digest
from .dataset.window_cache import atomic_json, digest
from .watch_dino import locked

AUDIT_PROOF_FORMAT = "independent-smplh-hm-quality-revision-delta-audit-v1"
PROFILE = "vision_profiles/psi0_256x480_v1_30hz"
FLAGS = ("parent_acceptance_inherited", "parameter_values_preserved", "camera_values_preserved",
         "image_join_values_preserved", "quality_masks_accepted", "ground_recomputed_accepted",
         "latest_motionexpert_accepted")
FLOOR_COLUMNS = {"observation.environment.floor_height", "observation.valid.floor", "observation.valid.floor_verified"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _owned(root, relative, *, exists=True):
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("Independent revision evidence requires owned relative paths")
    path = (root / value).resolve(strict=exists)
    if not path.is_relative_to(root) or (exists and not path.is_file()):
        raise ValueError("Independent revision evidence escapes its archive")
    return path


def _same_file(first, second, name):
    if not os.path.samefile(first, second):
        raise ValueError(f"Inherited {name} is not the immutable original hard link")


def _arrays(table):
    values = {}
    for name in table.schema.names:
        column = table[name]
        if isinstance(column, pa.ChunkedArray):
            column = column.combine_chunks()
        if column.null_count:
            raise ValueError(f"Unexpected null cells in revision column {name}")
        value = (column.values.to_numpy(zero_copy_only=False).reshape(len(table), column.type.list_size)
                 if pa.types.is_fixed_size_list(column.type) else column.to_numpy(zero_copy_only=False))
        values[name] = np.asarray(value)
    return values


def _assert_values(actual, expected, context):
    if actual.dtype != expected.dtype or actual.shape != expected.shape or not np.array_equal(actual, expected, equal_nan=True):
        raise ValueError(f"Revision value mismatch: {context}")


def _expected_masks(old, good):
    """Independent per-column rules; do not call the revision writer."""
    if (not np.array_equal(old["source.feet_valid"], old["source.observed.body_position"][:, [10, 11]])
            or not np.array_equal(old["source.feet_interpolated"], old["source.interpolated.body_position"][:, [10, 11]])):
        raise ValueError("Parent foot masks differ from the accepted body-observation protocol")
    expected = {}
    for component in ("position", "rotation"):
        available = old[f"estimated.valid.body_{component}"] & good[:, None]
        observed = old[f"source.observed.body_{component}"] & good[:, None]
        expected[f"estimated.valid.body_{component}"] = available
        expected[f"source.observed.body_{component}"] = observed
        expected[f"source.interpolated.body_{component}"] = old[f"source.interpolated.body_{component}"] & observed
        expected[f"estimated.completed.body_{component}"] = available & ~observed
        expected[f"quality.{component}_confidence"] = np.where(available, old[f"quality.{component}_confidence"], 0)
    expected["quality.observation_residual_valid"] = (old["quality.observation_residual_valid"]
        & expected["estimated.valid.body_position"] & expected["source.observed.body_position"])
    expected["source.feet_valid"] = old["source.feet_valid"] & expected["source.observed.body_position"][:, [10, 11]]
    expected["source.feet_interpolated"] = old["source.feet_interpolated"] & expected["source.interpolated.body_position"][:, [10, 11]]
    for side in ("left", "right"):
        available = old[f"observation.valid.hand_{side}_position"] & good[:, None]
        expected[f"observation.valid.hand_{side}_position"] = available
        expected[f"observation.interpolated.hand_{side}_position"] = old[f"observation.interpolated.hand_{side}_position"] & available
    if "smplh.valid.hand_pose" in old:
        expected["smplh.valid.hand_pose"] = old["smplh.valid.hand_pose"] & good[:, None]
    return expected


def _check_unchanged_quality_masks(parent, record, good):
    """Only new rejected-frame masks are read; accepted XYZ/FK stays inherited."""
    selected = np.flatnonzero(~good)
    if not len(selected):
        return 0
    path = parent.data_path(record["episode_index"])
    parquet = pq.ParquetFile(path)
    columns = {f"{prefix}.body_{component}" for component in ("position", "rotation")
               for prefix in ("estimated.valid", "source.observed", "source.interpolated", "estimated.completed")}
    columns.update({"quality.position_confidence", "quality.rotation_confidence", "quality.observation_residual_valid",
                    "source.feet_valid", "source.feet_interpolated"})
    columns.update({f"observation.{prefix}.hand_{side}_position"
                    for prefix in ("valid", "interpolated") for side in ("left", "right")})
    if "smplh.valid.hand_pose" in parquet.schema_arrow.names:
        columns.add("smplh.valid.hand_pose")
    old = parent.reader.read(path, selected, sorted(columns))
    expected = _expected_masks(old, np.zeros(len(selected), bool))
    for key, value in expected.items():
        _assert_values(old[key], value, f"producer-declared unchanged HM episode {record['episode_index']}, newly screened masks {key}")
    return len(selected)


def reference_ground(feet, feet_valid, root_z, root_valid, metadata):
    """Independent original ground policy, after gating the valid foot samples.

    This follows the historical pipeline stationary-foot DBSCAN definition.
    Only this new ground derivative is checked; accepted body FK is not rerun.
    """
    previous = metadata.get("ground", {})
    if previous.get("mode", "estimate") != "estimate":
        return deepcopy(previous)
    result = {"height_m": 0., "valid": False, "verified": False, "normal": [0., 0., 1.],
        "frame": "output_world", "mode": "estimate", "foot_joint_indices": [10, 11],
        "window_fallback": deepcopy(previous.get("window_fallback", {"mode": "mean_valid_feet", "offset_m": .02}))}
    if not metadata.get("coordinate_verified", False):
        result["reason"] = "output_world_vertical_axis_unverified"
        return result
    original = previous.get("parameters", {})
    parameters = {"static_speed_m_s": float(original.get("static_speed_m_s", .15)),
        "cluster_eps_m": float(original.get("cluster_eps_m", .005)),
        "cluster_min_samples": original.get("cluster_min_samples", 3),
        "toe_offset_m": float(original.get("toe_offset_m", .01)),
        "max_gap_s": float(original.get("max_gap_s", .075)), "noise_policy": "exclude_label_minus_one"}
    if (type(parameters["cluster_min_samples"]) is not int or parameters["cluster_min_samples"] < 1
            or any(not np.isfinite(parameters[key]) or parameters[key] <= 0 for key in (
                "static_speed_m_s", "cluster_eps_m", "max_gap_s"))
            or not np.isfinite(parameters["toe_offset_m"]) or parameters["toe_offset_m"] < 0):
        raise ValueError("Invalid inherited ground thresholds")
    result.update(method="stationary_foot_height_dbscan", parameters=parameters, clusters=[])
    times = np.arange(len(feet), dtype=np.float64) / 30
    if len(times) < 2:
        result["reason"] = "too_few_frames"
        return result
    valid = feet_valid & np.isfinite(feet).all(-1)
    delta = np.diff(times)
    edge_valid = valid[1:] & valid[:-1] & (delta[:, None] <= parameters["max_gap_s"])
    velocities = np.linalg.norm(feet[1:] - feet[:-1], axis=-1) / delta[:, None]
    edge_speeds = np.where(edge_valid, velocities, 0)
    peak = np.maximum(np.vstack((edge_speeds, np.zeros((1, 2)))),
                      np.vstack((np.zeros((1, 2)), edge_speeds)))
    seen = np.vstack((edge_valid, np.zeros((1, 2), bool))) | np.vstack((np.zeros((1, 2), bool), edge_valid))
    frame, foot = np.nonzero(valid & seen & (peak <= parameters["static_speed_m_s"]))
    height = feet[frame, foot, 2]
    result["candidate_count"] = len(height)
    if not len(height):
        result["reason"] = "no_stationary_feet"
        return result
    from sklearn.cluster import DBSCAN
    unique, inverse, weights = np.unique(height, return_inverse=True, return_counts=True)
    labels = DBSCAN(eps=parameters["cluster_eps_m"], min_samples=parameters["cluster_min_samples"]).fit(
        unique[:, None], sample_weight=weights).labels_[inverse]
    result["noise_count"] = int((labels == -1).sum())
    for label in np.unique(labels[labels >= 0]):
        members = labels == label
        frames = np.unique(frame[members])
        spans = np.diff(times[frames])
        roots = root_z[frames[root_valid[frames]]]
        result["clusters"].append({"median_height_m": float(np.median(height[members])),
            "root_median_height_m": float(np.median(roots)) if len(roots) else None,
            "sample_count": int(members.sum()), "support_s": float(spans[spans <= parameters["max_gap_s"]].sum()),
            "spread_m": float(np.ptp(height[members]))})
    if not result["clusters"]:
        result["reason"] = "no_density_height_cluster"
        return result
    chosen = min(result["clusters"], key=lambda item: item["median_height_m"])
    result.update(height_m=chosen["median_height_m"] - parameters["toe_offset_m"], valid=True,
        source="estimated_stationary_feet", reason="estimated", multiple_height_levels=any(
            item["median_height_m"] > chosen["median_height_m"] + .04 for item in result["clusters"]))
    return result


def _quality(root, receipt, parent_record, parent_record_path):
    if (receipt.get("format") != AUDIT_FORMAT or receipt.get("episode_index") != parent_record["episode_index"]
            or receipt.get("source_key") != parent_record["source_key"] or receipt.get("split") != parent_record["split"]
            or receipt.get("record_sha256") != file_sha256(parent_record_path)
            or receipt.get("motion_sha256") != parent_record["motion_sha256"]):
        raise ValueError("Independent quality receipt/source identity differs")
    original_identity = parent_record["metadata"].get("source_identity")
    if original_identity is not None and receipt.get("source_identity") != original_identity:
        raise ValueError("Quality native source identity differs from its accepted parent")
    path = _owned(root, receipt["mask_path"])
    if file_sha256(path) != receipt.get("mask_sha256"):
        raise ValueError("Independent quality sidecar SHA differs")
    with np.load(path, allow_pickle=False) as data:
        required = {"quality_valid", "native_valid", "fit_finite", "transition_valid", "reasons"}
        if not required <= set(data.files):
            raise ValueError("Quality sidecar lacks required masks")
        values = {key: data[key].copy() for key in required}
    count = parent_record["length"]
    if any(value.shape != (count,) for value in values.values()) or any(
            values[key].dtype != np.bool_ for key in required - {"reasons"}) or values["reasons"].dtype != np.uint16:
        raise ValueError("Quality mask shape/dtype differs")
    if hashlib.sha256(values["native_valid"].tobytes()).hexdigest() != receipt.get("native_valid_sha256"):
        raise ValueError("Native-valid quality upstream SHA differs")
    expected = values["native_valid"] & values["fit_finite"] & ((values["reasons"] & (8 | 16)) == 0)
    if not np.array_equal(values["quality_valid"], expected):
        raise ValueError("Quality frame masks differ from conservative rejection reasons")
    if count and (values["transition_valid"][0] or np.any(values["transition_valid"][1:]
            & ~(values["quality_valid"][1:] & values["quality_valid"][:-1]))):
        raise ValueError("Quality transition mask exceeds available endpoints")
    return values["quality_valid"]


def _check_rewritten(parent, candidate, old, new, good, receipt, batch_rows=4096):
    index = old["episode_index"]
    before, after = parent.data_path(index), candidate.data_path(index)
    original_file, revised_file = pq.ParquetFile(before), pq.ParquetFile(after)
    if (original_file.metadata.num_rows != old["length"] or revised_file.metadata.num_rows != old["length"]
            or original_file.schema_arrow != revised_file.schema_arrow):
        raise ValueError("Rewritten HM Parquet schema/row count differs")
    if file_sha256(after) != new["motion_sha256"]:
        raise ValueError("Rewritten HM motion SHA differs from its committed record")
    join_path = candidate.vision_cache._episodes[index]["artifacts"]["join.npz"][0]
    with np.load(join_path, allow_pickle=False) as saved:
        if set(saved.files) != set(JOIN_DTYPES):
            raise ValueError("Rewritten HM visual join columns differ")
        join = {key: saved[key].copy() for key in JOIN_DTYPES}
    for key, dtype in JOIN_DTYPES.items():
        if join[key].shape != (old["length"],) or join[key].dtype != np.dtype(dtype):
            raise ValueError("Rewritten HM visual join shape/dtype differs")
    collections = {key: [] for key in ("source.feet_world", "source.feet_valid", "smplh.betas", "smplh.trans", "estimated.valid.body_position")}
    offset, revised_cells, changed_columns = 0, 0, set()
    feet_changed = False
    left = original_file.iter_batches(batch_size=batch_rows, use_threads=False)
    right = revised_file.iter_batches(batch_size=batch_rows, use_threads=False)
    for old_batch, new_batch in zip_longest(left, right):
        if old_batch is None or new_batch is None or old_batch.num_rows != new_batch.num_rows:
            raise ValueError("Rewritten HM batch population differs")
        a, b = _arrays(old_batch), _arrays(new_batch)
        count = len(old_batch)
        expected = _expected_masks(a, good[offset:offset + count])
        for key in a:
            if key in FLOOR_COLUMNS:
                ground = new["metadata"]["ground"]
                value = (np.float32(ground["height_m"]) if key.endswith("floor_height") else
                    bool(ground["valid"] and (not key.endswith("floor_verified") or ground.get("verified", False))))
                _assert_values(b[key], np.full_like(a[key], value), f"episode {index}, {key}")
            else:
                _assert_values(b[key], expected.get(key, a[key]), f"episode {index}, rows {offset}:{offset+count}, {key}")
            if not np.array_equal(a[key], b[key], equal_nan=True):
                changed_columns.add(key)
                if key in expected:
                    revised_cells += int(np.count_nonzero(a[key] != b[key]))
        feet_changed |= not np.array_equal(a["source.feet_valid"], b["source.feet_valid"])
        for key in collections:
            collections[key].append(b[key].copy())
        for key in JOIN_DTYPES:
            source_key = "observation.valid.dino" if key == "source.valid.dino" else key
            expected_join = b.get(source_key)
            if expected_join is None and key == "source.video_frame_index":
                expected_join = np.arange(offset, offset + count, dtype=np.int64)
            if expected_join is None:
                raise ValueError(f"Rewritten HM lacks native visual join column {key}")
            _assert_values(join[key][offset:offset + count], expected_join, f"episode {index} native visual join {key}")
        offset += count
    if offset != old["length"] or receipt.get("revised_mask_cells") != revised_cells:
        raise ValueError("Rewritten HM mask-cell/frame population differs")
    if feet_changed:
        arrays = {key: np.concatenate(value) for key, value in collections.items()}
        root_z = (candidate.model.rest[0][None] + np.einsum("tb,cb->tc", arrays["smplh.betas"], candidate.model.shape_basis[0])
                  + arrays["smplh.trans"])[:, 2]
        ground = reference_ground(arrays["source.feet_world"].reshape(-1, 2, 3), arrays["source.feet_valid"], root_z,
                                  arrays["estimated.valid.body_position"][:, 0], old["metadata"])
    else:
        ground = old["metadata"].get("ground")
    if new["metadata"].get("ground") != ground:
        raise ValueError(f"Independent gated-foot ground differs at episode {index}")
    if new["metadata"]["hm_quality_revision"].get("ground_recomputed") is not feet_changed:
        raise ValueError("Revision ground-recomputed declaration differs from changed feet")
    return {"episode_index": index, "frames_compared": offset, "changed_columns": sorted(changed_columns),
            "revised_mask_cells": revised_cells, "ground_recomputed": feet_changed,
            "native_visual_join_all_rows_compared": True}


def _snapshot(parent, candidate, revision_path, quality_path):
    identities = []
    for role, archive in (("parent", parent), ("candidate", candidate)):
        files = {**archive._metadata_stamps, **archive._artifact_stamps,
                 **{archive._record_paths[index]: value for index, value in archive._record_stamps.items()}}
        for path in sorted(files, key=str):
            stat = path.stat()
            identities.append([role, str(path.relative_to(archive.root)), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns])
        if archive.vision_cache is not None:
            info = archive.vision_cache.manifest_path.stat()
            identities.append([role, str(archive.vision_cache.manifest_path),
                               info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns])
            for entry in archive.vision_cache._episodes.values():
                for path in [entry["receipt_path"], *[value[0] for value in entry["artifacts"].values()]]:
                    stat = path.stat()
                    identities.append([role, str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns])
    for path in sorted((candidate.root / ".pipeline/quality").iterdir()):
        if path.is_file():
            stat = path.stat()
            identities.append(["quality", str(path.relative_to(candidate.root)), stat.st_dev,
                               stat.st_ino, stat.st_size, stat.st_mtime_ns])
    for relative in (".pipeline/config.json", ".pipeline/revision-plan.json", "meta/ontology.json", "meta/summary.json",
                     ".pipeline/parent-publication/config.json"):
        path = candidate.root / relative
        if path.is_file():
            info = path.stat()
            identities.append(["preserved", relative, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns])
    return digest({"files": identities, "parent_publication_sha256": file_sha256(parent.state_path),
        "revision_manifest_sha256": file_sha256(revision_path), "quality_audit_sha256": file_sha256(quality_path)})


def _quality_stamps(root):
    return [(str(path.relative_to(root)), path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted((root / ".pipeline/quality").iterdir()) if path.is_file()]


def _contexts(parent_root, revision_root, parent_vision_root=None):
    root = Path(revision_root).resolve(strict=True)
    plan = json.loads(_owned(root, ".pipeline/revision-plan.json").read_text())
    parent_root = Path(parent_root).resolve(strict=True)
    if (parent_root == root or Path(plan.get("source_root", "")).resolve() != parent_root
            or Path(plan.get("output_root", "")).resolve() != root):
        raise ValueError("Revision plan names a different immutable parent")
    parent_vision = Path(parent_vision_root or plan["source_vision_root"]).resolve(strict=True)
    if parent_vision != Path(plan["source_vision_root"]).resolve():
        raise ValueError("Revision plan names a different original DINO profile")
    parent = SMPLHParameterArchive(parent_root, vision_profile_root=parent_vision)
    candidate = SMPLHParameterArchive(root, allow_unpublished=True, vision_profile_root=root / PROFILE)
    return parent, candidate, plan


def audit_revision(parent_root, revision_root, *, parent_vision_root=None, progress=None):
    root = Path(revision_root).resolve(strict=True)
    with locked(root / ".revision-build.lock"):
        return _audit_revision(parent_root, root, parent_vision_root=parent_vision_root, progress=progress)


def _audit_revision(parent_root, revision_root, *, parent_vision_root=None, progress=None):
    parent, candidate, plan = _contexts(parent_root, revision_root, parent_vision_root)
    root = candidate.root
    revision_path, quality_path = _owned(root, ".pipeline/hm-quality-revision.json"), _owned(root, ".pipeline/quality/report.json")
    revision, quality = json.loads(revision_path.read_text()), json.loads(quality_path.read_text())
    build = revision.get("build_binding", {})
    quality_sha = file_sha256(quality_path)
    build_sha = digest(build)
    quality_stamps = _quality_stamps(root)
    if (candidate.state.get("state") != "candidate" or candidate.state.get("independent_acceptance_pending") is not True
            or revision.get("format") != REVISION_FORMAT or revision.get("scope") != "full"
            or revision.get("binding") != candidate.binding or revision.get("archive_fingerprint") != candidate.archive_fingerprint
            or build != plan.get("binding") or revision.get("build_binding_sha256") != build_sha
            or plan.get("binding_sha256") != build_sha or build.get("scope") != "full"
            or build.get("parent_archive_binding") != parent.binding
            or build.get("parent_archive_fingerprint") != parent.archive_fingerprint
            or build.get("quality_report_sha256") != quality_sha
            or Path(revision.get("source_root", "")).resolve() != parent.root
            or Path(revision.get("output_root", "")).resolve() != root):
        raise ValueError("Revision candidate/manifest/parent/QC binding differs or is partial")
    if (quality.get("format") != AUDIT_FORMAT or quality.get("state") != "complete" or quality.get("diagnostic") is not False
            or quality.get("production_modified") is not False
            or quality.get("source_hm_population", {}).get("scope", quality.get("scope")) != "all_archive_HM"
            or Path(quality.get("archive_root", "")).resolve() != parent.root):
        raise ValueError("Revision requires complete original all-HM read-only quality audit")
    old_records, new_records = parent.records(), candidate.records()
    if len(old_records) != len(new_records) or parent.frames != candidate.frames:
        raise ValueError("Revision altered original archive population")
    if parent.split_fingerprint != candidate.split_fingerprint or parent.model_sha256 != candidate.model_sha256:
        raise ValueError("Revision changed original split or model")
    if (revision.get("model_sha256") != candidate.model_sha256 or revision.get("parent_model_sha256") != parent.model_sha256
            or revision.get("split_fingerprint") != candidate.split_fingerprint
            or revision.get("parent_split_fingerprint") != parent.split_fingerprint):
        raise ValueError("Revision declared model/split identity differs")
    old_info, new_info = deepcopy(parent.info), deepcopy(candidate.info)
    new_info.pop("quality_revision", None)
    if old_info != new_info:
        raise ValueError("Revision altered original schema/model metadata")
    preserved = {relative: file_sha256(parent.root / relative)
        for relative in ("meta/ontology.json", "meta/summary.json", ".pipeline/config.json")
        if (parent.root / relative).is_file()}
    if build.get("preserved_metadata_sha256") != preserved:
        raise ValueError("Revision preserved ontology/inventory/configuration binding differs")
    for relative in preserved:
        destination = ".pipeline/parent-publication/config.json" if relative == ".pipeline/config.json" else relative
        _same_file(parent.root / relative, _owned(root, destination), "ontology/inventory/parent configuration")
    standalone = json.loads(_owned(root, ".pipeline/config.json").read_text())
    if (standalone.get("format") != REVISION_FORMAT or standalone.get("standalone") is not True
            or standalone.get("requires_old_dense_archive") is not False
            or standalone.get("requires_parent_archive_at_training") is not False):
        raise ValueError("Revision standalone configuration still requires an obsolete archive")
    _same_file(parent.model_path, candidate.model_path, "SMPL-H model")
    for path in (parent.root / "meta").glob("splits*.json"):
        _same_file(path, root / "meta" / path.name, "source split")
    publication = _owned(root, ".pipeline/parent-publication/dataset-state.json")
    _same_file(parent.state_path, publication, "parent publication")
    if file_sha256(publication) != build.get("parent_publication_sha256"):
        raise ValueError("Inherited parent publication binding differs")
    _same_file(Path(plan["quality_report"]), quality_path, "quality audit")
    old_visual, new_visual = parent.vision_cache.manifest, candidate.vision_cache.manifest
    expected_visual = deepcopy(old_visual)
    expected_visual.update(source_root=str(root), source_binding=new_visual.get("source_binding"),
                           quality_revision_reuse=new_visual.get("quality_revision_reuse"))
    if old_visual.get("encoder") != new_visual.get("encoder") or new_visual != expected_visual:
        raise ValueError("Revision changed the frozen DINO encoder/image-processing declaration")
    quality_receipts = quality.get("episodes", [])
    qc = {item["episode_index"]: item for item in quality_receipts}
    required = {r["episode_index"] for r in old_records if r["source"] == "egoexo4d_hm"
        and r["metadata"].get("training_eligible") is not False and r["metadata"].get("canonical_training_eligible") is not False
        and not r["metadata"].get("quarantine_reasons", [])}
    if (len(qc) != len(quality_receipts) or set(qc) != required or quality.get("episodes_completed") != len(qc)
            or revision.get("quality_episodes") != sorted(qc) or revision.get("required_quality_episodes") != sorted(required)):
        raise ValueError("Quality audit/revision does not cover exactly the full eligible HM population")
    rewritten = revision.get("motion_rewritten_episodes", [])
    if (len(set(rewritten)) != len(rewritten) or not set(rewritten) <= required
            or revision.get("changed_episode_indices") != rewritten):
        raise ValueError("Rewritten HM episode scope differs")
    sources, splits, comparisons, inherited = {}, {name: {"episodes": 0, "frames": 0} for name in ("train", "val", "test")}, [], 0
    inherited_quality_mask_frames_checked = 0
    for old, new in zip(old_records, new_records):
        index = old["episode_index"]
        counts = sources.setdefault(old["source"], {"episodes": 0, "frames": 0})
        counts["episodes"] += 1; counts["frames"] += old["length"]
        split_counts = splits.setdefault(old["split"], {"episodes": 0, "frames": 0})
        split_counts["episodes"] += 1; split_counts["frames"] += old["length"]
        old_receipt_path = parent.owned_path(f"episodes/{index:08d}/receipt.json")
        receipt = json.loads(candidate.owned_path(f"episodes/{index:08d}/receipt.json").read_text())
        entry = receipt.get("quality_revision", {})
        if (entry.get("build_binding_sha256") != build_sha or entry.get("parent_record_sha256") != digest(old)
                or entry.get("parent_receipt_sha256") != file_sha256(old_receipt_path)
                or entry.get("motion_rewritten") is not (index in rewritten)):
            raise ValueError(f"Revision episode receipt parent/scope binding differs: {index}")
        expected = deepcopy(old)
        if index in qc:
            quality_receipt = qc[index]
            owned_qc = _owned(root, f".pipeline/quality/episode_{index:08d}.json")
            if json.loads(owned_qc.read_text()) != quality_receipt or quality_receipt.get("policy") != quality.get("policy"):
                raise ValueError("Quality episode receipt/policy differs from full report")
            descriptor = new["metadata"].get("hm_quality_revision", {})
            if (descriptor.get("format") != REVISION_FORMAT or descriptor.get("quality_mask_sha256") != quality_receipt["mask_sha256"]
                    or descriptor.get("native_valid_sha256") != quality_receipt["native_valid_sha256"]
                    or descriptor.get("quality_receipt_sha256") != digest(quality_receipt)
                    or descriptor.get("quality_report_sha256") != quality_sha
                    or descriptor.get("policy") != quality_receipt["policy"]
                    or descriptor.get("parent_ground") != old["metadata"].get("ground")
                    or entry.get("quality_mask_sha256") != quality_receipt["mask_sha256"]):
                raise ValueError("HM quality metadata provenance differs")
            good = _quality(root / ".pipeline/quality", quality_receipt, old, parent._record_paths[index])
            expected["metadata"]["hm_quality_revision"] = descriptor
            expected["metadata"]["ground"] = new["metadata"]["ground"]
            expected["motion_sha256"] = new["motion_sha256"]
        else:
            good = None
            if entry.get("quality_mask_sha256") is not None:
                raise ValueError("Unscreened original episode has a quality mask receipt")
        if new != expected:
            raise ValueError(f"Revision changed unauthorized episode metadata/identity: {index}")
        if index in rewritten:
            comparisons.append(_check_rewritten(parent, candidate, old, new, good, entry))
        else:
            if old["motion_sha256"] != new["motion_sha256"]:
                raise ValueError("Unchanged motion SHA was altered")
            _same_file(parent.data_path(index), candidate.data_path(index), "unchanged motion")
            if old["metadata"].get("ground") != new["metadata"].get("ground") or entry.get("revised_mask_cells") != 0:
                raise ValueError("Unchanged motion has revised masks or ground")
            if good is not None and new["metadata"]["hm_quality_revision"].get("ground_recomputed") is not False:
                raise ValueError("Unchanged feet ground must be inherited")
            if good is not None:
                inherited_quality_mask_frames_checked += _check_unchanged_quality_masks(parent, old, good)
            inherited += 1
        for name in ("features.npy", "join.npz"):
            _same_file(parent.vision_cache._episodes[index]["artifacts"][name][0],
                       candidate.vision_cache._episodes[index]["artifacts"][name][0], "DINO " + name)
        if entry.get("legacy_vision_retired") is not (index in parent._retired_vision):
            raise ValueError("Revision changed the legacy-vision retirement boundary")
        if index not in parent._retired_vision:
            _same_file(parent.owned_path(old["vision_path"]), candidate.owned_path(new["vision_path"]), "legacy vision")
        visual = json.loads(candidate.vision_cache._episodes[index]["receipt_path"].read_text())
        visual_reuse = visual.get("quality_revision_reuse", {})
        if (visual_reuse.get("build_binding_sha256") != build_sha
                or visual_reuse.get("parent_receipt_sha256") != file_sha256(parent.vision_cache._episodes[index]["receipt_path"])
                or visual_reuse.get("parent_manifest_sha256") != parent.vision_cache.metadata["manifest_sha256"]
                or visual_reuse.get("join_all_native_rows_checked") is not (index in rewritten)):
            raise ValueError("DINO receipt inherited binding or changed-row evidence differs")
        if progress is not None:
            progress(index, len(new_records), len(comparisons))
    population = {"episodes": len(new_records), "frames": candidate.frames, "sources": sources}
    if revision.get("population") != population or candidate.state.get("sources") != sources:
        raise ValueError("Revision source/split population declaration differs")
    if "meta/summary.json" in preserved:
        summary = json.loads((root / "meta/summary.json").read_text())
        if (summary.get("episodes") != len(new_records) or summary.get("frames") != candidate.frames
                or summary.get("sources") != sources or summary.get("splits") != splits):
            raise ValueError("Inherited archive inventory differs from original source/split/frame population")
    compatibility = _motionexpert_compatibility(candidate, rewritten)
    parent.assert_current(); candidate.assert_current()
    if _quality_stamps(root) != quality_stamps or file_sha256(quality_path) != quality_sha:
        raise ValueError("Quality evidence changed during independent revision audit")
    report = {"format": AUDIT_PROOF_FORMAT, "scope": "full", "passed": True, "failures": [],
        "created_at": _now(), "binding": candidate.binding, "population": population,
        "parent_archive_binding": parent.binding, "parent_archive_fingerprint": parent.archive_fingerprint,
        "validator_sha256": file_sha256(__file__), "source_split_population": splits,
        "quality_audit_sha256": quality_sha, "revision_manifest_sha256": file_sha256(revision_path),
        "artifact_snapshot_sha256": _snapshot(parent, candidate, revision_path, quality_path),
        "changed_episode_indices": rewritten, "changed_motion_comparisons": comparisons,
        "inherited_unchanged_motion_episodes": inherited, "unchanged_motion_full_numeric_QA_repeated": False,
        "inherited_HM_new_quality_bad_frame_masks_checked": inherited_quality_mask_frames_checked,
        "dino_payload_rehashed": False, "dino_join_rows_compared_only_for_rewritten_motion": True,
        "previous_physical_acceptance_repeated": False, "latest_motionexpert": compatibility,
        **{name: True for name in FLAGS}}
    return report


def _motionexpert_compatibility(candidate, rewritten):
    """A few changed reader windows, without old data requalification or GPU work."""
    eligible = [index for index in rewritten if candidate._records[index]["split"] == "train"]
    selected = list(eligible[:3] or rewritten[:3])
    if not selected:
        return {"method": "inherited_unchanged_motion", "windows": [], "gpu_checks_repeated": False}
    try:
        data = SMPLHMotionDataset(archive_root=candidate.root, split=candidate._records[selected[0]]["split"],
            allow_unpublished=True, episode_indices=selected,
            do_normalization=False, cond_img_feat=False, vision_profile_root=candidate.vision_cache.root,
            min_valid_image_fraction=.5, verified_archive=candidate)
    except ValueError as error:
        if not str(error).startswith("No usable complete SMPL-H windows"):
            raise
        return {"method": "changed_examples_have_no_retained_windows", "windows": [], "gpu_checks_repeated": False}
    windows = []
    for index in sorted({0, len(data)//2, len(data)-1}):
        sample = data[index]
        motion, trajectory = sample["misc"]["motion"], sample["misc"]["traj"]
        if motion.shape != (11, 245) or trajectory.shape != (11, 18) or not torch.isfinite(motion).all() or not torch.isfinite(trajectory).all():
            raise ValueError("Changed HM revision is incompatible with current MotionExpert245/18")
        if (not torch.equal(motion[:, 108:117], trajectory[:, 9:18])
                or not torch.equal(sample["misc"]["feature_valid_mask"][:, 108:117], sample["misc"]["trajectory_valid"][:, 9:18])):
            raise ValueError("Changed HM revision motion/trajectory global delta differs")
        windows.append({"episode_index": sample["misc"]["episode_index"], "start_frame": sample["misc"]["start_idx"]})
    return {"method": "changed_hm_CPU_reader_5hz_11frames_245motion_18trajectory",
            "windows": windows, "gpu_checks_repeated": False}


def _write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def publish_revision(parent_root, revision_root, independent_report, *, parent_vision_root=None):
    """Publish only a bound owned independent decision; no training is started."""
    root = Path(revision_root).resolve(strict=True)
    with locked(root / ".revision-build.lock"):
        return _publish_revision(parent_root, root, independent_report, parent_vision_root=parent_vision_root)


def _publish_revision(parent_root, revision_root, independent_report, *, parent_vision_root=None):
    parent, candidate, plan = _contexts(parent_root, revision_root, parent_vision_root)
    root = candidate.root
    supplied = Path(independent_report).resolve(strict=True)
    if not supplied.is_relative_to(root):
        raise ValueError("Independent revision report must be owned by the candidate archive")
    if candidate.state.get("state") != "candidate" or candidate.state.get("independent_acceptance_pending") is not True:
        raise ValueError("Only an unpublished candidate may be published once")
    report = json.loads(supplied.read_text())
    revision_path, quality_path = _owned(root, ".pipeline/hm-quality-revision.json"), _owned(root, ".pipeline/quality/report.json")
    if (report.get("format") != AUDIT_PROOF_FORMAT or report.get("passed") is not True
            or report.get("scope") != "full" or report.get("failures") != []
            or report.get("binding") != candidate.binding
            or report.get("validator_sha256") != file_sha256(__file__)
            or report.get("artifact_snapshot_sha256") != _snapshot(parent, candidate, revision_path, quality_path)
            or any(report.get(name) is not True for name in FLAGS)):
        raise ValueError("Independent revision audit is missing, partial, or stale")
    proof = lambda path: {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
    proofs = {"revision_manifest": proof(revision_path), "quality_audit": proof(quality_path),
        "independent_revision_audit": proof(supplied),
        "parent_publication": proof(_owned(root, ".pipeline/parent-publication/dataset-state.json"))}
    if plan["binding"].get("parent_acceptance_sha256"):
        proofs["parent_acceptance"] = proof(_owned(root, ".pipeline/parent-publication/acceptance-report.json"))
    top = {"format": ACCEPTANCE_FORMAT, "scope": "full", "passed": True, "failures": [],
        "population": report["population"], "binding": candidate.binding, "proofs": proofs,
        "created_at": _now(), "previous_physical_acceptance_repeated": False,
        **{name: True for name in FLAGS}}
    top_path = root / ".pipeline/acceptance-report.json"
    check_revision_acceptance(candidate, top, top_path)
    parent.assert_current(); candidate.assert_current()
    _write_new(top_path, top)
    state = dict(candidate.state, state="accepted", independent_acceptance_pending=False,
                 acceptance_report=str(top_path.relative_to(root)), acceptance_report_sha256=file_sha256(top_path),
                 accepted_at=_now(), formal_training_ready=False)
    atomic_json(candidate.state_path, state)
    return top


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--parent-vision-root", type=Path)
    parser.add_argument("--report", type=Path, help="Owned new report; default .pipeline/independent-revision-audit.json")
    parser.add_argument("--publish", action="store_true", help="Publish after independent delta acceptance; never start training")
    args = parser.parse_args()
    torch.set_num_threads(1); pa.set_cpu_count(1); pa.set_io_thread_count(1)
    root = args.archive_root.resolve(strict=True)
    output = (args.report or root / ".pipeline/independent-revision-audit.json").resolve()
    if not output.is_relative_to(root):
        parser.error("--report must be owned by the candidate archive")
    if output.exists():
        if not args.publish:
            raise FileExistsError("Independent audit already exists; no automatic repeat")
    else:
        def progress(index, total, rewritten):
            if index == 0 or (index+1) % 100 == 0 or index+1 == total:
                print(json.dumps({"episode_checked": index+1, "total": total, "rewritten_compared": rewritten}), flush=True)
        report = audit_revision(args.parent_root, root, parent_vision_root=args.parent_vision_root, progress=progress)
        _write_new(output, report)
    if args.publish:
        publish_revision(args.parent_root, root, output, parent_vision_root=args.parent_vision_root)
    print(json.dumps({"passed": True, "report": str(output), "published": bool(args.publish),
                      "training_started": False}), flush=True)


if __name__ == "__main__":
    main()
