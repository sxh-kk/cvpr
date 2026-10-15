"""Independently accept dual-source final-FK quality changes, never training.

Only newly rewritten motion is compared over all columns. The new head-specific
physical gate is independently recomputed for every eligible episode from the
preserved final SMPL-H parameters. Old source accuracy qualification and DINO
payload scans are inherited rather than repeated.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from . import validate_hm_revision as prior
from .dataset.fit_revision_acceptance import (ACCEPTANCE_FORMAT, AUDIT_FORMAT, REVISION_FORMAT,
    INDEPENDENT_AUDIT_FORMAT, ACCEPTANCE_FLAGS, check_fit_revision_acceptance, eligible)
from .dataset.smplh_archive import PARENTS, file_sha256
from .dataset.vision_cache import JOIN_DTYPES, SOURCE_JOIN
from .dataset.window_cache import atomic_json, digest
from .watch_dino import locked

FLAGS = ACCEPTANCE_FLAGS
PROFILE = prior.PROFILE
_owned, _same_file, _assert_values = prior._owned, prior._same_file, prior._assert_values
_write_new = prior._write_new


def _expected_masks(old, good, source):
    if source == "egoexo4d_hm":
        return prior._expected_masks(old, good)
    if source != "egosuite":
        raise ValueError("Final-FK screening supports exactly the two unified sources")
    expected = {}
    for component in ("position", "rotation"):
        valid = old[f"estimated.valid.body_{component}"] & good[:, None]
        expected[f"estimated.valid.body_{component}"] = valid
        expected[f"estimated.completed.body_{component}"] = valid & ~old[f"source.observed.body_{component}"]
        expected[f"quality.{component}_confidence"] = np.where(valid, old[f"quality.{component}_confidence"], 0)
    expected["quality.observation_residual_valid"] = (old["quality.observation_residual_valid"]
        & expected["estimated.valid.body_position"] & old["source.observed.body_position"])
    return expected


def independent_head_fk(arrays, rest, shape_basis):
    """Independent float64 Rodrigues/FK of the head chain, without model.decode."""
    from scipy.spatial.transform import Rotation
    count = len(arrays["smplh.trans"])
    betas = np.asarray(arrays["smplh.betas"], np.float64)
    shaped = rest[None] + np.einsum("tb,jcb->tjc", betas, shape_basis)
    pose = np.concatenate((arrays["smplh.root_orient"][:, None],
                           arrays["smplh.pose_body"].reshape(count, 21, 3)), axis=1).astype(np.float64)
    finite = np.isfinite(pose).all((1, 2)) & np.isfinite(betas).all(1) & np.isfinite(arrays["smplh.trans"]).all(1)
    head = np.full((count, 3), np.nan, np.float64)
    ids = np.flatnonzero(finite)
    if len(ids):
        # Root -> spine1 -> spine2 -> spine3 -> neck -> head (SMPL-H index15).
        chain = (0, 3, 6, 9, 12, 15)
        local = Rotation.from_rotvec(pose[ids][:, chain].reshape(-1, 3)).as_matrix().reshape(len(ids), 6, 3, 3)
        position = shaped[ids, 0] + arrays["smplh.trans"][ids]
        rotation = local[:, 0]
        for offset, joint in enumerate(chain[1:], start=1):
            parent = PARENTS[joint]
            position = position + np.einsum("tij,tj->ti", rotation, shaped[ids, joint] - shaped[ids, parent])
            rotation = rotation @ local[:, offset]
        head[ids] = position
    return head.astype(np.float32).astype(np.float64), finite



def independent_quality(head, camera, observed, native, finite, camera_valid, head_valid, times, policy):
    """Independent threshold/edge formulas; never calls the producer scorer."""
    required = {"max_head_camera_distance_m", "warning_head_camera_distance_m",
        "max_head_observation_distance_m", "warning_head_observation_distance_m",
        "relative_jump_margin_m", "relative_speed_allowance_m_s", "max_time_gap_s"}
    if set(policy) != required or any(not np.isfinite(value) or value <= 0 for value in policy.values()):
        raise ValueError("Independent final-FK policy is incomplete or invalid")
    if (policy["warning_head_camera_distance_m"] >= policy["max_head_camera_distance_m"]
            or policy["warning_head_observation_distance_m"] >= policy["max_head_observation_distance_m"]):
        raise ValueError("Independent final-FK warning thresholds exceed rejection thresholds")
    count = len(head)
    camera_checked = camera_valid & np.isfinite(camera).all(1)
    head_checked = head_valid & np.isfinite(observed).all(1)
    head_finite = np.isfinite(head).all(1)
    finite = finite & head_finite
    camera_distance_checked, observation_distance_checked = camera_checked & head_finite, head_checked & head_finite
    relative = np.full((count, 3), np.nan, np.float64)
    relative[camera_distance_checked] = head[camera_distance_checked] - camera[camera_distance_checked]
    distance, observation_distance = np.full(count, np.nan), np.full(count, np.nan)
    distance[camera_distance_checked] = np.linalg.norm(relative[camera_distance_checked], axis=1)
    observation_distance[observation_distance_checked] = np.linalg.norm(
        head[observation_distance_checked] - observed[observation_distance_checked], axis=1)
    far = camera_distance_checked & (distance > policy["max_head_camera_distance_m"])
    observation_far = observation_distance_checked & (observation_distance > policy["max_head_observation_distance_m"])
    reasons = np.zeros(count, np.uint16)
    for mask, bit in ((~native, 1), (~finite, 2), (~camera_checked, 4), (~head_checked, 8), (far, 16), (observation_far, 32)):
        reasons[mask] |= bit
    warnings = np.zeros(count, np.uint16)
    warnings[camera_distance_checked & (distance > policy["warning_head_camera_distance_m"])] |= 1
    warnings[observation_distance_checked & (observation_distance > policy["warning_head_observation_distance_m"])] |= 2
    base = native & finite & camera_checked & head_checked
    edge_checked, jump_edge = np.zeros(count, bool), np.zeros(count, bool)
    jump = np.full(count, np.nan)
    if count > 1:
        dt = times[1:] - times[:-1]
        time_ok = np.isfinite(dt) & (dt > 0) & (dt <= policy["max_time_gap_s"])
        reasons[1:][~time_ok] |= 128
        edge_checked[1:] = base[1:] & base[:-1] & time_ok
        ids = np.flatnonzero(edge_checked)
        jump[ids] = np.linalg.norm(relative[ids] - relative[ids-1], axis=1)
        jump_edge[ids] = jump[ids] > (policy["relative_jump_margin_m"]
            + policy["relative_speed_allowance_m_s"] * (times[ids] - times[ids-1]))
    jump_frames = jump_edge.copy()
    if count > 1:
        jump_frames[:-1] |= jump_edge[1:]
    reasons[jump_frames] |= 64
    valid = base & ~far & ~observation_far & ~jump_frames
    transition_valid = edge_checked & ~jump_edge
    if count > 1:
        transition_valid[1:] &= valid[:-1] & valid[1:]
    return {"quality_valid": valid, "native_valid": native.copy(), "fit_finite": finite,
        "newly_rejected": native & ~valid, "camera_checked": camera_checked,
        "head_observation_checked": head_checked, "distance_checked": camera_distance_checked,
        "head_observation_distance_checked": observation_distance_checked,
        "transition_checked": edge_checked, "transition_valid": transition_valid,
        "reasons": reasons, "warnings": warnings, "head_camera_distance_m": distance,
        "head_observation_distance_m": observation_distance, "relative_jump_m": jump}


def _head_check_worker(task):
    import pyarrow as pa
    pa.set_cpu_count(1); pa.set_io_thread_count(1); torch.set_num_threads(1)
    old, receipt = task["record"], task["receipt"]
    columns = ("timestamp", "smplh.root_orient", "smplh.pose_body", "smplh.trans", "smplh.betas",
        "estimated.valid.body_position", "estimated.valid.body_rotation", "observation.state.camera_world",
        "observation.valid.camera_position", "observation.state.head_world", "observation.valid.head_position")
    collected = {key: [] for key in ("head", "native", "finite", "camera", "camera_valid", "observed", "head_valid", "times")}
    for batch in pq.ParquetFile(task["motion_path"]).iter_batches(batch_size=4096, columns=list(columns), use_threads=False):
        arrays = prior._arrays(batch)
        head, finite = independent_head_fk(arrays, task["rest"], task["shape_basis"])
        values = {"head": head, "finite": finite,
            "native": arrays["estimated.valid.body_position"].any(1) | arrays["estimated.valid.body_rotation"].any(1),
            "camera": arrays["observation.state.camera_world"].astype(np.float64),
            "camera_valid": arrays["observation.valid.camera_position"],
            "observed": arrays["observation.state.head_world"].astype(np.float64),
            "head_valid": arrays["observation.valid.head_position"], "times": arrays["timestamp"].astype(np.float64)}
        for key, value in values.items():
            collected[key].append(value)
    values = {key: np.concatenate(parts) for key, parts in collected.items()}
    if len(values["head"]) != old["length"]:
        raise ValueError("Independent final-FK input population differs")
    expected = independent_quality(policy=task["policy"], **values)
    _quality(Path(task["quality_root"]), receipt, old, Path(task["record_path"]))
    path = _owned(Path(task["quality_root"]), receipt["mask_path"])
    with np.load(path, allow_pickle=False) as saved:
        for key, wanted in expected.items():
            if key not in saved.files:
                raise ValueError("Independent final-FK sidecar lacks metric " + key)
            found = saved[key]
            if wanted.dtype.kind == "f":
                passed = found.dtype == np.float64 and found.shape == wanted.shape and np.allclose(
                    found, wanted, rtol=1e-6, atol=1e-6, equal_nan=True)
            else:
                passed = found.dtype == wanted.dtype and np.array_equal(found, wanted)
            if not passed:
                raise ValueError(f"Independent final-FK recomputation differs: episode {old['episode_index']}, {key}")
    if receipt.get("summary", {}).get("newly_rejected_frames") != int(expected["newly_rejected"].sum()):
        raise ValueError("Independent final-FK rejection summary differs")
    kept = expected["quality_valid"]
    return {"episode_index": old["episode_index"], "source": old["source"], "frames": old["length"],
        "native_valid_frames": int(expected["native_valid"].sum()), "retained_frames": int(kept.sum()),
        "newly_rejected_frames": int(expected["newly_rejected"].sum()),
        "maximum_retained_head_camera_distance_m": float(expected["head_camera_distance_m"][kept].max()) if kept.any() else None,
        "maximum_retained_head_observation_distance_m": float(expected["head_observation_distance_m"][kept].max()) if kept.any() else None}


def _independent_population(parent, qc, quality_root, policy, *, workers=8, progress=None):
    records = {record["episode_index"]: record for record in parent.records()}
    def task(index):
        return {"record": records[index], "receipt": qc[index], "motion_path": str(parent.data_path(index)),
            "record_path": str(parent._record_paths[index]), "quality_root": str(quality_root), "policy": policy,
            "rest": parent.model.rest, "shape_basis": parent.model.shape_basis}
    results = []
    if workers == 1:
        for index in sorted(qc):
            results.append(_head_check_worker(task(index)))
    else:
        import multiprocessing
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            indices, pending = iter(sorted(qc)), {}
            exhausted = False
            while pending or not exhausted:
                while len(pending) < workers and not exhausted:
                    try:
                        index = next(indices)
                    except StopIteration:
                        exhausted = True
                        break
                    pending[pool.submit(_head_check_worker, task(index))] = index
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    results.append(future.result())
                    if progress is not None and (len(results) % 100 == 0 or len(results) == len(qc)):
                        print(json.dumps({"independent_head_gate_episodes_checked": len(results), "total": len(qc)}), flush=True)
    results.sort(key=lambda item: item["episode_index"])
    controls = {source: next((item for item in results if item["source"] == source and item["retained_frames"] > 0 and item["newly_rejected_frames"] == 0), None)
                for source in ("egosuite", "egoexo4d_hm")}
    return {"scope": "all_eligible_archive", "episodes_checked": len(results), "episode_indices": sorted(qc),
        "frames_checked": sum(item["frames"] for item in results), "workers": workers, "max_inflight_tasks": workers,
        "method": "independent_scipy_Rodrigues_head_chain_FK_from_preserved_final_parameters",
        "columns_scope": "final_SMPLH_parameters_parent_estimated_masks_camera_head_observation_timestamp_only",
        "new_head_specific_check": True, "old_source_accuracy_requalification_repeated": False,
        "newly_rejected_frames": sum(item["newly_rejected_frames"] for item in results),
        "focused_episodes": [item for item in results if item["episode_index"] in {4503, 1093, 9084, 7498, 8421}],
        "normal_controls": controls, "episodes": results}

def _check_rewritten(parent, candidate, old, new, good, receipt, batch_rows=4096):
    index = old["episode_index"]
    original, revised = pq.ParquetFile(parent.data_path(index)), pq.ParquetFile(candidate.data_path(index))
    if (original.schema_arrow != revised.schema_arrow or original.metadata.num_rows != old["length"]
            or revised.metadata.num_rows != old["length"] or file_sha256(candidate.data_path(index)) != new["motion_sha256"]):
        raise ValueError("Final-FK rewritten Parquet schema/population/SHA differs")
    join_path = candidate.vision_cache._episodes[index]["artifacts"]["join.npz"][0]
    with np.load(join_path, allow_pickle=False) as saved:
        if set(saved.files) != set(JOIN_DTYPES):
            raise ValueError("Rewritten native visual join columns differ")
        join = {key: saved[key].copy() for key in JOIN_DTYPES}
    if any(join[key].shape != (old["length"],) or join[key].dtype != np.dtype(dtype) for key, dtype in JOIN_DTYPES.items()):
        raise ValueError("Rewritten native visual join shape/dtype differs")
    _assert_values(join["source.valid.dino"], join["observation.valid.dino"],
                   f"episode {index} native visual validity")
    from itertools import zip_longest
    offset, revised_cells, changed = 0, 0, set()
    feet_changed = False
    collections = {key: [] for key in ("source.feet_world", "source.feet_valid", "smplh.betas", "smplh.trans", "estimated.valid.body_position")}
    for before, after in zip_longest(original.iter_batches(batch_size=batch_rows, use_threads=False),
                                    revised.iter_batches(batch_size=batch_rows, use_threads=False)):
        if before is None or after is None or before.num_rows != after.num_rows:
            raise ValueError("Rewritten batch populations differ")
        a, b = prior._arrays(before), prior._arrays(after)
        count = before.num_rows
        expected = _expected_masks(a, good[offset:offset+count], old["source"])
        for key in a:
            if key in prior.FLOOR_COLUMNS:
                ground = new["metadata"]["ground"]
                value = (np.float32(ground["height_m"]) if key.endswith("floor_height") else
                    bool(ground["valid"] and (not key.endswith("floor_verified") or ground.get("verified", False))))
                wanted = np.full_like(a[key], value)
            else:
                wanted = expected.get(key, a[key])
            _assert_values(b[key], wanted, f"episode {index}, rows {offset}:{offset+count}, {key}")
            if not np.array_equal(a[key], b[key], equal_nan=True):
                changed.add(key)
                if key in expected:
                    revised_cells += int(np.count_nonzero(a[key] != b[key]))
        feet_changed |= not np.array_equal(a["source.feet_valid"], b["source.feet_valid"])
        for key in collections:
            collections[key].append(b[key].copy())
        # Native video PTS/error are recomputed by DINO extraction and replace
        # the older motion derivatives at read time. Compare only the shared
        # source-row identity and validity contract; both payload versions are
        # separately preserved exactly (motion columns above, join hardlink).
        for key in (*SOURCE_JOIN, "observation.valid.dino"):
            wanted = b.get(key)
            if wanted is None and key == "source.video_frame_index":
                wanted = np.arange(offset, offset+count, dtype=np.int64)
            if wanted is None:
                raise ValueError("Rewritten motion lacks a native visual join column")
            _assert_values(join[key][offset:offset+count], wanted, f"episode {index} visual join {key}")
        offset += count
    if offset != old["length"] or receipt.get("revised_mask_cells") != revised_cells:
        raise ValueError("Rewritten mask-cell/frame population differs")
    if feet_changed:
        if old["source"] != "egoexo4d_hm":
            raise ValueError("EgoSuite independent foot observations must remain unchanged")
        arrays = {key: np.concatenate(value) for key, value in collections.items()}
        root_z = (candidate.model.rest[0][None] + np.einsum("tb,cb->tc", arrays["smplh.betas"], candidate.model.shape_basis[0]) + arrays["smplh.trans"])[:, 2]
        ground = prior.reference_ground(arrays["source.feet_world"].reshape(-1, 2, 3), arrays["source.feet_valid"],
            root_z, arrays["estimated.valid.body_position"][:, 0], old["metadata"])
    else:
        ground = old["metadata"].get("ground")
    if new["metadata"].get("ground") != ground or new["metadata"]["fit_quality_revision"].get("ground_recomputed") is not feet_changed:
        raise ValueError("Independent final-FK gated-foot ground differs")
    return {"episode_index": index, "frames_compared": offset, "changed_columns": sorted(changed),
            "revised_mask_cells": revised_cells, "ground_recomputed": feet_changed,
            "native_visual_join_all_rows_compared": True}


def _quality(root, receipt, old, record_path):
    if (receipt.get("format") != AUDIT_FORMAT or receipt.get("episode_index") != old["episode_index"]
            or receipt.get("source_key") != old["source_key"] or receipt.get("split") != old["split"]
            or receipt.get("source") != old["source"]
            or receipt.get("record_sha256") != file_sha256(record_path)
            or receipt.get("motion_sha256") != old["motion_sha256"]):
        raise ValueError("Final-FK quality receipt source identity differs")
    if (old["metadata"].get("source_identity") is not None
            and receipt.get("source_identity") != old["metadata"]["source_identity"]):
        raise ValueError("Final-FK quality native source identity differs")
    path = _owned(root, receipt["mask_path"])
    if file_sha256(path) != receipt.get("mask_sha256"):
        raise ValueError("Final-FK quality sidecar SHA differs")
    with np.load(path, allow_pickle=False) as saved:
        fields = ("quality_valid", "native_valid", "fit_finite", "newly_rejected", "transition_valid", "reasons")
        if not set(fields) <= set(saved.files):
            raise ValueError("Final-FK quality sidecar masks are incomplete")
        values = {key: saved[key].copy() for key in fields}
    if any(value.shape != (old["length"],) for value in values.values()) or any(
            values[key].dtype != np.bool_ for key in fields if key != "reasons") or values["reasons"].dtype != np.uint16:
        raise ValueError("Final-FK quality sidecar shape/dtype differs")
    if hashlib.sha256(values["native_valid"].tobytes()).hexdigest() != receipt.get("native_valid_sha256"):
        raise ValueError("Final-FK native-valid source SHA differs")
    return values


def _check_unchanged_masks(parent, old, newly_rejected):
    selected = np.flatnonzero(newly_rejected)
    if not len(selected):
        return 0
    columns = {f"{prefix}.body_{component}" for component in ("position", "rotation")
        for prefix in ("estimated.valid", "source.observed", "source.interpolated", "estimated.completed")}
    columns.update({"quality.position_confidence", "quality.rotation_confidence", "quality.observation_residual_valid",
                    "source.feet_valid", "source.feet_interpolated"})
    if old["source"] == "egoexo4d_hm":
        columns.update({f"observation.{prefix}.hand_{side}_position"
                        for prefix in ("valid", "interpolated") for side in ("left", "right")})
    path = parent.data_path(old["episode_index"])
    schema = pq.ParquetFile(path).schema_arrow.names
    if old["source"] == "egoexo4d_hm" and "smplh.valid.hand_pose" in schema:
        columns.add("smplh.valid.hand_pose")
    values = parent.reader.read(path, selected, sorted(columns))
    for key, expected in _expected_masks(values, np.zeros(len(selected), bool), old["source"]).items():
        _assert_values(values[key], expected, f"producer-declared unchanged episode {old['episode_index']} rejected masks {key}")
    return len(selected)


def _check_structure(parent, candidate, plan, revision, quality, revision_path, quality_path):
    root = candidate.root
    build = revision.get("build_binding", {})
    build_sha, quality_sha = digest(build), file_sha256(quality_path)
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
        raise ValueError("Final-FK revision candidate/parent/quality binding differs")
    old_records, new_records = parent.records(), candidate.records()
    required = {record["episode_index"] for record in old_records if eligible(record)}
    qc_items = quality.get("episodes", [])
    qc = {item["episode_index"]: item for item in qc_items}
    population = quality.get("source_population", {})
    if (quality.get("format") != AUDIT_FORMAT or quality.get("state") != "complete"
            or quality.get("diagnostic") is not False or quality.get("production_modified") is not False
            or not quality.get("finished_at") or Path(quality.get("archive_root", "")).resolve() != parent.root
            or population.get("scope") != "all_eligible_archive" or population.get("total") != len(old_records)
            or population.get("episodes", population.get("total")) != population.get("total")
            or population.get("eligible") != len(required) or len(population.get("excluded", [])) != len(old_records)-len(required)
            or {item.get("episode_index") for item in population.get("excluded", [])} != {record["episode_index"] for record in old_records}-required
            or len(qc) != len(qc_items) or set(qc) != required
            or quality.get("episodes_completed") != len(required) or quality.get("episodes_planned") != len(required)
            or revision.get("quality_episodes") != sorted(required)
            or revision.get("required_quality_episodes") != sorted(required)):
        raise ValueError("Final-FK quality audit does not cover the complete eligible dual-source population")
    if (len(old_records) != len(new_records) or parent.frames != candidate.frames
            or parent.split_fingerprint != candidate.split_fingerprint or parent.model_sha256 != candidate.model_sha256
            or revision.get("model_sha256") != candidate.model_sha256 or revision.get("parent_model_sha256") != parent.model_sha256
            or revision.get("split_fingerprint") != candidate.split_fingerprint
            or revision.get("parent_split_fingerprint") != parent.split_fingerprint):
        raise ValueError("Final-FK revision changed original population/split/model")
    a, b = deepcopy(parent.info), deepcopy(candidate.info)
    declaration = b.pop("quality_revision", None)
    parent_declaration = a.pop("quality_revision", None)
    if a != b or not isinstance(declaration, dict) or declaration.get("format") != REVISION_FORMAT:
        raise ValueError("Final-FK revision changed original schema metadata")
    if parent_declaration is not None and declaration.get("parent_revision") != parent_declaration:
        raise ValueError("Final-FK revision lost its original parent revision declaration")
    preserved = {relative: file_sha256(parent.root / relative)
        for relative in ("meta/ontology.json", "meta/summary.json", ".pipeline/config.json") if (parent.root / relative).is_file()}
    if build.get("preserved_metadata_sha256") != preserved:
        raise ValueError("Final-FK preserved ontology/inventory/configuration differs")
    for relative in preserved:
        destination = ".pipeline/parent-publication/config.json" if relative == ".pipeline/config.json" else relative
        _same_file(parent.root / relative, _owned(root, destination), "original metadata")
    config = json.loads(_owned(root, ".pipeline/config.json").read_text())
    if (config.get("format") != REVISION_FORMAT or config.get("standalone") is not True
            or config.get("requires_old_dense_archive") is not False or config.get("requires_parent_archive_at_training") is not False):
        raise ValueError("Final-FK standalone revision must not require obsolete archives")
    _same_file(parent.model_path, candidate.model_path, "SMPL-H model")
    for path in (parent.root / "meta").glob("splits*.json"):
        _same_file(path, root / "meta" / path.name, "source split")
    publication = _owned(root, ".pipeline/parent-publication/dataset-state.json")
    _same_file(parent.state_path, publication, "parent publication")
    if file_sha256(publication) != build.get("parent_publication_sha256"):
        raise ValueError("Final-FK inherited parent publication differs")
    parent_acceptance = parent.state.get("acceptance_report")
    if parent_acceptance:
        path = Path(parent_acceptance)
        path = path if path.is_absolute() else parent.root / path
        if path.is_file() and build.get("parent_acceptance_sha256") != file_sha256(path):
            raise ValueError("Final-FK inherited parent independent acceptance differs")
    _same_file(Path(plan["quality_report"]), quality_path, "quality audit")
    old_visual, new_visual = parent.vision_cache.manifest, candidate.vision_cache.manifest
    expected = deepcopy(old_visual)
    expected.update(source_root=str(root), output_root=str(candidate.vision_cache.root),
                    source_binding=new_visual.get("source_binding"), quality_revision_reuse=new_visual.get("quality_revision_reuse"))
    if (new_visual != expected or new_visual.get("quality_revision_reuse", {}).get("parent_profile_root") != str(parent.vision_cache.root)):
        raise ValueError("Final-FK revision changed frozen DINO/image-processing declaration")
    rewritten = revision.get("motion_rewritten_episodes", [])
    if len(set(rewritten)) != len(rewritten) or not set(rewritten) <= required or revision.get("changed_episode_indices") != rewritten:
        raise ValueError("Final-FK rewritten episode scope differs")
    return old_records, new_records, qc, rewritten, quality_sha, build_sha, preserved


def audit_revision(parent_root, revision_root, *, parent_vision_root=None, workers=8, progress=None):
    root = Path(revision_root).resolve(strict=True)
    if not 1 <= workers <= 8:
        raise ValueError("Independent final-FK workers must be between1 and8")
    with locked(root / ".revision-build.lock"):
        return _audit_revision(parent_root, root, parent_vision_root=parent_vision_root, workers=workers, progress=progress)


def _audit_revision(parent_root, revision_root, *, parent_vision_root=None, workers=8, progress=None):
    parent, candidate, plan = prior._contexts(parent_root, revision_root, parent_vision_root)
    root = candidate.root
    revision_path, quality_path = _owned(root, ".pipeline/fit-quality-revision.json"), _owned(root, ".pipeline/quality/report.json")
    revision, quality = json.loads(revision_path.read_text()), json.loads(quality_path.read_text())
    stamps = prior._quality_stamps(root)
    old_records, new_records, qc, rewritten, quality_sha, build_sha, preserved = _check_structure(
        parent, candidate, plan, revision, quality, revision_path, quality_path)
    # This new physical check deliberately covers every newly screened fit.
    physical = _independent_population(parent, qc, root / ".pipeline/quality", quality["policy"], workers=workers, progress=progress)
    sources, splits, comparisons = {}, {name: {"episodes": 0, "frames": 0} for name in ("train", "val", "test")}, []
    inherited, inherited_masks = 0, 0
    for old, new in zip(old_records, new_records):
        index = old["episode_index"]
        counts = sources.setdefault(old["source"], {"episodes": 0, "frames": 0})
        counts["episodes"] += 1; counts["frames"] += old["length"]
        splits[old["split"]]["episodes"] += 1; splits[old["split"]]["frames"] += old["length"]
        receipt = json.loads(candidate.owned_path(f"episodes/{index:08d}/receipt.json").read_text())
        entry = receipt.get("quality_revision", {})
        if (entry.get("build_binding_sha256") != build_sha or entry.get("parent_record_sha256") != digest(old)
                or entry.get("parent_receipt_sha256") != file_sha256(parent.owned_path(f"episodes/{index:08d}/receipt.json"))
                or entry.get("motion_rewritten") is not (index in rewritten)):
            raise ValueError("Final-FK episode receipt parent/scope differs")
        expected = deepcopy(old)
        good = None
        if index in qc:
            quality_receipt = qc[index]
            if (json.loads(_owned(root, f".pipeline/quality/episode_{index:08d}.json").read_text()) != quality_receipt
                    or quality_receipt.get("policy") != quality["policy"]
                    or entry.get("quality_mask_sha256") != quality_receipt["mask_sha256"]):
                raise ValueError("Final-FK episode quality receipt/policy differs")
            values = _quality(root / ".pipeline/quality", quality_receipt, old, parent._record_paths[index])
            good = values["quality_valid"]
            if index in rewritten:
                metadata = new["metadata"].get("fit_quality_revision", {})
                if (metadata.get("format") != REVISION_FORMAT or metadata.get("quality_mask_sha256") != quality_receipt["mask_sha256"]
                        or metadata.get("native_valid_sha256") != quality_receipt["native_valid_sha256"]
                        or metadata.get("quality_receipt_sha256") != digest(quality_receipt)
                        or metadata.get("quality_report_sha256") != quality_sha or metadata.get("policy") != quality["policy"]
                        or metadata.get("parent_ground") != old["metadata"].get("ground")):
                    raise ValueError("Final-FK rewritten metadata provenance differs")
                expected["metadata"]["fit_quality_revision"] = metadata
                expected["metadata"]["ground"] = new["metadata"]["ground"]
                expected["motion_sha256"] = new["motion_sha256"]
        elif entry.get("quality_mask_sha256") is not None:
            raise ValueError("Quarantined episode unexpectedly gained quality evidence")
        if new != expected:
            raise ValueError("Final-FK revision changed unauthorized record metadata/identity")
        if index in rewritten:
            comparisons.append(_check_rewritten(parent, candidate, old, new, good, entry))
        else:
            _same_file(parent.data_path(index), candidate.data_path(index), "unchanged motion")
            if new != old or entry.get("revised_mask_cells") != 0:
                raise ValueError("Unchanged motion must preserve the exact accepted record")
            if good is not None:
                inherited_masks += _check_unchanged_masks(parent, old, values["newly_rejected"])
            inherited += 1
        for name in ("features.npy", "join.npz"):
            _same_file(parent.vision_cache._episodes[index]["artifacts"][name][0],
                       candidate.vision_cache._episodes[index]["artifacts"][name][0], "DINO " + name)
        if entry.get("legacy_vision_retired") is not (index in parent._retired_vision):
            raise ValueError("Final-FK revision changed legacy vision retirement")
        if index not in parent._retired_vision:
            _same_file(parent.owned_path(old["vision_path"]), candidate.owned_path(new["vision_path"]), "legacy vision")
        visual = json.loads(candidate.vision_cache._episodes[index]["receipt_path"].read_text()).get("quality_revision_reuse", {})
        if (visual.get("build_binding_sha256") != build_sha
                or visual.get("parent_receipt_sha256") != file_sha256(parent.vision_cache._episodes[index]["receipt_path"])
                or visual.get("parent_manifest_sha256") != parent.vision_cache.metadata["manifest_sha256"]
                or visual.get("join_all_native_rows_checked") is not (index in rewritten)):
            raise ValueError("Final-FK DINO reuse receipt binding differs")
        if progress is not None:
            progress(index, len(new_records), len(comparisons))
    population = {"episodes": len(new_records), "frames": candidate.frames, "sources": sources}
    if revision.get("population") != population or candidate.state.get("sources") != sources:
        raise ValueError("Final-FK source/frame population differs")
    if "meta/summary.json" in preserved:
        summary = json.loads((root / "meta/summary.json").read_text())
        if summary.get("episodes") != len(new_records) or summary.get("frames") != candidate.frames or summary.get("sources") != sources or summary.get("splits") != splits:
            raise ValueError("Final-FK inherited inventory differs")
    compatibility = prior._motionexpert_compatibility(candidate, rewritten)
    parent.assert_current(); candidate.assert_current()
    if prior._quality_stamps(root) != stamps or file_sha256(quality_path) != quality_sha:
        raise ValueError("Final-FK evidence changed during independent acceptance")
    return {"format": INDEPENDENT_AUDIT_FORMAT, "scope": "full", "passed": True, "failures": [],
        "created_at": prior._now(), "binding": candidate.binding, "population": population,
        "parent_archive_binding": parent.binding, "parent_archive_fingerprint": parent.archive_fingerprint,
        "validator_sha256": file_sha256(__file__), "source_split_population": splits,
        "quality_audit_sha256": quality_sha, "revision_manifest_sha256": file_sha256(revision_path),
        "artifact_snapshot_sha256": prior._snapshot(parent, candidate, revision_path, quality_path),
        "changed_episode_indices": rewritten, "changed_motion_comparisons": comparisons,
        "inherited_unchanged_motion_episodes": inherited, "inherited_new_rejected_frame_masks_checked": inherited_masks,
        "unchanged_motion_full_numeric_QA_repeated": False, "previous_physical_acceptance_repeated": False,
        "new_head_specific_physical_gate": physical, "dino_payload_rehashed": False,
        "dino_join_rows_compared_only_for_rewritten_motion": True, "latest_motionexpert": compatibility,
        "retained_fits_certified_ground_truth": False,
        **{name: True for name in FLAGS}}


def publish_revision(parent_root, revision_root, independent_report, *, parent_vision_root=None):
    root = Path(revision_root).resolve(strict=True)
    with locked(root / ".revision-build.lock"):
        parent, candidate, plan = prior._contexts(parent_root, root, parent_vision_root)
        path = Path(independent_report).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Final-FK independent report must be owned by candidate")
        if candidate.state.get("state") != "candidate" or candidate.state.get("independent_acceptance_pending") is not True:
            raise ValueError("Only an unpublished final-FK candidate may be published once")
        report = json.loads(path.read_text())
        revision_path, quality_path = _owned(root, ".pipeline/fit-quality-revision.json"), _owned(root, ".pipeline/quality/report.json")
        if (report.get("format") != INDEPENDENT_AUDIT_FORMAT or report.get("scope") != "full"
                or report.get("passed") is not True or report.get("failures") != []
                or report.get("binding") != candidate.binding or report.get("validator_sha256") != file_sha256(__file__)
                or report.get("artifact_snapshot_sha256") != prior._snapshot(parent, candidate, revision_path, quality_path)
                or any(report.get(name) is not True for name in FLAGS)):
            raise ValueError("Final-FK independent revision audit is missing, partial, or stale")
        proof = lambda value: {"path": str(value.relative_to(root)), "sha256": file_sha256(value)}
        proofs = {"revision_manifest": proof(revision_path), "quality_audit": proof(quality_path),
            "independent_revision_audit": proof(path),
            "parent_publication": proof(_owned(root, ".pipeline/parent-publication/dataset-state.json"))}
        if plan["binding"].get("parent_acceptance_sha256"):
            proofs["parent_acceptance"] = proof(_owned(root, ".pipeline/parent-publication/acceptance-report.json"))
        top = {"format": ACCEPTANCE_FORMAT, "scope": "full", "passed": True, "failures": [],
            "population": report["population"], "binding": candidate.binding, "proofs": proofs,
            "created_at": prior._now(), "previous_physical_acceptance_repeated": False,
            **{name: True for name in FLAGS}}
        top_path = root / ".pipeline/acceptance-report.json"
        check_fit_revision_acceptance(candidate, top, top_path)
        parent.assert_current(); candidate.assert_current()
        _write_new(top_path, top)
        state = dict(candidate.state, state="accepted", independent_acceptance_pending=False,
            acceptance_report=str(top_path.relative_to(root)), acceptance_report_sha256=file_sha256(top_path),
            accepted_at=prior._now(), formal_training_ready=False)
        atomic_json(candidate.state_path, state)
        return top


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--parent-vision-root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    import pyarrow as pa
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    root = args.archive_root.resolve(strict=True)
    path = (args.report or root / ".pipeline/independent-fit-revision-audit.json").resolve()
    if not path.is_relative_to(root):
        parser.error("Independent fit report must be owned by the candidate")
    if path.exists():
        if not args.publish:
            raise FileExistsError("Final-FK independent report already exists; no automatic repeat")
    else:
        def progress(index, total, rewritten):
            if index == 0 or (index+1) % 100 == 0 or index+1 == total:
                print(json.dumps({"episode_checked": index+1, "total": total, "rewritten_compared": rewritten}), flush=True)
        report = audit_revision(args.parent_root, root, parent_vision_root=args.parent_vision_root,
                                workers=args.workers, progress=progress)
        _write_new(path, report)
    if args.publish:
        publish_revision(args.parent_root, root, path, parent_vision_root=args.parent_vision_root)
    print(json.dumps({"passed": True, "report": str(path), "published": args.publish, "training_started": False}))


if __name__ == "__main__":
    main()
