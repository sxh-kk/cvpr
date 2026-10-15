"""Candidate revision for final-FK quality, preserving independent observations.

This module shares the immutable writer, joins and original ground policy with
the preceding HM revision. It never edits its parent, publishes an acceptance,
or launches training. A separate checker must accept the finished candidate.
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

from . import revise_hm_archive as shared
from .dataset.smplh_archive import SMPLHParameterArchive, PARAMETER_SCHEMA_VERSION, canonical_json, file_sha256
from .dataset.vision_cache import storage_digest
from .dataset.window_cache import atomic_json, digest


AUDIT_FORMAT = "smplh-final-fk-quality-audit-v1"
REVISION_FORMAT = "smplh-fit-quality-revision-v1"
MANIFEST = ".pipeline/fit-quality-revision.json"
PROFILE = "vision_profiles/psi0_256x480_v1_30hz"
ALGORITHM_VERSION = 1


def apply_fit_quality_masks(arrays, good, source):
    """Invalidate estimates; source-aware masks retain independent capture data."""
    good = np.asarray(good)
    if good.dtype != np.bool_ or good.shape != (len(arrays["timestamp"]),):
        raise ValueError("Final-FK quality mask must contain boolean T")
    if source == "egoexo4d_hm":
        return shared.apply_quality_masks(arrays, good)
    if source != "egosuite":
        raise ValueError("Unknown final-FK source semantics")
    for foot_key, body_key in (("source.feet_valid", "source.observed.body_position"),
                               ("source.feet_interpolated", "source.interpolated.body_position")):
        if not np.array_equal(arrays[foot_key], arrays[body_key][:, [10, 11]]):
            raise ValueError("Original source foot masks differ from body observations")
    # Capture availability is independent of a rejected fitted body. In
    # particular, observed=True with estimated=False is a supported archive
    # state and does not imply that the observation was consumed as a target.
    result = dict(arrays)
    for component in ("position", "rotation"):
        available = arrays[f"estimated.valid.body_{component}"] & good[:, None]
        observed = arrays[f"source.observed.body_{component}"]
        result[f"estimated.valid.body_{component}"] = available
        result[f"estimated.completed.body_{component}"] = available & ~observed
        result[f"quality.{component}_confidence"] = np.where(
            available, arrays[f"quality.{component}_confidence"], 0)
    result["quality.observation_residual_valid"] = (arrays["quality.observation_residual_valid"]
        & result["estimated.valid.body_position"] & arrays["source.observed.body_position"])
    return result


def _load_quality(directory, receipt, record, record_bytes):
    if (receipt.get("format") != AUDIT_FORMAT or receipt.get("episode_index") != record["episode_index"]
            or receipt.get("source") != record["source"]
            or receipt.get("source_key") != record["source_key"] or receipt.get("split") != record["split"]
            or receipt.get("record_sha256") != hashlib.sha256(record_bytes).hexdigest()
            or receipt.get("motion_sha256") != record["motion_sha256"]):
        raise ValueError("Final-FK quality receipt differs from the accepted parent record")
    path = shared._owned(directory, receipt["mask_path"])
    if file_sha256(path) != receipt.get("mask_sha256"):
        raise ValueError("Final-FK quality mask SHA differs")
    with np.load(path, allow_pickle=False) as saved:
        required = ("quality_valid", "native_valid", "fit_finite", "newly_rejected", "transition_valid", "reasons")
        if not set(required) <= set(saved.files):
            raise ValueError("Final-FK sidecar lacks required quality evidence")
        quality = {name: saved[name].copy() for name in required}
    good = quality["quality_valid"]
    if any(value.shape != (record["length"],) for value in quality.values()) or any(
            quality[name].dtype != np.bool_ for name in required[:-1]) or quality["reasons"].dtype != np.uint16:
        raise ValueError("Final-FK quality mask shape/dtype differs")
    if hashlib.sha256(quality["native_valid"].tobytes()).hexdigest() != receipt.get("native_valid_sha256"):
        raise ValueError("Final-FK input-availability SHA differs")
    if np.any(good & ~(quality["native_valid"] & quality["fit_finite"])):
        raise ValueError("Final-FK quality cannot upgrade unavailable fits")
    rejected = quality["native_valid"] & ~good
    if (not np.array_equal(rejected, quality["newly_rejected"])
            or receipt.get("summary", {}).get("newly_rejected_frames") != int(rejected.sum())):
        raise ValueError("Final-FK newly rejected mask/count differs")
    return quality


def _write_motion(path, template, arrays):
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".motion.revision.tmp")
    if path.exists():
        raise ValueError("Uncommitted motion exists; never overwrite frozen artifacts")
    pq.write_table(shared._table_from_arrays(template, arrays), temporary, compression="zstd", row_group_size=512)
    saved = shared._columns(pq.read_table(temporary, use_threads=False))
    if set(saved) != set(arrays) or any(
            saved[name].dtype != value.dtype or saved[name].shape != value.shape
            or not np.array_equal(saved[name], value) for name, value in arrays.items()):
        raise ValueError("Final-FK revised Parquet failed exact readback")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _revise_episode(task):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    source, target = Path(task["source_root"]), Path(task["output_root"])
    original, quality_receipt = task["record"], task["quality_receipt"]
    index = original["episode_index"]
    resumed = shared._resume_episode(target, original, task["build_binding_sha256"], quality_receipt,
                                     source_root=source, source_vision_root=task["source_vision_root"])
    if resumed:
        return {"episode_index": index, "record": resumed[0], "revision": resumed[1], "resumed": True}
    shared._prepare_episode_output(task)
    record_path = shared._owned(source, f".pipeline/episodes/{index:08d}.json")
    original_bytes = record_path.read_bytes()
    if json.loads(original_bytes) != original:
        raise ValueError("Original final-FK episode changed after planning")
    receipt_path = shared._owned(source, f"episodes/{index:08d}/receipt.json")
    old_receipt = json.loads(receipt_path.read_text())
    motion = shared._owned(source, original["data_path"])
    if not shared._matches_stamp(motion, old_receipt["output_identities"]["motion"]):
        raise ValueError("Final-FK parent motion identity changed")
    record, arrays, rewritten, cells, ground_changed = deepcopy(original), None, False, 0, False
    destination = target / original["data_path"]
    if quality_receipt:
        quality = _load_quality(task["quality_dir"], quality_receipt, original, original_bytes)
        good = quality["quality_valid"]
        if quality["newly_rejected"].any():
            table = pq.read_table(motion, use_threads=False)
            before = shared._columns(table)
            arrays = apply_fit_quality_masks(before, good, original["source"])
            names = [name for name in arrays if not np.array_equal(arrays[name], before[name])]
            cells = sum(int(np.count_nonzero(arrays[name] != before[name])) for name in names)
            if names:
                ground_changed = not np.array_equal(arrays["source.feet_valid"], before["source.feet_valid"])
                if ground_changed:
                    if original["source"] != "egoexo4d_hm":
                        raise ValueError("Independent EgoSuite feet must remain unchanged")
                    ground = shared.estimate_revision_ground(arrays, record["metadata"], root_rest=task["root_rest"],
                                                            root_shape_basis=task["root_shape_basis"])
                    record["metadata"]["ground"] = ground
                    arrays["observation.environment.floor_height"] = np.full_like(
                        before["observation.environment.floor_height"], np.float32(ground["height_m"]))
                    arrays["observation.valid.floor"] = np.full_like(before["observation.valid.floor"], bool(ground["valid"]))
                    arrays["observation.valid.floor_verified"] = np.full_like(
                        before["observation.valid.floor_verified"], bool(ground["valid"] and ground.get("verified", False)))
                _write_motion(destination, table, arrays)
                rewritten = True
                record["motion_sha256"] = file_sha256(destination)
                # Unchanged episodes keep their entire record exactly equal to
                # the parent, enabling strictly proven raw-window cache reuse.
                record["metadata"]["fit_quality_revision"] = {
                    "format": REVISION_FORMAT, "quality_mask_sha256": quality_receipt["mask_sha256"],
                    "native_valid_sha256": quality_receipt["native_valid_sha256"],
                    "quality_receipt_sha256": digest(quality_receipt),
                    "quality_report_sha256": task["quality_report_sha256"], "policy": quality_receipt["policy"],
                    "ground_recomputed": ground_changed, "parent_ground": deepcopy(original["metadata"].get("ground")),
                    "capture_observations_preserved": original["source"] == "egosuite",
                    "mask_policy": ("estimated_body_only_independent_capture_observations_preserved"
                                    if original["source"] == "egosuite" else "fit_linked_body_hands_feet_gated")}
    if not rewritten:
        shared._hardlink(motion, destination)
    retired = bool(task["legacy_vision_retired"])
    if not retired:
        shared._hardlink(shared._owned(source, original["vision_path"]), target / original["vision_path"])
    visual, old_visual_receipt_path = shared._verify_visual_join(
        source, original, task["source_vision_root"], arrays=arrays, compare_rows=rewritten)
    prefix = f"episodes/{index:08d}"
    for name in ("features.npy", "join.npz"):
        shared._hardlink(shared._owned(task["source_vision_root"], f"{prefix}/{name}"), target / PROFILE / prefix / name)
    visual = deepcopy(visual)
    visual["source_record_sha256"] = storage_digest(record)
    visual["quality_revision_reuse"] = {
        "parent_receipt_sha256": file_sha256(old_visual_receipt_path),
        "parent_manifest_sha256": task["source_vision_manifest_sha256"],
        "build_binding_sha256": task["build_binding_sha256"], "join_all_native_rows_checked": rewritten,
        "join_evidence": ("changed_motion_all_native_rows_compared" if rewritten else "inherited_immutable_motion_and_visual"),
        "payload_reuse": "same_device_inode_hardlink", "features_content_rehashed": False}
    atomic_json(target / PROFILE / prefix / "receipt.json", visual)
    revision = {"build_binding_sha256": task["build_binding_sha256"], "parent_record_sha256": digest(original),
                "parent_receipt_sha256": file_sha256(receipt_path),
                "quality_mask_sha256": quality_receipt["mask_sha256"] if quality_receipt else None,
                "motion_rewritten": rewritten, "revised_mask_cells": cells, "ground_recomputed": ground_changed,
                "legacy_vision_retired": retired, "image_join_values_preserved": True}
    receipt = deepcopy(old_receipt)
    receipt.update(record=record, quality_revision=revision, completed_at=datetime.now(timezone.utc).isoformat(),
                   conversion_binding=digest({"parent": old_receipt["conversion_binding"], "revision": task["build_binding_sha256"]}))
    receipt["motion"]["sha256"] = record["motion_sha256"]
    receipt["output_identities"]["motion"] = shared._stamp(destination)
    if not retired:
        receipt["output_identities"]["vision"] = shared._stamp(target / original["vision_path"])
    if (record_path.read_bytes() != original_bytes
            or not shared._matches_stamp(motion, old_receipt["output_identities"]["motion"])):
        raise ValueError("Original final-FK source changed while writing")
    atomic_json(target / prefix / "receipt.json", receipt)
    atomic_json(target / f".pipeline/episodes/{index:08d}.json", record)
    return {"episode_index": index, "record": record, "revision": revision, "resumed": False}


def _parallel_tasks(pool, tasks, *, workers, worker_control):
    pending, cursor = {}, 0
    while pending or cursor < len(tasks):
        active = shared._active_workers(worker_control, workers)
        while cursor < len(tasks) and len(pending) < active:
            future = pool.submit(_revise_episode, tasks[cursor])
            pending[future] = tasks[cursor]["record"]["episode_index"]
            cursor += 1
        done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
        for future in done:
            pending.pop(future)
            yield future.result()


def build_revision(source_root, quality_report, output_root, *, source_vision_root=None,
                   workers=2, worker_control=None, resume=False, diagnostic=False):
    """Build a new candidate and preserve its immutable parent and visual bytes."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    source, output = Path(source_root).resolve(strict=True), Path(output_root).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Final-FK output must be separate from its immutable source")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Final-FK workers must be an integer from1 to8")
    quality_path = Path(quality_report).resolve(strict=True)
    quality_report_value = json.loads(quality_path.read_text())
    if (quality_report_value.get("format") != AUDIT_FORMAT or quality_report_value.get("state") != "complete"
            or Path(quality_report_value.get("archive_root", "")).resolve() != source
            or quality_report_value.get("production_modified") is not False):
        raise ValueError("Final-FK candidate requires a complete read-only source audit")
    population_scope = quality_report_value.get("source_population", {}).get("scope")
    if not diagnostic and (quality_report_value.get("diagnostic") is not False or population_scope != "all_eligible_archive"):
        raise ValueError("Formal final-FK revision requires every eligible source episode")
    if output.is_relative_to(quality_path.parent) or quality_path.parent.is_relative_to(output):
        raise ValueError("Final-FK output and independent quality directory must be separate")
    vision = Path(source_vision_root or source / PROFILE).resolve(strict=True)
    if output.is_relative_to(vision) or vision.is_relative_to(output):
        raise ValueError("Final-FK output overlaps its frozen DINO profile")
    if output.exists() and not resume and any(output.iterdir()):
        raise ValueError("Use an empty final-FK revision directory or --resume its frozen plan")
    output.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != output.stat().st_dev or vision.stat().st_dev != output.stat().st_dev:
        raise ValueError("Final-FK zero-copy reuse requires the same filesystem")
    with (output / ".revision-build.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        archive = SMPLHParameterArchive(source, vision_profile_root=vision)
        records = archive.records()
        entries = quality_report_value.get("episodes")
        if (not isinstance(entries, list) or quality_report_value.get("episodes_completed") != len(entries)
                or quality_report_value.get("episodes_planned") != len(entries)):
            raise ValueError("Final-FK audit receipt population differs")
        quality = {entry["episode_index"]: entry for entry in entries}
        required = {r["episode_index"] for r in records if shared._source_eligible(r)}
        population = quality_report_value["source_population"]
        excluded = population.get("excluded")
        if (population.get("total") != len(records)
                or population.get("episodes", len(records)) != len(records)
                or population.get("eligible") != len(required)
                or not isinstance(excluded, list)
                or len({entry.get("episode_index") for entry in excluded}) != len(excluded)
                or {entry.get("episode_index") for entry in excluded} != set(range(len(records))) - required
                or any(entry.get("source_key") != records[entry["episode_index"]]["source_key"] for entry in excluded)):
            raise ValueError("Final-FK audit source population differs from the accepted parent")
        if len(quality) != len(entries) or not set(quality) <= required or (not diagnostic and set(quality) != required):
            raise ValueError("Final-FK audit must exactly cover eligible episodes without changing quarantines")
        for index, receipt in quality.items():
            if receipt.get("policy") != quality_report_value.get("policy"):
                raise ValueError("Final-FK episode policy differs from the bound audit")
            _load_quality(quality_path.parent, receipt, records[index],
                          (source / f".pipeline/episodes/{index:08d}.json").read_bytes())
        binding = {"format": REVISION_FORMAT, "algorithm_version": ALGORITHM_VERSION,
                   "builder_sha256": file_sha256(__file__), "shared_io_sha256": file_sha256(shared.__file__),
                   "parent_archive_binding": archive.binding, "parent_archive_fingerprint": archive.archive_fingerprint,
                   "quality_report_sha256": file_sha256(quality_path), "policy": quality_report_value["policy"],
                   "source_vision_manifest_sha256": archive.vision_cache.metadata["manifest_sha256"],
                   "parent_publication_sha256": file_sha256(archive.state_path),
                   "preserved_metadata_sha256": {relative: file_sha256(source / relative) for relative in (
                       "meta/ontology.json", "meta/summary.json", ".pipeline/config.json") if (source / relative).is_file()},
                   "scope": "selected_episodes" if diagnostic else "full"}
        parent_report = archive.state.get("acceptance_report")
        if parent_report:
            parent_report = Path(parent_report)
            if not parent_report.is_absolute():
                parent_report = source / parent_report
            if parent_report.is_file():
                binding["parent_acceptance_sha256"] = file_sha256(parent_report)
        plan = {"format": REVISION_FORMAT, "binding": binding, "binding_sha256": digest(binding),
                "source_root": str(source), "output_root": str(output), "source_vision_root": str(vision),
                "quality_report": str(quality_path)}
        plan_path, state_path = output / ".pipeline/revision-plan.json", output / ".pipeline/dataset-state.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("Final-FK resume plan differs from the frozen source, policy or writer")
        if state_path.exists() and json.loads(state_path.read_text()).get("state") == "accepted":
            raise ValueError("Accepted final-FK archive is immutable")
        atomic_json(plan_path, plan)
        atomic_json(state_path, {"schema_version": PARAMETER_SCHEMA_VERSION, "state": "building",
                                "independent_acceptance_pending": True, "episodes": len(records), "frames": archive.frames})
        for descriptor in (archive.info["model"], archive.info.get("hand_basis")):
            if descriptor:
                shared._hardlink(archive.owned_path(descriptor["path"]), output / descriptor["path"])
        for path in sorted((source / "meta").glob("splits*.json")):
            shared._hardlink(path, output / "meta" / path.name)
        for relative in binding["preserved_metadata_sha256"]:
            destination = output / (".pipeline/parent-publication/config.json" if relative == ".pipeline/config.json" else relative)
            shared._hardlink(source / relative, destination)
        shared._hardlink(archive.state_path, output / ".pipeline/parent-publication/dataset-state.json")
        if parent_report and parent_report.is_file():
            shared._hardlink(parent_report, output / ".pipeline/parent-publication/acceptance-report.json")
        atomic_json(output / ".pipeline/config.json", {"format": REVISION_FORMAT, "schema_version": PARAMETER_SCHEMA_VERSION,
            "model_sha256": archive.model_sha256, "standalone": True, "requires_old_dense_archive": False,
            "requires_parent_archive_at_training": False, "revision_plan": ".pipeline/revision-plan.json",
            "quality_report": ".pipeline/quality/report.json", "parent_configuration": (
                ".pipeline/parent-publication/config.json" if ".pipeline/config.json" in binding["preserved_metadata_sha256"] else None)})
        shared._hardlink(quality_path, output / ".pipeline/quality/report.json")
        for index, receipt in quality.items():
            shared._hardlink(shared._owned(quality_path.parent, receipt["mask_path"]), output / ".pipeline/quality" / receipt["mask_path"])
            receipt_path = quality_path.parent / f"episode_{index:08d}.json"
            if not receipt_path.is_file() or json.loads(receipt_path.read_text()) != receipt:
                raise ValueError("Final-FK individual quality receipt differs from its complete audit")
            shared._hardlink(receipt_path, output / ".pipeline/quality" / receipt_path.name)
        info = deepcopy(archive.info)
        info["quality_revision"] = {"format": REVISION_FORMAT, "parent_revision": deepcopy(archive.info.get("quality_revision")),
            "parent_archive_fingerprint": archive.archive_fingerprint, "quality_report_sha256": binding["quality_report_sha256"],
            "build_binding_sha256": plan["binding_sha256"], "policy": quality_report_value["policy"],
            "parameter_values_preserved": True, "image_join_values_preserved": True,
            "observation_policy": "EgoSuite_capture_preserved_HM_fit_linked_gated",
            "ground_policy": "reestimate_only_changed_fit_linked_feet_then_observed_prefix_fallback"}
        atomic_json(output / "meta/info.json", info)
        tasks = [{"source_root": str(source), "output_root": str(output), "record": record,
                  "quality_receipt": quality.get(record["episode_index"]), "quality_dir": str(quality_path.parent),
                  "quality_report_sha256": binding["quality_report_sha256"], "build_binding_sha256": plan["binding_sha256"],
                  "source_vision_root": str(vision), "source_vision_manifest_sha256": binding["source_vision_manifest_sha256"],
                  "legacy_vision_retired": record["episode_index"] in archive._retired_vision,
                  "root_rest": archive.model.rest[0], "root_shape_basis": archive.model.shape_basis[0]} for record in records]
        results = {}
        def completed(result):
            results[result["episode_index"]] = result
            atomic_json(output / ".pipeline/revision-progress.json", {"state": "building", "completed": len(results),
                "total": len(tasks), "latest_episode": result["episode_index"], "updated_at": datetime.now(timezone.utc).isoformat()})
            print(json.dumps({"completed": len(results), "total": len(tasks), "episode_index": result["episode_index"]}), flush=True)
        if workers == 1:
            for task in tasks:
                completed(_revise_episode(task))
        else:
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ.setdefault(name, "1")
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                for result in _parallel_tasks(pool, tasks, workers=workers, worker_control=worker_control):
                    completed(result)
        archive.assert_current()
        revised = [results[i]["record"] for i in range(len(records))]
        for relative, expected in binding["preserved_metadata_sha256"].items():
            path = output / (".pipeline/parent-publication/config.json" if relative == ".pipeline/config.json" else relative)
            if file_sha256(path) != expected or file_sha256(source / relative) != expected:
                raise ValueError("Final-FK inherited ontology, inventory or provenance changed")
        old_table = pq.read_table(source / "meta/episodes.parquet", use_threads=False)
        rows = [{name: record[name] for name in old_table.column_names} for record in revised]
        metadata_path = output / "meta/episodes.parquet"
        temporary = metadata_path.with_name(".episodes.revision.tmp")
        pq.write_table(pa.Table.from_pylist(rows, schema=old_table.schema), temporary, compression="zstd")
        if pq.read_table(temporary).to_pylist() != rows:
            raise ValueError("Final-FK episode metadata writeback differs")
        os.replace(temporary, metadata_path)
        if archive._retirement_path.exists():
            retirement = json.loads(archive._retirement_path.read_text())
            retirement["info_sha256"] = file_sha256(output / "meta/info.json")
            retirement["records_manifest_sha256"] = hashlib.sha256(canonical_json(revised)).hexdigest()
            atomic_json(output / ".pipeline/retired-vision.json", retirement)
        visual = deepcopy(archive.vision_cache.manifest)
        profile_root = output / PROFILE
        visual.update(source_root=str(output), output_root=str(profile_root), source_binding={
            "info_sha256": file_sha256(output / "meta/info.json"), "episodes_metadata_sha256": file_sha256(metadata_path),
            "split_metadata_sha256": archive.binding["split_metadata_sha256"], "records_sha256": storage_digest(revised)})
        visual["quality_revision_reuse"] = {"format": REVISION_FORMAT, "parent_profile_root": str(vision),
            "parent_manifest_sha256": binding["source_vision_manifest_sha256"], "quality_report_sha256": binding["quality_report_sha256"],
            "payload_reuse": "same_device_inode_hardlink", "join_all_native_rows_checked": False,
            "changed_motion_native_join_checked": True, "unchanged_join_evidence": "inherited_immutable_motion_and_visual",
            "features_content_rehashed": False}
        atomic_json(profile_root / "manifest.json", visual)
        sources = {}
        for record in revised:
            counts = sources.setdefault(record["source"], {"episodes": 0, "frames": 0})
            counts["episodes"] += 1
            counts["frames"] += record["length"]
        atomic_json(state_path, {"schema_version": PARAMETER_SCHEMA_VERSION, "state": "candidate",
            "independent_acceptance_pending": True, "formal_training_ready": False,
            "episodes": len(revised), "frames": archive.frames, "sources": sources, "quality_revision": MANIFEST})
        candidate = SMPLHParameterArchive(output, allow_unpublished=True, vision_profile_root=profile_root)
        candidate.assert_current()
        changed = [i for i in sorted(results) if results[i]["revision"]["motion_rewritten"]]
        result = {"format": REVISION_FORMAT, "state": "candidate", "scope": binding["scope"],
            "independent_acceptance_pending": True, "build_binding": binding, "build_binding_sha256": plan["binding_sha256"],
            "binding": candidate.binding, "archive_fingerprint": candidate.archive_fingerprint,
            "source_root": str(source), "output_root": str(output), "vision_profile_root": str(profile_root),
            "population": {"episodes": len(revised), "frames": archive.frames, "sources": sources},
            "quality_episodes": sorted(quality), "required_quality_episodes": sorted(required),
            "changed_episode_indices": changed, "motion_rewritten_episodes": changed,
            "unchanged_motion_sha256": {str(i): revised[i]["motion_sha256"] for i in sorted(results) if i not in changed},
            "model_sha256": candidate.model_sha256, "parent_model_sha256": archive.model_sha256,
            "split_fingerprint": candidate.split_fingerprint, "parent_split_fingerprint": archive.split_fingerprint,
            "coordinate_values_preserved": True, "parameter_values_preserved": True, "camera_values_preserved": True,
            "image_join_values_preserved": True, "features_reextracted": False,
            "completed_at": datetime.now(timezone.utc).isoformat()}
        atomic_json(output / MANIFEST, result)
        atomic_json(output / ".pipeline/revision-progress.json", {"state": "candidate", "completed": len(records),
            "total": len(records), "independent_acceptance_pending": True, "updated_at": datetime.now(timezone.utc).isoformat()})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-vision-root")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worker-control", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    result = build_revision(args.source_root, args.quality_report, args.output_root,
        source_vision_root=args.source_vision_root, workers=args.workers, worker_control=args.worker_control,
        resume=args.resume, diagnostic=args.diagnostic)
    print(json.dumps({"state": result["state"], "output_root": result["output_root"],
                      "changed_episodes": len(result["changed_episode_indices"])}), flush=True)


if __name__ == "__main__":
    main()
