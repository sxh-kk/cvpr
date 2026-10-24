"""Losslessly package independently admitted TRAIN-only supplementary captures.

This is a separate, variable-size contract. It never changes quality100's
100-episode admission rule, invents missing actions, or certifies policy TEST.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid

import numpy as np

import pack_handover_quality100 as legacy
import collect_handover_quality100 as collector

CONTRACT = "cvpr-handover-supplement-lerobot-pack-v1"
HELDOUT = frozenset((39, 65, 80, 92, 94))
CONTRACT_FILES = ("source_initialization_receipt.json", "source_run_config.json",
                  "source_dataset_statistics.json", "source_state_info.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def template_contract_bindings(template, ready):
    bindings = {name: legacy.sha(template / name) for name in CONTRACT_FILES}
    require(all(digest == ready.get("payload_sha256", {}).get(name)
                for name, digest in bindings.items()),
            "Inherited checkpoint/normalization contract differs from template payload SHA")
    return bindings


def validate_pack_selection(items, plan, state):
    counts = Counter(item["audit"]["source_episode_index"] for item in items)
    require(state.get("capture_target_satisfied") is True
            and len(items) == plan["target_accepted_count"]
            and len(counts) >= 6 and max(counts.values(), default=0) <= 2,
            "Complete supplement pack requires its target, at least6 TRAIN ancestors and at most2 per ancestor")


def validate_admission(audit, *, allowed_ancestors):
    """Fail closed on incomplete 3D/contact/RGB evidence; no threshold overrides."""
    require(audit.get("contract") == collector.AUDIT_CONTRACT
            and audit.get("authority_revision") == collector.AUTHORITY_REVISION
            and audit.get("status") == "independent_capture_audit_passed"
            and audit.get("eligible_dataset") is True, "Capture has no independent admission")
    source = audit.get("source_episode_index")
    require(type(source) is int and source in allowed_ancestors and source not in HELDOUT,
            "Supplementary capture is not a protected-split TRAIN ancestor")
    physics = audit.get("physical_gate", {})
    geometry = physics.get("final_placement_geometric_validation", {})
    contact = physics.get("contact_evidence", {})
    require(physics.get("acceptance_gate_version") == "handover-physics-v2"
            and physics.get("verified_handover_success") is True
            and physics.get("eligible_physics") is True
            and physics.get("capture_validation_errors") == []
            and geometry.get("require_3d_evidence") is True
            and geometry.get("passed") is True and geometry.get("v2_passed") is True
            and contact.get("strict_handover_success") is True,
            "Full physical handover, release and necessary3D placement must pass")
    require(audit.get("renderer_pose_audit", {}).get("within_predeclared_render_pose_tolerances") is True,
            "Synchronized real RTX evidence is missing")
    require(isinstance(audit.get("bindings"), dict) and audit["bindings"], "Missing bound capture files")
    require(isinstance(audit.get("capture_content_sha256"), str)
            and len(audit["capture_content_sha256"]) == 64, "Missing unique capture identity")


def load_items(admissions, template, queue_root):
    """Rebind already accepted raw bytes, without rerunning admitted dynamics."""
    ready = legacy.read(template / "PACK_READY.json")
    require(ready.get("train_target_satisfied") is True and ready.get("train_episodes") == 100,
            "Template must be the accepted original100 TRAIN package")
    for relative in ("train/meta/info.json", "train/meta/modality.json"):
        require(legacy.sha(template / relative) == ready["payload_sha256"].get(relative),
                f"Inherited template metadata changed: {relative}")
    import collect_handover_supplement as supplement
    state, plan = supplement.verified_queue(queue_root)
    registered = {(queue_root / a["audit_path"]).resolve(): a
                  for a in state["attempts"] if a["status"] == "accepted"}
    source_meta_path = template / "train/meta/episodes.jsonl"
    expected_meta = ready["payload_sha256"].get("train/meta/episodes.jsonl")
    require(expected_meta and legacy.sha(source_meta_path) == expected_meta,
            "Template episode metadata differs from its admitted payload SHA")
    source_metadata = [json.loads(line) for line in source_meta_path.read_text().splitlines() if line.strip()]
    metadata_by_candidate = {e["candidate_id"]: e for e in source_metadata}
    require(len(metadata_by_candidate) == len(source_metadata) == 100, "Template repeats or loses an episode")
    source_by_id = {}
    for receipt in ready["episodes"]:
        admission = template / "accepted_evidence" / receipt["candidate_id"] / "admission.json"
        relative = str(admission.relative_to(template))
        require(legacy.sha(admission) == receipt["admission_sha256"] == ready["payload_sha256"].get(relative),
                "Template ancestor admission differs from its published SHA")
        old_audit = legacy.read(admission)
        e = metadata_by_candidate[receipt["candidate_id"]]
        source = e["source_identity"]
        require(e.get("split") == "train" and e.get("lineage_complete") is True
                and e["source_episode_index"] == old_audit["source_episode_index"] == source["source_episode_index"]
                and e["source_dataset_key"] == source["source_dataset_key"]
                and e["lineage_group_sources"] == [{"source_dataset_key": source["source_dataset_key"],
                                                     "source_episode_index": source["source_episode_index"]}],
                "Template source identity is not a single accepted TRAIN lineage")
        source_by_id.setdefault(source["source_episode_index"], source)
    allowed = set(map(int, ready["covered_ancestors"]))
    previous = {item["capture_content_sha256"] for item in ready["episodes"]}
    identities, candidates, items = set(), set(), []
    for path in admissions:
        path = Path(path).resolve(strict=True)
        require(path in registered, "Admission is not an accepted entry in the verified supplementary queue")
        entry = registered[path]
        require(legacy.sha(path) == entry["audit_sha256"], "Registered admission SHA changed")
        audit = legacy.read(path)
        validate_admission(audit, allowed_ancestors=allowed)
        require(audit["candidate_id"] == entry["candidate_id"]
                and audit["candidate_spec_sha256"] == entry["candidate_spec_sha256"]
                and audit["plan_sha256"] == state["plan_sha256"], "Queue/admission proposal identity mismatch")
        identity = audit["capture_content_sha256"]
        require(identity not in previous and identity not in identities,
                "A supplementary demonstration must be a new capture")
        candidate_id = audit["candidate_id"]
        require(candidate_id not in candidates, "Duplicate candidate admission")
        identities.add(identity); candidates.add(candidate_id)
        for filename, expected in audit["bindings"].items():
            raw = Path(filename)
            require(raw.is_file() and legacy.sha(raw) == expected,
                    f"Admitted physical/RGB payload changed: {raw}")
        def bound_named(name):
            matches = [Path(p) for p in audit["bindings"] if Path(p).name == name]
            require(len(matches) == 1, f"Admission must bind exactly one {name}")
            return matches[0]
        manifest_path, events_path = bound_named("capture.json"), bound_named("events.jsonl")
        manifest = legacy.read(manifest_path)
        capture = manifest_path.parent
        require(legacy.owned(capture, manifest["events_path"]) == events_path,
                "Events escape the admitted capture")
        require(manifest["candidate_id"] == candidate_id
                and manifest["candidate_spec_sha256"] == audit["candidate_spec_sha256"]
                and manifest["plan_sha256"] == audit["plan_sha256"]
                and manifest["source_episode_index"] == audit["source_episode_index"],
                "Capture/admission identity mismatch")
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        require(len(events) == audit["event_count"] == manifest["event_count"], "Capture row count changed")
        require(manifest.get("code_unchanged_during_run") is True
                and manifest.get("object_restores_during_control") is False,
                "Capture control identity was not accepted")
        source = source_by_id[audit["source_episode_index"]]
        declared = next(c["ancestor"] for c in plan["candidates"] if c["candidate_id"] == candidate_id)
        require(declared == source, "Supplement source identity differs from the accepted template")
        items.append({"audit_path": path, "audit_sha256": entry["audit_sha256"],
                      "audit": audit, "capture": capture, "manifest": manifest,
                      "events": events, "source_identity": source})
    validate_pack_selection(items, plan, state)
    return items, ready


def pack(admissions, template_pack, output, queue_root):
    import pyarrow.parquet as pq

    template = Path(template_pack).resolve(strict=True)
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink()
            and not output.resolve().is_relative_to(legacy.REPO)
            and not output.resolve().is_relative_to(template), "Use a fresh external output")
    queue_root = Path(queue_root).resolve(strict=True)
    require(not output.resolve().is_relative_to(queue_root), "Packed output must be outside the collection queue")
    items, template_ready = load_items(admissions, template, queue_root)
    queue_sha = legacy.sha(queue_root / "queue.json")
    contract_bindings = template_contract_bindings(template, template_ready)
    contract_args = [template / name for name in CONTRACT_FILES]
    identity, statistics = legacy.checkpoint_contract(*contract_args)
    staging = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    root = staging / "train"; offset = 0; episodes = []; receipts = []; shape = None; tasks = {}
    try:
        for number, item in enumerate(items):
            events, audit, manifest = item["events"], item["audit"], item["manifest"]
            eligible, history = legacy.eligible_currents(events)
            values = legacy.episode_values(events, number, offset, identity["state_joint_names"], identity["hand_joint_names"])
            instruction = manifest.get("instruction")
            require(isinstance(instruction, str) and instruction.strip(), "Missing real task instruction")
            task_index = tasks.setdefault(instruction, len(tasks))
            values["task_index"] = np.full(len(events), task_index, np.int64)
            values["task_description"] = [instruction] * len(events)
            values["source_episode_index"] = np.full(len(events), audit["source_episode_index"], np.int64)
            table = legacy.arrow_table(values)
            parquet = root / f"data/chunk-000/episode_{number:06d}.parquet"
            parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, parquet, compression="zstd")
            legacy.check_parquet_roundtrip(table, pq.read_table(parquet))
            video = legacy.encode_pngs(item["capture"], events,
                root / f"videos/chunk-000/{legacy.VIDEO_KEY}/episode_{number:06d}.mp4")
            if shape is None: shape = video["shape"]
            require(shape == video["shape"], "Camera geometry differs across new captures")
            evidence = staging / "accepted_evidence" / audit["candidate_id"]
            evidence.mkdir(parents=True)
            for name, source in (("admission.json", item["audit_path"]),
                                 ("capture.json", item["capture"] / "capture.json"),
                                 ("events.jsonl", item["capture"] / manifest["events_path"])):
                shutil.copyfile(source, evidence / name)
                expected = (item["audit_sha256"] if name == "admission.json"
                            else audit["bindings"][str(source)])
                require(legacy.sha(evidence / name) == expected, "Copied admission/capture bytes changed")
            source = item["source_identity"]
            episodes.append({"episode_index": number, "tasks": [instruction], "length": len(events),
                "candidate_id": audit["candidate_id"], "source_episode_index": audit["source_episode_index"],
                "source_dataset_key": source["source_dataset_key"], "source_identity": source,
                "lineage_group_sources": [{"source_dataset_key": source["source_dataset_key"],
                                            "source_episode_index": source["source_episode_index"]}],
                "lineage_complete": True, "split": "train", "split_restriction": "train_only",
                "history_only_prefix_rows": 30, "eligible_current_rows": eligible,
                "teacher_collection_warmup": audit["teacher_collection_warmup"],
                "diversity_receipt": audit.get("diversity_receipt"),
                "physics_validation": {"gate": "handover-physics-v2", "verified_handover_success": True}})
            receipts.append({"episode_index": number, "candidate_id": audit["candidate_id"],
                "admission_sha256": item["audit_sha256"], "capture_content_sha256": audit["capture_content_sha256"],
                "frames": len(events), "eligible_current_rows": eligible, "history_rows": history,
                "video": video, "parquet_sha256": legacy.sha(parquet)})
            offset += len(events)
        meta = root / "meta"; meta.mkdir(parents=True)
        info = legacy.read(template / "train/meta/info.json")
        require(info["features"][legacy.VIDEO_KEY]["shape"] == shape,
                "Supplement video dimensions differ from the inherited feature contract")
        info.update(total_episodes=len(episodes), total_frames=offset, total_videos=len(episodes),
                    total_tasks=len(tasks), splits={"train": f"0:{len(episodes)}"})
        legacy.write(meta / "info.json", info)
        legacy.write(meta / "stats.json", statistics); legacy.write(meta / "stats_psi0.json", statistics)
        shutil.copyfile(template / "train/meta/modality.json", meta / "modality.json")
        (meta / "episodes.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in episodes))
        (meta / "tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": t}) + "\n" for t, i in tasks.items()))
        for source in contract_args: shutil.copyfile(source, staging / source.name)
        reader = legacy.verify_lerobot_reader(root, episodes, [x["events"] for x in items])
        report = {"contract": CONTRACT, "status": "cpu_lossless_supplement_pack_complete",
            "created_utc": datetime.now(timezone.utc).isoformat(), "train_episodes": len(episodes),
            "total_frames": offset, "train_windows": sum(len(e["eligible_current_rows"]) for e in episodes),
            "accepted_per_ancestor": dict(Counter(e["source_episode_index"] for e in episodes)),
            "covered_ancestors": sorted({e["source_episode_index"] for e in episodes}),
            "capture_target_satisfied": True, "minimum_distinct_sources_satisfied": True,
            "template_contract_sha256": contract_bindings,
            "existing_VAL_sources_excluded": sorted(HELDOUT), "checkpoint_contract": identity,
            "episodes": receipts, "cpu_reader_validation": reader,
            "template_pack_sha256": legacy.sha(template / "PACK_READY.json"),
            "supplement_queue_sha256": queue_sha,
            "normalization_refitted": False, "train_payload_packed": True,
            "DINO_cache_generated": False, "training_started": False, "recovery_demonstrations": False,
            "final_ready": False, "final_ready_blockers": ["real_DINO_cache_pending", "production_training_reader_preflight_pending"],
            "independent_policy_TEST_certified": False,
            "scope": "New successful TRAIN reset-diversity executions; no failed-policy state recovery, new human source, or stronger exact-cavity claim",
            "source_sha256": legacy.sha(__file__)}
        report["payload_sha256"] = {str(p.relative_to(staging)): legacy.sha(p)
                                    for p in sorted(staging.rglob("*")) if p.is_file()}
        require(template_contract_bindings(template, template_ready) == contract_bindings,
                "Inherited checkpoint contract changed during packing")
        for item in items:
            require(legacy.sha(item["audit_path"]) == item["audit_sha256"],
                    "Registered admission changed while packing")
            for group in ("bindings", "source_sha256"):
                require(all(legacy.sha(path) == identity for path, identity in item["audit"][group].items()),
                        "Admitted source or raw payload changed while packing")
        require(legacy.sha(queue_root / "queue.json") == queue_sha,
                "Supplement queue changed during packaging; preserve staging and retry a fresh snapshot")
        legacy.write(staging / "PACK_READY.json", report)
        staging.rename(output)
        return report
    except Exception as error:
        legacy.write(staging / "FAILED.json", {"error": str(error), "output_requested": str(output), "final_ready": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", action="append", required=True, type=Path)
    parser.add_argument("--template-pack", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = pack(args.admission, args.template_pack, args.output, args.queue)
    print(json.dumps({k: report[k] for k in ("status", "train_episodes", "train_windows", "final_ready")}))


if __name__ == "__main__":
    main()
