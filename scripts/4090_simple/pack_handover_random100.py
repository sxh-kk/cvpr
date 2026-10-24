"""Pack exactly100 independently admitted, genuinely new random TRAIN runs.

This entrypoint never collects, repairs, reruns a failed pilot, merges old packs,
refits normalization or promotes the separate DINO/production readiness gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import uuid

import numpy as np

import pack_handover_quality100 as legacy

CONTRACT = "cvpr-handover-random100-lerobot-pack-v1"
REGISTRY_CONTRACT = "cvpr-handover-random100-admission-registry-v1"
ADMISSION_CONTRACT = "cvpr-handover-random100-independent-capture-audit-v1"
DIVERSITY_CONTRACT = "cvpr-handover-random100-diversity-audit-v1"
HELDOUT = frozenset((39, 65, 80, 92, 94))
EXCLUDED = HELDOUT | {38}
CONTRACT_FILES = ("source_initialization_receipt.json", "source_run_config.json",
                  "source_dataset_statistics.json", "source_state_info.json")
CHECKS = ("source_coverage", "scene_coverage", "near_duplicate", "stage_coverage", "actual_command_adaptation")
FOOTPRINT_CONTRACT = "cvpr-handover-actual-full-outer-footprint-final50-v1"
ADDITIONAL_DATASET_QUALITY = {"actual_full_outer_footprint_final50": {
    "window_states": 50, "minimum_outer_hull_margin_m": .005,
    "geometry": "actual_MJB_target_collision_vertices_and_container_outer_collision_hull",
    "exact_inner_cavity_claimed": False}}


def require(value, message):
    if not value:
        raise ValueError(message)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def candidate_identity(source, seed, scene_sha):
    require(type(source) is int and type(seed) is int and valid_sha(scene_sha), "Invalid candidate inputs")
    digest = hashlib.sha256(json.dumps([source, seed, scene_sha], separators=(",", ":")).encode()).hexdigest()
    return f"source{source:03d}-{digest[:24]}"


def validate_requirements(requirements):
    require(requirements.get("contract") == "cvpr-handover-random100-rebuild-requirements-v1"
            and requirements.get("user_requested_count") == 100
            and set(requirements.get("protected_VAL_sources", [])) == HELDOUT
            and requirements.get("truncated_source_excluded") == 38
            and requirements.get("minimum_distinct_accepted_original_sources", 0) >= 50
            and requirements.get("maximum_accepted_per_source") == 2
            and requirements.get("additional_dataset_quality") == ADDITIONAL_DATASET_QUALITY
            and requirements.get("formal_TRAIN_seed_range_inclusive") == [1, 2**30 - 1]
            and requirements.get("formal_scene_sampling", {}).get("unique_actual_scene_proposal_sha256") is True,
            "Frozen100/minimum50/cap2/VAL-and38 rebuild requirements missing")


def validate_selection(entries, requirements):
    validate_requirements(requirements)
    require(len(entries) == 100, "Formal package requires exactly100 accepted new executions")
    sources = [row.get("source_episode_index") for row in entries]
    require(all(type(s) is int and 0 <= s < 100 and s not in EXCLUDED for s in sources),
            "Raw TRAIN namespace excludes all5 VAL sources and truncated38")
    counts = Counter(sources)
    require(len(counts) >= requirements["minimum_distinct_accepted_original_sources"]
            and max(counts.values()) <= requirements["maximum_accepted_per_source"],
            "Insufficient independent original TRAIN coverage or repeated source exceeds cap2")
    for key in ("candidate_id", "run", "admission_path", "admission_sha256", "scene_proposal_sha256"):
        require(len({row.get(key) for row in entries}) == 100 and all(row.get(key) for row in entries),
                f"Duplicated/missing registered {key}")
    require(all(type(row.get("seed")) is int and 1 <= row["seed"] < 2**30
                and row.get("hand_condition") in ("declared", "inferred_effective")
                and valid_sha(row["admission_sha256"]) and valid_sha(row.get("scene_proposal_sha256"))
                and isinstance(row["candidate_id"], str)
                and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", row["candidate_id"])
                for row in entries), "Registry execution condition/identity missing")
    return counts


def validate_diversity(report, requirements_sha, registry_sha, entries):
    require(report.get("contract") == DIVERSITY_CONTRACT and report.get("passed") is True
            and report.get("requirements_sha256") == requirements_sha
            and report.get("registry_sha256") == registry_sha
            and all(report.get("checks", {}).get(key) is True for key in CHECKS)
            and report.get("admission_sha256_by_candidate") == {
                row["candidate_id"]: row["admission_sha256"] for row in entries},
            "Final physical/scene/motion diversity proof does not bind this complete100 registry")
    require(report.get("no_seed_only_diversity_claim") is True
            and isinstance(report.get("bindings"), dict) and report["bindings"],
            "Actual scene/motion evidence bindings required; seed-only diversity cannot pass")


def validate_admission(audit, entry):
    require(audit.get("contract") == ADMISSION_CONTRACT
            and audit.get("status") == "independent_capture_audit_passed"
            and audit.get("eligible_dataset") is True and audit.get("require_rgb") is True
            and audit.get("replay_expert") is True and audit.get("source_split") == "train"
            and audit.get("source_episode_index") == entry["source_episode_index"]
            and Path(audit.get("run", "")).resolve() == Path(entry["run"]).resolve()
            and audit.get("scene_proposal_sha256") == entry["scene_proposal_sha256"],
            "Require the independent new-online-execution RTX admission; old captures cannot enter")
    validate_full_footprint(audit.get("actual_full_outer_footprint_final50", {}))
    physical = audit.get("physical_gate", {})
    require(physical.get("acceptance_gate_version") == "handover-physics-v2"
            and physical.get("eligible_physics") is True and physical.get("verified_handover_success") is True
            and physical.get("capture_validation_errors") == []
            and physical.get("contact_evidence", {}).get("strict_handover_success") is True
            and physical.get("final_placement_geometric_validation", {}).get("v2_passed") is True
            and physical.get("final_placement_geometric_validation", {}).get("require_3d_evidence") is True,
            "Original complete physics-v2 handover/placement gate did not pass")
    rgb = audit.get("RGB_state_action_history_audit", {})
    require(audit.get("scene_readback_verified") is True
            and audit.get("expert_reproduction", {}).get("status") == "all_generated_commands_reproduced_on_CPU"
            and rgb.get("status") == "all_actual_RTX_events_pixels_actions_and_renderer_poses_passed"
            and valid_sha(rgb.get("capture_content_sha256"))
            and rgb.get("eligible_current_rows") and rgb.get("history_rows")
            and audit.get("quality_gate_redefined") is False
            and audit.get("source_fixed_tokens_used_as_new_labels") is False
            and isinstance(audit.get("bindings"), dict) and audit["bindings"],
            "Actual RTX scene/online commands/history lineage is incomplete")


def validate_full_footprint(receipt):
    """Require the independently rebuilt actual50-state collision-vertex gate."""
    require(receipt.get("contract") == FOOTPRINT_CONTRACT and receipt.get("passed") is True
            and type(receipt.get("checked_states")) is int and receipt["checked_states"] == 50
            and receipt.get("window_states") == 50
            and receipt.get("required_minimum_outer_hull_margin_m") == .005
            and receipt.get("all_states_all_vertices_inside_outer_hull") is True
            and receipt.get("AABB_or_source_bbox_used_as_vertices") is False
            and receipt.get("exact_inner_cavity_claimed") is False
            and receipt.get("original_handover_physics_v2_modified") is False
            and receipt.get("geometry_source") == "actual_MJB_compiled_collision_vertices_at_each_saved_integration_state"
            and receipt.get("physics_steps_performed") == 0,
            "Missing actual collision-vertex final50 dataset-quality contract; AABB is insufficient")
    rows = receipt.get("measurements", [])
    require(isinstance(rows, list) and len(rows) == 50, "Require every final50 actual collision-vertex measurement")
    margins = np.asarray([row.get("minimum_outer_hull_margin_m") for row in rows], dtype=np.float64)
    minimum = float(receipt.get("minimum_outer_hull_margin_m", float("nan")))
    require(np.isfinite(margins).all() and np.all(margins >= .005) and np.isfinite(minimum)
            and minimum >= .005 and abs(minimum - float(margins.min())) <= 1e-12
            and all(row.get("passed") is True and row.get("all_target_collision_vertices_inside_outer_hull") is True
                    and row.get("target_collision_vertex_count", 0) > 0
                    and row.get("container_collision_vertex_count", 0) > 0
                    and row.get("outer_hull_vertex_count", 0) >= 3 for row in rows),
            "Every actual final50 target footprint must retain at least5mm outer-hull margin")
    indices = np.asarray([row.get("physics_state_row") for row in rows], dtype=np.int64)
    times = np.asarray([row.get("sim_time_s") for row in rows], dtype=np.float64)
    require(np.array_equal(np.diff(indices), np.ones(49, np.int64)) and np.isfinite(times).all()
            and np.allclose(np.diff(times), .02, rtol=0, atol=1e-6), "Actual final50 footprint state clock is incomplete")


def verify_bindings(bindings):
    require(bindings and all(valid_sha(value) and Path(path).is_file() and legacy.sha(path) == value
                             for path, value in bindings.items()), "Bound source/evidence bytes changed")


def load_items(registry_path, requirements_path, diversity_path):
    from handover_random100_sources import SOURCE_KEY, ARCHIVE_SHA256
    registry_path, requirements_path, diversity_path = [Path(p).resolve(strict=True)
        for p in (registry_path, requirements_path, diversity_path)]
    registry, requirements, diversity = map(legacy.read, (registry_path, requirements_path, diversity_path))
    inputs = {str(p): legacy.sha(p) for p in (registry_path, requirements_path, diversity_path)}
    require(registry.get("contract") == REGISTRY_CONTRACT
            and registry.get("requirements_sha256") == inputs[str(requirements_path)]
            and Path(registry["requirements_path"]).resolve() == requirements_path, "Registry requirements binding differs")
    entries = registry["admissions"]
    validate_selection(entries, requirements)
    validate_diversity(diversity, inputs[str(requirements_path)], inputs[str(registry_path)], entries)
    verify_bindings(diversity["bindings"])
    items, identities = [], set()
    for entry in entries:
        run, admission = Path(entry["run"]).resolve(strict=True), Path(entry["admission_path"]).resolve(strict=True)
        require(legacy.sha(admission) == entry["admission_sha256"], "Registered admission SHA changed")
        audit = legacy.read(admission); validate_admission(audit, entry)
        verify_bindings(audit["bindings"]); verify_bindings(audit["source_sha256"])
        def bound(name):
            path = (run / name).resolve(strict=True)
            require(path.is_relative_to(run) and audit["bindings"].get(str(path)) == legacy.sha(path),
                    "Admission must bind actual run evidence: " + name)
            return path
        evidence = legacy.read(bound("source_evidence.json"))
        require(evidence.get("source_dataset_key") == SOURCE_KEY and evidence.get("archive_sha256") == ARCHIVE_SHA256
                and evidence.get("source_id") == entry["source_episode_index"] and evidence.get("split") == "train",
                "Raw archive namespace/ancestor differs; old24 template provenance cannot substitute")
        args = legacy.read(bound("execution_args.json"))
        require(args["source_episode"] == entry["source_episode_index"] and args["seed"] == entry["seed"]
                and args["hand_condition"] == entry["hand_condition"] and args["render_mode"] == "isaac",
                "Registered fresh execution args differ")
        manifest_path = bound("capture/capture.json"); manifest = legacy.read(manifest_path)
        capture = manifest_path.parent; events_path = legacy.owned(capture, manifest["events_path"])
        require(audit["bindings"].get(str(events_path)) == legacy.sha(events_path), "Captured events unbound")
        events = jsonl(events_path)
        require(len(events) == manifest["event_count"] == audit["RGB_state_action_history_audit"]["event_count"]
                and manifest.get("execution_contract") == "cvpr-handover-random100-online-execution-v1"
                and manifest.get("commands_generated_online") is True
                and manifest.get("source_episode_index") == entry["source_episode_index"]
                and manifest.get("scene_proposal_sha256") == entry["scene_proposal_sha256"],
                "Actual new RTX stream differs from its admission")
        identity = audit["RGB_state_action_history_audit"]["capture_content_sha256"]
        require(identity not in identities, "The same actual capture cannot appear twice")
        identities.add(identity)
        eligible, histories = legacy.eligible_currents(events)
        require(eligible == audit["RGB_state_action_history_audit"]["eligible_current_rows"]
                and histories == audit["RGB_state_action_history_audit"]["history_rows"], "Admitted real history/future changed")
        items.append(dict(entry=entry, audit=audit, admission=admission, run=run, capture=capture,
                          events=events, manifest=manifest, source_evidence=evidence, args=args,
                          eligible=eligible, histories=histories, capture_identity=identity))
    return items, requirements, diversity, inputs


def pack(registry_path, requirements_path, diversity_path, template_pack, output):
    import pyarrow.parquet as pq
    template = Path(template_pack).resolve(strict=True); output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink() and not output.resolve().is_relative_to(legacy.REPO)
            and not output.resolve().is_relative_to(template), "Use a fresh external random100 package")
    items, requirements, diversity, inputs = load_items(registry_path, requirements_path, diversity_path)
    require(all(not output.resolve().is_relative_to(item["run"]) for item in items), "Output overlaps a capture")
    require(all(not output.resolve().is_relative_to(Path(item["args"][name]).resolve())
                for item in items for name in ("raw_root", "data_root", "sonic_resources")),
            "Output overlaps immutable source/assets/official model resources")
    template_ready = legacy.read(template / "PACK_READY.json")
    template_bindings = {name: legacy.sha(template / name) for name in
                         (*CONTRACT_FILES, "train/meta/info.json", "train/meta/modality.json")}
    require(all(template_ready["payload_sha256"].get(name) == value for name, value in template_bindings.items()),
            "Original38k contract/template bytes differ from published payload SHA")
    identity, statistics = legacy.checkpoint_contract(*(template / name for name in CONTRACT_FILES))
    staging = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"; staging.mkdir(parents=True)
    root = staging / "train"; episodes, receipts, tasks = [], [], {}; offset = 0; shape = None
    try:
        frozen = staging / "rebuild_evidence"; frozen.mkdir()
        for name, path in (("registry.json", registry_path), ("REBUILD_REQUIREMENTS.json", requirements_path),
                           ("diversity.json", diversity_path)):
            shutil.copyfile(path, frozen / name)
            require(legacy.sha(frozen / name) == inputs[str(Path(path).resolve())], "Frozen rebuild evidence changed")
        for number, item in enumerate(items):
            entry, events = item["entry"], item["events"]
            values = legacy.episode_values(events, number, offset, identity["state_joint_names"], identity["hand_joint_names"])
            instruction = item["manifest"].get("instruction")
            require(isinstance(instruction, str) and instruction.strip(), "Missing real instruction")
            task_index = tasks.setdefault(instruction, len(tasks))
            values.update(task_index=np.full(len(events), task_index, np.int64), task_description=[instruction] * len(events),
                          source_episode_index=np.full(len(events), entry["source_episode_index"], np.int64))
            table = legacy.arrow_table(values)
            parquet = root / f"data/chunk-000/episode_{number:06d}.parquet"; parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, parquet, compression="zstd"); legacy.check_parquet_roundtrip(table, pq.read_table(parquet))
            video = legacy.encode_pngs(item["capture"], events,
                root / f"videos/chunk-000/{legacy.VIDEO_KEY}/episode_{number:06d}.mp4")
            if shape is None: shape = video["shape"]
            require(shape == video["shape"], "Actual camera dimensions differ")
            evidence = staging / "accepted_evidence" / entry["candidate_id"]; evidence.mkdir(parents=True)
            shutil.copyfile(item["admission"], evidence / "admission.json")
            require(legacy.sha(evidence / "admission.json") == entry["admission_sha256"], "Admission copy changed")
            # Preserve all smaller run authority, including actual states/model,
            # quality, config, scene, commands and code snapshots. Lossless video
            # replaces PNG duplication; their original hashes remain in admission.
            evidence_map = {}
            for path, expected in item["audit"]["bindings"].items():
                path = Path(path).resolve(strict=True)
                if path.is_relative_to(item["run"]) and path.suffix.lower() != ".png":
                    relative = path.relative_to(item["run"])
                    destination = evidence / "run" / relative; destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, destination); require(legacy.sha(destination) == expected, "Copied run evidence changed")
                    evidence_map[str(relative)] = str(destination.relative_to(staging))
            source = {"source_dataset_key": item["source_evidence"]["source_dataset_key"],
                      "source_episode_index": entry["source_episode_index"]}
            episodes.append(dict(episode_index=number, tasks=[instruction], length=len(events),
                candidate_id=entry["candidate_id"], source_episode_index=entry["source_episode_index"],
                source_dataset_key=source["source_dataset_key"], source_identity=source, lineage_group_sources=[source],
                lineage_complete=True, split="train", split_restriction="train_only", history_only_prefix_rows=30,
                eligible_current_rows=item["eligible"], seed=entry["seed"], hand_condition=entry["hand_condition"],
                scene_proposal_sha256=entry["scene_proposal_sha256"],
                label_origin="actual_new_online_expert_execution", new_human_demonstration_claim=False))
            receipts.append(dict(episode_index=number, candidate_id=entry["candidate_id"],
                admission_sha256=entry["admission_sha256"], capture_content_sha256=item["capture_identity"],
                frames=len(events), eligible_current_rows=item["eligible"], history_rows=item["histories"],
                video=video, parquet_sha256=legacy.sha(parquet), evidence_files=evidence_map))
            offset += len(events)
        meta = root / "meta"; meta.mkdir(parents=True)
        info = legacy.read(template / "train/meta/info.json")
        require(info["features"][legacy.VIDEO_KEY]["shape"] == shape, "Actual RGB differs from inherited contract")
        info.update(total_episodes=100, total_frames=offset, total_videos=100, total_tasks=len(tasks), splits={"train": "0:100"})
        legacy.write(meta / "info.json", info); legacy.write(meta / "stats.json", statistics); legacy.write(meta / "stats_psi0.json", statistics)
        shutil.copyfile(template / "train/meta/modality.json", meta / "modality.json")
        (meta / "episodes.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in episodes))
        (meta / "tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": t}) + "\n" for t, i in tasks.items()))
        for name in CONTRACT_FILES: shutil.copyfile(template / name, staging / name)
        reader = legacy.verify_lerobot_reader(root, episodes, [item["events"] for item in items])
        report = dict(contract=CONTRACT, status="cpu_lossless_random100_pack_complete", created_utc=datetime.now(timezone.utc).isoformat(),
            train_episodes=100, total_frames=offset, train_windows=sum(len(e["eligible_current_rows"]) for e in episodes),
            covered_ancestors=sorted({e["source_episode_index"] for e in episodes}),
            accepted_per_ancestor=dict(Counter(e["source_episode_index"] for e in episodes)), episodes=receipts,
            train_target_satisfied=True, minimum_distinct_sources_satisfied=True, train_payload_packed=True,
            existing_VAL_sources_excluded=sorted(HELDOUT), truncated_source_excluded=38,
            checkpoint_contract=identity, template_contract_sha256=template_bindings,
            template_pack_sha256=legacy.sha(template / "PACK_READY.json"), rebuild_input_sha256=inputs,
            normalization_refitted=False, DINO_cache_generated=False, training_started=False,
            old_captures_merged=False, new_human_demonstration_claim=False, cpu_reader_validation=reader,
            final_ready=False, final_ready_blockers=["real_DINO_cache_pending", "production_training_reader_preflight_pending"],
            independent_policy_TEST_certified=False, packer_sha256=legacy.sha(__file__))
        report["payload_sha256"] = {str(p.relative_to(staging)): legacy.sha(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        verify_bindings(inputs); verify_bindings(diversity["bindings"])
        verify_bindings({str(template / p): v for p, v in template_bindings.items()})
        for item in items:
            require(legacy.sha(item["admission"]) == item["entry"]["admission_sha256"], "Admission changed during packaging")
            verify_bindings(item["audit"]["bindings"]); verify_bindings(item["audit"]["source_sha256"])
        legacy.write(staging / "PACK_READY.json", report); staging.rename(output)
        return report
    except Exception as error:
        legacy.write(staging / "FAILED.json", dict(error=str(error), final_ready=False)); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("registry", "requirements", "diversity", "template-pack", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    report = pack(args.registry, args.requirements, args.diversity, args.template_pack, args.output)
    print(json.dumps({key: report[key] for key in ("status", "train_episodes", "train_windows", "final_ready")}))


if __name__ == "__main__":
    main()
