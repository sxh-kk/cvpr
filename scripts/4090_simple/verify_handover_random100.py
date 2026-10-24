"""Read-only CPU admission of the new100 pack, real DINO and production reader.

Only a separate verification report is written. PACK_READY remains immutable;
a successful report, bound to its exact pack/cache bytes, is the additional
production-readiness receipt. No policy/model is constructed or GPU initialized.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import verify_handover_supplement as shared
import pack_handover_random100 as packer

REPO = Path(__file__).resolve().parents[2]
CONTRACT = "cvpr-handover-random100-reader-verification-v1"
require, sha, read, jsonl = shared.require, shared.sha, shared.read, shared.jsonl
cpu_environment, runtime_paths = shared.cpu_environment, shared.runtime_paths
encoder_contract, validate_sample = shared.encoder_contract, shared.validate_sample


def inspect_pack(pack, *, expected_episodes=100):
    """Bind the portable pack without depending on original capture paths."""
    import numpy as np
    import pyarrow.parquet as pq
    from handover_random100_sources import SOURCE_KEY, ARCHIVE_SHA256
    pack = Path(pack).resolve(strict=True); ready = read(pack / "PACK_READY.json")
    require(expected_episodes == 100 and ready.get("contract") == packer.CONTRACT
            and ready.get("train_payload_packed") is True and ready.get("train_episodes") == 100
            and ready.get("train_target_satisfied") is True
            and ready.get("minimum_distinct_sources_satisfied") is True
            and ready.get("old_captures_merged") is False
            and ready.get("normalization_refitted") is False
            and set(ready.get("existing_VAL_sources_excluded", [])) == packer.HELDOUT
            and ready.get("truncated_source_excluded") == 38
            and ready.get("final_ready") is False,
            "Require the exact new100 immutable pack with separate DINO/production readiness")
    receipts = {str(pack / "PACK_READY.json"): sha(pack / "PACK_READY.json")}
    for relative, expected in ready["payload_sha256"].items():
        path = packer.legacy.owned(pack, relative)
        require(sha(path) == expected, "Packed bytes changed: " + relative)
        receipts[str(path)] = expected
    def bound(relative):
        path = packer.legacy.owned(pack, relative)
        require(str(path) in receipts, "Packed authority is absent from payload SHA: " + relative)
        return path
    registry = read(bound("rebuild_evidence/registry.json"))
    requirements = read(bound("rebuild_evidence/REBUILD_REQUIREMENTS.json"))
    diversity = read(bound("rebuild_evidence/diversity.json"))
    require(registry.get("contract") == packer.REGISTRY_CONTRACT, "Frozen registry contract differs")
    packer.validate_selection(registry["admissions"], requirements)
    packer.validate_diversity(diversity, sha(bound("rebuild_evidence/REBUILD_REQUIREMENTS.json")),
                              sha(bound("rebuild_evidence/registry.json")), registry["admissions"])
    source_entries = {entry["candidate_id"]: entry for entry in registry["admissions"]}
    episodes = jsonl(bound("train/meta/episodes.jsonl"))
    require(len(episodes) == len(ready["episodes"]) == 100, "Actual100 episode count differs")
    entries = {entry["episode_index"]: entry for entry in ready["episodes"]}
    require(set(entries) == set(range(100)) == {e["episode_index"] for e in episodes}
            and len({e["candidate_id"] for e in episodes}) == 100
            and {e["candidate_id"] for e in episodes} == set(source_entries), "Packed episode identity differs")
    require(set(ready["covered_ancestors"]) == {e["source_episode_index"] for e in episodes}, "Actual TRAIN ancestors differ")
    require(Counter(e["source_episode_index"] for e in episodes)
            == Counter(entry["source_episode_index"] for entry in registry["admissions"]), "Frozen ancestry counts differ")
    identity, _ = packer.legacy.checkpoint_contract(*(bound(name) for name in packer.CONTRACT_FILES))
    require(identity["bounds_sha256"] == ready["checkpoint_contract"]["bounds_sha256"], "Inherited38k bounds changed")
    tasks = {row["task"]: row["task_index"] for row in jsonl(bound("train/meta/tasks.jsonl"))}
    identities, offset = set(), 0
    for episode in sorted(episodes, key=lambda row: row["episode_index"]):
        index = episode["episode_index"]; entry = entries[index]; registered = source_entries[episode["candidate_id"]]
        source = {"source_dataset_key": SOURCE_KEY, "source_episode_index": registered["source_episode_index"]}
        require(episode.get("source_identity") == source and episode.get("lineage_group_sources") == [source]
                and episode.get("source_episode_index") == registered["source_episode_index"]
                and episode.get("source_dataset_key") == SOURCE_KEY and episode.get("split") == "train"
                and episode.get("split_restriction") == "train_only" and episode.get("lineage_complete") is True
                and episode.get("label_origin") == "actual_new_online_expert_execution"
                and episode.get("scene_proposal_sha256") == registered["scene_proposal_sha256"]
                and episode.get("hand_condition") == registered["hand_condition"] and episode.get("seed") == registered["seed"],
                "Protected actual new TRAIN lineage/condition differs")
        prefix = "accepted_evidence/" + episode["candidate_id"]
        admission = bound(prefix + "/admission.json"); audit = read(admission)
        require(sha(admission) == registered["admission_sha256"] == entry["admission_sha256"], "Admission SHA differs")
        packer.validate_admission(audit, registered)
        require(entry["candidate_id"] == episode["candidate_id"], "Episode receipt candidate differs")
        evidence_files = entry["evidence_files"]
        for relative, stored in evidence_files.items():
            require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                    "Unsafe copied run authority")
            path = bound(stored)
            require(sha(path) == audit["bindings"].get(str(Path(audit["run"]) / relative)),
                    "Copied run authority differs from independent admission")
        for name in ("source_evidence.json", "initialization_input.json", "execution_args.json", "environment_config.json",
                     "scene_proposal.json", "scene_readback.json", "scene_visual_application.json", "quality_samples.jsonl",
                     "expert_commands.jsonl", "trace.jsonl", "trajectory.npz", "physics_model.mjb", "result.json",
                     "physical_quality.json", "container_geometry_cache.json", "capture/capture.json", "capture/events.jsonl"):
            require(name in evidence_files, "Missing portable execution authority: " + name)
        evidence = read(bound(evidence_files["source_evidence.json"]))
        require(evidence.get("source_dataset_key") == SOURCE_KEY and evidence.get("archive_sha256") == ARCHIVE_SHA256
                and evidence.get("source_id") == registered["source_episode_index"], "Raw archive identity differs")
        events = jsonl(bound(evidence_files["capture/events.jsonl"]))
        eligible, histories = packer.legacy.eligible_currents(events)
        rgb = audit["RGB_state_action_history_audit"]
        require(eligible == entry["eligible_current_rows"] == episode["eligible_current_rows"] == rgb["eligible_current_rows"]
                and histories == entry["history_rows"] == rgb["history_rows"]
                and len(events) == episode["length"] == entry["frames"] == rgb["event_count"],
                "Real history/future availability changed")
        require(entry["capture_content_sha256"] == rgb["capture_content_sha256"]
                and entry["capture_content_sha256"] not in identities, "Duplicated actual capture content")
        identities.add(entry["capture_content_sha256"])
        values = packer.legacy.episode_values(events, index, offset, identity["state_joint_names"], identity["hand_joint_names"])
        instruction = episode["tasks"][0]
        values.update(task_index=np.full(len(events), tasks[instruction], np.int64), task_description=[instruction] * len(events),
                      source_episode_index=np.full(len(events), registered["source_episode_index"], np.int64))
        parquet = bound(f"train/data/chunk-000/episode_{index:06d}.parquet")
        packer.legacy.check_parquet_roundtrip(packer.legacy.arrow_table(values), pq.read_table(parquet))
        offset += len(events)
    require(offset == ready["total_frames"] and sum(len(e["eligible_current_rows"]) for e in episodes) == ready["train_windows"],
            "Actual frame/window totals differ")
    return pack, ready, episodes, entries, receipts


def make_production_subset(run_config, pack, dino_root, *, processor=None, no_aug=True):
    inspect_pack(pack)
    # This helper checks source-bound transform semantics, normalization and all
    # MotionHistoryIndex identities. Its6..12 gate only applies to its own pack
    # contract; the exact100 admission above is always used for this contract.
    return shared.make_production_subset(run_config, pack, dino_root, processor=processor, no_aug=no_aug)


def verify(pack, dino_root, reference_dino_root, run_config, output, *, expected_episodes=100):
    cpu_environment(); runtime_paths()
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from psi.data.lerobot.dino_cache import load_cache_manifest, load_episode_cache
    from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether
    require(not torch.cuda.is_initialized(), "CPU verifier must not initialize CUDA")
    pack, ready, episodes, entries, receipts = inspect_pack(pack, expected_episodes=expected_episodes)
    dino_root, reference_dino_root = Path(dino_root).resolve(strict=True), Path(reference_dino_root).resolve(strict=True)
    output = Path(output).resolve()
    require(not output.exists() and not any(output.is_relative_to(root) for root in (REPO, pack, dino_root, reference_dino_root)),
            "Use a fresh verification report outside all source/cache/repository trees")
    receipts[str(Path(run_config).resolve(strict=True))] = sha(run_config)
    manifest = load_cache_manifest(dino_root, pack / "train", expected_profile="256x320")
    reference = load_cache_manifest(reference_dino_root, expected_profile="256x320")
    require(dino_root != reference_dino_root and encoder_contract(manifest) == encoder_contract(reference),
            "New100 needs an independent cache with the inherited actual DINO model/profile/dtype contract")
    require(set(manifest["episode_indices"]) == {episode["episode_index"] for episode in episodes}, "DINO episode coverage differs")
    for root in (dino_root, reference_dino_root):
        receipts[str(root / "manifest.json")] = sha(root / "manifest.json")
    cache_reports = []
    for episode in episodes:
        index, length = episode["episode_index"], episode["length"]
        cached = load_episode_cache(dino_root, index, manifest=manifest, dataset_root=pack / "train", verify_checksums=True)
        require(cached.features.shape == (length, 5, 1024) and cached.valid.all()
                and np.array_equal(cached.frame_index, np.arange(length))
                and np.array_equal(cached.source_frame_index, np.arange(length))
                and np.allclose(cached.video_timestamp, np.arange(length) / 30, rtol=0, atol=1e-7)
                and np.allclose(cached.timestamp, cached.video_timestamp, rtol=0, atol=.25 / 30),
                "Actual DINO features/continuous30Hz frame identities differ")
        for start in range(0, length, 256):
            require(np.isfinite(cached.features[start:start + 256]).all(), "Nonfinite actual DINO feature")
        directory = dino_root / "episodes" / f"{index:08d}"
        for name in ("receipt.json", "features.npy", "frames.npz"):
            receipts[str(directory / name)] = sha(directory / name)
        parquet = pack / "train/data/chunk-000" / f"episode_{index:06d}.parquet"
        raw = pq.read_table(parquet, columns=["action_valid", "action", "action.body_token_v1_1", "action_mask"]).to_pydict()
        valid = np.asarray(raw["action_valid"], dtype=bool)
        hands, tokens, mask = (np.asarray(raw[key]) for key in ("action", "action.body_token_v1_1", "action_mask"))
        require(len(valid) == length and not valid[:30].any() and valid[30:].all()
                and np.isnan(hands[:30]).all() and np.isnan(tokens[:30]).all() and not mask[:30].any()
                and np.isfinite(hands[30:]).all() and np.isfinite(tokens[30:]).all()
                and mask[30:, :78].all() and not mask[30:, 78:].any(), "Real warmup/future labels or78+2 masks differ")
        cache_reports.append({"episode_index": index, "actual_feature_shape": list(cached.features.shape),
                              "dtype": str(cached.features.dtype), "every_feature_finite": True, "every_frame_valid": True})
        cached.features._mmap.close()
    subset, processor, cfg, normalization = make_production_subset(run_config, pack, dino_root, no_aug=True)
    history = subset.dataset.raw_dataset.motion_history
    collator = PaddedCollatorForTogether(model_max_length=processor.tokenizer.model_max_length,
        pad_token_id=processor.tokenizer.pad_token_id, padding_side="right")
    queries = []
    for episode in episodes:
        local = [int(i) for i in subset.indices if int(history.episode_ids[i]) == episode["episode_index"]]
        chosen = {local[0], local[len(local) // 2], local[-1]}
        audit = read(pack / "accepted_evidence" / episode["candidate_id"] / "admission.json")
        stages = audit["physical_gate"]["contact_evidence"].get("ordered_lifted_stages", {})
        for interval in stages.values():
            if isinstance(interval, dict) and type(interval.get("start_step")) is int:
                target_row = (50 + interval["start_step"]) * 30 / 50
                chosen.add(min(local, key=lambda i: abs(int(history.history_indices[i][-1]) - target_row)))
        for index in sorted(chosen):
            sample = subset.dataset[index]
            shapes = validate_sample(sample)
            require(np.array_equal(np.asarray(sample["motion_img_embs"]), history.get_history(index)),
                    "Actual transformed sample changed clean DINO history")
            batch = collator([sample]); validate_sample(batch, batched=True)
            require(torch.isfinite(batch["pixel_values"]).all().item(), "Actual current RGB/processor pixels are nonfinite")
            queries.append({"episode_index": episode["episode_index"], "filtered_dataset_index": index,
                "current_row": int(history.history_indices[index][-1]),
                "history_rows": history.history_indices[index].tolist(), "sample_shapes": shapes,
                "actual_current_RGB_and_production_collator_passed": True})
    for path, expected in receipts.items():
        require(sha(path) == expected, f"Read-only input changed during verification: {path}")
    require(not torch.cuda.is_initialized(), "CPU verifier unexpectedly initialized CUDA")
    report = {"contract": CONTRACT, "status": "real_DINO_and_production_reader_passed",
        "created_utc": datetime.now(timezone.utc).isoformat(), "pack_root": str(pack), "dino_root": str(dino_root),
        "reference_dino_root": str(reference_dino_root), "train_episodes": len(episodes), "train_windows": len(subset),
        "cache_episodes": cache_reports, "actual_reader_queries": queries, "normalization": normalization,
        "dataset_type": type(subset.dataset).__module__ + "." + type(subset.dataset).__name__,
        "raw_dataset_type": type(subset.dataset.raw_dataset).__module__ + "." + type(subset.dataset.raw_dataset).__name__,
        "source_sha256": receipts, "verifier_sha256": sha(__file__), "GPU_used": False, "CUDA_initialized": False,
        "model_constructed": False, "training_started": False, "old_packages_modified": False, "VAL_or_TEST_loaded": False,
        "scope": "Independent exact100 accepted new online-expert TRAIN executions; all real DINO values and all admitted history/future mappings checked, with actual CPU RGB/action/state/collator queries. No model forward/backward or policy success claim.",
        "PACK_READY_modified": False, "final_ready_automatically_promoted": False,
        "production_ready_for_bound_pack_and_cache": True, "formal100_quality_and_diversity_passed": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("pack-root", "dino-root", "reference-dino-root", "run-config", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.pack_root, args.dino_root, args.reference_dino_root, args.run_config, args.output)
    print(json.dumps({key: report[key] for key in
        ("status", "train_episodes", "train_windows", "GPU_used", "model_constructed", "production_ready_for_bound_pack_and_cache")}))


if __name__ == "__main__":
    main()
