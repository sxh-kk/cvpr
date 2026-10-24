"""Verify an independent supplement's real DINO and production reader on CPU.

The CLI never constructs a model, changes a package, or marks PACK_READY final.
``mix_training_subsets`` is an optional new-training hook: it keeps both packs
and DINO bindings separate and preserves the original ancestry probabilities.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
HELDOUT = frozenset((39, 65, 80, 92, 94))
CONTRACT = "cvpr-handover-supplement-reader-verification-v1"
REPACK_KEYS = ("image_keys", "state_keys", "action_keys", "instruction_key", "num_past_frames",
               "action_chunk_size", "pad_action_dim", "pad_state_dim", "goal_sampling",
               "state_temporal_jitter", "state_temporal_jitter_prob")
FIELD_KEYS = ("action_norm_type", "stat_action_keys", "stat_state_keys", "pad_action_dim",
              "pad_state_dim", "normalize_state", "use_norm_mask", "action_norm_masks")


def transform_contract(transform):
    if hasattr(transform, "model_dump"):
        transform = transform.model_dump(mode="json")
    return {"repack": {key: transform["repack"].get(key) for key in REPACK_KEYS},
            "field": {key: transform["field"].get(key) for key in FIELD_KEYS}}


def encoder_contract(manifest):
    keys = ("model_id", "weights_sha256", "local_file_sha256", "preprocessing", "image_process", "pooling", "token_order")
    return {"storage_dtype": manifest["storage_dtype"], "image_process": manifest["image_process"],
            "encoder": {key: manifest["encoder"].get(key) for key in keys}}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def cpu_environment():
    os.environ.update(CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True


def runtime_paths():
    for path in (REPO, REPO / "VLMExpert/src", REPO / "pipeline/src", Path(__file__).parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def inspect_pack(pack, *, expected_episodes=None):
    """Read and bind the independent package, without requiring its raw queue."""
    from pack_handover_supplement import validate_admission
    pack = Path(pack).resolve(strict=True)
    ready = read(pack / "PACK_READY.json")
    require(ready.get("contract") == "cvpr-handover-supplement-lerobot-pack-v1"
            and ready.get("train_payload_packed") is True, "Require the actual independent supplement pack")
    episodes = jsonl(pack / "train/meta/episodes.jsonl")
    require(episodes and ready.get("train_episodes") == len(episodes) == len(ready["episodes"]),
            "Supplement episode count differs")
    counts = Counter(episode["source_episode_index"] for episode in episodes)
    require(ready.get("capture_target_satisfied") is True and ready.get("minimum_distinct_sources_satisfied") is True
            and 6 <= len(episodes) <= 12 and len(counts) >= 6 and max(counts.values()) <= 2,
            "Supplement must satisfy the frozen complete6..12/at-least6-ancestors/cap2 contract")
    require(expected_episodes is None or len(episodes) == expected_episodes, "Unexpected actual supplement count")
    require(set(ready.get("existing_VAL_sources_excluded", [])) >= HELDOUT
            and ready.get("normalization_refitted") is False
            and ready.get("recovery_demonstrations") is False, "Protected TRAIN/normalization/scope contract differs")
    receipts = {str(pack / "PACK_READY.json"): sha(pack / "PACK_READY.json")}
    for relative, expected in ready["payload_sha256"].items():
        name = Path(relative)
        require(not name.is_absolute() and ".." not in name.parts, "Payload path escapes package")
        path = (pack / name).resolve(strict=True)
        require(path.is_relative_to(pack) and sha(path) == expected, f"Packed bytes changed: {relative}")
        receipts[str(path)] = expected
    entries = {entry["episode_index"]: entry for entry in ready["episodes"]}
    require(len(entries) == len(episodes)
            and {episode["episode_index"] for episode in episodes} == set(entries), "Duplicate/missing package episode identity")
    ancestors = {episode["source_episode_index"] for episode in episodes}
    require(not ancestors & HELDOUT and ancestors == set(ready["covered_ancestors"]), "Supplement ancestry differs")
    for episode in episodes:
        source, entry = episode["source_identity"], entries[episode["episode_index"]]
        require(episode.get("split") == "train" and episode.get("split_restriction") == "train_only"
                and episode.get("lineage_complete") is True and source["source_episode_index"] == episode["source_episode_index"]
                and episode["lineage_group_sources"] == [{"source_dataset_key": source["source_dataset_key"],
                                                          "source_episode_index": source["source_episode_index"]}],
                "Require single-ancestor protected TRAIN lineage")
        require(entry["candidate_id"] == episode["candidate_id"]
                and entry["eligible_current_rows"] == episode["eligible_current_rows"], "Accepted window metadata differs")
        path = pack / "accepted_evidence" / episode["candidate_id"] / "admission.json"
        require(str(path) in receipts and receipts[str(path)] == entry["admission_sha256"], "Admission SHA differs")
        audit = read(path)
        validate_admission(audit, allowed_ancestors=ancestors)
        require(audit["source_episode_index"] == episode["source_episode_index"]
                and audit["candidate_id"] == episode["candidate_id"]
                and audit["capture_content_sha256"] == entry["capture_content_sha256"], "Admission identity differs")
    return pack, ready, episodes, entries, receipts


def validate_mapping(history, candidates, columns, episodes, entries):
    """Match every production filtered index to admitted row/history identities."""
    import numpy as np
    metadata = {episode["episode_index"]: episode for episode in episodes}
    require(set(candidates) == set(metadata), "Production episode coverage differs")
    selected = []
    for episode_index in sorted(candidates):
        episode = metadata[episode_index]
        start, stop = history.source_episode_ranges[0][episode_index]
        require(stop - start == episode["length"], "Actual episode range differs")
        rows = []
        for index in candidates[episode_index]:
            original = int(history.original_indices[index]); row = original - start
            require(int(history.episode_ids[index]) == episode_index and start <= original < stop,
                    "Filtered Dataset index points to another episode")
            require(int(columns["frame_index"][original]) == row and row >= 30
                    and original + 30 <= stop and columns["action_valid"][original:original + 30].all(),
                    "Unavailable history/future action entered production supervision")
            require(np.array_equal(history.history_indices[index], entries[episode_index]["history_rows"][str(row)])
                    and max(history.history_indices[index]) == row,
                    "Actual six-frame production history differs from admission")
            rows.append(row); selected.append(int(index))
        require(rows == episode["eligible_current_rows"], "Production windows differ from admitted current rows")
    require(len(selected) == len(set(selected)), "Production repeats a filtered window")
    return selected


def validate_sample(sample, *, batched=False):
    import numpy as np
    prefix = (1,) if batched else ()
    for key, shape in (("actions", (30, 80)), ("actions_mask", (30, 80)),
                       ("states", (1, 45)), ("motion_img_embs", (6, 5, 1024))):
        value = np.asarray(sample[key])
        require(value.shape == prefix + shape and np.isfinite(value).all(), f"Invalid real {key} shape/value")
    mask = np.asarray(sample["actions_mask"])
    require(np.all(mask[..., :78] == 1) and np.all(mask[..., 78:] == 0), "Require78 valid and2 padded action channels")
    return {key: list(np.shape(sample[key])) for key in ("actions", "actions_mask", "states", "motion_img_embs")}


def make_production_subset(run_config, pack, dino_root, *, processor=None, no_aug=False):
    """Construct the actual configured Dataset, then its complete-future Subset."""
    runtime_paths()
    import torch
    from transformers import AutoProcessor
    from psi.config.train.posttrain_sonic_psi0_config import DynamicLaunchConfig
    from finetune_carrybox import eligible_dataset_rows, normalization_receipt
    pack = Path(pack).resolve(strict=True)
    ready = read(pack / "PACK_READY.json")
    if ready["contract"] == "cvpr-handover-supplement-lerobot-pack-v1":
        inspect_pack(pack)
    episodes = jsonl(pack / "train/meta/episodes.jsonl")
    config = deepcopy(read(run_config) if isinstance(run_config, (str, Path)) else run_config)
    source_config_path = pack / "source_run_config.json"
    require(sha(source_config_path) == ready["payload_sha256"].get("source_run_config.json"),
            "Inherited source transform configuration changed")
    require(transform_contract(config["data"]["transform"])
            == transform_contract(read(source_config_path)["data"]["transform"]),
            "Production action/state layout or normalization mode differs from the admitted source")
    config["data"].update(root_dir=str(pack), train_repo_ids=["train"], val_repo_ids=["train"],
                          dino_cache_root=str(Path(dino_root).resolve(strict=True)), task_sample_weights=[])
    config["data"]["transform"]["repack"]["action_mask_key"] = "action_mask"
    config["data"]["transform"]["field"]["stat_path"] = str(pack / "source_dataset_statistics.json")
    cfg = DynamicLaunchConfig.model_validate(config)
    norm = normalization_receipt(cfg.data.transform.field)
    require(norm["stat_file_sha256"] == ready["checkpoint_contract"]["statistics"]["sha256"]
            and norm["bounds_sha256"] == ready["checkpoint_contract"]["bounds_sha256"], "Inherited normalization differs")
    processor = processor or AutoProcessor.from_pretrained(cfg.model.model_name_or_path, local_files_only=True)
    dataset = cfg.data(split="train", transform_kwargs={"vlm_processor": processor, "no_aug": no_aug})
    candidates, columns, history = eligible_dataset_rows(dataset, len(episodes), source_frame_rule="ceil")
    entries = {entry["episode_index"]: entry for entry in ready["episodes"]}
    selected = validate_mapping(history, candidates, columns, episodes, entries)
    return torch.utils.data.Subset(dataset, selected), processor, cfg, norm


def ancestry_preserving_mix(old_weights, old_sources, new_sources, *, fraction=.5):
    """Split each covered ancestor's existing mass between its old/new windows."""
    require(type(fraction) in (int, float) and math.isfinite(fraction) and 0 < fraction < 1,
            "New source fraction must lie strictly between0 and1")
    weights = [float(value) for value in old_weights]
    require(len(weights) == len(old_sources) and weights and new_sources
            and all(math.isfinite(value) and value > 0 for value in weights), "Invalid mixed sampling inputs")
    require(all(type(source) is int and source not in HELDOUT for source in [*old_sources, *new_sources]),
            "Mixed sampling requires protected integer TRAIN ancestors")
    counts = Counter(new_sources)
    require(set(counts) <= set(old_sources), "Supplement adds an unregistered ancestor")
    total = sum(weights)
    prior = {source: sum(value for value, item in zip(weights, old_sources) if item == source) / total
             for source in set(old_sources)}
    mixed_old = [value / total * (1 - fraction if source in counts else 1)
                 for value, source in zip(weights, old_sources)]
    mixed_new = [prior[source] * fraction / counts[source] for source in new_sources]
    result = mixed_old + mixed_new
    require(all(math.isfinite(value) and value > 0 for value in result)
            and math.isclose(sum(result), 1, rel_tol=0, abs_tol=1e-10), "Mixed weights are invalid")
    return result, {"contract": "cvpr-handover-separate-pack-training-mixture-v1",
        "pack_count": 2, "physical_pack_merge_performed": False, "validation_changed": False,
        "new_fraction_per_covered_ancestor": fraction, "old_pack_probability_mass": sum(mixed_old),
        "supplement_probability_mass": sum(mixed_new), "ancestor_probability_mass": {str(k): v for k, v in sorted(prior.items())},
        "old_pack_policy": "critical sidecar probabilities retained within each ancestor",
        "supplement_policy": "uniform admitted windows within each covered ancestor",
        "scope": "Two separate real production Subsets with separate DINO/source bindings; no combined106 episode package"}


def mix_training_subsets(old_subset, new_subset, old_sidecar, *, seed=20261024, fraction=.5):
    """New-training hook returning a ConcatDataset, global sampler and receipt.

    Construct the two Subsets with ``make_production_subset`` separately. Keep
    the existing VAL dataset/loader. Install the returned sampler on the new
    plain training DataLoader before Accelerator.prepare; only Accelerator
    shards this identical global epoch stream across ranks.
    """
    runtime_paths()
    import torch
    from finetune_carrybox import eligible_dataset_rows, normalization_receipt
    from handover_critical_sampler import (_production_base, weights_for_subset, EpochWeightedSampler,
                                           load_sidecar, CriticalSidecar)
    sources, roots, caches = [], [], []
    normalizations, transforms, encoders = [], [], []
    from psi.data.lerobot.dino_cache import load_cache_manifest
    for subset in (old_subset, new_subset):
        base, resolved, history = _production_base(subset)
        root = Path(history.pack_info[0][1]).resolve(strict=True)
        episodes = jsonl(root / "meta/episodes.jsonl")
        ready = read(root.parent / "PACK_READY.json")
        entries = {entry["episode_index"]: entry for entry in ready["episodes"]}
        candidates, columns, actual_history = eligible_dataset_rows(base, len(episodes), source_frame_rule="ceil")
        expected = validate_mapping(actual_history, candidates, columns, episodes, entries)
        require(len(resolved) == len(expected) and set(resolved) == set(expected),
                "Mixed production Subset must contain all admitted windows exactly once")
        by_episode = {e["episode_index"]: e["source_episode_index"] for e in episodes}
        sources.append([by_episode[int(history.episode_ids[index])] for index in resolved])
        roots.append(str(root)); caches.append(str(Path(history.pack_info[0][0]).resolve(strict=True)))
        normalizations.append(normalization_receipt(base.transform.field))
        transforms.append(transform_contract(base.transform))
        encoders.append(encoder_contract(load_cache_manifest(caches[-1], root, expected_profile="256x320")))
    require(roots[0] != roots[1] and caches[0] != caches[1], "Keep supplementary source/cache independently bound")
    require(read(Path(roots[0]).parent / "PACK_READY.json")["contract"] == "cvpr-handover-quality100-lerobot-pack-v1"
            and read(Path(roots[1]).parent / "PACK_READY.json")["contract"] == "cvpr-handover-supplement-lerobot-pack-v1",
            "Expected original100 plus independently accepted supplementary pack")
    inspect_pack(Path(roots[1]).parent)
    require(all(normalizations[0][key] == normalizations[1][key] for key in ("stat_file_sha256", "bounds_sha256")),
            "Both production datasets must inherit identical source normalization")
    require(transforms[0] == transforms[1] and encoders[0] == encoders[1],
            "Both actual production datasets must share action/state transform and DINO contracts")
    for root, signature in zip(roots, transforms):
        source = Path(root).parent / "source_run_config.json"
        require(signature == transform_contract(read(source)["data"]["transform"]),
                "Actual mixed dataset transform differs from its admitted source")
    if not isinstance(old_sidecar, CriticalSidecar):
        old_sidecar = load_sidecar(old_sidecar, expected_window_count=None)
    require(old_sidecar.metadata.get("critical_focus_policy_revision") == "critical-conversions-v3",
            "The original100 component requires the reviewed v3 focus policy")
    old = weights_for_subset(old_subset, old_sidecar, expected_window_count=None)
    probabilities, receipt = ancestry_preserving_mix(old.tolist(), *sources, fraction=fraction)
    receipt.update(dataset_roots=roots, dino_roots=caches,
                   source_pack_ready_sha256=[sha(Path(root).parent / "PACK_READY.json") for root in roots],
                   source_window_counts=[len(old_subset), len(new_subset)],
                   original_critical_rows_sha256=old_sidecar.metadata["rows_sha256"],
                   original_critical_weights_digest=old_sidecar.metadata["weights_digest"],
                   both_subsets_cover_all_admitted_windows=True,
                   shared_production_transform_contract=transforms[0],
                   shared_DINO_encoder_contract=encoders[0],
                   normalization_bounds_sha256=normalizations[0]["bounds_sha256"],
                   fraction_is_explicit_experiment_setting_not_an_optimum_claim=True)
    return torch.utils.data.ConcatDataset([old_subset, new_subset]), EpochWeightedSampler(probabilities, seed=seed), receipt


def verify(pack, dino_root, reference_dino_root, run_config, output, *, expected_episodes=None):
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
            "Supplement needs an independent cache with the inherited actual DINO model/profile/dtype contract")
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
        "model_constructed": False, "training_started": False, "original100_modified": False, "VAL_or_TEST_loaded": False,
        "scope": "Independent variable-size accepted TRAIN supplement only; all real DINO values and all admitted history/future mappings checked, with actual CPU RGB/action/state/collator queries. No model forward/backward or policy success claim.",
        "PACK_READY_modified": False, "final_ready_automatically_promoted": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True, type=Path)
    parser.add_argument("--dino-root", required=True, type=Path)
    parser.add_argument("--reference-dino-root", required=True, type=Path)
    parser.add_argument("--run-config", required=True, type=Path, help="Existing localized production run_config.json; processor/config only")
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.pack_root, args.dino_root, args.reference_dino_root, args.run_config, args.output,
                    expected_episodes=args.expected_episodes)
    print(json.dumps({key: report[key] for key in ("status", "train_episodes", "train_windows", "GPU_used", "model_constructed")}))


if __name__ == "__main__":
    main()
