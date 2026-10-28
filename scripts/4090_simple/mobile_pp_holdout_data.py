"""Read-only episode95/5 views over the already accepted fixed Mobile100 pack.

The original archive and its train_only receipts stay unchanged. This module
records the user's new per-run role assignment, checks whole-episode lineage,
and selects existing production-reader rows. It does not convert/audit frames,
extract features, change normalization, load a model, or use a GPU.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid

CONTRACT = "cvpr-mobile-pp100-95train-5eval-split-v1"
TASK_ID = "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
EPISODES, TRAIN_EPISODES, EVAL_EPISODES, PROBE_EPISODES = 100, 95, 5, 5
TOTAL_FRAMES, TOTAL_WINDOWS = 61806, 55906
SOURCE_ROLE_OVERRIDE = {
    "source_archive_split": "train", "source_archive_split_restriction": "train_only",
    "new_run_roles": "95 TRAIN episodes; 5 held-out EVAL episodes; 5 TRAIN-probe episodes are a subset of TRAIN",
    "authorization": "explicit user request to repartition the original fixed100 Mobile episodes",
    "source_archive_modified": False, "previous_all100_task_checkpoint_allowed": False,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda word:
        (_ for _ in ()).throw(ValueError("Nonfinite JSON metadata: " + word)))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def canonical_uuid(value):
    require(isinstance(value, str), "Episode identity must be a canonical UUID string")
    try:
        parsed = str(uuid.UUID(value))
    except (ValueError, AttributeError) as error:
        raise ValueError("Invalid episode lineage UUID") from error
    require(parsed == value, "Episode UUID spelling must be canonical")
    return parsed


def episode_identities(episodes):
    require(isinstance(episodes, list) and len(episodes) == EPISODES,
            "Use all and only the fixed100 source episodes")
    identities, seen_indices, seen_uuids = [], set(), set()
    for episode in episodes:
        index, length = episode.get("episode_index"), episode.get("length")
        identity = canonical_uuid(episode.get("human_demo_uuid"))
        require(type(index) is int and index not in seen_indices and identity not in seen_uuids,
                "Duplicated original episode index or human-demo UUID")
        require(type(length) is int and length > 59 and episode.get("task_id") == TASK_ID
                and episode.get("split") == "train" and episode.get("split_restriction") == "train_only",
                "Source task/episode/archive role identity differs")
        eligible = episode.get("eligible_current_rows")
        require(isinstance(eligible, list) and eligible
                and all(type(row) is int and row >= 30 and row + 29 < length for row in eligible)
                and len(eligible) == len(set(eligible)), "Incomplete/duplicated admitted history/future rows")
        sources = episode.get("lineage_group_sources")
        require(episode.get("lineage_complete") is True and isinstance(sources, list) and sources,
                "Whole-episode lineage evidence is required for the holdout split")
        lineage = []
        for source in sources:
            require(isinstance(source, dict) and isinstance(source.get("source_namespace"), str)
                    and source["source_namespace"], "Lineage requires an explicit source namespace")
            lineage.append((source["source_namespace"], canonical_uuid(source.get("human_demo_uuid"))))
        require(len(lineage) == len(set(lineage)) and identity in {value for _, value in lineage},
                "Lineage must include this UUID exactly once")
        identities.append({"episode_index": index, "human_demo_uuid": identity, "length": length,
            "eligible_count": len(eligible), "lineage_identities": [list(item) for item in sorted(lineage)]})
        seen_indices.add(index)
        seen_uuids.add(identity)
    require(seen_indices == set(range(EPISODES)), "Original fixed100 episode indices must remain0..99")
    require(sum(e["length"] for e in identities) == TOTAL_FRAMES
            and sum(e["eligible_count"] for e in identities) == TOTAL_WINDOWS,
            "Fixed100 source frame/window totals changed")
    return sorted(identities, key=lambda value: value["episode_index"])


def lineage_set(identities, selected):
    return {tuple(source) for item in identities if item["episode_index"] in selected
            for source in item["lineage_identities"]}


def build_split(episodes, source_metadata_sha256, seed=42):
    """UUID hash order, independent of file order and any model results.

    Hash input is exactly UTF-8 ``str(seed) + ':' + canonical UUID``. The first
    five identities are EVAL, the next five are the clean TRAIN probe. A lineage
    crossing the boundary is a split-evidence error, never a data-quality reject.
    """
    require(valid_sha(source_metadata_sha256), "A SHA256 binding for original episode metadata is required")
    require(type(seed) is int, "Split seed must be an integer, not bool")
    identities = episode_identities(episodes)
    ordered = sorted(identities, key=lambda item: (
        hashlib.sha256((str(seed) + ":" + item["human_demo_uuid"]).encode("utf-8")).hexdigest(),
        item["human_demo_uuid"]))
    eval_ids = {item["episode_index"] for item in ordered[:EVAL_EPISODES]}
    train_ids = set(range(EPISODES)) - eval_ids
    probe_ids = {item["episode_index"] for item in ordered[EVAL_EPISODES:EVAL_EPISODES + PROBE_EPISODES]}
    require(len(train_ids) == TRAIN_EPISODES and len(eval_ids) == EVAL_EPISODES
            and len(probe_ids) == PROBE_EPISODES and probe_ids <= train_ids, "Wrong95/5/TRAIN-probe cardinality")
    require(not lineage_set(identities, train_ids) & lineage_set(identities, eval_ids),
            "A complete source lineage crosses TRAIN/EVAL; refuse this split without rejecting source data")
    result = {"contract": CONTRACT, "task_id": TASK_ID, "seed": seed,
        "source_metadata_sha256": source_metadata_sha256, "source_episode_count": EPISODES,
        "source_total_frames": TOTAL_FRAMES, "source_total_windows": TOTAL_WINDOWS,
        "split_method": "ascending sha256(UTF8(str(seed)+':'+canonical_UUID)); first5 EVAL, next5 TRAIN-probe",
        "split_order_uuids": [item["human_demo_uuid"] for item in ordered],
        "episodes_identity": identities, "source_role_override": dict(SOURCE_ROLE_OVERRIDE)}
    for prefix, indices in (("train", train_ids), ("eval", eval_ids), ("train_probe", probe_ids)):
        result[prefix + "_episode_indices"] = sorted(indices)
        result[prefix + "_uuids"] = [item["human_demo_uuid"] for item in identities if item["episode_index"] in indices]
    return result


def validate_split(episodes, manifest, source_metadata_sha256):
    require(isinstance(manifest, dict) and type(manifest.get("seed")) is int, "Missing split manifest/seed")
    expected = build_split(episodes, source_metadata_sha256, seed=manifest["seed"])
    require(canonical(manifest) == canonical(expected), "Frozen95/5 split or original metadata identity drifted")
    return expected


def _reader_rows(dataset, episodes, release, legacy, shared):
    """Cross-check admitted rows against both real reader clocks and each episode."""
    history = dataset.raw_dataset.motion_history
    require(history is not None and len(history.pack_info) == 1, "Require one original100 source pack")
    info = history.pack_info[0]
    require(len(info) == 3 and Path(info[0]).resolve(strict=True) == (release / "dino").resolve(strict=True)
            and Path(info[1]).resolve(strict=True) == (release / "packed/train").resolve(strict=True),
            "Reader/DINO roots differ from the accepted original Mobile release")
    admitted, _ = legacy.map_accepted_rows(history, episodes, TOTAL_WINDOWS)
    candidates, _, checked_history = shared.eligible_dataset_rows(dataset, expected_episodes=EPISODES,
                                                                 source_frame_rule="ceil")
    require(checked_history is history and set(candidates) == set(range(EPISODES)),
            "Production complete-window episode identity differs")
    all_rows = [int(row) for episode in sorted(candidates) for row in candidates[episode]]
    require(len(all_rows) == len(set(all_rows)) == TOTAL_WINDOWS and set(all_rows) == set(admitted),
            "Production source clock/future rows differ from independent admissions")
    by_episode = {}
    for episode in episodes:
        index = episode["episode_index"]
        rows = [int(row) for row in candidates[index]]
        require(all(0 <= row < len(history.original_indices) and int(history.episode_ids[row]) == index for row in rows),
                "Filtered reader index was attributed to another episode")
        start, stop = history.source_episode_ranges[0][index]
        originals = {int(history.original_indices[row]) for row in rows}
        require(stop - start == episode["length"]
                and originals == {start + row for row in episode["eligible_current_rows"]},
                "Original admitted rows changed within an episode")
        by_episode[index] = sorted(rows, key=lambda row: int(history.original_indices[row]))
    return by_episode, history


def datasets(config, processor, plan, legacy, shared):
    """Return augmented TRAIN95, full clean EVAL5, summary, and clean-reader bundle."""
    import torch
    release = Path(plan["release"]).resolve(strict=True)
    metadata = release / "packed/train/meta/episodes.jsonl"
    source_sha = sha_file(metadata)
    episodes = [json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]
    split_path = Path(plan["split_file"])
    require(split_path.is_file() and not split_path.is_symlink() and valid_sha(plan.get("split_sha256"))
            and sha_file(split_path) == plan["split_sha256"], "Frozen split file SHA differs")
    split = validate_split(episodes, read_json(split_path), source_sha)
    train_raw = config.data(split="train", transform_kwargs={"vlm_processor": processor})
    clean = config.data(split="val", transform_kwargs={"vlm_processor": processor, "no_aug": True})
    train_by_episode, train_history = _reader_rows(train_raw, episodes, release, legacy, shared)
    clean_by_episode, clean_history = _reader_rows(clean, episodes, release, legacy, shared)
    train_ids, eval_ids, probe_ids = (split[key + "_episode_indices"] for key in ("train", "eval", "train_probe"))
    train_rows = [row for index in train_ids for row in train_by_episode[index]]
    val_rows = [row for index in eval_ids for row in clean_by_episode[index]]
    probe_rows = [row for index in probe_ids for row in clean_by_episode[index]]
    original_train = {int(train_history.original_indices[row]) for row in train_rows}
    original_eval = {int(clean_history.original_indices[row]) for row in val_rows}
    original_probe = {int(clean_history.original_indices[row]) for row in probe_rows}
    require(not original_train & original_eval and not original_probe & original_eval
            and original_probe <= original_train, "TRAIN/EVAL original-row leakage or invalid TRAIN probe")
    lengths = {episode["episode_index"]: episode["length"] for episode in episodes}
    train_frames, eval_frames = (sum(lengths[index] for index in ids) for ids in (train_ids, eval_ids))
    require(len(train_rows) + len(val_rows) == TOTAL_WINDOWS and train_frames + eval_frames == TOTAL_FRAMES,
            "TRAIN95 plus EVAL5 no longer covers the fixed original100 archive")
    summary = {"contract": CONTRACT, "source_metadata_sha256": source_sha, "split_sha256": plan["split_sha256"],
        "source_episode_count": EPISODES, "source_total_frames": TOTAL_FRAMES, "source_total_windows": TOTAL_WINDOWS,
        "train_episodes": TRAIN_EPISODES, "validation_episodes": EVAL_EPISODES, "train_probe_episodes": PROBE_EPISODES,
        "train_windows": len(train_rows), "validation_windows": len(val_rows), "train_probe_windows": len(probe_rows),
        "train_frames": train_frames, "validation_frames": eval_frames,
        "train_windows_per_episode": {str(index): len(train_by_episode[index]) for index in train_ids},
        "validation_windows_per_episode": {str(index): len(clean_by_episode[index]) for index in eval_ids},
        "train_probe_windows_per_episode": {str(index): len(clean_by_episode[index]) for index in probe_ids},
        "validation_rows": val_rows, "train_probe_rows": probe_rows,
        "train_eval_uuid_disjoint": True, "train_eval_lineage_disjoint": True,
        "train_eval_original_rows_disjoint": True, "train_probe_subset_of_train": True,
        "counts_partition_sum_verified": True, "source_role_override": split["source_role_override"],
        "validation_role": "five_fixed_original_Mobile_episodes_full_complete_windows",
        "train_probe_role": "five_fixed_TRAIN_episodes_full_complete_windows_with_same_clean_reader_as_EVAL"}
    for key in ("train_episode_indices", "eval_episode_indices", "train_probe_episode_indices",
                "train_uuids", "eval_uuids", "train_probe_uuids"):
        summary[key] = split[key]
    require(sha_file(metadata) == source_sha and sha_file(split_path) == plan["split_sha256"],
            "Original episode metadata or frozen split changed while building readers")
    bundle = {"clean_dataset": clean,
        "val_rows_by_episode": {index: clean_by_episode[index] for index in eval_ids},
        "train_probe_rows_by_episode": {index: clean_by_episode[index] for index in probe_ids}}
    return torch.utils.data.Subset(train_raw, train_rows), torch.utils.data.Subset(clean, val_rows), summary, bundle
