"""CPU deployment of a selected checkpoint from the completed Mobile95/EVAL5 run.

This entry point has its own training, backup and deployment contracts. It reads
only accepted archive metadata, requires final20k acceptance and a published
model-only backup, and exports Motion parameters from the selected new joint
checkpoint. It does not start a service, run a model forward, or use a GPU.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def adjacent_module(name, filename):
    """Load a specific neighboring source file, never a search-path namesake."""
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


# The older Mobile entry supplies pure construction/strict-load functions only.
# Its task constants and validate_source/prepare are never altered or called.
export = adjacent_module("mobile95_export_bundle", "export_bundle.py")
_previous = {name: sys.modules.get(name) for name in ("export_bundle", "prepare_task_snapshot")}
try:
    sys.modules["export_bundle"] = export
    snapshot = adjacent_module("mobile95_task_snapshot", "prepare_task_snapshot.py")
    sys.modules["prepare_task_snapshot"] = snapshot
    generic = adjacent_module("mobile95_generic_construction", "prepare_mobile_pp_eval.py")
finally:
    for _name, _value in _previous.items():
        if _value is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _value
splitter = adjacent_module("mobile95_frozen_split", "mobile_pp_holdout_data.py")

sha256, write_json = export.sha256, export.write_json
CONTRACT = "cvpr-mobile-pp95-eval5-local-deployment-v1"
BACKUP_CONTRACT = "cvpr-mobile-pp95-eval5-local-checkpoint-backup-v1"
FINAL_CONTRACT = "cvpr-mobile-pp95-eval5-final-training-artifact-acceptance-v1"
PREPARATION_CONTRACT = "cvpr-mobile-pp95-eval5-a100-fresh20k-v1"
DIAGNOSTIC_CONTRACT = "cvpr-mobile-pp95-5-overfit-metrics-v1"
TASK_ID = "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
VALIDATION_ROLE = "heldout_mobile5_validation_not_independent_test"
EXPECTED_ROOT = "/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/pico_mobile_pp95_eval5_20k_20261028/run_001"
EXPECTED_PROJECT = (EXPECTED_ROOT + "/training/posttrain/"
    "mobile_pp95_eval5_joint38000_statekeep_8gpu.us.flow1000.cosine.lr1.0e-05.b64.gpus8.261028073258")
EXPECTED_CONFIG_SHA = "8464d90a6a7fa4c3b303f7132ad5b6f9738dc171f43a9fb8d6980a0f39012468"
EXPECTED_SPLIT_SHA = "0d87f6ae5634d07e312cfef51fa9e23e37ba70500b9ad69c4a577011457f8ac9"
EXPECTED_MODEL_BYTES = 12066487572
SOURCE_SHA = "12a71e046e221b83f0bdb0efb50f9ba82442045aca0efabef3ae337f185320b7"
SOURCE_CHECKPOINT = ("/gaozt-test1/datasets/UnifoLM/.pipeline/training/posttrain/"
    "dual_source_unifolm_joint_motion100_8gpu_recovery.us.flow1000.cosine.lr1.0e-04.b256.gpus8.261020043942/"
    "checkpoints/ckpt_38000/model.safetensors")
STAT_SHA = "931f2b536b2fd47e8d0964ad966202ccbda852d05ea88074be321987200a6ac4"
SAVE_STEPS = list(range(4000, 20001, 500))
DIAGNOSTIC_STEPS = list(range(0, 20001, 1000))
SELECTION_CANDIDATES = list(range(4000, 20001, 1000))
SELECTION_METRIC = "roles.val.metrics.full_fm_loss_episode_macro"
TRACKER_KEYS = ("entity", "group", "id", "name")
ROOT_METADATA = ("run_config.json", "argv.txt", "dataset_statistics.json", "task_initialization.json",
    "task_window_selection.json", "prepared_run_config.json", "prepared_argv.txt", "preparation.json",
    "training_state.json", "checkpoint_retention.json", "gpu_check_update20000.json", "episode_split.json",
    "FINAL_TRAINING_ACCEPTANCE_001.json")
DIAGNOSTIC_METADATA = tuple(f"mobile_diagnostics/update_{step:05d}.json" for step in DIAGNOSTIC_STEPS)
RELEASE_METADATA = ("packed/PACK_READY.json", "dino/manifest.json", "A100_PICO_DATASET_READY.json",
                    "dino/REMOTE_DINO_RELOCATION.json")
METADATA = ROOT_METADATA + DIAGNOSTIC_METADATA + tuple("release_metadata/" + name for name in RELEASE_METADATA)
HELPERS = ("prepare_mobile_pp_eval.py", "prepare_task_snapshot.py", "export_bundle.py", "mobile_pp_holdout_data.py")


def require(value, message):
    if not value:
        raise ValueError(message)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda word:
        (_ for _ in ()).throw(ValueError("Nonfinite JSON value: " + word)))


def same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def regular(root, relative):
    return generic.regular(root, relative)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def bound_file(source, final, key, filename, remote_path):
    record = final.get(key, {})
    require(record.get("path") == remote_path and valid_sha(record.get("sha256"))
        and sha256(regular(source, filename)) == record["sha256"], "Final acceptance file differs: " + key)


def validate_runtime_metadata(source, config, preparation, final):
    require(sha256(source / "prepared_run_config.json") == preparation.get("config_sha256") == EXPECTED_CONFIG_SHA
        and sha256(source / "prepared_argv.txt") == preparation.get("argv_sha256"), "Prepared95/5 config/argv SHA differs")
    expected = read(source / "prepared_run_config.json")
    require(expected["train"].get("hf_token") == ".hf_token"
        and set(expected["wandb"]) == set(TRACKER_KEYS)
        and all(expected["wandb"][key] is None for key in TRACKER_KEYS), "Unexpected prepared tracker/HF metadata")
    tracker = final.get("runtime_wandb", {})
    require(set(tracker) == set(TRACKER_KEYS) and same(config["wandb"], tracker), "Actual runtime tracker is not final-bound")
    expected["train"]["hf_token"] = "PosixPath('.hf_token')"
    expected["wandb"].update(tracker)
    require(same(expected, config), "Runtime configuration differs beyond recorded tracker/HF serialization")
    tokens = shlex.split((source / "prepared_argv.txt").read_text())
    require(tokens and tokens == shlex.split((source / "argv.txt").read_text()), "Runtime/prepared argv semantic tokens differ")
    return {"status": "passed", "prepared_config_sha256": EXPECTED_CONFIG_SHA,
        "runtime_config_sha256": sha256(source / "run_config.json"), "all_other_values_exact": True,
        "argv_tokens_exact": True, "allowed_runtime_fields": ["train.hf_token", *("wandb." + key for key in TRACKER_KEYS)]}


def validate_training_config(config, preparation, statistics):
    model, train = config["model"], config["train"]
    require(preparation.get("contract") == PREPARATION_CONTRACT and preparation.get("task_id") == TASK_ID
        and preparation.get("fresh_run") is True and preparation.get("initial_global_step") == 0
        and preparation.get("optimizer_scheduler_rng_restored") is False
        and preparation.get("source_step") == 38000 and preparation.get("source_checkpoint_sha256") == SOURCE_SHA
        and preparation.get("source_checkpoint") == SOURCE_CHECKPOINT
        and model.get("pretrained_joint_checkpoint") == SOURCE_CHECKPOINT,
        "Require fresh joint38000/task0; previous Mobile100 weights or restored state are forbidden")
    require((preparation.get("gpu_count"), preparation.get("per_gpu_batch"), preparation.get("gradient_accumulation_steps"),
             preparation.get("effective_global_batch"), preparation.get("max_optimizer_steps"), preparation.get("warmup_steps"))
            == (8, 8, 1, 64, 20000, 1000)
        and (preparation.get("train_episodes"), preparation.get("validation_episodes"), preparation.get("source_total_frames"),
             preparation.get("source_total_windows")) == (95, 5, 61806, 55906)
        and preparation.get("validation_role") == VALIDATION_ROLE
        and preparation.get("split_file") == EXPECTED_ROOT + "/episode_split.json"
        and preparation.get("split_sha256") == EXPECTED_SPLIT_SHA
        and preparation.get("diagnostic_steps") == DIAGNOSTIC_STEPS,
        "Require the bound eight-rank95/5 full20k experiment")
    require((model.get("action_architecture"), model.get("num_blocks"), model.get("action_dim"), model.get("odim"),
             model.get("action_chunk_size"), model.get("state_drop_prob"), model.get("state_null_token"))
            == ("dual_source_v1", 6, 80, 45, 30, 0., True)
        and model.get("tune_motion") is True and model.get("tune_vlm") is True and model.get("rtc") is False,
        "Require the complete six-block state-keep dual-source model")
    require((train.get("train_batch_size"), train.get("gradient_accumulation_steps"), train.get("max_training_steps"),
             train.get("warmup_steps"), train.get("checkpointing_steps"), train.get("preserve_checkpoint_every"),
             train.get("validation_steps")) == (8, 1, 20000, 1000, 20000, 500, 0)
        and train.get("resume_from_checkpoint") is None and train.get("skip_resumed_steps") is False
        and train.get("early_checkpoint_steps") == SAVE_STEPS[:-1]
        and preparation.get("checkpoint_policy", {}).get("model_steps") == SAVE_STEPS
        and preparation["checkpoint_policy"].get("full_recovery_states_to_keep") == 2,
        "Fresh optimization/save/diagnostic policy differs")
    repack, image = (config["data"]["transform"][key] for key in ("repack", "model"))
    require(repack.get("action_keys") == ["action.body_token_v1_1", "action[:14]"]
        and repack.get("state_keys") == ["observation.state"] and repack.get("action_mask_key") == "action_mask"
        and (repack.get("pad_action_dim"), repack.get("pad_state_dim")) == (80, 45)
        and (image.get("shared_image_profile"), image.get("motion_image_policy")) == ("256x320", "vlm_color_clean_motion"),
        "Measured state/action mask or clean DINO256x320 policy differs")
    require(sha256(statistics) == STAT_SHA,
            "Inherited normalization statistics changed")


def validate_windows(source, release):
    path = source / "episode_split.json"
    require(sha256(path) == EXPECTED_SPLIT_SHA, "Actual95/5 frozen split SHA differs")
    metadata = regular(release, "packed/train/meta/episodes.jsonl")
    episodes = [json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]
    split = splitter.validate_split(episodes, read(path), sha256(metadata))
    windows = read(source / "task_window_selection.json")
    require(windows.get("contract") == splitter.CONTRACT and windows.get("split_sha256") == EXPECTED_SPLIT_SHA
        and windows.get("source_metadata_sha256") == split["source_metadata_sha256"]
        and (windows.get("train_episodes"), windows.get("validation_episodes"), windows.get("train_probe_episodes")) == (95, 5, 5)
        and (windows.get("train_windows"), windows.get("validation_windows"), windows.get("train_probe_windows")) == (53342, 2564, 2435)
        and windows.get("source_total_frames") == 61806 and windows.get("source_total_windows") == 55906
        and windows.get("source_episode_count") == 100
        and same(windows.get("source_role_override"), split["source_role_override"]), "Wrong original95/5/probe complete-window summary")
    for key in ("train_episode_indices", "eval_episode_indices", "train_probe_episode_indices", "train_uuids", "eval_uuids", "train_probe_uuids"):
        require(same(windows.get(key), split[key]), "Training-window split membership differs: " + key)
    expected_train_counts = {str(item["episode_index"]): item["eligible_count"] for item in split["episodes_identity"]
                             if item["episode_index"] in split["train_episode_indices"]}
    require(same(windows.get("train_windows_per_episode"), expected_train_counts), "TRAIN95 episode-window coverage differs")
    require(all(windows.get(key) is True for key in ("train_eval_uuid_disjoint", "train_eval_lineage_disjoint",
        "train_eval_original_rows_disjoint", "train_probe_subset_of_train", "counts_partition_sum_verified")),
        "Missing original episode/UUID/lineage/row nonleakage proof")
    by_id = {episode["episode_index"]: episode for episode in episodes}
    role_rows = {}
    for role, prefix, rows_key in (("val", "eval", "validation"), ("train_probe", "train_probe", "train_probe")):
        indices = split[prefix + "_episode_indices"]
        rows, cursor, per_episode = windows[rows_key + "_rows"], 0, {}
        require(isinstance(rows, list) and all(type(row) is int and row >= 0 for row in rows)
            and len(rows) == len(set(rows)) == windows[rows_key + "_windows"], "Invalid diagnostic original filtered rows")
        for index in indices:
            count = len(by_id[index]["eligible_current_rows"])
            require(windows[rows_key + "_windows_per_episode"].get(str(index)) == count,
                    "Incomplete source episode windows in diagnostic view")
            selected = rows[cursor:cursor + count]
            require(selected == sorted(selected), "Diagnostic episode rows changed order")
            per_episode[str(index)] = selected
            cursor += count
        require(cursor == len(rows), "Diagnostic view did not retain all episode rows")
        role_rows[role] = per_episode
    require(not set(windows["validation_rows"]) & set(windows["train_probe_rows"]), "Clean diagnostic rows leaked across roles")
    require(windows.get("train_frames") == sum(by_id[i]["length"] for i in split["train_episode_indices"])
        and windows.get("validation_frames") == sum(by_id[i]["length"] for i in split["eval_episode_indices"])
        and windows["train_frames"] + windows["validation_frames"] == 61806, "95/5 frame partition differs")
    return split, windows, role_rows


def validate_diagnostics(source, final, role_rows):
    records = final.get("diagnostics", [])
    require(isinstance(records, list) and all(type(record.get("completed_update")) is int for record in records)
        and [record.get("completed_update") for record in records] == DIAGNOSTIC_STEPS,
            "Final acceptance must bind all21 completed-update diagnostics")
    scores = {}
    for record, relative in zip(records, DIAGNOSTIC_METADATA):
        require(record.get("path") == EXPECTED_PROJECT + "/" + relative
            and valid_sha(record.get("sha256")) and sha256(source / relative) == record["sha256"], "Diagnostic source/SHA differs")
        value = read(source / relative)
        require(value.get("contract") == DIAGNOSTIC_CONTRACT and type(value.get("completed_update")) is int
            and value["completed_update"] == record["completed_update"]
            and value.get("split_sha256") == EXPECTED_SPLIT_SHA and value.get("world_size") == 8
            and value.get("validation_role") == VALIDATION_ROLE, "Wrong checkpoint/split diagnostic provenance")
        for role, per_episode in role_rows.items():
            actual = value["roles"][role]
            require(set(actual["per_episode"]) == set(per_episode)
                and actual["counts"].get("episodes") == 5
                and actual["counts"].get("full_fm_windows") == sum(map(len, per_episode.values()))
                and actual["counts"].get("sampled_l1_windows") == 40, "Diagnostic role/complete episode coverage differs")
            losses = []
            for index, rows in per_episode.items():
                item = actual["per_episode"][index]
                require(item.get("full_fm_windows") == len(rows) and item.get("full_rows_sha256") == generic.digest(rows)
                    and item.get("sampled_rows") == [rows[i * (len(rows) - 1) // 7] for i in range(8)],
                    "Diagnostic original-row identity differs")
                loss = item.get("full_fm_loss")
                require(finite_number(loss) and loss >= 0, "Nonfinite/negative episode FM loss")
                losses.append(loss)
            macro = actual["metrics"].get("full_fm_loss_episode_macro")
            require(finite_number(macro) and math.isclose(macro, sum(losses) / 5, rel_tol=1e-12, abs_tol=1e-12)
                and same(value["metrics"].get(role + "_full_fm_loss_episode_macro"), macro), "Diagnostic macro/flat loss differs")
            require(all(finite_number(x) for x in actual["metrics"].values()), "Nonfinite role metrics")
        require(all(finite_number(x) for x in value["metrics"].values()), "Nonfinite diagnostic metrics")
        scores[record["completed_update"]] = value["roles"]["val"]["metrics"]["full_fm_loss_episode_macro"]
    require(final.get("selection_metric") == SELECTION_METRIC
        and final.get("selection_candidate_steps") == SELECTION_CANDIDATES
        and final.get("tie_break") == "lowest_step", "Explicit scored/saved17-model selection policy differs")
    best = min(SELECTION_CANDIDATES, key=lambda step: (scores[step], step))
    expected = {"early": 4000, "best_validation": best, "final": 20000}
    require(same(final.get("selections"), expected), "Best validation checkpoint was not selected from the actual17 scored saved models")
    return {"diagnostics": 21, "scored_saved_candidates": 17, "selection_metric": SELECTION_METRIC,
        "tie_break": "lowest_step", "selections": expected, "candidate_val_macro": {str(step): scores[step] for step in SELECTION_CANDIDATES}}


def validate_source(source, release, step, expected_final_acceptance_sha256, *, verify_weight=True):
    source, release = (Path(p).resolve(strict=True) for p in (source, release))
    require(type(step) is int and step in SAVE_STEPS and valid_sha(expected_final_acceptance_sha256), "Require a saved step and explicit final acceptance SHA")
    # Never inspect weights before the atomically published model-only receipt.
    receipt = read(regular(source, "backup_receipt.json"))
    require(receipt.get("contract") == BACKUP_CONTRACT and receipt.get("task_id") == TASK_ID
        and type(receipt.get("step")) is int and receipt["step"] == step and receipt.get("optimizer_copied") is False,
        "Require a completed95/5 model-only backup, not an old Mobile100/task10000 receipt")
    require(set(METADATA) <= set(receipt.get("metadata", {})), "Missing final95/5 source metadata bindings")
    for name in METADATA:
        require(sha256(regular(source, name)) == receipt["metadata"][name], "Source metadata changed: " + name)
    final_path = source / "FINAL_TRAINING_ACCEPTANCE_001.json"
    require(sha256(final_path) == expected_final_acceptance_sha256, "Actual final acceptance SHA differs")
    final = read(final_path)
    require(final.get("contract") == FINAL_CONTRACT and final.get("status") == "passed" and final.get("task_id") == TASK_ID
        and final.get("root") == EXPECTED_ROOT and final.get("project") == EXPECTED_PROJECT
        and final.get("completed_step") == 20000, "Require the completed real95/5 full20k final acceptance")
    for key, filename, remote in (("configuration", "prepared_run_config.json", EXPECTED_ROOT + "/run_config.json"),
        ("runtime_configuration", "run_config.json", EXPECTED_PROJECT + "/run_config.json"),
        ("split", "episode_split.json", EXPECTED_ROOT + "/episode_split.json"),
        ("preparation_file", "preparation.json", EXPECTED_ROOT + "/preparation.json"),
        ("state_file", "training_state.json", EXPECTED_ROOT + "/training_state.json"),
        ("retention_file", "checkpoint_retention.json", EXPECTED_PROJECT + "/checkpoint_retention.json"),
        ("health_file", "gpu_check_update20000.json", EXPECTED_PROJECT + "/gpu_check_update20000.json")):
        bound_file(source, final, key, filename, remote)
    require(final["configuration"]["sha256"] == EXPECTED_CONFIG_SHA and final["split"]["sha256"] == EXPECTED_SPLIT_SHA,
            "Wrong actual run configuration/split authority")
    config, preparation = read(source / "run_config.json"), read(source / "preparation.json")
    runtime = validate_runtime_metadata(source, config, preparation, final)
    validate_training_config(config, preparation, source / "dataset_statistics.json")
    state = read(source / "training_state.json")
    require(state.get("state") == "complete" and type(state.get("returncode")) is int and state["returncode"] == 0
        and state.get("run_directory") == EXPECTED_PROJECT and state.get("task_id") == TASK_ID
        and state.get("config_sha256") == EXPECTED_CONFIG_SHA and state.get("split_sha256") == EXPECTED_SPLIT_SHA
        and state.get("max_optimizer_steps") == 20000, "Training has not completed this bound20k run")
    initialization = read(source / "task_initialization.json")
    require(initialization.get("source_checkpoint") == SOURCE_CHECKPOINT and initialization.get("source_step") == 38000
        and initialization.get("strict_joint_loader") is True and initialization.get("fresh_optimizer_scheduler_rng") is True
        and initialization.get("global_batch") == 64 and initialization.get("backend", {}).get("backend") == "fused"
        and initialization["backend"].get("foreach") is False and initialization["backend"].get("initialized_parameter_states") == 0,
        "Strict fresh joint38000 initialization proof differs")
    retention = read(source / "checkpoint_retention.json")
    require(retention.get("saved_update") == 20000 and retention.get("all_models_preserved") is True
        and retention.get("all_saved_model_steps") == SAVE_STEPS and retention.get("full_recovery_steps") == [19500, 20000]
        and final.get("full_recovery_steps") == [19500, 20000]
        and same(receipt.get("completed_retention_marker"), retention), "Completed33-model/two-state retention differs")
    health = read(source / "gpu_check_update20000.json")
    require(health.get("status") == "passed" and health.get("completed_updates") == 20000
        and (health.get("gpu_count"), health.get("per_gpu_batch"), health.get("accumulation")) == (8, 8, 1)
        and health.get("all_rank_probe_values_equal") is True and health.get("actual_fused_flags_passed") is True
        and health.get("actual_actions_shapes_all_ranks") == [[8, 30, 80]] * 8,
        "Final20000 all-rank production health has not passed")
    split, windows, role_rows = validate_windows(source, release)
    selection = validate_diagnostics(source, final, role_rows)
    require(step in selection["selections"].values(), "Prepare only final-selected early/best-validation/final checkpoints")
    models = final.get("models", [])
    require(isinstance(models, list) and [item.get("step") for item in models] == SAVE_STEPS, "All33 saved model identities are required")
    for item in models:
        require(type(item["step"]) is int and item.get("path") == f"{EXPECTED_PROJECT}/checkpoints/ckpt_{item['step']}/model.safetensors"
            and item.get("size_bytes") == EXPECTED_MODEL_BYTES and valid_sha(item.get("sha256")), "Saved model authority differs")
    model = next(item for item in models if item["step"] == step)
    identity, copied = receipt.get("run_identity", {}), receipt.get("model", {})
    require(identity.get("run_directory") == EXPECTED_PROJECT and identity.get("run_config_sha256") == sha256(source / "run_config.json")
        and identity.get("preparation_sha256") == sha256(source / "preparation.json") and identity.get("split_sha256") == EXPECTED_SPLIT_SHA
        and copied.get("source_path") == model["path"] and copied.get("bytes") == model["size_bytes"]
        and all(copied.get(key) == model["sha256"] for key in ("source_sha256_before", "source_sha256_after", "local_sha256")),
        "Checkpoint before/after/local SHA or actual95/5 source identity differs")
    # Reuse only the old pure metadata validator for the unchanged original100.
    generic.validate_release(source, release, config, preparation)
    weights = regular(source, "model.safetensors")
    require(weights.stat().st_size == model["size_bytes"] and (not verify_weight or sha256(weights) == model["sha256"]),
            "Actual selected local checkpoint bytes differ")
    return source, config, preparation, receipt, {"final": final, "model": model, "selection": selection,
        "split": split, "windows": windows, "runtime_metadata": runtime}


def prepare(source, resources, output, repo, release, step, expected_final_acceptance_sha256):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU preparation requires CUDA_VISIBLE_DEVICES='' explicitly")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    source, config, preparation, receipt, authority = validate_source(source, release, step, expected_final_acceptance_sha256)
    original_stat = (source / "model.safetensors").stat()
    resources, repo, release = (Path(p).resolve(strict=True) for p in (resources, repo, release))
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink()
        and not any(output.resolve().is_relative_to(p) or p.is_relative_to(output.resolve()) for p in (repo, source, resources, release)),
        "Use a fresh independent deployment outside source/repository/resources/release")
    code = generic.validate_frozen_model_code(repo, preparation)
    construction = generic.validate_construction_metadata(resources)
    sys.path[:0] = [str(repo / "VLMExpert/src"), str(repo), str(repo / "pipeline/src"), str(repo / "VLMExpert")]
    base = resources / "joint20k"
    for path in (base / "qwen", base / "motion/joint_motion.ckpt", resources / "dinov2-with-registers-large", resources / "simple_local_usd_v1"):
        require(path.exists(), "Missing accepted construction resource: " + str(path))
    motion_metadata = generic.validate_motion_template_metadata(base / "motion/joint_motion.ckpt", construction["motion_template_metadata_sha256"])
    manifest = read(source / "release_metadata/dino/manifest.json")
    reference = read(base / "dino_training_contract.json")
    transition = generic.recorded_encoder_transition(reference["encoder"], manifest["encoder"])
    contract = generic.bind_training_motion_image_policy(reference, config)
    contract["encoder"] = export.canonical_visual_identity({"encoder": deepcopy(manifest["encoder"])})["encoder"]
    contract.update(online_checkpoint=str(resources / "dinov2-with-registers-large"),
        source_manifest_sha256=sha256(source / "release_metadata/dino/manifest.json"), mobile_split_sha256=EXPECTED_SPLIT_SHA,
        source_dataset="original Mobile100 clean cache, read-only episode TRAIN95/EVAL5 views; five TRAIN probes use the EVAL no_aug reader")
    replacements = {"model.model_name_or_path": str(base / "qwen"), "model.motion_checkpoint": str(output / "motion/joint_motion.ckpt"),
        "model.pretrained_joint_checkpoint": None, "data.root_dir": str(resources / "simple_local_usd_v1"), "data.dino_cache_root": None,
        "data.transform.field.stat_path": str(output / "dataset_statistics.json"), "train.output_dir": str(output / "unused_training"),
        "train.resume_from_checkpoint": None, "train.deepspeed_config": str(repo / "VLMExpert/scripts/deepspeed/zero3.json")}
    deployed, argv = snapshot.remap_config(config, (source / "argv.txt").read_text(), replacements)
    output.mkdir(parents=True)
    try:
        provenance = output / "provenance"
        for name in (*METADATA, "backup_receipt.json"):
            copied = provenance / name
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, copied)
        for name, value in (("runtime_metadata_validation", authority["runtime_metadata"]), ("model_code_validation", code),
            ("construction_metadata_validation", construction), ("motion_template_metadata_validation", motion_metadata),
            ("checkpoint_selection_validation", authority["selection"])):
            write_json(provenance / (name + ".json"), value)
        shutil.copy2(resources / "manifest.json", provenance / "construction_resources_manifest.json")
        write_json(output / "run_config.json", deployed)
        (output / "argv.txt").write_text(argv)
        shutil.copy2(source / "dataset_statistics.json", output / "dataset_statistics.json")
        target = output / f"checkpoints/ckpt_{step}/model.safetensors"
        export.link_or_copy(source / "model.safetensors", target)
        if (target.stat().st_dev, target.stat().st_ino) != (original_stat.st_dev, original_stat.st_ino):
            require(sha256(target) == authority["model"]["sha256"], "Copied checkpoint bytes changed")
        write_json(output / "dino_training_contract.json", contract)
        from psi.deploy.dual_source_runtime import load_dino_contract
        load_dino_contract(output / "dino_training_contract.json")
        structure = generic.strict_component_check(target, deployed, base / "qwen")
        write_json(output / "component_load_validation.json", structure)
        motion = export.export_motion(base / "motion/joint_motion.ckpt", target, output / "motion/joint_motion.ckpt",
            f"38000_then_mobile_pp95_eval5_task_{step}", reference["encoder"])
        require(motion.get("strict_meta_load") == "passed" and motion.get("exact_serialization_comparison") == "passed",
                "Selected new Motion export failed strict/exact CPU serialization")
        motion.update(parameter_source="selected_new_mobile_pp95_eval5_complete_joint_checkpoint",
            parameter_source_sha256=authority["model"]["sha256"], mobile_split_sha256=EXPECTED_SPLIT_SHA,
            previous_motion_parameters_used=False, construction_metadata_only_source=str(base / "motion/joint_motion.ckpt"),
            mobile95_actual_encoder_runtime_transition=transition)
        write_json(output / "motion/export_validation.json", motion)
        snapshot.validate_config(deployed, output / "argv.txt", repo)
        validate_source(source, release, step, expected_final_acceptance_sha256, verify_weight=False)
        current = (source / "model.safetensors").stat()
        require((original_stat.st_dev, original_stat.st_ino, original_stat.st_size, original_stat.st_mtime_ns)
            == (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns), "Source model changed during CPU preparation")
        import torch
        require(not torch.cuda.is_initialized(), "CPU preparation initialized CUDA")
        windows = authority["windows"]
        report = {"contract": CONTRACT, "task_id": TASK_ID, "status": "local_mobile_pp95_eval5_run_prepared",
            "step": step, "task_step": step, "source_unifolm_step": 38000, "initial_task_step": 0,
            "step_semantics": "mobile_pp95_eval5_task_optimizer_updates", "source_global_batch": 64,
            "source_project": EXPECTED_PROJECT, "source_prepared_root": EXPECTED_ROOT, "fresh_run": True,
            "source_prepared_config_sha256": EXPECTED_CONFIG_SHA, "mobile_split_sha256": EXPECTED_SPLIT_SHA,
            "final_acceptance_sha256": expected_final_acceptance_sha256, "checkpoint_selection": authority["selection"],
            "selected_roles": [key for key, value in authority["selection"]["selections"].items() if value == step],
            "train_episodes": 95, "validation_episodes": 5, "train_probe_episodes": 5,
            "train_windows": windows["train_windows"], "validation_windows": windows["validation_windows"],
            "train_probe_windows": windows["train_probe_windows"], "source_archive_episodes": 100,
            "source_role_override": authority["split"]["source_role_override"], "validation_role": VALIDATION_ROLE,
            **{key: authority["split"][key] for key in ("train_episode_indices", "eval_episode_indices",
                "train_probe_episode_indices", "train_uuids", "eval_uuids", "train_probe_uuids")},
            "source_snapshot": str(source), "run_dir": str(output), "joint_sha256": authority["model"]["sha256"],
            "state_drop_prob": 0., "num_blocks": 6, "source_backup_receipt_sha256": sha256(source / "backup_receipt.json"),
            "full_saved_components_strict_CPU_load": structure, "strict_motion_load": motion["strict_meta_load"],
            "motion_weights": motion["weights"], "previous_motion_parameters_used": False,
            "normalization_refitted": False, "statistics_sha256": STAT_SHA, "actual_DINO_runtime_transition": transition,
            "motion_image_policy": "vlm_color_clean_motion", "deployment_path_changes": replacements,
            "optimizer_rng_restored": False, "GPU_used": False, "CUDA_initialized": False, "policy_forward_performed": False,
            "gpu_validation": "not_run", "physical_task_success": "not_measured", "formal_benchmark_result": False,
            "checkpoint_save_completed": True, "training_completed": True,
            "source_metadata_sha256": {name: sha256(provenance / name) for name in (*METADATA, "backup_receipt.json")},
            "verifier_sha256": sha256(Path(__file__)), "shared_preparation_helpers_sha256": {name: sha256(HERE / name) for name in HELPERS},
            "source_model_bytes_verified": True, "source_model_identity_unchanged_during_prepare": True}
        write_json(output / "LOCAL_RUN_READY.json", report)
        return report
    except BaseException as error:
        write_json(output / "PREPARATION_FAILED.json", {"contract": CONTRACT, "status": "failed_incomplete", "error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "resources", "output", "release"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--step", type=int, choices=SAVE_STEPS, required=True)
    parser.add_argument("--expected-final-acceptance-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.resources, args.output, args.repo, args.release, args.step,
        args.expected_final_acceptance_sha256), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
