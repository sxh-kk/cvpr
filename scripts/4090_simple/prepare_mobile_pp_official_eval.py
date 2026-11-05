"""CPU-only deployment of an explicitly backed-up, user-stopped Mobile checkpoint.

This is a new contract for the 20261104 official-initialization100 run, not a
completed10000/20000 acceptance. Older entry points provide pure construction
helpers only. Every deployed VLM/action/Motion parameter comes from this selected
checkpoint; an old Motion file supplies non-tensor construction metadata only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def adjacent(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


export = adjacent("mobile_official_export", "export_bundle.py")
_previous = {name: sys.modules.get(name) for name in ("export_bundle", "prepare_task_snapshot")}
try:
    sys.modules["export_bundle"] = export
    snapshot = adjacent("mobile_official_snapshot", "prepare_task_snapshot.py")
    sys.modules["prepare_task_snapshot"] = snapshot
    generic = adjacent("mobile_official_construction", "prepare_mobile_pp_eval.py")
finally:
    for _name, _value in _previous.items():
        if _value is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _value

sha256, write_json = export.sha256, export.write_json
CONTRACT = "cvpr-mobile-pp-official100-stopped-local-deployment-v1"
BACKUP_CONTRACT = "cvpr-mobile-pp-official100-stopped-checkpoint-backup-v1"
PREPARATION_CONTRACT = "cvpr-mobile-official100-a100-bs4-fresh-v1"
TASK_ID = "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
VALIDATION_ROLE = "historical_handover_regression_only_not_mobile_validation"
EXPECTED_ROOT = "/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/mobile_pp_official100_training_20261104/run_001"
EXPECTED_PROJECT = (EXPECTED_ROOT + "/training/posttrain/"
    "mobile_pp_official100_joint38000_statekeep_bs4_8gpu.us.flow1000.cosine.lr1.0e-05.b32.gpus8.261104065646")
EXPECTED_RELEASE = "/gaozt-test1/datasets/SIMPLE_sonic_v1_1/pico_mobile_pp_20261101_official99_plus1_human100_01"
EXPECTED_READY_SHA = "672079bec26432011049f6a08939e17721814954dc3f99d63a51f46d88769426"
EXPECTED_CONFIG_SHA = "b8273ac16b704f6faecb34bf8a1f1735edd254fc163b5caff5f629ac2c3bd36f"
EXPECTED_RUNTIME_CONFIG_SHA = "25f8347470780954bb36d19f047208e4245268b0ea327cba6745f1b52a4bc241"
EXPECTED_PREPARATION_SHA = "80837e33e849e1c3ede6e9dfc34917b238c34a6426d27224fc71b03b8689dc56"
EXPECTED_STOP_SHA = "a59ff49947f565ad9015f8ab1401f4a993edb97e80e4ef7eb5fe9ddb29e6009a"
STOP_SOURCE = "/home/zxc/CVPR_Evaluation_runtime/mobile_pp_official100_training_20261104/STOPPED_BY_USER.json"
EXPECTED_MODEL_BYTES = 12066487572
SOURCE_SHA = "12a71e046e221b83f0bdb0efb50f9ba82442045aca0efabef3ae337f185320b7"
SOURCE_CHECKPOINT = ("/gaozt-test1/datasets/UnifoLM/.pipeline/training/posttrain/"
    "dual_source_unifolm_joint_motion100_8gpu_recovery.us.flow1000.cosine.lr1.0e-04.b256.gpus8.261020043942/"
    "checkpoints/ckpt_38000/model.safetensors")
STAT_SHA = "931f2b536b2fd47e8d0964ad966202ccbda852d05ea88074be321987200a6ac4"
MIGRATION_SHA = "95bbb9f4414c7a133e03f253c2219ed9683406e9569bd1084961a6c17f6d8e5a"
SAVE_STEPS = list(range(1000, 6001, 1000))
RELEASE_METADATA = ("packed/PACK_READY.json", "dino/manifest.json", "A100_PICO_DATASET_READY.json",
                    "dino/REMOTE_DINO_RELOCATION.json")
METADATA_SOURCES = {
    "prepared_run_config.json": EXPECTED_ROOT + "/run_config.json",
    "prepared_argv.txt": EXPECTED_ROOT + "/argv.txt",
    "preparation.json": EXPECTED_ROOT + "/preparation.json",
    "training_state.json": EXPECTED_ROOT + "/training_state.json",
    "STOPPED_BY_USER.json": STOP_SOURCE,
    "dataset_statistics.json": EXPECTED_RELEASE + "/packed/source_dataset_statistics.json",
    **{name: EXPECTED_PROJECT + "/" + name for name in (
        "run_config.json", "argv.txt", "task_initialization.json", "task_window_selection.json",
        "checkpoint_retention.json", "model_config_state_keep.json")},
    **{f"state_policy_migration_rank_{rank}.json": EXPECTED_PROJECT + f"/state_policy_migration_rank_{rank}.json"
       for rank in range(8)},
    **{"release_metadata/" + name: EXPECTED_RELEASE + "/" + name for name in RELEASE_METADATA},
}
METADATA = tuple(METADATA_SOURCES)
HELPERS = ("prepare_mobile_pp_eval.py", "prepare_task_snapshot.py", "export_bundle.py")
DEPLOYMENT_ARTIFACTS = ("run_config.json", "argv.txt", "dataset_statistics.json", "dino_training_contract.json",
                        "motion/joint_motion.ckpt", "motion/export_validation.json", "component_load_validation.json")


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value:
        (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def stat_identity(path):
    value = Path(path).stat()
    return {name: getattr(value, name) for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns")}


def validate_stat(value, size):
    require(isinstance(value, dict) and set(value) == {"st_dev", "st_ino", "st_size", "st_mtime_ns"}
            and all(type(v) is int and v >= 0 for v in value.values()) and value["st_size"] == size,
            "Incomplete checkpoint file identity")


def validate_runtime_metadata(source, config, preparation):
    require(sha256(source / "prepared_run_config.json") == preparation.get("config_sha256") == EXPECTED_CONFIG_SHA
            and sha256(source / "prepared_argv.txt") == preparation.get("argv_sha256"),
            "Prepared official100 configuration/argv differs")
    require(sha256(source / "run_config.json") == EXPECTED_RUNTIME_CONFIG_SHA,
            "Actual official100 runtime configuration identity differs")
    expected = read(source / "prepared_run_config.json")
    tracker_keys = ("entity", "group", "id", "name")
    require(expected["train"].get("hf_token") == ".hf_token"
            and all(expected["wandb"].get(key) is None for key in tracker_keys), "Unexpected prepared tracker/HF serialization")
    expected["train"]["hf_token"] = "PosixPath('.hf_token')"
    expected["wandb"].update({key: config["wandb"][key] for key in tracker_keys})
    require(same(config, expected), "Runtime config differs beyond actual tracker/HF serialization")
    tokens = shlex.split((source / "prepared_argv.txt").read_text())
    require(tokens and tokens == shlex.split((source / "argv.txt").read_text()), "Actual/prepared argv differ")
    require(same(read(source / "model_config_state_keep.json"), config["model"]),
            "Actual migrated model config differs from deployment semantics")
    return {"status": "passed", "prepared_config_sha256": EXPECTED_CONFIG_SHA,
            "runtime_config_sha256": EXPECTED_RUNTIME_CONFIG_SHA, "argv_tokens_exact": True}


def validate_training_config(config, preparation):
    require(preparation.get("contract") == PREPARATION_CONTRACT and preparation.get("task_id") == TASK_ID
            and preparation.get("release") == EXPECTED_RELEASE and preparation.get("expected_ready_sha256") == EXPECTED_READY_SHA
            and preparation.get("fresh_run") is True and preparation.get("initial_global_step") == 0
            and preparation.get("optimizer_scheduler_rng_restored") is False
            and preparation.get("source_step") == 38000 and preparation.get("source_checkpoint") == SOURCE_CHECKPOINT
            and preparation.get("source_checkpoint_sha256") == SOURCE_SHA,
            "Require this exact fresh joint38000 official100 run")
    require(tuple(preparation.get(key) for key in ("gpu_count", "per_gpu_batch", "gradient_accumulation_steps",
            "effective_global_batch", "train_episodes", "total_frames", "train_windows", "max_optimizer_steps"))
            == (8, 4, 1, 32, 100, 56423, 50523, 10000)
            and preparation.get("validation_role") == VALIDATION_ROLE and preparation.get("Mobile_validation_selected") is False,
            "Actual official100 TRAIN/BS4 provenance differs")
    model, train = config["model"], config["train"]
    require(tuple(model.get(key) for key in ("action_architecture", "num_blocks", "action_dim", "odim", "action_chunk_size",
            "state_drop_prob", "state_null_token", "state_feature_dropout")) == ("dual_source_v1", 6, 80, 45, 30, 0., True, 0.)
            and model.get("pretrained_joint_checkpoint") == SOURCE_CHECKPOINT
            and model.get("tune_vlm") is True and model.get("tune_motion") is True and model.get("rtc") is False,
            "State retention/six-block complete joint architecture differs")
    require(tuple(train.get(key) for key in ("train_batch_size", "gradient_accumulation_steps", "max_training_steps",
            "warmup_steps", "checkpointing_steps", "preserve_checkpoint_every")) == (4, 1, 10000, 1000, 1000, 1000)
            and train.get("resume_from_checkpoint") is None and train.get("skip_resumed_steps") is False,
            "Actual BS4/fresh optimizer/save cadence differs")
    repack, image, field = (config["data"]["transform"][key] for key in ("repack", "model", "field"))
    require(repack.get("action_keys") == ["action.body_token_v1_1", "action[:14]"]
            and repack.get("state_keys") == ["observation.state"] and repack.get("action_mask_key") == "action_mask"
            and (repack.get("pad_action_dim"), repack.get("pad_state_dim")) == (80, 45)
            and (image.get("shared_image_profile"), image.get("motion_image_policy")) == ("256x320", "vlm_color_clean_motion")
            and field.get("stat_path") == EXPECTED_RELEASE + "/packed/source_dataset_statistics.json",
            "Measured state/body64+hand14 or clean256x320 policy differs")


def validate_release(source, release, config, preparation):
    remote = source / "release_metadata"
    ready, pack = read(remote / "A100_PICO_DATASET_READY.json"), read(remote / "packed/PACK_READY.json")
    manifest, relocation = read(remote / "dino/manifest.json"), read(remote / "dino/REMOTE_DINO_RELOCATION.json")
    require(sha256(remote / "A100_PICO_DATASET_READY.json") == EXPECTED_READY_SHA
            and ready.get("contract") == "cvpr-pico-mobile-pp-production-verification-v1"
            and ready.get("task_id") == TASK_ID and ready.get("status") == "passed"
            and ready.get("remote_production_ready") is True and ready.get("DINO_relocation_pending") is False
            and ready.get("selection_policy") == "native_only" and ready.get("split_restriction") == "train_only"
            and ready.get("normalization_refitted") is False
            and (ready.get("train_episodes"), ready.get("total_frames"), ready.get("train_windows")) == (100, 56423, 50523),
            "Actual trained official100 production readiness differs")
    require((ready.get("release_root"), ready.get("pack_root"), ready.get("dino_root")) ==
            (EXPECTED_RELEASE, EXPECTED_RELEASE + "/packed", EXPECTED_RELEASE + "/dino"), "Remote READY paths differ")
    require(pack.get("contract") == "cvpr-pico-mobile-pp-lerobot-pack-v1" and pack.get("task_id") == TASK_ID
            and pack.get("selection_policy") == "native_only" and pack.get("normalization_refitted") is False
            and (pack.get("train_episodes"), pack.get("total_frames"), pack.get("train_windows")) == (100, 56423, 50523)
            and sha256(remote / "packed/PACK_READY.json") == ready.get("pack_ready_sha256") ==
                sha256(generic.regular(release, "packed/PACK_READY.json")), "Actual/local frozen pack differs")
    require(manifest.get("contract") == "unifolm-dino-native-frames-v1" and manifest.get("state") == "complete"
            and manifest.get("episode_indices") == list(range(100))
            and tuple(manifest.get("source", {}).get(key) for key in ("total_episodes", "total_frames", "fps")) == (100, 56423, 30)
            and sha256(remote / "dino/manifest.json") == ready.get("cache_manifest_sha256"),
            "Require the actual remote-rebound trained DINO manifest")
    require(relocation.get("contract") == "cvpr-pico-mobile-pp-dino-host-relocation-v1"
            and relocation.get("task_id") == TASK_ID and relocation.get("status") == "passed"
            and relocation.get("data_and_features_byte_identical") is True and relocation.get("PACK_READY_modified") is False
            and sha256(remote / "dino/REMOTE_DINO_RELOCATION.json") == ready.get("remote_relocation_binding", {}).get("sha256")
            and relocation.get("manifest_sha256") == ready["cache_manifest_sha256"]
            and relocation.get("pack_ready_sha256") == ready["pack_ready_sha256"]
            and relocation.get("original_manifest_sha256") == sha256(generic.regular(release, "dino/manifest.json")),
            "Actual remote DINO relocation binding differs")
    statistics = generic.regular(source, "dataset_statistics.json")
    require(sha256(statistics) == STAT_SHA == sha256(generic.regular(release, "packed/source_dataset_statistics.json"))
            and preparation.get("bindings_sha256", {}).get(EXPECTED_RELEASE + "/packed/source_dataset_statistics.json") == STAT_SHA,
            "Inherited original bounds/statistics differ")
    checkpoint = pack.get("checkpoint_contract", {})
    require(checkpoint.get("source_step") == 38000 and checkpoint.get("source_checkpoint_sha256") == SOURCE_SHA
            and checkpoint.get("bounds_refitted") is False and checkpoint.get("statistics", {}).get("sha256") == STAT_SHA,
            "Pack inherited normalization source differs")
    field = config["data"]["transform"]["field"]
    require(all(checkpoint.get("bounds_sha256", {}).get(name) == generic.digest(field[name])
                for name in ("action_min", "action_max", "state_min", "state_max")), "Actual model normalization bounds differ")
    return {"status": "passed", "ready_sha256": EXPECTED_READY_SHA,
            "training_manifest_sha256": ready["cache_manifest_sha256"], "normalization_refitted": False}


def validate_source(source, release, step, expected_model_sha256, *, verify_weight=True):
    require(type(step) is int and step in SAVE_STEPS, "Use an explicitly saved stopped-run checkpoint (1000..6000)")
    require(valid_sha(expected_model_sha256), "Supply the actual checkpoint SHA explicitly")
    source, release = (Path(path).resolve(strict=True) for path in (source, release))
    receipt = read(generic.regular(source, "backup_receipt.json"))
    require(receipt.get("contract") == BACKUP_CONTRACT and receipt.get("status") == "passed"
            and receipt.get("task_id") == TASK_ID and type(receipt.get("step")) is int and receipt["step"] == step
            and receipt.get("training_completed") is False and receipt.get("optimizer_copied") is False,
            "Require an independent model-only stopped-checkpoint backup")
    for name, original in METADATA_SOURCES.items():
        item = receipt.get("metadata", {}).get(name, {})
        path = generic.regular(source, name)
        require(item.get("source_path") == original and type(item.get("bytes")) is int
                and item["bytes"] == path.stat().st_size and item.get("sha256") == sha256(path),
                "Source metadata path/bytes/SHA binding differs: " + name)
    require(sha256(source / "STOPPED_BY_USER.json") == EXPECTED_STOP_SHA, "Original user-stop evidence differs")
    stopped = read(source / "STOPPED_BY_USER.json").get("receipt", {})
    require(stopped.get("status") == "stopped_by_user" and stopped.get("remaining_scoped_processes") == []
            and stopped.get("latest_confirmed_checkpoint_step") == 6000 and stopped.get("all_saved_models_retained") == SAVE_STEPS
            and stopped.get("raw_launcher_terminal_state", {}).get("run_directory") == EXPECTED_PROJECT,
            "The stopped run/available checkpoint identity differs")
    model = receipt.get("model", {})
    require(model.get("source_path") == EXPECTED_PROJECT + f"/checkpoints/ckpt_{step}/model.safetensors"
            and type(model.get("bytes")) is int and model["bytes"] == EXPECTED_MODEL_BYTES
            and all(model.get(key) == expected_model_sha256 for key in
                    ("source_sha256_before", "source_sha256_after", "local_sha256")), "Checkpoint source/before-after/local SHA differs")
    for name in ("source_stat_before", "source_stat_after", "local_stat"):
        validate_stat(model.get(name), model["bytes"])
    require(model["source_stat_before"] == model["source_stat_after"], "Remote checkpoint changed during backup")
    weights = generic.regular(source, "model.safetensors")
    require(stat_identity(weights) == model["local_stat"] and (not verify_weight or sha256(weights) == expected_model_sha256),
            "Actual local checkpoint bytes/file identity differs")
    require(sha256(source / "preparation.json") == EXPECTED_PREPARATION_SHA,
            "Original prepared source/code authority differs")
    preparation, config = read(source / "preparation.json"), read(source / "run_config.json")
    validate_training_config(config, preparation)
    runtime = validate_runtime_metadata(source, config, preparation)
    state, retention = read(source / "training_state.json"), read(source / "checkpoint_retention.json")
    require(state.get("state") == "failed" and type(state.get("returncode")) is int and state["returncode"] != 0
            and state.get("run_directory") == EXPECTED_PROJECT and state.get("task_id") == TASK_ID
            and state.get("expected_ready_sha256") == EXPECTED_READY_SHA
            and state.get("fresh_run") is True and state.get("initial_global_step") == 0
            and state.get("optimizer_scheduler_rng_restored") is False, "Require fresh terminal user-stopped training state")
    require(type(retention.get("saved_update")) is int and retention["saved_update"] == 6000
            and retention.get("task_id") == TASK_ID and retention.get("all_models_preserved") is True
            and retention.get("all_saved_model_steps") == SAVE_STEPS and retention.get("full_recovery_steps") == [5000, 6000],
            "Latest completed-save marker does not prove the selected checkpoint")
    remote = receipt.get("remote_snapshot", {})
    require(remote.get("scoped_training_processes_absent") is True
            and remote.get("training_state_sha256") == sha256(source / "training_state.json")
            and remote.get("retention_sha256") == sha256(source / "checkpoint_retention.json")
            and remote.get("stop_receipt_sha256") == EXPECTED_STOP_SHA, "Missing fresh remote stop/save snapshot binding")
    observed = datetime.fromisoformat(remote.get("observed_utc", "").replace("Z", "+00:00"))
    stopped_at = datetime.strptime(stopped["stopped_at_beijing"], "%Y-%m-%d %H:%M:%S %z")
    finished = datetime.fromisoformat(state.get("finished_utc", "").replace("Z", "+00:00"))
    require(observed.tzinfo is not None and finished.tzinfo is not None and observed >= max(stopped_at, finished),
            "Remote snapshot predates the user stop/terminal state")
    identity = receipt.get("run_identity", {})
    require(identity.get("run_directory") == EXPECTED_PROJECT
            and identity.get("prepared_root") == EXPECTED_ROOT
            and identity.get("run_config_sha256") == EXPECTED_RUNTIME_CONFIG_SHA
            and identity.get("preparation_sha256") == sha256(source / "preparation.json"), "Actual run/config/preparation binding differs")
    initialization = read(source / "task_initialization.json")
    require(initialization.get("source_checkpoint") == SOURCE_CHECKPOINT and initialization.get("source_step") == 38000
            and initialization.get("strict_joint_loader") is True and initialization.get("fresh_optimizer_scheduler_rng") is True
            and initialization.get("global_batch") == 32 and initialization.get("backend", {}).get("backend") == "fused"
            and initialization["backend"].get("foreach") is False and initialization["backend"].get("initialized_parameter_states") == 0,
            "Fresh complete-joint initialization differs")
    windows = read(source / "task_window_selection.json")
    require(windows.get("train_windows") == 50523 and windows.get("train_episode_indices") == list(range(100))
            and windows.get("train_task_id") == TASK_ID and windows.get("validation_role") == VALIDATION_ROLE
            and windows.get("Mobile_validation_selected") is False and windows.get("validation_windows") == 40
            and windows.get("validation_rows") == preparation.get("expected_validation_rows"), "Actual TRAIN/regression-only VAL differs")
    for rank in range(8):
        migration = read(source / f"state_policy_migration_rank_{rank}.json")
        require(migration.get("status") == "passed" and migration.get("source_probability") == .5
                and migration.get("target_probability") == 0. and migration.get("helper_sha256") == MIGRATION_SHA
                and migration.get("changed_state_tensors") == ["action_header._architecture_signature"]
                and migration.get("unchanged_learned_parameters") is True and migration.get("parameter_objects_preserved") is True
                and migration.get("all_other_state_tensors_exactly_unchanged") is True
                and migration.get("strict_target_head_roundtrip", {}).get("status") == "passed"
                and migration.get("source", {}).get("source_sha256") == SOURCE_SHA,
                "Actual state-retention migration evidence differs: rank " + str(rank))
    release_validation = validate_release(source, release, config, preparation)
    return source, config, preparation, receipt, {"runtime": runtime, "release": release_validation}


def prepare(source, resources, output, repo, release, step, expected_model_sha256):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Set CUDA_VISIBLE_DEVICES='' for CPU preparation")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    source, config, preparation, receipt, validations = validate_source(source, release, step, expected_model_sha256)
    before = stat_identity(source / "model.safetensors")
    source_metadata = {name: sha256(source / name) for name in (*METADATA, "backup_receipt.json")}
    resources, repo, release = (Path(path).resolve(strict=True) for path in (resources, repo, release))
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink()
            and not any(output.resolve().is_relative_to(path) or path.is_relative_to(output.resolve())
                        for path in (source, resources, repo, release)), "Use a fresh independent deployment directory")
    code = generic.validate_frozen_model_code(repo, preparation)
    construction = generic.validate_construction_metadata(resources)
    sys.path[:0] = [str(repo / "VLMExpert/src"), str(repo), str(repo / "pipeline/src"), str(repo / "VLMExpert")]
    base = resources / "joint20k"
    for path in (base / "qwen", base / "motion/joint_motion.ckpt", resources / "dinov2-with-registers-large",
                 resources / "simple_local_usd_v1"):
        require(path.exists(), "Missing accepted construction resource: " + str(path))
    motion_metadata = generic.validate_motion_template_metadata(base / "motion/joint_motion.ckpt",
                                                                construction["motion_template_metadata_sha256"])
    manifest = read(source / "release_metadata/dino/manifest.json")
    reference = read(base / "dino_training_contract.json")
    transition = generic.recorded_encoder_transition(reference["encoder"], manifest["encoder"])
    contract = generic.bind_training_motion_image_policy(reference, config)
    contract["encoder"] = export.canonical_visual_identity({"encoder": deepcopy(manifest["encoder"])})["encoder"]
    contract.update(online_checkpoint=str(resources / "dinov2-with-registers-large"),
        source_manifest_sha256=validations["release"]["training_manifest_sha256"],
        source_dataset="20261104 official-initialization Mobile100 train_only; historical Handover VAL is regression-only")
    replacements = {"model.model_name_or_path": str(base / "qwen"),
        "model.motion_checkpoint": str(output / "motion/joint_motion.ckpt"), "model.pretrained_joint_checkpoint": None,
        "data.root_dir": str(resources / "simple_local_usd_v1"), "data.dino_cache_root": None,
        "data.transform.field.stat_path": str(output / "dataset_statistics.json"),
        "train.output_dir": str(output / "unused_training"), "train.resume_from_checkpoint": None,
        "train.deepspeed_config": str(repo / "VLMExpert/scripts/deepspeed/zero3.json")}
    deployed, argv = snapshot.remap_config(config, (source / "argv.txt").read_text(), replacements)
    output.mkdir(parents=True)
    try:
        provenance = output / "provenance"
        for name in (*METADATA, "backup_receipt.json"):
            target = provenance / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
        for name, value in (("runtime_metadata_validation", validations["runtime"]), ("release_validation", validations["release"]),
                ("model_code_validation", code), ("construction_metadata_validation", construction),
                ("motion_template_metadata_validation", motion_metadata)):
            write_json(provenance / (name + ".json"), value)
        shutil.copy2(resources / "manifest.json", provenance / "construction_resources_manifest.json")
        write_json(output / "run_config.json", deployed)
        (output / "argv.txt").write_text(argv)
        shutil.copy2(source / "dataset_statistics.json", output / "dataset_statistics.json")
        target = output / f"checkpoints/ckpt_{step}/model.safetensors"
        export.link_or_copy(source / "model.safetensors", target)
        require(sha256(target) == expected_model_sha256, "Deployed checkpoint bytes changed")
        write_json(output / "dino_training_contract.json", contract)
        from psi.deploy.dual_source_runtime import load_dino_contract
        load_dino_contract(output / "dino_training_contract.json")
        structure = generic.strict_component_check(target, deployed, base / "qwen")
        write_json(output / "component_load_validation.json", structure)
        motion = export.export_motion(base / "motion/joint_motion.ckpt", target, output / "motion/joint_motion.ckpt",
                                     f"38000_then_official_mobile100_task_{step}", reference["encoder"])
        require(motion.get("strict_meta_load") == "passed" and motion.get("exact_serialization_comparison") == "passed",
                "Selected checkpoint Motion export failed strict/exact CPU serialization")
        motion.update(parameter_source="selected_20261104_official_mobile100_stopped_joint_checkpoint",
            parameter_source_sha256=expected_model_sha256, previous_motion_parameters_used=False,
            construction_metadata_only_source=str(base / "motion/joint_motion.ckpt"),
            actual_encoder_runtime_transition=transition)
        write_json(output / "motion/export_validation.json", motion)
        snapshot.validate_config(deployed, output / "argv.txt", repo)
        validate_source(source, release, step, expected_model_sha256, verify_weight=False)
        require(stat_identity(source / "model.safetensors") == before, "Source checkpoint changed during CPU preparation")
        require(all(sha256(source / name) == sha256(provenance / name) == expected
                    for name, expected in source_metadata.items()), "Source metadata changed during CPU preparation")
        import torch
        require(not torch.cuda.is_initialized(), "CPU preparation initialized CUDA")
        report = {"contract": CONTRACT, "task_id": TASK_ID,
            "status": "local_mobile_pp_official100_stopped_checkpoint_prepared", "step": step, "task_step": step,
            "step_semantics": "official_mobile100_task_optimizer_updates", "source_unifolm_step": 38000,
            "initial_task_step": 0, "source_global_batch": 32, "train_episodes": 100, "total_frames": 56423,
            "train_windows": 50523, "train_split": "train_only", "validation_role": VALIDATION_ROLE,
            "Mobile_validation_selected": False, "source_prepared_config_sha256": EXPECTED_CONFIG_SHA,
            "source_runtime_config_sha256": EXPECTED_RUNTIME_CONFIG_SHA, "source_project": EXPECTED_PROJECT,
            "joint_sha256": expected_model_sha256, "run_dir": str(output), "source_snapshot": str(source),
            "source_backup_receipt_sha256": sha256(source / "backup_receipt.json"),
            "source_metadata_sha256": source_metadata,
            "training_manifest_sha256": validations["release"]["training_manifest_sha256"], "state_drop_prob": 0.,
            "num_blocks": 6, "full_saved_components_strict_CPU_load": structure, "component_load_validation": structure,
            "strict_motion_load": motion["strict_meta_load"], "motion_weights": motion["weights"],
            "previous_motion_parameters_used": False, "statistics_sha256": STAT_SHA, "normalization_refitted": False,
            "motion_image_policy": "vlm_color_clean_motion", "actual_DINO_runtime_transition": transition,
            "deployment_path_changes": replacements, "training_completed": False, "stopped_by_user": True,
            "stop_receipt_sha256": EXPECTED_STOP_SHA, "checkpoint_save_completed": True, "optimizer_rng_restored": False,
            "GPU_used": False, "CUDA_initialized": False, "policy_forward_performed": False,
            "gpu_validation": "not_run", "physical_task_success": "not_measured", "source_model_bytes_verified": True,
            "source_model_identity_unchanged_during_prepare": True, "verifier_sha256": sha256(Path(__file__)),
            "deployment_artifacts": {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
                                     for name in DEPLOYMENT_ARTIFACTS},
            "shared_preparation_helpers_sha256": {name: sha256(HERE / name) for name in HELPERS}}
        write_json(output / "LOCAL_RUN_READY.json", report)
        return report
    except BaseException as error:
        write_json(output / "PREPARATION_FAILED.json", {"contract": CONTRACT, "status": "failed_incomplete", "error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "resources", "output", "release"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--step", type=int, default=6000)
    parser.add_argument("--expected-model-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.resources, args.output, args.repo, args.release,
                             args.step, args.expected_model_sha256), ensure_ascii=False))


if __name__ == "__main__":
    main()
