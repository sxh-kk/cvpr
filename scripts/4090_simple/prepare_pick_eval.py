"""CPU-only selected Pick task/checkpoint deployment from a completed10000 run.

The backup receipt binds one remote transmission SHA, its stable source stat,
the persisted local SHA, and every metadata source path. No Mobile/stopped-run
contract is accepted. Original Motion resources supply construction metadata
only; the selected joint provides every deployed VLM/action/Motion parameter.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import shutil
import re
import sys

import prepare_mobile_pp_official_eval as helpers

generic, export, snapshot = helpers.generic, helpers.export, helpers.snapshot
sha256, write_json = export.sha256, export.write_json
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONTRACT = "cvpr-pick-local-deployment-v1"
READY_STATUS = "local_pick_checkpoint_prepared"
BACKUP_CONTRACT = "cvpr-pick-checkpoint-backup-v1"
PREPARATION_CONTRACT = "cvpr-pico-pick-a800-bs4-fresh-v1"
EXPECTED_MODEL_BYTES = 12066487572
SOURCE_SHA = "12a71e046e221b83f0bdb0efb50f9ba82442045aca0efabef3ae337f185320b7"
STAT_SHA = "931f2b536b2fd47e8d0964ad966202ccbda852d05ea88074be321987200a6ac4"
MIGRATION_SHA = "95bbb9f4414c7a133e03f253c2219ed9683406e9569bd1084961a6c17f6d8e5a"
VALIDATION_ROLE = "historical_handover_regression_only_not_pick_validation"
SAVE_STEPS = (1000, 2000, 3000, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 8000, 9000, 10000)
SERVER_BASE = "/gaozt-test1/datasets/SIMPLE_sonic_v1_1"
QUEUE_ROOT = SERVER_BASE + "/.pipeline/pick299_training_20261110_001"
QUEUE_PLAN = Path('/home/zxc/CVPR_Evaluation_runtime/pick299_training_20261110/QUEUE_PLAN.json')
QUEUE_PLAN_SHA = '1ff8819e7b9ef6cdc80e00d42412b0c7cacf09224ec54f255a9e0858e1430a3f'
PROFILES = {
    'bend_pick': {'task_id': 'cvpr/G1SonicBendPickPico-v0',
        'source_task_id': 'simple/G1WholebodyBendPickMP-v0', 'counts': (100, 15722, 9822),
        'release_name': 'pico_bend_pick_20261110_human100_01',
        'pack_sha256': '3c8134b35124e5e50b0fd6c90110fa303ea98874489076f19efcf4ff33e553f3',
        'project_name': 'bend_pick_joint38000_statekeep_bs4_8gpu.us.flow1000.cosine.lr1.0e-05.b32.gpus8.261110144035'},
    'xmove_pick': {'task_id': 'cvpr/G1SonicXMovePickPico-v0',
        'source_task_id': 'simple/G1WholebodyXMovePickTeleop-v0', 'counts': (99, 19925, 14084),
        'release_name': 'pico_xmove_pick_20261110_human99_01',
        'pack_sha256': '704258475f482451b66108f4dadb33526a49b0b358dbd5fb899ebcb150bf3591',
        'project_name': 'xmove_pick_joint38000_statekeep_bs4_8gpu.us.flow1000.cosine.lr1.0e-05.b32.gpus8.261110223312'},
    'xmove_bend_pick': {'task_id': 'cvpr/G1SonicXMoveBendPickPico-v0',
        'source_task_id': 'simple/G1WholebodyXMoveBendPickTeleop-v0', 'counts': (100, 34569, 28669),
        'release_name': 'pico_xmove_bend_pick_20261110_human100_01',
        'pack_sha256': '742fdb53ac99fb85d8a8e5c9044e0f18757a59b070885ad4ae983530cbde215c',
        'project_name': None},
}
RELEASE_METADATA = ("packed/PACK_READY.json", "dino/manifest.json", "A100_PICO_DATASET_READY.json",
                    "dino/REMOTE_DINO_RELOCATION.json")
DEPLOYMENT_ARTIFACTS = ("run_config.json", "argv.txt", "dataset_statistics.json", "dino_training_contract.json",
    "motion/joint_motion.ckpt", "motion/export_validation.json", "component_load_validation.json")
HELPERS = ("prepare_mobile_pp_official_eval.py", "prepare_mobile_pp_eval.py", "prepare_task_snapshot.py", "export_bundle.py")
require, read, same, valid_sha = helpers.require, helpers.read, helpers.same, helpers.valid_sha
stat_identity, validate_stat = helpers.stat_identity, helpers.validate_stat


def local_release(path):
    """Accept the local release root or its read-only packed child."""
    path = Path(path).resolve(strict=True)
    return path.parent if (path / "PACK_READY.json").is_file() else path


class PickDeployment:
    """One explicit task/step authority; no process-global task routing."""

    def __init__(self, profile, step, source=None, queue_plan=QUEUE_PLAN):
        require(profile in PROFILES and type(step) is int and step in SAVE_STEPS,
            'Select an explicit Pick profile and one of its thirteen retained checkpoints')
        self.PROFILE, self.EXPECTED_STEP = profile, step
        item = PROFILES[profile]
        self.TASK_ID, self.SOURCE_TASK_ID = item['task_id'], item['source_task_id']
        self.COUNTS, self.PACK_SHA = tuple(item['counts']), item['pack_sha256']
        self.EXPECTED_ROOT = QUEUE_ROOT + '/runs/' + profile
        self.EXPECTED_RELEASE = SERVER_BASE + '/' + item['release_name']
        self.queue_plan_path = Path(queue_plan).resolve(strict=True)
        require(sha256(self.queue_plan_path) == QUEUE_PLAN_SHA, 'Independent authorized queue plan SHA differs')
        plan = read(self.queue_plan_path)
        require(plan.get('contract') == 'cvpr-pico-pick-three-model-a800-serial-queue-v1' and
            plan.get('training_authorized') is True and plan.get('remote_root') == QUEUE_ROOT and
            plan.get('checkpoint_steps') == list(SAVE_STEPS) and plan.get('per_task_optimizer_steps') == 10000 and
            plan.get('source_checkpoint_sha256') == SOURCE_SHA and plan.get('gpu_count') == 8,
            'Authorized training queue recipe differs')
        matches = [row for row in plan.get('tasks', []) if row.get('profile') == profile]
        require(len(matches) == 1 and matches[0].get('task_id') == self.TASK_ID and
            matches[0].get('release') == self.EXPECTED_RELEASE and
            tuple(matches[0].get(key) for key in ('episodes', 'frames', 'windows')) == self.COUNTS and
            matches[0].get('pack_ready_sha256') == self.PACK_SHA,
            'Selected profile is not the independently authorized dataset/run')
        if item['project_name'] is not None:
            self.EXPECTED_PROJECT = self.EXPECTED_ROOT + '/training/posttrain/' + item['project_name']
        else:
            require(source is not None, 'Completed XMoveBend source metadata is required to bind its generated run name')
            state = read(generic.regular(Path(source).resolve(strict=True), 'training_state.json'))
            project = state.get('run_directory')
            prefix = self.EXPECTED_ROOT + '/training/posttrain/' + profile + '_joint38000_statekeep_bs4_8gpu.us.flow1000.cosine.lr1.0e-05.b32.gpus8.'
            require(isinstance(project, str) and re.fullmatch(re.escape(prefix)+r'[0-9]{12}', project) is not None and
                state.get('task_id') == self.TASK_ID and state.get('state') == 'complete' and state.get('returncode') == 0,
                'Generated completed XMoveBend run escapes its authorized profile/recipe namespace')
            self.EXPECTED_PROJECT = project
        self.METADATA_SOURCES = {
            **{name: self.EXPECTED_ROOT+'/'+remote for name, remote in (
                ('prepared_run_config.json', 'run_config.json'), ('prepared_argv.txt', 'argv.txt'),
                ('preparation.json', 'preparation.json'), ('training_state.json', 'training_state.json'))},
            **{name: self.EXPECTED_PROJECT+'/'+name for name in ('dataset_statistics.json', 'run_config.json', 'argv.txt',
                'task_initialization.json', 'task_window_selection.json', 'checkpoint_retention.json',
                'model_config_state_keep.json', 'gpu_check_update10000.json')},
            **{f'state_policy_migration_rank_{rank}.json': self.EXPECTED_PROJECT+f'/state_policy_migration_rank_{rank}.json'
                for rank in range(8)},
            **{'release_metadata/'+name: self.EXPECTED_RELEASE+'/'+name for name in RELEASE_METADATA}}
        self.METADATA = tuple(self.METADATA_SOURCES)

    def recorded_encoder_transition(self, reference, actual):
        result = generic.recorded_encoder_transition(reference, actual)
        for value in result["versions"].values():
            value[self.PROFILE+"_training"] = value.pop("mobile100_training")
        result["actual_online_contract"] = "dino_training_contract.json retains the actual selected Pick encoder runtime identity"
        return result


    def validate_runtime_metadata(self, source, config, preparation):
        require(sha256(source / "prepared_run_config.json") == preparation.get("config_sha256")
                and sha256(source / "prepared_argv.txt") == preparation.get("argv_sha256"),
                "Prepared Pick configuration/argv binding differs")
        expected = read(source / "prepared_run_config.json")
        # Production serializes pathlib and tracker fields after argument parsing.
        require(expected["train"].get("hf_token") == ".hf_token"
                and all(expected["wandb"].get(k) is None for k in ("entity", "group", "id", "name")),
                "Unexpected prepared tracker/HF serialization")
        expected["train"]["hf_token"] = "PosixPath('.hf_token')"
        expected["wandb"].update({key: config["wandb"][key] for key in ("entity", "group", "id", "name")})
        require(same(config, expected), "Runtime configuration differs beyond Path/tracker serialization")
        require(shlex.split((source / "prepared_argv.txt").read_text()) == shlex.split((source / "argv.txt").read_text())
                and bool(shlex.split((source / "argv.txt").read_text()))
                and same(read(source / "model_config_state_keep.json"), config["model"]),
                "Actual argv or migrated model config differs")


    def validate_training(self, source, config, preparation):
        require(preparation.get("contract") == PREPARATION_CONTRACT and preparation.get("task_id") == self.TASK_ID
                and preparation.get("profile") == self.PROFILE and preparation.get("release") == self.EXPECTED_RELEASE
                and preparation.get("fresh_run") is True and preparation.get("initial_global_step") == 0
                and preparation.get("optimizer_scheduler_rng_restored") is False
                and preparation.get("source_step") == 38000 and preparation.get("source_checkpoint_sha256") == SOURCE_SHA
                and tuple(preparation.get(k) for k in ("gpu_count", "per_gpu_batch", "gradient_accumulation_steps",
                    "effective_global_batch", "train_episodes", "total_frames", "train_windows", "max_optimizer_steps"))
                    == (8, 4, 1, 32, *self.COUNTS, 10000)
                and preparation.get("validation_role") == VALIDATION_ROLE and preparation.get("Pick_validation_selected") is False
                and preparation.get("checkpoint_policy", {}).get("model_steps") == list(SAVE_STEPS),
                "Require this completed selected Pick fresh joint38000 BS4 training recipe")
        self.validate_runtime_metadata(source, config, preparation)
        model, train = config["model"], config["train"]
        require(tuple(model.get(k) for k in ("action_architecture", "num_blocks", "action_dim", "odim", "action_chunk_size",
                    "state_drop_prob", "state_null_token", "state_feature_dropout")) == ("dual_source_v1", 6, 80, 45, 30, 0., True, 0.)
                and model.get("pretrained_joint_checkpoint") == preparation.get("source_checkpoint")
                and model.get("tune_motion") is True and model.get("tune_vlm") is True and model.get("rtc") is False,
                "Require complete state-retention Pick joint; training RTC must remain false")
        require(tuple(train.get(k) for k in ("train_batch_size", "gradient_accumulation_steps", "max_training_steps", "warmup_steps",
                    "checkpointing_steps", "preserve_checkpoint_every")) == (4, 1, 10000, 1000, 1000, 500)
                and train.get("early_checkpoint_steps") == [4500, 5500, 6500]
                and train.get("resume_from_checkpoint") is None and train.get("skip_resumed_steps") is False,
                "Pick optimization/save recipe differs")
        repack, image = (config["data"]["transform"][k] for k in ("repack", "model"))
        require(repack.get("action_keys") == ["action.body_token_v1_1", "action[:14]"]
                and repack.get("state_keys") == ["observation.state"] and repack.get("action_mask_key") == "action_mask"
                and (repack.get("pad_action_dim"), repack.get("pad_state_dim")) == (80, 45)
                and (image.get("shared_image_profile"), image.get("motion_image_policy")) == ("256x320", "vlm_color_clean_motion"),
                "Measured token/state or training clean-history visual policy differs")
        state, initialization, windows, retention, health = (read(source / name) for name in ("training_state.json",
            "task_initialization.json", "task_window_selection.json", "checkpoint_retention.json", "gpu_check_update10000.json"))
        require(state.get("state") == "complete" and state.get("returncode") == 0 and state.get("run_directory") == self.EXPECTED_PROJECT
                and state.get("task_id") == self.TASK_ID and state.get("max_optimizer_steps") == 10000
                and state.get("expected_ready_sha256") == preparation.get("expected_ready_sha256"),
                "Pick training has not completed10000 updates")
        require(retention.get("all_saved_model_steps") == list(SAVE_STEPS) and retention.get("all_models_preserved") is True
                and retention.get("task_id") == self.TASK_ID and retention.get("saved_update") == 10000,
                "Completed selected Pick run does not retain all13 model saves")
        require(health.get("status") == "passed" and health.get("completed_updates") == 10000
                and (health.get("gpu_count"), health.get("per_gpu_batch"), health.get("accumulation")) == (8, 4, 1)
                and health.get("actual_fused_flags_passed") is True and health.get("all_rank_probe_values_equal") is True
                and health.get("actual_actions_shapes_all_ranks") == [[4, 30, 80]] * 8,
                "Final eight-rank Pick health evidence differs")
        require(initialization.get("source_checkpoint") == preparation["source_checkpoint"] and initialization.get("source_step") == 38000
                and initialization.get("strict_joint_loader") is True and initialization.get("fresh_optimizer_scheduler_rng") is True
                and initialization.get("global_batch") == 32 and initialization.get("backend", {}).get("backend") == "fused"
                and initialization.get("backend", {}).get("foreach") is False
                and initialization.get("backend", {}).get("initialized_parameter_states") == 0,
                "Strict complete joint/fresh optimizer initialization evidence differs")
        require(windows.get("train_windows") == self.COUNTS[2] and windows.get("train_episode_indices") == list(range(self.COUNTS[0]))
                and windows.get("train_task_id") == self.TASK_ID and windows.get("validation_role") == VALIDATION_ROLE
                and windows.get("Pick_validation_selected") is False and windows.get("validation_windows") == 40
                and windows.get("validation_rows") == preparation.get("expected_validation_rows"),
                "Pick TRAIN/history-only Handover regression window provenance differs")
        for rank in range(8):
            migration = read(source / f"state_policy_migration_rank_{rank}.json")
            require(migration.get("status") == "passed" and migration.get("helper_sha256") == MIGRATION_SHA
                    and migration.get("target_probability") == 0. and migration.get("unchanged_learned_parameters") is True
                    and migration.get("all_other_state_tensors_exactly_unchanged") is True
                    and migration.get("strict_target_head_roundtrip", {}).get("status") == "passed"
                    and migration.get("source", {}).get("source_sha256") == SOURCE_SHA
                    and migration.get("source", {}).get("all_serialized_tensors_exactly_equal_before_migration") is True,
                    "State-retention migration proof differs for rank" + str(rank))


    def validate_release(self, source, release, config, preparation):
        remote = source / "release_metadata"
        ready, pack, manifest, relocation = (read(remote / name) for name in
            ("A100_PICO_DATASET_READY.json", "packed/PACK_READY.json", "dino/manifest.json", "dino/REMOTE_DINO_RELOCATION.json"))
        require(sha256(remote / "A100_PICO_DATASET_READY.json") == preparation.get("expected_ready_sha256")
                and ready.get("contract") == "cvpr-pico-pick-production-verification-v1" and ready.get("task_id") == self.TASK_ID
                and ready.get("status") == "passed" and ready.get("remote_production_ready") is True
                and ready.get("DINO_relocation_pending") is False and ready.get("normalization_refitted") is False
                and ready.get("selection_policy") == "native_only" and ready.get("split_restriction") == "train_only"
                and tuple(ready.get(k) for k in ("train_episodes", "total_frames", "train_windows")) == self.COUNTS
                and (ready.get("release_root"), ready.get("pack_root"), ready.get("dino_root")) ==
                    (self.EXPECTED_RELEASE, self.EXPECTED_RELEASE + "/packed", self.EXPECTED_RELEASE + "/dino"),
                "Actual trained Pick A100 release identity differs")
        require(pack.get("contract") == "cvpr-pico-pick-lerobot-pack-v1" and pack.get("task_id") == self.TASK_ID
                and pack.get("selection_policy") == "native_only" and pack.get("normalization_refitted") is False
                and tuple(pack.get(k) for k in ("train_episodes", "total_frames", "train_windows")) == self.COUNTS
                and sha256(remote / "packed/PACK_READY.json") == ready.get("pack_ready_sha256") == self.PACK_SHA
                    == sha256(generic.regular(release, "packed/PACK_READY.json")),
                "Actual/local frozen Pick packed dataset differs")
        require(manifest.get("contract") == "unifolm-dino-native-frames-v1" and manifest.get("state") == "complete"
                and manifest.get("episode_indices") == list(range(self.COUNTS[0]))
                and tuple(manifest.get("source", {}).get(k) for k in ("total_episodes", "total_frames", "fps")) == (self.COUNTS[0], self.COUNTS[1], 30)
                and sha256(remote / "dino/manifest.json") == ready.get("cache_manifest_sha256"),
                "Actual trained clean Pick DINO metadata differs")
        require(relocation.get("contract") == "cvpr-pico-pick-dino-host-relocation-v1" and relocation.get("task_id") == self.TASK_ID
                and relocation.get("status") == "passed" and relocation.get("data_and_features_byte_identical") is True
                and relocation.get("PACK_READY_modified") is False
                and sha256(remote / "dino/REMOTE_DINO_RELOCATION.json") == ready.get("remote_relocation_binding", {}).get("sha256")
                and relocation.get("manifest_sha256") == ready["cache_manifest_sha256"]
                and relocation.get("pack_ready_sha256") == self.PACK_SHA
                and relocation.get("original_manifest_sha256") == sha256(generic.regular(release, "dino/manifest.json"))
                and same(read(release / "dino/manifest.json")["encoder"], manifest["encoder"]),
                "Pick DINO host relocation or encoder identity differs")
        field, checkpoint = config["data"]["transform"]["field"], pack.get("checkpoint_contract", {})
        require(field.get("stat_path") == self.EXPECTED_RELEASE + "/packed/source_dataset_statistics.json"
                and checkpoint.get("source_step") == 38000 and checkpoint.get("source_checkpoint_sha256") == SOURCE_SHA
                and checkpoint.get("bounds_refitted") is False
                and sha256(source / "dataset_statistics.json") == STAT_SHA == checkpoint.get("statistics", {}).get("sha256")
                    == sha256(generic.regular(release, "packed/source_dataset_statistics.json"))
                    == preparation.get("bindings_sha256", {}).get(field["stat_path"])
                and set(checkpoint.get("bounds_sha256", {})) == {"action_min", "action_max", "state_min", "state_max"}
                and all(generic.digest(field[name]) == value for name, value in checkpoint["bounds_sha256"].items()),
                "Inherited joint38000 normalization or bounds changed")
        return {"status": "passed", "training_manifest_sha256": ready["cache_manifest_sha256"],
                "metadata_only": True, "frame_or_feature_audit_repeated": False}


    def validate_source(self, source, release, step, expected_model_sha256, *, verify_weight=True):
        require(type(step) is int and step == self.EXPECTED_STEP and valid_sha(expected_model_sha256),
                "Use the explicitly selected Pick task checkpoint and its actual SHA")
        source, release = Path(source).resolve(strict=True), local_release(release)
        receipt = read(generic.regular(source, "backup_receipt.json"))
        require(receipt.get("contract") == BACKUP_CONTRACT and receipt.get("status") == "passed"
                and receipt.get("profile") == self.PROFILE and receipt.get("task_id") == self.TASK_ID and type(receipt.get("step")) is int and receipt["step"] == step
                and receipt.get("training_completed") is True and receipt.get("checkpoint_save_completed") is True
                and receipt.get("optimizer_copied") is False and set(receipt.get("metadata", {})) == set(self.METADATA),
                "Require this completed Pick run's independent model-only checkpoint backup")
        for name, original in self.METADATA_SOURCES.items():
            item, path = receipt["metadata"][name], generic.regular(source, name)
            require(item.get("source_path") == original and type(item.get("bytes")) is int
                    and item["bytes"] == path.stat().st_size and valid_sha(item.get("sha256")) and sha256(path) == item["sha256"],
                    "Source metadata path/bytes/SHA binding differs: " + name)
        model = receipt.get("model", {})
        require(model.get("source_path") == self.EXPECTED_PROJECT + f"/checkpoints/ckpt_{step}/model.safetensors"
                and type(model.get("bytes")) is int and model["bytes"] == EXPECTED_MODEL_BYTES
                and model.get("source_sha256") == model.get("local_sha256") == expected_model_sha256,
                "Checkpoint transmitted-source/persisted-local SHA differs")
        for name in ("source_stat_before", "source_stat_after", "local_stat"):
            validate_stat(model.get(name), EXPECTED_MODEL_BYTES)
        require(model["source_stat_before"] == model["source_stat_after"], "Remote checkpoint changed during transmission")
        weights = generic.regular(source, "model.safetensors")
        before = stat_identity(weights)
        require(before == model["local_stat"] and (not verify_weight or sha256(weights) == expected_model_sha256)
                and stat_identity(weights) == before, "Actual persisted checkpoint bytes/identity differs")
        identity = receipt.get("run_identity", {})
        require(identity.get("run_directory") == self.EXPECTED_PROJECT and identity.get("prepared_root") == self.EXPECTED_ROOT
                and identity.get("run_config_sha256") == receipt["metadata"]["run_config.json"]["sha256"]
                and identity.get("preparation_sha256") == receipt["metadata"]["preparation.json"]["sha256"],
                "Remote completed-run identity differs")
        config, preparation = read(source / "run_config.json"), read(source / "preparation.json")
        self.validate_training(source, config, preparation)
        release_validation = self.validate_release(source, release, config, preparation)
        return source, release, config, preparation, receipt, release_validation


    def prepare(self, source, resources, output, repo, release, step, expected_model_sha256):
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU preparation requires CUDA_VISIBLE_DEVICES='' explicitly")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        source, release, config, preparation, receipt, release_validation = self.validate_source(source, release, step, expected_model_sha256)
        source_before = stat_identity(source / "model.safetensors")
        source_hashes = {name: receipt["metadata"][name]["sha256"] for name in self.METADATA}
        source_hashes["backup_receipt.json"] = sha256(source / "backup_receipt.json")
        require(same(read(source / "backup_receipt.json"), receipt)
                and all(sha256(source / name) == expected for name, expected in source_hashes.items()),
                "Sealed metadata/backup changed while verifying the selected joint")
        resources, repo = Path(resources).resolve(strict=True), Path(repo).resolve(strict=True)
        output = Path(output).absolute()
        require(not output.exists() and not output.is_symlink() and not any(p.is_symlink() for p in output.parents)
                and not any(output.resolve().is_relative_to(p) or p.is_relative_to(output.resolve()) for p in (source, resources, repo, release)),
                "Use a fresh independent deployment directory")
        require(output.parent.is_dir() and output.parent.stat().st_dev == source.stat().st_dev,
                "Source/deployment must share one filesystem for the immutable12GB hardlink")
        code = generic.validate_frozen_model_code(repo, preparation)
        construction = generic.validate_construction_metadata(resources)
        sys.path[:0] = [str(repo / "VLMExpert/src"), str(repo), str(repo / "pipeline/src"), str(repo / "VLMExpert")]
        base = resources / "joint20k"
        motion_metadata = generic.validate_motion_template_metadata(base / "motion/joint_motion.ckpt", construction["motion_template_metadata_sha256"])
        manifest, reference = read(source / "release_metadata/dino/manifest.json"), read(base / "dino_training_contract.json")
        transition = self.recorded_encoder_transition(reference["encoder"], manifest["encoder"])
        contract = generic.bind_training_motion_image_policy(reference, config)
        contract["encoder"] = export.canonical_visual_identity({"encoder": deepcopy(manifest["encoder"])})["encoder"]
        contract.update(online_checkpoint=str(resources / "dinov2-with-registers-large"),
            source_manifest_sha256=release_validation["training_manifest_sha256"], source_dataset=f"completed {self.PROFILE} train_only")
        replacements = {"model.model_name_or_path": str(base / "qwen"), "model.motion_checkpoint": str(output / "motion/joint_motion.ckpt"),
            "model.pretrained_joint_checkpoint": None, "data.root_dir": str(resources / "simple_local_usd_v1"), "data.dino_cache_root": None,
            "data.transform.field.stat_path": str(output / "dataset_statistics.json"), "train.output_dir": str(output / "unused_training"),
            "train.resume_from_checkpoint": None, "train.deepspeed_config": str(repo / "VLMExpert/scripts/deepspeed/zero3.json")}
        deployed, argv = snapshot.remap_config(config, (source / "argv.txt").read_text(), replacements)
        require(deployed["model"]["rtc"] is False, "Do not rewrite training RTC; only serve test_time may enable it")
        output.mkdir(mode=0o700)
        try:
            import torch
            require(not torch.cuda.is_initialized(), "CPU preparation may not initialize CUDA")
            provenance = output / "provenance"
            for name in (*self.METADATA, "backup_receipt.json"):
                target = provenance / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, target)
            for name, value in (("release_validation", release_validation), ("model_code_validation", code),
                    ("construction_metadata_validation", construction), ("motion_template_metadata_validation", motion_metadata)):
                write_json(provenance / (name + ".json"), value)
            require(sha256(self.queue_plan_path) == QUEUE_PLAN_SHA, "Authorized queue plan changed during preparation")
            shutil.copy2(self.queue_plan_path, provenance / "authorized_queue_plan.json")
            shutil.copy2(resources / "manifest.json", provenance / "construction_resources_manifest.json")
            write_json(output / "run_config.json", deployed)
            (output / "argv.txt").write_text(argv)
            shutil.copy2(source / "dataset_statistics.json", output / "dataset_statistics.json")
            target = output / f"checkpoints/ckpt_{step}/model.safetensors"
            target.parent.mkdir(parents=True)
            os.link(source / "model.safetensors", target)  # Never duplicate12GB onto the small evaluation partition.
            write_json(output / "dino_training_contract.json", contract)
            from psi.deploy.dual_source_runtime import load_dino_contract
            load_dino_contract(output / "dino_training_contract.json")
            structure = generic.strict_component_check(target, deployed, base / "qwen")
            require(structure.get("status") == "passed" and structure.get("strict_VLM_load") is True
                    and structure.get("strict_action_load") is True, "Strict full VLM/action CPU loading failed")
            write_json(output / "component_load_validation.json", structure)
            motion = export.export_motion(base / "motion/joint_motion.ckpt", target, output / "motion/joint_motion.ckpt",
                                         f"38000_then_{self.PROFILE}_task_{step}", reference["encoder"])
            require(motion.get("strict_meta_load") == "passed" and motion.get("exact_serialization_comparison") == "passed",
                    "Selected Pick checkpoint Motion export is not strict/exact")
            motion.update(parameter_source=f"selected_completed_{self.PROFILE}_joint_checkpoint{step}", parameter_source_sha256=expected_model_sha256,
                previous_motion_parameters_used=False, construction_metadata_only_source=str(base / "motion/joint_motion.ckpt"),
                actual_encoder_runtime_transition=transition)
            write_json(output / "motion/export_validation.json", motion)
            snapshot.validate_config(deployed, output / "argv.txt", repo)
            self.validate_source(source, release, step, expected_model_sha256, verify_weight=False)
            require(stat_identity(source / "model.safetensors") == source_before
                    and all(sha256(source / name) == sha256(provenance / name) == value for name, value in source_hashes.items()),
                    "Source checkpoint/metadata changed during CPU preparation")
            require(sha256(self.queue_plan_path) == QUEUE_PLAN_SHA, "Authorized queue plan changed during preparation")
            require(not torch.cuda.is_initialized(), "CPU preparation initialized CUDA")
            report = {"contract": CONTRACT, "status": READY_STATUS, "profile": self.PROFILE, "task_id": self.TASK_ID, "source_task_id": self.SOURCE_TASK_ID,
                "step": step, "task_step": step, "step_semantics": "selected Pick task optimizer updates", "completed_training_optimizer_steps": 10000,
                "source_unifolm_step": 38000, "joint_sha256": expected_model_sha256, "run_dir": str(output), "source_project": self.EXPECTED_PROJECT,
                "authorized_queue_plan_sha256": QUEUE_PLAN_SHA, "authorized_prepared_root": self.EXPECTED_ROOT,
                "evaluation_control_scope": "actual_model_closed_loop; historical PICO modes are capture provenance only",
                "source_snapshot": str(source), "source_backup_receipt_sha256": sha256(source / "backup_receipt.json"),
                "source_metadata_sha256": source_hashes, "training_manifest_sha256": release_validation["training_manifest_sha256"],
                "train_episodes": self.COUNTS[0], "total_frames": self.COUNTS[1], "train_windows": self.COUNTS[2], "train_split": "train_only",
                "validation_role": VALIDATION_ROLE, "Pick_validation_selected": False, "state_drop_prob": 0., "num_blocks": 6,
                "component_load_validation": structure, "full_saved_components_strict_CPU_load": structure,
                "strict_motion_load": motion["strict_meta_load"], "motion_weights": motion["weights"], "previous_motion_parameters_used": False,
                "statistics_sha256": STAT_SHA, "normalization_refitted": False, "motion_image_policy": "vlm_color_clean_motion",
                "actual_DINO_runtime_transition": transition, "deployment_path_changes": replacements,
                "training_completed": True, "selected_checkpoint_is_final": step == 10000, "checkpoint_save_completed": True,
                "optimizer_rng_restored": False, "training_rtc_enabled": False, "GPU_used": False, "CUDA_initialized": False,
                "policy_forward_performed": False, "gpu_validation": "not_run", "physical_task_success": "not_measured",
                "source_model_bytes_verified": True, "source_model_identity_unchanged_during_prepare": True,
                "verifier_sha256": sha256(Path(__file__)),
                "deployment_artifacts": {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
                                         for name in DEPLOYMENT_ARTIFACTS},
                "shared_preparation_helpers_sha256": {name: sha256(HERE / name) for name in HELPERS}}
            write_json(output / "LOCAL_RUN_READY.json", report)
            return report
        except BaseException as error:
            write_json(output / "PREPARATION_FAILED.json", {"contract": CONTRACT, "status": "failed_incomplete", "error": repr(error)})
            raise



def prepare(source, resources, output, repo, release, profile, step, expected_model_sha256, *, queue_plan=QUEUE_PLAN):
    authority = PickDeployment(profile, step, source, queue_plan)
    return authority.prepare(source, resources, output, repo, release, step, expected_model_sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'resources', 'release', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--profile', choices=tuple(PROFILES), required=True)
    parser.add_argument('--step', type=int, choices=SAVE_STEPS, required=True)
    parser.add_argument('--queue-plan', type=Path, default=QUEUE_PLAN)
    parser.add_argument('--expected-model-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.resources, args.output, args.repo, args.release,
        args.profile, args.step, args.expected_model_sha256, queue_plan=args.queue_plan), ensure_ascii=False))


if __name__ == '__main__':
    main()
