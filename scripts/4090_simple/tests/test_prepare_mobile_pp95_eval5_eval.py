"""Isolated CPU metadata/selection gates and real tiny new-Motion export."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import torch
from safetensors.torch import save_file

SCRIPTS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mobile95_deploy_tests", SCRIPTS / "prepare_mobile_pp95_eval5_eval.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def package(name, **attributes):
    value = ModuleType(name)
    value.__path__ = []
    value.__dict__.update(attributes)
    return value


class TinyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2, 2))


class TinyHead(TinyModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.state_drop_prob = kwargs["state_drop_prob"]
        self.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(kwargs["action_num_blocks"])])
        self.register_buffer("_architecture_signature", torch.zeros(16, dtype=torch.int64))


def encoder(version="old"):
    return {"checkpoint": "/metadata/dino", "local_file_sha256": {"model.safetensors": "weights"},
        "torch_version": version, "transformers_version": version,
        "image_process": {"numpy_version": "1.26.4", "pillow_version": "11.3.0", "height": 256, "width": 320},
        "shape": [5, 1024], "token_order": ["cls", "register_0", "register_1", "register_2", "register_3"],
        "storage_dtype": "float32", "optimization": {"compute_precision": "bf16"}}


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.release, self.resources, self.repo, self.output = (self.root / name for name in
            ("source", "release", "resources", "repo", "output"))
        for path in (self.source, self.release, self.repo):
            path.mkdir()
        self.patchers = []
        self.addCleanup(lambda: [value.stop() for value in reversed(self.patchers)])
        self.saved_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self.saved_path))
        self.model = {"action_architecture": "dual_source_v1", "num_blocks": 6, "action_dim": 80, "odim": 45,
            "action_chunk_size": 30, "state_drop_prob": 0., "state_null_token": True, "tune_motion": True,
            "tune_vlm": True, "rtc": False, "pretrained_joint_checkpoint": deploy.SOURCE_CHECKPOINT,
            "hidden_dim": 8, "nhead": 2, "view_feature_dim": 4, "time_dim": 4, "qk_norm": "layer_norm", "dropout": 0.,
            "model_name_or_path": "/old/qwen", "motion_checkpoint": "/old/motion.ckpt"}
        self.prepared = {"model": self.model, "train": {"train_batch_size": 8, "gradient_accumulation_steps": 1,
            "max_training_steps": 20000, "warmup_steps": 1000, "checkpointing_steps": 20000,
            "preserve_checkpoint_every": 500, "validation_steps": 0, "resume_from_checkpoint": None,
            "skip_resumed_steps": False, "early_checkpoint_steps": deploy.SAVE_STEPS[:-1], "hf_token": ".hf_token",
            "output_dir": "/old/output", "deepspeed_config": "/old/zero3.json"},
            "wandb": {"project": "cvpr-handover-merged114", "resume": "never",
                **{key: None for key in deploy.TRACKER_KEYS}},
            "data": {"root_dir": "/old/data", "dino_cache_root": "/old/dino", "transform": {
                "repack": {"action_keys": ["action.body_token_v1_1", "action[:14]"], "state_keys": ["observation.state"],
                    "action_mask_key": "action_mask", "pad_action_dim": 80, "pad_state_dim": 45},
                "model": {"shared_image_profile": "256x320", "motion_image_policy": "vlm_color_clean_motion"},
                "field": {"stat_path": "/remote/stats", "action_min": [-1.], "action_max": [1.],
                    "state_min": [-2.], "state_max": [2.]}}}}
        self.write("prepared_run_config.json", self.prepared)
        self.start(patch.object(deploy, "EXPECTED_CONFIG_SHA", deploy.sha256(self.source / "prepared_run_config.json")))
        self.config = deepcopy(self.prepared)
        self.config["train"]["hf_token"] = "PosixPath('.hf_token')"
        self.config["wandb"].update(entity="", group="posttrain", id="actual-fixture-id", name="actual-fixture-project")
        self.write("run_config.json", self.config)
        for filename in ("argv.txt", "prepared_argv.txt"):
            (self.source / filename).write_text("--model.motion_checkpoint=/old/motion.ckpt\n--model.action_architecture=dual_source_v1\n")
        self.write("dataset_statistics.json", {"inherited": "fixture38k"})
        self.start(patch.object(deploy, "STAT_SHA", deploy.sha256(self.source / "dataset_statistics.json")))
        self.make_split_and_windows()
        self.preparation = {"contract": deploy.PREPARATION_CONTRACT, "task_id": deploy.TASK_ID, "fresh_run": True,
            "initial_global_step": 0, "optimizer_scheduler_rng_restored": False, "source_step": 38000,
            "source_checkpoint_sha256": deploy.SOURCE_SHA, "source_checkpoint": deploy.SOURCE_CHECKPOINT,
            "gpu_count": 8, "per_gpu_batch": 8, "gradient_accumulation_steps": 1, "effective_global_batch": 64,
            "max_optimizer_steps": 20000, "warmup_steps": 1000, "train_episodes": 95, "validation_episodes": 5,
            "source_total_frames": 61806, "source_total_windows": 55906, "validation_role": deploy.VALIDATION_ROLE,
            "split_file": deploy.EXPECTED_ROOT + "/episode_split.json", "split_sha256": deploy.EXPECTED_SPLIT_SHA,
            "diagnostic_steps": deploy.DIAGNOSTIC_STEPS,
            "checkpoint_policy": {"model_steps": deploy.SAVE_STEPS, "full_recovery_states_to_keep": 2},
            "config_sha256": deploy.EXPECTED_CONFIG_SHA, "argv_sha256": deploy.sha256(self.source / "prepared_argv.txt")}
        self.write("preparation.json", self.preparation)
        self.write("task_initialization.json", {"source_checkpoint": deploy.SOURCE_CHECKPOINT, "source_step": 38000,
            "strict_joint_loader": True, "fresh_optimizer_scheduler_rng": True, "global_batch": 64,
            "backend": {"backend": "fused", "foreach": False, "initialized_parameter_states": 0}})
        self.state = {"state": "complete", "returncode": 0, "run_directory": deploy.EXPECTED_PROJECT,
            "task_id": deploy.TASK_ID, "config_sha256": deploy.EXPECTED_CONFIG_SHA,
            "split_sha256": deploy.EXPECTED_SPLIT_SHA, "max_optimizer_steps": 20000}
        self.write("training_state.json", self.state)
        self.retention = {"saved_update": 20000, "all_models_preserved": True,
            "all_saved_model_steps": deploy.SAVE_STEPS, "full_recovery_steps": [19500, 20000]}
        self.write("checkpoint_retention.json", self.retention)
        self.write("gpu_check_update20000.json", {"status": "passed", "completed_updates": 20000, "gpu_count": 8,
            "per_gpu_batch": 8, "accumulation": 1, "all_rank_probe_values_equal": True,
            "actual_fused_flags_passed": True, "actual_actions_shapes_all_ranks": [[8, 30, 80]] * 8})
        self.make_weights()
        self.start(patch.object(deploy, "EXPECTED_MODEL_BYTES", (self.source / "model.safetensors").stat().st_size))
        self.model_sha = deploy.sha256(self.source / "model.safetensors")
        self.final = {"contract": deploy.FINAL_CONTRACT, "status": "passed", "task_id": deploy.TASK_ID,
            "root": deploy.EXPECTED_ROOT, "project": deploy.EXPECTED_PROJECT, "completed_step": 20000,
            "runtime_wandb": {key: self.config["wandb"][key] for key in deploy.TRACKER_KEYS},
            "full_recovery_steps": [19500, 20000],
            "models": [{"step": step, "path": f"{deploy.EXPECTED_PROJECT}/checkpoints/ckpt_{step}/model.safetensors",
                        "sha256": self.model_sha, "size_bytes": deploy.EXPECTED_MODEL_BYTES} for step in deploy.SAVE_STEPS],
            "selection_metric": deploy.SELECTION_METRIC, "selection_candidate_steps": deploy.SELECTION_CANDIDATES,
            "tie_break": "lowest_step", "selections": {"early": 4000, "best_validation": 6000, "final": 20000}}
        self.make_diagnostics()
        for relative in deploy.RELEASE_METADATA:
            self.write("release_metadata/" + relative, {"fixture": relative, "encoder": encoder("actual-mobile")})
        self.receipt = {"contract": deploy.BACKUP_CONTRACT, "task_id": deploy.TASK_ID, "step": 4000,
            "optimizer_copied": False, "completed_retention_marker": self.retention,
            "run_identity": {"run_directory": deploy.EXPECTED_PROJECT,
                "run_config_sha256": deploy.sha256(self.source / "run_config.json"),
                "preparation_sha256": deploy.sha256(self.source / "preparation.json"), "split_sha256": deploy.EXPECTED_SPLIT_SHA},
            "model": {"source_path": deploy.EXPECTED_PROJECT + "/checkpoints/ckpt_4000/model.safetensors",
                "bytes": deploy.EXPECTED_MODEL_BYTES, "source_sha256_before": self.model_sha,
                "source_sha256_after": self.model_sha, "local_sha256": self.model_sha}}
        self.seal()
        # The unchanged original100 byte/normalization/DINO relocation validator
        # already has independent tests. These fixtures contain no source cache.
        self.start(patch.object(deploy.generic, "validate_release", return_value={"status": "passed"}))
        self.start(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}))

    def start(self, patcher):
        self.patchers.append(patcher)
        return patcher.start()

    def write(self, filename, value):
        deploy.write_json(self.source / filename, value)

    def make_split_and_windows(self):
        self.episodes = []
        for i in range(100):
            uid = str(uuid.UUID(int=i + 1))
            length = 618 + int(i < 6)
            self.episodes.append({"episode_index": i, "human_demo_uuid": uid, "length": length,
                "task_id": deploy.TASK_ID, "split": "train", "split_restriction": "train_only",
                "eligible_current_rows": list(range(30, length - 29)), "lineage_complete": True,
                "lineage_group_sources": [{"source_namespace": "fixture", "human_demo_uuid": uid}]})
        first = deploy.splitter.build_split(self.episodes, "0" * 64)
        sizes = {index: (571 if j == 0 else 572) for j, index in enumerate(first["eval_episode_indices"])}
        sizes.update({index: 546 for index in first["train_probe_episode_indices"]})
        others = sorted(set(range(100)) - sizes.keys())
        sizes.update({index: 624 + int(j < 57) for j, index in enumerate(others)})
        for row in self.episodes:
            row["length"] = sizes[row["episode_index"]]
            row["eligible_current_rows"] = list(range(30, row["length"] - 29))
        path = self.release / "packed/train/meta/episodes.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in self.episodes))
        self.split = deploy.splitter.build_split(self.episodes, deploy.sha256(path))
        self.write("episode_split.json", self.split)
        self.start(patch.object(deploy, "EXPECTED_SPLIT_SHA", deploy.sha256(self.source / "episode_split.json")))
        self.rows_by_episode, offset = {}, 0
        for row in self.episodes:
            self.rows_by_episode[row["episode_index"]] = list(range(offset, offset + len(row["eligible_current_rows"])))
            offset += row["length"] - 30
        self.windows = {"contract": deploy.splitter.CONTRACT, "split_sha256": deploy.EXPECTED_SPLIT_SHA,
            "source_metadata_sha256": self.split["source_metadata_sha256"], "train_episodes": 95, "validation_episodes": 5,
            "train_probe_episodes": 5, "source_total_frames": 61806, "source_total_windows": 55906, "source_episode_count": 100,
            "source_role_override": self.split["source_role_override"]}
        for prefix, key in (("train", "train"), ("validation", "eval"), ("train_probe", "train_probe")):
            indices = self.split[key + "_episode_indices"]
            self.windows[prefix + "_windows"] = sum(len(self.rows_by_episode[i]) for i in indices)
            self.windows[prefix + "_windows_per_episode"] = {str(i): len(self.rows_by_episode[i]) for i in indices}
            self.windows[prefix + "_frames"] = sum(self.episodes[i]["length"] for i in indices)
            if prefix != "train":
                self.windows[prefix + "_rows"] = [row for i in indices for row in self.rows_by_episode[i]]
        for key in ("train_episode_indices", "eval_episode_indices", "train_probe_episode_indices", "train_uuids", "eval_uuids", "train_probe_uuids"):
            self.windows[key] = self.split[key]
        self.windows.update({key: True for key in ("train_eval_uuid_disjoint", "train_eval_lineage_disjoint",
            "train_eval_original_rows_disjoint", "train_probe_subset_of_train", "counts_partition_sum_verified")})
        self.write("task_window_selection.json", self.windows)

    def make_weights(self, state_drop=0.):
        model = self.model
        signature = [0x44534143, 1, 80, 30, 8, 6, 2, 4, 45, 4, 1, 2,
            struct.unpack("q", struct.pack("d", state_drop))[0], struct.unpack("q", struct.pack("d", 0.))[0], 11, 768]
        save_file({"vlm_model.weight": torch.full((2, 2), 3.), "action_header.weight": torch.full((2, 2), 5.),
            "action_header._architecture_signature": torch.tensor(signature, dtype=torch.int64),
            "motion_conditioner.model.weight": torch.full((2, 2), 9.)}, self.source / "model.safetensors")

    def make_diagnostics(self):
        for step in deploy.DIAGNOSTIC_STEPS:
            value = {"contract": deploy.DIAGNOSTIC_CONTRACT, "completed_update": step, "split_sha256": deploy.EXPECTED_SPLIT_SHA,
                "world_size": 8, "validation_role": deploy.VALIDATION_ROLE, "roles": {}, "metrics": {}}
            for role, prefix in (("val", "eval"), ("train_probe", "train_probe")):
                # Six and seven thousand tie. The lower saved step must win.
                loss = 1. if step in (6000, 7000) else 1. + abs(step - 6000) / 10000
                items = {}
                for index in self.split[prefix + "_episode_indices"]:
                    rows = self.rows_by_episode[index]
                    items[str(index)] = {"full_fm_loss": loss, "full_fm_windows": len(rows),
                        "full_rows_sha256": deploy.generic.digest(rows), "sampled_rows": [rows[i * (len(rows) - 1) // 7] for i in range(8)]}
                value["roles"][role] = {"per_episode": items, "metrics": {"full_fm_loss_episode_macro": loss},
                    "counts": {"episodes": 5, "full_fm_windows": sum(item["full_fm_windows"] for item in items.values()),
                               "sampled_l1_windows": 40}}
                value["metrics"][role + "_full_fm_loss_episode_macro"] = loss
            self.write(f"mobile_diagnostics/update_{step:05d}.json", value)

    def seal(self):
        for key, filename, remote in (("configuration", "prepared_run_config.json", deploy.EXPECTED_ROOT + "/run_config.json"),
            ("runtime_configuration", "run_config.json", deploy.EXPECTED_PROJECT + "/run_config.json"),
            ("split", "episode_split.json", deploy.EXPECTED_ROOT + "/episode_split.json"),
            ("preparation_file", "preparation.json", deploy.EXPECTED_ROOT + "/preparation.json"),
            ("state_file", "training_state.json", deploy.EXPECTED_ROOT + "/training_state.json"),
            ("retention_file", "checkpoint_retention.json", deploy.EXPECTED_PROJECT + "/checkpoint_retention.json"),
            ("health_file", "gpu_check_update20000.json", deploy.EXPECTED_PROJECT + "/gpu_check_update20000.json")):
            self.final[key] = {"path": remote, "sha256": deploy.sha256(self.source / filename)}
        self.final["diagnostics"] = [{"completed_update": step, "path": deploy.EXPECTED_PROJECT + "/" + filename,
            "sha256": deploy.sha256(self.source / filename)} for step, filename in zip(deploy.DIAGNOSTIC_STEPS, deploy.DIAGNOSTIC_METADATA)]
        self.write("FINAL_TRAINING_ACCEPTANCE_001.json", self.final)
        self.final_sha = deploy.sha256(self.source / "FINAL_TRAINING_ACCEPTANCE_001.json")
        self.receipt["metadata"] = {name: deploy.sha256(self.source / name) for name in deploy.METADATA}
        self.write("backup_receipt.json", self.receipt)

    def validate(self, step=4000):
        return deploy.validate_source(self.source, self.release, step, self.final_sha)

    def test_new95_source_passes_without_invented_old_state_fields(self):
        result = self.validate()
        self.assertEqual(result[4]["selection"]["selections"]["best_validation"], 6000)
        self.assertEqual(result[4]["windows"]["train_windows"], 53342)
        self.assertNotIn("fresh_run", self.state)
        self.assertEqual(deploy.generic.EXPECTED_STEP, 10000)
        self.assertEqual(Path(deploy.generic.__file__).parent, SCRIPTS)
        self.assertEqual(Path(deploy.export.__file__).parent, SCRIPTS)

    def test_backup_must_be_published_before_any_weight_read(self):
        (self.source / "backup_receipt.json").unlink()
        (self.source / "model.safetensors").write_bytes(b"streaming unfinished bytes")
        original = deploy.sha256
        def no_weights(path):
            self.assertNotEqual(Path(path).name, "model.safetensors")
            return original(path)
        with patch.object(deploy, "sha256", side_effect=no_weights), self.assertRaisesRegex(ValueError, "backup_receipt"):
            self.validate()

    def test_old_backup_task_step_and_model_before_after_mismatch_rejected(self):
        baseline = deepcopy(self.receipt)
        for section, key, value in ((None, "contract", deploy.generic.BACKUP_CONTRACT), (None, "task_id", "handover"),
                (None, "step", 10000), ("model", "source_sha256_after", "0" * 64),
                ("run_identity", "split_sha256", "0" * 64)):
            with self.subTest(key=key):
                self.receipt = deepcopy(baseline)
                (self.receipt if section is None else self.receipt[section])[key] = value
                self.write("backup_receipt.json", self.receipt)
                with self.assertRaises(ValueError): self.validate()

    def test_final_sha_and_real_root_project_33models_are_required(self):
        with self.assertRaisesRegex(ValueError, "acceptance SHA"):
            deploy.validate_source(self.source, self.release, 4000, "0" * 64)
        baseline = deepcopy(self.final)
        for key, value in (("root", "/old_mobile100"), ("project", deploy.generic.EXPECTED_RUN),
                           ("completed_step", 10000), ("models", baseline["models"][:-1])):
            self.final = deepcopy(baseline); self.final[key] = value; self.seal()
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate()

    def test_running_or_failed_training_and_final_health_are_refused(self):
        original = deepcopy(self.state)
        for key, value in (("state", "running"), ("returncode", 1), ("split_sha256", "0" * 64)):
            self.write("training_state.json", {**original, key: value}); self.seal()
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate()
        self.write("training_state.json", original)
        health = deploy.read(self.source / "gpu_check_update20000.json")
        health["actual_actions_shapes_all_ranks"][-1] = [4, 30, 80]
        self.write("gpu_check_update20000.json", health); self.seal()
        with self.assertRaisesRegex(ValueError, "production health"): self.validate()

    def test_final_preparation_authority_cannot_be_replaced_by_self_resealed_backup(self):
        changed = deepcopy(self.preparation)
        changed["code_sha256"] = {"/frozen/632968d/MotionExpert/model.py": "0" * 64}
        self.write("preparation.json", changed)
        new_sha = deploy.sha256(self.source / "preparation.json")
        self.receipt["run_identity"]["preparation_sha256"] = new_sha
        self.receipt["metadata"]["preparation.json"] = new_sha
        self.write("backup_receipt.json", self.receipt)
        # Neither the final acceptance nor the external expected SHA changed.
        with self.assertRaisesRegex(ValueError, "preparation_file"): self.validate()

    def test_fresh38k_no_resume_or_state_drop_visual_policy_drift(self):
        original = deepcopy(self.preparation)
        for key, value in (("source_checkpoint", "/mobile100/ckpt_10000/model.safetensors"),
                           ("optimizer_scheduler_rng_restored", True), ("initial_global_step", 10000)):
            self.write("preparation.json", {**original, key: value})
            self.receipt["run_identity"]["preparation_sha256"] = deploy.sha256(self.source / "preparation.json")
            self.seal()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "fresh joint"): self.validate()
        self.write("preparation.json", original)
        for section, key, value in (("model", "state_drop_prob", .5), ("train", "resume_from_checkpoint", "latest")):
            changed = deepcopy(self.config); changed[section][key] = value
            self.write("run_config.json", changed); self.seal()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Runtime configuration"): self.validate()

    def test_runtime_tracker_is_actual_final_bound_and_no_other_drift(self):
        changed = deepcopy(self.config); changed["wandb"]["id"] = "foreign"
        self.write("run_config.json", changed); self.seal()
        with self.assertRaisesRegex(ValueError, "tracker"): self.validate()
        self.final["runtime_wandb"] = {key: changed["wandb"][key] for key in deploy.TRACKER_KEYS}
        self.receipt["run_identity"]["run_config_sha256"] = deploy.sha256(self.source / "run_config.json")
        self.seal()
        self.validate()

    def test_prepared_wandb_project_resume_and_other_static_fields_cannot_drift(self):
        self.assertTrue(self.validate()[4]["runtime_metadata"]["all_other_values_exact"])
        for key, value in (("project", "foreign-project"), ("resume", "allow"), ("extra", True)):
            with self.subTest(key=key):
                changed = deepcopy(self.config)
                changed["wandb"][key] = value
                self.write("run_config.json", changed)
                self.receipt["run_identity"]["run_config_sha256"] = deploy.sha256(self.source / "run_config.json")
                self.seal()
                with self.assertRaisesRegex(ValueError, "Runtime configuration"):
                    self.validate()

    def test_original_split_uuid_windows_and_stats_cannot_drift(self):
        for key, value in (("validation_windows", 2563), ("train_probe_windows", 2434),
                           ("eval_uuids", ["foreign"] * 5), ("train_eval_lineage_disjoint", False)):
            changed = {**self.windows, key: value}
            self.write("task_window_selection.json", changed); self.seal()
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate()
        self.write("task_window_selection.json", self.windows)
        self.write("dataset_statistics.json", {"refitted": True}); self.seal()
        with self.assertRaisesRegex(ValueError, "normalization"): self.validate()

    def test_21_diagnostics_full_rows_native_macro_and_best_tie_are_checked(self):
        path = "mobile_diagnostics/update_06000.json"
        original = deploy.read(self.source / path)
        for change in ("row", "flat", "coverage"):
            value = deepcopy(original)
            if change == "row": next(iter(value["roles"]["val"]["per_episode"].values()))["full_rows_sha256"] = "0" * 64
            elif change == "flat": value["metrics"]["val_full_fm_loss_episode_macro"] = 0.
            else: value["roles"]["val"]["counts"]["full_fm_windows"] -= 1
            self.write(path, value); self.seal()
            with self.subTest(change=change), self.assertRaises(ValueError): self.validate()
        self.write(path, original)
        self.final["selections"]["best_validation"] = 7000; self.seal()
        with self.assertRaisesRegex(ValueError, "Best validation"): self.validate()
        self.final["selections"]["best_validation"] = 6000
        self.final["selection_candidate_steps"] = deploy.SAVE_STEPS; self.seal()
        with self.assertRaisesRegex(ValueError, "17-model"): self.validate()

    def test_only_selected_early_best_final_dynamic_sha_and_no_model_corruption(self):
        self.receipt["step"] = 4500
        self.receipt["model"]["source_path"] = deploy.EXPECTED_PROJECT + "/checkpoints/ckpt_4500/model.safetensors"
        self.write("backup_receipt.json", self.receipt)
        with self.assertRaisesRegex(ValueError, "final-selected"):
            self.validate(step=4500)
        self.receipt["step"] = 6000
        self.receipt["model"]["source_path"] = deploy.EXPECTED_PROJECT + "/checkpoints/ckpt_6000/model.safetensors"
        self.write("backup_receipt.json", self.receipt)
        self.assertEqual(self.validate(step=6000)[4]["model"]["sha256"], self.model_sha)
        path = self.source / "model.safetensors"
        before = path.stat(); payload = bytearray(path.read_bytes()); payload[-1] ^= 1; path.write_bytes(payload)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "local checkpoint bytes"): self.validate(step=6000)

    def test_all33_models_and_only_last2_full_states(self):
        self.retention["full_recovery_steps"] = [19000, 19500, 20000]
        self.write("checkpoint_retention.json", self.retention); self.seal()
        with self.assertRaisesRegex(ValueError, "two-state"): self.validate()

    def fake_modules(self):
        reader = lambda p: torch.load(p, map_location="cpu", weights_only=False)
        return {"transformers": package("transformers", AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace()),
                    Qwen3VLForConditionalGeneration=TinyModel),
            "psi": package("psi"), "psi.models": package("psi.models"), "psi.deploy": package("psi.deploy"),
            "psi.models.dual_source_action": package("psi.models.dual_source_action", DualSourceActionTransformer=TinyHead),
            "psi.models.checkpoint_aliases": package("psi.models.checkpoint_aliases", restore_shared_tensor_aliases=lambda model, value: value),
            "psi.deploy.dual_source_runtime": package("psi.deploy.dual_source_runtime", load_dino_contract=lambda p: deploy.read(p)),
            "MotionExpert": package("MotionExpert"), "MotionExpert.dataset": package("MotionExpert.dataset"),
            "MotionExpert.checkpoints": package("MotionExpert.checkpoints", read_checkpoint=reader,
                checkpoint_config=lambda value: SimpleNamespace(), validate_checkpoint_statistics=lambda *args: None),
            "MotionExpert.config": package("MotionExpert.config", as_dict=lambda cfg: {}, architecture_metadata=lambda cfg: {}),
            "MotionExpert.dataset.normalization": package("MotionExpert.dataset.normalization", MotionNormalizer=lambda stats: SimpleNamespace(stats=stats, clip=None)),
            "MotionExpert.model": package("MotionExpert.model", UniEgoMotion=TinyModel),
            "MotionExpert.visual_contract": package("MotionExpert.visual_contract",
                checkpoint_visual_contract=lambda value: value["motionexpert_visual_conditioning"], compare_visual_contracts=lambda *a, **k: {})}

    def make_resources(self):
        for relative in ("joint20k/qwen", "joint20k/motion", "dinov2-with-registers-large", "simple_local_usd_v1"):
            (self.resources / relative).mkdir(parents=True)
        deploy.write_json(self.resources / "manifest.json", {"fixture": True})
        self.reference = {"encoder": encoder(), "shared_image_profile": "256x320"}
        deploy.write_json(self.resources / "joint20k/dino_training_contract.json", self.reference)
        torch.save({"state_dict": {"weight": torch.full((2, 2), -7.)}, "motionexpert_normalization": {"mean": [0.]},
            "motionexpert_visual_conditioning": {"format": "motionexpert-visual-v1", "mode": "shared_rgb",
                "image_process": encoder()["image_process"], "encoder": encoder()}}, self.resources / "joint20k/motion/joint_motion.ckpt")

    def test_prepare_real_cpu_components_and_new_motion_only_keep95_provenance(self):
        self.make_resources()
        with patch.dict(sys.modules, self.fake_modules()), \
             patch.object(deploy.generic, "validate_frozen_model_code", return_value={"status": "passed"}), \
             patch.object(deploy.generic, "validate_construction_metadata", return_value={"motion_template_metadata_sha256": "fixture"}), \
             patch.object(deploy.generic, "validate_motion_template_metadata", return_value={"old_parameter_values_used": False}), \
             patch.object(deploy.snapshot, "validate_config", return_value=None):
            report = deploy.prepare(self.source, self.resources, self.output, self.repo, self.release, 4000, self.final_sha)
        self.assertEqual(report["contract"], deploy.CONTRACT)
        self.assertEqual(report["mobile_split_sha256"], deploy.EXPECTED_SPLIT_SHA)
        self.assertEqual(report["train_windows"], 53342)
        self.assertEqual(report["selected_roles"], ["early"])
        self.assertFalse(report["previous_motion_parameters_used"])
        self.assertFalse(report["GPU_used"])
        saved = torch.load(self.output / "motion/joint_motion.ckpt", weights_only=False)
        torch.testing.assert_close(saved["state_dict"]["weight"], torch.full((2, 2), 9.))
        self.assertTrue(report["full_saved_components_strict_CPU_load"]["strict_VLM_load"])
        self.assertTrue(report["full_saved_components_strict_CPU_load"]["strict_action_load"])
        self.assertEqual(deploy.read(self.output / "run_config.json")["model"]["state_drop_prob"], 0.)

    def test_strict_saved_state_signature_has_no_old_motion_fallback(self):
        self.make_weights(state_drop=.5)
        with patch.dict(sys.modules, self.fake_modules()), self.assertRaisesRegex(ValueError, "signature"):
            deploy.generic.strict_component_check(self.source / "model.safetensors", self.config, self.repo)

    def test_gpu_hidden_and_fresh_output_required(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), self.assertRaisesRegex(ValueError, "CPU preparation"):
            deploy.prepare(self.source, self.resources, self.output, self.repo, self.release, 4000, self.final_sha)
        self.resources.mkdir()
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "fresh independent"):
            deploy.prepare(self.source, self.resources, self.output, self.repo, self.release, 4000, self.final_sha)


if __name__ == "__main__":
    unittest.main()
