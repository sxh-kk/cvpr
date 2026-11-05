"""Stopped-run authority guards plus real tiny safetensors/Motion CPU export.

Construction metadata and tiny tensor helpers are shared with the previous
deployment tests. The old completed-run validator is never invoked here.
"""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


deploy = load("official_mobile_eval_test", SCRIPTS / "prepare_mobile_pp_official_eval.py")
legacy = load("official_mobile_tiny_fixture", Path(__file__).with_name("test_prepare_mobile_pp_eval.py"))


class StoppedDeploymentTests(unittest.TestCase):
    def setUp(self):
        fixture = self.fixture = legacy.DeploymentTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.source, self.release, self.resources, self.repo, self.output = (
            fixture.source, fixture.release, fixture.resources, fixture.repo, fixture.output)
        self.config = deepcopy(fixture.config)
        self.config["model"].update(pretrained_joint_checkpoint=deploy.SOURCE_CHECKPOINT, state_feature_dropout=0.)
        self.config["train"].update(train_batch_size=4, warmup_steps=1000, checkpointing_steps=1000, preserve_checkpoint_every=1000)
        self.config["data"]["transform"]["field"]["stat_path"] = deploy.EXPECTED_RELEASE + "/packed/source_dataset_statistics.json"
        prepared = deepcopy(self.config)
        prepared["train"]["hf_token"] = ".hf_token"
        for key in ("entity", "group", "id", "name"):
            prepared["wandb"][key] = None
        self.write("prepared_run_config.json", prepared)
        self.write("run_config.json", self.config)
        self.write("model_config_state_keep.json", self.config["model"])
        self.preparation = deepcopy(fixture.preparation)
        self.preparation.update(contract=deploy.PREPARATION_CONTRACT, source_checkpoint=deploy.SOURCE_CHECKPOINT,
            release=deploy.EXPECTED_RELEASE, per_gpu_batch=4, effective_global_batch=32,
            total_frames=56423, train_windows=50523, config_sha256=deploy.sha256(self.source / "prepared_run_config.json"),
            bindings_sha256={self.config["data"]["transform"]["field"]["stat_path"]: fixture.stat_sha})
        self.make_release()
        self.write("preparation.json", self.preparation)
        initialization = deepcopy(fixture.initialization)
        initialization.update(source_checkpoint=deploy.SOURCE_CHECKPOINT, global_batch=32)
        self.write("task_initialization.json", initialization)
        windows = deploy.read(self.source / "task_window_selection.json")
        windows["train_windows"] = 50523
        self.write("task_window_selection.json", windows)
        self.state = {"state": "failed", "returncode": 1, "run_directory": deploy.EXPECTED_PROJECT,
            "task_id": deploy.TASK_ID, "fresh_run": True, "initial_global_step": 0,
            "optimizer_scheduler_rng_restored": False, "expected_ready_sha256": self.preparation["expected_ready_sha256"],
            "finished_utc": "2026-11-04T10:48:46.530200+00:00"}
        self.write("training_state.json", self.state)
        self.retention = {"saved_update": 6000, "task_id": deploy.TASK_ID, "all_models_preserved": True,
            "all_saved_model_steps": deploy.SAVE_STEPS, "full_recovery_steps": [5000, 6000]}
        self.write("checkpoint_retention.json", self.retention)
        self.stop = {"returncode": 0, "receipt": {"status": "stopped_by_user", "remaining_scoped_processes": [],
            "stopped_at_beijing": "2026-11-04 18:48:47 +08:00", "latest_confirmed_checkpoint_step": 6000,
            "all_saved_models_retained": deploy.SAVE_STEPS,
            "raw_launcher_terminal_state": {"run_directory": deploy.EXPECTED_PROJECT}}}
        self.write("STOPPED_BY_USER.json", self.stop)
        for rank in range(8):
            self.write(f"state_policy_migration_rank_{rank}.json", {"status": "passed", "source_probability": .5,
                "target_probability": 0., "helper_sha256": deploy.MIGRATION_SHA,
                "changed_state_tensors": ["action_header._architecture_signature"], "unchanged_learned_parameters": True,
                "parameter_objects_preserved": True, "all_other_state_tensors_exactly_unchanged": True,
                "strict_target_head_roundtrip": {"status": "passed"}, "source": {"source_sha256": deploy.SOURCE_SHA}})
        self.model_sha = deploy.sha256(self.source / "model.safetensors")
        self.model_bytes = (self.source / "model.safetensors").stat().st_size
        self.receipt = {"contract": deploy.BACKUP_CONTRACT, "status": "passed", "task_id": deploy.TASK_ID,
            "step": 6000, "training_completed": False, "optimizer_copied": False,
            "run_identity": {"run_directory": deploy.EXPECTED_PROJECT, "prepared_root": deploy.EXPECTED_ROOT,
                "run_config_sha256": deploy.sha256(self.source / "run_config.json"),
                "preparation_sha256": deploy.sha256(self.source / "preparation.json")},
            "model": {"source_path": deploy.EXPECTED_PROJECT + "/checkpoints/ckpt_6000/model.safetensors",
                "bytes": self.model_bytes, "source_sha256_before": self.model_sha,
                "source_sha256_after": self.model_sha, "local_sha256": self.model_sha,
                **{key: deploy.stat_identity(self.source / "model.safetensors") for key in
                    ("source_stat_before", "source_stat_after", "local_stat")}},
            "metadata": {}, "remote_snapshot": {"observed_utc": "2026-11-05T01:00:00+00:00",
                "scoped_training_processes_absent": True,
                "training_state_sha256": deploy.sha256(self.source / "training_state.json"),
                "retention_sha256": deploy.sha256(self.source / "checkpoint_retention.json"),
                "stop_receipt_sha256": deploy.sha256(self.source / "STOPPED_BY_USER.json")}}
        for name in deploy.METADATA:
            self.rebind(name)
        self.patchers = []
        self.addCleanup(lambda: [value.stop() for value in reversed(self.patchers)])
        for name, value in (("EXPECTED_MODEL_BYTES", self.model_bytes),
                ("EXPECTED_CONFIG_SHA", self.preparation["config_sha256"]),
                ("EXPECTED_RUNTIME_CONFIG_SHA", deploy.sha256(self.source / "run_config.json")),
                ("EXPECTED_PREPARATION_SHA", deploy.sha256(self.source / "preparation.json")),
                ("EXPECTED_READY_SHA", self.preparation["expected_ready_sha256"]),
                ("EXPECTED_STOP_SHA", deploy.sha256(self.source / "STOPPED_BY_USER.json")),
                ("STAT_SHA", fixture.stat_sha)):
            self.start(patch.object(deploy, name, value))
        self.start(patch.object(deploy.generic, "RESOURCES_MANIFEST_SHA", deploy.sha256(self.resources / "manifest.json")))

    def start(self, patcher):
        self.patchers.append(patcher)
        return patcher.start()

    def write(self, name, value):
        deploy.write_json(self.source / name, value)

    def rebind(self, name):
        path = self.source / name
        self.receipt["metadata"][name] = {"source_path": deploy.METADATA_SOURCES[name],
            "bytes": path.stat().st_size, "sha256": deploy.sha256(path)}
        self.write("backup_receipt.json", self.receipt)

    def make_release(self):
        pack = deepcopy(self.fixture.pack)
        pack.update(total_frames=56423, train_windows=50523)
        self.write("release_metadata/packed/PACK_READY.json", pack)
        deploy.write_json(self.release / "packed/PACK_READY.json", pack)
        manifest = deepcopy(self.fixture.manifest)
        manifest["source"].update(total_frames=56423)
        self.write("release_metadata/dino/manifest.json", manifest)
        original = deepcopy(manifest)
        original["source"]["dataset_root"] = "/local/train"
        deploy.write_json(self.release / "dino/manifest.json", original)
        relocation = deepcopy(self.fixture.relocation)
        relocation.update(pack_ready_sha256=deploy.sha256(self.source / "release_metadata/packed/PACK_READY.json"),
            manifest_sha256=deploy.sha256(self.source / "release_metadata/dino/manifest.json"),
            original_manifest_sha256=deploy.sha256(self.release / "dino/manifest.json"))
        self.write("release_metadata/dino/REMOTE_DINO_RELOCATION.json", relocation)
        ready = {"contract": "cvpr-pico-mobile-pp-production-verification-v1", "task_id": deploy.TASK_ID,
            "status": "passed", "remote_production_ready": True, "DINO_relocation_pending": False,
            "selection_policy": "native_only", "split_restriction": "train_only", "normalization_refitted": False,
            "train_episodes": 100, "total_frames": 56423, "train_windows": 50523,
            "release_root": deploy.EXPECTED_RELEASE, "pack_root": deploy.EXPECTED_RELEASE + "/packed",
            "dino_root": deploy.EXPECTED_RELEASE + "/dino", "pack_ready_sha256": relocation["pack_ready_sha256"],
            "cache_manifest_sha256": relocation["manifest_sha256"], "remote_relocation_binding": {
                "sha256": deploy.sha256(self.source / "release_metadata/dino/REMOTE_DINO_RELOCATION.json")}}
        self.write("release_metadata/A100_PICO_DATASET_READY.json", ready)
        self.preparation["expected_ready_sha256"] = deploy.sha256(self.source / "release_metadata/A100_PICO_DATASET_READY.json")

    def validate(self, **kwargs):
        return deploy.validate_source(self.source, self.release, kwargs.get("step", 6000), kwargs.get("sha", self.model_sha))

    def test_stopped_checkpoint_accepts_without_completed_run_receipt(self):
        # Old completed-run receipts deliberately exist, but cannot provide authority.
        (self.source / "FINAL_TRAINING_ACCEPTANCE_001.json").unlink()
        self.assertEqual(self.validate()[4]["release"]["status"], "passed")

    def test_completed_or_stale_running_state_rejected_even_after_rebinding(self):
        for state, rc in (("complete", 0), ("running", None)):
            self.state.update(state=state, returncode=rc)
            self.write("training_state.json", self.state)
            self.rebind("training_state.json")
            with self.assertRaisesRegex(ValueError, "terminal user-stopped"):
                self.validate()

    def test_old_contract_and_changed_source_path_rejected(self):
        self.receipt["contract"] = deploy.generic.BACKUP_CONTRACT
        self.write("backup_receipt.json", self.receipt)
        with self.assertRaisesRegex(ValueError, "stopped-checkpoint backup"):
            self.validate()
        self.receipt["contract"] = deploy.BACKUP_CONTRACT
        self.receipt["metadata"]["training_state.json"]["source_path"] = "/stale/local/training_state.json"
        self.write("backup_receipt.json", self.receipt)
        with self.assertRaisesRegex(ValueError, "metadata path"):
            self.validate()

    def test_explicit_model_sha_and_stable_stats_required(self):
        for value in (None, "", "0" * 64):
            with self.assertRaises(ValueError):
                self.validate(sha=value)
        self.receipt["model"]["source_stat_after"]["st_mtime_ns"] += 1
        self.write("backup_receipt.json", self.receipt)
        with self.assertRaisesRegex(ValueError, "changed during backup"):
            self.validate()

    def test_bool_step_and_unsaved_checkpoint_rejected(self):
        for step in (True, 6500, 10000):
            with self.assertRaisesRegex(ValueError, "explicitly saved"):
                self.validate(step=step)

    def test_stop_evidence_is_pinned_and_not_fabricated_by_backup(self):
        self.stop["receipt"]["status"] = "completed"
        self.write("STOPPED_BY_USER.json", self.stop)
        self.rebind("STOPPED_BY_USER.json")
        with self.assertRaisesRegex(ValueError, "user-stop evidence"):
            self.validate()

    def test_preparation_cannot_rebind_the_actual_training_source_code(self):
        self.preparation["code_sha256"] = {"/different/code.py": "0" * 64}
        self.write("preparation.json", self.preparation)
        self.rebind("preparation.json")
        self.receipt["run_identity"]["preparation_sha256"] = deploy.sha256(self.source / "preparation.json")
        self.write("backup_receipt.json", self.receipt)
        with self.assertRaisesRegex(ValueError, "source/code authority"):
            self.validate()

    def test_remote_snapshot_must_follow_stop_and_terminal_state(self):
        for key, value in (("observed_utc", "2026-11-04T09:00:00+00:00"),
                ("scoped_training_processes_absent", False), ("retention_sha256", "0" * 64)):
            original = self.receipt["remote_snapshot"][key]
            self.receipt["remote_snapshot"][key] = value
            self.write("backup_receipt.json", self.receipt)
            with self.assertRaises(ValueError):
                self.validate()
            self.receipt["remote_snapshot"][key] = original

    def test_actual_statekeep_config_migration_and_count_guards(self):
        model = deepcopy(self.config["model"])
        model["state_drop_prob"] = .5
        self.write("model_config_state_keep.json", model)
        self.rebind("model_config_state_keep.json")
        with self.assertRaisesRegex(ValueError, "migrated model config"):
            self.validate()
        self.write("model_config_state_keep.json", self.config["model"])
        self.rebind("model_config_state_keep.json")
        migration = deploy.read(self.source / "state_policy_migration_rank_7.json")
        migration["unchanged_learned_parameters"] = False
        self.write("state_policy_migration_rank_7.json", migration)
        self.rebind("state_policy_migration_rank_7.json")
        with self.assertRaisesRegex(ValueError, "migration evidence"):
            self.validate()

    def test_local_dino_manifest_cannot_replace_remote_trained_manifest(self):
        (self.source / "release_metadata/dino/manifest.json").write_bytes((self.release / "dino/manifest.json").read_bytes())
        self.rebind("release_metadata/dino/manifest.json")
        with self.assertRaisesRegex(ValueError, "remote-rebound trained DINO"):
            self.validate()

    def test_missing_remote_relocation_and_symlink_metadata_rejected(self):
        target = self.source / "release_metadata/dino/REMOTE_DINO_RELOCATION.json"
        content = target.read_bytes()
        target.unlink()
        with self.assertRaises(ValueError):
            self.validate()
        elsewhere = self.fixture.root / "relocation.json"
        elsewhere.write_bytes(content)
        target.symlink_to(elsewhere)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.validate()

    def test_real_tiny_prepare_exports_selected_motion_not_old_parameters(self):
        modules = self.fixture.fake_model_modules()
        runtime = ModuleType("psi.deploy.dual_source_runtime")
        runtime.load_dino_contract = lambda path: deploy.read(path)
        modules["psi.deploy.dual_source_runtime"] = runtime
        with patch.dict(sys.modules, modules), patch.object(deploy.snapshot, "validate_config"), \
                patch.object(deploy.generic, "validate_source", side_effect=AssertionError("old task gate invoked")):
            report = deploy.prepare(self.source, self.resources, self.output, self.repo, self.release, 6000, self.model_sha)
        self.assertFalse(report["training_completed"])
        self.assertTrue(report["stopped_by_user"])
        self.assertTrue(report["checkpoint_save_completed"])
        self.assertEqual(report["task_step"], 6000)
        self.assertEqual(report["source_global_batch"], 32)
        self.assertEqual(report["joint_sha256"], self.model_sha)
        self.assertFalse(report["GPU_used"])
        exported = legacy.torch.load(self.output / "motion/joint_motion.ckpt", map_location="cpu", weights_only=False)
        self.assertTrue(legacy.torch.equal(exported["state_dict"]["weight"], legacy.torch.full((2, 2), 9.)))
        self.assertFalse((self.output / "optimizer.bin").exists())
        self.assertEqual(deploy.read(self.output / "run_config.json")["model"]["state_drop_prob"], 0.)
        self.assertEqual(report["training_manifest_sha256"], deploy.sha256(self.source / "release_metadata/dino/manifest.json"))
        self.assertEqual(report["component_load_validation"]["status"], "passed")
        self.assertEqual(set(report["deployment_artifacts"]), set(deploy.DEPLOYMENT_ARTIFACTS))
        for name, identity in report["deployment_artifacts"].items():
            self.assertEqual(identity["sha256"], deploy.sha256(self.output / name))
            self.assertEqual(identity["bytes"], (self.output / name).stat().st_size)
        # Actual independent supervisor consumes this READY, not a handwritten
        # duplicate. Its official-case fixture supplies only tiny source rows.
        campaign = load("official_mobile_campaign_fixture", Path(__file__).with_name("test_evaluate_mobile_pp30.py"))
        fixture = campaign.CampaignTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.args.deployment = self.output
        fixture.args.expected_model_sha256 = self.model_sha
        fixture.args.training_manifest = self.source / "release_metadata/dino/manifest.json"
        frozen = campaign.evaluate.freeze_inputs(fixture.args)
        self.assertEqual(len(frozen["cases"]), 30)
        self.assertEqual(frozen["model"]["sha256"], self.model_sha)

    def test_no_publish_when_source_identity_changes_during_prepare(self):
        modules = self.fixture.fake_model_modules()
        runtime = ModuleType("psi.deploy.dual_source_runtime")
        runtime.load_dino_contract = lambda path: deploy.read(path)
        modules["psi.deploy.dual_source_runtime"] = runtime
        original = deploy.export.export_motion
        def export_then_change(*args, **kwargs):
            result = original(*args, **kwargs)
            path = self.source / "model.safetensors"
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
            return result
        with patch.dict(sys.modules, modules), patch.object(deploy.snapshot, "validate_config"), \
                patch.object(deploy.export, "export_motion", side_effect=export_then_change):
            with self.assertRaisesRegex(ValueError, "file identity"):
                deploy.prepare(self.source, self.resources, self.output, self.repo, self.release, 6000, self.model_sha)
        self.assertFalse((self.output / "LOCAL_RUN_READY.json").exists())
        self.assertTrue((self.output / "PREPARATION_FAILED.json").exists())


if __name__ == "__main__":
    unittest.main()
