"""Bend authority guards and actual tiny CPU strict loading/Motion serialization."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import prepare_bend_pick_eval as deploy

spec = importlib.util.spec_from_file_location("bend_tiny_construction_fixture", Path(__file__).with_name("test_prepare_mobile_pp_eval.py"))
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


class BendDeploymentTests(unittest.TestCase):
    def setUp(self):
        f = self.fixture = fixture_module.DeploymentTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.source, self.release, self.resources, self.repo, self.output = f.source, f.release, f.resources, f.repo, f.output
        self.config = deepcopy(f.config)
        self.config["model"]["state_feature_dropout"] = 0.
        self.config["train"].update(train_batch_size=4, warmup_steps=1000, checkpointing_steps=1000,
            preserve_checkpoint_every=500, early_checkpoint_steps=[4500, 5500, 6500])
        self.config["data"]["transform"]["field"]["stat_path"] = deploy.EXPECTED_RELEASE + "/packed/source_dataset_statistics.json"
        self.preparation = deepcopy(f.preparation)
        self.preparation.update(contract=deploy.PREPARATION_CONTRACT, task_id=deploy.TASK_ID, profile="bend_pick",
            release=deploy.EXPECTED_RELEASE, per_gpu_batch=4, effective_global_batch=32,
            total_frames=15722, train_windows=9822, validation_role=deploy.VALIDATION_ROLE,
            Pick_validation_selected=False, checkpoint_policy={"model_steps": list(deploy.SAVE_STEPS)},
            bindings_sha256={self.config["data"]["transform"]["field"]["stat_path"]: f.stat_sha})
        prepared = deepcopy(self.config)
        prepared["train"]["hf_token"] = ".hf_token"
        for key in ("entity", "group", "id", "name"):
            prepared["wandb"][key] = None
        self.write("prepared_run_config.json", prepared)
        self.write("run_config.json", self.config)
        self.write("model_config_state_keep.json", self.config["model"])
        self.preparation["config_sha256"] = deploy.sha256(self.source / "prepared_run_config.json")
        pack = deepcopy(f.pack)
        pack.update(contract="cvpr-pico-pick-lerobot-pack-v1", task_id=deploy.TASK_ID, total_frames=15722, train_windows=9822)
        self.write("release_metadata/packed/PACK_READY.json", pack)
        deploy.write_json(self.release / "packed/PACK_READY.json", pack)
        manifest = deepcopy(f.manifest)
        manifest["source"].update(total_frames=15722)
        self.write("release_metadata/dino/manifest.json", manifest)
        original = deepcopy(manifest)
        original["source"]["dataset_root"] = "/original/bend/train"
        deploy.write_json(self.release / "dino/manifest.json", original)
        relocation = deepcopy(f.relocation)
        relocation.update(contract="cvpr-pico-pick-dino-host-relocation-v1", task_id=deploy.TASK_ID,
            pack_ready_sha256=deploy.sha256(self.source / "release_metadata/packed/PACK_READY.json"),
            manifest_sha256=deploy.sha256(self.source / "release_metadata/dino/manifest.json"),
            original_manifest_sha256=deploy.sha256(self.release / "dino/manifest.json"))
        self.write("release_metadata/dino/REMOTE_DINO_RELOCATION.json", relocation)
        ready = dict(contract="cvpr-pico-pick-production-verification-v1", task_id=deploy.TASK_ID,
            status="passed", remote_production_ready=True, DINO_relocation_pending=False, normalization_refitted=False,
            selection_policy="native_only", split_restriction="train_only", train_episodes=100, total_frames=15722,
            train_windows=9822, release_root=deploy.EXPECTED_RELEASE, pack_root=deploy.EXPECTED_RELEASE + "/packed",
            dino_root=deploy.EXPECTED_RELEASE + "/dino", pack_ready_sha256=relocation["pack_ready_sha256"],
            cache_manifest_sha256=relocation["manifest_sha256"], remote_relocation_binding={
                "sha256": deploy.sha256(self.source / "release_metadata/dino/REMOTE_DINO_RELOCATION.json")})
        self.write("release_metadata/A100_PICO_DATASET_READY.json", ready)
        self.preparation["expected_ready_sha256"] = deploy.sha256(self.source / "release_metadata/A100_PICO_DATASET_READY.json")
        self.write("preparation.json", self.preparation)
        initialization = deepcopy(f.initialization)
        initialization.update(global_batch=32)
        self.write("task_initialization.json", initialization)
        self.write("task_window_selection.json", dict(train_windows=9822, train_episode_indices=list(range(100)),
            train_task_id=deploy.TASK_ID, validation_role=deploy.VALIDATION_ROLE, Pick_validation_selected=False,
            validation_windows=40, validation_rows=list(range(40))))
        self.state = dict(state="complete", returncode=0, run_directory=deploy.EXPECTED_PROJECT, task_id=deploy.TASK_ID,
            max_optimizer_steps=10000, expected_ready_sha256=self.preparation["expected_ready_sha256"])
        self.write("training_state.json", self.state)
        self.write("checkpoint_retention.json", dict(all_saved_model_steps=list(deploy.SAVE_STEPS),
            all_models_preserved=True, task_id=deploy.TASK_ID, saved_update=10000))
        self.write("gpu_check_update10000.json", dict(status="passed", completed_updates=10000, gpu_count=8,
            per_gpu_batch=4, accumulation=1, actual_fused_flags_passed=True, all_rank_probe_values_equal=True,
            actual_actions_shapes_all_ranks=[[4, 30, 80]] * 8))
        for rank in range(8):
            self.write(f"state_policy_migration_rank_{rank}.json", dict(status="passed", helper_sha256=deploy.MIGRATION_SHA,
                target_probability=0., unchanged_learned_parameters=True, all_other_state_tensors_exactly_unchanged=True,
                strict_target_head_roundtrip={"status": "passed"}, source={"source_sha256": deploy.SOURCE_SHA,
                    "all_serialized_tensors_exactly_equal_before_migration": True}))
        self.model_sha = deploy.sha256(self.source / "model.safetensors")
        self.model_bytes = (self.source / "model.safetensors").stat().st_size
        model_stat = deploy.stat_identity(self.source / "model.safetensors")
        self.receipt = dict(contract=deploy.BACKUP_CONTRACT, status="passed", task_id=deploy.TASK_ID,
            step=5000, training_completed=True, checkpoint_save_completed=True, optimizer_copied=False,
            model=dict(source_path=deploy.EXPECTED_PROJECT + "/checkpoints/ckpt_5000/model.safetensors", bytes=self.model_bytes,
                source_sha256=self.model_sha, local_sha256=self.model_sha, source_stat_before=dict(model_stat),
                source_stat_after=dict(model_stat), local_stat=dict(model_stat)),
            metadata={}, run_identity=dict(run_directory=deploy.EXPECTED_PROJECT, prepared_root=deploy.EXPECTED_ROOT,
                run_config_sha256=deploy.sha256(self.source / "run_config.json"),
                preparation_sha256=deploy.sha256(self.source / "preparation.json")))
        for name in deploy.METADATA:
            self.rebind(name)
        self.patchers = []
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patchers)])
        self.start(patch.object(deploy, "EXPECTED_MODEL_BYTES", self.model_bytes))
        self.start(patch.object(deploy, "PACK_SHA", deploy.sha256(self.source / "release_metadata/packed/PACK_READY.json")))
        self.start(patch.object(deploy, "STAT_SHA", f.stat_sha))
        self.start(patch.object(deploy.generic, "RESOURCES_MANIFEST_SHA", deploy.sha256(self.resources / "manifest.json")))

    def start(self, patcher):
        self.patchers.append(patcher)
        return patcher.start()

    def write(self, name, value):
        deploy.write_json(self.source / name, value)

    def rebind(self, name):
        path = self.source / name
        self.receipt["metadata"][name] = dict(source_path=deploy.METADATA_SOURCES[name], bytes=path.stat().st_size,
                                             sha256=deploy.sha256(path))
        if name in ("run_config.json", "preparation.json"):
            self.receipt["run_identity"]["run_config_sha256" if name == "run_config.json" else "preparation_sha256"] = deploy.sha256(path)
        self.write("backup_receipt.json", self.receipt)

    def validate(self, **kwargs):
        return deploy.validate_source(self.source, self.release, kwargs.get("step", 5000), kwargs.get("sha", self.model_sha))

    def runtime_modules(self):
        modules = self.fixture.fake_model_modules()
        runtime = ModuleType("psi.deploy.dual_source_runtime")
        runtime.load_dino_contract = lambda path: deploy.read(path)
        modules["psi.deploy.dual_source_runtime"] = runtime
        return modules

    def test_completed_run_has_independent_bend5000_authority_without_stop_contract(self):
        (self.source / "FINAL_TRAINING_ACCEPTANCE_001.json").unlink()
        result = self.validate()
        self.assertTrue(result[4]["training_completed"])
        self.assertEqual(result[5]["status"], "passed")
        self.assertEqual(len(deploy.METADATA), 24)
        self.assertEqual(deploy.local_release(self.release / "packed"), self.release)

    def test_mobile_contract_wrong_task_unsaved_or_boolean_step_rejected(self):
        for key, value in (("contract", "cvpr-mobile-pp-official100-stopped-checkpoint-backup-v1"),
                           ("task_id", "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0"),
                           ("training_completed", False), ("checkpoint_save_completed", False)):
            original = self.receipt[key]
            self.receipt[key] = value
            self.write("backup_receipt.json", self.receipt)
            with self.assertRaisesRegex(ValueError, "independent model-only"):
                self.validate()
            self.receipt[key] = original
        self.write("backup_receipt.json", self.receipt)
        for step in (True, 10000, 6000):
            with self.assertRaisesRegex(ValueError, "explicitly selected"):
                self.validate(step=step)

    def test_actual_remote_transmission_sha_and_stable_source_stat_are_mandatory(self):
        self.receipt["model"]["source_sha256"] = "0" * 64
        self.write("backup_receipt.json", self.receipt)
        with self.assertRaisesRegex(ValueError, "transmitted-source"):
            self.validate()
        self.receipt["model"]["source_sha256"] = self.model_sha
        self.receipt["model"]["source_stat_after"]["st_mtime_ns"] += 1
        self.write("backup_receipt.json", self.receipt)
        with self.assertRaisesRegex(ValueError, "changed during transmission"):
            self.validate()

    def test_metadata_source_path_and_bytes_cannot_be_rebound_to_other_run(self):
        self.receipt["metadata"]["run_config.json"]["source_path"] = "/other/bend/run_config.json"
        self.write("backup_receipt.json", self.receipt)
        with self.assertRaisesRegex(ValueError, "metadata path/bytes/SHA"):
            self.validate()
        self.rebind("run_config.json")
        path = self.source / "run_config.json"
        path.write_bytes(path.read_bytes()+b" ")
        with self.assertRaisesRegex(ValueError, "metadata path/bytes/SHA"):
            self.validate()

    def test_running_state_or_missing_migration_cannot_claim_completed_training(self):
        self.state.update(state="running", returncode=None)
        self.write("training_state.json", self.state)
        self.rebind("training_state.json")
        with self.assertRaisesRegex(ValueError, "has not completed"):
            self.validate()
        self.state.update(state="complete", returncode=0)
        self.write("training_state.json", self.state)
        self.rebind("training_state.json")
        migration = deploy.read(self.source / "state_policy_migration_rank_7.json")
        migration["unchanged_learned_parameters"] = False
        self.write("state_policy_migration_rank_7.json", migration)
        self.rebind("state_policy_migration_rank_7.json")
        with self.assertRaisesRegex(ValueError, "migration proof"):
            self.validate()

    def test_rtc_and_visual_training_semantics_cannot_be_rewritten(self):
        for key, value in (("rtc", True), ("state_drop_prob", .5)):
            changed = deepcopy(self.config)
            changed["model"][key] = value
            with self.assertRaises(ValueError):
                deploy.validate_training(self.source, changed, self.preparation)
        contract = deploy.generic.bind_training_motion_image_policy(self.fixture.reference_contract, self.config)
        self.assertEqual(contract["motion_image_policy"], "vlm_color_clean_motion")
        changed = deepcopy(self.config)
        changed["data"]["transform"]["model"]["motion_image_policy"] = "shared_rgb"
        with self.assertRaisesRegex(ValueError, "vlm_color_clean_motion"):
            deploy.generic.bind_training_motion_image_policy(contract, changed)

    def test_real_tiny_cpu_prepare_hardlinks_joint_and_exports_selected_motion(self):
        with patch.dict(sys.modules, self.runtime_modules()), patch.object(deploy.snapshot, "validate_config"), \
                patch.object(deploy.generic, "validate_source", side_effect=AssertionError("old Mobile gate invoked")):
            report = deploy.prepare(self.source, self.resources, self.output, self.repo, self.release, 5000, self.model_sha)
        torch = fixture_module.torch
        exported = torch.load(self.output / "motion/joint_motion.ckpt", map_location="cpu", weights_only=False)
        self.assertTrue(torch.equal(exported["state_dict"]["weight"], torch.full((2, 2), 9.)))
        self.assertFalse(torch.equal(exported["state_dict"]["weight"], torch.full((2, 2), -7.)))
        target = self.output / "checkpoints/ckpt_5000/model.safetensors"
        self.assertEqual(target.stat().st_ino, (self.source / "model.safetensors").stat().st_ino)
        self.assertTrue(report["training_completed"])
        self.assertFalse(report["selected_checkpoint_is_final"])
        self.assertNotIn("stopped_by_user", report)
        self.assertEqual(report["contract"], deploy.CONTRACT)
        self.assertEqual(report["task_id"], deploy.TASK_ID)
        self.assertFalse(deploy.read(self.output / "run_config.json")["model"]["rtc"])
        self.assertEqual(deploy.read(self.output / "dino_training_contract.json")["motion_image_policy"], "vlm_color_clean_motion")
        self.assertTrue(report["component_load_validation"]["strict_VLM_load"])
        self.assertTrue(report["component_load_validation"]["strict_action_load"])
        self.assertFalse(report["previous_motion_parameters_used"])
        self.assertFalse(report["GPU_used"])
        self.assertEqual(set(report["deployment_artifacts"]), set(deploy.DEPLOYMENT_ARTIFACTS))
        for name, identity in report["deployment_artifacts"].items():
            self.assertEqual(identity["sha256"], deploy.sha256(self.output / name))

    def test_source_and_existing_outputs_are_preserved(self):
        self.output.mkdir()
        marker = self.output / "untouched"
        marker.write_bytes(b"existing")
        with self.assertRaisesRegex(ValueError, "fresh independent"):
            deploy.prepare(self.source, self.resources, self.output, self.repo, self.release, 5000, self.model_sha)
        self.assertEqual(marker.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
