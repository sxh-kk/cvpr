"""CPU-only contract and local deployment checks with tiny fake payloads."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("prepare_task_snapshot", SCRIPTS / "prepare_task_snapshot.py")
task = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task)


def fixture():
    previous = {
        "model": {"action_architecture": "dual_source_v1", "action_dim": 80,
                  "odim": 45, "action_chunk_size": 30, "num_blocks": 6,
                  "model_name_or_path": "/a100/qwen", "motion_checkpoint": "/a100/motion",
                  "tune_vlm": True, "tune_motion": True,
                  **task.PRETRAIN_LRS},
        "data": {"root_dir": "/a100/unifolm", "dino_cache_root": "/a100/dino",
                 "train_repo_ids": ["unifolm_train"], "val_repo_ids": ["unifolm_val"],
                 "transform": {"repack": {"action_keys": ["action.body_token_v1_1", "action[:14]"],
                     "state_keys": ["observation.state"], "pad_action_dim": 80, "pad_state_dim": 45},
                     "model": {"shared_image_profile": "256x320", "motion_image_policy": "vlm_color_clean_motion"},
                     "field": {"stat_path": "/a100/originalstats", "action_min": [0, 1],
                               "action_max": [1, 2], "state_min": [0], "state_max": [1]}}},
        "train": {"resume_from_checkpoint": None, "output_dir": "/a100/output",
                  "deepspeed_config": "/a100/repo/zero3.json"}}
    original = copy.deepcopy(previous)
    original["model"].update(task.TASK_LRS)
    original["model"]["pretrained_joint_checkpoint"] = "/a100/run/checkpoints/ckpt_38000/model.safetensors"
    original["data"].update(root_dir="/a100/handover_task50_30hz", dino_cache_root="/a100/dino/handover_task50_30hz",
                            train_repo_ids=["Handover-sonic11-train"], val_repo_ids=["Handover-sonic11-val"])
    original["data"]["transform"]["field"]["stat_path"] = "/a100/run/dataset_statistics.json"
    statistics = {"action": {"min": [0, 1], "max": [1, 2]}}
    identity = {"step": 6000, "task_step": 6000, "source_unifolm_step": 38000,
                "initial_task_step": 0, "save_completion_marker_verified": True,
                "training_modified": False, "dataset_contract": {
                    "contract": task.TASK_CONTRACT, "fps": 30, "train_episodes": 45,
                    "val_episodes": 5, "successful_teacher_episodes": 50}}
    initialization = {"source_step": 38000, "strict_joint_loader": True,
                      "fresh_optimizer_scheduler_rng": True, "global_batch": 32,
                      "source_checkpoint": original["model"]["pretrained_joint_checkpoint"]}
    fused = {"mode": "fresh_fused_restart_from38000", "source_step": 38000,
             "initial_global_step": 0, "optimizer_scheduler_rng_restored": False}
    return original, previous, statistics, copy.deepcopy(statistics), identity, initialization, fused


class TaskSnapshotTests(unittest.TestCase):
    def test_accepts_only_the_accepted_task_differences(self):
        values = fixture()
        values[4]["dataset_contract"]["evidence"] = "retained original acceptance record"
        before = copy.deepcopy(values)
        self.assertEqual(task.validate_contract(*values), 6000)
        self.assertEqual(values, before)

    def test_rejects_architecture_change(self):
        values = fixture()
        values[0]["model"]["num_blocks"] = 7
        with self.assertRaisesRegex(ValueError, "architecture/construction"):
            task.validate_contract(*values)

    def test_rejects_statistics_and_inline_bounds_changes(self):
        values = fixture()
        values[2]["action"]["min"][0] = -1
        with self.assertRaisesRegex(ValueError, "normalization"):
            task.validate_contract(*values)
        values = fixture()
        values[0]["data"]["transform"]["field"]["state_max"] = [9]
        with self.assertRaisesRegex(ValueError, "bounds"):
            task.validate_contract(*values)

    def test_rejects_token_version_or_visual_policy_changes(self):
        for section, key, value in (("repack", "action_keys", ["action.body_token", "action[:14]"]),
                                    ("model", "motion_image_policy", "shared_rgb")):
            values = fixture()
            values[0]["data"]["transform"][section][key] = value
            with self.assertRaisesRegex(ValueError, "contract changed"):
                task.validate_contract(*values)

    def test_rejects_unrelated_dataset_and_unproven_30hz(self):
        values = fixture()
        values[0]["data"]["train_repo_ids"] = ["Another-task"]
        with self.assertRaisesRegex(ValueError, "audited Handover"):
            task.validate_contract(*values)
        values = fixture()
        values[4]["dataset_contract"]["fps"] = 50
        with self.assertRaisesRegex(ValueError, "task50/30Hz"):
            task.validate_contract(*values)

    def test_rejects_mixed_task_and_unifolm_counters_or_incomplete_save(self):
        for key, value in (("step", 38000), ("source_unifolm_step", 40000),
                           ("initial_task_step", 2345), ("task_step", True),
                           ("save_completion_marker_verified", False), ("training_modified", True)):
            values = fixture()
            values[4][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "save-complete"):
                task.validate_contract(*values)

    def test_rejects_missing_three_module_initialization_evidence(self):
        for index, key, value in ((5, "strict_joint_loader", False),
                                  (6, "initial_global_step", 2345),
                                  (6, "optimizer_scheduler_rng_restored", True)):
            values = fixture()
            values[index][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "initialization evidence"):
                task.validate_contract(*values)

    def test_remaps_both_argument_forms_and_clears_old_joint_initialization(self):
        original = fixture()[0]
        argv = "train.py posttrain_sonic_psi0_config --model.pretrained-joint-checkpoint /a100/38k "
        argv += "--model.motion_checkpoint=/a100/motion --model.num-blocks=6 --model.no-rtc"
        replacements = {"model.pretrained_joint_checkpoint": None, "model.motion_checkpoint": "/local/newmotion"}
        config, rewritten = task.remap_config(original, argv, replacements)
        self.assertIsNone(config["model"]["pretrained_joint_checkpoint"])
        self.assertIn("--model.pretrained_joint_checkpoint=None", rewritten)
        self.assertIn("--model.motion_checkpoint=/local/newmotion", rewritten)
        self.assertIn("--model.num-blocks=6\n--model.no-rtc", rewritten)
        self.assertNotIn("/a100", rewritten)
        self.assertEqual(original["model"]["motion_checkpoint"], "/a100/motion")
        with self.assertRaisesRegex(ValueError, "Missing value"):
            task.remap_config(original, "--model.motion_checkpoint --model.no-rtc", replacements)

    def test_prepare_new_motion_from_task_joint_with_explicit_counter_semantics(self):
        values = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, resources, output, repo = [root / name for name in ("source", "resources", "deployment", "repo")]
            source.mkdir(); repo.mkdir()
            base = resources / "joint20k"
            (base / "provenance").mkdir(parents=True)
            for path in (base / "qwen", resources / "dinov2-with-registers-large", resources / "simple_local_usd_v1"):
                path.mkdir(parents=True)
            (base / "motion").mkdir()
            (base / "motion/joint_motion.ckpt").write_bytes(b"old metadata, not task weights")
            model = source / "model.safetensors"
            model.write_bytes(b"new task VLM Motion Action weights")
            values[4].update(model_bytes=model.stat().st_size, model_sha256=task.sha256(model))
            names = ("run_config.json", "dataset_statistics.json", "SNAPSHOT_READY.json",
                     "task_initialization.json", "task_fused_restart_initialization.json")
            for name, value in zip(names, (values[0], values[2], values[4], values[5], values[6])):
                task.write_json(source / name, value)
            (source / "argv.txt").write_text("train.py posttrain_sonic_psi0_config\n")
            task.write_json(base / "provenance/run_config.json", values[1])
            task.write_json(base / "dataset_statistics.json", values[3])
            task.write_json(base / "dino_training_contract.json", {"encoder": {"source": "immutable"}})
            def export(metadata, joint, target, label, encoder):
                self.assertEqual(joint.read_bytes(), model.read_bytes())
                self.assertEqual(label, "38000_then_task_finetune_6000")
                self.assertEqual(encoder, {"source": "immutable"})
                target.parent.mkdir(parents=True)
                target.write_bytes(b"extracted new Motion")
                return {"weights": f"joint_posttrain_step_{label}", "strict_meta_load": "passed"}
            with patch.object(task, "export_motion", side_effect=export), patch.object(task, "validate_config") as validator:
                report = task.prepare(source, resources, output, repo)
                validator.assert_called_once()
            self.assertEqual(report["task_step"], 6000)
            self.assertEqual(report["source_unifolm_step"], 38000)
            self.assertEqual(report["gpu_validation"], "not_run")
            self.assertEqual(model.stat().st_ino, (output / "checkpoints/ckpt_6000/model.safetensors").stat().st_ino)
            config = json.loads((output / "run_config.json").read_text())
            self.assertIsNone(config["model"]["pretrained_joint_checkpoint"])
            self.assertEqual(config["model"]["motion_checkpoint"], str(output / "motion/joint_motion.ckpt"))
            self.assertTrue((output / "provenance/task_initialization.json").exists())
            with self.assertRaisesRegex(ValueError, "fresh output"):
                task.prepare(source, resources, output, repo)


if __name__ == "__main__":
    unittest.main()
