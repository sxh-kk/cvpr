"""Task adaptation must retain the learned head and independent validation."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location("a100_task_finetune", DIRECTORY / "a100_task_finetune.py")
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


class TaskConfigurationTests(unittest.TestCase):
    def test_joint_initialization_keeps_all_three_components_and_new_optimizer(self):
        values = task.overrides(Path("/source"), Path("/data"), Path("/dino"), Path("/output"))
        self.assertEqual(values["model.pretrained_joint_checkpoint"], "/source/checkpoints/ckpt_40000/model.safetensors")
        self.assertIsNone(values["model.pretrained_action_header_path"])
        self.assertIsNone(values["train.resume_from_checkpoint"])
        self.assertTrue(values["model.tune_vlm"])
        self.assertTrue(values["model.tune_motion"])
        self.assertEqual(values["train.train_batch_size"] * 8 * values["train.gradient_accumulation_steps"], 32)
        self.assertEqual(values["train.max_training_steps"], 2000)
        self.assertEqual(values["data.transform.field.stat_path"], "/source/dataset_statistics.json")
        self.assertNotIn("model.action_architecture", values)
        self.assertNotIn("data.transform.repack.action_keys", values)
        self.assertNotIn("data.transform.field.action_min", values)
        # This otherwise-unused DDP field is still serialized and compared by
        # parent/worker parsers; a relative path changes with their different cwd.
        path = Path(values["train.deepspeed_config"])
        self.assertTrue(path.is_absolute())
        self.assertTrue(path.is_file())

    def test_five_small_group_learning_rates_are_explicit(self):
        values = task.overrides("/source", "/data", "/dino", "/output")
        self.assertEqual(values["train.learning_rate"], 1e-5)
        self.assertEqual(values["model.lang_backbone_lr"], 1e-7)
        self.assertEqual(values["model.vision_tower_lr"], 1e-6)
        self.assertEqual(values["model.mm_projector_lr"], 1e-5)
        self.assertEqual(values["model.motion_expert_lr"], 1e-6)

    def test_preserves_accepted_split_and_no_previous_wandb_identity(self):
        values = task.overrides("/source", "/data", "/dino", "/output")
        self.assertEqual(values["data.train_repo_ids"], ["Handover-sonic11-train"])
        self.assertEqual(values["data.val_repo_ids"], ["Handover-sonic11-val"])
        self.assertEqual(values["wandb.resume"], "never")
        self.assertIsNone(values["wandb.id"])
        self.assertEqual(values["log.report_to"], "wandb")

    def test_impossible_warmup_and_zero_budget_are_rejected(self):
        for steps, warmup in ((0, 0), (1, 1), (20, 20), (20, -1)):
            with self.subTest(steps=steps, warmup=warmup), self.assertRaises(ValueError):
                task.overrides("/source", "/data", "/dino", "/output", max_steps=steps, warmup=warmup)

    def test_fixed_validation_covers_all_episodes_and_divides_across_eight_ranks(self):
        candidates = {i: list(range(i * 100, i * 100 + 35)) for i in range(5)}
        selected = task.fixed_validation(candidates)
        self.assertEqual(len(selected), 40)
        self.assertEqual(len(set(selected)), 40)
        self.assertEqual(len(selected) % 8, 0)
        for i, rows in candidates.items():
            self.assertEqual(sum(row in rows for row in selected), 8)
            self.assertIn(rows[0], selected)
            self.assertIn(rows[-1], selected)

    def test_changed_data_cannot_reuse_previous_cpu_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "dataset"
            data.mkdir()
            task.shared.write_json(data / "ADAPTATION_READY.json", {"status": "example"})
            payload = data / "example.parquet"
            payload.write_bytes(b"original")
            task.shared.write_json(root / "data_check.json", {
                "status": "cpu_data_check_passed",
                "adaptation_receipt_sha256": task.shared.sha256(data / "ADAPTATION_READY.json"),
                "artifact_bindings": {str(payload): {"bytes": payload.stat().st_size,
                                                     "mtime_ns": payload.stat().st_mtime_ns}},
            })
            task.shared.write_json(root / "launch_gate.json", {
                "data_check_sha256": task.shared.sha256(root / "data_check.json")})
            plan = {"dataset_root": str(data)}
            self.assertEqual(task.verify_data_check(root, plan)["status"], "cpu_data_check_passed")
            (root / "launch_gate.json").unlink()
            self.assertEqual(task.verify_data_check(root, plan, require_launch_gate=False)["status"],
                             "cpu_data_check_passed")
            payload.write_bytes(b"changed_payload")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                task.verify_data_check(root, plan)


if __name__ == "__main__":
    unittest.main()
