"""CPU configuration and production-group invariants; these do not claim GPU success."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_liftbox_joint_gpu as check
import finetune_carrybox as shared


def alignment():
    model = dict(action_architecture="dual_source_v1", tune_vlm=True, tune_motion=True,
                 tune_mm_llm=False, tune_mm_mlp=False, tune_mm_vision=False,
                 lang_backbone_lr=1e-6, vision_tower_lr=1e-5, mm_projector_lr=1e-4,
                 motion_expert_lr=1e-5)
    lrs = [1e-4, 1e-6, 1e-5, 1e-4, 1e-5]
    return {"configuration": {"model": model, "train": {"learning_rate": 1e-4}},
            "optimizer_groups": [{"name": name, "peak_lr": lr, "trainable_parameters": i+2}
                                 for i, (name, lr) in enumerate(zip(check.GROUPS, lrs))]}


class JointCheckCPU(unittest.TestCase):
    def test_payload_byte_binding_rejects_changed_payload_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "features.npy"
            payload.write_bytes(b"accepted-cache-bytes")
            sha = shared.sha256(payload)
            self.assertEqual(check.verify_payload_bytes(root, {"features.npy": sha})["checked_files"], 1)
            payload.write_bytes(b"changed-cache-bytes")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                check.verify_payload_bytes(root, {"features.npy": sha})
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                check.verify_payload_bytes(root, {str(payload): shared.sha256(payload)})

    def test_five_actual_lrs_and_joint_flags_small_scope(self):
        overrides = check.joint_overrides(alignment(), Path("/tmp/cvpr-joint-fixture"), "fixed")
        self.assertTrue(overrides["model.tune_vlm"])
        self.assertTrue(overrides["model.tune_motion"])
        self.assertTrue(overrides["model.gradient_checkpointing"])
        self.assertTrue(overrides["train.training_safety_checks"])
        for component in ("llm", "vision", "mlp"):
            self.assertFalse(overrides["model.tune_mm_"+component])
        self.assertEqual((overrides["train.max_training_steps"], overrides["train.train_batch_size"],
                          overrides["train.gradient_accumulation_steps"]), (2,1,1))
        self.assertEqual(overrides["train.lr_scheduler_type"], "constant")
        for row in alignment()["optimizer_groups"]:
            self.assertEqual(overrides[check.LR_FIELDS[row["name"]]], row["peak_lr"])
        self.assertFalse(any(key.startswith("data.") for key in overrides))

    def test_malformed_remote_alignment_is_not_silently_defaulted(self):
        for mutate in (lambda a: a["optimizer_groups"].pop(),
                       lambda a: a["optimizer_groups"][0].update(peak_lr=float("nan")),
                       lambda a: a["configuration"]["model"].update(tune_motion=False),
                       lambda a: a["configuration"]["model"].update(tune_mm_vision=True),
                       lambda a: a["optimizer_groups"][1].update(trainable_parameters=0)):
            data = alignment(); mutate(data)
            with self.assertRaises(ValueError):
                check.alignment_groups(data)

    def test_rewrite_removes_old_freeze_flags_without_changing_data(self):
        original = ["train.py", shared.SCHEMA, "--model.no-tune-vlm", "--model.no-tune-motion",
                    "--data.train-repo-ids", "train", "--data.transform.repack.action-keys", "action.body_token_v1_1", "action",
                    "--train.learning-rate=1e-5"]
        argv = shared.rewrite_argv(original, check.joint_overrides(alignment(), Path("/tmp/joint"), "fixed"))
        self.assertNotIn("--model.no-tune-vlm", argv)
        self.assertNotIn("--model.no-tune-motion", argv)
        self.assertIn("--model.tune-vlm", argv)
        self.assertIn("--model.tune-motion", argv)
        start = argv.index("--data.transform.repack.action-keys")
        self.assertEqual(argv[start:start+3], ["--data.transform.repack.action-keys", "action.body_token_v1_1", "action"])

    def tiny_model(self):
        import torch
        model = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(i+2)) for i in range(5)])
        groups = [{"params": [param], "group_name": name, "lr": row["peak_lr"]}
                  for param, name, row in zip(model, check.GROUPS, alignment()["optimizer_groups"])]
        return model, torch.optim.AdamW(groups)

    def test_five_groups_require_exact_trainable_coverage_count_and_lr(self):
        model, optimizer = self.tiny_model()
        expected = check.alignment_groups(alignment())
        self.assertEqual(len(check.check_optimizer_groups(model, optimizer, expected, device_type="cpu")),5)
        optimizer.param_groups[0]["params"].append(optimizer.param_groups[1]["params"][0])
        with self.assertRaisesRegex(ValueError,"exactly once"):
            check.check_optimizer_groups(model,optimizer,expected,device_type="cpu")
        model, optimizer = self.tiny_model()
        optimizer.param_groups[0]["lr"] = 1e-8
        with self.assertRaisesRegex(ValueError,"counts/LR"):
            check.check_optimizer_groups(model,optimizer,expected,device_type="cpu")

    def test_all_five_hashes_and_nonfinite_post_update_rejection(self):
        import torch
        model, optimizer = self.tiny_model()
        before = check.group_parameter_hashes(model,optimizer)
        for parameter in model:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        after = check.group_parameter_hashes(model,optimizer)
        self.assertTrue(all(before[name] != after[name] for name in check.GROUPS))
        with torch.no_grad():
            model[0][0] = float("nan")
        with self.assertRaises(FloatingPointError):
            check.group_parameter_hashes(model,optimizer)

    def test_production_norm_and_loss_receipt_rejects_missing_zero_or_nan(self):
        norms = {name:1.0 for name in check.GROUPS}
        check.validate_update_metrics({"loss":1},norms)
        for bad in (float("nan"),float("inf"),0.0):
            changed=copy.deepcopy(norms);changed["motion_expert"]=bad
            with self.assertRaises(FloatingPointError):
                check.validate_update_metrics({"loss":1},changed)
        with self.assertRaises(FloatingPointError):
            check.validate_update_metrics({"loss":float("nan")},norms)
        norms.pop("vision_tower")
        with self.assertRaises(FloatingPointError):
            check.validate_update_metrics({"loss":1},norms)


if __name__ == "__main__":
    unittest.main()
