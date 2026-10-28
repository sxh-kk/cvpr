"""CPU checks for the new training control path; no model/data/GPU loading."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import a100_mobile_pp95_eval5 as launch


def fixtures():
    reference = {"exp": "old", "timestamp": "old", "seed": 42,
        "train": {"output_dir": "/old", "resume_from_checkpoint": None, "skip_resumed_steps": False,
            "max_training_steps": 10000, "warmup_steps": 1000, "train_batch_size": 8,
            "gradient_accumulation_steps": 1, "validation_steps": 1000, "max_checkpoints_to_keep": 100,
            "checkpointing_steps": 1000, "early_checkpoint_steps": [], "preserve_checkpoint_every": 1000,
            "learning_rate": 1e-5, "mixed_precision": "bf16"},
        "data": {"root_dir": "/old/data", "dino_cache_root": "/old/dino",
                 "transform": {"field": {"stat_path": "/old/stats"}}},
        "model": {"pretrained_joint_checkpoint": "/source/joint38000/model.safetensors", "state_drop_prob": 0.}}
    plan = {"contract": launch.CONTRACT, "task_id": launch.TASK_ID, "fresh_run": True,
        "initial_global_step": 0, "optimizer_scheduler_rng_restored": False, "source_step": 38000,
        "source_checkpoint_sha256": launch.SOURCE_SHA,
        "source_checkpoint": reference["model"]["pretrained_joint_checkpoint"],
        "gpus": list(range(8)), "gpu_count": 8, "effective_global_batch": 64, "per_gpu_batch": 8,
        "gradient_accumulation_steps": 1, "peak_learning_rates": launch.original.PEAK_LRS,
        "max_optimizer_steps": 20000, "warmup_steps": 1000, "train_episodes": 95, "validation_episodes": 5,
        "validation_role": launch.VALIDATION_ROLE, "diagnostic_steps": list(launch.DIAGNOSTIC_STEPS),
        "checkpoint_policy": {"model_steps": list(launch.SAVE_STEPS)},
        "release": "/fixture/release", "timestamp": "fresh"}
    root, config = Path("/independent/newrun"), copy.deepcopy(reference)
    for name, value in launch.config_overrides(root, Path(plan["release"]), plan["timestamp"]).items():
        cursor = config
        parts = name.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return reference, plan, root, config


class ConfigTests(unittest.TestCase):
    def test_exact_inheritance_and_only_new_split_budget_save_changes(self):
        reference, plan, root, config = fixtures()
        launch.validate_config(config, plan, reference, root)
        self.assertEqual(config["model"], reference["model"])
        self.assertEqual(config["train"]["max_training_steps"], 20000)
        self.assertEqual(config["train"]["validation_steps"], 0)
        self.assertEqual(config["train"]["warmup_steps"], 1000)

    def test_old_mobile_weight_resume_lr_or_role_drift_rejected(self):
        reference, plan, root, config = fixtures()
        for section, key, value in (("model", "pretrained_joint_checkpoint", "/mobile100/ckpt_10000/model.safetensors"),
                                   ("train", "resume_from_checkpoint", "latest"),
                                   ("train", "learning_rate", 2e-5), ("train", "mixed_precision", "fp16"),
                                   ("train", "validation_steps", 1000)):
            bad = copy.deepcopy(config)
            bad[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.validate_config(bad, plan, reference, root)
        for key, value in (("initial_global_step", 10000), ("train_episodes", 100),
                           ("validation_role", "old_handover"), ("optimizer_scheduler_rng_restored", True),
                           ("max_optimizer_steps", 10000)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.validate_config(config, dict(plan, **{key: value}), reference, root)

    def test_real_production_save_trigger_is_exactly_33_models(self):
        reference, plan, root, config = fixtures()
        path = SCRIPTS.parents[1] / "VLMExpert/src/psi/trainers/trainer.py"
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "should_save_training_checkpoint")
        scope = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        cfg = config["train"]
        saved = [step for step in range(1, 20001) if scope[node.name](step,
            cfg["checkpointing_steps"], cfg["max_training_steps"], tuple(cfg["early_checkpoint_steps"]))]
        self.assertEqual(saved, list(launch.SAVE_STEPS))
        self.assertEqual(len(saved), 33)
        self.assertEqual(saved[-2:], [19500, 20000])

    def test_restricted_inspector_must_not_launch_training(self):
        plan = {"training_cpus": list(range(128)), "training_nice": 0}
        with patch.object(launch.original.os, "sched_getaffinity", return_value={127}), \
             patch.object(launch.original.os, "getpriority", return_value=19), self.assertRaises(ValueError):
            launch.original.require_launch_runtime(plan)

    def test_launch_rehashes_joint_weights_before_gpu_or_starting_state(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.safetensors"
            model.write_bytes(b"changed joint model bytes")
            plan = {"source_checkpoint": str(model)}
            upstream = SimpleNamespace(selected_gpu_idle=Mock())
            with patch.object(launch, "load", return_value=(None, plan, None, upstream)), \
                 patch.object(launch, "require_cpu_check"), \
                 patch.object(launch.original, "require_launch_runtime", return_value={}):
                with self.assertRaisesRegex(ValueError, "joint38000 bytes"):
                    launch.run(root)
            upstream.selected_gpu_idle.assert_not_called()
            self.assertFalse((root / "training_state.json").exists())
            self.assertFalse((root / "launch_acceptance.json").exists())


class TrainerHookTests(unittest.TestCase):
    def test_baseline_then_exact_completed_updates_and_no_extra_evaluate(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            model = SimpleNamespace(action_header=SimpleNamespace(state_drop_prob=0,
                config=SimpleNamespace(state_drop_prob=0), last_state_drop_frac=0))
            class Base:
                def __init__(self):
                    self.project_dir = directory
                    self.accelerator = SimpleNamespace(is_main_process=True)
                    self.model_cfg = SimpleNamespace(state_drop_prob=0)
                def prepare(self, accelerator):
                    calls.append("prepared")
                    return "loader"
                def step(self, batch, global_step, local_step):
                    calls.append(("update", global_step + 1))
                    return True, {"loss": .5}
                def unwrap_model(self):
                    return model
                def log(self, metrics):
                    calls.append("baseline_logged")
            def diagnostic(trainer, bundle, completed, split):
                calls.append(("diagnostic", completed))
                return {"completed_update": completed, "split_sha256": split, "metrics": {"val_full_fm_loss_micro": 1.}}
            plan = {"migration_helper": "/unused", "split_sha256": "a" * 64}
            with patch.object(launch, "helper", return_value=SimpleNamespace(evaluate_diagnostics=diagnostic)), \
                 patch.object(launch, "module", return_value=SimpleNamespace()), \
                 patch.object(torch.distributed, "broadcast_object_list", side_effect=lambda value, src: None):
                cls = launch.trainer_class(plan, SimpleNamespace(), SimpleNamespace(trainer_class=lambda plan: Base))
                trainer = cls()
                self.assertEqual(trainer.prepare(trainer.accelerator), "loader")
                batch = {"states": torch.zeros(8, 1, 45)}
                for completed in [1, 999, 1000, 1001, 19999, 20000]:
                    trainer.step(batch, completed - 1, 0)
                self.assertEqual([v for v in calls if isinstance(v, tuple) and v[0] == "diagnostic"],
                                 [("diagnostic", 0), ("diagnostic", 1000), ("diagnostic", 20000)])
                self.assertLess(calls.index("prepared"), calls.index(("diagnostic", 0)))
                rows = [json.loads(line) for line in (Path(directory) / "mobile_diagnostics/curve.jsonl").read_text().splitlines()]
                self.assertEqual([row["completed_update"] for row in rows], [0, 1000, 20000])
                with self.assertRaises(ValueError):
                    trainer.save_checkpoint(2000)
                with self.assertRaises(ValueError):
                    trainer.record_diagnostics(1000)


if __name__ == "__main__":
    unittest.main()
