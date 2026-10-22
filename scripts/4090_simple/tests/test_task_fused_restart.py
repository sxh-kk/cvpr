"""CPU contracts for the explicitly fresh38k fused task and recovery saves."""
import ast
import copy
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel

DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIRECTORY))
_SPEC = importlib.util.spec_from_file_location("task_fused_restart_under_test", DIRECTORY / "a100_task_fused_restart.py")
restart = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(restart)
task = restart.task


class _Config:
    def __init__(self, values):
        self.values = copy.deepcopy(values)
        self.exp = self.values["exp"]
    def model_dump(self, mode="json"):
        return copy.deepcopy(self.values)


def _base():
    return _Config({"timestamp": "original", "exp": "handover_task50_joint38000", "seed": 20261022,
        "model": {"pretrained_joint_checkpoint": "/source/ckpt_38000/model.safetensors", "tune_vlm": True, "tune_motion": True},
        "data": {"root_dir": "/accepted/data", "dino_cache_root": "/accepted/dino"},
        "train": {"output_dir": "/parent/training", "max_training_steps": 20000, "warmup_steps": 1000,
            "checkpointing_steps": 4000, "validation_steps": 1000, "early_checkpoint_steps": [],
            "max_checkpoints_to_keep": 12, "preserve_checkpoint_every": None,
            "train_batch_size": 4, "gradient_accumulation_steps": 1, "mixed_precision": "bf16",
            "resume_from_checkpoint": None, "skip_resumed_steps": False, "learning_rate": 1e-5}})


def _derived(base, root, timestamp="new"):
    values = base.model_dump()
    values.update(timestamp=timestamp, exp=base.exp + "_fused_restart")
    values["train"].update(output_dir=str(root / "training"), **restart.ARCHIVE)
    return _Config(values)


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def _reports(root, parent_sha="a" * 64, model_sha="b" * 64):
    micro = {"status": "optimizer_microbenchmark_passed", "dtype": "float32", "device": "cuda:0",
        "candidate": "AdamW foreach=False fused=True", "recipe": {"source_sha256": parent_sha, "lrs": task.PEAK_LRS},
        "correctness": {"passed": True}, "post_timing_correctness": {"passed": True}}
    micro_path = root / "micro.json"
    _write(micro_path, micro)
    exact = {"passed": True, "strict_zero_tolerance": True}
    full = {"status": "controlled_optimizer_comparison_passed",
        "same_gradient_optimizer_comparison_passed": True, "postclip_gradients_exactly_equal": True,
        "all_model_and_optimizer_tensors_finite": True, "continuous_fused_real_gradient_steps_passed": True,
        "full_step_comparison_passed": False, "uncontrolled_full_step_comparison_passed": False,
        "source_config_sha256": parent_sha, "gpu_count": 1, "per_gpu_batch": 4,
        "source_checkpoint_files_sha256": {"model.safetensors": model_sha, "optimizer.bin": "c" * 64, "scheduler.bin": "d" * 64},
        "tolerances": {"parameter_moments_atol": 1e-8, "parameter_moments_rtol": 1e-5,
            "gradient_atol": 1e-7, "gradient_rtol": 1e-4, "loss_atol": 1e-6, "adam_steps": "exact"},
        "baseline_repetition": {"diagnostic_only": True, "passed": False, "comparison_complete": True},
        "same_gradient_comparisons": [{"passed": True, "execution_order": order,
            "preclip_replay": exact, "postclip_gradient_equality": exact}
            for order in (["standard", "fused"], ["fused", "standard"])],
        "additional_fused_steps": [{"all_parameters_and_adam_states_finite": True, "gradient_source": "own_real_backward",
                                    "restored_since_previous_step": index == 0} for index in range(3)],
        "peak_memory": {"peak_allocated_bytes": 100, "peak_reserved_bytes": 200, "gpu_total_bytes": 1000,
                        "minimum_device_free_bytes_after_update": 700},
        "microbenchmark_report": {"path": str(micro_path), "sha256": task.shared.sha256(micro_path), "status": micro["status"]}}
    full_path = root / "controlled.json"
    _write(full_path, full)
    return full_path, micro_path, full


def _production_function(name):
    """Execute the actual pure checkpoint predicate/retention body, not a copy.

    Importing the entire model trainer is unnecessary for this filesystem-only
    contract. AST extraction preserves its implementation and annotations.
    """
    source = task.REPO / "VLMExpert/src/psi/trainers/trainer.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"os": os, "shutil": shutil, "overwatch": Mock()}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


class FreshFusedRestartTests(unittest.TestCase):
    def test_only_naming_output_and_archive_policy_can_change(self):
        base, root = _base(), Path("/new/restart")
        restart.require_parent(base, {"source_step": 38000})
        derived = _derived(base, root)
        restart.require_config_delta(derived, base, root, "new")
        for category, key, value in (("train", "learning_rate", 2e-5), ("data", "dino_cache_root", "/other"),
                                     ("model", "tune_motion", False), ("train", "preserve_checkpoint_every", 1000)):
            changed = derived.model_dump()
            changed[category][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "outside naming/output"):
                restart.require_config_delta(_Config(changed), base, root, "new")

    def test_parent_cannot_be_resume_or_wrong_source_or_budget(self):
        for key, value in (("max_training_steps", 2000), ("resume_from_checkpoint", "/checkpoint"),
                           ("gradient_accumulation_steps", 2), ("early_checkpoint_steps", [20])):
            changed = _base().model_dump()
            changed["train"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "parent recipe"):
                restart.require_parent(_Config(changed), {"source_step": 38000})
        with self.assertRaises(ValueError):
            restart.require_parent(_base(), {"source_step": 40000})

    def test_gpu_gate_accepts_controlled_scope_without_faking_independent_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = _reports(Path(directory))
            gate = restart.validation_gate(*paths[:2], parent_config_sha256="a" * 64, source_model_sha256="b" * 64)
            self.assertIn("independent full-step equivalence remains unproven", gate["scope"])
            self.assertFalse(paths[2]["full_step_comparison_passed"])

    def test_gpu_gate_rejects_wrong_model_config_old_strict_or_loosened_tolerance(self):
        with tempfile.TemporaryDirectory() as directory:
            controlled, micro, original = _reports(Path(directory))
            changes = [("status", "passed"), ("source_config_sha256", "e" * 64),
                       ("full_step_comparison_passed", True), ("continuous_fused_real_gradient_steps_passed", False)]
            for key, value in changes:
                changed = copy.deepcopy(original)
                changed[key] = value
                _write(controlled, changed)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    restart.validation_gate(controlled, micro, parent_config_sha256="a" * 64, source_model_sha256="b" * 64)
            changed = copy.deepcopy(original)
            changed["tolerances"]["parameter_moments_atol"] = 1e-6
            _write(controlled, changed)
            with self.assertRaisesRegex(ValueError, "tolerances"):
                restart.validation_gate(controlled, micro, parent_config_sha256="a" * 64, source_model_sha256="b" * 64)
            _write(controlled, original)
            with self.assertRaisesRegex(ValueError, "another38k"):
                restart.validation_gate(controlled, micro, parent_config_sha256="a" * 64, source_model_sha256="f" * 64)

    def test_gpu_gate_requires_exact_postclip_preclip_and_real_consecutive_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            controlled, micro, original = _reports(Path(directory))
            mutations = [lambda d: d["same_gradient_comparisons"][0]["postclip_gradient_equality"].update(strict_zero_tolerance=False),
                lambda d: d["same_gradient_comparisons"][1]["preclip_replay"].update(passed=False),
                lambda d: d["same_gradient_comparisons"][1].update(execution_order=["standard", "fused"]),
                lambda d: d["baseline_repetition"].update(comparison_complete=False),
                lambda d: d.update(additional_fused_steps=d["additional_fused_steps"][:2]),
                lambda d: d["additional_fused_steps"][0].update(gradient_source="replayed_gradient"),
                lambda d: d["additional_fused_steps"][0].update(restored_since_previous_step=False),
                lambda d: d["additional_fused_steps"][1].update(restored_since_previous_step=True),
                lambda d: d["peak_memory"].update(minimum_device_free_bytes_after_update=0)]
            for mutate in mutations:
                changed = copy.deepcopy(original)
                mutate(changed)
                _write(controlled, changed)
                with self.assertRaises(ValueError):
                    restart.validation_gate(controlled, micro, parent_config_sha256="a" * 64, source_model_sha256="b" * 64)
            _write(controlled, original)
            micro.write_text(micro.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "exact CUDA microbenchmark"):
                restart.validation_gate(controlled, micro, parent_config_sha256="a" * 64, source_model_sha256="b" * 64)

    def test_load_requires_explicit_fused_backend_before_parent_access(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = {"mode": restart.MODE, "fresh_restart": True, "initial_global_step": 0,
                "optimizer_scheduler_rng_restored": False, "task_progress_restored": False,
                "unrecoverable_parent_updates": 2345}
            for backend in (None, "single_tensor", False):
                with self.subTest(backend=backend), patch.object(task, "load_prepared", return_value=(_base(), dict(plan, optimizer_backend=backend))), \
                     self.assertRaisesRegex(ValueError, "step-zero restart"):
                    restart.load(Path(directory))

    def test_prepare_and_reload_bind_failed_parent_evidence_and_copy_acceptance_without_data_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, output = root / "failed", root / "fresh"
            parent.mkdir()
            base = _base()
            _write(parent / "run_config.json", base.model_dump())
            _write(parent / "training_state.json", {"state": "failed", "observed_updates": 2345})
            (parent / "argv.txt").write_text(f"train.py\n{task.shared.SCHEMA}\n")
            (parent / "data_check.json").write_bytes(b'{"accepted":"existing data only"}\n')
            accepted = {"source_step": 38000, "source_checkpoint_sha256": "b" * 64}
            _write(parent / "preparation.json", accepted)
            controlled, micro, _ = _reports(root, parent_sha=task.shared.sha256(parent / "run_config.json"))
            def load_prepared(path):
                path = Path(path)
                return _Config(task.read_json(path / "run_config.json")), task.read_json(path / "preparation.json")
            def parse(path):
                argv = Path(path).read_text().split()
                flags = dict(token.removeprefix("--").split("=", 1) for token in argv if token.startswith("--") and "=" in token)
                self.assertEqual(flags["train.max-checkpoints-to-keep"], "2")
                self.assertEqual(flags["train.preserve-checkpoint-every"], "4000")
                start = argv.index("--train.early-checkpoint-steps") + 1
                recovery = []
                for token in argv[start:]:
                    if token.startswith("--"):
                        break
                    recovery.append(int(token))
                self.assertEqual(recovery, restart.ARCHIVE["early_checkpoint_steps"])
                return _derived(base, output, flags["timestamp"])
            args = SimpleNamespace(parent=parent, output=output, controlled_report=controlled, micro_report=micro)
            with patch.object(task, "load_prepared", side_effect=load_prepared), \
                 patch.object(task.shared, "parse_training_argv", side_effect=parse), \
                 patch.object(task, "verify_data_check") as verify, patch("builtins.print"):
                restart.prepare(args)
                config, plan = restart.load(output)
                self.assertEqual(plan["initial_global_step"], 0)
                self.assertEqual(plan["optimizer_backend"], "fused")
                self.assertFalse(plan["task_progress_restored"])
                self.assertEqual(config.model_dump()["train"]["learning_rate"], base.model_dump()["train"]["learning_rate"])
                self.assertEqual((output / "data_check.json").read_bytes(), (parent / "data_check.json").read_bytes())
                self.assertIn("--prepared", (output / "launch.sh").read_text())
                self.assertGreaterEqual(verify.call_count, 3)
                original_plan = (output / "preparation.json").read_bytes()
                for name in ("preparation.json", "run_config.json", "training_state.json", "data_check.json"):
                    path = parent / name
                    saved = path.read_bytes()
                    path.write_bytes(saved + b" ")
                    with self.subTest(tampered_parent=name), self.assertRaisesRegex(ValueError, "changed"):
                        restart.load(output)
                    path.write_bytes(saved)
                self.assertEqual((output / "preparation.json").read_bytes(), original_plan)

    def test_early20_every1k_recovery_and_all4k_formal_saves_use_production_predicate(self):
        predicate = _production_function("should_save_training_checkpoint")
        selected = [step for step in range(1, 20001) if predicate(step, 4000, 20000, restart.ARCHIVE["early_checkpoint_steps"])]
        self.assertEqual(selected, [20] + list(range(1000, 20001, 1000)))
        self.assertEqual([step for step in selected if step % 4000 == 0], [4000, 8000, 12000, 16000, 20000])
        self.assertTrue(all(step % 4000 for step in restart.ARCHIVE["early_checkpoint_steps"]))

    def test_production_retention_keeps_formal_points_and_only_new_run_recovery_files(self):
        save = _production_function("save_checkpoint")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old_run/checkpoints/ckpt_1000"
            old.mkdir(parents=True)
            (old / "untouched").write_text("old")
            project = root / "new_run"
            def save_state(path):
                Path(path).mkdir()
                (Path(path) / "tiny-payload").write_text("new")
            accelerator = SimpleNamespace(save_state=save_state, wait_for_everyone=Mock(), is_main_process=True)
            trainer = SimpleNamespace(project_dir=str(project), accelerator=accelerator,
                train_cfg=SimpleNamespace(max_checkpoints_to_keep=2, preserve_checkpoint_every=4000))
            for step in [20] + list(range(1000, 20001, 1000)):
                save(trainer, step)
            remaining = sorted(int(p.name[5:]) for p in (project / "checkpoints").iterdir())
            self.assertEqual(remaining, [4000, 8000, 12000, 16000, 19000, 20000])
            self.assertEqual((old / "untouched").read_text(), "old")

    def test_fused_selected_only_after_base_cuda_prepare_on_empty_optimizer(self):
        order = []
        class Base:
            def prepare(self, accelerator):
                order.append("base_prepare")
                self.optimizer = SimpleNamespace(state={})
                return sentinel.loader
        plan = {"source_checkpoint_sha256": "b" * 64, "controlled_gpu_acceptance": {"scope": "controlled"}}
        with patch.object(task, "task_trainer_class", return_value=Base), \
             patch.object(restart, "select_adamw_backend", side_effect=lambda opt, fused: order.append("select_fused") or {"backend": "fused"}) as select, \
             patch.object(restart, "optimizer_state_identity", return_value=[{"fused": True}]), \
             patch.object(task.shared, "write_json") as write:
            trainer = restart.trainer_class(plan)()
            trainer.project_dir = "/external/new"
            self.assertIs(trainer.prepare(SimpleNamespace(is_main_process=True)), sentinel.loader)
            self.assertEqual(order, ["base_prepare", "select_fused"])
            select.assert_called_once_with(trainer.optimizer, fused=True)
            receipt = write.call_args.args[1]
            self.assertEqual(receipt["initial_global_step"], 0)
            self.assertFalse(receipt["optimizer_scheduler_rng_restored"])
            self.assertEqual(receipt["unrecoverable_parent_updates"], 2345)

    def test_fused_memory_checks_follow_base_step_only_at_synced_updates2_and20(self):
        events = []
        class Base:
            def step(self, batch, global_step, local_step):
                events.append(("base", global_step))
                return batch["synced"], sentinel.metrics
        with patch.object(task, "task_trainer_class", return_value=Base), \
             patch.object(restart, "record_fused_gpu_health", side_effect=lambda trainer, completed: events.append(("health", completed))) as health:
            trainer = restart.trainer_class({})()
            for completed, synced in ((1, True), (2, False), (2, True), (3, True), (20, True), (21, True)):
                self.assertEqual(trainer.step({"synced": synced}, completed - 1, 0), (synced, sentinel.metrics))
            self.assertEqual([call.args[1] for call in health.call_args_list], [2, 20])
            self.assertEqual(events[2:4], [("base", 1), ("health", 2)])

    def test_actual_health_collective_checks_all_eight_flags_and_memory_without_gpu(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            def gather(row):
                rows = row.repeat(8, 1)
                rows[:, 0] = torch.arange(8)
                return rows.reshape(-1)
            accelerator = SimpleNamespace(process_index=0, is_main_process=True, gather=Mock(side_effect=gather))
            optimizer = SimpleNamespace(defaults={"fused": True, "foreach": False},
                                        param_groups=[{"fused": True, "foreach": False}])
            trainer = SimpleNamespace(device=torch.device("cpu"), optimizer=optimizer, accelerator=accelerator, project_dir=directory)
            with patch.object(torch.cuda, "mem_get_info", return_value=(700, 1000)) as memory, \
                 patch.object(torch.cuda, "max_memory_allocated", return_value=100), \
                 patch.object(torch.cuda, "max_memory_reserved", return_value=200):
                restart.record_fused_gpu_health(trainer, 2)
                receipt = task.read_json(Path(directory) / "gpu_check_fused_update2.json")
                self.assertEqual(len(receipt["ranks"]), 8)
                self.assertTrue(receipt["all_rank_actual_fused_flags_passed"])
                self.assertEqual(receipt["ranks"][7]["rank"], 7)
                self.assertEqual(receipt["ranks"][0]["peak_allocated_bytes"], 100)
                self.assertIn("only", receipt["scope"])
                optimizer.defaults["fused"] = False
                with self.assertRaisesRegex(RuntimeError, "Eight-rank fused"):
                    restart.record_fused_gpu_health(trainer, 20)
                optimizer.defaults["fused"] = True
                optimizer.param_groups[0]["foreach"] = True
                with self.assertRaises(RuntimeError):
                    restart.record_fused_gpu_health(trainer, 20)
                optimizer.param_groups[0]["foreach"] = False
                memory.return_value = (0, 1000)
                with self.assertRaises(RuntimeError):
                    restart.record_fused_gpu_health(trainer, 20)
                self.assertFalse((Path(directory) / "gpu_check_fused_update20.json").exists())

    def test_legacy_run_and_injected_run_choose_correct_name_and_train_entrypoints(self):
        for supplied in (None, DIRECTORY / "a100_task_fused_restart.py"):
            with self.subTest(entrypoint=supplied), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "model.safetensors"
                source.write_bytes(b"tiny-source")
                _write(root / "data_check.json", {})
                plan = {"source_checkpoint": str(source), "source_checkpoint_sha256": task.shared.sha256(source),
                        "config_sha256": "a" * 64, "timestamp": "test", "max_optimizer_steps": 20000,
                        "warmup_steps": 1000, "source_step": 38000}
                processes = [SimpleNamespace(pid=12345, wait=Mock(return_value=0)), SimpleNamespace(wait=Mock(return_value=0))]
                with patch.object(task, "load_prepared", return_value=(None, plan)), \
                     patch.object(task, "verify_data_check"), patch.object(task.subprocess, "check_output", return_value=""), \
                     patch.object(task.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="/new/run\n", stderr="")) as naming, \
                     patch.object(task.subprocess, "Popen", side_effect=processes) as spawn:
                    self.assertEqual(task.run(root, entrypoint=supplied), 0)
                    expected = str(Path(supplied or task.__file__).resolve())
                    self.assertEqual(naming.call_args.args[0][1], expected)
                    self.assertEqual(spawn.call_args_list[0].args[0][5], expected)

    def test_launch_inherits_bounded_rolling_diagnostics_and_restores_caller_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {"timestamp": "cpu-test"}
            diag = root / "fake_tmpfs_diagnostics"
            diag.mkdir()
            observed = {}
            def launch(path, entrypoint):
                observed.update({key: os.environ.get(key) for key in ("CUDA_LAUNCH_BLOCKING", "PSI0_ROLLING_DIAGNOSTICS_DIR", "PSI0_DIAGNOSTIC_BATCH_RANGE")})
                self.assertEqual(Path(entrypoint).resolve(), Path(restart.__file__).resolve())
                return 0
            with patch.dict(os.environ, {"CUDA_LAUNCH_BLOCKING": "old", "PSI0_DIAGNOSTIC_BATCH_RANGE": "1:2"}), \
                 patch.object(restart, "diagnostic_directory", return_value=(diag, 2 * 1024 ** 3)), \
                 patch.object(restart.subprocess, "check_output", side_effect=["fixture-revision\n", " M fixture.py\n"]), \
                 patch.object(task, "run", side_effect=launch):
                self.assertEqual(restart.launch(root, plan), 0)
                self.assertEqual(os.environ["CUDA_LAUNCH_BLOCKING"], "old")
                self.assertEqual(os.environ["PSI0_DIAGNOSTIC_BATCH_RANGE"], "1:2")
            self.assertEqual(observed, {"CUDA_LAUNCH_BLOCKING": "1", "PSI0_ROLLING_DIAGNOSTICS_DIR": str(diag), "PSI0_DIAGNOSTIC_BATCH_RANGE": None})
            provenance = task.read_json(root / "restart_launch_provenance.json")
            self.assertEqual(provenance["rolling_slots_per_rank"], 2)
            self.assertEqual(provenance["initial_global_step"], 0)
            self.assertIn("add overhead", provenance["scope"])
            self.assertEqual(provenance["git_status"], "M fixture.py")
            self.assertEqual(len(provenance["source_sha256"]), 15)
            self.assertIn("VLMExpert/src/psi/models/psi0.py", provenance["source_sha256"])
            self.assertIn("ImageProcess/processor.py", provenance["source_sha256"])

    def test_repeat_run_does_not_create_diagnostics_or_overwrite_first_provenance(self):
        for name in ("training_state.json", "restart_launch_provenance.json"):
            with self.subTest(existing=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / name).write_bytes(b"first-launch-evidence")
                with patch.object(restart, "load") as load, patch.object(restart, "diagnostic_directory") as diagnostic, \
                     patch.object(restart, "launch") as launch, self.assertRaisesRegex(ValueError, "already launched"):
                    restart.run(root)
                load.assert_not_called()
                diagnostic.assert_not_called()
                launch.assert_not_called()
                self.assertEqual((root / name).read_bytes(), b"first-launch-evidence")


if __name__ == "__main__":
    unittest.main()
