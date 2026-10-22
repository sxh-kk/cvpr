"""CPU contracts for full-model validation helpers, without CUDA/model imports."""
import importlib.util
from pathlib import Path
import random
import unittest

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "validate_task_fused_step.py"
spec = importlib.util.spec_from_file_location("validate_task_fused_step", SCRIPT)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class FullTaskValidationTests(unittest.TestCase):
    def make_optimizer(self):
        parameters = [torch.nn.Parameter(torch.ones(7, 9) * (i + 1)) for i in range(5)]
        groups = [{"params": [p], "group_name": name, "lr": lr}
            for p, name, lr in zip(parameters,
                ("action_header", "lang_backbone", "vision_tower", "mm_projector", "motion_expert"),
                (1e-5, 1e-7, 1e-6, 1e-5, 1e-6))]
        optimizer = torch.optim.AdamW(groups, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6, foreach=False)
        for p in parameters:
            p.grad = torch.full_like(p, .3)
        optimizer.step()
        return parameters, optimizer

    def test_optimizer_restore_keeps_source_counters_and_moments_independent(self):
        parameters, optimizer = self.make_optimizer()
        saved = validation.cpu_tree(optimizer.state_dict())
        source = validation.cpu_tree(saved)
        validation.restore_optimizer(optimizer, saved, "cpu", fused=False)
        for p in parameters:
            p.grad = torch.full_like(p, .6)
        optimizer.step()
        for key in saved["state"]:
            for field in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(saved["state"][key][field], source["state"][key][field]))
        self.assertEqual(optimizer.state[parameters[0]]["step"].item(), 2)

    def test_complete_comparison_checks_tail_none_pattern_and_exact_steps(self):
        reference = {"grad": torch.ones(31), "unused": None, "0/0/step": torch.tensor(4000.)}
        candidate = validation.cpu_tree(reference)
        self.assertTrue(validation.compare_maps(reference, candidate,
            atol=1e-8, rtol=1e-5, chunk_elements=7)["passed"])
        candidate["grad"][-1] += .1
        self.assertFalse(validation.compare_maps(reference, candidate,
            atol=1e-8, rtol=1e-5, chunk_elements=7)["passed"])
        candidate = validation.cpu_tree(reference)
        candidate["0/0/step"] += 1
        # A relative tolerance must never hide an off-by-one large Adam step.
        self.assertFalse(validation.compare_maps(reference, candidate,
            atol=1e-8, rtol=.01, chunk_elements=7)["passed"])
        candidate = validation.cpu_tree(reference)
        candidate["unused"] = torch.tensor(0.)
        self.assertFalse(validation.compare_maps(reference, candidate,
            atol=1e-8, rtol=1e-5, chunk_elements=7)["passed"])

    def test_saved_optimizer_recipe_shapes_and_scheduler_lr_are_bound(self):
        _, optimizer = self.make_optimizer()
        saved = validation.cpu_tree(optimizer.state_dict())
        names = [group["group_name"] for group in optimizer.param_groups]
        scheduler = {"_last_lr": [group["lr"] for group in optimizer.param_groups], "last_epoch": 32000}
        report = validation.validate_saved_optimizer(optimizer, saved, scheduler, names)
        self.assertEqual(report["initialized_parameter_states"], 5)
        scheduler["_last_lr"][0] *= 2
        with self.assertRaisesRegex(ValueError, "learning rates"):
            validation.validate_saved_optimizer(optimizer, saved, scheduler, names)

    def test_rng_replay_covers_python_numpy_and_torch(self):
        saved = {"random_state": random.getstate(), "numpy_random_seed": np.random.get_state(),
                 "torch_manual_seed": torch.get_rng_state().clone()}
        validation.restore_rng(saved, "cpu")
        a = (random.random(), np.random.rand(), torch.rand(7))
        validation.restore_rng(saved, "cpu")
        b = (random.random(), np.random.rand(), torch.rand(7))
        self.assertEqual(a[:2], b[:2])
        self.assertTrue(torch.equal(a[2], b[2]))

    def test_finite_guard_includes_adam_moments(self):
        parameters, optimizer = self.make_optimizer()
        model = torch.nn.ParameterList(parameters)
        validation.check_finite_model_optimizer(model, optimizer)
        optimizer.state[parameters[-1]]["exp_avg_sq"][-1, -1] = float("nan")
        with self.assertRaises(FloatingPointError):
            validation.check_finite_model_optimizer(model, optimizer)

    def test_gate_rejects_missing_actual_fullstep_or_checkpoint_identity(self):
        memory = {"peak_allocated_bytes": 100, "peak_reserved_bytes": 120,
                  "device_free_bytes_after_update": 1000}
        step = {"all_parameters_and_adam_states_finite": True, "memory": memory}
        report = {"comparisons": [{"passed": True, "branches": {"standard": {"step": step},
            "fused": {"step": step}}}], "additional_fused_steps": [step] * 3,
            "discarded_warmup": [{"finite": True, "memory": memory}] * 2,
            "source_checkpoint_files_sha256": {key: "a" * 64 for key in
                ("model.safetensors", "optimizer.bin", "scheduler.bin")},
            "source_config_sha256": "b" * 64, "runtime": {"gpu_total_bytes": 2000}}
        validation.finish_acceptance(report)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["full_step_comparison_passed"])
        incomplete = validation.cpu_tree(report)
        incomplete["additional_fused_steps"] = []
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validation.finish_acceptance(incomplete)
        unbound = validation.cpu_tree(report)
        unbound["source_checkpoint_files_sha256"].pop("optimizer.bin")
        with self.assertRaisesRegex(ValueError, "bound"):
            validation.finish_acceptance(unbound)


if __name__ == "__main__":
    unittest.main()
