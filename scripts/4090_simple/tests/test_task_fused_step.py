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
        self.assertTrue(report["uncontrolled_full_step_comparison_passed"])
        incomplete = validation.cpu_tree(report)
        incomplete["additional_fused_steps"] = []
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validation.finish_acceptance(incomplete)
        unbound = validation.cpu_tree(report)
        unbound["source_checkpoint_files_sha256"].pop("optimizer.bin")
        with self.assertRaisesRegex(ValueError, "bound"):
            validation.finish_acceptance(unbound)

    def test_replayed_preclip_snapshot_is_independent_and_exact(self):
        parameters, _ = self.make_optimizer()
        model = torch.nn.ParameterList(parameters)
        parameters[-1].grad = None
        saved = validation.snapshot_gradients(model)
        self.assertNotEqual(saved["0"].data_ptr(), parameters[0].grad.data_ptr())
        original = saved["0"].clone()
        parameters[0].grad.add_(.125)
        result = validation.replay_preclip_gradients(model, saved, chunk_elements=11)
        self.assertTrue(result["passed"])
        self.assertTrue(result["strict_zero_tolerance"])
        self.assertTrue(torch.equal(parameters[0].grad, original))
        torch.nn.utils.clip_grad_norm_(model.parameters(), .01)
        self.assertTrue(torch.equal(saved["0"], original))
        self.assertIsNone(parameters[-1].grad)

    def test_replay_validates_all_none_shape_dtype_and_devices_before_mutation(self):
        parameters, _ = self.make_optimizer()
        model = torch.nn.ParameterList(parameters)
        for bad in (None, torch.zeros(1), torch.zeros_like(parameters[-1], dtype=torch.float64),
                    torch.empty_like(parameters[-1], device="meta")):
            saved = validation.snapshot_gradients(model)
            saved["0"].add_(.5)  # This earlier valid entry must not be copied.
            saved["4"] = bad
            unchanged = parameters[0].grad.clone()
            with self.assertRaises(ValueError):
                validation.replay_preclip_gradients(model, saved, chunk_elements=13)
            self.assertTrue(torch.equal(parameters[0].grad, unchanged))

    def test_postclip_exact_gate_blocks_optimizer_before_state_or_params_change(self):
        parameters, optimizer = self.make_optimizer()
        model = torch.nn.ParameterList(parameters)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        saved = validation.snapshot_gradients(model)
        parameters[0].grad.view(-1)[-1] = torch.nextafter(parameters[0].grad.view(-1)[-1],
                                                        torch.tensor(float("inf")))
        comparison = validation.exact_postclip_comparison(saved, model, chunk_elements=7)
        self.assertFalse(comparison["passed"])
        self.assertTrue(comparison["strict_zero_tolerance"])
        before = parameters[0].detach().clone()
        old_step = optimizer.state[parameters[0]]["step"].clone()
        with self.assertRaisesRegex(RuntimeError, "was not executed"):
            validation.guarded_optimizer_step(optimizer, comparison)
        self.assertTrue(torch.equal(before, parameters[0]))
        self.assertTrue(torch.equal(old_step, optimizer.state[parameters[0]]["step"]))
        parameters[0].grad.copy_(saved["0"])
        comparison = validation.exact_postclip_comparison(saved, model, chunk_elements=7)
        validation.guarded_optimizer_step(optimizer, comparison)
        self.assertEqual(optimizer.state[parameters[0]]["step"].item(), old_step.item() + 1)

    def controlled_report(self):
        memory = {"peak_allocated_bytes": 100, "peak_reserved_bytes": 120,
                  "device_free_bytes_after_update": 1000}
        step = {"all_parameters_and_adam_states_finite": True, "memory": memory}
        branch = {"step": step}
        exact = {"passed": True, "strict_zero_tolerance": True}
        pairs = [{"passed": True, "comparison_complete": True, "execution_order": order,
            "preclip_structure": {"passed": True},
            "parameters_and_buffers": {"passed": True}, "adam_states_including_steps": {"passed": True},
            "preclip_replay": exact,
            "postclip_gradient_equality": exact, "branches": {"standard": branch, "fused": branch}}
            for order in (["standard", "fused"], ["fused", "standard"])]
        return validation.cpu_tree({"mode": "controlled", "same_gradient_comparisons": pairs,
            "baseline_repetition": {"comparison_complete": True, "passed": False,
                "branches": {"standard_first": branch, "standard_second": branch}},
            "additional_fused_steps": [{**step, "gradient_source": "own_real_backward",
                "restored_since_previous_step": i == 0} for i in range(3)],
            "discarded_warmup": [{"finite": True, "memory": memory}] * 2,
            "source_checkpoint_files_sha256": {name: "a" * 64 for name in
                ("model.safetensors", "optimizer.bin", "scheduler.bin")},
            "source_config_sha256": "b" * 64, "runtime": {"gpu_total_bytes": 2000}})

    def test_controlled_pass_is_separate_from_independent_and_strict_gates(self):
        report = self.controlled_report()
        validation.finish_controlled_acceptance(report)
        self.assertEqual(report["status"], "controlled_optimizer_comparison_passed")
        self.assertTrue(report["same_gradient_optimizer_comparison_passed"])
        self.assertTrue(report["postclip_gradients_exactly_equal"])
        self.assertTrue(report["continuous_fused_real_gradient_steps_passed"])
        self.assertFalse(report["full_step_comparison_passed"])
        self.assertFalse(report["uncontrolled_full_step_comparison_passed"])
        self.assertFalse(report["baseline_repetition"]["passed"])
        with self.assertRaisesRegex(ValueError, "Controlled"):
            validation.finish_acceptance(report)

    def test_controlled_gate_rejects_postclip_mismatch_and_noncontinuous_replay(self):
        report = self.controlled_report()
        report["same_gradient_comparisons"][1]["postclip_gradient_equality"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "input equality"):
            validation.finish_controlled_acceptance(report)
        report = self.controlled_report()
        report["additional_fused_steps"][1]["gradient_source"] = "shared_replay"
        with self.assertRaisesRegex(ValueError, "own real backward"):
            validation.finish_controlled_acceptance(report)
        report = self.controlled_report()
        report["additional_fused_steps"][1]["restored_since_previous_step"] = True
        with self.assertRaisesRegex(ValueError, "must not restore"):
            validation.finish_controlled_acceptance(report)
        report = self.controlled_report()
        report["same_gradient_comparisons"] = report["same_gradient_comparisons"][:1]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validation.finish_controlled_acceptance(report)

    def test_default_mode_and_numerical_thresholds_are_unchanged(self):
        args = validation.parse_args(["--prepared", "/prepared", "--checkpoint", "/checkpoint",
            "--output-dir", "/output", "--device", "cuda:0"])
        self.assertEqual(args.mode, "strict")
        self.assertEqual((args.atol, args.rtol, args.grad_atol, args.grad_rtol, args.loss_atol),
                         (1e-8, 1e-5, 1e-7, 1e-4, 1e-6))


if __name__ == "__main__":
    unittest.main()
