"""CPU contracts for the bounded optimizer benchmark; no CUDA initialization."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "benchmark_fused_adamw.py"
spec = importlib.util.spec_from_file_location("benchmark_fused_adamw", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class FusedAdamWBenchmarkTests(unittest.TestCase):
    def test_restored_states_are_nonzero_independent_and_keep_all_five_lrs(self):
        recipe = benchmark.optimizer_recipe()
        fixture = benchmark.make_fixture(recipe, "tiny", initial_step=938)
        a = benchmark.build_optimizer(fixture, recipe, "cpu", fused=False)
        b = benchmark.build_optimizer(fixture, recipe, "cpu", fused=False)
        self.assertEqual([g["group_name"] for g in a[1].param_groups], list(benchmark.GROUP_NAMES))
        self.assertEqual([g["lr"] for g in a[1].param_groups], list(benchmark.DEFAULT_LRS))
        for pa, pb in zip(a[0], b[0]):
            self.assertEqual(a[1].state[pa]["step"].item(), 938)
            self.assertGreater(a[1].state[pa]["exp_avg_sq"].min().item(), 0)
            self.assertNotEqual(a[1].state[pa]["exp_avg"].data_ptr(), b[1].state[pb]["exp_avg"].data_ptr())
        # Mutation of one optimizer must not contaminate either the other branch
        # or the CPU source fixture (load_state_dict can otherwise alias tensors).
        a[1].state[a[0][0]]["exp_avg"].zero_()
        self.assertGreater(b[1].state[b[0][0]]["exp_avg"].abs().max().item(), 0)
        self.assertGreater(fixture["optimizer_state"]["state"][0]["exp_avg"].abs().max().item(), 0)

    def test_nonzero_restore_multistep_none_grad_and_exact_moment_comparison(self):
        recipe = benchmark.optimizer_recipe()
        fixture = benchmark.make_fixture(recipe, "tiny")
        a = benchmark.build_optimizer(fixture, recipe, "cpu", fused=False)
        b = benchmark.build_optimizer(fixture, recipe, "cpu", fused=False)
        for update in range(1, 7):
            for parameters, optimizer in (a, b):
                benchmark.set_gradients(parameters, fixture, update)
                optimizer.step()
        result = benchmark.compare_branches(a, b, fixture, 6, atol=1e-8, rtol=1e-5, chunk_elements=9)
        self.assertTrue(result["passed"])
        self.assertEqual(result["tensors"][0]["expected_step"], 944)
        self.assertEqual(result["tensors"][2]["expected_step"], 940)
        self.assertTrue(all(t["checks"]["exp_avg_sq"]["max_abs"] == 0 for t in result["tensors"]))
        with torch.no_grad():
            b[1].state[b[0][0]]["exp_avg_sq"].view(-1)[-1] += 0.1
        self.assertFalse(benchmark.compare_branches(a, b, fixture, 6,
            atol=1e-8, rtol=1e-5, chunk_elements=9)["passed"])

    def test_fused_backend_survives_loading_an_unfused_state_dict(self):
        recipe = benchmark.optimizer_recipe()
        fixture = benchmark.make_fixture(recipe, "tiny")
        # No fused CPU step required: exercise the real native optimizer loader.
        parameters, optimizer = benchmark.build_optimizer(fixture, recipe, "cpu", fused=True)
        self.assertTrue(all(g["fused"] and g["foreach"] is False for g in optimizer.param_groups))
        self.assertEqual(optimizer.state[parameters[0]]["step"].item(), 938)
        self.assertNotIn("fused", fixture["optimizer_state"]["param_groups"][0])

    def test_recipe_uses_actual_optimizer_decay_not_model_decay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            path.write_text(json.dumps({"train": {"learning_rate": 1e-5,
                "lr_scheduler_kwargs": {"betas": [0.95, 0.999], "eps": 1e-8, "weight_decay": 1e-6}},
                "model": {"weight_decay": 0.01, "lang_backbone_lr": 1e-7,
                          "vision_tower_lr": 1e-6, "mm_projector_lr": 1e-5, "motion_expert_lr": 1e-6}}))
            recipe = benchmark.optimizer_recipe(path)
            self.assertEqual(recipe["kwargs"]["weight_decay"], 1e-6)
            self.assertEqual(recipe["kwargs"]["betas"], (0.95, 0.999))
            self.assertEqual(len(recipe["source_sha256"]), 64)

    def test_chunked_comparison_rejects_nonfinite_and_far_tail_difference(self):
        a, b = torch.ones(31), torch.ones(31)
        b[-1] = float("nan")
        result = benchmark.tensor_comparison(a, b, atol=1e-8, rtol=1e-5, chunk_elements=7)
        self.assertFalse(result["passed"])
        self.assertFalse(result["finite"])
        b[-1] = 2
        self.assertFalse(benchmark.tensor_comparison(a, b,
            atol=1e-8, rtol=1e-5, chunk_elements=7)["passed"])

    def test_cpu_harness_never_initializes_cuda_or_claims_fused_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cpu.json"
            argv = ["--device", "cpu", "--candidate", "standard", "--profile", "tiny",
                "--correctness-steps", "3", "--warmup", "1", "--iterations", "2", "--rounds", "2",
                "--output", str(output)]
            with patch.object(torch.cuda, "set_device", side_effect=AssertionError("CUDA forbidden")):
                self.assertEqual(benchmark.main(argv), 0)
            receipt = json.loads(output.read_text())
            self.assertEqual(receipt["status"], "cpu_harness_self_check_passed")
            self.assertTrue(receipt["correctness"]["passed"])
            self.assertIn("not production checkpoint", receipt["source_state"])
            with self.assertRaises(FileExistsError):
                benchmark.main(argv)


if __name__ == "__main__":
    unittest.main()
