"""Restoration must keep Adam moments, counters, recipe and unused states."""
import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adamw_task_resume import audit_optimizer_restoration, select_adamw_backend


def populated():
    parameters = [torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10)
                  for _ in range(6)]
    names = ("action_header", "lang_backbone", "vision_tower", "mm_projector", "motion_expert")
    groups = [{"params": [p], "group_name": name, "lr": lr}
              for p, name, lr in zip(parameters, names, (1e-5, 1e-7, 1e-6, 1e-5, 1e-6))]
    groups[-1]["params"].append(parameters[-1])  # Never active routed parameter.
    optimizer = torch.optim.AdamW(groups, foreach=False, betas=(.95, .999), eps=1e-8, weight_decay=1e-6)
    for index in range(4):
        for i, parameter in enumerate(parameters[:-1]):
            parameter.grad = torch.full_like(parameter, .1 + index) if i != 2 or index == 0 else None
        optimizer.step()
    return parameters, optimizer


class RestoreTests(unittest.TestCase):
    def test_nonzero_restoration_audit_checks_all_moments_and_individual_steps(self):
        parameters, optimizer = populated()
        saved = copy.deepcopy(optimizer.state_dict())
        result = audit_optimizer_restoration(optimizer, saved)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["state_tensors_compared"], 15)
        self.assertNotEqual(optimizer.state[parameters[0]]["step"], optimizer.state[parameters[2]]["step"])

    def test_backend_choice_preserves_moments_lr_and_unused_state(self):
        parameters, optimizer = populated()
        saved = copy.deepcopy(optimizer.state_dict())
        moments = {p: (state["exp_avg"], state["exp_avg_sq"]) for p, state in optimizer.state.items()}
        receipt = select_adamw_backend(optimizer, fused=False)
        self.assertEqual(receipt["initialized_parameter_states"], 5)
        self.assertFalse(receipt["moments_reinitialized"])
        self.assertNotIn(parameters[-1], optimizer.state)
        for p, (m, v) in moments.items():
            self.assertIs(optimizer.state[p]["exp_avg"], m)
            self.assertIs(optimizer.state[p]["exp_avg_sq"], v)
        after = optimizer.state_dict()
        for old, current in zip(saved["param_groups"], after["param_groups"]):
            self.assertEqual(current["lr"], old["lr"])
            self.assertEqual(current["betas"], old["betas"])
        for index, state in saved["state"].items():
            for name, tensor in state.items():
                self.assertTrue(torch.equal(tensor, after["state"][index][name]))

    def test_changed_moment_is_rejected(self):
        _, optimizer = populated()
        saved = copy.deepcopy(optimizer.state_dict())
        next(iter(saved["state"].values()))["exp_avg"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "differs"):
            audit_optimizer_restoration(optimizer, saved)

    def test_changed_recipe_and_missing_state_are_rejected(self):
        _, optimizer = populated()
        saved = copy.deepcopy(optimizer.state_dict())
        saved["param_groups"][0]["lr"] *= 2
        with self.assertRaisesRegex(ValueError, "options"):
            audit_optimizer_restoration(optimizer, saved)
        saved = copy.deepcopy(optimizer.state_dict())
        saved["state"].pop(next(iter(saved["state"])))
        with self.assertRaisesRegex(ValueError, "state"):
            audit_optimizer_restoration(optimizer, saved)

    def test_invalid_cuda_switch_does_not_mutate_cpu_state(self):
        _, optimizer = populated()
        before = copy.deepcopy(optimizer.state_dict())
        with self.assertRaisesRegex(ValueError, "CUDA FP32"):
            select_adamw_backend(optimizer, fused=True)
        self.assertEqual(optimizer.param_groups[0].get("fused"), before["param_groups"][0].get("fused"))
        self.assertEqual(audit_optimizer_restoration(optimizer, before)["status"], "passed")


if __name__ == "__main__":
    unittest.main()
