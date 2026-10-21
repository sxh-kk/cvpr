"""CPU numerical/serialization contracts; GPU transfers require separate acceptance."""
import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_master_adamw import CpuMasterAdamW


def groups(parameters):
    return [{"params": [p], "group_name": name, "lr": lr} for p, name, lr in zip(parameters,
        ("action_header", "lang_backbone", "vision_tower", "mm_projector", "motion_expert"),
        (1e-4, 1e-6, 1e-5, 1e-4, 1e-5))]


def parameters():
    return [torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3) * .07 + index)
            for index in range(5)]


class CpuMasterAdamWTests(unittest.TestCase):
    def test_native_adamw_exact_multigroup_parity_with_clip_and_none_grad(self):
        left, right = parameters(), parameters()
        kwargs = dict(betas=(.95, .999), eps=1e-8, weight_decay=1e-6, foreach=False)
        wrapped = CpuMasterAdamW(groups(left), **kwargs)
        native = torch.optim.AdamW(groups(right), **kwargs)
        self.assertTrue(all(g['params'][0] is p for g, p in zip(wrapped.param_groups, left)))
        self.assertEqual(wrapped.receipt()['groups'][0]['initialized_states'], 0)
        for step in range(4):
            for index, (a, b) in enumerate(zip(left, right)):
                gradient = torch.full_like(a, .2 * (index + 1) + step)
                a.grad = gradient.clone() if (step + index) % 3 else None
                b.grad = gradient.clone() if (step + index) % 3 else None
            for wa, na in zip(wrapped.param_groups, native.param_groups):
                torch.nn.utils.clip_grad_norm_(wa['params'], 1.)
                torch.nn.utils.clip_grad_norm_(na['params'], 1.)
            wrapped.step(); native.step()
            for a, b in zip(left, right):
                self.assertTrue(torch.equal(a, b))
                for key, expected in native.state.get(b, {}).items():
                    actual = wrapped.state[a][key]
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(actual.device.type, 'cpu')
            wrapped.zero_grad(); native.zero_grad()
        self.assertEqual(wrapped.completed_steps, 4)

    def test_none_grad_does_not_apply_decay_or_create_state(self):
        active, unused = torch.nn.Parameter(torch.tensor(2.)), torch.nn.Parameter(torch.tensor(3.))
        wrapped = CpuMasterAdamW([active, unused], lr=.1, weight_decay=.5, foreach=False)
        active.grad = torch.tensor(1.)
        wrapped.step()
        self.assertEqual(unused.item(), 3.)
        self.assertNotIn(unused, wrapped.state)
        self.assertEqual(wrapped.cpu_optimizer.param_groups[0]['params'][1].grad, None)

    def test_scheduler_and_model_then_optimizer_roundtrip_continue_exactly(self):
        left = parameters()
        wrapped = CpuMasterAdamW(groups(left), foreach=False, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(wrapped, lambda step: .8 ** step)
        for _ in range(2):
            for p in left:
                p.grad = torch.full_like(p, .3)
            wrapped.step(); scheduler.step(); wrapped.zero_grad()
        model_state = [p.detach().clone() for p in left]
        optimizer_state = copy.deepcopy(wrapped.state_dict())
        scheduler_state = copy.deepcopy(scheduler.state_dict())
        right = [torch.nn.Parameter(p.clone()) for p in model_state]
        restored = CpuMasterAdamW(groups(right), foreach=False, amsgrad=True)
        restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored, lambda step: .8 ** step)
        restored.load_state_dict(optimizer_state)
        restored_scheduler.load_state_dict(scheduler_state)
        for _ in range(3):
            for a, b in zip(left, right):
                a.grad = torch.full_like(a, .4); b.grad = a.grad.clone()
            wrapped.step(); scheduler.step(); wrapped.zero_grad()
            restored.step(); restored_scheduler.step(); restored.zero_grad()
        self.assertEqual(restored.completed_steps, 5)
        self.assertEqual(restored_scheduler.get_last_lr(), scheduler.get_last_lr())
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(left, right)))

    def test_wrapping_is_fresh_only_and_native_state_cannot_be_restored(self):
        p = torch.nn.Parameter(torch.ones(2))
        native = torch.optim.AdamW([p], foreach=False)
        wrapped = CpuMasterAdamW.from_optimizer(native)
        self.assertIs(wrapped.param_groups[0]['params'][0], p)
        with self.assertRaisesRegex(ValueError, 'Only CpuMasterAdamW state'):
            wrapped.load_state_dict(native.state_dict())
        p.grad = torch.ones_like(p); native.step()
        with self.assertRaisesRegex(ValueError, 'fresh native'):
            CpuMasterAdamW.from_optimizer(native)

    def test_fp32_and_checkpoint_layout_are_explicit(self):
        with self.assertRaisesRegex(ValueError, 'original FP32'):
            CpuMasterAdamW([torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))])
        source = CpuMasterAdamW(groups(parameters()), foreach=False)
        destination = CpuMasterAdamW(groups(parameters())[::-1], foreach=False)
        with self.assertRaisesRegex(ValueError, 'layout changed'):
            destination.load_state_dict(source.state_dict())

    def test_external_model_restore_refreshes_cpu_masters(self):
        p = torch.nn.Parameter(torch.tensor(1.))
        wrapped = CpuMasterAdamW([p], lr=.1, weight_decay=0, foreach=False)
        state = copy.deepcopy(wrapped.state_dict())
        with torch.no_grad():
            p.fill_(5.)
        wrapped.load_state_dict(state)  # standard model-before-optimizer restore order
        p.grad = torch.tensor(1.); wrapped.step()
        self.assertAlmostEqual(p.item(), 4.9, places=6)

    def test_accelerated_optimizer_fresh_roundtrip_and_accumulation(self):
        from accelerate import Accelerator
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2, mixed_precision='no')
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(.5)
        reference = copy.deepcopy(model)
        wrapped = CpuMasterAdamW(model.parameters(), lr=.01, foreach=False)
        native = torch.optim.AdamW(reference.parameters(), lr=.01, foreach=False)
        model, accelerated = accelerator.prepare(model, wrapped)
        for index in range(4):
            x = torch.tensor([[1., 2.]]) * (index + 1)
            with accelerator.accumulate(model):
                accelerator.backward(model(x).square().mean())
                accelerated.step(); accelerated.zero_grad()
            (reference(x).square().mean() / 2).backward()
            if (index + 1) % 2 == 0:
                native.step(); native.zero_grad()
            self.assertEqual(wrapped.completed_steps, (index + 1) // 2)
            self.assertTrue(torch.equal(model.weight, reference.weight))
        self.assertEqual(wrapped.receipt()['groups'][0]['state_devices'], ['cpu'])

    def test_accum4_with_accelerated_scheduler_matches_native_warmup_cosine(self):
        from accelerate import Accelerator
        import math
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=4, mixed_precision='no')
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(.5)
        reference = copy.deepcopy(model)
        wrapped = CpuMasterAdamW(model.parameters(), lr=.01, foreach=False)
        native = torch.optim.AdamW(reference.parameters(), lr=.01, foreach=False)
        def schedule(step):
            return step / 20 if step < 20 else .5 * (1 + math.cos(math.pi * (step - 20) / 280))
        scheduler = torch.optim.lr_scheduler.LambdaLR(wrapped, schedule)
        reference_scheduler = torch.optim.lr_scheduler.LambdaLR(native, schedule)
        model, accelerated, prepared_scheduler = accelerator.prepare(model, wrapped, scheduler)
        for index in range(12):
            x = torch.tensor([[1., 2.]]) * (index + 1)
            with accelerator.accumulate(model):
                accelerator.backward(model(x).square().mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.)
                accelerated.step(); prepared_scheduler.step(); accelerated.zero_grad()
            (reference(x).square().mean() / 4).backward()
            if (index + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
                native.step(); reference_scheduler.step(); native.zero_grad()
            self.assertEqual(wrapped.completed_steps, (index + 1) // 4)
            self.assertTrue(torch.equal(model.weight, reference.weight))
            self.assertEqual(scheduler.last_epoch, reference_scheduler.last_epoch)
            self.assertEqual(wrapped.param_groups[0]['lr'], native.param_groups[0]['lr'])


if __name__ == '__main__':
    unittest.main()
