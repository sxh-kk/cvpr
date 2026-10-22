"""CPU wiring checks for the actual fresh/resumed task trainer factories.

Only the expensive PosttrainTrainer implementation is replaced. The task's
factory methods execute unchanged; CUDA initialization and RNG calls are banned.
No real model, task data, checkpoint, worker pool or GPU is created.
"""
from contextlib import contextmanager, ExitStack
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel

from accelerate.data_loader import prepare_data_loader
import torch
from torch.utils.data import DataLoader, SequentialSampler


DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIRECTORY))
_SPEC = importlib.util.spec_from_file_location("task_fused_resume_under_test", DIRECTORY / "a100_task_fused_resume.py")
resume = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(resume)
task = resume.task
import replay_task_data


class _SizedLoader:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


class _PosttrainBoundary:
    """A CPU-only boundary patched at the same methods as PosttrainTrainer."""
    def init_models(self):
        raise AssertionError("Real model construction is forbidden")

    def prepare(self, accelerator):
        raise AssertionError("Real Accelerator/model preparation is forbidden")

    def create_optimizers(self):
        raise AssertionError("Real optimizer construction is forbidden")

    def create_dataloaders(self, train_dataset, val_dataset):
        raise AssertionError("Real task data loading is forbidden")

    def evaluate(self):
        raise AssertionError("Real model evaluation is forbidden")

    @property
    def num_steps_per_epoch(self):
        return 432


class _FreshGateBase(_PosttrainBoundary):
    def prepare(self, accelerator):
        raise AssertionError("Resumed preparation must bypass the fresh step-zero gate")


class TaskFusedResumeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        boundary = ModuleType("psi.trainers.posttrain")
        boundary.PosttrainTrainer = _PosttrainBoundary
        self.stack.enter_context(patch.dict(sys.modules, {"psi.trainers.posttrain": boundary}))
        for name in ("_lazy_init", "init", "set_device", "get_rng_state_all", "set_rng_state_all"):
            self.stack.enter_context(patch.object(torch.cuda, name,
                side_effect=AssertionError(f"CUDA {name} is forbidden in CPU wiring tests")))
        self.plan = {"resume_step": 4000, "resume_checkpoint": "/external/ckpt_4000",
                     "source_step": 38000, "source_checkpoint": "/external/ckpt_38000/model.safetensors"}

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def resumed(self, *, manager_factory=None):
        with patch.object(task, "task_trainer_class", return_value=_FreshGateBase) as factory:
            if manager_factory is None:
                cls = resume.resumed_trainer_class(self.plan)
            else:
                with patch.object(replay_task_data, "ReplayResumedMapDataLoader", manager_factory):
                    cls = resume.resumed_trainer_class(self.plan)
            factory.assert_called_once_with(self.plan)
        trainer = cls()
        trainer.cfg = SimpleNamespace(train=SimpleNamespace(resume_from_checkpoint=self.plan["resume_checkpoint"]))
        trainer.world_size = 8
        trainer.gradient_accumulation_steps = 1
        trainer.accelerator = SimpleNamespace(num_processes=8, is_main_process=True, process_index=0)
        trainer.project_dir = "/external/cpu-only-segment"
        trainer.train_dataloader = _SizedLoader(431)
        return trainer

    def fresh(self):
        # Execute the actual extracted factory, rather than the resumed test's
        # patched factory, to protect its original fresh-run behavior too.
        trainer = task.task_trainer_class(self.plan)()
        trainer.accelerator = SimpleNamespace(num_processes=8, is_main_process=True, process_index=3)
        trainer.project_dir = "/external/cpu-only-fresh"
        trainer.cfg = SimpleNamespace(seed=20261022)
        trainer.device = torch.device("cpu")
        return trainer

    def test_init_models_hides_resume_only_during_successful_construction(self):
        trainer = self.resumed()
        def construct(actual):
            self.assertIs(actual, trainer)
            self.assertIsNone(actual.cfg.train.resume_from_checkpoint)
            return sentinel.model
        with patch.object(_PosttrainBoundary, "init_models", autospec=True, side_effect=construct) as build:
            self.assertIs(trainer.init_models(), sentinel.model)
        build.assert_called_once_with(trainer)
        self.assertEqual(trainer.cfg.train.resume_from_checkpoint, self.plan["resume_checkpoint"])

    def test_init_models_restores_resume_if_model_construction_raises(self):
        trainer = self.resumed()
        def fail(actual):
            self.assertIsNone(actual.cfg.train.resume_from_checkpoint)
            raise RuntimeError("synthetic construction failure")
        with patch.object(_PosttrainBoundary, "init_models", autospec=True, side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "synthetic construction failure"):
                trainer.init_models()
        self.assertEqual(trainer.cfg.train.resume_from_checkpoint, self.plan["resume_checkpoint"])

    def test_prepare_calls_posttrain_directly_and_requires_eight_ranks_431_batches(self):
        trainer = self.resumed()
        loader = trainer.train_dataloader
        with patch.object(_PosttrainBoundary, "prepare", autospec=True, return_value=loader) as prepare:
            self.assertIs(trainer.prepare(trainer.accelerator), loader)
            prepare.assert_called_once_with(trainer, trainer.accelerator)
        for ranks, batches in ((7, 431), (8, 430), (8, 432)):
            with self.subTest(ranks=ranks, batches=batches):
                accelerator = SimpleNamespace(num_processes=ranks)
                with patch.object(_PosttrainBoundary, "prepare", return_value=_SizedLoader(batches)):
                    with self.assertRaisesRegex(ValueError, "eight-rank431"):
                        trainer.prepare(accelerator)

    def test_epoch_length_uses_real_prepared_shard_and_raw_loader_before_prepare(self):
        trainer = self.resumed()
        del trainer.train_dataloader
        self.assertEqual(trainer.num_steps_per_epoch, 432)  # Construction fallback only.
        trainer.train_dataloader = _SizedLoader(3452)
        self.assertEqual(trainer.num_steps_per_epoch, 431)
        raw = DataLoader(range(13808), batch_size=4, drop_last=True)
        trainer.train_dataloader = prepare_data_loader(raw, num_processes=8, process_index=7,
                                                       put_on_device=False, rng_types=[])
        self.assertEqual(trainer.num_steps_per_epoch, 431)

    def test_resume_manager_wires_double_argument_progress_without_calling_cuda(self):
        manager_factory = Mock(return_value=sentinel.manager)
        trainer = self.resumed(manager_factory=manager_factory)
        with patch.object(resume, "write") as write, patch.object(task, "now", return_value="cpu-test-time"):
            self.assertIs(trainer.create_resume_data_manager(4000), sentinel.manager)
            args, options = manager_factory.call_args
            self.assertEqual(args, (trainer.train_dataloader,))
            self.assertEqual(options["completed_microbatches"], 4000)
            self.assertIs(options["extra_rng_context"], resume.preserve_cuda_rng)
            callback = options["progress"]
            callback(1, 4000)
            write.assert_not_called()
            callback(100, 4000)
            callback(4000, 4000)
            self.assertEqual([call.args[1]["status"] for call in write.call_args_list], ["replaying", "complete"])
            self.assertEqual(write.call_args.args, (Path(trainer.project_dir) / "data_replay_progress.json", {
                "completed_microbatches_replayed": 4000, "target_microbatches": 4000,
                "generated_utc": "cpu-test-time", "status": "complete"}))
            write.reset_mock()
            trainer.accelerator.is_main_process = False
            callback(4000, 4000)
            write.assert_not_called()
        self.assertTrue(replay_task_data.ReplayResumedMapDataLoader.keep_prepared_loader)

    def test_resume_manager_rejects_changed_update_or_accumulation(self):
        manager_factory = Mock()
        trainer = self.resumed(manager_factory=manager_factory)
        with self.assertRaisesRegex(ValueError, "cursor"):
            trainer.create_resume_data_manager(3999)
        trainer.gradient_accumulation_steps = 2
        with self.assertRaisesRegex(ValueError, "cursor"):
            trainer.create_resume_data_manager(4000)
        manager_factory.assert_not_called()

    def test_fresh_factory_retains_five_peak_groups_and_empty_optimizer_state_gate(self):
        trainer = self.fresh()
        def create(actual):
            actual.optimizer = SimpleNamespace(
                param_groups=[{"group_name": name, "lr": lr} for name, lr in task.PEAK_LRS.items()], state={})
        with patch.object(_PosttrainBoundary, "create_optimizers", autospec=True, side_effect=create):
            trainer.create_optimizers()
        for bad in ("lr", "state"):
            create(trainer)
            if bad == "lr":
                trainer.optimizer.param_groups[0]["lr"] *= 2
            else:
                trainer.optimizer.state["existing"] = {"step": 3}
            with patch.object(_PosttrainBoundary, "create_optimizers"):
                with self.assertRaisesRegex(ValueError, "fresh optimizer groups"):
                    trainer.create_optimizers()

    def test_fresh_prepare_retains_zero_step_gate_and_fresh_receipt(self):
        trainer = self.fresh()
        trainer.resume_from_checkpoint = Mock(return_value=(0, None))
        loader = _SizedLoader(431)
        with patch.object(_PosttrainBoundary, "prepare", return_value=loader), patch.object(task.shared, "write_json") as write:
            self.assertIs(trainer.prepare(trainer.accelerator), loader)
            receipt = write.call_args.args[1]
            self.assertTrue(receipt["fresh_optimizer_scheduler_rng"])
            self.assertTrue(receipt["strict_joint_loader"])
            self.assertEqual(receipt["source_step"], 38000)
            self.assertEqual(receipt["five_peak_learning_rates"], task.PEAK_LRS)
            trainer.resume_from_checkpoint.return_value = (4000, "/external/ckpt_4000")
            with self.assertRaisesRegex(ValueError, "step zero"):
                trainer.prepare(trainer.accelerator)

    def test_fresh_validation_keeps_fixed_order_collator_and_isolated_rng(self):
        trainer = self.fresh()
        train_loader = SimpleNamespace(collate_fn=sentinel.collator)
        trainer.train_dataloader = train_loader
        with patch.object(_PosttrainBoundary, "create_dataloaders", autospec=True) as create:
            train, val = trainer.create_dataloaders(sentinel.train_data, list(range(40)))
            create.assert_called_once_with(trainer, sentinel.train_data, list(range(40)))
        self.assertIs(train, train_loader)
        self.assertEqual((val.batch_size, val.num_workers, val.drop_last), (1, 0, False))
        self.assertIsInstance(val.sampler, SequentialSampler)
        self.assertIs(val.collate_fn, sentinel.collator)
        entered = []
        @contextmanager
        def isolated(seed, device):
            self.assertEqual(seed, 20261022 + 100000 + 3)
            self.assertEqual(device, torch.device("cpu"))
            entered.append("enter")
            try:
                yield
            finally:
                entered.append("exit")
        with patch.object(task.shared, "isolated_rng", isolated), patch.object(_PosttrainBoundary, "evaluate", return_value=sentinel.losses):
            self.assertIs(trainer.evaluate(), sentinel.losses)
        self.assertEqual(entered, ["enter", "exit"])


if __name__ == "__main__":
    unittest.main()
