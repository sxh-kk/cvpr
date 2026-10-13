"""Periodic DDP metrics preserve sample weighting without changing gradients."""
import json
from types import SimpleNamespace

import pytorch_lightning as pl
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pytorch_lightning.loggers.logger import Logger
from torch.utils.data import DataLoader, TensorDataset

from MotionExpert.module.uem_module import UEM_Module
from MotionExpert.tests.test_image_only_forecast_training import conditions, module


def metric_module():
    result = UEM_Module.__new__(UEM_Module)
    pl.LightningModule.__init__(result)
    result.cfg = SimpleNamespace(TRAIN=SimpleNamespace(LOG_EVERY_N_STEPS=2))
    result._trainer = SimpleNamespace(global_step=0, fit_loop=SimpleNamespace(_should_accumulate=lambda: False))
    result.logged = {}
    result.log = lambda name, value, **kwargs: result.logged.update({name: float(value)})
    return result


def test_logging_optimization_does_not_change_loss_or_gradients():
    batch = {"y": conditions(), "misc": {"motion": torch.randn(2, 11, 245)}}
    results = []
    for batched in (False, True):
        subject = module(supervision="joint")
        subject.cfg.TRAIN.BATCHED_METRIC_SYNC = batched
        torch.manual_seed(13)
        loss = subject.training_step(batch, 0)
        loss.backward()
        results.append((loss.detach(), subject.model.scale.grad.clone()))
        if batched:
            assert not subject._train_metric_totals.requires_grad
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_step_cadence_and_epoch_include_unlogged_tail():
    subject = metric_module()
    for step, (value, count) in enumerate(((2., 3), (8., 1), (20., 1))):
        subject._trainer.global_step = step
        subject._log_training_metrics({"loss": torch.tensor(value)}, count)
        if step == 0:
            assert not subject.logged
    assert subject.logged == {"train/loss_step": 8.}
    subject.on_train_epoch_end()
    assert subject.logged["train/loss_epoch"] == pytest.approx(34 / 5)
    assert subject._train_metric_totals is None


def _distributed_metrics(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        subject = metric_module()
        batches = ((2., 3), (8., 1), (20., 1)) if rank == 0 else ((4., 2), (10., 3), (30., 1))
        for step, (value, count) in enumerate(batches):
            subject._trainer.global_step = step
            subject._log_training_metrics({"loss": torch.tensor(value), "mse": torch.tensor(value ** 2)}, count)
        subject.on_train_epoch_end()
        from pathlib import Path
        Path(output, f"rank-{rank}.json").write_text(json.dumps(subject.logged))
    finally:
        dist.destroy_process_group()


def test_distributed_metrics_weight_unequal_rank_batches_and_tail(tmp_path):
    mp.spawn(_distributed_metrics, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        logged = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert logged["train/loss_step"] == pytest.approx(9.5)
        assert logged["train/loss_epoch"] == pytest.approx(102 / 11)
        assert logged["train/mse_step"] == pytest.approx(91.)
        assert logged["train/mse_epoch"] == pytest.approx(1708 / 11)


def test_lightning_logs_across_epochs_with_gradient_accumulation(tmp_path):
    class CaptureLogger(Logger):
        name, version = "metrics", "0"

        def __init__(self):
            super().__init__()
            self.rows = []

        def log_hyperparams(self, params):
            pass

        def log_metrics(self, metrics, step=None):
            self.rows.append((step, metrics))

    class TinyModule(UEM_Module):
        def __init__(self):
            pl.LightningModule.__init__(self)
            self.cfg = SimpleNamespace(TRAIN=SimpleNamespace(LOG_EVERY_N_STEPS=2),
                                       MODEL=SimpleNamespace(CKPT_PATH=None))
            self.scale = torch.nn.Parameter(torch.tensor(1.))

        def _set_motion_expert_train_phase(self, epoch):
            pass

        def training_step(self, batch, batch_idx):
            loss = batch[0].mean() * self.scale
            self._log_training_metrics({"loss": loss}, len(batch[0]))
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.)

    logger = CaptureLogger()
    trainer = pl.Trainer(default_root_dir=tmp_path, accelerator="cpu", devices=1,
                         max_epochs=2, accumulate_grad_batches=2, log_every_n_steps=2,
                         logger=logger, enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, num_sanity_val_steps=0)
    trainer.fit(TinyModule(), DataLoader(TensorDataset(torch.arange(1., 6.)), batch_size=2))
    epoch_values = [row["train/loss_epoch"] for _, row in logger.rows if "train/loss_epoch" in row]
    step_values = [row["train/loss_step"] for _, row in logger.rows if "train/loss_step" in row]
    assert epoch_values == [3., 3.]
    assert step_values == [5., 5.]
