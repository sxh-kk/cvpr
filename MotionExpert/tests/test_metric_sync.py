"""Batched training logging preserves step cadence and full epoch averages."""

import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers.logger import Logger
from torch.utils.data import DataLoader, TensorDataset

from MotionExpert.module.uem_module import UEM_Module


class _MetricProbe:
    """Exercise production logging hooks without constructing the motion model."""

    _log_training_metrics = UEM_Module._log_training_metrics
    _reduce_training_metrics = staticmethod(UEM_Module._reduce_training_metrics)
    on_train_epoch_end = UEM_Module.on_train_epoch_end

    def __init__(self, interval=10):
        self.cfg = SimpleNamespace(TRAIN=SimpleNamespace(LOG_EVERY_N_STEPS=interval))
        self.accumulating = False
        self._trainer = SimpleNamespace(global_step=0, fit_loop=SimpleNamespace(
            _should_accumulate=lambda: self.accumulating))
        self.logs = []

    def log(self, name, value, **options):
        self.logs.append((name, value, options))


def _metrics(value):
    return {"loss": torch.tensor(value, dtype=torch.float32, requires_grad=True),
            "body_mse": torch.tensor(value * 2, dtype=torch.float32, requires_grad=True)}


def test_step_logs_only_on_optimizer_logging_interval_and_detaches_graph():
    probe = _MetricProbe()
    reductions = []
    probe._reduce_training_metrics = lambda packed: (
        reductions.append(packed.clone()) or packed[:-1] / packed[-1])
    for step in range(25):
        probe._trainer.global_step = step
        probe._log_training_metrics(_metrics(float(step + 1)), batch_size=4)
    assert len(reductions) == 2
    losses = [(value.item(), options) for name, value, options in probe.logs
              if name == "train/loss_step"]
    assert [value for value, _ in losses] == [10., 20.]
    assert all(options == {"on_step": True, "on_epoch": False,
                           "sync_dist": False, "batch_size": 4}
               for _, options in losses)
    assert not probe._train_metric_totals.requires_grad
    assert all(not value.requires_grad for _, value, _ in probe.logs)


def test_epoch_average_includes_unlogged_tail_and_weights_by_sample_count():
    probe = _MetricProbe(interval=2)
    for step, (value, size) in enumerate([(1., 4), (3., 2), (9., 1)]):
        probe._trainer.global_step = step
        probe._log_training_metrics(_metrics(value), batch_size=size)
    probe.on_train_epoch_end()
    epoch_logs = {name: value.item() for name, value, _ in probe.logs if name.endswith("_epoch")}
    assert epoch_logs == pytest.approx({"train/loss_epoch": 19 / 7,
                                       "train/body_mse_epoch": 38 / 7})
    assert [value.item() for name, value, _ in probe.logs if name == "train/loss_step"] == [3.]
    assert probe._train_metric_totals is None
    assert probe._train_metric_names is None
    log_count = len(probe.logs)
    probe.on_train_epoch_end()
    assert len(probe.logs) == log_count


def test_cadence_continues_across_epoch_boundary_and_resumed_global_step():
    probe = _MetricProbe(interval=10)
    for step in range(7):
        probe._trainer.global_step = step
        probe._log_training_metrics(_metrics(1.), batch_size=2)
    probe.on_train_epoch_end()
    assert not any(name.endswith("_step") for name, _, _ in probe.logs)
    for step in range(7, 12):
        probe._trainer.global_step = step
        probe._log_training_metrics(_metrics(float(step)), batch_size=2)
    assert [value.item() for name, value, _ in probe.logs if name == "train/loss_step"] == [9.]
    resumed = _MetricProbe(interval=10)
    resumed._trainer.global_step = 9739
    resumed._log_training_metrics(_metrics(4.), batch_size=2)
    assert [value.item() for name, value, _ in resumed.logs if name == "train/loss_step"] == [4.]


def test_accumulated_step_logs_last_microbatch_but_epoch_keeps_every_sample():
    probe = _MetricProbe(interval=2)
    for global_step, accumulating, value, size in [
        (0, True, 1., 4), (0, False, 3., 4),
        (1, True, 5., 4), (1, False, 7., 2),
    ]:
        probe._trainer.global_step = global_step
        probe.accumulating = accumulating
        probe._log_training_metrics(_metrics(value), batch_size=size)
    assert [value.item() for name, value, _ in probe.logs if name == "train/loss_step"] == [7.]
    probe.on_train_epoch_end()
    assert [value.item() for name, value, _ in probe.logs
            if name == "train/loss_epoch"] == pytest.approx([50 / 14])


def test_metric_name_change_fails_instead_of_mixing_columns():
    probe = _MetricProbe()
    probe._log_training_metrics(_metrics(1.), batch_size=2)
    with pytest.raises(ValueError, match="metric names changed"):
        probe._log_training_metrics({"another_metric": torch.tensor(2.)}, batch_size=2)


class _MemoryLogger(Logger):
    def __init__(self):
        super().__init__()
        self.records = []

    @property
    def name(self):
        return "metric-regression"

    @property
    def version(self):
        return "0"

    def log_hyperparams(self, params):
        pass

    def log_metrics(self, metrics, step):
        self.records.append((dict(metrics), step))


class _TinyMetricModule(pl.LightningModule):
    _log_training_metrics = UEM_Module._log_training_metrics
    _reduce_training_metrics = staticmethod(UEM_Module._reduce_training_metrics)
    on_train_epoch_end = UEM_Module.on_train_epoch_end

    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(TRAIN=SimpleNamespace(LOG_EVERY_N_STEPS=2))
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def training_step(self, batch, batch_idx):
        loss = self.weight * 0 + batch_idx + 1. + self.current_epoch * 10.
        self._log_training_metrics({"loss": loss}, len(batch[0]))
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=.1)


@pytest.mark.parametrize("accumulate, expected_steps", [(1, [2., 11., 13.]), (2, [3., 13.])])
def test_real_lightning_logger_keeps_cadence_and_complete_epoch_mean(tmp_path, accumulate, expected_steps):
    logger = _MemoryLogger()
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=2, logger=logger,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, log_every_n_steps=2,
                         accumulate_grad_batches=accumulate, default_root_dir=tmp_path)
    trainer.fit(_TinyMetricModule(), train_dataloaders=DataLoader(
        TensorDataset(torch.zeros(10, 1)), batch_size=4, num_workers=0))
    steps = [metrics["train/loss_step"] for metrics, _ in logger.records
             if "train/loss_step" in metrics]
    epochs = [metrics["train/loss_epoch"] for metrics, _ in logger.records
              if "train/loss_epoch" in metrics]
    assert steps == pytest.approx(expected_steps)
    assert epochs == pytest.approx([1.8, 11.8])


def _distributed_metric_worker(rank, init_path, output):
    from datetime import timedelta
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_path}",
                                         rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        probe = _MetricProbe(interval=2)
        # Unequal local counts exercise the global numerator/count reduction.
        for step, (value, size) in enumerate([(1. + rank * 4, 2 + rank),
                                               (3. + rank * 4, 1 + rank)]):
            probe._trainer.global_step = step
            probe._log_training_metrics(_metrics(value), batch_size=size)
        probe.on_train_epoch_end()
        Path(output, f"rank{rank}.json").write_text(json.dumps(
            {name: value.item() for name, value, _ in probe.logs}))
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_reduction_uses_sample_weighted_step_and_epoch_means(tmp_path):
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_distributed_metric_worker, args=(
        rank, str(tmp_path / "rendezvous"), str(tmp_path))) for rank in range(2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(60)
        assert all(process.exitcode == 0 for process in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
    expected = {"train/loss_step": 17 / 3, "train/body_mse_step": 34 / 3,
                "train/loss_epoch": 34 / 8, "train/body_mse_epoch": 68 / 8}
    for rank in range(2):
        assert json.loads((tmp_path / f"rank{rank}.json").read_text()) == pytest.approx(expected)
