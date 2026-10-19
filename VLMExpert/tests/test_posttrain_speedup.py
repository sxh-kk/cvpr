"""Real CPU Accelerator/Posttrain paths; no CUDA or full-model acceptance claim."""
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import socket
from types import SimpleNamespace

from accelerate import Accelerator
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

from psi.config.config import TrainConfig
from psi.trainers.posttrain import PosttrainTrainer


GROUPS = ("action_header", "lang_backbone", "vision_tower", "mm_projector", "motion_expert")


class _Rows(Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {"row": index, "value": torch.tensor([index / 32, 1.0, (index % 3) / 5]),
                "target": torch.tensor([0.1, -0.2, 0.3])}


class _FiveGroups(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(3, 3) * 0.1) for _ in GROUPS])

    def forward(self, value):
        for weight in self.weights:
            value = value + 0.1 * (value @ weight)
        # Exercise RNG preservation across loader preparation and microbatches.
        return value + torch.rand_like(value) * 0.001


class _TinyPosttrain(PosttrainTrainer):
    def __init__(self, directory, *, keep_cpu=False, update_metrics=False, length=16):
        train = TrainConfig(keep_train_batch_on_cpu=keep_cpu, sync_metrics_on_update_only=update_metrics,
                            mixed_precision="no", training_safety_checks=True,
                            gradient_accumulation_steps=2, max_grad_norm=0.25)
        self.cfg = SimpleNamespace(train=train, data=SimpleNamespace(),
                                   model=SimpleNamespace(action_architecture="dual_source_v1"),
                                   log=SimpleNamespace(log_freq=100, report_to=None))
        self.device = torch.device("cpu")
        self._directory = str(directory)
        self.model = _FiveGroups()
        groups = [{"params": [parameter], "group_name": name, "lr": 0.001 * (i + 1)}
                  for i, (name, parameter) in enumerate(zip(GROUPS, self.model.parameters()))]
        self.optimizer = torch.optim.AdamW(groups, foreach=False, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1 / (1 + step / 10))
        self.train_dataloader = DataLoader(_Rows(length), batch_size=2, shuffle=True,
                                           generator=torch.Generator().manual_seed(745), num_workers=0)
        self.val_dataloader = DataLoader(_Rows(4), batch_size=2, num_workers=0)
        self._grad_norms, self._grad_norm_act = {}, None
        self.inject_loss_at = None
        self.last_local_loss = None

    @property
    def project_dir(self):
        return self._directory

    def forward_and_loss(self, model, batch):
        # The actual trainer must have captured the CPU inputs before forward.
        snapshot = self._training_safety.snapshots[-1]
        assert snapshot["batch"]["value"].device.type == "cpu"
        torch.testing.assert_close(snapshot["batch"]["value"], batch["value"], rtol=0, atol=0)
        prediction = model(batch["value"])
        loss = (prediction - batch["target"]).square().mean()
        if self.local_step == self.inject_loss_at:
            loss = loss * float("nan")
        self.last_local_loss = loss.detach().clone()
        return {"loss": loss, "vlm_feat_norm": prediction.detach().norm()}


def _seed():
    random.seed(105)
    np.random.seed(106)
    torch.manual_seed(107)


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _digest(model):
    return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()


def _healthy(directory, *, keep_cpu, update_metrics, distributed=False):
    _seed()
    accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=2)
    trainer = _TinyPosttrain(directory, keep_cpu=keep_cpu, update_metrics=update_metrics,
                             length=32 if distributed else 16)
    loader = trainer.prepare(accelerator)
    assert loader.device is None if keep_cpu else loader.device.type == "cpu"
    # Train-only override must not change the validation path.
    assert trainer.val_dataloader.device.type == "cpu"
    calls = {"loss_gate": 0, "gradient_gate": 0, "metric_gather": 0}
    guard = trainer._training_safety
    loss_gate, gradient_gate, gather = guard.check_loss, guard.check_gradients, accelerator.gather

    def checked_loss(value):
        calls["loss_gate"] += 1
        return loss_gate(value)

    def checked_gradients(values):
        calls["gradient_gate"] += 1
        return gradient_gate(values)

    def gathered(value):
        calls["metric_gather"] += 1
        return gather(value)

    guard.check_loss, guard.check_gradients, accelerator.gather = checked_loss, checked_gradients, gathered
    rows, metrics, hashes = [], [], []
    update = 0
    first = [p.detach().clone() for p in trainer.model.parameters()]
    for local, batch in enumerate(loader):
        assert batch["value"].device.type == "cpu"
        rows.extend(batch["row"].tolist())
        synced, result = trainer.step(batch, update, local)
        assert synced == (local % 2 == 1)
        if not synced:
            assert result == {} if update_metrics else "loss" in result
            if local == 0:
                for before, actual in zip(first, trainer.model.parameters()):
                    torch.testing.assert_close(before, actual, rtol=0, atol=0)
        else:
            # Metrics keep the LAST microbatch rather than an accumulation mean.
            expected = gather(trainer.last_local_loss).mean().item()
            assert result["loss"] == expected
            metrics.append(result)
            digest = _digest(trainer.model)
            if distributed:
                rank_hashes = [None, None]
                dist.all_gather_object(rank_hashes, digest)
                assert len(set(rank_hashes)) == 1
            hashes.append(digest)
            update += 1
    assert update == 4
    assert calls == {"loss_gate": 8, "gradient_gate": 4, "metric_gather": 4 if update_metrics else 8}
    assert not (Path(directory) / "diagnostics").exists()
    assert len(guard.snapshots) == 2
    for snapshot in guard.snapshots:
        assert set(snapshot["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
        assert snapshot["rng"]["torch_cuda"] is None
    result = {"weights": _cpu_copy(trainer.model.state_dict()), "optimizer": _cpu_copy(trainer.optimizer.state_dict()),
              "scheduler": _cpu_copy(trainer.lr_scheduler.state_dict()), "metrics": metrics, "rows": rows,
              "rng": torch.get_rng_state().clone(), "hashes": hashes, "calls": calls}
    assert not torch.cuda.is_initialized()
    accelerator.free_memory()
    return result


def _assert_equivalent(left, right):
    def equal(a, b):
        assert type(a) is type(b)
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for aa, bb in zip(a, b):
                equal(aa, bb)
        elif isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            assert a == b
    for key in ("weights", "optimizer", "scheduler", "metrics", "rows", "rng", "hashes"):
        equal(left[key], right[key])


@pytest.mark.parametrize("keep_cpu,update_metrics", [(True, False), (False, True), (True, True)])
def test_real_accumulation_two_flags_preserve_updates_rng_last_micro_metrics(tmp_path, keep_cpu, update_metrics):
    baseline = _healthy(tmp_path / "baseline", keep_cpu=False, update_metrics=False)
    optimized = _healthy(tmp_path / "optimized", keep_cpu=keep_cpu, update_metrics=update_metrics)
    _assert_equivalent(baseline, optimized)


@pytest.mark.parametrize("failure", ["loss_first", "loss_second", "gradient_inf"])
def test_optimized_actual_step_keeps_finite_gates_and_complete_cpu_diagnostics(tmp_path, failure):
    _seed()
    accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=2)
    trainer = _TinyPosttrain(tmp_path, keep_cpu=True, update_metrics=True)
    trainer.prepare(accelerator)
    if failure.startswith("loss"):
        trainer.inject_loss_at = 0 if failure == "loss_first" else 1
    else:
        next(trainer.model.parameters()).register_hook(
            lambda gradient: torch.full_like(gradient, float("inf")) if trainer.local_step == 1 else gradient)
    before = _cpu_copy(trainer.model.state_dict())
    batches = [torch.utils.data.default_collate([_Rows(4)[index], _Rows(4)[index + 1]]) for index in (0, 2)]
    with pytest.raises(FloatingPointError):
        for local, batch in enumerate(batches):
            trainer.step(batch, 0, local)
    torch.testing.assert_close(before, trainer.model.state_dict(), rtol=0, atol=0)
    assert not trainer.optimizer.state
    assert trainer.lr_scheduler.state_dict()["last_epoch"] == 0
    files = list((tmp_path / "diagnostics").glob("*.pt"))
    assert len(files) == 1
    dump = torch.load(files[0], map_location="cpu", weights_only=False)
    assert dump["stage"] == ("gradients" if failure == "gradient_inf" else "loss")
    assert dump["details"]["nonfinite_ranks"] == [0]
    assert len(dump["microbatches"]) == (1 if failure == "loss_first" else 2)
    for snapshot, original in zip(dump["microbatches"], batches):
        torch.testing.assert_close(snapshot["batch"], original, rtol=0, atol=0)
        assert set(snapshot["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
        assert snapshot["rng"]["torch_cpu"].device.type == "cpu"
    assert not torch.cuda.is_initialized()
    accelerator.free_memory()


def _distributed_worker(rank, directory, port):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    root = Path(directory)
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=45))
    try:
        baseline = _healthy(root / f"baseline_rank{rank}", keep_cpu=False, update_metrics=False, distributed=True)
        optimized = _healthy(root / f"optimized_rank{rank}", keep_cpu=True, update_metrics=True, distributed=True)
        _assert_equivalent(baseline, optimized)
        # Both optimized gates must still fail collectively on a rank-local
        # numerical error, including the non-update microbatch's loss gate.
        for stage, failed_rank in (("loss", 1), ("gradients", 0)):
            _seed()
            accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=2)
            trainer = _TinyPosttrain(root / stage, keep_cpu=True, update_metrics=True, length=8)
            loader = trainer.prepare(accelerator)
            if stage == "loss" and rank == failed_rank:
                trainer.inject_loss_at = 0
            if stage == "gradients" and rank == failed_rank:
                unscale = accelerator.unscale_gradients

                def local_inf_after_ddp_and_unscale(*args, **kwargs):
                    result = unscale(*args, **kwargs)
                    # Inject after DDP reduction so only this rank is invalid;
                    # a backward hook would correctly propagate Inf to both.
                    with torch.no_grad():
                        next(p.grad for p in trainer.model.parameters() if p.grad is not None).fill_(float("inf"))
                    return result

                accelerator.unscale_gradients = local_inf_after_ddp_and_unscale
            before = _cpu_copy(trainer.model.state_dict())
            iterator = iter(loader)
            with pytest.raises(FloatingPointError):
                for local, batch in enumerate(iterator):
                    synced, result = trainer.step(batch, 0, local)
                    assert not synced and result == {}
            torch.testing.assert_close(trainer.model.state_dict(), before, rtol=0, atol=0)
            assert not trainer.optimizer.state
            files = list((root / stage / "diagnostics").glob("*.pt"))
            assert len(files) == 2  # Production CPU barrier completed both writes.
            dump = torch.load(trainer._training_safety.last_dump, map_location="cpu", weights_only=False)
            assert dump["details"]["nonfinite_ranks"] == [failed_rank]
            assert len(dump["microbatches"]) == (1 if stage == "loss" else 2)
            assert all(s["batch"]["value"].device.type == "cpu" for s in dump["microbatches"])
            # Finish only the small CPU iterator so Accelerator releases its
            # active-loader bookkeeping before the next independent case.
            list(iterator)
            accelerator.free_memory()
        (root / f"rank{rank}.json").write_text(json.dumps({"rows": optimized["rows"], "hashes": optimized["hashes"],
                                                         "calls": optimized["calls"], "collective_failures": ["loss", "gradients"]}))
    finally:
        dist.destroy_process_group()


def test_two_real_cpu_gloo_ranks_preserve_sharding_rng_and_four_consistent_updates(tmp_path):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    mp.spawn(_distributed_worker, args=(str(tmp_path), port), nprocs=2, join=True)
    reports = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert set(reports[0]["rows"]).isdisjoint(reports[1]["rows"])
    assert sorted(reports[0]["rows"] + reports[1]["rows"]) == list(range(32))
    assert reports[0]["hashes"] == reports[1]["hashes"]
    assert all(report["collective_failures"] == ["loss", "gradients"] for report in reports)


@pytest.mark.parametrize("max_norm", [None, 0.25])
def test_batched_clip_metrics_preserve_mixed_precision_norms_and_gradients(tmp_path, max_norm):
    results = []
    for enabled in (False, True):
        trainer = _TinyPosttrain(tmp_path, update_metrics=enabled)
        trainer.train_cfg.max_grad_norm = max_norm
        parameters = [torch.nn.Parameter(torch.ones(3, dtype=torch.float64 if i % 2 else torch.float32))
                      for i in range(5)]
        for i, parameter in enumerate(parameters):
            parameter.grad = torch.arange(1, 4, dtype=parameter.dtype) * (i + 0.125)
        trainer.model = torch.nn.ParameterList(parameters)
        trainer.optimizer = torch.optim.SGD([{"params": [parameter], "group_name": name}
                                            for name, parameter in zip(GROUPS, parameters)], lr=0.01)
        trainer.accelerator = SimpleNamespace(clip_grad_norm_=torch.nn.utils.clip_grad_norm_)
        norms = trainer.clip_grad_norms()
        results.append((norms, [p.grad.clone() for p in parameters]))
    assert results[0][0] == results[1][0]
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)


def test_defaults_leave_both_optimizations_disabled():
    cfg = TrainConfig()
    assert cfg.keep_train_batch_on_cpu is False
    assert cfg.sync_metrics_on_update_only is False
