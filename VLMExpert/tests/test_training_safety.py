"""CPU gates through the actual Posttrain methods; no production-model/GPU claim."""

import contextlib
from datetime import timedelta
import json
from pathlib import Path
import pickle
import random
import time
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from psi.config.config import TrainConfig
from psi.trainers import posttrain
from psi.trainers.training_safety import TrainingSafetyGuard


def _load_dump(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_capture_is_detached_bounded_preserves_rng_and_has_no_healthy_disk_writes(tmp_path):
    guard = TrainingSafetyGuard(tmp_path / "diagnostics", device="cpu", accumulation_steps=2)
    before = (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone())
    value = torch.tensor([1.0], requires_grad=True)
    for local_step in range(3):
        guard.capture({"value": value, "array": np.array([local_step])}, global_step=0, local_step=local_step)
        guard.check_loss(torch.tensor(1.0))
    assert len(guard.snapshots) == 2
    assert [item["local_step"] for item in guard.snapshots] == [1, 2]
    saved = guard.snapshots[-1]["batch"]["value"]
    assert saved.device.type == "cpu" and not saved.requires_grad and saved.grad_fn is None
    with torch.no_grad():
        value.add_(4)
    torch.testing.assert_close(saved, torch.tensor([1.0]))
    assert before[0] == pickle.dumps(random.getstate())
    assert before[1] == pickle.dumps(np.random.get_state())
    assert torch.equal(before[2], torch.get_rng_state())
    assert not (tmp_path / "diagnostics").exists()
    guard.capture({"value": value}, global_step=1, local_step=3)
    assert len(guard.snapshots) == 1


def test_dump_uses_cached_cpu_batch_and_current_cuda_rng_without_querying_cuda_again(tmp_path, monkeypatch):
    # Only the RNG API is simulated; this test allocates no CUDA tensors.
    cuda_rng = torch.arange(8, dtype=torch.uint8)
    requested = []
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: requested.append(str(device)) or cuda_rng)
    guard = TrainingSafetyGuard(tmp_path, device="cuda:3")
    guard.capture({"value": torch.tensor([3.0]), "nested": [torch.tensor([2])]}, global_step=9, local_step=19)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda *_: pytest.fail("CUDA queried after failure"))
    guard.stage = "forward"
    path = guard.dump("CUDA error: device-side assert triggered")
    payload = _load_dump(path)
    assert requested == ["cuda:3"]
    assert (payload["global_step"], payload["local_step"], payload["rank"], payload["stage"]) == (9, 19, 0, "forward")
    snapshot = payload["microbatches"][0]
    assert set(snapshot["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert torch.equal(snapshot["rng"]["torch_cuda"], cuda_rng)
    assert snapshot["batch"]["value"].device.type == "cpu"
    assert guard.dump("second exception handler") == path
    assert len(list(tmp_path.glob("*.pt"))) == 1


def _distributed_gate_worker(rank, root):
    root = Path(root)
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        guard = TrainingSafetyGuard(root / "diagnostics", device="cpu", rank=rank, world_size=2)
        dump = guard.dump

        def delayed_dump(*args, **kwargs):
            if rank == 1:
                time.sleep(0.2)
            return dump(*args, **kwargs)

        guard.dump = delayed_dump
        outcomes = []
        for step, stage in enumerate(("loss", "gradients")):
            guard.capture({"rank_input": torch.tensor([rank])}, global_step=step, local_step=step)
            try:
                if stage == "loss":
                    guard.check_loss(torch.tensor(float("nan") if rank == 1 else 1.0))
                else:
                    parameter = torch.nn.Parameter(torch.ones(2))
                    parameter.grad = torch.tensor([1.0, float("inf") if rank == 1 else 2.0])
                    guard.check_gradients([parameter])
            except FloatingPointError:
                payload = _load_dump(guard.last_dump)
                assert payload["stage"] == stage and payload["details"]["nonfinite_ranks"] == [1]
                assert payload["rank"] == rank and len(payload["microbatches"]) == 1
                # Even the fast rank cannot return/raise before the slow rank
                # has committed its CPU diagnostic.
                assert len(list((root / "diagnostics").glob(f"*_{stage}.pt"))) == 2
                outcomes.append(stage)
        (root / f"rank{rank}.json").write_text(json.dumps(outcomes))
    finally:
        dist.destroy_process_group()


def test_two_actual_cpu_ranks_agree_on_loss_and_gradient_failure_and_both_dump(tmp_path):
    mp.spawn(_distributed_gate_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)] == [
        ["loss", "gradients"], ["loss", "gradients"]]
    assert len(list((tmp_path / "diagnostics").glob("*.pt"))) == 4


class _Optimizer:
    def __init__(self, parameters, events):
        self.optimizer = torch.optim.SGD(parameters, lr=0.1)
        self.param_groups = self.optimizer.param_groups
        self.events = events

    def step(self):
        self.events.append("step")
        self.optimizer.step()

    def zero_grad(self):
        self.events.append("zero_grad")
        self.optimizer.zero_grad()


def _trainer(tmp_path, monkeypatch, *, enabled=True, failure=None):
    events = []

    class TracedGuard(TrainingSafetyGuard):
        def capture(self, *args, **kwargs):
            events.append("capture")
            return super().capture(*args, **kwargs)

        def check_loss(self, *args):
            events.append("check_loss")
            return super().check_loss(*args)

        def check_gradients(self, *args):
            events.append("check_gradients")
            return super().check_gradients(*args)

    monkeypatch.setattr(posttrain, "TrainingSafetyGuard", TracedGuard)
    clip = torch.nn.utils.clip_grad_norm_

    def traced_clip(*args, **kwargs):
        events.append("clip")
        return clip(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", traced_clip)
    model = torch.nn.Linear(2, 1, bias=False)
    if failure == "gradients":
        model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
    optimizer = _Optimizer(model.parameters(), events)

    def forward_and_loss(model, batch):
        events.append("forward")
        if failure == "exception":
            raise RuntimeError("CUDA error: device-side assert triggered")
        loss = model(batch["value"]).square().mean()
        return {"loss": loss * float("nan") if failure == "loss" else loss}

    def backward(loss):
        events.append("backward")
        loss.backward()

    def unscale(_optimizer):
        events.append("unscale")
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(2)

    accelerator = SimpleNamespace(
        process_index=0, num_processes=1, sync_gradients=True, is_main_process=False,
        accumulate=lambda _: contextlib.nullcontext(), backward=backward, unscale_gradients=unscale,
        clip_grad_norm_=traced_clip, gather=lambda value: value.detach(),
    )
    trainer = SimpleNamespace(
        train_cfg=TrainConfig(training_safety_checks=enabled, max_grad_norm=1.0, gradient_accumulation_steps=2),
        cfg=SimpleNamespace(log=SimpleNamespace(log_freq=100, report_to=None)), device=torch.device("cpu"),
        project_dir=str(tmp_path), model=model, optimizer=optimizer, accelerator=accelerator,
        lr_scheduler=SimpleNamespace(step=lambda: events.append("schedule")),
        global_step=1, local_step=2, forward_autocast=contextlib.nullcontext,
        forward_and_loss=forward_and_loss, _grad_norms={}, _grad_norm_act=None,
        lr=0.1, group_lrs=lambda: {}, grad_norm_metrics=lambda: {},
    )
    trainer.clip_grad_norms = lambda **kwargs: posttrain.PosttrainTrainer.clip_grad_norms(trainer, **kwargs)
    trainer._get_training_safety = lambda: posttrain.PosttrainTrainer._get_training_safety(trainer)
    trainer._training_step = lambda *args, **kwargs: posttrain.PosttrainTrainer._training_step(trainer, *args, **kwargs)
    return trainer, events


@pytest.mark.parametrize("failure", ["loss", "gradients", "exception"])
def test_actual_trainer_failure_never_reaches_update_and_preserves_cpu_diagnostic(tmp_path, monkeypatch, failure):
    trainer, events = _trainer(tmp_path, monkeypatch, failure=failure)
    before = trainer.model.weight.detach().clone()
    with pytest.raises(FloatingPointError if failure != "exception" else RuntimeError):
        posttrain.PosttrainTrainer.training_step(trainer, {"value": torch.ones(2, 2)})
    assert "clip" not in events and "step" not in events and "schedule" not in events
    if failure == "loss":
        assert "backward" not in events
    if failure == "gradients":
        assert events.index("unscale") < events.index("check_gradients")
    torch.testing.assert_close(trainer.model.weight, before)
    dumps = list((tmp_path / "diagnostics").glob("*.pt"))
    assert len(dumps) == 1
    payload = _load_dump(dumps[0])
    assert payload["stage"] == ("forward" if failure == "exception" else failure)
    torch.testing.assert_close(payload["microbatches"][0]["batch"]["value"], torch.ones(2, 2))


def test_actual_trainer_checks_loss_then_unscaled_gradients_before_clipping_and_update(tmp_path, monkeypatch):
    trainer, events = _trainer(tmp_path, monkeypatch)
    posttrain.PosttrainTrainer.training_step(trainer, {"value": torch.ones(2, 2)})
    assert events == ["capture", "forward", "check_loss", "backward", "unscale", "check_gradients",
                      "clip", "step", "schedule", "zero_grad"]
    assert not (tmp_path / "diagnostics").exists()


def test_actual_legacy_trainer_keeps_original_path_when_safety_is_disabled(tmp_path, monkeypatch):
    trainer, events = _trainer(tmp_path, monkeypatch, enabled=False)
    posttrain.PosttrainTrainer.training_step(trainer, {"value": torch.ones(2, 2)})
    assert events == ["forward", "backward", "clip", "step", "schedule", "zero_grad"]
    assert not hasattr(trainer, "_training_safety")


def test_default_is_compatible_and_unsupported_parallel_modes_fail_explicitly():
    assert TrainConfig().training_safety_checks is False
    with pytest.raises(ValidationError, match="DDP only"):
        TrainConfig(training_safety_checks=True, data_parallel="fsdp")


def test_real_cpu_accelerator_accumulation_two_is_bitexact_with_and_without_safety(tmp_path):
    from accelerate import Accelerator

    def run(enabled):
        accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=2)
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.25)
        optimizer = torch.optim.SGD([{"params": model.parameters(), "group_name": "action_header"}],
                                    lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        calls = {"unscale": 0, "clip": 0}
        unscale = accelerator.unscale_gradients

        def counted_unscale(*args, **kwargs):
            calls["unscale"] += 1
            return unscale(*args, **kwargs)

        accelerator.unscale_gradients = counted_unscale
        trainer = SimpleNamespace(
            train_cfg=TrainConfig(training_safety_checks=enabled, max_grad_norm=1.0, gradient_accumulation_steps=2),
            cfg=SimpleNamespace(log=SimpleNamespace(log_freq=100, report_to=None)), device=accelerator.device,
            project_dir=str(tmp_path / str(enabled)), model=model, optimizer=optimizer, accelerator=accelerator,
            lr_scheduler=scheduler, global_step=0, local_step=0, forward_autocast=accelerator.autocast,
            forward_and_loss=lambda model, batch: {"loss": model(batch["value"]).square().mean()},
            _grad_norms={}, _grad_norm_act=None, lr=0.1, group_lrs=lambda: {}, grad_norm_metrics=lambda: {},
        )

        def counted_clip(**kwargs):
            calls["clip"] += 1
            return posttrain.PosttrainTrainer.clip_grad_norms(trainer, **kwargs)

        trainer.clip_grad_norms = counted_clip
        trainer._get_training_safety = lambda: posttrain.PosttrainTrainer._get_training_safety(trainer)
        trainer._training_step = lambda *args, **kwargs: posttrain.PosttrainTrainer._training_step(trainer, *args, **kwargs)
        before = model.weight.detach().clone()
        synchronized = []
        for microstep in range(2):
            trainer.local_step = microstep
            result = posttrain.PosttrainTrainer.training_step(trainer, {"value": torch.ones(2, 2) * (microstep + 1)})
            synchronized.append(result[0])
            if microstep == 0:
                torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
                assert not optimizer.state
        assert synchronized == [False, True]
        assert calls == {"unscale": 1, "clip": 1}
        assert scheduler.state_dict()["last_epoch"] == 1
        if enabled:
            assert len(trainer._training_safety.snapshots) == 2
            assert not (Path(trainer.project_dir) / "diagnostics").exists()
        result = (model.weight.detach().clone(), optimizer.state_dict(), scheduler.state_dict())
        accelerator.free_memory()
        return result

    baseline, guarded = run(False), run(True)
    torch.testing.assert_close(baseline[0], guarded[0], rtol=0, atol=0)
    assert baseline[1]["param_groups"] == guarded[1]["param_groups"]
    for key in baseline[1]["state"]:
        torch.testing.assert_close(baseline[1]["state"][key]["momentum_buffer"],
                                   guarded[1]["state"][key]["momentum_buffer"], rtol=0, atol=0)
    assert baseline[2] == guarded[2]


def test_prepare_initializes_cpu_coordinator_before_any_first_batch(tmp_path):
    events = []

    class Loader:
        def __iter__(self):
            events.append("first_batch")
            assert events == ["safety", "first_batch"]
            return iter([{"sample": 1}])

    config = TrainConfig(training_safety_checks=True, overfit_single_batch=True)
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(train=config), train_cfg=config, data_cfg=SimpleNamespace(),
        model=object(), optimizer=object(), lr_scheduler=object(), train_dataloader=Loader(),
    )
    accelerator = SimpleNamespace(prepare=lambda *items: items if len(items) > 1 else items[0])

    def get_safety():
        assert trainer.accelerator is accelerator
        events.append("safety")

    trainer._get_training_safety = get_safety
    loader = posttrain.PosttrainTrainer.prepare(trainer, accelerator)
    assert next(iter(loader)) == {"sample": 1}
    assert events == ["safety", "first_batch"]
