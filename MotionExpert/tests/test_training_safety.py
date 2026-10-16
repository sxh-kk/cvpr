"""Safety gates reject bad data/updates collectively without hiding clipping."""

import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from MotionExpert.config import smplh_motion_expert_config
from MotionExpert.training_safety import (TrainingSafety, TrainingSafetyError,
                                         gradient_norm, validate_safety_precision)


def _guard(**overrides):
    config = smplh_motion_expert_config().TRAIN
    for key, value in overrides.items():
        setattr(config, "SAFETY_" + key.upper(), value)
    return TrainingSafety(config)


def _loss_check(guard, loss, **kwargs):
    return guard.check_forward({"loss": torch.as_tensor(loss)}, device=torch.device("cpu"), **kwargs)


@pytest.mark.parametrize("count", [100000., float("nan"), float("inf")])
def test_forward_ignores_valid_counts(count):
    guard = _guard()
    normal = guard.check_forward({"loss": torch.tensor(.1), "body_valid_count": torch.tensor(count)},
                                 device="cpu")
    assert not normal["reasons"]
    assert normal["global_loss_component_abs_max"] == pytest.approx(.1)
    assert normal["global_objective_loss_abs_max"] == pytest.approx(.1)


@pytest.mark.parametrize("name", ["global_rotation_mse", "global_mse", "moe_aux_loss"])
def test_finite_group_or_auxiliary_diagnostic_217_keeps_its_max_without_objective_stop(name):
    guard = _guard()
    result = guard.check_forward({"loss": torch.tensor(.1), name: torch.tensor([.01, 217.])},
                                  device="cpu", training=True)
    assert not result["reasons"]
    assert result["global_loss_component_abs_max"] == 217.
    assert result["global_objective_loss_abs_max"] == pytest.approx(.1)
    assert list(guard.history) == pytest.approx([.1])


@pytest.mark.parametrize("name", ["loss", "flow_loss", "per_sample_flow_loss"])
@pytest.mark.parametrize("shape", [(), (1,), (2,)])
def test_objective_gate_uses_explicit_names_for_scalar_singleton_and_per_sample_values(name, shape):
    value = torch.full(shape, 101.)
    if shape == (2,):
        value[0] = .1  # The batch mean is below 100; the corrupt sample must stop.
    metrics = {"loss": torch.tensor(.1), "global_rotation_mse": torch.tensor(217.), name: value}
    result = _guard().check_forward(metrics, device="cpu")
    assert result["reasons"] == ["objective loss exceeds absolute limit"]
    assert result["global_objective_loss_abs_max"] == 101.
    assert result["global_loss_component_abs_max"] == 217.


@pytest.mark.parametrize("name", ["global_rotation_mse", "moe_aux_loss"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_group_or_auxiliary_still_stops_when_magnitude_gates_disabled(name, bad):
    result = _guard(enabled=False).check_forward({"loss": torch.tensor(.1), name: torch.tensor([bad])},
                                                 device="cpu")
    assert result["reasons"] == ["non-finite loss or loss component"]
    assert result["global_objective_loss_abs_max"] == pytest.approx(.1)


def test_forward_uses_one_small_collective_for_both_magnitude_scopes(monkeypatch):
    from MotionExpert import training_safety
    sizes = []
    def reduce(values):
        sizes.append(values.numel())
        return values
    monkeypatch.setattr(training_safety, "collective_max", reduce)
    result = _guard().check_forward({"loss": torch.tensor(.1), "global_rotation_mse": torch.tensor(217.)},
                                    device="cpu")
    assert not result["reasons"]
    assert sizes == [7]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_always_stops_even_when_magnitude_gates_disabled(bad):
    result = _loss_check(_guard(enabled=False), bad)
    assert "non-finite loss or loss component" in result["reasons"]


def test_rolling_gate_needs_healthy_warmup_and_keeps_baseline_on_resume():
    guard = _guard(rolling_window=4, rolling_warmup_steps=3, rolling_multiplier=10.)
    for loss in (.1, .11, .09):
        assert not _loss_check(guard, loss, training=True)["reasons"]
    restored = _guard(rolling_window=4, rolling_warmup_steps=3, rolling_multiplier=10.)
    restored.load_state_dict(guard.state_dict())
    assert _loss_check(restored, 2., training=True)["reasons"] == ["loss exceeds healthy rolling baseline"]
    assert len(restored.history) == 3  # A failed batch never poisons the baseline.
    assert not _loss_check(restored, 2., training=False)["reasons"]
    assert len(restored.history) == 3  # Validation never changes training history.


def test_hard_gate_is_active_from_first_step_and_token_rms_is_checked():
    assert _loss_check(_guard(), 1000., training=True)["reasons"]
    result = _loss_check(_guard(), .1, activation_rms=torch.tensor([1., 101.]))
    assert result["reasons"] == ["hidden token RMS exceeds absolute limit"]


def test_nonfinite_prediction_cannot_hide_behind_zero_label_masks():
    result = _loss_check(_guard(), .1, prediction=torch.tensor([0., float("nan")]))
    assert result["reasons"] == ["non-finite model prediction"]


def test_gradient_gate_uses_unclipped_values_and_does_not_modify_gradients():
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    parameter.grad = torch.tensor([3000., 4000.])
    original = parameter.grad.clone()
    guard = _guard()
    assert gradient_norm(optimizer, "cpu").item() == pytest.approx(5000.)
    result = guard.check_gradients(optimizer, device="cpu")
    assert result["reasons"] == ["unscaled gradient norm exceeds absolute limit"]
    assert parameter.grad.equal(original)
    torch.nn.utils.clip_grad_norm_([parameter], 1.)
    assert not guard.check_gradients(optimizer, device="cpu")["reasons"]
    parameter.grad[0] = float("nan")
    assert "non-finite unscaled gradient norm" in guard.check_gradients(optimizer, device="cpu")["reasons"]


def test_precision_guard_only_rejects_fused_fp16_on_gpu():
    config = SimpleNamespace(PRECISION="16-mixed", FUSED_ADAMW=True)
    with pytest.raises(ValueError, match="unscaled"):
        validate_safety_precision(config, use_gpu=True)
    validate_safety_precision(config, use_gpu=False)
    config.PRECISION = "bf16-mixed"
    validate_safety_precision(config, use_gpu=True)


def _distributed_gates_worker(rank, world, init_path, output):
    from datetime import timedelta
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank,
                                         world_size=world, timeout=timedelta(seconds=30))
    try:
        guard = _guard()
        healthy = [_loss_check(guard, .1 + rank * .01, training=True) for _ in range(3)]
        diagnostic = guard.check_forward({"loss": torch.tensor(.1),
            "global_rotation_mse": torch.tensor([.1 if rank == 0 else 217.])}, device="cpu")
        # Rank zero sees an ordinary batch. Rank one must stop both ranks before
        # either can enter a mismatched backward collective.
        forward = guard.check_forward({"loss": torch.tensor(.1),
            "per_sample_flow_loss": torch.tensor([.1 if rank == 0 else 101.])}, device="cpu", training=True)
        group_nan = guard.check_forward({"loss": torch.tensor(.1),
            "global_rotation_mse": torch.tensor(.1 if rank == 0 else float("nan"))}, device="cpu")
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        parameter.grad = torch.tensor([.5 if rank == 0 else float("nan")])
        gradient = guard.check_gradients(optimizer, device="cpu")
        Path(output, f"rank{rank}.json").write_text(json.dumps({
            "healthy": [result["reasons"] for result in healthy],
            "forward": forward["reasons"], "gradient": gradient["reasons"],
            "diagnostic": diagnostic, "group_nan": group_nan["reasons"],
            "history": list(guard.history)}))
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_cpu_gates_reach_identical_decision_for_one_bad_rank(tmp_path):
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_distributed_gates_worker,
                                args=(rank, 2, str(tmp_path / "rendezvous"), str(tmp_path))) for rank in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(45)
    try:
        assert all(process.exitcode == 0 for process in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
    results = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert results[0] == results[1]
    assert results[0]["healthy"] == [[], [], []]
    assert not results[0]["diagnostic"]["reasons"]
    assert results[0]["diagnostic"]["global_loss_component_abs_max"] == 217.
    assert results[0]["diagnostic"]["global_objective_loss_abs_max"] == pytest.approx(.1)
    assert results[0]["forward"] == ["objective loss exceeds absolute limit"]
    assert results[0]["group_nan"] == ["non-finite loss or loss component"]
    assert "non-finite unscaled gradient norm" in results[0]["gradient"]


class _TinySafetyModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.1))
        self.feature_calls = 0

    def forward(self, x, t, y, return_features=False):
        prediction = self.weight.expand_as(x)
        if return_features:
            self.feature_calls += 1
            hidden = self.weight.expand(x.shape[0], x.shape[1], 768)
            return prediction, hidden, hidden.mean(1)
        return prediction

    def get_moe_auxiliary_loss(self):
        return None


@pytest.fixture
def tiny_module(tmp_path, monkeypatch):
    from MotionExpert.module import uem_module
    from MotionExpert.tests.fixtures import smplh_stats
    monkeypatch.setattr(uem_module, "UniEgoMotion", _TinySafetyModel)
    config = smplh_motion_expert_config()
    config.DATA.STATS_PATH = str(tmp_path / "stats.pt")
    config.TRAIN.EXP_PATH = str(tmp_path / "outside-repo-output")
    torch.save(smplh_stats(config), config.DATA.STATS_PATH)
    module = uem_module.UEM_Module(config)
    module.log = lambda *args, **kwargs: None
    return module


def _batch():
    from MotionExpert.tests.fixtures import conditioning
    return {"y": conditioning(batch=2, frames=11), "misc": {
        "motion": torch.zeros(2, 11, 245), "seq_name": ["egosuite/test", "hm/test"],
        "start_idx": torch.tensor([0, 6]), "source": ["egosuite", "egoexo4d_hm"]}}


def test_healthy_step_has_no_disk_io_and_validation_always_checks_pooled_hidden(tiny_module):
    module = tiny_module
    module.on_validation_epoch_start()
    assert module.cfg.MODEL.GLOBAL_TOKEN_COUNT == 0
    module.validation_step(_batch(), 0)
    assert module.model.feature_calls == 1
    assert not Path(module.cfg.TRAIN.EXP_PATH).exists()
    module.training_step(_batch(), 0).backward()
    optimizer = torch.optim.SGD(module.parameters(), lr=.1)
    module.on_before_optimizer_step(optimizer)
    assert not Path(module.cfg.TRAIN.EXP_PATH).exists()
    assert module._safety_microbatches == []


def test_failure_saves_exact_batch_and_forward_noise_state_without_model_ckpt(tiny_module):
    module = tiny_module
    module.model.weight.data.fill_(100.)
    batch = _batch()
    before = module.model.weight.detach().clone()
    with pytest.raises(TrainingSafetyError, match="before optimizer update"):
        module.training_step(batch, 7)
    assert module.model.weight.equal(before)
    root = Path(module.cfg.TRAIN.EXP_PATH) / "anomalies"
    summary_file, = root.glob("*/summary.json")
    summary = json.loads(summary_file.read_text())
    assert summary["batch_idx"] == 7
    assert summary["optimizer_update_applied"] is False
    assert summary["batch_summary"]["misc"]["seq_name"] == ["egosuite/test", "hm/test"]
    payload = torch.load(summary_file.parent / "batch.pt", weights_only=False)
    assert payload["batch"]["misc"]["motion"].equal(batch["misc"]["motion"])
    assert payload["flow_diagnostics"]["x_t"].shape == (2, 11, 245)
    assert "torch_cpu" in payload["rng_before_step"]
    assert len(payload["accumulated_microbatches"]) == 1
    assert not list(root.rglob("*.ckpt"))


def test_accumulated_gradient_failure_prevents_adamw_and_preserves_all_microbatches(tiny_module):
    module = tiny_module
    optimizer = torch.optim.AdamW(module.parameters(), lr=.1)
    before = module.model.weight.detach().clone()
    for index in range(2):
        module.training_step(_batch(), index).backward()
    module.model.weight.grad.fill_(2000.)
    with pytest.raises(TrainingSafetyError, match="gradient norm"):
        module.on_before_optimizer_step(optimizer)
        optimizer.step()  # This update must never execute.
    assert module.model.weight.equal(before)
    assert optimizer.state == {}
    assert module.model.weight.grad.item() == 2000.  # Not hidden by clipping.
    batch_file, = Path(module.cfg.TRAIN.EXP_PATH).glob("anomalies/*/batch.pt")
    payload = torch.load(batch_file, weights_only=False)
    assert [context["batch_idx"] for context in payload["accumulated_microbatches"]] == [0, 1]


def _distributed_sample_hook_worker(rank, world, init_path, directory, kind):
    from datetime import timedelta
    from MotionExpert.module import uem_module
    from MotionExpert import evaluate
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank,
                                         world_size=world, timeout=timedelta(seconds=30))
    try:
        uem_module.UniEgoMotion = _TinySafetyModel
        config = smplh_motion_expert_config()
        config.DATA.STATS_PATH = str(Path(directory) / "stats.pt")
        config.TRAIN.EXP_PATH = str(Path(directory) / "outside-repo-output")
        module = uem_module.UEM_Module(config)
        module._trainer = SimpleNamespace(
            current_epoch=4, global_step=9, sanity_checking=False, is_global_zero=rank == 0,
            datamodule=SimpleNamespace(val_dataset=object()))
        logged, calls = {}, []
        module.log = lambda name, value, **kwargs: logged.update({name: float(value)})
        def probe(*args, **kwargs):
            calls.append(rank)
            assert rank == 0
            if kind == "raise":
                raise ValueError("Injected non-finite sampled motion")
            return {"fore/test_metric": float("nan") if kind == "nan" else .125}
        evaluate.sampled_validation = probe
        module.model.train()
        module.on_validation_epoch_start()
        error = None
        try:
            module.on_validation_epoch_end()
        except TrainingSafetyError as exception:
            error = str(exception)
        Path(directory, f"sample-{rank}.json").write_text(json.dumps({
            "failed": error is not None, "mode_restored": module.model.training,
            "calls": calls, "logged": logged}))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("kind", ["raise", "nan", "healthy"])
def test_periodic_sampling_failure_is_shared_before_any_rank_resumes(tmp_path, kind):
    from MotionExpert.tests.fixtures import smplh_stats
    torch.save(smplh_stats(), tmp_path / "stats.pt")
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_distributed_sample_hook_worker,
                                args=(rank, 2, str(tmp_path / "rendezvous"), str(tmp_path), kind))
                 for rank in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(45)
    try:
        assert all(process.exitcode == 0 for process in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
    results = [json.loads((tmp_path / f"sample-{rank}.json").read_text()) for rank in range(2)]
    assert [value["failed"] for value in results] == [kind != "healthy"] * 2
    assert all(value["mode_restored"] for value in results)
    assert results[0]["calls"] == [0] and results[1]["calls"] == []
    if kind == "healthy":
        assert results[0]["logged"] == {"val_sample/fore/test_metric": .125}
        assert not Path(tmp_path / "outside-repo-output").exists()
    else:
        reports = list((tmp_path / "outside-repo-output").glob("anomalies/*/summary.json"))
        assert len(reports) == 2
        assert {json.loads(path.read_text())["rank"] for path in reports} == {0, 1}


def test_periodic_failure_records_current_probe_identity_and_seed(tiny_module, monkeypatch):
    from MotionExpert import evaluate
    module = tiny_module
    dataset = SimpleNamespace(get_sample=lambda index, task: _batch())
    module._trainer = SimpleNamespace(
        current_epoch=4, global_step=9, sanity_checking=False, is_global_zero=True,
        datamodule=SimpleNamespace(val_dataset=dataset))
    def fail(module, probe, **kwargs):
        probe.get_sample(5, task="fore")
        raise ValueError("Injected sampled decode failure")
    monkeypatch.setattr(evaluate, "sampled_validation", fail)
    module.on_validation_epoch_start()
    with pytest.raises(TrainingSafetyError, match="periodic sampled validation"):
        module.on_validation_epoch_end()
    path, = Path(module.cfg.TRAIN.EXP_PATH).glob("anomalies/*/summary.json")
    summary = json.loads(path.read_text())
    assert summary["mode"] == "val_sample" and summary["batch_idx"] == 5
    assert summary["sampling_seed"] == 67 and summary["sampling_task"] == "fore"
    assert summary["batch_summary"]["misc"]["seq_name"] == ["egosuite/test", "hm/test"]
