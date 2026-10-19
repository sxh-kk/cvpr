"""CPU AdamW execution overrides across real production checkpoint restores."""

from copy import deepcopy
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("accelerate")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer

from psi.config.config import DataConfig, LaunchConfig, LoggingConfig, TrainConfig, WandbConfig
from psi.config.model_psi0 import Psi0ModelConfig
from psi.config.transform import DataTransform, FieldTransform, Psi0ModelTransform
from psi.config.transform_psi0_sonic import SonicRepackTransform
from psi.trainers import trainer as trainer_module
from psi.trainers.posttrain import PosttrainTrainer


class _FiveGroups(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(3, 2).double() for _ in range(5)])
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, inputs):
        outputs = [(index + 1) * torch.tanh(layer(inputs)) for index, layer in enumerate(self.layers)]
        return self.dropout(torch.stack(outputs).sum(0))


def _optimizer(model, foreach=False):
    names = ("action_header", "lang_backbone", "vision_tower", "mm_projector", "motion_expert")
    groups = [{"params": list(layer.parameters()), "group_name": name, "lr": (index + 1) * 1e-4}
              for index, (name, layer) in enumerate(zip(names, model.layers))]
    return torch.optim.AdamW(groups, foreach=foreach, betas=(0.8, 0.97), weight_decay=0.002, eps=1e-9)


def _assert_tree(actual, expected, *, exact=True):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0 if exact else 1e-12,
                                   atol=0 if exact else 1e-14)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree(actual[key], expected[key], exact=exact)
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            _assert_tree(value, reference, exact=exact)
    else:
        assert actual == expected


def _without_foreach(state):
    state = deepcopy(state)
    for group in state["param_groups"]:
        group.pop("foreach", None)
    return state


def _rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def _one_update(model, optimizer, scheduler=None, accelerator=None):
    # All three generators affect the batch; Dropout also consumes Torch RNG.
    inputs = torch.randn(7, 3, dtype=torch.float64) + random.random()
    target = torch.tensor(np.random.normal(size=(7, 2)), dtype=torch.float64)
    optimizer.zero_grad(set_to_none=True)
    loss = (model(inputs) - target).square().mean()
    if accelerator is None:
        loss.backward()
    else:
        accelerator.backward(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss.detach().clone()


def _populated_optimizer(wrapped=False):
    model = _FiveGroups()
    optimizer = _optimizer(model)
    _one_update(model, optimizer)
    if wrapped:
        optimizer = _accelerated_optimizer(optimizer)
    return optimizer


def _accelerated_optimizer(optimizer):
    Accelerator(cpu=True, mixed_precision="no")
    return AcceleratedOptimizer(optimizer, device_placement=False)


def _raw(optimizer):
    return optimizer.optimizer if isinstance(optimizer, AcceleratedOptimizer) else optimizer


@pytest.mark.parametrize("wrapped", [False, True])
def test_none_preserves_groups_defaults_and_adam_state(wrapped):
    optimizer = _populated_optimizer(wrapped)
    raw = _raw(optimizer)
    raw.defaults["foreach"] = True
    raw.param_groups[-1]["foreach"] = None
    before = deepcopy(raw.state_dict()), deepcopy(raw.defaults)
    assert trainer_module.apply_optimizer_foreach_override(optimizer, None) is False
    _assert_tree(raw.state_dict(), before[0])
    _assert_tree(raw.defaults, before[1])


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("requested", [False, True])
def test_explicit_override_changes_all_five_groups_without_rebuilding_state(wrapped, requested):
    optimizer = _populated_optimizer(wrapped)
    raw = _raw(optimizer)
    raw.defaults["foreach"] = not requested
    for group, flag in zip(raw.param_groups, (None, True, False, None, True)):
        group["foreach"] = flag
    before = deepcopy(raw.state_dict())
    parameter_ids = [[id(parameter) for parameter in group["params"]] for group in raw.param_groups]
    slot_ids = {id(parameter): {key: id(value) for key, value in slots.items()}
                for parameter, slots in raw.state.items()}
    assert trainer_module.apply_optimizer_foreach_override(optimizer, requested) is True
    assert raw.defaults["foreach"] is requested
    assert all(group["foreach"] is requested for group in raw.param_groups)
    assert parameter_ids == [[id(parameter) for parameter in group["params"]] for group in raw.param_groups]
    assert slot_ids == {id(parameter): {key: id(value) for key, value in slots.items()}
                        for parameter, slots in raw.state.items()}
    _assert_tree(_without_foreach(raw.state_dict()), _without_foreach(before))


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("fused_location", ["defaults", "last_group"])
def test_fused_conflict_is_rejected_before_any_group_or_default_changes(wrapped, fused_location):
    optimizer = _populated_optimizer(wrapped)
    raw = _raw(optimizer)
    if fused_location == "defaults":
        raw.defaults["fused"] = True
    else:
        raw.param_groups[-1]["fused"] = True
    before = deepcopy(raw.state_dict()), deepcopy(raw.defaults)
    with pytest.raises(ValueError, match="fused"):
        trainer_module.apply_optimizer_foreach_override(optimizer, True)
    _assert_tree(raw.state_dict(), before[0])
    _assert_tree(raw.defaults, before[1])


def test_explicit_false_remains_compatible_with_fused_adamw():
    optimizer = _populated_optimizer()
    optimizer.defaults["fused"] = True
    optimizer.param_groups[-1]["fused"] = True
    assert trainer_module.apply_optimizer_foreach_override(optimizer, False) is True
    assert optimizer.defaults["foreach"] is False
    assert optimizer.defaults["fused"] is True
    assert all(group["foreach"] is False for group in optimizer.param_groups)


@pytest.mark.parametrize("wrapped", [False, True])
def test_non_adamw_optimizer_is_rejected_without_mutation(wrapped):
    raw = torch.optim.SGD(_FiveGroups().parameters(), lr=0.01, momentum=0.9)
    optimizer = _accelerated_optimizer(raw) if wrapped else raw
    before = deepcopy(raw.state_dict())
    with pytest.raises(TypeError, match="AdamW"):
        trainer_module.apply_optimizer_foreach_override(optimizer, True)
    _assert_tree(raw.state_dict(), before)


def test_arbitrary_optimizer_attribute_does_not_make_a_supported_wrapper():
    raw = _populated_optimizer()
    before = deepcopy(raw.state_dict())
    with pytest.raises(TypeError):
        trainer_module.apply_optimizer_foreach_override(SimpleNamespace(optimizer=raw), True)
    _assert_tree(raw.state_dict(), before)


@pytest.mark.parametrize("requested", [0, 1, "true"])
def test_override_requires_an_actual_boolean(requested):
    with pytest.raises(TypeError, match="bool"):
        trainer_module.apply_optimizer_foreach_override(_populated_optimizer(), requested)


def _trainer(tmp_path, *, foreach, data_parallel="ddp"):
    cfg = LaunchConfig(
        exp="cpu_foreach_resume", timestamp="bounded", auto_tag_run=False,
        log=LoggingConfig(), wandb=WandbConfig(),
        train=TrainConfig(name="posttrain", output_dir=str(tmp_path), max_training_steps=7,
                          train_batch_size=1, warmup_ratio=None, optimizer_foreach=foreach,
                          data_parallel=data_parallel, preserve_checkpoint_every=None),
        data=DataConfig(transform=DataTransform(repack=SonicRepackTransform(), field=FieldTransform(),
                                               model=Psi0ModelTransform())),
        model=Psi0ModelConfig(action_dim=80, odim=45, action_chunk_size=30),
    )
    trainer = PosttrainTrainer(cfg, torch.device("cpu"))
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    model = _FiveGroups()
    optimizer = _optimizer(model, foreach=foreach)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
    trainer.model, trainer.optimizer, trainer.lr_scheduler = accelerator.prepare(model, optimizer, scheduler)
    trainer.accelerator = accelerator
    return trainer


@pytest.mark.parametrize("data_parallel,requested,saved_foreach", [
    ("ddp", True, False), ("ddp", False, True), ("ddp", None, False),
    ("fsdp", True, False), ("deepspeed", True, False),
])
def test_production_resume_preserves_state_rng_and_continuation(
    tmp_path, monkeypatch, data_parallel, requested, saved_foreach,
):
    # Imports must resolve to this checkout, including in an isolated worktree.
    checkout = Path(__file__).resolve().parents[2]
    assert Path(trainer_module.__file__).resolve().is_relative_to(checkout)
    before_test_rng = _rng()
    try:
        random.seed(181)
        np.random.seed(182)
        torch.manual_seed(183)
        saved = _trainer(tmp_path / "source", foreach=saved_foreach)
        for _ in range(3):
            _one_update(saved.model, saved.optimizer, saved.lr_scheduler, saved.accelerator)
        checkpoint = Path(saved.save_checkpoint(3))
        expected_model = deepcopy(saved.model.state_dict())
        expected_optimizer = deepcopy(saved.optimizer.state_dict())
        expected_scheduler = deepcopy(saved.lr_scheduler.state_dict())
        rng_path = checkpoint / "random_states_0.pkl"
        serialized_rng = torch.load(rng_path, map_location="cpu", weights_only=False)
        expected_rng = {"python": serialized_rng["random_state"],
                        "numpy": serialized_rng["numpy_random_seed"],
                        "torch": serialized_rng["torch_manual_seed"]}
        _assert_tree(_rng(), expected_rng)
        # Accelerate1.7 CPU save omits a key its loader unconditionally reads.
        # Repair only this test-created CPU RNG fixture; no CUDA state is mocked.
        serialized_rng.setdefault("torch_cuda_manual_seed", [])
        torch.save(serialized_rng, rng_path)
        reference_steps = []
        for _ in range(4):
            loss = _one_update(saved.model, saved.optimizer, saved.lr_scheduler, saved.accelerator)
            reference_steps.append((loss, deepcopy(saved.model.state_dict()),
                                    deepcopy(saved.optimizer.state_dict()),
                                    deepcopy(saved.lr_scheduler.state_dict()), _rng()))

        restored = _trainer(tmp_path / "restored", foreach=requested, data_parallel=data_parallel)
        restored.cfg.train.resume_from_checkpoint = str(checkpoint)
        calls = []
        original = trainer_module.apply_optimizer_foreach_override

        def record_override(optimizer, value):
            calls.append(value)
            return original(optimizer, value)

        monkeypatch.setattr(trainer_module, "apply_optimizer_foreach_override", record_override)
        step, path = restored.resume_from_checkpoint()
        assert (step, path) == (3, str(checkpoint))
        should_override = data_parallel == "ddp" and requested is not None
        assert calls == ([requested] if should_override else [])
        effective = requested if should_override else saved_foreach
        assert all(group["foreach"] is effective for group in restored.optimizer.param_groups)
        assert restored.optimizer.optimizer.defaults["foreach"] is requested
        _assert_tree(restored.model.state_dict(), expected_model)
        _assert_tree(_without_foreach(restored.optimizer.state_dict()), _without_foreach(expected_optimizer))
        _assert_tree(restored.lr_scheduler.state_dict(), expected_scheduler)
        _assert_tree(_rng(), expected_rng)
        assert all(int(slots["step"]) == 3 for slots in restored.optimizer.state.values())
        for expected_loss, model_state, optimizer_state, scheduler_state, rng_state in reference_steps:
            actual_loss = _one_update(restored.model, restored.optimizer, restored.lr_scheduler, restored.accelerator)
            torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-12, atol=1e-14)
            # Foreach is algebraically identical; FP64 permits harmless operation-order rounding.
            _assert_tree(restored.model.state_dict(), model_state, exact=False)
            _assert_tree(_without_foreach(restored.optimizer.state_dict()),
                         _without_foreach(optimizer_state), exact=False)
            _assert_tree(restored.lr_scheduler.state_dict(), scheduler_state)
            _assert_tree(_rng(), rng_state)
        assert all(int(slots["step"]) == 7 for slots in restored.optimizer.state.values())
        assert not torch.cuda.is_initialized()
    finally:
        random.setstate(before_test_rng["python"])
        np.random.set_state(before_test_rng["numpy"])
        torch.set_rng_state(before_test_rng["torch"])
