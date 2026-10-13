"""Small CPU checks for Lightning sampling and persistent resume schedules."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from MotionExpert.config import smplh_motion_expert_config
from MotionExpert.dataset.motion_layout import SMPLH_REPRE_TYPE
from MotionExpert.ema import EMAOptimizer
from MotionExpert.evaluate import sampled_validation
from MotionExpert.module import uem_module
from MotionExpert.tests.fixtures import conditioning, smplh_stats
from MotionExpert.utils.task_conditioning import apply_task_conditioning


def _identity_motion(frames=11):
    motion = torch.zeros(frames, 245)
    body = motion[:, :108].reshape(frames, 12, 9)
    body[..., 0] = body[..., 4] = 1
    motion[:, 108] = motion[:, 112] = 1
    return motion


class _TinyGroupedModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block = torch.nn.Module()
        self.block.ff = torch.nn.Module()
        self.block.ff.routed_experts = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(.1))])
        self.block.ff.shared = torch.nn.Linear(1, 1, bias=False)
        self.last_y = None
        self.calls = 0

    def forward(self, x, t, y, cond_scale=None, **kwargs):
        self.last_y = y
        self.calls += 1
        return _identity_motion(x.shape[1]).to(x)[None].expand_as(x)


@pytest.fixture
def make_module(tmp_path, monkeypatch):
    monkeypatch.setattr(uem_module, "UniEgoMotion", _TinyGroupedModel)
    stats_path = tmp_path / "stats.pt"
    torch.save(smplh_stats(), stats_path)

    def make(cfg=None):
        cfg = smplh_motion_expert_config() if cfg is None else cfg
        cfg.DATA.STATS_PATH = str(stats_path)
        return uem_module.UEM_Module(cfg)

    return make


class _TinyValidationDataset:
    fps = 5
    forecast_prefix = 6
    repre_type = SMPLH_REPRE_TYPE
    normalizer = None

    def __len__(self):
        return 1

    def get_sample(self, index, task):
        y = {key: value[0] for key, value in conditioning(batch=1, frames=11).items()}
        y = apply_task_conditioning(y, task, trajectory_condition=False)
        return {"y": y, "misc": {
            "motion": _identity_motion(), "num_real_frames": 11,
            "feature_valid_mask": torch.ones(11, 245, dtype=torch.bool),
            "world_from_window": torch.eye(4), "floor_transform": torch.eye(4).repeat(11, 1, 1),
            "floor_height": 0.,
        }}


def test_lightning_sample_and_actual_sampled_validation_accept_task_ids(make_module):
    module = make_module()
    dataset = _TinyValidationDataset()
    y = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
         for key, value in dataset.get_sample(0, "fore")["y"].items()}
    assert isinstance(y["task_id"], torch.Tensor)
    prediction = module.sample(y)
    assert prediction.shape == (1, 11, 245)
    assert module.model.last_y["img_mask"][:, 6:].bool().all()
    assert module.model.last_y["traj_mask"].bool().all()
    assert not y["traj"].equal(torch.zeros_like(y["traj"]))
    result = sampled_validation(module, dataset, max_samples=1)
    assert result["fore/sample_count"] == 1
    assert result["fore/world_decode_failure"] == 0
    assert result["fore/body_local_mpjpe_m"] == pytest.approx(0.)


def test_fifth_epoch_validation_hook_calls_sampler_and_restores_model_mode(make_module):
    module = make_module()
    module._trainer = SimpleNamespace(
        current_epoch=4, sanity_checking=False, is_global_zero=True,
        datamodule=SimpleNamespace(val_dataset=_TinyValidationDataset()),
    )
    logged = {}
    module.log = lambda name, value, **kwargs: logged.update({name: value})
    module.on_validation_epoch_start()
    module.model.train()
    module.on_validation_epoch_end()
    assert module.model.calls == module.cfg.FLOW.NUM_STEPS
    assert module.model.training
    assert logged["val_sample/fore/sample_count"] == 1


@pytest.mark.parametrize("scheduler_name,epoch", [
    ("cosine_warmup", 2),  # Warmup and pre-unfreeze phase.
    ("cosine_warmup", 4),  # Shared branch ramp.
    ("cosine_warmup", 11),  # Cosine decay.
    ("step", 6),  # Next step crosses a decay boundary.
    ("step", 7),
])
@pytest.mark.parametrize("ema", [False, True])
@pytest.mark.parametrize("use_checkpoint_lr", [False, True])
def test_resume_preserves_schedule_position_and_rebases_future_steps(
        make_module, scheduler_name, epoch, ema, use_checkpoint_lr):
    old_cfg = smplh_motion_expert_config()
    old_cfg.MODEL.INIT_CKPT_PATH = "/unit-test/current-model.pt"
    old_cfg.MODEL.MOTION_EXPERT.SHARED_UNFREEZE_EPOCH = 3
    old_cfg.MODEL.MOTION_EXPERT.UNFREEZE_RAMP_EPOCHS = 4
    old_cfg.MODEL.MOTION_EXPERT.ROUTED_LR_SCALE = .25
    old_cfg.MODEL.MOTION_EXPERT.UNFROZEN_LR_SCALE = 1.5
    old_cfg.TRAIN.LR = 1e-3
    old_cfg.TRAIN.WEIGHT_DECAY = .05
    old_cfg.TRAIN.SCHEDULER = scheduler_name
    old_cfg.TRAIN.NUM_EPOCHS = 57  # StepLR step_size=7.
    old_cfg.TRAIN.WARMUP_EPOCHS = 5
    old_cfg.TRAIN.SCHEDULER_TOTAL_EPOCHS = 17
    original = make_module(old_cfg)
    optimizers, schedulers = original.configure_optimizers()
    old_optimizer, old_scheduler = optimizers[0], schedulers[0]["scheduler"]
    for _ in range(epoch):
        old_optimizer.step()
        old_scheduler.step()
    optimizer_state = deepcopy(old_optimizer.state_dict())
    scheduler_state = deepcopy(old_scheduler.state_dict())
    old_lrs = [group["lr"] for group in old_optimizer.param_groups]
    old_optimizer.step()
    old_scheduler.step()
    old_next_lrs = [group["lr"] for group in old_optimizer.param_groups]

    cfg = deepcopy(old_cfg)
    cfg.MODEL.CKPT_PATH = "/unit-test/resume.ckpt"
    cfg.TRAIN.LR = 2e-4
    cfg.TRAIN.WEIGHT_DECAY = .02
    cfg.TRAIN.USE_CKPT_LR = use_checkpoint_lr
    module = make_module(cfg)
    optimizers, schedulers = module.configure_optimizers()
    raw_optimizer, scheduler = optimizers[0], schedulers[0]["scheduler"]
    if ema:
        optimizer = EMAOptimizer(raw_optimizer, device=torch.device("cpu"))
        optimizer.load_state_dict({
            "opt": optimizer_state,
            "ema": tuple(parameter.detach().clone() for parameter in module.model.parameters()),
            "current_step": 321, "decay": .99, "every_n_steps": 1,
        })
    else:
        optimizer = raw_optimizer
        optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)
    module._trainer = SimpleNamespace(
        current_epoch=epoch, global_step=321, optimizers=[optimizer],
        lr_scheduler_configs=[SimpleNamespace(scheduler=scheduler)],
    )
    original_phase = deepcopy(module.cfg.MODEL.__dict__)
    module.on_train_start()
    ratio = 1. if use_checkpoint_lr else cfg.TRAIN.LR / old_cfg.TRAIN.LR
    expected = [lr * ratio for lr in old_lrs]
    assert [group["lr"] for group in raw_optimizer.param_groups] == pytest.approx(expected)
    assert scheduler.get_last_lr() == pytest.approx(expected)
    base = old_cfg.TRAIN.LR if use_checkpoint_lr else cfg.TRAIN.LR
    assert scheduler.base_lrs == pytest.approx([base * .25, base * 1.5])
    assert [group["initial_lr"] for group in raw_optimizer.param_groups] == pytest.approx(scheduler.base_lrs)
    decay = old_cfg.TRAIN.WEIGHT_DECAY if use_checkpoint_lr else cfg.TRAIN.WEIGHT_DECAY
    assert [group["weight_decay"] for group in raw_optimizer.param_groups] == [decay, decay]
    assert scheduler.last_epoch == epoch
    assert module.current_epoch == epoch and module.global_step == 321
    assert module.cfg.MODEL.__dict__ == original_phase
    if ema:
        assert optimizer.current_step == 321

    # Future scheduler advancement must continue at the new base, including
    # after a save/load round-trip, rather than recovering checkpoint base_lrs.
    saved_optimizer = deepcopy(optimizer.state_dict())
    saved_scheduler = deepcopy(scheduler.state_dict())
    serialized_groups = (saved_optimizer["opt"] if ema else saved_optimizer)["param_groups"]
    assert [group["lr"] for group in serialized_groups] == pytest.approx(expected)
    assert [group["initial_lr"] for group in serialized_groups] == pytest.approx(scheduler.base_lrs)
    assert saved_scheduler["base_lrs"] == pytest.approx(scheduler.base_lrs)
    assert saved_scheduler["_last_lr"] == pytest.approx(expected)
    optimizer.load_state_dict(saved_optimizer)
    scheduler.load_state_dict(saved_scheduler)
    raw_optimizer.step()
    scheduler.step()
    assert scheduler.last_epoch == epoch + 1
    assert scheduler.get_last_lr() == pytest.approx([lr * ratio for lr in old_next_lrs])
    assert [group["lr"] for group in raw_optimizer.param_groups] == pytest.approx(scheduler.get_last_lr())
