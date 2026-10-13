"""Six historical images condition joint motion targets, never clean motion inputs."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from MotionExpert.config import motion_expert_config, smplh_motion_expert_config
from MotionExpert.model import uniegomotion as model_module
from MotionExpert.module.uem_module import UEM_Module
from MotionExpert.mydiffusion.flow_matching import FlowMatching
from MotionExpert.utils.task_conditioning import apply_task_conditioning
from MotionExpert.tests.test_dense_model import InspectDecoder


class ContextPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.3))
        self.inputs, self.conditions = [], []

    def forward(self, x, t, y, **kwargs):
        self.inputs.append(x.detach().clone())
        self.conditions.append(y)
        return (x.mean(dim=1, keepdim=True) * self.scale).expand_as(x)

    def get_moe_auxiliary_loss(self):
        return None


def conditions(batch=2):
    return {
        "valid_frames": torch.ones(batch, 11, dtype=torch.bool),
        "img_embs": torch.randn(batch, 11, 5, 1024),
        "valid_img_embs": torch.ones(batch, 11, dtype=torch.bool),
        "traj": torch.randn(batch, 11, 18),
        "trajectory_valid": torch.ones(batch, 11, 18, dtype=torch.bool),
        "feature_valid_mask": torch.ones(batch, 11, 245, dtype=torch.bool),
    }


def module(supervision="future_only"):
    result = UEM_Module.__new__(UEM_Module)
    nn.Module.__init__(result)
    result.cfg = motion_expert_config()
    result.cfg.DATA.FPS = 5
    result.cfg.DATA.WINDOW = 11
    result.cfg.DATA.FORECAST_PREFIX = 6
    result.cfg.DATA.TRAIN_TASK = "fore"
    result.cfg.DATA.TRAJECTORY_CONDITION_PROB = 0.0
    result.cfg.DATA.MOTION_SUPERVISION = supervision
    result.cfg.MODEL.GLOBAL_TOKEN_COUNT = 0
    result.model = ContextPrediction()
    result.flow = FlowMatching()
    result.log = lambda *args, **kwargs: None
    result._validation_source_metrics = {}
    return result


def test_future_only_experiment_keeps_all_six_images_and_hides_all_trajectory():
    original = conditions()
    result = apply_task_conditioning(original, "fore", forecast_prefix=6,
                                     trajectory_condition=False, noise_only_history=True)
    assert result["img_mask"].tolist() == [[0] * 6 + [1] * 5] * 2
    assert bool(result["traj_mask"].all())
    assert result["loss_mask"].tolist() == [[0] * 6 + [1] * 5] * 2
    assert result["motion_generation_mask"].tolist() == [[False] * 6 + [True] * 5] * 2
    assert "img_mask" not in original and "motion_generation_mask" not in original


def test_explicit_new_training_does_not_draw_original_image_dropout():
    with patch.object(model_module, "MoEDecoderBlock", InspectDecoder):
        model = model_module.UniEgoMotion(motion_expert_config(), dropout=0).train()
    y = apply_task_conditioning(conditions(batch=1), "fore", forecast_prefix=6,
                               trajectory_condition=False, supervise_history=True)
    with patch("torch.rand", side_effect=AssertionError("No whole/single-frame dropout")):
        model(torch.zeros(1, 11, 245), torch.ones(1), y)
    mask = model.tsfm[-1].call["context_mask"][0, 0, 0]
    assert bool(mask[:31].all()) and not bool(mask[31:].any())
    assert model.tsfm[-1].call["router_conditions"]["ego_valid_frames"].tolist() == [[True] * 6 + [False] * 5]


def test_history_truth_and_history_label_validity_cannot_change_training():
    torch.manual_seed(43)
    model, flow = ContextPrediction(), FlowMatching()
    original = conditions(batch=1)
    y = apply_task_conditioning(original, "fore", forecast_prefix=6,
                               trajectory_condition=False, noise_only_history=True)
    target = torch.randn(1, 11, 245, requires_grad=True)
    noise, t = torch.randn_like(target), torch.tensor([.4])
    expected = flow.training_losses(model, target, {"y": y}, noise=noise, t=t,
                                    return_diagnostics=True)
    changed_target = target.detach().clone()
    changed_target[:, :6] = 999
    changed_y = deepcopy(y)
    changed_y["feature_valid_mask"][:, :6] = False
    actual = flow.training_losses(model, changed_target, {"y": changed_y}, noise=noise, t=t,
                                  return_diagnostics=True)
    for name in ("x_t", "model_output", "loss", "local_mse", "global_mse"):
        torch.testing.assert_close(actual[name], expected[name])
    torch.testing.assert_close(expected["x_t"][:, :6], noise[:, :6])
    torch.testing.assert_close(expected["x_t"][:, 6:], .6 * target[:, 6:] + .4 * noise[:, 6:])
    expected["loss"].sum().backward()
    assert not bool(target.grad[:, :6].any())
    assert bool(target.grad[:, 6:].abs().sum() > 0)
    assert expected["body_position_valid_count"].item() == 5 * 12 * 3


def test_sampling_keeps_the_same_history_gaussian_at_every_euler_step():
    model, flow = ContextPrediction(), FlowMatching(num_steps=4)
    y = apply_task_conditioning(conditions(batch=1), "fore", forecast_prefix=6,
                               trajectory_condition=False, noise_only_history=True)
    noise = torch.randn(1, 11, 245)
    outputs = list(flow.sample_loop_progressive(model, noise.shape, {"y": y}, noise=noise))
    assert len(model.inputs) == 4
    for state in model.inputs + [step["sample"] for step in outputs]:
        torch.testing.assert_close(state[:, :6], noise[:, :6])
    assert not torch.equal(outputs[-1]["sample"][:, 6:], noise[:, 6:])


def test_generation_domain_is_not_derived_from_missing_future_labels():
    y = conditions(batch=1)
    y["feature_valid_mask"][:, 7:] = False
    result = apply_task_conditioning(y, "fore", forecast_prefix=6,
                                    trajectory_condition=False, noise_only_history=True)
    assert bool(result["motion_generation_mask"][:, 6:].all())
    result["feature_valid_mask"][:] = False
    flow, model = FlowMatching(num_steps=2), ContextPrediction()
    noise = torch.ones(1, 11, 245)
    output = flow.sample_loop(model, noise.shape, {"y": result}, noise=noise)
    torch.testing.assert_close(output[:, :6], noise[:, :6])
    assert not torch.equal(output[:, 6:], noise[:, 6:])


def test_future_only_train_validation_and_sample_apply_the_same_deployment_masks():
    subject, y = module(), conditions()
    batch = {"y": y, "misc": {"motion": torch.randn(2, 11, 245)}}
    with patch("torch.rand", side_effect=AssertionError("Default image-only task needs no modality draw")):
        subject.training_step(batch, 0)
        subject.validation_step(batch, 0)
        subject.sample(y)
    for received in subject.model.conditions:
        assert bool(received["traj_mask"].all())
        assert received["img_mask"].tolist() == [[0] * 6 + [1] * 5] * 2
        assert received["motion_generation_mask"].tolist() == [[False] * 6 + [True] * 5] * 2


def test_privileged_training_option_never_turns_on_trajectory_in_validation():
    subject = module()
    subject.cfg.DATA.TRAJECTORY_CONDITION_PROB = 1.0
    y = conditions()
    assert not bool(subject._task_conditioning(y, "train")["traj_mask"][:, :6].any())
    assert bool(subject._task_conditioning(y, "val")["traj_mask"].all())


def test_mixed_smplh_training_opens_only_selected_history_and_keeps_joint_targets():
    subject = module(supervision="joint")
    subject.cfg = smplh_motion_expert_config()
    # Start from the same explicit forecast masks produced by the Dataset.
    y = apply_task_conditioning(conditions(), "fore", forecast_prefix=6,
                               supervise_history=True)
    batch = {"y": y, "misc": {"motion": torch.randn(2, 11, 245)}}
    with patch("torch.rand", return_value=torch.tensor([.25, .75])) as draw:
        subject.training_step(batch, 0)
    draw.assert_called_once()
    train = subject.model.conditions[-1]
    assert train["traj_mask"].tolist() == [[0] * 6 + [1] * 5, [1] * 11]
    assert train["img_mask"].tolist() == [[0] * 6 + [1] * 5] * 2
    assert bool(train["loss_mask"].all())
    assert "motion_generation_mask" not in train
    assert subject.model.inputs[-1].shape == (2, 11, 245)
    with patch("torch.rand", side_effect=AssertionError("Validation cannot draw a trajectory condition")):
        validation = subject._task_conditioning(y, "val")
    assert bool(validation["traj_mask"].all())
    assert validation["img_mask"].tolist() == [[0] * 6 + [1] * 5] * 2
    assert bool(validation["loss_mask"].all())


def test_mixed_training_probability_does_not_redefine_motion_statistics():
    from MotionExpert.tests.test_smplh_contract import smplh_stats
    from MotionExpert.dataset.normalization import MotionNormalizer
    from MotionExpert.checkpoints import validate_checkpoint_statistics
    cfg = smplh_motion_expert_config()
    assert cfg.DATA.TRAJECTORY_CONDITION_PROB == .5
    stats = smplh_stats(cfg)
    normalizer = MotionNormalizer(stats)
    validate_checkpoint_statistics(cfg, {}, normalizer)
    cfg.DATA.TRAJECTORY_CONDITION_PROB = 0.
    validate_checkpoint_statistics(cfg, {}, normalizer)


def test_legacy_task_masks_and_reconstruction_generation_remain_compatible():
    y = conditions(batch=1)
    legacy = apply_task_conditioning(y, "fore", forecast_prefix=6)
    assert "motion_generation_mask" not in legacy
    assert not bool(legacy["traj_mask"][:, :6].any())
    for task in ("recon", "gen"):
        current = apply_task_conditioning(y, task, forecast_prefix=6,
                                         trajectory_condition=False, noise_only_history=True)
        assert bool(current["motion_generation_mask"].all())
    subject = module()
    subject.cfg.DATA.TRAIN_TASK = None
    assert subject._task_conditioning(y, "train") == y


@pytest.mark.parametrize("invalid", [0, 1, torch.ones(2), torch.ones(2, 11, dtype=torch.bool)])
def test_trajectory_condition_rejects_ambiguous_values(invalid):
    with pytest.raises(ValueError, match="trajectory_condition"):
        apply_task_conditioning(conditions(), "fore", forecast_prefix=6, trajectory_condition=invalid)


def test_flow_rejects_stale_generation_mask_shapes_and_non_boolean_values():
    flow, model = FlowMatching(), ContextPrediction()
    for invalid in (torch.ones(1, 11), torch.ones(1, 10, dtype=torch.bool)):
        with pytest.raises(ValueError, match="motion_generation_mask"):
            flow.training_losses(model, torch.zeros(1, 11, 245),
                {"y": {"valid_frames": torch.ones(1, 11), "motion_generation_mask": invalid}})


def test_joint_task_supervises_both_history_and_future_without_freezing_latents():
    original = conditions()
    result = apply_task_conditioning(original, "fore", forecast_prefix=6,
                                     trajectory_condition=False, supervise_history=True)
    assert bool(result["loss_mask"].all())
    assert bool(result["traj_mask"].all())
    assert result["img_mask"].tolist() == [[0] * 6 + [1] * 5] * 2
    assert "motion_generation_mask" not in result


def test_joint_history_and_future_have_gradients_and_use_standard_fm_noising():
    model, flow = ContextPrediction(), FlowMatching()
    y = apply_task_conditioning(conditions(batch=1), "fore", forecast_prefix=6,
                               trajectory_condition=False, supervise_history=True)
    target = torch.randn(1, 11, 245, requires_grad=True)
    noise, t = torch.randn_like(target), torch.tensor([.4])
    terms = flow.training_losses(model, target, {"y": y}, noise=noise, t=t,
                                 return_diagnostics=True)
    torch.testing.assert_close(terms["x_t"], .6 * target + .4 * noise)
    terms["loss"].sum().backward()
    assert bool(target.grad[:, :6].abs().sum() > 0)
    assert bool(target.grad[:, 6:].abs().sum() > 0)
    assert terms["body_position_valid_count"].item() == 11 * 12 * 3
    assert "motion" not in y and "history_motion" not in y


def test_joint_sampling_updates_history_and_future_throughout_denoising():
    model, flow = ContextPrediction(), FlowMatching(num_steps=4)
    y = apply_task_conditioning(conditions(batch=1), "fore", forecast_prefix=6,
                               trajectory_condition=False, supervise_history=True)
    noise = torch.randn(1, 11, 245)
    outputs = list(flow.sample_loop_progressive(model, noise.shape, {"y": y}, noise=noise))
    assert len(model.inputs) == 4
    assert not torch.equal(model.inputs[1][:, :6], noise[:, :6])
    assert not torch.equal(model.inputs[1][:, 6:], noise[:, 6:])
    assert not torch.equal(outputs[-1]["sample"][:, :6], noise[:, :6])
    assert not torch.equal(outputs[-1]["sample"][:, 6:], noise[:, 6:])


def test_joint_module_uses_same_image_only_conditions_for_train_validation_sampling():
    subject, y = module("joint"), conditions()
    batch = {"y": y, "misc": {"motion": torch.randn(2, 11, 245)}}
    with patch("torch.rand", side_effect=AssertionError("No random condition dropout")):
        subject.training_step(batch, 0)
        subject.validation_step(batch, 0)
        subject.sample(y)
    for index, received in enumerate(subject.model.conditions):
        assert bool(received["traj_mask"].all())
        assert received["img_mask"].tolist() == [[0] * 6 + [1] * 5] * 2
        assert "motion_generation_mask" not in received
        if index < 2:  # Training and validation losses include historical targets.
            assert bool(received["loss_mask"].all())
    # Sampling evaluation still marks only the future task region for metrics.
    assert subject.model.conditions[-1]["loss_mask"].tolist() == [[0] * 6 + [1] * 5] * 2


def test_joint_task_clears_stale_future_only_generation_mask():
    previous = apply_task_conditioning(conditions(), "fore", forecast_prefix=6,
                                      trajectory_condition=False, noise_only_history=True)
    joint = apply_task_conditioning(previous, "fore", forecast_prefix=6,
                                   trajectory_condition=False, supervise_history=True)
    assert "motion_generation_mask" not in joint
    assert bool(joint["loss_mask"].all())
    assert "motion_generation_mask" in previous


def test_joint_history_cannot_be_both_supervised_and_fixed_noise():
    with pytest.raises(ValueError, match="Historical supervision"):
        apply_task_conditioning(conditions(), "fore", forecast_prefix=6,
                               supervise_history=True, noise_only_history=True)
    with pytest.raises(ValueError, match="supervise_history"):
        apply_task_conditioning(conditions(), "fore", forecast_prefix=6, supervise_history=1)
