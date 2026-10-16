"""Differentiable one-call features using real condition assembly and tiny decoders."""

from unittest.mock import patch

import pytest
import torch

from MotionExpert.model import uniegomotion as model_module
from MotionExpert.tests.fixtures import conditioning, tiny_expert
from MotionExpert.tests.fixtures import InspectDecoder


@pytest.fixture
def expert():
    torch.set_num_threads(1)
    result = tiny_expert()
    # Exercise real image/time projections and motion pooling without MoEs.
    with patch.object(model_module, "MoEDecoderBlock", InspectDecoder):
        result.model = model_module.UniEgoMotion(result.cfg, dropout=0).eval()
    return result


def test_forward_features_updates_motion_parameters_and_observed_images(expert):
    images = conditioning(batch=1, frames=6)["img_embs"].requires_grad_()
    expert.train()
    optimizer = torch.optim.AdamW(expert.parameters(), lr=1e-3)
    previous = expert.model.embed_clip_cond.weight.detach().clone()
    with patch.object(expert.model, "forward", wraps=expert.model.forward) as forward:
        output = expert.forward_features({"img_embs": images}, seed=62)
    assert forward.call_count == 1
    assert expert.training
    assert output["one_step_hidden"].shape == (1, 11, 768)
    assert output["one_step_global"].shape == (1, 768)
    torch.testing.assert_close(output["one_step_global"], output["one_step_hidden"].mean(1),
                               rtol=0, atol=0)
    assert not any("global_token" in name for name, _ in expert.model.named_parameters())
    noise, time, y = forward.call_args.args
    assert noise.shape == (1, 11, 245)
    torch.testing.assert_close(time, torch.ones(1))
    assert y["img_mask"][:, 6:].bool().all()
    assert not y["valid_img_embs"][:, 6:].any()
    assert not y["img_embs"][:, 6:].any()
    assert y["traj_mask"].bool().all()
    assert not y["trajectory_valid"].any()
    loss = output["one_step_hidden"].square().mean() + output["one_step_global"].square().mean()
    loss.backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()
    assert images.grad.abs().sum() > 0
    assert expert.model.embed_clip_cond.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(expert.model.embed_clip_cond.weight, previous)


def test_future_images_and_ground_truth_are_hidden_and_have_zero_gradient(expert):
    observed = conditioning(batch=1, frames=6)["img_embs"]
    expected = expert.forward_features({"img_embs": observed}, seed=62)
    full = conditioning(batch=1, frames=11)
    full["img_embs"][:, :6] = observed
    full["img_embs"][:, 6:] = float("nan")
    full["img_embs"].requires_grad_()
    full["traj"][:] = float("nan")
    full["traj"].requires_grad_()
    full["feature_valid_mask"] = torch.zeros(1, 11, 245, dtype=torch.bool)
    actual = expert.forward_features(full, seed=62)
    for key in ("one_step_hidden", "one_step_global"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    actual["one_step_global"].square().mean().backward()
    assert torch.isfinite(full["img_embs"].grad).all()
    assert full["img_embs"].grad[:, :6].abs().sum() > 0
    assert not full["img_embs"].grad[:, 6:].any()
    assert full["traj"].grad is None or not full["traj"].grad.any()
    # Future image availability and GT label validity are not feature conditions.
    full["valid_img_embs"][:, 6:] = False
    full["valid_frames"][:, 6:] = False
    again = expert.forward_features(full, seed=62)
    torch.testing.assert_close(again["one_step_global"], expected["one_step_global"], rtol=0, atol=0)


def test_frozen_dino_features_still_update_motion_network(expert):
    with torch.no_grad():
        images = conditioning(batch=1, frames=6)["img_embs"] * 2
    assert not images.requires_grad
    expert.train()
    output = expert.forward_features({"img_embs": images}, seed=62)
    output["one_step_global"].square().mean().backward()
    assert expert.model.embed_clip_cond.weight.grad is not None
    assert expert.model.embed_clip_cond.weight.grad.abs().sum() > 0


def test_frozen_interface_matches_one_step_without_retaining_gradients(expert):
    images = conditioning(batch=1, frames=6)["img_embs"].requires_grad_()
    expert.eval()
    actual = expert.forward_features({"img_embs": images}, seed=62)
    frozen = expert.extract_features({"img_embs": images}, task="fore", output_frames=11, seed=62)
    for key in ("one_step_hidden", "one_step_global"):
        torch.testing.assert_close(frozen[key], actual[key], rtol=0, atol=0)
        assert actual[key].requires_grad
        assert not frozen[key].requires_grad
        assert not frozen[key].is_inference()
    # Frozen tensors remain usable by a trainable downstream head.
    head = torch.nn.Linear(768, 1)
    head(frozen["one_step_global"]).sum().backward()
    assert head.weight.grad is not None
    assert all(parameter.grad is None for parameter in expert.parameters())


def test_mode_parameter_and_state_delegation_preserves_model_keys(expert):
    assert expert.train() is expert and expert.training and expert.model.training
    assert expert.model.pos_enc.dropout.training
    assert expert.eval() is expert and not expert.training
    assert not expert.model.pos_enc.dropout.training
    assert list(expert.parameters()) == list(expert.model.parameters())
    assert dict(expert.named_parameters()).keys() == dict(expert.model.named_parameters()).keys()
    assert expert.state_dict().keys() == expert.model.state_dict().keys()
    assert not expert.load_state_dict(expert.state_dict()).missing_keys
    assert expert.requires_grad_(False) is expert
    assert not any(parameter.requires_grad for parameter in expert.parameters())
    expert.requires_grad_(True)
    assert all(parameter.requires_grad for parameter in expert.parameters())


def test_default_forward_checks_fixed_history_length(expert):
    images = conditioning(batch=1, frames=5)["img_embs"]
    with pytest.raises(ValueError, match="6 observed image frames"):
        expert.forward_features({"img_embs": images})
    with pytest.raises(KeyError, match="img_embs"):
        expert.forward_features({})


@pytest.mark.parametrize("frames", [6, 11])
def test_future_validity_cannot_replace_missing_historical_observations(expert, frames):
    supplied = conditioning(batch=2, frames=frames)
    supplied["valid_frames"][0, :6] = False
    # The second sample and all future frames are valid; neither may rescue
    # the first sample's missing historical prefix.
    with pytest.raises(ValueError, match="valid historical observation"):
        expert.forward_features(supplied, seed=62)
