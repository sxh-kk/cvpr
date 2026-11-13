"""Velocity regression and reverse-time Euler integration share one convention."""

import pytest
import torch

from MotionExpert.mydiffusion.flow_matching import FlowMatching


class ConstantVelocity(torch.nn.Module):
    def __init__(self, velocity):
        super().__init__()
        self.velocity = torch.nn.Parameter(velocity.clone())
        self.inputs = []
        self.times = []
        self.conditions = []

    def forward(self, x, t, y):
        self.inputs.append(x.detach().clone())
        self.times.append(t.detach().clone())
        self.conditions.append(y)
        return self.velocity.expand_as(x)


def pair():
    generator = torch.Generator().manual_seed(143)
    clean = torch.randn(2, 11, 245, generator=generator)
    noise = torch.randn(clean.shape, generator=generator)
    y = {"valid_frames": torch.ones(2, 11, dtype=torch.bool)}
    return clean, noise, y


@pytest.mark.parametrize("time", [0., .2, .8, 1.])
def test_training_regresses_velocity_without_time_dependent_loss_scaling(time):
    clean, noise, y = pair()
    target_velocity = noise - clean
    # A constant residual must have the same MSE at every time, including zero.
    # This distinguishes direct velocity regression from an x0 objective or
    # a clean-prediction residual converted by division through the flow time.
    model = ConstantVelocity(target_velocity + .25)
    result = FlowMatching().training_losses(
        model, clean, {"y": y}, noise=noise, t=torch.full((2,), time),
        return_diagnostics=True,
    )
    expected_state = (1 - time) * clean + time * noise
    torch.testing.assert_close(model.inputs[0], expected_state)
    torch.testing.assert_close(result["target_velocity"], target_velocity)
    torch.testing.assert_close(result["model_output"], model.velocity)
    torch.testing.assert_close(result["pred_velocity"], model.velocity)
    torch.testing.assert_close(result["pred_xstart"], expected_state - time * model.velocity)
    torch.testing.assert_close(result["loss"], torch.full((2,), .25 ** 2))
    result["loss"].mean().backward()
    assert torch.isfinite(model.velocity.grad).all()
    assert model.velocity.grad.abs().sum() > 0


@pytest.mark.parametrize("steps", [1, 10])
def test_perfect_velocity_integrates_noise_to_clean_with_negative_time_steps(steps):
    clean, noise, y = pair()
    model = ConstantVelocity(noise - clean)
    result, estimates = FlowMatching(num_steps=steps).sample_loop(
        model, clean.shape, model_kwargs={"y": y}, noise=noise,
        return_all_pred_xstart=True,
    )
    assert len(model.inputs) == steps
    assert len(estimates) == steps
    for index, (state, time) in enumerate(zip(model.inputs, model.times)):
        expected_time = torch.full((2,), 1 - index / steps)
        torch.testing.assert_close(time, expected_time)
        t = time[:, None, None]
        torch.testing.assert_close(state, (1 - t) * clean + t * noise,
                                   rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(estimates[index], clean, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(result, clean, rtol=1e-5, atol=1e-6)
    assert not result.requires_grad


def test_generation_mask_preserves_noise_context_and_velocity_supervision_region():
    clean, noise, y = pair()
    generation = torch.zeros(2, 11, dtype=torch.bool)
    generation[:, 6:] = True
    y["motion_generation_mask"] = generation
    model = ConstantVelocity(noise - clean)
    flow = FlowMatching(num_steps=3)
    losses = flow.training_losses(
        model, clean, {"y": y}, noise=noise, t=torch.tensor([0., .4]),
        return_diagnostics=True,
    )
    torch.testing.assert_close(model.inputs[0][:, :6], noise[:, :6])
    assert not losses["target_velocity"][:, :6].any()
    torch.testing.assert_close(losses["loss"], torch.zeros(2))
    model.inputs.clear()
    model.times.clear()
    generated = flow.sample_loop(model, clean.shape, model_kwargs={"y": y}, noise=noise)
    for state in model.inputs:
        torch.testing.assert_close(state[:, :6], noise[:, :6], rtol=0, atol=0)
    torch.testing.assert_close(generated[:, :6], noise[:, :6], rtol=0, atol=0)
    torch.testing.assert_close(generated[:, 6:], clean[:, 6:], rtol=1e-5, atol=1e-6)


def test_sampling_discards_supervised_feature_validity_without_mutating_the_caller():
    clean, noise, y = pair()
    y["feature_valid_mask"] = torch.zeros_like(clean, dtype=torch.bool)
    model = ConstantVelocity(noise - clean)
    result = FlowMatching(num_steps=2).sample_loop(
        model, clean.shape, model_kwargs={"y": y}, noise=noise,
    )
    assert "feature_valid_mask" in y
    assert not y["feature_valid_mask"].any()
    assert all("feature_valid_mask" not in condition for condition in model.conditions)
    torch.testing.assert_close(result, clean, rtol=1e-5, atol=1e-6)


def test_no_supervised_frames_contribute_finite_zero_loss_and_gradients():
    clean, noise, y = pair()
    y["valid_frames"][:] = False
    model = ConstantVelocity(torch.full_like(clean, .25))
    result = FlowMatching().training_losses(
        model, clean, {"y": y}, noise=noise, t=torch.zeros(2),
    )
    torch.testing.assert_close(result["loss"], torch.zeros(2))
    result["loss"].sum().backward()
    assert torch.isfinite(model.velocity.grad).all()
    assert not model.velocity.grad.any()
