"""Loss masks must not replace unknown generated motion with zero inputs."""

import torch

from MotionExpert.mydiffusion.flow_matching import FlowMatching


class CaptureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.full((245,), 0.25))
        self.input = None

    def forward(self, x, t, y):
        self.input = x.detach().clone()
        return x * self.gain


def inputs():
    generator = torch.Generator().manual_seed(62)
    target = torch.randn(2, 11, 245, generator=generator)
    noise = torch.randn(target.shape, generator=generator)
    label_mask = torch.ones_like(target, dtype=torch.bool)
    label_mask[:, 6:, 117:180] = False
    y = {"valid_frames": torch.ones(2, 11, dtype=torch.bool),
         "loss_mask": torch.ones(2, 11, dtype=torch.bool),
         "feature_valid_mask": label_mask}
    return target, noise, y


def test_t1_model_input_is_full_noise_independent_of_label_availability():
    target, noise, y = inputs()
    target[~y["feature_valid_mask"]] = float("nan")
    model = CaptureModel()
    flow = FlowMatching()
    flow.training_losses(model, target, {"y": y}, noise=noise, t=torch.ones(2))
    torch.testing.assert_close(model.input, noise, rtol=0, atol=0)
    alternative = dict(y, feature_valid_mask=torch.ones_like(target, dtype=torch.bool))
    flow.training_losses(model, torch.nan_to_num(target), {"y": alternative},
                         noise=noise, t=torch.ones(2))
    torch.testing.assert_close(model.input, noise, rtol=0, atol=0)


def test_unknown_channels_keep_noise_at_intermediate_times():
    target, noise, y = inputs()
    t = torch.tensor([0.2, 0.7])
    model = CaptureModel()
    diagnostics = FlowMatching().training_losses(
        model, target, {"y": y}, noise=noise, t=t, return_diagnostics=True)
    valid = y["feature_valid_mask"]
    interpolated = (1 - t[:, None, None]) * target + t[:, None, None] * noise
    torch.testing.assert_close(model.input[valid], interpolated[valid], rtol=0, atol=0)
    torch.testing.assert_close(model.input[~valid], noise[~valid], rtol=0, atol=0)
    torch.testing.assert_close(diagnostics["target_velocity"][valid],
                               (noise - target)[valid], rtol=0, atol=0)
    assert not diagnostics["target_velocity"][~valid].any()


def test_masked_nan_targets_preserve_finite_loss_and_gradients():
    target, noise, y = inputs()
    # A whole unsupervised output channel makes zero output-gradient testable
    # without reusing the same gain on supervised historical frames.
    y["feature_valid_mask"][:, :, 180:183] = False
    valid = y["feature_valid_mask"]
    bad_target = target.masked_fill(~valid, float("nan")).requires_grad_()
    model, reference = CaptureModel(), CaptureModel()
    flow = FlowMatching()
    t = torch.tensor([0.25, 0.8])
    loss = flow.training_losses(model, bad_target, {"y": y}, noise=noise, t=t)["loss"].mean()
    clean_loss = flow.training_losses(reference, target, {"y": y}, noise=noise, t=t)["loss"].mean()
    torch.testing.assert_close(loss, clean_loss, rtol=0, atol=0)
    loss.backward()
    clean_loss.backward()
    assert torch.isfinite(model.input).all()
    assert torch.isfinite(model.gain.grad).all()
    assert torch.isfinite(bad_target.grad).all()
    assert not bad_target.grad[~valid].any()
    assert not model.gain.grad[180:183].any()
    torch.testing.assert_close(model.gain.grad, reference.gain.grad, rtol=0, atol=0)


def test_fully_valid_channels_keep_original_flow_and_weighted_objective():
    target, noise, y = inputs()
    y["feature_valid_mask"][:] = True
    t = torch.tensor([0.3, 0.65])
    model = CaptureModel()
    result = FlowMatching().training_losses(model, target, {"y": y}, noise=noise,
                                            t=t, return_diagnostics=True)
    interpolated = (1 - t[:, None, None]) * target + t[:, None, None] * noise
    torch.testing.assert_close(model.input, interpolated, rtol=0, atol=0)
    torch.testing.assert_close(result["target_velocity"], noise - target, rtol=0, atol=0)
    weights = torch.ones(245)
    weights[108:117] = 8
    expected = ((interpolated * model.gain - target).square() * weights).sum((1, 2)) / (11 * weights.sum())
    torch.testing.assert_close(result["loss"], expected)
