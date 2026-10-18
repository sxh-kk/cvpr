"""Pure-Torch dual-source math, RTC, state and checkpoint contracts."""

from dataclasses import dataclass
import math

import pytest
import torch
from torch import nn

from psi.models.dual_source_action import (
    ARCHITECTURE_ID, DualSourceActionBlock, DualSourceActionTransformer,
)
from psi.models.vlm_features import VLMFeatures, masked_mean_vlm_tokens


@dataclass
class _MotionFeatures:
    tokens: torch.Tensor

    @property
    def p_m(self):
        return self.tokens.mean(1)


def _head(**changes):
    config = dict(action_dim=3, action_pred_horizon=4, action_hidden_dim=16,
                  action_num_blocks=2, action_nheads=4, view_feature_dim=12, odim=5,
                  time_dim=8, state_null_token=True, dropout=0.0)
    return DualSourceActionTransformer(**{**config, **changes})


def _inputs(batch=2):
    views = torch.randn(batch, 1, 5, 12)
    mask = torch.ones(batch, 5, dtype=torch.bool)
    mask[0, -2:] = False
    vlm = VLMFeatures(views, masked_mean_vlm_tokens(views, mask), mask, views[:, 0])
    return dict(
        timestep=torch.full((batch,), 10.0), vlm_attn_mask=mask,
        joint_attention_kwargs=dict(
            action_hidden_embeds=torch.randn(batch, 4, 3), views=views,
            vlm_features=vlm, motion_features=_MotionFeatures(torch.randn(batch, 11, 768)),
            obs=torch.randn(batch, 2, 5), traj2ds=None,
        ),
    )


def _activate_gates(head):
    # A nonzero gate makes every attention memory observable to the assertions.
    for block in head.transformer_blocks:
        with torch.no_grad():
            bias = block.mod_head[-1].bias.view(4, 3, -1)
            bias[:, 2].fill_(0.2)


def test_zero_modulation_starts_as_identity_without_zero_output_projections():
    block = DualSourceActionBlock(16, 4, "rms_norm", 0)
    x = torch.randn(2, 5, 16, requires_grad=True)
    condition = torch.randn_like(x, requires_grad=True)
    result = block(x, condition, torch.randn(2, 7, 16), torch.randn(2, 11, 16),
                   torch.ones(2, 7, dtype=torch.bool))
    torch.testing.assert_close(result, x, rtol=0, atol=0)
    result.square().sum().backward()
    grad = block.mod_head[-1].weight.grad.view(4, 3, 16, 16)
    assert grad[:, 2].abs().sum() > 0
    assert grad[:, :2].count_nonzero() == 0
    assert condition.grad.count_nonzero() == 0
    for branch in (block.self_attn, block.vlm_attn, block.motion_attn):
        assert branch.to_out[0].weight.count_nonzero() > 0


class _AffineBranch(nn.Module):
    def __init__(self, factor, offset, name, calls):
        super().__init__()
        self.factor, self.offset, self.name, self.calls = factor, offset, name, calls

    def forward(self, value, *args):
        self.calls.append((self.name, value.detach().clone(), args))
        return self.factor * value + self.offset


def test_four_residuals_are_serial_and_have_independent_gamma_beta_gate():
    width = 4
    block = DualSourceActionBlock(width, 2, None, 0)
    block.norms = nn.ModuleList([nn.Identity() for _ in range(4)])
    calls = []
    branch_values = [(0.2, 0.1), (-0.3, 0.2), (0.5, -0.4), (0.7, 0.6)]
    for name, (factor, offset) in zip(("self_attn", "vlm_attn", "motion_attn", "ff_act"), branch_values):
        setattr(block, name, _AffineBranch(factor, offset, name, calls))
    controls = [(0.1, 0.2, 0.3), (-0.2, 0.3, 0.4), (0.4, -0.1, 0.5), (0.2, 0.5, -0.3)]
    with torch.no_grad():
        block.mod_head[-1].bias.copy_(torch.tensor(controls).repeat_interleave(width, dim=1).flatten())
    x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) / 10
    v, m = torch.randn(1, 5, width), torch.randn(1, 11, width)
    mask = torch.ones(1, 5, dtype=torch.bool)
    actual = block(x, torch.randn_like(x), v, m, mask)
    expected = x
    for (factor, offset), (gamma, beta, gate), (_, branch_input, _) in zip(branch_values, controls, calls):
        adapted = (1 + gamma) * expected + beta
        torch.testing.assert_close(branch_input, adapted)
        expected = expected + gate * (factor * adapted + offset)
    torch.testing.assert_close(actual, expected)
    assert [name for name, _, _ in calls] == ["self_attn", "vlm_attn", "motion_attn", "ff_act"]
    assert calls[1][2][0] is v and calls[1][2][1] is mask
    assert calls[2][2][0] is m


def test_masks_discard_nan_padding_and_memories_stay_fixed_between_blocks():
    head = _head().eval()
    _activate_gates(head)
    inputs = _inputs()
    context = inputs["joint_attention_kwargs"]
    views, mask = context["views"], inputs["vlm_attn_mask"]
    expected = head(**inputs).action
    views[0, 0, -2:] = float("nan")
    actual = head(**inputs, return_dict=False).action
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    seen = []
    handles = [block.register_forward_pre_hook(lambda _, args: seen.append(args[2:]))
               for block in head.transformer_blocks]
    head(**inputs)
    for handle in handles:
        handle.remove()
    assert seen[0][0] is seen[1][0] and seen[0][1] is seen[1][1]
    assert torch.isnan(views[0, 0, -2:]).all()
    assert context["action_hidden_embeds"].shape == (2, 4, 3)  # input dict not popped
    assert torch.equal(mask, context["vlm_features"].attention_mask)


def test_rtc_uses_per_token_time_and_clean_state_time_zero():
    head = _head().eval()
    inputs = _inputs()
    inputs["timestep"] = torch.tensor([[0., 0., 4., 9.], [1., 2., 3., 4.]])
    seen = []
    hook = head.time_ins_embed.register_forward_pre_hook(lambda _, args: seen.append(args[0].clone()))
    result = head(**inputs).action
    hook.remove()
    assert result.shape == (2, 4, 3)
    torch.testing.assert_close(seen[0], torch.cat((torch.zeros(2, 1), inputs["timestep"]), 1))
    scalar = _inputs()
    scalar["timestep"] = torch.tensor([8., 9.])
    per_token = {**scalar, "timestep": scalar["timestep"][:, None].expand(-1, 4)}
    torch.testing.assert_close(head(**scalar).action, head(**per_token).action, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
def test_explicit_model_cast_preserves_time_phase_precision_and_distinguishes_999_1000(dtype):
    head = _head().to(dtype=dtype).eval()
    embedding = head.time_ins_embed
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert embedding.frequencies.dtype == expected_dtype
    reference = torch.exp(torch.arange(4, dtype=expected_dtype) * (-math.log(10000) / 3))
    torch.testing.assert_close(embedding.frequencies, reference, rtol=0, atol=0)
    encodings = []
    handle = embedding.out_net[0].register_forward_pre_hook(lambda _, args: encodings.append(args[0].clone()))
    result = embedding(torch.tensor([[0., 999., 1000.]]))
    handle.remove()
    assert torch.isfinite(result).all()
    assert not torch.equal(encodings[0][:, 1], encodings[0][:, 2])
    assert not torch.equal(result[:, 1], result[:, 2])
    zero_reference = torch.tensor([1., 1., 1., 1., 0., 0., 0., 0.], dtype=dtype)
    torch.testing.assert_close(encodings[0][0, 0], zero_reference, rtol=0, atol=0)
    assert "time_ins_embed.frequencies" in head.state_dict()
    # Repeated dtype changes must reconstruct frequencies, never reuse a
    # BF16-rounded intermediate buffer.
    head.to(dtype=torch.float32)
    torch.testing.assert_close(head.time_ins_embed.frequencies, _head().time_ins_embed.frequencies, rtol=0, atol=0)


def test_state_drop_and_missing_state_use_explicit_null_without_unused_parameters():
    head = _head(state_drop_prob=1.0).train()
    _activate_gates(head)
    inputs = _inputs()
    dropped = head(**inputs).action
    assert head.last_state_drop_frac == 1.0
    missing = {**inputs, "joint_attention_kwargs": {**inputs["joint_attention_kwargs"], "obs": None}}
    torch.testing.assert_close(dropped, head(**missing).action)
    head(**missing).action.square().mean().backward()
    assert all(parameter.grad is not None for parameter in head.parameters())
    no_null = _head(state_null_token=False).eval()
    with pytest.raises(ValueError, match="obs=None"):
        no_null(**missing)
    with pytest.raises(ValueError, match="requires state_null_token"):
        _head(state_drop_prob=0.1, state_null_token=False)


def test_cpu_autocast_and_gradients_reach_both_memories_and_fusion_after_gate_initialization():
    head = _head().train()
    _activate_gates(head)
    with torch.no_grad():
        for block in head.transformer_blocks:
            block.mod_head[-1].weight.normal_(std=0.01)
    inputs = _inputs()
    context = inputs["joint_attention_kwargs"]
    context["views"].requires_grad_()
    context["vlm_features"] = VLMFeatures(
        context["views"], masked_mean_vlm_tokens(context["views"], inputs["vlm_attn_mask"]),
        inputs["vlm_attn_mask"], context["views"][:, 0],
    )
    context["motion_features"].tokens.requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = head(**inputs).action
        loss = prediction.float().square().mean()
    loss.backward()
    assert torch.isfinite(prediction).all()
    for value in (context["views"].grad, context["motion_features"].tokens.grad,
                  head.f_fuse[0].weight.grad, head.action_proj_out.linear.weight.grad):
        assert torch.isfinite(value).all() and value.abs().sum() > 0
    torch.testing.assert_close(context["views"].grad[0, 0, -2:], torch.zeros(2, 12))


def test_tensor_only_checkpoint_roundtrip_and_nested_strict_false_reject_legacy_or_partial():
    original, restored = _head(), _head()
    state = original.state_dict()
    assert original.architecture_id == ARCHITECTURE_ID
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    restored.load_state_dict(state)
    original.eval(); restored.eval()
    inputs = _inputs()
    torch.testing.assert_close(original(**inputs).action, restored(**inputs).action, rtol=0, atol=0)
    parent = nn.Module()
    parent.add_module("action_header", _head())
    with pytest.raises(RuntimeError, match="legacy joint-attention"):
        parent.load_state_dict({"action_header.attn.to_q.weight": torch.randn(16, 16)}, strict=False)
    partial = dict(state)
    partial.pop("state_proj.weight")
    with pytest.raises(RuntimeError, match="Incomplete dual-source"):
        _head().load_state_dict(partial, strict=False)
    with pytest.raises(RuntimeError, match="matching configuration"):
        _head(dropout=0.1).load_state_dict(state, strict=False)


@pytest.mark.parametrize("kind", ["layers", "motion_count", "pool_shape", "state_shape", "time_shape", "mask"])
def test_shape_and_mask_mismatches_are_rejected(kind):
    head = _head()
    inputs = _inputs()
    context = inputs["joint_attention_kwargs"]
    if kind == "layers":
        context["views"] = context["views"].expand(-1, 2, -1, -1)
    elif kind == "motion_count":
        context["motion_features"].tokens = torch.randn(2, 6, 768)
    elif kind == "pool_shape":
        views = context["views"]
        context["vlm_features"] = VLMFeatures(views, torch.zeros(2, 1, 13), inputs["vlm_attn_mask"], views[:, 0])
    elif kind == "state_shape":
        context["obs"] = torch.zeros(2, 5)
    elif kind == "time_shape":
        inputs["timestep"] = torch.zeros(2, 5)
    else:
        inputs["vlm_attn_mask"] = torch.ones(2, 5, dtype=torch.bool)
    with pytest.raises(ValueError):
        head(**inputs)
