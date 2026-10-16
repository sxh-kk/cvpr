"""Attention regressions for the residual-magnitude amplification mechanism."""

import pytest
import torch
import torch.nn.functional as F

from MotionExpert.model.core import Attention


def test_self_attention_projects_the_same_normalized_state_for_qkv():
    torch.manual_seed(1)
    block = Attention(16, 4, 0).eval()
    state = (torch.randn(2, 12, 16) * 1000 + 500).requires_grad_()
    observed = []
    handle = block.attn.register_forward_pre_hook(lambda module, args: observed.append(args))
    output = block(state, kv_x=state)
    handle.remove()
    query, keys, values = observed[0]
    assert query is keys and keys is values
    torch.testing.assert_close(query, block.norm(state))
    assert output.abs().max() < 10
    output.square().mean().backward()
    assert torch.isfinite(state.grad).all() and state.grad.abs().sum() > 0


def test_cross_attention_normalizes_context_without_the_query_affine_transform():
    torch.manual_seed(2)
    block = Attention(16, 4, 0, cross_attention=True).eval()
    with torch.no_grad():
        block.norm.weight.fill_(3)
        block.norm.bias.fill_(7)
    query = torch.randn(2, 12, 16)
    context = torch.randn(2, 56, 16) * 1000 + 500
    observed = []
    handle = block.attn.register_forward_pre_hook(lambda module, args: observed.append(args))
    output = block(query, kv_x=context)
    handle.remove()
    actual_query, keys, values = observed[0]
    torch.testing.assert_close(actual_query, block.norm(query))
    torch.testing.assert_close(keys, F.layer_norm(context, (16,)))
    assert keys is values
    assert keys.mean().abs() < 1e-5
    assert torch.isfinite(output).all()
    assert set(block.state_dict()) == {
        "norm.weight", "norm.bias", "attn.in_proj_weight", "attn.in_proj_bias",
        "attn.out_proj.weight", "attn.out_proj.bias",
    }


@pytest.mark.parametrize("cross_attention", [False, True])
def test_normalized_attention_removes_raw_residual_or_context_scale(cross_attention):
    torch.manual_seed(3)
    block = Attention(16, 4, 0, cross_attention=cross_attention).eval()
    state = torch.randn(2, 12, 16)
    context = torch.randn(2, 56, 16) if cross_attention else state
    expected = block(state, kv_x=context)
    if cross_attention:
        actual = block(state, kv_x=context * 1000 + 100)
    else:
        changed = state * 1000 + 100
        actual = block(changed, kv_x=changed)
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=1e-5)


def test_cross_attention_padding_cannot_reintroduce_large_invalid_context():
    torch.manual_seed(4)
    block = Attention(16, 4, 0, cross_attention=True).eval()
    query = torch.randn(2, 12, 16)
    context = torch.randn(2, 56, 16)
    valid = torch.ones(2, 1, 1, 56, dtype=torch.bool)
    valid[..., 20:] = False
    expected = block(query, kv_x=context, mask=valid)
    changed = context.clone()
    changed[:, 20:] = torch.randn_like(changed[:, 20:]) * 1e8
    actual = block(query, kv_x=changed, mask=valid)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
