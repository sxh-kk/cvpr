"""Real tiny image/text Qwen and action-head equivalence, entirely on CPU.

The optimization narrows the discarded vocabulary projection, never the
observation tokens, pooling, state token or downstream action computation.
"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError
import torch
from torch import nn
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from psi.config.model_psi0 import Psi0ModelConfig
from psi.models.frozen_motion import MotionFeatures
from psi.models.psi0 import Psi0Model
from psi.trainers.posttrain import PosttrainTrainer


class _UnusedMotionFacade:
    """No motion producer is executed; both models use identical fixed memory."""

    def __init__(self):
        self.model = nn.Linear(1, 768)

    def forward_features(self, *args, **kwargs):
        raise AssertionError("This regression supplies already-computed motion tokens")


def _model(skip, *, tied, checkpointed):
    qwen_config = Qwen3VLConfig(
        text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=16, max_position_embeddings=64, use_cache=False,
                         attention_dropout=0.0, tie_word_embeddings=tied,
                         rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]}),
        vision_config=dict(depth=2, hidden_size=16, intermediate_size=32, num_heads=2,
                           patch_size=2, spatial_merge_size=2, temporal_patch_size=1,
                           out_hidden_size=32, num_position_embeddings=16,
                           deepstack_visual_indexes=[0]),
        image_token_id=120, video_token_id=121, vision_start_token_id=122,
        vision_end_token_id=123, tie_word_embeddings=tied,
    )
    cfg = Psi0ModelConfig(
        action_architecture="dual_source_v1", n_conditions=0,
        tune_vlm=True, tune_motion=False, vlm_skip_unused_logits=skip,
        gradient_checkpointing=checkpointed, action_dim=4, action_chunk_size=3,
        hidden_dim=16, num_blocks=2, nhead=2, view_feature_dim=32, odim=3,
        time_dim=8, dropout=0.0, state_feature_dropout=0.0,
        state_drop_prob=0.5, state_null_token=True,
    )
    model = Psi0Model(cfg, Qwen3VLForConditionalGeneration(qwen_config),
                      motion_expert=_UnusedMotionFacade()).train()
    # Open both context and pooled-summary gradient paths immediately, instead
    # of waiting for zero-initialized AdaLN gates to train in this tiny fixture.
    with torch.no_grad():
        for block in model.action_header.transformer_blocks:
            nn.init.normal_(block.mod_head[-1].weight, std=0.01)
            for branch in range(4):
                block.mod_head[-1].bias[(3 * branch + 2) * 16:(3 * branch + 3) * 16] = 0.2
        nn.init.normal_(model.action_header.action_proj_out.adaLN_modulation[-1].weight, std=0.01)
        model.vlm_model.model.language_model.norm.weight.fill_(1.7)
    trainer = SimpleNamespace(model=model, model_cfg=cfg,
                              _vlm_group_of=PosttrainTrainer._vlm_group_of)
    trainer.vlm_trainable_components = lambda: PosttrainTrainer.vlm_trainable_components(trainer)
    PosttrainTrainer.apply_vlm_trainability(trainer)
    return model


def _batch(padding):
    # Two actual image prefixes, each image has grid1x4x4 -> four merged tokens.
    rows = [[2, 122, 120, 120, 120, 120, 123, 17, 5, 9, 11],
            [3, 122, 120, 120, 120, 120, 123, 7, 19]]
    padded, masks = [], []
    for row in rows:
        pad = [0] * (12 - len(row))
        padded.append(row + pad if padding == "right" else pad + row)
        masks.append([1] * len(row) + [0] * len(pad) if padding == "right"
                     else [0] * len(pad) + [1] * len(row))
    return dict(
        input_ids=torch.tensor(padded), attention_mask=torch.tensor(masks, dtype=torch.bool),
        pixel_values=torch.randn(32, 12), image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 4]]),
        action_samples=torch.randn(2, 3, 4), states=torch.randn(2, 1, 3),
        timestep=torch.tensor([128.0, 749.0]),
        motion_features=MotionFeatures(torch.randn(2, 11, 768)), return_vlm_features=True,
    )


def _assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for first, second in zip(left, right):
            _assert_tree_equal(first, second)
    else:
        assert left == right


def test_opt_in_defaults_to_false_and_rejects_legacy_config_and_model_boundary():
    assert Psi0ModelConfig().vlm_skip_unused_logits is False
    assert Psi0ModelConfig(action_architecture="dual_source_v1", n_conditions=0).vlm_skip_unused_logits is False
    with pytest.raises(ValidationError, match="requires action_architecture=dual_source_v1"):
        Psi0ModelConfig(vlm_skip_unused_logits=True)
    with pytest.raises(ValueError, match="requires action_architecture=dual_source_v1"):
        Psi0Model(SimpleNamespace(action_architecture="psi0_joint", vlm_skip_unused_logits=True), nn.Identity())


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
@pytest.mark.parametrize("checkpointed", [False, True], ids=["plain", "checkpointed"])
@pytest.mark.parametrize("padding", ["right", "left"])
def test_real_multimodal_tokens_action_gradients_and_three_updates_are_unchanged(tied, checkpointed, padding):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(4941)
        baseline = _model(False, tied=tied, checkpointed=checkpointed)
        optimized = _model(True, tied=tied, checkpointed=checkpointed)
        optimized.load_state_dict(baseline.state_dict(), strict=True)
        _assert_tree_equal(baseline.state_dict(), optimized.state_dict())
        assert (baseline.vlm_model.lm_head.weight is baseline.vlm_model.get_input_embeddings().weight) is tied
        assert optimized.vlm_model.lm_head.weight.requires_grad is tied
        batch = _batch(padding)
        target = torch.randn(2, 3, 4)
        optimizers = [torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4,
                                        weight_decay=0.01, foreach=False)
                      for model in (baseline, optimized)]
        projections, prefixes, norm_inputs, norm_outputs = [[], []], [[], []], [[], []], [[], []]
        handles = []
        for index, model in enumerate((baseline, optimized)):
            def record_projection(_module, inputs, index=index):
                projections[index].append(tuple(inputs[0].shape))

            def record_prefix(_module, _inputs, kwargs, index=index):
                prefixes[index].append(dict(kwargs))

            def record_norm(_module, inputs, output, index=index):
                norm_inputs[index].append(inputs[0].detach().clone())
                norm_outputs[index].append(output.detach().clone())

            handles.append(model.vlm_model.lm_head.register_forward_pre_hook(record_projection))
            handles.append(model.vlm_model.register_forward_pre_hook(record_prefix, with_kwargs=True))
            handles.append(model.vlm_model.model.language_model.norm.register_forward_hook(record_norm))
        try:
            for _update in range(3):
                rng = torch.get_rng_state()
                results, losses = [], []
                expected_rng = None
                for index, (model, optimizer) in enumerate(zip((baseline, optimized), optimizers)):
                    torch.set_rng_state(rng)
                    optimizer.zero_grad(set_to_none=True)
                    result = model(**batch)
                    result.vlm_features.tokens.retain_grad()
                    result.vlm_features.pooled_tokens.retain_grad()
                    loss = (result.action - target).square().sum(1).mean()
                    loss.backward()
                    results.append(result)
                    losses.append(loss.detach())
                    if expected_rng is None:
                        expected_rng = torch.get_rng_state()
                    else:
                        assert torch.equal(torch.get_rng_state(), expected_rng)
                    assert result.vlm_features.attention_mask is batch["attention_mask"]
                    assert result.vlm_features.tokens.shape == (2, 1, 12, 32)
                    assert result.vlm_features.p_v.shape == (2, 32)
                    torch.testing.assert_close(result.vlm_features.tokens[:, 0], norm_inputs[index][-1], rtol=0, atol=0)
                    assert not torch.equal(result.vlm_features.tokens[:, 0], norm_outputs[index][-1])
                _assert_tree_equal(losses[0], losses[1])
                for attribute in ("tokens", "pooled_tokens", "p_v"):
                    _assert_tree_equal(getattr(results[0].vlm_features, attribute),
                                       getattr(results[1].vlm_features, attribute))
                _assert_tree_equal(results[0].action, results[1].action)
                for attribute in ("tokens", "pooled_tokens"):
                    gradient = getattr(results[0].vlm_features, attribute).grad
                    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
                    _assert_tree_equal(gradient, getattr(results[1].vlm_features, attribute).grad)
                first, second = dict(baseline.named_parameters()), dict(optimized.named_parameters())
                assert first.keys() == second.keys()
                for name in first:
                    assert first[name].requires_grad == second[name].requires_grad
                    if first[name].grad is None:
                        assert second[name].grad is None
                    else:
                        assert torch.isfinite(first[name].grad).all()
                        _assert_tree_equal(first[name].grad, second[name].grad)
                assert baseline.vlm_model.model.visual.patch_embed.proj.weight.grad.abs().sum() > 0
                assert baseline.vlm_model.get_input_embeddings().weight.grad.abs().sum() > 0
                assert baseline.vlm_model.model.language_model.norm.weight.grad is None
                for optimizer in optimizers:
                    optimizer.step()
                _assert_tree_equal(baseline.state_dict(), optimized.state_dict())
                _assert_tree_equal(optimizers[0].state_dict(), optimizers[1].state_dict())
            assert projections[0] == [(2, 12, 32)] * 3
            assert projections[1] == [(2, 1, 32)] * 3
            assert all("logits_to_keep" not in kwargs for kwargs in prefixes[0])
            assert all(kwargs["logits_to_keep"] == 1 for kwargs in prefixes[1])
            assert all(kwargs["output_hidden_states"] and kwargs["use_cache"] is False
                       for stream in prefixes for kwargs in stream)
        finally:
            for handle in handles:
                handle.remove()
