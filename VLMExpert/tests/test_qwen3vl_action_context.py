"""Real tiny Qwen3-VL: preserve Psi0's decoder tokens, freeze only unused norm.

No checkpoints, tokenizer downloads, images or GPU are needed. The producer is
the installed Qwen3VLForConditionalGeneration, not a model double that returns
a different hidden-state convention.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from psi.models.psi0 import Psi0Model
from psi.trainers.finetune import FinetuneTrainer
from psi.trainers.posttrain import PosttrainTrainer
from psi.trainers.sonic import SonicTrainer


def _tiny_qwen():
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=16, max_position_embeddings=64,
                         rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]},
                         attention_dropout=0.0, use_cache=False),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32,
                           num_heads=2, patch_size=2, spatial_merge_size=2,
                           temporal_patch_size=1, out_hidden_size=32,
                           num_position_embeddings=4, deepstack_visual_indexes=[0]),
        image_token_id=120, video_token_id=121, vision_start_token_id=122,
        vision_end_token_id=123, tie_word_embeddings=False,
    )
    model = Qwen3VLForConditionalGeneration(config).train()
    # Make normalized and original tokens observably different, even if a
    # future initialization produces near-unit hidden-state variance.
    with torch.no_grad():
        model.model.language_model.norm.weight.fill_(1.7)
    return model


def _context_model(vlm):
    # Use the actual extraction method without constructing an unrelated
    # MotionExpert or action head. This test isolates the VLM token contract.
    model = Psi0Model.__new__(Psi0Model)
    torch.nn.Module.__init__(model)
    model.vlm_model = vlm
    model.uses_dual_source = True
    model.tune_vlm = True
    model.vlm_layer_indices = None
    return model


def _loss(features):
    return features.tokens.square().mean() + features.p_v.square().mean()


@pytest.mark.parametrize("trainer_type", [PosttrainTrainer, FinetuneTrainer, SonicTrainer],
                         ids=["posttrain", "finetune", "sonic"])
@pytest.mark.parametrize("gradient_checkpointing", [False, True], ids=["plain", "checkpointed"])
def test_real_outer_qwen_tokens_bypass_final_norm_and_trainer_freeze_preserves_values(
    trainer_type, gradient_checkpointing,
):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(394)
        vlm = _tiny_qwen()
        model = _context_model(vlm)
        prefix = dict(input_ids=torch.tensor([[2, 5, 9, 12, 17, 0], [3, 8, 13, 19, 0, 0]]),
                      attention_mask=torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]]),
                      pixel_values=None, image_grid_thw=None)
        norm = vlm.model.language_model.norm
        norm_io = {}

        def observe_norm(_module, inputs, output):
            norm_io["input"] = inputs[0].detach().clone()
            norm_io["output"] = output.detach().clone()

        observer = norm.register_forward_hook(observe_norm)
        try:
            assert norm.weight.requires_grad
            before = model.extract_vlm_features(**prefix)
            assert before.tokens.shape == (2, 1, 6, 32)
            torch.testing.assert_close(before.tokens[:, 0], norm_io["input"], rtol=0, atol=0)
            assert not torch.equal(before.tokens[:, 0], norm_io["output"])
            _loss(before).backward()
            assert norm.weight.grad is None  # Unused even while requires_grad=True.
            embedding = vlm.get_input_embeddings().weight
            query = vlm.model.language_model.layers[-1].self_attn.q_proj.weight
            for parameter in (embedding, query):
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                assert parameter.grad.abs().sum() > 0
            saved_tokens = before.tokens.detach().clone()
            saved_pool = before.p_v.detach().clone()
            vlm.zero_grad(set_to_none=True)

            cfg = SimpleNamespace(action_architecture="dual_source_v1", tune_vlm=True,
                                  tune_mm_llm=False, tune_mm_vision=False, tune_mm_mlp=False,
                                  gradient_checkpointing=gradient_checkpointing)
            trainer = SimpleNamespace(model=model, model_cfg=cfg,
                                      _vlm_group_of=trainer_type._vlm_group_of)
            trainer.vlm_trainable_components = lambda: trainer_type.vlm_trainable_components(trainer)
            assert trainer_type.apply_vlm_trainability(trainer) == {
                "lang_backbone", "vision_tower", "mm_projector"}
            assert not norm.weight.requires_grad
            assert not vlm.lm_head.weight.requires_grad
            assert all(parameter.requires_grad for name, parameter in vlm.named_parameters()
                       if name not in {"model.language_model.norm.weight", "lm_head.weight"})
            after = model.extract_vlm_features(**prefix)
            torch.testing.assert_close(after.tokens, saved_tokens, rtol=0, atol=0)
            torch.testing.assert_close(after.p_v, saved_pool, rtol=0, atol=0)
            _loss(after).backward()
            assert norm.weight.grad is None
            for parameter in (embedding, query):
                assert parameter.requires_grad and parameter.grad is not None
                assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        finally:
            observer.remove()
