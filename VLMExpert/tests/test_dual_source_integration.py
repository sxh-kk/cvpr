"""Real dual-source tensors/Psi0 methods with small frozen expert doubles.

The optional Qwen/diffusers packages and production checkpoints are absent in
this test runtime. Only pretrained experts and the diffusion scheduler are
doubled; action-head operations and Psi0 tensor-routing methods are real.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from psi.models.dual_source_action import DualSourceActionTransformer
from psi.models.frozen_motion import FrozenMotionConditioner, FrozenMotionFeatures

from test_vlm_features import SAMPLERS, _Processor, _Scheduler, _VLM, _prefix, build_psi0_harness


class _MotionExpert:
    def __init__(self):
        self.model = nn.Sequential(nn.Linear(1, 768), nn.Dropout(0.9))
        self.cfg = SimpleNamespace(DATA=SimpleNamespace(FPS=5, WINDOW=11, FORECAST_PREFIX=6))
        self.calls = []

    def forward_features(self, conditioning, **kwargs):
        self.calls.append(dict(conditioning=conditioning, grad_enabled=torch.is_grad_enabled(),
                               training=self.model.training, **kwargs))
        image = conditioning["img_embs"].to(device=self.device, dtype=self.dtype)
        summary = image.mean(dim=(1, 2, 3)).unsqueeze(-1)
        hidden = self.model(summary).unsqueeze(1).expand(-1, 11, -1)
        generator = torch.Generator(device=self.device).manual_seed(kwargs["seed"])
        noise = torch.randn(hidden.shape, generator=generator, device=self.device, dtype=self.dtype)
        return {"one_step_hidden": hidden + noise * 0.1,
                "one_step_global": torch.full_like(hidden[:, 0], 12345)}


def _images(batch=2):
    return torch.arange(batch * 6 * 5 * 1024, dtype=torch.float32).reshape(batch, 6, 5, 1024) / 10000


@pytest.fixture(scope="module")
def dual_harness():
    return build_psi0_harness(include_init=True, extra_globals={
        "DualSourceActionTransformer": DualSourceActionTransformer,
    })


def _build_model(dual_harness, *, vlm_model=None, motion_expert=None, **overrides):
    config = dict(action_architecture="dual_source_v1", use_dit=False, combined_temb=False,
                  use_film=False, tune_vlm=False, tune_motion=False, vlm_layer_indices=None, n_conditions=0,
                  pooled_text_encoder=None, action_dim=2, action_chunk_size=4,
                  hidden_dim=16, num_blocks=2, nhead=2, view_feature_dim=4, odim=3,
                  time_dim=8, qk_norm="rms_norm", state_drop_prob=0.0,
                  state_null_token=False, dropout=0.0, motion_feature_seed=17)
    config.update(overrides)
    expert = _MotionExpert() if motion_expert is None else motion_expert
    model_cfg = SimpleNamespace(**config)
    model = dual_harness(model_cfg, _VLM() if vlm_model is None else vlm_model,
                         motion_expert=expert)
    model.test_model_cfg = model_cfg
    model.device = torch.device("cpu")
    model.vlm_processor = _Processor()
    model.noise_scheduler = _Scheduler()
    model.action_horizon = config["action_chunk_size"]
    model.action_dim = config["action_dim"]
    model.test_motion_expert = expert
    model.head_calls = []
    model.action_header.register_forward_pre_hook(
        lambda module, args, kwargs: model.head_calls.append(kwargs), with_kwargs=True)
    return model


@pytest.fixture
def dual_model(dual_harness):
    return _build_model(dual_harness)


def _forward_args(batch=2):
    return dict(**{key: value[:batch] for key, value in _prefix().items()},
                action_samples=torch.randn(batch, 4, 2), states=torch.randn(batch, 1, 3),
                timestep=torch.full((batch,), 500.0))


def _open_gates(model):
    with torch.no_grad():
        for block in model.action_header.transformer_blocks:
            block.mod_head[-1].weight.fill_(0.01)
            block.mod_head[-1].bias.view(4, 3, -1)[:, 2].fill_(0.15)


def test_real_psi0_forward_freezes_both_experts_and_updates_both_cross_attention_branches(dual_model):
    model = dual_model.train()
    _open_gates(model)
    assert model.training and model.action_header.training
    assert not model.vlm_model.training and not model.motion_conditioner.training
    assert not model.test_motion_expert.model.training
    frozen_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                     if not name.startswith("action_header.")}
    updated = [model.action_header.vlm_context_proj[-1].weight,
               model.action_header.motion_context_proj[-1].weight,
               model.action_header.vlm_pool_proj[-1].weight,
               model.action_header.motion_pool_proj[-1].weight]
    head_before = [parameter.detach().clone() for parameter in updated]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    images = _images().requires_grad_()

    result = model(**_forward_args(), motion_img_embs=images, return_vlm_features=True)
    result.action.square().mean().backward()

    assert result.action.shape == (2, 4, 2)
    assert torch.isfinite(result.action).all()
    assert not result.vlm_features.tokens.requires_grad
    assert not result.vlm_features.p_v.requires_grad
    assert not result.motion_features.tokens.requires_grad
    torch.testing.assert_close(result.vlm_features.tokens[:, 0], model.vlm_model.hidden_states[-1])
    assert model.vlm_model.calls[-1]["grad_enabled"] is False
    assert model.vlm_model.calls[-1]["training"] is False
    call = model.head_calls[-1]
    assert call["joint_attention_kwargs"]["vlm_features"] is result.vlm_features
    assert call["joint_attention_kwargs"]["motion_features"] is result.motion_features
    for parameter in updated:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    for parameter in model.action_header.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    for block in model.action_header.transformer_blocks:
        for attention in (block.vlm_attn, block.motion_attn):
            assert torch.isfinite(attention.to_v.weight.grad).all()
            assert attention.to_v.weight.grad.abs().sum() > 0
    assert images.grad is None
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in frozen_before:
            assert not parameter.requires_grad and parameter.grad is None
            torch.testing.assert_close(parameter, frozen_before[name], rtol=0, atol=0)
    for parameter, before in zip(updated, head_before):
        assert not torch.equal(parameter, before)


@pytest.mark.parametrize("sampler_name", SAMPLERS)
@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda', but CUDA is not available.*:UserWarning")
def test_real_dual_sampling_reuses_both_memories_for_every_denoising_step(dual_model, sampler_name):
    model = dual_model.eval()
    _open_gates(model)
    arguments = dict(observations=[[object()]], states=torch.zeros(1, 1, 3), instructions=["1 2 3"],
                     num_inference_steps=3, traj2ds=None, motion_img_embs=_images(1))
    if sampler_name != "predict_action":
        arguments.update(prev_actions=torch.full((1, 4, 2), 0.3), inference_delay=1)
        if sampler_name == "predict_action_with_training_rtc_flow":
            arguments.update(max_delay=2)
        else:
            arguments.update(execution_horizon=2)
    sampler = getattr(model, sampler_name)

    torch.manual_seed(42)
    default = sampler(**arguments)
    torch.manual_seed(42)
    detailed = sampler(**arguments, return_vlm_features=True)

    assert isinstance(default, torch.Tensor)
    torch.testing.assert_close(default, detailed.action, rtol=0, atol=0)
    assert torch.isfinite(detailed.action).all() and detailed.action.shape == (1, 4, 2)
    assert len(model.vlm_model.calls) == len(model.test_motion_expert.calls) == 2
    assert len(model.head_calls) == 6
    for call in model.head_calls[3:]:
        context = call["joint_attention_kwargs"]
        assert context["vlm_features"] is detailed.vlm_features
        assert context["views"] is detailed.vlm_features.tokens
        assert context["motion_features"] is detailed.motion_features
        assert context["motion_features"].tokens.shape == (1, 11, 768)
        assert call["vlm_attn_mask"] is detailed.vlm_features.attention_mask
    assert not detailed.vlm_features.tokens.requires_grad
    assert not detailed.motion_features.tokens.requires_grad


def test_real_training_rtc_keeps_known_prefix_and_state_at_zero_time(dual_model):
    model = dual_model.eval()
    times = []
    model.action_header.time_ins_embed.register_forward_pre_hook(lambda module, args: times.append(args[0].clone()))
    result = model.predict_action_with_training_rtc_flow(
        observations=[[object()], [object()]], states=torch.zeros(2, 1, 3), instructions=["1 2", "3 4 5"],
        num_inference_steps=3, traj2ds=None, motion_img_embs=_images(),
        prev_actions=torch.full((2, 4, 2), 0.3), inference_delay=1, max_delay=2,
        return_vlm_features=True,
    )
    assert len(times) == 3
    for condition_times in times:
        assert condition_times.shape == (2, 5)
        assert torch.count_nonzero(condition_times[:, :2]) == 0
        assert torch.all(condition_times[:, 2:] > 0)
    mask = result.vlm_features.attention_mask
    torch.testing.assert_close(mask, torch.tensor([[True, True, False], [True, True, True]]))
    assert result.action.shape == (2, 4, 2)
    assert len(model.vlm_model.calls) == len(model.test_motion_expert.calls) == 1


def test_reusable_motion_features_skip_extraction_and_preserve_default_forward_actions(dual_model):
    model = dual_model.eval()
    arguments = _forward_args()
    features = model.extract_motion_features(motion_img_embs=_images())
    first = model(**arguments, motion_features=features)
    second = model(**arguments, motion_features=features, return_vlm_features=True)
    assert first.vlm_features is None and first.motion_features is None
    assert second.motion_features is features
    torch.testing.assert_close(first.action, second.action, rtol=0, atol=0)
    assert len(model.test_motion_expert.calls) == 1
    assert len(model.vlm_model.calls) == 2


@pytest.mark.parametrize("inputs, exception, message", [
    ({}, ValueError, "requires motion_img_embs"),
    ({"motion_features": torch.zeros(1, 11, 768)}, TypeError, "MotionFeatures"),
    ({"motion_img_embs": torch.zeros(1, 6, 5, 1024),
      "motion_features": FrozenMotionFeatures(torch.zeros(1, 11, 768))}, ValueError, "either"),
    ({"motion_img_embs": torch.zeros(1, 11, 5, 1024)}, ValueError, "observed DINO"),
])
def test_real_forward_rejects_missing_or_ambiguous_motion_before_vlm(dual_model, inputs, exception, message):
    with pytest.raises(exception, match=message):
        dual_model(**_forward_args(), **inputs)
    assert not dual_model.vlm_model.calls


def test_complete_model_checkpoint_roundtrip_contains_frozen_experts_and_architecture_signature(dual_model, dual_harness):
    model = dual_model.eval()
    _open_gates(model)
    arguments = _forward_args()
    state = model.state_dict()
    assert "action_header._architecture_signature" in state
    assert "vlm_model.embedding.weight" in state
    assert "motion_conditioner.model.0.weight" in state
    restored = _build_model(dual_harness).eval()
    restored.load_state_dict(state)
    original_result = model(**arguments, motion_img_embs=_images())
    restored_result = restored(**arguments, motion_img_embs=_images())
    torch.testing.assert_close(original_result.action, restored_result.action, rtol=0, atol=0)
    for name, parameter in restored.named_parameters():
        assert parameter.requires_grad == name.startswith("action_header.")
    legacy = {key: value for key, value in state.items() if key != "action_header._architecture_signature"}
    with pytest.raises(RuntimeError, match="legacy joint-attention"):
        restored.load_state_dict(legacy, strict=False)


@pytest.mark.parametrize("override, message", [
    ({"vlm_layer_indices": [0, 2]}, "final VLM layer"),
    ({"combined_temb": True}, "legacy"),
    ({"n_conditions": 1}, "n_conditions=0"),
])
def test_actual_psi0_constructor_rejects_configuration_that_breaks_dual_contract(dual_harness, override, message):
    with pytest.raises(ValueError, match=message):
        _build_model(dual_harness, **override)


def test_motion_conditioner_freezes_expert_and_uses_complete_hidden_mean():
    expert = _MotionExpert()
    conditioner = FrozenMotionConditioner(expert, seed=17)
    conditioner.train()
    images = _images().requires_grad_()
    rng = torch.random.get_rng_state().clone()

    first = conditioner(images)
    second = conditioner(images)

    assert not conditioner.training and not expert.model.training
    assert all(not parameter.requires_grad for parameter in conditioner.parameters())
    assert not first.tokens.requires_grad and not torch.is_inference(first.tokens)
    assert first.tokens.shape == (2, 11, 768)
    torch.testing.assert_close(first.p_m, first.tokens.float().mean(1))
    torch.testing.assert_close(first.tokens, second.tokens, rtol=0, atol=0)
    assert not torch.equal(first.tokens[:, 0], first.tokens[:, -1])
    assert torch.equal(rng, torch.random.get_rng_state())
    for call in expert.calls:
        assert set(call["conditioning"]) == {"img_embs"}
        torch.testing.assert_close(call["conditioning"]["img_embs"], images, rtol=0, atol=0)
        assert not call["conditioning"]["img_embs"].requires_grad
        assert not call["grad_enabled"] and not call["training"]
        assert call["seed"] == 17 and call["task"] == "fore"
        assert call["forecast_prefix"] == 6 and call["output_frames"] == 11
    projection = nn.Linear(768, 4)
    projection(first.tokens).square().mean().backward()
    assert torch.isfinite(projection.weight.grad).all()
    assert projection.weight.grad.abs().sum() > 0
    assert images.grad is None
    assert all(parameter.grad is None for parameter in expert.model.parameters())


@pytest.mark.parametrize("shape", [(1, 11, 5, 1024), (1, 6, 5120), (1, 6, 4, 1024), (0, 6, 5, 1024)])
def test_motion_conditioner_rejects_incompatible_observed_history(shape):
    expert = _MotionExpert()
    with pytest.raises(ValueError, match="observed DINO"):
        FrozenMotionConditioner(expert)(torch.zeros(shape))
    assert not expert.calls


def test_cached_inference_motion_tokens_become_autograd_usable_constants():
    with torch.inference_mode():
        cached = FrozenMotionFeatures(torch.randn(1, 11, 768))
    assert not torch.is_inference(cached.tokens)
    assert not cached.tokens.requires_grad
    projection = nn.Linear(768, 2)
    projection(cached.tokens).square().mean().backward()
    assert torch.isfinite(projection.weight.grad).all()


def test_motion_conditioner_rejects_nonfinite_history_before_expert_call():
    expert = _MotionExpert()
    images = _images(1)
    images[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        FrozenMotionConditioner(expert)(images)
    assert not expert.calls
