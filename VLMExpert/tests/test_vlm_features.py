"""CPU pooling and actual Psi0 method wiring with lightweight model doubles.

The AST harness executes methods from psi0.py without importing its optional
Qwen/diffusers dependencies. These checks do not validate real model inference.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from psi.models.vlm_features import VLMFeatures, masked_mean_vlm_tokens


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_masked_mean_discards_nonfinite_padding_and_preserves_gradients(dtype):
    tokens = torch.tensor(
        [[[[2, 4], [float("nan"), float("inf")], [6, 8], [float("-inf"), 0]],
          [[20, 40], [float("inf"), float("nan")], [60, 80], [0, float("-inf")]]]],
        dtype=dtype,
        requires_grad=True,
    )
    mask = torch.tensor([[True, False, True, False]])

    pooled = masked_mean_vlm_tokens(tokens, mask)

    assert pooled.dtype == dtype
    assert pooled.shape == (1, 2, 2)
    torch.testing.assert_close(pooled, tokens.new_tensor([[[4, 6], [40, 60]]]))
    pooled.sum().backward()
    expected_grad = mask[:, None, :, None].expand_as(tokens).to(dtype) / 2
    torch.testing.assert_close(tokens.grad, expected_grad)
    assert torch.isfinite(tokens.grad).all()


def test_half_pooling_accumulates_without_half_sum_overflow():
    tokens = torch.full((2, 1, 128, 3), 60000, dtype=torch.float16)
    pooled = masked_mean_vlm_tokens(tokens)
    assert torch.isfinite(pooled).all()
    torch.testing.assert_close(pooled, tokens[:, :, 0])


def test_double_pooling_preserves_precision_beyond_float32():
    tokens = torch.tensor([1e10, 1, -1e10], dtype=torch.float64).view(1, 1, 3, 1)
    pooled = masked_mean_vlm_tokens(tokens)
    torch.testing.assert_close(pooled, torch.full((1, 1, 1), 1 / 3, dtype=torch.float64))


@pytest.mark.parametrize(
    ("tokens", "mask", "message"),
    [
        (torch.ones(2, 3, 4), None, "floating-point"),
        (torch.ones(1, 1, 2, 3, dtype=torch.int64), None, "floating-point"),
        (torch.ones(1, 0, 2, 3), None, "nonempty"),
        (torch.ones(1, 1, 2, 3), torch.ones(1, 3), "shape"),
        (torch.ones(2, 1, 2, 3), torch.tensor([[1, 0], [0, 0]]), "at least one"),
        (torch.ones(1, 1, 2, 3), torch.tensor([[1, 2]]), "0/1"),
        (torch.ones(1, 1, 2, 3), torch.tensor([[1, float("nan")]]), "0/1"),
        (torch.ones(1, 1, 2, 3), torch.ones(1, 2, device="meta"), "same device"),
    ],
)
def test_pooling_rejects_invalid_inputs(tokens, mask, message):
    with pytest.raises(ValueError, match=message):
        masked_mean_vlm_tokens(tokens, mask)


@pytest.mark.parametrize("views", [1, 3])
def test_p_v_preserves_batch_and_selected_layer_dimensions(views):
    tokens = torch.arange(2 * views * 4 * 5, dtype=torch.float32).reshape(2, views, 4, 5)
    pooled = masked_mean_vlm_tokens(tokens)
    features = VLMFeatures(tokens, pooled, None, tokens[:, -1])

    expected = tokens.mean(dim=2)
    torch.testing.assert_close(features.pooled_tokens, expected)
    assert features.p_v.shape == ((2, 5) if views == 1 else (2, views, 5))
    torch.testing.assert_close(features.p_v, expected[:, 0] if views == 1 else expected)


class _Batch(dict):
    def to(self, device):
        return _Batch({key: value.to(device) for key, value in self.items()})


class _Processor:
    tokenizer = SimpleNamespace(pad_token_id=0)

    def apply_chat_template(self, messages, **kwargs):
        return next(item["text"] for item in messages[0]["content"] if item["type"] == "text")

    def __call__(self, *, text, images, **kwargs):
        ids = torch.tensor([[int(token) for token in text[0].split()]])
        return _Batch(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            pixel_values=torch.ones(len(images), 4),
            image_grid_thw=torch.ones(len(images), 3, dtype=torch.long),
        )


def _vision_info(messages, **kwargs):
    images = [item["image"] for group in messages for message in group
              for item in message["content"] if item["type"] == "image"]
    return images, None


class _VLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(128).reshape(32, 4) / 100)
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append({**kwargs, "grad_enabled": torch.is_grad_enabled(), "training": self.training})
        hidden = self.embedding(kwargs["input_ids"])
        self.hidden_states = (hidden - 10, hidden + 5, hidden * 2)
        return SimpleNamespace(hidden_states=self.hidden_states)


class _LegacyHeader(nn.Module):
    """Ignore the new branch, retain the old token/CLIP paths and sample grads."""

    last_state_drop_frac = 0.125

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        context = kwargs["joint_attention_kwargs"]
        views = context["views"]
        mask = kwargs.get("vlm_attn_mask")
        if mask is None:
            mask = torch.ones(views.shape[0], views.shape[2], dtype=torch.bool, device=views.device)
        valid_context = views.masked_fill(~mask[:, None, :, None].bool(), 0)
        summary = valid_context.sum(dim=(1, 2, 3)) / (mask.sum(-1) * views.shape[1] * views.shape[-1])
        actions = context["action_hidden_embeds"] * 0.2 + summary[:, None, None] * 0.01
        clip = kwargs.get("pooled_projections")
        if clip is not None:
            actions = actions + clip.mean(-1)[:, None, None] * 0.1
        return SimpleNamespace(action=actions)


class _Scheduler:
    def set_timesteps(self, steps):
        self.timesteps = torch.arange(steps, 0, -1, dtype=torch.float32)
        self.sigmas = torch.linspace(1, 0, steps + 1)

    def index_for_timestep(self, timestep):
        return int((self.timesteps == timestep).nonzero()[0, 0])

    def step(self, *, model_output, timestep, sample):
        return SimpleNamespace(prev_sample=sample - 0.2 * model_output)


@pytest.fixture(scope="module")
def psi0_harness():
    source = Path(__file__).parents[1] / "src/psi/models/psi0.py"
    tree = ast.parse(source.read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Psi0Model")
    names = {
        "_select_vlm_views", "extract_vlm_features", "forward", "_collate_vlm_batch",
        "predict_action", "predict_action_with_training_rtc_flow", "predict_action_with_rtc_flow",
        "predict_action_with_rtc_flow_naive_inpaint", "_create_soft_mask",
    }
    methods = [copy.deepcopy(node) for node in model.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == names
    harness = ast.ClassDef(name="Psi0Harness", bases=[ast.Attribute(value=ast.Name(id="nn", ctx=ast.Load()),
                          attr="Module", ctx=ast.Load())], keywords=[], body=methods, decorator_list=[])
    output = copy.deepcopy(next(node for node in tree.body
                                if isinstance(node, ast.ClassDef) and node.name == "HumanFoundationModelOutput"))
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             output, harness], type_ignores=[])
    namespace = dict(__name__=__name__, torch=torch, nn=nn, dataclass=dataclass, BaseOutput=object,
                     VLMFeatures=VLMFeatures, masked_mean_vlm_tokens=masked_mean_vlm_tokens,
                     pad_sequence=pad_sequence, process_vision_info=_vision_info,
                     overwatch=SimpleNamespace(warning=lambda *args, **kwargs: None))
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["Psi0Harness"]


@pytest.fixture
def model(psi0_harness):
    model = psi0_harness()
    model.device = torch.device("cpu")
    model.vlm_model = _VLM()
    model.vlm_processor = _Processor()
    model.action_header = _LegacyHeader()
    model.noise_scheduler = _Scheduler()
    model.vlm_layer_indices = None
    model.action_horizon = 4
    model.action_dim = 2
    return model


def _prefix():
    return dict(input_ids=torch.tensor([[1, 2, 0], [3, 4, 5]]),
                attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
                pixel_values=torch.ones(2, 4), image_grid_thw=torch.ones(2, 3, dtype=torch.long))


@pytest.mark.parametrize("layers", [None, [0, 2]])
def test_real_extraction_keeps_layer_selection_and_train_grad_state(model, layers):
    model.vlm_layer_indices = layers
    model.eval()
    prefix = _prefix()
    features = model.extract_vlm_features(**prefix)

    for key, value in prefix.items():
        assert model.vlm_model.calls[-1][key] is value
    selected = [-1] if layers is None else layers
    for view, layer in enumerate(selected):
        torch.testing.assert_close(features.tokens[:, view], model.vlm_model.hidden_states[layer])
        for batch, valid in enumerate(prefix["attention_mask"].bool()):
            torch.testing.assert_close(features.pooled_tokens[batch, view],
                                       model.vlm_model.hidden_states[layer][batch, valid].mean(0))
    assert features.last_hidden_state is model.vlm_model.hidden_states[-1]
    assert features.attention_mask is prefix["attention_mask"]
    assert features.p_v.shape == ((2, 4) if layers is None else (2, 2, 4))
    assert features.p_v.requires_grad
    assert model.vlm_model.calls[-1]["grad_enabled"]
    assert not model.vlm_model.calls[-1]["training"]


def test_real_forward_retains_original_tokens_clip_and_optional_output(model):
    model.train()
    prefix = _prefix()
    clip = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    arguments = dict(**prefix, action_samples=torch.zeros(2, 4, 2), states=torch.zeros(2, 1, 3),
                     timestep=torch.ones(2), pooled_projections=clip)
    default = model(**arguments)
    detailed = model(**arguments, return_vlm_features=True)

    assert default.vlm_features is None
    features = detailed.vlm_features
    call = model.action_header.calls[-1]
    assert call["joint_attention_kwargs"]["views"] is features.tokens
    assert call["joint_attention_kwargs"]["vlm_features"] is features
    assert call["pooled_projections"] is clip
    assert call["vlm_attn_mask"] is prefix["attention_mask"]
    torch.testing.assert_close(features.tokens[:, 0], features.last_hidden_state)
    torch.testing.assert_close(default.action, detailed.action, rtol=0, atol=0)
    without_clip = model(**{**arguments, "pooled_projections": None}, return_vlm_features=True)
    torch.testing.assert_close(detailed.action - without_clip.action,
                               (clip.mean(-1)[:, None, None] * 0.1).expand_as(detailed.action))
    torch.testing.assert_close(features.p_v, without_clip.vlm_features.p_v)
    assert detailed.state_drop_frac == 0.125
    assert detailed.to_tuple() == (None, detailed.action)
    assert not detailed.vlm_feat_norm.requires_grad
    norms = features.last_hidden_state.detach().float().norm(dim=-1)
    expected_norm = norms[prefix["attention_mask"].bool()].mean()
    torch.testing.assert_close(detailed.vlm_feat_norm, expected_norm)
    (detailed.action.sum() + features.p_v.sum()).backward()
    assert model.vlm_model.embedding.weight.grad[1:].abs().sum() > 0
    torch.testing.assert_close(model.vlm_model.embedding.weight.grad[0], torch.zeros(4))
    assert model.vlm_model.calls[-1]["training"]


SAMPLERS = ["predict_action", "predict_action_with_training_rtc_flow", "predict_action_with_rtc_flow",
            "predict_action_with_rtc_flow_naive_inpaint"]


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda', but CUDA is not available.*:UserWarning")
@pytest.mark.parametrize("sampler_name", SAMPLERS)
def test_real_sampling_optional_features_preserve_actions_and_one_vlm_pass_per_chunk(model, sampler_name):
    model.eval()
    arguments = dict(observations=[[object()]], states=torch.zeros(1, 1, 3), instructions=["1 2 3"],
                     num_inference_steps=3, traj2ds=None)
    if sampler_name != "predict_action":
        arguments.update(prev_actions=torch.full((1, 4, 2), 0.3), inference_delay=1)
        if sampler_name == "predict_action_with_training_rtc_flow":
            arguments.update(max_delay=2)
        else:
            arguments.update(execution_horizon=2)
    sampler = getattr(model, sampler_name)

    torch.manual_seed(42)
    default = sampler(**arguments)
    assert isinstance(default, torch.Tensor)
    assert len(model.vlm_model.calls) == 1
    assert len(model.action_header.calls) == 3
    torch.manual_seed(42)
    detailed = sampler(**arguments, return_vlm_features=True)

    torch.testing.assert_close(default, detailed.action, rtol=0, atol=0)
    assert detailed.action.shape == (1, 4, 2)
    assert detailed.vlm_features.p_v.shape == (1, 4)
    assert torch.isfinite(detailed.action).all()
    assert len(model.vlm_model.calls) == 2
    assert len(model.action_header.calls) == 6
    assert not detailed.vlm_features.p_v.requires_grad
    for call in model.action_header.calls[3:]:
        context = call["joint_attention_kwargs"]
        assert context["vlm_features"] is detailed.vlm_features
        assert context["views"] is detailed.vlm_features.tokens


@pytest.mark.parametrize("sampler_name", SAMPLERS[:2])
def test_real_batched_sampling_right_pads_prefix_and_keeps_pool_mask(model, sampler_name):
    arguments = dict(observations=[[object()], [object()]], states=torch.zeros(2, 1, 3),
                     instructions=["1 2", "3 4 5"], num_inference_steps=2, traj2ds=None,
                     return_vlm_features=True)
    if sampler_name == "predict_action_with_training_rtc_flow":
        arguments.update(prev_actions=torch.zeros(2, 4, 2), inference_delay=1, max_delay=2)
    result = getattr(model, sampler_name)(**arguments)

    mask = result.vlm_features.attention_mask
    torch.testing.assert_close(mask, torch.tensor([[True, True, False], [True, True, True]]))
    assert result.action.shape == (2, 4, 2)
    assert len(model.vlm_model.calls) == 1
    for batch, valid in enumerate(mask):
        torch.testing.assert_close(result.vlm_features.p_v[batch],
                                   result.vlm_features.tokens[batch, 0, valid].mean(0))
    for call in model.action_header.calls:
        assert call["vlm_attn_mask"] is mask
