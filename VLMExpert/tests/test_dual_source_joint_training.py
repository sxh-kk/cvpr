"""Joint-training autograd through real Psi0 methods and the dual-source head.

Qwen and MotionExpert producers are small trainable doubles. These tests check
tensor routing and updates, not production-checkpoint accuracy or throughput.
"""

from __future__ import annotations

import ast
from collections import Counter
import copy
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from psi.models.dual_source_action import DualSourceActionTransformer
from psi.models.frozen_motion import MotionFeatures

from test_dual_source_integration import _MotionExpert, _build_model, _forward_args, _images, _open_gates
from test_vlm_features import SAMPLERS, _Processor, _Scheduler, _prefix, build_psi0_harness


class _Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(4, 4)
        self.merger = nn.Linear(4, 4)

    def forward(self, value):
        return self.merger(torch.nn.functional.silu(self.patch_embed(value)))


class _JointVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = _Vision()
        self.model.language_model = nn.Module()
        language = self.model.language_model
        language.embed_tokens = nn.Embedding(32, 4)
        language.block = nn.Linear(4, 4)
        language.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 32, bias=False)
        self.config = SimpleNamespace(use_cache=False, text_config=SimpleNamespace(use_cache=False, hidden_size=4))
        self.calls = []

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def forward(self, **kwargs):
        self.calls.append(dict(grad_enabled=torch.is_grad_enabled(), training=self.training))
        language = self.model.language_model
        embedding = language.embed_tokens(kwargs["input_ids"])
        image = kwargs["pixel_values"].reshape(embedding.shape[0], -1, 4).mean(1)
        merged = embedding + self.model.visual(image)[:, None]
        hidden = torch.nn.functional.silu(language.block(merged)) + merged
        self.hidden_states = (embedding, hidden, language.norm(hidden))
        return SimpleNamespace(hidden_states=self.hidden_states)


class _MotionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(1, 768)
        self.dropout = nn.Dropout(0.1)
        # Production one-step hidden extraction also leaves the SMPL-H output
        # projection unread; it must not become an unused trainable parameter.
        self.output_process = nn.Linear(768, 245)

    def forward(self, value):
        return self.dropout(self.backbone(value))


class _JointMotionExpert(_MotionExpert):
    def __init__(self):
        super().__init__()
        self.model = _MotionNetwork()


@pytest.fixture(scope="module")
def joint_harness():
    return build_psi0_harness(include_init=True, extra_globals={
        "DualSourceActionTransformer": DualSourceActionTransformer,
    })


def _joint_model(joint_harness, *, vlm_model=None, **flags):
    torch.manual_seed(91)
    model = _build_model(joint_harness, vlm_model=_JointVLM() if vlm_model is None else vlm_model,
                         motion_expert=_JointMotionExpert(),
                         **{**dict(tune_vlm=True, tune_motion=True), **flags})
    _open_gates(model)
    return model


def _nonzero_finite(gradient):
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_joint_forward_reaches_both_experts_final_vlm_norm_and_both_pooled_summaries(joint_harness):
    model = _joint_model(joint_harness).train()
    vlm = model.vlm_model
    motion = model.motion_conditioner.model
    assert vlm.training and motion.training and model.action_header.training
    assert not vlm.lm_head.weight.requires_grad
    assert not motion.output_process.weight.requires_grad
    assert vlm.model.language_model.norm.weight.requires_grad
    updated = [vlm.get_input_embeddings().weight, vlm.model.visual.patch_embed.weight,
               vlm.model.visual.merger.weight, vlm.model.language_model.norm.weight,
               motion.backbone.weight, model.action_header.action_proj_out.linear.weight]
    before = [parameter.detach().clone() for parameter in updated]
    unused = [vlm.lm_head.weight, motion.output_process.weight]
    unused_before = [parameter.detach().clone() for parameter in unused]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3, weight_decay=0)
    images = _images().requires_grad_()

    result = model(**_forward_args(), motion_img_embs=images, return_vlm_features=True)
    result.vlm_features.tokens.retain_grad()
    result.vlm_features.pooled_tokens.retain_grad()
    result.motion_features.tokens.retain_grad()
    result.motion_features.p_m.retain_grad()
    result.action.square().mean().backward()

    assert vlm.calls[-1] == {"grad_enabled": True, "training": True}
    assert model.test_motion_expert.calls[-1]["grad_enabled"]
    assert model.test_motion_expert.calls[-1]["training"]
    torch.testing.assert_close(result.vlm_features.tokens[:, 0], vlm.hidden_states[-1])
    _nonzero_finite(result.vlm_features.pooled_tokens.grad)
    _nonzero_finite(result.motion_features.p_m.grad)
    _nonzero_finite(result.motion_features.tokens.grad[:, :6])
    _nonzero_finite(result.motion_features.tokens.grad[:, 6:])
    torch.testing.assert_close(result.vlm_features.tokens.grad[0, 0, -1], torch.zeros(4))
    assert images.grad is None  # Frozen DINO features are observation constants.
    for parameter in updated:
        _nonzero_finite(parameter.grad)
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    optimizer.step()
    for parameter, previous in zip(updated, before):
        assert not torch.equal(parameter, previous)
    for parameter, previous in zip(unused, unused_before):
        assert parameter.grad is None
        torch.testing.assert_close(parameter, previous, rtol=0, atol=0)


def test_motion_features_mean_retains_all_eleven_token_gradients():
    tokens = torch.randn(2, 11, 768, requires_grad=True)
    features = MotionFeatures(tokens)
    assert features.tokens is tokens and features.p_m.requires_grad
    assert features.p_m is features.p_m
    torch.testing.assert_close(features.p_m, tokens.mean(1))
    features.p_m.sum().backward()
    torch.testing.assert_close(tokens.grad, torch.full_like(tokens, 1 / 11))


def test_frozen_dino_inference_outputs_can_feed_trainable_motion_backbone(joint_harness):
    model = _joint_model(joint_harness).train()
    with torch.inference_mode():
        images = _images()
    assert torch.is_inference(images)

    result = model(**_forward_args(), motion_img_embs=images)
    result.action.square().mean().backward()

    _nonzero_finite(model.motion_conditioner.model.backbone.weight.grad)
    conditioned_images = model.test_motion_expert.calls[-1]["conditioning"]["img_embs"]
    assert not torch.is_inference(conditioned_images)
    assert not conditioned_images.requires_grad and images.grad is None
    torch.testing.assert_close(conditioned_images, images, rtol=0, atol=0)


def test_tied_vocabulary_head_does_not_freeze_the_language_input_embedding(joint_harness):
    vlm = _JointVLM()
    vlm.lm_head.weight = vlm.get_input_embeddings().weight
    model = _joint_model(joint_harness, vlm_model=vlm).train()
    assert vlm.lm_head.weight is vlm.get_input_embeddings().weight
    assert vlm.get_input_embeddings().weight.requires_grad
    result = model(**_forward_args(), motion_img_embs=_images())
    result.action.square().mean().backward()
    _nonzero_finite(vlm.get_input_embeddings().weight.grad)


def test_graph_connected_features_can_be_reused_within_one_training_loss(joint_harness):
    model = _joint_model(joint_harness).train()
    vlm = model.extract_vlm_features(**_prefix())
    motion = model.extract_motion_features(motion_img_embs=_images())
    result = model(**_forward_args(), vlm_features=vlm, motion_features=motion, return_vlm_features=True)
    assert result.vlm_features is vlm and result.motion_features is motion
    result.action.square().mean().backward()
    _nonzero_finite(model.vlm_model.get_input_embeddings().weight.grad)
    _nonzero_finite(model.motion_conditioner.model.backbone.weight.grad)
    assert len(model.vlm_model.calls) == len(model.test_motion_expert.calls) == 1


@pytest.mark.parametrize("source", ["vlm", "motion"])
@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode], ids=["no_grad", "inference"])
def test_detached_or_inference_caches_cannot_silently_disable_joint_training(joint_harness, source, context):
    model = _joint_model(joint_harness).eval()
    with context():
        cached = (model.extract_vlm_features(**_prefix()) if source == "vlm"
                  else model.extract_motion_features(motion_img_embs=_images()))
    model.train()
    arguments = _forward_args()
    if source == "vlm":
        arguments.update(vlm_features=cached, motion_img_embs=_images())
    else:
        arguments.update(motion_features=cached)
    with pytest.raises(ValueError, match="grad|train|detach|graph|inference"):
        model(**arguments)


@pytest.mark.parametrize("sampler_name", SAMPLERS)
@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda', but CUDA is not available.*:UserWarning")
def test_trainable_sources_sample_without_grad_and_run_once_per_chunk(joint_harness, sampler_name):
    model = _joint_model(joint_harness).eval()
    arguments = dict(observations=[[object()]], states=torch.zeros(1, 1, 3), instructions=["1 2 3"],
                     num_inference_steps=3, traj2ds=None, motion_img_embs=_images(1), return_vlm_features=True)
    if sampler_name != "predict_action":
        arguments.update(prev_actions=torch.full((1, 4, 2), 0.3), inference_delay=1)
        if sampler_name == "predict_action_with_training_rtc_flow":
            arguments.update(max_delay=2)
        else:
            arguments.update(execution_horizon=2)

    result = getattr(model, sampler_name)(**arguments)

    assert torch.isfinite(result.action).all() and not result.action.requires_grad
    assert not result.vlm_features.tokens.requires_grad and not result.vlm_features.p_v.requires_grad
    assert not result.motion_features.tokens.requires_grad and not result.motion_features.p_m.requires_grad
    assert len(model.vlm_model.calls) == len(model.test_motion_expert.calls) == 1
    assert not model.vlm_model.calls[-1]["grad_enabled"]
    assert not model.vlm_model.calls[-1]["training"]
    assert not model.test_motion_expert.calls[-1]["grad_enabled"]
    assert not model.test_motion_expert.calls[-1]["training"]
    assert len(model.head_calls) == 3
    for call in model.head_calls:
        assert call["joint_attention_kwargs"]["vlm_features"] is result.vlm_features
        assert call["joint_attention_kwargs"]["motion_features"] is result.motion_features
    model.train()
    assert model.vlm_model.training and model.motion_conditioner.model.training
    assert model.vlm_model.get_input_embeddings().weight.requires_grad
    assert model.motion_conditioner.model.backbone.weight.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("tune_vlm,tune_motion", [(False, False), (True, False), (False, True)])
def test_independent_freezing_flags_remain_available(joint_harness, tune_vlm, tune_motion):
    model = _joint_model(joint_harness, tune_vlm=tune_vlm, tune_motion=tune_motion).train()
    assert model.vlm_model.training == tune_vlm
    assert model.motion_conditioner.model.training == tune_motion
    result = model(**_forward_args(), motion_img_embs=_images(), return_vlm_features=True)
    assert result.vlm_features.tokens.requires_grad == tune_vlm
    assert result.motion_features.tokens.requires_grad == tune_motion
    assert result.vlm_features.p_v.requires_grad == tune_vlm
    assert result.motion_features.p_m.requires_grad == tune_motion
    result.action.square().mean().backward()
    for parameter, trainable in ((model.vlm_model.get_input_embeddings().weight, tune_vlm),
                                 (model.motion_conditioner.model.backbone.weight, tune_motion)):
        assert parameter.requires_grad == trainable
        if trainable:
            _nonzero_finite(parameter.grad)
        else:
            assert parameter.grad is None


def test_global_no_grad_is_respected_without_permanently_freezing_trainable_sources(joint_harness):
    model = _joint_model(joint_harness).train()
    with torch.no_grad():
        result = model(**_forward_args(), motion_img_embs=_images(), return_vlm_features=True)
    assert not result.action.requires_grad
    assert not result.vlm_features.tokens.requires_grad and not result.motion_features.tokens.requires_grad
    assert model.vlm_model.get_input_embeddings().weight.requires_grad
    assert model.motion_conditioner.model.backbone.weight.requires_grad


def _optimizer_trainer(module_name, class_name):
    path = Path(__file__).parents[1] / "src/psi/trainers" / f"{module_name}.py"
    tree = ast.parse(path.read_text())
    source = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = {"apply_vlm_trainability", "_vlm_group_of", "vlm_trainable_components", "create_optimizers"}
    methods = [copy.deepcopy(node) for node in source.body
               if isinstance(node, ast.FunctionDef) and node.name in selected]
    assert {node.name for node in methods} == selected
    harness = ast.ClassDef(name="TrainerHarness", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             harness], type_ignores=[])
    namespace = dict(torch=torch, overwatch=SimpleNamespace(info=lambda *args, **kwargs: None))
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["TrainerHarness"]()


@pytest.fixture(params=[("posttrain", "PosttrainTrainer"), ("finetune", "FinetuneTrainer"),
                        ("sonic", "SonicTrainer")], ids=["posttrain", "finetune", "sonic"])
def actual_optimizer_trainer(request):
    return _optimizer_trainer(*request.param)


def test_real_trainer_optimizer_covers_each_trainable_parameter_once_with_source_learning_rates(
    joint_harness, actual_optimizer_trainer,
):
    model = _joint_model(joint_harness).train()
    trainer = actual_optimizer_trainer
    trainer.model = model
    trainer.model_cfg = SimpleNamespace(
        action_architecture="dual_source_v1", tune_vlm=True,
        tune_mm_llm=False, tune_mm_vision=False, tune_mm_mlp=False, gradient_checkpointing=False,
        lang_backbone_lr=1e-6, vision_tower_lr=1e-5, mm_projector_lr=1e-4, motion_expert_lr=1e-5,
    )
    trainer.cfg = SimpleNamespace(train=SimpleNamespace(
        learning_rate=1e-4, lr_scheduler_kwargs={"weight_decay": 0.0}, optimizer_foreach=False,
    ))
    assert trainer.apply_vlm_trainability() == {"lang_backbone", "vision_tower", "mm_projector"}
    assert model.vlm_model.model.language_model.norm.weight.requires_grad
    assert not model.vlm_model.lm_head.weight.requires_grad
    assert not model.motion_conditioner.model.output_process.weight.requires_grad
    trainer.create_optimizers()
    optimizer = trainer.optimizer
    grouped_ids = Counter(id(parameter) for group in optimizer.param_groups for parameter in group["params"])
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert set(grouped_ids) == trainable_ids
    assert set(grouped_ids.values()) == {1}
    assert optimizer.param_groups[0]["group_name"] == "action_header"
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    expected = {
        "action_header": (model.action_header.action_proj_out.linear.weight, 1e-4),
        "lang_backbone": (model.vlm_model.model.language_model.norm.weight, 1e-6),
        "vision_tower": (model.vlm_model.model.visual.patch_embed.weight, 1e-5),
        "mm_projector": (model.vlm_model.model.visual.merger.weight, 1e-4),
        "motion_expert": (model.motion_conditioner.model.backbone.weight, 1e-5),
    }
    before = {}
    assert set(groups) == set(expected)
    for name, (parameter, lr) in expected.items():
        assert groups[name]["lr"] == lr
        assert id(parameter) in {id(item) for item in groups[name]["params"]}
        before[name] = parameter.detach().clone()
    result = model(**_forward_args(), motion_img_embs=_images())
    result.action.square().mean().backward()
    optimizer.step()
    for name, (parameter, _) in expected.items():
        _nonzero_finite(parameter.grad)
        assert not torch.equal(parameter, before[name])


@pytest.mark.parametrize("tune_vlm", [True, False])
def test_legacy_sonic_keeps_single_vlm_group_and_historical_group_order(tune_vlm):
    trainer = _optimizer_trainer("sonic", "SonicTrainer")
    model = nn.Module()
    model.action_header = nn.Linear(4, 2)
    model.action_header.bias.requires_grad_(False)
    model.vlm_model = _JointVLM()
    model.extra = nn.Parameter(torch.ones(1), requires_grad=False)
    trainer.model = model
    trainer.model_cfg = SimpleNamespace(
        action_architecture="psi0_joint", tune_vlm=tune_vlm,
        tune_mm_llm=False, tune_mm_vision=True, tune_mm_mlp=False, gradient_checkpointing=False,
        lang_backbone_lr=1e-6, vision_tower_lr=1e-5, mm_projector_lr=1e-4, motion_expert_lr=1e-5,
    )
    trainer.cfg = SimpleNamespace(train=SimpleNamespace(
        learning_rate=1e-4, lr_scheduler_kwargs={"weight_decay": 0.0}, optimizer_foreach=False,
    ))
    components = trainer.apply_vlm_trainability()
    assert components == ({"lang_backbone", "vision_tower", "mm_projector"} if tune_vlm else set())
    assert not model.vlm_model.model.language_model.norm.weight.requires_grad
    trainer.create_optimizers()
    groups = trainer.optimizer.param_groups
    assert [group["group_name"] for group in groups] == (
        ["action_header", "vlm_model", "other"] if tune_vlm else ["action_header", "other"])
    for group in groups:
        assert group["lr"] == (1e-6 if group["group_name"] == "vlm_model" else 1e-4)
    grouped_ids = Counter(id(parameter) for group in groups for parameter in group["params"])
    # Sonic legacy optimizer layouts included frozen parameters too. Keep this
    # membership so optimizer checkpoints resume with identical group lengths.
    expected = {id(parameter) for parameter in model.action_header.parameters()} | {id(model.extra)}
    if tune_vlm:
        expected |= {id(parameter) for parameter in model.vlm_model.parameters()}
    assert set(grouped_ids) == expected
    assert set(grouped_ids.values()) == {1}


@pytest.mark.parametrize("tied_head", [False, True], ids=["untied_saved_head", "tied_omitted_alias"])
def test_actual_pretrained_loader_preserves_untied_weights_and_restores_only_a_tied_missing_alias(
    joint_harness, tmp_path, monkeypatch, tied_head,
):
    from safetensors.torch import save_file

    def make_vlm(config=None):
        vlm = _JointVLM()
        if tied_head:
            vlm.lm_head.weight = vlm.get_input_embeddings().weight
        return vlm

    original = _joint_model(joint_harness, vlm_model=make_vlm()).eval()
    if not tied_head:
        with torch.no_grad():
            original.vlm_model.lm_head.weight.fill_(7)
        assert not torch.equal(original.vlm_model.lm_head.weight,
                               original.vlm_model.get_input_embeddings().weight)
    state = dict(original.state_dict())
    if tied_head:
        state.pop("vlm_model.lm_head.weight")  # safetensors removes a duplicated tied alias.
    checkpoint = tmp_path / "checkpoints/ckpt_1/model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    save_file(state, str(checkpoint))
    config = original.test_model_cfg
    config.model_name_or_path = "test-egodex-vlm"
    config.noise_scheduler = "flow"
    config.train_diffusion_steps = 1000

    # Execute the complete real classmethod body, replacing optional pretrained
    # factories and the scheduler import, while reading an actual safetensors file.
    source_path = Path(__file__).parents[1] / "src/psi/models/psi0.py"
    tree = ast.parse(source_path.read_text())
    source_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Psi0Model")
    method = copy.deepcopy(next(node for node in source_class.body
                                if isinstance(node, ast.FunctionDef) and node.name == "from_pretrained"))
    method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             method], type_ignores=[])

    class Scheduler(_Scheduler):
        def __init__(self, **kwargs):
            pass

    scheduler_name = "diffusers.schedulers.scheduling_flow_match_euler_discrete"
    for name in ("diffusers", "diffusers.schedulers", scheduler_name):
        fake_module = ModuleType(name)
        fake_module.__path__ = []
        monkeypatch.setitem(sys.modules, name, fake_module)
    sys.modules[scheduler_name].FlowMatchEulerDiscreteScheduler = Scheduler
    utilities = ModuleType("transformers.utils")
    utilities.is_flash_attn_2_available = lambda: False
    monkeypatch.setitem(sys.modules, "transformers.utils", utilities)
    namespace = dict(
        os=os, torch=torch, Qwen3VLForConditionalGeneration=make_vlm,
        AutoConfig=SimpleNamespace(from_pretrained=lambda path: SimpleNamespace(
            vision_config=SimpleNamespace(), text_config=SimpleNamespace())),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda path: _Processor()),
        Psi0Model=lambda model_cfg, vlm_model: joint_harness(
            model_cfg, vlm_model, motion_expert=_JointMotionExpert()),
        overwatch=SimpleNamespace(info=lambda *args, **kwargs: None),
    )
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)
    restored = namespace["from_pretrained"](
        joint_harness, str(tmp_path), 1, SimpleNamespace(model=config), torch.device("cpu"))

    torch.testing.assert_close(restored.vlm_model.lm_head.weight,
                               original.vlm_model.lm_head.weight.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(restored.vlm_model.get_input_embeddings().weight,
                               original.vlm_model.get_input_embeddings().weight.to(torch.bfloat16), rtol=0, atol=0)
    assert (restored.vlm_model.lm_head.weight is restored.vlm_model.get_input_embeddings().weight) == tied_head
    torch.testing.assert_close(restored.action_header.action_proj_out.linear.weight,
                               original.action_header.action_proj_out.linear.weight, rtol=0, atol=0)
    torch.testing.assert_close(restored.motion_conditioner.model.backbone.weight,
                               original.motion_conditioner.model.backbone.weight, rtol=0, atol=0)
