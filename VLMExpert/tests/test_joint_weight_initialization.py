"""CPU joint-initialization contracts using tiny modules and real safetensors.

The production strict loader, alias restoration, posttrain initialization and
optimizer grouping execute here. Tiny pretrained factories replace the 3B
networks; this is not a real-checkpoint or GPU training acceptance test.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
from safetensors.torch import save_file

from psi.config.config import TrainConfig
from psi.config.model_psi0 import Psi0ModelConfig
import psi.models.psi0 as model_module
from psi.trainers.posttrain import PosttrainTrainer


class TinyVLM(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or SimpleNamespace(text_config=SimpleNamespace(hidden_size=2))
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(4, 2)
        self.model.language_model.norm = nn.LayerNorm(2)
        self.model.visual = nn.Module()
        self.model.visual.patch_embed = nn.Linear(2, 2)
        self.model.visual.merger = nn.Linear(2, 2)
        self.lm_head = nn.Linear(2, 4, bias=False)

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens


class TinyMotion(nn.Module):
    def __init__(self, trainable):
        super().__init__()
        self.trainable = trainable
        self.model = nn.Module()
        self.model.backbone = nn.Linear(2, 2)
        self.model.pos_enc = nn.Module()
        self.model.pos_enc.register_buffer("pe", torch.tensor([.1234567]))
        self.model.embed_timestep = nn.Module()
        self.model.embed_timestep.pos_enc = self.model.pos_enc
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.model.train(mode if self.trainable else False)
        self.model.requires_grad_(self.trainable)
        return self


class TinyAction(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.register_buffer("_architecture_signature", torch.tensor([1, 2]))


class TinyJoint(nn.Module):
    def __init__(self, model_cfg, vlm_model):
        super().__init__()
        self.tune_vlm = model_cfg.tune_vlm
        self.vlm_model = vlm_model
        self.motion_conditioner = TinyMotion(model_cfg.tune_motion)
        self.action_header = TinyAction()

    def train(self, mode=True):
        super().train(mode)
        if not self.tune_vlm:
            self.vlm_model.eval()
        return self

    def forward(self):
        source = self.vlm_model.get_input_embeddings()(torch.tensor([0])).float()
        source = source + self.motion_conditioner.model.backbone(torch.ones(1, 2))
        return self.action_header.linear(source)


class TestJointWeightInitialization(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.qwen = self.root / "qwen"
        self.qwen.mkdir()
        (self.qwen / "config.json").write_text("{}")
        self.motion = self.root / "motion.ckpt"
        self.motion.write_bytes(b"construction metadata path fixture; tiny factory bypasses expert loading")
        self.checkpoint = self.root / "run/checkpoints/ckpt_28000/model.safetensors"
        self.checkpoint.parent.mkdir(parents=True)
        source_cfg = self.config()
        self.original = TinyJoint(source_cfg, TinyVLM())
        with torch.no_grad():
            for index, parameter in enumerate(self.original.parameters()):
                parameter.fill_(.1234567 + index / 100)
        self.write_checkpoint()

    def config(self, **overrides):
        values = dict(
            action_architecture="dual_source_v1", n_conditions=0,
            model_name_or_path=str(self.qwen), motion_checkpoint=str(self.motion),
            tune_vlm=False, tune_motion=False,
        )
        values.update(overrides)
        return Psi0ModelConfig(**values)

    def write_checkpoint(self, *, missing=None):
        values = {key: value.clone() for key, value in self.original.state_dict().items()}
        # The real loader must recover this proven shared alias, never fabricate
        # an independent weight missing from a checkpoint.
        values.pop("motion_conditioner.model.embed_timestep.pos_enc.pe")
        if missing is not None:
            values.pop(missing)
        save_file(values, str(self.checkpoint))

    def trainer(self, *, train_overrides=None, **model_overrides):
        model_overrides.setdefault("pretrained_joint_checkpoint", str(self.checkpoint))
        trainer = PosttrainTrainer.__new__(PosttrainTrainer)
        trainer.cfg = SimpleNamespace(
            model=self.config(**model_overrides),
            train=TrainConfig(**(train_overrides or {})),
        )
        return trainer

    def factories(self):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(model_module, "Psi0Model", TinyJoint))
        stack.enter_context(mock.patch.object(model_module, "Qwen3VLForConditionalGeneration", TinyVLM))
        stack.enter_context(mock.patch.object(model_module.AutoConfig, "from_pretrained", return_value=SimpleNamespace(
            vision_config=SimpleNamespace(), text_config=SimpleNamespace(hidden_size=2))))
        stack.enter_context(mock.patch.object(model_module.AutoProcessor, "from_pretrained", return_value=SimpleNamespace(
            tokenizer=SimpleNamespace(model_max_length=128, pad_token_id=0))))
        return stack

    def test_old_config_default_remains_none_and_legacy_init_is_unchanged(self):
        self.assertIsNone(Psi0ModelConfig().pretrained_joint_checkpoint)
        trainer = self.trainer(pretrained_joint_checkpoint=None)
        vlm = TinyVLM()
        trainer.init_qwen3vl_models = mock.Mock(return_value=vlm)
        with mock.patch("psi.trainers.posttrain.Psi0Model", wraps=TinyJoint) as factory:
            trainer.init_models()
            factory.assert_called_once_with(model_cfg=trainer.model_cfg, vlm_model=vlm)
        trainer.init_qwen3vl_models.assert_called_once()

    def test_actual_strict_loader_preserves_header_motion_aliases_and_uses_local_vlm_config_only(self):
        trainer = self.trainer()
        trainer.init_qwen3vl_models = mock.Mock(side_effect=AssertionError("Must not load initial HF VLM weights"))
        with self.factories():
            trainer.init_models()
        trainer.init_qwen3vl_models.assert_not_called()
        self.assertEqual(next(trainer.model.vlm_model.parameters()).dtype, torch.bfloat16)
        self.assertFalse(any(parameter.requires_grad for parameter in trainer.model.vlm_model.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in trainer.model.motion_conditioner.parameters()))
        for prefix in ("action_header.", "motion_conditioner."):
            for name, expected in self.original.state_dict().items():
                if name.startswith(prefix):
                    torch.testing.assert_close(trainer.model.state_dict()[name], expected, rtol=0, atol=0)
        self.assertIs(trainer.model.motion_conditioner.model.pos_enc,
                      trainer.model.motion_conditioner.model.embed_timestep.pos_enc)
        self.assertFalse(hasattr(trainer, "optimizer"))
        self.assertFalse(hasattr(trainer, "lr_scheduler"))
        self.assertEqual(trainer.resume_from_checkpoint(), (0, None))

    def test_action_only_new_optimizer_updates_only_action_and_keeps_sources_frozen(self):
        trainer = self.trainer(train_overrides={"learning_rate": 1e-3, "optimizer_foreach": False})
        with self.factories():
            trainer.init_models()
        trainer.model.train()
        self.assertFalse(trainer.model.vlm_model.training)
        self.assertFalse(trainer.model.motion_conditioner.model.training)
        self.assertTrue(trainer.model.action_header.training)
        trainer.create_optimizer_and_scheduler(num_training_steps=4)
        self.assertEqual([group["group_name"] for group in trainer.optimizer.param_groups], ["action_header"])
        self.assertFalse(trainer.optimizer.state)
        self.assertLessEqual(trainer.lr_scheduler.last_epoch, 0)
        frozen = {name: value.clone() for name, value in trainer.model.state_dict().items()
                  if not name.startswith("action_header.")}
        before = trainer.model.action_header.linear.weight.detach().clone()
        for _ in range(2):
            loss = trainer.model().square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                for parameter in trainer.model.action_header.parameters()))
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
        self.assertFalse(torch.equal(before, trainer.model.action_header.linear.weight))
        for name, expected in frozen.items():
            torch.testing.assert_close(trainer.model.state_dict()[name], expected, rtol=0, atol=0)

    def test_trainable_vlm_keeps_original_fp32_values_without_bf16_roundtrip(self):
        trainer = self.trainer()
        trainer.model_cfg.tune_vlm = True
        with self.factories():
            trainer.init_models()
        self.assertEqual(next(trainer.model.vlm_model.parameters()).dtype, torch.float32)
        for name, expected in self.original.vlm_model.state_dict().items():
            torch.testing.assert_close(trainer.model.vlm_model.state_dict()[name], expected, rtol=0, atol=0)

    def test_missing_independent_weight_is_never_silently_initialized(self):
        for missing in ("action_header.linear.bias", "motion_conditioner.model.backbone.weight",
                        "vlm_model.model.language_model.norm.weight"):
            with self.subTest(missing=missing):
                self.write_checkpoint(missing=missing)
                with self.factories(), self.assertRaisesRegex(RuntimeError, "Missing key"):
                    self.trainer().init_models()

    def test_ambiguous_resume_lora_and_partial_header_initialization_rejected_before_load(self):
        for train_values, model_values, message in (
            ({"resume_from_checkpoint": "/previous/run"}, {}, "mutually exclusive"),
            ({"lora": True}, {}, "LoRA"),
            ({}, {"pretrained_action_header_path": "/header"}, "separate action-header"),
        ):
            with self.subTest(message=message):
                trainer = self.trainer(train_overrides=train_values, **model_values)
                with mock.patch("psi.trainers.posttrain.Psi0Model.from_pretrained") as load:
                    with self.assertRaisesRegex(ValueError, message):
                        trainer.init_models()
                    load.assert_not_called()

    def test_nonlocal_inputs_and_nonstandard_checkpoint_layout_fail_before_model_allocation(self):
        for overrides, message in (
            ({"pretrained_joint_checkpoint": "model.safetensors"}, "absolute local file"),
            ({"pretrained_joint_checkpoint": str(self.motion)}, "must use"),
            ({"model_name_or_path": "Qwen/Qwen3-VL-2B-Instruct"}, "absolute local Qwen"),
            ({"motion_checkpoint": "/nonexistent/motion.ckpt"}, "local Motion"),
        ):
            with self.subTest(overrides=overrides):
                trainer = self.trainer(**overrides)
                with mock.patch("psi.trainers.posttrain.Psi0Model.from_pretrained") as load:
                    with self.assertRaisesRegex(ValueError, message):
                        trainer.init_models()
                    load.assert_not_called()
        with self.assertRaisesRegex(ValueError, "requires action_architecture"):
            Psi0ModelConfig(pretrained_joint_checkpoint=str(self.checkpoint))


if __name__ == "__main__":
    unittest.main()
