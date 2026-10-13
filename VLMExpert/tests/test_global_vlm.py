import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, PropertyMock, patch

import torch
import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    Qwen2TokenizerFast, Qwen2VLImageProcessor,
    Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Qwen3VLVideoProcessor,
)

from psi.models.global_vlm import (
    build_vlm_inputs, extract_global_features, initialize_vlm_tokens,
    load_global_backbone, select_global_features, validate_vlm_tokens,
)
from psi.tokenizer.fast_action_tokenizer import FastActionTokenizer
from psi.trainers.pretrain_checkpoint import (
    resolve_pretrain_checkpoint, save_checkpoint_metadata, validate_checkpoint_metadata,
)
from psi.trainers.qwen3vl_mixin import PaddedCollatorForActionPrediction


CHAT_TEMPLATE = (
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% for item in message['content'] %}"
    "{% if item['type'] == 'image' %}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
    "{% else %}{{ item['text'] }}{% endif %}{% endfor %}"
    "{{ '<|im_end|>\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def make_model_processor():
    special = ["<unk>", "<pad>", "<|im_start|>", "<|im_end|>",
               "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
    vocab = {token: index for index, token in enumerate(
        special + ["user", "assistant", "pick", "place", "the", "red", "blue", "object"]
    )}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>",
        eos_token="<|im_end|>", additional_special_tokens=special[2:],
    )
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=16, min_pixels=1024, max_pixels=4096),
        tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor(), chat_template=CHAT_TEMPLATE,
    )
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=192, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                         head_dim=16, max_position_embeddings=512,
                         rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]},
                         pad_token_id=vocab["<pad>"], eos_token_id=vocab["<|im_end|>"],
                         tie_word_embeddings=True),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                           out_hidden_size=32, num_position_embeddings=16,
                           patch_size=16, deepstack_visual_indexes=[0]),
        image_token_id=vocab["<|image_pad|>"], video_token_id=vocab["<|video_pad|>"],
        vision_start_token_id=vocab["<|vision_start|>"],
        vision_end_token_id=vocab["<|vision_end|>"], tie_word_embeddings=True,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3VLForConditionalGeneration(config).eval()
    return model, processor


class GlobalVLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.model, self.processor = make_model_processor()
        self.token_id = initialize_vlm_tokens(
            self.model, self.processor, bins=4, use_global_token=True, model_max_length=256,
        )
        self.views = [[Image.new("RGB", (32, 32), "red")],
                      [Image.new("RGB", (64, 32), "blue"), Image.new("RGB", (32, 32), "red")]]
        self.instructions = ["pick", "place the blue object"]
        self.answers = ["<|a_0|><|a_1|>", "<|a_2|><|a_3|>"]

    def inputs(self, answers=True):
        return build_vlm_inputs(
            self.processor, self.views, self.instructions,
            global_token_id=self.token_id, action_texts=self.answers if answers else None,
        )

    def test_vocabulary_is_idempotent_and_preserves_special_tokens(self):
        weight = self.model.get_input_embeddings().weight
        initial = weight.detach().clone()
        self.assertIs(weight, self.model.get_output_embeddings().weight)
        self.assertNotEqual(self.token_id, weight.shape[0] - 1)
        FastActionTokenizer.register_tokens(self.processor.tokenizer, 4)
        self.assertIn(self.token_id, self.processor.tokenizer.all_special_ids)
        initialize_vlm_tokens(self.model, self.processor, bins=4, use_global_token=True,
                              model_max_length=256, require_global_checkpoint=True)
        torch.testing.assert_close(initial, self.model.get_input_embeddings().weight, rtol=0, atol=0)
        self.assertEqual(validate_vlm_tokens(self.model, self.processor, require_global=True), self.token_id)
        for token in ("<|image_pad|>", "<|action_start|>", "<|action_end|>"):
            self.assertIn(self.processor.tokenizer.convert_tokens_to_ids(token), self.processor.tokenizer.all_special_ids)

    def test_initialization_does_not_shrink_or_reinitialize_reserved_rows(self):
        model, processor = make_model_processor()
        before = model.get_input_embeddings().weight.detach().clone()
        ordinary_ids = [i for i in range(len(processor.tokenizer)) if i not in processor.tokenizer.all_special_ids]
        expected_mean = before[ordinary_ids].mean(0)
        token_id = initialize_vlm_tokens(model, processor, bins=4, use_global_token=True)
        torch.testing.assert_close(model.get_input_embeddings().weight[token_id], expected_mean)
        self.assertEqual(model.get_input_embeddings().weight.shape, before.shape)
        torch.testing.assert_close(model.get_input_embeddings().weight[:len(ordinary_ids)], before[:len(ordinary_ids)])

    def test_baseline_and_gradient_checkpointing_remain_trainable(self):
        model, processor = make_model_processor()
        initialize_vlm_tokens(model, processor, bins=4, use_global_token=False, model_max_length=256)
        self.assertNotIn("<GLOBAL>", processor.tokenizer.get_vocab())
        self.assertIsNone(validate_vlm_tokens(model, processor))
        batch = build_vlm_inputs(processor, self.views, self.instructions, action_texts=self.answers)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        loss = model(**batch, use_cache=False).loss
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.get_input_embeddings().weight.grad.norm().item(), 0)

    def test_prefix_labels_and_ragged_multiview_collation(self):
        full, prefix = self.inputs(), self.inputs(False)
        self.assertEqual(full["image_grid_thw"].shape, (3, 3))
        self.assertEqual(full["pixel_values"].ndim, 2)
        for i in range(2):
            position = full["input_ids"][i].eq(self.token_id).nonzero().item()
            prefix_length = prefix["attention_mask"][i].sum().item()
            self.assertEqual(position + 1, prefix_length)
            torch.testing.assert_close(full["input_ids"][i, :prefix_length], prefix["input_ids"][i, :prefix_length])
            self.assertTrue(full["labels"][i, :position + 1].eq(-100).all())
            self.assertEqual(full["labels"][i, position + 1], self.model.config.fast_token_ids[i * 2])
        self.assertTrue(full["labels"][full["attention_mask"] == 0].eq(-100).all())
        samples = [build_vlm_inputs(self.processor, [views], [instruction],
                                    global_token_id=self.token_id, action_texts=[answer])
                   for views, instruction, answer in zip(self.views, self.instructions, self.answers)]
        collator = PaddedCollatorForActionPrediction(256, self.processor.tokenizer.pad_token_id,
                                                    global_token_id=self.token_id)
        collated = collator(samples)
        for key in full:
            torch.testing.assert_close(collated[key], full[key])

    def test_first_action_loss_trains_global_input_and_updates_embedding(self):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.train()
        inputs = self.inputs()
        captured = []

        def capture(module, args, output):
            output.retain_grad()
            captured.append(output)

        handle = self.model.get_input_embeddings().register_forward_hook(capture)
        output = self.model(**inputs, use_cache=False)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        handle.remove()
        mask = inputs["input_ids"].eq(self.token_id)
        self.assertGreater(captured[0].grad[mask].norm().item(), 0)
        before = self.model.get_input_embeddings().weight[self.token_id].detach().clone()
        torch.optim.SGD(self.model.parameters(), lr=0.01).step()
        self.assertFalse(torch.equal(before, self.model.get_input_embeddings().weight[self.token_id]))

    def test_causal_global_matches_prefix_and_ignores_answer_changes(self):
        full = self.inputs()
        with torch.no_grad():
            output = self.model(**{k: v for k, v in full.items() if k != "labels"},
                                output_hidden_states=True, logits_to_keep=1, use_cache=False)
        selected = select_global_features(output.hidden_states[-1], full["input_ids"],
                                          full["attention_mask"], self.token_id)
        prefix = extract_global_features(self.model, self.processor, self.views, self.instructions)
        torch.testing.assert_close(selected.global_hidden, prefix.global_hidden, atol=1e-5, rtol=1e-5)
        for i, position in enumerate(selected.global_index):
            self.assertFalse(selected.ordinary_mask[i, position:].any())
        self.answers = list(reversed(self.answers))
        changed = self.inputs()
        with torch.no_grad():
            output = self.model(**{k: v for k, v in changed.items() if k != "labels"},
                                output_hidden_states=True, logits_to_keep=1, use_cache=False)
        swapped = select_global_features(output.hidden_states[-1], changed["input_ids"],
                                         changed["attention_mask"], self.token_id)
        torch.testing.assert_close(selected.global_hidden, swapped.global_hidden, atol=1e-5, rtol=1e-5)

    def test_truncation_only_shortens_instruction_and_never_silently_cuts_answer(self):
        short = build_vlm_inputs(self.processor, self.views[:1], ["pick"],
                                 global_token_id=self.token_id, action_texts=self.answers[:1])
        limit = short["input_ids"].shape[1] + 2
        long = build_vlm_inputs(self.processor, self.views[:1], ["pick " * 100],
                                global_token_id=self.token_id, action_texts=self.answers[:1],
                                model_max_length=limit, max_instruction_tokens=3)
        prefix = build_vlm_inputs(self.processor, self.views[:1], ["pick " * 100],
                                  global_token_id=self.token_id, model_max_length=limit,
                                  max_instruction_tokens=3)
        torch.testing.assert_close(long["input_ids"][:, :prefix["input_ids"].shape[1]], prefix["input_ids"])
        torch.testing.assert_close(long["labels"][long["labels"] != -100], short["labels"][short["labels"] != -100])
        self.assertLessEqual(long["input_ids"].shape[1], limit)
        with self.assertRaisesRegex(ValueError, "not truncated"):
            build_vlm_inputs(self.processor, self.views[:1], [""], global_token_id=self.token_id,
                              action_texts=self.answers[:1], model_max_length=2)
        with self.assertRaisesRegex(ValueError, "refusing to truncate"):
            PaddedCollatorForActionPrediction(2, self.processor.tokenizer.pad_token_id)([short])

    def test_missing_duplicate_and_mismatched_global_are_rejected(self):
        prefix = self.inputs(False)
        ids = prefix["input_ids"].clone()
        ids[ids == self.token_id] = self.processor.tokenizer.pad_token_id
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_global_features(torch.zeros(*ids.shape, 32), ids, prefix["attention_mask"], self.token_id)
        with self.assertRaisesRegex(ValueError, "must not occur"):
            build_vlm_inputs(self.processor, self.views[:1], ["pick <GLOBAL>"], global_token_id=self.token_id)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            initialize_vlm_tokens(self.model, self.processor, bins=4, use_global_token=False)
        self.model.config.global_token_id += 1
        with self.assertRaisesRegex(ValueError, "metadata"):
            validate_vlm_tokens(self.model, self.processor, require_global=True)

    def test_checkpoint_export_reload_and_he_initialization(self):
        source_features = extract_global_features(self.model, self.processor, self.views, self.instructions)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoints" / "ckpt_200000"
            self.model.save_pretrained(checkpoint)
            save_checkpoint_metadata(checkpoint, self.model, self.processor,
                                     {"stage": "egodex", "completed_steps": 200000})
            self.assertEqual(resolve_pretrain_checkpoint(root), checkpoint)
            module_path = Path(__file__).resolve().parents[1] / "scripts/save_pretrain_qwen3vl_backbone.py"
            spec = importlib.util.spec_from_file_location("export_global", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            exported = root / "exported"
            module.export_pretrained(checkpoint, exported, dtype=torch.float32, max_shard_size="30KB")
            loaded, processor = load_global_backbone(exported)
            initialize_vlm_tokens(loaded, processor, bins=4, use_global_token=True,
                                  model_max_length=256, require_global_checkpoint=True)
            self.assertEqual(loaded.config.global_token_id, self.token_id)
            self.assertEqual(loaded.get_input_embeddings().weight.shape, self.model.get_input_embeddings().weight.shape)
            torch.testing.assert_close(loaded.get_input_embeddings().weight, self.model.get_input_embeddings().weight)
            features = extract_global_features(loaded, processor, self.views, self.instructions)
            torch.testing.assert_close(source_features.global_hidden, features.global_hidden, rtol=1e-5, atol=1e-6)
            with self.assertRaisesRegex(ValueError, "same pretraining stage"):
                validate_checkpoint_metadata(exported, loaded, processor, expected_stage="he")
            processor.tokenizer.add_tokens(["unexpected"])
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                validate_checkpoint_metadata(exported, loaded, processor)

    def test_stage_resume_uses_completed_updates_without_off_by_one(self):
        from psi.trainers.pretrain import PretrainTrainer

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "ckpt_2"
            save_checkpoint_metadata(checkpoint, self.model, self.processor,
                                     {"stage": "egodex", "completed_steps": 2})
            trainer = object.__new__(PretrainTrainer)
            trainer.cfg = SimpleNamespace(
                model=SimpleNamespace(pretraining_stage="egodex"),
                train=SimpleNamespace(max_training_steps=10),
            )
            trainer.model = self.model
            trainer.vlm_processor = self.processor
            trainer._resume_checkpoint = checkpoint
            trainer.accelerator = Mock()
            trainer.accelerator.unwrap_model.return_value = self.model
            step, path = trainer.resume_from_checkpoint()
            self.assertEqual(step, 2)
            trainer.accelerator.load_state.assert_called_once_with(str(checkpoint))
            trainer._resume_checkpoint = None
            self.assertEqual(trainer.resume_from_checkpoint(), (0, None))

    def test_accelerate_accumulates_updates_and_saves_reloadable_checkpoint(self):
        from accelerate import Accelerator
        from psi.config.model_qwen3vl import Qwen3VLModelConfig
        from psi.config.tokenizer import FastActionTokenizerConfig
        from psi.trainers.pretrain import PretrainTrainer

        trainer = object.__new__(PretrainTrainer)
        trainer.cfg = SimpleNamespace(
            model=Qwen3VLModelConfig(action_tokenizer=FastActionTokenizerConfig(bins=4),
                                    use_global_token=True, global_token_id=self.token_id),
            train=SimpleNamespace(max_grad_norm=1.0, data_parallel="ddp", train_batch_size=2,
                                  gradient_accumulation_steps=2, max_checkpoints_to_keep=1,
                                  max_training_steps=10),
            log=SimpleNamespace(log_freq=1, report_to=None),
            model_dump=lambda **kwargs: {},
        )
        trainer.accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
        trainer.device = torch.device("cpu")
        trainer.vlm_processor = self.processor
        trainer._model_source = "tiny-qwen"
        trainer.global_step = 0
        trainer._grad_norm_vlm = None
        trainer._previous_samples_seen = 0
        trainer._local_samples_seen = 0
        trainer.maxmin = SimpleNamespace(denormalize=lambda values: values)
        start = self.model.config.fast_token_ids[0]
        trainer.action_tokenizer = SimpleNamespace(
            n_bins=4, action_token_begin_idx=start,
            decode_token_ids_to_actions=lambda ids: np.asarray(ids, dtype=np.float32) - start,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        trainer.model, trainer.optimizer, trainer.lr_scheduler = trainer.accelerator.prepare(
            self.model, optimizer, scheduler,
        )
        before = self.model.get_input_embeddings().weight.detach().clone()
        sync, metrics = trainer.training_step(self.inputs())
        self.assertFalse(sync)
        torch.testing.assert_close(before, self.model.get_input_embeddings().weight, rtol=0, atol=0)
        sync, metrics = trainer.training_step(self.inputs())
        self.assertTrue(sync)
        self.assertEqual(trainer._local_samples_seen, 4)
        self.assertGreater(metrics["global_input_grad_norm"], 0)
        self.assertFalse(torch.equal(before, self.model.get_input_embeddings().weight))
        with TemporaryDirectory() as directory:
            trainer._previous_samples_seen = 7
            with patch.object(PretrainTrainer, "project_dir", new_callable=PropertyMock, return_value=directory):
                path = Path(trainer.save_checkpoint(1))
            metadata = validate_checkpoint_metadata(path, self.model, self.processor)
            self.assertEqual(metadata["samples_seen_in_stage"], 11)
            self.assertEqual(metadata["global_batch_size"], 4)
            loaded, processor = load_global_backbone(path)
            torch.testing.assert_close(loaded.get_input_embeddings().weight, self.model.get_input_embeddings().weight)
            expected = self.model.get_input_embeddings().weight.detach().clone()
            with torch.no_grad():
                self.model.get_input_embeddings().weight.add_(1)
            trainer._resume_checkpoint = path
            self.assertEqual(trainer.resume_from_checkpoint(), (1, str(path)))
            self.assertEqual(trainer._previous_samples_seen, 11)
            self.assertEqual(trainer._local_samples_seen, 0)
            torch.testing.assert_close(expected, self.model.get_input_embeddings().weight)

    def test_he_trainer_loads_global_checkpoint_without_vocab_growth(self):
        from psi.config.model_qwen3vl import Qwen3VLModelConfig
        from psi.config.tokenizer import FastActionTokenizerConfig
        from psi.trainers.pretrain import PretrainTrainer

        with TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            save_checkpoint_metadata(directory, self.model, self.processor,
                                     {"stage": "egodex", "completed_steps": 200000})
            trainer = object.__new__(PretrainTrainer)
            trainer.Da = 48
            trainer.cfg = SimpleNamespace(
                model=Qwen3VLModelConfig(
                    model_name_or_path=directory, action_tokenizer=FastActionTokenizerConfig(bins=4),
                    use_global_token=True, require_global_checkpoint=True, pretraining_stage="he",
                    model_max_length=256, attn_implementation="sdpa", tune_mm_vision=True,
                ),
                train=SimpleNamespace(resume_from_checkpoint=None, lora=False),
                data=SimpleNamespace(chunk_size=1),
            )
            with patch("psi.trainers.pretrain.FastActionTokenizer") as fast:
                fast.return_value = SimpleNamespace(n_bins=4)
                trainer.init_models()
            self.assertIsNone(trainer._resume_checkpoint)
            self.assertEqual(trainer.model.config.global_token_id, self.token_id)
            self.assertEqual(trainer.model.config.text_config.vocab_size, self.model.config.text_config.vocab_size)
            torch.testing.assert_close(trainer.model.get_input_embeddings().weight.float(),
                                        self.model.get_input_embeddings().weight.to(torch.bfloat16).float())

    def test_offline_evaluation_and_linear_probes(self):
        module_path = Path(__file__).resolve().parents[1] / "scripts/eval_global_vlm.py"
        spec = importlib.util.spec_from_file_location("evaluate_global", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        action_tokenizer = Mock()
        action_tokenizer.side_effect = lambda target: "<|a_0|><|a_1|>"
        action_tokenizer.decode_token_ids_to_actions.return_value = np.zeros((1, 1, 2), dtype=np.float32)
        records = [{"id": str(i), "images": views, "instruction": instruction,
                    "normalized_actions": [[0.0, 1.0]]}
                   for i, (views, instruction) in enumerate(zip(self.views, self.instructions))]
        metrics, features = module.evaluate_samples(self.model, self.processor, action_tokenizer,
                                                    records, max_new_tokens=2)
        self.assertTrue(np.isfinite(metrics["teacher_forcing_cross_entropy"]))
        self.assertTrue(metrics["generation_enabled"])
        self.assertEqual(features["global"].shape, (2, 32))
        self.assertLess(metrics["global_prefix_max_abs_error"], 1e-5)
        train = {**features, "ids": ["train-0", "train-1"]}
        scores = module.probe_features(train, features)
        self.assertIn("global_swapped_mae", scores)
        with self.assertRaisesRegex(ValueError, "overlap"):
            module.probe_features(features, features)


if __name__ == "__main__":
    unittest.main()
