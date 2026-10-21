"""CPU contracts for the standalone local adaptation launcher; no model/GPU loading."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import random
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

LAUNCHER = Path(__file__).resolve().parents[1] / "finetune_carrybox.py"
spec = importlib.util.spec_from_file_location("finetune_carrybox", LAUNCHER)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherContracts(unittest.TestCase):
    def test_training_quality_gate_rejects_before_any_cuda_or_model_import(self):
        with mock.patch("torch.cuda.set_device", side_effect=AssertionError("No GPU")):
            with self.assertRaisesRegex(RuntimeError, "training is blocked"):
                launcher.train_local(None, [], None, None)

    def test_cli_gate_rejects_missing_or_failed_reports_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            failed = output / "training_result.json"
            for existing_failure in (False, True):
                if existing_failure:
                    failed.write_text('{"status":"failed","completed_updates":0}\n')
                before = failed.read_bytes() if failed.exists() else None
                argv = [str(LAUNCHER), "--base-run", "/missing/source", "--dataset-root", "/missing/data",
                        "--dino-cache-root", "/missing/cache", "--output", str(output), "--train"]
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(launcher, "prepare", side_effect=AssertionError("Must block before preparation")):
                    with self.assertRaisesRegex(RuntimeError, "deprecated measured-mode0"):
                        launcher.main()
                self.assertEqual(before, failed.read_bytes() if failed.exists() else None)

    def test_tensor_hash_zero_dimensional_bfloat16_and_other_dtypes(self):
        for dtype in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                      torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128):
            with self.subTest(dtype=dtype):
                scalar = torch.tensor(1, dtype=dtype)
                scalar_signature = launcher.tensor_signature(scalar)
                self.assertEqual(scalar_signature, launcher.tensor_signature(scalar.clone()))
                self.assertNotEqual(scalar_signature, launcher.tensor_signature(scalar.reshape(1)))
        array = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).T
        self.assertEqual(launcher.tensor_signature(array), launcher.tensor_signature(array.contiguous()))

    def test_rewrite_removes_complete_groups_and_negative_booleans(self):
        argv = ["scripts/train.py", launcher.SCHEMA, "--model.tune-vlm", "--train.early-checkpoint-steps", "10", "20",
                "--model.no-rtc", "--data.train_repo_ids=old", "--seed=7"]
        result = launcher.rewrite_argv(argv, {"model.tune_vlm": False, "train.early_checkpoint_steps": [],
                                            "data.train_repo_ids": ["train"]})
        self.assertNotIn("10", result)
        self.assertNotIn("--model.tune-vlm", result)
        self.assertIn("--model.no-tune-vlm", result)
        self.assertIn("--model.no-rtc", result)
        self.assertEqual(result.count("--train.early-checkpoint-steps"), 1)
        self.assertIn("--seed=7", result)

    def test_fixed_windows_equal_five_episodes_and_deterministic(self):
        candidates = {episode: list(range(30, 101)) for episode in range(5)}
        selected = launcher.uniform_windows(candidates)
        self.assertEqual(selected, launcher.uniform_windows(candidates))
        self.assertEqual(len(selected), 20)
        for episode in candidates:
            self.assertEqual([row for ep, row in selected if ep == episode], [30, 53, 76, 100])
        with self.assertRaises(ValueError):
            launcher.uniform_windows({0: list(range(100))})
        with self.assertRaises(ValueError):
            launcher.uniform_windows(candidates, 21)

    def test_future_filter_rejects_tail_and_intermediate_gap(self):
        frames = np.arange(90)
        timestamps = frames / 30
        source_frames = frames * 50 // 30
        source_episodes = np.repeat(42, 90)
        self.assertTrue(launcher.future_is_complete(30, 90, frames, timestamps, source_frames, source_episodes))
        self.assertTrue(launcher.future_is_complete(60, 90, frames, timestamps, source_frames, source_episodes))
        self.assertFalse(launcher.future_is_complete(61, 90, frames, timestamps, source_frames, source_episodes))
        source_frames[45] += 1
        self.assertFalse(launcher.future_is_complete(30, 90, frames, timestamps, source_frames, source_episodes))

    def test_future_float32_timestamps_after_sixteen_seconds(self):
        frames = np.arange(1800)
        source_frames = frames * 50 // 30
        timestamps = (frames / 30).astype(np.float32)
        source_episodes = np.repeat(42, len(frames))
        self.assertTrue(launcher.future_is_complete(600, 1800, frames, timestamps, source_frames, source_episodes))
        self.assertTrue(launcher.future_is_complete(1500, 1800, frames, timestamps, source_frames, source_episodes))
        timestamps[615] += .001
        self.assertFalse(launcher.future_is_complete(600, 1800, frames, timestamps, source_frames, source_episodes))

    def test_denormalized_metric_uses_original_unclipped_label(self):
        prediction = np.ones((30, 80))
        normalized = np.ones((30, 80))
        original = np.ones((30, 80)) * 7
        result = launcher.action_metrics(prediction, normalized, original, lambda value: value * 2)
        self.assertEqual(result["body_normalized_action_l1"], 0)
        self.assertEqual(result["hand_denormalized_action_l1"], 5)
        prediction[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            launcher.action_metrics(prediction, normalized, original, lambda value: value)

    def test_lifecycle_before_zero_after_exact_final_update(self):
        lifecycle = launcher.ComparisonLifecycle(2)
        events = []
        with self.assertRaises(RuntimeError):
            lifecycle.update(0, True)
        lifecycle.before(lambda: events.append(("before", lifecycle.completed)))
        lifecycle.update(0, False)
        lifecycle.update(0, True)
        with self.assertRaises(RuntimeError):
            lifecycle.after(lambda: None)
        lifecycle.update(1, True)
        lifecycle.after(lambda: events.append(("after", lifecycle.completed)))
        self.assertEqual(events, [("before", 0), ("after", 2)])
        with self.assertRaises(RuntimeError):
            lifecycle.before(lambda: None)
        with self.assertRaises(RuntimeError):
            lifecycle.update(2, True)

    def test_only_action_optimizer_and_frozen_hash_evidence(self):
        model = torch.nn.Module()
        model.vlm_model = torch.nn.Linear(2, 2).to(torch.bfloat16).requires_grad_(False)
        model.motion_conditioner = torch.nn.Linear(2, 2).requires_grad_(False)
        model.action_header = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.action_header.parameters(), lr=1e-5)
        report = launcher.optimizer_contract(model, optimizer)
        self.assertEqual(report["fresh_state_entries"], 0)
        initial = launcher.component_hashes(model)
        model.action_header(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        after = launcher.component_hashes(model)
        self.assertEqual(initial["vlm_model"], after["vlm_model"])
        self.assertEqual(initial["motion_conditioner"], after["motion_conditioner"])
        self.assertNotEqual(initial["action_header"], after["action_header"])
        with self.assertRaises(ValueError):
            launcher.optimizer_contract(model, optimizer)

    def test_fixed_eval_reuses_noise_preserves_rng_scheduler_and_training_mode(self):
        class FakeTrainer:
            device = torch.device("cpu")
            completed_updates = 0
            def __init__(self):
                self.model = torch.nn.Linear(1, 1)
                self.noise_scheduler = {"timesteps": [99]}
                self.maxmin = argparse.Namespace(denormalize=lambda value: value * 2)
            def unwrap_model(self):
                return self.model
            def forward_autocast(self):
                return contextlib.nullcontext()
            def forward_and_loss(self, model, batch):
                return {"loss": torch.rand(1).sum() + torch.randn_like(batch["actions"]).square().mean()
                        + np.random.rand() + random.random()}
            def inference(self, model, batch):
                self.noise_scheduler["timesteps"] = [1, 2, 3]
                return torch.randn_like(batch["actions"]) + np.random.rand() + random.random()
        trainer = FakeTrainer()
        batch = {"actions": torch.zeros(1, 30, 80), "raw_actions": [np.zeros((30, 80), dtype=np.float32)],
                 "input_ids": torch.tensor([[1, 2]])}
        manifest = [{"ordinal": 0, "episode": 0, "fm_seed": 1, "inference_seed": 2,
                     "input_sha256": launcher.tensor_signature(batch)}]
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(output=Path(temporary))
            torch.manual_seed(800)
            rng = torch.random.get_rng_state().clone()
            py_rng, np_rng = random.getstate(), np.random.get_state()
            before = launcher.evaluate_fixed_test(trainer, [batch], manifest, "before", args)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertEqual(py_rng, random.getstate())
            self.assertTrue(np.array_equal(np_rng[1], np.random.get_state()[1]))
            self.assertEqual(np_rng[2:], np.random.get_state()[2:])
            trainer.completed_updates = 300
            after = launcher.evaluate_fixed_test(trainer, [batch], manifest, "after", args)
            self.assertEqual(before["mean"], after["mean"])
            self.assertEqual(trainer.noise_scheduler, {"timesteps": [99]})
            self.assertTrue(trainer.model.training)
            self.assertTrue(np.array_equal(np.load(args.output / "test_before_actions.npz")["predictions"],
                                           np.load(args.output / "test_after_actions.npz")["predictions"]))
            batch["input_ids"][0, 0] = 5
            with self.assertRaises(RuntimeError):
                launcher.evaluate_fixed_test(trainer, [batch], manifest, "invalid", args)
            self.assertEqual(py_rng, random.getstate())
            self.assertTrue(np.array_equal(np_rng[1], np.random.get_state()[1]))
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            batch["input_ids"][0, 0] = 1
            def failed_forward(*unused):
                random.random()
                np.random.rand()
                torch.rand(2)
                raise ValueError("evaluation failed after RNG consumption")
            with mock.patch.object(trainer, "forward_and_loss", side_effect=failed_forward):
                with self.assertRaises(ValueError):
                    launcher.evaluate_fixed_test(trainer, [batch], manifest, "failed", args)
            self.assertEqual(py_rng, random.getstate())
            self.assertTrue(np.array_equal(np_rng[1], np.random.get_state()[1]))
            self.assertEqual(np_rng[2:], np.random.get_state()[2:])
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertTrue(trainer.model.training)

    def test_rng_isolation_restores_all_cpu_generators_on_exception(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
        with self.assertRaises(ValueError):
            with launcher.isolated_rng(9):
                random.random()
                np.random.rand()
                torch.rand(10)
                raise ValueError("dataset construction failed")
        self.assertEqual(py_rng, random.getstate())
        self.assertTrue(np.array_equal(np_rng[1], np.random.get_state()[1]))
        self.assertEqual(np_rng[2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(torch_rng, torch.random.get_rng_state()))

    def test_config_overrides_leave_model_layout_and_field_unchanged(self):
        args = argparse.Namespace(output=Path("/tmp/adaptation"), dataset_root=Path("/tmp/data"),
                                  dino_cache_root=Path("/tmp/dino"), seed=20261021, batch_size=4,
                                  grad_accum=4, max_steps=300, lr=1e-5, warmup_steps=20)
        overrides = launcher.make_overrides(args, Path("/tmp/run/checkpoints/ckpt_28000/model.safetensors"), "now")
        self.assertFalse(overrides["model.tune_vlm"])
        self.assertFalse(overrides["model.tune_motion"])
        self.assertEqual(overrides["train.early_checkpoint_steps"], [])
        self.assertGreater(overrides["train.checkpointing_steps"], args.max_steps)
        self.assertEqual(overrides["train.num_workers"], 2)
        self.assertEqual(overrides["train.dataloader_start_method"], "spawn")
        self.assertIsNone(overrides["log.report_to"])
        self.assertFalse(overrides["auto_tag_run"])
        self.assertFalse(any(key.startswith("data.transform.field.") for key in overrides))
        self.assertFalse(any(key in overrides for key in ("model.action_dim", "model.odim", "model.num_blocks")))

    def test_prepare_only_full_config_and_replay_without_loading_models(self):
        """Tiny on-disk provenance fixture; the model bytes are deliberately not a model."""
        from psi.data.lerobot import dino_cache
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, data, cache, output = (root / name for name in ("base", "data", "cache", "output"))
            base.mkdir()
            statistics = {"action.body_token_v1_1": {"min": [-1.] * 64, "max": [1.] * 64},
                          "action": {"min": [-1.] * 14, "max": [1.] * 14},
                          "observation.state": {"min": [-1.] * 43, "max": [1.] * 43}}
            launcher.write_json(base / "dataset_statistics.json", statistics)
            encoder = {"model_id": "fixture", "weights_sha256": "fixture", "local_file_sha256": {},
                       "preprocessing": {}, "image_process": {}}
            launcher.write_json(base / "dino_training_contract.json", {"encoder": encoder})
            argv = ["scripts/train.py", launcher.SCHEMA, "--exp=source", "--train.name=posttrain",
                    "--data.root-dir=" + str(data), "--data.train-repo-ids=train", "--data.val-repo-ids=val",
                    "--data.transform.repack.pad-action-dim=80", "--data.transform.repack.pad-state-dim=45",
                    "--data.transform.repack.action-keys", "action.body_token_v1_1", "action[:14]",
                    "--data.transform.field.stat-path=" + str(base / "dataset_statistics.json"),
                    "--data.transform.field.stat-action-keys", "action.body_token_v1_1", "action[:14]",
                    "--data.transform.field.pad-action-dim=80", "--data.transform.field.pad-state-dim=45",
                    "--data.transform.model.motion-image-policy=vlm_color_clean_motion",
                    "--model.action-architecture=dual_source_v1", "--model.action-chunk-size=30",
                    "--model.action-dim=80", "--model.odim=45", "--model.n-conditions=0"]
            (base / "argv.txt").write_text("\n".join(argv))
            source_config = launcher.parse_training_argv(base / "argv.txt")
            launcher.write_json(base / "run_config.json", source_config.model_dump(mode="json"))
            checkpoint = base / "checkpoints/ckpt_28000/model.safetensors"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"CPU preparation fixture, never loaded")
            launcher.write_json(base / "LOCAL_RUN_READY.json", {"status": "local_joint_run_prepared", "step": 28000,
                                "optimizer_rng_restored": False, "joint_sha256": launcher.sha256(checkpoint)})
            launcher.write_json(data / "CONVERSION_READY.json", {"status": "cpu_conversion_complete",
                                "complete_50_episode_conversion": True, "legacy_token_used": False,
                                "all_numeric_finite": True, "all_output_rgb_verified_lossless": True})
            launcher.write_json(data / "STATE_ACTION_CONTRACT.json", {"fixture": True})
            offset = 0
            manifests = {}
            for split, count in (("train", 40), ("val", 5), ("test", 5)):
                episode_path = data / split / "meta/episodes.jsonl"
                episode_path.parent.mkdir(parents=True)
                episode_path.write_text("\n".join(json.dumps({"episode_index": episode,
                                        "source_episode_index": offset + episode}) for episode in range(count)))
                launcher.write_json(data / split / "meta/info.json", {"fps": 30})
                launcher.write_json(cache / split / "manifest.json", {"fixture": True})
                manifests[str(cache / split)] = {"encoder": encoder, "episode_indices": list(range(count))}
                offset += count
            args = argparse.Namespace(base_run=base, dataset_root=data, dino_cache_root=cache, output=output,
                                      max_steps=300, batch_size=4, grad_accum=4, lr=1e-5, warmup_steps=20,
                                      seed=20261021, test_windows=20, train=False, prepare_only=True)
            original_argv = sys.argv[:]
            with mock.patch.object(dino_cache, "load_cache_manifest", side_effect=lambda path, *a, **k: manifests[str(path)]), \
                    mock.patch("torch.cuda.set_device", side_effect=AssertionError("CPU-only")):
                config, replay, report = launcher.prepare(args)
                second, _, second_report = launcher.prepare(args)
            self.assertEqual(original_argv, sys.argv)
            self.assertEqual(config.model_dump(mode="json"), second.model_dump(mode="json"))
            self.assertEqual(report["timestamp"], second_report["timestamp"])
            self.assertEqual(config.train.resume_from_checkpoint, None)
            self.assertEqual(config.train.early_checkpoint_steps, ())
            self.assertEqual(config.model.pretrained_joint_checkpoint, str(checkpoint))
            self.assertEqual(config.data.transform.field.model_dump(mode="json"),
                             source_config.data.transform.field.model_dump(mode="json"))
            self.assertEqual(replay[1], launcher.SCHEMA)
            self.assertEqual(config.model.action_architecture, "dual_source_v1")
            self.assertFalse(config.model.tune_vlm)
            self.assertIsNone(config.log.report_to)
            self.assertFalse((output / "training_result.json").exists())


if __name__ == "__main__":
    unittest.main()
