"""Tiny CPU checks for new Mobile overfit metric arithmetic and isolation."""
import contextlib
import copy
import importlib.util
import json
from pathlib import Path
import pickle
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("mobile_overfit_test", SCRIPTS / "mobile_pp_overfit_metrics.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
SPLIT_SHA = "a" * 64


def rng_identity():
    return (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()),
            torch.get_rng_state().numpy().tobytes())


class TinyDataset(Dataset):
    def __init__(self):
        self.rows, self.by_episode = [], {}
        for episode in range(10):
            self.by_episode[episode] = list(range(len(self.rows), len(self.rows) + 8 + episode))
            self.rows.extend((episode, row) for row in self.by_episode[episode])
        self.bad_mask = False

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        episode, row = self.rows[index]
        actions = torch.full((30, 80), float(row + 1))
        mask = torch.ones((30, 80), dtype=torch.bool)
        mask[:, 78:] = False
        if self.bad_mask:
            mask[:, 78] = True
        return {"actions": actions, "actions_mask": mask,
                "row": row, "episode": episode}


class TinyTrainer:
    def __init__(self, *, stochastic=False, failure=None):
        self.device = torch.device("cpu")
        self.accelerator = SimpleNamespace(num_processes=1)
        self.train_cfg = SimpleNamespace(val_batch_size=3)
        self.train_dataloader = SimpleNamespace(collate_fn=default_collate)
        self.model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout(.3))
        self.model.train()
        self.model[0].eval()  # Preserve heterogeneous frozen-module modes too.
        self.stochastic, self.failure = stochastic, failure
        self.fm_batches, self.inference_batches = [], []

    def unwrap_model(self):
        return self.model

    def forward_autocast(self):
        return contextlib.nullcontext()

    def noise(self):
        value = random.random() + float(np.random.rand()) + float(torch.rand(()))
        return value if self.stochastic else 0.

    def check(self, batch):
        assert not self.model.training and not torch.is_grad_enabled()
        assert len(batch["episode"].unique()) == 1, "A batch crossed episode boundaries"

    def forward_and_loss(self, model, batch):
        self.check(batch)
        self.fm_batches.append(batch["row"].tolist())
        noise = self.noise()
        if self.failure == "fm":
            raise RuntimeError("synthetic FM failure")
        loss = batch["actions"][:, 0, 0].mean() + noise
        if self.failure == "nonfinite":
            loss = torch.tensor(float("nan"))
        return {"loss": loss}

    def inference(self, model, batch):
        self.check(batch)
        self.inference_batches.append(batch["row"].tolist())
        noise = self.noise()
        if self.failure == "inference":
            raise RuntimeError("synthetic inference failure")
        predicted = batch["actions"].clone()
        error = batch["actions"][:, :1, :1] + noise
        predicted[:, :, :64] += error
        predicted[:, :, 64:78] += 2 * error
        predicted[:, :, 78:] += 1e9  # Padding must have no influence on L1.
        return predicted


class CaptureReduction(Exception):
    pass


class OverfitMetricsTests(unittest.TestCase):
    def setUp(self):
        self.dataset = TinyDataset()
        self.bundle = {"clean_dataset": self.dataset,
            "val_rows_by_episode": {i: self.dataset.by_episode[i] for i in range(5)},
            "train_probe_rows_by_episode": {i: self.dataset.by_episode[i] for i in range(5, 10)}}
        # This test isolates metric logic; real trainer device/string decoding is
        # the unchanged psi.utils function called by the production helper.
        self.batch_patch = patch.object(metrics, "_prepare_batch", lambda batch, device: batch)
        self.batch_patch.start()
        self.addCleanup(self.batch_patch.stop)

    def evaluate(self, trainer=None, update=0, sha=SPLIT_SHA, bundle=None):
        return metrics.evaluate_diagnostics(trainer or TinyTrainer(), bundle or self.bundle, update, sha)

    def expected(self, role):
        groups = self.bundle[role + "_rows_by_episode"]
        all_values = [row + 1 for rows in groups.values() for row in rows]
        episode_means = [sum(row + 1 for row in rows) / len(rows) for rows in groups.values()]
        sampled = [[rows[i * (len(rows) - 1) // 7] + 1 for i in range(8)] for rows in groups.values()]
        return (sum(all_values) / len(all_values), sum(episode_means) / 5,
                sum(sum(values) for values in sampled) / 40)

    def test_full_loss_is_batch_size_weighted_and_episode_macro_is_distinct(self):
        result = self.evaluate()
        for role in metrics.ROLES:
            micro, macro, _ = self.expected(role)
            self.assertAlmostEqual(result["metrics"][role + "_full_fm_loss_micro"], micro)
            self.assertAlmostEqual(result["metrics"][role + "_full_fm_loss_episode_macro"], macro)
            self.assertNotEqual(micro, macro)
        self.assertEqual(result["roles"]["val"]["counts"]["full_fm_windows"], 50)
        self.assertEqual(result["roles"]["train_probe"]["counts"]["full_fm_windows"], 75)

    def test_sampled_l1_is_exactly_uniform_eight_each_and_excludes_padding(self):
        result = self.evaluate()
        for role in metrics.ROLES:
            _, _, expected_l1 = self.expected(role)
            self.assertAlmostEqual(result["metrics"][role + "_sampled_normalized_body_token_l1_micro"], expected_l1)
            self.assertAlmostEqual(result["metrics"][role + "_sampled_normalized_hand_joint_l1_micro"], 2 * expected_l1)
            counts = result["roles"][role]["counts"]
            self.assertEqual(counts["sampled_l1_windows"], 40)
            self.assertEqual(counts["body_elements"], 40 * 30 * 64)
            self.assertEqual(counts["hand_elements"], 40 * 30 * 14)
            for episode, record in result["roles"][role]["per_episode"].items():
                rows = self.bundle[role + "_rows_by_episode"][int(episode)]
                self.assertEqual(record["sampled_rows"], [rows[i * (len(rows) - 1) // 7] for i in range(8)])
        json.dumps(result, allow_nan=False)

    def test_full_loss_and_sampled_inference_are_separate_no_aug_reader_passes(self):
        trainer = TinyTrainer()
        original_rows = copy.deepcopy({k: v for k, v in self.bundle.items() if k != "clean_dataset"})
        result = self.evaluate(trainer)
        self.assertEqual(sum(map(len, trainer.fm_batches)), len(self.dataset))
        self.assertEqual(sum(map(len, trainer.inference_batches)), 80)
        self.assertEqual(original_rows, {k: v for k, v in self.bundle.items() if k != "clean_dataset"})
        self.assertFalse(result["padding_channels_in_metrics"])
        self.assertIn("all eligible", result["full_fm_scope"])
        self.assertIn("40 per role", result["sampled_l1_scope"])

    def test_checkpoint_independent_seed_and_all_training_rng_and_modes_restored(self):
        trainer = TinyTrainer(stochastic=True)
        before = rng_identity()
        modes = [m.training for m in trainer.model.modules()]
        first = self.evaluate(trainer, update=0)
        self.assertEqual(before, rng_identity())
        self.assertEqual(modes, [m.training for m in trainer.model.modules()])
        second = self.evaluate(trainer, update=1000)
        self.assertEqual(first["metrics"], second["metrics"])
        changed = self.evaluate(trainer, update=1000, sha="b" * 64)
        self.assertNotEqual(first["metrics"], changed["metrics"])
        self.assertEqual(before, rng_identity())

    def test_fm_and_inference_exceptions_restore_rng_and_heterogeneous_modes(self):
        for failure in ("fm", "inference", "nonfinite"):
            with self.subTest(failure=failure):
                trainer = TinyTrainer(stochastic=True, failure=failure)
                before, modes = rng_identity(), [m.training for m in trainer.model.modules()]
                with self.assertRaises((RuntimeError, ValueError)):
                    self.evaluate(trainer)
                self.assertEqual(before, rng_identity())
                self.assertEqual(modes, [m.training for m in trainer.model.modules()])

    def test_cuda_rng_context_touches_only_explicit_rank_device_without_seed_all(self):
        before = rng_identity()
        local_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
        with patch.object(torch.cuda, "get_rng_state", return_value=local_state) as get, \
             patch.object(torch.cuda, "set_rng_state") as restore, \
             patch.object(torch.cuda, "device", return_value=contextlib.nullcontext()) as device, \
             patch.object(torch.cuda, "manual_seed") as seed, \
             patch.object(torch.cuda, "manual_seed_all", side_effect=AssertionError("nonlocal seed")), \
             patch.object(torch.cuda, "device_count", side_effect=AssertionError("nonlocal enumeration")), \
             patch.object(torch, "manual_seed", side_effect=AssertionError("global seed")):
            with self.assertRaisesRegex(RuntimeError, "synthetic"), metrics._isolated_rng(91, "cuda:3"):
                random.random(); np.random.rand(); torch.rand(())
                raise RuntimeError("synthetic local CUDA-context failure")
            get.assert_called_once_with(3)
            device.assert_called_once_with(3)
            seed.assert_called_once_with(91)
            self.assertIs(restore.call_args.args[0], local_state)
            self.assertEqual(restore.call_args.args[1], 3)
        self.assertEqual(before, rng_identity())

    def test_invalid_rows_episode_overlap_masks_and_split_identity_are_rejected(self):
        overlap = copy.copy(self.bundle)
        overlap["train_probe_rows_by_episode"] = {**self.bundle["train_probe_rows_by_episode"]}
        overlap["train_probe_rows_by_episode"][5] = self.bundle["val_rows_by_episode"][0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.evaluate(bundle=overlap)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.evaluate(sha="not-a-sha")
        self.dataset.bad_mask = True
        with self.assertRaisesRegex(ValueError, "padding"):
            self.evaluate()

    def test_uneven_rank_shards_have_no_duplicates_or_padding_and_one_final_reduce(self):
        world, partials, full_rows, sampled_rows = 16, [], [], []
        for rank in range(world):
            trainer = TinyTrainer()
            def capture(packed, count):
                self.assertEqual(count, world)
                partials.append(packed.clone())
                raise CaptureReduction()
            with patch.object(metrics, "_rank_world", return_value=(rank, world)), \
                 patch.object(metrics, "_reduce_sum", side_effect=capture) as reducer:
                with self.assertRaises(CaptureReduction):
                    self.evaluate(trainer)
                self.assertEqual(reducer.call_count, 1)
            full_rows.extend(row for batch in trainer.fm_batches for row in batch)
            sampled_rows.extend(row for batch in trainer.inference_batches for row in batch)
        self.assertEqual(sorted(full_rows), list(range(len(self.dataset))))
        self.assertEqual(len(sampled_rows), len(set(sampled_rows)))
        self.assertEqual(len(sampled_rows), 80)
        combined = torch.stack(partials).sum(0)
        def reduce(packed, count):
            packed.copy_(combined)
        with patch.object(metrics, "_rank_world", return_value=(0, world)), \
             patch.object(metrics, "_reduce_sum", side_effect=reduce) as reducer:
            distributed = self.evaluate()
            self.assertEqual(reducer.call_count, 1)
        serial = self.evaluate()
        self.assertEqual(serial["metrics"], distributed["metrics"])
        self.assertEqual(distributed["world_size"], world)

    def test_rank_local_failure_joins_final_collective_and_remote_failure_rejects(self):
        trainer = TinyTrainer(failure="fm")
        with patch.object(metrics, "_rank_world", return_value=(0, 8)), \
             patch.object(metrics, "_reduce_sum") as reducer:
            with self.assertRaisesRegex(RuntimeError, "collectively"):
                self.evaluate(trainer)
            self.assertEqual(reducer.call_count, 1)
            self.assertEqual(reducer.call_args.args[0][-1, 0].item(), 1)
        def remote_failure(packed, world):
            packed[-1, 0] = 1
        with patch.object(metrics, "_rank_world", return_value=(0, 8)), \
             patch.object(metrics, "_reduce_sum", side_effect=remote_failure):
            with self.assertRaisesRegex(RuntimeError, "another rank failed"):
                self.evaluate()


if __name__ == "__main__":
    unittest.main()
