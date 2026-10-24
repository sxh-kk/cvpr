"""CPU checks for integrity, filtered-index mapping and new-run loader sampling."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.utils.data import DataLoader, Dataset, Subset

SPEC = importlib.util.spec_from_file_location(
    "handover_critical_sampler", Path(__file__).resolve().parents[1] / "handover_critical_sampler.py")
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


class FakeProductionDataset(Dataset):
    def __init__(self, train):
        self.raw_dataset = SimpleNamespace(motion_history=SimpleNamespace(
            # The DINO/history filter order deliberately differs from Parquet order.
            original_indices=[133, 31, 132, 30, 131, 32, 130, 33],
            source_episode_ranges=[{0: (0, 100), 1: (100, 200)}],
            pack_info=[("unused-cache", str(train), {})]))

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return index


def fixture(root):
    pack = root / "packed"
    train = pack / "train"
    (train / "meta").mkdir(parents=True)
    write_json(pack / "PACK_READY.json", {"contract": "test-quality100", "actual_labels_unchanged": True})
    episodes = [{"episode_index": 0, "source_episode_index": 9, "split": "train", "length": 100,
                 "eligible_current_rows": [30, 32]},
                {"episode_index": 1, "source_episode_index": 17, "split": "train", "length": 100,
                 "eligible_current_rows": [31, 33]}]
    (train / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    directory = root / "sidecar"
    directory.mkdir()
    rows = []
    for ep, row, source, stage, weight in ((0, 30, 9, "lift", .1), (0, 32, 9, "lift", .2),
                                         (1, 31, 17, "handover", .3), (1, 33, 17, "placement", .4)):
        rows.append({"episode_index": ep, "current_row": row, "source_episode_index": source,
                     "critical_stages": [stage], "weight": weight, "base_weight": .05,
                     "focus_weights": {stage: weight - .05}, "observed_stage": "observed-from-past",
                     "sim_time_s": .005 + row / 30})
    metadata = {"contract": adapter.CONTRACT, "window_count": len(rows),
                "source_dataset": str(train), "source_pack_ready_sha256": sha(pack / "PACK_READY.json"),
                "source_metadata_sha256": sha(train / "meta/episodes.jsonl"),
                "physical_receipt": {"raw_path": "/builder-only/not-on-consumer/raw.jsonl", "sha256": "a" * 64}}
    save(directory, rows, metadata)
    base = FakeProductionDataset(train)
    subset = Subset(base, [3, 0, 5, 4])
    return directory, train, rows, metadata, base, subset


def save(directory, rows, metadata):
    (directory / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    metadata["window_count"] = len(rows)
    metadata["rows_sha256"] = sha(directory / "rows.jsonl")
    metadata["weights_digest"] = adapter.weights_digest(rows)
    write_json(directory / "metadata.json", metadata)


class CriticalSamplerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.directory, self.train, self.rows, self.meta, self.base, self.subset = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def load(self, **kwargs):
        return adapter.load_sidecar(self.directory, expected_window_count=4, **kwargs)

    def test_integrity_and_relocated_consumer_do_not_require_raw_builder_paths(self):
        moved = self.root / "relocated-packed"
        shutil.copytree(self.train.parent, moved)
        sidecar = self.load(dataset_root=moved)
        weights = adapter.weights_for_subset(self.subset, sidecar, dataset_root=moved, expected_window_count=4)
        self.assertTrue(torch.allclose(weights, torch.tensor([.1, .4, .2, .3], dtype=torch.float64)))

    def test_nonidentity_history_and_subset_mapping(self):
        weights = adapter.weights_for_subset(self.subset, self.directory, expected_window_count=4)
        self.assertEqual(weights.tolist(), [.1, .4, .2, .3])

    def test_nested_subset_mapping_retains_outer_order(self):
        inner = Subset(self.base, [0, 3, 4, 5, 6])
        outer = Subset(inner, [3, 2, 1, 0])
        weights = adapter.weights_for_subset(outer, self.directory, expected_window_count=4)
        self.assertEqual(weights.tolist(), [.2, .3, .1, .4])

    def test_raw_rows_file_tamper_rejected(self):
        with (self.directory / "rows.jsonl").open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "rows SHA"):
            self.load()

    def test_probability_digest_tamper_rejected(self):
        self.meta["weights_digest"] = "0" * 64
        write_json(self.directory / "metadata.json", self.meta)
        with self.assertRaisesRegex(ValueError, "probability digest"):
            self.load()

    def test_wrong_expected_release_size_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected release"):
            adapter.load_sidecar(self.directory)

    def test_source_metadata_tamper_rejected(self):
        with (self.train / "meta/episodes.jsonl").open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "episodes metadata"):
            self.load(dataset_root=self.train)

    def test_pack_receipt_tamper_rejected(self):
        write_json(self.train.parent / "PACK_READY.json", {"different": True})
        with self.assertRaisesRegex(ValueError, "PACK_READY"):
            self.load(dataset_root=self.train)

    def test_duplicate_row_rejected(self):
        self.rows[1] = deepcopy(self.rows[0])
        save(self.directory, self.rows, self.meta)
        with self.assertRaisesRegex(ValueError, "Repeated"):
            self.load()

    def test_zero_or_negative_probability_rejected(self):
        for weight in (0., -.1):
            rows = deepcopy(self.rows)
            rows[0]["weight"] = weight
            save(self.directory, rows, self.meta)
            with self.assertRaises(ValueError):
                self.load()

    def test_unbacked_focus_tag_rejected(self):
        self.rows[0]["critical_stages"] = []
        save(self.directory, self.rows, self.meta)
        with self.assertRaisesRegex(ValueError, "not backed"):
            self.load()

    def test_heldout_ancestor_rejected(self):
        self.rows[0]["source_episode_index"] = 39
        save(self.directory, self.rows, self.meta)
        with self.assertRaisesRegex(ValueError, "heldout"):
            self.load()

    def test_missing_subset_row_rejected(self):
        subset = Subset(self.base, [3, 0, 5])
        with self.assertRaisesRegex(ValueError, "every admitted window"):
            adapter.weights_for_subset(subset, self.directory, expected_window_count=4)

    def test_extra_unadmitted_subset_row_rejected(self):
        subset = Subset(self.base, [3, 0, 5, 4, 1])
        with self.assertRaisesRegex(ValueError, "non-admitted"):
            adapter.weights_for_subset(subset, self.directory, expected_window_count=4)

    def test_repeated_dataset_row_rejected(self):
        subset = Subset(self.base, [3, 0, 5, 4, 3])
        with self.assertRaisesRegex(ValueError, "repeats"):
            adapter.weights_for_subset(subset, self.directory, expected_window_count=4)

    def test_source_ancestor_mismatch_rejected_even_with_valid_sidecar_hashes(self):
        self.rows[0]["source_episode_index"] = 18
        save(self.directory, self.rows, self.meta)
        with self.assertRaisesRegex(ValueError, "eligible rows/ancestors"):
            self.load(dataset_root=self.train)

    def test_sampler_same_seed_and_epoch_is_global_rank_independent(self):
        rank0 = adapter.EpochWeightedSampler([.1, .2, .3, .4], seed=0, num_samples=128)
        rank1 = adapter.EpochWeightedSampler([.1, .2, .3, .4], seed=0, num_samples=128)
        first = list(rank0)
        self.assertEqual(first, list(rank1))
        self.assertEqual(first, list(rank0))
        self.assertEqual(len(rank0), 128)
        rank0.set_epoch(1)
        rank1.set_epoch(1)
        self.assertNotEqual(first, list(rank0))
        self.assertEqual(list(rank0), list(rank1))
        # External batch sharding covers the global stream, without rank slicing inside the sampler.
        blocks = [first[i:i + 8] for i in range(0, len(first), 8)]
        shard0, shard1 = blocks[::2], blocks[1::2]
        rebuilt = [value for pair in zip(shard0, shard1) for block in pair for value in block]
        self.assertEqual(first, rebuilt)

    def test_sampler_empirical_mass_tracks_critical_probabilities(self):
        sampler = adapter.EpochWeightedSampler([.05, .15, .8], seed=91, num_samples=20000)
        values = torch.tensor(list(sampler))
        observed = torch.bincount(values, minlength=3).double() / len(values)
        self.assertTrue(torch.allclose(observed, torch.tensor([.05, .15, .8], dtype=torch.float64), atol=.015, rtol=0))

    def test_sampler_invalid_weights_and_epoch_rejected(self):
        for weights in ([0, 1], [-1, 2], [float("nan"), 1], [float("inf"), 1], []):
            with self.assertRaises(ValueError):
                adapter.EpochWeightedSampler(weights, seed=1)
        with self.assertRaises(ValueError):
            adapter.EpochWeightedSampler([1], seed=1).set_epoch(-1)

    def trainer(self, *, num_workers=0):
        g = torch.Generator().manual_seed(17)
        options = dict(batch_size=2, shuffle=True, drop_last=True, collate_fn=list,
                       generator=g, pin_memory=False, num_workers=num_workers)
        if num_workers:
            options.update(prefetch_factor=3, persistent_workers=True, multiprocessing_context="spawn")
        train = DataLoader(self.subset, **options)
        val = DataLoader([1, 2, 3], batch_size=1)
        return SimpleNamespace(train_dataloader=train, val_dataloader=val, cfg=SimpleNamespace(seed=0), completed_updates=0)

    def test_install_preserves_loader_options_and_validation(self):
        trainer = self.trainer(num_workers=1)
        old, validation = trainer.train_dataloader, trainer.val_dataloader
        receipt = adapter.install_critical_sampler(trainer, self.directory, expected_window_count=4)
        new = trainer.train_dataloader
        self.assertIs(new.dataset, old.dataset)
        self.assertIs(new.collate_fn, old.collate_fn)
        self.assertIs(new.generator, old.generator)
        self.assertEqual(new.batch_size, old.batch_size)
        self.assertEqual(new.drop_last, old.drop_last)
        self.assertEqual(new.prefetch_factor, old.prefetch_factor)
        self.assertEqual(new.persistent_workers, old.persistent_workers)
        self.assertEqual(new.multiprocessing_context.get_start_method(), "spawn")
        self.assertIs(trainer.val_dataloader, validation)
        self.assertIs(new.sampler, trainer.train_sampler)
        self.assertEqual(trainer.train_sampler.seed, 0)
        self.assertEqual(new.sampler.weights.tolist(), [.1, .4, .2, .3])
        self.assertFalse(receipt["validation_loader_changed"])
        self.assertFalse(receipt["stage_labels_are_model_inputs"])

    def test_fresh_loader_can_iterate_without_touching_val(self):
        trainer = self.trainer()
        validation = trainer.val_dataloader
        adapter.install_critical_sampler(trainer, self.directory, seed=23, expected_window_count=4)
        batches = list(trainer.train_dataloader)
        self.assertEqual(sum(map(len, batches)), 4)
        self.assertIs(trainer.val_dataloader, validation)
        self.assertEqual(list(validation), [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])])

    def test_active_or_repeated_installation_rejected(self):
        trainer = self.trainer()
        trainer.completed_updates = 1
        with self.assertRaisesRegex(ValueError, "fresh plain loader"):
            adapter.install_critical_sampler(trainer, self.directory, expected_window_count=4)
        trainer.completed_updates = 0
        adapter.install_critical_sampler(trainer, self.directory, expected_window_count=4)
        with self.assertRaisesRegex(ValueError, "fresh plain loader"):
            adapter.install_critical_sampler(trainer, self.directory, expected_window_count=4)

    def test_pending_resume_is_rejected_before_it_loads_optimizer_state(self):
        trainer = self.trainer()
        trainer.cfg.train = SimpleNamespace(resume_from_checkpoint="checkpoint_1000")
        with self.assertRaisesRegex(ValueError, "fresh plain loader"):
            adapter.install_critical_sampler(trainer, self.directory, expected_window_count=4)


if __name__ == "__main__":
    unittest.main()
