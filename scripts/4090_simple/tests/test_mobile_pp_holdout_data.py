"""Frozen episode partitioning with tiny metadata and real production row maps."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


holdout = load("holdout_data_test", "mobile_pp_holdout_data.py")
legacy = load("holdout_legacy_mapper_test", "a100_handover_merged114.py")
shared = load("holdout_real_clock_filter_test", "finetune_carrybox.py")


def episodes():
    result = []
    for index in range(100):
        identity = str(uuid.UUID(int=index + 1))
        result.append({"episode_index": index, "human_demo_uuid": identity, "length": 64,
            "task_id": holdout.TASK_ID, "split": "train", "split_restriction": "train_only",
            "eligible_current_rows": list(range(30, 35)), "lineage_complete": True,
            "lineage_group_sources": [{"source_namespace": "live_pico_human", "human_demo_uuid": identity}]})
    return result


class Metadata:
    def __init__(self, values):
        self.values = values
        self.column_names = list(values)

    def with_format(self, format):
        assert format == "numpy"
        return self

    def __getitem__(self, key):
        return self.values[key]


class TinyDataset:
    def __init__(self, release, source, no_aug):
        self.no_aug = no_aug
        self.episodes = source
        ranges, originals, owners = {}, [], []
        frames, times, source_frames, source_episodes, labels = [], [], [], [], []
        offset = 0
        for episode in source:
            index, length = episode["episode_index"], episode["length"]
            ranges[index] = (offset, offset + length)
            originals.extend(offset + row for row in range(30, length))
            owners.extend([index] * (length - 30))
            frames.extend(range(length))
            times.extend(row / 30 for row in range(length))
            source_frames.extend((row * 50 + 29) // 30 for row in range(length))
            source_episodes.extend([1000000 + index] * length)
            labels.extend(row >= 30 for row in range(length))
            offset += length
        self.history = SimpleNamespace(pack_info=[(str(release / "dino"), str(release / "packed/train"), {})],
            source_episode_ranges=[ranges], original_indices=np.array(originals, dtype=np.int64),
            episode_ids=np.array(owners, dtype=np.int64))
        columns = Metadata({"frame_index": np.array(frames), "timestamp": np.array(times, dtype=np.float32),
            "source_frame_index": np.array(source_frames), "source_episode_index": np.array(source_episodes),
            "action_valid": np.array(labels, dtype=bool)})
        self.raw_dataset = SimpleNamespace(motion_history=self.history,
            base_dataset=SimpleNamespace(hf_dataset=columns))

    def __len__(self):
        return len(self.history.original_indices)

    def __getitem__(self, filtered):
        index = int(self.history.episode_ids[filtered])
        original = int(self.history.original_indices[filtered])
        start, _ = self.history.source_episode_ranges[0][index]
        return {"episode": index, "current_row": original - start,
                "uuid": self.episodes[index]["human_demo_uuid"], "no_aug": self.no_aug}


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        self.source = episodes()
        # Keep fixed100/95/5 semantics; shrink only source lengths/window counts
        # inside this isolated fixture so real clock/future mapping stays small.
        self.patches = [patch.object(holdout, "TOTAL_FRAMES", 6400), patch.object(holdout, "TOTAL_WINDOWS", 500)]
        for value in self.patches:
            value.start()
            self.addCleanup(value.stop)
        self.sha = "b" * 64
        self.split = holdout.build_split(self.source, self.sha)

    def test_exact_uuid_hash_95_5_and_next_five_train_probe_are_order_independent(self):
        ranked = sorted(self.source, key=lambda row: hashlib.sha256(("42:" + row["human_demo_uuid"]).encode()).hexdigest())
        expected_eval = sorted(row["episode_index"] for row in ranked[:5])
        expected_probe = sorted(row["episode_index"] for row in ranked[5:10])
        self.assertEqual(self.split["eval_episode_indices"], expected_eval)
        self.assertEqual(self.split["train_probe_episode_indices"], expected_probe)
        self.assertEqual(len(self.split["train_episode_indices"]), 95)
        self.assertTrue(set(expected_probe) <= set(self.split["train_episode_indices"]))
        self.assertFalse(set(self.split["train_uuids"]) & set(self.split["eval_uuids"]))
        self.assertEqual(self.split, holdout.build_split(list(reversed(self.source)), self.sha))
        self.assertNotEqual(self.split["eval_uuids"], holdout.build_split(self.source, self.sha, 43)["eval_uuids"])
        self.assertFalse(self.split["source_role_override"]["source_archive_modified"])
        self.assertFalse(self.split["source_role_override"]["previous_all100_task_checkpoint_allowed"])

    def test_duplicate_uuid_index_or_missing_lineage_cannot_be_frozen(self):
        for field in ("human_demo_uuid", "episode_index", "lineage_complete"):
            bad = deepcopy(self.source)
            bad[1][field] = False if field == "lineage_complete" else bad[0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                holdout.build_split(bad, self.sha)

    def test_lineage_shared_across_train_eval_refuses_split_not_source_data(self):
        bad = deepcopy(self.source)
        train, evaluation = self.split["train_episode_indices"][0], self.split["eval_episode_indices"][0]
        bad[train]["lineage_group_sources"].append(deepcopy(bad[evaluation]["lineage_group_sources"][0]))
        with self.assertRaisesRegex(ValueError, "without rejecting source data"):
            holdout.build_split(bad, self.sha)

    def test_validate_split_recomputes_selection_source_sha_and_types(self):
        self.assertEqual(holdout.validate_split(self.source, self.split, self.sha), self.split)
        for mutation in ("eval", "sha", "seed_type", "override"):
            bad = deepcopy(self.split)
            if mutation == "eval":
                bad["eval_episode_indices"][0] = bad["train_episode_indices"][0]
            elif mutation == "sha":
                bad["source_metadata_sha256"] = "0" * 64
            elif mutation == "seed_type":
                bad["seed"] = True
            else:
                bad["source_role_override"]["source_archive_modified"] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                holdout.validate_split(self.source, bad, self.sha)

    def test_last_complete_future_row_is_included_but_repeated_tail_cannot_enter(self):
        self.assertEqual(self.split["source_total_windows"], 500)
        bad = deepcopy(self.source)
        bad[0]["eligible_current_rows"][-1] = 35  # 35+29==64, outside this episode.
        with self.assertRaisesRegex(ValueError, "history/future"):
            holdout.build_split(bad, self.sha)

    def fixture(self, root, *, corrupt=None):
        release = root / "release"
        (release / "packed/train/meta").mkdir(parents=True)
        (release / "dino").mkdir()
        metadata = release / "packed/train/meta/episodes.jsonl"
        metadata.write_text("".join(json.dumps(row) + "\n" for row in self.source))
        split = holdout.build_split(self.source, holdout.sha_file(metadata))
        split_path = root / "split.json"
        split_path.write_text(json.dumps(split, indent=2))
        calls, created = [], {}
        def data(*, split, transform_kwargs):
            calls.append((split, deepcopy(transform_kwargs)))
            result = TinyDataset(release, self.source, no_aug=transform_kwargs.get("no_aug", split != "train"))
            if split == "val" and corrupt == "duplicate_original":
                result.history.original_indices[0] = result.history.original_indices[1]
            elif split == "val" and corrupt == "clock_gap":
                result.raw_dataset.base_dataset.hf_dataset.values["source_frame_index"][32] += 10
            elif split == "val" and corrupt == "wrong_episode":
                result.history.episode_ids[0] = 1
            created[split] = result
            return result
        plan = {"release": str(release), "split_file": str(split_path), "split_sha256": holdout.sha_file(split_path)}
        return SimpleNamespace(data=data), plan, split, calls, created

    def test_real_admission_and_clock_mapping_cover95_5_with_one_shared_clean_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            config, plan, split, calls, created = self.fixture(Path(directory))
            metadata = Path(plan["release"]) / "packed/train/meta/episodes.jsonl"
            before = holdout.sha_file(metadata)
            train, val, summary, bundle = holdout.datasets(config, "processor", plan, legacy, shared)
            self.assertEqual(len(train), 95 * 5)
            self.assertEqual(len(val), 5 * 5)
            self.assertEqual(summary["train_windows"] + summary["validation_windows"], 500)
            self.assertEqual(summary["train_frames"] + summary["validation_frames"], 6400)
            self.assertEqual(summary["train_probe_windows"], 25)
            self.assertEqual([call[0] for call in calls], ["train", "val"])
            self.assertEqual(calls[1][1], {"vlm_processor": "processor", "no_aug": True})
            self.assertIs(bundle["clean_dataset"], created["val"])
            self.assertIs(val.dataset, bundle["clean_dataset"])
            train_uuids = {train[i]["uuid"] for i in range(len(train))}
            eval_uuids = {val[i]["uuid"] for i in range(len(val))}
            self.assertEqual(train_uuids, set(split["train_uuids"]))
            self.assertEqual(eval_uuids, set(split["eval_uuids"]))
            self.assertFalse(train_uuids & eval_uuids)
            self.assertFalse(train[0]["no_aug"])
            self.assertTrue(val[0]["no_aug"])
            for index, rows in bundle["val_rows_by_episode"].items():
                self.assertEqual([bundle["clean_dataset"][row]["current_row"] for row in rows], [30, 31, 32, 33, 34])
                self.assertTrue(all(bundle["clean_dataset"][row]["episode"] == index for row in rows))
            for index, rows in bundle["train_probe_rows_by_episode"].items():
                self.assertIn(index, split["train_probe_episode_indices"])
                self.assertTrue(all(bundle["clean_dataset"][row]["no_aug"] for row in rows))
                self.assertIn(bundle["clean_dataset"][rows[0]]["uuid"], train_uuids)
            self.assertEqual(holdout.sha_file(metadata), before)

    def test_reader_duplicates_clock_gaps_or_wrong_episode_assignment_refuse_leakage(self):
        for corrupt in ("duplicate_original", "clock_gap", "wrong_episode"):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                config, plan, _, _, _ = self.fixture(Path(directory), corrupt=corrupt)
                with self.assertRaises(ValueError):
                    holdout.datasets(config, None, plan, legacy, shared)

    def test_split_file_sha_mismatch_stops_before_readers_construct(self):
        with tempfile.TemporaryDirectory() as directory:
            config, plan, _, calls, _ = self.fixture(Path(directory))
            Path(plan["split_file"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "split file SHA"):
                holdout.datasets(config, None, plan, legacy, shared)
            self.assertEqual(calls, [])

    def test_metadata_byte_change_cannot_reuse_the_frozen_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            config, plan, _, calls, _ = self.fixture(Path(directory))
            metadata = Path(plan["release"]) / "packed/train/meta/episodes.jsonl"
            metadata.write_text(metadata.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "identity drifted"):
                holdout.datasets(config, None, plan, legacy, shared)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
