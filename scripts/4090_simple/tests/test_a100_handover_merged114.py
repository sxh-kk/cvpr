"""Admission mapping and weight preservation must survive the114/8GPU adaptation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest

PATH = Path(__file__).resolve().parents[1]/"a100_handover_merged114.py"
spec = importlib.util.spec_from_file_location("merged114_launcher", PATH)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class AcceptedWindowTests(unittest.TestCase):
    def setUp(self):
        self.history = SimpleNamespace(pack_info=[{}], source_episode_ranges=[{0: (0, 64), 1: (64, 128)}],
            original_indices=list(range(30, 64))+list(range(94, 128)))
        self.episodes = [{"episode_index": i, "split": "train", "length": 64,
                          "eligible_current_rows": [30, 31, 32, 33, 34]} for i in range(2)]

    def test_metadata_rows_map_through_actual_history_filter(self):
        rows, counts = launch.map_accepted_rows(self.history, self.episodes, 10)
        self.assertEqual(rows, [0, 1, 2, 3, 4, 34, 35, 36, 37, 38])
        self.assertEqual(counts, {"0": 5, "1": 5})

    def test_duplicated_admission_row_rejected(self):
        self.episodes[0]["eligible_current_rows"].append(30)
        with self.assertRaises(ValueError):
            launch.map_accepted_rows(self.history, self.episodes, 10)

    def test_tail_without_complete_future_rejected(self):
        self.episodes[0]["eligible_current_rows"].append(35)
        with self.assertRaises(ValueError):
            launch.map_accepted_rows(self.history, self.episodes, 11)

    def test_missing_real_history_window_rejected(self):
        self.history.original_indices.remove(30)
        with self.assertRaises(ValueError):
            launch.map_accepted_rows(self.history, self.episodes, 10)


class RecoveryRetentionTests(unittest.TestCase):
    def test_all_models_remain_but_only_newest_two_full_recovery_states(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in range(1000, 10001, 1000):
                p = root/f"ckpt_{step}"
                p.mkdir()
                for name in ["model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl", "config.json"]:
                    (p/name).write_bytes(f"{step}-{name}".encode())
            record = launch.retain_model_weights(root)
            self.assertEqual(record["all_saved_model_steps"], list(range(1000, 10001, 1000)))
            self.assertEqual(record["full_recovery_steps"], [9000, 10000])
            for step in record["all_saved_model_steps"]:
                p = root/f"ckpt_{step}"
                self.assertEqual((p/"model.safetensors").read_bytes(), f"{step}-model.safetensors".encode())
                self.assertTrue((p/"config.json").exists())
                self.assertEqual((p/"optimizer.bin").exists(), step >= 9000)
                self.assertEqual((p/"random_states_0.pkl").exists(), step >= 9000)
            self.assertEqual(launch.retain_model_weights(root)["removed_old_recovery_payloads"], [])

    def test_symlink_checkpoint_cannot_clean_outside_run(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base/"new"
            root.mkdir()
            old = base/"old"
            old.mkdir()
            (old/"model.safetensors").write_bytes(b"old model")
            (old/"optimizer.bin").write_bytes(b"old optimizer")
            (root/"ckpt_1000").symlink_to(old, target_is_directory=True)
            for step in [2000, 3000]:
                (root/f"ckpt_{step}").mkdir()
            with self.assertRaises(ValueError):
                launch.retain_model_weights(root)
            self.assertEqual((old/"optimizer.bin").read_bytes(), b"old optimizer")

    def test_keep_zero_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                launch.retain_model_weights(Path(directory), keep_full=0)


class RemoteAcceptanceTests(unittest.TestCase):
    def make_ready(self, root):
        for name in ["packed/PACK_READY.json", "dino/manifest.json", "dino/MERGED114_REINDEX.json", "actual_reader.json"]:
            p = root/name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"status": "passed"}))
        ready = {"contract": "cvpr-handover-merged114-a100-ready-v1", "status": "ready_for_training",
            "train_episodes": 114, "total_frames": 49089, "train_windows": 42363,
            "production_report": "actual_reader.json", "payload_sha256": {name: launch.sha(root/name) for name in
                ["packed/PACK_READY.json", "dino/manifest.json", "dino/MERGED114_REINDEX.json", "actual_reader.json"]}}
        (root/"A100_DATASET_READY.json").write_text(json.dumps(ready))
        return ready

    def test_actual_remote_cache_and_reader_hashes_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_ready(root)
            self.assertEqual(launch.validate_remote_ready(root)["train_episodes"], 114)
            (root/"dino/manifest.json").write_text('{"cache":"different relocation"}')
            with self.assertRaises(ValueError):
                launch.validate_remote_ready(root)

    def test_local_ready_alone_cannot_authorize_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"MERGED114_DATASET_READY.json").write_text('{"status":"ready_for_training"}')
            with self.assertRaises(FileNotFoundError):
                launch.validate_remote_ready(root)

    def test_failed_actual_reader_not_accepted_even_with_correct_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready = self.make_ready(root)
            (root/"actual_reader.json").write_text('{"status":"failed"}')
            ready["payload_sha256"]["actual_reader.json"] = launch.sha(root/"actual_reader.json")
            (root/"A100_DATASET_READY.json").write_text(json.dumps(ready))
            with self.assertRaises(ValueError):
                launch.validate_remote_ready(root)


class FrozenImportTests(unittest.TestCase):
    def test_frozen_code_moves_ahead_of_launcher_even_if_already_on_pythonpath(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = root/"frozen"
            scripts = code/"scripts/4090_simple"
            scripts.mkdir(parents=True)
            (scripts/"finetune_carrybox.py").write_text("identity = 'accepted frozen helper'\n")
            shadow = root/"new_launcher_dir"
            shadow.mkdir()
            (shadow/"finetune_carrybox.py").write_text("raise RuntimeError('wrong unsealed import')\n")
            upstream = root/"upstream.py"
            upstream.write_text(f"from pathlib import Path\nCODE = Path({str(code)!r})\nimport finetune_carrybox as shared\n")
            migration = root/"migration.py"
            migration.write_text("# sealed helper\n")
            plan = {"frozen_code": str(code), "upstream_runner": str(upstream),
                    "upstream_runner_sha256": launch.sha(upstream), "migration_helper": str(migration),
                    "migration_helper_sha256": launch.sha(migration)}
            previous_path = sys.path[:]
            previous_module = sys.modules.pop("finetune_carrybox", None)
            try:
                sys.path[:] = [str(shadow), str(scripts)]+previous_path
                result = launch.external(plan)
                self.assertEqual(result.shared.identity, "accepted frozen helper")
                self.assertEqual(sys.path[0], str(scripts))
            finally:
                sys.path[:] = previous_path
                sys.modules.pop("finetune_carrybox", None)
                if previous_module is not None:
                    sys.modules["finetune_carrybox"] = previous_module

if __name__ == "__main__":
    unittest.main()
