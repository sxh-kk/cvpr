"""Temporary collection fixtures only: never move/discard any real episode."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pico_episode_history import DISCARD_CONTRACT, EpisodeHistory
from test_audit_pico_teleop_episode import add_deferred_quality, make_episode


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n")


def hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()
            and path.name not in ("collection_record.json", "operator_discard.json")}


class EpisodeHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index = self.root / "pico_episode_history.json"
        self.history = EpisodeHistory(self.index)

    def episode(self, *, failure=None, session="session"):
        identity = str(uuid.uuid4())
        directory = self.root / session / "episodes" / identity
        directory.mkdir(parents=True)
        make_episode(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        # Fixture's own UUID is an actual random identity; give it that path.
        actual = directory.with_name(metadata["human_demo_uuid"])
        directory.rename(actual)
        directory = actual
        if failure:
            if failure == "rejected":
                outcome = json.loads((directory / "episode.json").read_text())
                outcome["state"].update(status="rejected", pending_audit=False, reasons=["lag=0.106s"])
                write(directory / "episode.json", outcome)
            elif failure == "force":
                path = directory / "applied_force_samples.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[25]["all_zero"] = False
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            elif failure == "physical":
                path = directory / "quality_samples.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                for row in rows:
                    row["contacts"]["left_hand"] = False
                    row["actual_target_xyz_m"][2] = .55
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                outcome = json.loads((directory / "episode.json").read_text())
                outcome["physical_gate"] = {"eligible_physics": True}  # Must not be trusted.
                write(directory / "episode.json", outcome)
        return directory

    def test_empty_cross_instance_pointer_and_restart_preserve_number_and_evidence(self):
        self.assertEqual(self.history.status()["status"], "empty")
        self.assertEqual(self.history.discard_previous()["status"], "empty")
        source = self.episode()
        original = hashes(source)
        remembered = self.history.remember(source)
        self.assertEqual(remembered["status"], "ready", remembered)
        self.assertEqual(remembered["number"], 1)
        self.assertEqual(remembered["disposition"], "kept")
        self.assertTrue(remembered["available"])
        actual = self.root / "kept/001"
        self.assertEqual(Path(remembered["path"]), actual)
        self.assertTrue(source.is_symlink())
        self.assertEqual(source.resolve(), actual)
        self.assertEqual(hashes(actual), original)
        self.assertEqual(EpisodeHistory(self.index).status(), remembered)
        self.assertEqual(EpisodeHistory(self.index).remember(source), remembered)
        self.assertEqual(len((self.root / "collection_records.jsonl").read_text().splitlines()), 1)
        record = json.loads((actual / "collection_record.json").read_text())
        self.assertTrue(record["physical_report"]["eligible_physics"])
        self.assertFalse(record["accepted_new_dataset"])
        self.assertFalse(record["eligible_for_packing"])

    def test_manual_discard_moves_actual_folder_updates_only_source_alias_and_is_idempotent(self):
        earlier = self.episode(session="old")
        self.history.remember(earlier)
        latest = self.episode(session="new")
        original = hashes(latest)
        self.history.remember(latest)
        result = EpisodeHistory(self.index).discard_previous()
        self.assertEqual(result["number"], 2)
        self.assertEqual(result["disposition"], "discarded")
        self.assertTrue(result["discarded"])
        self.assertFalse(result["available"])
        self.assertFalse(result["already_discarded"])
        self.assertFalse((self.root / "kept/002").exists())
        self.assertFalse((self.root / "kept/002").is_symlink())
        target = self.root / "discarded/002"
        self.assertEqual(latest.resolve(), target)
        self.assertEqual(hashes(target), original)
        marker = json.loads((target / "operator_discard.json").read_text())
        self.assertEqual(marker["contract"], DISCARD_CONTRACT)
        self.assertEqual(marker["human_demo_uuid"], result["uuid"])
        self.assertTrue(marker["operator_discarded"])
        marker_hash = hashlib.sha256((target / "operator_discard.json").read_bytes()).hexdigest()
        again = EpisodeHistory(self.index).discard_previous()
        self.assertTrue(again["already_discarded"])
        self.assertEqual(again["uuid"], result["uuid"])
        self.assertEqual(again["number"], 2)
        self.assertEqual(marker_hash, hashlib.sha256((target / "operator_discard.json").read_bytes()).hexdigest())
        self.assertEqual(earlier.resolve(), self.root / "kept/001")
        self.assertEqual(len((self.root / "collection_records.jsonl").read_text().splitlines()), 3)

    def test_recompute_physics_incomplete_and_force_failures_route_discarded_with_chinese_reasons(self):
        expected = {"physical":"未完成右手→双手→左手交接", "rejected":"录制中断", "force":"200Hz外力"}
        for number, (failure, text) in enumerate(expected.items(), start=1):
            source = self.episode(failure=failure, session=failure)
            original = hashes(source)
            result = self.history.remember(source)
            self.assertEqual(result["number"], number, result)
            self.assertEqual(result["disposition"], "discarded", result)
            self.assertIn(text, "；".join(result["reasons"]))
            self.assertEqual(source.resolve(), self.root / "discarded" / f"{number:03d}")
            self.assertEqual(hashes(source.resolve()), original)
        again = self.history.discard_previous()
        self.assertTrue(again["already_discarded"])
        self.assertEqual(again["number"], 3)

    def test_active_synthetic_foreign_and_symlink_paths_leave_pointer_and_files_untouched(self):
        valid = self.episode(session="valid")
        pointer = self.history.remember(valid)
        for kind in ("active", "synthetic", "symlink", "foreign"):
            with self.subTest(kind=kind):
                source = self.episode(session=kind)
                outcome = json.loads((source / "episode.json").read_text())
                if kind == "active":
                    outcome["state"]["status"] = "recording"
                    write(source / "episode.json", outcome)
                elif kind == "synthetic":
                    outcome["source_kind"] = "synthetic"
                    write(source / "episode.json", outcome)
                elif kind == "symlink":
                    alias = source.with_name("unregistered_alias")
                    alias.symlink_to(source, target_is_directory=True)
                    source = alias
                else:
                    outside = tempfile.TemporaryDirectory()
                    self.addCleanup(outside.cleanup)
                    source = Path(outside.name)
                self.assertEqual(self.history.remember(source)["status"], "error")
                self.assertEqual(self.history.status(), pointer)
        escaped_index = self.root / "link_index.json"
        escaped_index.symlink_to(self.index)
        self.assertEqual(EpisodeHistory(escaped_index).status()["status"], "error")

    def test_pending_move_and_post_alias_io_failures_recover_without_duplicate_numbers(self):
        for failure_method in ("rename", "_log"):
            with self.subTest(failure_method=failure_method):
                scope = self.root / failure_method
                scope.mkdir()
                history = EpisodeHistory(scope / "pico_episode_history.json")
                source = self.episode(session=failure_method + "_source")
                destination = scope / source.name
                source.rename(destination)
                source = destination
                original = hashes(source)
                if failure_method == "rename":
                    context = patch("pico_episode_history._rename_no_replace", side_effect=OSError("injected rename error"))
                else:
                    context = patch.object(history, "_log", side_effect=OSError("injected log error"))
                with context:
                    failed = history.remember(source)
                self.assertEqual(failed["status"], "error")
                self.assertIsNone(failed["disposition"])
                recovered = EpisodeHistory(history.index_path).status()
                self.assertEqual(recovered["status"], "ready", recovered)
                self.assertEqual(recovered["number"], 1)
                self.assertEqual(hashes(Path(recovered["path"])), original)
                self.assertEqual(source.resolve(), scope / "kept/001")
                self.assertEqual(len((scope / "collection_records.jsonl").read_text().splitlines()), 1)
                self.assertEqual(json.loads(history.index_path.read_text())["next_number"], 2)

    def test_collision_and_mutated_original_receipt_fail_without_overwrite(self):
        source = self.episode()
        original = hashes(source)
        (self.root / "kept/001").mkdir(parents=True)
        self.assertEqual(self.history.remember(source)["status"], "error")
        self.assertFalse(source.is_symlink())
        self.assertEqual(hashes(source), original)
        (self.root / "kept/001").rmdir()
        result = self.history.remember(source)
        self.assertEqual(result["number"], 1)
        target = Path(result["path"])
        outcome = json.loads((target / "episode.json").read_text())
        outcome["mutated"] = True
        write(target / "episode.json", outcome)
        self.assertEqual(self.history.discard_previous()["status"], "error")
        self.assertEqual(source.resolve(), target)
        self.assertFalse((target / "operator_discard.json").exists())

    def test_rejected_minimal_receipt_missing_metadata_and_short_prefix_still_get_numbers(self):
        missing = self.episode(session="missing_metadata")
        (missing / "metadata.json").unlink()
        outcome = json.loads((missing / "episode.json").read_text())
        outcome["state"] = {"status":"rejected", "pending_audit":False}
        write(missing / "episode.json", outcome)
        result = self.history.remember(missing)
        self.assertEqual(result["number"], 1, result)
        self.assertEqual(result["uuid"], missing.name)
        self.assertEqual(result["disposition"], "discarded")
        self.assertIn("录制中断", "；".join(result["reasons"]))
        short = self.episode(session="short")
        path = short / "quality_samples.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:25]) + "\n")
        result = self.history.remember(short)
        self.assertEqual(result["number"], 2, result)
        self.assertEqual(result["disposition"], "discarded")
        self.assertIn("记录不完整", "；".join(result["reasons"]))

    def test_finalize_errors_cannot_be_rescued_by_a_complete_successful_quality_file(self):
        for key in ("physical_gate_error", "finalization_errors"):
            with self.subTest(key=key):
                source = self.episode(session=key)
                outcome = json.loads((source / "episode.json").read_text())
                outcome[key] = "injected finalization failure"
                write(source / "episode.json", outcome)
                result = self.history.remember(source)
                self.assertEqual(result["disposition"], "discarded", result)
                self.assertIn("失败", "；".join(result["reasons"]))

    def test_original_solved_geometry_three_sha_and_recomputed_rows_are_required(self):
        for corruption in ("none", "cache_after_write", "missing_receipt", "extra_receipt", "wrong_sha", "recomputed_rows"):
            with self.subTest(corruption=corruption):
                source = self.episode(session=corruption)
                add_deferred_quality(source)
                if corruption == "cache_after_write":
                    path = source / "container_geometry_cache.json"
                    cache = json.loads(path.read_text())
                    cache["collision_geom_count"] += 1
                    write(path, cache)
                elif corruption in ("missing_receipt", "extra_receipt", "wrong_sha"):
                    path = source / "episode.json"
                    outcome = json.loads(path.read_text())
                    if corruption == "missing_receipt":
                        outcome.pop("quality_geometry_sources")
                    elif corruption == "extra_receipt":
                        outcome["quality_geometry_sources"]["unrelated"] = "a" * 64
                    else:
                        outcome["quality_geometry_sources"]["cache_sha256"] = "b" * 64
                    write(path, outcome)
                elif corruption == "recomputed_rows":
                    # Refresh the raw SHA deliberately: content equivalence to
                    # the completed geometry still needs independent checking.
                    path = source / "raw_quality_samples.jsonl"
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[-1]["original_live_support"]["container_support_weight_fraction"] = .75
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                    outcome = json.loads((source / "episode.json").read_text())
                    outcome["quality_geometry_sources"]["raw_samples_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    write(source / "episode.json", outcome)
                result = self.history.remember(source)
                self.assertNotEqual(result["status"], "error", result)
                if corruption == "none":
                    self.assertEqual(result["disposition"], "kept", result)
                else:
                    self.assertEqual(result["disposition"], "discarded", result)
                    self.assertIn("验收失败", "；".join(result["reasons"]))

    def test_first_index_write_failure_after_new_attempt_cannot_revert_to_discarding_old_pointer(self):
        import pico_episode_history as history_module
        older = self.episode(session="older")
        old = self.history.remember(older)
        latest = self.episode(session="new_workflow")
        receipt = json.loads((latest / "episode.json").read_text())
        receipt["capture_started"] = True
        write(latest / "episode.json", receipt)
        real_atomic = history_module._atomic
        def fail_initial_index(path, value, **kwargs):
            if path == self.index:
                raise OSError("injected first index write failure")
            return real_atomic(path, value, **kwargs)
        with patch("pico_episode_history._atomic", side_effect=fail_initial_index):
            result = self.history.remember(latest)
        self.assertEqual(result["status"], "error")
        self.assertFalse(latest.is_symlink())
        restarted = EpisodeHistory(self.index)
        self.assertEqual(restarted.status()["status"], "error")
        self.assertEqual(restarted.discard_previous()["status"], "error")
        self.assertEqual(older.resolve(), Path(old["path"]))
        self.assertFalse((Path(old["path"]) / "operator_discard.json").exists())
        recovered = restarted.remember(latest)
        self.assertEqual(recovered["status"], "ready", recovered)
        self.assertEqual(recovered["number"], 2)
        self.assertEqual(restarted.status()["uuid"], latest.name)
        discarded = restarted.discard_previous()
        self.assertEqual(discarded["number"], 2)
        self.assertEqual(older.resolve(), self.root / "kept/001")

    def test_unregistered_legacy_attempt_without_explicit_marker_does_not_block_pointer(self):
        source = self.episode(session="current")
        original = self.history.remember(source)
        self.episode(session="legacy")
        self.assertEqual(EpisodeHistory(self.index).status(), original)


if __name__ == "__main__":
    unittest.main()
