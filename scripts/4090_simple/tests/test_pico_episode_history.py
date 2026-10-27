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
from pico_episode_history import DISCARD_CONTRACT, EpisodeHistory, MOBILE_TASK_ID, MOBILE_OFFICIAL_SOURCE_SHA256
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

    def background_receipt(self, directory):
        add_deferred_quality(directory)
        outcome_path = directory / "episode.json"
        outcome = json.loads(outcome_path.read_text())
        sources = outcome.pop("quality_geometry_sources")
        outcome["geometry_pending"] = True
        outcome["quality_geometry_expected_cache_sha256"] = sources["cache_sha256"]
        write(outcome_path, outcome)
        metadata = json.loads((directory / "metadata.json").read_text())
        receipt = {"contract":"cvpr-pico-quality-geometry-completion-v1", "status":"completed",
            "episode_json_sha256":hashlib.sha256(outcome_path.read_bytes()).hexdigest(),
            "metadata_sha256":hashlib.sha256((directory / "metadata.json").read_bytes()).hexdigest(),
            "human_demo_uuid":metadata["human_demo_uuid"], "source_kind":metadata["source_kind"],
            "quality_geometry_sources":sources,"accepted_new_dataset":False}
        write(directory / "quality_geometry_completion.json", receipt)
        return receipt

    def test_separate_background_receipt_binds_immutable_episode_and_geometry(self):
        directory = self.episode()
        self.background_receipt(directory)
        original = hashes(directory)
        result = self.history.remember(directory)
        self.assertEqual(result["disposition"], "kept", result)
        self.assertEqual(hashes(Path(result["path"])), original)
        invalid = self.episode(session="second")
        receipt = self.background_receipt(invalid)
        receipt["episode_json_sha256"] = "0" * 64
        write(invalid / "quality_geometry_completion.json", receipt)
        result = self.history.remember(invalid)
        self.assertEqual(result["disposition"], "discarded")
        self.assertTrue(any("回执" in reason for reason in result["reasons"]))

    def test_worker_discards_exact_older_uuid_without_changing_latest_or_pending_attempt(self):
        first = self.episode(session="first")
        a = self.history.remember(first)
        second = self.episode(session="second")
        b = self.history.remember(second)
        self.assertEqual(self.history.remember(first)["uuid"], a["uuid"])
        pending = self.episode(session="pending")
        outcome = json.loads((pending / "episode.json").read_text())
        outcome["capture_started"] = True
        write(pending / "episode.json", outcome)
        self.assertEqual(self.history.discard_previous()["status"], "error")
        result = self.history.discard_uuid(a["uuid"])
        self.assertEqual(result["uuid"], a["uuid"])
        self.assertTrue(result["discarded"])
        self.assertTrue(self.history.discard_uuid(a["uuid"])["already_discarded"])
        self.assertEqual(json.loads(self.index.read_text())["latest_uuid"], b["uuid"])
        self.assertTrue((self.root / "kept/002").is_dir())
        self.assertFalse(pending.is_symlink())

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


class MobileNativeReselectionTests(unittest.TestCase):
    """Only tiny temporary CPU fixtures; never reclassify a real collection."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index = self.root / "pico_episode_history.json"
        self.history = EpisodeHistory(self.index)

    def mobile(self, *, native=True, legacy_discard=False, technical=None):
        directory = self.root / "session" / "episodes" / str(uuid.uuid4())
        directory.mkdir(parents=True)
        make_episode(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        actual = directory.with_name(metadata["human_demo_uuid"])
        directory.rename(actual); directory = actual
        metadata.update(task_id=MOBILE_TASK_ID, quality_geometry_mode="deferred_geometry_with_original_solved_support",
            actual_task_source_sha256=MOBILE_OFFICIAL_SOURCE_SHA256, actual_native_success_criteria=.9)
        metadata.pop("initial_target_z_m")
        write(directory / "metadata.json", metadata)
        rows = [json.loads(line) for line in (directory / "quality_samples.jsonl").read_text().splitlines()]
        # No root, contact, lift or placement geometry is needed for native selection.
        rows = [{key:row[key] for key in ("completed_steps", "sim_time_s", "solver_stage")} for row in rows]
        for index, row in enumerate(rows):
            row["official_task_success"] = bool(native and index == 5)
        if technical == "missing_native":
            rows[-1].pop("official_task_success")
        (directory / "raw_quality_samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (directory / "quality_samples.jsonl").unlink()
        if technical == "force":
            force = directory / "applied_force_samples.jsonl"
            values = [json.loads(line) for line in force.read_text().splitlines()]
            values[2]["all_zero"] = False
            force.write_text("".join(json.dumps(row) + "\n" for row in values))
        if technical == "truncated":
            outcome = json.loads((directory / "episode.json").read_text())
            outcome["state"].update(status="rejected", pending_audit=False, reasons=["interrupted"])
            write(directory / "episode.json", outcome)
        if legacy_discard:
            with patch("pico_episode_history._physical_precheck", return_value=(
                    ["旧附加搬运门"], {"eligible_physics":False,"acceptance_gate_version":"legacy_mobile_extra"})):
                result = self.history.remember(directory)
        else:
            result = self.history.remember(directory)
        self.assertNotEqual(result["status"], "error", result)
        return directory, result

    def test_native_true_without_legacy_fields_or_geometry_is_kept_with_source_identity(self):
        source, result = self.mobile()
        self.assertEqual(result["disposition"], "kept", result)
        self.assertEqual(result["selection_policy"], "native_only")
        self.assertEqual(result["selection_status"], "selected")
        self.assertTrue(result["official_task_success_observed"])
        record = json.loads((source / "collection_record.json").read_text())
        self.assertEqual(record["physical_report"]["native_flag_source_file"], "raw_quality_samples.jsonl")
        self.assertEqual(record["physical_report"]["excluded_unsupervised_warmup_ticks50"], 0)
        self.assertEqual(set(record["evaluator_source_sha256"]), {"pico_mobile_quality.py", "pico_episode_history.py"})
        self.assertFalse(record["accepted_new_dataset"])
        self.assertTrue(record["physical_report"]["task_identity_validation"]["passed"])

    def test_missing_or_changed_official_source_and_threshold_are_technical_not_task_failure(self):
        from pico_episode_history import _physical_precheck
        source, _ = self.mobile()
        baseline = json.loads((source / "metadata.json").read_text())
        outcome = json.loads((source / "episode.json").read_text())
        for key, value in (("actual_task_source_sha256", None), ("actual_task_source_sha256", "a"*64),
                           ("actual_native_success_criteria", None), ("actual_native_success_criteria", .8),
                           ("actual_native_success_criteria", True)):
            metadata = dict(baseline)
            if value is None:
                metadata.pop(key)
            else:
                metadata[key] = value
            reasons, report = _physical_precheck(source, outcome, metadata)
            self.assertTrue(report["technical_blocked"], report)
            self.assertFalse(report["task_identity_validation"]["passed"])
            self.assertIsNone(report["official_task_success"])
            self.assertNotIn("官方任务成功标志未出现", reasons)

    def test_native_false_is_task_failure_and_invalid_recording_is_technical_blocked(self):
        _, failed = self.mobile(native=False)
        self.assertEqual(failed["disposition"], "discarded")
        self.assertEqual(failed["selection_status"], "official_task_not_successful")
        self.assertIn("官方任务成功标志未出现", failed["reasons"])
        for technical in ("missing_native", "truncated", "force"):
            _, blocked = self.mobile(technical=technical)
            self.assertEqual(blocked["status"], "technical_blocked", blocked)
            self.assertEqual(blocked["selection_status"], "technical_evidence_incomplete")
            self.assertIsNone(blocked["official_task_success"])
            self.assertNotIn("官方任务成功标志未出现", blocked["reasons"])

    def test_reselection_preserves_raw_files_numbers_index_receipts_and_original_alias(self):
        source, original = self.mobile(legacy_discard=True)
        _, latest = self.mobile()
        old_index = self.index.read_bytes()
        old_journal = self.history.log_path.read_bytes()
        old_record = (source / "collection_record.json").read_bytes()
        before = hashes(source)
        state = json.loads(old_index)
        snapshot = self.history.snapshot_mobile_native([source.name])
        operation = str(uuid.uuid4())
        result = self.history.reselect_mobile_native(snapshot, operation_uuid=operation)
        self.assertEqual(result["status"], "committed", result)
        self.assertEqual(result["selected_uuids"], [source.name])
        target = self.root / "kept/001"
        self.assertEqual(source.resolve(), target)
        self.assertFalse((self.root / "discarded/001").exists())
        self.assertEqual(hashes(target), before)
        archive = Path(result["archive"])
        self.assertEqual((archive / "before_index.json").read_bytes(), old_index)
        self.assertEqual((archive / "before_collection_records.jsonl").read_bytes(), old_journal)
        self.assertEqual((archive / source.name / "collection_record.before.json").read_bytes(), old_record)
        after = json.loads(self.index.read_text())
        self.assertEqual(after["next_number"], state["next_number"])
        self.assertEqual(after["latest_uuid"], latest["uuid"])
        self.assertEqual(after["entries"][latest["uuid"]], state["entries"][latest["uuid"]])
        again = self.history.reselect_mobile_native(snapshot, operation_uuid=operation)
        self.assertEqual(again, result)
        journal = [json.loads(line) for line in self.history.log_path.read_text().splitlines()]
        self.assertEqual(sum(row["operation_id"].startswith("reselect_mobile_native:") for row in journal), 1)
        self.assertEqual(self.history.remember(source)["disposition"], "kept")

    def test_changed_global_snapshot_never_silently_adds_later_record(self):
        source, _ = self.mobile(legacy_discard=True)
        snapshot = self.history.snapshot_mobile_native([source.name])
        self.mobile()
        old = self.index.read_bytes()
        result = self.history.reselect_mobile_native(snapshot, operation_uuid=str(uuid.uuid4()))
        self.assertEqual(result["status"], "error")
        self.assertIn("快照已过期", result["error"])
        self.assertEqual(self.index.read_bytes(), old)
        self.assertFalse(self.history.reselection_path.exists())
        self.assertEqual(source.resolve(), self.root / "discarded/001")

    def test_manual_discard_is_independent_and_is_not_restored_by_native_true(self):
        source, original = self.mobile()
        self.history.discard_uuid(original["uuid"])
        marker = (source / "operator_discard.json").read_bytes()
        snapshot = self.history.snapshot_mobile_native([source.name])
        self.assertTrue(snapshot["selected"][0]["manual_discard_intent"])
        result = self.history.reselect_mobile_native(snapshot, operation_uuid=str(uuid.uuid4()))
        self.assertEqual(result["status"], "committed", result)
        self.assertEqual(result["results"][0]["disposition"], "discarded")
        self.assertEqual(result["results"][0]["selection_status"], "operator_discarded")
        self.assertEqual((source / "operator_discard.json").read_bytes(), marker)
        self.assertEqual(self.history.remember(source)["disposition"], "discarded")

    def test_technical_blocked_reselection_keeps_existing_location_and_cannot_become_available(self):
        source, _ = self.mobile(legacy_discard=True, technical="missing_native")
        snapshot = self.history.snapshot_mobile_native([source.name])
        result = self.history.reselect_mobile_native(snapshot, operation_uuid=str(uuid.uuid4()))
        self.assertEqual(result["status"], "committed", result)
        self.assertEqual(source.resolve(), self.root / "discarded/001")
        self.assertEqual(result["results"][0]["selection_status"], "technical_evidence_incomplete")
        self.assertEqual(self.history.status()["status"], "technical_blocked")
        self.assertFalse(self.history.status()["available"])

    def test_previously_kept_but_now_technical_blocked_is_not_available_or_task_failed(self):
        source, original = self.mobile()
        raw = source / "raw_quality_samples.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        rows[-1].pop("official_task_success")
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        snapshot = self.history.snapshot_mobile_native([source.name])
        result = self.history.reselect_mobile_native(snapshot, operation_uuid=str(uuid.uuid4()))
        self.assertEqual(result["status"], "committed", result)
        self.assertEqual(source.resolve(), Path(original["path"]))
        view = self.history.status()
        self.assertEqual(view["status"], "technical_blocked")
        self.assertFalse(view["available"])
        self.assertIsNone(view["official_task_success"])
        self.assertNotIn("官方任务成功标志未出现", view["reasons"])

    def test_all_false_can_move_previously_kept_to_discarded_without_renumbering(self):
        source, _ = self.mobile(native=False, legacy_discard=True)
        # Explicitly simulate the old keep decision, retaining an honest complete all-false archive.
        import pico_episode_history as module
        path = source.resolve(); kept = self.root / "kept/001"; kept.parent.mkdir(exist_ok=True)
        module._rename_no_replace(path, kept)
        record = json.loads((kept / "collection_record.json").read_text())
        record.update(disposition="kept", episode_directory=str(kept)); write(kept / "collection_record.json", record)
        index = json.loads(self.index.read_text()); index["entries"][source.name].update(disposition="kept",actual_relative_path="kept/001")
        write(self.index,index); source.unlink(); source.symlink_to(kept)
        snapshot = self.history.snapshot_mobile_native([source.name])
        result = self.history.reselect_mobile_native(snapshot, operation_uuid=str(uuid.uuid4()))
        self.assertEqual(result["status"], "committed", result)
        self.assertEqual(source.resolve(), self.root / "discarded/001")
        self.assertEqual(json.loads(self.index.read_text())["next_number"], 2)

    def test_reselection_recovers_after_rename_alias_journal_and_index_write_failure(self):
        import pico_episode_history as module
        for method in ("rename", "alias", "journal", "index"):
            with self.subTest(method=method):
                source, original = self.mobile(legacy_discard=True)
                snapshot = self.history.snapshot_mobile_native([source.name])
                operation = str(uuid.uuid4())
                if method == "rename":
                    context = patch("pico_episode_history._rename_no_replace", side_effect=OSError("injected rename"))
                elif method == "alias":
                    context = patch.object(self.history, "_alias", side_effect=OSError("injected alias"))
                elif method == "journal":
                    context = patch.object(self.history, "_log", side_effect=OSError("injected journal"))
                else:
                    actual = module._atomic
                    def fail_index(path, value, **kwargs):
                        if path == self.index:
                            raise OSError("injected index")
                        return actual(path,value,**kwargs)
                    context = patch("pico_episode_history._atomic", side_effect=fail_index)
                with context:
                    failed = self.history.reselect_mobile_native(snapshot, operation_uuid=operation)
                self.assertEqual(failed["status"], "error", failed)
                self.assertTrue(self.history.reselection_path.exists())
                recovered = EpisodeHistory(self.index).status()
                self.assertNotEqual(recovered["status"], "error", recovered)
                self.assertEqual(source.resolve(), self.root / "kept" / f'{original["number"]:03d}')
                result = self.history.reselect_mobile_native(snapshot, operation_uuid=operation)
                self.assertEqual(result["status"], "committed", result)
                self.assertFalse(self.history.reselection_path.exists())

    def test_handover_and_occupied_destination_are_not_mutated(self):
        source, _ = self.mobile(legacy_discard=True)
        occupied = self.root / "kept/001"; occupied.mkdir(parents=True)
        (occupied / "unrelated.txt").write_text("preserve")
        snapshot = self.history.snapshot_mobile_native([source.name])
        result = self.history.reselect_mobile_native(snapshot, operation_uuid=str(uuid.uuid4()))
        self.assertEqual(result["status"], "error")
        self.assertEqual((occupied / "unrelated.txt").read_text(), "preserve")
        self.assertEqual(source.resolve(), self.root / "discarded/001")
        metadata = json.loads((source / "metadata.json").read_text())
        metadata["task_id"] = "simple/G1WholebodyHandoverTeleop-v0"; write(source / "metadata.json", metadata)
        self.assertEqual(self.history.snapshot_mobile_native([source.name])["status"], "error")

    def test_changed_native_evidence_after_snapshot_is_rejected_before_transaction(self):
        source, _ = self.mobile(legacy_discard=True)
        snapshot = self.history.snapshot_mobile_native([source.name])
        raw = source / "raw_quality_samples.jsonl"
        raw.write_text(raw.read_text().replace('"official_task_success": true', '"official_task_success": false'))
        result = self.history.reselect_mobile_native(snapshot, operation_uuid=str(uuid.uuid4()))
        self.assertEqual(result["status"], "error")
        self.assertFalse(self.history.reselection_path.exists())


if __name__ == "__main__":
    unittest.main()
