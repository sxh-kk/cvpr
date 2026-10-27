"""Isolated CPU fixtures exercise the queue; never touch real recordings."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pico_episode_review as review
from pico_episode_history import EpisodeHistory
from test_audit_pico_teleop_episode import make_episode, add_deferred_quality


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n")


class Crash(BaseException):
    pass


class ReviewQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index = self.root / "pico_episode_history.json"
        self.queue = review.ReviewQueue(self.index)

    def episode(self, session="session", *, synthetic=False, completed_geometry=False):
        directory = self.root / session / "episodes" / "temporary"
        directory.mkdir(parents=True)
        make_episode(directory)
        add_deferred_quality(directory)
        outcome = review._read(directory / "episode.json")
        metadata = review._read(directory / "metadata.json")
        identity = metadata["human_demo_uuid"]
        outcome.update(geometry_pending=True, capture_started=True,
            quality_geometry_expected_cache_sha256=metadata["container_geometry_cache_sha256"])
        outcome.pop("quality_geometry_sources")
        if synthetic:
            outcome["source_kind"] = "synthetic_diagnostic"
            metadata["source_kind"] = "synthetic_diagnostic"
            write(directory / "metadata.json", metadata)
        write(directory / "episode.json", outcome)
        actual = directory.with_name(identity)
        directory.rename(actual)
        if not completed_geometry:
            (actual / "quality_samples.jsonl").unlink()
        return actual

    def process(self, count=1):
        for _ in range(count):
            self.assertTrue(self.queue._process_one())

    def test_fifo_old_completion_does_not_replace_new_pending_and_seals_immutable(self):
        first, second = self.episode("a"), self.episode("b")
        originals = {path: ((path / "episode.json").read_bytes(), (path / "metadata.json").read_bytes())
                     for path in (first, second)}
        a, b = self.queue.submit(first), self.queue.submit(second)
        self.assertEqual((a["sequence"], b["sequence"]), (1, 2))
        self.assertFalse(a["available"])
        self.process()
        status = self.queue.snapshot()
        self.assertEqual(status["latest_finished_uuid"], second.name)
        self.assertEqual(status["latest_episode"]["uuid"], second.name)
        self.assertTrue(status["latest_episode"]["pending"])
        self.assertEqual(status["latest_completed"]["uuid"], first.name)
        self.process()
        self.assertEqual(self.queue.snapshot()["latest_episode"]["number"], 2)
        for path, original in originals.items():
            self.assertEqual((path / "episode.json").read_bytes(), original[0])
            self.assertEqual((path / "metadata.json").read_bytes(), original[1])
            self.assertEqual(review._read(path / "quality_geometry_completion.json")["status"], "completed")
        self.assertFalse(self.queue.snapshot()["accepted_new_dataset"])

    def test_slow_geometry_never_holds_queue_lock_or_blocks_poll_and_next_submit(self):
        first = self.episode("a")
        self.queue.submit(first)
        second = self.episode("b")
        entered, release = threading.Event(), threading.Event()
        original = review.complete_geometry
        def slow(*args):
            entered.set()
            self.assertTrue(release.wait(3))
            return original(*args)
        failures = []
        def work():
            try:
                self.queue._process_one()
            except BaseException as error:
                failures.append(error)
        with patch.object(review, "complete_geometry", slow):
            thread = threading.Thread(target=work)
            thread.start()
            self.assertTrue(entered.wait(3))
            start = time.monotonic()
            receipt = self.queue.submit(second)
            for _ in range(20):
                self.queue.snapshot()
            self.assertLess(time.monotonic() - start, .3)
            self.assertEqual(receipt["uuid"], second.name)
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(failures)
        self.assertEqual(self.queue.snapshot()["latest_finished_uuid"], second.name)

    def test_pending_and_late_discard_permanent_exact_uuid(self):
        first = self.episode("a")
        self.queue.submit(first)
        self.queue.request_discard(first.name)
        self.assertTrue(review.discard_requested(first))
        self.process()
        result = self.queue.snapshot()["latest_episode"]
        self.assertEqual(result["disposition"], "discarded")
        self.assertEqual(self.queue.snapshot()["pending_count"], 0)
        self.assertFalse(result["available"])
        self.assertFalse(self.queue.request_discard(first.name)["pending"])
        second = self.episode("b")
        self.queue.submit(second)
        self.process()
        self.queue.request_discard(second.name)
        self.assertTrue(review.discard_requested(second))
        self.process()
        self.assertEqual(self.queue.snapshot()["latest_episode"]["disposition"], "discarded")
        self.assertTrue(review.discard_requested(second))
        self.assertEqual(self.queue.submit(second)["disposition"], "discarded")
        self.assertEqual(review.ReviewQueue(self.index).submit(second)["disposition"], "discarded")

    def test_discard_legacy_without_job_or_latest_completed_none(self):
        directory = self.episode(completed_geometry=True)
        # Legacy path uses synchronous completed sources, predating any queue.
        outcome = review._read(directory / "episode.json")
        outcome.pop("geometry_pending")
        outcome["quality_geometry_sources"] = {key: review._sha(directory / name) for key, name in (
            ("raw_samples_sha256", "raw_quality_samples.jsonl"), ("cache_sha256", "container_geometry_cache.json"),
            ("completed_samples_sha256", "quality_samples.jsonl"))}
        write(directory / "episode.json", outcome)
        result = EpisodeHistory(self.index).remember(directory)
        self.assertEqual(result["disposition"], "kept", result)
        control = self.queue._load()
        control.update(latest_finished_uuid=directory.name, latest_episode=review._summary(result), latest_completed=None)
        self.queue._publish(control)
        self.queue.request_discard(directory.name)
        self.assertTrue(review.discard_requested(directory))
        self.process()
        self.assertEqual(self.queue.snapshot()["latest_episode"]["disposition"], "discarded")
        self.assertEqual(self.queue.snapshot()["pending_count"], 0)

    def test_first_queue_write_failure_restart_keeps_new_uuid_barrier_and_recovers(self):
        older = self.episode("a")
        self.queue.submit(older)
        self.process()
        newer = self.episode("b")
        original = review._atomic
        def fail_control(path, value):
            if Path(path) == self.queue.spool / "queue.json":
                raise OSError("fixture disk failure")
            return original(path, value)
        with patch.object(review, "_atomic", fail_control):
            self.assertEqual(self.queue.submit(newer)["status"], "error")
        restarted = review.ReviewQueue(self.index)
        status = restarted.snapshot()
        self.assertEqual(status["status"], "recovering")
        self.assertEqual(status["latest_finished_uuid"], newer.name)
        self.assertFalse(status["latest_episode"]["available"])
        self.assertTrue(restarted._process_one())
        self.assertEqual(restarted.snapshot()["latest_episode"]["number"], 2)

    def test_crash_after_completed_job_before_status_recovers_result_transaction(self):
        directory = self.episode()
        self.queue.submit(directory)
        original = self.queue._publish
        def crash_publish(control, **kwargs):
            if control.get("latest_completed"):
                raise Crash()
            return original(control, **kwargs)
        with patch.object(self.queue, "_publish", crash_publish):
            with self.assertRaises(Crash):
                self.queue._process_one()
        restarted = review.ReviewQueue(self.index)
        self.assertEqual(restarted.snapshot()["status"], "recovering")
        restarted._recover()
        self.assertEqual(restarted.snapshot()["pending_count"], 0)
        self.assertEqual(restarted.snapshot()["latest_episode"]["number"], 1)
        self.assertEqual(review._read(self.index)["next_number"], 2)

    def test_crash_after_quality_before_receipt_and_after_history_rename_idempotent(self):
        directory = self.episode()
        self.queue.submit(directory)
        seal = (directory / "episode.json").read_bytes()
        original_atomic = review._atomic
        def crash_receipt(path, value):
            if Path(path).name == "quality_geometry_completion.json":
                raise Crash()
            return original_atomic(path, value)
        with patch.object(review, "_atomic", crash_receipt):
            with self.assertRaises(Crash):
                self.queue._process_one()
        self.assertTrue((directory / "quality_samples.jsonl").exists())
        with patch.object(EpisodeHistory, "_alias", side_effect=Crash):
            with self.assertRaises(Crash):
                self.queue._process_one()
        self.assertFalse(directory.exists())
        self.assertTrue((self.root / "kept/001").is_dir())
        restarted = review.ReviewQueue(self.index)
        self.assertTrue(restarted._process_one())
        self.assertTrue(directory.is_symlink())
        self.assertEqual((directory / "episode.json").read_bytes(), seal)
        self.assertEqual(review._read(self.index)["next_number"], 2)
        self.assertEqual(restarted.snapshot()["pending_count"], 0)

    def test_discard_io_failure_never_marks_intent_or_job_completed(self):
        directory = self.episode()
        self.queue.submit(directory)
        self.queue.request_discard(directory.name)
        with patch.object(EpisodeHistory, "discard_uuid", return_value={"status": "error", "error": "fixture I/O"}):
            self.process()
        job_path = self.queue.spool / "jobs" / (directory.name + ".json")
        job = review._read(job_path)
        self.assertEqual(job["status"], "error")
        self.assertTrue(self.queue.snapshot()["latest_episode"]["pending"])
        self.assertFalse(review._read(self.queue.spool / "discard_requests" / (directory.name + ".json")).get("completed"))
        job["retry_after"] = 0
        write(job_path, job)
        self.process()
        self.assertEqual(self.queue.snapshot()["latest_episode"]["disposition"], "discarded")

    def test_duplicate_same_sha_idempotent_changed_seal_or_metadata_rejected(self):
        directory = self.episode()
        a = self.queue.submit(directory)
        self.assertEqual(self.queue.submit(directory)["sequence"], a["sequence"])
        metadata = directory / "metadata.json"
        metadata.write_bytes(metadata.read_bytes() + b" ")
        self.assertEqual(self.queue.submit(directory)["status"], "error")
        self.process()
        self.assertEqual(self.queue.snapshot()["latest_episode"]["status"], "error")
        self.assertFalse(self.index.exists())

    def test_geometry_cache_expected_identity_and_error_routes_discarded(self):
        directory = self.episode()
        outcome = review._read(directory / "episode.json")
        outcome["quality_geometry_expected_cache_sha256"] = "0" * 64
        write(directory / "episode.json", outcome)
        self.queue.submit(directory)
        self.process()
        result = self.queue.snapshot()["latest_episode"]
        self.assertEqual(result["disposition"], "discarded", result)
        receipt = review._read(directory / "quality_geometry_completion.json")
        self.assertEqual(receipt["status"], "error")
        self.assertTrue(any("几何" in reason for reason in result["reasons"]))

    def test_diagnostic_queue_never_human_collection_and_paths_cannot_escape(self):
        directory = self.episode(synthetic=True)
        self.assertEqual(self.queue.submit(directory)["status"], "error")
        self.assertEqual(self.queue.submit(directory, collection_enabled=False)["status"], "pending")
        self.process()
        self.assertEqual(self.queue.snapshot()["latest_episode"]["disposition"], "diagnostic")
        self.assertFalse(self.index.exists())
        self.assertFalse(directory.is_symlink())
        outside = self.root.parent / "missing-foreign-episode"
        self.assertEqual(self.queue.submit(outside)["status"], "error")
        alias = self.root / "evil"
        alias.symlink_to(directory, target_is_directory=True)
        self.assertEqual(self.queue.submit(alias)["status"], "error")

    def test_poll_reads_only_bounded_status_even_while_heavy_history_lock_is_held(self):
        directory = self.episode()
        self.queue.submit(directory)
        with self.queue._locked(), patch.object(review, "_sha", side_effect=AssertionError("no hashing")), \
                patch.object(Path, "glob", side_effect=AssertionError("no scan")):
            started = time.monotonic()
            status = self.queue.snapshot()
            self.assertLess(time.monotonic() - started, .1)
        self.assertEqual(status["latest_finished_uuid"], directory.name)
        (self.queue.spool / "status.json").write_bytes(b" " * (review.MAX_STATUS_BYTES + 1))
        self.assertEqual(self.queue.snapshot()["status"], "error")

    def test_broken_symlink_or_mismatched_marker_intent_fails_closed(self):
        directory = self.episode()
        self.queue.submit(directory)
        marker = directory / "review_queue_submission.json"
        original = review._read(marker)
        wrong = dict(original, uuid="00000000-0000-0000-0000-000000000000")
        write(marker, wrong)
        with self.assertRaises(ValueError):
            review.discard_requested(directory)
        marker.unlink()
        marker.symlink_to(directory / "missing")
        with self.assertRaises(ValueError):
            review.discard_requested(directory)
        marker.unlink()
        write(marker, original)
        intent = self.queue.spool / "discard_requests" / (directory.name + ".json")
        intent.symlink_to(self.root / "missing")
        with self.assertRaises(ValueError):
            review.discard_requested(directory)

    def test_late_intent_during_history_commit_cannot_publish_kept(self):
        directory = self.episode()
        self.queue.submit(directory)
        original = EpisodeHistory.remember
        def late_intent(history, path):
            result = original(history, path)
            self.queue.request_discard(directory.name)
            return result
        with patch.object(EpisodeHistory, "remember", late_intent):
            self.process()
        self.assertEqual(self.queue.snapshot()["latest_episode"]["disposition"], "discarded")
        self.assertEqual(self.queue.snapshot()["pending_count"], 0)
        self.assertFalse((self.root / "kept/001").exists())

    def test_old_result_and_manual_result_crash_barriers_preserve_newest_uuid(self):
        first, second = self.episode("a"), self.episode("b")
        self.queue.submit(first)
        self.queue.submit(second)
        original = self.queue._publish
        def crash_publish(control, **kwargs):
            if control.get("latest_completed"):
                raise Crash()
            return original(control, **kwargs)
        with patch.object(self.queue, "_publish", crash_publish):
            with self.assertRaises(Crash):
                self.queue._process_one()
        restarted = review.ReviewQueue(self.index)
        self.assertEqual(restarted.snapshot()["latest_finished_uuid"], second.name)
        restarted._recover()
        self.assertEqual(restarted.snapshot()["latest_finished_uuid"], second.name)
        self.queue = restarted
        self.process()
        self.queue.request_discard(first.name)
        with patch.object(self.queue, "_publish", side_effect=Crash):
            with self.assertRaises(Crash):
                self.queue._process_one()
        restarted = review.ReviewQueue(self.index)
        self.assertEqual(restarted.snapshot()["latest_finished_uuid"], second.name)
        restarted._recover()
        self.assertEqual(restarted.snapshot()["latest_finished_uuid"], second.name)
        self.assertEqual(restarted.snapshot()["pending_count"], 0)
        self.assertFalse((self.root / "kept/001").exists())
        self.assertTrue((self.root / "kept/002").exists())

    def test_startup_recovers_sealed_attempt_when_even_first_barrier_write_failed(self):
        first = self.episode("a")
        self.queue.submit(first)
        self.process()
        newer = self.episode("b")
        original = review._atomic
        def fail_first(path, value):
            if Path(path).name == "recovery_required.json":
                raise OSError("fixture first write failed")
            return original(path, value)
        with patch.object(review, "_atomic", fail_first):
            self.assertEqual(self.queue.submit(newer)["status"], "error")
        self.assertFalse((self.queue.spool / "jobs" / (newer.name + ".json")).exists())
        # start makes its cached pointer unavailable until the worker's scan.
        with patch.object(subprocess, "Popen") as popen:
            popen.return_value.poll.return_value = None
            self.queue.start()
        self.assertIsNone(self.queue.snapshot()["latest_finished_uuid"])
        self.assertFalse(self.queue.snapshot()["latest_episode"]["available"])
        restarted = review.ReviewQueue(self.index)
        restarted._discover_orphans()
        self.assertEqual(restarted.snapshot()["latest_finished_uuid"], newer.name)
        self.assertTrue(restarted._process_one())
        self.assertEqual(restarted.snapshot()["latest_episode"]["number"], 2)

    def test_recovered_older_orphan_never_repoints_newer_latest_and_is_numbered_first(self):
        older = self.episode("old")
        original = review._atomic
        def fail_first(path, value):
            if Path(path).name == "recovery_required.json":
                raise OSError("fixture first write failed")
            return original(path, value)
        with patch.object(review, "_atomic", fail_first):
            self.assertEqual(self.queue.submit(older)["status"], "error")
        newer = self.episode("new")
        self.queue.submit(newer)
        self.queue._discover_orphans()
        self.assertEqual(self.queue.snapshot()["latest_finished_uuid"], newer.name)
        self.process()
        self.assertEqual(self.queue.snapshot()["latest_finished_uuid"], newer.name)
        self.assertEqual(review._read(older / "collection_record.json")["number"], 1)
        self.process()
        self.assertEqual(self.queue.snapshot()["latest_episode"]["number"], 2)
        self.assertEqual(self.queue.snapshot()["latest_finished_uuid"], newer.name)

    def test_older_submit_barrier_preserves_completed_manually_discarded_newest(self):
        older = self.episode("old")
        outcome = review._read(older / "episode.json")
        outcome["capture_started"] = False  # Legacy record, explicitly submitted later.
        write(older / "episode.json", outcome)
        newer = self.episode("new")
        self.queue.submit(newer)
        self.process()
        self.queue.request_discard(newer.name)
        self.process()
        self.assertEqual(self.queue.snapshot()["latest_episode"]["disposition"], "discarded")
        original = review._atomic
        def fail_control(path, value):
            if Path(path).name == "queue.json":
                raise OSError("fixture control write failure")
            return original(path, value)
        with patch.object(review, "_atomic", fail_control):
            self.assertEqual(self.queue.submit(older)["status"], "error")
        self.assertEqual(review.ReviewQueue(self.index).snapshot()["latest_finished_uuid"], newer.name)

    def test_real_cpu_worker_subprocess_priority_single_core_and_singleton(self):
        directory = self.episode()
        self.queue.submit(directory)
        environment = dict(os.environ, PYTHONNOUSERSITE="1")
        args = [sys.executable, str(Path(review.__file__)), "--worker", "--once",
                "--index", str(self.index), "--spool", str(self.queue.spool)]
        with (self.queue.spool / "worker.lock").open("a+b") as stream:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
            child = subprocess.run(args, env=environment, capture_output=True, text=True, timeout=15)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertFalse(self.index.exists())
        child = subprocess.run(args, env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr)
        status = self.queue.snapshot()
        self.assertEqual(status["pending_count"], 0, status)
        self.assertEqual(status["worker"]["nice"], 19)
        self.assertEqual(len(status["worker"]["cpus"]), 1)
        self.assertEqual(status["worker"]["threads_per_library"], 1)
        self.assertTrue(status["worker"]["cpu_only"])
        self.assertFalse(status["worker"]["ready"])


    def _queue_file_bytes(self):
        return {str(path.relative_to(self.queue.spool)): path.read_bytes()
                for path in self.queue.spool.rglob("*") if path.is_file()}

    def _new_selection_view(self, identity):
        index = review._read(self.index)
        index["entries"][identity].update(selection_policy="native_only", selection_status="selected",
            official_task_success_observed=True, acceptance_gate_version="mobile-native-task-success-v2")
        write(self.index, index)

    def _owned_worker_command(self, *, index=None):
        return b"\0".join(os.fsencode(value) for value in (sys.executable,
            str(Path(review.__file__).resolve()), "--worker", "--index", str(index or self.index),
            "--spool", str(self.queue.spool))) + b"\0"

    def test_rotate_only_at_job_boundary_holds_both_locks_and_keeps_pending_job(self):
        directory = self.episode()
        self.queue.submit(directory)
        control = self.queue._load()
        control["worker"] = {"pid": 123456, "ready": True, "cpu_only": True}
        self.queue._publish(control)
        job_path = self.queue.spool / "jobs" / (directory.name + ".json")
        original_job, original_seal = job_path.read_bytes(), (directory / "episode.json").read_bytes()
        def terminate(pid):
            self.assertEqual(pid, 123456)
            for path in (self.queue.spool / "queue.lock", EpisodeHistory(self.index).lock_path):
                with path.open("a+b") as lock:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        def start():
            for path in (self.queue.spool / "queue.lock", EpisodeHistory(self.index).lock_path):
                with path.open("a+b") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(lock, fcntl.LOCK_UN)
            return self.queue
        self.queue._process = object()
        with patch.object(self.queue, "_terminate_owned_worker", side_effect=terminate) as kill, \
                patch.object(self.queue, "start", side_effect=start) as restart:
            result = self.queue.rotate_worker_at_boundary()
        self.assertTrue(result["rotated"])
        self.assertFalse(result["collector_modified"])
        self.assertEqual(result["previous_worker"]["pid"], 123456)
        kill.assert_called_once_with(123456); restart.assert_called_once_with()
        self.assertIsNone(self.queue._process)
        self.assertEqual(job_path.read_bytes(), original_job)
        self.assertEqual((directory / "episode.json").read_bytes(), original_seal)
        self.assertEqual(self.queue.snapshot()["latest_finished_uuid"], directory.name)
        self.assertEqual(self.queue.snapshot()["pending_count"], 1)
        self.assertFalse(self.queue.snapshot()["worker"]["ready"])
        self.assertTrue(self.queue.snapshot()["worker"]["reload_requested"])

    def test_rotate_with_running_job_never_signals_or_mutates(self):
        directory = self.episode(); self.queue.submit(directory)
        job_path = self.queue.spool / "jobs" / (directory.name + ".json")
        job = review._read(job_path); job["status"] = "running"; write(job_path, job)
        before = self._queue_file_bytes()
        with patch.object(self.queue, "_terminate_owned_worker") as kill, \
                patch.object(self.queue, "start") as restart, \
                patch.object(EpisodeHistory, "_locked", side_effect=AssertionError("must not enter History")):
            result = self.queue.rotate_worker_at_boundary()
        self.assertEqual(result, {"rotated": False, "reason": "active_review_job"})
        kill.assert_not_called(); restart.assert_not_called()
        self.assertEqual(self._queue_file_bytes(), before)

    def test_rotate_recovery_and_startup_barriers_never_signal(self):
        directory = self.episode(); self.queue.submit(directory)
        for name in ("recovery_required.json", "startup_scan_required.json"):
            with self.subTest(name=name):
                barrier = self.queue.spool / name
                write(barrier, {"fixture": True})
                before = self._queue_file_bytes()
                with patch.object(self.queue, "_terminate_owned_worker") as kill, \
                        patch.object(self.queue, "start") as restart:
                    result = self.queue.rotate_worker_at_boundary()
                self.assertEqual(result, {"rotated": False, "reason": "queue_recovery_pending"})
                kill.assert_not_called(); restart.assert_not_called()
                self.assertEqual(self._queue_file_bytes(), before)
                barrier.unlink()

    def test_rotate_can_retire_without_starting_a_worker(self):
        with (self.queue.spool / "worker.lock").open("a+b") as old_worker:
            fcntl.flock(old_worker, fcntl.LOCK_EX)
            with patch.object(self.queue, "_terminate_owned_worker") as kill, \
                    patch.object(self.queue, "start") as restart, \
                    patch.object(review.time, "monotonic", side_effect=AssertionError("must not wait")):
                result = self.queue.rotate_worker_at_boundary(start_new=False)
        self.assertTrue(result["rotated"])
        kill.assert_not_called(); restart.assert_not_called()

    def test_rotate_waits_for_old_singleton_outside_data_locks_before_start(self):
        with (self.queue.spool / "worker.lock").open("a+b") as old_worker:
            fcntl.flock(old_worker, fcntl.LOCK_EX)
            def release_old_lock(_seconds):
                restart.assert_not_called()
                for path in (self.queue.spool / "queue.lock", EpisodeHistory(self.index).lock_path):
                    with path.open("a+b") as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(lock, fcntl.LOCK_UN)
                fcntl.flock(old_worker, fcntl.LOCK_UN)
            with patch.object(self.queue, "_terminate_owned_worker") as kill, \
                    patch.object(self.queue, "start") as restart, \
                    patch.object(review.time, "sleep", side_effect=release_old_lock) as wait:
                result = self.queue.rotate_worker_at_boundary()
        self.assertTrue(result["rotated"])
        kill.assert_not_called(); restart.assert_called_once_with()
        wait.assert_called_once_with(.02)

    def test_rotate_singleton_release_timeout_does_not_start_replacement(self):
        with (self.queue.spool / "worker.lock").open("a+b") as old_worker:
            fcntl.flock(old_worker, fcntl.LOCK_EX)
            with patch.object(self.queue, "_terminate_owned_worker") as kill, \
                    patch.object(self.queue, "start") as restart, \
                    patch.object(review.time, "monotonic", side_effect=(0., 3.)), \
                    patch.object(review.time, "sleep") as wait:
                with self.assertRaisesRegex(TimeoutError, "replacement not started"):
                    self.queue.rotate_worker_at_boundary()
        kill.assert_not_called(); restart.assert_not_called(); wait.assert_not_called()

    def test_signal_helper_binds_pidfd_and_checks_owned_exact_command_arguments(self):
        pid, descriptor = 123456, 987654
        with patch.object(review.os, "pidfd_open", return_value=descriptor, create=True) as opened, \
                patch.object(review.signal, "pidfd_send_signal", create=True) as signal_worker, \
                patch.object(review.os, "kill") as kill, patch.object(review.os, "close") as close, \
                patch.object(Path, "read_bytes", return_value=self._owned_worker_command()) as read:
            self.assertTrue(self.queue._terminate_owned_worker(pid))
        opened.assert_called_once_with(pid)
        signal_worker.assert_called_once_with(descriptor, review.signal.SIGTERM)
        kill.assert_not_called(); close.assert_called_once_with(descriptor)
        read.assert_called_once_with()

    def test_signal_helper_refuses_other_queue_or_collector_without_signaling(self):
        commands = (self._owned_worker_command(index=self.root / "other.json"),
                    self._owned_worker_command().replace(b"--worker", b"--collector"),
                    self._owned_worker_command().replace(os.fsencode(Path(review.__file__).resolve()), b"collector.py"))
        for command in commands:
            with self.subTest(command=command), \
                    patch.object(review.os, "pidfd_open", return_value=987654, create=True), \
                    patch.object(review.signal, "pidfd_send_signal", create=True) as signal_worker, \
                    patch.object(review.os, "kill") as kill, patch.object(review.os, "close") as close, \
                    patch.object(Path, "read_bytes", return_value=command):
                with self.assertRaises(ValueError): self.queue._terminate_owned_worker(123456)
                signal_worker.assert_not_called(); kill.assert_not_called()
                close.assert_called_once_with(987654)
        for pid in (None, True, 0, 1, -1, "123456"):
            with self.subTest(pid=pid), patch.object(review.os, "pidfd_open", create=True) as opened:
                with self.assertRaises(ValueError): self.queue._terminate_owned_worker(pid)
                opened.assert_not_called()

    def test_signal_helper_missing_worker_is_safe_without_fallback_signal(self):
        with patch.object(review.os, "pidfd_open", side_effect=ProcessLookupError, create=True), \
                patch.object(review.signal, "pidfd_send_signal", create=True) as signal_worker, \
                patch.object(review.os, "kill") as kill:
            self.assertFalse(self.queue._terminate_owned_worker(123456))
        signal_worker.assert_not_called(); kill.assert_not_called()

    def test_refresh_updates_same_uuid_from_history_and_preserves_new_pending_latest(self):
        first = self.episode("a"); self.queue.submit(first); self.process()
        second = self.episode("b"); self.queue.submit(second)
        self._new_selection_view(first.name)
        index_before, original_files, status_before = self.index.read_bytes(), self._queue_file_bytes(), self.queue.snapshot()
        seals = {name: (first / name).read_bytes() for name in
                 ("episode.json", "metadata.json", "collection_record.json", "quality_geometry_completion.json")}
        revision = str(uuid.uuid4())
        result = self.queue.refresh_reselection_results([first.name], revision)
        self.assertTrue(result["refreshed"])
        self.assertFalse(result["accepted_new_dataset"])
        self.assertEqual(result["revision_id"], revision)
        self.assertEqual(result["updated_results"], 1)
        after = self.queue.snapshot()
        self.assertEqual(after["latest_finished_uuid"], second.name)
        self.assertEqual(after["latest_episode"], status_before["latest_episode"])
        self.assertEqual(after["pending_uuids"], status_before["pending_uuids"])
        self.assertEqual(after["latest_completed"]["uuid"], first.name)
        self.assertEqual(after["latest_completed"]["number"], 1)
        self.assertEqual(after["latest_completed"]["sequence"], 1)
        self.assertEqual(after["latest_completed"]["selection_policy"], "native_only")
        self.assertEqual(after["latest_completed"]["selection_status"], "selected")
        self.assertTrue(after["latest_completed"]["official_task_success_observed"])
        self.assertEqual(after["latest_completed"]["acceptance_gate_version"], "mobile-native-task-success-v2")
        self.assertEqual(self.queue._load()["next_sequence"], 3)
        self.assertEqual(self.index.read_bytes(), index_before)
        for name, before in seals.items(): self.assertEqual((first / name).read_bytes(), before)
        self.assertEqual((self.queue.spool / "jobs" / (second.name + ".json")).read_bytes(),
                         original_files["jobs/" + second.name + ".json"])
        archive = self.queue.spool / "reselections" / revision
        for name in ("queue.json", "status.json", "jobs/" + first.name + ".json"):
            self.assertEqual((archive / name).read_bytes(), original_files[name])
        self.assertFalse((archive / "jobs" / (second.name + ".json")).exists())
        self.assertEqual(review._read(archive / "refresh_complete.json"), result)
        job = review._read(self.queue.spool / "jobs" / (first.name + ".json"))
        self.assertEqual(job["reselection_revision"], revision)
        self.assertEqual(job["result"], after["latest_completed"])

    def test_refresh_latest_completed_updates_all_caches_and_is_idempotent(self):
        directory = self.episode(); self.queue.submit(directory); self.process()
        self._new_selection_view(directory.name)
        revision = str(uuid.uuid4())
        self.queue.refresh_reselection_results([directory.name], revision)
        status = self.queue.snapshot()
        self.assertEqual(status["latest_episode"], status["latest_completed"])
        self.assertEqual(status["latest_results"][-1], status["latest_completed"])
        before = self._queue_file_bytes()
        with patch.object(self.queue, "_publish", side_effect=AssertionError("no rewrite")), \
                patch.object(EpisodeHistory, "_view", side_effect=AssertionError("already refreshed")):
            result = self.queue.refresh_reselection_results([directory.name], revision)
        self.assertTrue(result["already_refreshed"])
        self.assertEqual(self._queue_file_bytes(), before)
        with self.assertRaises(ValueError): self.queue.refresh_reselection_results([str(uuid.uuid4())], revision)
        self.assertEqual(self._queue_file_bytes(), before)

    def test_refresh_rejects_pending_running_and_missing_targets_before_any_mutation(self):
        first = self.episode("a"); self.queue.submit(first); self.process()
        second = self.episode("b"); self.queue.submit(second)
        job_path = self.queue.spool / "jobs" / (second.name + ".json")
        for state in ("pending", "running", "missing"):
            with self.subTest(state=state):
                if state == "missing": job_path.unlink()
                else:
                    job = review._read(job_path); job["status"] = state; write(job_path, job)
                before = self._queue_file_bytes()
                with patch.object(EpisodeHistory, "_view", side_effect=AssertionError("preflight first")):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        self.queue.refresh_reselection_results([first.name, second.name], str(uuid.uuid4()))
                self.assertEqual(self._queue_file_bytes(), before)
                self.assertFalse((self.queue.spool / "reselections").exists())

    def test_refresh_rejects_noncanonical_or_duplicate_identity_and_recovery_barrier(self):
        directory = self.episode(); self.queue.submit(directory); self.process()
        for identities, revision in (([], str(uuid.uuid4())), ([directory.name, directory.name], str(uuid.uuid4())),
                                     (["bad-uuid"], str(uuid.uuid4())), ([directory.name], "bad-revision")):
            before = self._queue_file_bytes()
            with self.assertRaises(ValueError): self.queue.refresh_reselection_results(identities, revision)
            self.assertEqual(self._queue_file_bytes(), before)
        write(self.queue.spool / "recovery_required.json", {"fixture": True})
        before = self._queue_file_bytes()
        with self.assertRaises(ValueError):
            self.queue.refresh_reselection_results([directory.name], str(uuid.uuid4()))
        self.assertEqual(self._queue_file_bytes(), before)

    def test_refresh_rejects_completed_job_with_pending_operator_discard_without_mutation(self):
        directory = self.episode(); self.queue.submit(directory); self.process()
        self.queue.request_discard(directory.name)
        self.assertEqual(self.queue.snapshot()["latest_episode"]["status"], "pending_discard")
        self.assertEqual(review._read(self.queue.spool / "jobs" / (directory.name + ".json"))["status"], "completed")
        before = self._queue_file_bytes()
        with self.assertRaises(ValueError):
            self.queue.refresh_reselection_results([directory.name], str(uuid.uuid4()))
        self.assertEqual(self._queue_file_bytes(), before)
        self.assertFalse((self.queue.spool / "reselections").exists())
        self.assertTrue(review.discard_requested(directory))

    def test_refresh_crash_recovers_same_jobs_without_renumbering_or_overwriting_old_snapshots(self):
        directory = self.episode(); self.queue.submit(directory); self.process()
        self._new_selection_view(directory.name)
        index_before, original_files = self.index.read_bytes(), self._queue_file_bytes()
        revision = str(uuid.uuid4())
        with patch.object(self.queue, "_publish", side_effect=Crash):
            with self.assertRaises(Crash):
                self.queue.refresh_reselection_results([directory.name], revision)
        self.assertTrue((self.queue.spool / "recovery_required.json").exists())
        restarted = review.ReviewQueue(self.index)
        self.assertEqual(restarted.snapshot()["status"], "recovering")
        with patch.object(restarted, "_publish", side_effect=Crash):
            with self.assertRaises(Crash): restarted._recover()
        with self.assertRaises(ValueError):
            restarted.refresh_reselection_results([directory.name], revision)
        restarted._recover()
        self.assertFalse((restarted.spool / "recovery_required.json").exists())
        self.assertEqual(restarted.snapshot()["latest_episode"]["selection_policy"], "native_only")
        self.assertEqual(restarted.snapshot()["latest_episode"]["number"], 1)
        self.assertEqual(review._read(self.index)["next_number"], 2)
        self.assertEqual(self.index.read_bytes(), index_before)
        archive = restarted.spool / "reselections" / revision
        for name in ("queue.json", "status.json", "jobs/" + directory.name + ".json"):
            self.assertEqual((archive / name).read_bytes(), original_files[name])
        self.assertTrue(restarted.refresh_reselection_results([directory.name], revision)["already_refreshed"])

    def test_mobile_geometry_is_not_recomputed_or_required_and_old_receipt_is_preserved(self):
        directory = self.episode()
        metadata = review._read(directory / "metadata.json")
        metadata["task_id"] = "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
        write(directory / "metadata.json", metadata)
        receipt_path = directory / "quality_geometry_completion.json"
        old_receipt = b'{"status":"error","error":"preserved old extra geometry failure"}\n'
        receipt_path.write_bytes(old_receipt)
        self.queue.submit(directory)
        job = review._read(self.queue.spool / "jobs" / (directory.name + ".json"))
        (directory / "container_geometry_cache.json").unlink()
        with patch("pico_simple_teleop.complete_quality_geometry", side_effect=AssertionError("no extra geometry")):
            result = review.complete_geometry(directory, job)
        self.assertEqual(result["status"], "not_required")
        self.assertEqual(result["selection_policy"], "native_only")
        self.assertEqual(receipt_path.read_bytes(), old_receipt)
        self.assertFalse((directory / "quality_samples.jsonl").exists())
        metadata["task_id"] = "changed-after-seal"
        write(directory / "metadata.json", metadata)
        with self.assertRaises(ValueError): review.complete_geometry(directory, job)

    def test_nonpending_geometry_still_verifies_metadata_sha_before_short_circuit(self):
        for mobile in (True, False):
            with self.subTest(mobile=mobile):
                directory = self.episode("nonpending-" + str(mobile))
                outcome = review._read(directory / "episode.json")
                outcome["geometry_pending"] = False
                write(directory / "episode.json", outcome)
                metadata = review._read(directory / "metadata.json")
                if mobile:
                    metadata["task_id"] = "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
                    write(directory / "metadata.json", metadata)
                self.queue.submit(directory)
                job = review._read(self.queue.spool / "jobs" / (directory.name + ".json"))
                with patch("pico_simple_teleop.complete_quality_geometry", side_effect=AssertionError("no geometry")):
                    result = review.complete_geometry(directory, job)
                if mobile:
                    self.assertEqual(result["status"], "not_required")
                else:
                    self.assertIsNone(result)
                metadata["tampered_after_submission"] = True
                write(directory / "metadata.json", metadata)
                with self.assertRaisesRegex(ValueError, "metadata SHA"):
                    review.complete_geometry(directory, job)


if __name__ == "__main__":
    unittest.main()
