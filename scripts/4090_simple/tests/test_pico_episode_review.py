"""Isolated CPU fixtures exercise the queue; never touch real recordings."""
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


if __name__ == "__main__":
    unittest.main()
