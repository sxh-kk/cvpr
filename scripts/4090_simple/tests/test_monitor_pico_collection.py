"""Standard-library CPU tests; no collector, SDK, simulation or GPU startup."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monitor_pico_collection as monitor


def fake_stat(pid, parent=1, start=100, ticks=10):
    values = ["0"] * 40
    for key, value in {0: "S", 1: parent, 11: ticks, 12: 0, 16: 19, 17: 1,
                       19: start, 21: 10, 36: 26}.items():
        values[key] = str(value)
    return f"{pid} (name ) with spaces) " + " ".join(values)


def fake_proc(root, pid, parent=1, start=100, children="", collector=False):
    base = root / str(pid); (base / "task" / str(pid)).mkdir(parents=True, exist_ok=True)
    (base / "stat").write_text(fake_stat(pid, parent, start))
    (base / "task" / str(pid) / "children").write_text(children)
    (base / "status").write_text("voluntary_ctxt_switches:\t5\nnonvoluntary_ctxt_switches:\t2\n")
    (base / "io").write_text("rchar: 10\nwrite_bytes: 100\n")
    (base / "schedstat").write_text("100 200 1\n")
    (base / "cmdline").write_bytes(b"python\0/root/pico_simple_teleop.py\0" if collector else b"bash\0")


class MonitorTests(unittest.TestCase):
    def test_urls_are_local_read_only_status_only(self):
        for url in ("http://127.0.0.1:43100/status.json", "http://localhost:43100/status.json",
                    "http://[::1]:43100/status.json"):
            self.assertEqual(monitor.validate_url(url), url)
        for url in ("https://127.0.0.1:43100/status.json", "http://example.com:80/status.json",
                    "http://127.0.0.1:43100/command", "http://127.0.0.1:43100/status.json?command=quit",
                    "http://user:password@127.0.0.1:43100/status.json"):
            with self.assertRaises(ValueError):
                monitor.validate_url(url)

    def test_file_cache_reads_only_changes_and_rejects_oversize_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / "result.json"; path.write_text('{"a":1}')
            cache = monitor.ChangedJSON(max_bytes=32)
            self.assertEqual(cache.read(path), ({"a": 1}, None))
            with patch.object(monitor.os, "open", side_effect=AssertionError("unchanged file must not reopen")):
                self.assertEqual(cache.read(path), ({"a": 1}, None))
            path.write_text('{"a":22}')
            self.assertEqual(cache.read(path), ({"a": 22}, None))
            path.write_bytes(b" " * 33)
            self.assertIn("size_limit", cache.read(path)[1])
            link = root / "link"; link.symlink_to(path)
            self.assertEqual(cache.read(link)[1], "symlink_rejected")

    def test_sample_rotation_is_three_bounded_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); log = monitor.RotatingJSONL(root / "samples.jsonl", max_bytes=48)
            for i in range(20):
                log.append({"value": i})
            files = list(root.iterdir())
            self.assertEqual(len(files), 3)
            self.assertTrue(all(path.stat().st_size <= 48 for path in files))
            self.assertEqual(json.loads(log.path.read_text().splitlines()[-1]), {"value": 19})

    def test_owned_process_tree_pid_identity_and_cpu_io_context(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake_proc(root, 100, children="101 102")
            fake_proc(root, 101, parent=100, start=101, collector=True)
            # A stale child PID now owned by somebody else must not be adopted.
            fake_proc(root, 102, parent=999, start=102)
            family = monitor.ProcessFamily(100, root)
            first = family.sample(10.)
            self.assertTrue(first["launcher_alive"])
            self.assertEqual({p["pid"] for p in first["processes"]}, {100, 101})
            child = next(p for p in first["processes"] if p["pid"] == 101)
            self.assertTrue(child["is_collector"])
            self.assertEqual(child["status"]["nonvoluntary_ctxt_switches"], 2)
            self.assertEqual(child["io"]["write_bytes"], 100)
            (root / "101/stat").write_text(fake_stat(101, parent=100, start=101, ticks=30))
            second = family.sample(11.)
            self.assertGreater(next(p for p in second["processes"] if p["pid"] == 101)["cpu_percent"], 0)
            (root / "101/stat").write_text(fake_stat(101, parent=999, start=999))
            self.assertNotIn(101, {p["pid"] for p in family.sample(12.)["processes"]})

    def test_no_tracking_and_startup_device_gaps_are_not_faults_and_deduplicate(self):
        detector = monitor.Events()
        status = {"status": "paused", "tracking": {"valid": False, "reasons": ["no_tracking_packet"]},
            "pico_receive_telemetry": {"recent_receive_gaps_ge_50ms": [{"sequence": 10, "host_monotonic_s": 1., "gap_s": 9.}],
                "last_device_event": {"kind": "device", "status": "offline", "host_monotonic_s": 2.}}}
        first = detector.observe(status, {}, "/round/live")
        self.assertEqual(len(first), 2)
        self.assertTrue(all(row["severity"] == "diagnostic" for row in first))
        self.assertEqual(detector.observe(status, {}, "/round/live"), [])

    def test_historical_review_is_not_this_session_fault_and_real_rejection_is_distinct(self):
        detector = monitor.Events()
        old = {"status": "technical_blocked", "uuid": "old", "source_path": "/round/old/episodes/old",
            "selection_status": "technical_evidence_incomplete"}
        self.assertEqual(detector.observe({"last_episode": old}, {}, "/round/new"), [])
        new = {**old, "uuid": "new", "source_path": "/round/new/episodes/new"}
        self.assertEqual(detector.observe({"last_episode": new}, {}, "/round/new")[0]["kind"], "review_technical")
        result = {"rejections": [{"reason": "lag=.101", "tick50": 5}]}
        self.assertEqual(len(detector.observe({}, result, "/round/new")), 1)
        self.assertEqual(detector.observe({}, result, "/round/new"), [])
        result["rejections"].append({"reason": "lag=.101", "tick50": 6})
        self.assertEqual(len(detector.observe({}, result, "/round/new")), 1)

    def test_status_reader_total_wait_single_flight_and_late_response_discard(self):
        release, calls = threading.Event(), []
        def slow(url, timeout):
            calls.append(url); release.wait(1.)
            return {"status": {"status": "paused"}}
        reader = monitor.StatusReader("http://localhost:43100/status.json", .02)
        try:
            with patch.object(monitor, "get_status", side_effect=slow):
                start = time.monotonic()
                with self.assertRaises(TimeoutError):
                    reader.read()
                self.assertLess(time.monotonic() - start, .3)
                with self.assertRaisesRegex(TimeoutError, "outstanding"):
                    reader.read()
                self.assertEqual(len(calls), 1)
                release.set(); reader.thread.join(1.)
            with patch.object(monitor, "get_status", return_value={"status": {"status": "following"}}):
                self.assertEqual(reader.read()["status"]["status"], "following")
        finally:
            release.set(); reader.thread.join(1.)

    def make_monitor(self, root):
        session = root / "collection/live"; session.mkdir(parents=True)
        value = monitor.Monitor(round_root=root, launcher_pid=os.getpid(), output=root / "monitor",
            session_output=session, interface="lo")
        return value, session

    def test_singleton_and_original_output_guard_do_not_modify_raw_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); value, session = self.make_monitor(root)
            try:
                with self.assertRaisesRegex(RuntimeError, "already owns"):
                    monitor.Monitor(round_root=root, launcher_pid=os.getpid(), output=root / "another")
            finally:
                value.close()
            (session / "result.json").write_text("original evidence")
            with self.assertRaisesRegex(ValueError, "original evidence"):
                monitor.Monitor(round_root=root, launcher_pid=os.getpid(), output=session)
            self.assertEqual((session / "result.json").read_text(), "original evidence")
            with self.assertRaisesRegex(ValueError, "inside the original"):
                monitor.Monitor(round_root=root, launcher_pid=os.getpid(), output=session / "new_monitor",
                    session_output=session)
            self.assertFalse((session / "new_monitor").exists())

    def test_fault_snapshot_ring_uses_preceding60s_and_shutdown_does_not_signal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); value, session = self.make_monitor(root)
            status = {"status": {"output": str(session), "status": "paused",
                "tracking": {"reasons": ["no_tracking_packet"]}}}
            try:
                with patch.object(value.reader, "read", return_value=status), \
                     patch.object(monitor, "system_sample", return_value={}):
                    for i in range(62):
                        self.assertIsNone(value.poll(float(i)))
                    self.assertFalse((value.output / "snapshot.json").exists())
                    status["status"]["error"] = "Cannot maintain real time: lag=.102s"
                    value.poll(62.)
                    receipt = json.loads((value.output / "snapshot.json").read_text())
                    self.assertEqual(receipt["samples"][0]["monotonic_s"], 2.)
                    self.assertEqual(len(receipt["samples"]), 61)
                    value.poll(63.)
                    self.assertEqual(len((value.output / "events.jsonl").read_text().splitlines()), 1)
                with patch.object(monitor.os, "kill", side_effect=AssertionError("must never signal collector")):
                    value.close("monitor_signal")
                self.assertTrue(json.loads((value.output / "latest.json").read_text())["monitor_stopped"])
            finally:
                value.close()

    def test_collector_exit_grace_and_complete_round_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); value, session = self.make_monitor(root)
            status = {"status": {"output": str(session), "status": "stopped"}}
            try:
                with patch.object(value.reader, "read", return_value=status), \
                     patch.object(monitor, "system_sample", return_value={}):
                    self.assertIsNone(value.poll(1.))
                    self.assertIsNone(value.poll(30.9))
                    self.assertEqual(value.poll(31.), "collector_exited_30s")
                    status["status"]["mobile_collection_progress"] = {"status": "complete", "pending_uuid": None}
                    self.assertEqual(value.poll(32.), "round_complete")
            finally:
                value.close()

    def test_wrong_http_session_cannot_override_explicit_output_or_create_old_faults(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); value, session = self.make_monitor(root)
            foreign = {"ui_session_id": "old", "status": {"output": str(root / "collection/old"),
                "status": "stopped", "error": "historical error", "mobile_collection_progress": {"status": "complete"}}}
            try:
                with patch.object(value.reader, "read", return_value=foreign), patch.object(monitor, "system_sample", return_value={}):
                    self.assertIsNone(value.poll(1.))
                    self.assertEqual(value.session, session)
                    self.assertEqual([row["kind"] for row in value.latest["events"]], ["status_session_mismatch"])
                    self.assertTrue(all(row["severity"] == "diagnostic" for row in value.latest["events"]))
                    self.assertIsNone(value.exit_since)
                current = {"ui_session_id": "new", "status": {"output": str(session), "status": "paused"}}
                with patch.object(value.reader, "read", return_value=current), patch.object(monitor, "system_sample", return_value={}):
                    value.poll(2.)
                    current["status"] = {"status": "resetting"}
                    value.poll(3.)
                    self.assertEqual(value.session, session)
                    self.assertTrue(value.latest["http"]["accepted_session"])
                    self.assertEqual(value.latest["events"], [])
            finally:
                value.close()

    def test_dynamic_owned_session_output_conflict_stops_before_sampling_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); active = root / "collection/live"; active.mkdir(parents=True)
            value = monitor.Monitor(round_root=root, launcher_pid=os.getpid(), output=active / "monitor")
            status = {"ui_session_id": "owned", "status": {"output": str(active), "status": "paused"}}
            family = {"launcher_alive": True, "processes": [{"pid": 123, "state": "S", "is_collector": True,
                "collector_output": str(active)}]}
            try:
                with patch.object(value.reader, "read", return_value=status), \
                     patch.object(value.family, "sample", return_value=family):
                    with self.assertRaises(monitor.MonitorOutputConflict):
                        value.poll(1.)
                self.assertFalse((value.output / "samples.jsonl").exists())
                self.assertFalse((value.output / "latest.json").exists())
            finally:
                value.close()


if __name__ == "__main__":
    unittest.main()
