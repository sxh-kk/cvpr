"""Offline monitoring checks: throughput, reconstruction, ETA and dead processes."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import monitor_egosuite as monitor


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.state = {
            "schema_version": 1, "run_id": "first", "pid": 123,
            "phase": "downloading", "network_bytes": 0, "reconstructed_bytes": 0,
            "total_bytes": 1000, "total_files": 2, "completed_files": 0,
            "completed_hours": 0, "selected_hours": 2, "active_files": [], "updated_at": 1000,
        }
        self.network, self.reconstruction = monitor.RateTracker(), monitor.RateTracker()

    def write(self, **changes):
        self.state.update(changes)
        path = self.output / ".egosuite-download/progress.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state))

    def snapshot(self, now):
        return monitor.snapshot(self.output, self.network, self.reconstruction, now=now, wall_now=1000)

    @patch.object(monitor, "process_alive", return_value=True)
    def test_speed_uses_network_bytes_while_progress_and_eta_use_reconstruction(self, alive):
        self.write()
        self.assertIsNone(self.snapshot(100)["network_speed_bytes_per_second"])
        self.write(network_bytes=200, reconstructed_bytes=500)
        report = self.snapshot(102)
        self.assertEqual(report["network_speed_bytes_per_second"], 100)
        self.assertEqual(report["average_network_bytes_per_second"], 100)
        self.assertEqual(report["reconstruction_speed_bytes_per_second"], 250)
        self.assertEqual(report["progress_percent"], 50)
        self.assertEqual(report["file_progress_percent"], 0)
        self.assertEqual(report["eta_seconds"], 2)
        self.assertIn("当前下载速度：100.00 B/s", monitor.render(report))
        self.write(network_bytes=240, reconstructed_bytes=1000, completed_files=2, phase="complete")
        report = self.snapshot(104)
        self.assertEqual(report["progress_percent"], 100)
        self.assertEqual(report["file_progress_percent"], 100)
        self.assertEqual(report["eta_seconds"], 0)

    @patch.object(monitor, "process_alive", return_value=True)
    def test_retry_regression_does_not_create_negative_speed_or_false_eta(self, alive):
        self.write(network_bytes=100, reconstructed_bytes=600)
        self.snapshot(100)
        self.write(network_bytes=200, reconstructed_bytes=100)
        report = self.snapshot(102)
        self.assertEqual(report["network_speed_bytes_per_second"], 50)
        self.assertEqual(report["reconstruction_speed_bytes_per_second"], 0)
        self.assertIsNone(report["eta_seconds"])
        self.write(network_bytes=300, reconstructed_bytes=300)
        report = self.snapshot(104)
        self.assertEqual(report["eta_seconds"], 7)
        self.write(run_id="restarted", network_bytes=0, reconstructed_bytes=300)
        report = self.snapshot(106)
        self.assertIsNone(report["network_speed_bytes_per_second"])
        self.assertEqual(report["average_network_bytes_per_second"], 0)

    @patch.object(monitor, "process_alive", return_value=False)
    def test_dead_process_is_reported_and_eta_is_absent(self, alive):
        self.write(network_bytes=10, reconstructed_bytes=100)
        self.snapshot(100)
        self.write(network_bytes=20, reconstructed_bytes=200)
        report = self.snapshot(102)
        self.assertIn("进程已退出", report["status"])
        self.assertIsNone(report["eta_seconds"])

    def test_monitor_can_start_before_downloader_and_once_emits_json(self):
        report = self.snapshot(100)
        self.assertEqual(report["phase"], "waiting")
        self.assertIn("等待", monitor.render(report))
        output = StringIO()
        with redirect_stdout(output):
            code = monitor.main(["--output", str(self.output), "--once", "--interval", "0.2", "--json"])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["network_speed_bytes_per_second"], 0)
        self.assertFalse((self.output / ".egosuite-download").exists())

    @patch.object(monitor, "process_alive", return_value=True)
    def test_estimated_selection_and_actual_completed_time_are_labeled(self, alive):
        self.write(hours_are_estimated=True, duration_sample_count=32, completed_hours=0.05)
        text = monitor.render(self.snapshot(100))
        self.assertIn("样本 32 段", text)
        self.assertIn("已完成实际录制时长：0.050 h", text)
        self.assertIn("清单估算时长：2.000 h", text)

    @patch.object(monitor, "process_alive", return_value=True)
    def test_monitor_displays_actual_bucket_and_prefix(self, alive):
        self.write(bucket="LightwheelAI/EgoStandard", prefix="EgoStandard-body/mcap")
        self.assertIn("hf://buckets/LightwheelAI/EgoStandard/EgoStandard-body/mcap/",
                      monitor.render(self.snapshot(100)))

    def test_pid_start_time_guard_prevents_mistaking_reused_pid_for_downloader(self):
        fields = ["S"] + ["0"] * 18 + ["12345"]
        with patch.object(Path, "read_text", return_value="123 (fake name) " + " ".join(fields)):
            self.assertTrue(monitor.process_alive(123, 12345))
            self.assertFalse(monitor.process_alive(123, 12346))
        fields[0] = "Z"
        with patch.object(Path, "read_text", return_value="123 (fake name) " + " ".join(fields)):
            self.assertFalse(monitor.process_alive(123, 12345))


if __name__ == "__main__":
    unittest.main()
