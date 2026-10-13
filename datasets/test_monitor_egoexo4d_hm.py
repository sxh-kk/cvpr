"""Offline tests for bytes, temporary-file accounting and speed estimates."""

import base64
import hashlib
from pathlib import Path
import tempfile
import unittest

from monitor_egoexo4d_hm import ProgressScanner, RateTracker, snapshot


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = {"hm_files": [{"relative_path": "take/body.npz", "size": 32768}],
                     "s3_files": [{"relative_path": "takes/take/rgb.mp4", "size": 65536, "part": "takes",
                                    "source_path": "s3://slow-fixture/takes/take/rgb.mp4"}]}
        self.scanner = ProgressScanner(self.root, self.plan)

    def write(self, relative, size):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
        return path

    def test_hf_old_and_hashed_partial_names_are_deduplicated(self):
        prefix = "EgoExo4D-HM/.cache/huggingface/download/take/"
        hashed = base64.urlsafe_b64encode(hashlib.sha1(b"body.npz.metadata").digest()).decode()
        self.write(prefix + "body.npz.oldetag.incomplete", 4096)
        self.write(prefix + hashed + ".newetag.incomplete", 8192)
        result = self.scanner.scan()
        self.assertEqual(result["present_bytes"], 8192)
        self.assertEqual(result["complete_files"], 0)
        self.assertEqual(result["partial_files"], 1)
        self.assertEqual(result["total_bytes"], 98304)

    def test_completed_destination_wins_over_stale_partials(self):
        self.write("EgoExo4D-HM/take/body.npz", 32768)
        self.write("EgoExo4D-HM/.cache/huggingface/download/take/body.npz.etag.incomplete", 4096)
        self.write("EgoExo4D/takes/take/rgb.mp4.part.random", 8192)
        self.write("EgoExo4D/takes/take/rgb.mp4.part", 4096)
        result = self.scanner.scan()
        self.assertEqual(result["present_bytes"], 32768 + 8192)
        self.assertEqual(result["complete_files"], 1)
        self.assertEqual(result["partial_files"], 1)
        self.write("EgoExo4D/takes/take/rgb.mp4", 65536)
        result = self.scanner.scan()
        self.assertEqual(result["present_bytes"], 98304)
        self.assertEqual(result["complete_files"], 2)
        self.assertEqual(result["partial_files"], 0)

    def test_sparse_preallocation_is_not_counted_as_downloaded(self):
        path = self.root / "EgoExo4D/takes/take/rgb.mp4.part.random"
        path.parent.mkdir(parents=True)
        with path.open("wb") as handle:
            handle.truncate(65536)
            handle.write(b"x" * 4096)
        stat = path.stat()
        if stat.st_blocks * 512 >= stat.st_size:
            self.skipTest("test filesystem does not expose sparse allocation")
        result = self.scanner.scan()
        self.assertEqual(result["present_bytes"], min(stat.st_size, stat.st_blocks * 512))
        self.assertLess(result["present_bytes"], 65536)
        self.assertEqual(result["complete_files"], 0)

    def test_wrong_size_and_unrelated_files_do_not_count_as_completed(self):
        self.write("EgoExo4D-HM/take/body.npz", 4096)
        self.write("EgoExo4D/takes/take/other.mp4", 65536)
        self.write("EgoExo4D-HM/.cache/huggingface/download/take/unrelated.etag.incomplete", 32768)
        result = self.scanner.scan()
        self.assertEqual(result["complete_files"], 0)
        self.assertEqual(result["present_bytes"], 4096)

    def test_rates_stall_and_retry(self):
        tracker = RateTracker(window=60)
        self.assertIsNone(tracker.update(100, 1024)["speed_bytes_per_second"])
        report = tracker.update(102, 5120)
        self.assertEqual(report["speed_bytes_per_second"], 2048)
        self.assertEqual(report["average_bytes_per_second"], 2048)
        self.assertEqual(tracker.update(104, 5120)["speed_bytes_per_second"], 0)
        # Deleting a partial during retry resets the ETA window.
        report = tracker.update(106, 0)
        self.assertEqual(report["speed_bytes_per_second"], 0)
        self.assertEqual(report["eta_speed_bytes_per_second"], 0)
        self.assertEqual(tracker.update(108, 4096)["eta_speed_bytes_per_second"], 2048)

    def test_stall_ages_out_of_eta_window(self):
        tracker = RateTracker(window=60)
        tracker.update(0, 0)
        tracker.update(5, 4096)
        self.assertEqual(tracker.update(70, 4096)["eta_speed_bytes_per_second"], 0)

    def test_incomplete_without_process_has_no_eta(self):
        report = snapshot(self.scanner, RateTracker(), requested_pid=2147483647)
        self.assertEqual(report["processes"], [])
        self.assertIsNone(report["eta_seconds"])
        self.assertIn("未找到下载进程", report["status"])

    def test_complete_without_process_reports_finished(self):
        self.write("EgoExo4D-HM/take/body.npz", 32768)
        self.write("EgoExo4D/takes/take/rgb.mp4", 65536)
        report = snapshot(self.scanner, RateTracker(), requested_pid=2147483647)
        self.assertEqual(report["progress_percent"], 100)
        self.assertEqual(report["eta_seconds"], 0)
        self.assertIn("已全部落盘", report["status"])

    def test_exclusion_progress_finishes_without_excluded_files(self):
        self.write("EgoExo4D-HM/take/body.npz", 32768)
        self.write("EgoExo4D/takes/take/rgb.mp4.part.random", 4096)
        scanner = ProgressScanner(self.root, self.plan, ["slow-fixture"])
        report = snapshot(scanner, RateTracker(), requested_pid=2147483647)
        self.assertEqual(report["total_bytes"], 32768)
        self.assertEqual(report["present_bytes"], 32768)
        self.assertEqual(report["total_files"], 1)
        self.assertEqual(report["partial_files"], 0)
        self.assertEqual(report["progress_percent"], 100)
        self.assertEqual(report["excluded_files"], 1)
        self.assertEqual(report["excluded_total_bytes"], 65536)
        self.assertEqual(report["excluded_buckets"], ["slow-fixture"])
        self.assertEqual(len(self.plan["s3_files"]), 1)
        self.assertIn("已全部落盘", report["status"])

    def test_monitor_rejects_unknown_excluded_bucket(self):
        with self.assertRaisesRegex(ValueError, "清单中没有"):
            ProgressScanner(self.root, self.plan, ["slow-fixture-typo"])


if __name__ == "__main__":
    unittest.main()
