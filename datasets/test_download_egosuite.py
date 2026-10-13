"""Offline checks: range-only planning, resumable writes, retries and Xet progress.

Run with the download virtualenv:
    python -m unittest discover -s /cvpr/datasets -p 'test_*egosuite.py' -v
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer

import download_egosuite as download


def recording(seconds=60, topics=None, payload_size=2):
    buffer = BytesIO()
    writer = Writer(buffer, compression=CompressionType.NONE)
    writer.start()
    for topic in sorted(download.REQUIRED_TOPICS if topics is None else topics):
        channel = writer.register_channel(topic, "json", 0)
        for timestamp in (1_000_000_000, int((1 + seconds) * 1e9)):
            writer.add_message(channel, timestamp, b"x" * payload_size, timestamp)
    writer.finish()
    return buffer.getvalue()


def file_row(name, data, seconds=60):
    return {"path": f"{download.PREFIX}/{name}.mcap", "size": len(data),
            "xet_hash": "a" * 64, "duration_s": seconds, "message_count": 10}


def selection(rows, hours=None):
    total = sum(row["duration_s"] for row in rows) / 3600
    return {"bucket": download.BUCKET, "prefix": download.PREFIX, "seed": 42,
            "target_hours": total if hours is None else hours, "recorded_hours": total,
            "total_bytes": sum(row["size"] for row in rows), "files": rows}


def report(network, reconstructed):
    return SimpleNamespace(total_transfer_bytes_completed=network, total_bytes_completed=reconstructed)


class FakeRangeClient:
    def __init__(self, data, status=206):
        self.data, self.status, self.body_reads, self.ranges = data, status, 0, []

    @contextmanager
    def stream(self, method, url, *, headers, **kwargs):
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        self.ranges.append((start, end))
        def iter_bytes(**kwargs):
            self.body_reads += 1
            yield self.data[start:end + 1]
        yield SimpleNamespace(status_code=self.status,
                              headers={"Content-Range": f"bytes {start}-{end}/{len(self.data)}"},
                              iter_bytes=iter_bytes)


class OfflineDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.logs = StringIO()
        self.stdout = redirect_stdout(self.logs)
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def fake_api(self, rows):
        api = Mock()
        api.get_bucket_paths_info.return_value = [SimpleNamespace(**row) for row in rows]
        return api

    def prepare(self, rows, progress):
        with patch.object(download.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 2**30)):
            return download.prepare_download(self.output, selection(rows), self.fake_api(rows), progress)

    def test_summary_probe_only_reads_ranges_and_refuses_whole_file_response(self):
        # Deliberately large message payloads: summaries must use a small fraction.
        data = recording(payload_size=128 * 1024)
        fs = SimpleNamespace(
            resolve_path=lambda path, **kwargs: SimpleNamespace(unresolve=lambda: path),
            url=lambda path: "https://example.invalid/mcap",
            _api=SimpleNamespace(_build_hf_headers=lambda: {}),
        )
        client = FakeRangeClient(data)
        with patch("huggingface_hub.utils.get_session", return_value=client), \
                patch("huggingface_hub.utils.hf_raise_for_status"):
            range_class = download.summary_file_class()
            with range_class(fs, "fake.mcap", size=len(data), block_size=64 * 1024) as stream:
                row = download.summary_row(make_reader(stream).get_summary(),
                                           SimpleNamespace(**file_row("one", data)))
            self.assertEqual(row["duration_s"], 60)
            self.assertLess(sum(end - start + 1 for start, end in client.ranges), len(data) // 4)
            client.status, client.body_reads = 200, 0
            with range_class(fs, "fake.mcap", size=len(data), block_size=64 * 1024) as stream:
                with self.assertRaisesRegex(ValueError, "ignored Range"):
                    make_reader(stream).get_summary()
            self.assertEqual(client.body_reads, 0)

    def test_missing_required_topics_are_rejected(self):
        data = recording(topics=["/pose/body"])
        with self.assertRaisesRegex(ValueError, "Missing required topics"):
            download.summary_row(make_reader(BytesIO(data)).get_summary(),
                                 SimpleNamespace(**file_row("missing", data)))

    def test_planning_selects_whole_episodes_and_reuses_selection_without_network(self):
        rows = [file_row(str(index), b"fake", seconds=120) for index in range(5)]
        api = self.fake_api(rows)
        api.list_bucket_tree.return_value = [SimpleNamespace(**row) for row in rows]
        with download.Progress(self.output, 0.05) as progress, \
                patch("huggingface_hub.HfApi", return_value=api), \
                patch.object(download, "probe", side_effect=lambda file: vars(file)):
            result = download.plan(self.output, 0.05, 42, 2, progress, duration_mode="exact")
            self.assertEqual(len(result["files"]), 2)
            self.assertAlmostEqual(result["recorded_hours"], 240 / 3600)
            api.list_bucket_tree.side_effect = AssertionError("must reuse saved selection")
            self.assertEqual(download.plan(self.output, 0.05, 42, 2, progress, duration_mode="exact"), result)
            self.assertEqual(progress.state["selected_files"], 2)
        self.assertEqual(list(self.output.rglob("*.mcap")), [])

    def test_insufficient_hours_does_not_save_selection(self):
        rows = [file_row("short", b"fake")]
        api = self.fake_api(rows)
        api.list_bucket_tree.return_value = [SimpleNamespace(**rows[0])]
        with download.Progress(self.output, 250) as progress, \
                patch("huggingface_hub.HfApi", return_value=api), \
                patch.object(download, "probe", side_effect=lambda file: vars(file)):
            with self.assertRaisesRegex(ValueError, "Only"):
                download.plan(self.output, 250, 42, 2, progress, duration_mode="exact")
        self.assertFalse((download.control_dir(self.output) / "selection.json").exists())
        self.assertEqual(list(self.output.rglob("*.mcap")), [])

    def test_cached_sample_avoids_all_new_duration_requests(self):
        rows = [file_row(str(index), b"fake", seconds=120) for index in range(100)]
        directory = download.control_dir(self.output)
        directory.mkdir()
        (directory / "durations.jsonl").write_text("\n".join(json.dumps(row) for row in rows[:32]))
        api = self.fake_api(rows)
        api.list_bucket_tree.return_value = [SimpleNamespace(**row) for row in rows]
        with download.Progress(self.output, 0.2) as progress, \
                patch("huggingface_hub.HfApi", return_value=api), \
                patch.object(download, "probe", side_effect=AssertionError("must use cached samples")) as probe:
            result = download.plan(self.output, 0.2, 42, 2, progress)
            self.assertEqual(len(result["files"]), 6)
            self.assertEqual(result["duration_sample_count"], 32)
            self.assertTrue(result["hours_are_estimated"])
            self.assertEqual(probe.call_count, 0)

    def test_new_sampling_is_bounded_and_downloads_all_if_estimated_source_is_small(self):
        rows = [file_row(str(index), b"fake", seconds=120) for index in range(100)]
        api = self.fake_api(rows)
        api.list_bucket_tree.return_value = [SimpleNamespace(**row) for row in rows]
        with download.Progress(self.output, 100) as progress, \
                patch("huggingface_hub.HfApi", return_value=api), \
                patch.object(download, "probe", side_effect=lambda file: vars(file)) as probe:
            result = download.plan(self.output, 100, 42, 2, progress, sample_files=4)
            self.assertEqual(probe.call_count, 4)
            self.assertEqual(len(result["files"]), 100)
            self.assertTrue(result["estimated_source_exhausted"])
            self.assertLess(result["recorded_hours"], result["target_hours"])
            download.validate_selection(result, self.output)
        self.assertEqual(list(self.output.rglob("*.mcap")), [])

    def test_estimated_duration_is_replaced_by_actual_local_time_and_survives_restart(self):
        data = recording(seconds=60)
        row = {**file_row("estimated", data, seconds=120), "duration_estimated": True}
        def backend(api, row, path, callback):
            path.write_bytes(data)
            callback(report(len(data), len(data)))
        with download.Progress(self.output, 2 / 60) as progress:
            _, done = self.prepare([row], progress)
            download.download_one(self.output, row, None, progress, done, threading.Event(), 0, 0, backend)
            self.assertAlmostEqual(progress.state["completed_hours"], 1 / 60)
            self.assertEqual(done[row["path"]]["actual_duration_s"], 60)
        with download.Progress(self.output, 2 / 60) as restarted:
            pending, done = self.prepare([row], restarted)
            self.assertEqual(pending, [])
            self.assertAlmostEqual(restarted.state["completed_hours"], 1 / 60)

    def test_simultaneous_downloads_are_journaled_and_skipped_on_restart(self):
        data = recording()
        rows = [file_row(str(index), data) for index in range(4)]
        def backend(api, row, path, callback):
            callback(report(7, len(data) // 2))
            path.write_bytes(data)
            callback(report(11, len(data)))
        with download.Progress(self.output, 4 / 60) as progress:
            pending, done = self.prepare(rows, progress)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(download.download_one, self.output, row, None, progress,
                                           done, threading.Event(), 0, 0, backend) for row in pending]
                for future in futures:
                    future.result()
            self.assertEqual(progress.state["completed_files"], 4)
            self.assertEqual(progress.state["network_bytes"], 44)
            self.assertEqual(progress.state["completed_bytes"], 4 * len(data))
            self.assertFalse(progress.active)
        with download.Progress(self.output, 4 / 60) as restarted:
            pending, done = self.prepare(rows, restarted)
            self.assertEqual(pending, [])
            self.assertEqual(restarted.state["resumed_files"], 4)
            self.assertEqual(len(done), 4)
        self.assertEqual(list(self.output.rglob("*.part")), [])

    def test_retry_keeps_network_counter_monotonic_and_does_not_mark_partial_done(self):
        data = recording()
        row = file_row("retry", data)
        calls, counters = [], []
        def backend(api, row, path, callback):
            calls.append(path)
            callback(report(100, len(data) // 2))
            counters.append(progress.state["network_bytes"])
            if len(calls) == 1:
                path.write_bytes(b"unfinished")
                raise ConnectionError("temporary disconnect")
            path.write_bytes(data)
            callback(report(150, len(data)))
        with download.Progress(self.output, 1 / 60) as progress:
            _, done = self.prepare([row], progress)
            download.download_one(self.output, row, None, progress, done, threading.Event(), 1, 0, backend)
            self.assertEqual(counters, [100, 200])
            self.assertEqual(progress.state["network_bytes"], 250)
            self.assertEqual(progress.state["completed_files"], 1)
            self.assertEqual(download.local_path(self.output, row["path"]).read_bytes(), data)

    def test_crash_recovery_requires_verified_journal_not_part_size(self):
        data = recording()
        rows = [file_row("verified", data), file_row("unverified", data)]
        done = {}
        for row in rows:
            destination = download.local_path(self.output, row["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            part = destination.with_name(destination.name + ".part")
            part.write_bytes(data)
            if "unverified" not in row["path"]:
                done[row["path"]] = {"state": "verified", "xet_hash": row["xet_hash"],
                                     "mtime_ns": part.stat().st_mtime_ns}
        download.atomic_json(download.control_dir(self.output) / "completed.json", done)
        with download.Progress(self.output, 2 / 60) as progress:
            pending, journal = self.prepare(rows, progress)
            self.assertEqual(pending, [rows[1]])
            self.assertEqual(progress.state["completed_files"], 1)
            self.assertEqual(journal[rows[0]["path"]]["state"], "complete")

    def test_changed_source_existing_unverified_file_and_low_disk_fail_before_download(self):
        row = file_row("one", recording())
        api = self.fake_api([row])
        api.get_bucket_paths_info.return_value[0].xet_hash = "b" * 64
        with download.Progress(self.output, 1 / 60) as progress:
            with self.assertRaisesRegex(ValueError, "Remote file changed"):
                download.prepare_download(self.output, selection([row]), api, progress)
            destination = download.local_path(self.output, row["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"do not overwrite")
            with self.assertRaisesRegex(ValueError, "unverified"):
                self.prepare([row], progress)
            self.assertEqual(destination.read_bytes(), b"do not overwrite")
            destination.unlink()
            with patch.object(download.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(ValueError, "plus 5 GiB"):
                    download.prepare_download(self.output, selection([row]), self.fake_api([row]), progress)

    def test_bad_local_recording_never_becomes_complete(self):
        data = recording(topics=["/pose/body"])
        row = file_row("missing", data)
        def backend(api, row, path, callback):
            path.write_bytes(data)
            callback(report(len(data), len(data)))
        with download.Progress(self.output, 1 / 60) as progress:
            _, done = self.prepare([row], progress)
            with self.assertRaisesRegex(ValueError, "Missing required topics"):
                download.download_one(self.output, row, None, progress, done, threading.Event(), 2, 0, backend)
            self.assertEqual(progress.state["completed_files"], 0)
            self.assertFalse(done)
            self.assertFalse(download.local_path(self.output, row["path"]).exists())

    def test_native_xet_receives_progress_callback_and_validates_metadata(self):
        row = file_row("native", recording())
        api = Mock()
        api.get_bucket_file_metadata.return_value = SimpleNamespace(
            size=row["size"], xet_file_data=SimpleNamespace(file_hash=row["xet_hash"],
                                                         refresh_route="https://example.invalid/token"))
        session, callback = MagicMock(), Mock()
        group = session.new_file_download_group.return_value.__enter__.return_value
        destination = self.output / "native.part"
        with patch("huggingface_hub.utils._xet.get_xet_session", return_value=session), \
                patch("huggingface_hub.utils.build_hf_headers", return_value={"user-agent": "test"}):
            download.xet_download(api, row, destination, callback)
            self.assertIs(session.new_file_download_group.call_args.kwargs["progress_callback"], callback)
            self.assertEqual(group.start_download_file.call_args.args[1], str(destination))
            api.get_bucket_file_metadata.return_value.size += 1
            with self.assertRaisesRegex(ValueError, "Remote file changed"):
                download.xet_download(api, row, destination, callback)
            self.assertEqual(session.new_file_download_group.call_count, 1)

    def test_lock_and_path_guard(self):
        with download.output_lock(self.output):
            with self.assertRaisesRegex(ValueError, "already using"):
                with download.output_lock(self.output):
                    self.fail("second lock acquired")
        for path in ("/tmp/a.mcap", f"{download.PREFIX}/../a.mcap", "other/body/a.mcap"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                download.local_path(self.output, path)

    def test_egostandard_source_and_rejection_of_old_egopro_selection(self):
        self.assertEqual(download.BUCKET, "LightwheelAI/EgoStandard")
        self.assertEqual(download.PREFIX, "EgoStandard-body/mcap")
        old = selection([file_row("one", recording())])
        old["bucket"] = "LightwheelAI/EgoPro"
        old["prefix"] = "EgoProStandard-body/mcap"
        with self.assertRaisesRegex(ValueError, "Selection source mismatch"):
            download.validate_selection(old, self.output)
        with download.Progress(self.output, 200):
            state = json.loads((download.control_dir(self.output) / "progress.json").read_text())
            self.assertEqual(state["bucket"], "LightwheelAI/EgoStandard")
            self.assertEqual(state["prefix"], "EgoStandard-body/mcap")

    def test_preflight_retries_network_failures_and_retains_verified_local_progress(self):
        data = recording()
        row = file_row("already_downloaded", data)
        destination = download.local_path(self.output, row["path"])
        destination.parent.mkdir(parents=True)
        destination.write_bytes(data)
        marker = {"xet_hash": row["xet_hash"], "mtime_ns": destination.stat().st_mtime_ns,
                  "state": "complete", "actual_duration_s": 60}
        download.atomic_json(download.control_dir(self.output) / "completed.json", {row["path"]: marker})
        api = Mock()
        calls = []
        with download.Progress(self.output, 1 / 60) as progress:
            def remote(*args):
                calls.append(True)
                self.assertEqual(progress.state["completed_files"], 1)
                self.assertEqual(progress.state["completed_bytes"], len(data))
                if len(calls) <= 2:
                    raise ConnectionError("temporary TLS disconnect")
                return [SimpleNamespace(**row)]
            api.get_bucket_paths_info.side_effect = remote
            pending, done = download.prepare_download(self.output, selection([row]), api, progress,
                                                       retries=2, retry_delay=0)
            self.assertEqual(len(calls), 3)
            self.assertEqual(pending, [])
            self.assertEqual(progress.state["resumed_files"], 1)
            self.assertIsNone(progress.state["last_error"])

    def test_preflight_auth_failure_is_not_retried(self):
        row = file_row("one", recording())
        failure = RuntimeError("authentication denied")
        failure.response = SimpleNamespace(status_code=401)
        api = Mock()
        api.get_bucket_paths_info.side_effect = failure
        with download.Progress(self.output, 1 / 60) as progress:
            with self.assertRaisesRegex(RuntimeError, "authentication denied"):
                download.prepare_download(self.output, selection([row]), api, progress,
                                          retries=20, retry_delay=0)
        self.assertEqual(api.get_bucket_paths_info.call_count, 1)


if __name__ == "__main__":
    unittest.main()
