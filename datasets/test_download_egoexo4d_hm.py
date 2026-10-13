"""Offline tests: matching, RGB filtering, dry-run, retries, and resuming."""

from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import download_egoexo4d_hm as downloader


def npz_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in downloader.HM_FIELDS:
            archive.writestr(name + ".npy", b"fixture")
    return buffer.getvalue()


def take(name):
    return {"take_name": name, "take_uid": name + "-uid", "capture_uid": "capture-uid",
            "root_dir": "takes/" + name,
            "capture": {"cameras": [{"cam_id": "aria01", "is_ego": True, "device_type": "aria"},
                                     {"cam_id": "cam01", "is_ego": False, "device_type": "hero10"}]},
            "frame_aligned_videos": {
                "aria01": {"rgb": {"relative_path": "frame_aligned_videos/aria01_214-1.mp4"},
                           "slam-left": {"relative_path": "frame_aligned_videos/aria01_1201-1.mp4"}},
                "cam01": {"0": {"relative_path": "frame_aligned_videos/cam01.mp4"}}}}


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.downloads = []
        self.fail_downloads = False

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[f"s3://{Bucket}/{Key}"])}

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.objects[f"s3://{Bucket}/{Key}"])}

    def download_file(self, bucket, key, target, Config=None):
        self.downloads.append(key)
        if self.fail_downloads:
            raise IOError("simulated interrupted transfer")
        Path(target).write_bytes(self.objects[f"s3://{bucket}/{key}"])

    def add(self, relative, payload=b"data", views=None):
        uri = "s3://fixture/" + relative
        self.objects[uri] = payload
        return {"source_path": uri, "relative_path": relative, "size": len(payload), "views": views}

    def add_manifest(self, part, rows):
        self.objects[f"{downloader.S3_BASE}/v2/{part}/manifest.json"] = json.dumps(rows).encode()


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "downloads"
        self.args = downloader.parse_args(["--output-dir", str(self.root), "--take", "alpha"])
        self.s3 = FakeS3()
        self.hm = npz_bytes()
        self.api_calls = []
        self.hm_calls = []
        info = SimpleNamespace(sha="a" * 40, siblings=[
            SimpleNamespace(rfilename=f"{name}/{downloader.HM_FILE}", size=len(self.hm))
            for name in ("alpha", "beta")])

        def repo_info(*args, **kwargs):
            self.api_calls.append(kwargs)
            return info

        self.api = SimpleNamespace(repo_info=repo_info)
        self.s3.add_manifest("metadata", [{"uid": None, "paths": [
            self.s3.add("takes.json", json.dumps([take("alpha"), take("beta")]).encode()),
            self.s3.add("captures.json", b"[]")]}])
        for part in downloader.RAW_PARTS:
            rows = []
            if part == "captures":
                rows = [{"uid": "capture-uid", "paths": [
                    self.s3.add("captures/session/timesync.csv", b"timestamps"),
                    self.s3.add("captures/session/survey.csv", b"unneeded")]}]
            else:
                for name in ("alpha", "beta"):
                    if part == "takes":
                        paths = [self.s3.add(f"takes/{name}/frame_aligned_videos/{file}", views=views)
                                 for file, views in [("aria01_214-1.mp4", ["ego"]),
                                                     ("aria01_1201-1.mp4", ["ego"]),
                                                     ("cam01.mp4", ["exo"])]]
                    elif part == "take_trajectory":
                        paths = [self.s3.add(f"takes/{name}/trajectory/closed_loop_trajectory.csv", views=["ego"])]
                    else:
                        paths = [self.s3.add(f"takes/{name}/aria01_noimagestreams.vrs", views=["ego"])]
                    rows.append({"uid": name + "-uid", "paths": paths})
            self.s3.add_manifest(part, rows)

    def hf_download(self, *args, **kwargs):
        self.hm_calls.append(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.hm)
        return str(path)

    def run_download(self):
        with redirect_stdout(io.StringIO()):
            downloader.run(self.args, self.s3, self.api, self.hf_download, None)

    def assign_bucket(self, part, bucket):
        uri = f"{downloader.S3_BASE}/v2/{part}/manifest.json"
        rows = json.loads(self.s3.objects[uri])
        for row in rows:
            for item in row["paths"]:
                previous = item["source_path"]
                item["source_path"] = f"s3://{bucket}/{item['relative_path']}"
                self.s3.objects[item["source_path"]] = self.s3.objects[previous]
        self.s3.add_manifest(part, rows)

    def test_exact_take_rgb_and_capture_matching(self):
        with redirect_stdout(io.StringIO()):
            plan, _ = downloader.build_plan(self.args, self.s3, self.api)
        paths = {item["relative_path"] for item in plan["s3_files"]}
        self.assertEqual(paths, {"takes.json", "captures.json",
                                "takes/alpha/frame_aligned_videos/aria01_214-1.mp4",
                                "takes/alpha/trajectory/closed_loop_trajectory.csv",
                                "takes/alpha/aria01_noimagestreams.vrs",
                                "captures/session/timesync.csv"})
        self.assertEqual(len(plan["hm_files"]), 1)

    def test_dry_run_writes_nothing(self):
        self.args.dry_run = True
        self.run_download()
        self.assertFalse(self.root.exists())
        self.assertEqual(self.s3.downloads, [])
        self.assertEqual(self.hm_calls, [])

    def test_auto_route_dry_run_never_constructs_or_starts_probes(self):
        self.args.dry_run = self.args.auto_route = True
        factory = Mock(side_effect=AssertionError("dry-run must not start routing"))
        with redirect_stdout(io.StringIO()):
            downloader.run(self.args, self.s3, self.api, self.hf_download, None, factory)
        factory.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_router_receives_only_pending_included_files(self):
        self.assign_bucket("takes", "excluded-fixture")
        self.args.exclude_bucket = ["excluded-fixture"]
        self.args.auto_route = True
        factory = Mock(return_value=nullcontext(None))
        with redirect_stdout(io.StringIO()):
            downloader.run(self.args, self.s3, self.api, self.hf_download, None, factory)
        files = factory.call_args.args[0]
        self.assertEqual(len(files), 3)
        self.assertTrue(all(item["part"] != "metadata" for item in files))
        self.assertTrue(all(not item["source_path"].startswith("s3://excluded-fixture/") for item in files))
        factory.reset_mock()
        with redirect_stdout(io.StringIO()):
            downloader.run(self.args, self.s3, self.api, self.hf_download, None, factory)
        factory.assert_not_called()

    def test_failed_file_retries_with_latest_route(self):
        item = self.s3.add("retry.bin", b"data")
        failing = FakeS3()
        failing.fail_downloads = True
        router = Mock()
        router.transfer.return_value = nullcontext()
        router.client_for.side_effect = [("direct", failing), ("proxy", self.s3)]
        with patch.object(downloader.time, "sleep"):
            status = downloader.download_s3_file(None, item, self.root, None, router)
        self.assertEqual(status, "完成（代理）")
        self.assertEqual(router.client_for.call_count, 2)
        self.assertEqual((self.root / "retry.bin").read_bytes(), b"data")

    def test_completed_file_does_not_trigger_routing(self):
        item = self.s3.add("done.bin", b"data")
        self.root.mkdir(parents=True)
        (self.root / "done.bin").write_bytes(b"data")
        router = Mock()
        self.assertEqual(downloader.download_s3_file(None, item, self.root, None, router), "跳过")
        router.transfer.assert_not_called()

    def test_probe_cli_rejects_invalid_values(self):
        for option, value in (("--probe-interval", "0"), ("--probe-interval", "nan"),
                              ("--probe-mib", "0"), ("--probe-mib", "100"), ("--probe-timeout", "0")):
            with self.subTest(option=option, value=value), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                downloader.parse_args([option, value])

    def test_resume_pins_revision_and_skips_completed_files(self):
        self.run_download()
        self.assertEqual(len(self.s3.downloads), 4)
        self.run_download()
        self.assertEqual(len(self.s3.downloads), 4)
        self.assertEqual(self.api_calls[-1]["revision"], "a" * 40)
        self.assertEqual(len(self.hm_calls), 1)
        self.assertEqual(self.hm_calls[0]["filename"], f"alpha/{downloader.HM_FILE}")
        self.assertEqual(self.hm_calls[0]["revision"], "a" * 40)

    def test_truncated_hm_file_is_replaced(self):
        self.run_download()
        path = self.root / "EgoExo4D-HM" / "alpha" / downloader.HM_FILE
        path.write_bytes(b"interrupted")
        self.run_download()
        self.assertEqual(path.read_bytes(), self.hm)
        self.assertEqual(len(self.hm_calls), 2)

    def test_hm_failure_prevents_raw_downloads(self):
        def interrupted(*args, **kwargs):
            raise IOError("simulated interrupted HM transfer")

        with patch.object(downloader.time, "sleep"), redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "HM 文件下载失败"):
            downloader.run(self.args, self.s3, self.api, interrupted, None)
        self.assertEqual(self.s3.downloads, [])
        self.assertFalse((self.root / "EgoExo4D-HM/alpha" / downloader.HM_FILE).exists())

    def test_hm_downloads_only_selected_manifest_paths(self):
        selected = [{"relative_path": f"selected_{i}/{downloader.HM_FILE}", "size": len(self.hm)}
                    for i in range(300)]
        plan = {"hm_files": selected, "hm_repo": downloader.HM_REPO, "hm_revision": "a" * 40}
        with redirect_stdout(io.StringIO()):
            downloader.download_hm(plan, self.root / "EgoExo4D-HM", 4, self.hf_download)
        self.assertEqual({call["filename"] for call in self.hm_calls}, {item["relative_path"] for item in selected})
        self.assertEqual(len(self.hm_calls), len(selected))
        self.assertTrue(all(call["revision"] == plan["hm_revision"] for call in self.hm_calls))

    def test_resume_plan_avoids_all_catalog_requests(self):
        self.run_download()
        self.args.resume = True
        self.args.take = None
        with patch.object(self.api, "repo_info", side_effect=AssertionError("must not request HF catalog")), \
             patch.object(self.s3, "get_object", side_effect=AssertionError("must not request S3 catalog")):
            self.run_download()
        self.assertEqual(len(self.hm_calls), 1)
        self.assertEqual(len(self.s3.downloads), 4)

    def test_excluded_bucket_is_deferred_and_can_be_resumed_later(self):
        self.assign_bucket("takes", "slow-fixture")
        # A similar name must not be excluded by prefix matching.
        self.assign_bucket("captures", "slow-fixture-backup")
        self.args.exclude_bucket = ["slow-fixture"]
        self.run_download()
        video = self.root / "EgoExo4D/takes/alpha/frame_aligned_videos/aria01_214-1.mp4"
        self.assertFalse(video.exists())
        self.assertTrue((self.root / "EgoExo4D/captures/session/timesync.csv").is_file())
        self.assertTrue((self.root / "EgoExo4D-HM/alpha" / downloader.HM_FILE).is_file())
        self.assertEqual(len(self.s3.downloads), 3)
        saved = json.loads((self.root / "download_plan.json").read_text())
        self.assertEqual(len(saved["s3_files"]), 6)
        self.assertTrue(any(item["source_path"].startswith("s3://slow-fixture/") for item in saved["s3_files"]))

        self.args.resume = True
        self.args.take = None
        self.args.exclude_bucket = []
        with patch.object(self.api, "repo_info", side_effect=AssertionError("must use saved plan")), \
             patch.object(self.s3, "get_object", side_effect=AssertionError("must use local metadata")):
            self.run_download()
        self.assertEqual(video.read_bytes(), b"data")
        self.assertEqual(len(self.s3.downloads), 4)
        self.assertEqual(len(self.hm_calls), 1)
        self.assertEqual(json.loads((self.root / "download_plan.json").read_text()), saved)

    def test_exclusion_preserves_existing_files_and_partial_data(self):
        self.run_download()
        self.args.resume = True
        self.args.take = None
        self.args.exclude_bucket = ["fixture"]
        video = self.root / "EgoExo4D/takes/alpha/frame_aligned_videos/aria01_214-1.mp4"
        partial = video.with_name(video.name + ".part.old")
        partial.write_bytes(b"unfinished bytes")
        saved = (self.root / "download_plan.json").read_bytes()
        output = io.StringIO()
        with redirect_stdout(output):
            downloader.run(self.args, self.s3, self.api, self.hf_download, None)
        self.assertEqual(video.read_bytes(), b"data")
        self.assertEqual(partial.read_bytes(), b"unfinished bytes")
        self.assertEqual((self.root / "download_plan.json").read_bytes(), saved)
        self.assertEqual(len(self.s3.downloads), 4)
        self.assertIn("本次筛选范围下载完成", output.getvalue())

    def test_exclusion_dry_run_writes_nothing(self):
        self.args.dry_run = True
        self.args.exclude_bucket = ["fixture"]
        self.run_download()
        self.assertFalse(self.root.exists())
        self.assertEqual(self.s3.downloads, [])
        self.assertEqual(self.hm_calls, [])

    def test_unknown_excluded_bucket_is_rejected_before_downloads(self):
        self.args.exclude_bucket = ["fixture-typo"]
        with self.assertRaisesRegex(ValueError, "清单中没有"):
            self.run_download()
        self.assertEqual(self.s3.downloads, [])
        self.assertEqual(self.hm_calls, [])
        self.assertFalse((self.root / "download_plan.json").exists())

    def test_multiple_exclusions_keep_metadata_and_do_not_mutate_plan(self):
        args = downloader.parse_args(["--resume", "--exclude-bucket", "one", "two", "--exclude-bucket", "one"])
        plan = {"s3_files": [
            {"source_path": "s3://one/takes/a.mp4", "part": "takes"},
            {"source_path": "s3://two/takes/b.vrs", "part": "take_vrs_noimagestream"},
            {"source_path": "s3://one/takes.json", "part": "metadata"},
        ]}
        original = json.dumps(plan, sort_keys=True)
        selected, excluded = downloader.filter_s3_files(plan, args.exclude_bucket)
        self.assertEqual(selected, [plan["s3_files"][2]])
        self.assertEqual(len(excluded), 2)
        self.assertEqual(json.dumps(plan, sort_keys=True), original)

    def test_resume_requires_complete_local_metadata(self):
        self.run_download()
        self.args.resume = True
        self.args.take = None
        (self.root / "EgoExo4D/takes.json").write_bytes(b"interrupted")
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "takes.json"):
            downloader.load_saved_plan(self.args)

    def test_resume_rejects_different_revision(self):
        self.run_download()
        self.args.hm_revision = "b" * 40
        with self.assertRaisesRegex(ValueError, "HM commit"):
            downloader.load_saved_plan(self.args)

    def test_running_download_is_detected_before_catalog_requests(self):
        with downloader.download_lock(self.root), \
             patch.object(self.api, "repo_info", side_effect=AssertionError("must check lock first")), \
             redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "已有下载进程"):
            downloader.run(self.args, self.s3, self.api, self.hf_download, None)

    def test_truncated_file_is_replaced(self):
        self.run_download()
        video = self.root / "EgoExo4D/takes/alpha/frame_aligned_videos/aria01_214-1.mp4"
        video.write_bytes(b"broken")
        self.run_download()
        self.assertEqual(video.read_bytes(), b"data")
        self.assertEqual(len(self.s3.downloads), 5)

    def test_missing_timesync_for_one_capture_is_rejected(self):
        rows = [{"uid": "capture-uid", "paths": [self.s3.add("captures/session/survey.csv")]}]
        with self.assertRaisesRegex(ValueError, "captures"):
            downloader.select_raw(rows, "captures", [take("alpha")])

    def test_failed_transfer_does_not_leave_a_completed_file(self):
        self.s3.fail_downloads = True
        with patch.object(downloader.time, "sleep"), self.assertRaises(RuntimeError):
            self.run_download()
        self.assertFalse((self.root / "EgoExo4D/takes/alpha/aria01_noimagestreams.vrs").exists())
        self.assertEqual(list(self.root.rglob("*.part")), [])

    def test_unknown_take_and_path_escape_are_rejected(self):
        self.args.take = ["missing"]
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "不存在"):
            downloader.build_plan(self.args, self.s3, self.api)
        with self.assertRaises(ValueError):
            downloader.local_path(self.root, "../outside")


if __name__ == "__main__":
    unittest.main()
