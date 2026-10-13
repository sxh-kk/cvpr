"""Offline checks for route decisions, bounded probes, and task scheduling."""

import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import egoexo4d_routes as routes


def sample(bucket="fixture", size=32 * routes.MIB):
    return {"source_path": f"s3://{bucket}/folder/object.bin", "size": size}


def speeds(direct=1, proxy=2):
    return {"direct": routes.ProbeResult(direct * routes.MIB, 1),
            "proxy": routes.ProbeResult(proxy * routes.MIB, 1)}


class PolicyTests(unittest.TestCase):
    def test_two_consecutive_wins_are_required(self):
        policy = routes.RoutePolicy()
        policy.observe(speeds(), 0)
        self.assertEqual(policy.route, "direct")
        policy.observe(speeds(), 120)
        self.assertEqual(policy.route, "proxy")

    def test_small_difference_does_not_switch(self):
        policy = routes.RoutePolicy()
        for now in (0, 120, 240, 360):
            policy.observe(speeds(proxy=1.2), now)
        self.assertEqual(policy.route, "direct")
        self.assertEqual(policy.wins, 0)

    def test_a_win_is_reset_by_a_losing_sample(self):
        policy = routes.RoutePolicy()
        for now, pair in ((0, speeds()), (120, speeds(proxy=0.5)), (240, speeds())):
            policy.observe(pair, now)
        self.assertEqual(policy.route, "direct")
        self.assertEqual(policy.wins, 1)

    def test_switch_back_respects_cooldown(self):
        policy = routes.RoutePolicy()
        policy.observe(speeds(), 0)
        policy.observe(speeds(), 120)
        policy.observe(speeds(direct=4), 130)
        policy.observe(speeds(direct=4), 140)
        self.assertEqual(policy.route, "proxy")
        policy.observe(speeds(direct=4), 360)
        self.assertEqual(policy.route, "direct")

    def test_current_failure_switches_without_waiting(self):
        policy = routes.RoutePolicy(last_switch=0)
        pair = speeds()
        pair["direct"] = routes.ProbeResult(error="timeout")
        policy.observe(pair, 1)
        self.assertEqual(policy.route, "proxy")

    def test_two_failures_preserve_current_route(self):
        policy = routes.RoutePolicy(route="proxy")
        failed = routes.ProbeResult(error="timeout")
        policy.observe({"direct": failed, "proxy": failed}, 1000)
        self.assertEqual(policy.route, "proxy")

    def test_buckets_choose_independently(self):
        first, second = routes.RoutePolicy(), routes.RoutePolicy()
        for now in (0, 120):
            first.observe(speeds(), now)
            second.observe(speeds(direct=4), now)
        self.assertEqual(first.route, "proxy")
        self.assertEqual(second.route, "direct")


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.session = Mock()
        with patch.object(routes.shutil, "which", return_value="/usr/bin/curl"):
            self.prober = routes.CurlRangeProber(self.session, "http://127.0.0.1:17891")

    def test_proxy_validation_rejects_credentials_and_non_http_urls(self):
        for url in ("socks5://localhost:1234", "http://user:secret@localhost:1234",
                    "http://localhost:bad", "http://localhost:1234/path", "http://localhost:1234?token=x"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                routes.validate_proxy(url)

    def test_explicit_proxy_settings_override_environment(self):
        direct = self.prober._config("https://example.org", "direct")
        proxy = self.prober._config("https://example.org", "proxy")
        self.assertIn('proxy = ""', direct)
        self.assertIn('noproxy = "*"', direct)
        self.assertIn('proxy = "http://127.0.0.1:17891"', proxy)
        self.assertIn('noproxy = ""', proxy)

    def test_presigned_url_is_only_on_stdin_and_probe_has_a_deadline(self):
        url = "https://example.org/object?X-Amz-Signature=private"
        response = SimpleNamespace(returncode=0, stdout="206 2048 0.5", stderr="")
        with patch.object(routes.subprocess, "run", return_value=response) as run:
            result = self.prober._range(url, "direct", 1024, 2048)
        command = run.call_args.args[0]
        self.assertNotIn(url, command)
        self.assertEqual(command[1], "--disable")
        self.assertIn("--max-time", command)
        self.assertEqual(command[command.index("--max-filesize") + 1], "2048")
        self.assertEqual(command[command.index("--range") + 1], "1024-3071")
        self.assertEqual(run.call_args.kwargs["timeout"], 14)
        self.assertIn(url, run.call_args.kwargs["input"])
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(result.speed, 4096)

    def test_timed_out_partial_range_measures_slow_route(self):
        response = SimpleNamespace(returncode=28, stdout="206 8192 12.0", stderr="timeout")
        with patch.object(self.prober, "_invoke", return_value=response):
            result = self.prober._range("url", "direct", 0, routes.MIB)
        self.assertTrue(result.ok)
        self.assertAlmostEqual(result.speed, 8192 / 12)

    def test_error_pages_and_unbounded_responses_are_not_speed_samples(self):
        for stdout in ("403 4096 1.0", "200 4096 1.0", "206 4097 1.0", "206 0 1.0", "206 4096 0"):
            with self.subTest(stdout=stdout):
                response = SimpleNamespace(returncode=0, stdout=stdout, stderr="PRIVATE")
                with patch.object(self.prober, "_invoke", return_value=response):
                    result = self.prober._range("private-url", "direct", 0, 4096)
                self.assertFalse(result.ok)
                self.assertNotIn("PRIVATE", result.error)

    def test_subprocess_timeout_and_errors_do_not_expose_url(self):
        for failure in (subprocess.TimeoutExpired(["curl"], 14), OSError("private-url")):
            with self.subTest(failure=type(failure).__name__):
                with patch.object(self.prober, "_invoke", side_effect=failure):
                    result = self.prober._range("private-url", "proxy", 0, 4096)
                self.assertFalse(result.ok)
                self.assertNotIn("private-url", result.error)

    def test_region_is_taken_from_denied_head_and_cached(self):
        response = SimpleNamespace(returncode=0, stdout="HTTP/1.1 403 Forbidden\r\nx-amz-bucket-region: us-west-1\r\n\r\n")
        with patch.object(self.prober, "_invoke", return_value=response) as invoke:
            self.assertEqual(self.prober._region("fixture"), "us-west-1")
            self.assertEqual(self.prober._region("fixture"), "us-west-1")
        self.assertEqual(invoke.call_count, 1)
        self.assertLessEqual(invoke.call_args.args[-1], 5)

    def test_region_discovery_can_use_proxy(self):
        responses = [SimpleNamespace(returncode=28, stdout=""),
                     SimpleNamespace(returncode=0, stdout="x-amz-bucket-region: ap-south-1\r\n")]
        with patch.object(self.prober, "_invoke", side_effect=responses) as invoke:
            self.assertEqual(self.prober._region("fixture"), "ap-south-1")
        self.assertEqual([call.args[1] for call in invoke.call_args_list], ["direct", "proxy"])

    def test_both_routes_probe_same_rotating_range(self):
        with patch.object(self.prober, "_signed_url", return_value="private-url"), \
             patch.object(self.prober, "_range", return_value=routes.ProbeResult(2048, 1)) as read:
            result = self.prober.compare(sample(), 3)
        calls = read.call_args_list
        self.assertEqual({call.args[1] for call in calls}, {"direct", "proxy"})
        self.assertEqual({call.args[2:] for call in calls}, {(6 * routes.MIB, 2 * routes.MIB)})
        self.assertTrue(all(value.ok for value in result.values()))

    def test_small_objects_are_never_probed_past_their_end(self):
        with patch.object(self.prober, "_signed_url", return_value="url"), \
             patch.object(self.prober, "_range", return_value=routes.ProbeResult(100, 1)) as read:
            self.prober.compare(sample(size=100), 3)
        self.assertTrue(all(call.args[2:] == (0, 100) for call in read.call_args_list))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.status = Path(self.temp.name) / "route_status.json"
        self.prober = Mock()
        self.prober.compare.return_value = speeds()
        self.clients = {"direct": object(), "proxy": object()}

    def controller(self, files=None, **kwargs):
        return routes.AutoRouteController(self.clients, self.prober, files or [sample()],
                                           self.status, log=Mock(), **kwargs)

    def test_selection_affects_only_subsequent_client_requests(self):
        controller = self.controller()
        route, original = controller.client_for("fixture")
        controller._probe("fixture", 0)
        controller._probe("fixture", 1)
        self.assertEqual(route, "direct")
        self.assertIs(original, self.clients["direct"])
        self.assertEqual(controller.client_for("fixture"), ("proxy", self.clients["proxy"]))

    def test_status_contains_speeds_and_choices_without_signed_urls(self):
        controller = self.controller()
        controller._probe("fixture", 0)
        data = json.loads(self.status.read_text())
        self.assertEqual(data["buckets"]["fixture"]["proxy"]["mib_per_second"], 2)
        self.assertNotIn("source_path", self.status.read_text())
        self.assertNotIn("Signature", self.status.read_text())

    def test_selects_largest_available_sample_per_bucket(self):
        controller = self.controller([sample(size=100), sample(size=200), sample("other", 50)])
        self.assertEqual(controller.samples["fixture"]["size"], 200)
        self.assertEqual(set(controller.samples), {"fixture", "other"})

    def test_probe_failure_keeps_downloader_usable(self):
        self.prober.compare.side_effect = RuntimeError("private URL must not leak")
        controller = self.controller()
        controller._probe("fixture", 0)
        self.assertEqual(controller.client_for("fixture")[0], "direct")
        self.assertNotIn("private", self.status.read_text())

    def test_only_active_buckets_are_probed_and_interval_is_respected(self):
        done = threading.Event()
        self.prober.compare.side_effect = lambda *_: (done.set() or speeds())
        controller = self.controller([sample(), sample("inactive")], interval=120)
        with controller:
            with controller.transfer("fixture"):
                self.assertTrue(done.wait(timeout=2))
        self.assertEqual(self.prober.compare.call_count, 1)
        self.assertEqual(self.prober.compare.call_args.args[0]["source_path"], sample()["source_path"])
        self.assertEqual(controller.active["fixture"], 0)
        self.assertFalse(json.loads(self.status.read_text())["running"])

    def test_scheduling_resumes_when_a_bucket_becomes_due(self):
        first, second = threading.Event(), threading.Event()

        def compare(*_):
            (first if not first.is_set() else second).set()
            return speeds()

        self.prober.compare.side_effect = compare
        controller = self.controller(interval=120)
        with controller:
            with controller.transfer("fixture"):
                self.assertTrue(first.wait(2))
                with controller.lock:
                    controller.due["fixture"] = time.monotonic() - 1
                controller.wakeup.set()
                self.assertTrue(second.wait(2))
        self.assertEqual(self.prober.compare.call_count, 2)


if __name__ == "__main__":
    unittest.main()
