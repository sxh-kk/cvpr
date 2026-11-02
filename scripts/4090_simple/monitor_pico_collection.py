"""Read-only1Hz PICO collection monitor; standard library only.

No SDK, simulator, GPU API or control endpoint is used. Only this monitor's CPU
affinity/niceness changes. Samples retain at most three8MiB JSONL files; the
latest real fault atomically replaces snapshot.json with its preceding60s.
Subsecond stalls can be missed by1Hz sampling. Retained collector rejections and
callback telemetry improve visibility but do not establish a network/SDK/CPU
root cause. SIGINT/SIGTERM stop only this monitor.
"""
from __future__ import annotations

import argparse
from collections import deque, OrderedDict
import fcntl
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import threading
import time
from urllib.parse import urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, ProxyHandler, Request

CONTRACT = "cvpr-pico-read-only-collection-monitor-v1"
MAX_BYTES = 8 * 1024 * 1024
STATUS_BYTES = 1024 * 1024
OPERATOR_REASONS = ("operator_pause", "operator_reset", "operator_discard", "session_ended_without_save")


class MonitorOutputConflict(ValueError):
    pass


def small(value, depth=0):
    """Bound observations, never retain raw input packets or full histories."""
    if depth > 7:
        return "<depth limit>"
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:2048]
    if isinstance(value, dict):
        return {str(k)[:128]: small(v, depth + 1) for k, v in list(value.items())[:96]
                if k not in ("raw_packet", "source_raw_packet", "pico_calibration", "environment_config")}
    if isinstance(value, (list, tuple)):
        return [small(v, depth + 1) for v in value[-128:]]
    return str(value)[:256]


def encode(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    if path.is_symlink():
        raise ValueError("Monitor output must not be a symlink")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encode(value)); stream.flush()
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class RotatingJSONL:
    def __init__(self, path, max_bytes=MAX_BYTES, copies=3):
        self.path, self.max_bytes, self.copies = Path(path), max_bytes, copies

    def append(self, value):
        payload = encode(value)
        if len(payload) > self.max_bytes:
            raise ValueError("One monitor observation exceeds its bounded log")
        paths = [self.path] + [self.path.with_name(self.path.name + "." + str(i)) for i in range(1, self.copies)]
        if any(path.is_symlink() for path in paths):
            raise ValueError("Monitor log must not be a symlink")
        if self.path.exists() and self.path.stat().st_size + len(payload) > self.max_bytes:
            for index in range(self.copies - 1, 0, -1):
                if paths[index - 1].exists():
                    os.replace(paths[index - 1], paths[index])
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "ab") as stream:
            stream.write(payload)


def session_path(round_root, value):
    if not isinstance(value, str) or not Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("Session output must be an absolute path under the specified round")
    path = Path(value)
    if not path.is_relative_to(round_root) or path == round_root:
        raise ValueError("Session output is outside this round")
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Session output traverses a symlink")
    return path


class ChangedJSON:
    def __init__(self, max_bytes=MAX_BYTES):
        self.max_bytes, self.cache = max_bytes, OrderedDict()

    def read(self, path):
        path = Path(path)
        if path.is_symlink():
            return None, "symlink_rejected"
        try:
            before = path.stat()
            signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            if not stat.S_ISREG(before.st_mode) or before.st_size > self.max_bytes:
                return None, "not_regular_or_size_limit"
            previous = self.cache.get(str(path))
            if previous and previous[0] == signature:
                return previous[1], None
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                actual = os.fstat(stream.fileno())
                if (actual.st_dev, actual.st_ino, actual.st_size, actual.st_mtime_ns) != signature:
                    return None, "changed_while_opening"
                payload = stream.read(self.max_bytes + 1)
                after = os.fstat(stream.fileno())
            if len(payload) > self.max_bytes or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != signature:
                return None, "changed_while_reading"
            result = json.loads(payload, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            if not isinstance(result, dict):
                return None, "not_object"
            self.cache[str(path)] = (signature, result)
            while len(self.cache) > 6:
                self.cache.popitem(last=False)
            return result, None
        except (OSError, ValueError) as error:
            return None, type(error).__name__


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def validate_url(url):
    parts = urlsplit(url)
    try:
        local = parts.hostname == "localhost" or ipaddress.ip_address(parts.hostname or "").is_loopback
    except ValueError:
        local = False
    if (parts.scheme != "http" or not local or parts.username or parts.password
            or parts.path != "/status.json" or parts.query or parts.fragment or parts.port is None):
        raise ValueError("Monitor only permits a loopback HTTP /status.json GET with explicit port")
    return url


def get_status(url, timeout=.4):
    if not 0 < timeout <= .5:
        raise ValueError("HTTP timeout must be positive and at most0.5s")
    validate_url(url)
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(Request(url, method="GET", headers={"Cache-Control": "no-store"}), timeout=timeout) as response:
        payload = response.read(STATUS_BYTES + 1)
    if len(payload) > STATUS_BYTES:
        raise ValueError("Status response exceeds1MiB")
    value = json.loads(payload, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite status")))
    if not isinstance(value, dict) or not isinstance(value.get("status"), dict):
        raise ValueError("Invalid status envelope")
    return value


class StatusReader:
    """One daemon GET at a time, with an independent wall-clock wait budget."""
    def __init__(self, url, timeout):
        self.url, self.timeout, self.thread = url, timeout, None

    def read(self):
        if self.thread is not None and self.thread.is_alive():
            raise TimeoutError("Previous status GET still outstanding; no second request started")
        outcome = []
        def request():
            try:
                outcome.append((True, get_status(self.url, self.timeout)))
            except Exception as error:
                outcome.append((False, error))
        self.thread = threading.Thread(target=request, daemon=True, name="monitor-status-get")
        self.thread.start()
        self.thread.join(self.timeout)
        if self.thread.is_alive():
            raise TimeoutError("Status GET exceeded monitor wall-clock wait budget")
        passed, value = outcome[0]
        if not passed:
            raise value
        return value


def read_text(path, maximum=16384):
    with Path(path).open("r") as stream:
        return stream.read(maximum)


def proc_stat(text):
    close = text.rfind(")")
    if close < 0:
        raise ValueError("Malformed proc stat")
    fields = text[close + 2:].split()
    return {"state": fields[0], "ppid": int(fields[1]), "cpu_ticks": int(fields[11]) + int(fields[12]),
        "nice": int(fields[16]), "threads": int(fields[17]), "start_ticks": int(fields[19]),
        "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"), "processor": int(fields[36])}


def proc_record(pid, proc_root=Path("/proc")):
    base = proc_root / str(pid)
    result = proc_stat(read_text(base / "stat"))
    result["pid"] = pid
    for filename in ("io", "status"):
        try:
            values = {}
            for line in read_text(base / filename).splitlines():
                key, _, value = line.partition(":")
                if filename == "io" or key in ("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches"):
                    values[key] = int(value.strip().split()[0])
            result[filename] = values
        except (OSError, ValueError):
            result[filename] = None
    try:
        result["schedstat_cpu_runqueue_ns_timeslices"] = [int(v) for v in read_text(base / "schedstat").split()[:3]]
    except (OSError, ValueError):
        result["schedstat_cpu_runqueue_ns_timeslices"] = None
    try:
        with (base / "cmdline").open("rb") as stream:
            arguments = stream.read(8192).split(b"\0")
        result["is_collector"] = any(arg.endswith(b"/pico_simple_teleop.py") or arg == b"pico_simple_teleop.py" for arg in arguments)
        if result["is_collector"]:
            for index, arg in enumerate(arguments):
                if arg == b"--output" and index + 1 < len(arguments):
                    result["collector_output"] = os.fsdecode(arguments[index + 1])
                elif arg.startswith(b"--output="):
                    result["collector_output"] = os.fsdecode(arg.split(b"=", 1)[1])
    except OSError:
        result["is_collector"] = False
    # Verify identity after the other /proc reads. Never report a reused PID.
    if proc_stat(read_text(base / "stat"))["start_ticks"] != result["start_ticks"]:
        raise ValueError("PID changed during observation")
    return result


class ProcessFamily:
    def __init__(self, launcher_pid, proc_root=Path("/proc")):
        self.launcher_pid, self.proc_root = launcher_pid, Path(proc_root)
        self.known, self.previous = {}, {}
        try:
            self.known[launcher_pid] = proc_record(launcher_pid, self.proc_root)["start_ticks"]
        except (OSError, ValueError):
            pass
        self.launcher_identity = self.known.get(launcher_pid)

    def sample(self, now):
        pending, records, visited = [(pid, None) for pid in self.known], [], set()
        while pending and len(visited) < 256:
            pid, parent = pending.pop()
            if pid in visited:
                continue
            visited.add(pid)
            try:
                result = proc_record(pid, self.proc_root)
                if pid in self.known and result["start_ticks"] != self.known[pid]:
                    continue
                if pid not in self.known and (parent is None or result["ppid"] != parent
                        or proc_stat(read_text(self.proc_root / str(parent) / "stat"))["start_ticks"] != self.known.get(parent)):
                    continue
                self.known[pid] = result["start_ticks"]
                previous = self.previous.get((pid, result["start_ticks"]))
                if previous and now > previous[0]:
                    result["cpu_percent"] = 100 * (result["cpu_ticks"] - previous[1]) / os.sysconf("SC_CLK_TCK") / (now - previous[0])
                self.previous[(pid, result["start_ticks"])] = (now, result["cpu_ticks"])
                records.append(result)
                # Only walk children of an authenticated, owned descendant.
                for thread in list((self.proc_root / str(pid) / "task").iterdir())[:512]:
                    try:
                        pending.extend((int(v), pid) for v in read_text(thread / "children").split())
                    except (OSError, ValueError):
                        continue
            except (OSError, ValueError, IndexError):
                continue
        self.known = {row["pid"]: row["start_ticks"] for row in records}
        launcher_alive = any(row["pid"] == self.launcher_pid and row["start_ticks"] == self.launcher_identity
            and row["state"] not in ("Z", "X") for row in records)
        return {"launcher_alive": launcher_alive, "launcher_pid": self.launcher_pid,
            "launcher_start_ticks": self.launcher_identity, "processes": records,
            "descendant_limit": 256}


def system_sample(round_root, interface):
    result = {"pressure": {}, "interface": interface, "network": {}}
    for kind in ("cpu", "memory", "io"):
        try:
            result["pressure"][kind] = read_text(Path("/proc/pressure") / kind, 4096).strip()
        except OSError as error:
            result["pressure"][kind] = type(error).__name__
    try:
        usage = shutil.disk_usage(round_root)
        result["disk"] = dict(zip(("total_bytes", "used_bytes", "free_bytes"), usage))
    except OSError as error:
        result["disk"] = {"error": type(error).__name__}
    for key in ("rx_bytes", "tx_bytes", "rx_packets", "tx_packets", "rx_dropped", "tx_dropped", "rx_errors", "tx_errors"):
        try:
            result["network"][key] = int(read_text(Path("/sys/class/net") / interface / "statistics" / key, 64))
        except (OSError, ValueError):
            result["network"][key] = None
    return result


def result_summary(value):
    result = {key: value[key] for key in ("status", "error", "last_error", "traceback", "cleanup_errors",
        "episode_finalize_error", "last_loop_wall_timing", "maximum_lag_s", "wall_timing",
        "mobile_collection_progress", "pico_receive_telemetry") if key in value}
    result["latest_rejections"] = (value.get("rejections") or [])[-4:]
    return small(result)


class Events:
    def __init__(self):
        self.seen = OrderedDict()

    def emit(self, result, kind, detail, session, severity="diagnostic", identity=None):
        key = hashlib.sha256(encode([session, kind, detail if identity is None else identity])).hexdigest()
        if key in self.seen:
            return
        self.seen[key] = True
        while len(self.seen) > 4096:
            self.seen.popitem(last=False)
        result.append({"kind": kind, "severity": severity, "session_output": session,
            "event_id": key, "detail": small(detail)})

    def observe(self, status, result, session):
        events = []
        for source in (status, result):
            for key in ("error", "last_error", "cleanup_errors", "episode_finalize_error"):
                value = source.get(key)
                if value and not (isinstance(value, str) and value.startswith(OPERATOR_REASONS)):
                    self.emit(events, "collector_error", {"field": key, "value": small(value)}, session,
                              "fault", identity=small(value))
            if source.get("status") == "error":
                self.emit(events, "collector_error_status", {"status": "error"}, session, "fault")
            for rejection in source.get("rejections", []) if isinstance(source.get("rejections"), list) else []:
                if not isinstance(rejection, dict):
                    continue
                reason = str(rejection.get("reason", ""))
                severity = "diagnostic" if reason.startswith(OPERATOR_REASONS) else "fault"
                self.emit(events, "collector_rejection", rejection, session, severity)
            for key in ("last_episode", "review_result"):
                review = source.get(key)
                source_path = review.get("source_path") if isinstance(review, dict) else None
                belongs = (isinstance(source_path, str) and Path(source_path).is_absolute()
                    and ".." not in Path(source_path).parts and session is not None
                    and Path(source_path).is_relative_to(Path(session)) and Path(source_path) != Path(session))
                if belongs and (review.get("technical_blocked") is True
                        or review.get("status") in ("technical_blocked", "error")
                        or review.get("selection_status") == "technical_evidence_incomplete"):
                    self.emit(events, "review_technical", review, session, "fault",
                              identity=[review.get("uuid"), review.get("status"), review.get("error"), review.get("reasons")])
            progress = source.get("mobile_collection_progress")
            if isinstance(progress, dict) and progress.get("status") == "error" and progress.get("error"):
                self.emit(events, "collection_progress_error", progress, session, "fault",
                          identity=[progress.get("pending_uuid"), progress["error"]])
            telemetry = source.get("pico_receive_telemetry")
            if isinstance(telemetry, dict):
                for gap in telemetry.get("recent_receive_gaps_ge_50ms", []):
                    if isinstance(gap, dict):
                        self.emit(events, "source_callback_gap", gap, session, identity=[gap.get("sequence"), gap.get("host_monotonic_s")])
                device = telemetry.get("last_device_event")
                if isinstance(device, dict) and device:
                    self.emit(events, "source_device_event", device, session)
        return events


class Monitor:
    def __init__(self, *, round_root, launcher_pid, output, interface="wlp18s0",
                 url="http://127.0.0.1:43100/status.json", session_output=None, timeout=.4):
        self.round_root = Path(round_root).resolve(strict=True)
        self.output = session_path(self.round_root, str(Path(output).absolute()))
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,32}", interface) or interface in (".", ".."):
            raise ValueError("Invalid network interface name")
        if type(launcher_pid) is not int or launcher_pid <= 0:
            raise ValueError("Need the explicitly owned launcher PID")
        if not 0 < timeout <= .5:
            raise ValueError("Timeout must be at most0.5s")
        self.url, self.timeout, self.interface = validate_url(url), timeout, interface
        self.reader = StatusReader(self.url, timeout)
        self.session = session_path(self.round_root, str(session_output)) if session_output else None
        self.fixed_session = self.session
        if self.session and (self.output == self.session or self.output.is_relative_to(self.session)):
            raise ValueError("Monitor output cannot be inside the original collector session")
        self._unsafe_output = False
        self.lock_fd = os.open(self.round_root / ".pico_collection_monitor.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self.lock_fd)
            raise RuntimeError("A monitor already owns this collection round") from None
        try:
            marker = self.output / "monitor_identity.json"
            if self.output.exists() and any(self.output.iterdir()):
                if marker.is_symlink() or not marker.is_file() or json.loads(read_text(marker)).get("contract") != CONTRACT:
                    raise ValueError("Existing output is not an owned monitor directory; original evidence is untouched")
            self.output.mkdir(parents=True, exist_ok=True)
            atomic_json(marker, {"contract": CONTRACT, "launcher_pid": launcher_pid, "monitor_pid": os.getpid(),
                "poll_period_s": 1., "http_timeout_s": timeout, "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "nice": os.getpriority(os.PRIO_PROCESS, 0), "external_processes_modified": False})
        except BaseException:
            os.close(self.lock_fd); self.lock_fd = None
            raise
        self.family = ProcessFamily(launcher_pid)
        self.files, self.events, self.ring = ChangedJSON(), Events(), deque(maxlen=61)
        self.samples = RotatingJSONL(self.output / "samples.jsonl")
        self.event_log = RotatingJSONL(self.output / "events.jsonl")
        self.stop = threading.Event()
        self.http_seen, self.collector_seen, self.exit_since = False, False, None
        self.latest, self.ui_instance = None, None

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        sample = {"contract": CONTRACT, "observed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "monotonic_s": now, "monitor_pid": os.getpid(), "http": {}, "files": {}}
        status, result, raw_result, events = {}, {}, {}, []
        sample["process_family"] = self.family.sample(now)
        owned_outputs = {row.get("collector_output") for row in sample["process_family"]["processes"]
                         if row.get("is_collector") and row["state"] not in ("Z", "X")}
        try:
            envelope = self.reader.read()
            status = envelope["status"]
            instance = envelope.get("ui_session_id")
            candidate = status.get("output")
            accepted = False
            if candidate is not None:
                try:
                    candidate = session_path(self.round_root, candidate)
                    accepted = (candidate == self.fixed_session if self.fixed_session else str(candidate) in owned_outputs)
                    if accepted and (self.output == candidate or self.output.is_relative_to(candidate)):
                        self._unsafe_output = True
                        raise MonitorOutputConflict("Active collector session contains monitor output; monitor stopped without further writes")
                except MonitorOutputConflict:
                    raise
                except ValueError:
                    accepted = False
                if accepted:
                    self.session = candidate
            elif self.http_seen and instance == self.ui_instance:
                # Saving/resetting snapshots may omit output; only an already
                # authenticated UI instance can retain its session binding.
                accepted = True
            if accepted:
                self.http_seen, self.ui_instance = True, instance
            else:
                self.events.emit(events, "status_session_mismatch", {"reported_output": str(candidate),
                    "expected_output": str(self.fixed_session) if self.fixed_session else None,
                    "ui_session_id": instance}, str(self.session) if self.session else None,
                    identity=[str(candidate), instance])
                status = {}
            sample["http"] = {"available": True, "accepted_session": accepted, "ui_session_id": instance}
        except MonitorOutputConflict:
            raise
        except Exception as error:
            status = {}
            sample["http"] = {"available": False, "error": type(error).__name__ + ": " + str(error)[:256]}
            if self.http_seen:
                self.events.emit(events, "status_unreachable", sample["http"], str(self.session), "fault",
                                 identity=type(error).__name__)
        if self.session:
            value, error = self.files.read(self.session / "result.json")
            sample["files"]["result.json"] = {"read_error": error}
            if error in ("symlink_rejected", "not_regular_or_size_limit"):
                self.events.emit(events, "monitor_input_limited", {"file": "result.json", "reason": error},
                    str(self.session), "fault")
            if value:
                raw_result, result = value, result_summary(value)
        sample["session_output"] = str(self.session) if self.session else None
        sample["collector_status"] = small({key: status[key] for key in ("status", "output", "recording",
            "error", "last_error", "tracking", "last_episode", "review_result", "mobile_collection_progress",
            "last_loop_wall_timing", "pico_receive_telemetry", "mobile_planner") if key in status})
        sample["session_result"] = result
        sample["system"] = system_sample(self.round_root, self.interface)
        events.extend(self.events.observe(status, raw_result, sample["session_output"]))
        alive = [row for row in sample["process_family"]["processes"] if row["is_collector"] and row["state"] not in ("Z", "X")]
        self.collector_seen = self.collector_seen or bool(alive)
        exited = (not sample["process_family"]["launcher_alive"]
            or (self.collector_seen and not alive)
            or status.get("status") == "stopped" or result.get("status") in ("stopped", "error"))
        progress = status.get("mobile_collection_progress") or result.get("mobile_collection_progress") or {}
        complete = progress.get("status") == "complete" and progress.get("pending_uuid") is None
        if exited and not complete:
            if self.exit_since is None:
                self.exit_since = now
                self.events.emit(events, "collector_exit_observed", {"launcher_alive": sample["process_family"]["launcher_alive"],
                    "collector_seen": self.collector_seen, "exit_grace_s": 30}, sample["session_output"], "fault")
        else:
            self.exit_since = None
        stop_reason = "round_complete" if complete else "collector_exited_30s" if self.exit_since is not None and now - self.exit_since >= 30 else None
        sample["events"] = events
        sample["monitor_stop_reason"] = stop_reason
        self.ring.append(sample)
        while self.ring and now - self.ring[0]["monotonic_s"] > 60:
            self.ring.popleft()
        self.samples.append(sample)
        for event in events:
            self.event_log.append({"observed_utc": sample["observed_utc"], "monotonic_s": now, **event})
        if any(event["severity"] == "fault" for event in events):
            atomic_json(self.output / "snapshot.json", {"contract": CONTRACT, "trigger_events": events,
                "scope": "latest fault and preceding at most60s of1Hz observations; root cause is unproven",
                "samples": list(self.ring)})
        self.latest = sample
        atomic_json(self.output / "latest.json", sample)
        return stop_reason

    def close(self, reason="monitor_signal"):
        try:
            if self.latest is not None and not self._unsafe_output:
                atomic_json(self.output / "latest.json", {**self.latest, "monitor_stopped": True, "monitor_stop_reason": reason})
        finally:
            if self.lock_fd is not None:
                os.close(self.lock_fd); self.lock_fd = None

    def run(self):
        def stopped(signum, frame):
            self.stop.set()
        handlers = {sig: signal.signal(sig, stopped) for sig in (signal.SIGINT, signal.SIGTERM)}
        reason = "monitor_signal"
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                reason = self.poll() or "monitor_signal"
                if reason != "monitor_signal":
                    break
                self.stop.wait(max(0., 1. - (time.monotonic() - started)))
        except BaseException:
            reason = "monitor_error"
            raise
        finally:
            try:
                self.close(reason)
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
        return reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-root", type=Path, required=True)
    parser.add_argument("--launcher-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interface", default="wlp18s0")
    parser.add_argument("--url", default="http://127.0.0.1:43100/status.json")
    parser.add_argument("--session-output", type=Path)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    allowed = os.sched_getaffinity(0)
    cpu = max(allowed) if args.cpu is None else args.cpu
    if cpu not in allowed:
        parser.error("Monitor CPU must belong to this process's inherited affinity")
    os.sched_setaffinity(0, {cpu})
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    monitor = Monitor(**{key: value for key, value in vars(args).items() if key != "cpu"})
    print(json.dumps({"contract": CONTRACT, "monitor_pid": os.getpid(), "cpu": cpu, "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "external_processes_modified": False, "output": str(monitor.output)}), flush=True)
    print(json.dumps({"monitor_stopped": True, "reason": monitor.run()}), flush=True)


if __name__ == "__main__":
    main()
