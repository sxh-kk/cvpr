"""Bounded S3 range probes and per-bucket routing for the HM downloader.

Presigned URLs are passed to curl on stdin, never on the command line or in
status files. Probe bytes go to /dev/null. Existing transfers keep their client.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import urlsplit

MIB = 1024 ** 2
ROUTE_NAMES = {"direct": "直连", "proxy": "代理"}


def validate_proxy(url):
    parts = urlsplit(url)
    if (parts.scheme not in {"http", "https"} or not parts.hostname
            or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment or parts.path not in {"", "/"}):
        raise ValueError("--route-proxy 需要不含账号密码的 HTTP(S) 代理地址")
    try:
        parts.port
    except ValueError as exc:
        raise ValueError("--route-proxy 端口无效") from exc
    return url


@dataclass(frozen=True)
class ProbeResult:
    received_bytes: int = 0
    seconds: float = 0.0
    error: str = ""

    @property
    def speed(self):
        return self.received_bytes / self.seconds if self.ok else 0.0

    @property
    def ok(self):
        return not self.error and self.received_bytes > 0 and self.seconds > 0

    def summary(self):
        return {**asdict(self), "mib_per_second": self.speed / MIB}


@dataclass
class RoutePolicy:
    route: str = "direct"
    candidate: str = ""
    wins: int = 0
    last_switch: float = float("-inf")
    margin: float = 0.30
    required_wins: int = 2
    cooldown: float = 240.0

    def observe(self, results, now):
        current = results[self.route]
        other_route = "proxy" if self.route == "direct" else "direct"
        other = results[other_route]
        switch = False
        reason = "保持当前线路"
        if other.ok and not current.ok:
            switch, reason = True, "当前线路探测失败"
        elif current.ok and other.ok and other.speed >= current.speed * (1 + self.margin):
            self.wins = self.wins + 1 if self.candidate == other_route else 1
            self.candidate = other_route
            reason = f"候选 {ROUTE_NAMES[other_route]} {self.wins}/{self.required_wins}"
            if self.wins >= self.required_wins and now - self.last_switch >= self.cooldown:
                switch, reason = True, "连续测速更快"
            elif self.wins >= self.required_wins:
                reason = "切换冷却期内"
        else:
            self.candidate, self.wins = "", 0
            if not current.ok and not other.ok:
                reason = "两条线路探测失败，保留当前线路"
        if switch:
            self.route, self.last_switch = other_route, now
            self.candidate, self.wins = "", 0
            reason = f"切换至{ROUTE_NAMES[self.route]}：{reason}"
        return reason


class CurlRangeProber:
    def __init__(self, session, proxy, probe_bytes=2 * MIB, timeout=12):
        self.session = session
        self.proxy = validate_proxy(proxy)
        self.probe_bytes = probe_bytes
        self.timeout = timeout
        self.curl = shutil.which("curl")
        if self.curl is None:
            raise RuntimeError("--auto-route 需要 curl，请先安装 curl")
        self.regions = {}
        self.signers = {}
        self.signer_lock = threading.Lock()

    def _config(self, url, route):
        # curl reads this from stdin. Explicit settings also override NO_PROXY
        # and environment proxy variables without changing the process env.
        proxy = self.proxy if route == "proxy" else ""
        no_proxy = "" if route == "proxy" else "*"
        return "\n".join(f"{key} = {json.dumps(value)}" for key, value in
                         (("url", url), ("proxy", proxy), ("noproxy", no_proxy))) + "\n"

    def _invoke(self, url, route, extra, timeout):
        command = [self.curl, "--disable", "--silent", "--show-error",
                   "--connect-timeout", str(min(3, timeout)), "--max-time", str(timeout),
                   *extra, "--config", "-"]
        return subprocess.run(command, input=self._config(url, route), text=True,
                              capture_output=True, timeout=timeout + 2, check=False)

    def _region(self, bucket):
        if bucket in self.regions:
            return self.regions[bucket]
        # S3 returns the region header even if this account cannot list a
        # bucket. The global endpoint is used only for this bounded HEAD.
        for route in ("direct", "proxy"):
            try:
                response = self._invoke(f"https://{bucket}.s3.amazonaws.com/", route,
                                        ["--head"], min(5, self.timeout))
            except (subprocess.SubprocessError, OSError):
                continue
            match = re.search(r"^x-amz-bucket-region:\s*([a-z0-9-]+)\s*$",
                              response.stdout, re.IGNORECASE | re.MULTILINE)
            if response.returncode == 0 and match:
                region = match.group(1)
                self.regions[bucket] = region
                return region
        raise RuntimeError("无法探测 S3 桶所在区域")

    def _signed_url(self, bucket, key):
        from botocore.config import Config

        region = self._region(bucket)
        # Session.client is constructed under a lock; clients are then reused.
        with self.signer_lock:
            if region not in self.signers:
                self.signers[region] = self.session.client(
                    "s3", region_name=region,
                    config=Config(signature_version="s3v4", proxies={},
                                  s3={"addressing_style": "virtual"}))
            return self.signers[region].generate_presigned_url(
                "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=60)

    def _range(self, url, route, start, length):
        try:
            response = self._invoke(url, route,
                                    ["--output", os.devnull, "--max-filesize", str(length),
                                     "--range", f"{start}-{start + length - 1}",
                                     "--write-out", "%{http_code} %{size_download} %{time_total}"],
                                    self.timeout)
            status, received, elapsed = response.stdout.strip().split()
            received, elapsed = int(received), float(elapsed)
            if response.returncode not in {0, 28}:
                return ProbeResult(error=f"curl 退出码 {response.returncode}")
            if status != "206":
                return ProbeResult(error=f"HTTP {status}，未返回 Range 数据")
            if not 0 < received <= length or elapsed <= 0:
                return ProbeResult(error="未收到有效数据")
            # A deadline-expired partial 206 is a useful low-throughput sample.
            return ProbeResult(received, elapsed)
        except subprocess.TimeoutExpired:
            return ProbeResult(error="探测超过总时限")
        except (OSError, ValueError, subprocess.SubprocessError):
            return ProbeResult(error="无法执行或解析探测")

    def compare(self, item, round_number=0):
        location = urlsplit(item["source_path"])
        size = int(item["size"])
        if location.scheme != "s3" or not location.netloc or size < 1:
            error = ProbeResult(error="探测样本无效")
            return {"direct": error, "proxy": error}
        length = min(size, self.probe_bytes)
        # Rotate the range; both routes see exactly the same object and bytes.
        start = (round_number * length) % (size - length + 1)
        try:
            url = self._signed_url(location.netloc, location.path.lstrip("/"))
        except Exception:
            # Credentials or presigned URLs may appear in SDK exceptions.
            error = ProbeResult(error="无法获取桶区域或签名探测 URL")
            return {"direct": error, "proxy": error}
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {route: executor.submit(self._range, url, route, start, length)
                       for route in ("direct", "proxy")}
            return {route: future.result() for route, future in futures.items()}


class AutoRouteController:
    def __init__(self, clients, prober, files, status_path, interval=120, log=print):
        self.clients = clients
        self.prober = prober
        self.status_path = Path(status_path)
        self.interval = interval
        self.log = log
        self.samples = {}
        for item in files:
            bucket = urlsplit(item["source_path"]).netloc
            if item["size"] > 0 and item["size"] > self.samples.get(bucket, {}).get("size", 0):
                self.samples[bucket] = item
        self.policies = {bucket: RoutePolicy(cooldown=2 * interval) for bucket in self.samples}
        self.active = {bucket: 0 for bucket in self.samples}
        self.due = {bucket: 0.0 for bucket in self.samples}
        self.rounds = {bucket: 0 for bucket in self.samples}
        self.measurements = {}
        self.inflight = set()
        self.lock = threading.Lock()
        self.wakeup = threading.Event()
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        self.thread = threading.Thread(target=self._loop, name="s3-route-probes", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.wakeup.set()
        if self.thread is not None:
            self.thread.join()
        with self.lock:
            self._save_locked(running=False)

    @contextmanager
    def transfer(self, bucket):
        with self.lock:
            self.active[bucket] = self.active.get(bucket, 0) + 1
        self.wakeup.set()
        try:
            yield
        finally:
            with self.lock:
                self.active[bucket] -= 1

    def client_for(self, bucket):
        with self.lock:
            policy = self.policies.get(bucket)
            route = policy.route if policy else "direct"
            return route, self.clients[route]

    def _save_locked(self, running=True):
        data = {"pid": os.getpid(), "running": running, "updated_at": time.time(),
                "probe_interval": self.interval, "buckets": {bucket: {
                    "route": policy.route, "active_files": self.active.get(bucket, 0),
                    "candidate": policy.candidate, "candidate_wins": policy.wins,
                    **self.measurements.get(bucket, {})}
                    for bucket, policy in self.policies.items()}}
        try:
            temporary = self.status_path.with_name(self.status_path.name + ".tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(self.status_path)
        except OSError:
            # Optional status output must not interrupt dataset downloads.
            pass

    def _probe(self, bucket, round_number):
        try:
            results = self.prober.compare(self.samples[bucket], round_number)
        except Exception:
            failure = ProbeResult(error="探测异常")
            results = {"direct": failure, "proxy": failure}
        with self.lock:
            reason = self.policies[bucket].observe(results, time.monotonic())
            self.measurements[bucket] = {"probed_at": time.time(), "reason": reason,
                                         **{route: result.summary() for route, result in results.items()}}
            current = self.policies[bucket].route
            self.inflight.discard(bucket)
            self._save_locked()
        speeds = "，".join(f"{ROUTE_NAMES[route]} {result.speed / MIB:.2f} MiB/s"
                           if result.ok else f"{ROUTE_NAMES[route]}失败（{result.error}）"
                           for route, result in results.items())
        self.log(f"[选路] {bucket}：{speeds}；{reason}；当前{ROUTE_NAMES[current]}", flush=True)
        self.wakeup.set()

    def _loop(self):
        # At most two buckets are probed simultaneously, each with two routes.
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="s3-route-probe") as executor:
            while not self.stop.is_set():
                self.wakeup.clear()
                now = time.monotonic()
                with self.lock:
                    available = max(0, 2 - len(self.inflight))
                    ready = sorted((bucket for bucket in self.samples
                                    if self.active.get(bucket, 0) and bucket not in self.inflight
                                    and self.due[bucket] <= now), key=lambda bucket: self.due[bucket])[:available]
                    for bucket in ready:
                        self.inflight.add(bucket)
                        round_number = self.rounds[bucket]
                        self.rounds[bucket] += 1
                        self.due[bucket] = now + self.interval
                        executor.submit(self._probe, bucket, round_number)
                self.wakeup.wait(timeout=1)


def make_router(session, args, files):
    """Called after the directory lock and dry-run check, before raw transfers."""
    from botocore.config import Config

    prober = CurlRangeProber(session, args.route_proxy,
                            probe_bytes=int(args.probe_mib * MIB), timeout=args.probe_timeout)
    clients = {route: session.client("s3", config=Config(
        max_pool_connections=args.workers, connect_timeout=10, read_timeout=30,
        retries={"total_max_attempts": 3, "mode": "standard"},
        proxies={} if route == "direct" else {"http": args.route_proxy, "https": args.route_proxy}))
        for route in ("direct", "proxy")}
    return AutoRouteController(clients, prober, files, args.output_dir.resolve() / "route_status.json",
                               interval=args.probe_interval)
