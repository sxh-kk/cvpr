#!/usr/bin/env python3
"""Download about 250 recorded hours of formal EgoStandard full-body MCAP files.

Examples (Python >= 3.10; install requirements-egosuite.txt):
    python /cvpr/datasets/download_egosuite.py plan --hours 250
    python /cvpr/datasets/download_egosuite.py download
    python /cvpr/datasets/download_egosuite.py run --hours 250 --workers 4

The default destination is /gaozt-test1/datasets/EgoSuite. By default, cached
durations or at most --sample-files MCAP summaries estimate the mean episode
length. A fixed seed selects whole episodes for approximately --hours. Actual
durations are read locally after download. --duration-mode exact restores
per-file planning. These are recorded hours, not valid supervision hours.
Planning reads bounded HTTP ranges, never whole MCAP files. Existing selection
and completion journals from the earlier standalone script are supported.

HF_TOKEN / an existing HF login and inherited proxy/CA settings are used. No
interactive login is started. Network bytes come from Xet progress callbacks;
reconstructed file bytes are tracked separately. JSON state is written for
monitor_egosuite.py, including during planning, retry, failure and interruption.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import sys
import threading
import time
from types import SimpleNamespace
import uuid

BUCKET = "LightwheelAI/EgoStandard"
PREFIX = "EgoStandard-body/mcap"
DEFAULT_OUTPUT_DIR = Path("/gaozt-test1/datasets/EgoSuite")
REQUIRED_TOPICS = {
    "/pose/body", "/pose/left_hand", "/pose/right_hand",
    "/sensor/camera/head_left/video", "/sensor/camera/head_left/extrinsic",
}
LOCAL = threading.local()


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def control_dir(output):
    return output / ".egosuite-download"


def local_path(output, remote):
    relative = PurePosixPath(remote)
    if relative.is_absolute() or ".." in relative.parts or not remote.startswith(PREFIX + "/"):
        raise ValueError(f"Invalid source path: {remote}")
    destination = output.joinpath(*relative.parts)
    if not destination.resolve().is_relative_to(output.resolve()):
        raise ValueError(f"Destination escapes output directory: {remote}")
    return destination


@contextmanager
def output_lock(output):
    directory = control_dir(output)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "download.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another planner/downloader is already using this output directory") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def process_start_ticks(pid):
    try:
        return int((Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def safe_error(exc):
    message = str(exc)
    if "Connection refused" in message or "connection refused" in message:
        return "连接被拒绝，请检查当前终端的代理地址、代理监听端口和 SSH 转发。"
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status in (401, 403):
        return f"HF 返回 {status}，请检查已有登录 / HF_TOKEN 以及 EgoStandard 数据访问权限。"
    message = re.sub(r"hf_[A-Za-z0-9]+", "<token-hidden>", message)
    message = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?<query-hidden>", message)
    return f"{type(exc).__name__}: {message.splitlines()[0] if message else 'operation failed'}"[:600]


class Progress:
    def __init__(self, output, target_hours):
        self.output = output
        self.path = control_dir(output) / "progress.json"
        self.lock = threading.RLock()
        self.active = {}
        self.last_write = 0.0
        self.stop_heartbeat = threading.Event()
        self.state = {
            "schema_version": 1, "run_id": uuid.uuid4().hex,
            "pid": os.getpid(), "process_start_ticks": process_start_ticks(os.getpid()),
            "bucket": BUCKET, "prefix": PREFIX,
            "output_dir": str(output), "started_at": time.time(),
            "phase": "starting", "target_hours": target_hours,
            "total_files": 0, "total_bytes": 0, "selected_hours": 0.0,
            "completed_files": 0, "completed_bytes": 0, "completed_hours": 0.0,
            "network_bytes": 0, "resumed_files": 0,
            "candidates_total": 0, "candidates_examined": 0,
            "selected_files": 0, "skipped_files": 0, "last_error": None,
            "hours_are_estimated": False, "duration_sample_count": 0,
        }

    def publish(self, force=False):
        with self.lock:
            if not force and time.monotonic() - self.last_write < 1:
                return
            snapshot = {
                **self.state, "updated_at": time.time(),
                "reconstructed_bytes": self.state["completed_bytes"] + sum(
                    item["reconstructed_bytes"] for item in self.active.values()),
                "active_files": [{key: value for key, value in item.items() if not key.startswith("_")}
                                 for item in self.active.values()],
            }
            atomic_json(self.path, snapshot)
            self.last_write = time.monotonic()

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            self.publish(force=True)

    def begin_file(self, row, attempt):
        with self.lock:
            self.active[row["path"]] = {
                "path": row["path"], "size": row["size"], "attempt": attempt,
                "reconstructed_bytes": 0, "_last_network": 0,
            }
            self.publish(force=True)

    def transfer(self, path, report):
        with self.lock:
            task = self.active[path]
            network = int(report.total_transfer_bytes_completed)
            self.state["network_bytes"] += max(0, network - task["_last_network"])
            task["_last_network"] = network
            task["reconstructed_bytes"] = min(task["size"], max(0, int(report.total_bytes_completed)))
            self.publish()

    def finish_file(self, row, actual=None):
        with self.lock:
            self.active.pop(row["path"], None)
            self.state["completed_files"] += 1
            self.state["completed_bytes"] += row["size"]
            self.state["completed_hours"] += (actual or row)["duration_s"] / 3600
            self.publish(force=True)

    def failed_file(self, path, error):
        with self.lock:
            if path in self.active:
                self.active[path]["error"] = error
            self.state["last_error"] = error
            self.publish(force=True)

    def __enter__(self):
        self.publish(force=True)
        def heartbeat():
            while not self.stop_heartbeat.wait(2):
                self.publish(force=True)
        self.thread = threading.Thread(target=heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_heartbeat.set()
        self.thread.join()
        self.publish(force=True)


def summary_file_class():
    from huggingface_hub.hf_file_system import HfFileSystemFile
    from huggingface_hub.utils import get_session, hf_raise_for_status

    class SummaryRangeFile(HfFileSystemFile):
        def _fetch_range(self, start, end):
            if end <= start:
                return b""
            amount = end - start
            used = getattr(self, "summary_bytes", 0)
            if used + amount > 16 * 1024 * 1024:
                raise ValueError("MCAP summary exceeds the 16 MiB probe limit")
            headers = {
                **self.fs._api._build_hf_headers(),
                "Range": f"bytes={start}-{end - 1}", "Accept-Encoding": "identity",
            }
            with get_session().stream("GET", self.url(), headers=headers, timeout=30,
                                      follow_redirects=True) as response:
                hf_raise_for_status(response)
                expected = f"bytes {start}-{end - 1}/"
                if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                    raise ValueError("Server ignored Range; refusing a whole-file probe")
                parts, received = [], 0
                for part in response.iter_bytes(chunk_size=64 * 1024):
                    received += len(part)
                    if received > amount:
                        raise ValueError("Range response exceeded requested length")
                    parts.append(part)
                if received != amount:
                    raise ValueError("Truncated Range response")
            self.summary_bytes = used + amount
            return b"".join(parts)

    return SummaryRangeFile


def summary_row(summary, file):
    if summary is None or summary.statistics is None:
        raise ValueError("No MCAP Statistics summary")
    missing = REQUIRED_TOPICS - {channel.topic for channel in summary.channels.values()}
    if missing:
        raise ValueError(f"Missing required topics: {', '.join(sorted(missing))}")
    stats = summary.statistics
    duration = (stats.message_end_time - stats.message_start_time) / 1e9
    if not math.isfinite(duration) or duration <= 0 or stats.message_count <= 0:
        raise ValueError("Empty or invalid recording time span")
    return {"path": file.path, "size": file.size, "xet_hash": file.xet_hash,
            "duration_s": duration, "message_count": stats.message_count}


def probe(file):
    from huggingface_hub import HfFileSystem
    from mcap.reader import make_reader
    if not hasattr(LOCAL, "fs"):
        LOCAL.fs = HfFileSystem()
        LOCAL.range_class = summary_file_class()
    with LOCAL.range_class(LOCAL.fs, f"buckets/{BUCKET}/{file.path}", mode="rb", size=file.size,
                           block_size=64 * 1024, cache_type="readahead") as stream:
        summary = make_reader(stream).get_summary()
    return summary_row(summary, file)


def validate_selection(selection, output):
    if selection.get("bucket") != BUCKET or selection.get("prefix") != PREFIX:
        raise ValueError(f"Selection source mismatch: expected {BUCKET}/{PREFIX}; "
                         "archive the old selection before switching datasets")
    target = selection.get("target_hours", 0)
    if not math.isfinite(target) or target <= 0:
        raise ValueError("Invalid requested hours in selection")
    rows = selection.get("files", [])
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Empty selection or duplicate paths")
    for row in rows:
        local_path(output, row["path"])
        if int(row["size"]) <= 0 or not math.isfinite(row["duration_s"]) or row["duration_s"] <= 0:
            raise ValueError("Invalid size or duration in selection")
    if sum(row["size"] for row in rows) != selection["total_bytes"]:
        raise ValueError("Selection byte total does not match files")
    hours = sum(row["duration_s"] for row in rows) / 3600
    estimated_all = (selection.get("duration_mode") == "sample" and selection.get("estimated_source_exhausted")
                     and len(rows) == selection.get("candidates_total"))
    if not math.isclose(hours, selection["recorded_hours"], rel_tol=1e-8) or (hours < target and not estimated_all):
        raise ValueError("Selection does not meet the requested recorded hours")
    return rows


def print_summary(selection, output):
    label = "估算约" if selection.get("hours_are_estimated") else "实测"
    print(f"清单：{len(selection['files'])} 个 episode，{label} {selection['recorded_hours']:.3f} h，"
          f"{selection['total_bytes'] / 2**30:.2f} GiB\n"
          f"清单位置：{control_dir(output) / 'selection.json'}", flush=True)


def sampled_selection(output, files, cache, hours, seed, workers, sample_files, progress):
    """Bound remote probing; cached random durations need no new MCAP requests."""
    samples = [cache[(item.path, item.xet_hash)] for item in files
               if (item.path, item.xet_hash) in cache and cache[(item.path, item.xet_hash)]["size"] == item.size]
    progress.update(phase="sampling", candidates_total=len(files), duration_sample_count=len(samples),
                    candidates_examined=len(samples), hours_are_estimated=True)
    needed = max(0, min(sample_files, len(files)) - len(samples))
    candidates = [item for item in files if (item.path, item.xet_hash) not in cache
                  or cache[(item.path, item.xet_hash)]["size"] != item.size][:needed]
    cache_path = control_dir(output) / "durations.jsonl"
    if candidates:
        with ThreadPoolExecutor(max_workers=workers) as executor, cache_path.open("a") as cache_file:
            futures = {executor.submit(probe, item): item for item in candidates}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    row = future.result()
                except ValueError as exc:
                    progress.update(skipped_files=progress.state["skipped_files"] + 1)
                    print(f"跳过抽样 {item.path}: {safe_error(exc)}", flush=True)
                    continue
                samples.append(row)
                cache[(item.path, item.xet_hash)] = row
                cache_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                cache_file.flush()
                progress.update(duration_sample_count=len(samples), candidates_examined=len(samples))
    if not samples:
        raise ValueError("No valid duration samples; cannot estimate recorded hours")
    mean_seconds = sum(row["duration_s"] for row in samples) / len(samples)
    selected, seconds = [], 0.0
    for item in files:
        known = cache.get((item.path, item.xet_hash))
        if known is not None and known["size"] == item.size:
            row = {**known, "duration_estimated": False}
        else:
            row = {"path": item.path, "size": item.size, "xet_hash": item.xet_hash,
                   "duration_s": mean_seconds, "duration_estimated": True}
        selected.append(row)
        seconds += row["duration_s"]
        if seconds >= hours * 3600:
            break
    estimated = any(row["duration_estimated"] for row in selected)
    result = {
        "bucket": BUCKET, "prefix": PREFIX, "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_hours": hours, "seed": seed, "recorded_hours": seconds / 3600,
        "duration_definition": "cached/remote sample mean; actual time measured locally after download",
        "duration_mode": "sample", "hours_are_estimated": estimated,
        "duration_sample_count": len(samples), "mean_episode_seconds": mean_seconds,
        "candidates_total": len(files), "estimated_source_exhausted": seconds < hours * 3600,
        "total_bytes": sum(row["size"] for row in selected), "files": selected,
    }
    if seconds < hours * 3600:
        print(f"正式目录全量估算约 {seconds / 3600:.1f} h，低于目标 {hours:g} h；已选择该目录全部文件。", flush=True)
    print(f"使用 {len(samples)} 段时长样本，平均 {mean_seconds:.1f}s/段；其余文件不再查询远端时长。", flush=True)
    return result


def plan(output, hours, seed, workers, progress, duration_mode="sample", sample_files=32):
    from huggingface_hub import HfApi
    path = control_dir(output) / "selection.json"
    if path.exists():
        selection = json.loads(path.read_text())
        validate_selection(selection, output)
        if selection["target_hours"] != hours or selection["seed"] != seed:
            raise ValueError("Existing selection uses different hours/seed; choose a new output directory")
        progress.update(phase="planned", total_files=len(selection["files"]),
                        total_bytes=selection["total_bytes"], selected_hours=selection["recorded_hours"],
                        selected_files=len(selection["files"]),
                        hours_are_estimated=selection.get("hours_are_estimated", False),
                        duration_sample_count=selection.get("duration_sample_count", 0))
        print_summary(selection, output)
        return selection
    progress.update(phase="listing")
    print(f"读取正式目录：hf://buckets/{BUCKET}/{PREFIX}/", flush=True)
    files = sorted((item for item in HfApi().list_bucket_tree(BUCKET, prefix=PREFIX, recursive=True)
                    if item.path.endswith(".mcap")), key=lambda item: item.path)
    if not files:
        raise ValueError("No MCAP files found")
    random.Random(seed).shuffle(files)
    cache_path = control_dir(output) / "durations.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            try:
                row = json.loads(line)
                if row["duration_s"] > 0 and math.isfinite(row["duration_s"]):
                    cache[(row["path"], row["xet_hash"])] = row
            except (ValueError, KeyError, TypeError):
                continue
    if duration_mode == "sample":
        result = sampled_selection(output, files, cache, hours, seed, workers, sample_files, progress)
        validate_selection(result, output)
        atomic_json(path, result)
        progress.update(phase="planned", total_files=len(result["files"]), total_bytes=result["total_bytes"],
                        selected_files=len(result["files"]), selected_hours=result["recorded_hours"],
                        hours_are_estimated=result["hours_are_estimated"])
        print_summary(result, output)
        return result
    selected, seconds, examined, skipped = [], 0.0, 0, 0
    progress.update(phase="planning", candidates_total=len(files))
    with ThreadPoolExecutor(max_workers=workers) as executor, cache_path.open("a") as cache_file:
        for start in range(0, len(files), workers):
            batch = files[start:start + workers]
            futures = {item.path: executor.submit(probe, item) for item in batch
                       if (item.path, item.xet_hash) not in cache}
            resolved = []
            for item in batch:
                key = (item.path, item.xet_hash)
                examined += 1
                try:
                    row = cache.get(key)
                    if row is None or row["size"] != item.size:
                        row = futures[item.path].result() if item.path in futures else probe(item)
                        cache_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        cache_file.flush()
                        cache[key] = row
                    resolved.append(row)
                except ValueError as exc:
                    skipped += 1
                    print(f"跳过 {item.path}: {safe_error(exc)}", flush=True)
                progress.update(candidates_examined=examined, skipped_files=skipped)
            for row in resolved:
                selected.append(row)
                seconds += row["duration_s"]
                progress.update(selected_files=len(selected), selected_hours=seconds / 3600)
                if seconds >= hours * 3600:
                    break
            print(f"已检查 {examined}/{len(files)}，选取 {len(selected)} 段 / {seconds / 3600:.3f} h", flush=True)
            if seconds >= hours * 3600:
                break
    if seconds < hours * 3600:
        raise ValueError(f"Only {seconds / 3600:.3f} h available with required topics; "
                         "no complete selection saved or MCAP data downloaded")
    selection = {
        "bucket": BUCKET, "prefix": PREFIX, "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_hours": hours, "seed": seed, "recorded_hours": seconds / 3600,
        "duration_definition": "MCAP Statistics message_end_time - message_start_time",
        "total_bytes": sum(row["size"] for row in selected), "files": selected,
    }
    validate_selection(selection, output)
    atomic_json(path, selection)
    progress.update(phase="planned", total_files=len(selected), total_bytes=selection["total_bytes"])
    print_summary(selection, output)
    return selection


def verified_file(path, row, marker):
    try:
        stat = path.stat()
        return path.is_file() and stat.st_size == row["size"] and marker.get("xet_hash") == row["xet_hash"] and marker.get("mtime_ns") == stat.st_mtime_ns
    except OSError:
        return False


def completed_duration(path, row, marker):
    duration = marker.get("actual_duration_s")
    if duration is None and row.get("duration_estimated"):
        from mcap.reader import make_reader
        with path.open("rb") as handle:
            duration = summary_row(make_reader(handle).get_summary(), SimpleNamespace(**row))["duration_s"]
    return duration if duration is not None else row["duration_s"]


def prepare_download(output, selection, api, progress, retries=3, retry_delay=3):
    rows = validate_selection(selection, output)
    done_path = control_dir(output) / "completed.json"
    done = json.loads(done_path.read_text()) if done_path.exists() else {}
    existing = []
    for row in rows:
        destination = local_path(output, row["path"])
        marker = done.get(row["path"], {})
        if verified_file(destination, row, marker):
            existing.append({**row, "duration_s": completed_duration(destination, row, marker)})
    progress.update(phase="checking", total_files=len(rows), total_bytes=selection["total_bytes"],
                    selected_hours=selection["recorded_hours"], target_hours=selection["target_hours"],
                    selected_files=len(rows), hours_are_estimated=selection.get("hours_are_estimated", False),
                    duration_sample_count=selection.get("duration_sample_count", 0),
                    completed_files=len(existing), resumed_files=len(existing),
                    completed_bytes=sum(row["size"] for row in existing),
                    completed_hours=sum(row["duration_s"] for row in existing) / 3600)
    for attempt in range(1, retries + 2):
        try:
            current = {item.path: item for item in api.get_bucket_paths_info(BUCKET, [row["path"] for row in rows])}
            progress.update(last_error=None)
            break
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            permanent = isinstance(exc, (ValueError, ImportError, TypeError, AttributeError))
            if permanent or (status is not None and status not in (408, 429, 500, 502, 503, 504)) or attempt > retries:
                raise
            delay = min(60, retry_delay * 2 ** min(attempt - 1, 10))
            error = f"HF 远端版本检查：{safe_error(exc)}；{delay:g}s 后重试 {attempt}/{retries}。"
            progress.update(last_error=error)
            print(error, flush=True)
            time.sleep(delay)
    pending, complete = [], []
    for row in rows:
        remote = current.get(row["path"])
        if remote is None or remote.xet_hash != row["xet_hash"] or remote.size != row["size"]:
            raise ValueError(f"Remote file changed or disappeared: {row['path']}")
        destination = local_path(output, row["path"])
        marker = done.get(row["path"], {})
        part = destination.with_name(destination.name + ".part")
        if not destination.exists() and marker.get("state") == "verified" and verified_file(part, row, marker):
            part.replace(destination)
            marker["state"] = "complete"
            atomic_json(done_path, done)
        if verified_file(destination, row, marker):
            complete.append({**row, "duration_s": completed_duration(destination, row, marker)})
        elif destination.exists():
            raise ValueError(f"Existing file is unverified; refusing overwrite: {destination}")
        else:
            pending.append(row)
    required = sum(row["size"] for row in pending)
    free = shutil.disk_usage(output).free
    if required and free < required + 5 * 2**30:
        raise ValueError(f"Need {required / 2**30:.2f} GiB plus 5 GiB free; available {free / 2**30:.2f} GiB")
    progress.update(completed_files=len(complete), resumed_files=len(complete),
                    completed_bytes=sum(row["size"] for row in complete),
                    completed_hours=sum(row["duration_s"] for row in complete) / 3600)
    return pending, done


def xet_download(api, row, destination, callback):
    """Use the pinned HF/Xet bucket implementation with an explicit callback."""
    from hf_xet import XetFileInfo
    from huggingface_hub.utils import build_hf_headers
    from huggingface_hub.utils._xet import get_xet_session, xet_headers_without_auth
    metadata = api.get_bucket_file_metadata(BUCKET, row["path"])
    if metadata.size != row["size"] or metadata.xet_file_data.file_hash != row["xet_hash"]:
        raise ValueError(f"Remote file changed during download: {row['path']}")
    headers = build_hf_headers(library_name="egosuite-downloader", library_version="1.0")
    with get_xet_session().new_file_download_group(
        token_refresh_url=metadata.xet_file_data.refresh_route,
        token_refresh_headers=headers, custom_headers=xet_headers_without_auth(headers),
        progress_callback=callback,
    ) as group:
        group.start_download_file(XetFileInfo(row["xet_hash"], row["size"]), str(destination))


def download_one(output, row, api, progress, done, stop, retries, retry_delay, backend=xet_download):
    destination = local_path(output, row["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    for attempt in range(1, retries + 2):
        if stop.is_set():
            raise InterruptedError("Download cancelled")
        progress.begin_file(row, attempt)
        try:
            backend(api, row, part, lambda report, *_: progress.transfer(row["path"], report))
            if part.stat().st_size != row["size"]:
                raise ValueError(f"Downloaded size mismatch: {row['path']}")
            from mcap.reader import make_reader
            remote = SimpleNamespace(**row)
            with part.open("rb") as handle:
                actual = summary_row(make_reader(handle).get_summary(), remote)
            if not row.get("duration_estimated") and not math.isclose(actual["duration_s"], row["duration_s"], rel_tol=1e-8, abs_tol=1e-6):
                raise ValueError(f"Downloaded duration differs from selection: {row['path']}")
            with progress.lock:
                # Journal a verified part before rename; either side of a crash
                # is recoverable without treating an arbitrary full-size part as done.
                done[row["path"]] = {"xet_hash": row["xet_hash"], "size": row["size"],
                                     "mtime_ns": part.stat().st_mtime_ns, "state": "verified",
                                     "actual_duration_s": actual["duration_s"]}
                atomic_json(control_dir(output) / "completed.json", done)
                part.replace(destination)
                done[row["path"]]["state"] = "complete"
                atomic_json(control_dir(output) / "completed.json", done)
                progress.finish_file(row, actual)
            print(f"完成 [{progress.state['completed_files']}/{progress.state['total_files']}] {row['path']}", flush=True)
            return
        except Exception as exc:
            progress.failed_file(row["path"], safe_error(exc))
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if isinstance(exc, (ValueError, ImportError)) or status in (400, 401, 403, 404) or attempt > retries or stop.is_set():
                raise
            print(f"重试 {attempt}/{retries}: {row['path']} ({safe_error(exc)})", flush=True)
            if stop.wait(min(60, retry_delay * 2 ** (attempt - 1))):
                raise InterruptedError("Download cancelled")


def download(output, selection, workers, retries, retry_delay, progress):
    from huggingface_hub import HfApi
    from huggingface_hub.utils._xet import abort_xet_session
    api = HfApi()
    pending, done = prepare_download(output, selection, api, progress, retries, retry_delay)
    progress.update(phase="downloading", last_error=None)
    stop = threading.Event()
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = []
    try:
        futures = [executor.submit(download_one, output, row, api, progress, done, stop,
                                   retries, retry_delay) for row in pending]
        for future in as_completed(futures):
            future.result()
    except BaseException:
        stop.set()
        abort_xet_session()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    progress.update(phase="complete", last_error=None)
    print(f"下载完成：实际录制时长 {progress.state['completed_hours']:.3f} h，{output / PREFIX}", flush=True)


def check_dependencies():
    expected = {"huggingface_hub": "2.1.1", "hf_xet": "1.7.0", "mcap": "1.5.0"}
    for name, version in expected.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = "missing"
        if installed != version:
            raise ValueError(f"Need {name}=={version} (found {installed}); install requirements-egosuite.txt")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("command", nargs="?", choices=["run", "plan", "download"], default="run")
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hours", type=float, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4, help="同时下载的文件数")
    parser.add_argument("--probe-workers", type=int, default=8, help="时长查询并发数")
    parser.add_argument("--duration-mode", choices=["sample", "exact"], default="sample", help="抽样估算 / 逐文件精确查询")
    parser.add_argument("--sample-files", type=int, default=32, help="抽样目标数；已有足够缓存时不再查询远端时长")
    parser.add_argument("--retries", type=int, default=3, help="HF 版本检查及每个文件失败后的重试次数")
    parser.add_argument("--retry-delay", type=float, default=3, help="首次重试等待秒数；后续递增")
    parser.add_argument("--dry-run", action="store_true", help="只生成清单，不下载完整 MCAP")
    args = parser.parse_args(argv)
    if not math.isfinite(args.hours) or args.hours <= 0 or min(args.workers, args.probe_workers) < 1:
        parser.error("hours、workers 和 probe-workers 必须大于零")
    if args.retries < 0 or not math.isfinite(args.retry_delay) or args.retry_delay < 0:
        parser.error("retries 和 retry-delay 不能小于零")
    if args.sample_files < 1:
        parser.error("sample-files 必须大于零")
    if args.dry_run:
        args.command = "plan"
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir.resolve()
    try:
        check_dependencies()
        with output_lock(output), Progress(output, args.hours) as progress:
            try:
                if args.command in ("plan", "run"):
                    selection = plan(output, args.hours, args.seed, args.probe_workers, progress,
                                     args.duration_mode, args.sample_files)
                else:
                    selection = json.loads((control_dir(output) / "selection.json").read_text())
                if args.command != "plan":
                    download(output, selection, args.workers, args.retries, args.retry_delay, progress)
            except KeyboardInterrupt:
                progress.update(phase="interrupted", last_error="用户中断；重跑会保留已完成文件。")
                raise
            except Exception as exc:
                progress.update(phase="error", last_error=safe_error(exc))
                raise
        return 0
    except KeyboardInterrupt:
        print("\n下载已中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {safe_error(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
