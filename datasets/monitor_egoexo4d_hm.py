#!/usr/bin/env python3
"""Monitor the existing HM downloader without extra packages or credentials.

    python /cvpr/datasets/monitor_egoexo4d_hm.py
    python /cvpr/datasets/monitor_egoexo4d_hm.py --pid 58416 --interval 5
    python /cvpr/datasets/monitor_egoexo4d_hm.py --once --interval 2
    python /cvpr/datasets/monitor_egoexo4d_hm.py --exclude-bucket ego4d-indiana

Progress comes from download_plan.json and file sizes, including HF/S3 partial
files. Speeds estimate bytes written to the dataset directory, not network
traffic. Sparse temporary files count allocated blocks to avoid treating a
preallocated logical file size as a completed transfer. Completed-file counts
check size only, matching the downloader's resume rule.
"""

from __future__ import annotations

import argparse
import base64
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time

from download_egoexo4d_hm import filter_s3_files

DEFAULT_OUTPUT_DIR = Path("/gaozt-test1/datasets/EgoExo-4D-HM")
PARTS = ("HM", "metadata", "takes", "take_trajectory", "take_vrs_noimagestream", "captures")
STATE_NAMES = {"R": "运行", "S": "等待", "D": "等待 I/O", "T": "暂停", "t": "暂停", "Z": "已退出"}


@dataclass(frozen=True)
class FileSpec:
    path: str
    part: str
    size: int


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="监控 HM 下载的进度、落盘速度、ETA 和进程状态（无需第三方依赖）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="下载存储目录")
    parser.add_argument("--pid", type=int, help="指定下载进程；默认自动寻找同一存储目录的下载进程")
    parser.add_argument("--interval", type=float, default=5, help="刷新间隔，秒")
    parser.add_argument("--once", action="store_true", help="采样两次后输出一次，包含速度")
    parser.add_argument("--no-clear", action="store_true", help="每次追加输出，适合保存日志")
    parser.add_argument("--json", action="store_true", help="每次输出一行 JSON，便于其他程序读取")
    parser.add_argument("--exclude-bucket", nargs="+", action="extend", default=[], metavar="BUCKET",
                        help="按下载器的相同排除范围统计进度；完整清单保留")
    args = parser.parse_args(argv)
    if not 0.2 <= args.interval <= 60:
        parser.error("--interval 必须在 0.2 到 60 秒之间")
    if args.pid is not None and args.pid < 1:
        parser.error("--pid 必须大于零")
    return args


class ProgressScanner:
    def __init__(self, root, plan, excluded_buckets=()):
        self.root = Path(root)
        selected, excluded = filter_s3_files(plan, excluded_buckets)
        self.excluded_buckets = sorted(set(excluded_buckets))
        self.excluded_files = len(excluded)
        self.excluded_total_bytes = sum(item["size"] or 0 for item in excluded)
        self.files = {}
        self.hf_partial = {}
        self.unknown_files = 0
        for folder, items in (("EgoExo4D-HM", plan["hm_files"]), ("EgoExo4D", selected)):
            for item in items:
                relative = PurePosixPath(item["relative_path"])
                if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                    raise ValueError(f"清单包含无效路径：{relative}")
                if item.get("size") is None:
                    self.unknown_files += 1
                    continue
                spec = FileSpec(f"{folder}/{relative}", "HM" if folder == "EgoExo4D-HM" else item["part"], int(item["size"]))
                if spec.size < 0:
                    raise ValueError(f"清单包含无效大小：{relative}")
                self.files[spec.path] = spec
                if folder == "EgoExo4D-HM":
                    parent = f"{folder}/.cache/huggingface/download/{relative.parent}"
                    metadata_name = relative.name + ".metadata"
                    short_hash = base64.urlsafe_b64encode(hashlib.sha1(metadata_name.encode()).digest()).decode()
                    # New HF versions hash the metadata filename; older ones use
                    # the original filename. Both add an etag and .incomplete.
                    for prefix in (relative.name, short_hash):
                        self.hf_partial[(parent, prefix)] = spec.path
        # Walk only branches that can contain expected files or HF partials.
        self.directories = set()
        for directory in [str(PurePosixPath(path).parent) for path in self.files] + [key[0] for key in self.hf_partial]:
            path = PurePosixPath(directory)
            self.directories.add(str(path))
            self.directories.update(str(parent) for parent in path.parents if str(parent) != ".")

    def identify(self, relative):
        if relative in self.files:
            return relative, True
        if relative.startswith("EgoExo4D-HM/.cache/huggingface/download/") and relative.endswith(".incomplete"):
            path = PurePosixPath(relative)
            prefix = path.name[:-len(".incomplete")]
            # Remove the etag from the temporary filename.
            result = self.hf_partial.get((str(path.parent), prefix.rsplit(".", 1)[0]))
            result = result or self.hf_partial.get((str(path.parent), prefix))
            return (result, False) if result else (None, False)
        base, marker, suffix = relative.rpartition(".part")
        if marker and (not suffix or suffix.startswith(".")) and base in self.files:
            return base, False
        return None, False

    @staticmethod
    def allocated_bytes(stat):
        return min(stat.st_size, getattr(stat, "st_blocks", (stat.st_size + 511) // 512) * 512)

    def scan(self):
        # Keep maxima by destination: stale partials and atomic renames must not
        # double count bytes; a complete destination always wins.
        present, complete = {}, set()
        for folder in ("EgoExo4D-HM", "EgoExo4D"):
            base = self.root / folder
            if not base.is_dir():
                continue
            for directory, dirs, names in os.walk(base):
                parent = Path(directory).relative_to(self.root).as_posix()
                dirs[:] = [name for name in dirs if f"{parent}/{name}" in self.directories]
                for name in names:
                    key, final = self.identify(f"{parent}/{name}")
                    if key is None:
                        continue
                    try:
                        stat = (Path(directory) / name).stat()
                    except (FileNotFoundError, OSError):
                        # Concurrent moves/deletions are normal while downloading.
                        continue
                    expected = self.files[key].size
                    if final and stat.st_size == expected:
                        complete.add(key)
                        present[key] = expected
                    elif key not in complete:
                        amount = min(expected, self.allocated_bytes(stat))
                        present[key] = max(present.get(key, 0), amount)
        parts = {part: dict(total_bytes=0, present_bytes=0, complete_files=0, total_files=0, partial_files=0)
                 for part in PARTS}
        for key, spec in self.files.items():
            group = parts.setdefault(spec.part, dict(total_bytes=0, present_bytes=0, complete_files=0, total_files=0, partial_files=0))
            group["total_bytes"] += spec.size
            group["total_files"] += 1
            group["present_bytes"] += present.get(key, 0)
            group["complete_files"] += key in complete
            group["partial_files"] += key in present and key not in complete
        totals = {field: sum(group[field] for group in parts.values()) for field in next(iter(parts.values()))}
        return {**totals, "parts": parts, "unknown_files": self.unknown_files,
                "excluded_buckets": self.excluded_buckets, "excluded_files": self.excluded_files,
                "excluded_total_bytes": self.excluded_total_bytes}


class RateTracker:
    def __init__(self, window=60):
        self.window = window
        self.samples = deque()
        self.started = None
        self.initial_bytes = None

    def update(self, now, amount):
        if self.started is None:
            self.started, self.initial_bytes = now, amount
        previous = self.samples[-1] if self.samples else None
        current = None if previous is None or now <= previous[0] else max(0, amount - previous[1]) / (now - previous[0])
        # A retry can delete a partial file. Reset the ETA window on regression
        # so its replacement is not interpreted as negative transfer speed.
        if previous and amount < previous[1]:
            self.samples.clear()
        self.samples.append((now, amount))
        while len(self.samples) > 2 and self.samples[1][0] <= now - self.window:
            self.samples.popleft()
        oldest = self.samples[0]
        smooth = max(0, amount - oldest[1]) / (now - oldest[0]) if now > oldest[0] else 0
        average = max(0, amount - self.initial_bytes) / (now - self.started) if now > self.started else 0
        return {"speed_bytes_per_second": current, "average_bytes_per_second": average,
                "eta_speed_bytes_per_second": smooth, "monitor_seconds": now - self.started}


def downloader_root(pid, args):
    # Inspect only the output directory argument; never read credentials or env.
    value = str(DEFAULT_OUTPUT_DIR)
    for i, arg in enumerate(args):
        if arg == "--output-dir" and i + 1 < len(args):
            value = args[i + 1]
        elif arg.startswith("--output-dir="):
            value = arg.split("=", 1)[1]
    path = Path(value)
    if not path.is_absolute():
        path = (Path("/proc") / str(pid) / "cwd").resolve() / path
    return path.resolve()


def process_info(root, requested_pid=None):
    processes = []
    candidates = [Path("/proc") / str(requested_pid)] if requested_pid else Path("/proc").iterdir()
    for path in candidates:
        if not path.name.isdigit():
            continue
        try:
            args = path.joinpath("cmdline").read_bytes().decode(errors="replace").strip("\0").split("\0")
            if not any(Path(arg).name == "download_egoexo4d_hm.py" for arg in args):
                continue
            if requested_pid is None and downloader_root(path.name, args) != root:
                continue
            # /proc/stat's process name may itself contain spaces or parentheses.
            fields = path.joinpath("stat").read_text().rsplit(")", 1)[1].split()
            uptime = float(Path("/proc/uptime").read_text().split()[0])
            elapsed = max(0, uptime - int(fields[19]) / os.sysconf("SC_CLK_TCK"))
            processes.append({"pid": int(path.name), "state": fields[0],
                              "state_text": STATE_NAMES.get(fields[0], fields[0]),
                              "elapsed_seconds": elapsed,
                              "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE")})
        except (OSError, ValueError, IndexError):
            continue
    return sorted(processes, key=lambda item: item["pid"])


def snapshot(scanner, rates, requested_pid=None):
    report = scanner.scan()
    report.update(rates.update(time.monotonic(), report["present_bytes"]))
    report["timestamp"] = datetime.now().astimezone().isoformat(timespec="seconds")
    report["output_dir"] = str(scanner.root)
    report["processes"] = process_info(scanner.root, requested_pid)
    report["progress_percent"] = 100 * report["present_bytes"] / report["total_bytes"] if report["total_bytes"] else 0
    remaining = report["total_bytes"] - report["present_bytes"]
    speed = report["eta_speed_bytes_per_second"]
    report["eta_seconds"] = remaining / speed if speed > 0 else None
    if report["complete_files"] == report["total_files"] and not report["unknown_files"]:
        report["status"] = "清单文件已全部落盘（按大小检查）"
        report["eta_seconds"] = 0
    elif not report["processes"]:
        report["status"] = "未找到下载进程，数据尚未下完"
        report["eta_seconds"] = None
    elif report["parts"]["HM"]["complete_files"] < report["parts"]["HM"]["total_files"]:
        report["status"] = "HM 下载 / 准备"
    else:
        report["status"] = "原始视频、轨迹及同步文件下载 / 准备"
    return report


def human_bytes(amount):
    if amount is None:
        return "--"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024


def duration(seconds):
    if seconds is None:
        return "--"
    seconds = int(max(0, seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return (f"{days}天 " if days else "") + f"{hours:02}:{minutes:02}:{seconds:02}"


def render(report):
    percent = report["progress_percent"]
    filled = min(30, int(percent * 30 / 100))
    lines = [f"EgoExo4D-HM 下载监控  {report['timestamp']}", f"目录：{report['output_dir']}",
             f"状态：{report['status']}"]
    if report["excluded_buckets"]:
        lines.append(f"监控筛选范围：排除 {', '.join(report['excluded_buckets'])}，"
                     f"暂缓 {report['excluded_files']} 文件 / {human_bytes(report['excluded_total_bytes'])}；以下进度按筛选范围计算。")
    for process in report["processes"]:
        lines.append(f"进程：PID {process['pid']}  {process['state_text']} ({process['state']})  "
                     f"已运行 {duration(process['elapsed_seconds'])}  内存 {human_bytes(process['rss_bytes'])}")
    lines.extend(["", f"总进度：[{('#' * filled).ljust(30, '-')}] {percent:.3f}%  "
                  f"{human_bytes(report['present_bytes'])} / {human_bytes(report['total_bytes'])}",
                  f"完整文件：{report['complete_files']}/{report['total_files']}  临时/未完整文件：{report['partial_files']}",
                  f"当前速度：{human_bytes(report['speed_bytes_per_second'])}/s  "
                  f"监控期间均速：{human_bytes(report['average_bytes_per_second'])}/s  "
                  f"预计剩余：{duration(report['eta_seconds'])}", "",
                  f"{'类别':26} {'完整文件':>13} {'已落盘 GiB / 总 GiB':>25} {'进度':>9}"])
    for part, group in report["parts"].items():
        if not group["total_files"]:
            continue
        ratio = group["present_bytes"] / group["total_bytes"] if group["total_bytes"] else 0
        lines.append(f"{part:26} {group['complete_files']:6}/{group['total_files']:<6} "
                     f"{group['present_bytes'] / 1024**3:10.3f} / {group['total_bytes'] / 1024**3:<10.3f} {100 * ratio:8.3f}%")
    if report["unknown_files"]:
        lines.append(f"另有 {report['unknown_files']} 个大小未知的文件，未计入进度。")
    lines.extend(["", "速度按目录字节增量估算，计入临时文件；均速从启动监控时计算，ETA 使用最近约 60 秒均速。",
                  "首轮速度为 --；Ctrl+C 退出监控。"])
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    root = args.output_dir.resolve()
    try:
        plan = json.loads((root / "download_plan.json").read_text())
        scanner, rates = ProgressScanner(root, plan, args.exclude_bucket), RateTracker()
        report = snapshot(scanner, rates, args.pid)
        while True:
            if args.once:
                time.sleep(args.interval)
                report = snapshot(scanner, rates, args.pid)
            if args.json:
                print(json.dumps(report, ensure_ascii=False), flush=True)
            else:
                if sys.stdout.isatty() and not args.no_clear and not args.once:
                    print("\033[2J\033[H", end="")
                print(render(report), flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
            report = snapshot(scanner, rates, args.pid)
    except KeyboardInterrupt:
        print("\n监控已停止。", file=sys.stderr)
        return 0
    except FileNotFoundError:
        print(f"找不到 {root / 'download_plan.json'}；下载脚本生成清单后即可监控。", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"监控失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
