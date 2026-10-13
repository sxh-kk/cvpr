#!/usr/bin/env python3
"""Monitor EgoSuite selection/download progress using only the standard library.

    python /cvpr/datasets/monitor_egosuite.py
    python /cvpr/datasets/monitor_egosuite.py --once --interval 2
    python /cvpr/datasets/monitor_egosuite.py --json --no-clear

Network speed is measured from the downloader's Xet transfer counters, including
retries. File progress uses reconstructed bytes, including active files, plus
verified completed files. Compressed/cached transfer bytes and reconstructed
bytes differ; ETA uses the latter. Average speed starts when monitoring begins.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

DEFAULT_OUTPUT_DIR = Path("/gaozt-test1/datasets/EgoSuite")
PHASES = {
    "starting": "启动", "listing": "读取远程文件目录", "planning": "查询时长并生成清单",
    "sampling": "抽样估算时长并生成清单",
    "planned": "清单已生成", "checking": "检查远端版本、已有文件和容量",
    "downloading": "下载中", "complete": "下载完成", "error": "下载失败", "interrupted": "下载已中断",
    "waiting": "等待下载脚本生成状态文件",
}


def process_alive(pid, expected_start=None):
    try:
        fields = (Path("/proc") / str(int(pid)) / "stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] != "Z" and (expected_start is None or int(fields[19]) == expected_start)
    except (OSError, ValueError, TypeError, IndexError):
        return False


class RateTracker:
    def __init__(self, window=60):
        self.window = window
        self.samples = deque()
        self.run_id = None
        self.started = None
        self.initial = 0

    def update(self, now, amount, run_id):
        if self.run_id != run_id or self.started is None:
            self.samples.clear()
            self.run_id, self.started, self.initial = run_id, now, amount
        previous = self.samples[-1] if self.samples else None
        current = None if previous is None or now <= previous[0] else max(0, amount - previous[1]) / (now - previous[0])
        if previous is not None and amount < previous[1]:
            self.samples.clear()
        self.samples.append((now, amount))
        while len(self.samples) > 2 and self.samples[1][0] <= now - self.window:
            self.samples.popleft()
        first = self.samples[0]
        smooth = max(0, amount - first[1]) / (now - first[0]) if now > first[0] else 0
        average = max(0, amount - self.initial) / (now - self.started) if now > self.started else 0
        return {"current": current, "smooth": smooth, "average": average}


def read_state(output):
    path = output / ".egosuite-download/progress.json"
    try:
        state = json.loads(path.read_text())
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported progress schema")
        return state
    except FileNotFoundError:
        return {"phase": "waiting", "run_id": None, "network_bytes": 0, "reconstructed_bytes": 0,
                "completed_files": 0, "total_files": 0, "total_bytes": 0, "active_files": []}


def snapshot(output, network_rates, reconstruction_rates, *, now=None, wall_now=None):
    state = read_state(output)
    now = time.monotonic() if now is None else now
    wall_now = time.time() if wall_now is None else wall_now
    network = network_rates.update(now, state["network_bytes"], state.get("run_id"))
    reconstruction = reconstruction_rates.update(now, state["reconstructed_bytes"], state.get("run_id"))
    total = state["total_bytes"]
    present = min(total, max(0, state["reconstructed_bytes"])) if total else 0
    alive = process_alive(state.get("pid"), state.get("process_start_ticks"))
    status = PHASES.get(state["phase"], state["phase"])
    terminal = state["phase"] in ("complete", "error", "interrupted", "planned", "waiting")
    if not alive and not terminal:
        status = f"下载进程已退出；上次阶段：{status}"
    remaining = max(0, total - present)
    eta = remaining / reconstruction["smooth"] if alive and reconstruction["smooth"] > 0 else None
    if state["phase"] == "complete":
        eta = 0
    stale = max(0, wall_now - state.get("updated_at", wall_now))
    return {
        **state, "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_dir": str(output), "process_alive": alive, "status": status,
        "progress_percent": 100 * present / total if total else 0,
        "file_progress_percent": 100 * state["completed_files"] / state["total_files"] if state["total_files"] else 0,
        "network_speed_bytes_per_second": network["current"],
        "average_network_bytes_per_second": network["average"],
        "reconstruction_speed_bytes_per_second": reconstruction["current"],
        "eta_seconds": eta, "state_age_seconds": stale,
    }


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
    filled = min(30, max(0, int(percent * 30 / 100)))
    lines = [f"EgoSuite 正式 body MCAP 下载监控  {report['timestamp']}",
             f"目录：{report['output_dir']}", f"状态：{report['status']}"]
    if report.get("bucket") and report.get("prefix"):
        lines.append(f"数据源：hf://buckets/{report['bucket']}/{report['prefix']}/")
    if report.get("pid"):
        lines.append(f"PID：{report['pid']}  {'运行中' if report['process_alive'] else '已退出'}  "
                     f"状态更新距今：{report['state_age_seconds']:.1f}s")
    estimate_label = "估算" if report.get("hours_are_estimated") else "实测"
    if report["phase"] in ("listing", "planning", "sampling", "planned"):
        lines.extend([
            f"时长查询：{report.get('candidates_examined', 0)}/{report.get('candidates_total', 0)} 个文件  "
            f"跳过：{report.get('skipped_files', 0)}",
            f"已选取：{report.get('selected_files', report['total_files'])} 段，{estimate_label} "
            f"{report.get('selected_hours', 0):.3f}/{report.get('target_hours', 250):.3f} h",
        ])
    if report.get("hours_are_estimated"):
        lines.append(f"清单时长为抽样估算（样本 {report.get('duration_sample_count', 0)} 段）；完成时长从本地 MCAP 实测。")
    lines.extend([
        "", f"容量进度：[{('#' * filled).ljust(30, '-')}] {percent:.3f}%  "
        f"{human_bytes(report['reconstructed_bytes'])} / {human_bytes(report['total_bytes'])}",
        f"完整文件：{report['completed_files']}/{report['total_files']}  "
        f"文件进度：{report['file_progress_percent']:.3f}%  同时下载：{len(report['active_files'])}",
        f"已完成实际录制时长：{report.get('completed_hours', 0):.3f} h  "
        f"清单{estimate_label}时长：{report.get('selected_hours', 0):.3f} h",
        f"当前下载速度：{human_bytes(report['network_speed_bytes_per_second'])}/s  "
        f"监控期间均速：{human_bytes(report['average_network_bytes_per_second'])}/s",
        f"文件重建速度：{human_bytes(report['reconstruction_speed_bytes_per_second'])}/s  "
        f"预计剩余：{duration(report['eta_seconds'])}",
        f"本次进程累计网络传输：{human_bytes(report['network_bytes'])}",
    ])
    for item in report["active_files"]:
        ratio = item["reconstructed_bytes"] / item["size"] if item["size"] else 0
        lines.append(f"  {100 * ratio:6.2f}%  尝试 {item['attempt']}  {item['path']}")
        if item.get("error"):
            lines.append(f"    最近错误：{item['error']}")
    if report.get("last_error"):
        lines.append(f"最近错误：{report['last_error']}")
    lines.extend(["", "下载速度来自 HF/Xet 网络传输计数；容量进度含正在重建的文件，重试流量也计入网络速度。",
                  "首轮速度为 --；Ctrl+C 只停止监控。"])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--interval", type=float, default=5, help="刷新间隔（秒）")
    parser.add_argument("--once", action="store_true", help="采样两次，输出一次（包含速度）")
    parser.add_argument("--json", action="store_true", help="输出 JSONL，便于保存或对接程序")
    parser.add_argument("--no-clear", action="store_true", help="追加输出，适合日志")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or not 0.2 <= args.interval <= 60:
        parser.error("interval 必须在 0.2 到 60 秒之间")
    output = args.output_dir.resolve()
    network, reconstruction = RateTracker(), RateTracker()
    try:
        report = snapshot(output, network, reconstruction)
        while True:
            if args.once:
                time.sleep(args.interval)
                report = snapshot(output, network, reconstruction)
            if args.json:
                print(json.dumps(report, ensure_ascii=False), flush=True)
            else:
                if sys.stdout.isatty() and not args.no_clear and not args.once:
                    print("\033[2J\033[H", end="")
                print(render(report), flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
            report = snapshot(output, network, reconstruction)
    except KeyboardInterrupt:
        print("\n监控已停止。", file=sys.stderr)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"监控失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
