#!/usr/bin/env python3
"""Read-only task-finetuning losses and wall-clock ETA from local offline W&B.

The observer never imports a trainer, initializes W&B, or signals a process.
Update timestamps include observed validation/checkpoint pauses between updates;
the ETA is an estimate, not a throughput or convergence acceptance result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import training_loss_curve as curve

LR_KEYS = ("action_header", "lang_backbone", "vision_tower", "mm_projector", "motion_expert")
USER_STOP_STATES = ("switched", "stopped_by_user")
TERMINAL_STATES = ("completed", "failed", *USER_STOP_STATES)
CSV_FIELDS = ["completed_update", "timestamp_unix_s", "train_loss", "val_loss",
              *[f"lr_{name}" for name in LR_KEYS]]


def read_history(source):
    """Use the existing snapshot reader/CRC scanner without opening W&B writable."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    with Path(source).open("rb") as handle:
        snapshot = handle.read(os.fstat(handle.fileno()).st_size)
    stream = curve._SnapshotReader(snapshot)
    store = DataStore()
    store._fp, store._size_bytes, store._opened_for_scan = stream, len(snapshot), True
    store._read_header()
    rows, incomplete = {}, False
    accepted = {"train/loss": "train_loss", "eval/loss": "val_loss", "_timestamp": "timestamp_unix_s",
                **{f"train/lr_{name}": f"lr_{name}" for name in LR_KEYS}}
    while True:
        stream.short_read = False
        start = stream.tell()
        try:
            payload = store.scan_data()
        except (AssertionError, EOFError) as error:
            if not stream.short_read:
                raise ValueError(f"Corrupt W&B record at byte {start}") from error
            incomplete = True
            break
        if payload is None:
            incomplete = stream.tell() > start
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(payload)
        if not record.HasField("history"):
            continue
        values = {}
        for item in record.history.item:
            key = "/".join(item.nested_key) if item.nested_key else item.key
            if key in accepted or key == "_step":
                values[key] = json.loads(item.value_json)
        if not any(key in values for key in ("train/loss", "eval/loss")):
            continue
        step = values.get("_step")
        if type(step) is not int or step < 0:
            raise ValueError("Loss record needs a zero-based integer optimizer _step")
        row = rows.setdefault(step + 1, {"completed_update": step + 1})
        for key, field in accepted.items():
            if key not in values:
                continue
            value = values[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Nonfinite/invalid {key} at update {step + 1}")
            # Evaluation may be a separate record at the same optimizer step.
            # ETA uses the TRAIN log timestamp, not a later evaluation timestamp.
            if field == "timestamp_unix_s" and "train/loss" not in values:
                continue
            row[field] = value
    return [rows[k] for k in sorted(rows)], {
        "source": str(Path(source).resolve()), "source_snapshot_bytes": len(snapshot),
        "source_snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
        "tail_record_not_yet_flushed": incomplete,
    }


def console_snapshot(path):
    if not Path(path).is_file():
        return {"completed": 0, "maximum": 0, "success": False, "failure": False}
    with Path(path).open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - 131072))
        text = handle.read().decode(errors="replace")
    matches = re.findall(r"Training steps:[^\r\n]*?\|\s*(\d+)/(\d+)", text)
    completed, maximum = map(int, matches[-1]) if matches else (0, 0)
    return {"completed": completed, "maximum": maximum, "success": "Happy Ending!" in text,
            "failure": bool(re.search(r"Traceback \(most recent call last\)|ChildFailedError|CUDA out of memory", text))}


def unix_time(value):
    if not value:
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError("time must include timezone")
        return result.timestamp()
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError(f"Invalid timezone-aware timestamp {value!r}") from error


def make_progress(state, rows, console, *, now_s=None, eta_window=50, eta_min_updates=5):
    now_s = time.time() if now_s is None else now_s
    maximum = state.get("max_optimizer_steps", state.get("max_training_steps"))
    warmup = state.get("warmup_steps", 0)
    if type(maximum) is not int or maximum <= 0 or type(warmup) is not int or not 0 <= warmup < maximum:
        raise ValueError("State needs positive max_optimizer_steps and valid warmup_steps")
    if console["maximum"] not in (0, maximum):
        raise ValueError("Console budget differs from bound task run")
    train = [row for row in rows if "train_loss" in row]
    completed = max(console["completed"], max((row["completed_update"] for row in train), default=0))
    if completed > maximum:
        raise ValueError("Observed updates exceed the bound task run budget")
    declared = state.get("state", "prepared")
    failed = declared in ("failed", "error") or state.get("returncode") not in (None, 0) or console["failure"]
    done = completed == maximum and (console["success"] or declared in ("complete", "completed"))
    status = (declared if declared in USER_STOP_STATES else
              "failed" if failed and not done else "completed" if done else
              "finalizing" if completed == maximum else "running" if completed else "waiting")
    stamped = [row for row in train if "timestamp_unix_s" in row]
    started = unix_time(state.get("started_utc"))
    elapsed_origin = "launcher_started_utc"
    if started is None and stamped:
        started = stamped[0]["timestamp_unix_s"]
        elapsed_origin = "first_flushed_train_update_excludes_startup"
    finished = unix_time(state.get("finished_utc")) if status in TERMINAL_STATES else None
    elapsed = max(0.0, (finished if finished is not None else now_s) - started) if started is not None else None
    after_warmup = [row for row in stamped if row["completed_update"] >= max(1, warmup)]
    recent = after_warmup[-(eta_window + 1):]
    intervals = [(b["timestamp_unix_s"] - a["timestamp_unix_s"],
                  b["completed_update"] - a["completed_update"])
                 for a, b in zip(recent, recent[1:])
                 if b["timestamp_unix_s"] > a["timestamp_unix_s"]]
    covered = recent[-1]["completed_update"] - recent[0]["completed_update"] if len(recent) >= 2 else 0
    estimate_ready = covered >= eta_min_updates and len(intervals) >= eta_min_updates
    rate = sum(pair[0] for pair in intervals) / sum(pair[1] for pair in intervals) if estimate_ready else None
    remaining = (maximum - completed) * rate if status == "running" and rate is not None else None
    if status == "completed":
        remaining = 0.0
    validation = [row for row in rows if "val_loss" in row]
    return {"schema_version": 1, "state": status,
            "generated_utc": datetime.fromtimestamp(now_s, timezone.utc).isoformat(),
            "run_timestamp": state["timestamp"], "run_directory": str(Path(state["run_directory"]).resolve()),
            "completed_optimizer_updates": completed, "max_optimizer_updates": maximum,
            "progress_fraction": completed / maximum, "elapsed_seconds": elapsed,
            "elapsed_origin": elapsed_origin if started is not None else None,
            "latest_train_loss": train[-1]["train_loss"] if train else None,
            "latest_validation_loss": validation[-1]["val_loss"] if validation else None,
            "latest_train_loss_update": train[-1]["completed_update"] if train else None,
            "latest_validation_loss_update": validation[-1]["completed_update"] if validation else None,
            "recent_seconds_per_optimizer_update": rate, "eta_sample_intervals": len(intervals),
            "estimated_remaining_seconds": remaining,
            "estimated_finish_utc": datetime.fromtimestamp(now_s + remaining, timezone.utc).isoformat()
                if remaining is not None and status == "running" else None,
            "eta_scope": "Recent post-warmup TRAIN metric-row wall time (W&B may merge same-step evaluation into the row), including observed validation/checkpoint pauses; future pauses and final save may differ.",
            "eta_wait_reason": None if remaining is not None else
                "training_switched" if status == "switched" else
                "training_stopped_by_user" if status == "stopped_by_user" else
                "training_failed" if status == "failed" else
                "awaiting_training_finalization" if status == "finalizing" else
                "training_not_started" if completed == 0 else "waiting_for_post_warmup_timestamps",
            "loss_semantics": "Production logged rank-mean loss at optimizer update; with accumulation it is the last microbatch, not an average over every accumulated sample.",
            "step_mapping": "completed_update = W&B zero-based _step + 1; console shows completed updates",
            "validation_step_semantics": "Production validation occurs after update 1, 1+validation_steps, ... and the final update.",
            "stop_reason": state.get("stop_reason"),
            "superseded_by_run_directory": state.get("superseded_by_run_directory"),
            "latest_group_lrs": {name: train[-1].get(f"lr_{name}") for name in LR_KEYS} if train else {},
            "cuda_visible_devices": "", "training_modified": False}


def render_plot(csv_path, target):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    with Path(csv_path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for field, label, style in (("train_loss", "Train loss", "-"), ("val_loss", "Validation loss", "o-")):
        values = [(int(row["completed_update"]), float(row[field])) for row in rows if row[field]]
        if values:
            axis.plot(*zip(*values), style, label=label, linewidth=1, markersize=3)
    if not rows:
        axis.text(.5, .5, "Waiting for task training updates", transform=axis.transAxes, ha="center")
    if axis.lines:
        axis.legend()
    axis.set(xlabel="Completed optimizer updates", ylabel="Logged loss", title="Task joint finetuning")
    axis.grid(alpha=.2)
    figure.tight_layout()
    target = Path(target)
    temporary = target.with_name(f".{target.stem}.{os.getpid()}.tmp.png")
    try:
        figure.savefig(temporary, dpi=150)
        os.replace(temporary, target)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


def refresh(args):
    state = json.loads(args.state.read_text())
    run = {"run_timestamp": state["timestamp"], "run_directory": str(Path(state["run_directory"]).resolve())}
    if not Path(state["run_directory"]).is_absolute():
        raise ValueError("run_directory must be absolute")
    marker = args.output_dir / "progress.json"
    if marker.is_file():
        previous = json.loads(marker.read_text())
        if any(previous.get(key) != value for key, value in run.items()):
            raise ValueError("Output already belongs to a different training run")
    sources = sorted((Path(run["run_directory"]) / "wandb").glob("offline-run-*/*.wandb"))
    if len(sources) > 1:
        raise ValueError("Expected at most one local W&B file for a fresh task run")
    rows, metadata = read_history(sources[0]) if sources else ([], {"source": None})
    progress = make_progress(state, rows, console_snapshot(state["log"]), eta_window=args.eta_window,
                             eta_min_updates=args.eta_min_updates)
    progress.update(metadata)
    curve.atomic_text(args.output_dir / "losses.csv", curve.csv_text(CSV_FIELDS, rows))
    if not args.no_plot:
        child_env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                         MPLCONFIGDIR=str(args.output_dir / ".matplotlib"))
        child_env.pop("PYTHONPATH", None)
        subprocess.run([args.plot_python, str(Path(__file__).resolve()), "--plot-only",
                        str(args.output_dir / "losses.csv"), str(args.output_dir / "loss.png")],
                       check=True, env=child_env, timeout=60, capture_output=True)
    curve.atomic_json(marker, progress)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--eta-window", type=int, default=50)
    parser.add_argument("--eta-min-updates", type=int, default=5)
    parser.add_argument("--plot-python", default=sys.executable)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-only", nargs=2, metavar=("CSV", "PNG"))
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if args.plot_only:
        render_plot(*args.plot_only)
        return 0
    if not args.state or not args.output_dir:
        parser.error("--state and --output-dir are required")
    if args.poll_seconds < 1 or args.eta_window < args.eta_min_updates or args.eta_min_updates < 2:
        parser.error("poll-seconds >=1 and eta-window >= eta-min-updates >=2 required")
    args.state, args.output_dir = args.state.resolve(), args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "monitor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A task monitor already owns this output directory")
        while True:
            try:
                progress = refresh(args)
                print(json.dumps(progress, ensure_ascii=False), flush=True)
                curve.atomic_json(args.output_dir / "monitor_health.json", {
                    "state": "ok", "generated_utc": curve.utc_now(), "last_error": None})
            except Exception as error:
                curve.atomic_json(args.output_dir / "monitor_health.json", {
                    "state": "error", "generated_utc": curve.utc_now(),
                    "last_error": f"{type(error).__name__}: {error}"})
                if args.once:
                    raise
                print(f"Task monitor retry: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                try:
                    terminal = json.loads(args.state.read_text()).get("state") in (
                        "complete", "error", *TERMINAL_STATES)
                except (OSError, ValueError):
                    terminal = False
                if terminal:
                    return 1  # A failed final refresh must not keep a dead run under observation.
                time.sleep(args.poll_seconds)
                continue
            if args.once or progress["state"] in TERMINAL_STATES:
                return 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
