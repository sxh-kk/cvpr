#!/usr/bin/env python3
"""Read local W&B loss records or render their summary, without training imports.

The two modes deliberately have separate optional dependencies: extraction needs
wandb; plotting needs matplotlib. Both can inspect --help with only the stdlib.
All source files are read only. A run timestamp and directory bind the outputs to
one training run; this command never logs in, syncs W&B, or initializes a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any

WAIT_FOR_DATA = 75
SUMMARY_NAME = "loss_summary.json"
PLOT_METADATA_NAME = "loss_plot_metadata.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, content: str) -> None:
    """Publish one complete artifact; the summary/plot marker is published last."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def csv_text(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def load_run(state_path: Path, run_timestamp: str) -> dict[str, Any]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("timestamp") != run_timestamp:
        raise ValueError("Training state timestamp changed; refusing to mix training runs")
    run_directory = state.get("run_directory")
    if not isinstance(run_directory, str) or not Path(run_directory).is_absolute():
        raise ValueError("Training state must contain an absolute run_directory")
    return {"run_timestamp": run_timestamp, "run_directory": str(Path(run_directory).resolve())}


def check_binding(document: dict[str, Any], run: dict[str, Any]) -> None:
    if document.get("run_timestamp") != run["run_timestamp"]:
        raise ValueError("Output timestamp differs from the requested training run")
    recorded_directory = document.get("run_directory")
    # Older one-off summaries omitted run_directory; their exact W&B source still
    # identifies the run, allowing this CLI to refresh the existing user's plot.
    if recorded_directory is None:
        source = document.get("source")
        if not isinstance(source, str) or not Path(source).resolve().is_relative_to(
            Path(run["run_directory"]) / "wandb"
        ):
            raise ValueError("Existing output has no verifiable training run directory")
    elif str(Path(recorded_directory).resolve()) != run["run_directory"]:
        raise ValueError("Output directory is already bound to a different training run")


class _SnapshotReader(io.BytesIO):
    """Remember incomplete reads so a live final record is not called corruption."""

    short_read = False

    def read(self, size: int = -1) -> bytes:
        result = super().read(size)
        if size > 0 and len(result) < size:
            self.short_read = True
        return result


def read_wandb_losses(source: Path) -> tuple[dict[int, float], dict[str, Any]]:
    # Importing the parser does not create a run. Avoid open_for_scan(), which
    # opens r+b in W&B: parsing an in-memory rb snapshot guarantees read-only I/O.
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    with source.open("rb") as handle:
        snapshot = handle.read(os.fstat(handle.fileno()).st_size)
    stream = _SnapshotReader(snapshot)
    store = DataStore()
    store._fp = stream
    store._size_bytes = len(snapshot)
    store._opened_for_scan = True
    store._read_header()
    losses: dict[int, float] = {}
    duplicates = 0
    incomplete_tail = False
    while True:
        stream.short_read = False
        start_offset = stream.tell()
        try:
            payload = store.scan_data()
        except (AssertionError, EOFError) as exc:
            if not stream.short_read:
                raise ValueError(f"Corrupt W&B record at byte {start_offset}: {exc}") from exc
            incomplete_tail = True
            break
        if payload is None:
            # An unfinished fragmented record can return None after consuming its
            # first fragment. A clean EOF consumes no payload bytes.
            incomplete_tail = stream.tell() > start_offset
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(payload)
        if not record.HasField("history"):
            continue
        row = {}
        for item in record.history.item:
            key = "/".join(item.nested_key) if item.nested_key else item.key
            if key in ("_step", "train/loss"):
                row[key] = json.loads(item.value_json)
        if "train/loss" not in row:
            continue
        step = row.get("_step")
        if type(step) is not int or step < 0:
            raise ValueError("train/loss without a valid zero-based optimizer _step")
        update = step + 1
        duplicates += update in losses
        losses[update] = row["train/loss"]
    for update, value in losses.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Invalid final train/loss at update {update}: {value!r}")
    return losses, {
        "duplicates_replaced_by_last": duplicates,
        "tail_record_not_yet_flushed": incomplete_tail,
        "source_snapshot_bytes": len(snapshot),
        "source_snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
    }


def summarize_losses(losses: dict[int, float], window: int) -> dict[str, Any]:
    if window <= 0:
        raise ValueError("window must be positive")
    latest = max(losses, default=0)
    missing = [step for step in range(1, latest + 1) if step not in losses]
    bins = []
    for start in range(1, latest + 1, window):
        end = min(start + window - 1, latest)
        updates = [step for step in range(start, end + 1) if step in losses]
        if not updates:
            continue
        values = [losses[step] for step in updates]
        bins.append({
            "start_update": start,
            "end_update": end,
            "count": len(values),
            "complete": len(values) == window,
            "continuous": len(values) == end - start + 1,
            "mean_loss": statistics.fmean(values),
            "min_loss": min(values),
            "max_loss": max(values),
            "std_loss": statistics.pstdev(values),
        })
    complete = [row for row in bins if row["complete"]]
    return {
        "latest_flushed_update": latest,
        "latest_complete_update": max((row["end_update"] for row in complete), default=0),
        "distinct_loss_records": len(losses),
        "missing_updates": missing,
        "window_size": window,
        "complete_windows": len(complete),
        "bins": bins,
    }


def extract(args: argparse.Namespace, run: dict[str, Any]) -> int:
    output = args.output_dir
    for name in (SUMMARY_NAME, PLOT_METADATA_NAME):
        path = output / name
        if path.exists():
            check_binding(json.loads(path.read_text(encoding="utf-8")), run)
    files = sorted((Path(run["run_directory"]) / "wandb").glob("offline-run-*/*.wandb"))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one local W&B file for this run, got {len(files)}")
    losses, scan_metadata = read_wandb_losses(files[0])
    report = {
        "schema_version": 1,
        **run,
        "generated_utc": utc_now(),
        "source": str(files[0]),
        "loss_semantics": (
            "Mean of logged train/loss for each optimizer-update window; each train/loss "
            "is the final accumulation microbatch mean over all ranks, not all accumulated samples."
        ),
        "step_mapping": "completed_update = W&B zero-based _step + 1",
        **scan_metadata,
        **summarize_losses(losses, args.window),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(output / "loss_per_update.csv", csv_text(
        ["completed_update", "logged_train_loss"],
        [{"completed_update": step, "logged_train_loss": value} for step, value in sorted(losses.items())],
    ))
    fields = ["start_update", "end_update", "count", "complete", "continuous", "mean_loss", "min_loss", "max_loss", "std_loss"]
    atomic_text(output / "loss_100step.csv", csv_text(fields, report["bins"]))
    atomic_json(output / SUMMARY_NAME, report)
    print(json.dumps({key: report[key] for key in (
        "run_timestamp", "latest_flushed_update", "latest_complete_update", "complete_windows",
        "duplicates_replaced_by_last", "tail_record_not_yet_flushed",
    )}, ensure_ascii=False))
    return 0


def plot(args: argparse.Namespace, run: dict[str, Any]) -> int:
    report = json.loads((args.output_dir / SUMMARY_NAME).read_text(encoding="utf-8"))
    check_binding(report, run)
    marker_path = args.output_dir / PLOT_METADATA_NAME
    if marker_path.exists():
        check_binding(json.loads(marker_path.read_text(encoding="utf-8")), run)
    if report.get("window_size") != args.window:
        raise ValueError("Summary window_size differs from --window; extract it again")
    complete = [row for row in report["bins"] if row["complete"]]
    last_complete = max((row["end_update"] for row in complete), default=0)
    if not complete or last_complete < (args.minimum_complete_update or 0):
        print(json.dumps({"status": "waiting", "latest_complete_update": last_complete,
                          "minimum_complete_update": args.minimum_complete_update}, ensure_ascii=False))
        return WAIT_FOR_DATA
    # Do not even import matplotlib until the requested full window is ready.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt
    from matplotlib.ticker import MaxNLocator

    font = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = ["DejaVu Sans", font_manager.FontProperties(fname=str(font)).get_name()]
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#334155", "text.color": "#0f172a",
                         "xtick.color": "#475569", "ytick.color": "#475569", "svg.fonttype": "none"})
    window = args.window
    recent_start = 6 * window + 1
    partial = [row for row in report["bins"]
               if not row["complete"] and row.get("continuous", False)
               and row["end_update"] == report["latest_flushed_update"]
               and row["start_update"] > last_complete]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), gridspec_kw={"width_ratios": [1.08, 1]})
    fig.patch.set_facecolor("white")
    fig.suptitle(f"UnifoLM 后训练：每{window}步记录的平均 Loss", x=0.075, ha="left", y=0.975,
                 fontsize=19, fontweight="bold")
    fig.text(0.075, 0.895,
             f"连续、不重叠分组：1–{window}、{window+1}–{2*window}…  |  已记录 {report['latest_flushed_update']:,} 步  |  {len(complete)} 个完整分组",
             fontsize=10.5, color="#64748b")
    color = "#0f766e"
    for index, ax in enumerate(axes):
        rows = complete if index == 0 else [row for row in complete if row["start_update"] >= recent_start]
        recent_available = bool(rows)
        if not rows:
            rows = complete
        # Retain NaN gaps for missing/incomplete interior windows, rather than
        # visually connecting nonadjacent completed groups as if data existed.
        x, y = [], []
        previous_end = None
        for row in rows:
            if previous_end is not None and row["start_update"] != previous_end + 1:
                x.append(row["start_update"] - 1)
                y.append(math.nan)
            x.append(row["end_update"])
            y.append(row["mean_loss"])
            previous_end = row["end_update"]
        ax.plot(x, y, color=color, linewidth=2.6, marker="o", markersize=5.3,
                markeredgecolor="white", markeredgewidth=0.8, label=f"完整{window}步均值")
        if partial:
            tail = partial[-1]
            adjacent = rows[-1]["end_update"] + 1 == tail["start_update"]
            tail_x = [rows[-1]["end_update"], tail["end_update"]] if adjacent else [tail["end_update"]]
            tail_y = [rows[-1]["mean_loss"], tail["mean_loss"]] if adjacent else [tail["mean_loss"]]
            ax.plot(tail_x, tail_y, linestyle="--", color="#d97706", linewidth=1.6,
                    marker="o", markersize=4.7, label=f"尾段均值（{tail['count']}/{window}步）")
        title = "完整趋势" if index == 0 else (
            f"近期趋势放大（第{recent_start}步起）" if recent_available else "近期趋势放大"
        )
        ax.set_title(title, loc="left", fontsize=12.5, pad=13, fontweight="bold")
        ax.set_xlabel("已完成的优化器更新步数", labelpad=9)
        ax.set_ylabel("平均训练 Loss", labelpad=8)
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(axis="both", length=0, pad=7)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        ax.margins(x=0.09, y=0.16)
        if index == 0:
            ax.set_ylim(bottom=0)
            ax.annotate(f"{y[0]:.2f}", (x[0], y[0]), xytext=(7, 7), textcoords="offset points",
                        color=color, fontsize=10)
        else:
            for row in rows[-5:]:
                ax.annotate(f"{row['mean_loss']:.3f}", (row["end_update"], row["mean_loss"]),
                            xytext=(0, 10), textcoords="offset points", color=color, fontsize=9, ha="center")
            ax.legend(loc="upper right", frameon=False, fontsize=9)
    last = complete[-1]
    fig.text(0.075, 0.045,
             f"最近完整分组：第{last['start_update']}–{last['end_update']}步，平均 Loss = {last['mean_loss']:.4f}",
             fontsize=11, color=color, fontweight="bold")
    fig.text(0.075, 0.007,
             "来源：本地 W&B 完整精度记录；每步 Loss 沿用当前训练记录口径（最后一个累积 microbatch 的全卡均值）。",
             fontsize=8.8, color="#64748b")
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.22, top=0.78, wspace=0.25)
    temporary_paths = []
    try:
        for extension in ("png", "svg"):
            descriptor, temporary = tempfile.mkstemp(prefix=".loss_100step.", suffix=f".{extension}", dir=args.output_dir)
            os.close(descriptor)
            temporary_paths.append((Path(temporary), args.output_dir / f"loss_100step.{extension}"))
            fig.savefig(temporary, dpi=170, facecolor="white", bbox_inches="tight")
        for temporary, destination in temporary_paths:
            os.replace(temporary, destination)
        atomic_json(args.output_dir / PLOT_METADATA_NAME, {
            "schema_version": 1,
            **run,
            "window_size": window,
            "latest_complete_update": last_complete,
            "latest_flushed_update": report["latest_flushed_update"],
            "data_generated_utc": report["generated_utc"],
            "rendered_utc": utc_now(),
        })
    finally:
        plt.close(fig)
        for temporary, _ in temporary_paths:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"status": "rendered", "latest_complete_update": last_complete,
                      "png": str(args.output_dir / "loss_100step.png"),
                      "svg": str(args.output_dir / "loss_100step.svg")}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("extract", "plot"))
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-timestamp", required=True)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--minimum-complete-update", type=int, default=None,
                        help="Plot only: return 75 without changing plots until this full-window endpoint is reached")
    args = parser.parse_args(argv)
    if args.window < 1:
        parser.error("--window must be positive")
    if args.minimum_complete_update is not None:
        if args.mode != "plot" or args.minimum_complete_update < 0:
            parser.error("--minimum-complete-update must be nonnegative and is only valid in plot mode")
    try:
        run = load_run(args.state, args.run_timestamp)
        return extract(args, run) if args.mode == "extract" else plot(args, run)
    except (OSError, ValueError, ImportError) as exc:
        print(f"training_loss_curve: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
