"""Plot completed-epoch TensorBoard losses without importing the training model.

Run this file directly to keep the monitor independent of MotionExpert's package
initialization. It reads events only; it never loads weights or signals training.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import threading


DEFAULT_POINTER = Path(
    "/gaozt-test1/datasets/MixEgoMotion/.pipeline/current_motionexpert_training.json")
FORMAT = "motionexpert-loss-plot-v1"


def read_run(run_pointer=None, run_dir=None):
    """Resolve the target once; a later pointer update cannot mix training runs."""
    if run_dir is not None:
        result = {"run_dir": str(Path(run_dir).resolve())}
    else:
        pointer = Path(run_pointer or DEFAULT_POINTER).resolve()
        source = json.loads(pointer.read_text(encoding="utf-8"))
        result = {"run_dir": str(Path(source["output_dir"]).resolve()),
                  "run_pointer": str(pointer), "training_pid": source.get("pid"),
                  "training_started_at": source.get("started_at")}
    return result


def build_history(epoch_events, train_events, val_events=()):
    """Use actual epoch means and exact-step epoch labels, never inferred steps."""
    epochs = {}
    for order, event in enumerate(epoch_events):
        value = float(event.value)
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            continue
        key = (event.wall_time, order)
        if event.step not in epochs or key > epochs[event.step][0]:
            epochs[event.step] = (key, int(value) + 1)

    def latest(events):
        result = {}
        for order, event in enumerate(events):
            if event.step not in epochs or not math.isfinite(float(event.value)):
                continue
            epoch = epochs[event.step][1]
            key = (event.wall_time, order)
            if epoch not in result or key > result[epoch][0]:
                result[epoch] = (key, event)
        return result

    train, validation = latest(train_events), latest(val_events)
    return [{"epoch": epoch, "step": int(train[epoch][1].step),
             "train_loss": float(train[epoch][1].value),
             "val_loss": float(validation[epoch][1].value) if epoch in validation else None}
            for epoch in sorted(train)]


def select_plot_history(history, last_plotted_epoch=None, every_epochs=1):
    if every_epochs < 1:
        raise ValueError("every_epochs must be positive")
    if not history:
        return []
    latest = history[-1]["epoch"]
    if last_plotted_epoch is None or latest // every_epochs > last_plotted_epoch // every_epochs:
        cutoff = latest
    else:
        # Include late validation points without advancing an unselected epoch.
        cutoff = min(latest, last_plotted_epoch)
    return [row for row in history if row["epoch"] <= cutoff]


def directory_identity(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino]


def atomic_write(path, payload):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def render(history, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    end = history[-1]["epoch"]
    train = {row["epoch"]: row["train_loss"] for row in history}
    x = list(range(1, end + 1))
    figure, axis = plt.subplots(figsize=(10, 5.5), layout="constrained")
    try:
        axis.plot(x, [train.get(epoch, math.nan) for epoch in x], color="#2563eb",
                  linewidth=2, label="Train (epoch mean)")
        validation = [row for row in history if row["val_loss"] is not None]
        if validation:
            axis.plot([row["epoch"] for row in validation],
                      [row["val_loss"] for row in validation], color="#ea580c",
                      linewidth=1.7, marker="o", markersize=4, label="Validation")
        axis.set(title=f"MotionExpert loss through epoch {end}", xlabel="Epoch", ylabel="Loss")
        axis.set_xlim(1, max(2, end))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
        axis.grid(alpha=.22)
        axis.legend(frameon=False)
        figure.text(.99, .01, f"Latest train loss: {history[-1]['train_loss']:.6f}",
                    ha="right", fontsize=9, color="#475569")
        image = io.BytesIO()
        figure.savefig(image, format="png", dpi=180)
        atomic_write(Path(output_dir) / "loss_curve.png", image.getvalue())
    finally:
        plt.close(figure)


class LossPlotter:
    def __init__(self, run, output_dir=None, every_epochs=1):
        if every_epochs < 1:
            raise ValueError("every_epochs must be positive")
        self.run = dict(run)
        self.run_dir = Path(run["run_dir"])
        self.identity = directory_identity(self.run_dir)
        self.output_dir = Path(output_dir) if output_dir else self.run_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.every_epochs = every_epochs
        self.status = {}
        status_path = self.output_dir / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if (status.get("run_dir") != str(self.run_dir)
                    or status.get("run_directory_identity") != self.identity
                    or status.get("training_started_at") != run.get("training_started_at")):
                raise ValueError("Plot output belongs to another training run; select a new output directory")
            self.status = status
            if status.get("pid") != os.getpid() or status.get("every_epochs") != every_epochs:
                self.status.update(pid=os.getpid(), every_epochs=every_epochs)
                atomic_write(status_path,
                             (json.dumps(self.status, ensure_ascii=False, indent=2) + "\n").encode())

    def check_run(self):
        if directory_identity(self.run_dir) != self.identity:
            raise RuntimeError("Training directory was replaced; stopping to avoid mixing runs")
        if self.run.get("run_pointer") and self.run.get("training_started_at"):
            current = json.loads(Path(self.run["run_pointer"]).read_text(encoding="utf-8"))
            if (Path(current["output_dir"]).resolve() == self.run_dir
                    and current.get("started_at") != self.run["training_started_at"]):
                raise RuntimeError("Training run changed in the same directory; select a new plot output")

    def update(self, accumulator):
        self.check_run()
        accumulator.Reload()
        tags = accumulator.Tags()["scalars"]
        if "epoch" not in tags or "train/loss_epoch" not in tags:
            return False
        history = build_history(accumulator.Scalars("epoch"),
                                accumulator.Scalars("train/loss_epoch"),
                                accumulator.Scalars("val/loss") if "val/loss" in tags else ())
        rows = select_plot_history(history, self.status.get("last_plotted_epoch"), self.every_epochs)
        if not rows:
            return False
        signature = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        if (signature == self.status.get("history_sha256")
                and (self.output_dir / "loss_curve.png").is_file()
                and (self.output_dir / "loss_history.csv").is_file()):
            return False
        self.check_run()
        render(rows, self.output_dir)
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=["epoch", "step", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(rows)
        atomic_write(self.output_dir / "loss_history.csv", text.getvalue().encode())
        self.status = {"format": FORMAT, **self.run, "run_directory_identity": self.identity,
                       "pid": os.getpid(), "every_epochs": self.every_epochs,
                       "last_plotted_epoch": rows[-1]["epoch"], "train_loss": rows[-1]["train_loss"],
                       "plotted_epoch_count": len(rows), "history_sha256": signature,
                       "missing_epochs": sorted(set(range(1, rows[-1]["epoch"] + 1))
                                                - {row["epoch"] for row in rows}),
                       "plot_path": str((self.output_dir / "loss_curve.png").resolve()),
                       "updated_at": datetime.now(timezone.utc).isoformat()}
        atomic_write(self.output_dir / "status.json",
                     (json.dumps(self.status, ensure_ascii=False, indent=2) + "\n").encode())
        print(json.dumps({"epoch": rows[-1]["epoch"], "train_loss": rows[-1]["train_loss"],
                          "plot": self.status["plot_path"]}), flush=True)
        return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--run-pointer", type=Path, help=f"Default: {DEFAULT_POINTER}")
    target.add_argument("--run-dir", type=Path, help="Explicit TensorBoard training output directory")
    parser.add_argument("--output-dir", type=Path, help="Default: RUN_DIR/plots")
    parser.add_argument("--every-epochs", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=30.)
    parser.add_argument("--once", action="store_true", help="Update once and exit")
    args = parser.parse_args(argv)
    if args.every_epochs < 1 or not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("every-epochs and poll-seconds must be positive")
    # This monitor is CPU-only, including optional TensorBoard dependencies.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    run = read_run(args.run_pointer, args.run_dir)
    output = args.output_dir or Path(run["run_dir"]) / "plots"
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".watch_loss.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another loss plot monitor already owns this output directory")
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        plotter = LossPlotter(run, output, args.every_epochs)
        reader = EventAccumulator(run["run_dir"], size_guidance={"scalars": 0})
        stop = threading.Event()
        for name in (signal.SIGINT, signal.SIGTERM):
            signal.signal(name, lambda *_: stop.set())
        while not stop.is_set():
            changed = plotter.update(reader)
            if args.once:
                if not changed and not plotter.status:
                    parser.error("No completed epoch loss is available yet")
                return 0
            stop.wait(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
