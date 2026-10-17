"""Epoch loss plotting consumes completed TensorBoard epochs without Torch."""

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def plotter():
    path = Path(__file__).resolve().parents[1] / "watch_loss.py"
    spec = importlib.util.spec_from_file_location("motionexpert_loss_plot_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def event(step, value, wall_time=None):
    return SimpleNamespace(step=step, value=value,
                           wall_time=float(step if wall_time is None else wall_time))


def history(epochs):
    return [{"epoch": epoch, "step": epoch * 100 - 1,
             "train_loss": 1 / epoch, "val_loss": None}
            for epoch in range(1, epochs + 1)]


def test_completed_epoch_losses_pair_by_exact_step_and_use_one_based_epochs(plotter):
    result = plotter.build_history(
        [event(9, 0), event(19, 1), event(29, 2)],
        [event(29, .3), event(9, .9), event(19, .5)],
        [event(19, .6)],
    )
    assert result == [
        {"epoch": 1, "step": 9, "train_loss": .9, "val_loss": None},
        {"epoch": 2, "step": 19, "train_loss": .5, "val_loss": .6},
        {"epoch": 3, "step": 29, "train_loss": .3, "val_loss": None},
    ]


def test_partial_event_flush_waits_for_matching_epoch_without_guessing(plotter):
    epochs = [event(9, 0), event(14, 1), event(29, 2)]
    losses = [event(9, .9), event(19, .5), event(29, .3)]
    result = plotter.build_history(epochs, losses)
    assert [row["epoch"] for row in result] == [1, 3]
    assert [row["step"] for row in result] == [9, 29]

    result = plotter.build_history(epochs + [event(19, 1)], losses)
    assert [row["epoch"] for row in result] == [1, 2, 3]


def test_validation_never_pairs_with_a_nearby_step(plotter):
    result = plotter.build_history(
        [event(9, 0), event(19, 1)],
        [event(9, .9), event(19, .5)],
        [event(18, 42.), event(19, .6)],
    )
    assert result[0]["val_loss"] is None
    assert result[1]["val_loss"] == pytest.approx(.6)


def test_resume_uses_latest_wall_time_for_repeated_epoch_and_duplicate_step(plotter):
    # A resumed event file is not necessarily encountered last in file order.
    epochs = [event(29, 1, 30), event(19, 1, 20), event(9, 0, 10)]
    losses = [event(29, .7, 30), event(19, .8, 20),
              event(9, .6, 12), event(9, .9, 10)]
    result = plotter.build_history(epochs, losses,
                                   [event(29, .75, 31), event(29, 9., 29)])
    assert result == [
        {"epoch": 1, "step": 9, "train_loss": .6, "val_loss": None},
        {"epoch": 2, "step": 29, "train_loss": .7, "val_loss": .75},
    ]


def test_empty_or_unpaired_logs_produce_no_completed_epoch(plotter):
    assert plotter.build_history([], []) == []
    assert plotter.build_history([event(9, 0)], []) == []
    assert plotter.build_history([], [event(9, .9)]) == []
    assert plotter.build_history([event(9, 0)], [event(10, .9)]) == []


def test_default_plot_interval_includes_every_completed_epoch(plotter):
    rows = history(9)
    assert plotter.select_plot_history(rows) == rows
    assert plotter.select_plot_history(rows, last_plotted_epoch=8) == rows
    assert plotter.select_plot_history([]) == []


def test_optional_interval_waits_for_next_boundary_after_initial_backfill(plotter):
    assert plotter.select_plot_history(history(9), every_epochs=5) == history(9)
    assert plotter.select_plot_history(
        history(9), last_plotted_epoch=5, every_epochs=5) == history(5)
    assert plotter.select_plot_history(
        history(10), last_plotted_epoch=5, every_epochs=5) == history(10)


def test_read_run_resolves_explicit_directory_without_pointer(tmp_path, plotter):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = plotter.read_run(run_dir=str(run_dir))
    assert result["run_dir"] == str(run_dir.resolve())


def test_read_run_binds_pointer_once_instead_of_following_later_replacement(tmp_path, plotter):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    pointer = tmp_path / "current.json"
    pointer.write_text(json.dumps({"output_dir": str(first), "started_at": "first-start"}))
    bound = plotter.read_run(run_pointer=str(pointer))
    monitor = plotter.LossPlotter(bound, output_dir=tmp_path / "plots")
    pointer.write_text(json.dumps({"output_dir": str(second), "started_at": "second-start"}))
    assert bound["run_dir"] == str(first.resolve())
    assert monitor.run_dir == first.resolve()
    monitor.check_run()
    assert plotter.read_run(run_pointer=str(pointer))["run_dir"] == str(second.resolve())


def test_replacing_run_directory_stops_monitor_before_reading_other_runs(tmp_path, plotter):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monitor = plotter.LossPlotter({"run_dir": str(run_dir)},
                                   output_dir=tmp_path / "plots")
    run_dir.rename(tmp_path / "retired-run")
    run_dir.mkdir()
    with pytest.raises(RuntimeError, match="replaced"):
        monitor.update(None)


def test_output_cannot_be_reused_for_a_different_run(tmp_path, plotter):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    output = tmp_path / "plots"
    output.mkdir()
    (output / "status.json").write_text(json.dumps({
        "run_dir": str(first),
        "run_directory_identity": plotter.directory_identity(first),
    }))
    with pytest.raises(ValueError, match="another training run"):
        plotter.LossPlotter({"run_dir": str(second)}, output_dir=output)


def test_same_directory_new_start_time_stops_existing_monitor(tmp_path, plotter):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pointer = tmp_path / "current.json"
    pointer.write_text(json.dumps({"output_dir": str(run_dir), "started_at": "first-start"}))
    monitor = plotter.LossPlotter(plotter.read_run(run_pointer=pointer),
                                   output_dir=tmp_path / "plots")
    identity = plotter.directory_identity(run_dir)
    pointer.write_text(json.dumps({"output_dir": str(run_dir), "started_at": "second-start"}))
    assert plotter.directory_identity(run_dir) == identity
    with pytest.raises(RuntimeError, match="changed in the same directory"):
        monitor.update(None)


def test_same_directory_new_start_time_cannot_reuse_previous_plot_status(tmp_path, plotter):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output = tmp_path / "plots"
    output.mkdir()
    (output / "status.json").write_text(json.dumps({
        "run_dir": str(run_dir),
        "run_directory_identity": plotter.directory_identity(run_dir),
        "training_started_at": "first-start",
    }))
    with pytest.raises(ValueError, match="another training run"):
        plotter.LossPlotter({"run_dir": str(run_dir),
                              "training_started_at": "second-start"}, output_dir=output)


def test_real_tensorboard_png_csv_late_validation_and_idempotent_updates(tmp_path, plotter):
    pytest.importorskip("matplotlib")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    run_dir = tmp_path / "run"
    writer = EventFileWriter(str(run_dir))
    output = tmp_path / "plots"
    monitor = plotter.LossPlotter({"run_dir": str(run_dir)}, output_dir=output)
    reader = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})

    def add(step, tag, value, wall_time):
        writer.add_event(Event(step=step, wall_time=wall_time,
                               summary=Summary(value=[Summary.Value(
                                   tag=tag, simple_value=value)])))

    def csv_rows():
        with (output / "loss_history.csv").open(newline="") as stream:
            return list(csv.DictReader(stream))

    try:
        # A batch value must never become the epoch mean curve.
        add(8, "train/loss_step", 999., 100.)
        add(9, "train/loss_epoch", .75, 101.)
        writer.flush()
        assert monitor.update(reader) is False  # epoch tag has not flushed yet
        assert not (output / "loss_curve.png").exists()

        add(9, "epoch", 0., 102.)
        writer.flush()
        assert monitor.update(reader) is True
        png = output / "loss_curve.png"
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        first = csv_rows()
        assert len(first) == 1
        assert first[0]["epoch"] == "1"
        assert float(first[0]["train_loss"]) == pytest.approx(.75)
        assert first[0]["val_loss"] == ""
        mtime = png.stat().st_mtime_ns
        assert monitor.update(reader) is False
        assert png.stat().st_mtime_ns == mtime

        # Validation may flush after the train epoch; update the same endpoint.
        add(9, "val/loss", .875, 103.)
        writer.flush()
        assert monitor.update(reader) is True
        assert float(csv_rows()[0]["val_loss"]) == pytest.approx(.875)
        assert monitor.status["last_plotted_epoch"] == 1

        add(19, "epoch", 1., 104.)
        add(19, "train/loss_epoch", .5, 105.)
        writer.flush()
        assert monitor.update(reader) is True
        rows = csv_rows()
        assert [int(row["epoch"]) for row in rows] == [1, 2]
        assert [float(row["train_loss"]) for row in rows] == pytest.approx([.75, .5])
        status = json.loads((output / "status.json").read_text())
        assert status["last_plotted_epoch"] == 2
        assert status["plotted_epoch_count"] == 2
        assert status["missing_epochs"] == []
        assert monitor.update(reader) is False

        # Restarting the monitor does not rewrite identical completed history.
        status["pid"] = -1
        (output / "status.json").write_text(json.dumps(status))
        mtime = png.stat().st_mtime_ns
        csv_mtime = (output / "loss_history.csv").stat().st_mtime_ns
        resumed = plotter.LossPlotter({"run_dir": str(run_dir)}, output_dir=output)
        resumed_status = json.loads((output / "status.json").read_text())
        assert resumed_status["pid"] == os.getpid()
        assert resumed_status["updated_at"] == status["updated_at"]
        assert png.stat().st_mtime_ns == mtime
        assert (output / "loss_history.csv").stat().st_mtime_ns == csv_mtime
        assert resumed.update(reader) is False
        assert png.stat().st_mtime_ns == mtime
    finally:
        writer.close()


def test_real_tensorboard_reload_purges_restarted_tail_across_event_files(tmp_path, plotter):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.compat.proto.event_pb2 import Event, SessionLog
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    run_dir = tmp_path / "run"

    def add_epoch(writer, step, epoch, loss, wall_time):
        writer.add_event(Event(step=step, wall_time=wall_time, summary=Summary(value=[
            Summary.Value(tag="epoch", simple_value=epoch),
            Summary.Value(tag="train/loss_epoch", simple_value=loss),
        ])))

    writer = EventFileWriter(str(run_dir), filename_suffix=".first")
    try:
        add_epoch(writer, 9, 0, .75, 100.)
        add_epoch(writer, 19, 1, .5, 101.)
        add_epoch(writer, 29, 2, .4, 102.)
        writer.flush()
    finally:
        writer.close()
    reader = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    reader.Reload()
    initial = plotter.build_history(reader.Scalars("epoch"),
                                     reader.Scalars("train/loss_epoch"))
    assert [row["epoch"] for row in initial] == [1, 2, 3]

    # Resume replaces the old tail, rather than keeping epoch 3 from that tail.
    writer = EventFileWriter(str(run_dir), filename_suffix=".resumed")
    try:
        writer.add_event(Event(step=19, wall_time=200.,
                               session_log=SessionLog(status=SessionLog.START)))
        add_epoch(writer, 19, 1, .25, 201.)
        writer.flush()
    finally:
        writer.close()
    assert len(list(run_dir.glob("events.out.tfevents.*"))) == 2
    reader.Reload()
    resumed = plotter.build_history(reader.Scalars("epoch"),
                                     reader.Scalars("train/loss_epoch"))
    assert [row["epoch"] for row in resumed] == [1, 2]
    assert [row["step"] for row in resumed] == [9, 19]
    assert [row["train_loss"] for row in resumed] == pytest.approx([.75, .25])
