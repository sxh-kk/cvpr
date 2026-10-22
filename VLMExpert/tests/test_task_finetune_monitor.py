"""CPU checks for real task loss records, waiting state, and measured ETA."""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/analysis/monitor_task_finetune.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("task_finetune_monitor", SCRIPT)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
sys.path.pop(0)


def state(tmp_path, **overrides):
    return {"timestamp": "task-50", "run_directory": str(tmp_path / "run"),
            "log": str(tmp_path / "train.log"), "max_optimizer_steps": 100,
            "warmup_steps": 10, **overrides}


CONSOLE = {"completed": 0, "maximum": 0, "success": False, "failure": False}


def test_waiting_does_not_invent_loss_elapsed_or_eta(tmp_path):
    progress = monitor.make_progress(state(tmp_path), [], CONSOLE, now_s=1000)
    assert progress["state"] == "waiting"
    assert progress["estimated_remaining_seconds"] is None
    assert progress["latest_train_loss"] is None
    assert progress["elapsed_seconds"] is None
    assert progress["eta_wait_reason"] == "training_not_started"


def test_postwarmup_eta_includes_actual_checkpoint_validation_pause(tmp_path):
    # Nine seconds ordinary update time plus a 20-second checkpoint/val pause.
    rows = [{"completed_update": step, "train_loss": 1 / step,
             "timestamp_unix_s": 1000 + 2 * step + (20 if step >= 15 else 0)}
            for step in range(1, 21)]
    progress = monitor.make_progress(state(tmp_path), rows, CONSOLE, now_s=1061)
    assert progress["completed_optimizer_updates"] == 20
    assert progress["recent_seconds_per_optimizer_update"] == 4
    assert progress["estimated_remaining_seconds"] == 320
    assert progress["elapsed_seconds"] == 59
    assert progress["estimated_finish_utc"]
    assert progress["training_modified"] is False


def test_warmup_and_too_few_flushed_timestamps_delay_eta(tmp_path):
    for last in (5, 14):
        rows = [{"completed_update": step, "train_loss": 1., "timestamp_unix_s": 1000 + step}
                for step in range(1, last + 1)]
        progress = monitor.make_progress(state(tmp_path), rows, CONSOLE, now_s=1100)
        assert progress["state"] == "running"
        assert progress["estimated_remaining_seconds"] is None


def test_validation_only_rows_do_not_inflate_completed_updates(tmp_path):
    rows = [{"completed_update": 3, "train_loss": .5}, {"completed_update": 11, "val_loss": .7}]
    progress = monitor.make_progress(state(tmp_path), rows, CONSOLE, now_s=1000)
    assert progress["completed_optimizer_updates"] == 3
    assert progress["latest_validation_loss_update"] == 11
    assert progress["latest_train_loss"] == .5
    assert progress["estimated_remaining_seconds"] is None


def test_terminal_success_and_failure_never_offer_live_eta(tmp_path):
    rows = [{"completed_update": 100, "train_loss": .1, "timestamp_unix_s": 1100}]
    success = {**CONSOLE, "completed": 100, "maximum": 100, "success": True}
    done = monitor.make_progress(state(tmp_path), rows, success, now_s=1200)
    assert done["state"] == "completed" and done["estimated_remaining_seconds"] == 0
    failed = monitor.make_progress(state(tmp_path, state="failed", returncode=1), rows[:0], CONSOLE)
    assert failed["state"] == "failed" and failed["estimated_remaining_seconds"] is None


def test_finished_wall_clock_does_not_keep_growing_when_reopened(tmp_path):
    binding = state(tmp_path, state="complete", started_utc="2026-10-22T00:00:00+00:00",
                    finished_utc="2026-10-22T00:10:00+00:00")
    console = {**CONSOLE, "completed": 100, "maximum": 100, "success": True}
    assert monitor.make_progress(binding, [], console, now_s=2_000_000_000)["elapsed_seconds"] == 600


def test_full_update_count_waits_for_process_finalization(tmp_path):
    progress = monitor.make_progress(state(tmp_path), [], {**CONSOLE, "completed": 100, "maximum": 100})
    assert progress["state"] == "finalizing"
    assert progress["estimated_remaining_seconds"] is None
    assert progress["eta_wait_reason"] == "awaiting_training_finalization"


def test_budget_mismatch_and_invalid_time_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="budget"):
        monitor.make_progress(state(tmp_path), [], {**CONSOLE, "maximum": 200})
    with pytest.raises(ValueError, match="exceed"):
        monitor.make_progress(state(tmp_path), [], {**CONSOLE, "completed": 101})
    with pytest.raises(ValueError, match="timezone-aware"):
        monitor.make_progress(state(tmp_path, started_utc="2026-10-22T10:00:00"), [], CONSOLE)


@pytest.fixture
def wandb_source(tmp_path):
    datastore = pytest.importorskip("wandb.sdk.internal.datastore")
    proto = pytest.importorskip("wandb.proto.wandb_internal_pb2")

    def write(rows):
        path = tmp_path / "run.wandb"
        store = datastore.DataStore()
        store.open_for_write(str(path))
        try:
            for values in rows:
                record = proto.Record()
                for key, value in values.items():
                    item = record.history.item.add()
                    item.key, item.value_json = key, json.dumps(value)
                store.write(record)
        finally:
            store.close()
        return path
    return write


def test_real_wandb_train_eval_lr_merge_retains_train_timestamp(wandb_source):
    path = wandb_source([
        {"_step": 0, "_timestamp": 1001, "train/loss": 2., "train/lr_motion_expert": 1e-6},
        {"_step": 0, "_timestamp": 1020, "eval/loss": 3.},
        {"_step": 1, "_timestamp": 1022, "train/loss": 1.},
    ])
    before = path.read_bytes(), path.stat().st_mtime_ns
    rows, metadata = monitor.read_history(path)
    assert rows[0] == {"completed_update": 1, "timestamp_unix_s": 1001,
                       "train_loss": 2., "val_loss": 3., "lr_motion_expert": 1e-6}
    assert rows[1]["completed_update"] == 2
    assert metadata["tail_record_not_yet_flushed"] is False
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)


def test_live_partial_record_retains_only_complete_rows(wandb_source):
    path = wandb_source([{"_step": 0, "train/loss": 2.}, {"_step": 1, "train/loss": 1.}])
    path.write_bytes(path.read_bytes()[:-3])
    rows, metadata = monitor.read_history(path)
    assert [row["completed_update"] for row in rows] == [1]
    assert metadata["tail_record_not_yet_flushed"] is True


@pytest.mark.parametrize("invalid", [math.nan, True])
def test_nonfinite_or_boolean_loss_rejected(wandb_source, invalid):
    path = wandb_source([{"_step": 0, "train/loss": invalid}])
    with pytest.raises(ValueError, match="Nonfinite/invalid"):
        monitor.read_history(path)


def test_cli_waiting_outputs_csv_json_without_optional_dependencies(tmp_path):
    binding = tmp_path / "state.json"
    binding.write_text(json.dumps(state(tmp_path)))
    output = tmp_path / "monitor"
    process = subprocess.run([sys.executable, "-S", str(SCRIPT), "--state", str(binding),
                              "--output-dir", str(output), "--once", "--no-plot"],
                             capture_output=True, text=True, timeout=10)
    assert process.returncode == 0, process.stderr
    progress = json.loads((output / "progress.json").read_text())
    assert progress["state"] == "waiting" and progress["estimated_remaining_seconds"] is None
    assert (output / "losses.csv").read_text().splitlines() == [",".join(monitor.CSV_FIELDS)]
    assert json.loads((output / "monitor_health.json").read_text())["state"] == "ok"


def test_refresh_refuses_mixing_different_runs(tmp_path):
    binding = tmp_path / "state.json"
    binding.write_text(json.dumps(state(tmp_path)))
    output = tmp_path / "out"
    output.mkdir()
    args = SimpleNamespace(state=binding, output_dir=output, eta_window=50, eta_min_updates=5,
                           no_plot=True, plot_python=sys.executable)
    monitor.refresh(args)
    binding.write_text(json.dumps(state(tmp_path, timestamp="other")))
    with pytest.raises(ValueError, match="different training run"):
        monitor.refresh(args)


def test_failed_final_refresh_does_not_wait_forever_after_training_exit(tmp_path):
    binding = tmp_path / "state.json"
    binding.write_text(json.dumps(state(tmp_path, state="failed", returncode=1, max_optimizer_steps=0)))
    output = tmp_path / "monitor"
    process = subprocess.run([sys.executable, "-S", str(SCRIPT), "--state", str(binding),
                              "--output-dir", str(output), "--no-plot", "--poll-seconds", "1"],
                             capture_output=True, text=True, timeout=5)
    assert process.returncode == 1
    health = json.loads((output / "monitor_health.json").read_text())
    assert health["state"] == "error" and "max_optimizer_steps" in health["last_error"]
