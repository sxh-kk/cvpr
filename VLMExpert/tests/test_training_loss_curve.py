"""CPU regressions for the independent local W&B loss reader/plot boundary.

Real W&B protobuf records and small temporary datastore files exercise the parser;
no training imports, existing run artifacts, models, GPUs, or renderer are used.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analysis/training_loss_curve.py"


@pytest.fixture
def curve():
    spec = importlib.util.spec_from_file_location("training_loss_curve", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def make_wandb_source(tmp_path):
    # The production helper keeps this dependency lazy; pure summary/binding
    # tests still run in environments that have no W&B package installed.
    datastore = pytest.importorskip("wandb.sdk.internal.datastore")
    protobuf = pytest.importorskip("wandb.proto.wandb_internal_pb2")
    counter = 0

    def write(rows):
        nonlocal counter
        counter += 1
        path = tmp_path / f"sample-{counter}.wandb"
        store = datastore.DataStore()
        store.open_for_write(str(path))
        try:
            for values in rows:
                record = protobuf.Record()
                for key, value in values.items():
                    item = record.history.item.add()
                    item.key = key
                    item.value_json = json.dumps(value)
                store.write(record)
        finally:
            store.close()
        return path

    return write


def test_continuous_windows_and_missing_interior(curve):
    losses = {step: float(step) for step in range(1, 306)}
    del losses[150]
    report = curve.summarize_losses(losses, 100)
    assert report["missing_updates"] == [150]
    assert [row["complete"] for row in report["bins"]] == [True, False, True, False]
    assert [row["continuous"] for row in report["bins"]] == [True, False, True, True]
    assert report["latest_complete_update"] == 300
    assert report["bins"][0]["mean_loss"] == 50.5
    assert report["bins"][-1]["count"] == 5


def test_custom_window_and_empty(curve):
    assert curve.summarize_losses({}, 3)["latest_complete_update"] == 0
    assert curve.summarize_losses({step: 1.0 for step in range(1, 8)}, 3)["complete_windows"] == 2
    with pytest.raises(ValueError):
        curve.summarize_losses({}, 0)


def test_real_protobuf_duplicate_last_and_evaluation_ignored(curve, make_wandb_source):
    path = make_wandb_source([
        {"_step": 0, "train/loss": 2.0},
        {"_step": 1, "train/loss": 4.0},
        {"_step": 0, "train/loss": 6.0},
        {"_step": 1, "eval/loss": 99.0},
    ])
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    losses, metadata = curve.read_wandb_losses(path)
    assert losses == {1: 6.0, 2: 4.0}
    assert metadata["duplicates_replaced_by_last"] == 1
    assert metadata["tail_record_not_yet_flushed"] is False
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)


def test_partial_live_tail_keeps_only_complete_records(curve, make_wandb_source):
    path = make_wandb_source([
        {"_step": 0, "train/loss": 2.0},
        {"_step": 1, "train/loss": 4.0},
    ])
    path.write_bytes(path.read_bytes()[:-3])
    losses, metadata = curve.read_wandb_losses(path)
    assert losses == {1: 2.0}
    assert metadata["tail_record_not_yet_flushed"] is True


def test_complete_corruption_rejected(curve, make_wandb_source):
    from wandb.sdk.internal.datastore import LEVELDBLOG_HEADER_LEN

    path = make_wandb_source([
        {"_step": 0, "train/loss": 2.0},
        {"_step": 1, "train/loss": 4.0},
    ])
    damaged = bytearray(path.read_bytes())
    # Locate the first complete record through the actual reader's file-header
    # parser, then flip a payload byte while preserving its length/checksum.
    from wandb.sdk.internal.datastore import DataStore
    import io

    reader = DataStore()
    reader._fp = io.BytesIO(damaged)
    reader._read_header()
    damaged[reader._fp.tell() + LEVELDBLOG_HEADER_LEN] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(ValueError, match="Corrupt W&B record"):
        curve.read_wandb_losses(path)


def test_final_nonfinite_rejected(curve, make_wandb_source):
    path = make_wandb_source([
        {"_step": 0, "train/loss": 2.0},
        {"_step": 0, "train/loss": math.nan},
    ])
    with pytest.raises(ValueError, match="Invalid final train/loss"):
        curve.read_wandb_losses(path)


def test_boolean_step_rejected(curve, make_wandb_source):
    path = make_wandb_source([{"_step": True, "train/loss": 2.0}])
    with pytest.raises(ValueError, match="zero-based"):
        curve.read_wandb_losses(path)


def test_run_binding_both_new_and_legacy(curve, tmp_path):
    run = {"run_timestamp": "abc", "run_directory": str(tmp_path)}
    curve.check_binding({"run_timestamp": "abc", "run_directory": str(tmp_path)}, run)
    curve.check_binding({
        "run_timestamp": "abc", "source": str(tmp_path / "wandb/offline-run-x/run.wandb")
    }, run)
    for invalid in [
        {"run_timestamp": "wrong", "run_directory": str(tmp_path)},
        {"run_timestamp": "abc", "run_directory": str(tmp_path / "other-run")},
        {"run_timestamp": "abc", "source": str(tmp_path / "other-run/wandb/run.wandb")},
    ]:
        with pytest.raises(ValueError):
            curve.check_binding(invalid, run)


def test_wait_without_matplotlib_preserves_all_artifacts(tmp_path):
    state = tmp_path / "state.json"
    run_directory = str(tmp_path / "run")
    state.write_text(json.dumps({"timestamp": "abc", "run_directory": run_directory}))
    report = {
        "run_timestamp": "abc", "run_directory": run_directory,
        "window_size": 100,
        "bins": [{"start_update": 1, "end_update": 100, "complete": True}],
        "latest_flushed_update": 110,
    }
    (tmp_path / "loss_summary.json").write_text(json.dumps(report))
    artifacts = [tmp_path / name for name in ("loss_100step.png", "loss_100step.svg")]
    for path in artifacts:
        path.write_bytes(b"preserve")
    marker = tmp_path / "loss_plot_metadata.json"
    marker.write_text(json.dumps({"run_timestamp": "abc", "run_directory": run_directory}))
    artifacts.append(marker)
    before = {path.name: path.read_bytes() for path in artifacts}
    result = subprocess.run([
        sys.executable, "-S", str(SCRIPT), "--mode", "plot", "--state", str(state),
        "--output-dir", str(tmp_path), "--run-timestamp", "abc",
        "--minimum-complete-update", "200",
    ], capture_output=True, text=True, timeout=15)
    assert result.returncode == 75, (result.stdout, result.stderr)
    assert before == {path.name: path.read_bytes() for path in artifacts}


def test_cli_wrong_run_stops_before_optional_imports(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"timestamp": "abc", "run_directory": str(tmp_path / "run")}))
    output = tmp_path / "output"
    result = subprocess.run([
        sys.executable, "-S", str(SCRIPT), "--mode", "extract", "--state", str(state),
        "--output-dir", str(output), "--run-timestamp", "wrong",
    ], capture_output=True, text=True, timeout=15)
    assert result.returncode == 1
    assert "timestamp changed" in result.stderr
    assert not output.exists()
