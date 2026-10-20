"""CPU regressions for the independent local W&B loss reader/plot boundary.

Real W&B protobuf records and small temporary datastore files exercise the parser;
no training imports, existing run artifacts, models, GPUs, or renderer are used.
"""
from __future__ import annotations

import importlib.util
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

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


def _history_run(curve, tmp_path, through=100):
    prefix_path = tmp_path / "prefix.csv"
    prefix_path.write_text(curve.csv_text(
        ["completed_update", "logged_train_loss"],
        [{"completed_update": step, "logged_train_loss": float(step)}
         for step in range(1, through + 1)],
    ))
    prefix = {
        "path": str(prefix_path), "sha256": hashlib.sha256(prefix_path.read_bytes()).hexdigest(),
        "through_update": through, "source_run_timestamp": "original",
        "source_run_directory": str(tmp_path / "original-run"),
    }
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "timestamp": "resumed", "run_directory": str(tmp_path / "resumed-run"),
        "loss_history_prefix": prefix,
    }))
    return state, curve.load_run(state, "resumed")


def _install_wandb_source(run, source):
    destination = Path(run["run_directory"]) / "wandb/offline-run-fixture/run.wandb"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())
    return destination


def test_resume_extract_joins_frozen_history_and_real_protobuf(curve, tmp_path, make_wandb_source):
    _, run = _history_run(curve, tmp_path)
    # The previous run's records after the resume checkpoint must never be read.
    old_source = Path(run["loss_history_prefix"]["source_run_directory"]) / "wandb/offline-run-old/run.wandb"
    old_source.parent.mkdir(parents=True)
    old_source.write_bytes(b"previous abandoned run is not a loss source")
    source = _install_wandb_source(run, make_wandb_source([
        *[{"_step": step - 1, "train/loss": float(step)} for step in range(101, 206)],
        {"_step": 100, "train/loss": 101.5},
    ]))
    prefix_path = Path(run["loss_history_prefix"]["path"])
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in (source, prefix_path, old_source)}
    output = tmp_path / "output"
    assert curve.extract(SimpleNamespace(output_dir=output, window=100), run) == 0
    report = json.loads((output / "loss_summary.json").read_text())
    assert report["loss_history_prefix"] == run["loss_history_prefix"]
    assert report["source"] == str(source)
    assert report["latest_flushed_update"] == 205
    assert report["latest_complete_update"] == 200
    assert report["missing_updates"] == []
    assert report["distinct_loss_records"] == 205
    assert report["duplicates_replaced_by_last"] == 1
    assert [row["count"] for row in report["bins"]] == [100, 100, 5]
    assert report["bins"][0]["mean_loss"] == 50.5
    assert report["bins"][1]["mean_loss"] == pytest.approx(150.505)
    with (output / "loss_per_update.csv").open() as handle:
        raw = list(csv.DictReader(handle))
    assert [int(row["completed_update"]) for row in raw] == list(range(1, 206))
    assert float(raw[100]["logged_train_loss"]) == 101.5
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}


def test_resume_extract_keeps_a_missing_new_update_visible(curve, tmp_path, make_wandb_source):
    _, run = _history_run(curve, tmp_path)
    _install_wandb_source(run, make_wandb_source([
        {"_step": step - 1, "train/loss": 1.0} for step in range(102, 201)
    ]))
    output = tmp_path / "output"
    assert curve.extract(SimpleNamespace(output_dir=output, window=100), run) == 0
    report = json.loads((output / "loss_summary.json").read_text())
    assert report["missing_updates"] == [101]
    assert report["latest_complete_update"] == 100
    assert report["bins"][1]["complete"] is False


@pytest.mark.parametrize("overlap_step", [0, 99])
def test_resume_refuses_new_run_overlap_without_changing_outputs(
    curve, tmp_path, make_wandb_source, overlap_step,
):
    _, run = _history_run(curve, tmp_path)
    _install_wandb_source(run, make_wandb_source([
        {"_step": overlap_step, "train/loss": 999.0},
        {"_step": 100, "train/loss": 1.0},
    ]))
    output = tmp_path / "output"
    output.mkdir()
    summary = output / "loss_summary.json"
    summary.write_text(json.dumps(run))
    raw = output / "loss_per_update.csv"
    raw.write_bytes(b"preserve")
    before = {path: path.read_bytes() for path in (summary, raw)}
    with pytest.raises(ValueError, match="at or before.*boundary"):
        curve.extract(SimpleNamespace(output_dir=output, window=100), run)
    assert before == {path: path.read_bytes() for path in before}


def test_resume_prefix_hash_is_checked_before_reading_wandb(curve, tmp_path):
    _, run = _history_run(curve, tmp_path)
    Path(run["loss_history_prefix"]["path"]).write_text("changed")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="SHA256"):
        curve.extract(SimpleNamespace(output_dir=output, window=100), run)
    assert not output.exists()


@pytest.mark.parametrize("rows", [
    [(1, 1.0), (3, 3.0)],  # Missing interior row.
    [(1, 1.0), (1, 2.0), (3, 3.0)],  # Duplicates are not resumable history.
    [(2, 2.0), (1, 1.0), (3, 3.0)],
    [(1, 1.0), (2, 2.0)],  # Short endpoint.
    [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)],  # Abandoned post-checkpoint loss.
    [(1, math.nan), (2, 2.0), (3, 3.0)],
    [(1, math.inf), (2, 2.0), (3, 3.0)],
])
def test_frozen_history_requires_finite_continuous_exact_prefix(curve, tmp_path, rows):
    _, run = _history_run(curve, tmp_path, through=3)
    prefix = run["loss_history_prefix"]
    path = Path(prefix["path"])
    path.write_text(curve.csv_text(["completed_update", "logged_train_loss"], [
        {"completed_update": step, "logged_train_loss": value} for step, value in rows
    ]))
    prefix["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="continuous|non-finite"):
        curve.read_history_prefix(prefix)


@pytest.mark.parametrize("content", [
    "update,loss\n1,1\n",
    "completed_update,logged_train_loss,extra\n1,1,x\n",
    "completed_update,logged_train_loss\n1,1,x\n",
    "completed_update,logged_train_loss\n1\n",
    "completed_update,logged_train_loss\n1.0,1\n",
])
def test_frozen_history_rejects_wrong_columns_and_malformed_rows(curve, tmp_path, content):
    _, run = _history_run(curve, tmp_path, through=1)
    prefix = run["loss_history_prefix"]
    path = Path(prefix["path"])
    path.write_text(content)
    prefix["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="columns|invalid"):
        curve.read_history_prefix(prefix)


@pytest.mark.parametrize("field,value", [
    ("path", "relative.csv"), ("source_run_directory", "relative-run"),
    ("sha256", "g" * 64), ("sha256", "0" * 63),
    ("through_update", True), ("through_update", 0), ("source_run_timestamp", ""),
])
def test_load_run_rejects_invalid_prefix_metadata(curve, tmp_path, field, value):
    state, _ = _history_run(curve, tmp_path)
    document = json.loads(state.read_text())
    document["loss_history_prefix"][field] = value
    state.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="prefix"):
        curve.load_run(state, "resumed")


@pytest.mark.parametrize("field,value", [
    ("path", "/different/prefix.csv"), ("sha256", "a" * 64),
    ("through_update", 200), ("source_run_timestamp", "another-source"),
    ("source_run_directory", "/different/source"),
])
def test_existing_outputs_cannot_be_rebound_to_different_history(curve, tmp_path, field, value):
    _, run = _history_run(curve, tmp_path)
    report = json.loads(json.dumps(run))
    curve.check_binding(report, run)
    report["loss_history_prefix"][field] = value
    with pytest.raises(ValueError, match="history prefix"):
        curve.check_binding(report, run)
    with pytest.raises(ValueError, match="history prefix"):
        curve.check_binding({key: value for key, value in run.items() if key != "loss_history_prefix"}, run)
    with pytest.raises(ValueError, match="history prefix"):
        curve.check_binding(run, {key: value for key, value in run.items() if key != "loss_history_prefix"})


def test_plot_rejects_different_history_before_optional_imports(curve, tmp_path):
    state, run = _history_run(curve, tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    report = json.loads(json.dumps(run))
    report["loss_history_prefix"]["sha256"] = "0" * 64
    (output / "loss_summary.json").write_text(json.dumps(report))
    image = output / "loss_100step.png"
    image.write_bytes(b"preserve")
    result = subprocess.run([
        sys.executable, "-S", str(SCRIPT), "--mode", "plot", "--state", str(state),
        "--output-dir", str(output), "--run-timestamp", "resumed",
    ], capture_output=True, text=True, timeout=15)
    assert result.returncode == 1
    assert "history prefix" in result.stderr
    assert image.read_bytes() == b"preserve"


def test_extract_without_history_preserves_original_step_mapping(curve, tmp_path, make_wandb_source):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"timestamp": "fresh", "run_directory": str(tmp_path / "fresh-run")}))
    run = curve.load_run(state, "fresh")
    assert "loss_history_prefix" not in run
    _install_wandb_source(run, make_wandb_source([
        {"_step": step - 1, "train/loss": 2.0} for step in range(1, 101)
    ]))
    output = tmp_path / "output"
    assert curve.extract(SimpleNamespace(output_dir=output, window=100), run) == 0
    report = json.loads((output / "loss_summary.json").read_text())
    assert "loss_history_prefix" not in report
    assert report["latest_complete_update"] == report["distinct_loss_records"] == 100
