"""Read-only token extraction binding and diagnostic throttle regressions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from MotionExpert import probe_tokens
from MotionExpert.probe_prior import file_stamp
from MotionExpert.tests.test_prior_probe_reuse import baseline_job


def token_baseline(tmp_path):
    base, cache, _, job, _ = baseline_job(tmp_path)
    repo = Path(probe_tokens.__file__).resolve().parents[1]
    source = repo / "MotionExpert/probe_prior.py"
    (base / "sources").mkdir()
    (base / "sources/probe_prior-extraction.py").write_bytes(source.read_bytes())
    plan = json.loads((base / "plan.json").read_text())
    plan["input"] = "six historical DINO frames only; all245 Gaussian channels; t=1; no trajectory"
    plan["source_files"]["MotionExpert/probe_prior.py"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (base / "plan.json").write_text(json.dumps(plan))
    return base, cache, job


def test_prepare_preserves_baseline_and_allows_one_physical_gpu(tmp_path):
    base, cache, job = token_baseline(tmp_path)
    before = {str(path.relative_to(base)): file_stamp(path) for path in base.rglob("*") if path.is_file()}
    plan, entries = probe_tokens.prepare(base, cache, job, gpus=[3], batch_size=128, sleep_seconds=1)
    assert plan["epoch"] == 100 and plan["gpus"] == [3]
    assert plan["logical_shards"] == [0]  # Fixture baseline uses one logical shard.
    assert len(entries) == plan["windows"]
    for name in ("selection.json", "targets.npz", "dino.npy", "epoch100.npy", "dino-linear-probe.pt"):
        assert (job / name).resolve() == base / name
    assert json.loads((job / "control.json").read_text()) == {"paused": False, "sleep_seconds": 1.0}
    assert before == {str(path.relative_to(base)): file_stamp(path) for path in base.rglob("*") if path.is_file()}
    probe_tokens.assert_bindings(plan)
    (base / "epoch100.npy").touch()
    with pytest.raises(ValueError, match="artifact changed"):
        probe_tokens.assert_bindings(plan)


@pytest.mark.parametrize("mismatch", ("acceptance", "selection", "source", "checkpoint", "cache"))
def test_prepare_rejects_baseline_mismatches_without_creating_output(tmp_path, mismatch):
    base, cache, job = token_baseline(tmp_path)
    if mismatch == "acceptance":
        (base / "review.json").write_text('{"passed": false}')
    elif mismatch == "selection":
        entries = json.loads((base / "selection.json").read_text())
        entries[0]["start_frame"] = 6
        (base / "selection.json").write_text(json.dumps(entries))
    elif mismatch == "source":
        (base / "sources/probe_prior-extraction.py").write_text("invalid source")
    elif mismatch == "checkpoint":
        plan = json.loads((base / "plan.json").read_text())
        Path(plan["checkpoints"]["100"]).touch()
    else:
        cache.manifest_sha256 = "foreign"
    with pytest.raises(ValueError):
        probe_tokens.prepare(base, cache, job, gpus=[0])
    assert not list(job.iterdir())


@pytest.mark.parametrize("kwargs", ({"gpus": []}, {"gpus": [0, 0]}, {"gpus": [-1]},
    {"gpus": [0], "batch_size": 0}, {"gpus": [0], "sleep_seconds": float("nan")},
    {"gpus": [0], "memory_fraction": 0}))
def test_prepare_rejects_invalid_runtime_limits(tmp_path, kwargs):
    base, cache, job = token_baseline(tmp_path)
    with pytest.raises(ValueError, match="Invalid GPU"):
        probe_tokens.prepare(base, cache, job, **kwargs)
    assert not list(job.iterdir())


@pytest.mark.parametrize("batch_size", (16, 32, 64, 256))
def test_paired_extraction_rejects_changed_batch_shapes(tmp_path, batch_size):
    base, cache, job = token_baseline(tmp_path)
    with pytest.raises(ValueError, match="original baseline batch_size"):
        probe_tokens.prepare(base, cache, job, gpus=[0], batch_size=batch_size)
    assert not list(job.iterdir())


def test_extract_guard_rejects_edited_batch_metadata(tmp_path):
    base, cache, job = token_baseline(tmp_path)
    plan, _ = probe_tokens.prepare(base, cache, job, gpus=[0])
    plan["batch_size"] = 16
    with pytest.raises(ValueError, match="original baseline batch shapes"):
        probe_tokens.assert_bindings(plan)


def test_live_throttle_is_scoped_to_diagnostic(tmp_path):
    assert probe_tokens.read_control(tmp_path, 1) == (False, 1)
    (tmp_path / "control.json").write_text('{"paused": true, "sleep_seconds": 0.3}')
    assert probe_tokens.read_control(tmp_path, 1) == (True, .3)
    for control in ({"paused": "yes"}, {"sleep_seconds": -1}, {"sleep_seconds": 31}, {"sleep_seconds": True}):
        (tmp_path / "control.json").write_text(json.dumps(control))
        with pytest.raises(ValueError, match="Diagnostic control"):
            probe_tokens.read_control(tmp_path, 1)


def test_ast_semantics_ignore_comments_but_reject_changes():
    first = "def f(x):\n    # a comment\n    return x + 1\n"
    second = "def f(x):\n    return x + 1\n"
    assert probe_tokens._function_ast(first, "f") == probe_tokens._function_ast(second, "f")
    assert probe_tokens._function_ast(first, "f") != probe_tokens._function_ast(second.replace("+ 1", "+ 2"), "f")


def test_foreign_plan_does_not_overwrite_existing_progress(tmp_path):
    job = tmp_path / "foreign"
    job.mkdir()
    old = {"stage": "another diagnostic"}
    (job / "progress.json").write_text(json.dumps(old))
    (job / "plan.json").write_text(json.dumps({"format": "foreign", "job_dir": str(job)}))
    with pytest.raises(ValueError, match="another protocol"):
        probe_tokens.main(["--job-dir", str(job), "--stage", "extract"])
    assert json.loads((job / "progress.json").read_text()) == old


def test_valid_owned_extract_failure_records_own_job(tmp_path, monkeypatch):
    base, cache, job = token_baseline(tmp_path)
    probe_tokens.prepare(base, cache, job, gpus=[0])
    monkeypatch.setattr(probe_tokens, "assert_bindings", lambda plan: (_ for _ in ()).throw(ValueError("changed")))
    with pytest.raises(ValueError, match="changed"):
        probe_tokens.main(["--job-dir", str(job), "--stage", "extract"])
    progress = json.loads((job / "progress.json").read_text())
    assert progress["stage"] == "failed" and "changed" in progress["error"]
